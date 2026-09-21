//! Database wrapper for SlateDB.
//!
//! This provides a unified interface for both read-write and read-only database access.
//! Encryption is handled at the SlateDB level via BlockTransformer, so this wrapper
//! just passes through operations.

use crate::fs::errors::FsError;
use crate::fs::key_codec::{BranchId, KeyCodec, KeyPrefix};
use crate::fs::metrics::SegmentFootprintDelta;
use anyhow::Result;
use arc_swap::ArcSwap;
use bytes::Bytes;
use futures::stream::StreamExt;
use slatedb::config::{DurabilityLevel, PutOptions, ReadOptions, ScanOptions, WriteOptions};
use slatedb::{CacheTarget, DbCacheManagerOps, DbReader, WriteBatch};
use slatedb_common::metrics::DefaultMetricsRecorder;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio_stream::Stream;

/// Wrapper for SlateDB handle that can be either read-write or read-only.
pub enum SlateDbHandle {
    ReadWrite(Arc<slatedb::Db>),
    ReadOnly(ArcSwap<DbReader>),
}

impl Clone for SlateDbHandle {
    fn clone(&self) -> Self {
        match self {
            SlateDbHandle::ReadWrite(db) => SlateDbHandle::ReadWrite(db.clone()),
            SlateDbHandle::ReadOnly(reader) => {
                SlateDbHandle::ReadOnly(ArcSwap::new(reader.load_full()))
            }
        }
    }
}

impl SlateDbHandle {
    pub fn is_read_only(&self) -> bool {
        matches!(self, SlateDbHandle::ReadOnly(_))
    }
}

/// Outcome of a [`Db::warm_metadata`] pass.
#[cfg(test)]
#[derive(Debug, Default, Clone, Copy)]
pub struct WarmStats {
    /// Metadata SSTs the warm fan-out touched.
    pub ssts: usize,
    /// Of those, how many had at least one target fail (counted, not fatal).
    pub failed: usize,
}

/// Tracks which metadata SSTs have already been warmed, so that across manifest
/// changes (L0 flushes and compactions, which swap SSTs in and out) each SST is
/// warmed exactly once. Generic over the id type purely so the diff can be unit
/// tested with plain ids; production instantiates it with slatedb's `SsTableId`.
struct WarmTracker<Id> {
    seen: std::collections::HashSet<Id>,
    last_manifest_id: u64,
}

impl<Id: Eq + std::hash::Hash + Copy> WarmTracker<Id> {
    fn new() -> Self {
        Self {
            seen: std::collections::HashSet::new(),
            last_manifest_id: 0,
        }
    }

    /// Given a manifest id and that manifest's live metadata SST ids, return the
    /// ids not yet warmed (recording them as warmed) and forget ids that are no
    /// longer live. Returns empty when the manifest id is unchanged, so a status
    /// notification that didn't change the manifest (e.g. a durability advance) is
    /// a no-op.
    fn plan(&mut self, manifest_id: u64, current: impl Iterator<Item = Id>) -> Vec<Id> {
        if manifest_id == self.last_manifest_id {
            return Vec::new();
        }
        self.last_manifest_id = manifest_id;
        let current: std::collections::HashSet<Id> = current.collect();
        // Drop retired ids so the set can't grow without bound (and a future SST
        // that reuses a retired id would warm again).
        self.seen.retain(|id| current.contains(id));
        current
            .into_iter()
            .filter(|id| self.seen.insert(*id))
            .collect()
    }
}

/// DST override for [`exit_on_write_error`]: in the simulation the "process"
/// is one instance inside the test, so death becomes a task panic: the
/// instance's workers stop and the test goes on to verify what the crash
/// left behind.
#[doc(hidden)]
pub static DST_PANIC_ON_WRITE_ERROR: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fatal handler for SlateDB write errors.
/// After a write failure, the database state is unknown. Exit and let
/// the eventual orchestrator restart the service to rebuild from a known-good state.
pub fn exit_on_write_error(err: impl std::fmt::Display) -> ! {
    if DST_PANIC_ON_WRITE_ERROR.load(std::sync::atomic::Ordering::Relaxed) {
        panic!("dst: simulated process death on write error: {err}");
    }
    tracing::error!("Fatal write error, exiting: {}", err);
    std::process::exit(1)
}

enum TxOp {
    Put(Bytes, Bytes),
    Delete(Bytes),
}

/// Usage-stats adjustment riding along with a transaction. Deltas commute, so
/// the commit worker can aggregate them per shard across a whole batch and
/// persist one absolute shard value, without any per-operation locking.
pub(crate) struct StatsDelta {
    pub(crate) inode_id: u64,
    pub(crate) bytes: i64,
    pub(crate) inodes: i64,
}

/// Transaction for batching database writes.
///
/// Ops are recorded as a flat vector so the commit coordinator can replay
/// several transactions into a single merged `WriteBatch` via [`apply_to`].
pub struct Transaction {
    ops: Vec<TxOp>,
    inode_cache_invalidations: Vec<u64>,
    directory_entry_cache_invalidations: Vec<(u64, Bytes)>,
    stats_deltas: Vec<StatsDelta>,
    /// Per-segment counter adjustments (segcount key, `(live_delta, total_delta)`),
    /// aggregated by the commit worker into one absolute `(live, total)` per
    /// segment. Same lock-free pattern as `stats_deltas`.
    seg_deltas: Vec<(Bytes, (i64, i64))>,
    /// Footprint debit for raw segment-counter rows deleted by this transaction.
    segcount_delete_delta: SegmentFootprintDelta,
    /// Pins FrameLoc publication from assignment through commit.
    extent_ref_guard: Option<ExtentRefGuard>,
    dedup_entry: Option<crate::dedup::DedupEntry>,
}

/// Cloneable without re-acquiring the non-reentrant read lock.
pub(crate) type ExtentRefGuard = Arc<tokio::sync::OwnedRwLockReadGuard<()>>;

impl Transaction {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            inode_cache_invalidations: Vec::new(),
            directory_entry_cache_invalidations: Vec::new(),
            stats_deltas: Vec::new(),
            seg_deltas: Vec::new(),
            segcount_delete_delta: SegmentFootprintDelta::default(),
            extent_ref_guard: None,
            dedup_entry: None,
        }
    }

    pub(crate) fn has_extent_ref_guard(&self) -> bool {
        self.extent_ref_guard.is_some()
    }

    pub(crate) fn hold_extent_ref_guard(&mut self, guard: ExtentRefGuard) {
        debug_assert!(
            self.extent_ref_guard.is_none(),
            "a transaction may hold only one extent reference guard"
        );
        self.extent_ref_guard = Some(guard);
    }

    pub(crate) fn take_extent_ref_guard(&mut self) -> Option<ExtentRefGuard> {
        self.extent_ref_guard.take()
    }

    /// Attach the result published after this transaction applies.
    pub fn set_dedup_result(
        &mut self,
        op_id: crate::dedup::OpId,
        result: crate::dedup::DedupResult,
    ) {
        self.dedup_entry =
            crate::dedup::has_op_id(&op_id).then_some(crate::dedup::DedupEntry { op_id, result });
    }

    pub(crate) fn take_dedup_entry(&mut self) -> Option<crate::dedup::DedupEntry> {
        self.dedup_entry.take()
    }

    pub fn put_bytes(&mut self, key: &Bytes, value: Bytes) {
        self.ops.push(TxOp::Put(key.clone(), value));
    }

    pub fn delete_bytes(&mut self, key: &Bytes) {
        self.ops.push(TxOp::Delete(key.clone()));
    }

    /// Delete one raw segment-counter row and publish the matching footprint
    /// debit after the transaction applies.
    pub(crate) fn delete_segcount(&mut self, key: &Bytes, live: u64, total: u64) {
        self.delete_bytes(key);
        self.segcount_delete_delta.merge(SegmentFootprintDelta::new(
            -1,
            -i64::try_from(total).unwrap_or(i64::MAX),
            -i64::try_from(live).unwrap_or(i64::MAX),
        ));
    }

    pub(crate) fn invalidate_cached_inode(&mut self, inode_id: u64) {
        self.inode_cache_invalidations.push(inode_id);
    }

    pub(crate) fn take_inode_cache_invalidations(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.inode_cache_invalidations)
    }

    pub(crate) fn invalidate_cached_directory_entry(&mut self, dir_id: u64, name: Bytes) {
        self.directory_entry_cache_invalidations
            .push((dir_id, name));
    }

    pub(crate) fn take_directory_entry_cache_invalidations(&mut self) -> Vec<(u64, Bytes)> {
        std::mem::take(&mut self.directory_entry_cache_invalidations)
    }

    /// Record a usage-stats adjustment for `inode_id`'s shard, materialized
    /// by the commit worker. No-op deltas are dropped so callers can pass
    /// computed differences unconditionally.
    pub fn add_stats_delta(&mut self, inode_id: u64, bytes: i64, inodes: i64) {
        if bytes != 0 || inodes != 0 {
            self.stats_deltas.push(StatsDelta {
                inode_id,
                bytes,
                inodes,
            });
        }
    }

    pub(crate) fn take_stats_deltas(&mut self) -> Vec<StatsDelta> {
        std::mem::take(&mut self.stats_deltas)
    }

    /// Record live/total byte adjustments for a segment's counter (`segcount_key`),
    /// materialized as an absolute `(live, total)` by the commit worker. A frame
    /// write credits both (`+len, +len`); an overwrite/delete debits live only
    /// (`-len, 0`), keeping `total` monotonic. All-zero deltas drop.
    pub fn add_seg_delta(&mut self, segcount_key: &Bytes, live_delta: i64, total_delta: i64) {
        if live_delta != 0 || total_delta != 0 {
            self.seg_deltas
                .push((segcount_key.clone(), (live_delta, total_delta)));
        }
    }

    pub(crate) fn take_seg_deltas(&mut self) -> Vec<(Bytes, (i64, i64))> {
        std::mem::take(&mut self.seg_deltas)
    }

    pub(crate) fn take_segcount_delete_delta(&mut self) -> SegmentFootprintDelta {
        std::mem::take(&mut self.segcount_delete_delta)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of staged key operations (puts and deletes).
    #[cfg(test)]
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Replay this transaction's ops into `target`. SlateDB's `WriteBatch`
    /// already dedupes per key, so calling this on multiple transactions
    /// produces one merged batch with last-write-wins per key.
    ///
    /// Side channels must be drained by the write coordinator first.
    pub(crate) fn apply_to(self, target: &mut WriteBatch) {
        for op in self.into_ops() {
            match op {
                TxOp::Put(k, v) => target.put_bytes(k, v),
                TxOp::Delete(k) => target.delete(k),
            }
        }
    }

    /// Consume into the staged op vector after the side-channel assertions.
    /// [`Db::apply_transaction`] uses this to remap ops for a branch view
    /// before they reach the batch.
    fn into_ops(self) -> Vec<TxOp> {
        self.assert_side_channels_drained();
        self.ops
    }

    fn assert_side_channels_drained(&self) {
        assert!(
            self.inode_cache_invalidations.is_empty(),
            "inode cache invalidations would be dropped: commit inode mutations through the \
             WriteCoordinator"
        );
        assert!(
            self.directory_entry_cache_invalidations.is_empty(),
            "directory-entry cache invalidations would be dropped: commit namespace mutations \
             through the WriteCoordinator"
        );
        assert!(
            self.stats_deltas.is_empty(),
            "stats deltas would be dropped: commit a stats-bearing transaction through the \
             WriteCoordinator"
        );
        assert!(
            self.seg_deltas.is_empty(),
            "seg_deltas would be dropped: commit a seg_delta-bearing txn through the \
             WriteCoordinator"
        );
        assert_eq!(
            self.segcount_delete_delta,
            SegmentFootprintDelta::default(),
            "segment footprint delta would be dropped: commit through the WriteCoordinator"
        );
        assert!(
            self.extent_ref_guard.is_none(),
            "extent reference guard would be dropped before commit"
        );
        assert!(
            self.dedup_entry.is_none(),
            "dedup result would be dropped: commit a deduplicated transaction through the \
             WriteCoordinator"
        );
    }

    /// Like [`apply_to`](Self::apply_to) but also returns the ops as `ReplOp`s
    /// for shipping. In apply order; replaying in seqno-then-op order on the
    /// standby reproduces the merged batch's last-write-wins result.
    pub(crate) fn apply_to_collecting(
        self,
        target: &mut WriteBatch,
    ) -> Vec<crate::replication::ReplOp> {
        use crate::replication::ReplOp;
        self.assert_side_channels_drained();
        let mut ops = Vec::with_capacity(self.ops.len());
        for op in self.ops {
            match op {
                TxOp::Put(k, v) => {
                    target.put_bytes(k.clone(), v.clone());
                    ops.push(ReplOp::Put(k, v));
                }
                TxOp::Delete(k) => {
                    target.delete(k.clone());
                    ops.push(ReplOp::Delete(k));
                }
            }
        }
        ops
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::new()
    }
}

/// Database wrapper providing a unified interface for SlateDB operations.
///
/// With BlockTransformer handling encryption at the SlateDB level, this wrapper
/// simply passes through operations without additional encryption/decryption.
pub struct Db {
    inner: SlateDbHandle,
    metrics_recorder: Option<Arc<DefaultMetricsRecorder>>,
    /// HA leader lease. `Some` only under replication; reads/writes are refused
    /// while invalid so a deposed node never serves stale data. `None` (ungated)
    /// in single-node mode.
    lease: Option<Arc<crate::replication::Lease>>,
    /// Latest data-db status snapshot. A fenced SlateDB handle publishes its
    /// close reason here; locally initiated close is gated earlier by `closing`.
    status: Option<tokio::sync::watch::Receiver<slatedb::DbStatus>>,
    /// Flush barrier. Every commit takes a read lock; the seal+flush durability
    /// sequence (flush coordinator, segment reclaim) takes the write lock. This
    /// keeps the flush from durably capturing a `FrameLoc` whose segment is still
    /// the un-PUT open buffer: a commit that overlaps a flush lands *after* it, so
    /// its pointer is never in the flushed set referencing an un-sealed segment.
    flush_barrier: Arc<tokio::sync::RwLock<()>>,
    /// Rejects cache hits and writers once local close begins.
    closing: AtomicBool,
    /// The basin branch this handle serves (see [`Self::with_branch`]).
    /// [`BranchId::ROOT`] is the volume root and behaves exactly as before
    /// branches existed.
    branch: BranchId,
}

/// Admission to one database write while holding the flush barrier's read side.
///
/// Acquiring the permit performs all potentially blocking lease and flush-barrier
/// waits. Callers can then establish short-lived cache invalidation windows
/// immediately before the atomic SlateDB apply.
#[must_use = "an admitted write must be applied or explicitly abandoned"]
pub(crate) struct WritePermit<'a> {
    db: &'a Db,
    _flush_guard: tokio::sync::RwLockReadGuard<'a, ()>,
}

impl WritePermit<'_> {
    pub(crate) async fn write_with_options(
        self,
        batch: WriteBatch,
        options: &WriteOptions,
    ) -> Result<u64> {
        let result = self.db.write_admitted(batch, options).await;
        drop(self);
        result
    }
}

impl Db {
    pub fn new(
        db: Arc<slatedb::Db>,
        metrics_recorder: Option<Arc<DefaultMetricsRecorder>>,
    ) -> Self {
        let status = Some(db.subscribe());
        Self {
            inner: SlateDbHandle::ReadWrite(db),
            metrics_recorder,
            lease: None,
            status,
            flush_barrier: Arc::new(tokio::sync::RwLock::new(())),
            closing: AtomicBool::new(false),
            branch: BranchId::ROOT,
        }
    }

    pub fn new_read_only(db_reader: ArcSwap<DbReader>) -> Self {
        Self {
            inner: SlateDbHandle::ReadOnly(db_reader),
            metrics_recorder: None,
            lease: None,
            status: None,
            flush_barrier: Arc::new(tokio::sync::RwLock::new(())),
            closing: AtomicBool::new(false),
            branch: BranchId::ROOT,
        }
    }

    /// Restrict this handle to a basin branch's view of the keyspace (see the
    /// branch-dimension comment in [`crate::fs::key_codec`]). Keys passed to a
    /// branched handle must be built by the matching
    /// [`KeyCodec::for_branch`] codec. Point reads of scoped kinds then
    /// resolve nearest-writer-wins: the branch's own key, then a branch
    /// tombstone (present means deleted-in-branch), then the parent key.
    /// Writes land in the branch's key range, and deletes of parent-visible
    /// keys become branch tombstones instead of real deletes (see
    /// [`Self::apply_transaction`]). Unscoped kinds and [`BranchId::ROOT`]
    /// behave exactly as before branches existed.
    pub fn with_branch(mut self, branch: BranchId) -> Self {
        self.branch = branch;
        self
    }

    pub fn branch(&self) -> BranchId {
        self.branch
    }

    /// The flush barrier (see the field). The seal+flush durability sequence holds
    /// the *write* lock across `seal_open()` + `flush()`; commits hold a read lock.
    pub fn flush_barrier(&self) -> Arc<tokio::sync::RwLock<()>> {
        Arc::clone(&self.flush_barrier)
    }

    /// Latest SlateDB sequence durably published to object storage.
    pub(crate) fn durable_seq(&self) -> u64 {
        self.status
            .as_ref()
            .map_or(0, |status| status.borrow().durable_seq)
    }

    /// Manifest id of the live in-memory manifest snapshot, for the
    /// flush-time index (`KeyPrefix::FlushTime`). Read directly from the db
    /// state (not the status watch, whose notification can lag the flush that
    /// published the manifest). `None` on a read-only handle, which has no
    /// flush path.
    pub(crate) fn current_manifest_id(&self) -> Option<u64> {
        match &self.inner {
            SlateDbHandle::ReadWrite(db) => Some(db.manifest().id()),
            SlateDbHandle::ReadOnly(_) => None,
        }
    }

    /// Best-effort bookkeeping write that returns its error to the caller
    /// instead of taking the process-fatal serving path
    /// ([`exit_on_write_error`]). For writes whose loss is tolerable — the
    /// flush-time index is one: a missed row just engages the manifest-listing
    /// fallback in [`crate::fork_manager::ForkManager::manifest_at_time`].
    pub(crate) async fn try_put(&self, key: &Bytes, value: &[u8]) -> Result<()> {
        match &self.inner {
            SlateDbHandle::ReadWrite(db) => db
                .put_with_options(key, value, &PutOptions::default(), &WriteOptions::default())
                .await
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!("best-effort put failed: {e}")),
            SlateDbHandle::ReadOnly(_) => Err(FsError::ReadOnlyFilesystem.into()),
        }
    }

    /// Attach the HA leader lease; reads/writes are then refused while it is
    /// invalid. Single-node `Db`s have no lease and are never gated.
    pub fn with_lease(mut self, lease: Arc<crate::replication::Lease>) -> Self {
        self.lease = Some(lease);
        self
    }

    /// Check current serving authority.
    #[inline]
    fn check_lease(&self) -> Result<()> {
        self.check_serving_authority().map_err(Into::into)
    }

    /// Enforce the HA gate for reads satisfied without touching SlateDB.
    #[inline]
    pub(crate) fn check_serving_authority(&self) -> Result<(), FsError> {
        if self.is_deposed() {
            Err(FsError::LeaderLeaseExpired)
        } else {
            Ok(())
        }
    }

    /// Wait through recoverable suspension for admitted or durability work.
    /// Terminal revocation and database fencing remain errors. It does not
    /// authorize new requests or successful responses.
    async fn check_internal_lease(&self) -> Result<()> {
        if let Some(lease) = &self.lease
            && !lease.wait_until_internal_work_is_permitted().await
        {
            return Err(FsError::LeaderLeaseExpired.into());
        }
        if let Some(status) = &self.status
            && status.borrow().close_reason.is_some()
        {
            return Err(FsError::LeaderLeaseExpired.into());
        }
        Ok(())
    }

    #[inline]
    fn check_closing(&self) -> Result<()> {
        if self.closing.load(Ordering::Acquire) {
            return Err(FsError::ShuttingDown.into());
        }
        Ok(())
    }

    /// Called under the flush barrier's write lock immediately before close.
    pub fn mark_closing(&self) {
        self.closing.store(true, Ordering::Release);
    }

    /// Whether a protocol adapter may emit a successful response.
    #[inline]
    pub fn permits_successful_response(&self) -> bool {
        !self.is_deposed()
    }

    /// Waits for terminal HA lease revocation. Without a lease, this does not complete.
    pub async fn serving_authority_lost(&self) {
        match &self.lease {
            Some(lease) => lease.revoked().await,
            None => std::future::pending().await,
        }
    }

    /// Permanently revoke the attached serving lease.
    pub fn revoke_lease(&self) {
        if let Some(lease) = &self.lease {
            lease.revoke();
        }
    }

    /// True once no longer the serving leader: the lease is invalid, or the data
    /// db has been closed (fenced by a takeover).
    #[inline]
    fn is_deposed(&self) -> bool {
        if self.closing.load(Ordering::Acquire) {
            return true;
        }
        if let Some(lease) = &self.lease
            && !lease.is_valid()
        {
            return true;
        }
        if let Some(status) = &self.status
            && status.borrow().close_reason.is_some()
        {
            return true;
        }
        false
    }

    pub fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }

    pub async fn get_bytes(&self, key: &[u8]) -> Result<Option<Bytes>> {
        self.get_bytes_at(key, DurabilityLevel::Memory).await
    }

    /// Point read for commit-worker internal work.
    ///
    /// Unlike a serving read, this waits through a recoverable lease
    /// suspension. Commit preparation uses it before write admission so a
    /// temporary authority gap does not turn a safe retry into an I/O error.
    pub(crate) async fn get_bytes_internal(&self, key: &[u8]) -> Result<Option<Bytes>> {
        self.check_internal_lease().await?;
        self.check_closing()?;
        let result = self
            .get_bytes_at_unchecked(key, DurabilityLevel::Memory)
            .await?;
        self.check_internal_lease().await?;
        self.check_closing()?;
        Ok(result)
    }

    /// Point read seeing only object-storage-durable data.
    pub async fn get_bytes_durable(&self, key: &[u8]) -> Result<Option<Bytes>> {
        self.get_bytes_at(key, DurabilityLevel::Remote).await
    }

    /// Point read of the exact stored key for commit-worker internal work,
    /// with NO branch-view fallback: unlike [`Self::get_bytes_internal`], a
    /// parent-visible row does NOT make the key present. Segment-counter
    /// ownership needs the distinction — a branch debits only counters that
    /// exist in its own scope (see `write_coordinator::stage_seg_deltas`).
    pub(crate) async fn get_bytes_own_internal(&self, key: &[u8]) -> Result<Option<Bytes>> {
        self.check_internal_lease().await?;
        self.check_closing()?;
        self.raw_get(key, DurabilityLevel::Memory).await
    }

    async fn get_bytes_at(
        &self,
        key: &[u8],
        durability_filter: DurabilityLevel,
    ) -> Result<Option<Bytes>> {
        self.check_lease()?;
        let result = self.get_bytes_at_unchecked(key, durability_filter).await?;

        // Do not publish a point-read result obtained across serving-authority
        // loss. Cache loaders rely on this check before admitting the value.
        self.check_lease()?;
        Ok(result)
    }

    async fn get_bytes_at_unchecked(
        &self,
        key: &[u8],
        durability_filter: DurabilityLevel,
    ) -> Result<Option<Bytes>> {
        let result = self.raw_get(key, durability_filter).await?;
        if result.is_some() || self.branch.is_root() {
            return Ok(result);
        }
        // Basin-branch fallback for scoped kinds: the branch never wrote this
        // key. A branch tombstone means the branch deleted the parent-visible
        // key (report it absent); otherwise resolve through the parent.
        let codec = KeyCodec::for_branch(self.branch);
        if !matches!(codec.peek_kind(key), Some(kind) if kind.is_scoped()) {
            return Ok(result);
        }
        let Some(tombstone) = codec.branch_tombstone_key(key) else {
            // Scoped kind but not a well-formed key of this branch: serve the
            // raw (absent) result rather than guess at a parent key.
            return Ok(result);
        };
        if self.raw_get(&tombstone, durability_filter).await?.is_some() {
            return Ok(None);
        }
        let Some(parent) = codec.strip_branch(key) else {
            return Ok(None);
        };
        self.raw_get(&parent, durability_filter).await
    }

    /// Point read of the exact stored key, with no branch-view resolution.
    async fn raw_get(
        &self,
        key: &[u8],
        durability_filter: DurabilityLevel,
    ) -> Result<Option<Bytes>> {
        let read_options = ReadOptions {
            durability_filter,
            cache_blocks: true,
            ..Default::default()
        };

        let result = match &self.inner {
            SlateDbHandle::ReadWrite(db) => db.get_with_options(key, &read_options).await?,
            SlateDbHandle::ReadOnly(reader_swap) => {
                let reader = reader_swap.load();
                reader.get_with_options(key, &read_options).await?
            }
        };
        Ok(result)
    }

    /// Scan a key range.
    ///
    /// Under a branch-view handle ([`Self::with_branch`]) with a range that
    /// lies within one branch-scoped kind (the shape every
    /// [`KeyCodec::for_branch`] range has), the stream is the *merged* branch
    /// view of the range: the parent's rows plus the branch's own rows,
    /// merged as a streaming k-way merge of the two underlying scans
    /// (parent prefix + branch prefix), never a materialize-both-then-sort.
    /// Ordering is by key *suffix* — the bytes after `domain || kind` on the
    /// parent side and after `domain || kind || branch` on the branch side,
    /// compared lexicographically ([`KeyCodec::scoped_suffix`]). A branch row
    /// shadows the parent row with the same suffix; a parent row shadowed by
    /// a branch tombstone ([`KeyCodec::branch_tombstone_key`]) is suppressed;
    /// tombstone rows themselves are never emitted (they live in their own
    /// kind outside the scanned range). Surviving parent rows are re-keyed
    /// into the branch layout ([`KeyCodec::adopt_parent_key`]), so every
    /// emitted key parses under the branch codec.
    ///
    /// The tombstone suppression set is prefetched with a single bounded
    /// range scan over the branch's tombstone kind mapped from the parent
    /// bounds ([`KeyCodec::branch_tombstone_bound`]) — one extra scan per
    /// merged scan, never a point lookup per candidate row.
    ///
    /// Root handles, unscoped kinds, and ranges that do not lie within one
    /// scoped kind (e.g. whole-database scans) behave exactly as before
    /// branches existed: a raw scan of the stored keys.
    pub async fn scan<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send + '_>>> {
        self.scan_at(range, DurabilityLevel::Memory).await
    }

    /// Scan seeing only object-storage-durable data. The branch-view merge
    /// (see [`Self::scan`]) applies at this durability level as well: all
    /// three underlying scans (parent, branch, tombstones) use it.
    pub async fn scan_durable<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send + '_>>> {
        self.scan_at(range, DurabilityLevel::Remote).await
    }

    /// Scan the exact stored range with no branch-view resolution: every
    /// scope's rows, as stored, never merged or re-keyed (the inverse of
    /// [`Self::scan`]'s branch view). Volume-wide maintenance that must see
    /// across branch scopes uses this — the orphan sweep's segcount census,
    /// a branch's own-scope reclaim scan, branch-data deletion. Serving
    /// scans never should. On a root handle this is byte-for-byte
    /// [`Self::scan`].
    pub(crate) async fn scan_raw<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send + '_>>> {
        self.scan_raw_at(range, DurabilityLevel::Memory).await
    }

    /// [`Self::scan_raw`] seeing only object-storage-durable data.
    pub(crate) async fn scan_durable_raw<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send + '_>>> {
        self.scan_raw_at(range, DurabilityLevel::Remote).await
    }

    async fn scan_raw_at<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
        durability_filter: DurabilityLevel,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send + '_>>> {
        self.check_lease()?;
        let scan_options = ScanOptions {
            durability_filter,
            read_ahead_bytes: 4 * 1024 * 1024,
            cache_blocks: true,
            max_fetch_tasks: 4,
            ..Default::default()
        };
        let iter = self.raw_scan(range, &scan_options).await?;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<(Bytes, Bytes)>>(32);
        tokio::spawn(forward_scan(iter, tx));
        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn scan_at<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
        durability_filter: DurabilityLevel,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send + '_>>> {
        self.check_lease()?;
        let scan_options = ScanOptions {
            durability_filter,
            read_ahead_bytes: 4 * 1024 * 1024,
            cache_blocks: true,
            max_fetch_tasks: 4,
            ..Default::default()
        };

        let codec = KeyCodec::for_branch(self.branch);
        if let Some((parent_start, parent_end)) =
            branch_parent_range(&codec, range.start_bound(), range.end_bound())
        {
            // Merged branch view: a streaming suffix-ordered merge of the
            // parent's range and the branch's range (see [`BranchMerge`]).
            let parent = self
                .raw_scan(
                    BoundsRange(parent_start.clone(), parent_end.clone()),
                    &scan_options,
                )
                .await?;
            let branch = self.raw_scan(range, &scan_options).await?;
            let tombstones = self
                .branch_tombstone_suffixes(&codec, &parent_start, &parent_end, &scan_options)
                .await?;
            let merge = BranchMerge::new(parent, branch, tombstones, codec);
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<(Bytes, Bytes)>>(32);
            tokio::spawn(forward_scan(merge, tx));
            return Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)));
        }

        let iter = self.raw_scan(range, &scan_options).await?;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<(Bytes, Bytes)>>(32);
        tokio::spawn(forward_scan(iter, tx));
        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    /// Prefix scan that consults SlateDB SST filters to skip non-matching SSTs.
    ///
    /// `seek_to` (a full key within the prefix) is pushed down as the scan's
    /// lower bound, so SlateDB prunes sorted runs entirely below the resume
    /// point at setup instead of opening them and seeking forward.
    /// `read_ahead_bytes` controls SlateDB's read-ahead within the iterator.
    ///
    /// Branch-view handles merge the prefix's parent scope and branch scope
    /// exactly like [`Self::scan`] (same ordering, shadowing, and tombstone
    /// suppression); `seek_to` is mapped into both scopes, and the tombstone
    /// set is one prefix scan of the branch's tombstone rows for the scanned
    /// suffix prefix.
    pub async fn scan_prefix(
        &self,
        prefix: Bytes,
        seek_to: Option<Bytes>,
        read_ahead_bytes: usize,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<(Bytes, Bytes)>> + Send>>> {
        self.check_lease()?;
        let scan_options = ScanOptions {
            durability_filter: DurabilityLevel::Memory,
            read_ahead_bytes,
            cache_blocks: true,
            max_fetch_tasks: 4,
            ..Default::default()
        };

        let codec = KeyCodec::for_branch(self.branch);
        let scoped = if self.branch.is_root() {
            None
        } else {
            codec
                .scoped_suffix(&prefix)
                .map(Bytes::copy_from_slice)
        };
        if let Some(suffix_prefix) = scoped {
            let kind = codec
                .peek_kind(&prefix)
                .expect("scoped_suffix implies a decodable kind");
            // Merged branch view of a scoped prefix: parent prefix scan +
            // branch prefix scan + one tombstone prefix scan, streamed
            // through the suffix-ordered merge (see [`BranchMerge`]).
            let parent_prefix = codec.strip_branch(&prefix).ok_or_else(|| {
                anyhow::anyhow!("scan_prefix prefix is not a well-formed key of this branch")
            })?;
            let (branch_suffix, parent_suffix) = match &seek_to {
                Some(key) => {
                    debug_assert!(
                        key.starts_with(prefix.as_ref()),
                        "scan_prefix seek_to must fall within the prefix"
                    );
                    let parent_key = codec.strip_branch(key).ok_or_else(|| {
                        anyhow::anyhow!("scan_prefix seek_to is not a key of this branch")
                    })?;
                    debug_assert!(parent_key.starts_with(parent_prefix.as_ref()));
                    (
                        key.slice(prefix.len()..),
                        parent_key.slice(parent_prefix.len()..),
                    )
                }
                None => (Bytes::new(), Bytes::new()),
            };
            let parent = self
                .raw_scan_prefix(parent_prefix, parent_suffix, &scan_options)
                .await?;
            let branch = self
                .raw_scan_prefix(prefix.clone(), branch_suffix, &scan_options)
                .await?;
            let mut tombstone_iter = self
                .raw_scan_prefix(
                    codec.branch_tombstone_prefix(kind, &suffix_prefix),
                    Bytes::new(),
                    &scan_options,
                )
                .await?;
            let tombstones = collect_tombstone_suffixes(&codec, &mut tombstone_iter).await?;
            let merge = BranchMerge::new(parent, branch, tombstones, codec);
            return Ok(Box::pin(futures::stream::unfold(
                merge,
                |mut merge| async move {
                    match merge.next_kv().await {
                        Ok(Some(kv)) => Some((Ok(kv), merge)),
                        Ok(None) => None,
                        Err(e) => Some((Err(e), merge)),
                    }
                },
            )));
        }

        // Push the optional resume point down as the subrange's lower bound
        // rather than scanning from the prefix start and seeking forward.
        // SlateDB selects only sorted runs whose key range overlaps the scan
        // range, so a tighter lower bound prunes SSTs that sit entirely below
        // the resume point before they are ever opened. `seek_to` is a full key
        // (`prefix ++ suffix`); the subrange is a prefix-relative suffix, so we
        // strip the prefix. An empty suffix reproduces the full-prefix scan.
        let suffix = match &seek_to {
            Some(key) => {
                debug_assert!(
                    key.starts_with(prefix.as_ref()),
                    "scan_prefix seek_to must fall within the prefix"
                );
                key.slice(prefix.len()..)
            }
            None => Bytes::new(),
        };

        let iter = self.raw_scan_prefix(prefix, suffix, &scan_options).await?;

        Ok(Box::pin(futures::stream::unfold(
            iter,
            |mut iter| async move {
                match iter.next().await {
                    Ok(Some(kv)) => Some((Ok((kv.key, kv.value)), iter)),
                    Ok(None) => None,
                    Err(e) => Some((Err(e.into()), iter)),
                }
            },
        )))
    }

    /// One raw range scan against the underlying handle with no branch-view
    /// resolution — the merge's operands and the non-merged path share it.
    async fn raw_scan<R: slatedb::ByteRangeBounds + Send>(
        &self,
        range: R,
        scan_options: &ScanOptions,
    ) -> Result<slatedb::DbIterator> {
        Ok(match &self.inner {
            SlateDbHandle::ReadWrite(db) => db.scan_with_options(range, scan_options).await?,
            SlateDbHandle::ReadOnly(reader_swap) => {
                reader_swap.load().scan_with_options(range, scan_options).await?
            }
        })
    }

    /// [`Self::raw_scan`] for a prefix plus a `suffix..` subrange.
    async fn raw_scan_prefix(
        &self,
        prefix: Bytes,
        suffix: Bytes,
        scan_options: &ScanOptions,
    ) -> Result<slatedb::DbIterator> {
        Ok(match &self.inner {
            SlateDbHandle::ReadWrite(db) => {
                db.scan_prefix_with_options(prefix, suffix.., scan_options)
                    .await?
            }
            SlateDbHandle::ReadOnly(reader_swap) => {
                reader_swap
                    .load()
                    .scan_prefix_with_options(prefix, suffix.., scan_options)
                    .await?
            }
        })
    }

    /// This branch's tombstone-shadowed suffixes within a merged scan's
    /// parent-view bounds, in ascending order. One bounded range scan per
    /// merged scan — never a point lookup per candidate parent row.
    async fn branch_tombstone_suffixes(
        &self,
        codec: &KeyCodec,
        parent_start: &Bound<Bytes>,
        parent_end: &Bound<Bytes>,
        scan_options: &ScanOptions,
    ) -> Result<Vec<Bytes>> {
        let map = |bound: &Bound<Bytes>, what: &str| -> Result<Bound<Bytes>> {
            Ok(match bound {
                Bound::Included(b) => Bound::Included(
                    codec
                        .branch_tombstone_bound(b)
                        .ok_or_else(|| anyhow::anyhow!("cannot map merged-scan {what} bound"))?,
                ),
                Bound::Excluded(b) => Bound::Excluded(
                    codec
                        .branch_tombstone_bound(b)
                        .ok_or_else(|| anyhow::anyhow!("cannot map merged-scan {what} bound"))?,
                ),
                Bound::Unbounded => Bound::Unbounded,
            })
        };
        let start = map(parent_start, "start")?;
        let end = map(parent_end, "end")?;
        let mut iter = self.raw_scan(BoundsRange(start, end), scan_options).await?;
        collect_tombstone_suffixes(codec, &mut iter).await
    }

    /// Returns the committed batch's SlateDB seqnum, mapped to `durable_seq` to
    /// advance the standby's prune watermark.
    pub async fn write_with_options(
        &self,
        batch: WriteBatch,
        options: &WriteOptions,
    ) -> Result<u64> {
        self.acquire_write_permit()
            .await?
            .write_with_options(batch, options)
            .await
    }

    /// Complete the potentially blocking admission phase for a database write.
    pub(crate) async fn acquire_write_permit(&self) -> Result<WritePermit<'_>> {
        if self.is_read_only() {
            return Err(FsError::ReadOnlyFilesystem.into());
        }
        self.check_internal_lease().await?;
        // Read side of the flush barrier: a commit overlapping a seal+flush waits
        // for it, so its pointer lands after the flush, never in the flushed set.
        let flush_guard = self.flush_barrier.read().await;
        // Recheck authority after waiting on the flush barrier.
        self.check_internal_lease().await?;
        self.check_closing()?;
        Ok(WritePermit {
            db: self,
            _flush_guard: flush_guard,
        })
    }

    async fn write_admitted(&self, batch: WriteBatch, options: &WriteOptions) -> Result<u64> {
        match &self.inner {
            SlateDbHandle::ReadWrite(db) => match db.write_with_options(batch, options).await {
                Ok(handle) => Ok(handle.seqnum()),
                Err(e) => exit_on_write_error(e),
            },
            SlateDbHandle::ReadOnly(_) => unreachable!(),
        }
    }

    pub fn new_transaction(&self) -> Result<Transaction, FsError> {
        if self.is_read_only() {
            return Err(FsError::ReadOnlyFilesystem);
        }
        Ok(Transaction::new())
    }

    /// Replay `txn`'s staged ops into `target`, remapping them for this
    /// handle's branch view (see [`Self::with_branch`]) so the whole
    /// transaction — including any branch tombstones its deletes imply —
    /// commits as one atomic `WriteBatch`. For [`BranchId::ROOT`] this is
    /// exactly [`Transaction::apply_to`]. For a non-root branch:
    ///
    /// - `put` writes the branch key as staged (callers build keys with the
    ///   matching [`KeyCodec::for_branch`] codec);
    /// - `delete` of a scoped key becomes (a) a real delete when the key
    ///   exists in the branch's own range, (b) a branch-tombstone write when
    ///   it exists only in the parent, (c) a no-op when it exists in neither;
    /// - `delete` of an unscoped (volume-level) key stays a plain delete.
    pub async fn apply_transaction(
        &self,
        txn: Transaction,
        target: &mut WriteBatch,
    ) -> Result<()> {
        if self.branch.is_root() {
            txn.apply_to(target);
            return Ok(());
        }
        self.check_internal_lease().await?;
        self.check_closing()?;
        let codec = KeyCodec::for_branch(self.branch);
        for op in txn.into_ops() {
            match op {
                TxOp::Put(k, v) => target.put_bytes(k, v),
                TxOp::Delete(k) => self.stage_branch_delete(&codec, k, target).await?,
            }
        }
        Ok(())
    }

    /// Map one staged delete for a non-root branch (see
    /// [`Self::apply_transaction`]). Reads are raw point lookups: the
    /// branch-owned check must not fall back to the parent, or every
    /// parent-visible key would look branch-owned.
    async fn stage_branch_delete(
        &self,
        codec: &KeyCodec,
        key: Bytes,
        target: &mut WriteBatch,
    ) -> Result<()> {
        if !matches!(codec.peek_kind(&key), Some(kind) if kind.is_scoped()) {
            target.delete(key);
            return Ok(());
        }
        if self.raw_get(&key, DurabilityLevel::Memory).await?.is_some() {
            // (a) Branch-owned: a real delete of the branch key.
            target.delete(key);
            return Ok(());
        }
        let Some(parent) = codec.strip_branch(&key) else {
            return Err(anyhow::anyhow!(
                "branch-scoped delete of a key that is not a well-formed key of this branch"
            ));
        };
        if self.raw_get(&parent, DurabilityLevel::Memory).await?.is_some() {
            // (b) Parent-owned: shadow it with a branch tombstone; nothing is
            // deleted. `strip_branch` already validated the key's shape.
            let tombstone = codec
                .branch_tombstone_key(&key)
                .expect("strip_branch accepted this scoped key");
            target.put_bytes(tombstone, KeyCodec::branch_tombstone_value());
        }
        // (c) Absent in both scopes: no-op.
        Ok(())
    }

    /// Write a single key. Branch-view handles ([`Self::with_branch`]) write
    /// the key exactly as given: scoped keys arrive already branch-qualified
    /// from the caller's [`KeyCodec::for_branch`] codec, so a branch's `put`
    /// lands in its own key range and never touches the parent's.
    pub async fn put_with_options(
        &self,
        key: &Bytes,
        value: &[u8],
        put_options: &PutOptions,
        write_options: &WriteOptions,
    ) -> Result<()> {
        if self.is_read_only() {
            return Err(FsError::ReadOnlyFilesystem.into());
        }
        let _flush_guard = self.flush_barrier.read().await;
        self.check_closing()?;

        match &self.inner {
            SlateDbHandle::ReadWrite(db) => {
                if let Err(e) = db
                    .put_with_options(key, value, put_options, write_options)
                    .await
                {
                    exit_on_write_error(e);
                }
            }
            SlateDbHandle::ReadOnly(_) => unreachable!(),
        }

        Ok(())
    }

    pub async fn flush(&self) -> Result<()> {
        if self.is_read_only() {
            return Err(FsError::ReadOnlyFilesystem.into());
        }
        // Recoverable suspension pauses durability work; terminal revocation
        // rejects it.
        self.check_internal_lease().await?;
        self.check_closing()?;

        match &self.inner {
            SlateDbHandle::ReadWrite(db) => {
                if let Err(e) = db.flush().await {
                    exit_on_write_error(e);
                }
            }
            SlateDbHandle::ReadOnly(_) => unreachable!(),
        }
        Ok(())
    }

    pub fn slatedb_metrics(&self) -> Option<Arc<DefaultMetricsRecorder>> {
        self.metrics_recorder.clone()
    }

    /// Concurrency cap for the warm fan-out. Each task issues at most a couple of
    /// object-store GETs (filter + index), so a small cap drains thousands of
    /// SSTs quickly without crowding the serving path off the object store.
    const WARM_CONCURRENCY: usize = 16;

    /// Cache targets for a metadata warm. Filters + index are always warmed
    /// (small, bounded, and they gate every point lookup); `warm_data` adds the
    /// data blocks (the whole `meta` segment is metadata, so the full range).
    fn warm_targets(warm_data: bool) -> Vec<CacheTarget> {
        let mut targets = vec![CacheTarget::Filters, CacheTarget::Index];
        if warm_data {
            targets.push(CacheTarget::data::<&[u8], _>(..));
        }
        targets
    }

    /// One-shot warm of the whole metadata segment, returning what it touched.
    /// Production keeps the cache warm with [`warm_metadata_watch`](Self::warm_metadata_watch)
    /// instead (which also re-warms after compactions); this primitive exists for
    /// tests that assert the cache effect of a single warm.
    ///
    /// Best-effort and side-effect-only: a per-SST failure is counted, not
    /// propagated. A no-op on a volume with no metadata segment yet (a fresh DB
    /// before its first flush) or without a block cache. Not lease-gated.
    #[cfg(test)]
    pub async fn warm_metadata(&self, warm_data: bool) -> WarmStats {
        let targets = Self::warm_targets(warm_data);

        // Snapshot the metadata segment's SST ids; the manifest borrow ends with
        // the collect (ids are `Copy`), before the fan-out.
        let manifest = match &self.inner {
            SlateDbHandle::ReadWrite(db) => db.manifest(),
            SlateDbHandle::ReadOnly(reader) => reader.load().manifest(),
        };
        let Some(segment) = manifest.segment(crate::fs::key_codec::META_DOMAIN) else {
            tracing::info!("metadata cache warm skipped: no metadata segment on this volume");
            return WarmStats::default();
        };
        let ids: Vec<_> = segment
            .l0()
            .iter()
            .chain(
                segment
                    .compacted()
                    .iter()
                    .flat_map(|run| run.sst_views().iter()),
            )
            .map(|view| view.sst.id)
            .collect();

        let total = ids.len();
        let failed = futures::stream::iter(ids)
            .map(|id| {
                let targets = &targets;
                async move {
                    match &self.inner {
                        SlateDbHandle::ReadWrite(db) => db.warm_sst(id, targets).await,
                        SlateDbHandle::ReadOnly(reader) => {
                            reader.load_full().warm_sst(id, targets).await
                        }
                    }
                }
            })
            .buffer_unordered(Self::WARM_CONCURRENCY)
            .filter(|r| std::future::ready(r.is_err()))
            .count()
            .await;

        WarmStats {
            ssts: total,
            failed,
        }
    }

    /// A fresh status subscription (manifest + durability updates) for the
    /// read-write handle, or `None` for a read-only open (which has no block
    /// cache to keep warm).
    pub fn subscribe_status(&self) -> Option<tokio::sync::watch::Receiver<slatedb::DbStatus>> {
        match &self.inner {
            SlateDbHandle::ReadWrite(db) => Some(db.subscribe()),
            SlateDbHandle::ReadOnly(_) => None,
        }
    }

    /// Keep the metadata block cache warm for the life of the process.
    ///
    /// Warms the meta segment once up front, then re-warms newly-appeared SSTs
    /// whenever the manifest changes. Compactions (and L0 flushes) replace meta
    /// SSTs with cold ones, and the compactor shares no block cache, so without
    /// this the startup warm decays and metadata reads pay the cold object-store
    /// cost again right after every compaction. The set of already-warmed ids is
    /// diffed against each new manifest, so each SST is warmed exactly once and an
    /// unchanged manifest is a no-op. Runs until `shutdown`.
    pub async fn warm_metadata_watch(
        &self,
        warm_data: bool,
        mut status: tokio::sync::watch::Receiver<slatedb::DbStatus>,
        shutdown: tokio_util::sync::CancellationToken,
    ) {
        let targets = Self::warm_targets(warm_data);
        let mut tracker = WarmTracker::new();
        let mut first = true;

        loop {
            let new_ids: Vec<_> = {
                let snapshot = status.borrow();
                let manifest = &snapshot.current_manifest;
                match manifest.segment(crate::fs::key_codec::META_DOMAIN) {
                    None => Vec::new(),
                    Some(segment) => {
                        let live = segment
                            .l0()
                            .iter()
                            .chain(
                                segment
                                    .compacted()
                                    .iter()
                                    .flat_map(|run| run.sst_views().iter()),
                            )
                            .map(|view| view.sst.id);
                        tracker.plan(manifest.id(), live)
                    }
                }
            };

            if !new_ids.is_empty() {
                let count = new_ids.len();
                let failed = futures::stream::iter(new_ids)
                    .map(|id| {
                        let targets = &targets;
                        async move {
                            match &self.inner {
                                SlateDbHandle::ReadWrite(db) => db.warm_sst(id, targets).await,
                                SlateDbHandle::ReadOnly(reader) => {
                                    reader.load_full().warm_sst(id, targets).await
                                }
                            }
                        }
                    })
                    .buffer_unordered(Self::WARM_CONCURRENCY)
                    .filter(|r| std::future::ready(r.is_err()))
                    .count()
                    .await;
                // The first pass is the startup warm (whole segment); later passes
                // are the post-compaction / post-flush deltas.
                if first {
                    tracing::info!("metadata cache warm: {count} SSTs ({failed} failed)");
                } else {
                    tracing::debug!("metadata cache re-warm: {count} new SSTs ({failed} failed)");
                }
            }
            first = false;

            tokio::select! {
                _ = shutdown.cancelled() => break,
                changed = status.changed() => {
                    if changed.is_err() {
                        break; // sender dropped: the db is closing
                    }
                }
            }
        }
    }

    pub async fn close(&self) -> Result<()> {
        self.mark_closing();
        match &self.inner {
            SlateDbHandle::ReadWrite(db) => {
                if let Err(e) = db.close().await {
                    exit_on_write_error(e);
                }
            }
            SlateDbHandle::ReadOnly(reader_swap) => {
                let reader = reader_swap.load();
                reader.close().await?
            }
        }
        Ok(())
    }
}

/// The scan forwarder's input
trait ScanSource: Send {
    fn next_kv(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<(Bytes, Bytes)>>> + Send;
}

impl ScanSource for slatedb::DbIterator {
    async fn next_kv(&mut self) -> Result<Option<(Bytes, Bytes)>> {
        Ok(self.next().await?.map(|kv| (kv.key, kv.value)))
    }
}

/// Pump `iter` into `tx` until end-of-range, a dropped consumer, or an error.
async fn forward_scan<S: ScanSource>(
    mut iter: S,
    tx: tokio::sync::mpsc::Sender<Result<(Bytes, Bytes)>>,
) {
    loop {
        match iter.next_kv().await {
            Ok(Some(kv)) => {
                if tx.send(Ok(kv)).await.is_err() {
                    break; // consumer dropped the stream
                }
            }
            Ok(None) => break,
            Err(e) => {
                let _ = tx.send(Err(e)).await;
                break;
            }
        }
    }
}

/// A byte range from explicit owned bounds, for scans whose bounds are
/// computed rather than caller-supplied (the merged branch view's
/// parent-range and tombstone-range scans).
struct BoundsRange(Bound<Bytes>, Bound<Bytes>);

impl slatedb::ByteRangeBounds for BoundsRange {
    fn start_bound(&self) -> Bound<&[u8]> {
        self.0.as_ref().map(|b| b.as_ref())
    }

    fn end_bound(&self) -> Bound<&[u8]> {
        self.1.as_ref().map(|b| b.as_ref())
    }
}

/// Position of a scan bound relative to a branch's slice of one scoped kind
/// (`kind || branch` up to `kind || branch + 1`).
enum SliceBound {
    /// At or below the slice's first key.
    Before,
    /// Inside the slice; the same bound with the branch bytes stripped (the
    /// parent-view bound).
    Within(Bytes),
    /// At or past the first key past the slice.
    After,
}

/// Classify a scan bound against `codec`'s branch slice of `kind`. `None`
/// when the bound is not in `kind`'s domain+kind space at all.
fn slice_bound(codec: &KeyCodec, kind: KeyPrefix, key: &[u8]) -> Option<SliceBound> {
    if codec.peek_kind(key) != Some(kind) {
        return None;
    }
    let kind_off = codec.kind_offset(kind);
    let rb = key.get(kind_off + 1..)?;
    let branch_be = codec.branch().0.to_be_bytes();
    if rb.len() < std::mem::size_of::<u32>() {
        // A strict prefix of the branch-id bytes: it sorts with the branch id
        // they prefix — at the slice start or past it.
        return Some(if rb <= &branch_be[..rb.len()] {
            SliceBound::Before
        } else {
            SliceBound::After
        });
    }
    let embedded = u32::from_be_bytes(rb[..std::mem::size_of::<u32>()].try_into().ok()?);
    Some(match embedded.cmp(&codec.branch().0) {
        std::cmp::Ordering::Less => SliceBound::Before,
        std::cmp::Ordering::Equal => SliceBound::Within(codec.strip_branch(key)?),
        std::cmp::Ordering::Greater => SliceBound::After,
    })
}

/// Map a branch-view scan range to the equivalent parent-view bounds (see
/// [`Db::scan`]). `Some` only when both bounds lie within this branch's
/// slice of one scoped kind — the shape every range built from a
/// [`KeyCodec::for_branch`] key or prefix range has. `None` (root handle,
/// unscoped kind, whole-keyspace or cross-kind range) falls back to the raw
/// stored view, exactly as before branches existed.
fn branch_parent_range(
    codec: &KeyCodec,
    start: Bound<&[u8]>,
    end: Bound<&[u8]>,
) -> Option<(Bound<Bytes>, Bound<Bytes>)> {
    if codec.branch().is_root() {
        return None;
    }
    let (start_key, start_included) = match start {
        Bound::Included(k) => (k, true),
        Bound::Excluded(k) => (k, false),
        // Whole-keyspace scans span kinds; merging them is ill-defined.
        Bound::Unbounded => return None,
    };
    let kind = codec.peek_kind(start_key)?;
    if !kind.is_scoped() {
        return None;
    }
    let kind_off = codec.kind_offset(kind);
    let kind_start = Bytes::copy_from_slice(&start_key[..kind_off + 1]);
    let mut kind_end = kind_start.to_vec();
    *kind_end.last_mut().expect("kind byte present") = u8::from(kind) + 1;
    let kind_end = Bytes::from(kind_end);

    let parent_start = match slice_bound(codec, kind, start_key)? {
        SliceBound::Before => Bound::Included(kind_start.clone()),
        SliceBound::Within(parent) if start_included => Bound::Included(parent),
        SliceBound::Within(parent) => Bound::Excluded(parent),
        // Past the slice: the merged range is empty.
        SliceBound::After => Bound::Included(kind_end.clone()),
    };
    let parent_end = match end {
        Bound::Included(k) => Some((k, true)),
        Bound::Excluded(k) => Some((k, false)),
        // An unbounded end spans into later kinds; keep the raw view.
        Bound::Unbounded => return None,
    }
    .and_then(|(end_key, included)| {
        let flavor = |b: Bytes| {
            if included {
                Bound::Included(b)
            } else {
                Bound::Excluded(b)
            }
        };
        // `domain || kind + 1` is the exclusive end of the kind — possibly an
        // unscoped or unknown kind byte, so compare bytes, not kinds.
        if end_key.len() == kind_off + 1
            && end_key.starts_with(&kind_start[..kind_off])
            && end_key[kind_off] == u8::from(kind) + 1
        {
            return Some(flavor(Bytes::copy_from_slice(end_key)));
        }
        match slice_bound(codec, kind, end_key)? {
            // At or below the slice start: the merged range is empty.
            SliceBound::Before => Some(Bound::Excluded(kind_start)),
            SliceBound::Within(parent) => Some(flavor(parent)),
            SliceBound::After => Some(Bound::Excluded(kind_end)),
        }
    })?;
    Some((parent_start, parent_end))
}

/// Collect the shadowed suffixes of every branch-tombstone row in `iter`
/// (already restricted to the relevant range), ascending — the suppression
/// set of a merged branch scan.
async fn collect_tombstone_suffixes(
    codec: &KeyCodec,
    iter: &mut slatedb::DbIterator,
) -> Result<Vec<Bytes>> {
    let mut suffixes = Vec::new();
    while let Some(kv) = iter.next().await? {
        let (_, suffix) = codec
            .parse_branch_tombstone_key(&kv.key)
            .ok_or_else(|| anyhow::anyhow!("tombstone range held a malformed row"))?;
        suffixes.push(Bytes::copy_from_slice(suffix));
    }
    Ok(suffixes)
}

/// Streaming merge of the two scopes that make up a branch's view of one
/// scanned range (see [`Db::scan`]): the parent's rows (root layout) and the
/// branch's own rows (branch layout), plus the branch's prefetched tombstone
/// set for the range. Both inputs are already ordered by their keys, hence
/// by suffix (each has a constant `domain || kind [|| branch]` header), so
/// this is a classic two-way merge: it holds at most one row per side and
/// never buffers a scope.
///
/// The parent scan is over the shared kind range, so it also covers every
/// branch's slice (branch ids sort as key payload inside the kind): rows
/// carrying a non-zero branch field are dropped from the parent side — the
/// branch stream supplies this branch's own rows, and siblings' rows are
/// never this branch's to merge. Root rows of every scoped kind begin their
/// suffix with a big-endian id or timestamp whose high 4 bytes are zero
/// (inode ids below 2^32, timestamps before 2106), so a non-zero field can
/// only be a branch id.
///
/// Output ordering is by key suffix, parent and branch suffixes compared
/// lexicographically; equal suffixes collapse to the branch row
/// (nearest-writer-wins). Parent rows whose suffix is tombstoned are
/// suppressed. Surviving parent rows are re-keyed into the branch layout so
/// every emitted key parses under the branch codec — merged-scan consumers
/// (orphan/tombstone walks, extent scans, dir-entry listings) only ever hold
/// the branch codec.
struct BranchMerge<P, B> {
    parent: P,
    branch: B,
    parent_head: Option<(Bytes, Bytes)>,
    branch_head: Option<(Bytes, Bytes)>,
    parent_done: bool,
    branch_done: bool,
    /// Tombstone-shadowed suffixes, ascending (scan order); binary-searched
    /// per surviving parent row.
    tombstones: Vec<Bytes>,
    codec: KeyCodec,
    root_codec: KeyCodec,
    /// Latch an iterator error: report it once, then end the stream instead
    /// of re-polling a failed iterator (see `forward_scan`'s contract).
    failed: bool,
}

impl<P: ScanSource, B: ScanSource> BranchMerge<P, B> {
    fn new(parent: P, branch: B, tombstones: Vec<Bytes>, codec: KeyCodec) -> Self {
        debug_assert!(
            tombstones.is_sorted(),
            "tombstone suffixes arrive in scan order"
        );
        Self {
            parent,
            branch,
            parent_head: None,
            branch_head: None,
            parent_done: false,
            branch_done: false,
            tombstones,
            codec,
            root_codec: KeyCodec::new(),
            failed: false,
        }
    }

    fn parent_suffix<'a>(&self, key: &'a [u8]) -> Result<&'a [u8]> {
        self.root_codec
            .scoped_suffix(key)
            .ok_or_else(|| anyhow::anyhow!("parent-scope row outside the merged kind"))
    }

    fn branch_suffix<'a>(&self, key: &'a [u8]) -> Result<&'a [u8]> {
        self.codec
            .scoped_suffix(key)
            .ok_or_else(|| anyhow::anyhow!("branch-scope row outside the merged kind"))
    }

    fn is_tombstoned(&self, suffix: &[u8]) -> bool {
        self.tombstones
            .binary_search_by(|t| t.as_ref().cmp(suffix))
            .is_ok()
    }

    /// Whether a row from the parent-scope scan is actually branch-qualified
    /// (carries a non-zero branch field) and therefore not a parent row at
    /// all (see the struct comment).
    fn is_branch_qualified(&self, key: &[u8]) -> Result<bool> {
        let suffix = self.parent_suffix(key)?;
        let Some(field) = suffix.get(..std::mem::size_of::<u32>()) else {
            return Ok(false);
        };
        Ok(u32::from_be_bytes(field.try_into().expect("4 bytes")) != 0)
    }

    async fn next_merged(&mut self) -> Result<Option<(Bytes, Bytes)>> {
        loop {
            if !self.parent_done && self.parent_head.is_none() {
                self.parent_head = self.parent.next_kv().await?;
                self.parent_done = self.parent_head.is_none();
                if let Some((key, _)) = &self.parent_head
                    && self.is_branch_qualified(key)?
                {
                    self.parent_head = None;
                    continue;
                }
            }
            if !self.branch_done && self.branch_head.is_none() {
                self.branch_head = self.branch.next_kv().await?;
                self.branch_done = self.branch_head.is_none();
            }
            let take_parent = match (&self.parent_head, &self.branch_head) {
                (None, None) => return Ok(None),
                (None, Some(_)) => return Ok(self.branch_head.take()),
                (Some(_), None) => true,
                (Some((parent_key, _)), Some((branch_key, _))) => {
                    match self
                        .parent_suffix(parent_key)?
                        .cmp(self.branch_suffix(branch_key)?)
                    {
                        std::cmp::Ordering::Less => true,
                        std::cmp::Ordering::Greater => return Ok(self.branch_head.take()),
                        // Same logical key: the branch's row shadows the
                        // parent's (nearest-writer-wins).
                        std::cmp::Ordering::Equal => {
                            self.parent_head = None;
                            return Ok(self.branch_head.take());
                        }
                    }
                }
            };
            debug_assert!(take_parent);
            let (key, value) = self.parent_head.take().expect("parent head checked above");
            if self.is_tombstoned(self.parent_suffix(&key)?) {
                continue;
            }
            let key = self
                .codec
                .adopt_parent_key(&key)
                .ok_or_else(|| anyhow::anyhow!("parent-scope row outside the merged kind"))?;
            return Ok(Some((key, value)));
        }
    }
}

impl<P: ScanSource, B: ScanSource> ScanSource for BranchMerge<P, B> {
    async fn next_kv(&mut self) -> Result<Option<(Bytes, Bytes)>> {
        if self.failed {
            return Ok(None);
        }
        match self.next_merged().await {
            Err(e) => {
                self.failed = true;
                Err(e)
            }
            ok => ok,
        }
    }
}

#[cfg(test)]
mod warm_tracker_tests {
    use super::WarmTracker;

    fn sorted(mut v: Vec<u64>) -> Vec<u64> {
        v.sort_unstable();
        v
    }

    // The tracker warms each SST once across the manifest changes a long-running
    // leader sees: the initial open, L0 flushes (add an SST), and compactions
    // (retire several SSTs, add new ones). Only genuinely-new ids are warmed.
    #[test]
    fn warms_new_ssts_once_across_flushes_and_compactions() {
        let mut t = WarmTracker::<u64>::new();

        // Initial manifest: warm everything.
        assert_eq!(
            sorted(t.plan(1, [10, 11, 12].into_iter())),
            vec![10, 11, 12]
        );

        // A status notification that didn't bump the manifest id (e.g. a
        // durability advance): nothing to do.
        assert!(t.plan(1, [10, 11, 12].into_iter()).is_empty());

        // An L0 flush adds one SST: only the new one is warmed.
        assert_eq!(sorted(t.plan(2, [10, 11, 12, 13].into_iter())), vec![13]);

        // A compaction retires 11, 12, 13 into a new run 20 and keeps 10: only the
        // compaction output is cold, so only it is warmed.
        assert_eq!(sorted(t.plan(3, [10, 20].into_iter())), vec![20]);

        // Re-presenting the same set after another change warms nothing.
        assert!(t.plan(4, [10, 20].into_iter()).is_empty());

        // An id that was retired and reappears is treated as new (its blocks are
        // no longer guaranteed cached), so it warms again.
        assert_eq!(sorted(t.plan(5, [10, 20, 11].into_iter())), vec![11]);
    }
}

#[cfg(test)]
mod lease_gate_tests {
    use super::*;
    use crate::replication::Lease;
    use std::time::Duration;

    async fn open_inner() -> Arc<slatedb::Db> {
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        Arc::new(
            slatedb::DbBuilder::new(slatedb::object_store::path::Path::from("data"), store)
                .build()
                .await
                .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lease_gates_reads_and_writes() {
        let lease = Lease::new();
        let db = Db::new(open_inner().await, None).with_lease(lease.clone());
        let key = Bytes::from_static(b"k");

        // Invalid lease (never renewed): reads are refused.
        assert!(
            db.get_bytes(&key).await.is_err(),
            "read must be refused while the lease is invalid"
        );

        // Refused by the gate before reaching the db, so no write is attempted.
        let mut batch = WriteBatch::new();
        batch.put_bytes(key.clone(), Bytes::from_static(b"v"));
        assert!(
            db.write_with_options(batch, &WriteOptions::default())
                .await
                .is_err(),
            "write must be refused while the lease is invalid"
        );

        // Valid lease: reads serve (the key is absent, so None).
        lease.renew(Duration::from_millis(500));
        assert!(db.get_bytes(&key).await.unwrap().is_none());

        // Admitted local apply waits through recoverable suspension.
        lease.suspend_for_tests();
        assert!(!db.permits_successful_response());
        assert!(db.get_bytes(&key).await.is_err());
        let mut batch = WriteBatch::new();
        batch.put_bytes(key.clone(), Bytes::from_static(b"v"));
        let options = WriteOptions::default();
        let write = db.write_with_options(batch, &options);
        tokio::pin!(write);
        assert!(matches!(
            futures::poll!(write.as_mut()),
            std::task::Poll::Pending
        ));
        assert!(lease.recover_for_tests(Duration::from_millis(500)));
        write.await.unwrap();
        assert_eq!(
            db.get_bytes(&key).await.unwrap(),
            Some(Bytes::from_static(b"v"))
        );

        // Internal commit preparation waits through the same recoverable
        // suspension instead of inheriting the fail-fast serving-read gate.
        lease.suspend_for_tests();
        let internal_read = db.get_bytes_internal(&key);
        tokio::pin!(internal_read);
        assert!(matches!(
            futures::poll!(internal_read.as_mut()),
            std::task::Poll::Pending
        ));
        assert!(lease.recover_for_tests(Duration::from_millis(500)));
        assert_eq!(internal_read.await.unwrap(), Some(Bytes::from_static(b"v")));

        lease.suspend_for_tests();
        let flush = db.flush();
        tokio::pin!(flush);
        assert!(matches!(
            futures::poll!(flush.as_mut()),
            std::task::Poll::Pending
        ));
        assert!(lease.recover_for_tests(Duration::from_millis(500)));
        flush.await.unwrap();

        // Terminal revocation cannot be reversed by a later renewal.
        db.revoke_lease();
        lease.renew(Duration::from_millis(500));
        assert!(!lease.is_valid());
        assert!(
            db.get_bytes(&key).await.is_err(),
            "read must be refused after the lease is revoked"
        );
        assert!(
            db.flush().await.is_err(),
            "flush must be refused after the lease is revoked"
        );
    }
}

#[cfg(test)]
mod scan_error_tests {
    use super::*;
    use futures::StreamExt;
    use slatedb::config::PutOptions;
    use std::collections::VecDeque;

    /// A scripted [`ScanSource`]; yields its items then end-of-range.
    struct Scripted(VecDeque<Result<Option<(Bytes, Bytes)>>>);

    impl ScanSource for Scripted {
        async fn next_kv(&mut self) -> Result<Option<(Bytes, Bytes)>> {
            self.0.pop_front().unwrap_or(Ok(None))
        }
    }

    // A mid-scan error must surface as an Err item on the stream, never as a
    // silently truncated clean end: read_range would serve the missing extents
    // as fabricated zeros, and delete_range would skip their segment debits (a
    // permanent leak).
    #[tokio::test]
    async fn forwarder_delivers_a_mid_scan_error_instead_of_truncating() {
        let kv = |k: &str| {
            (
                Bytes::copy_from_slice(k.as_bytes()),
                Bytes::from_static(b"v"),
            )
        };
        let script = VecDeque::from([
            Ok(Some(kv("a"))),
            Ok(Some(kv("b"))),
            Err(anyhow::anyhow!("mid-scan read failure")),
            // Past the error: must never be served.
            Ok(Some(kv("c"))),
        ]);
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tokio::spawn(forward_scan(Scripted(script), tx));

        let items: Vec<_> = tokio_stream::wrappers::ReceiverStream::new(rx)
            .collect()
            .await;
        assert!(
            items.len() == 3 && items[2].is_err(),
            "two rows then the delivered error, nothing after; got {} item(s), \
             last ok={:?}",
            items.len(),
            items.last().map(|i| i.is_ok())
        );
        assert!(items[0].is_ok() && items[1].is_ok());
    }

    // The forwarder streams a range wider than the eager prefetch window
    // (max_fetch_tasks x read_ahead_bytes) to completion over a cold reopen —
    // the paged path production scans rarely cross.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scan_streams_a_multi_window_range_completely() {
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let path = slatedb::object_store::path::Path::from("data");
        const ROWS: usize = 8192;
        let value = vec![7u8; 4096];
        let settings = || slatedb::config::Settings {
            compactor_options: None,
            ..Default::default()
        };
        {
            let db = Db::new(
                Arc::new(
                    slatedb::DbBuilder::new(path.clone(), store.clone())
                        .with_settings(settings())
                        .build()
                        .await
                        .unwrap(),
                ),
                None,
            );
            for i in 0..ROWS {
                let key = Bytes::from(format!("k{i:05}"));
                db.put_with_options(
                    &key,
                    &value,
                    &PutOptions::default(),
                    &WriteOptions::default(),
                )
                .await
                .unwrap();
            }
            db.flush().await.unwrap();
            db.close().await.unwrap();
        }
        let db = Db::new(
            Arc::new(
                slatedb::DbBuilder::new(path, store)
                    .with_settings(settings())
                    .build()
                    .await
                    .unwrap(),
            ),
            None,
        );
        let mut stream = db
            .scan(Bytes::new()..Bytes::from_static(&[0xff]))
            .await
            .unwrap();
        let rows = tokio::time::timeout(std::time::Duration::from_secs(60), async {
            let mut rows = 0usize;
            while let Some(item) = stream.next().await {
                item.unwrap();
                rows += 1;
            }
            rows
        })
        .await
        .expect("scan drain hung");
        assert_eq!(rows, ROWS);
    }
}

#[cfg(test)]
mod branch_tests {
    //! Branch-view behavior of [`Db::with_branch`]: fallback reads, branch
    //! tombstones, branch-scoped transaction application, and the branch
    //! registry. All against an in-memory SlateDB, mirroring the style of
    //! the lease-gate tests above.
    use super::*;
    use crate::fs::key_codec::{BranchId, KeyCodec};
    use slatedb::config::PutOptions;

    const BRANCH: BranchId = BranchId(1);

    async fn open_inner() -> Arc<slatedb::Db> {
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        Arc::new(
            slatedb::DbBuilder::new(slatedb::object_store::path::Path::from("data"), store)
                .build()
                .await
                .unwrap(),
        )
    }

    /// A root handle and a branch handle over the same underlying database.
    async fn open_pair() -> (Db, Db) {
        let inner = open_inner().await;
        let root = Db::new(inner.clone(), None);
        let branch = Db::new(inner, None).with_branch(BRANCH);
        (root, branch)
    }

    fn root_codec() -> KeyCodec {
        KeyCodec::new()
    }

    fn branch_codec() -> KeyCodec {
        KeyCodec::for_branch(BRANCH)
    }

    async fn put(db: &Db, key: &Bytes, value: &[u8]) {
        db.put_with_options(key, value, &PutOptions::default(), &WriteOptions::default())
            .await
            .unwrap();
    }

    /// Stage `txn` through the branch view and commit it as one WriteBatch.
    async fn commit(db: &Db, txn: Transaction) {
        let mut batch = WriteBatch::new();
        db.apply_transaction(txn, &mut batch).await.unwrap();
        db.write_with_options(batch, &WriteOptions::default())
            .await
            .unwrap();
    }

    fn value(bytes: &[u8]) -> Option<Bytes> {
        Some(Bytes::copy_from_slice(bytes))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn point_reads_resolve_branch_then_tombstone_then_parent() {
        let (root, branch) = open_pair().await;
        let root_key: Bytes = root_codec().inode_key(5).into();
        let branch_key: Bytes = branch_codec().inode_key(5).into();

        // Parent-written, branch-untouched: the branch falls back to the
        // parent key, at both durability levels.
        put(&root, &root_key, b"parent").await;
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), value(b"parent"));
        root.flush().await.unwrap();
        assert_eq!(
            branch.get_bytes_durable(&branch_key).await.unwrap(),
            value(b"parent")
        );

        // A branch write shadows the parent without touching it.
        put(&branch, &branch_key, b"branch").await;
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), value(b"branch"));
        assert_eq!(root.get_bytes(&root_key).await.unwrap(), value(b"parent"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn branch_writes_are_invisible_to_the_parent() {
        let (root, branch) = open_pair().await;
        let root_key: Bytes = root_codec().inode_key(6).into();
        let branch_key: Bytes = branch_codec().inode_key(6).into();

        put(&branch, &branch_key, b"branch-only").await;
        assert_eq!(
            branch.get_bytes(&branch_key).await.unwrap(),
            value(b"branch-only")
        );
        // The parent's own key space is untouched: no value materialized
        // under the root key.
        assert_eq!(root.get_bytes(&root_key).await.unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn branch_delete_of_parent_key_shadows_with_a_tombstone() {
        let (root, branch) = open_pair().await;
        let codec = branch_codec();
        let root_key: Bytes = root_codec().inode_key(7).into();
        let branch_key: Bytes = codec.inode_key(7).into();
        let tombstone = codec.branch_tombstone_key(&branch_key).unwrap();

        put(&root, &root_key, b"parent").await;
        let mut txn = Transaction::new();
        txn.delete_bytes(&branch_key);
        commit(&branch, txn).await;

        // Hidden from the branch only; the parent's row is untouched and the
        // tombstone row — not a delete — is what landed in the batch.
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), None);
        assert_eq!(root.get_bytes(&root_key).await.unwrap(), value(b"parent"));
        assert!(branch.raw_get(&tombstone, DurabilityLevel::Memory).await.unwrap().is_some());

        // The tombstone keeps shadowing even if the parent later deletes its
        // own row: the branch's view is decided nearest-writer-first.
        let mut parent_delete = Transaction::new();
        parent_delete.delete_bytes(&root_key);
        commit(&root, parent_delete).await;
        assert_eq!(root.get_bytes(&root_key).await.unwrap(), None);
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn branch_delete_of_branch_owned_key_is_a_real_delete() {
        let (_root, branch) = open_pair().await;
        let codec = branch_codec();
        let branch_key: Bytes = codec.inode_key(8).into();
        let tombstone = codec.branch_tombstone_key(&branch_key).unwrap();

        put(&branch, &branch_key, b"branch").await;
        let mut txn = Transaction::new();
        txn.delete_bytes(&branch_key);
        commit(&branch, txn).await;

        assert!(branch.raw_get(&branch_key, DurabilityLevel::Memory).await.unwrap().is_none());
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), None);
        // No parent row existed, so no tombstone was needed or written.
        assert!(branch.raw_get(&tombstone, DurabilityLevel::Memory).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn branch_delete_of_absent_key_is_a_no_op() {
        let (_root, branch) = open_pair().await;
        let codec = branch_codec();
        let branch_key: Bytes = codec.inode_key(9).into();
        let tombstone = codec.branch_tombstone_key(&branch_key).unwrap();

        // A sentinel put keeps the batch non-empty (SlateDB rejects empty
        // batches); the delete itself must stage nothing.
        let mut txn = Transaction::new();
        txn.delete_bytes(&branch_key);
        txn.put_bytes(&codec.segcount_key(1, 1), Bytes::from_static(b"s"));
        commit(&branch, txn).await;

        assert!(branch.raw_get(&tombstone, DurabilityLevel::Memory).await.unwrap().is_none());
        assert!(branch.raw_get(&branch_key, DurabilityLevel::Memory).await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_key_branch_transaction_applies_atomically_in_one_batch() {
        let (root, branch) = open_pair().await;
        let codec = branch_codec();
        let k1_root: Bytes = root_codec().inode_key(10).into();
        let k2_root: Bytes = root_codec().inode_key(11).into();
        let k1_branch: Bytes = codec.inode_key(10).into();
        let k2_branch: Bytes = codec.inode_key(11).into();
        let k3_branch: Bytes = codec.inode_key(12).into();
        let k3_root: Bytes = root_codec().inode_key(12).into();

        put(&root, &k1_root, b"p1").await;
        put(&root, &k2_root, b"p2").await;

        // One transaction, one batch: shadow k1 with a branch value, put a
        // branch-only k3, and delete k2 (parent-owned -> tombstone).
        let mut txn = Transaction::new();
        txn.put_bytes(&k1_branch, Bytes::from_static(b"b1"));
        txn.put_bytes(&k3_branch, Bytes::from_static(b"b3"));
        txn.delete_bytes(&k2_branch);
        commit(&branch, txn).await;

        assert_eq!(branch.get_bytes(&k1_branch).await.unwrap(), value(b"b1"));
        assert_eq!(branch.get_bytes(&k3_branch).await.unwrap(), value(b"b3"));
        assert_eq!(branch.get_bytes(&k2_branch).await.unwrap(), None);
        // Parent rows are byte-for-byte intact.
        assert_eq!(root.get_bytes(&k1_root).await.unwrap(), value(b"p1"));
        assert_eq!(root.get_bytes(&k2_root).await.unwrap(), value(b"p2"));
        assert_eq!(root.get_bytes(&k3_root).await.unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reput_after_tombstone_then_redelete_stays_consistent() {
        let (root, branch) = open_pair().await;
        let codec = branch_codec();
        let root_key: Bytes = root_codec().inode_key(13).into();
        let branch_key: Bytes = codec.inode_key(13).into();

        put(&root, &root_key, b"parent").await;

        // Branch deletes the parent row (tombstone), then re-creates the key
        // in its own scope: the branch key wins over the stale tombstone.
        let mut txn = Transaction::new();
        txn.delete_bytes(&branch_key);
        commit(&branch, txn).await;
        put(&branch, &branch_key, b"branch").await;
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), value(b"branch"));

        // Deleting again is a real delete of the branch-owned key; the older
        // tombstone then correctly shadows the parent row once more.
        let mut txn = Transaction::new();
        txn.delete_bytes(&branch_key);
        commit(&branch, txn).await;
        assert!(branch.raw_get(&branch_key, DurabilityLevel::Memory).await.unwrap().is_none());
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), None);
        assert_eq!(root.get_bytes(&root_key).await.unwrap(), value(b"parent"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn root_branch_transactions_delete_for_real() {
        let (root, _branch) = open_pair().await;
        let key: Bytes = root_codec().inode_key(14).into();

        put(&root, &key, b"v").await;
        let mut txn = Transaction::new();
        txn.delete_bytes(&key);
        commit(&root, txn).await;
        assert_eq!(root.get_bytes(&key).await.unwrap(), None);

        // No branch-tombstone rows anywhere in the keyspace.
        let codec = root_codec();
        let (start, end) = codec.prefix_range(crate::fs::key_codec::KeyPrefix::BranchTombstone);
        let mut stream = root.scan(start..end).await.unwrap();
        assert!(stream.next().await.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unscoped_keys_are_shared_globally_across_views() {
        let (root, branch) = open_pair().await;
        let codec = root_codec();
        let stats_key = codec.stats_shard_key(0);

        // Same global key from both views: written via the branch handle,
        // read via the root handle, and vice versa.
        put(&branch, &stats_key, b"stats").await;
        assert_eq!(root.get_bytes(&stats_key).await.unwrap(), value(b"stats"));
        let counter_key = codec.system_counter_key();
        put(&root, &counter_key, &KeyCodec::encode_u64(41)).await;
        assert_eq!(
            branch.get_bytes(&counter_key).await.unwrap(),
            value(&KeyCodec::encode_u64(41))
        );

        // Deleting an unscoped key through a branch view is a real delete,
        // not a tombstone.
        let mut txn = Transaction::new();
        txn.delete_bytes(&stats_key);
        commit(&branch, txn).await;
        assert_eq!(root.get_bytes(&stats_key).await.unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn registry_create_list_delete_with_unique_ids() {
        let (root, _branch) = open_pair().await;

        let id_a = root.create_branch("alpha").await.unwrap();
        let id_b = root.create_branch("beta").await.unwrap();
        assert_eq!(id_a, BranchId(1));
        assert_eq!(id_b, BranchId(2));

        // Duplicate names are rejected; empty names are rejected.
        assert!(root.create_branch("alpha").await.is_err());
        assert!(root.create_branch("").await.is_err());

        let listed = root.list_branches().await.unwrap();
        let names: Vec<_> = listed.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta"]);
        assert_eq!(listed[0].1.id, 1);
        assert!(listed[0].1.created_at > 0);
        assert_eq!(listed[1].1.id, 2);

        // Deletion removes the row only; re-creation gets a fresh id (ids
        // are never reused, even across deletion).
        assert!(root.delete_branch("alpha").await.unwrap());
        assert!(!root.delete_branch("alpha").await.unwrap());
        assert!(!root.delete_branch("missing").await.unwrap());
        let id_a2 = root.create_branch("alpha").await.unwrap();
        assert_eq!(id_a2, BranchId(3));

        let listed = root.list_branches().await.unwrap();
        let names: Vec<_> = listed.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta"]);
        assert_eq!(listed[0].1.id, 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fallback_reads_survive_flush_to_ssts() {
        let (root, branch) = open_pair().await;
        let root_key: Bytes = root_codec().dir_entry_key(3, b"file").into();
        let branch_key: Bytes = branch_codec().dir_entry_key(3, b"file").into();

        // Resolution must work from SSTs, not just the memtable.
        put(&root, &root_key, b"entry").await;
        root.flush().await.unwrap();
        assert_eq!(branch.get_bytes(&branch_key).await.unwrap(), value(b"entry"));
        assert_eq!(
            branch.get_bytes_durable(&branch_key).await.unwrap(),
            value(b"entry")
        );
    }

    /// Collect a scan stream, unwrapping each row.
    async fn collect<S>(stream: S) -> Vec<(Bytes, Bytes)>
    where
        S: Stream<Item = Result<(Bytes, Bytes)>> + Unpin,
    {
        let mut out = Vec::new();
        tokio::pin!(stream);
        while let Some(item) = stream.next().await {
            out.push(item.unwrap());
        }
        out
    }

    /// `(suffix, value)` pairs of a collected merged scan, suffixes under the
    /// branch codec (every emitted key must parse as this branch's).
    fn merged_suffixes(rows: &[(Bytes, Bytes)]) -> Vec<(Vec<u8>, Bytes)> {
        let codec = branch_codec();
        rows.iter()
            .map(|(k, v)| {
                (
                    codec
                        .scoped_suffix(k)
                        .unwrap_or_else(|| panic!("emitted key not in the branch layout: {k:02x?}"))
                        .to_vec(),
                    v.clone(),
                )
            })
            .collect()
    }

    fn dir_suffix(dir: u64, name: &[u8]) -> Vec<u8> {
        let mut s = dir.to_be_bytes().to_vec();
        s.extend_from_slice(name);
        s
    }

    /// Stage a branch-view delete through the transaction path (tombstone or
    /// real delete, as [`Db::apply_transaction`] decides).
    async fn branch_delete(db: &Db, key: &Bytes) {
        let mut txn = Transaction::new();
        txn.delete_bytes(key);
        commit(db, txn).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_shadows_suppresses_and_orders_by_suffix() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        // Parent entries in dir 1: a, c, e.
        for name in [b"a".as_slice(), b"c", b"e"] {
            put(&root, &rc.dir_entry_key(1, name), &[b'p', name[0]]).await;
        }
        // Branch entries: b, and a shadowing c. Branch deletes parent's a
        // (tombstone). Also prove a stale tombstone does not suppress a
        // later branch row: delete then re-create d in the branch.
        put(&branch, &bc.dir_entry_key(1, b"b"), b"bb").await;
        put(&branch, &bc.dir_entry_key(1, b"c"), b"bc").await;
        put(&root, &rc.dir_entry_key(1, b"d"), b"pd").await;
        branch_delete(&branch, &bc.dir_entry_key(1, b"a")).await;
        branch_delete(&branch, &bc.dir_entry_key(1, b"d")).await;
        put(&branch, &bc.dir_entry_key(1, b"d"), b"bd").await;

        let (start, end) = bc.prefix_range(crate::fs::key_codec::KeyPrefix::DirEntry);
        let rows = collect(branch.scan(start..end).await.unwrap()).await;
        let merged = merged_suffixes(&rows);
        assert_eq!(
            merged,
            vec![
                (dir_suffix(1, b"b"), Bytes::from_static(b"bb")),
                (dir_suffix(1, b"c"), Bytes::from_static(b"bc")),
                (dir_suffix(1, b"d"), Bytes::from_static(b"bd")),
                (dir_suffix(1, b"e"), Bytes::from_static(b"pe")),
            ],
            "a tombstoned, c shadowed, d branch-recreated, b branch-only, e parent-only"
        );
        // Tombstone rows themselves are never emitted.
        assert!(
            rows.iter()
                .all(|(k, _)| bc.peek_kind(k) != Some(crate::fs::key_codec::KeyPrefix::BranchTombstone))
        );

        // The root handle's per-directory prefix scan sees exactly the
        // parent's rows, unchanged (branch rows sort outside the root
        // layout's directory prefix).
        let root_rows = collect(
            root.scan_prefix(Bytes::from(rc.dir_entry_prefix(1)), None, 4096)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            root_rows.len(),
            4,
            "a, c, d, e — branch writes and tombstones are invisible"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_handles_empty_scopes() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();
        let kind = crate::fs::key_codec::KeyPrefix::DirEntry;

        // Both empty: no rows, no error.
        let (start, end) = bc.prefix_range(kind);
        assert!(collect(branch.scan(start..end).await.unwrap()).await.is_empty());

        // Empty branch scope: parent rows flow through, re-keyed.
        put(&root, &rc.dir_entry_key(2, b"x"), b"px").await;
        let (start, end) = bc.prefix_range(kind);
        let rows = collect(branch.scan(start..end).await.unwrap()).await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![(dir_suffix(2, b"x"), Bytes::from_static(b"px"))]
        );

        // Empty parent scope: branch rows only.
        put(&branch, &bc.dir_entry_key(3, b"y"), b"by").await;
        let prefix = Bytes::from(bc.dir_entry_prefix(3));
        let rows = collect(branch.scan_prefix(prefix, None, 4096).await.unwrap()).await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![(dir_suffix(3, b"y"), Bytes::from_static(b"by"))]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_rename_hides_old_name_and_shows_new() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        put(&root, &rc.dir_entry_key(4, b"old"), b"entry").await;
        // Branch rename: tombstone the old name, put the new name (what a
        // branch-codec unlink + add pair stages through apply_transaction).
        branch_delete(&branch, &bc.dir_entry_key(4, b"old")).await;
        put(&branch, &bc.dir_entry_key(4, b"new"), b"entry").await;

        let prefix = Bytes::from(bc.dir_entry_prefix(4));
        let rows = collect(branch.scan_prefix(prefix, None, 4096).await.unwrap()).await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![(dir_suffix(4, b"new"), Bytes::from_static(b"entry"))]
        );

        let root_prefix = Bytes::from(rc.dir_entry_prefix(4));
        let root_rows = collect(root.scan_prefix(root_prefix, None, 4096).await.unwrap()).await;
        assert_eq!(root_rows.len(), 1);
        assert_eq!(root_rows[0].0, rc.dir_entry_key(4, b"old"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_sees_parent_rows_written_after_the_branch_exists() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        put(&branch, &bc.dir_entry_key(5, b"branch"), b"b").await;
        // The parent keeps writing; fallback covers anything the branch never
        // shadowed or tombstoned.
        put(&root, &rc.dir_entry_key(5, b"parent-new"), b"p").await;

        let prefix = Bytes::from(bc.dir_entry_prefix(5));
        let rows = collect(branch.scan_prefix(prefix, None, 4096).await.unwrap()).await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![
                (dir_suffix(5, b"branch"), Bytes::from_static(b"b")),
                (dir_suffix(5, b"parent-new"), Bytes::from_static(b"p")),
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_prefix_isolates_nested_directory_prefixes() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        // Two directories' entries interleave on disk (same kind); each
        // prefix scan must merge only its own directory.
        put(&root, &rc.dir_entry_key(6, b"pa"), b"p").await;
        put(&root, &rc.dir_entry_key(7, b"px"), b"p").await;
        put(&branch, &bc.dir_entry_key(6, b"bb"), b"b").await;
        put(&branch, &bc.dir_entry_key(7, b"by"), b"b").await;
        // Tombstones are per-suffix: deleting dir-6's "gone" must not touch
        // dir-7's same-named entry.
        put(&root, &rc.dir_entry_key(6, b"gone"), b"p").await;
        put(&root, &rc.dir_entry_key(7, b"gone"), b"p").await;
        branch_delete(&branch, &bc.dir_entry_key(6, b"gone")).await;

        let rows = collect(
            branch
                .scan_prefix(Bytes::from(bc.dir_entry_prefix(6)), None, 4096)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![
                (dir_suffix(6, b"bb"), Bytes::from_static(b"b")),
                (dir_suffix(6, b"pa"), Bytes::from_static(b"p")),
            ]
        );
        let rows = collect(
            branch
                .scan_prefix(Bytes::from(bc.dir_entry_prefix(7)), None, 4096)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![
                (dir_suffix(7, b"by"), Bytes::from_static(b"b")),
                (dir_suffix(7, b"gone"), Bytes::from_static(b"p")),
                (dir_suffix(7, b"px"), Bytes::from_static(b"p")),
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_prefix_resumes_from_seek_to_in_both_scopes() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        for name in [b"a".as_slice(), b"m", b"z"] {
            put(&root, &rc.dir_entry_key(8, name), b"p").await;
        }
        for name in [b"b".as_slice(), b"n"] {
            put(&branch, &bc.dir_entry_key(8, name), b"b").await;
        }

        // Resume at "m" (a parent-only row): both scopes skip below it.
        let seek_to = bc.dir_entry_key(8, b"m");
        let rows = collect(
            branch
                .scan_prefix(Bytes::from(bc.dir_entry_prefix(8)), Some(seek_to), 4096)
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![
                (dir_suffix(8, b"m"), Bytes::from_static(b"p")),
                (dir_suffix(8, b"n"), Bytes::from_static(b"b")),
                (dir_suffix(8, b"z"), Bytes::from_static(b"p")),
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_covers_extent_style_subranges() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        // Parent extents 1..=5 of inode 9; branch shadows 3 and adds 7;
        // branch deletes 4 (tombstone).
        for idx in 1..=5u64 {
            put(&root, &rc.extent_key(9, idx).into(), &[b'p', idx as u8]).await;
        }
        put(&branch, &bc.extent_key(9, 3).into(), b"b3").await;
        put(&branch, &bc.extent_key(9, 7).into(), b"b7").await;
        branch_delete(&branch, &bc.extent_key(9, 4).into()).await;

        let start: Bytes = bc.extent_key(9, 2).into();
        let end: Bytes = bc.extent_key(9, 6).into();
        let rows = collect(branch.scan(start..end).await.unwrap()).await;
        let parsed: Vec<(u64, Bytes)> = rows
            .iter()
            .map(|(k, v)| {
                (
                    bc.parse_extent_key(k)
                        .expect("merged extent keys parse under the branch codec"),
                    v.clone(),
                )
            })
            .collect();
        assert_eq!(
            parsed,
            vec![
                (2, Bytes::from_static(b"p\x02")),
                (3, Bytes::from_static(b"b3")),
                (5, Bytes::from_static(b"p\x05")),
            ],
            "in-range rows only, branch shadows 3, tombstoned 4 suppressed"
        );

        // The whole-kind scan additionally sees the branch-only extent 7.
        let (start, end) = bc.prefix_range(crate::fs::key_codec::KeyPrefix::Extent);
        let rows = collect(branch.scan(start..end).await.unwrap()).await;
        assert_eq!(rows.len(), 5);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn merged_scan_works_at_durable_level_after_flush() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        put(&root, &rc.dir_entry_key(9, b"kept"), b"p").await;
        put(&root, &rc.dir_entry_key(9, b"gone"), b"p").await;
        branch_delete(&branch, &bc.dir_entry_key(9, b"gone")).await;
        put(&branch, &bc.dir_entry_key(9, b"new"), b"b").await;
        root.flush().await.unwrap();

        let (start, end) = bc.prefix_range(crate::fs::key_codec::KeyPrefix::DirEntry);
        let rows = collect(branch.scan_durable(start..end).await.unwrap()).await;
        assert_eq!(
            merged_suffixes(&rows),
            vec![
                (dir_suffix(9, b"kept"), Bytes::from_static(b"p")),
                (dir_suffix(9, b"new"), Bytes::from_static(b"b")),
            ],
            "the merge resolves from SSTs, not just the memtable"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unscoped_kind_scans_are_unmerged_on_branch_handles() {
        let (root, branch) = open_pair().await;
        let rc = root_codec();
        let bc = branch_codec();

        // An unscoped kind is volume-global: both handles scan the same raw
        // rows (no merge, no re-keying).
        put(&branch, &bc.stats_shard_key(0), b"stats").await;
        let (start, end) = bc.prefix_range(crate::fs::key_codec::KeyPrefix::Stats);
        let rows = collect(branch.scan(start..end).await.unwrap()).await;
        assert_eq!(rows, vec![(rc.stats_shard_key(0), Bytes::from_static(b"stats"))]);

        // The branch-tombstone kind itself is unscoped: cleanup tooling can
        // enumerate tombstone rows raw, even on a branch handle.
        put(&root, &rc.dir_entry_key(10, b"x"), b"p").await;
        branch_delete(&branch, &bc.dir_entry_key(10, b"x")).await;
        let (start, end) = bc.prefix_range(crate::fs::key_codec::KeyPrefix::BranchTombstone);
        let rows = collect(branch.scan(start..end).await.unwrap()).await;
        assert_eq!(rows.len(), 1);
        assert_eq!(
            bc.parse_branch_tombstone_key(&rows[0].0).unwrap().0,
            crate::fs::key_codec::KeyPrefix::DirEntry
        );
    }
}

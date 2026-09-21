//! Writable forks: branch a running volume's state into an independent,
//! writable clone that shares all of the parent's existing objects.
//!
//! Fork creation is two-phase — `zerofs fork create` is a cheap pointer;
//! the fork's state materializes on its first mount:
//!
//! **Phase 1 — registration** ([`ForkManager::create_fork`] with
//! `barrier = false`, the default; ~50ms):
//!
//! 1. A durable checkpoint pins the branch point on the parent's *current
//!    durable manifest*, taken directly on the parent database
//!    (`CheckpointScope::Durable`) WITHOUT the seal+flush barrier. The branch
//!    point can therefore lag HEAD by up to the flush interval — the
//!    documented trade for a cheap create (`--barrier` forks exactly now).
//! 2. A pending-materialization record is written to
//!    `<fork db path>/.zerofs_fork_pending.json` (see [`PendingFork`]), and a
//!    registry entry goes into the parent's LSM, in that order. No slatedb
//!    clone, no key copy, no fork database open, no parent epoch bump.
//!
//! **Phase 2 — materialization** (automatic, at the fork's first open;
//! see [`materialize_fork_if_pending`], hooked into startup before the
//! fork's database opens):
//!
//! 1. SlateDB's clone builds a new, writable database at the fork's db path
//!    from the pinned checkpoint, referencing the parent's SSTs as
//!    `external_ssts` — a shallow, O(manifest) copy-on-write branch. The
//!    clone pins its source on the parent's manifest, which also pauses
//!    reclamation of the parent's referenced segments (same protection a
//!    persistent checkpoint gives).
//! 2. The wrapped encryption key is copied to the fork's db path (keys are
//!    per-path), and the fork's freshly cloned database is opened once to
//!    write its [`ForkInfo`] lineage record into its own LSM. The pending
//!    marker is deleted last, so every step is retryable: a crash
//!    mid-materialization re-runs from the top (the clone is idempotent).
//!
//! `barrier = true` runs both phases inline at create time (the pre-lazy
//! behavior): the checkpoint is taken under the seal+flush barrier (see
//! [`CheckpointManager`]), so the fork branches exactly the state at the
//! call and is fully materialized before `create` returns — at ~1s instead
//! of ~50ms.
//!
//! Either way the fork is then served like any other volume: point a
//! `[storage] url` at the fork's db path and `zerofs run`; startup reads the
//! lineage back from the fork's LSM to route segment reads across ancestors
//! (see [`crate::segment_path_router`]).

use crate::checkpoint_manager::CheckpointManager;
use crate::db::SlateDbHandle;
use crate::fork_info::{FORKS_INFIX, ForkAncestor, ForkInfo};
use crate::fs::key_codec::KeyCodec;
use crate::key_management;
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use object_store::{ObjectStore, ObjectStoreExt};
use serde::{Deserialize, Serialize};
use slatedb::admin::Admin;
use slatedb::admin::AdminBuilder;
use slatedb::admin::CloneSourceSpec;
use slatedb::config::{CheckpointOptions, CheckpointScope, PutOptions, WriteOptions};
use slatedb::object_store::path::Path;
use std::sync::Arc;
use uuid::Uuid;

/// Object name of the pending-materialization marker under a fork's db path:
/// a lazily created fork carries one until its first open materializes it.
pub const PENDING_FORK_FILENAME: &str = ".zerofs_fork_pending.json";

/// Pending-materialization record: everything the fork's first open needs to
/// build the fork database that lazy fork creation deferred (see the module
/// docs). Written before the parent's registry entry, so a listed fork
/// always has its marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingFork {
    pub fork_info: ForkInfo,
    pub source_checkpoint_id: Uuid,
    #[allow(dead_code)] // diagnostics; deletes key off `source_checkpoint_id`
    pub source_checkpoint_name: String,
    /// Whether fork creation itself pinned the source checkpoint (the default
    /// branch point and `--at` forks); deleting a pending fork then releases
    /// it. False for a user-named `--from-checkpoint`, which the user owns.
    pub source_checkpoint_owned: bool,
}

impl PendingFork {
    fn path(fork_db_path: &Path) -> Path {
        fork_db_path.clone().join(PENDING_FORK_FILENAME)
    }

    /// Read the pending-materialization marker under `fork_db_path`; `None`
    /// when absent (non-fork volume, eager fork, or already materialized).
    async fn load(
        object_store: &Arc<dyn ObjectStore>,
        fork_db_path: &Path,
    ) -> Result<Option<Self>> {
        match object_store.get(&Self::path(fork_db_path)).await {
            Ok(result) => {
                let bytes = result.bytes().await?;
                let pending = serde_json::from_slice(&bytes)
                    .context("parsing pending-fork materialization record")?;
                Ok(Some(pending))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(anyhow!("Failed to read pending-fork marker: {}", e)),
        }
    }
}

/// The resolved branch point of a fork: the parent checkpoint the clone
/// starts from, and the fork's first writer epoch above it.
struct BranchPoint {
    checkpoint_id: Uuid,
    checkpoint_name: String,
    /// The fork's first writer epoch: the source manifest's writer epoch + 1.
    /// The fork's first writable open bumps the writer epoch it inherits from
    /// the source manifest, so its own segments start one epoch above that
    /// manifest's epoch — which is NOT necessarily the live parent's current
    /// epoch when forking an older state.
    base_epoch: u64,
    /// Whether fork creation owns the checkpoint (see
    /// [`PendingFork::source_checkpoint_owned`]).
    owned: bool,
}

pub struct ForkManager {
    db_handle: SlateDbHandle,
    parent_db_path: Path,
    object_store: Arc<dyn ObjectStore>,
    /// The volume's SST block transformer, needed to open a freshly cloned
    /// fork database for the one-time lineage write (SST blocks are encrypted
    /// with the same key on every volume of a lineage).
    block_transformer: Arc<dyn slatedb::BlockTransformer>,
    checkpoint_manager: Arc<CheckpointManager>,
    admin: Admin,
}

impl ForkManager {
    pub fn new(
        db_handle: SlateDbHandle,
        parent_db_path: Path,
        object_store: Arc<dyn ObjectStore>,
        block_transformer: Arc<dyn slatedb::BlockTransformer>,
        checkpoint_manager: Arc<CheckpointManager>,
    ) -> Self {
        let admin = AdminBuilder::new(parent_db_path.clone(), Arc::clone(&object_store)).build();
        Self {
            db_handle,
            parent_db_path,
            object_store,
            block_transformer,
            checkpoint_manager,
            admin,
        }
    }

    /// Create a writable fork named `name` from `from_checkpoint` (or from a
    /// fresh checkpoint of the current durable state when `None`). When `at`
    /// is given, the fork branches the volume as of the last manifest flushed
    /// before that timestamp (point-in-time fork); see
    /// [`ForkManager::manifest_at_time`].
    ///
    /// `barrier` selects the creation mode (see the module docs):
    ///
    /// - `false` (the CLI/RPC default) — **lazy**: registration only
    ///   (branch-point checkpoint + pending-materialization record + registry
    ///   entry, ~50ms). The clone, key copy, and lineage write defer to the
    ///   fork's first open ([`materialize_fork_if_pending`]). Without the
    ///   seal+flush barrier the branch point can lag HEAD by up to the flush
    ///   interval.
    /// - `true` — **eager**: the checkpoint is taken under the seal+flush
    ///   barrier and the fork is fully materialized before this call returns
    ///   (~1s).
    pub async fn create_fork(
        &self,
        name: &str,
        from_checkpoint: Option<String>,
        at: Option<DateTime<Utc>>,
        barrier: bool,
    ) -> Result<ForkInfo> {
        let name = name.trim();
        validate_fork_name(name)?;

        let SlateDbHandle::ReadWrite(parent_db) = &self.db_handle else {
            return Err(anyhow!(
                "Cannot create forks in read-only mode. Start the server without --read-only or --checkpoint flags."
            ));
        };

        let codec = KeyCodec::new();
        let fork_db_path = ForkInfo::db_path(self.parent_db_path.as_ref(), name);
        if self
            .registry_value(parent_db, &codec.fork_registry_key(name))
            .await?
            .is_some()
        {
            return Err(anyhow!("A fork named '{}' already exists", name));
        }

        let branch_point = self
            .resolve_branch_point(parent_db, name, from_checkpoint.as_deref(), at, barrier)
            .await?;

        // The parent's own lineage (if it is itself a fork) extends the
        // ancestor chain the fork records.
        let parent_info = ForkInfo::load(&self.db_handle).await?;
        let mut ancestors = parent_info
            .as_ref()
            .map(|info| info.ancestors.clone())
            .unwrap_or_default();
        ancestors.push(ForkAncestor {
            db_path: self.parent_db_path.as_ref().to_string(),
            base_epoch: parent_info.map(|info| info.base_epoch).unwrap_or(0),
        });

        let info = ForkInfo {
            name: name.to_string(),
            parent_db_path: self.parent_db_path.as_ref().to_string(),
            base_epoch: branch_point.base_epoch,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            ancestors,
        };

        if barrier {
            // Eager: materialize now, from the barriered branch point.
            materialize_fork_db(
                &self.object_store,
                &self.block_transformer,
                &info,
                branch_point.checkpoint_id,
            )
            .await?;
        } else {
            // Lazy: publish the pending-materialization record the fork's
            // first open will consume.
            let pending = PendingFork {
                fork_info: info.clone(),
                source_checkpoint_id: branch_point.checkpoint_id,
                source_checkpoint_name: branch_point.checkpoint_name,
                source_checkpoint_owned: branch_point.owned,
            };
            let bytes = serde_json::to_vec(&pending)?;
            self.object_store
                .put(
                    &PendingFork::path(&Path::from(fork_db_path.clone())),
                    bytes.into(),
                )
                .await
                .map_err(|e| anyhow!("Failed to write pending-fork marker: {}", e))?;
        }

        // Registry entry LAST, into the parent's LSM. Eager: a crash can now
        // leave a fork whose lineage is durable but unlisted (an orphan the
        // operator can re-register), never a registry entry for a fork whose
        // lineage record is missing. Lazy: a crash can leave a pending marker
        // without a registry entry (an orphan `fork delete` can't see but the
        // first open still materializes), never a listed fork without its
        // marker.
        parent_db
            .put_with_options(
                &codec.fork_registry_key(name),
                &info.encode()?,
                &PutOptions::default(),
                &WriteOptions::default(),
            )
            .await
            .map_err(|e| anyhow!("Failed to register fork '{}': {}", name, e))?;

        Ok(info)
    }

    /// Resolve the fork's branch point: an existing named checkpoint, a
    /// manifest chosen by timestamp (point-in-time fork), or a fresh
    /// checkpoint of the current durable state. `barrier` decides whether
    /// the fresh checkpoint is taken under the seal+flush barrier
    /// ([`CheckpointManager`], eager) or directly on the parent database
    /// (lazy — `CheckpointScope::Durable` checkpoints the already-durable
    /// manifest, so the branch point can lag HEAD by up to the flush
    /// interval).
    async fn resolve_branch_point(
        &self,
        parent_db: &slatedb::Db,
        name: &str,
        from_checkpoint: Option<&str>,
        at: Option<DateTime<Utc>>,
        barrier: bool,
    ) -> Result<BranchPoint> {
        if let Some(at) = at {
            let manifest_id = self.manifest_at_time(at).await?;
            let checkpoint_name = format!("pitr-{name}-{}", at.timestamp());
            let result = self
                .admin
                .create_detached_checkpoint_at(
                    manifest_id,
                    &CheckpointOptions {
                        lifetime: None,
                        source: None,
                        name: Some(checkpoint_name.clone()),
                    },
                )
                .await
                .map_err(|e| anyhow!("Failed to pin historical manifest {}: {}", manifest_id, e))?;
            let manifest = self
                .admin
                .read_manifest(Some(manifest_id))
                .await
                .map_err(|e| anyhow!("Failed to read historical manifest: {}", e))?
                .ok_or_else(|| anyhow!("Historical manifest {} not found", manifest_id))?;
            return Ok(BranchPoint {
                checkpoint_id: result.id,
                checkpoint_name,
                base_epoch: manifest.writer_epoch() + 1,
                owned: true,
            });
        }

        let (checkpoint_id, checkpoint_name, owned) = match from_checkpoint.map(str::trim) {
            Some("") | None => {
                let checkpoint_name = format!("fork-{name}-{}", Uuid::new_v4().simple());
                if barrier {
                    self.checkpoint_manager
                        .create_checkpoint(&checkpoint_name)
                        .await
                        .map(|cp| (cp.id, cp.name, true))?
                } else {
                    // Lazy: checkpoint the already-durable manifest directly,
                    // skipping the seal+flush barrier (and its ~0.2-0.5s).
                    let result = parent_db
                        .create_checkpoint(
                            CheckpointScope::Durable,
                            &CheckpointOptions {
                                lifetime: None,
                                source: None,
                                name: Some(checkpoint_name.clone()),
                            },
                        )
                        .await
                        .map_err(|e| anyhow!("Failed to create branch-point checkpoint: {}", e))?;
                    (result.id, checkpoint_name, true)
                }
            }
            Some(checkpoint_name) => self
                .checkpoint_manager
                .get_checkpoint_info(checkpoint_name)
                .await?
                .map(|cp| (cp.id, cp.name, false))
                .ok_or_else(|| anyhow!("Checkpoint '{}' not found", checkpoint_name))?,
        };

        let checkpoints = self
            .admin
            .list_checkpoints(None)
            .await
            .map_err(|e| anyhow!("Failed to list checkpoints: {}", e))?;
        let source = checkpoints
            .into_iter()
            .find(|cp| cp.id == checkpoint_id)
            .ok_or_else(|| anyhow!("Checkpoint '{}' no longer exists", checkpoint_name))?;
        let source_manifest = self
            .admin
            .read_manifest(Some(source.manifest_id))
            .await
            .map_err(|e| anyhow!("Failed to read checkpoint manifest: {}", e))?
            .ok_or_else(|| anyhow!("Checkpoint manifest not found"))?;

        Ok(BranchPoint {
            checkpoint_id,
            checkpoint_name,
            base_epoch: source_manifest.writer_epoch() + 1,
            owned,
        })
    }

    /// The id of the last manifest flushed at or before `target`.
    ///
    /// Two resolution paths:
    ///
    /// 1. **Flush-time index** (fast path): the flush coordinator records each
    ///    flush's wall-clock time in the volume's own LSM
    ///    (`KeyPrefix::FlushTime`, see [`crate::fs::flush_coordinator`]), so a
    ///    timestamp resolves to the greatest indexed manifest id at or before
    ///    `target` with no object-store I/O.
    /// 2. **Manifest listing** (fallback): engages when the index has no entry
    ///    at or before `target` — volumes written before the index existed, or
    ///    a row lost to the index's best-effort write. Manifests are
    ///    immutable, monotonically numbered objects under
    ///    `<db path>/manifest/`, each stamped with the object store's
    ///    last-modified time (the flush publication time), so listing them
    ///    reproduces the same answer.
    pub async fn manifest_at_time(&self, target: DateTime<Utc>) -> Result<u64> {
        if let Some(id) = self.manifest_at_time_from_index(target).await? {
            return Ok(id);
        }
        self.manifest_at_time_from_listing(target).await
    }

    /// Index path of [`Self::manifest_at_time`]: the greatest manifest id
    /// whose recorded flush time is at or before `target`; `None` when the
    /// index has no such entry (the listing fallback engages). The index is
    /// keyed by manifest id and scanned in full — one small row per flush.
    async fn manifest_at_time_from_index(&self, target: DateTime<Utc>) -> Result<Option<u64>> {
        // Index timestamps are (epoch seconds, nanos); a pre-epoch target
        // cannot match.
        let Ok(target_secs) = u64::try_from(target.timestamp()) else {
            return Ok(None);
        };
        let target_time = (target_secs, target.timestamp_subsec_nanos());
        let codec = KeyCodec::new();
        let scan_options = slatedb::config::ScanOptions {
            durability_filter: slatedb::config::DurabilityLevel::Memory,
            cache_blocks: true,
            ..Default::default()
        };
        let mut iter = match &self.db_handle {
            SlateDbHandle::ReadWrite(db) => db
                .scan_prefix_with_options(
                    codec.flush_time_prefix(),
                    bytes::Bytes::new()..,
                    &scan_options,
                )
                .await
                .map_err(|e| anyhow!("Failed to scan flush-time index: {}", e))?,
            SlateDbHandle::ReadOnly(reader) => reader
                .load()
                .scan_prefix_with_options(
                    codec.flush_time_prefix(),
                    bytes::Bytes::new()..,
                    &scan_options,
                )
                .await
                .map_err(|e| anyhow!("Failed to scan flush-time index: {}", e))?,
        };
        let mut best: Option<u64> = None;
        while let Some(kv) = iter
            .next()
            .await
            .map_err(|e| anyhow!("Failed to scan flush-time index: {}", e))?
        {
            let Some(manifest_id) = codec.parse_flush_time_key(&kv.key) else {
                continue;
            };
            let Some(flushed_at) = KeyCodec::decode_flush_time(&kv.value) else {
                continue;
            };
            if flushed_at <= target_time && best.is_none_or(|b| manifest_id > b) {
                best = Some(manifest_id);
            }
        }
        Ok(best)
    }

    /// Listing path of [`Self::manifest_at_time`]: the last manifest object
    /// under `<db path>/manifest/` whose last-modified time is at or before
    /// `target`.
    async fn manifest_at_time_from_listing(&self, target: DateTime<Utc>) -> Result<u64> {
        let prefix = Path::from(format!("{}/manifest", self.parent_db_path));
        let mut stream = self.object_store.list(Some(&prefix));
        let mut best: Option<(u64, DateTime<Utc>)> = None;
        {
            use futures::TryStreamExt;
            while let Some(meta) = stream.try_next().await? {
                let Some(id) = parse_manifest_id(&meta.location) else {
                    continue;
                };
                let last_modified: DateTime<Utc> = meta.last_modified.into();
                if last_modified > target {
                    continue;
                }
                if best.is_none_or(|(best_id, _)| id > best_id) {
                    best = Some((id, last_modified));
                }
            }
        }
        best.map(|(id, _)| id).ok_or_else(|| {
            anyhow!(
                "No manifest on '{}' at or before {}",
                self.parent_db_path,
                target
            )
        })
    }

    /// List the direct forks of this volume: a prefix scan of the fork
    /// registry in this volume's own LSM (see [`crate::fork_info`]).
    pub async fn list_forks(&self) -> Result<Vec<ForkInfo>> {
        let prefix = KeyCodec::new().fork_registry_prefix();
        let scan_options = slatedb::config::ScanOptions {
            durability_filter: slatedb::config::DurabilityLevel::Memory,
            cache_blocks: true,
            ..Default::default()
        };
        let mut iter = match &self.db_handle {
            SlateDbHandle::ReadWrite(db) => db
                .scan_prefix_with_options(prefix, bytes::Bytes::new().., &scan_options)
                .await
                .map_err(|e| anyhow!("Failed to scan fork registry: {}", e))?,
            SlateDbHandle::ReadOnly(reader) => reader
                .load()
                .scan_prefix_with_options(prefix, bytes::Bytes::new().., &scan_options)
                .await
                .map_err(|e| anyhow!("Failed to scan fork registry: {}", e))?,
        };
        let mut forks = Vec::new();
        while let Some(kv) = iter
            .next()
            .await
            .map_err(|e| anyhow!("Failed to scan fork registry: {}", e))?
        {
            forks.push(ForkInfo::decode(&kv.value)?);
        }
        forks.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.name.cmp(&b.name)));
        Ok(forks)
    }

    /// Delete the fork named `name`: release the GC pin it holds on this
    /// volume's manifest, remove every object under the fork's db path, and
    /// drop its registry entry from this volume's LSM.
    ///
    /// **Stop the fork's server first.** SlateDB fencing is single-writer:
    /// a fork server still running against the fork's db path would
    /// re-fence the delete's manifest reads and could recreate state
    /// (flushed manifests, new checkpoints) under a half-deleted prefix.
    ///
    /// A fork that has forks of its own is refused until those children
    /// are deleted: their external SST references point at this fork's db
    /// path, so deleting it would orphan them.
    ///
    /// Deleting a fork never affects its parent or siblings: the fork's
    /// external SSTs stay at the parent's path untouched, and every object
    /// delete is scoped to the fork's own db path (`<parent>/forks/<name>`
    /// — object-store prefix listings match on a path-segment basis, so a
    /// sibling like `forks/<name>2` is never in scope).
    pub async fn delete_fork(&self, name: &str) -> Result<()> {
        use futures::TryStreamExt;

        let name = name.trim();
        validate_fork_name(name)?;

        let SlateDbHandle::ReadWrite(parent_db) = &self.db_handle else {
            return Err(anyhow!(
                "Cannot delete forks in read-only mode. Start the server without --read-only or --checkpoint flags."
            ));
        };

        let codec = KeyCodec::new();
        let registry_key = codec.fork_registry_key(name);
        if self
            .registry_value(parent_db, &registry_key)
            .await?
            .is_none()
        {
            return Err(anyhow!("Fork '{}' not found", name));
        }

        let fork_db_path = Path::from(ForkInfo::db_path(self.parent_db_path.as_ref(), name));

        // Refuse while children anchor on this fork. Their slatedb clones
        // exist under `<fork db path>/forks/` from the moment of creation
        // (before the registry write), so a prefix listing catches orphans
        // a registry scan would miss.
        let children_prefix = Path::from(format!("{fork_db_path}/{FORKS_INFIX}"));
        let mut children = self.object_store.list(Some(&children_prefix));
        if children.try_next().await?.is_some() {
            return Err(anyhow!(
                "Fork '{}' has forks of its own; delete its forks first",
                name
            ));
        }

        let fork_admin =
            AdminBuilder::new(fork_db_path.clone(), Arc::clone(&self.object_store)).build();

        // A fork still pending materialization (lazy create, never opened)
        // has no slatedb database to delete: release its creation-owned
        // source checkpoint, any pin a crashed mid-materialization clone
        // took, and every object under its db path (the pending marker, at
        // minimum), then unregister.
        if let Some(pending) = PendingFork::load(&self.object_store, &fork_db_path).await? {
            if pending.source_checkpoint_owned {
                // Idempotent, so a retried delete is fine.
                self.admin
                    .delete_checkpoint(pending.source_checkpoint_id)
                    .await
                    .map_err(|e| {
                        anyhow!("Failed to delete fork '{}'s source checkpoint: {}", name, e)
                    })?;
            }
            self.release_parent_pin(&fork_admin, name).await?;
            let mut remaining = self.object_store.list(Some(&fork_db_path));
            while let Some(meta) = remaining.try_next().await? {
                object_store::ObjectStoreExt::delete(&*self.object_store, &meta.location).await?;
            }
            // Registry entry LAST (see the materialized path below).
            parent_db
                .delete_with_options(&registry_key, &WriteOptions::default())
                .await
                .map_err(|e| anyhow!("Failed to unregister fork '{}': {}", name, e))?;
            return Ok(());
        }

        self.release_parent_pin(&fork_admin, name).await?;

        // Delete the fork's database. slatedb's delete_db removes every
        // object under the fork's db path (manifests, SSTs, WAL) behind a
        // `.deleting` marker that makes a crash mid-delete resumable, and
        // refuses a prefix that was never a slatedb dir. External SSTs live
        // at the parent's path, outside this prefix, so they are untouched.
        fork_admin
            .delete_db(true)
            .await
            .map_err(|e| anyhow!("Failed to delete fork '{}'s database: {}", name, e))?;

        // Anything else under the fork's own prefix that isn't slatedb's —
        // the wrapped encryption key, the bucket-id marker, zerofs
        // `segments/` — goes too. Scoped to the fork's db path exactly:
        // prefix listings match whole path segments, so the parent prefix
        // and sibling forks are out of scope.
        let mut remaining = self.object_store.list(Some(&fork_db_path));
        while let Some(meta) = remaining.try_next().await? {
            object_store::ObjectStoreExt::delete(&*self.object_store, &meta.location).await?;
        }

        // Registry entry LAST: a crash before this point leaves the entry
        // in place, and a retried delete resumes from whatever objects
        // remain (delete_db is idempotent).
        parent_db
            .delete_with_options(&registry_key, &WriteOptions::default())
            .await
            .map_err(|e| anyhow!("Failed to unregister fork '{}': {}", name, e))?;

        Ok(())
    }

    /// Release the GC pin a materialized (or partially materialized) fork
    /// holds on this volume's manifest: the clone pinned an unnamed
    /// checkpoint on this volume, recorded in the fork's own manifest as
    /// this parent's `external_dbs` entry (`final_checkpoint_id`). While it
    /// exists, the parent's segment reclamation pauses (any persistent
    /// checkpoint protects segments indefinitely), so this is what resumes
    /// it. A fork too corrupt to read has effectively lost its pin record;
    /// warn and proceed (slatedb's delete_db strips the pin itself when it
    /// can read the manifest).
    async fn release_parent_pin(&self, fork_admin: &Admin, name: &str) -> Result<()> {
        match fork_admin.read_manifest(None).await {
            Ok(Some(manifest)) => {
                for external_db in manifest.external_dbs() {
                    if external_db.path != self.parent_db_path.as_ref() {
                        continue;
                    }
                    if let Some(pin) = external_db.final_checkpoint_id {
                        self.admin.delete_checkpoint(pin).await.map_err(|e| {
                            anyhow!(
                                "Failed to release fork '{}'s pin on the parent: {}",
                                name,
                                e
                            )
                        })?;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(
                    "Could not read fork '{}'s manifest to release its pin ({}); proceeding",
                    name,
                    e
                );
            }
        }
        Ok(())
    }

    /// Point-read of one registry key on the parent's (writable) database.
    async fn registry_value(
        &self,
        db: &slatedb::Db,
        key: &bytes::Bytes,
    ) -> Result<Option<bytes::Bytes>> {
        let read_options = slatedb::config::ReadOptions {
            durability_filter: slatedb::config::DurabilityLevel::Memory,
            cache_blocks: true,
            ..Default::default()
        };
        db.get_with_options(key, &read_options)
            .await
            .map_err(|e| anyhow!("Failed to read fork registry: {}", e))
    }
}

/// Build the fork's database at its db path: slatedb clone from the source
/// checkpoint, wrapped-key copy, then the one-time lineage write into the
/// fork's own LSM. Shared by eager fork creation and first-open
/// materialization; every step is idempotent (the clone is retryable per
/// slatedb's clone builder), so a crash mid-way re-runs cleanly.
async fn materialize_fork_db(
    object_store: &Arc<dyn ObjectStore>,
    block_transformer: &Arc<dyn slatedb::BlockTransformer>,
    info: &ForkInfo,
    source_checkpoint_id: Uuid,
) -> Result<()> {
    let fork_db_path = Path::from(ForkInfo::db_path(&info.parent_db_path, &info.name));
    let parent_db_path = Path::from(info.parent_db_path.clone());

    // Shallow-copy the metadata database. External SSTs stay referenced
    // from the parent's path; nothing moves.
    let admin = AdminBuilder::new(fork_db_path.clone(), Arc::clone(object_store)).build();
    admin
        .create_clone_builder_from_source(CloneSourceSpec::with_checkpoint(
            parent_db_path.clone(),
            source_checkpoint_id,
        ))
        .build()
        .await
        .map_err(|e| anyhow!("Failed to clone database for fork '{}': {}", info.name, e))?;

    // Encryption keys are per db path; the fork reads the parent's
    // segments and SSTs, so it needs the same wrapped key.
    key_management::copy_wrapped_key(object_store, &parent_db_path, &fork_db_path)
        .await
        .context("Failed to copy encryption key to fork")?;

    // Open the freshly cloned fork database once to persist the lineage
    // record into the fork's own LSM. Opened with the parent's durability
    // posture (no WAL; explicit flush) so the record is SST-durable before
    // the fork is served. This open consumes the fork's first writer epoch
    // (base_epoch), so the fork's serving open writes segments at
    // base_epoch + 1 — routing compares `>= base_epoch`, so the gap is
    // harmless.
    let settings = slatedb::config::Settings {
        wal_enabled: false,
        flush_interval: None,
        compactor_options: None,
        garbage_collector_options: None,
        compression_codec: None, // handled by the block transformer
        ..Default::default()
    };
    let fork_db = slatedb::DbBuilder::new(fork_db_path.clone(), Arc::clone(object_store))
        .with_settings(settings)
        .with_block_transformer(Arc::clone(block_transformer))
        .with_filter_policies(crate::fs::filter_policy::filter_policies())
        .with_segment_extractor(Arc::new(crate::segment_extractor::ZeroFsSegmentExtractor))
        .build()
        .await
        .map_err(|e| anyhow!("Failed to open fork database '{}': {}", fork_db_path, e))?;
    info.save(&fork_db).await?;
    fork_db
        .flush()
        .await
        .map_err(|e| anyhow!("Failed to flush fork lineage record: {}", e))?;
    fork_db
        .close()
        .await
        .map_err(|e| anyhow!("Failed to close fork database: {}", e))?;
    Ok(())
}

/// Pre-key-init phase of pending-fork materialization: run the slatedb
/// clone and copy the parent's wrapped key into the fork's db path, WITHOUT
/// touching the lineage record or the marker. Must run before the volume's
/// encryption key is loaded — otherwise a first open would generate a fresh
/// wrong key, then materialization would write fork SSTs the parent's key
/// cannot read (and the parent's SSTs become undecryptable to the fork).
/// Idempotent: the slatedb clone is retryable and the key copy overwrites
/// with identical bytes.
pub async fn materialize_fork_storage_if_pending(
    object_store: &Arc<dyn ObjectStore>,
    db_path: &Path,
) -> Result<bool> {
    let Some(pending) = PendingFork::load(object_store, db_path).await? else {
        return Ok(false);
    };
    let parent_db_path = Path::from(pending.fork_info.parent_db_path.clone());
    let admin = AdminBuilder::new(db_path.clone(), Arc::clone(object_store)).build();
    admin
        .create_clone_builder_from_source(CloneSourceSpec::with_checkpoint(
            parent_db_path.clone(),
            pending.source_checkpoint_id,
        ))
        .build()
        .await
        .map_err(|e| {
            anyhow!(
                "Failed to clone database for fork '{}': {}",
                pending.fork_info.name,
                e
            )
        })?;
    key_management::copy_wrapped_key(object_store, &parent_db_path, db_path)
        .await
        .context("Failed to copy encryption key to fork")?;
    Ok(true)
}

/// Materialize a lazily created fork at its first open (see the module
/// docs). Reads the pending-materialization marker under `db_path`; absent
/// marker => no-op (`Ok(false)`), so this is cheap to call on every volume
/// startup. Otherwise runs the deferred clone, key copy, and lineage write,
/// then deletes the marker — last, so a crash mid-materialization re-runs
/// from the top on the next open. Returns `Ok(true)` when this call
/// materialized the fork.
pub async fn materialize_fork_if_pending(
    object_store: &Arc<dyn ObjectStore>,
    db_path: &Path,
    block_transformer: Arc<dyn slatedb::BlockTransformer>,
) -> Result<bool> {
    let Some(pending) = PendingFork::load(object_store, db_path).await? else {
        return Ok(false);
    };
    materialize_fork_db(
        object_store,
        &block_transformer,
        &pending.fork_info,
        pending.source_checkpoint_id,
    )
    .await?;
    object_store
        .delete(&PendingFork::path(db_path))
        .await
        .map_err(|e| anyhow!("Failed to delete pending-fork marker: {}", e))?;
    Ok(true)
}

/// The manifest sequence id in a `<db path>/manifest/<id>.manifest` key.
fn parse_manifest_id(location: &Path) -> Option<u64> {
    let filename = location.filename()?;
    let id = filename.strip_suffix(".manifest")?;
    id.parse().ok()
}

fn validate_fork_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(anyhow!("Fork name cannot be empty"));
    }
    let valid = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.');
    if !valid {
        return Err(anyhow!(
            "Fork name '{}' may only contain ASCII letters, digits, '-', '_' and '.'",
            name
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_transformer::ZeroFsBlockTransformer;
    use crate::config::CompressionConfig;
    use crate::fs::ZeroFS;
    use crate::fs::errors::FsError;
    use crate::fs::permissions::Credentials;
    use crate::fs::types::AuthContext;
    use crate::fs::types::SetAttributes;
    use crate::segment_path_router::SegmentPathRouter;
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use slatedb::DbBuilder;
    use std::fmt::{self, Display, Formatter};
    use std::sync::atomic::{AtomicU64, Ordering};

    const TEST_KEY: [u8; 32] = [7u8; 32];
    const TEST_PASSWORD: &str = "fork-test-password";

    /// Object-store wrapper that counts `list` calls, so tests can prove the
    /// flush-time index resolves a point-in-time manifest lookup without an
    /// object-store listing (and that the fallback does list).
    #[derive(Debug)]
    struct ListCountingStore {
        inner: Arc<dyn ObjectStore>,
        list_calls: Arc<AtomicU64>,
    }

    impl ListCountingStore {
        fn wrap(inner: Arc<dyn ObjectStore>) -> (Arc<dyn ObjectStore>, Arc<AtomicU64>) {
            let list_calls = Arc::new(AtomicU64::new(0));
            (
                Arc::new(Self {
                    inner,
                    list_calls: Arc::clone(&list_calls),
                }),
                list_calls,
            )
        }
    }

    impl Display for ListCountingStore {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "ListCountingStore({})", self.inner)
        }
    }

    #[async_trait]
    impl ObjectStore for ListCountingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            self.inner.get_opts(location, options).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.list_calls.fetch_add(1, Ordering::Relaxed);
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    /// Open a parent volume on a list-counting store: fs, fork manager, the
    /// list-call counter, the db path, and the db handle. Opened with the
    /// production `wal_enabled: false` posture so a flush publishes a
    /// manifest, which is what the flush-time index records.
    async fn counting_parent_volume() -> (
        Arc<ZeroFS>,
        Arc<ForkManager>,
        Arc<AtomicU64>,
        Path,
        SlateDbHandle,
    ) {
        let (object_store, list_calls) =
            ListCountingStore::wrap(Arc::new(InMemory::new()) as Arc<dyn ObjectStore>);
        let db_path = Path::from("vol");
        key_management::load_or_init_encryption_key(
            &object_store,
            &db_path,
            crate::secrets::EncryptionPassword::try_new(TEST_PASSWORD).unwrap(),
            false,
        )
        .await
        .unwrap();
        let settings = slatedb::config::Settings {
            wal_enabled: false,
            // No periodic background flush: the coordinator's flush must be
            // the one that publishes (and records) the manifest.
            flush_interval: None,
            ..Default::default()
        };
        let (fs, db_handle) =
            open_volume_with_settings(Arc::clone(&object_store), db_path.clone(), settings).await;
        let (_, fork_manager) = managers(
            db_handle.clone(),
            db_path.clone(),
            Arc::clone(&object_store),
            &fs,
        );
        (fs, fork_manager, list_calls, db_path, db_handle)
    }

    fn test_creds() -> Credentials {
        Credentials {
            uid: 0,
            gid: 0,
            gid_known: true,
            groups: [0; 16],
            groups_count: 1,
            groups_complete: true,
        }
    }

    fn root_auth() -> AuthContext {
        AuthContext {
            uid: 0,
            gid: 0,
            gid_known: true,
            gids: Vec::new(),
            groups_complete: true,
        }
    }

    /// Open a writable volume (slatedb + ZeroFS) the way production wires it:
    /// the fork lineage is read from the volume's own LSM after the database
    /// opens (see `StartupContext::open_db`), and segment reads go through a
    /// SegmentPathRouter built from it.
    async fn open_volume(
        object_store: Arc<dyn ObjectStore>,
        db_path: Path,
    ) -> (Arc<ZeroFS>, SlateDbHandle) {
        open_volume_with_settings(object_store, db_path, slatedb::config::Settings::default()).await
    }

    /// [`open_volume`] with explicit slatedb settings (e.g. the production
    /// `wal_enabled: false` posture, under which a flush publishes a manifest).
    async fn open_volume_with_settings(
        object_store: Arc<dyn ObjectStore>,
        db_path: Path,
        settings: slatedb::config::Settings,
    ) -> (Arc<ZeroFS>, SlateDbHandle) {
        let block_transformer: Arc<dyn slatedb::BlockTransformer> =
            ZeroFsBlockTransformer::try_new_arc(&TEST_KEY, CompressionConfig::default())
                .expect("test key should be lockable");
        // Mirrors the production startup hook (StartupContext::open_db): a
        // lazily created fork materializes at its first open.
        materialize_fork_if_pending(&object_store, &db_path, Arc::clone(&block_transformer))
            .await
            .unwrap();
        let slatedb = Arc::new(
            DbBuilder::new(db_path.clone(), Arc::clone(&object_store))
                .with_settings(settings)
                .with_block_transformer(block_transformer)
                .with_filter_policies(crate::fs::filter_policy::filter_policies())
                .with_segment_extractor(Arc::new(crate::segment_extractor::ZeroFsSegmentExtractor))
                .build()
                .await
                .unwrap(),
        );
        let db_handle = SlateDbHandle::ReadWrite(slatedb);
        let fork_info = ForkInfo::load(&db_handle).await.unwrap();
        let segment_store: Arc<dyn ObjectStore> = Arc::new(SegmentPathRouter::new(
            Arc::clone(&object_store),
            db_path.clone(),
            fork_info.as_ref(),
        ));
        let fs = Arc::new(
            ZeroFS::try_new(
                db_handle.clone(),
                u64::MAX,
                None,
                false,
                false,
                None,
                None,
                Arc::new(crate::dedup::DedupCache::new()),
                None,
                crate::object_trace::ObjectTracer::new(),
                segment_store,
                crate::frame_codec::FrameCodec::try_new(
                    &TEST_KEY,
                    crate::segment::SEGMENT_INFO,
                    CompressionConfig::default(),
                )
                .expect("test key should be lockable"),
                None,
                fork_info.as_ref().map(|info| info.base_epoch),
            )
            .await
            .unwrap(),
        );
        fs.start_reclaim_drainer();
        (fs, db_handle)
    }

    fn managers(
        db_handle: SlateDbHandle,
        db_path: Path,
        object_store: Arc<dyn ObjectStore>,
        fs: &Arc<ZeroFS>,
    ) -> (Arc<CheckpointManager>, Arc<ForkManager>) {
        let checkpoint_manager = Arc::new(CheckpointManager::new(
            db_handle.clone(),
            db_path.clone(),
            Arc::clone(&object_store),
        ));
        let fc = fs.flush_coordinator.clone();
        checkpoint_manager.set_pre_flush(Arc::new(move || {
            let fc = fc.clone();
            Box::pin(async move {
                fc.flush()
                    .await
                    .map_err(|e| anyhow!("seal+flush failed: {:?}", e))
            })
        }));
        let block_transformer: Arc<dyn slatedb::BlockTransformer> =
            ZeroFsBlockTransformer::try_new_arc(&TEST_KEY, CompressionConfig::default())
                .expect("test key should be lockable");
        let fork_manager = Arc::new(ForkManager::new(
            db_handle,
            db_path,
            object_store,
            block_transformer,
            Arc::clone(&checkpoint_manager),
        ));
        (checkpoint_manager, fork_manager)
    }

    async fn new_parent_volume() -> (
        Arc<ZeroFS>,
        Arc<ForkManager>,
        Arc<dyn ObjectStore>,
        Path,
        SlateDbHandle,
    ) {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let db_path = Path::from("vol");
        key_management::load_or_init_encryption_key(
            &object_store,
            &db_path,
            crate::secrets::EncryptionPassword::try_new(TEST_PASSWORD).unwrap(),
            false,
        )
        .await
        .unwrap();
        let (fs, db_handle) = open_volume(Arc::clone(&object_store), db_path.clone()).await;
        let (_, fork_manager) = managers(
            db_handle.clone(),
            db_path.clone(),
            Arc::clone(&object_store),
            &fs,
        );
        (fs, fork_manager, object_store, db_path, db_handle)
    }

    #[tokio::test]
    async fn fork_reads_parent_and_writes_are_isolated() {
        let (parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"hello.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"hello from parent"))
            .await
            .unwrap();

        let info = fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();
        assert_eq!(info.base_epoch, 2, "fork starts one epoch above the parent");
        assert_eq!(info.ancestors.len(), 1);

        let forks = fork_manager.list_forks().await.unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].name, "f1");

        let (fork_fs, fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let loaded = ForkInfo::load(&fork_db)
            .await
            .unwrap()
            .expect("lineage persisted in the fork's LSM");
        assert_eq!(loaded.base_epoch, 2);

        // The fork sees the parent's file (same inode id: the LSM is cloned).
        let (data, _) = fork_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"hello from parent");

        // Writes in the fork are visible to the fork and invisible to the parent.
        let (fork_file_id, _) = fork_fs
            .create(&creds, 0, b"fork-only.txt", &SetAttributes::default())
            .await
            .unwrap();
        fork_fs
            .write(&auth, fork_file_id, 0, &Bytes::from_static(b"fork write"))
            .await
            .unwrap();
        let (data, _) = fork_fs
            .read_file(&auth, fork_file_id, 0, 1024)
            .await
            .unwrap();
        assert_eq!(&data[..], b"fork write");
        assert!(matches!(
            parent_fs.lookup(&creds, 0, b"fork-only.txt").await,
            Err(FsError::NotFound)
        ));

        // And the fork still sees new parent reads as of clone time (nothing
        // written after the clone point leaks in either direction).
        let (after_id, _) = parent_fs
            .create(&creds, 0, b"after-fork.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, after_id, 0, &Bytes::from_static(b"after"))
            .await
            .unwrap();
        assert!(matches!(
            fork_fs.lookup(&creds, 0, b"after-fork.txt").await,
            Err(FsError::NotFound)
        ));
    }

    #[tokio::test]
    async fn fork_of_fork_reads_across_the_chain() {
        let (parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"base.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"base"))
            .await
            .unwrap();

        // Fork f1 and write into it.
        fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();
        let (f1_fs, f1_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let (f1_file_id, _) = f1_fs
            .create(&creds, 0, b"f1.txt", &SetAttributes::default())
            .await
            .unwrap();
        f1_fs
            .write(&auth, f1_file_id, 0, &Bytes::from_static(b"f1 write"))
            .await
            .unwrap();

        // Fork g from f1; it must read across both ancestors.
        let (_, g_manager) = managers(
            f1_db.clone(),
            Path::from("vol/forks/f1"),
            Arc::clone(&object_store),
            &f1_fs,
        );
        let g_info = g_manager.create_fork("g", None, None, true).await.unwrap();
        // f1 was opened twice before the checkpoint (the lineage write at
        // creation, then the serving open above), so its writer epoch is 3.
        assert_eq!(g_info.base_epoch, 4);
        assert_eq!(g_info.ancestors.len(), 2);

        let (g_fs, _g_db) = open_volume(
            Arc::clone(&object_store),
            Path::from("vol/forks/f1/forks/g"),
        )
        .await;

        let (data, _) = g_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"base", "g reads the root volume's file");
        let (data, _) = g_fs.read_file(&auth, f1_file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"f1 write", "g reads its parent fork's file");
    }

    /// Forking an OLD checkpoint must compute the fork's base epoch from the
    /// checkpoint's manifest, not from the live parent's current epoch: the
    /// clone inherits the checkpoint-era epoch, so its own segments start one
    /// above that, and the router must agree.
    #[tokio::test]
    async fn fork_from_old_checkpoint_routes_own_writes() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let db_path = Path::from("vol");
        key_management::load_or_init_encryption_key(
            &object_store,
            &db_path,
            crate::secrets::EncryptionPassword::try_new(TEST_PASSWORD).unwrap(),
            false,
        )
        .await
        .unwrap();
        let creds = test_creds();
        let auth = root_auth();

        // Parent, first open (writer epoch 1): write file A, checkpoint it.
        let (parent_fs, parent_db) = open_volume(Arc::clone(&object_store), db_path.clone()).await;
        let (file_a, _) = parent_fs
            .create(&creds, 0, b"a.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_a, 0, &Bytes::from_static(b"old state"))
            .await
            .unwrap();
        let (checkpoint_manager, _) = managers(
            parent_db.clone(),
            db_path.clone(),
            Arc::clone(&object_store),
            &parent_fs,
        );
        checkpoint_manager.create_checkpoint("old").await.unwrap();
        if let SlateDbHandle::ReadWrite(db) = &parent_db {
            db.close().await.unwrap();
        }

        // Parent restarts (writer epoch bumps to 2) and writes file B.
        let (parent_fs, parent_db) = open_volume(Arc::clone(&object_store), db_path.clone()).await;
        let (file_b, _) = parent_fs
            .create(&creds, 0, b"b.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_b, 0, &Bytes::from_static(b"new state"))
            .await
            .unwrap();

        // Fork from the OLD checkpoint: base epoch must come from the
        // checkpoint's manifest (1), not the live parent's epoch (2).
        let (_, fork_manager) = managers(
            parent_db.clone(),
            db_path.clone(),
            Arc::clone(&object_store),
            &parent_fs,
        );
        let info = fork_manager
            .create_fork("past", Some("old".to_string()), None, true)
            .await
            .unwrap();
        assert_eq!(
            info.base_epoch, 2,
            "fork of an epoch-1 checkpoint owns epoch 2 onward"
        );

        // The fork sees the old state, not file B, and its own writes at
        // epoch >= 2 route to itself (not to the parent).
        let (fork_fs, _fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/past")).await;
        let (data, _) = fork_fs.read_file(&auth, file_a, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"old state");
        assert!(matches!(
            fork_fs.lookup(&creds, 0, b"b.txt").await,
            Err(FsError::NotFound)
        ));

        let (file_c, _) = fork_fs
            .create(&creds, 0, b"c.txt", &SetAttributes::default())
            .await
            .unwrap();
        fork_fs
            .write(&auth, file_c, 0, &Bytes::from_static(b"fork write"))
            .await
            .unwrap();
        let (data, _) = fork_fs.read_file(&auth, file_c, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"fork write", "fork reads back its own writes");
        assert!(matches!(
            parent_fs.lookup(&creds, 0, b"c.txt").await,
            Err(FsError::NotFound)
        ));
    }

    /// A point-in-time fork: --at resolves the last manifest flushed before
    /// the timestamp and branches exactly that state.
    #[tokio::test]
    async fn fork_at_timestamp_branches_the_state_at_that_time() {
        let (parent_fs, fork_manager, object_store, parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        // State A, flushed to a manifest.
        let (file_a, _) = parent_fs
            .create(&creds, 0, b"a.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_a, 0, &Bytes::from_static(b"state A"))
            .await
            .unwrap();
        let (checkpoint_manager, _) = managers(
            _parent_db.clone(),
            parent_path.clone(),
            Arc::clone(&object_store),
            &parent_fs,
        );
        checkpoint_manager.create_checkpoint("a").await.unwrap();

        // The branch point: the newest manifest on the store right now.
        let target = {
            use futures::TryStreamExt;
            let prefix = slatedb::object_store::path::Path::from("vol/manifest");
            let mut stream = object_store.list(Some(&prefix));
            let mut newest: Option<chrono::DateTime<chrono::Utc>> = None;
            while let Some(meta) = stream.try_next().await.unwrap() {
                let lm: chrono::DateTime<chrono::Utc> = meta.last_modified.into();
                if newest.is_none_or(|n| lm > n) {
                    newest = Some(lm);
                }
            }
            newest.expect("a manifest exists")
        };

        // State B, written after the branch point.
        let (file_b, _) = parent_fs
            .create(&creds, 0, b"b.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_b, 0, &Bytes::from_static(b"state B"))
            .await
            .unwrap();
        parent_fs.flush_coordinator.flush().await.unwrap();

        let info = fork_manager
            .create_fork("pit", None, Some(target), true)
            .await
            .unwrap();
        let (fork_fs, _fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/pit")).await;

        let (data, _) = fork_fs.read_file(&auth, file_a, 0, 1024).await.unwrap();
        assert_eq!(
            &data[..],
            b"state A",
            "fork sees the state at the timestamp"
        );
        assert!(matches!(
            fork_fs.lookup(&creds, 0, b"b.txt").await,
            Err(FsError::NotFound)
        ));
        assert_eq!(info.base_epoch, 2);
        assert!(
            fork_manager
                .create_fork("too-early", None, Some(chrono::DateTime::UNIX_EPOCH), true)
                .await
                .is_err()
        );
    }

    /// The flush-time index resolves a point-in-time lookup with no
    /// object-store listing at all.
    #[tokio::test]
    async fn manifest_at_time_resolves_through_the_flush_time_index() {
        let (parent_fs, fork_manager, list_calls, _db_path, db_handle) =
            counting_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"a.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"state A"))
            .await
            .unwrap();
        parent_fs.flush_coordinator.flush().await.unwrap();

        // The coordinator writes the index row after replying to the flush, so
        // poll briefly for it. Far-future target: every recorded row qualifies.
        let target = Utc::now() + chrono::Duration::hours(1);
        let indexed = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if let Some(id) = fork_manager
                    .manifest_at_time_from_index(target)
                    .await
                    .unwrap()
                {
                    break id;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("flush-time index row recorded");

        let lists_before = list_calls.load(Ordering::Relaxed);
        let resolved = fork_manager.manifest_at_time(target).await.unwrap();
        assert_eq!(resolved, indexed);
        assert_eq!(
            list_calls.load(Ordering::Relaxed),
            lists_before,
            "index hit must not list the object store"
        );

        // The index row names the manifest the flush published.
        let SlateDbHandle::ReadWrite(db) = &db_handle else {
            panic!("parent volume is writable");
        };
        assert!(indexed <= db.subscribe().borrow().current_manifest.id());
        assert!(indexed >= 1);
    }

    /// A volume whose flush-time index is empty (flushed before the index
    /// existed — here simulated by flushing the raw slatedb handle, bypassing
    /// the flush coordinator) still resolves via the manifest listing.
    #[tokio::test]
    async fn manifest_at_time_falls_back_to_listing_when_the_index_is_empty() {
        let (parent_fs, fork_manager, list_calls, _db_path, db_handle) =
            counting_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"a.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"state A"))
            .await
            .unwrap();
        // Flush WITHOUT the flush coordinator: manifests exist, no index row.
        let SlateDbHandle::ReadWrite(db) = &db_handle else {
            panic!("parent volume is writable");
        };
        db.flush().await.unwrap();
        let current_manifest = db.subscribe().borrow().current_manifest.id();

        let target = Utc::now() + chrono::Duration::hours(1);
        assert!(
            fork_manager
                .manifest_at_time_from_index(target)
                .await
                .unwrap()
                .is_none(),
            "no flush coordinator flush => empty index"
        );

        let lists_before = list_calls.load(Ordering::Relaxed);
        let resolved = fork_manager.manifest_at_time(target).await.unwrap();
        assert!(
            list_calls.load(Ordering::Relaxed) > lists_before,
            "empty index engages the listing fallback"
        );
        // Background slatedb flushes may publish newer manifests after the
        // explicit one; the fallback resolves the newest at or before target.
        assert!(resolved >= current_manifest);
    }

    /// The registry lives in the parent's LSM and the lineage in the fork's
    /// own LSM: a fork reopened from cold (no metadata side-file anywhere)
    /// still lists from the parent and routes segment reads across the
    /// lineage.
    #[tokio::test]
    async fn reopened_fork_reads_its_lineage_from_the_lsm() {
        let (parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"base.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"base"))
            .await
            .unwrap();

        fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();
        let forks = fork_manager.list_forks().await.unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].name, "f1");

        // First open: write fork-local data and close the fork's database.
        let (fork_fs, fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let (fork_file_id, _) = fork_fs
            .create(&creds, 0, b"fork.txt", &SetAttributes::default())
            .await
            .unwrap();
        fork_fs
            .write(&auth, fork_file_id, 0, &Bytes::from_static(b"fork data"))
            .await
            .unwrap();
        fork_fs.flush_coordinator.flush().await.unwrap();
        if let SlateDbHandle::ReadWrite(db) = &fork_db {
            db.close().await.unwrap();
        }
        drop(fork_fs);

        // Reopen: open_volume loads the lineage from the fork's own LSM the
        // way startup does, so ancestor and own segments still route.
        let (fork_fs, fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let loaded = ForkInfo::load(&fork_db)
            .await
            .unwrap()
            .expect("lineage survived the reopen");
        assert_eq!(loaded.base_epoch, 2);

        let (data, _) = fork_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"base", "reopened fork reads the parent's file");
        let (data, _) = fork_fs
            .read_file(&auth, fork_file_id, 0, 1024)
            .await
            .unwrap();
        assert_eq!(&data[..], b"fork data", "reopened fork reads its own file");
    }

    #[tokio::test]
    async fn delete_fork_removes_the_fork_and_leaves_parent_and_siblings() {
        use futures::StreamExt;
        let (parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"base.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"parent data"))
            .await
            .unwrap();

        fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();
        // A sibling whose name extends f1's: prefix scoping must never catch it.
        fork_manager
            .create_fork("f12", None, None, true)
            .await
            .unwrap();

        // Write data inside f1, then stop it (the operator contract for delete).
        let (f1_fs, f1_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let (fork_file_id, _) = f1_fs
            .create(&creds, 0, b"fork-only.txt", &SetAttributes::default())
            .await
            .unwrap();
        f1_fs
            .write(&auth, fork_file_id, 0, &Bytes::from_static(b"fork data"))
            .await
            .unwrap();
        f1_fs.flush_coordinator.flush().await.unwrap();
        if let SlateDbHandle::ReadWrite(db) = &f1_db {
            db.close().await.unwrap();
        }
        drop(f1_fs);

        fork_manager.delete_fork("f1").await.unwrap();

        // The registry no longer lists f1; the sibling stays.
        let forks = fork_manager.list_forks().await.unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].name, "f12");

        // Nothing is left under the fork's own prefix...
        let remaining: Vec<_> = object_store
            .list(Some(&Path::from("vol/forks/f1")))
            .collect()
            .await;
        assert!(
            remaining.is_empty(),
            "fork's object-store prefix is empty after delete: {remaining:?}"
        );
        // ...while the sibling fork's objects are untouched.
        let sibling: Vec<_> = object_store
            .list(Some(&Path::from("vol/forks/f12")))
            .collect()
            .await;
        assert!(!sibling.is_empty(), "sibling fork's objects are intact");

        // Parent data is intact and readable.
        let (data, _) = parent_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"parent data");

        // The sibling fork still reads across the lineage.
        let (f12_fs, _f12_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f12")).await;
        let (data, _) = f12_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"parent data", "sibling still reads the parent");

        // The name is free again.
        fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();
        let mut names: Vec<String> = fork_manager
            .list_forks()
            .await
            .unwrap()
            .iter()
            .map(|f| f.name.clone())
            .collect();
        names.sort();
        assert_eq!(names, vec!["f1".to_string(), "f12".to_string()]);
    }

    /// Deleting a fork deletes the unnamed checkpoint its clone pinned in the
    /// parent's manifest — the pin that pauses the parent's segment
    /// reclamation — while the named branch-point checkpoint survives.
    #[tokio::test]
    async fn delete_fork_releases_the_pin_on_the_parent_manifest() {
        let (_parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;

        fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();

        let fork_admin =
            AdminBuilder::new(Path::from("vol/forks/f1"), Arc::clone(&object_store)).build();
        let manifest = fork_admin
            .read_manifest(None)
            .await
            .unwrap()
            .expect("fork manifest exists");
        let pin = manifest
            .external_dbs()
            .iter()
            .find(|db| db.path == "vol")
            .and_then(|db| db.final_checkpoint_id)
            .expect("fork pins a final checkpoint in the parent");

        let before = fork_manager.admin.list_checkpoints(None).await.unwrap();
        assert!(
            before.iter().any(|cp| cp.id == pin),
            "pin present in the parent's manifest before delete"
        );

        fork_manager.delete_fork("f1").await.unwrap();

        let after = fork_manager.admin.list_checkpoints(None).await.unwrap();
        assert!(
            !after.iter().any(|cp| cp.id == pin),
            "pin released by the delete"
        );
        assert!(
            after.iter().any(|cp| cp
                .name
                .as_deref()
                .is_some_and(|n| n.starts_with("fork-f1-"))),
            "the named branch-point checkpoint survives"
        );
    }

    #[tokio::test]
    async fn delete_fork_refuses_while_children_exist() {
        let (_parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;

        fork_manager
            .create_fork("f1", None, None, true)
            .await
            .unwrap();
        let (f1_fs, f1_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let (_, g_manager) = managers(
            f1_db.clone(),
            Path::from("vol/forks/f1"),
            Arc::clone(&object_store),
            &f1_fs,
        );
        g_manager.create_fork("g", None, None, true).await.unwrap();

        let err = fork_manager.delete_fork("f1").await.unwrap_err();
        assert!(
            err.to_string().contains("forks of its own"),
            "unexpected error: {err}"
        );

        // Children first, then the parent deletes cleanly.
        g_manager.delete_fork("g").await.unwrap();
        fork_manager.delete_fork("f1").await.unwrap();
        assert!(fork_manager.list_forks().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_fork_errors_for_an_unknown_fork() {
        let (_fs, fork_manager, _store, _path, _db) = new_parent_volume().await;
        let err = fork_manager.delete_fork("nope").await.unwrap_err();
        assert!(
            err.to_string().contains("not found"),
            "unexpected error: {err}"
        );
    }

    /// Lazy create registers the fork without materializing it: registry
    /// entry + pending marker only — no clone, no wrapped key, no fork
    /// database open, no parent epoch bump. The first open materializes.
    #[tokio::test]
    async fn lazy_fork_create_registers_then_materializes_on_first_open() {
        use futures::StreamExt;
        let (parent_fs, fork_manager, object_store, _parent_path, parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"hello.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"hello from parent"))
            .await
            .unwrap();
        // The lazy branch point is the durable manifest: publish it.
        parent_fs.flush_coordinator.flush().await.unwrap();

        let SlateDbHandle::ReadWrite(db) = &parent_db else {
            panic!("parent volume is writable");
        };
        let parent_epoch_before = db.subscribe().borrow().current_manifest.writer_epoch();

        let info = fork_manager
            .create_fork("f1", None, None, false)
            .await
            .unwrap();
        assert_eq!(info.base_epoch, 2);
        assert_eq!(info.ancestors.len(), 1);

        // Registered in the parent's LSM...
        let forks = fork_manager.list_forks().await.unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].name, "f1");

        // ...with a pending marker as the ONLY object under the fork's db
        // path: no clone manifests, no SSTs, no wrapped key...
        let objects: Vec<_> = object_store
            .list(Some(&Path::from("vol/forks/f1")))
            .collect()
            .await;
        assert_eq!(objects.len(), 1, "only the pending marker: {objects:?}");
        assert_eq!(
            objects[0].as_ref().unwrap().location,
            PendingFork::path(&Path::from("vol/forks/f1"))
        );

        // ...and the parent's writer epoch untouched (no fork database open).
        let parent_epoch_after = db.subscribe().borrow().current_manifest.writer_epoch();
        assert_eq!(parent_epoch_before, parent_epoch_after);

        // First open materializes: the fork reads the parent's file, and its
        // writes stay isolated.
        let (fork_fs, fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        let loaded = ForkInfo::load(&fork_db)
            .await
            .unwrap()
            .expect("lineage persisted in the fork's LSM");
        assert_eq!(loaded.base_epoch, 2);
        let (data, _) = fork_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"hello from parent");
        let (fork_file_id, _) = fork_fs
            .create(&creds, 0, b"fork-only.txt", &SetAttributes::default())
            .await
            .unwrap();
        fork_fs
            .write(&auth, fork_file_id, 0, &Bytes::from_static(b"fork write"))
            .await
            .unwrap();
        assert!(matches!(
            parent_fs.lookup(&creds, 0, b"fork-only.txt").await,
            Err(FsError::NotFound)
        ));

        // The marker is gone, and the clone now lives under the fork's path.
        let pending = PendingFork::load(&object_store, &Path::from("vol/forks/f1"))
            .await
            .unwrap();
        assert!(pending.is_none(), "materialization cleared the marker");
        let manifests: Vec<_> = object_store
            .list(Some(&Path::from("vol/forks/f1/manifest")))
            .collect()
            .await;
        assert!(!manifests.is_empty(), "materialization cloned the database");
    }

    /// Without the seal+flush barrier, the lazy branch point is the
    /// already-durable manifest: writes not yet flushed at create time are
    /// NOT part of the fork (the documented trade for a cheap create).
    #[tokio::test]
    async fn lazy_fork_create_branches_the_durable_state_without_a_barrier() {
        let (parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"unflushed.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"not yet durable"))
            .await
            .unwrap();
        // Deliberately NO flush: the durable manifest does not know the file.

        fork_manager
            .create_fork("f1", None, None, false)
            .await
            .unwrap();
        let (fork_fs, _fork_db) =
            open_volume(Arc::clone(&object_store), Path::from("vol/forks/f1")).await;
        assert!(matches!(
            fork_fs.lookup(&creds, 0, b"unflushed.txt").await,
            Err(FsError::NotFound)
        ));
    }

    /// Deleting a pending fork removes the registry entry, the pending
    /// marker, and the creation-owned source checkpoint — no slatedb
    /// database was ever built.
    #[tokio::test]
    async fn delete_pending_fork_removes_registry_marker_and_source_checkpoint() {
        use futures::StreamExt;
        let (_parent_fs, fork_manager, object_store, _parent_path, _parent_db) =
            new_parent_volume().await;

        fork_manager
            .create_fork("f1", None, None, false)
            .await
            .unwrap();
        let checkpoints = fork_manager.admin.list_checkpoints(None).await.unwrap();
        assert!(
            checkpoints.iter().any(|cp| cp
                .name
                .as_deref()
                .is_some_and(|n| n.starts_with("fork-f1-"))),
            "source checkpoint pinned at create"
        );

        fork_manager.delete_fork("f1").await.unwrap();

        assert!(fork_manager.list_forks().await.unwrap().is_empty());
        let remaining: Vec<_> = object_store
            .list(Some(&Path::from("vol/forks/f1")))
            .collect()
            .await;
        assert!(remaining.is_empty(), "pending fork's prefix is empty");
        let checkpoints = fork_manager.admin.list_checkpoints(None).await.unwrap();
        assert!(
            !checkpoints.iter().any(|cp| cp
                .name
                .as_deref()
                .is_some_and(|n| n.starts_with("fork-f1-"))),
            "creation-owned source checkpoint released"
        );

        // The name is free again.
        fork_manager
            .create_fork("f1", None, None, false)
            .await
            .unwrap();
        assert_eq!(fork_manager.list_forks().await.unwrap().len(), 1);
    }

    /// A pending fork cut from a user-named checkpoint must not take the
    /// user's checkpoint down with it.
    #[tokio::test]
    async fn delete_pending_fork_keeps_a_user_owned_source_checkpoint() {
        let (parent_fs, fork_manager, object_store, parent_path, parent_db) =
            new_parent_volume().await;
        let (checkpoint_manager, _) = managers(
            parent_db.clone(),
            parent_path.clone(),
            Arc::clone(&object_store),
            &parent_fs,
        );
        checkpoint_manager.create_checkpoint("mine").await.unwrap();

        fork_manager
            .create_fork("f1", Some("mine".to_string()), None, false)
            .await
            .unwrap();
        fork_manager.delete_fork("f1").await.unwrap();

        assert!(
            checkpoint_manager
                .get_checkpoint_info("mine")
                .await
                .unwrap()
                .is_some(),
            "user-owned source checkpoint survives"
        );
    }

    /// `materialize_fork_if_pending` is a no-op for volumes without a marker
    /// and idempotent once materialized.
    #[tokio::test]
    async fn materialize_fork_if_pending_noops_without_a_marker() {
        let (_parent_fs, fork_manager, object_store, parent_path, _parent_db) =
            new_parent_volume().await;
        let block_transformer: Arc<dyn slatedb::BlockTransformer> =
            ZeroFsBlockTransformer::try_new_arc(&TEST_KEY, CompressionConfig::default())
                .expect("test key should be lockable");

        // A non-fork volume has no marker.
        assert!(
            !materialize_fork_if_pending(
                &object_store,
                &parent_path,
                Arc::clone(&block_transformer)
            )
            .await
            .unwrap()
        );

        fork_manager
            .create_fork("f1", None, None, false)
            .await
            .unwrap();
        let fork_path = Path::from("vol/forks/f1");
        assert!(
            materialize_fork_if_pending(&object_store, &fork_path, Arc::clone(&block_transformer))
                .await
                .unwrap(),
            "first call materializes"
        );
        assert!(
            !materialize_fork_if_pending(&object_store, &fork_path, block_transformer)
                .await
                .unwrap(),
            "already materialized: no-op"
        );
    }

    /// In-process timing: lazy create does strictly less work than the
    /// barriered create (no clone, no key copy, no fork database open).
    #[tokio::test]
    async fn lazy_create_is_cheaper_than_barrier_create() {
        let (_parent_fs, fork_manager, _store, _path, _db) = new_parent_volume().await;

        let start = std::time::Instant::now();
        fork_manager
            .create_fork("lazy", None, None, false)
            .await
            .unwrap();
        let lazy = start.elapsed();

        let start = std::time::Instant::now();
        fork_manager
            .create_fork("eager", None, None, true)
            .await
            .unwrap();
        let eager = start.elapsed();

        eprintln!("create_fork timing: lazy={lazy:?} barrier={eager:?}");
        assert!(
            lazy < eager,
            "lazy create ({lazy:?}) skips the barrier create's ({eager:?}) clone and lineage write"
        );
    }
    /// Regression: a first open must materialize the clone and inherited key
    /// BEFORE the volume's encryption key is loaded; otherwise the open
    /// generates a fresh wrong key and the fork writes SSTs the parent's key
    /// cannot read (and cannot read the parent's SSTs).
    #[tokio::test]
    async fn storage_materialize_before_key_load_yields_the_parent_key() {
        let (parent_fs, fork_manager, object_store, parent_path, _parent_db) =
            new_parent_volume().await;
        let creds = test_creds();
        let auth = root_auth();

        let (file_id, _) = parent_fs
            .create(&creds, 0, b"hello.txt", &SetAttributes::default())
            .await
            .unwrap();
        parent_fs
            .write(&auth, file_id, 0, &Bytes::from_static(b"hello from parent"))
            .await
            .unwrap();
        // Lazy branch points capture the durable manifest: flush first so the
        // write is included (the barrier-free path's documented trade).
        parent_fs.flush_coordinator.flush().await.unwrap();

        fork_manager
            .create_fork("l1", None, None, false)
            .await
            .unwrap();
        let fork_path = Path::from("vol/forks/l1");

        // The production order: materialize storage (clone + inherited key
        // copy) BEFORE the encryption key is loaded.
        assert!(
            materialize_fork_storage_if_pending(&object_store, &fork_path)
                .await
                .unwrap()
        );
        let fork_key = key_management::load_wrapped_key_from_object_store(
            &object_store,
            &fork_path,
        )
        .await
        .unwrap()
        .expect("fork has the inherited wrapped key after storage materialization");
        let parent_key = key_management::load_wrapped_key_from_object_store(
            &object_store,
            &parent_path,
        )
        .await
        .unwrap()
        .expect("parent has its wrapped key");
        assert_eq!(
            bincode::serialize(&fork_key).unwrap(),
            bincode::serialize(&parent_key).unwrap(),
            "inherited wrapped key must be the parent's, byte for byte"
        );

        // Full materialization via the production open hook, then decode.
        let (fork_fs, _fork_db) = open_volume(Arc::clone(&object_store), fork_path.clone()).await;
        let (data, _) = fork_fs.read_file(&auth, file_id, 0, 1024).await.unwrap();
        assert_eq!(&data[..], b"hello from parent");
    }
}

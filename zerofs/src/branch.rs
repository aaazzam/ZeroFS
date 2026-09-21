//! Basin-branch registry: branch metadata as LSM rows in the volume's own
//! database.
//!
//! A basin branch is a fork-like namespace *inside* the volume's LSM rather
//! than a cloned database (see `docs/basin-lineage.md` and the
//! branch-dimension comment in [`crate::fs::key_codec`]). Creating one is
//! O(1): bump the branch-id allocator (a `System` counter key in the global
//! layout) and write one `KeyPrefix::Branch` registry row, both in a single
//! `WriteBatch`. No database clone, no manifest duplication, no object-store
//! writes beyond the LSM itself. Branch 0 (the volume root) is always
//! implicit and unlisted.
//!
//! Registry values follow the keyspace's versioning rules: a version byte
//! followed by JSON, like the fork records in [`crate::fork_info`].
//!
//! Deletion is two-phase: [`Db::delete_branch`] removes the registry row
//! (instant; the branch is gone from serving), [`Db::delete_branch_data`]
//! drops its key ranges (each scoped kind's `[kind || branch, kind ||
//! branch + 1)` slice plus its branch-tombstone rows). A branch's bytes are
//! key-space isolated, not SST-isolated: the dropped rows become dead bytes
//! in shared SSTs until compaction rewrites them. Segment OBJECTS under the
//! shared `segments/` namespace are not deleted here — their segcount rows
//! die with the range drop, and the orphan sweep reclaims the objects once
//! no live scope holds a counter for them.

use crate::db::{Db, SlateDbHandle};
use crate::fs::key_codec::{BranchId, KeyCodec, KeyPrefix};
use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use slatedb::WriteBatch;
use slatedb::config::WriteOptions;
use tokio_stream::StreamExt;

/// Version byte preceding the JSON payload of every branch-registry value.
/// Durable data: never reuse a lower number with a different layout.
const RECORD_VERSION: u8 = 1;

/// Persistent record of one basin branch, stored as the value of its
/// `KeyPrefix::Branch` registry key (`meta || BRANCH || name`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchRecord {
    /// The branch's key-space id; embedded in every scoped key the branch
    /// writes. Never reused, even after the branch is deleted.
    pub id: u32,
    /// Wall-clock epoch seconds at creation.
    pub created_at: u64,
}

impl BranchRecord {
    /// Encode for LSM storage: a version byte followed by JSON.
    pub fn encode(&self) -> Result<Bytes> {
        let json = serde_json::to_vec(self)?;
        let mut out = Vec::with_capacity(1 + json.len());
        out.push(RECORD_VERSION);
        out.extend_from_slice(&json);
        Ok(Bytes::from(out))
    }

    /// Decode a stored branch record, rejecting unknown versions rather than
    /// silently misreading a future layout.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (version, payload) = bytes
            .split_first()
            .ok_or_else(|| anyhow!("empty branch record"))?;
        if *version != RECORD_VERSION {
            return Err(anyhow!(
                "unsupported branch record version {version} (expected {RECORD_VERSION})"
            ));
        }
        serde_json::from_slice(payload).context("parsing branch record")
    }
}

impl Db {
    /// Create a basin branch: allocate a fresh id from the volume-level
    /// branch counter and write the registry row, atomically, in one batch.
    /// O(1) and milliseconds — no data is copied. Branch ids are never
    /// reused, so a branch deleted and re-created under the same name gets a
    /// new id.
    ///
    /// Like every other metadata mutation today, this assumes the volume's
    /// single writer: the counter's read-modify-write is not CAS-guarded.
    pub async fn create_branch(&self, name: &str) -> Result<BranchId> {
        if name.is_empty() {
            return Err(anyhow!("branch name must not be empty"));
        }
        let codec = KeyCodec::new();
        let counter_key = codec.branch_counter_key();
        let registry_key = codec.branch_registry_key(name);

        if self.get_bytes(&registry_key).await?.is_some() {
            return Err(anyhow!("branch {name:?} already exists"));
        }
        let last = self
            .get_bytes(&counter_key)
            .await?
            .map(|raw| {
                KeyCodec::decode_u64(&raw).ok_or_else(|| anyhow!("corrupt branch id counter"))
            })
            .transpose()?
            .unwrap_or(0);
        let id = last
            .checked_add(1)
            .filter(|&id| id <= u64::from(u32::MAX))
            .ok_or_else(|| anyhow!("branch id space exhausted"))?;

        let record = BranchRecord {
            id: id as u32,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        let mut batch = WriteBatch::new();
        batch.put_bytes(counter_key, KeyCodec::encode_u64(id));
        batch.put_bytes(registry_key, record.encode()?);
        self.write_with_options(batch, &WriteOptions::default()).await?;
        Ok(BranchId(id as u32))
    }

    /// Every registered basin branch as `(name, record)`, ordered by name.
    /// The volume root (branch 0) is implicit and never listed.
    pub async fn list_branches(&self) -> Result<Vec<(String, BranchRecord)>> {
        let codec = KeyCodec::new();
        let prefix = codec.branch_registry_prefix();
        let mut stream = self.scan_prefix(prefix.clone(), None, 64 * 1024).await?;
        let mut branches = Vec::new();
        while let Some(item) = stream.next().await {
            let (key, value) = item?;
            let name = std::str::from_utf8(&key[prefix.len()..])
                .context("branch registry key with non-UTF-8 name")?
                .to_string();
            branches.push((name, BranchRecord::decode(&value)?));
        }
        Ok(branches)
    }

    /// Resolve a registered branch name to its id, or `None` if no such
    /// branch exists. Branch 0 (the volume root) is implicit and never
    /// resolves through the registry.
    pub async fn resolve_branch(&self, name: &str) -> Result<Option<BranchId>> {
        let codec = KeyCodec::new();
        let key = codec.branch_registry_key(name);
        let Some(raw) = self.get_bytes(&key).await? else {
            return Ok(None);
        };
        Ok(Some(BranchId(BranchRecord::decode(&raw)?.id)))
    }

    /// Remove a branch's registry row. Returns `false` if no such branch was
    /// registered. This removes only the metadata; the branch's keys and
    /// tombstones remain in the LSM (orphaned from serving) until
    /// [`Self::delete_branch_data`] reclaims them.
    pub async fn delete_branch(&self, name: &str) -> Result<bool> {
        let codec = KeyCodec::new();
        let registry_key = codec.branch_registry_key(name);
        if self.get_bytes(&registry_key).await?.is_none() {
            return Ok(false);
        }
        let mut batch = WriteBatch::new();
        batch.delete(registry_key);
        self.write_with_options(batch, &WriteOptions::default()).await?;
        Ok(true)
    }

    /// Delete every key branch `id` owns: each branch-scoped kind's
    /// `[kind || branch, kind || branch + 1)` slice (see
    /// [`KeyCodec::prefix_range`]) plus the branch's tombstone rows
    /// (`BRANCH_TOMBSTONE || branch || ...`). The registry row and the id
    /// counter are untouched — pair this with [`Self::delete_branch`], which
    /// must run FIRST so no serving handle can still resolve the branch
    /// while its keys disappear. Returns the number of rows deleted.
    ///
    /// Scans are raw (never the branch-merged view), so this works from any
    /// handle, root or branch. Segment objects are deliberately left behind
    /// for the orphan sweep (see the module doc). Deleting the root's data
    /// is refused: branch 0 is the volume itself.
    pub async fn delete_branch_data(&self, id: BranchId) -> Result<u64> {
        if id.is_root() {
            return Err(anyhow!("refusing to delete the root branch's data"));
        }
        /// Keys per committed `WriteBatch` during the range drop. Bounds a
        /// single batch's memory without bounding total deleted work.
        const DELETE_BATCH_KEYS: usize = 4096;

        let codec = KeyCodec::for_branch(id);
        let mut ranges = Vec::with_capacity(8);
        for kind in [
            KeyPrefix::Inode,
            KeyPrefix::DirEntry,
            KeyPrefix::DirCookie,
            KeyPrefix::Tombstone,
            KeyPrefix::Orphan,
            KeyPrefix::SegCount,
            KeyPrefix::Extent,
        ] {
            ranges.push(codec.prefix_range(kind));
        }
        let tombstone_range = codec
            .branch_tombstone_range()
            .expect("non-root branches have a tombstone range");
        ranges.push(tombstone_range);

        let mut deleted = 0u64;
        for (start, end) in ranges {
            let mut stream = self.scan_raw(start..end).await?;
            let mut batch = WriteBatch::new();
            let mut staged = 0usize;
            while let Some(item) = stream.next().await {
                let (key, _) = item?;
                batch.delete(key);
                staged += 1;
                deleted += 1;
                if staged == DELETE_BATCH_KEYS {
                    self.write_with_options(batch, &WriteOptions::default())
                        .await?;
                    batch = WriteBatch::new();
                    staged = 0;
                }
            }
            if staged > 0 {
                self.write_with_options(batch, &WriteOptions::default())
                    .await?;
            }
        }
        Ok(deleted)
    }
}

/// Resolve `--branch <name>` against the branch registry of an opened
/// volume, for `zerofs run --branch`. The name must already exist — branch
/// creation goes through the admin RPC (`zerofs branch create`) on the
/// parent server, never implicitly at serve time.
pub async fn resolve_branch_for_serving(handle: &SlateDbHandle, name: &str) -> Result<BranchId> {
    let db = match handle.clone() {
        SlateDbHandle::ReadWrite(db) => Db::new(db, None),
        SlateDbHandle::ReadOnly(reader) => Db::new_read_only(reader),
    };
    db.resolve_branch(name)
        .await?
        .ok_or_else(|| {
            anyhow!(
                "branch {name:?} does not exist in this volume; create it with \
                 `zerofs branch create {name}` against the running parent server"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn open_db() -> Db {
        let store: Arc<dyn object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        Db::new(
            Arc::new(
                slatedb::DbBuilder::new(slatedb::object_store::path::Path::from("data"), store)
                    .build()
                    .await
                    .unwrap(),
            ),
            None,
        )
    }

    async fn put(db: &Db, key: &Bytes) {
        db.put_with_options(
            key,
            b"v",
            &slatedb::config::PutOptions::default(),
            &WriteOptions::default(),
        )
        .await
        .unwrap();
    }

    async fn has(db: &Db, key: &Bytes) -> bool {
        db.get_bytes(key).await.unwrap().is_some()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn resolve_branch_finds_registered_ids() {
        let db = open_db().await;
        assert_eq!(db.resolve_branch("nope").await.unwrap(), None);
        let id = db.create_branch("yes").await.unwrap();
        assert_eq!(db.resolve_branch("yes").await.unwrap(), Some(id));
        assert!(db.delete_branch("yes").await.unwrap());
        assert_eq!(db.resolve_branch("yes").await.unwrap(), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn delete_branch_data_drops_exactly_the_branchs_keys() {
        let db = open_db().await;
        let victim_id = db.create_branch("victim").await.unwrap();
        let sibling_id = db.create_branch("sibling").await.unwrap();
        let root = KeyCodec::new();
        let victim = KeyCodec::for_branch(victim_id);
        let sibling = KeyCodec::for_branch(sibling_id);

        // Seed rows in every scoped kind for the victim, the sibling, and
        // the root — same logical suffixes, so only scope distinguishes them.
        let mut victim_keys = vec![
            victim.inode_key(7).into(),
            victim.dir_entry_key(1, b"file"),
            victim.dir_cookie_counter_key(1),
            victim.tombstone_key(9, 7),
            victim.orphan_key(7),
            victim.segcount_key(3, 4),
            victim.extent_key(7, 0).into(),
        ];
        victim_keys.push(
            victim
                .branch_tombstone_key(victim.inode_key(7).as_ref())
                .unwrap(),
        );
        for key in &victim_keys {
            put(&db, key).await;
        }
        let keep_keys: Vec<Bytes> = vec![
            root.inode_key(7).into(),
            root.segcount_key(3, 4),
            root.extent_key(7, 0).into(),
            sibling.inode_key(7).into(),
            sibling.segcount_key(3, 4),
            sibling
                .branch_tombstone_key(sibling.inode_key(7).as_ref())
                .unwrap(),
        ];
        for key in &keep_keys {
            put(&db, key).await;
        }

        // The root's data is not deletable through this path.
        assert!(db.delete_branch_data(BranchId::ROOT).await.is_err());

        let deleted = db.delete_branch_data(victim_id).await.unwrap();
        assert_eq!(
            deleted,
            victim_keys.len() as u64,
            "every seeded victim row is dropped"
        );
        for key in &victim_keys {
            assert!(!has(&db, key).await, "victim key survived: {key:02x?}");
        }
        for key in &keep_keys {
            assert!(has(&db, key).await, "foreign key deleted: {key:02x?}");
        }

        // Idempotent: a second drop finds nothing.
        assert_eq!(db.delete_branch_data(victim_id).await.unwrap(), 0);
    }

    #[test]
    fn branch_record_encoding_round_trips_and_checks_version() {
        let record = BranchRecord {
            id: 7,
            created_at: 1_700_000_000,
        };
        let encoded = record.encode().unwrap();
        assert_eq!(BranchRecord::decode(&encoded).unwrap(), record);

        assert!(BranchRecord::decode(&[]).is_err(), "empty record rejected");
        let mut future = encoded.to_vec();
        future[0] = RECORD_VERSION + 1;
        assert!(
            BranchRecord::decode(&future).is_err(),
            "unknown version rejected"
        );
        assert!(
            BranchRecord::decode(&encoded[..1]).is_err(),
            "version byte without payload rejected"
        );
    }
}

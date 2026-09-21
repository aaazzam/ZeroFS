//! Fork lineage metadata, persisted as first-class records in the LSM.
//!
//! A fork is a writable clone of its parent volume's LSM (see
//! [`crate::fork_manager`]). Because segment keys embed the writer epoch
//! (`segments/{shard}/{epoch}/{counter}`), the fork's read path can tell which
//! volume wrote any segment a `FrameLoc` refers to: epochs at or above
//! `base_epoch` are the fork's own writes, anything older belongs to the
//! nearest ancestor whose `base_epoch` does not exceed it. [`ForkInfo`] is the
//! record that makes that routing decision durable across restarts and across
//! forks-of-forks.
//!
//! The record lives in two LSMs, under the keyspace's own versioning rules
//! (see [`crate::fs::key_codec`]):
//!
//! - **Lineage** — a single `KeyPrefix::ForkLineage` record in the fork's OWN
//!   database. Written when the fork materializes (eager forks: at creation;
//!   lazy forks: at the fork's first open — see [`crate::fork_manager`] for
//!   the two-phase lifecycle), read back on every startup of the fork to
//!   build the segment path router. A non-fork volume has no record.
//! - **Registry** — a `KeyPrefix::ForkRegistry/<fork name>` record per fork in
//!   the PARENT's database, so `list_forks` is a prefix scan of the parent's
//!   own LSM. It is written strictly after the lineage record (eager) or the
//!   pending-materialization marker (lazy): a crash can leave a fork whose
//!   lineage/marker exists but no registry entry (an unlisted fork), never a
//!   registry entry pointing at a fork whose materialization state is
//!   missing.
//!
//! Both values are a version byte followed by JSON, so a future format change
//! bumps the version and migrates reads.

use crate::db::SlateDbHandle;
use crate::fs::key_codec::KeyCodec;
use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use slatedb::config::{DurabilityLevel, PutOptions, ReadOptions, WriteOptions};

/// Infix under a volume's db path holding its direct forks.
pub const FORKS_INFIX: &str = "forks";

/// Version byte preceding the JSON payload of every fork record. Durable
/// data: never reuse a lower number with a different layout.
const RECORD_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkAncestor {
    pub db_path: String,
    pub base_epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkInfo {
    pub name: String,
    pub parent_db_path: String,
    /// First writer epoch of this fork. Segments written by the fork itself
    /// have `epoch >= base_epoch`; older epochs belong to ancestors.
    pub base_epoch: u64,
    pub created_at: u64,
    /// Ancestors ordered root-first; the last entry is the direct parent.
    pub ancestors: Vec<ForkAncestor>,
}

impl ForkInfo {
    pub fn db_path(parent_db_path: &str, name: &str) -> String {
        format!("{parent_db_path}/{FORKS_INFIX}/{name}")
    }

    /// Encode for LSM storage: a version byte followed by JSON.
    pub fn encode(&self) -> Result<Bytes> {
        let json = serde_json::to_vec(self)?;
        let mut out = Vec::with_capacity(1 + json.len());
        out.push(RECORD_VERSION);
        out.extend_from_slice(&json);
        Ok(Bytes::from(out))
    }

    /// Decode a stored fork record, rejecting unknown versions rather than
    /// silently misreading a future layout.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let (version, payload) = bytes
            .split_first()
            .ok_or_else(|| anyhow!("empty fork record"))?;
        if *version != RECORD_VERSION {
            return Err(anyhow!(
                "unsupported fork record version {version} (expected {RECORD_VERSION})"
            ));
        }
        serde_json::from_slice(payload).context("parsing fork record")
    }

    /// Read this volume's own lineage record; `None` for a non-fork volume.
    pub async fn load(db: &SlateDbHandle) -> Result<Option<Self>> {
        let key = KeyCodec::new().fork_lineage_key();
        let read_options = ReadOptions {
            durability_filter: DurabilityLevel::Memory,
            cache_blocks: true,
            ..Default::default()
        };
        let value = match db {
            SlateDbHandle::ReadWrite(db) => db.get_with_options(&key, &read_options).await?,
            SlateDbHandle::ReadOnly(reader) => {
                reader.load().get_with_options(&key, &read_options).await?
            }
        };
        value.map(|bytes| Self::decode(&bytes)).transpose()
    }

    /// Write this fork's lineage record into its own (freshly cloned)
    /// database. The caller must flush before closing: the record has to be
    /// durable before the parent's registry entry is published (see
    /// [`crate::fork_manager`]).
    pub async fn save(&self, db: &slatedb::Db) -> Result<()> {
        let key = KeyCodec::new().fork_lineage_key();
        db.put_with_options(
            &key,
            &self.encode()?,
            &PutOptions::default(),
            &WriteOptions::default(),
        )
        .await
        .map_err(|e| anyhow!("failed to write fork lineage record: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_info() -> ForkInfo {
        ForkInfo {
            name: "agent-1".to_string(),
            parent_db_path: "vol".to_string(),
            base_epoch: 4,
            created_at: 1_700_000_000,
            ancestors: vec![ForkAncestor {
                db_path: "vol".to_string(),
                base_epoch: 0,
            }],
        }
    }

    #[test]
    fn fork_record_encoding_round_trips_and_checks_version() {
        let info = test_info();
        let encoded = info.encode().unwrap();
        let loaded = ForkInfo::decode(&encoded).unwrap();
        assert_eq!(loaded.name, "agent-1");
        assert_eq!(loaded.base_epoch, 4);
        assert_eq!(loaded.ancestors.len(), 1);

        assert!(ForkInfo::decode(&[]).is_err(), "empty record rejected");
        let mut future = encoded.to_vec();
        future[0] = RECORD_VERSION + 1;
        assert!(
            ForkInfo::decode(&future).is_err(),
            "unknown version rejected"
        );
    }

    #[tokio::test]
    async fn fork_info_round_trip_through_the_lsm() {
        let store: Arc<dyn slatedb::object_store::ObjectStore> =
            Arc::new(slatedb::object_store::memory::InMemory::new());
        let path = slatedb::object_store::path::Path::from("vol/forks/agent-1");

        // Write the lineage record and close, then prove it survives a reopen.
        {
            let db = slatedb::DbBuilder::new(path.clone(), Arc::clone(&store))
                .build()
                .await
                .unwrap();
            test_info().save(&db).await.unwrap();
            db.flush().await.unwrap();
            db.close().await.unwrap();
        }
        {
            let db = slatedb::DbBuilder::new(path, Arc::clone(&store))
                .build()
                .await
                .unwrap();
            let handle = SlateDbHandle::ReadWrite(Arc::new(db));
            let loaded = ForkInfo::load(&handle)
                .await
                .unwrap()
                .expect("lineage record present");
            assert_eq!(loaded.name, "agent-1");
            assert_eq!(loaded.base_epoch, 4);
            assert_eq!(loaded.ancestors.len(), 1);
            handle_close(handle).await;
        }

        // A non-fork volume has no lineage record.
        let db = slatedb::DbBuilder::new(
            slatedb::object_store::path::Path::from("vol"),
            Arc::clone(&store),
        )
        .build()
        .await
        .unwrap();
        let handle = SlateDbHandle::ReadWrite(Arc::new(db));
        assert!(
            ForkInfo::load(&handle).await.unwrap().is_none(),
            "non-fork volume has no lineage record"
        );
        handle_close(handle).await;
    }

    async fn handle_close(handle: SlateDbHandle) {
        if let SlateDbHandle::ReadWrite(db) = handle {
            db.close().await.unwrap();
        }
    }
}

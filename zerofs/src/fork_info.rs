//! Fork lineage metadata, persisted per fork at `<fork db path>/.zerofs_fork.json`.
//!
//! A fork is a writable clone of its parent volume's LSM (see
//! [`crate::fork_manager`]). Because segment keys embed the writer epoch
//! (`segments/{shard}/{epoch}/{counter}`), the fork's read path can tell which
//! volume wrote any segment a `FrameLoc` refers to: epochs at or above
//! `base_epoch` are the fork's own writes, anything older belongs to the
//! nearest ancestor whose `base_epoch` does not exceed it. [`ForkInfo`] is the
//! record that makes that routing decision durable across restarts and across
//! forks-of-forks.

use anyhow::{Context, Result, anyhow};
use object_store::{ObjectStore, ObjectStoreExt};
use serde::{Deserialize, Serialize};
use slatedb::object_store::path::Path;
use std::sync::Arc;

pub const FORK_INFO_FILENAME: &str = ".zerofs_fork.json";
/// Infix under a volume's db path holding its direct forks.
pub const FORKS_INFIX: &str = "forks";

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

    fn info_path(db_path: &str) -> Path {
        Path::from(format!("{db_path}/{FORK_INFO_FILENAME}"))
    }

    /// Load this volume's fork metadata, or `None` when it is not a fork.
    pub async fn load(object_store: &Arc<dyn ObjectStore>, db_path: &str) -> Result<Option<Self>> {
        match object_store.get(&Self::info_path(db_path)).await {
            Ok(result) => {
                let bytes = result.bytes().await?;
                let info = serde_json::from_slice(&bytes).context("parsing fork info")?;
                Ok(Some(info))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(anyhow!("failed to read fork info: {e}")),
        }
    }

    pub async fn save(&self, object_store: &Arc<dyn ObjectStore>, db_path: &str) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(self)?;
        object_store
            .put(&Self::info_path(db_path), bytes.into())
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[tokio::test]
    async fn fork_info_round_trip() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let info = ForkInfo {
            name: "agent-1".to_string(),
            parent_db_path: "vol".to_string(),
            base_epoch: 4,
            created_at: 1_700_000_000,
            ancestors: vec![ForkAncestor {
                db_path: "vol".to_string(),
                base_epoch: 0,
            }],
        };
        info.save(&store, "vol/forks/agent-1").await.unwrap();

        let loaded = ForkInfo::load(&store, "vol/forks/agent-1")
            .await
            .unwrap()
            .expect("fork info present");
        assert_eq!(loaded.name, "agent-1");
        assert_eq!(loaded.base_epoch, 4);
        assert_eq!(loaded.ancestors.len(), 1);
        assert!(
            ForkInfo::load(&store, "vol").await.unwrap().is_none(),
            "non-fork volume has no fork info"
        );
    }
}

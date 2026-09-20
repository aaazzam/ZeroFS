//! Writable forks: branch a running volume's state into an independent,
//! writable clone that shares all of the parent's existing objects.
//!
//! A fork is created in three steps:
//!
//! 1. A durable checkpoint pins the parent's state (existing
//!    [`CheckpointManager`] semantics: seal + flush under the barrier first).
//! 2. SlateDB's clone builds a new, writable database at the fork's db path,
//!    referencing the parent's SSTs as `external_ssts` — a shallow,
//!    O(manifest) copy-on-write branch. The clone pins its source on the
//!    parent's manifest, which also pauses reclamation of the parent's
//!    referenced segments (same protection a persistent checkpoint gives).
//! 3. The wrapped encryption key is copied to the fork's db path (keys are
//!    per-path) and a [`ForkInfo`] lineage record is written, which the fork's
//!    startup uses to route segment reads across the lineage (see
//!    [`crate::segment_path_router`]).
//!
//! The fork is then served like any other volume: point a `[storage] url` at
//! the fork's db path and `zerofs run`.

use crate::checkpoint_manager::CheckpointManager;
use crate::db::SlateDbHandle;
use crate::fork_info::{ForkAncestor, ForkInfo};
use crate::key_management;
use anyhow::{Context, Result, anyhow};
use object_store::{ObjectStore, ObjectStoreExt};
use slatedb::admin::Admin;
use slatedb::admin::AdminBuilder;
use slatedb::admin::CloneSourceSpec;
use slatedb::object_store::path::Path;
use std::sync::Arc;
use uuid::Uuid;

pub struct ForkManager {
    db_handle: SlateDbHandle,
    parent_db_path: Path,
    object_store: Arc<dyn ObjectStore>,
    checkpoint_manager: Arc<CheckpointManager>,
    admin: Admin,
}

impl ForkManager {
    pub fn new(
        db_handle: SlateDbHandle,
        parent_db_path: Path,
        object_store: Arc<dyn ObjectStore>,
        checkpoint_manager: Arc<CheckpointManager>,
    ) -> Self {
        let admin = AdminBuilder::new(parent_db_path.clone(), Arc::clone(&object_store)).build();
        Self {
            db_handle,
            parent_db_path,
            object_store,
            checkpoint_manager,
            admin,
        }
    }

    /// Create a writable fork named `name` from `from_checkpoint` (or from a
    /// fresh checkpoint of the current durable state when `None`).
    pub async fn create_fork(&self, name: &str, from_checkpoint: Option<String>) -> Result<ForkInfo> {
        let name = name.trim();
        validate_fork_name(name)?;

        if matches!(&self.db_handle, SlateDbHandle::ReadOnly(_)) {
            return Err(anyhow!(
                "Cannot create forks in read-only mode. Start the server without --read-only or --checkpoint flags."
            ));
        }

        let fork_db_path = ForkInfo::db_path(self.parent_db_path.as_ref(), name);
        if ForkInfo::load(&self.object_store, &fork_db_path)
            .await?
            .is_some()
        {
            return Err(anyhow!("A fork named '{}' already exists", name));
        }

        // Resolve the branch point: an existing named checkpoint, or a fresh
        // one so the fork starts from a consistent durable cut of HEAD. The
        // fork's first writable open bumps the writer epoch it inherits from
        // the checkpoint's manifest, so its own segments start one epoch above
        // that manifest's epoch — which is NOT necessarily the live parent's
        // current epoch when forking an older checkpoint.
        let checkpoint = match from_checkpoint.as_deref().map(str::trim) {
            Some("") | None => {
                let checkpoint_name = format!("fork-{name}-{}", Uuid::new_v4().simple());
                self.checkpoint_manager
                    .create_checkpoint(&checkpoint_name)
                    .await?
            }
            Some(checkpoint_name) => self
                .checkpoint_manager
                .get_checkpoint_info(checkpoint_name)
                .await?
                .ok_or_else(|| anyhow!("Checkpoint '{}' not found", checkpoint_name))?,
        };
        let checkpoints = self
            .admin
            .list_checkpoints(None)
            .await
            .map_err(|e| anyhow!("Failed to list checkpoints: {}", e))?;
        let source = checkpoints
            .into_iter()
            .find(|cp| cp.id == checkpoint.id)
            .ok_or_else(|| anyhow!("Checkpoint '{}' no longer exists", checkpoint.name))?;
        let source_manifest = self
            .admin
            .read_manifest(Some(source.manifest_id))
            .await
            .map_err(|e| anyhow!("Failed to read checkpoint manifest: {}", e))?
            .ok_or_else(|| anyhow!("Checkpoint manifest not found"))?;
        let base_epoch = source_manifest.writer_epoch() + 1;

        let parent_info =
            ForkInfo::load(&self.object_store, self.parent_db_path.as_ref()).await?;
        let mut ancestors = parent_info
            .as_ref()
            .map(|info| info.ancestors.clone())
            .unwrap_or_default();
        ancestors.push(ForkAncestor {
            db_path: self.parent_db_path.as_ref().to_string(),
            base_epoch: parent_info.map(|info| info.base_epoch).unwrap_or(0),
        });

        // Shallow-copy the metadata database. External SSTs stay referenced
        // from the parent's path; nothing moves.
        let admin = AdminBuilder::new(
            Path::from(fork_db_path.clone()),
            Arc::clone(&self.object_store),
        )
        .build();
        admin
            .create_clone_builder_from_source(CloneSourceSpec::with_checkpoint(
                self.parent_db_path.clone(),
                checkpoint.id,
            ))
            .build()
            .await
            .map_err(|e| anyhow!("Failed to clone database for fork '{}': {}", name, e))?;

        // Encryption keys are per db path; the fork reads the parent's
        // segments and SSTs, so it needs the same wrapped key.
        key_management::copy_wrapped_key(
            &self.object_store,
            &self.parent_db_path,
            &Path::from(fork_db_path.clone()),
        )
        .await
        .context("Failed to copy encryption key to fork")?;

        let info = ForkInfo {
            name: name.to_string(),
            parent_db_path: self.parent_db_path.as_ref().to_string(),
            base_epoch,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            ancestors,
        };
        info.save(&self.object_store, &fork_db_path).await?;

        Ok(info)
    }

    /// List the direct forks of this volume.
    pub async fn list_forks(&self) -> Result<Vec<ForkInfo>> {
        let prefix = Path::from(format!("{}/{}", self.parent_db_path, crate::fork_info::FORKS_INFIX));
        let mut stream = self.object_store.list(Some(&prefix));
        let mut forks = Vec::new();
        {
            use futures::TryStreamExt;
            while let Some(meta) = stream.try_next().await? {
                if !meta.location.as_ref().ends_with(crate::fork_info::FORK_INFO_FILENAME) {
                    continue;
                }
                let bytes = self
                    .object_store
                    .get(&meta.location)
                    .await?
                    .bytes()
                    .await?;
                forks.push(
                    serde_json::from_slice::<ForkInfo>(&bytes)
                        .context("parsing fork info from object store")?,
                );
            }
        }
        forks.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.name.cmp(&b.name)));
        Ok(forks)
    }
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
    use crate::fs::types::SetAttributes;
    use crate::fs::permissions::Credentials;
    use crate::segment_path_router::SegmentPathRouter;
    use bytes::Bytes;
    use crate::fs::types::AuthContext;
    use object_store::memory::InMemory;
    use slatedb::DbBuilder;

    const TEST_KEY: [u8; 32] = [7u8; 32];
    const TEST_PASSWORD: &str = "fork-test-password";

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
    /// segment reads through a SegmentPathRouter rooted at the volume's db
    /// path, ancestor lineage taken from `fork_info` when present.
    async fn open_volume(
        object_store: Arc<dyn ObjectStore>,
        db_path: Path,
        fork_info: Option<ForkInfo>,
    ) -> (Arc<ZeroFS>, SlateDbHandle) {
        let block_transformer: Arc<dyn slatedb::BlockTransformer> =
            ZeroFsBlockTransformer::try_new_arc(&TEST_KEY, CompressionConfig::default())
                .expect("test key should be lockable");
        let slatedb = Arc::new(
            DbBuilder::new(db_path.clone(), Arc::clone(&object_store))
                .with_block_transformer(block_transformer)
                .with_filter_policies(crate::fs::filter_policy::filter_policies())
                .with_segment_extractor(Arc::new(crate::segment_extractor::ZeroFsSegmentExtractor))
                .build()
                .await
                .unwrap(),
        );
        let db_handle = SlateDbHandle::ReadWrite(slatedb);
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
        let fork_manager = Arc::new(ForkManager::new(
            db_handle,
            db_path,
            object_store,
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
        let (fs, db_handle) = open_volume(Arc::clone(&object_store), db_path.clone(), None).await;
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

        let info = fork_manager.create_fork("f1", None).await.unwrap();
        assert_eq!(info.base_epoch, 2, "fork starts one epoch above the parent");
        assert_eq!(info.ancestors.len(), 1);

        let loaded = ForkInfo::load(&object_store, "vol/forks/f1")
            .await
            .unwrap()
            .expect("fork info persisted");
        assert_eq!(loaded.base_epoch, 2);

        let forks = fork_manager.list_forks().await.unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].name, "f1");

        let (fork_fs, _fork_db) = open_volume(
            Arc::clone(&object_store),
            Path::from("vol/forks/f1"),
            Some(loaded),
        )
        .await;

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
        let (data, _) = fork_fs.read_file(&auth, fork_file_id, 0, 1024).await.unwrap();
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
        fork_manager.create_fork("f1", None).await.unwrap();
        let f1_info = ForkInfo::load(&object_store, "vol/forks/f1")
            .await
            .unwrap()
            .unwrap();
        let (f1_fs, f1_db) = open_volume(
            Arc::clone(&object_store),
            Path::from("vol/forks/f1"),
            Some(f1_info),
        )
        .await;
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
        let g_info = g_manager.create_fork("g", None).await.unwrap();
        assert_eq!(g_info.base_epoch, 3);
        assert_eq!(g_info.ancestors.len(), 2);

        let g_loaded = ForkInfo::load(&object_store, "vol/forks/f1/forks/g")
            .await
            .unwrap()
            .unwrap();
        let (g_fs, _g_db) = open_volume(
            Arc::clone(&object_store),
            Path::from("vol/forks/f1/forks/g"),
            Some(g_loaded),
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
        let (parent_fs, parent_db) =
            open_volume(Arc::clone(&object_store), db_path.clone(), None).await;
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
        let (parent_fs, parent_db) =
            open_volume(Arc::clone(&object_store), db_path.clone(), None).await;
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
            .create_fork("past", Some("old".to_string()))
            .await
            .unwrap();
        assert_eq!(
            info.base_epoch, 2,
            "fork of an epoch-1 checkpoint owns epoch 2 onward"
        );

        // The fork sees the old state, not file B, and its own writes at
        // epoch 2 route to itself (not to the parent).
        let fork_info = ForkInfo::load(&object_store, "vol/forks/past")
            .await
            .unwrap()
            .unwrap();
        let (fork_fs, _fork_db) = open_volume(
            Arc::clone(&object_store),
            Path::from("vol/forks/past"),
            Some(fork_info),
        )
        .await;
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
}

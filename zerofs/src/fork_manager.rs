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
//!    per-path), the fork's freshly cloned database is opened once to write
//!    its [`ForkInfo`] lineage record into its own LSM (durable before the
//!    next step), and a registry entry for the fork is written into the
//!    parent's LSM (see [`crate::fork_info`] for the two record types and the
//!    crash-ordering guarantee).
//!
//! The fork is then served like any other volume: point a `[storage] url` at
//! the fork's db path and `zerofs run`; startup reads the lineage back from
//! the fork's LSM to route segment reads across ancestors (see
//! [`crate::segment_path_router`]).

use crate::checkpoint_manager::CheckpointManager;
use crate::db::SlateDbHandle;
use crate::fork_info::{FORKS_INFIX, ForkAncestor, ForkInfo};
use crate::fs::key_codec::KeyCodec;
use crate::key_management;
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use object_store::ObjectStore;
use slatedb::admin::Admin;
use slatedb::admin::AdminBuilder;
use slatedb::admin::CloneSourceSpec;
use slatedb::config::{CheckpointOptions, PutOptions, WriteOptions};
use slatedb::object_store::path::Path;
use std::sync::Arc;
use uuid::Uuid;

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
    pub async fn create_fork(
        &self,
        name: &str,
        from_checkpoint: Option<String>,
        at: Option<DateTime<Utc>>,
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

        // Resolve the branch point: an existing named checkpoint, a manifest
        // chosen by timestamp (point-in-time fork), or a fresh checkpoint of
        // the current durable state. The fork's first writable open bumps the
        // writer epoch it inherits from the source manifest, so its own
        // segments start one epoch above that manifest's epoch — which is NOT
        // necessarily the live parent's current epoch when forking an older
        // state.
        let (checkpoint_id, base_epoch) = if let Some(at) = at {
            let manifest_id = self.manifest_at_time(at).await?;
            let result = self
                .admin
                .create_detached_checkpoint_at(
                    manifest_id,
                    &CheckpointOptions {
                        lifetime: None,
                        source: None,
                        name: Some(format!("pitr-{name}-{}", at.timestamp())),
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
            (result.id, manifest.writer_epoch() + 1)
        } else {
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
            (checkpoint.id, source_manifest.writer_epoch() + 1)
        };

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
                checkpoint_id,
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

        // Open the freshly cloned fork database once to persist the lineage
        // record into the fork's own LSM. Opened with the parent's durability
        // posture (no WAL; explicit flush) so the record is SST-durable before
        // the registry entry below publishes the fork. This open consumes the
        // fork's first writer epoch (base_epoch), so the fork's serving open
        // writes segments at base_epoch + 1 — routing compares `>= base_epoch`,
        // so the gap is harmless.
        {
            let settings = slatedb::config::Settings {
                wal_enabled: false,
                flush_interval: None,
                compactor_options: None,
                garbage_collector_options: None,
                compression_codec: None, // handled by the block transformer
                ..Default::default()
            };
            let fork_db = slatedb::DbBuilder::new(
                Path::from(fork_db_path.clone()),
                Arc::clone(&self.object_store),
            )
            .with_settings(settings)
            .with_block_transformer(Arc::clone(&self.block_transformer))
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
        }

        // Registry entry LAST, into the parent's LSM: a crash can now leave a
        // fork whose lineage is durable but unlisted (an orphan the operator
        // can re-register), never a registry entry for a fork whose lineage
        // record is missing.
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

        let info = fork_manager.create_fork("f1", None, None).await.unwrap();
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
        fork_manager.create_fork("f1", None, None).await.unwrap();
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
        let g_info = g_manager.create_fork("g", None, None).await.unwrap();
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
            .create_fork("past", Some("old".to_string()), None)
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
            .create_fork("pit", None, Some(target))
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
                .create_fork("too-early", None, Some(chrono::DateTime::UNIX_EPOCH))
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

        fork_manager.create_fork("f1", None, None).await.unwrap();
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

}

//! Segment read-path routing across fork lineage.
//!
//! ZeroFS namespaces segment objects under the owning database's db path via a
//! [`object_store::prefix::PrefixStore`], so sibling databases never collide
//! on `segments/{shard}/{epoch}/{counter}` keys. A fork's LSM, however,
//! contains `FrameLoc`s written by its ancestors — and those objects live
//! under the *ancestors'* db paths.
//!
//! [`SegmentPathRouter`] is an [`ObjectStore`] decorator that resolves each
//! `segments/` key to the volume that wrote it, using the writer epoch encoded
//! in the key: an epoch at or above the fork's base epoch is the fork's own
//! write; anything older belongs to the nearest ancestor whose base epoch does
//! not exceed it. Writes, deletes, and lists always stay on the local volume,
//! so a fork's reclamation can never touch an ancestor's objects.

use crate::fork_info::ForkInfo;
use futures::stream::{BoxStream, TryStreamExt};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

/// Routes `segments/` object keys to the db path of the volume that wrote
/// them; every other path resolves under the local volume's db path.
pub struct SegmentPathRouter {
    inner: Arc<dyn ObjectStore>,
    own_prefix: Path,
    own_base_epoch: u64,
    /// `(base_epoch, db_path)` pairs, sorted ascending by base epoch.
    ancestors: Vec<(u64, Path)>,
}

impl SegmentPathRouter {
    /// Equivalent to [`object_store::prefix::PrefixStore`] when `fork_info` is
    /// `None` (a non-fork volume): everything resolves under `own_prefix`.
    pub fn new(inner: Arc<dyn ObjectStore>, own_prefix: Path, fork_info: Option<&ForkInfo>) -> Self {
        let (own_base_epoch, ancestors) = match fork_info {
            Some(info) => {
                let mut ancestors: Vec<(u64, Path)> = info
                    .ancestors
                    .iter()
                    .map(|a| (a.base_epoch, Path::from(a.db_path.clone())))
                    .collect();
                ancestors.sort_by_key(|(base_epoch, _)| *base_epoch);
                (info.base_epoch, ancestors)
            }
            None => (0, Vec::new()),
        };
        Self {
            inner,
            own_prefix,
            own_base_epoch,
            ancestors,
        }
    }

    /// The db path that owns the segment written at `epoch`.
    fn owner_prefix(&self, epoch: u64) -> &Path {
        if epoch >= self.own_base_epoch {
            return &self.own_prefix;
        }
        // Ancestors are sorted ascending: the writer of `epoch` is the last
        // ancestor whose base epoch does not exceed it. Older than every
        // recorded base epoch means the root ancestor wrote it.
        let mut owner = self.ancestors.first().map(|(_, path)| path);
        for (base_epoch, path) in &self.ancestors {
            if *base_epoch > epoch {
                break;
            }
            owner = Some(path);
        }
        owner.unwrap_or(&self.own_prefix)
    }

    /// The epoch encoded in a `segments/{shard}/{epoch}/{counter}` key.
    fn key_epoch(path: &Path) -> Option<u64> {
        let mut parts = path.parts();
        match parts.next() {
            Some(first) if first.as_ref() == "segments" => {}
            _ => return None,
        }
        let _shard = parts.next()?;
        let epoch = parts.next()?;
        u64::from_str_radix(epoch.as_ref(), 16).ok()
    }

    fn resolve(&self, path: &Path) -> Path {
        let prefix = match Self::key_epoch(path) {
            Some(epoch) => self.owner_prefix(epoch),
            None => &self.own_prefix,
        };
        join(prefix, path)
    }

    fn local(&self, path: &Path) -> Path {
        join(&self.own_prefix, path)
    }
}

fn join(prefix: &Path, path: &Path) -> Path {
    let mut out = prefix.clone();
    for part in path.parts() {
        out = out.join(part.as_ref());
    }
    out
}

impl Debug for SegmentPathRouter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SegmentPathRouter({}, base_epoch={}, ancestors={})",
            self.own_prefix,
            self.own_base_epoch,
            self.ancestors.len()
        )
    }
}

impl Display for SegmentPathRouter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[async_trait::async_trait]
impl ObjectStore for SegmentPathRouter {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner
            .put_opts(&self.local(location), payload, opts)
            .await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner
            .put_multipart_opts(&self.local(location), opts)
            .await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(&self.resolve(location), options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        let own_prefix = self.own_prefix.clone();
        let locations = locations.map_ok(move |path| join(&own_prefix, &path));
        self.inner
            .delete_stream(Box::pin(locations) as BoxStream<'static, _>)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let empty = Path::default();
        let local = self.local(prefix.unwrap_or(&empty));
        self.inner.list(Some(&local))
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        let empty = Path::default();
        let local = self.local(prefix.unwrap_or(&empty));
        self.inner.list_with_delimiter(Some(&local)).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner
            .copy_opts(&self.local(from), &self.local(to), options)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fork_info::{ForkAncestor, ForkInfo};
    use bytes::Bytes;
    use object_store::ObjectStoreExt;
    use object_store::memory::InMemory;

    fn router(
        own_base_epoch: u64,
        ancestors: Vec<(u64, &str)>,
    ) -> (Arc<dyn ObjectStore>, SegmentPathRouter) {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let info = ForkInfo {
            name: "fork".to_string(),
            parent_db_path: ancestors
                .last()
                .map(|(_, p)| p.to_string())
                .unwrap_or_default(),
            base_epoch: own_base_epoch,
            created_at: 0,
            ancestors: ancestors
                .iter()
                .map(|(base_epoch, db_path)| ForkAncestor {
                    db_path: db_path.to_string(),
                    base_epoch: *base_epoch,
                })
                .collect(),
        };
        let router = SegmentPathRouter::new(
            store.clone(),
            Path::from("vol/forks/fork"),
            Some(&info),
        );
        (store, router)
    }

    fn seg_key(epoch: u64, counter: u64) -> Path {
        Path::from(format!(
            "segments/{:02x}/{:016x}/{:016x}",
            counter & 0xff, epoch, counter
        ))
    }

    #[tokio::test]
    async fn reads_route_by_epoch_writes_stay_local() {
        let (store, router) = router(5, vec![(0, "vol"), (3, "vol/forks/a")]);

        // Writes always land under the local prefix.
        router
            .put(&seg_key(6, 1), Bytes::from_static(b"mine").into())
            .await
            .unwrap();
        let local = Path::from(format!("vol/forks/fork/{}", seg_key(6, 1)));
        assert!(store.head(&local).await.is_ok());

        // Ancestor objects written by hand resolve by epoch.
        for (epoch, owner) in [(1, "vol"), (3, "vol/forks/a"), (4, "vol/forks/a")] {
            let path = Path::from(format!("{}/{}", owner, seg_key(epoch, 7)));
            store
                .put(&path, Bytes::from_static(b"ancestor").into())
                .await
                .unwrap();
            let got = router.get(&seg_key(epoch, 7)).await.unwrap().bytes().await.unwrap();
            assert_eq!(got, Bytes::from_static(b"ancestor"), "epoch {epoch}");
        }
    }

    #[tokio::test]
    async fn non_fork_routes_everything_local() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let router = SegmentPathRouter::new(store.clone(), Path::from("vol"), None);
        router
            .put(&seg_key(9, 1), Bytes::from_static(b"x").into())
            .await
            .unwrap();
        let got = router.get(&seg_key(9, 1)).await.unwrap().bytes().await.unwrap();
        assert_eq!(got, Bytes::from_static(b"x"));
    }
}

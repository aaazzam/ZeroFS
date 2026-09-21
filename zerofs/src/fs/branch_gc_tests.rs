//! Filesystem-level proof of branch-aware garbage collection over one shared
//! volume: a basin branch's data lives in the parent's LSM and `segments/`
//! namespace, so every GC path must be scope-correct in both directions —
//! the parent's sweep must keep a live branch's objects, branch deletion
//! must free exactly the branch's keys and objects, a branch's tombstone
//! cleaner must not debit parent-owned segment counters, and a branch's
//! reclaim cycle must see only its own segcounts.
//!
//! Both filesystems share one in-memory slatedb (the branch built with
//! [`ZeroFS::new_with_slatedb_for_branch`]). They also share one writer
//! epoch, so the branch's segment store is constructed at
//! [`BRANCH_SEGMENT_EPOCH`] to keep its `(epoch, counter)` segids — and thus
//! its `segments/` object keys — disjoint from the root's (in production
//! each serving process gets its own epoch from the manifest; only tests
//! share one open).

use crate::db::{Db, SlateDbHandle};
use crate::fs::errors::FsError;
use crate::fs::key_codec::{BranchId, KeyCodec};
use crate::fs::store::extent::reclaim::cycle::{self, CyclePolicy, SegmentProtection};
use crate::fs::test_util::test_creds;
use crate::fs::types::SetAttributes;
use crate::fs::{EXTENT_SIZE, TombstoneCleaner, ZeroFS};
use crate::segment::{FrameLoc, Segid};
use crate::test_helpers::test_helpers_mod::test_auth;
use bytes::Bytes;
use futures::StreamExt;
use std::sync::Arc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

/// Segment writer epoch for the branch filesystem (see the module doc).
/// Epoch 0 also sorts every branch segment before the root store's reclaim
/// cutoff, keeping the branch's objects sweep-eligible in these tests.
const BRANCH_SEGMENT_EPOCH: u64 = 0;

/// The fs-level auth context for reads/writes (test_auth() builds the NFS
/// wire type, which converts into it).
fn fs_auth() -> crate::fs::types::AuthContext {
    (&test_auth()).into()
}

/// A root filesystem and a branch filesystem over one shared in-memory
/// slatedb, with a parent-owned data file already written and flushed.
///
/// The parent's files are created BEFORE the branch filesystem is
/// constructed: the two instances seed their in-memory inode-id watermarks
/// from the same global counter at construction, so parent allocations must
/// precede (and never follow) branch construction, or both would mint the
/// same inode ids.
async fn open_root_and_branch_with_parent_data()
-> (ZeroFS, ZeroFS, BranchId, Arc<dyn slatedb::object_store::ObjectStore>) {
    use crate::block_transformer::ZeroFsBlockTransformer;
    use crate::config::CompressionConfig;
    use slatedb::BlockTransformer;
    use slatedb::DbBuilder;
    use slatedb::object_store::path::Path;

    let test_key = [0u8; 32];
    let object_store: Arc<dyn slatedb::object_store::ObjectStore> =
        Arc::new(slatedb::object_store::memory::InMemory::new());
    let block_transformer: Arc<dyn BlockTransformer> =
        ZeroFsBlockTransformer::try_new_arc(&test_key, CompressionConfig::default())
            .expect("test key should be lockable");
    let slatedb = Arc::new(
        DbBuilder::new(Path::from("branch-gc-test"), object_store.clone())
            .with_block_transformer(block_transformer)
            .with_filter_policies(crate::fs::filter_policy::filter_policies())
            .with_segment_extractor(Arc::new(crate::segment_extractor::ZeroFsSegmentExtractor))
            .build()
            .await
            .unwrap(),
    );
    let segment_codec = || {
        crate::frame_codec::FrameCodec::try_new(
            &test_key,
            crate::segment::SEGMENT_INFO,
            CompressionConfig::default(),
        )
        .expect("test key should be lockable")
    };

    let root = ZeroFS::new_with_slatedb(
        SlateDbHandle::ReadWrite(slatedb.clone()),
        u64::MAX,
        None,
        false,
        object_store.clone(),
        segment_codec(),
    )
    .await
    .unwrap();

    // Parent-owned files: an 11-extent data file (large enough that unlink
    // defers to tombstone cleanup) and an empty file written to later.
    let auth = fs_auth();
    let creds = test_creds();
    let (parent_file, _) = root
        .create(&creds, 0, b"parent-file", &SetAttributes::default())
        .await
        .unwrap();
    root.write(&auth, parent_file, 0, &parent_file_data())
        .await
        .unwrap();
    root.create(&creds, 0, b"parent-later", &SetAttributes::default())
        .await
        .unwrap();
    root.flush_coordinator.flush().await.unwrap();

    let branch_id = root.db.create_branch("b1").await.unwrap();
    let branch = ZeroFS::new_with_slatedb_for_branch(
        SlateDbHandle::ReadWrite(slatedb),
        branch_id,
        u64::MAX,
        None,
        false,
        object_store.clone(),
        segment_codec(),
        Some(BRANCH_SEGMENT_EPOCH),
    )
    .await
    .unwrap();

    (root, branch, branch_id, object_store)
}

/// Content of the parent-owned data file: 11 extents (> the inline-delete
/// threshold, so its unlink goes through tombstone cleanup).
fn parent_file_data() -> Bytes {
    Bytes::from(vec![7u8; 11 * EXTENT_SIZE])
}

fn cleaner_for(fs: &ZeroFS) -> TombstoneCleaner {
    TombstoneCleaner::new(
        fs.tombstone_store.clone(),
        fs.extent_store.clone(),
        Arc::clone(&fs.stats),
    )
}

/// `(live, total)` of a segment's counter read through the given codec's
/// scope (a raw read on the shared database; `None` when the scope has no
/// such row).
async fn segcount_in_scope(db: &Db, codec: &KeyCodec, segid: Segid) -> Option<(u64, u64)> {
    db.get_bytes(&codec.segcount_key(segid.epoch, segid.counter))
        .await
        .unwrap()
        .and_then(|b| KeyCodec::decode_segcount(&b))
}

/// The segment holding a file's extent 0, resolved in the given codec's
/// scope through the given filesystem's view.
async fn file_segment(fs: &ZeroFS, codec: &KeyCodec, inode: u64) -> Segid {
    let key = codec.extent_key(inode, 0);
    let raw = fs
        .db
        .get_bytes(key.as_ref())
        .await
        .unwrap()
        .expect("extent 0 must exist");
    FrameLoc::decode(&raw).expect("extent value is a FrameLoc").segid
}

/// Sorted `segments/` object keys currently in the object store.
async fn segment_object_keys(
    object_store: &Arc<dyn slatedb::object_store::ObjectStore>,
) -> Vec<String> {
    use slatedb::object_store::ObjectStore as _;
    let stream = object_store.list(Some(&slatedb::object_store::path::Path::from("segments")));
    let mut out: Vec<String> = stream
        .map(|meta| meta.unwrap().location.to_string())
        .collect()
        .await;
    out.sort();
    out
}

fn segids_of(objects: &[String]) -> Vec<Segid> {
    objects
        .iter()
        .map(|k| Segid::from_object_key(k).expect("a segments/ object key"))
        .collect()
}

/// Entry names of a full readdir of the root directory, excluding `.`/`..`.
async fn root_entry_names(fs: &ZeroFS) -> Vec<Vec<u8>> {
    let result = fs
        .readdir(&(&test_auth()).into(), 0, 0, 1000)
        .await
        .unwrap();
    assert!(result.end, "test listings fit in one page");
    let mut names: Vec<Vec<u8>> = result.entries[2..].iter().map(|e| e.name.clone()).collect();
    names.sort();
    names
}

/// End-to-end branch serving and GC over one shared volume: branch
/// read/write isolation, parent-after-branch visibility, branch deletion of
/// a parent-owned file with the parent's counters intact, merged readdir,
/// and Db-level branch deletion followed by the orphan sweep reclaiming
/// exactly the branch's segment objects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn branch_serving_and_gc_end_to_end() {
    let (root, branch, branch_id, object_store) = open_root_and_branch_with_parent_data().await;
    let auth = fs_auth();
    let creds = test_creds();
    let root_codec = KeyCodec::new();
    let branch_codec = KeyCodec::for_branch(branch_id);
    let parent_file = root.lookup(&creds, 0, b"parent-file").await.unwrap();
    let parent_later = root.lookup(&creds, 0, b"parent-later").await.unwrap();

    // 1. Branch writes are invisible to the parent but read back in-branch.
    let (branch_file, _) = branch
        .create(&creds, 0, b"branch-file", &SetAttributes::default())
        .await
        .unwrap();
    let branch_data = Bytes::from(vec![3u8; EXTENT_SIZE]);
    branch.write(&auth, branch_file, 0, &branch_data).await.unwrap();
    branch.flush_coordinator.flush().await.unwrap();
    assert_eq!(
        root.lookup(&creds, 0, b"branch-file").await,
        Err(FsError::NotFound)
    );
    assert_eq!(
        branch
            .read_file(&auth, branch_file, 0, EXTENT_SIZE as u32)
            .await
            .unwrap()
            .0
            .as_ref(),
        &branch_data[..]
    );

    // 2. A parent write after the branch exists is visible to the branch.
    // (Flush seals + PUTs the parent's open segment; a separate process —
    // which the branch fs stands in for — can only read sealed segments.)
    let later_data = Bytes::from(vec![9u8; 100]);
    root.write(&auth, parent_later, 0, &later_data).await.unwrap();
    root.flush_coordinator.flush().await.unwrap();
    assert_eq!(
        branch.lookup(&creds, 0, b"parent-later").await.unwrap(),
        parent_later
    );
    assert_eq!(
        branch
            .read_file(&auth, parent_later, 0, 100)
            .await
            .unwrap()
            .0
            .as_ref(),
        &later_data[..]
    );

    // 3. The branch deletes the parent-owned file (deferred tombstone); the
    // cleaner drops the branch's extent pointers but does NOT debit the
    // parent-owned segment's counter, and materializes no branch copy of it.
    let parent_seg = file_segment(&root, &root_codec, parent_file).await;
    let parent_counter_before = segcount_in_scope(&root.db, &root_codec, parent_seg)
        .await
        .expect("parent segment counter exists");
    branch
        .remove(&auth, 0, b"parent-file")
        .await
        .unwrap();
    cleaner_for(&branch).run().await.unwrap();
    assert_eq!(
        segcount_in_scope(&root.db, &root_codec, parent_seg).await,
        Some(parent_counter_before),
        "branch deletion must not debit the parent-owned segment"
    );
    assert!(
        segcount_in_scope(&root.db, &branch_codec, parent_seg)
            .await
            .is_none(),
        "no branch-scope counter may be materialized for a parent-owned segment"
    );
    // The parent's data is untouched; the branch's view hides the file.
    assert_eq!(
        root.read_file(&auth, parent_file, 0, parent_file_data().len() as u32)
            .await
            .unwrap()
            .0,
        parent_file_data()
    );
    assert_eq!(
        branch.lookup(&creds, 0, b"parent-file").await,
        Err(FsError::NotFound)
    );

    // 4. Merged readdir: the branch's entries plus the parent's survivors.
    assert_eq!(
        root_entry_names(&branch).await,
        vec![b"branch-file".to_vec(), b"parent-later".to_vec()]
    );
    assert_eq!(
        root_entry_names(&root).await,
        vec![
            b"parent-file".to_vec(),
            b"parent-later".to_vec()
        ]
    );

    // 5. The branch deleting its OWN file debits its OWN counters.
    let (own_file, _) = branch
        .create(&creds, 0, b"branch-owned", &SetAttributes::default())
        .await
        .unwrap();
    branch
        .write(&auth, own_file, 0, &Bytes::from(vec![5u8; 11 * EXTENT_SIZE]))
        .await
        .unwrap();
    branch.flush_coordinator.flush().await.unwrap();
    let own_seg = file_segment(&branch, &branch_codec, own_file).await;
    let own_counter_before = segcount_in_scope(&root.db, &branch_codec, own_seg)
        .await
        .expect("branch segment counter exists");
    assert_eq!(own_counter_before.0, own_counter_before.1);
    assert!(own_counter_before.0 > 0);
    branch
        .remove(&auth, 0, b"branch-owned")
        .await
        .unwrap();
    cleaner_for(&branch).run().await.unwrap();
    let own_counter_after = segcount_in_scope(&root.db, &branch_codec, own_seg)
        .await
        .expect("branch counter row survives the debit");
    assert_eq!(
        own_counter_after,
        (0, own_counter_before.1),
        "the branch's own segment is debited to zero live (total stays monotonic)"
    );

    // 6. While the branch is registered, the parent's orphan sweep keeps
    // every segment object: the census sees both scopes' counters.
    let objects_before = segment_object_keys(&object_store).await;
    let epochs: Vec<u64> = segids_of(&objects_before).iter().map(|s| s.epoch).collect();
    assert!(
        epochs.contains(&BRANCH_SEGMENT_EPOCH),
        "precondition: branch segments exist: {objects_before:?}"
    );
    let sweep = cycle::sweep_orphans(&root.extent_store, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(sweep.deleted(), 0, "a live branch's objects are not orphans");
    for key in &objects_before {
        assert!(
            segment_object_keys(&object_store).await.contains(key),
            "the sweep deleted {key} while every scope's counters were live"
        );
    }

    // 7. Db-level branch deletion (what the DeleteBranch RPC does): the
    // registry row goes first, then the branch's key ranges — segcounts
    // included — and the orphan sweep then reclaims exactly the branch's
    // segment objects.
    assert!(root.db.delete_branch("b1").await.unwrap());
    let removed = root.db.delete_branch_data(branch_id).await.unwrap();
    assert!(removed > 0, "the branch owned rows before deletion");
    assert!(
        segcount_in_scope(&root.db, &branch_codec, own_seg)
            .await
            .is_none(),
        "the branch's segcount rows die with its key-range drop"
    );
    assert!(
        segcount_in_scope(&root.db, &root_codec, parent_seg)
            .await
            .is_some(),
        "the parent's segcount rows survive the branch's deletion"
    );

    let sweep = cycle::sweep_orphans(&root.extent_store, &CancellationToken::new())
        .await
        .unwrap();
    assert!(
        sweep.deleted() >= 2,
        "branch-file's and branch-owned's segments are reclaimed, got {sweep:?}"
    );
    let remaining = segids_of(&segment_object_keys(&object_store).await);
    assert!(
        remaining.iter().all(|s| s.epoch != BRANCH_SEGMENT_EPOCH),
        "no branch-epoch object survives the sweep: {remaining:?}"
    );
    assert!(!remaining.is_empty(), "parent data objects are untouched");
    assert_eq!(
        root.read_file(&auth, parent_later, 0, 100)
            .await
            .unwrap()
            .0
            .as_ref(),
        &later_data[..]
    );
}

/// A reclaim cycle driven by a branch filesystem scans only the branch's
/// segcount scope: it reclaims the branch's dead segments and never touches
/// the parent's rows or objects (and vice versa).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn branch_reclaim_cycle_sees_only_its_own_segcounts() {
    let (root, branch, branch_id, object_store) = open_root_and_branch_with_parent_data().await;
    let auth = fs_auth();
    let creds = test_creds();
    let branch_codec = KeyCodec::for_branch(branch_id);

    // The branch writes and deletes its own file (small enough that the
    // unlink debits inline), leaving its segment dead at (0 live).
    let (doomed, _) = branch
        .create(&creds, 0, b"doomed", &SetAttributes::default())
        .await
        .unwrap();
    branch
        .write(&auth, doomed, 0, &Bytes::from(vec![1u8; EXTENT_SIZE]))
        .await
        .unwrap();
    branch.flush_coordinator.flush().await.unwrap();
    let doomed_seg = file_segment(&branch, &branch_codec, doomed).await;
    branch.remove(&auth, 0, b"doomed").await.unwrap();

    let policy = CyclePolicy {
        repack_min_dead_percent: 95,
        job_bytes: 1 << 20,
        max_concurrent_repacks: 1,
    };
    let expired = || {
        std::future::ready(Ok(SegmentProtection::Until(Instant::now())))
    };

    // The branch's cycle reclaims exactly its own dead segment.
    let outcome = cycle::run(&branch.extent_store, expired, policy, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(
        (outcome.deleted, outcome.relocated),
        (1, 0),
        "only the branch's own dead segment is reclaimed"
    );
    assert!(
        !segment_object_keys(&object_store)
            .await
            .contains(&doomed_seg.object_key()),
        "the branch's dead segment object is gone"
    );
    assert!(
        segcount_in_scope(&root.db, &branch_codec, doomed_seg)
            .await
            .is_none(),
        "the reclaimed segment's branch counter is dropped"
    );
    // The parent's live segment object is untouched by the branch's cycle.
    assert!(
        segment_object_keys(&object_store)
            .await
            .iter()
            .any(|k| !k.contains(&format!("/{BRANCH_SEGMENT_EPOCH}/"))),
        "parent segments survive the branch's reclaim cycle"
    );

    // The root's cycle sees only its own (live) counters: nothing is
    // reclaimed, and the parent's file still reads.
    let outcome = cycle::run(&root.extent_store, expired, policy, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!((outcome.deleted, outcome.relocated), (0, 0));
    let parent_file = root.lookup(&creds, 0, b"parent-file").await.unwrap();
    assert_eq!(
        root.read_file(&auth, parent_file, 0, parent_file_data().len() as u32)
            .await
            .unwrap()
            .0,
        parent_file_data()
    );
}

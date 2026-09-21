//! Filesystem-level proof of merged branch scans: a basin branch's namespace
//! operations (create/rename/remove) and its readdir run entirely through
//! the branch view, while the parent's view of the same volume stays
//! byte-identical. Both filesystems share one in-memory slatedb; the branch
//! filesystem is built with [`ZeroFS::new_with_slatedb_for_branch`], so its
//! stores, commit worker, and directory listings all go through the branch
//! codec and [`crate::db::Db::with_branch`]'s merged scans.

use crate::db::SlateDbHandle;
use crate::fs::ZeroFS;
use crate::fs::test_util::test_creds;
use crate::fs::types::SetAttributes;
use crate::test_helpers::test_helpers_mod::test_auth;
use std::sync::Arc;

/// A root filesystem and a branch filesystem over one shared in-memory
/// slatedb. The branch is registered (O(1), see [`crate::branch`]) before
/// the branch filesystem is constructed.
async fn open_root_and_branch() -> (ZeroFS, ZeroFS) {
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
        DbBuilder::new(Path::from("branch-view-fs-test"), object_store.clone())
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

    let branch_id = root.db.create_branch("test-branch").await.unwrap();
    let branch = ZeroFS::new_with_slatedb_for_branch(
        SlateDbHandle::ReadWrite(slatedb),
        branch_id,
        u64::MAX,
        None,
        false,
        object_store,
        segment_codec(),
        // Metadata-only tests: no segment writes through the branch fs, so
        // sharing the writer epoch (and its segid counter) is harmless.
        None,
    )
    .await
    .unwrap();

    (root, branch)
}

/// Entry names of a full readdir of `dirid`, excluding `.` and `..`.
async fn entry_names(fs: &ZeroFS, dirid: u64) -> Vec<Vec<u8>> {
    let result = fs
        .readdir(&(&test_auth()).into(), dirid, 0, 1000)
        .await
        .unwrap();
    assert!(result.end, "test listings fit in one page");
    result.entries[2..].iter().map(|e| e.name.clone()).collect()
}

fn assert_names(mut actual: Vec<Vec<u8>>, mut expected: Vec<&[u8]>) {
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn branch_readdir_merges_parent_and_branch_namespaces() {
    let (root, branch) = open_root_and_branch().await;

    // The parent's tree, written before the branch exists.
    root.create(&test_creds(), 0, b"deleted-by-branch", &SetAttributes::default())
        .await
        .unwrap();
    root.create(&test_creds(), 0, b"renamed-by-branch", &SetAttributes::default())
        .await
        .unwrap();
    let (parent_shadowed_id, _) = root
        .create(&test_creds(), 0, b"shadowed", &SetAttributes::default())
        .await
        .unwrap();
    let (sub_id, _) = root
        .mkdir(&test_creds(), 0, b"sub", &SetAttributes::default())
        .await
        .unwrap();
    root.create(&test_creds(), sub_id, b"nested-kept", &SetAttributes::default())
        .await
        .unwrap();
    root.create(&test_creds(), sub_id, b"nested-renamed", &SetAttributes::default())
        .await
        .unwrap();

    // The parent keeps writing after the branch exists: fallback makes the
    // new entry visible to the branch.
    let (parent_new_id, _) = root
        .create(&test_creds(), 0, b"parent-new", &SetAttributes::default())
        .await
        .unwrap();

    // The branch's own mutations: delete a parent file, rename a parent
    // file, shadow a parent file (rename over its name), create a new file,
    // and rename inside a nested directory.
    branch
        .remove(&(&test_auth()).into(), 0, b"deleted-by-branch")
        .await
        .unwrap();
    branch
        .rename(&(&test_auth()).into(), 0, b"renamed-by-branch", 0, b"renamed")
        .await
        .unwrap();
    branch
        .create(&test_creds(), 0, b"tmp-shadow", &SetAttributes::default())
        .await
        .unwrap();
    branch
        .rename(&(&test_auth()).into(), 0, b"tmp-shadow", 0, b"shadowed")
        .await
        .unwrap();
    let (branch_new_id, _) = branch
        .create(&test_creds(), 0, b"branch-new", &SetAttributes::default())
        .await
        .unwrap();
    branch
        .rename(
            &(&test_auth()).into(),
            sub_id,
            b"nested-renamed",
            sub_id,
            b"nested-renamed-b",
        )
        .await
        .unwrap();

    // The branch sees the merged namespace: parent's surviving entries plus
    // its own, with deletions hidden and renames applied.
    assert_names(
        entry_names(&branch, 0).await,
        vec![
            b"branch-new",
            b"parent-new",
            b"renamed",
            b"shadowed",
            b"sub",
        ],
    );
    assert_names(
        entry_names(&branch, sub_id).await,
        vec![b"nested-kept", b"nested-renamed-b"],
    );

    // The parent's view is untouched: no branch-created names, its deleted
    // and renamed entries intact.
    assert_names(
        entry_names(&root, 0).await,
        vec![
            b"deleted-by-branch",
            b"parent-new",
            b"renamed-by-branch",
            b"shadowed",
            b"sub",
        ],
    );
    assert_names(
        entry_names(&root, sub_id).await,
        vec![b"nested-kept", b"nested-renamed"],
    );

    // The shadowing entry really is the branch's inode, not the parent's;
    // the parent's own inode is untouched underneath.
    let branch_view = branch
        .lookup(&test_creds(), 0, b"shadowed")
        .await
        .unwrap();
    assert_ne!(branch_view, parent_shadowed_id);
    assert_eq!(
        root.lookup(&test_creds(), 0, b"shadowed").await.unwrap(),
        parent_shadowed_id
    );

    // Point lookups agree with the listings: the branch-deleted file is
    // gone for the branch only, and each view resolves its own new entries.
    assert_eq!(
        branch.lookup(&test_creds(), 0, b"deleted-by-branch").await,
        Err(crate::fs::errors::FsError::NotFound)
    );
    assert!(
        root.lookup(&test_creds(), 0, b"deleted-by-branch")
            .await
            .is_ok()
    );
    assert_eq!(
        branch.lookup(&test_creds(), 0, b"parent-new").await.unwrap(),
        parent_new_id
    );
    assert_eq!(
        branch.lookup(&test_creds(), 0, b"branch-new").await.unwrap(),
        branch_new_id
    );
    assert_eq!(
        root.lookup(&test_creds(), 0, b"branch-new").await,
        Err(crate::fs::errors::FsError::NotFound)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn branch_readdir_paginates_and_handles_empty_scopes() {
    let (root, branch) = open_root_and_branch().await;

    // Parent: a, b, c, d and an empty directory. The branch deletes b and
    // adds e, so its merged listing is a, c, d, e (name-ordered).
    for name in [b"a".as_slice(), b"b", b"c", b"d"] {
        root.create(&test_creds(), 0, name, &SetAttributes::default())
            .await
            .unwrap();
    }
    let (empty_id, _) = root
        .mkdir(&test_creds(), 0, b"empty-parent-dir", &SetAttributes::default())
        .await
        .unwrap();
    branch
        .remove(&(&test_auth()).into(), 0, b"b")
        .await
        .unwrap();
    branch
        .create(&test_creds(), 0, b"e", &SetAttributes::default())
        .await
        .unwrap();
    let (branch_dir_id, _) = branch
        .mkdir(&test_creds(), 0, b"branch-dir", &SetAttributes::default())
        .await
        .unwrap();

    // Page the branch's listing two entries at a time.
    let auth = &(&test_auth()).into();
    let names_of = |result: &crate::fs::types::ReadDirResult| {
        result
            .entries
            .iter()
            .map(|e| e.name.clone())
            .collect::<Vec<Vec<u8>>>()
    };
    let page1 = branch.readdir(auth, 0, 0, 4).await.unwrap();
    assert!(!page1.end);
    assert_eq!(
        names_of(&page1),
        vec![
            b".".to_vec(),
            b"..".to_vec(),
            b"a".to_vec(),
            b"branch-dir".to_vec()
        ]
    );
    let page2 = branch
        .readdir(auth, 0, page1.entries.last().unwrap().cookie, 2)
        .await
        .unwrap();
    assert!(!page2.end);
    assert_eq!(names_of(&page2), vec![b"c".to_vec(), b"d".to_vec()]);
    let page3 = branch
        .readdir(auth, 0, page2.entries.last().unwrap().cookie, 100)
        .await
        .unwrap();
    assert!(page3.end);
    assert_eq!(
        names_of(&page3),
        vec![b"e".to_vec(), b"empty-parent-dir".to_vec()]
    );

    // Empty scopes: a parent-empty directory lists as `. ..` in the branch,
    // and the branch's own directory is invisible to the parent.
    assert!(entry_names(&branch, empty_id).await.is_empty());
    assert!(entry_names(&branch, branch_dir_id).await.is_empty());
    assert_eq!(
        root.lookup(&test_creds(), 0, b"branch-dir").await,
        Err(crate::fs::errors::FsError::NotFound)
    );
}

//! Proof of concept for "Basin lineages for ZeroFS forks"
//! (see `docs/basin-lineage.md`).
//!
//! Instead of cloning the volume's LSM into a new db path per fork (what the
//! `fork-support` branch ships), this test keeps ALL lineages in ONE SlateDB
//! database and scopes every mutable key with a leading branch id — the same
//! trick s2-lite uses to host thousands of independent streams in one basin
//! (`lite/src/backend/kv/mod.rs`: `StreamRecordData(StreamId, StreamPosition)`
//! etc., stream id embedded in every key).
//!
//! Key layout used here (mirroring `zerofs/src/fs/key_codec.rs`):
//!
//!   lineage-scoped mutable kinds:  [branch: u64 BE] || <zerofs key>
//!     - INODE:    branch || b"meta" || 0x01 || inode
//!     - SEGCOUNT: branch || b"meta" || 0x09 || epoch || counter
//!   basin-wide singleton kinds (unscoped, like s2-lite's per-stream meta):
//!     - b"basin" || 0x01 || branch  -> parent branch id (BRANCH_META)
//!     - b"basin" || 0x02 || branch  -> fencing token    (BRANCH_FENCE)
//!     - b"basin" || 0x03 || branch  -> trim watermark   (BRANCH_TRIM)
//!
//! Branch 0 is the root lineage. A read for branch B resolves B's own key
//! first, then walks the ancestor chain rootward — LSM point lookups, never
//! object-store 404 probing (the analogue of `segment_path_router.rs`'s
//! epoch-routed reads, but at the key level).

use bytes::Bytes;
use slatedb::Db;
use slatedb::IsolationLevel;
use slatedb::object_store::memory::InMemory;
use slatedb::object_store::path::Path;
use std::sync::Arc;

const ROOT: u64 = 0;

const KIND_INODE: u8 = 0x01;
const KIND_SEGCOUNT: u8 = 0x09;

const BASIN_BRANCH_META: u8 = 0x01;
const BASIN_FENCE: u8 = 0x02;
const BASIN_TRIM: u8 = 0x03;

// ---------------------------------------------------------------------------
// Key encoding
// ---------------------------------------------------------------------------

fn inode_key(branch: u64, inode: u64) -> Bytes {
    let mut k = Vec::with_capacity(8 + 4 + 1 + 8);
    k.extend_from_slice(&branch.to_be_bytes());
    k.extend_from_slice(b"meta");
    k.push(KIND_INODE);
    k.extend_from_slice(&inode.to_be_bytes());
    Bytes::from(k)
}

fn segcount_key(branch: u64, epoch: u64, counter: u64) -> Bytes {
    let mut k = Vec::with_capacity(8 + 4 + 1 + 16);
    k.extend_from_slice(&branch.to_be_bytes());
    k.extend_from_slice(b"meta");
    k.push(KIND_SEGCOUNT);
    k.extend_from_slice(&epoch.to_be_bytes());
    k.extend_from_slice(&counter.to_be_bytes());
    Bytes::from(k)
}

/// Half-open range covering every segcount row owned by `branch` — the
/// per-branch trim task scans exactly this and nothing else.
fn segcount_prefix_range(branch: u64) -> (Bytes, Bytes) {
    let mut start = Vec::with_capacity(8 + 4 + 1);
    start.extend_from_slice(&branch.to_be_bytes());
    start.extend_from_slice(b"meta");
    start.push(KIND_SEGCOUNT);
    let mut end = start.clone();
    *end.last_mut().unwrap() += 1;
    (Bytes::from(start), Bytes::from(end))
}

fn basin_key(kind: u8, branch: u64) -> Bytes {
    let mut k = Vec::with_capacity(5 + 1 + 8);
    k.extend_from_slice(b"basin");
    k.push(kind);
    k.extend_from_slice(&branch.to_be_bytes());
    Bytes::from(k)
}

fn branch_meta_key(branch: u64) -> Bytes {
    basin_key(BASIN_BRANCH_META, branch)
}

fn fence_key(branch: u64) -> Bytes {
    basin_key(BASIN_FENCE, branch)
}

fn trim_key(branch: u64) -> Bytes {
    basin_key(BASIN_TRIM, branch)
}

// ---------------------------------------------------------------------------
// Branch / lineage operations on one shared Db
// ---------------------------------------------------------------------------

/// O(1) branch creation: one row. No clone, no manifest copy, no second Db.
async fn create_branch(db: &Db, branch: u64, parent: u64) {
    db.put(branch_meta_key(branch), Bytes::from(parent.to_be_bytes().to_vec()))
        .await
        .unwrap();
}

/// Root-first ancestor chain of `branch`, ending in `branch` itself.
async fn lineage_of(db: &Db, branch: u64) -> Vec<u64> {
    let mut chain = vec![branch];
    let mut cur = branch;
    while cur != ROOT {
        let parent = db
            .get(branch_meta_key(cur))
            .await
            .unwrap()
            .map(|v| u64::from_be_bytes(v.as_ref().try_into().unwrap()))
            .expect("branch meta present");
        chain.push(parent);
        cur = parent;
    }
    chain.reverse();
    chain
}

/// Lineage-aware read: nearest writer wins. Each hop is an LSM point lookup
/// (bloom-filterable), ordered self -> parent -> ... -> root.
async fn read_inode(db: &Db, lineage: &[u64], inode: u64) -> Option<Bytes> {
    for branch in lineage.iter().rev() {
        if let Some(v) = db.get(inode_key(*branch, inode)).await.unwrap() {
            return Some(v);
        }
    }
    None
}

/// Branch-scoped writer identity, the analogue of s2-lite's
/// `StreamFencingToken`: opening a writer on a branch bumps its token inside
/// a serializable transaction, so exactly one writer holds the latest token
/// even if two servers race to open the same branch.
async fn open_writer(db: &Db, branch: u64) -> u64 {
    let txn = db.begin(IsolationLevel::SerializableSnapshot).await.unwrap();
    let token = txn
        .get(fence_key(branch))
        .await
        .unwrap()
        .map(|v| u64::from_be_bytes(v.as_ref().try_into().unwrap()))
        .unwrap_or(0)
        + 1;
    txn.put(fence_key(branch), Bytes::from(token.to_be_bytes().to_vec()))
        .unwrap();
    txn.commit().await.unwrap();
    token
}

/// A guarded write: the writer must still hold the branch's current token.
/// Returns false when the writer has been fenced (s2-lite's
/// `AppendConditionFailedError::FencingTokenMismatch`).
async fn guarded_put(db: &Db, branch: u64, token: u64, key: Bytes, value: Bytes) -> bool {
    let txn = db.begin(IsolationLevel::SerializableSnapshot).await.unwrap();
    let current = txn
        .get(fence_key(branch))
        .await
        .unwrap()
        .map(|v| u64::from_be_bytes(v.as_ref().try_into().unwrap()))
        .unwrap_or(0);
    if current != token {
        txn.rollback();
        return false;
    }
    txn.put(key, value).unwrap();
    txn.commit().await.unwrap();
    true
}

/// Per-branch trim: delete this branch's own dead segcount rows at or below
/// the branch's trim watermark. Sibling branches' rows are structurally out
/// of range — no global pause, no `SegmentProtection::Indefinite`.
async fn trim_branch(db: &Db, branch: u64) -> usize {
    let watermark = db
        .get(trim_key(branch))
        .await
        .unwrap()
        .map(|v| u64::from_be_bytes(v.as_ref().try_into().unwrap()))
        .unwrap_or(u64::MAX);
    let (start, end) = segcount_prefix_range(branch);
    let mut it = db.scan(start..end).await.unwrap();
    let mut trimmed = 0;
    while let Some(kv) = it.next().await.unwrap() {
        let epoch = u64::from_be_bytes(kv.key[13..21].try_into().unwrap());
        if epoch > watermark {
            continue;
        }
        // live == 0 in the (live, total) counter encoding
        let live = u64::from_le_bytes(kv.value[..8].try_into().unwrap());
        if live == 0 {
            db.delete(kv.key).await.unwrap();
            trimmed += 1;
        }
    }
    trimmed
}

fn segcount_value(live: u64, total: u64) -> Bytes {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&live.to_le_bytes());
    v.extend_from_slice(&total.to_le_bytes());
    Bytes::from(v)
}

async fn open_db() -> Db {
    let store = Arc::new(InMemory::new());
    Db::open(Path::from("basin"), store).await.unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_lineages_share_one_db_and_reads_resolve_through_base() {
    let db = open_db().await;

    // Root lineage writes inodes 1 and 2 (branch 0 prefix).
    db.put(inode_key(ROOT, 1), Bytes::from_static(b"root-v1"))
        .await
        .unwrap();
    db.put(inode_key(ROOT, 2), Bytes::from_static(b"root-only"))
        .await
        .unwrap();

    // Fork A (branch 1) and fork B (branch 2) off the root: O(1) each.
    create_branch(&db, 1, ROOT).await;
    create_branch(&db, 2, ROOT).await;

    // Branch 1 overwrites inode 1 and creates inode 3, under its own prefix.
    db.put(inode_key(1, 1), Bytes::from_static(b"branch-a-v2"))
        .await
        .unwrap();
    db.put(inode_key(1, 3), Bytes::from_static(b"branch-a-only"))
        .await
        .unwrap();

    // Branch 2 creates inode 4 only.
    db.put(inode_key(2, 4), Bytes::from_static(b"branch-b-only"))
        .await
        .unwrap();

    // Force state down to SSTs: lineage resolution must survive compaction.
    db.flush().await.unwrap();

    let la = lineage_of(&db, 1).await;
    let lb = lineage_of(&db, 2).await;
    assert_eq!(la, vec![ROOT, 1]);
    assert_eq!(lb, vec![ROOT, 2]);

    // Branch A sees its override + base state.
    assert_eq!(
        read_inode(&db, &la, 1).await.as_deref(),
        Some(&b"branch-a-v2"[..])
    );
    assert_eq!(
        read_inode(&db, &la, 2).await.as_deref(),
        Some(&b"root-only"[..])
    );
    assert_eq!(
        read_inode(&db, &la, 3).await.as_deref(),
        Some(&b"branch-a-only"[..])
    );
    assert_eq!(read_inode(&db, &la, 4).await, None);

    // Branch B sees the base, NOT branch A's overlay.
    assert_eq!(
        read_inode(&db, &lb, 1).await.as_deref(),
        Some(&b"root-v1"[..])
    );
    assert_eq!(read_inode(&db, &lb, 3).await, None);
    assert_eq!(
        read_inode(&db, &lb, 4).await.as_deref(),
        Some(&b"branch-b-only"[..])
    );

    // The root is untouched by both forks.
    let root_lineage = [ROOT];
    assert_eq!(
        read_inode(&db, &root_lineage, 1).await.as_deref(),
        Some(&b"root-v1"[..])
    );
    assert_eq!(read_inode(&db, &root_lineage, 3).await, None);
}

#[tokio::test]
async fn branch_scoped_fencing_token_fences_stale_writer() {
    let db = open_db().await;
    create_branch(&db, 1, ROOT).await;

    // Two servers race to open branch 1 for write: both get distinct tokens,
    // but only the later one is current.
    let t1 = open_writer(&db, 1).await;
    let t2 = open_writer(&db, 1).await;
    assert!(t2 > t1);

    // The stale writer is fenced off; the current writer proceeds.
    assert!(
        !guarded_put(&db, 1, t1, inode_key(1, 7), Bytes::from_static(b"stale")).await,
        "stale token must be fenced"
    );
    assert!(
        guarded_put(&db, 1, t2, inode_key(1, 7), Bytes::from_static(b"current")).await,
        "current token writes"
    );

    let lineage = lineage_of(&db, 1).await;
    assert_eq!(
        read_inode(&db, &lineage, 7).await.as_deref(),
        Some(&b"current"[..])
    );

    // Fencing is branch-scoped: a sibling branch's writers are independent.
    create_branch(&db, 2, ROOT).await;
    let t = open_writer(&db, 2).await;
    assert_eq!(t, 1, "branch 2 has its own token sequence");
}

#[tokio::test]
async fn per_branch_trim_collects_own_rows_without_pinning_siblings() {
    let db = open_db().await;
    create_branch(&db, 1, ROOT).await;

    // Root lineage (epoch 1): one dead segment, one live.
    db.put(segcount_key(ROOT, 1, 1), segcount_value(0, 1000))
        .await
        .unwrap();
    db.put(segcount_key(ROOT, 1, 2), segcount_value(500, 1000))
        .await
        .unwrap();

    // Branch 1 (epoch 2): one dead segment of its own.
    db.put(segcount_key(1, 2, 1), segcount_value(0, 2000))
        .await
        .unwrap();

    db.flush().await.unwrap();

    // The root's trim task runs while branch 1 exists. In the clone-per-path
    // model any fork pins parent reclaim indefinitely
    // (`SegmentProtection::Indefinite` in reclaim/driver.rs); here the root's
    // watermark advances independently and collects its own dead row.
    assert_eq!(trim_branch(&db, ROOT).await, 1);
    assert!(db.get(segcount_key(ROOT, 1, 1)).await.unwrap().is_none());
    assert!(
        db.get(segcount_key(ROOT, 1, 2)).await.unwrap().is_some(),
        "live root segment untouched"
    );
    assert!(
        db.get(segcount_key(1, 2, 1)).await.unwrap().is_some(),
        "sibling branch's rows are out of the root's scan range"
    );

    // Branch 1's trim task is equally independent — and its watermark can be
    // capped so epochs above it survive (time-travel floor for that branch).
    db.put(trim_key(1), Bytes::from(1u64.to_be_bytes().to_vec()))
        .await
        .unwrap();
    assert_eq!(
        trim_branch(&db, 1).await,
        0,
        "watermark 1 retains branch 1's epoch-2 rows"
    );
    db.put(trim_key(1), Bytes::from(2u64.to_be_bytes().to_vec()))
        .await
        .unwrap();
    assert_eq!(trim_branch(&db, 1).await, 1);
    assert!(db.get(segcount_key(1, 2, 1)).await.unwrap().is_none());
}

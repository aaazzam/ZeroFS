# Basin lineages for ZeroFS forks

**Status:** design document + proof of concept (`zerofs/tests/basin_poc.rs`, 3 tests, passing).
**Question:** what if a fork were a *key-prefix dimension inside the parent's LSM* instead of a cloned database at a new db path?

References to ZeroFS are paths in this repo (branch `fork-support`); references to
s2-lite are paths under `lite/` in `github.com/s2-streamstore/s2`.

> **Note on sources.** Code citations describe the *committed* `fork-support`
> branch (fork lineage in `.zerofs_fork.json`, `ForkInfo::load/save` against the
> object store). The working tree currently contains *uncommitted* work (from a
> parallel effort) moving that metadata into the LSM itself as new
> `FORK_LINEAGE`/`FORK_REGISTRY` key kinds in `key_codec.rs`. That change is a
> step in this document's direction — fork metadata as LSM records rather than
> sidecar objects — but it is still clone-per-path: the records live in the
> fork's *own cloned* database. Nothing in this design depends on either
> variant; where it matters (§6) both are noted.

---

## 0. The two models in one paragraph each

**Clone-per-path (shipped, `fork-support`):** `ForkManager::create_fork`
(`zerofs/src/fork_manager.rs`) checkpoints the parent, then uses SlateDB's clone
to build a new writable database at `<parent_db_path>/forks/<name>` referencing
the parent's SSTs as `external_ssts`. The fork inherits the parent's
`writer_epoch` (+1 on first open), writes its own segment objects under its own
db path, and reads ancestor segments through `SegmentPathRouter`
(`zerofs/src/segment_path_router.rs`), which routes `segments/{shard}/{epoch}/{counter}`
reads to the ancestor whose `base_epoch` ≤ the epoch in the key. Fork lineage is
persisted in `.zerofs_fork.json` (`zerofs/src/fork_info.rs`).

**Basin lineage (proposed):** one SlateDB database hosts the root volume and
*all* of its forks. Every mutable key gains a branch-id dimension, exactly the
way s2-lite hosts thousands of independent streams in one LSM by embedding
`StreamId` in every key (`lite/src/backend/kv/mod.rs`: `StreamMeta`,
`StreamTailPosition`, `StreamFencingToken`, `StreamTrimPoint`,
`StreamRecordData(StreamId, StreamPosition)`). A branch is a few metadata rows;
a read resolves "nearest writer in my lineage wins"; per-branch fencing tokens
and trim watermarks replace whole-volume epoch fencing and blanket GC pinning.

---

## 1. Motivation: where clone-per-path hurts

The clone itself is cheap — O(manifest) bytes, no data movement. The costs are
structural, and they scale with the *number of live forks*:

| Cost at N live forks | Clone-per-path | Basin lineage |
|---|---|---|
| Open databases | N full SlateDB `Db` instances (each with memtable, block cache, manifest poller, compactor state) | 1 |
| Manifest objects | N+1 manifests, each fenced by conditional-put | 1 |
| Fork creation | O(manifest) + clone pin + key copy + `ForkInfo` write | O(1) rows (see PoC `create_branch`) |
| Parent segment GC | **fully paused** while any fork/clone pin or persistent checkpoint exists (`SegmentProtection::Indefinite`, `zerofs/src/fs/store/extent/reclaim/driver.rs:44-46`) | per-branch trim watermarks; root reclaims while branches exist |
| Cross-fork GC | each fork's reclaim skips ancestor-owned segments (`cycle.rs:253`, `epoch < base_epoch`) and no fork may delete an ancestor's objects — ancestor dead bytes are collectible only when *all* descendants are gone | one store, union-GC: a segment is dead when no lineage's watermark still covers it |
| Failure isolation | strong: fork can live on another server/bucket | weak: one store, one manifest, shared SSTs |

Quantified, order-of-magnitude:

- **N = 100 forks.** Clone model: 100 `Db` instances ≈ 100 memtables (each
  sized for the volume's write buffer), 100 manifest-poll loops issuing LIST/GET
  against S3 on their own cadence, 100 compactor loops. Feasible but heavy; in
  practice you'd shard forks across servers, which is an operations answer, not
  a storage answer. Basin model: 100 rows in `BRANCH_META`, one memtable, one
  compactor. Read cost rises by ≤ (lineage depth − 1) bloom-filter probes per
  point lookup.
- **N = 1,000.** Clone model: 1,000 manifest pollers is already an S3
  request-rate problem, and parent GC has effectively been paused since the
  first fork — dead segments accumulate for the *lifetime of the fork fleet*.
  Basin model: unchanged; trim watermarks advance per branch.
- **N = 10,000** (the "fork per agent run / per CI job" regime): clone model is
  infeasible as colocated databases; basin model is the s2-lite operating point
  (thousands of streams per basin is the design center).

The motivating asymmetry: forks are *cheap to create* in both models, but in the
clone model they are *expensive to keep alive* and *paralyzing to GC*. Basin
lineages make liveness free and GC incremental.

---

## 2. Proposed key model

### 2.1 The branch dimension

Today every key is `[b"meta"|b"extent"] || kind || id...`
(`zerofs/src/fs/key_codec.rs`). Insert the branch id after the domain+kind and
before the id:

```
INODE:    b"meta"  || 0x01 || branch: u64 BE || inode_id
EXTENT:   b"extent"|| 0xFE || branch: u64 BE || inode_id || extent_index
SEGCOUNT: b"meta"  || 0x09 || branch: u64 BE || epoch || counter
```

Why not branch-first? Two reasons:

1. **Segment extraction.** `ZeroFsSegmentExtractor`
   (`zerofs/src/segment_extractor.rs`) routes on the *leading* `b"meta"` /
   `b"extent"` domain into separate LSM trees. Branch-first would silently land
   all keys in one extractor bucket unless the extractor learned to skip 8
   bytes. Kind-then-branch keeps routing byte-identical to today. (The PoC uses
   branch-first for readability only.)
2. **Per-kind scans stay per-branch.** Every existing full-kind scan (reclaim's
   segcount scan `segcount_prefix_range`, tombstone cleanup, orphan drain)
   becomes a `[kind || branch]` prefix scan per branch — no global scans that
   must be filtered.

Branch 0 is the root lineage; non-fork volumes are byte-compatible with today
modulo the inserted 8 zero bytes (a format bump — see §6).

Basin-wide singleton rows, unscoped (s2-lite's per-stream meta analogues):

```
b"basin" || 0x01 || branch  -> BranchMeta { parent, branch_point, fence_token }
b"basin" || 0x02 || branch  -> fencing token (u64, CAS'd; see §3)
b"basin" || 0x03 || branch  -> trim watermark (see §4)
```

### 2.2 Read resolution without 404 probing

The existing trick to preserve: **segment object keys already embed the writer
epoch**, and `SegmentPathRouter::owner_prefix` resolves the owning volume
*arithmetically* from that epoch — no probing, no existence checks. The basin
model generalizes that from object paths to LSM keys: ownership and visibility
must be *decidable from the key and the lineage metadata*, never discovered by
trial reads.

Point read (inode, extent, dir entry) for branch B with lineage `[root, ..., B]`:

```
for branch in lineage.iter().rev():        # self, parent, ..., root
    if let Some(v) = db.get(kind_key(branch, id)): return v
```

Each hop is an LSM point lookup; misses are bloom-filter hits, so a depth-d
read costs ~d filter probes and 1 block fetch. This is the same shape as
s2-lite, where a stream's records are exactly the `[StreamId]` prefix and no
ancestor probing exists at all (streams have no lineage — ZeroFS forks do, so
we pay d probes where s2-lite pays 1).

**Range scans are the hard part.** `read_file` scans
`extent || inode_id || [0..∞)`; readdir scans a dir-entry prefix. With a branch
dimension these become d prefix scans (one per lineage member) merged with
nearest-writer-wins. That is a real engineering cost ZeroFS doesn't have today
— but it is bounded: d is lineage depth (small; 1–3 in practice), each sub-scan
is prefix-contiguous, and the merge is a k-way heap over sorted iterators.

### 2.3 The snapshot problem (branch-point isolation)

A clone-fork is a *frozen snapshot*: the parent's later writes go to the
parent's memtable/SSTs and never appear in the fork. In a shared LSM, the
parent keeps overwriting the same `branch=0` keys, so a naive `branch 0`
fallback read would see *post-branch* parent state. Three options:

- **(a) Materialize-on-read (copy-on-read):** on first access through B, copy
  the parent's row into B's key range, then serve locally. Mutations under B
  always write B's range. Simple, no SlateDB changes, but read amplification
  and a subtle "touched vs. untouched" bookkeeping problem for deletes.
- **(b) MVCC-in-key:** value version = a volume-global sequence number appended
  to the key (`... || id || seq`); `BranchMeta.branch_point` records the
  parent's seq at fork time; resolution reads the greatest visible version ≤
  the ancestor's cut. Point gets become short range scans ("max seq ≤ cut"),
  deletes become versioned tombstones. Principled, but every key grows and
  every read becomes a scan; old versions need their own GC.
- **(c) Hybrid (recommended):** branch-point *metadata* uses (a) — metadata is
  small and fork-time materialization of the parent metadata range is bounded —
  while branch-point *data* needs nothing at all, because extents are
  content-addressed by `FrameLoc` into immutable segment objects. The parent's
  later writes allocate *new* `FrameLoc`s; they never mutate segments the
  fork's extents point at. Segment immutability, which the clone model relies
  on for `external_ssts` sharing, gives forks snapshot-isolated *data* for free.
  Only inode/dir/segcount rows need the copy-on-read treatment.

This is the sharpest difference from s2-lite: streams are append-only with a
trim point, so "the past" is explicitly retained or trimmed; ZeroFS metadata is
mutable, so snapshot semantics must be constructed. The PoC
(`two_lineages_share_one_db_and_reads_resolve_through_base`) demonstrates the
static-base case; (c) is the design for the live-parent case.

---

## 3. Write isolation and fencing

Today: one writer per volume, fenced by the manifest's `writer_epoch`
conditional-put; segments embed the epoch, so a fenced writer's straggler
objects are recognizable and its segcount rows (`epoch < base_epoch` on a fork)
are excluded from reclaim.

In the basin model the manifest epoch still fences the *whole database* — it
cannot distinguish "server X writes branch 3" from "server Y writes branch 7".
Two sub-designs:

**3a. Colocated writers (single server hosts all branches).** Process identity
*is* the fence, exactly as s2-lite does it: append fencing is enforced in
memory by the per-stream `Streamer` state machine
(`lite/src/backend/streamer.rs`, `sequence_records` rejects a mismatched token
before any write), and the token row is only for recovery. ZeroFS already
serializes per-volume writes through its flush coordinator; per-branch writers
behind one coordinator need no new machinery. This covers "thousands of cheap
branches on one server" — the motivating regime.

**3b. Distributed writers (any server may open any branch).** Per-branch
fencing tokens, the s2-lite `StreamFencingToken` pattern
(`lite/src/backend/kv/stream_fencing_token.rs`): `basin || 0x02 || branch`
holds the current token; opening a writer CASes `token+1` in a
`SerializableSnapshot` transaction; every guarded write batch re-checks the
token. The PoC (`branch_scoped_fencing_token_fences_stale_writer`) shows this
working on today's SlateDB API: two opens yield ordered tokens, the stale
writer's guarded put aborts, the current writer's commits. Honest caveat:
SlateDB transactions give snapshot-isolation conflict detection, so the CAS is
correct, but unlike the manifest conditional-put there is no *object-store-level*
fence — a partitioned writer that never touches the token row again could still
have in-flight segment objects. That is already true today (segment objects are
not conditional); the epoch-in-key convention is what makes stragglers
identifiable. Keep it: segments become `segments/{branch}/{shard}/{epoch}/{counter}`,
so ownership stays arithmetically decidable and `SegmentPathRouter` collapses
entirely — every read is local to the one db path.

Writer epochs can remain volume-global (allocated from the manifest) even in
3b; they fence the store, branch tokens fence the key ranges.

---

## 4. GC and reclaim: trim watermarks instead of blanket pinning

Today's coupling is total: `checkpoint_protection`
(`reclaim/driver.rs`) returns `SegmentProtection::Indefinite` if *any*
persistent checkpoint or clone pin exists, so one fork stops segment deletion
for the entire lineage — and a fork's own reclaim ignores ancestor-owned
segments (`cycle.rs:253`). Dead bytes in the root outlive every fork.

s2-lite's replacement (`lite/src/backend/bgtasks/stream_trim.rs`): a stream's
trim point is a durable row; a background task pages through pending trim
points (`tick_stream_trim`), deletes that stream's record rows below the
watermark in batches, and finalizes by clearing the trim row in a serializable
transaction that re-validates it (`finalize_trim` — a stale task can't
over-delete a newer trim point). Deletion is per-stream, incremental,
resumable, and never blocks sibling streams.

Basin-lineage ZeroFS, directly mapped:

- `basin || 0x03 || branch` = the branch's trim watermark: the highest segment
  epoch (within the branch's own epoch scope) whose dead segments are
  collectible.
- A branch's reclaim scans only `[0x09 || branch]` segcount rows — the same
  prefix scan as today's `segcount_prefix_range`, narrowed by branch — and
  deletes only `segments/{branch}/...` objects. The `epoch < base_epoch` guard
  disappears because ownership is the key prefix, not an epoch comparison.
- **Union-GC of shared ancestors.** The root's segments are referenced by
  descendant lineages' extents. A root segment is collectible when its live
  bytes drop to zero *across every lineage that can see it* — i.e., when no
  branch whose `branch_point` covers it still holds references. Two
  implementations: (i) refcount in the segcount value: fork materialization
  (§2.3c) credits the root's counters for inherited live frames, debit on
  branch delete; or (ii) watermarks only: the root's reclaim treats
  `min(branch_point over live branches)` as a moving floor, exactly analogous
  to today's checkpoint retention horizon in `checkpoint_protection`, but
  *per segment epoch* instead of a global pause.
- **What becomes collectible, when:**
  - Clone model: nothing, anywhere in the lineage, while any pin exists;
    everything below the horizon once the last pin drops.
  - Basin model: each branch's own dead segments as soon as its watermark
    passes them (PoC: `per_branch_trim_collects_own_rows_without_pinning_siblings`
    — root trims its dead row while branch 1 exists, branch 1's rows are
    structurally out of range, and branch 1's watermark independently caps what
    *its* task may delete). Root-shared segments become collectible when the
    last referencing branch is deleted or trimmed past them — bounded by the
    slowest branch, not paused by it.

The honest cost: (i) needs careful credit/debit accounting on fork create and
branch delete; (ii) retains root bytes until the *oldest* live branch point
advances, which can again approach pinning if a long-lived fork is never
trimmed. But "paused indefinitely by default" becomes "retained while a named
branch needs it, reclaimable per-branch otherwise".

---

## 5. Checkpoints and time travel

- **Checkpoint (basin model):** a SlateDB checkpoint of the *single* manifest
  atomically pins every branch — a cross-branch consistent snapshot, which the
  clone model cannot provide at all (each fork has its own manifest; there is
  no consistent point across volumes). For segments, a checkpoint of branch B
  records `(B, current epoch watermark)`; reclaim honors it as today's
  checkpoint retention, but per branch.
- **Fork-from-checkpoint:** create the branch row with `branch_point` = the
  checkpoint's sequence/watermark. In §2.3(c) terms, metadata is materialized
  from the checkpointed state, extents reference already-pinned segments. Same
  semantics as `create_fork(name, Some(checkpoint))` today, O(manifest) →
  O(metadata rows).
- **`--at` (point-in-time fork):** today this lists `manifest/` objects and
  picks the last flush before the timestamp (`ForkManager::manifest_at_time`).
  In the basin model there is still exactly one manifest chain, so the same
  trick works verbatim — pick the manifest, take its per-branch watermarks as
  the branch point. Arguably *better*: the chosen point is consistent across
  all branches by construction.

---

## 6. Migration path

- **Key format.** Inserting 8 branch bytes changes every key. Live migration
  of an existing volume means rewriting the LSM (or a format flag with
  dual-read: keys without the branch prefix decode as branch 0 — feasible
  because the inserted bytes are at a fixed offset after a fixed domain+kind
  preamble; read path tries both widths, write path emits only the new width,
  compaction converges the format). ZeroFS already tolerates dual-width value
  decoding (`decode_segcount`'s 8/16-byte legacy path) — same playbook.
- **Segment objects.** `segments/{branch}/...` for new writes; existing
  `segments/{shard}/{epoch}/{counter}` objects decode as branch 0 — the same
  epoch arithmetic `SegmentPathRouter` uses, now mapping epoch→(branch 0).
- **Coexistence.** Keep both. Clone-forks remain the right tool for
  *detach/export*: moving a fork to another bucket, another server fleet, or
  another trust domain. A basin branch can be *exported* into a clone-fork by
  materializing its visible key range (the resolution iterator of §2.2 already
  enumerates exactly that range) into a fresh db path — a O(branch size) copy
  doing what SlateDB's clone does for a whole Db. `fork create --mode=branch`
  (default, O(1)) vs `--mode=clone` (detachable). Existing forks keep working
  untouched — whether their lineage is recorded as `.zerofs_fork.json` (committed
  branch) or as `FORK_LINEAGE`/`FORK_REGISTRY` LSM records (uncommitted WIP),
  since either way the record lives in the fork's own cloned database and the
  basin model never reads it.
- **SlateDB changes required:** none. The PoC runs on the pinned rev
  (`4793fb9`) using only `Db::open`, `put/get/delete/scan/flush`, and
  `begin(SerializableSnapshot)`. Nice-to-haves, all optional: prefix-aware
  bloom/filter policy on the branch dimension (ZeroFS already wires custom
  `filter_policies` and a segment extractor — extending both to skip/parse the
  branch bytes is local); a persisted branch-point *snapshot* handle so
  branch-point reads could use SlateDB snapshots instead of materialization
  (would replace §2.3(a) with seqnum-scoped reads — the cleanest long-term
  answer if SlateDB ever exposes durable, addressable snapshots).

---

## 7. Risks and honest tradeoffs

1. **Compaction coupling.** After compaction, many branches share SSTs.
   Isolation is *key-space* isolation only: branch deletion leaves logically
   dead bytes in shared SSTs until compaction drops them (s2-lite has the same
   property — trimmed records persist in shared SSTs until rewritten); one
   corrupted SST or one bad compaction affects all branches; there is no
   per-branch storage accounting without key-space sampling. Soft-delete a
   branch instantly, reclaim its bytes at compaction cadence.
2. **Blast radius.** One manifest, one memtable flush pipeline, one block
   cache. A branch with a write hot-spot contends with siblings; a poisoned
   manifest kills the whole lineage. The clone model's N manifests are N
   independent failure domains.
3. **Snapshot semantics are constructed, not free** (§2.3). Clone-forks get
   freeze-by-copy from `external_ssts`; basin branches need materialize-on-read
   or MVCC-in-key for mutable metadata. This is the single biggest design
   complexity the clone model avoids.
4. **No object-store fence for distributed branch writers** (§3b) — token CAS
   is LSM-level. Today the manifest conditional-put is a genuine S3-level
   fence. Mitigations mirror today: epochs in segment keys make stragglers
   identifiable; colocated writers (3a) sidestep it.
5. **Crash semantics.** Mid-materialization fork: half-copied metadata rows in
   the branch's range. Resolution is nearest-writer-wins, so a partially
   materialized branch would *shadow the parent inconsistently*. Fix: branch
   row starts in state `Materializing`; readers treat it as transparent
   (fall through to parent) until the row flips to `Active` — a one-row state
   machine, committed last, idempotent on replay (same shape as s2-lite's
   resumable `BasinDeletionPending` cursor).
6. **Lineage depth.** Read cost grows with depth; fork-of-fork-of-fork chains
   accumulate bloom probes per point get and sub-scans per range. Depth cap or
   periodic *rebase* (materialize a branch's full visible state into its own
   range, reset lineage to `[root, B]`).
7. **When clone-forks are genuinely better:** separate buckets/accounts,
   separate servers, separate encryption domains (today keys are per db path
   and `copy_wrapped_key` gives the fork the parent's key — in the basin model
   all branches share one wrapped key, period), hard capacity/accounting
   isolation, and detach/export. Basin lineages are the answer to *many cheap
   branches of one trust domain*, not to *independence*.

---

## 8. Proof of concept

`zerofs/tests/basin_poc.rs` — self-contained, one SlateDB `Db` on an in-memory
object store, no ZeroFS internals. It demonstrates the three load-bearing
mechanisms:

1. **`two_lineages_share_one_db_and_reads_resolve_through_base`** — root +
   two sibling branches as prefixed key ranges; per-lineage reads resolve
   base+branch with nearest-writer-wins; isolation holds in both directions;
   resolution survives `db.flush()` (i.e., works from SSTs, not just the
   memtable).
2. **`branch_scoped_fencing_token_fences_stale_writer`** — s2-lite's
   `StreamFencingToken` as a branch-scoped row: racing opens get ordered
   tokens via serializable transactions, the stale writer is fenced, sibling
   branches have independent token sequences.
3. **`per_branch_trim_collects_own_rows_without_pinning_siblings`** — s2-lite's
   trim task as per-branch watermark + prefix-scoped scan: the root reclaims a
   dead segment *while a branch exists* (impossible today under
   `SegmentProtection::Indefinite`), never touches sibling rows, and honors a
   branch-local watermark floor.

Run: `cargo test --test basin_poc` from `zerofs/` — 3 passed.

Deliberately not demonstrated (future work, all discussed above): live-parent
snapshot isolation (§2.3), merged range scans (§2.2), union refcounting for
shared segments (§4), rebase (§7.6).

---

## Appendix: key files

ZeroFS (this repo):
- `zerofs/src/fs/key_codec.rs` — LSM key layout (`meta`/`extent` domains, kind bytes, segcount)
- `zerofs/src/fork_manager.rs` — clone-per-path fork creation
- `zerofs/src/fork_info.rs` — lineage record, `base_epoch` routing contract
- `zerofs/src/segment_path_router.rs` — epoch-routed ancestor segment reads
- `zerofs/src/segment_store.rs` — `base_epoch`, segment ownership scope
- `zerofs/src/fs/store/extent/reclaim/driver.rs` — `SegmentProtection::Indefinite` blanket pinning
- `zerofs/src/fs/store/extent/reclaim/cycle.rs` — per-segment live-byte reclaim; `epoch < base_epoch` ancestor skip (line 253)
- `zerofs/tests/basin_poc.rs` — the PoC

s2-lite (`github.com/s2-streamstore/s2`, `lite/`):
- `lite/src/backend/kv/mod.rs` — basin key layout: `KeyType` ordinals, `StreamId`-scoped keys
- `lite/src/backend/kv/stream_fencing_token.rs`, `stream_trim_point.rs` — per-stream fencing token / trim point rows
- `lite/src/backend/streamer.rs` — in-memory fencing enforcement at sequence time
- `lite/src/backend/bgtasks/stream_trim.rs` — resumable per-stream background trim (`tick_stream_trim`, `finalize_trim`)
- `lite/src/backend/store.rs` — `db_txn_commit_durable` and serializable-txn helpers

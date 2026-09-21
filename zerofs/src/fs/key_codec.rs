use super::errors::FsError;
use super::inode::InodeId;
use crate::replication::types::{HaStamp, ShipSeqno, SoloHistory, WriterEpoch};
use bytes::Bytes;

// Key layout for the underlying LSM.
//
// Every key is [b"meta" | b"extent"] + [kind: 1] + [branch?: 4] + [id: 8] + ...
//
// The leading domain prefix is what slatedb's segment extractor routes on:
// all metadata kinds land in the `b"meta"` segment, bulk extent pointers in
// `b"extent"`. Each segment is an independent LSM tree, so metadata churn and
// metadata compaction and bulk-data repacking never share an L0 list or lifecycle.
// Metadata/extent isolation is structural, not lexicographic.
//
// Within the meta segment, kind-byte values determine block-level adjacency.
// A `lookup()` touches both INODE and DIR_ENTRY: with kind bytes 0x01/0x02
// adjacent, their entries land in neighbouring blocks of the same meta-segment
// SST, so the read can reuse the same block-cache index/filter entries.
// Similarly, DIR_ENTRY/DIR_SCAN/DIR_COOKIE (0x02/0x03/0x04) are kept adjacent
// for directory operations.
//
// Kind byte assignments (one byte each):
//   0x01 INODE         hot metadata, point-keyed by inode_id
//   0x02 DIR_ENTRY     hot metadata, lookup by (dir_id, name)
//   0x03 DIR_SCAN      hot metadata, ordered scan by (dir_id, cookie)
//   0x04 DIR_COOKIE    per-directory cookie counter
//   0x05 STATS         shard-keyed fs-wide counters
//   0x06 SYSTEM        rare config (e.g. next-inode counter)
//   0x07 TOMBSTONE     deferred-deletion entries, scanned only by tombstone cleanup
//   0x08 ORPHAN        open-unlinked inodes pending reclaim, drained at startup
//   0x09 SEGCOUNT      per-segment (live, total) byte counters, segid-keyed; drives segment reclamation
//   0x0A FORK_LINEAGE  single record holding this volume's fork lineage (present only in forks)
//   0x0B FORK_REGISTRY per-fork registry entry, name-keyed (present only in parents of forks)
//   0x0C FLUSH_TIME    flush-time index: manifest_id -> flush wall-clock time, one row per flush
//   0x0D BRANCH_TOMBSTONE branch-delete marker: shadows a parent-visible key in one branch
//   0x0E BRANCH        branch registry entry, name-keyed; value is versioned JSON {id, created_at}
//   0xFE EXTENT        bulk file data — the only kind in the extent segment
//
// # Branch dimension (basin branches)
//
// A basin branch is a fork-like namespace *inside* this volume's LSM: branch
// creation is O(1) metadata (see [`crate::branch`]), and a branch's reads fall
// back to the parent (branch 0, the volume root) for keys the branch never
// wrote (see [`crate::db::Db::with_branch`]). For a [`KeyCodec`] built with
// [`KeyCodec::for_branch`] and a non-root [`BranchId`], keys of the
// *branch-scoped* kinds gain a 4-byte big-endian branch id immediately after
// the kind byte:
//
// ```text
// branch 0 (unchanged layout):  domain || kind || suffix
// branch > 0, scoped kinds:     domain || kind || branch: u32 BE || suffix
// ```
//
// Branch 0 emits exactly the historical layout, so existing volumes are
// byte-compatible with no format bump. The branch id sits after domain+kind
// (not first) so the segment extractor's domain routing is unchanged and every
// per-kind scan stays a `[kind || branch]` prefix scan per branch.
//
// Scoped vs. unscoped kinds:
//   - Scoped (branch id present for branch > 0): INODE, DIR_ENTRY, DIR_COOKIE,
//     TOMBSTONE, ORPHAN, SEGCOUNT, and EXTENT (the extent domain). These carry
//     per-branch namespace and segment-ownership state.
//   - Unscoped (global layout for every branch): STATS, SYSTEM, DIR_SCAN,
//     FORK_LINEAGE, FORK_REGISTRY, FLUSH_TIME, BRANCH_TOMBSTONE, and BRANCH.
//     These are volume-level state (counters, config, fork/branch bookkeeping).
//     BRANCH_TOMBSTONE keys embed the branch id they shadow for as key payload
//     (`meta || BRANCH_TOMBSTONE || branch || shadowed kind || shadowed
//     suffix`), so the kind itself needs no branch dimension.

const PREFIX_INODE: u8 = 0x01;
const PREFIX_DIR_ENTRY: u8 = 0x02;
const PREFIX_DIR_SCAN: u8 = 0x03;
const PREFIX_DIR_COOKIE: u8 = 0x04;
const PREFIX_STATS: u8 = 0x05;
const PREFIX_SYSTEM: u8 = 0x06;
const PREFIX_TOMBSTONE: u8 = 0x07;
const PREFIX_ORPHAN: u8 = 0x08;
const PREFIX_SEGCOUNT: u8 = 0x09;
const PREFIX_FORK_LINEAGE: u8 = 0x0A;
const PREFIX_FORK_REGISTRY: u8 = 0x0B;
const PREFIX_FLUSH_TIME: u8 = 0x0C;
const PREFIX_BRANCH_TOMBSTONE: u8 = 0x0D;
const PREFIX_BRANCH: u8 = 0x0E;
const PREFIX_EXTENT: u8 = 0xFE;

const SYSTEM_COUNTER_KEY: &[u8; 6] = b"meta\x06\x01";
// HA: the highest shipped replication batch seqno (with its writer epoch) that
// has been flushed into this data db. Written atomically with each shipped
// batch so a promoted standby can prune its tail to exactly what the db already
// holds (see write_coordinator + takeover replay).
const SYSTEM_HA_SEQNO_KEY: &[u8; 6] = b"meta\x06\x02";
// Durability lineage token (see fsync-honesty / ZeroFS::lineage_token). A single
// u64 identifying the current unbroken durable lineage. Regenerated at a cold
// bootstrap or a Solo-tainted takeover; carried forward unchanged at an untainted
// takeover (so a clean failover keeps a client's fsync transparent).
const SYSTEM_LINEAGE_KEY: &[u8; 6] = b"meta\x06\x03";
// Solo taint: set to the lineage token that was live when the leader first
// downgraded to Solo replication. A takeover reads it to decide keep-vs-regenerate
// the lineage token (taint == stored lineage => the lineage may be missing acked
// Solo writes => regenerate, so those writes' fsync fails instead of reporting success).
const SYSTEM_TAINT_KEY: &[u8; 6] = b"meta\x06\x04";
// Wall-clock epoch-seconds (u64 LE via encode_u64) of the last completed slow
// orphan sweep.
const SYSTEM_ORPHAN_SWEEP_KEY: &[u8; 6] = b"meta\x06\x05";
// Basin-branch id allocator: the highest branch id handed out so far (u64 LE
// via encode_u64; only the u32 range is used). Global layout, volume-level: it
// lives on branch 0 like every other System row. Never decremented, so ids
// stay unique even across branch deletion and re-creation.
const SYSTEM_BRANCH_COUNTER_KEY: &[u8; 6] = b"meta\x06\x06";

/// Version byte preceding the timestamp payload of every flush-time index
/// value (see [`KeyCodec::encode_flush_time`]). Durable data: never reuse a
/// lower number with a different layout.
const FLUSH_TIME_RECORD_VERSION: u8 = 1;

/// Version byte used as the entire value of every branch-tombstone row (see
/// [`KeyCodec::branch_tombstone_key`]). The row's *presence* is the signal —
/// "this branch deleted the parent-visible key" — so the value is just a
/// version byte, future-proofing the payload the same way as the fork records.
const BRANCH_TOMBSTONE_RECORD_VERSION: u8 = 1;

const U64_SIZE: usize = std::mem::size_of::<u64>();

/// Bytes the branch id contributes to a scoped key under a non-root branch.
const BRANCH_ID_SIZE: usize = std::mem::size_of::<u32>();

/// Domain prefix for any metadata kind.
pub const META_DOMAIN: &[u8] = b"meta";
/// Domain prefix for bulk extent data.
pub const EXTENT_DOMAIN: &[u8] = b"extent";

const INODE_KEY_SIZE: usize = META_DOMAIN.len() + 1 + U64_SIZE;
const EXTENT_KEY_SIZE: usize = EXTENT_DOMAIN.len() + 1 + U64_SIZE * 2;

const MAX_INODE_KEY_SIZE: usize = INODE_KEY_SIZE + BRANCH_ID_SIZE;
const MAX_EXTENT_KEY_SIZE: usize = EXTENT_KEY_SIZE + BRANCH_ID_SIZE;

/// Identifies a basin branch within a volume's LSM. `BranchId(0)` is the
/// volume root: it uses the historical key layout exactly (no branch bytes),
/// so pre-branch volumes are byte-compatible. See the keyspace comment at the
/// top of this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BranchId(pub u32);

impl BranchId {
    /// The volume root: the implicit branch every pre-branch volume consists of.
    pub const ROOT: BranchId = BranchId(0);

    pub fn is_root(self) -> bool {
        self == Self::ROOT
    }
}

macro_rules! fixed_key {
    ($name:ident, $max:ident) => {
        /// Equality, ordering, and hashing act on the key bytes, not the
        /// fixed-size backing array (whose padding is not part of the key).
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            bytes: [u8; $max],
            len: usize,
        }

        impl $name {
            fn new(bytes: [u8; $max], len: usize) -> Self {
                debug_assert!(len <= $max);
                Self { bytes, len }
            }
        }

        impl AsRef<[u8]> for $name {
            fn as_ref(&self) -> &[u8] {
                &self.bytes[..self.len]
            }
        }

        impl From<$name> for Bytes {
            fn from(key: $name) -> Self {
                Self::copy_from_slice(key.as_ref())
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.as_ref() == other.as_ref()
            }
        }
        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.as_ref().cmp(other.as_ref())
            }
        }

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.as_ref().hash(state);
            }
        }
    };
}

fixed_key!(InodeKey, MAX_INODE_KEY_SIZE);
fixed_key!(ExtentKey, MAX_EXTENT_KEY_SIZE);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyPrefix {
    Inode,
    Extent,
    DirEntry,
    DirScan,
    Tombstone,
    Orphan,
    Stats,
    System,
    DirCookie,
    SegCount,
    ForkLineage,
    ForkRegistry,
    FlushTime,
    BranchTombstone,
    Branch,
}

impl TryFrom<u8> for KeyPrefix {
    type Error = ();

    fn try_from(byte: u8) -> Result<Self, Self::Error> {
        match byte {
            PREFIX_INODE => Ok(Self::Inode),
            PREFIX_EXTENT => Ok(Self::Extent),
            PREFIX_DIR_ENTRY => Ok(Self::DirEntry),
            PREFIX_DIR_SCAN => Ok(Self::DirScan),
            PREFIX_TOMBSTONE => Ok(Self::Tombstone),
            PREFIX_ORPHAN => Ok(Self::Orphan),
            PREFIX_STATS => Ok(Self::Stats),
            PREFIX_SYSTEM => Ok(Self::System),
            PREFIX_DIR_COOKIE => Ok(Self::DirCookie),
            PREFIX_SEGCOUNT => Ok(Self::SegCount),
            PREFIX_FORK_LINEAGE => Ok(Self::ForkLineage),
            PREFIX_FORK_REGISTRY => Ok(Self::ForkRegistry),
            PREFIX_FLUSH_TIME => Ok(Self::FlushTime),
            PREFIX_BRANCH_TOMBSTONE => Ok(Self::BranchTombstone),
            PREFIX_BRANCH => Ok(Self::Branch),
            _ => Err(()),
        }
    }
}

impl From<KeyPrefix> for u8 {
    fn from(prefix: KeyPrefix) -> Self {
        match prefix {
            KeyPrefix::Inode => PREFIX_INODE,
            KeyPrefix::Extent => PREFIX_EXTENT,
            KeyPrefix::DirEntry => PREFIX_DIR_ENTRY,
            KeyPrefix::DirScan => PREFIX_DIR_SCAN,
            KeyPrefix::Tombstone => PREFIX_TOMBSTONE,
            KeyPrefix::Orphan => PREFIX_ORPHAN,
            KeyPrefix::Stats => PREFIX_STATS,
            KeyPrefix::System => PREFIX_SYSTEM,
            KeyPrefix::DirCookie => PREFIX_DIR_COOKIE,
            KeyPrefix::SegCount => PREFIX_SEGCOUNT,
            KeyPrefix::ForkLineage => PREFIX_FORK_LINEAGE,
            KeyPrefix::ForkRegistry => PREFIX_FORK_REGISTRY,
            KeyPrefix::FlushTime => PREFIX_FLUSH_TIME,
            KeyPrefix::BranchTombstone => PREFIX_BRANCH_TOMBSTONE,
            KeyPrefix::Branch => PREFIX_BRANCH,
        }
    }
}

impl KeyPrefix {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inode => "INODE",
            Self::Extent => "EXTENT",
            Self::DirEntry => "DIR_ENTRY",
            Self::DirScan => "DIR_SCAN",
            Self::Tombstone => "TOMBSTONE",
            Self::Orphan => "ORPHAN",
            Self::Stats => "STATS",
            Self::System => "SYSTEM",
            Self::DirCookie => "DIR_COOKIE",
            Self::SegCount => "SEGCOUNT",
            Self::ForkLineage => "FORK_LINEAGE",
            Self::ForkRegistry => "FORK_REGISTRY",
            Self::FlushTime => "FLUSH_TIME",
            Self::BranchTombstone => "BRANCH_TOMBSTONE",
            Self::Branch => "BRANCH",
        }
    }

    /// Whether keys of this kind carry the branch-id dimension under a
    /// non-root branch (see the keyspace comment at the top of this file).
    /// Unscoped kinds are volume-level state and keep the global layout for
    /// every branch.
    pub fn is_scoped(self) -> bool {
        matches!(
            self,
            Self::Inode
                | Self::DirEntry
                | Self::DirCookie
                | Self::Tombstone
                | Self::Orphan
                | Self::SegCount
                | Self::Extent
        )
    }

    fn domain(self) -> &'static [u8] {
        match self {
            KeyPrefix::Extent => EXTENT_DOMAIN,
            _ => META_DOMAIN,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ParsedKey {
    DirScan { cookie: u64 },
    Tombstone { inode_id: InodeId },
    Orphan { inode_id: InodeId },
    Unknown,
}

/// Per-volume key encoder/decoder. Every volume uses the segmented layout, a
/// `b"meta"`/`b"extent"` domain prefix that the slatedb segment extractor
/// routes on. The only state is the [`BranchId`]: [`KeyCodec::new`] builds the
/// root-branch codec (today's exact layout); [`KeyCodec::for_branch`] builds a
/// codec whose scoped-kind keys embed the branch id (see the keyspace comment
/// at the top of this file).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KeyCodec {
    branch: BranchId,
}

impl KeyCodec {
    pub fn new() -> Self {
        Self {
            branch: BranchId::ROOT,
        }
    }

    /// Codec for a basin branch. [`BranchId::ROOT`] produces the exact same
    /// layout as [`KeyCodec::new`]; any other branch inserts
    /// `branch: u32 BE` after the kind byte of every scoped-kind key.
    pub fn for_branch(branch: BranchId) -> Self {
        Self { branch }
    }

    pub fn branch(&self) -> BranchId {
        self.branch
    }

    /// Number of bytes the domain prefix contributes for `prefix`.
    pub fn domain_len(&self, prefix: KeyPrefix) -> usize {
        prefix.domain().len()
    }

    /// Byte offset where the kind byte lives for `prefix`.
    pub fn kind_offset(&self, prefix: KeyPrefix) -> usize {
        self.domain_len(prefix)
    }

    /// Bytes the branch id contributes to keys of `prefix` under this codec:
    /// 4 for scoped kinds under a non-root branch, 0 otherwise.
    fn branch_bytes(&self, prefix: KeyPrefix) -> usize {
        if prefix.is_scoped() && !self.branch.is_root() {
            BRANCH_ID_SIZE
        } else {
            0
        }
    }

    /// Byte offset where the id portion lives for `prefix`. Used by raw
    /// key consumers (tests, verifiers) that need to slice key bytes
    /// without going through a typed parse_*.
    pub fn id_offset(&self, prefix: KeyPrefix) -> usize {
        self.kind_offset(prefix) + 1 + self.branch_bytes(prefix)
    }

    /// Push the domain prefix (if any) plus the kind byte onto `key`,
    /// followed by the branch id for scoped kinds under a non-root branch.
    fn push_prefix(&self, key: &mut Vec<u8>, prefix: KeyPrefix) {
        key.extend_from_slice(prefix.domain());
        key.push(u8::from(prefix));
        if self.branch_bytes(prefix) == BRANCH_ID_SIZE {
            key.extend_from_slice(&self.branch.0.to_be_bytes());
        }
    }

    /// Total bytes in a complete inode key.
    pub fn inode_key_size(&self) -> usize {
        INODE_KEY_SIZE + self.branch_bytes(KeyPrefix::Inode)
    }

    /// Total bytes in a complete extent key.
    pub fn extent_key_size(&self) -> usize {
        EXTENT_KEY_SIZE + self.branch_bytes(KeyPrefix::Extent)
    }

    /// Total bytes in a complete tombstone key.
    pub fn tombstone_key_size(&self) -> usize {
        self.id_offset(KeyPrefix::Tombstone) + U64_SIZE * 2
    }

    /// Total bytes in a complete orphan key.
    pub fn orphan_key_size(&self) -> usize {
        self.id_offset(KeyPrefix::Orphan) + U64_SIZE
    }

    pub fn inode_key(&self, inode_id: InodeId) -> InodeKey {
        let mut bytes = [0; MAX_INODE_KEY_SIZE];
        bytes[..META_DOMAIN.len()].copy_from_slice(META_DOMAIN);
        bytes[self.kind_offset(KeyPrefix::Inode)] = PREFIX_INODE;
        if self.branch_bytes(KeyPrefix::Inode) == BRANCH_ID_SIZE {
            let off = self.kind_offset(KeyPrefix::Inode) + 1;
            bytes[off..off + BRANCH_ID_SIZE].copy_from_slice(&self.branch.0.to_be_bytes());
        }
        let id_offset = self.id_offset(KeyPrefix::Inode);
        bytes[id_offset..id_offset + U64_SIZE].copy_from_slice(&inode_id.to_be_bytes());
        InodeKey::new(bytes, id_offset + U64_SIZE)
    }

    pub fn extent_key(&self, inode_id: InodeId, extent_index: u64) -> ExtentKey {
        let mut bytes = [0; MAX_EXTENT_KEY_SIZE];
        bytes[..EXTENT_DOMAIN.len()].copy_from_slice(EXTENT_DOMAIN);
        bytes[self.kind_offset(KeyPrefix::Extent)] = PREFIX_EXTENT;
        if self.branch_bytes(KeyPrefix::Extent) == BRANCH_ID_SIZE {
            let off = self.kind_offset(KeyPrefix::Extent) + 1;
            bytes[off..off + BRANCH_ID_SIZE].copy_from_slice(&self.branch.0.to_be_bytes());
        }
        let id_offset = self.id_offset(KeyPrefix::Extent);
        bytes[id_offset..id_offset + U64_SIZE].copy_from_slice(&inode_id.to_be_bytes());
        bytes[id_offset + U64_SIZE..id_offset + U64_SIZE * 2]
            .copy_from_slice(&extent_index.to_be_bytes());
        ExtentKey::new(bytes, id_offset + U64_SIZE * 2)
    }

    pub fn parse_extent_key(&self, key: &[u8]) -> Option<u64> {
        let expected = self.extent_key_size();
        if key.len() != expected {
            return None;
        }
        let kind_off = self.kind_offset(KeyPrefix::Extent);
        if !key.starts_with(EXTENT_DOMAIN) {
            return None;
        }
        if key[kind_off] != PREFIX_EXTENT {
            return None;
        }
        let extent_off = self.id_offset(KeyPrefix::Extent) + U64_SIZE;
        let extent_bytes: [u8; U64_SIZE] = key[extent_off..expected].try_into().ok()?;
        Some(u64::from_be_bytes(extent_bytes))
    }

    /// `(inode_id, extent_index)` for an extent key, for the HA standby rebuilding a
    /// shipped segment's directory on takeover.
    pub fn parse_extent_key_full(&self, key: &[u8]) -> Option<(InodeId, u64)> {
        let expected = self.extent_key_size();
        if key.len() != expected {
            return None;
        }
        let kind_off = self.kind_offset(KeyPrefix::Extent);
        if !key.starts_with(EXTENT_DOMAIN) {
            return None;
        }
        if key[kind_off] != PREFIX_EXTENT {
            return None;
        }
        let id_off = self.id_offset(KeyPrefix::Extent);
        let inode = u64::from_be_bytes(key[id_off..id_off + U64_SIZE].try_into().ok()?);
        let extent = u64::from_be_bytes(key[id_off + U64_SIZE..expected].try_into().ok()?);
        Some((inode, extent))
    }

    /// Key for a segment's live-byte counter, keyed by its `(epoch, counter)` segid.
    /// Big-endian id so a prefix scan visits segments in creation order (like every
    /// other id key); the value is `(live, total)` via [`Self::encode_segcount`].
    pub fn segcount_key(&self, epoch: u64, counter: u64) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::SegCount) + U64_SIZE * 2);
        self.push_prefix(&mut key, KeyPrefix::SegCount);
        key.extend_from_slice(&epoch.to_be_bytes());
        key.extend_from_slice(&counter.to_be_bytes());
        Bytes::from(key)
    }

    /// `(epoch, counter)` of a segcount key, for the reclaim / rebuild keyspace scan.
    pub fn parse_segcount_key(&self, key: &[u8]) -> Option<(u64, u64)> {
        let id_off = self.id_offset(KeyPrefix::SegCount);
        if key.len() != id_off + U64_SIZE * 2
            || !key.starts_with(META_DOMAIN)
            || key[self.kind_offset(KeyPrefix::SegCount)] != PREFIX_SEGCOUNT
        {
            return None;
        }
        let epoch = u64::from_be_bytes(key[id_off..id_off + U64_SIZE].try_into().ok()?);
        let counter = u64::from_be_bytes(key[id_off + U64_SIZE..].try_into().ok()?);
        Some((epoch, counter))
    }

    /// Half-open `[start, end)` covering every segcount key, for a full scan.
    pub fn segcount_prefix_range(&self) -> (Bytes, Bytes) {
        self.prefix_range(KeyPrefix::SegCount)
    }

    /// `(epoch, counter)` of a segcount key in ANY branch's layout — root
    /// (`kind || epoch || counter`) or branch-qualified (`kind || branch ||
    /// epoch || counter`); the embedded branch id is skipped, not validated.
    /// Volume-wide maintenance that must see every scope's counters at once
    /// (the orphan sweep's liveness census) uses this; per-scope scans use
    /// [`Self::parse_segcount_key`]. Suffix length disambiguates the layouts:
    /// branch-qualified keys are exactly [`BRANCH_ID_SIZE`] bytes longer.
    pub fn parse_any_segcount_key(key: &[u8]) -> Option<(u64, u64)> {
        let root = KeyCodec::new();
        let base = root.id_offset(KeyPrefix::SegCount);
        if !key.starts_with(META_DOMAIN)
            || key.get(root.kind_offset(KeyPrefix::SegCount)) != Some(&PREFIX_SEGCOUNT)
        {
            return None;
        }
        let suffix = key.get(base..)?;
        let ids = match suffix.len() {
            n if n == U64_SIZE * 2 => suffix,
            n if n == BRANCH_ID_SIZE + U64_SIZE * 2 => &suffix[BRANCH_ID_SIZE..],
            _ => return None,
        };
        let epoch = u64::from_be_bytes(ids[..U64_SIZE].try_into().ok()?);
        let counter = u64::from_be_bytes(ids[U64_SIZE..].try_into().ok()?);
        Some((epoch, counter))
    }

    /// Key for this volume's own fork-lineage record: a single record per
    /// database, present only in forks (see [`crate::fork_info`]). A fork's
    /// startup reads it to route segment reads across the lineage.
    pub fn fork_lineage_key(&self) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::ForkLineage));
        self.push_prefix(&mut key, KeyPrefix::ForkLineage);
        Bytes::from(key)
    }

    /// Key for the registry entry naming a direct fork of this volume. One
    /// entry per fork lives in the *parent's* database, so `list_forks` is a
    /// prefix scan of the parent's own LSM.
    pub fn fork_registry_key(&self, name: &str) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::ForkRegistry) + name.len());
        self.push_prefix(&mut key, KeyPrefix::ForkRegistry);
        key.extend_from_slice(name.as_bytes());
        Bytes::from(key)
    }

    /// Prefix covering every fork-registry entry, for `scan_prefix`.
    pub fn fork_registry_prefix(&self) -> Bytes {
        let mut prefix = Vec::with_capacity(self.id_offset(KeyPrefix::ForkRegistry));
        self.push_prefix(&mut prefix, KeyPrefix::ForkRegistry);
        Bytes::from(prefix)
    }

    /// Flush-time index entry: `manifest_id -> flush wall-clock time`, one row
    /// per completed flush, written by the flush coordinator after the flush
    /// barrier (see [`crate::fs::flush_coordinator`]). Point-in-time forks
    /// resolve a timestamp to the greatest indexed manifest id at or before it
    /// (see [`crate::fork_manager::ForkManager::manifest_at_time`]). The
    /// manifest id is big-endian so a prefix scan visits flushes in
    /// publication order.
    pub fn flush_time_key(&self, manifest_id: u64) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::FlushTime) + U64_SIZE);
        self.push_prefix(&mut key, KeyPrefix::FlushTime);
        key.extend_from_slice(&manifest_id.to_be_bytes());
        Bytes::from(key)
    }

    /// Manifest id of a flush-time index key.
    pub fn parse_flush_time_key(&self, key: &[u8]) -> Option<u64> {
        let id_off = self.id_offset(KeyPrefix::FlushTime);
        if key.len() != id_off + U64_SIZE
            || !key.starts_with(META_DOMAIN)
            || key[self.kind_offset(KeyPrefix::FlushTime)] != PREFIX_FLUSH_TIME
        {
            return None;
        }
        Some(u64::from_be_bytes(key[id_off..].try_into().ok()?))
    }

    /// Prefix covering every flush-time index entry, for `scan_prefix`.
    pub fn flush_time_prefix(&self) -> Bytes {
        let mut prefix = Vec::with_capacity(self.id_offset(KeyPrefix::FlushTime));
        self.push_prefix(&mut prefix, KeyPrefix::FlushTime);
        Bytes::from(prefix)
    }

    /// Encode a flush-time index value: a version byte followed by the flush
    /// time as `(epoch seconds, sub-second nanos)`, both little-endian. The
    /// version byte follows the fork-record convention (`0x01 || payload`, see
    /// [`crate::fork_info`]); the payload itself is fixed-width LE integers
    /// like every other scalar value in this keyspace (see
    /// [`Self::encode_u64`]) — JSON would buy nothing for two integers.
    ///
    /// Sub-second precision matters: a point-in-time target can fall within
    /// the same second as a later flush, and second granularity would let
    /// that later flush's manifest compare `<=` the target and be wrongly
    /// selected. Index writes are best-effort and idempotent (keyed by
    /// manifest id), so a re-recorded flush simply overwrites its row.
    pub fn encode_flush_time(epoch_seconds: u64, subsec_nanos: u32) -> Bytes {
        let mut v = Vec::with_capacity(1 + U64_SIZE + 4);
        v.push(FLUSH_TIME_RECORD_VERSION);
        v.extend_from_slice(&epoch_seconds.to_le_bytes());
        v.extend_from_slice(&subsec_nanos.to_le_bytes());
        Bytes::from(v)
    }

    /// Decode a flush-time index value into `(epoch seconds, sub-second
    /// nanos)`, rejecting unknown versions rather than silently misreading a
    /// future layout.
    pub fn decode_flush_time(data: &[u8]) -> Option<(u64, u32)> {
        let (&version, payload) = data.split_first()?;
        if version != FLUSH_TIME_RECORD_VERSION || payload.len() != U64_SIZE + 4 {
            return None;
        }
        let secs = u64::from_le_bytes(payload[..U64_SIZE].try_into().ok()?);
        let nanos = u32::from_le_bytes(payload[U64_SIZE..].try_into().ok()?);
        Some((secs, nanos))
    }

    pub fn dir_entry_key(&self, dir_id: InodeId, name: &[u8]) -> Bytes {
        let mut key =
            Vec::with_capacity(self.id_offset(KeyPrefix::DirEntry) + U64_SIZE + name.len());
        self.push_prefix(&mut key, KeyPrefix::DirEntry);
        key.extend_from_slice(&dir_id.to_be_bytes());
        key.extend_from_slice(name);
        Bytes::from(key)
    }

    /// Prefix covering every dir-entry of `dir_id`, ordered by name, for
    /// `scan_prefix`. Branch codecs embed the branch id, so a branch's
    /// dir-entry prefix scan sees only its own rows unless the database
    /// merges scopes (see [`crate::db::Db::scan_prefix`]).
    pub fn dir_entry_prefix(&self, dir_id: InodeId) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(self.id_offset(KeyPrefix::DirEntry) + U64_SIZE);
        self.push_prefix(&mut prefix, KeyPrefix::DirEntry);
        prefix.extend_from_slice(&dir_id.to_be_bytes());
        prefix
    }

    pub fn dir_scan_key(&self, dir_id: InodeId, cookie: u64) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::DirScan) + U64_SIZE * 2);
        self.push_prefix(&mut key, KeyPrefix::DirScan);
        key.extend_from_slice(&dir_id.to_be_bytes());
        key.extend_from_slice(&cookie.to_be_bytes());
        Bytes::from(key)
    }

    pub fn dir_scan_prefix(&self, dir_id: InodeId) -> Vec<u8> {
        let mut prefix = Vec::with_capacity(self.id_offset(KeyPrefix::DirScan) + U64_SIZE);
        self.push_prefix(&mut prefix, KeyPrefix::DirScan);
        prefix.extend_from_slice(&dir_id.to_be_bytes());
        prefix
    }

    /// Build a key for resuming dir scan from a specific cookie
    pub fn dir_scan_resume_key(&self, dir_id: InodeId, resume_after_cookie: u64) -> Bytes {
        let mut key = self.dir_scan_prefix(dir_id);
        key.extend_from_slice(&(resume_after_cookie + 1).to_be_bytes());
        Bytes::from(key)
    }

    /// Key for storing next cookie counter per directory
    pub fn dir_cookie_counter_key(&self, dir_id: InodeId) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::DirCookie) + U64_SIZE);
        self.push_prefix(&mut key, KeyPrefix::DirCookie);
        key.extend_from_slice(&dir_id.to_be_bytes());
        Bytes::from(key)
    }

    pub fn tombstone_key(&self, timestamp: u64, inode_id: InodeId) -> Bytes {
        let mut key = Vec::with_capacity(self.tombstone_key_size());
        self.push_prefix(&mut key, KeyPrefix::Tombstone);
        key.extend_from_slice(&timestamp.to_be_bytes());
        key.extend_from_slice(&inode_id.to_be_bytes());
        Bytes::from(key)
    }

    /// Key for an orphan-set entry. The inode_id alone keys the entry (no
    /// timestamp, unlike tombstones): presence signals "open-unlinked, pending
    /// reclaim", so it must be a point key the reclaim path can delete by id.
    pub fn orphan_key(&self, inode_id: InodeId) -> Bytes {
        let mut key = Vec::with_capacity(self.orphan_key_size());
        self.push_prefix(&mut key, KeyPrefix::Orphan);
        key.extend_from_slice(&inode_id.to_be_bytes());
        Bytes::from(key)
    }

    pub fn stats_shard_key(&self, shard_id: usize) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::Stats) + U64_SIZE);
        self.push_prefix(&mut key, KeyPrefix::Stats);
        key.extend_from_slice(&(shard_id as u64).to_be_bytes());
        Bytes::from(key)
    }

    pub fn system_counter_key(&self) -> Bytes {
        Bytes::from_static(SYSTEM_COUNTER_KEY)
    }

    /// Key for HA provenance flushed atomically with each replicated leader
    /// batch. The stamp records the writer epoch, acknowledged ship, Solo
    /// history, and highest locally applied ship attempt; takeover validates the
    /// volatile tail and exact-result ledger against it.
    pub fn ha_seqno_key(&self) -> Bytes {
        Bytes::from_static(SYSTEM_HA_SEQNO_KEY)
    }

    pub(crate) fn encode_ha_stamp(stamp: &HaStamp) -> Bytes {
        let mut v = Vec::with_capacity(U64_SIZE * 5);
        v.extend_from_slice(&stamp.writer_epoch().get().to_le_bytes());
        v.extend_from_slice(&stamp.last_shipped().map_or(0, ShipSeqno::get).to_le_bytes());
        v.extend_from_slice(&stamp.solo_history().commits_since_last_ship().to_le_bytes());
        v.extend_from_slice(
            &stamp
                .applied_through()
                .map_or(0, ShipSeqno::get)
                .to_le_bytes(),
        );
        v.extend_from_slice(&u64::from(stamp.solo_history().ran_solo()).to_le_bytes());
        Bytes::from(v)
    }

    /// Key for the durability lineage token (a single u64).
    pub fn lineage_key(&self) -> Bytes {
        Bytes::from_static(SYSTEM_LINEAGE_KEY)
    }

    /// Key for the Solo taint (the lineage token that went Solo).
    pub fn taint_key(&self) -> Bytes {
        Bytes::from_static(SYSTEM_TAINT_KEY)
    }

    /// Key for the last-orphan-sweep wall-clock timestamp (epoch seconds, a u64 via
    /// [`Self::encode_u64`]).
    pub fn last_orphan_sweep_key(&self) -> Bytes {
        Bytes::from_static(SYSTEM_ORPHAN_SWEEP_KEY)
    }

    pub fn encode_u64(value: u64) -> Bytes {
        Bytes::copy_from_slice(&value.to_le_bytes())
    }

    pub fn decode_u64(data: &[u8]) -> Option<u64> {
        if data.len() != U64_SIZE {
            return None;
        }
        Some(u64::from_le_bytes(data[..U64_SIZE].try_into().ok()?))
    }

    /// Encode a segment counter value: `live` bytes (decrements as frames die) and
    /// `total` bytes (cumulative frame bytes ever appended, monotonic). Reclamation reads
    /// `live/total` as the segment's live fraction straight from this value, so the
    /// counter-based reclaim never has to list the object to get its size.
    pub fn encode_segcount(live: u64, total: u64) -> Bytes {
        let mut b = [0u8; U64_SIZE * 2];
        b[..U64_SIZE].copy_from_slice(&live.to_le_bytes());
        b[U64_SIZE..].copy_from_slice(&total.to_le_bytes());
        Bytes::copy_from_slice(&b)
    }

    /// Decode a segment counter value. A legacy 8-byte value (live only, pre-`total`)
    /// decodes as `(live, live)`: it treats the whole segment as live until its
    /// counter is next rewritten, which is the safe (over-count) direction.
    pub fn decode_segcount(data: &[u8]) -> Option<(u64, u64)> {
        if data.len() == U64_SIZE * 2 {
            let live = u64::from_le_bytes(data[..U64_SIZE].try_into().ok()?);
            let total = u64::from_le_bytes(data[U64_SIZE..].try_into().ok()?);
            Some((live, total))
        } else if data.len() == U64_SIZE {
            let live = u64::from_le_bytes(data[..U64_SIZE].try_into().ok()?);
            Some((live, live))
        } else {
            None
        }
    }

    /// Decode and validate a durable HA provenance stamp.
    ///
    /// The 16- and 24-byte layouts predate the applied frontier and Solo-history
    /// latch. They are durable data and can survive a coordinated binary upgrade,
    /// so decode them conservatively: the acknowledged ship is known applied, but
    /// the term cannot prove it never ran Solo and therefore cannot authorize
    /// promotion retry grace.
    pub(crate) fn decode_ha_stamp(data: &[u8]) -> Option<HaStamp> {
        if data.len() != U64_SIZE * 2 && data.len() != U64_SIZE * 3 && data.len() != U64_SIZE * 5 {
            return None;
        }
        let writer_epoch = WriterEpoch::new(u64::from_le_bytes(data[..U64_SIZE].try_into().ok()?))?;
        let last_shipped = ShipSeqno::new(u64::from_le_bytes(
            data[U64_SIZE..U64_SIZE * 2].try_into().ok()?,
        ));
        if data.len() == U64_SIZE * 2 {
            return HaStamp::new(
                writer_epoch,
                last_shipped,
                SoloHistory::ever(0),
                last_shipped,
            )
            .ok();
        }
        let solo = u64::from_le_bytes(data[U64_SIZE * 2..U64_SIZE * 3].try_into().ok()?);
        if data.len() == U64_SIZE * 3 {
            return HaStamp::new(
                writer_epoch,
                last_shipped,
                SoloHistory::ever(solo),
                last_shipped,
            )
            .ok();
        }
        let applied_through = ShipSeqno::new(u64::from_le_bytes(
            data[U64_SIZE * 3..U64_SIZE * 4].try_into().ok()?,
        ));
        let solo_ever = u64::from_le_bytes(data[U64_SIZE * 4..].try_into().ok()?);
        let ran_solo = match solo_ever {
            0 => false,
            1 => true,
            _ => return None,
        };
        let solo_history = SoloHistory::from_parts(solo, ran_solo)?;
        HaStamp::new(writer_epoch, last_shipped, solo_history, applied_through).ok()
    }

    pub fn parse_key(&self, key: &[u8]) -> ParsedKey {
        let kind = match self.peek_kind(key) {
            Some(k) => k,
            None => return ParsedKey::Unknown,
        };
        let id_off = self.id_offset(kind);

        match kind {
            KeyPrefix::DirScan => {
                let expected = id_off + U64_SIZE * 2;
                if key.len() != expected {
                    return ParsedKey::Unknown;
                }
                if let Ok(cookie_bytes) = key[id_off + U64_SIZE..expected].try_into() {
                    let cookie = u64::from_be_bytes(cookie_bytes);
                    ParsedKey::DirScan { cookie }
                } else {
                    ParsedKey::Unknown
                }
            }
            KeyPrefix::Tombstone => {
                let expected = self.tombstone_key_size();
                if key.len() != expected {
                    return ParsedKey::Unknown;
                }
                if let Ok(id_bytes) = key[id_off + U64_SIZE..expected].try_into() {
                    ParsedKey::Tombstone {
                        inode_id: u64::from_be_bytes(id_bytes),
                    }
                } else {
                    ParsedKey::Unknown
                }
            }
            KeyPrefix::Orphan => {
                let expected = self.orphan_key_size();
                if key.len() != expected {
                    return ParsedKey::Unknown;
                }
                if let Ok(id_bytes) = key[id_off..expected].try_into() {
                    ParsedKey::Orphan {
                        inode_id: u64::from_be_bytes(id_bytes),
                    }
                } else {
                    ParsedKey::Unknown
                }
            }
            _ => ParsedKey::Unknown,
        }
    }

    /// Decode the kind byte from a stored key. Returns `None` if the key
    /// is too short, lacks the expected domain prefix, or carries a kind
    /// byte we don't recognize. Layout-independent: only the domain and
    /// kind bytes are read, so any branch's codec can decode any branch's
    /// keys.
    pub(crate) fn peek_kind(&self, key: &[u8]) -> Option<KeyPrefix> {
        // Dispatch on the leading domain prefix to pick which kind byte to read.
        if let Some(rest) = key.strip_prefix(EXTENT_DOMAIN) {
            let kind = KeyPrefix::try_from(*rest.first()?).ok()?;
            return (kind == KeyPrefix::Extent).then_some(kind);
        }
        if let Some(rest) = key.strip_prefix(META_DOMAIN) {
            let kind = KeyPrefix::try_from(*rest.first()?).ok()?;
            return (kind != KeyPrefix::Extent).then_some(kind);
        }
        None
    }

    pub fn encode_counter(value: u64) -> Bytes {
        Bytes::copy_from_slice(&value.to_le_bytes())
    }

    pub fn decode_counter(data: &[u8]) -> Result<u64, FsError> {
        if data.len() != U64_SIZE {
            return Err(FsError::InvalidData);
        }
        let bytes: [u8; U64_SIZE] = data.try_into().map_err(|_| FsError::InvalidData)?;
        Ok(u64::from_le_bytes(bytes))
    }

    pub fn encode_dir_entry(inode_id: InodeId, cookie: u64) -> Bytes {
        let mut value = Vec::with_capacity(U64_SIZE * 2);
        value.extend_from_slice(&inode_id.to_le_bytes());
        value.extend_from_slice(&cookie.to_le_bytes());
        Bytes::from(value)
    }

    pub fn decode_dir_entry(data: &[u8]) -> Result<(InodeId, u64), FsError> {
        if data.len() < U64_SIZE * 2 {
            return Err(FsError::InvalidData);
        }
        let inode_bytes: [u8; U64_SIZE] = data[..U64_SIZE]
            .try_into()
            .map_err(|_| FsError::InvalidData)?;
        let cookie_bytes: [u8; U64_SIZE] = data[U64_SIZE..U64_SIZE * 2]
            .try_into()
            .map_err(|_| FsError::InvalidData)?;
        Ok((
            u64::from_le_bytes(inode_bytes),
            u64::from_le_bytes(cookie_bytes),
        ))
    }

    pub fn encode_tombstone_size(size: u64) -> Bytes {
        Bytes::copy_from_slice(&size.to_le_bytes())
    }

    pub fn decode_tombstone_size(data: &[u8]) -> Result<u64, FsError> {
        if data.len() != U64_SIZE {
            return Err(FsError::InvalidData);
        }
        let bytes: [u8; U64_SIZE] = data.try_into().map_err(|_| FsError::InvalidData)?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Half-open `[start, end)` range covering every key of `prefix` *under
    /// this codec's branch*. For a non-root branch and a scoped kind the
    /// range brackets exactly this branch's slice of the kind (`kind ||
    /// branch` up to `kind || branch + 1`); for the root branch or unscoped
    /// kinds it is the whole kind, exactly as before the branch dimension
    /// existed (`end` is the prefix bytes with the kind byte incremented, so
    /// the range stays within the domain segment).
    pub fn prefix_range(&self, prefix: KeyPrefix) -> (Bytes, Bytes) {
        let mut start = Vec::with_capacity(self.id_offset(prefix));
        self.push_prefix(&mut start, prefix);
        let mut end = start.clone();
        if self.branch_bytes(prefix) == BRANCH_ID_SIZE && self.branch.0 < u32::MAX {
            // End at this branch's successor: the range covers this branch's
            // keys and no sibling branch's. The kind byte sits at
            // `kind_offset`; the 4 branch bytes follow it.
            let branch_off = self.kind_offset(prefix) + 1;
            end[branch_off..branch_off + BRANCH_ID_SIZE]
                .copy_from_slice(&(self.branch.0 + 1).to_be_bytes());
        } else {
            // The kind byte we just pushed is at `kind_offset`. The end of
            // the range is the same bytes with that kind byte incremented by
            // 1 (and no branch bytes, for unscoped kinds and the root).
            end.truncate(self.kind_offset(prefix) + 1);
            let last_idx = end.len() - 1;
            end[last_idx] += 1;
        }
        (Bytes::from(start), Bytes::from(end))
    }

    /// Key marking that this codec's branch deleted the parent-visible key
    /// `shadowed` (a full scoped key built by this same branch codec):
    /// `meta || BRANCH_TOMBSTONE || branch || shadowed kind || shadowed
    /// suffix`. Presence of the row is the whole signal; the value is
    /// [`Self::branch_tombstone_value`]. Returns `None` if `shadowed` is not
    /// a well-formed scoped key of this branch.
    pub fn branch_tombstone_key(&self, shadowed: &[u8]) -> Option<Bytes> {
        let (kind, suffix) = self.split_scoped_key(shadowed)?;
        let mut key =
            Vec::with_capacity(META_DOMAIN.len() + 1 + BRANCH_ID_SIZE + 1 + suffix.len());
        key.extend_from_slice(META_DOMAIN);
        key.push(PREFIX_BRANCH_TOMBSTONE);
        key.extend_from_slice(&self.branch.0.to_be_bytes());
        key.push(u8::from(kind));
        key.extend_from_slice(suffix);
        Some(Bytes::from(key))
    }

    /// Value written with every branch-tombstone row: a bare version byte
    /// (presence is the signal; see [`BRANCH_TOMBSTONE_RECORD_VERSION`]).
    pub fn branch_tombstone_value() -> Bytes {
        Bytes::from_static(&[BRANCH_TOMBSTONE_RECORD_VERSION])
    }

    /// Prefix covering this branch's tombstone rows that shadow keys of
    /// `kind` whose suffix starts with `suffix_prefix` (`meta ||
    /// BRANCH_TOMBSTONE || branch || kind || suffix_prefix`, mirroring
    /// [`Self::branch_tombstone_key`]). A merged branch scan prefetches this
    /// range once to decide which parent-scope rows to suppress, instead of
    /// a point lookup per candidate row.
    pub fn branch_tombstone_prefix(&self, kind: KeyPrefix, suffix_prefix: &[u8]) -> Bytes {
        let mut key =
            Vec::with_capacity(META_DOMAIN.len() + 1 + BRANCH_ID_SIZE + 1 + suffix_prefix.len());
        key.extend_from_slice(META_DOMAIN);
        key.push(PREFIX_BRANCH_TOMBSTONE);
        key.extend_from_slice(&self.branch.0.to_be_bytes());
        key.push(u8::from(kind));
        key.extend_from_slice(suffix_prefix);
        Bytes::from(key)
    }

    /// The tombstone-range equivalent of a parent-view scan bound (see
    /// [`Self::branch_tombstone_prefix`]): the bound's bytes from its kind
    /// byte onward, re-rooted under this branch's tombstone header. The kind
    /// byte is copied verbatim, so a bound at `kind + 1` (the exclusive end
    /// of a whole-kind scan) maps to the end of that kind's tombstone space
    /// even when `kind + 1` is itself an unscoped or unknown kind. Returns
    /// `None` on a root codec or a bound outside both domains.
    pub fn branch_tombstone_bound(&self, parent_bound: &[u8]) -> Option<Bytes> {
        if self.branch.is_root() {
            return None;
        }
        let from_kind = parent_bound
            .strip_prefix(META_DOMAIN)
            .or_else(|| parent_bound.strip_prefix(EXTENT_DOMAIN))?;
        if from_kind.is_empty() {
            return None;
        }
        let mut key =
            Vec::with_capacity(META_DOMAIN.len() + 1 + BRANCH_ID_SIZE + from_kind.len());
        key.extend_from_slice(META_DOMAIN);
        key.push(PREFIX_BRANCH_TOMBSTONE);
        key.extend_from_slice(&self.branch.0.to_be_bytes());
        key.extend_from_slice(from_kind);
        Some(Bytes::from(key))
    }

    /// Half-open `[start, end)` covering every BRANCH_TOMBSTONE row of this
    /// codec's branch (`meta || BRANCH_TOMBSTONE || branch` up to `branch +
    /// 1`), for dropping them when the branch itself is deleted. Returns
    /// `None` on a root codec: the root has no branch-tombstone rows, and the
    /// whole-kind range would cover OTHER branches' rows.
    pub fn branch_tombstone_range(&self) -> Option<(Bytes, Bytes)> {
        if self.branch.is_root() {
            return None;
        }
        let mut start =
            Vec::with_capacity(META_DOMAIN.len() + 1 + BRANCH_ID_SIZE);
        start.extend_from_slice(META_DOMAIN);
        start.push(PREFIX_BRANCH_TOMBSTONE);
        start.extend_from_slice(&self.branch.0.to_be_bytes());
        let mut end = start.clone();
        if self.branch.0 < u32::MAX {
            let branch_off = META_DOMAIN.len() + 1;
            end[branch_off..branch_off + BRANCH_ID_SIZE]
                .copy_from_slice(&(self.branch.0 + 1).to_be_bytes());
        } else {
            // u32::MAX has no successor; bracket with the next kind byte.
            end.truncate(META_DOMAIN.len() + 1);
            let last_idx = end.len() - 1;
            end[last_idx] += 1;
        }
        Some((Bytes::from(start), Bytes::from(end)))
    }

    /// Split a branch-tombstone row's key into the shadowed kind and suffix
    /// (the inverse of the key half of [`Self::branch_tombstone_key`]).
    /// Returns `None` for another branch's rows or malformed keys.
    pub fn parse_branch_tombstone_key<'a>(
        &self,
        key: &'a [u8],
    ) -> Option<(KeyPrefix, &'a [u8])> {
        let rest = key.strip_prefix(META_DOMAIN)?;
        let (&kind_byte, rest) = rest.split_first()?;
        if kind_byte != PREFIX_BRANCH_TOMBSTONE {
            return None;
        }
        let branch: [u8; BRANCH_ID_SIZE] = rest.get(..BRANCH_ID_SIZE)?.try_into().ok()?;
        if BranchId(u32::from_be_bytes(branch)) != self.branch {
            return None;
        }
        let rest = &rest[BRANCH_ID_SIZE..];
        let (&shadowed_kind, suffix) = rest.split_first()?;
        let shadowed_kind = KeyPrefix::try_from(shadowed_kind).ok()?;
        Some((shadowed_kind, suffix))
    }

    /// The suffix of a scoped key as this codec sees it: the bytes after
    /// `domain || kind` for the root codec, or `domain || kind || branch` for
    /// a non-root codec (with the embedded branch id validated). This is the
    /// comparator key for merged branch scans: a parent row and a branch row
    /// name the same logical entry exactly when their suffixes are equal.
    /// Returns `None` for unrecognized or unscoped keys and for another
    /// branch's keys.
    pub fn scoped_suffix<'a>(&self, key: &'a [u8]) -> Option<&'a [u8]> {
        let kind = self.peek_kind(key)?;
        if !kind.is_scoped() {
            return None;
        }
        if self.branch_bytes(kind) == BRANCH_ID_SIZE {
            let branch_off = self.kind_offset(kind) + 1;
            let branch: [u8; BRANCH_ID_SIZE] = key
                .get(branch_off..branch_off + BRANCH_ID_SIZE)?
                .try_into()
                .ok()?;
            if BranchId(u32::from_be_bytes(branch)) != self.branch {
                return None;
            }
        }
        key.get(self.id_offset(kind)..)
    }

    /// The branch-layout form of a parent (root-layout) scoped key: the same
    /// bytes with this codec's branch id inserted after the kind byte — the
    /// inverse of [`Self::strip_branch`]. Merged branch scans re-key
    /// surviving parent rows this way so every emitted key parses under the
    /// branch codec. Returns `None` on a root codec or for unrecognized or
    /// unscoped keys.
    pub fn adopt_parent_key(&self, parent: &[u8]) -> Option<Bytes> {
        if self.branch.is_root() {
            return None;
        }
        let kind = self.peek_kind(parent)?;
        if !kind.is_scoped() {
            return None;
        }
        let kind_off = self.kind_offset(kind);
        let mut key = Vec::with_capacity(parent.len() + BRANCH_ID_SIZE);
        key.extend_from_slice(&parent[..kind_off + 1]);
        key.extend_from_slice(&self.branch.0.to_be_bytes());
        key.extend_from_slice(&parent[kind_off + 1..]);
        Some(Bytes::from(key))
    }

    /// The parent (root-branch) form of a scoped key built by this branch
    /// codec: the same bytes with the 4 branch bytes removed. Returns `None`
    /// for unrecognized, unscoped, root-branch, or other-branch keys — the
    /// caller must not silently treat those as parent keys.
    pub fn strip_branch(&self, key: &[u8]) -> Option<Bytes> {
        let (kind, suffix) = self.split_scoped_key(key)?;
        let kind_off = self.kind_offset(kind);
        let mut parent = Vec::with_capacity(key.len() - BRANCH_ID_SIZE);
        parent.extend_from_slice(&key[..kind_off + 1]);
        parent.extend_from_slice(suffix);
        Some(Bytes::from(parent))
    }

    /// Split a scoped key built by this branch codec into its kind and the
    /// suffix after the branch id, validating the domain, kind, scope, and
    /// embedded branch id. The root branch has no branch bytes to split.
    fn split_scoped_key<'a>(&self, key: &'a [u8]) -> Option<(KeyPrefix, &'a [u8])> {
        if self.branch.is_root() {
            return None;
        }
        let kind = self.peek_kind(key)?;
        if !kind.is_scoped() {
            return None;
        }
        let branch_off = self.kind_offset(kind) + 1;
        let branch_bytes: [u8; BRANCH_ID_SIZE] =
            key.get(branch_off..branch_off + BRANCH_ID_SIZE)?.try_into().ok()?;
        if BranchId(u32::from_be_bytes(branch_bytes)) != self.branch {
            return None;
        }
        Some((kind, &key[branch_off + BRANCH_ID_SIZE..]))
    }

    /// Key for the registry entry naming a basin branch of this volume:
    /// `meta || BRANCH || name`, global layout (the BRANCH kind is unscoped —
    /// the registry itself is volume-level state). The value is a versioned
    /// JSON [`crate::branch::BranchRecord`].
    pub fn branch_registry_key(&self, name: &str) -> Bytes {
        let mut key = Vec::with_capacity(self.id_offset(KeyPrefix::Branch) + name.len());
        self.push_prefix(&mut key, KeyPrefix::Branch);
        key.extend_from_slice(name.as_bytes());
        Bytes::from(key)
    }

    /// Prefix covering every branch-registry entry, for `scan_prefix`.
    pub fn branch_registry_prefix(&self) -> Bytes {
        let mut prefix = Vec::with_capacity(self.id_offset(KeyPrefix::Branch));
        self.push_prefix(&mut prefix, KeyPrefix::Branch);
        Bytes::from(prefix)
    }

    /// Key for the branch-id allocator counter (highest id handed out so far,
    /// a u64 via [`Self::encode_u64`]). Global layout, volume-level state.
    pub fn branch_counter_key(&self) -> Bytes {
        Bytes::from_static(SYSTEM_BRANCH_COUNTER_KEY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dir_scan_parsing() {
        let codec = KeyCodec::new();
        let dir_id = 10u64;
        let cookie = 42u64;
        let key = codec.dir_scan_key(dir_id, cookie);

        match codec.parse_key(&key) {
            ParsedKey::DirScan {
                cookie: parsed_cookie,
            } => {
                assert_eq!(parsed_cookie, cookie);
            }
            _ => panic!("Failed to parse dir scan key"),
        }
    }

    #[test]
    fn test_tombstone_parsing() {
        let codec = KeyCodec::new();
        let timestamp = 123456u64;
        let inode_id = 789u64;
        let key = codec.tombstone_key(timestamp, inode_id);

        match codec.parse_key(&key) {
            ParsedKey::Tombstone {
                inode_id: parsed_id,
            } => {
                assert_eq!(parsed_id, inode_id);
            }
            _ => panic!("Failed to parse tombstone key"),
        }
    }

    #[test]
    fn test_extent_parsing() {
        let codec = KeyCodec::new();
        let inode_id = 7u64;
        let extent_index = 99u64;
        let key = codec.extent_key(inode_id, extent_index);
        assert_eq!(codec.parse_extent_key(key.as_ref()), Some(extent_index));
    }

    #[test]
    fn test_layout_routing() {
        let codec = KeyCodec::new();
        let inode_key = codec.inode_key(0);
        assert!(inode_key.as_ref().starts_with(META_DOMAIN));
        assert_eq!(inode_key.as_ref()[META_DOMAIN.len()], PREFIX_INODE);

        let extent_key = codec.extent_key(0, 0);
        assert!(extent_key.as_ref().starts_with(EXTENT_DOMAIN));
        assert_eq!(extent_key.as_ref()[EXTENT_DOMAIN.len()], PREFIX_EXTENT);

        let tombstone = codec.tombstone_key(0, 0);
        assert!(tombstone.starts_with(META_DOMAIN));

        // No metadata key should be misrouted into the extent domain.
        assert!(!inode_key.as_ref().starts_with(EXTENT_DOMAIN));
        assert!(!tombstone.starts_with(EXTENT_DOMAIN));
    }

    #[test]
    fn segcount_key_roundtrips_and_orders() {
        let codec = KeyCodec::new();
        for (e, c) in [(0u64, 0u64), (5, 0x23b), (7, 255), (u64::MAX, u64::MAX)] {
            let k = codec.segcount_key(e, c);
            assert_eq!(k.len(), META_DOMAIN.len() + 1 + 16);
            assert!(k.starts_with(META_DOMAIN));
            assert_eq!(k[META_DOMAIN.len()], PREFIX_SEGCOUNT);
            assert_eq!(codec.parse_segcount_key(&k), Some((e, c)));
        }
        // Big-endian id => creation-order scan.
        assert!(codec.segcount_key(5, 10).as_ref() < codec.segcount_key(5, 11).as_ref());
        assert!(codec.segcount_key(5, u64::MAX).as_ref() < codec.segcount_key(6, 0).as_ref());
        // The prefix range brackets every segcount key and excludes other kinds.
        let (start, end) = codec.segcount_prefix_range();
        let sc = codec.segcount_key(9, 9);
        assert!(sc.as_ref() >= start.as_ref() && sc.as_ref() < end.as_ref());
        let ino = codec.inode_key(9);
        assert!(!(ino.as_ref() >= start.as_ref() && ino.as_ref() < end.as_ref()));
        assert_eq!(codec.parse_segcount_key(ino.as_ref()), None);
    }

    #[test]
    fn test_value_encoding() {
        let counter = 12345u64;
        let encoded = KeyCodec::encode_counter(counter);
        let decoded = KeyCodec::decode_counter(&encoded).unwrap();
        assert_eq!(decoded, counter);

        let inode_id = 999u64;
        let cookie = 42u64;
        let encoded = KeyCodec::encode_dir_entry(inode_id, cookie);
        let (decoded_id, decoded_cookie) = KeyCodec::decode_dir_entry(&encoded).unwrap();
        assert_eq!(decoded_id, inode_id);
        assert_eq!(decoded_cookie, cookie);

        let size = 1024u64;
        let encoded = KeyCodec::encode_tombstone_size(size);
        let decoded = KeyCodec::decode_tombstone_size(&encoded).unwrap();
        assert_eq!(decoded, size);
    }

    #[test]
    fn test_ha_stamp_encoding() {
        let (epoch, seqno, solo, applied_through) = (7u64, 123456u64, 3u64, 123458u64);
        let stamp = HaStamp::new(
            WriterEpoch::new(epoch).unwrap(),
            ShipSeqno::new(seqno),
            SoloHistory::ever(solo),
            ShipSeqno::new(applied_through),
        )
        .unwrap();
        let encoded = KeyCodec::encode_ha_stamp(&stamp);
        assert_eq!(KeyCodec::decode_ha_stamp(&encoded), Some(stamp));
        // A transitional four-field layout never existed and remains invalid.
        assert_eq!(KeyCodec::decode_ha_stamp(&encoded[..32]), None);
        // Durable legacy layouts migrate conservatively: the shipped seqno is
        // known applied, but missing Solo history disables retry grace.
        let legacy_three_field = HaStamp::new(
            WriterEpoch::new(epoch).unwrap(),
            ShipSeqno::new(seqno),
            SoloHistory::ever(solo),
            ShipSeqno::new(seqno),
        )
        .unwrap();
        assert_eq!(
            KeyCodec::decode_ha_stamp(&encoded[..24]),
            Some(legacy_three_field)
        );
        let legacy_two_field = HaStamp::new(
            WriterEpoch::new(epoch).unwrap(),
            ShipSeqno::new(seqno),
            SoloHistory::ever(0),
            ShipSeqno::new(seqno),
        )
        .unwrap();
        assert_eq!(
            KeyCodec::decode_ha_stamp(&encoded[..16]),
            Some(legacy_two_field)
        );
        let mut legacy_solo_from_birth = Vec::with_capacity(U64_SIZE * 3);
        legacy_solo_from_birth.extend_from_slice(&epoch.to_le_bytes());
        legacy_solo_from_birth.extend_from_slice(&0u64.to_le_bytes());
        legacy_solo_from_birth.extend_from_slice(&2u64.to_le_bytes());
        let legacy_solo_from_birth_stamp = HaStamp::new(
            WriterEpoch::new(epoch).unwrap(),
            None,
            SoloHistory::ever(2),
            None,
        )
        .unwrap();
        assert_eq!(
            KeyCodec::decode_ha_stamp(&legacy_solo_from_birth),
            Some(legacy_solo_from_birth_stamp),
            "a pre-upgrade writer may have committed Solo before its first ship"
        );
        // Wrong length is rejected, not silently misread.
        assert_eq!(KeyCodec::decode_ha_stamp(&encoded[..8]), None);
        assert_eq!(KeyCodec::decode_ha_stamp(&[]), None);

        let mut zero_epoch = encoded.to_vec();
        zero_epoch[..8].copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            KeyCodec::decode_ha_stamp(&zero_epoch),
            None,
            "zero is the absence sentinel, not a valid HA writer epoch"
        );

        let mut regressed = encoded.to_vec();
        regressed[24..32].copy_from_slice(&(seqno - 1).to_le_bytes());
        assert_eq!(
            KeyCodec::decode_ha_stamp(&regressed),
            None,
            "an applied frontier behind an acknowledged ship is invalid"
        );

        let mut invalid_solo_ever = encoded.to_vec();
        invalid_solo_ever[32..].copy_from_slice(&2u64.to_le_bytes());
        assert_eq!(
            KeyCodec::decode_ha_stamp(&invalid_solo_ever),
            None,
            "the durable Solo-history bit accepts only canonical boolean values"
        );

        let never_solo = HaStamp::new(
            WriterEpoch::new(epoch).unwrap(),
            ShipSeqno::new(seqno),
            SoloHistory::never(),
            ShipSeqno::new(seqno),
        )
        .unwrap();
        let inconsistent_solo_history = KeyCodec::encode_ha_stamp(&never_solo);
        let mut inconsistent_solo_history = inconsistent_solo_history.to_vec();
        inconsistent_solo_history[16..24].copy_from_slice(&1u64.to_le_bytes());
        assert_eq!(
            KeyCodec::decode_ha_stamp(&inconsistent_solo_history),
            None,
            "a positive current Solo count cannot claim the term never ran Solo"
        );

        // The HA-stamp key is distinct from the inode counter (both System-prefixed).
        let codec = KeyCodec::new();
        assert_ne!(codec.ha_seqno_key(), codec.system_counter_key());
    }

    #[test]
    fn last_orphan_sweep_key_is_distinct_system_key() {
        let codec = KeyCodec::new();
        let k = codec.last_orphan_sweep_key();
        assert!(k.starts_with(META_DOMAIN));
        assert_eq!(k[codec.kind_offset(KeyPrefix::System)], PREFIX_SYSTEM);
        // Must not alias any other System-subtyped key.
        for other in [
            codec.system_counter_key(),
            codec.ha_seqno_key(),
            codec.lineage_key(),
            codec.taint_key(),
        ] {
            assert_ne!(k, other);
        }
    }

    #[test]
    fn flush_time_keys_order_by_manifest_id_and_roundtrip() {
        let codec = KeyCodec::new();
        let key = codec.flush_time_key(42);
        assert!(key.starts_with(META_DOMAIN));
        assert_eq!(
            key[codec.kind_offset(KeyPrefix::FlushTime)],
            PREFIX_FLUSH_TIME
        );
        assert_eq!(codec.parse_flush_time_key(&key), Some(42));

        // Big-endian manifest id => publication-order scan.
        assert!(codec.flush_time_key(2).as_ref() < codec.flush_time_key(10).as_ref());

        // The prefix brackets every index entry and nothing else.
        let prefix = codec.flush_time_prefix();
        assert!(key.starts_with(&prefix));
        assert!(!codec.fork_registry_key("f1").starts_with(&prefix));
        assert!(!codec.fork_lineage_key().starts_with(&prefix));
        assert!(!codec.inode_key(42).as_ref().starts_with(&prefix));
        assert_eq!(
            codec.parse_flush_time_key(codec.inode_key(42).as_ref()),
            None
        );

        // Value: version byte + (epoch seconds, nanos), roundtripping and
        // rejecting unknown versions and wrong lengths.
        let encoded = KeyCodec::encode_flush_time(1_700_000_000, 42);
        assert_eq!(
            KeyCodec::decode_flush_time(&encoded),
            Some((1_700_000_000, 42))
        );
        let mut future = encoded.to_vec();
        future[0] = FLUSH_TIME_RECORD_VERSION + 1;
        assert_eq!(KeyCodec::decode_flush_time(&future), None);
        assert_eq!(KeyCodec::decode_flush_time(&encoded[..4]), None);
        assert_eq!(KeyCodec::decode_flush_time(&[]), None);
    }

    #[test]
    fn fork_keys_live_in_the_meta_domain() {
        let codec = KeyCodec::new();
        let lineage = codec.fork_lineage_key();
        assert!(lineage.starts_with(META_DOMAIN));
        assert_eq!(lineage[META_DOMAIN.len()], PREFIX_FORK_LINEAGE);

        let registry = codec.fork_registry_key("agent-1");
        assert!(registry.starts_with(META_DOMAIN));
        assert_eq!(registry[META_DOMAIN.len()], PREFIX_FORK_REGISTRY);
        assert!(registry.ends_with(b"agent-1"));

        // The registry prefix brackets every registry entry and nothing else.
        let prefix = codec.fork_registry_prefix();
        assert!(registry.starts_with(&prefix));
        assert!(!codec.inode_key(9).as_ref().starts_with(&prefix));
        assert!(!codec.segcount_key(1, 1).starts_with(&prefix));
        assert!(!lineage.starts_with(&prefix));
    }

    #[test]
    fn test_invalid_key_parsing() {
        let codec = KeyCodec::new();
        assert!(matches!(codec.parse_key(&[]), ParsedKey::Unknown));
        assert!(matches!(codec.parse_key(&[0xFF]), ParsedKey::Unknown));
        assert!(matches!(
            codec.parse_key(&[u8::from(KeyPrefix::Inode)]),
            ParsedKey::Unknown
        ));
        let inode_key = codec.inode_key(1);
        assert!(matches!(
            codec.parse_key(inode_key.as_ref()),
            ParsedKey::Unknown
        ));
    }

    const TEST_BRANCH: BranchId = BranchId(0x01020304);

    fn branch_codec() -> KeyCodec {
        KeyCodec::for_branch(TEST_BRANCH)
    }

    /// Exact expected bytes for `domain || kind || [branch?] || suffix`.
    fn expected_key(domain: &[u8], kind: u8, branch: Option<BranchId>, suffix: &[u8]) -> Vec<u8> {
        let mut v = domain.to_vec();
        v.push(kind);
        if let Some(branch) = branch {
            v.extend_from_slice(&branch.0.to_be_bytes());
        }
        v.extend_from_slice(suffix);
        v
    }

    #[test]
    fn branch_zero_is_byte_identical_to_the_legacy_layout() {
        let codec = KeyCodec::new();
        assert_eq!(codec.branch(), BranchId::ROOT);
        // The root codec and an explicit `for_branch(ROOT)` codec agree, and
        // both emit exactly `domain || kind || suffix` with no branch bytes.
        for c in [KeyCodec::new(), KeyCodec::for_branch(BranchId::ROOT)] {
            assert_eq!(
                c.inode_key(7).as_ref(),
                expected_key(META_DOMAIN, PREFIX_INODE, None, &7u64.to_be_bytes()).as_slice()
            );
            assert_eq!(
                c.extent_key(7, 9).as_ref(),
                expected_key(
                    EXTENT_DOMAIN,
                    PREFIX_EXTENT,
                    None,
                    &[7u64.to_be_bytes(), 9u64.to_be_bytes()].concat(),
                )
                .as_slice()
            );
            assert_eq!(
                c.segcount_key(3, 4).as_ref(),
                expected_key(
                    META_DOMAIN,
                    PREFIX_SEGCOUNT,
                    None,
                    &[3u64.to_be_bytes(), 4u64.to_be_bytes()].concat(),
                )
                .as_slice()
            );
            assert_eq!(c.inode_key_size(), INODE_KEY_SIZE);
            assert_eq!(c.extent_key_size(), EXTENT_KEY_SIZE);
        }
        assert_eq!(codec, KeyCodec::default());
    }

    #[test]
    fn scoped_kinds_embed_the_branch_id_after_the_kind_byte() {
        let codec = branch_codec();
        let b = Some(TEST_BRANCH);

        assert_eq!(
            codec.inode_key(7).as_ref(),
            expected_key(META_DOMAIN, PREFIX_INODE, b, &7u64.to_be_bytes()).as_slice()
        );
        assert_eq!(
            codec.dir_entry_key(7, b"name").as_ref(),
            expected_key(
                META_DOMAIN,
                PREFIX_DIR_ENTRY,
                b,
                &[&7u64.to_be_bytes()[..], b"name"].concat(),
            )
            .as_slice()
        );
        assert_eq!(
            codec.dir_cookie_counter_key(7).as_ref(),
            expected_key(META_DOMAIN, PREFIX_DIR_COOKIE, b, &7u64.to_be_bytes()).as_slice()
        );
        assert_eq!(
            codec.tombstone_key(11, 7).as_ref(),
            expected_key(
                META_DOMAIN,
                PREFIX_TOMBSTONE,
                b,
                &[11u64.to_be_bytes(), 7u64.to_be_bytes()].concat(),
            )
            .as_slice()
        );
        assert_eq!(
            codec.orphan_key(7).as_ref(),
            expected_key(META_DOMAIN, PREFIX_ORPHAN, b, &7u64.to_be_bytes()).as_slice()
        );
        assert_eq!(
            codec.segcount_key(3, 4).as_ref(),
            expected_key(
                META_DOMAIN,
                PREFIX_SEGCOUNT,
                b,
                &[3u64.to_be_bytes(), 4u64.to_be_bytes()].concat(),
            )
            .as_slice()
        );
        assert_eq!(
            codec.extent_key(7, 9).as_ref(),
            expected_key(
                EXTENT_DOMAIN,
                PREFIX_EXTENT,
                b,
                &[7u64.to_be_bytes(), 9u64.to_be_bytes()].concat(),
            )
            .as_slice()
        );

        assert_eq!(codec.inode_key_size(), INODE_KEY_SIZE + BRANCH_ID_SIZE);
        assert_eq!(codec.extent_key_size(), EXTENT_KEY_SIZE + BRANCH_ID_SIZE);
    }

    #[test]
    fn unscoped_kinds_keep_the_global_layout_under_a_branch_codec() {
        let codec = branch_codec();
        let root = KeyCodec::new();

        // Every unscoped key builder must emit byte-identical keys for the
        // root codec and a branch codec.
        assert_eq!(codec.dir_scan_key(7, 42), root.dir_scan_key(7, 42));
        assert_eq!(
            codec.dir_scan_resume_key(7, 42),
            root.dir_scan_resume_key(7, 42)
        );
        assert_eq!(codec.stats_shard_key(3), root.stats_shard_key(3));
        assert_eq!(codec.system_counter_key(), root.system_counter_key());
        assert_eq!(codec.ha_seqno_key(), root.ha_seqno_key());
        assert_eq!(codec.lineage_key(), root.lineage_key());
        assert_eq!(codec.taint_key(), root.taint_key());
        assert_eq!(
            codec.last_orphan_sweep_key(),
            root.last_orphan_sweep_key()
        );
        assert_eq!(codec.fork_lineage_key(), root.fork_lineage_key());
        assert_eq!(codec.fork_registry_key("f"), root.fork_registry_key("f"));
        assert_eq!(codec.flush_time_key(9), root.flush_time_key(9));
        assert_eq!(
            codec.branch_registry_key("b"),
            root.branch_registry_key("b")
        );
        assert_eq!(codec.branch_counter_key(), root.branch_counter_key());

        // None of them grew branch bytes.
        assert_eq!(
            codec.dir_scan_key(7, 42).len(),
            META_DOMAIN.len() + 1 + U64_SIZE * 2
        );
        assert_eq!(
            codec.branch_registry_key("b").as_ref(),
            expected_key(META_DOMAIN, PREFIX_BRANCH, None, b"b").as_slice()
        );
    }

    #[test]
    fn is_scoped_matches_the_documented_split() {
        for kind in [
            KeyPrefix::Inode,
            KeyPrefix::DirEntry,
            KeyPrefix::DirCookie,
            KeyPrefix::Tombstone,
            KeyPrefix::Orphan,
            KeyPrefix::SegCount,
            KeyPrefix::Extent,
        ] {
            assert!(kind.is_scoped(), "{kind:?} must be branch-scoped");
        }
        for kind in [
            KeyPrefix::DirScan,
            KeyPrefix::Stats,
            KeyPrefix::System,
            KeyPrefix::ForkLineage,
            KeyPrefix::ForkRegistry,
            KeyPrefix::FlushTime,
            KeyPrefix::BranchTombstone,
            KeyPrefix::Branch,
        ] {
            assert!(!kind.is_scoped(), "{kind:?} must keep the global layout");
        }
    }

    #[test]
    fn id_offset_shifts_only_for_scoped_kinds_under_a_branch() {
        let root = KeyCodec::new();
        let codec = branch_codec();
        for kind in [
            KeyPrefix::Inode,
            KeyPrefix::DirEntry,
            KeyPrefix::DirCookie,
            KeyPrefix::Tombstone,
            KeyPrefix::Orphan,
            KeyPrefix::SegCount,
            KeyPrefix::Extent,
        ] {
            assert_eq!(
                codec.id_offset(kind),
                root.id_offset(kind) + BRANCH_ID_SIZE,
                "{kind:?}"
            );
            assert_eq!(codec.kind_offset(kind), root.kind_offset(kind), "{kind:?}");
        }
        for kind in [KeyPrefix::DirScan, KeyPrefix::FlushTime, KeyPrefix::Branch] {
            assert_eq!(codec.id_offset(kind), root.id_offset(kind), "{kind:?}");
        }
    }

    #[test]
    fn strip_branch_recovers_the_parent_key() {
        let codec = branch_codec();
        let root = KeyCodec::new();

        for (branched, parent) in [
            (
                Bytes::from(codec.inode_key(7)),
                Bytes::from(root.inode_key(7)),
            ),
            (
                codec.dir_entry_key(7, b"name"),
                root.dir_entry_key(7, b"name"),
            ),
            (codec.segcount_key(3, 4), root.segcount_key(3, 4)),
            (
                Bytes::from(codec.extent_key(7, 9)),
                Bytes::from(root.extent_key(7, 9)),
            ),
            (codec.tombstone_key(1, 7), root.tombstone_key(1, 7)),
            (codec.orphan_key(7), root.orphan_key(7)),
        ] {
            assert_eq!(codec.strip_branch(&branched), Some(parent));
        }

        // Root keys have no branch bytes to strip.
        assert_eq!(root.strip_branch(&root.inode_key(7).as_ref()), None);
        // Unscoped keys are never stripped.
        assert_eq!(codec.strip_branch(&codec.dir_scan_key(7, 42)), None);
        assert_eq!(codec.strip_branch(&codec.branch_registry_key("b")), None);
        // Foreign-branch keys are not this branch's to strip.
        let other = KeyCodec::for_branch(BranchId(0x0A0B0C0D));
        assert_eq!(codec.strip_branch(other.inode_key(7).as_ref()), None);
        // Garbage is rejected, never silently mangled.
        assert_eq!(codec.strip_branch(&[]), None);
        assert_eq!(codec.strip_branch(&[0xFF; 32]), None);
        assert_eq!(codec.strip_branch(&codec.inode_key(7).as_ref()[..8]), None);
    }

    #[test]
    fn branch_tombstone_key_marks_the_shadowed_parent_key() {
        let codec = branch_codec();
        let shadowed = codec.inode_key(7);
        let tombstone = codec
            .branch_tombstone_key(shadowed.as_ref())
            .expect("scoped branch key");
        assert_eq!(
            tombstone.as_ref(),
            expected_key(
                META_DOMAIN,
                PREFIX_BRANCH_TOMBSTONE,
                Some(TEST_BRANCH),
                &[PREFIX_INODE, 0, 0, 0, 0, 0, 0, 0, 7],
            )
            .as_slice()
        );

        // Extent keys tombstone into the meta domain too; the shadowed kind
        // byte (0xFE) preserves which domain the shadowed key lived in.
        let extent_tombstone = codec
            .branch_tombstone_key(codec.extent_key(7, 9).as_ref())
            .expect("scoped extent key");
        assert_eq!(
            extent_tombstone.as_ref(),
            expected_key(
                META_DOMAIN,
                PREFIX_BRANCH_TOMBSTONE,
                Some(TEST_BRANCH),
                &[
                    &[PREFIX_EXTENT][..],
                    &7u64.to_be_bytes(),
                    &9u64.to_be_bytes()
                ]
                .concat(),
            )
            .as_slice()
        );

        // The value is a bare version byte; presence is the signal.
        assert_eq!(
            KeyCodec::branch_tombstone_value().as_ref(),
            &[BRANCH_TOMBSTONE_RECORD_VERSION]
        );

        // Unscoped, root, and foreign-branch keys cannot be tombstoned here.
        assert_eq!(codec.branch_tombstone_key(&codec.dir_scan_key(7, 42)), None);
        let root = KeyCodec::new();
        assert_eq!(root.branch_tombstone_key(root.inode_key(7).as_ref()), None);
        let other = KeyCodec::for_branch(BranchId(9));
        assert_eq!(codec.branch_tombstone_key(other.inode_key(7).as_ref()), None);
    }

    #[test]
    fn parse_any_segcount_key_reads_every_scope() {
        let root = KeyCodec::new();
        let branch = KeyCodec::for_branch(BranchId(7));

        // Root-layout and branch-layout rows both decode; the branch id is
        // skipped, not validated (a census cares about (epoch, counter)).
        assert_eq!(
            KeyCodec::parse_any_segcount_key(&root.segcount_key(3, 4)),
            Some((3, 4))
        );
        assert_eq!(
            KeyCodec::parse_any_segcount_key(&branch.segcount_key(3, 4)),
            Some((3, 4))
        );

        // Other kinds and malformed rows are rejected.
        assert_eq!(KeyCodec::parse_any_segcount_key(root.inode_key(3).as_ref()), None);
        assert_eq!(KeyCodec::parse_any_segcount_key(b"meta\x09\x01"), None);
        assert_eq!(
            KeyCodec::parse_any_segcount_key(&branch.segcount_key(3, 4)[..8]),
            None
        );
    }

    #[test]
    fn branch_tombstone_range_brackets_exactly_one_branch() {
        let codec = KeyCodec::for_branch(BranchId(3));
        let sibling = KeyCodec::for_branch(BranchId(4));

        // The root codec has no tombstone range (the whole-kind range would
        // cover other branches' rows).
        assert_eq!(KeyCodec::new().branch_tombstone_range(), None);

        let (start, end) = codec.branch_tombstone_range().unwrap();
        let mine = codec
            .branch_tombstone_key(codec.inode_key(1).as_ref())
            .unwrap();
        assert!(mine.as_ref() >= start.as_ref() && mine.as_ref() < end.as_ref());
        let foreign = sibling
            .branch_tombstone_key(sibling.inode_key(1).as_ref())
            .unwrap();
        assert!(!(foreign.as_ref() >= start.as_ref() && foreign.as_ref() < end.as_ref()));

        // u32::MAX has no successor; the range falls back to the next kind.
        let max = KeyCodec::for_branch(BranchId(u32::MAX));
        let (start, end) = max.branch_tombstone_range().unwrap();
        let mine = max
            .branch_tombstone_key(max.inode_key(1).as_ref())
            .unwrap();
        assert!(mine.as_ref() >= start.as_ref() && mine.as_ref() < end.as_ref());
    }

    #[test]
    fn branch_prefix_range_brackets_exactly_one_branch() {
        let codec = KeyCodec::for_branch(BranchId(1));
        let root = KeyCodec::new();
        let sibling = KeyCodec::for_branch(BranchId(2));

        let (start, end) = codec.prefix_range(KeyPrefix::SegCount);
        let mine = codec.segcount_key(9, 9);
        assert!(mine.as_ref() >= start.as_ref() && mine.as_ref() < end.as_ref());
        // Neither the parent's nor a sibling's segcount keys fall in range.
        for foreign in [root.segcount_key(9, 9), sibling.segcount_key(9, 9)] {
            assert!(
                !(foreign.as_ref() >= start.as_ref() && foreign.as_ref() < end.as_ref()),
                "foreign-branch key leaked into the branch prefix range"
            );
        }

        // The root codec's range covers the whole kind, branches included.
        let (root_start, root_end) = root.prefix_range(KeyPrefix::SegCount);
        for key in [root.segcount_key(1, 1), mine.clone()] {
            assert!(key.as_ref() >= root_start.as_ref() && key.as_ref() < root_end.as_ref());
        }

        // Same bracketing in the extent domain.
        let (start, end) = codec.prefix_range(KeyPrefix::Extent);
        assert!(codec.extent_key(1, 0).as_ref() >= start.as_ref()
            && codec.extent_key(1, 0).as_ref() < end.as_ref());
        assert!(!(root.extent_key(1, 0).as_ref() >= start.as_ref()
            && root.extent_key(1, 0).as_ref() < end.as_ref()));
        assert!(!(sibling.extent_key(1, 0).as_ref() >= start.as_ref()
            && sibling.extent_key(1, 0).as_ref() < end.as_ref()));
    }

    #[test]
    fn scoped_branched_keys_sort_by_branch_then_suffix() {
        let b1 = KeyCodec::for_branch(BranchId(1));
        let b2 = KeyCodec::for_branch(BranchId(2));
        // Within one branch, suffix order is preserved.
        assert!(b1.inode_key(7).as_ref() < b1.inode_key(8).as_ref());
        assert!(b1.segcount_key(5, 10).as_ref() < b1.segcount_key(5, 11).as_ref());
        // Branches never interleave: all of branch 1 sorts before branch 2.
        assert!(b1.inode_key(u64::MAX).as_ref() < b2.inode_key(0).as_ref());
        assert!(b1.segcount_key(u64::MAX, u64::MAX).as_ref() < b2.segcount_key(0, 0).as_ref());
        // The parent (no branch bytes) sorts before any branch's slice: its
        // suffix's first byte compares against the branch id's high byte.
        let root = KeyCodec::new();
        assert!(root.inode_key(0).as_ref() < b1.inode_key(0).as_ref());
    }

    #[test]
    fn branch_codecs_parse_their_own_scoped_keys() {
        let codec = branch_codec();
        let root = KeyCodec::new();

        // Segcount roundtrips through the branch codec; the root codec
        // rejects the widened key (wrong length) and vice versa.
        let key = codec.segcount_key(3, 4);
        assert_eq!(codec.parse_segcount_key(&key), Some((3, 4)));
        assert_eq!(root.parse_segcount_key(&key), None);
        assert_eq!(codec.parse_segcount_key(&root.segcount_key(3, 4)), None);

        let extent = codec.extent_key(7, 9);
        assert_eq!(codec.parse_extent_key(extent.as_ref()), Some(9));
        assert_eq!(codec.parse_extent_key_full(extent.as_ref()), Some((7, 9)));
        assert_eq!(root.parse_extent_key(extent.as_ref()), None);

        // parse_key is branch-aware through id_offset as well.
        match codec.parse_key(&codec.tombstone_key(11, 7)) {
            ParsedKey::Tombstone { inode_id } => assert_eq!(inode_id, 7),
            other => panic!("expected Tombstone, got {other:?}"),
        }
        match codec.parse_key(&codec.orphan_key(7)) {
            ParsedKey::Orphan { inode_id } => assert_eq!(inode_id, 7),
            other => panic!("expected Orphan, got {other:?}"),
        }
    }

    #[test]
    fn branch_registry_keys_are_global_and_well_bracketed() {
        let codec = KeyCodec::new();
        let entry = codec.branch_registry_key("agent-1");
        assert!(entry.starts_with(META_DOMAIN));
        assert_eq!(entry[META_DOMAIN.len()], PREFIX_BRANCH);
        assert!(entry.ends_with(b"agent-1"));

        let prefix = codec.branch_registry_prefix();
        assert!(entry.starts_with(&prefix));
        assert!(!codec.fork_registry_key("agent-1").starts_with(&prefix));
        assert!(!codec.inode_key(9).as_ref().starts_with(&prefix));
        assert!(!codec.branch_counter_key().starts_with(&prefix));

        // The id counter is a System-subtyped key distinct from its siblings.
        let counter = codec.branch_counter_key();
        assert_eq!(counter[codec.kind_offset(KeyPrefix::System)], PREFIX_SYSTEM);
        for other in [
            codec.system_counter_key(),
            codec.ha_seqno_key(),
            codec.lineage_key(),
            codec.taint_key(),
            codec.last_orphan_sweep_key(),
        ] {
            assert_ne!(counter, other);
        }
    }

    #[test]
    fn branched_fixed_keys_compare_and_hash_by_key_bytes() {
        let codec = branch_codec();
        let root = KeyCodec::new();
        // Same logical key, different layouts: unequal, ordered by raw bytes.
        assert_ne!(codec.inode_key(7), root.inode_key(7));
        assert_eq!(codec.inode_key(7), codec.inode_key(7));
        assert!(root.inode_key(7) < codec.inode_key(7));
        assert!(codec.extent_key(7, 1) < codec.extent_key(7, 2));
        let mut set = std::collections::HashSet::new();
        set.insert(codec.inode_key(7));
        assert!(set.contains(&codec.inode_key(7)));
        assert!(!set.contains(&root.inode_key(7)));
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        // Big-endian id encoding is the reason lexicographic key order matches
        // numeric order; every range scan (extent reads, dir listing, reclamation) leans on
        // it. The prose comments assert this layout; nothing tested it until now.
        #[test]
        fn extent_key_roundtrips_and_orders(
            a in (any::<u64>(), any::<u64>()),
            b in (any::<u64>(), any::<u64>()),
        ) {
            let codec = KeyCodec::new();
            let ka = codec.extent_key(a.0, a.1);
            let kb = codec.extent_key(b.0, b.1);
            prop_assert_eq!(codec.parse_extent_key(ka.as_ref()), Some(a.1));
            prop_assert_eq!(codec.parse_extent_key(kb.as_ref()), Some(b.1));
            prop_assert_eq!(ka.as_ref().cmp(kb.as_ref()), a.cmp(&b));
        }

        #[test]
        fn tombstone_key_roundtrips_and_orders(
            a in (any::<u64>(), any::<u64>()),
            b in (any::<u64>(), any::<u64>()),
        ) {
            let codec = KeyCodec::new();
            let ka = codec.tombstone_key(a.0, a.1);
            let kb = codec.tombstone_key(b.0, b.1);
            match codec.parse_key(&ka) {
                ParsedKey::Tombstone { inode_id } => prop_assert_eq!(inode_id, a.1),
                other => prop_assert!(false, "expected Tombstone, got {:?}", other),
            }
            // Ordered by (timestamp, inode_id): cleanup scans tombstones in time order.
            prop_assert_eq!(ka.as_ref().cmp(kb.as_ref()), a.cmp(&b));
        }

        #[test]
        fn dir_scan_key_roundtrips_and_orders(
            a in (any::<u64>(), any::<u64>()),
            b in (any::<u64>(), any::<u64>()),
        ) {
            let codec = KeyCodec::new();
            let ka = codec.dir_scan_key(a.0, a.1);
            let kb = codec.dir_scan_key(b.0, b.1);
            match codec.parse_key(&ka) {
                ParsedKey::DirScan { cookie } => prop_assert_eq!(cookie, a.1),
                other => prop_assert!(false, "expected DirScan, got {:?}", other),
            }
            prop_assert_eq!(ka.as_ref().cmp(kb.as_ref()), a.cmp(&b));
        }

        // A resume key must land exactly on the next cookie, so a paged scan
        // continues strictly after `cookie` with no skipped or repeated entry.
        #[test]
        fn dir_scan_resume_is_next_cookie(dir in any::<u64>(), cookie in 0u64..u64::MAX) {
            let codec = KeyCodec::new();
            let resume = codec.dir_scan_resume_key(dir, cookie);
            prop_assert_eq!(&resume, &codec.dir_scan_key(dir, cookie + 1));
            prop_assert!(resume.as_ref() > codec.dir_scan_key(dir, cookie).as_ref());
        }

        #[test]
        fn orphan_key_roundtrips(ino in any::<u64>()) {
            let codec = KeyCodec::new();
            match codec.parse_key(&codec.orphan_key(ino)) {
                ParsedKey::Orphan { inode_id } => prop_assert_eq!(inode_id, ino),
                other => prop_assert!(false, "expected Orphan, got {:?}", other),
            }
        }

        // parse_extent_key must reject every non-extent key, including any that
        // happens to match the extent key's byte length (only the domain/kind
        // bytes distinguish them).
        #[test]
        fn non_extent_keys_never_parse_as_extent(
            ino in any::<u64>(),
            x in any::<u64>(),
            name in prop::collection::vec(any::<u8>(), 0..40),
        ) {
            let codec = KeyCodec::new();
            prop_assert_eq!(codec.parse_extent_key(codec.inode_key(ino).as_ref()), None);
            prop_assert_eq!(codec.parse_extent_key(&codec.tombstone_key(x, ino)), None);
            prop_assert_eq!(codec.parse_extent_key(&codec.orphan_key(ino)), None);
            prop_assert_eq!(codec.parse_extent_key(&codec.dir_scan_key(ino, x)), None);
            prop_assert_eq!(codec.parse_extent_key(&codec.dir_entry_key(ino, &name)), None);
        }

        #[test]
        fn value_codecs_roundtrip(x in 1u64..=u64::MAX, y in any::<u64>()) {
            prop_assert_eq!(KeyCodec::decode_u64(&KeyCodec::encode_u64(x)), Some(x));
            prop_assert_eq!(KeyCodec::decode_counter(&KeyCodec::encode_counter(x)).unwrap(), x);
            prop_assert_eq!(
                KeyCodec::decode_tombstone_size(&KeyCodec::encode_tombstone_size(x)).unwrap(),
                x
            );
            let last_shipped = ShipSeqno::new(y);
            let stamp = HaStamp::new(
                WriterEpoch::new(x).unwrap(),
                last_shipped,
                SoloHistory::ever(x),
                last_shipped,
            ).unwrap();
            prop_assert_eq!(
                KeyCodec::decode_ha_stamp(&KeyCodec::encode_ha_stamp(&stamp)),
                Some(stamp)
            );
            prop_assert_eq!(
                KeyCodec::decode_dir_entry(&KeyCodec::encode_dir_entry(x, y)).unwrap(),
                (x, y)
            );
        }

        // The half-open prefix range must contain every extent key and no metadata
        // key, so an extent-domain scan can never read or remove metadata.
        #[test]
        fn prefix_range_isolates_extents(ino in any::<u64>(), idx in any::<u64>(), x in any::<u64>()) {
            let codec = KeyCodec::new();
            let (start, end) = codec.prefix_range(KeyPrefix::Extent);
            let extent = codec.extent_key(ino, idx);
            prop_assert!(extent.as_ref() >= start.as_ref() && extent.as_ref() < end.as_ref());

            let meta = codec.tombstone_key(x, ino);
            prop_assert!(
                !(meta.as_ref() >= start.as_ref() && meta.as_ref() < end.as_ref()),
                "a metadata key leaked into the extent prefix range"
            );
        }
    }

    #[test]
    fn scoped_suffix_strips_domain_kind_and_branch() {
        let root = KeyCodec::new();
        let branch = KeyCodec::for_branch(BranchId(7));

        // Root codec: suffix follows `domain || kind`.
        let root_entry = root.dir_entry_key(3, b"name");
        let mut expected = 3u64.to_be_bytes().to_vec();
        expected.extend_from_slice(b"name");
        assert_eq!(root.scoped_suffix(&root_entry).unwrap(), expected.as_slice());

        // Branch codec: suffix follows `domain || kind || branch`, and equals
        // the parent's suffix for the same logical entry.
        let branch_entry = branch.dir_entry_key(3, b"name");
        assert_eq!(
            branch.scoped_suffix(&branch_entry).unwrap(),
            root.scoped_suffix(&root_entry).unwrap()
        );

        // Wrong branch, unscoped kinds, and junk are rejected.
        assert!(
            KeyCodec::for_branch(BranchId(8))
                .scoped_suffix(&branch_entry)
                .is_none()
        );
        assert!(root.scoped_suffix(&root.stats_shard_key(0)).is_none());
        assert!(root.scoped_suffix(b"nonsense").is_none());
        // The root codec does not read a branch key as a longer root key.
        assert!(root.scoped_suffix(&branch_entry).is_some_and(|s| {
            // Root layout reads the branch bytes as suffix payload; that is
            // fine (the codecs serve different scopes) but must never equal
            // the parent suffix.
            s != root.scoped_suffix(&root_entry).unwrap()
        }));
    }

    #[test]
    fn adopt_parent_key_inverts_strip_branch() {
        let branch = KeyCodec::for_branch(BranchId(9));
        let root = KeyCodec::new();

        for (parent, branch_key) in [
            (
                root.dir_entry_key(1, b"file"),
                branch.dir_entry_key(1, b"file"),
            ),
            (
                Bytes::from(root.inode_key(42)),
                Bytes::from(branch.inode_key(42)),
            ),
            (
                Bytes::from(root.extent_key(5, 6)),
                Bytes::from(branch.extent_key(5, 6)),
            ),
        ] {
            assert_eq!(branch.strip_branch(&branch_key).unwrap(), parent);
            assert_eq!(branch.adopt_parent_key(&parent).unwrap(), branch_key);
        }

        // Root codec and unscoped keys have nothing to adopt.
        assert!(root.adopt_parent_key(&root.dir_entry_key(1, b"x")).is_none());
        assert!(
            branch
                .adopt_parent_key(&root.stats_shard_key(0))
                .is_none()
        );
    }

    #[test]
    fn branch_tombstone_prefix_covers_exactly_one_kinds_shadows() {
        let codec = KeyCodec::for_branch(BranchId(3));
        let dir3_suffix = {
            let mut s = 3u64.to_be_bytes().to_vec();
            s.extend_from_slice(b"old-name");
            s
        };
        let shadowed = codec.dir_entry_key(3, b"old-name");
        let tombstone = codec.branch_tombstone_key(&shadowed).unwrap();

        // The row parses back to its shadowed (kind, suffix).
        assert_eq!(
            codec.parse_branch_tombstone_key(&tombstone).unwrap(),
            (KeyPrefix::DirEntry, dir3_suffix.as_slice())
        );

        // The (kind, suffix-prefix) range for a directory covers the row;
        // sibling kinds and other branches' rows fall outside.
        let prefix = codec.branch_tombstone_prefix(KeyPrefix::DirEntry, &3u64.to_be_bytes());
        assert!(tombstone.starts_with(&prefix));
        let inode_prefix = codec.branch_tombstone_prefix(KeyPrefix::Inode, b"");
        assert!(!tombstone.starts_with(&inode_prefix));
        assert!(
            KeyCodec::for_branch(BranchId(4))
                .parse_branch_tombstone_key(&tombstone)
                .is_none()
        );
        assert!(
            KeyCodec::new()
                .parse_branch_tombstone_key(&tombstone)
                .is_none()
        );
    }

    #[test]
    fn branch_tombstone_bound_re_roots_parent_bounds() {
        let codec = KeyCodec::for_branch(BranchId(2));
        let root = KeyCodec::new();

        // A within-kind bound: kind byte and suffix are preserved verbatim.
        let bound = root.dir_entry_key(8, b"n");
        let tomb_bound = codec.branch_tombstone_bound(&bound).unwrap();
        assert_eq!(
            codec.parse_branch_tombstone_key(&tomb_bound).unwrap(),
            (KeyPrefix::DirEntry, root.scoped_suffix(&bound).unwrap())
        );

        // A kind-end bound (`domain || kind + 1`) maps to the end of that
        // kind's tombstone space even though 0x03 (DIR_SCAN) is unscoped.
        let kind_end = Bytes::from_static(b"meta\x03");
        let tomb_end = codec.branch_tombstone_bound(&kind_end).unwrap();
        let entry_tombstone = codec
            .branch_tombstone_key(&codec.dir_entry_key(1, b"a"))
            .unwrap();
        assert!(entry_tombstone.as_ref() < tomb_end.as_ref());
        let scan_tombstone_prefix = codec.branch_tombstone_prefix(KeyPrefix::DirScan, b"");
        assert!(scan_tombstone_prefix.as_ref() >= tomb_end.as_ref());

        // Extent-domain bounds re-root into the meta-domain tombstone space.
        let extent_bound: Bytes = root.extent_key(1, 0).into();
        assert!(
            codec
                .branch_tombstone_bound(&extent_bound)
                .unwrap()
                .starts_with(META_DOMAIN)
        );

        // Root codec and out-of-domain bounds are rejected.
        assert!(root.branch_tombstone_bound(&bound).is_none());
        assert!(codec.branch_tombstone_bound(b"other").is_none());
    }

    #[test]
    fn dir_entry_prefix_orders_entries_by_name() {
        let codec = KeyCodec::for_branch(BranchId(5));
        let prefix = codec.dir_entry_prefix(4);
        assert!(codec.dir_entry_key(4, b"a").starts_with(&prefix));
        assert!(codec.dir_entry_key(4, b"z").starts_with(&prefix));
        assert!(!codec.dir_entry_key(5, b"a").starts_with(&prefix));
        // Byte order of names is the scan order within the prefix.
        assert!(codec.dir_entry_key(4, b"a") < codec.dir_entry_key(4, b"b"));
        // Branch isolation: another branch's entries are not covered.
        assert!(
            !KeyCodec::for_branch(BranchId(6))
                .dir_entry_key(4, b"a")
                .starts_with(&prefix)
        );
        // Root codec's prefix is the historical layout (no branch bytes).
        let root_prefix = KeyCodec::new().dir_entry_prefix(4);
        assert!(KeyCodec::new().dir_entry_key(4, b"a").starts_with(&root_prefix));
        assert!(!codec.dir_entry_key(4, b"a").starts_with(&root_prefix));
    }
}

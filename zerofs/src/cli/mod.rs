use crate::config::Settings;
use crate::rpc::client::RpcClient;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

pub mod branch;
pub mod checkpoint;
pub mod debug;
pub mod fatrace;
pub mod flush;
pub mod fork;
mod init;
pub mod monitor;
pub mod otrace;
pub mod password;
pub mod server;

#[derive(Parser)]
#[command(name = "zerofs")]
#[command(author, version, about = "The Filesystem That Makes S3 your Primary Storage", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate a default configuration file
    Init {
        /// Output path for the config file, or "-" to write to stdout
        #[arg(default_value = "zerofs.toml")]
        path: PathBuf,
    },
    /// Run the filesystem server
    Run {
        #[arg(short, long)]
        config: PathBuf,
        /// Open the filesystem in read-only mode
        #[arg(long, conflicts_with = "checkpoint")]
        read_only: bool,
        /// Open from a specific checkpoint by name (read-only mode)
        #[arg(long, conflicts_with = "read_only")]
        checkpoint: Option<String>,
        /// Serve this basin branch of the volume instead of the root. The
        /// branch must already exist (create it with `zerofs branch create`
        /// against the running parent server). Not supported together with
        /// [replication].
        #[arg(long)]
        branch: Option<String>,
    },
    /// Change the encryption password
    ///
    /// Reads new password from stdin. Examples:
    ///
    /// echo "newpassword" | zerofs change-password -c config.toml
    ///
    /// zerofs change-password -c config.toml < password.txt
    ChangePassword {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Debug commands for inspecting the database
    Debug {
        #[command(subcommand)]
        subcommand: DebugCommands,
    },
    /// Checkpoint management commands
    Checkpoint {
        #[command(subcommand)]
        subcommand: CheckpointCommands,
    },
    /// Fork management commands: writable clones of this volume
    Fork {
        #[command(subcommand)]
        subcommand: ForkCommands,
    },
    /// Basin branch commands: O(1) branch namespaces inside this volume
    Branch {
        #[command(subcommand)]
        subcommand: BranchCommands,
    },
    /// Trace file system operations in real-time
    Fatrace {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Trace object store requests in real-time
    Otrace {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Flush pending writes to storage
    Flush {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Monitor filesystem activity in real-time
    Monitor {
        #[arg(short, long)]
        config: PathBuf,
        /// Stats refresh interval in milliseconds
        #[arg(long, default_value = "250")]
        interval: u32,
    },
    /// Mount a ZeroFS 9P export as a local filesystem (FUSE client)
    ///
    /// Connects to a running ZeroFS 9P server and exposes it at a local mount
    /// point. The server may be local or remote. Examples:
    ///
    /// zerofs mount 127.0.0.1:5564 /mnt/zerofs
    ///
    /// zerofs mount unix:/tmp/zerofs.9p.sock /mnt/zerofs
    #[cfg(target_os = "linux")]
    Mount {
        /// 9P server address: host[:port], tcp://host:port, or unix:/path/to.sock
        target: String,
        /// Local directory to mount at
        mountpoint: PathBuf,
        /// Mount read-only
        #[arg(long)]
        read_only: bool,
        /// Who may access the mount: `owner` (only the mounting user), `root`
        /// (owner + root), or `all` (any user). `root`/`all` need
        /// `user_allow_other` in /etc/fuse.conf unless mounting as root.
        #[arg(long, value_enum, default_value_t = crate::mount::MountAccess::Owner)]
        access: crate::mount::MountAccess,
        /// Maximum 9P message size in bytes
        #[arg(long, default_value_t = 10 * 1024 * 1024)]
        msize: u32,
        /// Use a writeback page cache: writes are buffered and flushed
        /// asynchronously (higher throughput, looser cross-client coherence).
        /// Pass `--writeback false` to write through synchronously instead.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        writeback: bool,
        /// Allow consistency-relaxing client/kernel caches for speed: the
        /// open+read prefetch fold, cached symlink targets, a 1s attribute cache,
        /// and page-cached reads. Pass `--relaxed-consistency false` for strict
        /// consistency, where every read and lookup hits the server (direct I/O,
        /// no attribute cache) and writes are synchronous (implies write-through).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        relaxed_consistency: bool,
        /// Root the mount at this server-side directory (a path from the
        /// filesystem root, e.g. /volumes/pvc-1) instead of the whole
        /// filesystem. The directory must already exist.
        #[arg(long)]
        aname: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum DebugCommands {
    /// List all keys in the database
    ListKeys {
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum CheckpointCommands {
    /// Create a new checkpoint
    Create {
        #[arg(short, long)]
        config: PathBuf,
        /// Name for the checkpoint (must be unique)
        name: String,
    },
    /// List all checkpoints
    List {
        #[arg(short, long)]
        config: PathBuf,
    },
    /// Delete a checkpoint by name
    Delete {
        #[arg(short, long)]
        config: PathBuf,
        /// Checkpoint name to delete
        name: String,
    },
    /// Get checkpoint information
    Info {
        #[arg(short, long)]
        config: PathBuf,
        /// Checkpoint name to query
        name: String,
    },
}

#[derive(Subcommand)]
pub enum BranchCommands {
    /// Create a basin branch of this volume (O(1); no data is copied)
    Create {
        #[arg(short, long)]
        config: PathBuf,
        /// Name for the branch (must be unique among this volume's branches)
        name: String,
    },
    /// Delete a branch, its registry entry, and all of its data
    Delete {
        #[arg(short, long)]
        config: PathBuf,
        /// Branch name to delete
        name: String,
    },
    /// List this volume's basin branches
    List {
        #[arg(short, long)]
        config: PathBuf,
    },
}

#[derive(Subcommand)]
pub enum ForkCommands {
    /// Create a writable fork of this volume
    Create {
        #[arg(short, long)]
        config: PathBuf,
        /// Name for the fork (must be unique among this volume's forks)
        name: String,
        /// Named checkpoint to fork from; defaults to the current durable state
        #[arg(long)]
        from_checkpoint: Option<String>,
        /// Point-in-time fork: fork the volume as of the last manifest
        /// flushed before this RFC 3339 timestamp
        #[arg(long)]
        at: Option<String>,
        /// Seal+flush barrier at the branch point and full materialization
        /// before create returns (~1s). Default is lazy creation (~50ms):
        /// the fork materializes at its first open, and the branch point can
        /// lag HEAD by up to the flush interval
        #[arg(long)]
        barrier: bool,
    },
    /// Delete a fork, releasing its pin on the parent's reclamation
    Delete {
        #[arg(short, long)]
        config: PathBuf,
        /// Fork name to delete
        name: String,
    },
    /// List this volume's forks
    List {
        #[arg(short, long)]
        config: PathBuf,
    },
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

pub async fn connect_rpc_client(config_path: &Path) -> Result<RpcClient> {
    let (settings, _) = Settings::from_file(config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    let rpc_config = settings
        .servers
        .rpc
        .as_ref()
        .context("RPC server not configured in config file")?;

    RpcClient::connect_from_config(rpc_config)
        .await
        .context("Failed to connect to RPC server. Is the server running?")
}

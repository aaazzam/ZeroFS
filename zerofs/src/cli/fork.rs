use super::connect_rpc_client;
use anyhow::Result;
use comfy_table::{Table, presets::UTF8_FULL};
use std::path::Path;

pub async fn create_fork(
    config_path: &Path,
    name: &str,
    from_checkpoint: Option<String>,
    at: Option<String>,
) -> Result<()> {
    let at = at
        .map(|t| {
            chrono::DateTime::parse_from_rfc3339(&t)
                .map(|t| t.to_utc())
                .map_err(|e| anyhow::anyhow!("invalid --at timestamp '{t}': {e}"))
        })
        .transpose()?;
    let client = connect_rpc_client(config_path).await?;
    let fork = client.create_fork(name, from_checkpoint, at).await?;

    println!("✓ Fork created successfully!");
    println!("  Name: {}", fork.name);
    println!("  Fork db path: {}", fork.db_path);
    println!("  Parent db path: {}", fork.parent_db_path);
    println!("  Base epoch: {}", fork.base_epoch);
    println!();
    println!("Serve the fork with a config whose [storage] url points at the fork db path, e.g.:");
    println!("  url = \"<backend>://<bucket>/{}/\"", fork.db_path);

    Ok(())
}

pub async fn list_forks(config_path: &Path) -> Result<()> {
    let client = connect_rpc_client(config_path).await?;
    let forks = client.list_forks().await?;

    if forks.is_empty() {
        println!("No forks found.");
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Name", "Fork db path", "Parent db path", "Base epoch"]);

    for fork in forks {
        table.add_row(vec![
            fork.name,
            fork.db_path,
            fork.parent_db_path,
            fork.base_epoch.to_string(),
        ]);
    }

    println!("{table}");
    Ok(())
}

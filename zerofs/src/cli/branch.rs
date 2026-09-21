use super::connect_rpc_client;
use anyhow::Result;
use comfy_table::{Table, presets::UTF8_FULL};
use std::path::Path;

pub async fn create_branch(config_path: &Path, name: &str) -> Result<()> {
    let client = connect_rpc_client(config_path).await?;
    let branch = client.create_branch(name).await?;

    println!("✓ Branch created successfully!");
    println!("  Name: {}", branch.name);
    println!("  Id: {}", branch.id);
    println!();
    println!("Serve the branch with the same config as this volume plus --branch, e.g.:");
    println!("  zerofs run -c <config> --branch {}", branch.name);

    Ok(())
}

pub async fn list_branches(config_path: &Path) -> Result<()> {
    let client = connect_rpc_client(config_path).await?;
    let branches = client.list_branches().await?;

    if branches.is_empty() {
        println!("No branches found.");
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Name", "Id", "Created at"]);

    for branch in branches {
        let created_at = branch
            .created_at
            .and_then(|t| chrono::DateTime::from_timestamp(t.seconds, t.nanos as u32))
            .map(|t| t.to_rfc3339())
            .unwrap_or_default();
        table.add_row(vec![branch.name, branch.id.to_string(), created_at]);
    }

    println!("{table}");
    Ok(())
}

pub async fn delete_branch(config_path: &Path, name: &str) -> Result<()> {
    let client = connect_rpc_client(config_path).await?;
    client.delete_branch(name).await?;

    println!("✓ Branch '{}' deleted successfully!", name);
    Ok(())
}

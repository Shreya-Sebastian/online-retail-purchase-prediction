output "workspace_url" {
  description = "Open this to reach the Databricks workspace"
  value       = "https://${azurerm_databricks_workspace.dbw.workspace_url}"
}

output "access_connector_id" {
  description = "Paste into Catalog > External data > Credentials (Azure managed identity)"
  value       = azurerm_databricks_access_connector.uc.id
}

output "raw_container_url" {
  description = "URL for the Unity Catalog external location"
  value       = "abfss://${azurerm_storage_container.raw.name}@${azurerm_storage_account.lake.name}.dfs.core.windows.net/"
}

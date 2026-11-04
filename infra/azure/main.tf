# Azure resources for running this project on Azure Databricks.
#
# Creates: a resource group, a Data Lake Storage Gen2 account with a private "raw" container,
# an Azure Databricks workspace (premium, needed for Unity Catalog), and an access connector
# (a managed identity) that Unity Catalog uses to reach the storage account without keys.
#
#   az login
#   terraform init
#   terraform plan  -var="subscription_id=..." -var="storage_account_name=..."
#   terraform apply -var="subscription_id=..." -var="storage_account_name=..."
#   terraform destroy ...   # removes everything again
#
# Everything lives in one resource group, so deleting that group in the portal also cleans up.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.20"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_group
  location = var.location
}

# Data Lake Storage Gen2 = a storage account with the hierarchical namespace switched on
resource "azurerm_storage_account" "lake" {
  name                            = var.storage_account_name
  resource_group_name             = azurerm_resource_group.rg.name
  location                        = azurerm_resource_group.rg.location
  account_kind                    = "StorageV2"
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  is_hns_enabled                  = true
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
}

resource "azurerm_storage_container" "raw" {
  name                  = "raw"
  storage_account_id    = azurerm_storage_account.lake.id
  container_access_type = "private"
}

resource "azurerm_databricks_workspace" "dbw" {
  name                        = var.workspace_name
  resource_group_name         = azurerm_resource_group.rg.name
  location                    = azurerm_resource_group.rg.location
  sku                         = "premium"
  managed_resource_group_name = "${var.resource_group}-managed"
}

# The identity Unity Catalog uses for storage access (registered later as a storage credential)
resource "azurerm_databricks_access_connector" "uc" {
  name                = "${var.workspace_name}-connector"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  identity {
    type = "SystemAssigned"
  }
}

# Least privilege: the connector may read and write blobs in this one storage account
resource "azurerm_role_assignment" "connector_blob" {
  scope                = azurerm_storage_account.lake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_databricks_access_connector.uc.identity[0].principal_id
}

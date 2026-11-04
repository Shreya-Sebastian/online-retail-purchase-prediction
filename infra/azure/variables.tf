variable "subscription_id" {
  description = "Your Azure subscription id (Portal > Subscriptions)"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "westeurope"
}

variable "resource_group" {
  description = "Resource group that holds everything"
  type        = string
  default     = "rg-custprop"
}

variable "storage_account_name" {
  description = "Globally unique, 3 to 24 lowercase letters and digits"
  type        = string
}

variable "workspace_name" {
  description = "Azure Databricks workspace name"
  type        = string
  default     = "dbw-custprop"
}

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Virtual Network
resource "azurerm_virtual_network" "vnet" {
  name                = "goldensignals-vnet"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  address_space       = ["10.0.0.0/16"]
}

# AKS Subnet
resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Application Gateway Subnet
resource "azurerm_subnet" "appgw" {
  name                 = "appgw-subnet"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.2.0/24"]
}

resource "azurerm_resource_group" "rg" {
  name     = "goldensignals-rg"
  location = "eastus"
  tags = {
    Environment = "Development"
    Application = "GoldenSignalsAI"
  }
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "goldensignals-aks"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "goldensignals"
  kubernetes_version  = "1.27.3"

  default_node_pool {
    name                = "default"
    node_count          = 2
    vm_size            = "Standard_D2s_v3"
    enable_auto_scaling = false
    os_disk_size_gb    = 50
    vnet_subnet_id     = azurerm_subnet.aks.id
    
    enable_host_encryption = false
    
    enable_node_public_ip = false
    
    tags = {
      Environment = "Development"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin     = "azure"
    load_balancer_sku  = "standard"
    service_cidr       = "10.1.0.0/16"
    dns_service_ip     = "10.1.0.10"
    docker_bridge_cidr = "172.17.0.1/16"
  }

  role_based_access_control {
    enabled = true
    azure_active_directory {
      managed = true
    }
  }

  tags = {
    Environment = "Development"
    Application = "GoldenSignalsAI"
  }
}

# Azure Container Registry
resource "azurerm_container_registry" "acr" {
  name                = "goldensignalsacr"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard"
  admin_enabled       = true

  network_rule_set {
    default_action = "Allow"
  }
}

# Grant AKS access to ACR
resource "azurerm_role_assignment" "aks_acr" {
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                           = azurerm_container_registry.acr.id
  skip_service_principal_aad_check = true
}

# Azure Key Vault
resource "azurerm_key_vault" "kv" {
  name                        = "goldensignals-kv"
  location                    = azurerm_resource_group.rg.location
  resource_group_name         = azurerm_resource_group.rg.name
  enabled_for_disk_encryption = true
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
  sku_name                   = "standard"

  network_acls {
    default_action = "Allow"
    bypass         = "AzureServices"
  }
}

# Azure Monitor Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "law" {
  name                = "goldensignals-law"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = "Development"
    Application = "GoldenSignalsAI"
  }
}

# Application Insights
resource "azurerm_application_insights" "appinsights" {
  name                = "goldensignals-appinsights"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.law.id

  tags = {
    Environment = "Development"
    Application = "GoldenSignalsAI"
  }
}

# Azure Monitor Action Group
resource "azurerm_monitor_action_group" "critical" {
  name                = "goldensignals-critical"
  resource_group_name = azurerm_resource_group.rg.name
  short_name          = "critical"

  email_receiver {
    name          = "admin"
    email_address = "admin@goldensignals.ai"
  }

  webhook_receiver {
    name        = "slack"
    service_uri = var.slack_webhook_url
  }
}

output "aks_credentials_command" {
  value = "az aks get-credentials --resource-group ${azurerm_resource_group.rg.name} --name ${azurerm_kubernetes_cluster.aks.name}"
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "application_insights_key" {
  value = azurerm_application_insights.appinsights.instrumentation_key
  sensitive = true
} 
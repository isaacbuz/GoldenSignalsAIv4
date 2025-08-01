resource "azurerm_public_ip" "agw" {
  name                = "goldensignals-agw-pip"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = ["1", "2", "3"]
}

resource "azurerm_application_gateway" "agw" {
  name                = "goldensignals-agw"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }

  gateway_ip_configuration {
    name      = "gateway-ip-config"
    subnet_id = azurerm_subnet.appgw.id
  }

  frontend_port {
    name = "https-port"
    port = 443
  }

  frontend_port {
    name = "http-port"
    port = 80
  }

  frontend_ip_configuration {
    name                 = "frontend-ip-config"
    public_ip_address_id = azurerm_public_ip.agw.id
  }

  ssl_policy {
    policy_type = "Predefined"
    policy_name = "AppGwSslPolicy20170401S"
  }

  ssl_certificate {
    name     = "goldensignals-cert"
    key_vault_secret_id = azurerm_key_vault_certificate.goldensignals.secret_id
  }

  # HTTP to HTTPS redirect configuration
  http_listener {
    name                           = "http-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name            = "http-port"
    protocol                      = "Http"
  }

  redirect_configuration {
    name                 = "http-to-https"
    redirect_type       = "Permanent"
    target_listener_name = "https-listener"
    include_path        = true
    include_query_string = true
  }

  # Frontend path-based routing
  url_path_map {
    name                               = "frontend-paths"
    default_backend_address_pool_name  = "frontend-pool"
    default_backend_http_settings_name = "frontend-settings"

    path_rule {
      name                       = "static-content"
      paths                      = ["/static/*", "/*.ico", "/*.png", "/*.jpg", "/*.svg"]
      backend_address_pool_name  = "frontend-static-pool"
      backend_http_settings_name = "frontend-static-settings"
    }

    path_rule {
      name                       = "api"
      paths                      = ["/api/*"]
      backend_address_pool_name  = "backend-pool"
      backend_http_settings_name = "backend-settings"
    }
  }

  # Static content backend pool
  backend_address_pool {
    name = "frontend-static-pool"
  }

  # API backend pool
  backend_address_pool {
    name = "backend-pool"
  }

  # Static content settings with compression and caching
  backend_http_settings {
    name                  = "frontend-static-settings"
    cookie_based_affinity = "Disabled"
    port                  = 8080
    protocol             = "Http"
    request_timeout      = 30
    probe_name           = "frontend-static-probe"

    compression {
      enabled = true
      mime_types = [
        "text/html",
        "text/css",
        "text/javascript",
        "application/javascript",
        "application/json",
        "image/svg+xml"
      ]
      min_response_size_bytes = 1024
    }

    custom_response_headers = {
      "X-Content-Type-Options"  = "nosniff"
      "X-Frame-Options"         = "SAMEORIGIN"
      "X-XSS-Protection"        = "1; mode=block"
      "Strict-Transport-Security" = "max-age=31536000; includeSubDomains"
      "Cache-Control"           = "public, max-age=31536000"
    }
  }

  # API backend settings
  backend_http_settings {
    name                  = "backend-settings"
    cookie_based_affinity = "Disabled"
    port                  = 8000
    protocol             = "Http"
    request_timeout      = 60
    probe_name           = "backend-probe"

    custom_response_headers = {
      "X-Content-Type-Options"  = "nosniff"
      "X-Frame-Options"         = "DENY"
      "X-XSS-Protection"        = "1; mode=block"
      "Strict-Transport-Security" = "max-age=31536000; includeSubDomains"
    }
  }

  # Static content health probe
  probe {
    name                = "frontend-static-probe"
    host                = "127.0.0.1"
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3
    protocol            = "Http"
    path                = "/static/health.txt"
    match {
      status_code = ["200-399"]
    }
  }

  # API health probe
  probe {
    name                = "backend-probe"
    host                = "127.0.0.1"
    interval            = 30
    timeout             = 30
    unhealthy_threshold = 3
    protocol            = "Http"
    path                = "/api/health"
    match {
      status_code = ["200-399"]
    }
  }

  # WAF custom rules
  waf_configuration {
    enabled                  = true
    firewall_mode           = "Prevention"
    rule_set_type           = "OWASP"
    rule_set_version        = "3.2"
    file_upload_limit_mb    = 100
    max_request_body_size_kb = 128

    disabled_rule_group {
      rule_group_name = "REQUEST-931-APPLICATION-ATTACK-RFI"
      rules          = ["931130"]
    }

    disabled_rule_group {
      rule_group_name = "REQUEST-942-APPLICATION-ATTACK-SQLI"
      rules          = ["942440", "942450"]
    }

    custom_rules {
      name      = "RequireCSRFToken"
      priority  = 1
      rule_type = "MatchRule"
      match_conditions {
        match_variables {
          variable_name = "RequestHeaders"
          selector     = "X-CSRF-Token"
        }
        operator           = "Equal"
        negation_condition = true
        match_values       = [""]
      }
      action = "Block"
    }

    custom_rules {
      name      = "BlockHighRiskCountries"
      priority  = 2
      rule_type = "MatchRule"
      match_conditions {
        match_variables {
          variable_name = "RemoteAddr"
        }
        operator           = "GeoMatch"
        match_values       = ["KP", "IR", "CU", "SD", "SY"]
      }
      action = "Block"
    }
  }

  http_listener {
    name                           = "https-listener"
    frontend_ip_configuration_name = "frontend-ip-config"
    frontend_port_name            = "https-port"
    protocol                      = "Https"
    ssl_certificate_name          = "goldensignals-cert"
  }

  request_routing_rule {
    name                       = "https-rule"
    rule_type                 = "Basic"
    http_listener_name        = "https-listener"
    backend_address_pool_name = "frontend-pool"
    backend_http_settings_name = "frontend-settings"
    priority                  = 100
  }

  identity {
    type = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.agw.id]
  }

  tags = {
    Environment = "Production"
    Application = "GoldenSignalsAI"
  }
}

resource "azurerm_user_assigned_identity" "agw" {
  name                = "goldensignals-agw-identity"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
}

# Grant Application Gateway access to Key Vault
resource "azurerm_key_vault_access_policy" "agw" {
  key_vault_id = azurerm_key_vault.kv.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.agw.principal_id

  secret_permissions = [
    "Get"
  ]

  certificate_permissions = [
    "Get"
  ]
}

# Create TLS certificate in Key Vault
resource "azurerm_key_vault_certificate" "goldensignals" {
  name         = "goldensignals-cert"
  key_vault_id = azurerm_key_vault.kv.id

  certificate_policy {
    issuer_parameters {
      name = "Self"
    }

    key_properties {
      exportable = true
      key_size   = 2048
      key_type   = "RSA"
      reuse_key  = true
    }

    lifetime_action {
      action {
        action_type = "AutoRenew"
      }

      trigger {
        days_before_expiry = 30
      }
    }

    secret_properties {
      content_type = "application/x-pkcs12"
    }

    x509_certificate_properties {
      key_usage = [
        "cRLSign",
        "dataEncipherment",
        "digitalSignature",
        "keyAgreement",
        "keyCertSign",
        "keyEncipherment",
      ]

      subject            = "CN=*.goldensignals.ai"
      validity_in_months = 12

      subject_alternative_names {
        dns_names = [
          "trading.goldensignals.ai",
          "dashboard.goldensignals.ai"
        ]
      }
    }
  }
}

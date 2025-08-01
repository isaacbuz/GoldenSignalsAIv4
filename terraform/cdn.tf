resource "azurerm_cdn_profile" "frontend" {
  name                = "goldensignals-cdn"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "Premium_Verizon"

  tags = {
    Environment = "Production"
    Application = "GoldenSignalsAI"
  }
}

resource "azurerm_cdn_endpoint" "frontend" {
  name                = "goldensignals-cdn-endpoint"
  profile_name        = azurerm_cdn_profile.frontend.name
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  origin {
    name       = "frontend-origin"
    host_name  = azurerm_public_ip.agw.ip_address
    http_port  = 80
    https_port = 443
  }

  origin_host_header = "trading.goldensignals.ai"

  delivery_rule {
    name  = "EnforceHTTPS"
    order = 1

    request_scheme_condition {
      operator     = "Equal"
      match_values = ["HTTP"]
    }

    url_redirect_action {
      redirect_type = "Found"
      protocol      = "Https"
    }
  }

  delivery_rule {
    name  = "CacheStaticFiles"
    order = 2

    file_extension_condition {
      operator     = "Equal"
      match_values = ["css", "js", "jpg", "jpeg", "png", "gif", "svg", "ico", "woff", "woff2"]
    }

    cache_expiration_action {
      behavior = "Override"
      duration = "7.00:00:00"
    }
  }

  delivery_rule {
    name  = "CompressContent"
    order = 3

    file_extension_condition {
      operator     = "Equal"
      match_values = ["css", "js", "html", "json", "svg"]
    }

    url_file_extension_condition {
      operator     = "Equal"
      match_values = ["css", "js", "html", "json", "svg"]
    }

    compression_enabled_action {
      compression = "Enabled"
    }
  }

  optimization_type = "GeneralWebDelivery"

  global_delivery_rule {
    cache_expiration_action {
      behavior = "SetIfMissing"
      duration = "1.00:00:00"
    }

    modify_response_header_action {
      action = "Append"
      name   = "X-Content-Type-Options"
      value  = "nosniff"
    }

    modify_response_header_action {
      action = "Append"
      name   = "X-Frame-Options"
      value  = "SAMEORIGIN"
    }

    modify_response_header_action {
      action = "Append"
      name   = "X-XSS-Protection"
      value  = "1; mode=block"
    }

    modify_response_header_action {
      action = "Append"
      name   = "Strict-Transport-Security"
      value  = "max-age=31536000; includeSubDomains"
    }
  }

  is_http_allowed                     = false
  is_https_allowed                    = true
  is_compression_enabled              = true
  content_types_to_compress          = ["text/plain", "text/html", "text/css", "text/javascript", "application/x-javascript", "application/javascript", "application/json", "application/xml"]
  querystring_caching_behaviour      = "IgnoreQueryString"
  response_timeout_seconds           = 30
}

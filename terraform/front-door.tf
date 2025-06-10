# Commented out for future use
# # DDoS Protection Plan
# resource "azurerm_network_ddos_protection_plan" "ddos" {
#   name                = "goldensignals-ddos-plan"
#   location            = azurerm_resource_group.rg.location
#   resource_group_name = azurerm_resource_group.rg.name
# }
# 
# # Front Door Profile
# resource "azurerm_cdn_frontdoor_profile" "fd" {
#   name                = "goldensignals-fd"
#   resource_group_name = azurerm_resource_group.rg.name
#   sku_name            = "Premium_AzureFrontDoor"
# 
#   tags = {
#     Environment = "Production"
#     Application = "GoldenSignalsAI"
#   }
# }
# 
# # Front Door Endpoint
# resource "azurerm_cdn_frontdoor_endpoint" "fd" {
#   name                     = "goldensignals-endpoint"
#   cdn_frontdoor_profile_id = azurerm_cdn_frontdoor_profile.fd.id
# }
# 
# # Front Door Origin Group
# resource "azurerm_cdn_frontdoor_origin_group" "fd" {
#   name                     = "goldensignals-origin-group"
#   cdn_frontdoor_profile_id = azurerm_cdn_frontdoor_profile.fd.id
#   session_affinity_enabled = true
# 
#   load_balancing {
#     sample_size                 = 4
#     successful_samples_required = 3
#     additional_latency_in_ms    = 50
#   }
# 
#   health_probe {
#     interval_in_seconds = 100
#     path                = "/health"
#     protocol            = "Https"
#     request_type        = "HEAD"
#   }
# }
# 
# # Front Door Origin
# resource "azurerm_cdn_frontdoor_origin" "appgw" {
#   name                          = "appgw-origin"
#   cdn_frontdoor_origin_group_id = azurerm_cdn_frontdoor_origin_group.fd.id
#   enabled                       = true
# 
#   host_name          = azurerm_public_ip.agw.ip_address
#   http_port          = 80
#   https_port         = 443
#   priority           = 1
#   weight             = 1000
#   certificate_name_check_enabled = true
# }
# 
# # Front Door Route
# resource "azurerm_cdn_frontdoor_route" "fd" {
#   name                          = "goldensignals-route"
#   cdn_frontdoor_endpoint_id     = azurerm_cdn_frontdoor_endpoint.fd.id
#   cdn_frontdoor_origin_group_id = azurerm_cdn_frontdoor_origin_group.fd.id
#   enabled                       = true
# 
#   forwarding_protocol    = "HttpsOnly"
#   https_redirect_enabled = true
#   patterns_to_match     = ["/*"]
#   supported_protocols    = ["Http", "Https"]
# 
#   cache {
#     query_string_caching_behavior = "IgnoreQueryString"
#     compression_enabled          = true
#     content_types_to_compress    = ["text/html", "text/css", "application/javascript"]
#   }
# }
# 
# # WAF Policy
# resource "azurerm_cdn_frontdoor_firewall_policy" "fd" {
#   name                = "goldensignals-waf-policy"
#   resource_group_name = azurerm_resource_group.rg.name
#   sku_name            = azurerm_cdn_frontdoor_profile.fd.sku_name
#   enabled             = true
#   mode               = "Prevention"
# 
#   managed_rule {
#     type    = "DefaultRuleSet"
#     version = "1.0"
#   }
# 
#   managed_rule {
#     type    = "Microsoft_BotManagerRuleSet"
#     version = "1.0"
#   }
# 
#   custom_rule {
#     name                           = "BlockHighRiskCountries"
#     enabled                        = true
#     priority                       = 1
#     rate_limit_duration_in_minutes = 1
#     rate_limit_threshold          = 10
#     type                          = "MatchRule"
#     action                        = "Block"
# 
#     match_condition {
#       match_variable     = "RemoteAddr"
#       operator           = "GeoMatch"
#       negation_condition = false
#       match_values       = ["KP", "IR", "CU", "SD", "SY"]
#     }
#   }
# 
#   custom_rule {
#     name                           = "RateLimitRule"
#     enabled                        = true
#     priority                       = 2
#     rate_limit_duration_in_minutes = 1
#     rate_limit_threshold          = 1000
#     type                          = "RateLimitRule"
#     action                        = "Block"
# 
#     match_condition {
#       match_variable     = "RequestUri"
#       operator           = "Any"
#       negation_condition = false
#     }
#   }
# }
# 
# # Associate WAF Policy with Front Door
# resource "azurerm_cdn_frontdoor_security_policy" "fd" {
#   name                     = "goldensignals-security-policy"
#   cdn_frontdoor_profile_id = azurerm_cdn_frontdoor_profile.fd.id
# 
#   security_policies {
#     firewall {
#       cdn_frontdoor_firewall_policy_id = azurerm_cdn_frontdoor_firewall_policy.fd.id
# 
#       association {
#         patterns_to_match = ["/*"]
#         domain {
#           cdn_frontdoor_domain_id = azurerm_cdn_frontdoor_endpoint.fd.id
#         }
#       }
#     }
#   }
# } 
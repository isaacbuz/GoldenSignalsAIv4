# Service Mesh Configuration
resource "helm_release" "istio" {
  name       = "istio"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "base"
  namespace  = "istio-system"
  create_namespace = true

  set {
    name  = "global.mtls.enabled"
    value = true
  }
}

resource "helm_release" "istio_ingress" {
  name       = "istio-ingress"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "gateway"
  namespace  = "istio-system"
  depends_on = [helm_release.istio]
}

# Kiali Dashboard
resource "helm_release" "kiali" {
  name       = "kiali"
  repository = "https://kiali.org/helm-charts"
  chart      = "kiali-operator"
  namespace  = "istio-system"
  depends_on = [helm_release.istio]

  values = [
    <<-EOT
    cr:
      spec:
        deployment:
          accessible_namespaces: ["*"]
        auth:
          strategy: "anonymous"
        external_services:
          prometheus:
            url: "http://prometheus-server.monitoring.svc:9090"
    EOT
  ]
}

# Service Mesh Monitoring
resource "helm_release" "jaeger" {
  name       = "jaeger"
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger-operator"
  namespace  = "istio-system"
  depends_on = [helm_release.istio]
}

# Circuit Breaker and Retry Policies
resource "kubernetes_manifest" "circuit_breaker" {
  manifest = {
    apiVersion = "networking.istio.io/v1alpha3"
    kind       = "DestinationRule"
    metadata = {
      name      = "circuit-breaker"
      namespace = "goldensignals"
    }
    spec = {
      host = "goldensignalsai-backend"
      trafficPolicy = {
        connectionPool = {
          tcp = {
            maxConnections = 100
          }
          http = {
            http1MaxPendingRequests = 1
            maxRequestsPerConnection = 1
          }
        }
        outlierDetection = {
          consecutive5xxErrors = 5
          interval            = "30s"
          baseEjectionTime    = "30s"
          maxEjectionPercent  = 100
        }
      }
    }
  }
}

# Retry Policy
resource "kubernetes_manifest" "retry_policy" {
  manifest = {
    apiVersion = "networking.istio.io/v1alpha3"
    kind       = "VirtualService"
    metadata = {
      name      = "retry-policy"
      namespace = "goldensignals"
    }
    spec = {
      hosts = ["*"]
      http = [{
        route = [{
          destination = {
            host = "goldensignalsai-backend"
          }
        }]
        retries = {
          attempts = 3
          perTryTimeout = "2s"
          retryOn = "connect-failure,refused-stream,unavailable,cancelled,resource-exhausted"
        }
      }]
    }
  }
}

# Service Mesh Metrics
resource "kubernetes_manifest" "metrics" {
  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    metadata = {
      name      = "istio-metrics"
      namespace = "monitoring"
      labels = {
        release = "prometheus"
      }
    }
    spec = {
      selector = {
        matchLabels = {
          app = "istio-proxy"
        }
      }
      namespaceSelector = {
        any = true
      }
      endpoints = [{
        port = "http-envoy-prom"
        path = "/stats/prometheus"
      }]
    }
  }
} 
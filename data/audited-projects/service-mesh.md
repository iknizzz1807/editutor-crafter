# AUDIT & FIX: service-mesh

## CRITIQUE
- **No Control Plane**: The project describes a data plane (sidecar proxy) but completely omits the control plane. Without a control plane (like Istiod or a simplified equivalent), there's no mechanism to distribute routing rules, certificate authority roots, or service discovery configuration to sidecars. The sidecars would need hardcoded configuration, which defeats the purpose of a mesh.
- **Elevated Privileges Not Addressed**: iptables manipulation requires CAP_NET_ADMIN or root. The project never mentions this, nor does it address how the sidecar gets these privileges (init container, CNI plugin, etc.). A learner will hit permission errors immediately.
- **55 Hours is Unrealistic for 4 Milestones**: Each milestone is estimated at 14 hours, but Traffic Interception alone (iptables + transparent proxy + protocol detection) is a multi-week effort for most developers. The estimates are suspiciously uniform.
- **Consistent Hashing Deliverable in M4 but Not in AC**: M4 ACs mention round-robin, least-connections, weighted, and health-aware routing. Consistent hashing is in the concepts and pitfalls but not in the ACs, yet it's in the project essence. This is inconsistent.
- **mTLS Without CA Infrastructure**: M3 says 'Generate unique X.509 certificates for each service instance' but doesn't specify who signs them. Without a Certificate Authority (which would live in the control plane), you can't do mTLS. Self-signed certs per service don't enable mutual trust.
- **SPIFFE ID Without SPIRE**: M3 mentions SPIFFE identity integration but provides no guidance on implementing or integrating with a SPIFFE runtime (SPIRE). SPIFFE is a specification, not a library you drop in.
- **Protocol Detection is Non-Trivial**: M1 casually mentions 'Parse and identify HTTP and gRPC protocols from intercepted traffic' as one AC among four. Protocol detection from raw bytes (especially distinguishing HTTP/1.1, HTTP/2, and TLS ClientHello) is a significant engineering challenge that deserves more attention.
- **Java Listed as Recommended but Impractical**: Building a transparent proxy with iptables integration and low-level socket options (SO_ORIGINAL_DST) in Java is extremely painful. Go, Rust, or C++ are the only practical choices.
- **No Observability/Metrics Milestone**: A service mesh's primary value proposition is observability (request metrics, distributed tracing). The project completely ignores this.

## FIXED YAML
```yaml
id: service-mesh
name: Service Mesh
description: Sidecar proxy with control plane for service-to-service communication
difficulty: advanced
estimated_hours: "60-80"
essence: >
  Layer-7 traffic interception via iptables/TPROXY redirection, dynamic endpoint
  resolution through control-plane-distributed configuration, cryptographic
  identity via mutual TLS with a centralized Certificate Authority, and
  configurable load balancing algorithms—all orchestrated by a control plane
  that pushes configuration to sidecar proxies.
why_important: >
  Service meshes like Istio and Linkerd are the standard for securing and
  observing microservice communication. Understanding both the data plane
  (sidecar proxy) and control plane (configuration distribution) is essential
  for debugging production issues, optimizing performance, and designing
  secure service architectures.
learning_outcomes:
  - Implement transparent traffic interception using iptables with proper privilege management
  - Build protocol detection for HTTP/1.1, HTTP/2, and raw TCP from intercepted bytes
  - Design a control plane that distributes routing configuration and CA certificates to sidecars
  - Integrate with service discovery (Kubernetes, Consul) for dynamic endpoint resolution
  - Implement mutual TLS with a centralized CA and automated certificate rotation
  - Build multiple load balancing algorithms including consistent hashing
  - Implement request-level observability (metrics, access logs, trace header propagation)
skills:
  - Network Traffic Interception (iptables, TPROXY)
  - Transparent Proxy Implementation
  - Control Plane Design (xDS-like configuration distribution)
  - Service Discovery Integration
  - Mutual TLS and Certificate Authority
  - Certificate Lifecycle Management
  - Load Balancing Algorithms
  - Protocol Detection (HTTP/1.1, HTTP/2, TLS)
  - Request-Level Observability
tags:
  - advanced
  - distributed-systems
  - envoy
  - istio
  - microservices
  - mtls
  - networking
  - service-mesh
  - sidecar
architecture_doc: architecture-docs/service-mesh/index.md
languages:
  recommended:
    - Go
    - Rust
    - C++
  also_possible: []
resources:
  - name: "Envoy Proxy Documentation"
    url: "https://www.envoyproxy.io/docs"
    type: documentation
  - name: "Linux TPROXY Documentation"
    url: "https://docs.kernel.org/networking/tproxy.html"
    type: documentation
  - name: "Go and Proxy Servers Tutorial"
    url: "https://eli.thegreenplace.net/2022/go-and-proxy-servers-part-1-http-proxies/"
    type: tutorial
  - name: "Envoy xDS Protocol"
    url: "https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol"
    type: documentation
  - name: "SPIFFE Specification"
    url: "https://spiffe.io/docs/latest/spiffe-about/overview/"
    type: documentation
prerequisites:
  - type: project
    id: api-gateway
  - type: project
    id: circuit-breaker
  - type: skill
    name: "Linux networking (iptables, network namespaces)"
milestones:
  - id: service-mesh-m1
    name: "Traffic Interception & Protocol Detection"
    description: >
      Intercept inbound and outbound traffic transparently using iptables
      REDIRECT/TPROXY rules and detect the application protocol (HTTP/1.1,
      HTTP/2, TLS, raw TCP) from intercepted bytes. Handle privilege
      requirements via an init container or CAP_NET_ADMIN.
    acceptance_criteria:
      - "iptables rules redirect all inbound TCP traffic on specified ports to the sidecar proxy listening port using REDIRECT or TPROXY target"
      - "iptables OWNER module (--uid-owner) excludes the proxy process's own traffic from redirection, preventing redirect loops"
      - "Separate ip6tables rules handle IPv6 traffic if dual-stack is required"
      - "Proxy recovers the original destination address using getsockopt(SO_ORIGINAL_DST) on redirected connections"
      - "Protocol detection reads the first bytes of a connection and distinguishes: TLS ClientHello (0x16 0x03), HTTP/1.x (method line), HTTP/2 connection preface ('PRI * HTTP/2.0'), and falls back to raw TCP pass-through for unrecognized protocols"
      - "Init script or init container sets up iptables rules before the application starts, requiring only CAP_NET_ADMIN (not full root)"
      - "Integration test: application makes HTTP request to external service; proxy intercepts, logs the request, and forwards it transparently without application code changes"
      - "Pass-through test: unrecognized protocol traffic is forwarded without modification or error"
    pitfalls:
      - "Redirect loops: if the proxy's own outbound connections are redirected back to itself, the proxy deadlocks; always exclude proxy UID/GID with iptables OWNER module"
      - "SO_ORIGINAL_DST is Linux-specific and not available on macOS; development requires a Linux VM or container"
      - "IPv6 requires separate ip6tables rules; forgetting this breaks IPv6-only services silently"
      - "Protocol detection must be non-destructive: bytes read for detection must be replayed to the upstream handler (use buffered peek, not consume)"
      - "CAP_NET_ADMIN is required for iptables; without it, rule installation fails with EPERM. Document the privilege model explicitly."
    concepts:
      - Transparent proxying with iptables REDIRECT and TPROXY targets
      - SO_ORIGINAL_DST socket option for destination recovery
      - iptables OWNER module for loop prevention
      - Protocol detection from connection preamble bytes
      - Network namespaces and init container privilege model
    deliverables:
      - "iptables rule installer script/init container with REDIRECT rules and OWNER exclusion"
      - "Transparent proxy listener accepting redirected connections and recovering original destination"
      - "Protocol detector classifying connections as HTTP/1.1, HTTP/2, TLS, or TCP pass-through"
      - "Integration test demonstrating transparent interception without application changes"
    estimated_hours: "12-16"

  - id: service-mesh-m2
    name: "Control Plane & Service Discovery"
    description: >
      Build a minimal control plane that discovers service endpoints from
      Kubernetes or Consul and pushes routing configuration to sidecar
      proxies via a gRPC or HTTP streaming API (simplified xDS). Sidecars
      maintain a local cache of endpoints.
    acceptance_criteria:
      - "Control plane watches Kubernetes Endpoints (or Consul service catalog) and detects endpoint additions and removals in real-time via watch/long-poll API"
      - "Control plane exposes a gRPC or HTTP streaming API that sidecars connect to for receiving endpoint updates (simplified Endpoint Discovery Service)"
      - "Sidecar proxy receives endpoint updates from control plane and maintains a local endpoint cache keyed by service name"
      - "Endpoint health status is tracked; unhealthy endpoints are excluded from load balancing selection"
      - "Sidecar reconnects to control plane automatically with exponential backoff if the stream disconnects"
      - "Local cache serves endpoint lookups even when control plane is temporarily unavailable (graceful degradation)"
      - "Cache staleness test: disconnect control plane for 60s, verify sidecar continues routing to last-known endpoints; reconnect and verify cache is refreshed within 5 seconds"
      - "DNS-based fallback: if service is not found in control plane, resolve via DNS as a last resort"
    pitfalls:
      - "Watch connections drop silently without TCP keepalive; implement application-level heartbeat or gRPC keepalive"
      - "Stale cache continues routing to dead endpoints; implement a TTL or version-based invalidation"
      - "DNS TTL conflicts with dynamic discovery; if DNS cache TTL is 30s but the endpoint changed 1s ago, stale routing occurs"
      - "Control plane must handle many sidecar connections; use streaming (not polling) to reduce API server load"
    concepts:
      - Control plane architecture (simplified Istiod)
      - xDS-like configuration distribution protocol
      - Service discovery watch APIs (Kubernetes, Consul)
      - Local endpoint cache with TTL and version tracking
      - Graceful degradation when control plane is unavailable
    deliverables:
      - "Control plane service watching Kubernetes Endpoints or Consul catalog"
      - "Configuration distribution API (gRPC stream or HTTP SSE) pushing endpoint updates to sidecars"
      - "Sidecar endpoint cache with version tracking and TTL-based staleness detection"
      - "Reconnection logic with exponential backoff for control plane disconnections"
      - "DNS fallback resolution for services not registered in the control plane"
    estimated_hours: "12-16"

  - id: service-mesh-m3
    name: "mTLS with Centralized CA"
    description: >
      Implement mutual TLS between services using a centralized Certificate
      Authority (CA) in the control plane that signs per-service certificates.
      Handle automatic certificate rotation without dropping active connections.
    acceptance_criteria:
      - "Control plane runs a CA that issues X.509 certificates signed by a root CA; each sidecar receives a unique certificate with a SPIFFE-format SAN (spiffe://trust-domain/ns/namespace/sa/service-account)"
      - "Sidecar generates a private key locally and sends a Certificate Signing Request (CSR) to the control plane CA; private key never leaves the sidecar"
      - "All inter-service connections use mTLS: sidecar presents its certificate and verifies the peer's certificate chain against the CA root"
      - "Certificate rotation: sidecar requests a new certificate before the current one reaches 80% of its lifetime; new connections use the new certificate while existing connections continue with the old one until they close"
      - "Identity verification test: service A connects to service B; service B's sidecar verifies that service A's certificate SAN matches an allowed identity; connections from unauthorized identities are rejected with TLS handshake failure"
      - "Clock skew tolerance: certificates have a NotBefore set 5 minutes in the past to tolerate clock differences between nodes"
    pitfalls:
      - "Certificate rotation during active requests drops connections if the TLS context is swapped atomically; use dynamic TLS credential reloading (e.g., tls.Config GetCertificate callback in Go) to serve the new cert for new connections while old connections finish"
      - "Missing SAN (Subject Alternative Name) in certificates causes modern TLS libraries to reject them even if CN matches; always set SAN"
      - "Clock skew between nodes makes valid certificates appear expired; set NotBefore conservatively in the past"
      - "Private key generated on the control plane and transmitted to the sidecar is a security risk; always generate keys locally and use CSR flow"
      - "Root CA key compromise breaks the entire mesh; store root CA key in HSM or use an intermediate CA for signing"
    concepts:
      - X.509 certificates with SPIFFE-format SANs
      - Certificate Signing Request (CSR) workflow
      - TLS handshake and mutual authentication
      - Dynamic certificate reloading for zero-downtime rotation
      - Trust domain and identity-based authorization
    deliverables:
      - "Control plane CA: CSR signing endpoint issuing X.509 certificates with SPIFFE SANs"
      - "Sidecar CSR flow: local key generation, CSR submission, certificate installation"
      - "mTLS enforcement: sidecar requires mutual certificate verification on all connections"
      - "Certificate rotation: automatic renewal at 80% lifetime with graceful handoff"
      - "Identity verification: peer SAN validation against authorization policy"
    estimated_hours: "14-18"

  - id: service-mesh-m4
    name: "Load Balancing Algorithms"
    description: >
      Implement multiple load balancing algorithms: round-robin, least
      connections, weighted round-robin, and consistent hashing with virtual
      nodes. All algorithms must be health-aware.
    acceptance_criteria:
      - "Round-robin distributes requests evenly across N healthy endpoints; after N requests, each endpoint has received exactly 1 request (verified by test)"
      - "Least-connections routes each request to the endpoint with the fewest active in-flight connections, tracked atomically"
      - "Weighted round-robin distributes requests proportional to configured weights; endpoint with weight=3 receives 3x traffic of weight=1 endpoint (within 5% over 1000 requests)"
      - "Consistent hashing maps request keys (e.g., user ID header) to endpoints using a hash ring with configurable virtual nodes (default 150 per endpoint); adding/removing an endpoint remaps at most 1/N of keys"
      - "All algorithms exclude endpoints marked unhealthy by the service discovery health status"
      - "Weight=0 is handled correctly: endpoint is excluded from selection, not causing division by zero"
      - "Slow-start: newly added endpoint receives gradually increasing traffic over a configurable warm-up period to avoid overwhelming a cold cache"
    pitfalls:
      - "Consistent hashing ring must be rebuilt when endpoints change; use sorted virtual node array with binary search for O(log V) lookup"
      - "Least-connections can thundering-herd to a recovered endpoint that suddenly has 0 connections; combine with slow-start"
      - "Weight=0 causes division by zero in naive weighted round-robin; treat weight=0 as 'do not route'"
      - "Atomic counter for active connections must handle both increment (on request start) and decrement (on response/error); missing decrement on error path causes permanent count inflation"
    concepts:
      - Consistent hashing with virtual nodes
      - Least-connections tracking with atomic counters
      - Weighted round-robin distribution
      - Health-aware endpoint selection
      - Slow-start for newly added endpoints
    deliverables:
      - "Round-robin algorithm with health-aware endpoint filtering"
      - "Least-connections algorithm with atomic in-flight tracking"
      - "Weighted round-robin with configurable per-endpoint weights"
      - "Consistent hashing with virtual nodes and key-based routing"
      - "Slow-start ramping for newly added endpoints"
      - "Algorithm selection configurable per-service via control plane"
    estimated_hours: "10-14"

  - id: service-mesh-m5
    name: "Request Observability"
    description: >
      Instrument the sidecar proxy to emit per-request metrics (latency,
      status codes, throughput), structured access logs, and propagate
      distributed tracing headers.
    acceptance_criteria:
      - "Sidecar emits Prometheus-compatible metrics: request count, request latency histogram, and error rate, labeled by source service, destination service, HTTP method, and response code"
      - "Structured access logs record timestamp, source, destination, method, path, response code, latency, and bytes transferred for every proxied request"
      - "Distributed tracing headers (e.g., X-Request-ID, traceparent from W3C Trace Context) are propagated from inbound to outbound requests; if no trace header exists, one is generated"
      - "Metrics endpoint exposes /metrics in Prometheus exposition format, scraped by a Prometheus instance in test"
      - "Observability overhead test: sidecar adds less than 1ms p99 latency overhead for proxied requests under 1000 RPS load"
    pitfalls:
      - "High-cardinality labels (e.g., full URL path) cause metric explosion and OOM; use parameterized route patterns"
      - "Logging every request at high throughput overwhelms disk I/O; use sampling or async buffered writes"
      - "Trace header propagation must be protocol-aware: HTTP/1.1 headers are different from gRPC metadata"
    concepts:
      - RED metrics (Rate, Errors, Duration)
      - Prometheus exposition format
      - W3C Trace Context propagation
      - Structured access logging
    deliverables:
      - "Prometheus metrics exporter with request count, latency histogram, and error rate per service pair"
      - "Structured access log writer with configurable output (stdout, file, or syslog)"
      - "Trace header propagation middleware for HTTP and gRPC"
      - "Performance benchmark verifying sub-millisecond proxy overhead"
    estimated_hours: "10-14"
```
# AUDIT & FIX: load-balancer-basic

## CRITIQUE
- **Technical Inaccuracy (Confirmed):** IP hash for sticky sessions is unreliable behind NAT, corporate proxies, or CDNs where thousands of users share a single IP. This causes severe load imbalance. Cookie-based affinity is the standard approach and should be the primary teaching mechanism. IP hash can be mentioned as a simple but flawed alternative.
- **Logical Gap (Confirmed):** Connection pooling to backends is completely missing. Opening a new TCP connection for every proxied request will exhaust ephemeral ports and file descriptors under any meaningful load. This should be in M1, not an afterthought.
- **AC Over-Precision vs Under-Precision:** M2 says 'less than 5% skew over 1000 requests' which is good and measurable, but M1 says 'All proxied requests are logged' without specifying log format or where logs go.
- **Header Handling Gaps:** M1 mentions X-Forwarded-For and X-Forwarded-Proto but misses critical headers: Host header rewriting, Connection/hop-by-hop header stripping, and Content-Length validation.
- **Missing Graceful Degradation:** What happens when ALL backends are unhealthy? The current design doesn't specify behavior — should it return 503, queue requests, or use a stale backend?
- **M4 Scope Creep:** Four algorithms plus runtime hot-swapping in a single milestone is too much. This should be split or simplified.
- **Pitfall Weakness:** 'Not forwarding all headers' is vague. The real pitfall is forwarding hop-by-hop headers (Connection, Keep-Alive, Transfer-Encoding, Upgrade) to the backend, which is a protocol violation.
- **Missing Metrics:** No mention of observability — request latency histograms, error rates, backend response times. A load balancer without metrics is undebuggable.

## FIXED YAML
```yaml
id: load-balancer-basic
name: Load Balancer (Basic)
description: >-
  Build an HTTP reverse proxy load balancer with round-robin distribution,
  connection pooling, active health checks, and multiple balancing algorithms.
difficulty: intermediate
estimated_hours: "18-25"
essence: >-
  HTTP reverse proxying with connection pooling to backends, round-robin and
  weighted request distribution, active health monitoring with failure
  thresholds, and graceful degradation when backends fail.
why_important: >-
  Building a load balancer teaches core distributed systems concepts that
  power production infrastructure — reverse proxy mechanics, connection
  management, failure detection, and algorithmic traffic distribution — skills
  directly applicable to DevOps, backend engineering, and systems architecture.
learning_outcomes:
  - Implement HTTP reverse proxy with proper header handling and connection pooling
  - Design round-robin distribution with thread-safe backend rotation
  - Build active health check system with configurable thresholds and recovery
  - Handle all-backends-down graceful degradation with 503 responses
  - Implement weighted round-robin and least-connections algorithms
  - Manage backend connection pools to prevent socket exhaustion
  - Strip hop-by-hop headers and add X-Forwarded-* headers correctly
  - Debug connection timeout and backend failure scenarios
skills:
  - Reverse Proxy Design
  - HTTP Protocol Handling
  - Connection Pool Management
  - Health Check Monitoring
  - Request Routing Algorithms
  - Failure Detection and Recovery
  - Hop-by-hop Header Handling
  - Concurrent Programming
tags:
  - go
  - health-checks
  - intermediate
  - javascript
  - networking
  - python
  - round-robin
architecture_doc: architecture-docs/load-balancer-basic/index.md
languages:
  recommended:
    - Go
    - Python
    - JavaScript
  also_possible:
    - Rust
    - Java
resources:
  - name: Build Your Own Load Balancer
    url: https://codingchallenges.fyi/challenges/challenge-load-balancer/
    type: tutorial
  - name: Load Balancer in Go
    url: https://kasvith.me/posts/lets-create-a-simple-lb-go/
    type: tutorial
  - name: RFC 7230 - HTTP/1.1 Message Syntax (Hop-by-hop headers)""
    url: https://datatracker.ietf.org/doc/html/rfc7230#section-6.1
    type: specification
prerequisites:
  - type: skill
    name: HTTP protocol
  - type: skill
    name: TCP networking
  - type: skill
    name: Concurrency basics
milestones:
  - id: load-balancer-basic-m1
    name: HTTP Reverse Proxy with Connection Pooling
    description: >-
      Build an HTTP reverse proxy that forwards requests to a single backend
      server using a connection pool. Properly handle headers including
      stripping hop-by-hop headers and adding X-Forwarded-* headers.
    acceptance_criteria:
      - Incoming HTTP requests on the proxy port are parsed and forwarded to a configured backend preserving method, path, query string, and body
      - "Hop-by-hop headers (Connection, Keep-Alive, Transfer-Encoding, Upgrade, Proxy-Authorization) are stripped before forwarding"
      - X-Forwarded-For, X-Forwarded-Proto, and X-Forwarded-Host headers are added to proxied requests
      - Backend response status code, headers, and body are returned to the originating client
      - Connection errors to the backend return 502 Bad Gateway with a descriptive JSON error body
      - "A connection pool of configurable size (default 10) is maintained per backend to reuse TCP connections"
      - Proxied requests are logged with: timestamp, client IP, method, path, backend address, response status, and latency in milliseconds
    pitfalls:
      - Forwarding hop-by-hop headers to the backend: this is an HTTP protocol violation that causes Connection header confusion
      - Opening a new TCP connection per request: without connection pooling, ephemeral port exhaustion occurs under load (~64K ports)
      - Not handling chunked transfer-encoding: if the backend sends chunked responses, the proxy must forward them correctly
      - Blocking on slow backend responses without a timeout: set a backend response timeout (default 30s) to avoid hanging the proxy
    concepts:
      - Reverse proxy mechanics
      - HTTP header classification (end-to-end vs hop-by-hop)
      - Connection pooling and reuse
      - Upstream timeout management
    skills:
      - HTTP protocol handling
      - TCP connection pool implementation
      - Header manipulation
      - Error response generation
    deliverables:
      - HTTP proxy server accepting client requests and forwarding to a single backend
      - Connection pool manager maintaining persistent connections to the backend
      - Header processing module (strip hop-by-hop, add X-Forwarded-*)
      - Structured request logging with latency measurement
    estimated_hours: "4-5"

  - id: load-balancer-basic-m2
    name: Round-Robin Distribution
    description: >-
      Distribute requests across multiple backend servers using a round-robin
      algorithm. Handle dynamic backend list configuration and skip unhealthy
      backends.
    acceptance_criteria:
      - "Backend server list is configurable via a JSON/YAML config file loaded at startup"
      - Round-robin algorithm cycles through backends sequentially, wrapping from last to first
      - Distribution is verified as even: over 1000 requests to N backends, each backend receives 1000/N ± 5% requests
      - "Counter increment is thread-safe using atomic operations (verified by 50+ concurrent request test with no race detector warnings)"
      - Division by zero is prevented: if the healthy backend list is empty, the proxy returns 503 Service Unavailable
      - Backends can be added or removed via a config reload endpoint (POST /admin/reload) without restarting the proxy
    pitfalls:
      - Race condition on the round-robin counter: use atomic increment, not lock-read-increment-unlock
      - Modulo with zero backends: if all backends are removed or unhealthy, the modulo operation panics
      - Not resetting or adjusting the counter when backends change: can cause temporary uneven distribution
      - Config reload not being atomic: partially loaded config can cause routing to nonexistent backends
    concepts:
      - Round-robin algorithm
      - Atomic operations for shared counters
      - Graceful degradation (503 when no backends)
      - Hot configuration reload
    skills:
      - Atomic/lock-free counter implementation
      - Configuration file parsing and reloading
      - Thread-safe backend list management
      - Graceful error handling
    deliverables:
      - Round-robin selector with atomic counter
      - Backend pool manager with add/remove operations
      - Configuration file parser and hot-reload endpoint
      - 503 fallback when no healthy backends are available
    estimated_hours: "3-4"

  - id: load-balancer-basic-m3
    name: Active Health Checks
    description: >-
      Implement active health checking that periodically probes backends
      and removes unhealthy ones from rotation. Include recovery detection
      to restore backends after they come back.
    acceptance_criteria:
      - "Health checks send HTTP GET to each backend's configurable health path (default /) at a configurable interval (default 10s)"
      - Backends failing N consecutive checks (configurable, default 3) are marked unhealthy and removed from rotation
      - Backends passing M consecutive checks (configurable, default 2) are restored to healthy and re-added to rotation
      - Health check timeout (default 5s), interval, healthy threshold, and unhealthy threshold are all configurable per backend
      - "When all backends are unhealthy, the proxy returns 503 Service Unavailable (not a panic or hang)"
      - "GET /admin/health returns JSON showing each backend's status, last check time, and consecutive pass/fail count"
      - Health checks are jittered (± 20% of interval) to avoid synchronized probing bursts
    pitfalls:
      - Thundering herd on recovery: when a backend recovers, all queued/waiting requests flood it simultaneously
      - Health checks overwhelming small backends: if the check interval is 1s and there are 100 backends, that is 100 checks/second from one proxy
      - No jitter: all health checks firing at exactly the same time creates periodic CPU/network spikes
      - Not handling the all-backends-down case: many implementations crash with division-by-zero or null pointer
    concepts:
      - Active health checking with failure thresholds
      - Recovery detection with consecutive success counting
      - Jitter for distributed probe timing
      - Graceful degradation under total backend failure
    skills:
      - Background task scheduling with jitter
      - State machine for backend health status
      - Concurrent health check execution
      - Admin API implementation
    deliverables:
      - Health check prober with configurable path, interval, and timeout
      - Backend health state machine (healthy/unhealthy) with threshold transitions
      - Recovery detector restoring backends after consecutive successes
      - Admin health status API endpoint
    estimated_hours: "4-5"

  - id: load-balancer-basic-m4
    name: Additional Algorithms
    description: >-
      Implement weighted round-robin and least-connections algorithms.
      Address sticky sessions using cookie-based affinity (not IP hash,
      which fails behind NAT/proxies).
    acceptance_criteria:
      - Weighted round-robin distributes requests proportionally to integer weights: a backend with weight 3 receives 3x the requests of weight 1 (verified over 1000 requests, ± 10%)
      - "Least-connections routes each request to the backend with the fewest active (in-flight) connections, with active count decremented when the backend response completes"
      - Cookie-based sticky sessions: proxy sets a session cookie (e.g., X-LB-Backend) on first request; subsequent requests with that cookie route to the same backend
      - "If the sticky backend is unhealthy, the request is routed to a healthy backend and a new cookie is set"
      - "Active algorithm is selected via configuration; changing the algorithm via POST /admin/algorithm takes effect for new requests without restart"
    pitfalls:
      - IP hash fails behind NAT/corporate proxies: thousands of users share one IP, creating massive imbalance. Use cookie-based affinity instead.
      - Least-connections not decrementing on response completion: active count grows monotonically, defeating the algorithm
      - "Weighted round-robin with large weights causing integer overflow in the accumulated weight counter"
      - Sticky session cookie not having an expiry: sessions stick forever even if the backend pool changes
    concepts:
      - Weighted round-robin with weight accumulation
      - Least-connections tracking
      - Cookie-based session affinity (vs flawed IP hash)
      - Algorithm hot-swapping
    skills:
      - Weight-based distribution algorithms
      - Active connection tracking with atomic counters
      - HTTP cookie management
      - Runtime configuration changes
    deliverables:
      - Weighted round-robin algorithm with configurable per-backend weights
      - Least-connections algorithm with atomic active connection tracking
      - Cookie-based sticky session implementation with health-aware fallback
      - Algorithm selection API with runtime switching
    estimated_hours: "5-6"
```
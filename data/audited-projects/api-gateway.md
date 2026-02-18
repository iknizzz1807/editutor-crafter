# AUDIT & FIX: api-gateway

## CRITIQUE
- CRITICAL: The audit correctly identifies that distributed rate limiting (explicitly mentioned in the essence as 'distributed rate limiting via token bucket algorithms') is completely absent from the milestones. M3 has 'Rate limiting per client' in deliverables but no dedicated AC or milestone for the algorithm implementation.
- M1 AC includes 'Circuit breaker opens after repeated failures' which is a complex pattern that deserves its own deliverable focus, but it's buried in M1 alongside basic routing. Circuit breaker should be its own focused deliverable.
- M1 AC mentions 'Health checks detect unhealthy backends' but doesn't specify the health check mechanism (active polling vs passive failure counting) or measurable thresholds.
- M2 AC 'Request aggregation combines responses from multiple backend calls into a single client response' is an advanced pattern that significantly increases scope. No AC specifies timeout handling or partial failure behavior for aggregation.
- M2 AC 'Request body transformation rewrites JSON payloads according to configurable mapping rules' — no specification of the mapping rule format.
- M3 AC mixes authentication methods (JWT, API key, OAuth2) without clear precedence rules. No AC specifies what happens when multiple auth methods are present.
- M3 mentions 'Auth caching stores validated tokens' but the pitfall says 'Caching tokens too long misses revocations.' No AC specifies cache TTL or revocation check mechanism.
- M4 AC 'Plugin architecture allows loading custom middleware modules without modifying the gateway core code' — this is a massive feature. No AC specifies the plugin interface contract, error isolation, or performance constraints.
- M4 AC 'Dynamic configuration reload applies routing and plugin changes without restarting' — no measurable criterion for zero-downtime reload (e.g., no dropped connections during reload).
- No mention of connection pooling to backends anywhere in AC, despite it being mentioned in learning outcomes.
- No mention of HTTP/2 support despite it being listed in skills.
- The estimated hours (50-70) seem tight given the breadth: routing + load balancing + circuit breaker + transformation + aggregation + 3 auth methods + observability + plugins + hot reload.

## FIXED YAML
```yaml
id: api-gateway
name: API Gateway
description: >-
  Build an API gateway with reverse proxy routing, distributed rate limiting,
  authentication, request transformation, and observability.
difficulty: advanced
estimated_hours: 65
essence: >-
  HTTP reverse proxy architecture implementing request routing with load
  balancing, distributed rate limiting via token bucket algorithm with Redis
  backend, middleware-based authentication pipelines with JWT validation, and
  circuit breaker patterns for backend failure isolation — all with structured
  observability instrumentation.
why_important: >-
  Building this teaches critical distributed systems patterns used in
  production infrastructure at scale, including request multiplexing, circuit
  breaking, rate limiting algorithms, and observability instrumentation that
  form the backbone of modern microservices architectures.
learning_outcomes:
  - Implement reverse proxy routing with load balancing across backend instances
  - Design token bucket rate limiting algorithm with distributed state in Redis
  - Build authentication middleware with JWT validation and API key support
  - Create HTTP request/response transformation pipelines
  - Implement circuit breaker pattern for backend failure isolation
  - Build observability layer with structured logging, metrics, and distributed tracing propagation
  - Handle connection pooling for efficient backend communication
skills:
  - Reverse Proxy Design
  - Rate Limiting Algorithms
  - JWT Authentication
  - Circuit Breaker Pattern
  - Distributed Tracing
  - Middleware Architecture
  - Connection Pool Management
tags:
  - advanced
  - api
  - authentication
  - rate-limiting
  - routing
architecture_doc: architecture-docs/api-gateway/index.md
languages:
  recommended:
    - Go
    - Rust
  also_possible:
    - Node.js
    - Java
resources:
  - name: Kong Gateway
    url: https://docs.konghq.com/
    type: reference
  - name: NGINX as API Gateway
    url: https://www.nginx.com/blog/deploying-nginx-plus-as-an-api-gateway/
    type: article
  - name: Token Bucket Algorithm
    url: https://en.wikipedia.org/wiki/Token_bucket
    type: reference
prerequisites:
  - type: skill
    name: REST APIs
  - type: skill
    name: TCP/HTTP networking
  - type: skill
    name: Load balancing concepts
milestones:
  - id: api-gateway-m1
    name: Reverse Proxy & Routing
    description: >-
      Route incoming HTTP requests to appropriate backend services with load
      balancing, health checks, and connection pooling.
    estimated_hours: 15
    concepts:
      - Reverse proxy: accept client request, forward to backend, return response
      - Load balancing algorithms: round-robin, weighted round-robin, least connections
      - Active health checks: periodic HTTP requests to backend health endpoint
      - Connection pooling to backends for reduced TCP handshake overhead
      - "X-Forwarded-For, X-Real-IP, and Via header propagation"
    skills:
      - HTTP protocol implementation
      - Load balancing algorithms
      - Service health monitoring
      - Connection pool management
    acceptance_criteria:
      - Path-based routing maps URL prefixes to backend services (e.g., /api/users/* -> user-service: 8080) configured via YAML or JSON config file
      - "Host-based routing directs requests to different backends based on the Host header (e.g., api.example.com -> service-a, admin.example.com -> service-b)"
      - "Load balancing distributes requests across multiple backend instances using configurable strategy (round-robin by default); measured distribution is within 10% of expected ratio across 1000 requests"
      - "Active health checks poll each backend's /health endpoint every 10 seconds (configurable); unhealthy backends (3 consecutive failures) are removed from the pool and re-added after 2 consecutive successes"
      - "X-Forwarded-For header is appended with the client's IP; X-Forwarded-Proto is set based on the incoming connection scheme"
      - "Connection pooling maintains persistent connections to each backend with configurable pool size (default 100 per backend); idle connections are closed after configurable timeout"
      - "Backend timeout is configurable per route (default 30s); requests exceeding timeout return 504 Gateway Timeout"
    pitfalls:
      - Not forwarding original client IP in X-Forwarded-For header
      - Health check requests blocking the request handling goroutine/thread
      - Memory leaks from unclosed backend connections on error paths
      - Thundering herd when all requests rush to a backend that just recovered
      - Not handling backend connection refused (ECONNREFUSED) gracefully
    deliverables:
      - Route configuration loader (YAML/JSON) mapping path/host patterns to backends
      - Path-based and host-based request routing engine
      - Round-robin and weighted load balancer implementations
      - Active health check poller with configurable interval and threshold
      - Connection pool manager with idle timeout and max size
      - Request forwarding with header propagation (X-Forwarded-For, X-Forwarded-Proto)

  - id: api-gateway-m2
    name: Rate Limiting & Circuit Breaker
    description: >-
      Implement distributed rate limiting using the token bucket algorithm with
      Redis backend, and circuit breaker pattern for backend failure isolation.
    estimated_hours: 15
    concepts:
      - Token bucket algorithm: tokens replenish at a fixed rate, each request consumes one token
      - Distributed rate limiting: store bucket state in Redis using atomic Lua scripts
      - Sliding window rate limiting as alternative: count requests in time windows
      - Circuit breaker states: closed (normal), open (failing), half-open (testing recovery)
      - Rate limit scoping by client IP, API key, or JWT subject
    skills:
      - Rate limiting algorithm implementation
      - Redis Lua scripting for atomic operations
      - Circuit breaker state machine
      - Distributed state management
    acceptance_criteria:
      - "Token bucket rate limiter enforces configurable requests-per-second per client (identified by IP, API key, or JWT subject); bucket state is stored in Redis using an atomic Lua script that checks and decrements in a single round-trip"
      - Rate limit headers are returned on every response: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset (Unix timestamp); 429 Too Many Requests is returned when limit is exceeded
      - Rate limit configuration is per-route: different routes can have different limits (e.g., /api/search -> 10/s, /api/posts -> 100/s)
      - "Circuit breaker opens after N consecutive failures (configurable, default 5) to a backend; while open, requests immediately return 503 Service Unavailable without contacting the backend"
      - "After a configurable cooldown (default 30s), circuit breaker transitions to half-open and allows one test request; if it succeeds, circuit closes; if it fails, circuit re-opens"
      - "Rate limiting works correctly across multiple gateway instances sharing the same Redis backend (verified by running 2 gateway instances and confirming aggregate rate does not exceed the limit)"
    pitfalls:
      - Non-atomic Redis check-then-decrement: race condition allows burst above limit
      - "Not using Lua script or Redis transaction for atomic token bucket operation"
      - Circuit breaker not resetting failure count on successful requests when closed
      - Rate limit key with high cardinality (per-path + per-user) exhausting Redis memory
      - Clock skew between gateway instances affecting sliding window accuracy
    deliverables:
      - Token bucket rate limiter with Redis Lua script for atomic check-and-decrement
      - Per-route rate limit configuration with client identification strategy
      - Rate limit response headers (X-RateLimit-Limit, Remaining, Reset)
      - Circuit breaker state machine (closed, open, half-open) with configurable thresholds
      - Circuit breaker metrics (state transitions, failure counts)
      - Sliding window rate limiter as alternative implementation

  - id: api-gateway-m3
    name: Authentication & Request Transformation
    description: >-
      Implement centralized authentication (JWT, API key) and request/response
      transformation middleware.
    estimated_hours: 15
    concepts:
      - JWT validation: signature verification (HS256/RS256), expiration, issuer, audience checks
      - API key lookup and validation from header or query parameter
      - Authentication result propagation: pass user context to backends via headers
      - Request/response transformation: header manipulation, body rewriting, URL rewriting
    skills:
      - JWT token validation
      - API key authentication
      - HTTP header manipulation
      - JSON body transformation
    acceptance_criteria:
      - "JWT validation middleware verifies HS256 or RS256 signatures, rejects expired tokens (with configurable clock skew tolerance of 30s), and validates issuer and audience claims against per-route configuration"
      - "API key authentication accepts keys from X-API-Key header or api_key query parameter; keys are validated against a store (Redis or database); invalid or missing keys return 401 with no information leakage about which keys exist"
      - "Per-route auth configuration specifies which auth method(s) are required (JWT, API key, or none); routes can require specific JWT scopes or roles"
      - "Authenticated user context (user ID, roles, scopes) is passed to backend services via X-User-ID, X-User-Roles, X-User-Scopes headers; these headers are stripped from incoming client requests to prevent spoofing"
      - "Header manipulation middleware adds, removes, or modifies headers on requests and responses per configurable rules"
      - "URL rewriting modifies the request path before forwarding (e.g., strip /api/v1 prefix); query parameters can be added or removed"
      - "Request and response body transformation rewrites JSON payloads according to configurable field mapping rules (rename, remove, add fields)"
      - "Auth validation results are cached in-memory with configurable TTL (default 60s) per token/key to avoid repeated validation overhead; cache is bounded to prevent memory exhaustion"
    pitfalls:
      - Not validating JWT audience and issuer allows tokens from other services
      - API keys appearing in access logs or error messages
      - Not stripping authentication headers from client requests before backend forwarding (header spoofing)
      - Caching auth results too long misses token revocation
      - Content-Length mismatch after body transformation causing truncated responses
      - Large request body buffering exhausting memory (should stream or enforce size limit)
    deliverables:
      - JWT validation middleware (signature, expiration, issuer, audience, scopes)
      - API key authentication middleware with secure key lookup
      - Per-route authentication configuration
      - User context header injection (X-User-ID, X-User-Roles) with anti-spoofing
      - Header manipulation middleware (add, remove, modify)
      - URL rewriting middleware
      - JSON body transformation pipeline with field mapping rules
      - Auth result cache with bounded size and configurable TTL

  - id: api-gateway-m4
    name: Observability & Plugin System
    description: >-
      Add structured logging, Prometheus metrics, distributed tracing
      propagation, and a plugin architecture for extensibility.
    estimated_hours: 20
    concepts:
      - Structured logging: JSON format with request ID, method, path, status, latency
      - Prometheus metrics: request count, latency histograms, error rates by route
      - Distributed tracing: propagate W3C traceparent/tracestate headers
      - Plugin architecture: middleware chain with defined interface contract
      - Configuration hot-reload: apply config changes without dropping in-flight requests
    skills:
      - Metrics collection and exposition
      - Distributed tracing propagation
      - Structured logging best practices
      - Plugin system design
      - Graceful configuration reload
    acceptance_criteria:
      - Structured JSON access logs include: request_id (generated UUID), method, path, query, status_code, response_time_ms, client_ip, user_agent, upstream_service, and upstream_response_time_ms for every proxied request
      - Prometheus-compatible /metrics endpoint exposes: gateway_requests_total (counter by route, method, status), gateway_request_duration_seconds (histogram by route with 0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10 second buckets), gateway_active_connections (gauge)
      - Distributed tracing: if incoming request has W3C traceparent header, it is propagated to backend; if absent, a new trace ID is generated and traceparent is injected
      - Plugin interface defines: OnRequest(req) -> (req, error), OnResponse(req, resp) -> (resp, error), and Name() -> string methods; plugins are loaded from configuration and executed in defined order
      - Plugin errors are isolated: a panicking or erroring plugin does not crash the gateway; the error is logged and the request continues through the remaining middleware chain
      - "Configuration reload on SIGHUP applies new routes, rate limits, and plugin configuration without dropping in-flight requests; new requests use updated config while in-flight requests complete with old config"
      - "Sensitive data (Authorization headers, API keys, request bodies) is NOT included in access logs by default; a configurable redaction list controls which headers are masked"
    pitfalls:
      - High cardinality labels in Prometheus metrics (e.g., per-path with path parameters) causing memory explosion
      - Logging sensitive data (Authorization header, API keys, passwords) in access logs
      - Trace sampling too aggressive loses important error traces
      - Plugin panic crashes entire gateway if not recovered
      - Configuration reload race condition causing inconsistent state between routes and auth config
    deliverables:
      - Structured JSON access log middleware with configurable field selection
      - Prometheus metrics endpoint with request count, latency histogram, and active connections
      - W3C Trace Context header propagation middleware
      - Plugin interface definition with OnRequest and OnResponse hooks
      - Plugin loader reading plugin chain from configuration
      - Plugin error isolation with panic recovery
      - SIGHUP configuration hot-reload with graceful transition
      - Sensitive header redaction in logs
```
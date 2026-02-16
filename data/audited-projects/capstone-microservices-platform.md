# AUDIT & FIX: capstone-microservices-platform

## CRITIQUE
- **REST-to-gRPC Transcoding Missing Protobuf Mapping**: The AC requires 'Request/response transformation (REST → gRPC)' but never mentions the protobuf-to-JSON mapping definitions or gRPC-JSON transcoding annotations required to make this work. Without `.proto` HTTP annotations or a manual mapping layer, this transformation is undefined.
- **Canary Rollback Requires Metrics Integration Not Detailed**: Milestone 5 requires 'Automatic rollback triggered by error rate spike in canary' but the metrics pipeline for detecting this spike in real-time and triggering the rollback is not detailed in either M4 (Observability) or M5 (CI/CD). This is a non-trivial control loop.
- **Database Migration Not Addressed**: M5 pitfall mentions 'Database migration coordination across services' but no milestone addresses schema evolution or backward-compatible migrations. For independent deployability, this is critical.
- **Saga Compensation Failure Not Addressed**: M3 pitfall mentions 'Compensation failure leaving inconsistent state' but no AC requires handling this case. What happens when the compensating action itself fails? Dead letter queues? Manual intervention? This is the hardest part of sagas.
- **Service Mesh Mentioned in Skills but Never Built**: 'Service Mesh' is listed as a skill but no milestone implements sidecar proxies, mTLS, or service mesh concepts.
- **4 Services May Be Too Many for Scope**: Building 4 real microservices with separate databases, gRPC APIs, health checks, plus all the infrastructure (gateway, saga, observability, CI/CD) in 100-150 hours is extremely ambitious. Two services would demonstrate the same patterns.
- **No Data Store Milestone**: Database-per-service pattern is required but there's no milestone for setting up databases, defining schemas, or implementing data access layers.
- **Observability Milestone Missing Alerting AC**: The AC mentions metrics and dashboards but 'alerting' (mentioned in learning outcomes) has no specific AC.

## FIXED YAML
```yaml
id: capstone-microservices-platform
name: "Capstone: Production Microservices Platform"
description: >
  Build a complete microservices platform with API gateway, service discovery,
  circuit breaking, distributed tracing, saga-based transactions, and CI/CD—
  integrated into a working e-commerce system with 4 services.
difficulty: expert
estimated_hours: "110-160"
essence: >
  Orchestrating independent services through dynamic discovery, API gateway
  routing with protocol transcoding, distributed tracing correlation, circuit
  breaking with rate limiting, saga-based transaction compensation with
  idempotency, and automated deployment pipelines—while maintaining
  system-wide observability, resilience, and independent deployability.
why_important: >
  This is how production systems are actually built at scale. Integrating all
  the patterns you learned individually into a working platform teaches you
  the hardest part of distributed systems: making everything work together
  reliably under failure conditions.
learning_outcomes:
  - Decompose a domain into bounded-context microservices with database-per-service
  - Integrate API gateway with service discovery for dynamic routing and protocol transcoding
  - Implement end-to-end distributed tracing with W3C Trace Context across all services
  - Apply circuit breaker and rate limiting patterns at the gateway layer
  - Build saga-based distributed transactions with compensation and idempotency
  - Monitor system health with centralized logging, metrics, and alerting
  - Build CI/CD pipelines with blue-green deployment and metrics-driven canary releases
skills:
  - Microservices Architecture
  - API Gateway & Protocol Transcoding
  - Distributed Tracing
  - CI/CD & Canary Deployment
  - Observability (Logs, Metrics, Traces)
  - Saga Pattern & Compensation
  - Circuit Breaker & Rate Limiting
tags:
  - microservices
  - distributed-systems
  - capstone
  - production-ready
  - expert
architecture_doc: architecture-docs/capstone-microservices-platform/index.md
languages:
  recommended:
    - Go
    - Java
  also_possible:
    - Python
    - Rust
    - TypeScript
resources:
  - name: "Microservices Patterns"
    url: https://microservices.io/patterns/
    type: documentation
  - name: "Building Microservices (Sam Newman)"
    url: https://samnewman.io/books/building_microservices_2nd_edition/
    type: book
  - name: "The Twelve-Factor App"
    url: https://12factor.net/
    type: article
  - name: "gRPC-JSON Transcoding"
    url: https://cloud.google.com/endpoints/docs/grpc/transcoding
    type: documentation
prerequisites:
  - type: project
    id: api-gateway
    name: "API Gateway"
  - type: project
    id: service-discovery
    name: "Service Discovery"
  - type: project
    id: circuit-breaker
    name: "Circuit Breaker Pattern"
  - type: project
    id: rate-limiter
    name: "Rate Limiter"
  - type: project
    id: distributed-tracing
    name: "Distributed Tracing System"
  - type: project
    id: ci-cd-pipeline
    name: "CI/CD Pipeline Builder"
milestones:
  - id: capstone-microservices-platform-m1
    name: "Service Decomposition & Discovery"
    description: >
      Decompose an e-commerce domain into 4 services (Users, Products, Orders,
      Payments) with database-per-service, gRPC communication, and service
      discovery registration.
    acceptance_criteria:
      - "Four independent services each with a separate database (database-per-service pattern enforced)"
      - "gRPC service definitions (.proto files) are versioned and include HTTP annotation mappings for JSON transcoding"
      - "All services register with service discovery on startup with health metadata and deregister gracefully on shutdown"
      - "Inter-service communication uses gRPC with automatic service address lookup from the discovery registry"
      - "Health check endpoints expose /healthz (liveness) and /readyz (readiness) returning structured health status"
      - "Each service's bounded context is documented: owned entities, exposed APIs, and explicitly prohibited cross-service database access"
    pitfalls:
      - "Distributed monolith: services that must be deployed together due to tight coupling defeat the purpose"
      - "Service discovery stale entries from crashed services cause routing failures—TTL-based eviction needed"
      - "Shared database anti-pattern: services reading each other's tables creates hidden coupling"
      - "Proto file versioning: breaking changes in .proto files break all consumers—use additive-only changes"
    concepts:
      - Bounded contexts and domain-driven design
      - Database-per-service pattern
      - Service discovery with health checks
      - gRPC with HTTP transcoding annotations
    skills:
      - Service decomposition
      - gRPC and protobuf design
      - Service discovery integration
      - Database schema design
    deliverables:
      - "4 microservices with documented bounded contexts and data ownership"
      - "gRPC .proto files with versioned APIs and HTTP annotation mappings for transcoding"
      - "Service registry integration with heartbeat-based health checking and automatic deregistration"
      - "Database setup per service with migration scripts"
    estimated_hours: "22-30"

  - id: capstone-microservices-platform-m2
    name: "API Gateway & Resilience"
    description: >
      Build the API gateway that routes external HTTP requests to internal gRPC
      services via discovery, with REST-to-gRPC transcoding, rate limiting,
      and circuit breakers.
    acceptance_criteria:
      - "Gateway routes HTTP requests to gRPC services using discovery lookup for service address resolution"
      - "REST-to-gRPC transcoding converts JSON request bodies to protobuf messages using .proto HTTP annotations or explicit mapping definitions"
      - "Per-client rate limiting enforces configurable tiers (e.g., free: 100 req/min, pro: 1000 req/min) identified by API key"
      - "Circuit breaker per downstream service transitions through closed → open → half-open states with configurable failure threshold (default 50% over 10-second window)"
      - "Gateway returns appropriate HTTP status codes: 429 for rate limit, 503 for open circuit, 504 for timeout"
      - "Gateway does not become a single point of failure: multiple gateway instances can run behind a load balancer"
    pitfalls:
      - "Gateway becoming a performance bottleneck—must be stateless and horizontally scalable"
      - "Circuit breaker masking intermittent issues—log every circuit state transition with reason"
      - "Rate limit bypass via distributed clients using different API keys—enforce per-account limits"
      - "REST-to-gRPC transcoding silently drops unknown JSON fields—must validate against proto schema"
    concepts:
      - API gateway pattern
      - Protocol transcoding (REST ↔ gRPC)
      - Rate limiting tiers
      - Circuit breaker state machine
    skills:
      - HTTP/gRPC proxy implementation
      - Rate limiting algorithms
      - Circuit breaker implementation
      - Protobuf-JSON mapping
    deliverables:
      - "API gateway with HTTP-to-gRPC routing via discovery lookup"
      - "REST-to-gRPC transcoding layer with proto-based JSON mapping"
      - "Per-client tiered rate limiter with API key identification"
      - "Per-service circuit breaker with configurable thresholds and state transition logging"
    estimated_hours: "20-28"

  - id: capstone-microservices-platform-m3
    name: "Distributed Transactions & Saga"
    description: >
      Implement the order creation flow as an orchestrated saga spanning
      Orders, Products (inventory), and Payments services, with compensation
      on failure and idempotency for retry safety.
    acceptance_criteria:
      - "Saga orchestrator coordinates: create order → reserve inventory → process payment → confirm order"
      - "Payment failure triggers compensating actions: release inventory reservation, cancel order—in reverse order"
      - "Saga state (current step, completed steps, compensation status) is persisted to survive orchestrator crashes"
      - "Idempotency keys (request IDs) on all saga step operations prevent duplicate charges or double reservations on retry"
      - "Compensation handler failure is retried with exponential backoff; after max retries, the saga is moved to a dead letter queue for manual resolution"
      - "Concurrent sagas for the same inventory item are serialized or use optimistic locking to prevent overselling"
    pitfalls:
      - "Compensation failure leaving permanently inconsistent state—must have dead letter queue and alerting"
      - "Race condition between saga steps: inventory reserved but payment processed against stale balance"
      - "Long-running sagas holding inventory reservations block other orders—use reservation TTL with automatic release"
      - "Saga orchestrator is a single point of failure if not designed for crash recovery"
    concepts:
      - Orchestrated saga pattern
      - Compensating transactions
      - Idempotency keys
      - Dead letter queues
      - Eventual consistency
    skills:
      - Saga orchestrator design
      - Compensation handler implementation
      - Idempotency enforcement
      - Crash recovery for stateful workflows
    deliverables:
      - "Saga orchestrator with persistent step tracking and crash recovery"
      - "Compensating action handlers for inventory release and payment reversal"
      - "Idempotency layer using request IDs to prevent duplicate operations"
      - "Dead letter queue for sagas with irrecoverable compensation failures"
      - "Reservation TTL with automatic release for long-running sagas"
    estimated_hours: "22-30"

  - id: capstone-microservices-platform-m4
    name: "Observability Stack"
    description: >
      Integrate distributed tracing (W3C Trace Context), centralized logging
      with trace correlation, metrics collection, and alerting across all services.
    acceptance_criteria:
      - "Every request gets a W3C Trace Context trace ID propagated across all gRPC service calls as metadata"
      - "Each service emits spans with parent-child relationships; the full trace shows the request's path through all services"
      - "Centralized log aggregation collects logs from all services with trace ID correlation for cross-service debugging"
      - "RED metrics (Rate, Errors, Duration) are collected per service with p50, p95, and p99 latency percentiles"
      - "Alert rules trigger notifications when error rate exceeds threshold (e.g., >5% over 5 minutes) or p99 latency exceeds SLO"
      - "Dashboard displays service dependency map, request flow traces, per-service health, and active alerts"
    pitfalls:
      - "Trace context lost in async operations (message queues, background jobs)—must propagate via message headers"
      - "Log volume explosion under high traffic—implement sampling for traces (e.g., 10% of requests)"
      - "Dashboard too slow with high-cardinality metrics—pre-aggregate at collection time"
      - "Alert fatigue from noisy thresholds—tune alert sensitivity with historical baseline data"
    concepts:
      - W3C Trace Context standard
      - Distributed tracing with spans
      - RED metrics (Rate, Errors, Duration)
      - Log correlation with trace IDs
      - Alerting rules and SLOs
    skills:
      - Distributed tracing instrumentation
      - Log aggregation and correlation
      - Metrics collection and percentile computation
      - Alert rule configuration
      - Dashboard construction
    deliverables:
      - "W3C Trace Context propagation across all service boundaries with span creation"
      - "Centralized logging with trace ID correlation linking logs to distributed traces"
      - "RED metrics collection with latency percentiles per service"
      - "Alert rules for error rate and latency SLO violations with notification endpoints"
      - "Service dependency map visualization showing call graphs and latency between services"
    estimated_hours: "20-25"

  - id: capstone-microservices-platform-m5
    name: "CI/CD, Blue-Green & Canary Deployment"
    description: >
      Build independent CI/CD pipelines per service with blue-green deployment,
      canary releases with metrics-driven automatic rollback, and database
      migration coordination.
    acceptance_criteria:
      - "Each service has an independent CI pipeline: lint → test → build container image → push to registry"
      - "Blue-green deployment runs two environments; traffic is switched atomically after health check validation of the new environment"
      - "Canary release controller routes a configurable percentage (starting at 5%) of traffic to the new version"
      - "Canary promotion: if error rate and p99 latency remain within SLO bounds for a configurable observation window (default 10 minutes), traffic percentage increases automatically to 100%"
      - "Automatic rollback: if canary error rate exceeds 2x the baseline or p99 exceeds SLO, traffic is immediately shifted back to 100% stable version"
      - "Rollback mechanism uses the observability stack's metrics API (from M4) as the data source for promotion/rollback decisions"
      - "Database migrations are backward-compatible (expand-contract pattern): new code works with old schema and old code works with new schema"
    pitfalls:
      - "Canary with incompatible API versions causes errors for routed clients—enforce backward compatibility"
      - "Rollback does not revert database schema changes—expand-contract pattern is mandatory"
      - "Metrics lag between canary deployment and error detection—observation window must exceed metrics pipeline latency"
      - "Blue-green requires 2x infrastructure during deployment—budget for it"
      - "CI pipeline without integration tests misses inter-service contract breakages"
    concepts:
      - Independent deployability
      - Blue-green deployment
      - Canary releases with metrics-driven promotion
      - Expand-contract database migration
      - Automatic rollback control loop
    skills:
      - CI/CD pipeline construction
      - Traffic shifting mechanisms
      - Metrics-driven deployment automation
      - Database migration strategies
    deliverables:
      - "Independent CI pipeline per service with container image publishing"
      - "Blue-green deployment controller with health-validated traffic switching"
      - "Canary release controller with gradual traffic shifting and observation windows"
      - "Metrics-driven rollback controller using observability API for error rate and latency monitoring"
      - "Database migration framework enforcing backward-compatible expand-contract pattern"
    estimated_hours: "22-30"
```
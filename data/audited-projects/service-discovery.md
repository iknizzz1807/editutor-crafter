# AUDIT & FIX: service-discovery

## CRITIQUE
- **Technical Inaccuracy (Confirmed):** TTL and Heartbeats are presented as separate mechanisms in Milestone 1's deliverables. In reality, heartbeats are the mechanism that refreshes a TTL lease. The TTL defines the expiry window; the heartbeat resets it. Conflating them as independent features is misleading.
- **Logical Gap (Confirmed):** No Watch/notification mechanism exists. Polling-only discovery is fundamentally inefficient. Production systems (Consul, etcd, ZooKeeper) all provide watch/event-stream capabilities. This is a critical missing milestone.
- **Logical Gap (Confirmed):** The bootstrapping problem is completely ignored. How does a service find the discovery service itself? Without addressing this (static config, DNS, environment variables), the system is incomplete.
- **Milestone Ordering Issue:** Milestone 3 (HTTP API) should be Milestone 1 or at least concurrent with it. You cannot register or look up services without an API surface. The current M1 talks about 'Registration API' in deliverables but M3 is where the API is actually defined. This is contradictory.
- **Difficulty Mismatch:** Tagged as 'beginner' but includes concepts like consistent hashing, optimistic locking, watch mechanisms, and network partition handling — these are solidly intermediate topics.
- **Acceptance Criteria Weakness:** M1 AC says 'List all healthy instances' but health checking isn't introduced until M2. This creates a forward dependency that breaks sequential milestone progression.
- **Pitfall Weakness:** Pitfalls are shallow one-liners ('Race conditions', 'Error handling') with no actionable insight. They should describe *what* races and *why* they matter.
- **Missing Security:** No mention of authentication/authorization for the registry API. Any service can register or deregister anything.

## FIXED YAML
```yaml
id: service-discovery
name: Service Discovery
description: >-
  Build a service discovery system with dynamic registration, heartbeat-based
  TTL leases, HTTP API, health monitoring, watch notifications, and client
  bootstrapping.
difficulty: intermediate
estimated_hours: "15-20"
essence: >-
  Dynamic service registration and lookup using an in-memory registry with
  heartbeat-refreshed TTL leases, HTTP API for CRUD operations, active health
  monitoring, and push-based watch notifications to maintain an
  eventually-consistent directory of available service instances.
why_important: >-
  Building this teaches you the foundational infrastructure pattern behind
  production microservices platforms like Kubernetes and Consul, giving you
  practical experience with distributed systems challenges like failure
  detection, TTL-based lease management, event-driven notifications, and
  bootstrapping problems that are critical for backend and infrastructure
  engineering roles.
learning_outcomes:
  - Implement service registration with heartbeat-refreshed TTL leases
  - Design an HTTP REST API for registry CRUD operations
  - Build active health check mechanisms with configurable intervals and failure thresholds
  - Handle concurrent registry updates using proper synchronization
  - Implement watch/event-stream mechanisms for real-time discovery updates
  - Address the bootstrapping problem of how clients find the discovery service itself
  - Debug stale entry cleanup and TTL expiration edge cases
  - Test failure scenarios including service crashes and network timeouts
skills:
  - REST API design
  - Health check patterns
  - Service registry architecture
  - Concurrent data structures
  - Heartbeat-based TTL lease protocols
  - HTTP server implementation
  - Failure detection algorithms
  - Event streaming (SSE/long-poll)
tags:
  - intermediate
  - consul
  - dns-based
  - go
  - health-checks
  - javascript
  - python
  - registry
  - service
architecture_doc: architecture-docs/service-discovery/index.md
languages:
  recommended:
    - Go
    - Python
    - JavaScript
  also_possible:
    - Java
    - Rust
resources:
  - name: Service Discovery Patterns
    url: https://microservices.io/patterns/service-registry.html
    type: article
  - name: Consul Service Discovery - HashiCorp Docs
    url: https://developer.hashicorp.com/consul/docs/use-case/service-discovery
    type: documentation
  - name: Consul Service Discovery Beginners Guide
    url: https://devopscube.com/consul-service-discovery-beginners-guide/
    type: tutorial
prerequisites:
  - type: skill
    name: HTTP basics
  - type: skill
    name: Networking concepts
  - type: skill
    name: Concurrency basics
milestones:
  - id: service-discovery-m1
    name: HTTP API and In-Memory Registry
    description: >-
      Build the HTTP API surface and in-memory registry for service
      registration, deregistration, and lookup. Services register with a
      name, host, port, and metadata. Each registration returns a unique
      instance ID.
    acceptance_criteria:
      - "POST /services accepts {name, host, port, metadata} and returns a unique instance ID (UUID)"
      - "DELETE /services/{name}/{id} deregisters the specified instance and returns 204 on success or 404 if not found"
      - "GET /services/{name} returns a JSON list of all registered instances for that service name"
      - "GET /services returns a JSON object listing all registered service names and their instance counts"
      - Duplicate registration from the same host:port for the same service name is rejected with 409 Conflict
      - All registry mutations are thread-safe under concurrent requests verified by a load test of 100+ concurrent registrations
      - Input validation rejects missing required fields (name, host, port) with 400 Bad Request and descriptive error
    pitfalls:
      - "Not generating truly unique instance IDs: using sequential integers allows ID collisions across restarts"
      - "Race condition on concurrent register + deregister for the same instance: use a mutex or lock-free map"
      - "Forgetting to validate port ranges (1-65535) and hostname format, allowing garbage data in the registry"
    concepts:
      - Service registry pattern
      - Instance identity via UUIDs
      - RESTful CRUD operations
      - Thread-safe in-memory data structures
    skills:
      - RESTful endpoint design
      - HTTP request/response handling
      - JSON marshaling and unmarshaling
      - Concurrent map or dictionary usage
      - Input validation
    deliverables:
      - HTTP server with registration, deregistration, and lookup endpoints
      - In-memory registry data structure (concurrent map of service name to instance list)
      - Input validation middleware rejecting malformed requests
      - Unique instance ID generation using UUIDs
    estimated_hours: "3-4"

  - id: service-discovery-m2
    name: Heartbeat-Based TTL Leases
    description: >-
      Add TTL-based leases refreshed by heartbeats. Services must periodically
      send heartbeat requests to keep their registration alive. If a heartbeat
      is not received within the TTL window, the instance is automatically
      removed from the registry.
    acceptance_criteria:
      - Each registration accepts an optional TTL parameter (default 30s) defining the lease duration
      - "PUT /services/{name}/{id}/heartbeat refreshes the TTL lease for the specified instance, returning 200 or 404"
      - A background goroutine/thread runs every N seconds (configurable) and removes instances whose TTL has expired
      - Expired instances are logged with timestamp, service name, and instance ID before removal
      - "GET /services/{name} excludes expired instances (instances past TTL but not yet reaped are still excluded)"
      - TTL expiration is accurate to within 2x the background sweep interval
    pitfalls:
      - "Setting sweep interval too long causes stale instances to persist in lookup results for extended periods"
      - "Not using monotonic clocks for TTL calculation: wall-clock adjustments (NTP) can cause premature or delayed expiry"
      - "Memory growth if expired instances are only marked but never actually deleted from the data structure"
      - "Heartbeat endpoint not being idempotent: multiple rapid heartbeats should simply reset the timer, not create side effects"
    concepts:
      - TTL lease model
      - Heartbeats as TTL refresh mechanism (not a separate system)
      - Background reaper/sweeper pattern
      - Monotonic vs wall-clock time
    skills:
      - Background task scheduling
      - Time-based expiration logic
      - Monotonic clock usage
      - Lease management patterns
    deliverables:
      - Heartbeat endpoint that resets the TTL timer for a registered instance
      - Background sweeper task that removes expired instances periodically
      - Configurable TTL per instance and configurable sweep interval
      - Logging of all TTL expirations for observability
    estimated_hours: "3-4"

  - id: service-discovery-m3
    name: Active Health Checks
    description: >-
      Add active health checking where the registry probes registered
      instances at configurable intervals. Instances failing consecutive
      health checks are marked unhealthy and excluded from lookup results.
      This is complementary to TTL-based expiry.
    acceptance_criteria:
      - "Services may register a health check URL (e.g., http://host:port/health) at registration time"
      - Registry polls each instance's health endpoint at a configurable interval (default 10s)
      - Instances failing N consecutive checks (configurable, default 3) are marked unhealthy
      - "GET /services/{name} returns only healthy instances by default; a ?include_unhealthy=true query parameter returns all"
      - Unhealthy instances that pass M consecutive checks (configurable, default 2) are restored to healthy
      - Health check HTTP timeout is configurable (default 5s) and a timeout counts as a failure
      - "GET /services/{name}/{id}/health returns the current health status, last check time, and consecutive failure count"
    pitfalls:
      - "Health check timeout too short (< 1s) causes false positives during normal GC pauses or load spikes"
      - "Not distinguishing between 'no health check URL registered' (assume healthy) and 'health check failed' (mark unhealthy)"
      - "Overwhelming backend services with health checks: ensure check intervals are reasonable and jittered across instances"
      - "Network partition false positives: the registry loses connectivity to an instance but the instance is actually fine"
    concepts:
      - Active vs passive health checking
      - Failure threshold with consecutive failure counting
      - Health state machine (healthy → suspect → unhealthy → recovering → healthy)
      - Jitter in polling intervals
    skills:
      - HTTP client with timeout configuration
      - State machine implementation
      - Periodic polling with jitter
      - Concurrent health check execution
    deliverables:
      - Health check executor that probes instance health endpoints concurrently
      - Configurable check interval, timeout, healthy threshold, and unhealthy threshold
      - Health status API endpoint per instance
      - Jittered polling to avoid thundering herd on health check cycles
    estimated_hours: "3-4"

  - id: service-discovery-m4
    name: Watch Mechanism and Bootstrapping
    description: >-
      Implement a watch/event-stream mechanism so clients receive push
      notifications when services register, deregister, or change health
      status. Also address the bootstrapping problem: how clients find
      the discovery service itself.
    acceptance_criteria:
      - "GET /services/{name}/watch returns a Server-Sent Events (SSE) stream of registration/deregistration/health-change events"
      - Events include event type (register, deregister, health_change), instance details, and a monotonic event sequence number
      - Clients reconnecting with Last-Event-ID header receive all events since that sequence number (catch-up)
      - Discovery service address is configurable via environment variable, CLI flag, or static config file (bootstrapping)
      - Watch connections that are idle for >60s receive a keepalive comment to prevent proxy/firewall timeout
      - "At least 100 concurrent watch connections are supported without degrading registration API latency by more than 10%"
    pitfalls:
      - "Unbounded event history growth: implement a circular buffer or max-event-count with compaction"
      - "SSE reconnection thundering herd: clients should use exponential backoff with jitter on reconnect"
      - "Not sending SSE keepalive comments: intermediate proxies will close idle connections after 30-60s"
      - "Ignoring the bootstrap problem entirely: if the discovery service itself is unreachable, the whole system fails"
    concepts:
      - Server-Sent Events (SSE) protocol
      - Event sourcing for change notification
      - Bootstrapping and well-known endpoints
      - Connection keepalive
    skills:
      - SSE server implementation
      - Event buffering and replay
      - Connection lifecycle management
      - Configuration management for bootstrapping
    deliverables:
      - SSE endpoint streaming real-time registry change events
      - Event sequence numbering with catch-up replay on reconnect
      - Keepalive mechanism for long-lived SSE connections
      - Bootstrapping documentation and configuration for discovering the discovery service
    estimated_hours: "4-5"

```
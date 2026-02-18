# AUDIT & FIX: circuit-breaker

## CRITIQUE
- **Audit Finding Confirmed (Bulkhead Conflation):** M2 AC includes 'Bulkhead enforces concurrency limit per downstream service independently.' The bulkhead pattern is a distinct resilience pattern that limits concurrent access to a resource. While it complements circuit breakers, conflating them in the same milestone without clear separation is architecturally misleading. The AC should either be removed, clearly labeled as a bonus/companion pattern, or placed in a separate section.
- **M1 Thread Safety Vagueness:** AC says 'Thread-safe implementation handles concurrent request processing correctly' but provides no measurable criteria. How do you verify thread safety? This needs a concrete test: e.g., concurrent goroutines/threads exercising the circuit breaker with no data races detected.
- **M1 Missing Timeout AC:** The state machine requires a timeout for the open state before transitioning to half-open. This is mentioned in deliverables ('Configurable timeout duration for open state') but not in the ACs.
- **M2 Sliding Window Missing from M1:** The basic circuit breaker (M1) uses a 'configurable failure threshold count' (simple counter). M2 introduces the sliding window. But the transition isn't clear—does M2 replace M1's counter, or is it an alternative? This should be explicit.
- **M3 Dashboard AC Without Implementation Path:** M3 AC says 'Dashboard displays current circuit states for all registered services' but the project has no frontend milestone. A dashboard is a significant deliverable that needs more specification (CLI? HTTP endpoint? Prometheus metrics?).
- **Missing Concurrency Testing:** No AC in any milestone requires a load test or concurrent stress test to verify the circuit breaker behaves correctly under high concurrency.
- **Pitfall Quality:** M2 pitfall 'Fallback also failing (need fallback for fallback)' is flippant. The real pattern is to return a static/cached response as the ultimate fallback, not to chain fallbacks infinitely.

## FIXED YAML
```yaml
id: circuit-breaker
name: Circuit Breaker Pattern
description: >-
  Implement a production-quality circuit breaker library for resilient
  microservices communication with configurable failure detection, fallback
  strategies, and observability.
difficulty: intermediate
estimated_hours: "15-25"
essence: >-
  State machine managing failure thresholds and timeout windows to
  automatically block requests to failing services, preventing cascading
  failures across distributed systems through closed, open, and half-open
  state transitions with sliding window failure rate tracking.
why_important: >-
  Circuit breakers are critical for production microservices resilience,
  preventing cascading failures that can take down entire systems and
  teaching you how to build fault-tolerant distributed architectures.
learning_outcomes:
  - Implement state transitions between closed, open, and half-open circuit states
  - Design sliding window algorithms for tracking failure rates over time
  - Build timeout mechanisms and automatic recovery detection logic
  - Integrate circuit breakers transparently with HTTP and gRPC clients
  - Configure failure thresholds, timeout periods, and half-open probe limits
  - Implement fallback strategies for graceful degradation during outages
  - Test resilience patterns using failure injection under concurrent load
  - Expose circuit breaker metrics for observability and alerting
skills:
  - Fault Tolerance Patterns
  - State Machine Design
  - Failure Rate Tracking
  - Distributed Systems Resilience
  - Timeout Management
  - Chaos Engineering
  - Service Degradation
  - Observability Metrics
tags:
  - failure-detection
  - fallback
  - fault-tolerance
  - go
  - half-open
  - intermediate
  - java
  - resilience
architecture_doc: architecture-docs/circuit-breaker/index.md
languages:
  recommended:
    - Go
    - Java
  also_possible:
    - Python
    - TypeScript
resources:
  - name: Circuit Breaker Pattern - Martin Fowler
    url: https://martinfowler.com/bliki/CircuitBreaker.html
    type: article
  - name: Hystrix Wiki (Netflix)
    url: https://github.com/Netflix/Hystrix/wiki
    type: reference
  - name: Resilience4j (modern Hystrix alternative)""
    url: https://resilience4j.readme.io/docs/circuitbreaker
    type: documentation
prerequisites:
  - type: skill
    name: Concurrency primitives (mutexes, atomics)
  - type: skill
    name: HTTP client usage
  - type: skill
    name: Basic microservices concepts
milestones:
  - id: circuit-breaker-m1
    name: Core State Machine
    description: >-
      Implement the three-state circuit breaker (closed, open, half-open)
      with failure counting, configurable thresholds, and thread-safe
      concurrent access.
    acceptance_criteria:
      - "Closed state passes all requests through to the downstream function/service"
      - "Closed-to-open transition occurs when consecutive failure count reaches the configurable threshold"
      - "Open state rejects all requests immediately without calling downstream, returning a circuit-open error"
      - "Open-to-half-open transition occurs automatically after a configurable timeout duration elapses"
      - "Half-open state allows exactly one probe request through; success transitions to closed, failure transitions back to open"
      - "Failure count resets to zero on a successful request in the closed state"
      - Implementation is thread-safe: a race detector (e.g., Go's -race flag) reports zero data races under concurrent load of at least 100 goroutines/threads
      - "All thresholds (failure count, timeout duration, half-open probe limit) are configurable at construction time"
    pitfalls:
      - "Race conditions on state transitions without proper mutex or atomic operations"
      - "Not resetting the failure count on success in the closed state causes premature tripping"
      - "Half-open state allowing multiple concurrent probe requests instead of exactly one (or a configured limit)"
      - "Timer for open-to-half-open transition not being reset when the circuit re-opens from half-open"
      - "Using wall-clock time instead of monotonic clock for timeout measurement"
    concepts:
      - Finite state machine with three states
      - Failure counting and threshold detection
      - Timeout-based state transitions
      - Thread-safe shared mutable state
    skills:
      - State machine implementation
      - Concurrent programming with mutexes/atomics
      - Timeout handling with monotonic time
      - Configurable threshold design
    deliverables:
      - CircuitBreaker struct/class with closed, open, and half-open states
      - Execute/Call method wrapping downstream calls with circuit logic
      - Configurable failure threshold, open timeout, and half-open probe limit
      - Concurrent test demonstrating thread safety under load
    estimated_hours: "5-8"

  - id: circuit-breaker-m2
    name: Sliding Window, Fallbacks & Metrics
    description: >-
      Replace simple failure counting with a time-based sliding window,
      add fallback strategies for graceful degradation, and expose
      observability metrics.
    acceptance_criteria:
      - "Sliding window tracks failure rate over a configurable time period (e.g., last 60 seconds) using time-bucketed counters"
      - "Circuit opens when the failure rate exceeds a configurable percentage threshold (e.g., 50%) within the sliding window, with a minimum request volume requirement"
      - "Error classification distinguishes retriable errors (timeouts, 503s) from non-retriable errors (400s, 404s); only retriable errors count toward the failure rate"
      - "A configurable fallback function executes when the circuit is open, returning a degraded response instead of an error"
      - "If the fallback function itself fails, a static/cached default response is returned (never an unhandled error)"
      - Metrics are exposed via a method/endpoint: current state, failure count, success count, failure rate percentage, and time since last state transition
      - "State change events (closed→open, open→half-open, half-open→closed) emit observable notifications (callback, channel, or log)"
    pitfalls:
      - "Sliding window bucket rotation must handle the case where no requests arrive for multiple bucket periods"
      - "Fallback functions that make network calls can also fail; always have a terminal static fallback"
      - "Error classifier being too aggressive (classifying transient 503 as permanent) causes unnecessary circuit opens"
      - "Minimum request volume prevents the circuit from opening on a single failure in a low-traffic window"
      - "Metrics collection adding overhead to every request; use atomic counters, not mutexes, for hot-path metrics"
    concepts:
      - Time-bucketed sliding window for rate calculation
      - Error classification and categorization
      - Fallback chain pattern
      - Observable state transitions
      - Minimum volume threshold
    skills:
      - Time-series data structures
      - Graceful degradation patterns
      - Error categorization logic
      - Metrics collection with atomics
    deliverables:
      - Sliding window implementation replacing simple failure counter
      - Error classifier interface with default implementation
      - Fallback function registration with static default fallback
      - Metrics accessor returning current circuit breaker state and statistics
      - State change notification mechanism
    estimated_hours: "6-10"

  - id: circuit-breaker-m3
    name: Client Integration & Chaos Testing
    description: >-
      Integrate the circuit breaker transparently with HTTP and gRPC clients,
      implement per-service isolation, and verify behavior with chaos testing.
    acceptance_criteria:
      - "HTTP client wrapper automatically applies circuit breaker to all outgoing requests to a configured host/service"
      - "gRPC client interceptor applies circuit breaker to RPC calls transparently without modifying calling code"
      - "Each downstream service has its own independent circuit breaker instance; one service's failures do not affect another's circuit"
      - "Chaos test injects configurable failure rates into a mock downstream service and verifies the circuit opens after the configured threshold"
      - "Chaos test verifies the circuit transitions to half-open after timeout and recovers to closed on successful probe"
      - "Chaos test runs under concurrent load (50+ simultaneous requests) and verifies no requests leak through an open circuit"
      - "Circuit breaker state is queryable via an HTTP endpoint or CLI command for each registered service (e.g., GET /circuits returns JSON with all circuit states)"
    pitfalls:
      - "Creating a circuit breaker per-request instead of per-service defeats the purpose; instances must be shared across calls to the same service"
      - "Chaos testing in production without safeguards (feature flags, percentage-based rollout) can cause real outages"
      - "Not exposing circuit state for debugging makes production incidents impossible to diagnose"
      - "Test flakiness from timing-dependent assertions; use controllable clocks or generous tolerances"
      - "Bulkhead (concurrency limiting) is a complementary but distinct pattern; if implemented, clearly separate it from circuit breaker logic"
    concepts:
      - Transparent middleware integration
      - Per-service circuit isolation
      - Chaos/fault injection testing
      - Controllable time in tests
      - Observability endpoints
    skills:
      - HTTP client middleware integration
      - gRPC interceptor implementation
      - Chaos testing with fault injection
      - Per-service circuit isolation
      - Integration testing under concurrency
    deliverables:
      - HTTP client wrapper with automatic circuit breaker application
      - gRPC interceptor applying circuit breaker to outgoing RPCs
      - Per-service circuit breaker registry with independent instances
      - Chaos test suite verifying open/half-open/closed transitions under load
      - Circuit state query endpoint returning all registered circuit states as JSON
    estimated_hours: "6-10"
```
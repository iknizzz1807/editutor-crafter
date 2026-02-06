# Circuit Breaker Pattern: Design Document


## Overview

A resilient microservices communication system that prevents cascade failures by automatically failing fast when downstream services become unhealthy, implementing state-based request routing with configurable recovery mechanisms. This system solves the key architectural challenge of maintaining service availability and responsiveness during partial system failures by providing automatic fault tolerance and graceful degradation capabilities.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

### Mental Model: Electrical Circuit Breakers

Think of the circuit breaker pattern like the electrical circuit breaker in your home's electrical panel. When you plug too many high-power devices into one circuit, the electrical current exceeds the safe threshold, and the circuit breaker automatically "trips" — cutting power to prevent fires or damage to your home's wiring. The breaker stays open (no power flows) until someone manually resets it after removing the overload condition.

In software systems, a **circuit breaker** serves the same protective function. Instead of protecting against electrical overload, it protects against **cascade failures** in distributed systems. When a downstream service becomes unhealthy (responding slowly, returning errors, or timing out), the circuit breaker "trips" and stops sending requests to that service temporarily. This prevents the unhealthy service from bringing down all the upstream services that depend on it.

The key insight is that **failing fast is often better than failing slow**. Just as an electrical circuit breaker prevents a house fire by cutting power immediately when dangerous conditions are detected, a software circuit breaker prevents system-wide outages by isolating failing components before they can cascade their failures throughout the entire system.

Unlike electrical circuit breakers that require manual reset, software circuit breakers can be **self-healing**. They periodically test whether the downstream service has recovered, automatically "closing" the circuit and resuming normal traffic flow when the service becomes healthy again.

### The Cascade Failure Problem

Modern microservices architectures create complex webs of service dependencies. A typical e-commerce application might have an order service that calls a payment service, inventory service, and shipping service. Each of these services may depend on additional services like user authentication, product catalog, and external payment gateways. This creates a **dependency chain** where the failure of any single service can potentially impact multiple upstream services.

Consider a concrete scenario: the payment service begins experiencing high latency due to database connection pool exhaustion. Each payment request now takes 30 seconds to timeout instead of the normal 200ms response time. Without circuit breakers, here's what happens:

1. **Request Accumulation**: The order service continues sending requests to the slow payment service. Since each request takes 30 seconds to fail, requests start accumulating in the order service's thread pool and connection pool.

2. **Resource Exhaustion**: The order service's threads become blocked waiting for payment responses. New incoming order requests cannot be processed because all threads are stuck waiting for the payment service timeouts.

3. **Upstream Cascade**: The order service itself becomes unresponsive. The web API gateway that calls the order service starts experiencing timeouts. Its threads become blocked, and soon the entire web frontend becomes unresponsive.

4. **System-Wide Outage**: What began as a database connection issue in the payment service has now brought down the order service, API gateway, and web frontend. Customers cannot browse products, view their cart, or perform any actions — even though the product catalog, user authentication, and inventory services are perfectly healthy.

The **root cause** is that traditional timeout mechanisms are insufficient for preventing cascade failures. Timeouts only determine how long to wait for a response, but they don't prevent the system from continuing to make doomed requests to failing services. Each timeout period consumes valuable resources (threads, connections, memory) that could be used for healthy requests.

> The circuit breaker pattern solves this by **failing fast** — detecting when a service is unhealthy and immediately rejecting requests without consuming resources or waiting for timeouts. This **preserves system capacity** for processing requests that have a chance of succeeding.

Circuit breakers also enable **graceful degradation**. Instead of returning generic errors when a service is unavailable, they can trigger fallback mechanisms that provide alternative functionality. For example, when the payment service circuit is open, the order service might save orders as "pending payment" and allow customers to complete the purchase later, rather than showing them a cryptic error message.

**Cascade failures** are particularly dangerous because they often exhibit **positive feedback loops**. As more services become overloaded due to slow downstream dependencies, the overall system load increases, making recovery more difficult. Circuit breakers break these feedback loops by isolating failing components and allowing healthy parts of the system to continue operating normally.

### Existing Fault Tolerance Approaches

Before implementing circuit breakers, it's important to understand how they complement and improve upon existing fault tolerance mechanisms. Most systems already use timeouts, retries, and bulkhead isolation, but each approach has limitations when used alone.

| Approach | How It Works | Strengths | Weaknesses | When Circuit Breakers Help |
|----------|--------------|-----------|------------|----------------------------|
| **Timeouts** | Set maximum wait time for service responses. If no response within timeout, fail the request. | Simple to implement. Prevents indefinite blocking. Configurable per service. | Still consumes resources during timeout period. All requests must wait full timeout before failing. No learning from failure patterns. | Circuit breakers eliminate timeout waits by failing immediately when service is known to be unhealthy. Prevents resource consumption during known outage periods. |
| **Retries** | Automatically retry failed requests with exponential backoff and jitter. | Can handle transient network issues. Simple configuration. Works well for intermittent failures. | Increases load on already-failing services. Can worsen cascade failures. No distinction between retryable and permanent failures. | Circuit breakers prevent retries to services in known failure states. Allow retries only when service shows signs of recovery in half-open state. |
| **Bulkhead Isolation** | Separate thread pools and connection pools for different downstream services. | Prevents one slow service from exhausting all system resources. Good resource isolation. | No reduction in requests to failing services. Still attempts to call unhealthy dependencies. Complex resource allocation decisions. | Circuit breakers reduce request volume to failing services, making bulkhead isolation more effective. Combination provides both resource isolation and request reduction. |
| **Load Balancing** | Distribute requests across multiple service instances. Remove unhealthy instances from rotation. | Handles individual instance failures. Can route around problematic nodes. Good for horizontal scaling. | Doesn't help when entire service type is failing. Load balancer health checks often too slow. May route traffic to degraded instances. | Circuit breakers provide faster failure detection than load balancer health checks. Work at service level rather than instance level. Can protect against service-wide issues. |
| **Rate Limiting** | Control the rate of incoming requests to prevent overload. | Protects services from traffic spikes. Simple to understand and configure. | Doesn't distinguish between healthy and unhealthy periods. May reject valid requests during normal operation. No failure detection mechanism. | Circuit breakers provide dynamic rate limiting based on service health. Zero requests during open state, normal rate during closed state, limited probing during half-open. |

> **Key Insight**: Circuit breakers are not a replacement for timeouts, retries, and bulkheads — they are a **coordination mechanism** that makes these other patterns more effective by providing **system-wide failure intelligence**.

The fundamental difference is that traditional approaches are **reactive** — they respond to individual request failures as they occur. Circuit breakers are **proactive** — they learn from patterns of failures and take preventive action to avoid future failures.

**Traditional Fault Tolerance** treats each request independently:
1. Send request to downstream service
2. Wait up to timeout period for response  
3. If timeout, retry with exponential backoff
4. If all retries fail, return error to caller
5. Repeat this entire process for the next request

**Circuit Breaker Enhanced Fault Tolerance** maintains system-wide failure state:
1. Check circuit breaker state before sending request
2. If circuit is closed (healthy), proceed normally with timeout and retry logic
3. If circuit is open (unhealthy), fail immediately without network call
4. If circuit is half-open (testing), allow limited requests to probe recovery
5. Use success/failure outcomes to update circuit state for future requests

This **state-based approach** is what enables circuit breakers to prevent cascade failures. Instead of each request independently discovering that a service is failing (consuming resources in the process), the circuit breaker maintains this knowledge and shares it across all requests.

> **Architecture Decision: Why Circuit Breakers Over Pure Timeout/Retry**
> - **Context**: We need to prevent cascade failures while maintaining good user experience during partial system failures
> - **Options Considered**: 
>   1. Aggressive timeouts with no retries (fast failure, but poor handling of transient issues)
>   2. Conservative timeouts with extensive retries (good for transient issues, but worsens cascade failures)  
>   3. Circuit breakers with configurable timeouts and fallbacks (adaptive behavior based on service health)
> - **Decision**: Implement circuit breakers as the primary fault tolerance mechanism, with timeouts and retries as secondary mechanisms
> - **Rationale**: Circuit breakers provide the benefits of both fast failure (during outages) and resilient retry logic (during recovery), while adding intelligent failure pattern recognition that pure timeout/retry approaches cannot provide
> - **Consequences**: More complex implementation and configuration, but significantly better cascade failure prevention and system observability

The next sections will detail how to implement a circuit breaker system that provides this intelligent, state-based fault tolerance while integrating seamlessly with existing timeout and retry mechanisms.

### Implementation Guidance

This section provides concrete guidance for implementing the circuit breaker concepts discussed above, using Go as the primary implementation language.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Recommended For Beginners |
|-----------|---------------|-----------------|---------------------------|
| **HTTP Client** | `net/http` with `http.Client` | Custom transport with connection pooling | net/http with reasonable timeouts |
| **Concurrency Control** | `sync.Mutex` for state protection | `sync.RWMutex` for read-heavy workloads | sync.Mutex for simplicity |
| **Time Management** | `time.Time` and `time.Duration` | Custom clock interface for testing | time.Time with clock interface |
| **Metrics Storage** | In-memory maps with mutex protection | Thread-safe concurrent data structures | In-memory with sync.Map |
| **Configuration** | Hardcoded constants in structs | External config files (JSON/YAML) | Struct fields with sensible defaults |
| **Logging** | Standard library `log` package | Structured logging with `logrus` or `zap` | Standard log for initial implementation |

#### Recommended File Structure

```
circuit-breaker/
├── cmd/
│   ├── server/
│   │   └── main.go                    ← Demo HTTP server
│   └── client/
│       └── main.go                    ← Demo client with circuit breaker
├── internal/
│   ├── circuitbreaker/
│   │   ├── circuit.go                 ← Core circuit breaker logic
│   │   ├── circuit_test.go            ← Unit tests
│   │   ├── config.go                  ← Configuration structures
│   │   ├── metrics.go                 ← Metrics collection
│   │   └── states.go                  ← State definitions
│   ├── slidingwindow/
│   │   ├── window.go                  ← Time-based sliding window
│   │   └── window_test.go             ← Window tests
│   ├── fallback/
│   │   ├── fallback.go                ← Fallback function registry
│   │   └── bulkhead.go                ← Bulkhead pattern implementation
│   └── integration/
│       ├── http_client.go             ← HTTP client wrapper
│       ├── grpc_interceptor.go        ← gRPC interceptor
│       └── service_registry.go        ← Per-service circuit management
├── pkg/
│   └── circuitbreaker/
│       └── circuit.go                 ← Public API
├── testdata/
│   ├── mock_server.go                 ← Mock services for testing
│   └── chaos_scenarios.go             ← Chaos testing helpers
├── examples/
│   ├── basic_usage.go                 ← Simple usage examples
│   └── advanced_patterns.go           ← Complex integration patterns
└── README.md
```

#### Infrastructure Starter Code

**Clock Interface for Testable Time Management** (`internal/circuitbreaker/clock.go`):

```go
package circuitbreaker

import "time"

// Clock provides an abstraction over time operations for testing.
// Production code uses RealClock, tests can use MockClock.
type Clock interface {
    Now() time.Time
    After(d time.Duration) <-chan time.Time
}

// RealClock implements Clock using the standard time package.
type RealClock struct{}

func (RealClock) Now() time.Time {
    return time.Now()
}

func (RealClock) After(d time.Duration) <-chan time.Time {
    return time.After(d)
}

// MockClock allows tests to control time progression.
type MockClock struct {
    current time.Time
    timers  []mockTimer
}

type mockTimer struct {
    when time.Time
    ch   chan time.Time
}

func NewMockClock(start time.Time) *MockClock {
    return &MockClock{current: start}
}

func (m *MockClock) Now() time.Time {
    return m.current
}

func (m *MockClock) After(d time.Duration) <-chan time.Time {
    ch := make(chan time.Time, 1)
    m.timers = append(m.timers, mockTimer{
        when: m.current.Add(d),
        ch:   ch,
    })
    return ch
}

func (m *MockClock) Advance(d time.Duration) {
    m.current = m.current.Add(d)
    // Trigger any timers that should fire
    for i := len(m.timers) - 1; i >= 0; i-- {
        timer := m.timers[i]
        if timer.when.Before(m.current) || timer.when.Equal(m.current) {
            timer.ch <- m.current
            close(timer.ch)
            // Remove from slice
            m.timers = append(m.timers[:i], m.timers[i+1:]...)
        }
    }
}
```

**Basic HTTP Mock Server** (`testdata/mock_server.go`):

```go
package testdata

import (
    "fmt"
    "net/http"
    "net/http/httptest"
    "sync/atomic"
    "time"
)

// MockServer provides controllable HTTP responses for testing circuit breakers.
type MockServer struct {
    server      *httptest.Server
    failureRate int32  // percentage 0-100
    latency     int64  // milliseconds
    requestCount int64
}

func NewMockServer() *MockServer {
    m := &MockServer{}
    m.server = httptest.NewServer(http.HandlerFunc(m.handler))
    return m
}

func (m *MockServer) URL() string {
    return m.server.URL
}

func (m *MockServer) Close() {
    m.server.Close()
}

// SetFailureRate sets percentage of requests that should return 500 errors.
func (m *MockServer) SetFailureRate(percentage int) {
    atomic.StoreInt32(&m.failureRate, int32(percentage))
}

// SetLatency sets artificial delay for all responses in milliseconds.
func (m *MockServer) SetLatency(ms int) {
    atomic.StoreInt64(&m.latency, int64(ms))
}

// RequestCount returns total number of requests received.
func (m *MockServer) RequestCount() int {
    return int(atomic.LoadInt64(&m.requestCount))
}

// Reset clears failure rate, latency, and request count.
func (m *MockServer) Reset() {
    atomic.StoreInt32(&m.failureRate, 0)
    atomic.StoreInt64(&m.latency, 0)
    atomic.StoreInt64(&m.requestCount, 0)
}

func (m *MockServer) handler(w http.ResponseWriter, r *http.Request) {
    count := atomic.AddInt64(&m.requestCount, 1)
    
    // Apply artificial latency
    if latency := atomic.LoadInt32(&m.latency); latency > 0 {
        time.Sleep(time.Duration(latency) * time.Millisecond)
    }
    
    // Apply failure rate
    failureRate := atomic.LoadInt32(&m.failureRate)
    if failureRate > 0 && (count % 100) < int64(failureRate) {
        w.WriteHeader(http.StatusInternalServerError)
        fmt.Fprintf(w, `{"error": "simulated failure", "request_id": %d}`, count)
        return
    }
    
    // Success response
    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, `{"message": "success", "request_id": %d}`, count)
}
```

#### Core Logic Skeleton

**Circuit Breaker State Definitions** (`internal/circuitbreaker/states.go`):

```go
package circuitbreaker

import (
    "context"
    "time"
)

// State represents the current state of a circuit breaker.
type State int

const (
    // StateClosed indicates the circuit is healthy and requests flow normally.
    StateClosed State = iota
    // StateOpen indicates the circuit is unhealthy and requests fail immediately.
    StateOpen
    // StateHalfOpen indicates the circuit is testing recovery with limited requests.
    StateHalfOpen
)

func (s State) String() string {
    switch s {
    case StateClosed:
        return "closed"
    case StateOpen:
        return "open"
    case StateHalfOpen:
        return "half-open"
    default:
        return "unknown"
    }
}

// CircuitBreaker represents the main circuit breaker implementation.
type CircuitBreaker struct {
    // TODO: Add fields for configuration, current state, metrics, etc.
    // Hint: You'll need mutex protection, failure counters, timers
}

// Execute runs the given function with circuit breaker protection.
// This is the main entry point that callers use.
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // TODO 1: Check current circuit state
    // TODO 2: If open, check if timeout period has elapsed (transition to half-open?)
    // TODO 3: If half-open, check if we should allow this request (limited concurrency)
    // TODO 4: If closed or half-open allowing request, execute the function
    // TODO 5: Record the result (success/failure) and update circuit state accordingly
    // TODO 6: Handle state transitions based on success/failure patterns
    // TODO 7: Return appropriate result or circuit breaker error
    // Hint: Use defer to ensure metrics are always updated
    panic("implement me")
}

// State returns the current state of the circuit breaker.
func (cb *CircuitBreaker) State() State {
    // TODO: Return current state with proper locking
    panic("implement me")
}

// Metrics returns current circuit breaker metrics.
func (cb *CircuitBreaker) Metrics() *Metrics {
    // TODO: Return copy of current metrics with proper locking
    panic("implement me")
}
```

#### Language-Specific Hints

**Concurrency Safety in Go:**
- Use `sync.RWMutex` for protecting circuit breaker state — most operations are reads (checking state), fewer are writes (state transitions)
- Consider `sync.atomic` for simple counters that are frequently updated (request counts, failure counts)
- Be careful with timer cleanup — use `time.Timer.Stop()` and drain the channel if needed

**Error Handling Patterns:**
- Define custom error types for different circuit breaker states: `ErrCircuitOpen`, `ErrCircuitHalfOpenRateLimited`
- Use `errors.Is()` for error classification — some errors should trip the circuit, others shouldn't
- Wrap downstream errors with context: `fmt.Errorf("circuit breaker: %w", downstreamErr)`

**Testing with Time:**
- Never use `time.Sleep()` in tests — use the clock interface and `MockClock.Advance()`
- For integration tests with real HTTP clients, use `httptest.Server` with controllable delays and failure rates
- Test state transitions by triggering the exact number of failures/successes needed for thresholds

#### Milestone Checkpoint

After implementing the basic circuit breaker (Milestone 1), verify functionality with this test sequence:

**Manual Verification Steps:**
1. **Test Closed State**: Start with a healthy mock server. Send 10 requests through the circuit breaker. All should succeed and circuit should remain closed.

2. **Test Open Transition**: Configure mock server for 100% failures. Send requests until failure threshold is reached. Verify circuit opens and subsequent requests fail immediately without hitting the server.

3. **Test Half-Open Transition**: Wait for the timeout period (or advance mock clock). Send a request and verify circuit transitions to half-open. Verify limited concurrency in half-open state.

4. **Test Recovery**: Configure mock server back to 0% failures. Send successful requests through half-open circuit until success threshold is reached. Verify circuit closes and normal operation resumes.

**Expected Command Line Output:**
```bash
$ go test ./internal/circuitbreaker/... -v
=== RUN   TestCircuitBreakerStateTransitions
    circuit_test.go:45: Circuit initial state: closed
    circuit_test.go:52: Circuit after 5 failures: closed  
    circuit_test.go:58: Circuit after 10 failures: open
    circuit_test.go:65: Circuit after timeout: half-open
    circuit_test.go:72: Circuit after successful probe: closed
--- PASS: TestCircuitBreakerStateTransitions (0.05s)
```

**Signs Something Is Wrong:**
- **Circuit never opens**: Check failure threshold configuration and ensure failures are being counted correctly
- **Circuit never closes**: Verify success counting logic in half-open state and success threshold configuration  
- **Race condition panics**: Add proper mutex protection around state reads/writes
- **Goroutine leaks**: Ensure timers are properly cleaned up and don't create unbounded goroutines


## Goals and Non-Goals

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

The circuit breaker pattern implementation requires carefully balancing comprehensive fault tolerance capabilities with practical constraints around performance, complexity, and maintenance overhead. This section establishes clear boundaries for what the system must accomplish, what quality attributes it must exhibit, and what features are explicitly excluded from scope to maintain focus on the core reliability engineering challenges.

Understanding these goals and non-goals is critical for making informed architectural decisions throughout the implementation. Without clear boundaries, circuit breaker systems often suffer from scope creep, attempting to solve every possible failure scenario and becoming overly complex, difficult to configure, and prone to their own failures. By establishing explicit priorities and constraints, we can build a system that solves the cascade failure problem effectively while remaining operationally manageable.

> **Key Design Principle:** A circuit breaker that is too complex to understand or configure correctly will itself become a source of system failures. Clarity and operational simplicity are as important as functional correctness.

### Functional Goals

The circuit breaker system must provide comprehensive fault isolation and recovery capabilities that address the fundamental problems of cascade failures in distributed systems. These functional requirements define the core behaviors that make the system effective at preventing upstream service degradation when downstream services become unhealthy.

#### State Management and Transitions

The system must implement a robust three-state circuit breaker that provides predictable failure isolation and recovery testing. The state machine forms the foundation of all circuit breaker behavior, making its correctness and reliability paramount.

| State | Primary Behavior | Request Handling | Metrics Collection |
|-------|------------------|------------------|--------------------|
| `StateClosed` | Normal operation with failure tracking | Pass all requests to downstream service | Record success/failure counts and response times |
| `StateOpen` | Fail-fast protection during service degradation | Immediately reject requests without downstream calls | Track rejection counts and maintain failure context |
| `StateHalfOpen` | Limited recovery testing with graduated rollback | Allow configurable number of test requests through | Monitor test request outcomes for state transition decisions |

The state transitions must be deterministic and based on clear, configurable criteria. When the failure threshold is exceeded in the closed state, the circuit must immediately transition to open, preventing additional load on the failing service. After the configured timeout period expires, the circuit must automatically transition to half-open to begin recovery testing. The half-open state must collect sufficient evidence of service health before returning to normal operation.

> **Decision: Deterministic State Transitions**
> - **Context**: Circuit breaker reliability depends on predictable state changes based on observable conditions
> - **Options Considered**: Time-based transitions only, failure-count-based only, hybrid approach with both criteria
> - **Decision**: Hybrid approach combining failure thresholds with timeout-based recovery testing
> - **Rationale**: Failure thresholds provide immediate protection when services degrade, while timeout-based recovery prevents circuits from staying open indefinitely during transient issues
> - **Consequences**: More complex configuration but significantly more effective at handling different failure patterns

#### Failure Detection and Classification

The system must accurately distinguish between errors that indicate downstream service health problems versus errors that should not influence circuit breaker state. This classification capability is essential for preventing false positives that would unnecessarily trip circuit breakers.

| Error Category | Circuit Impact | Examples | Handling Strategy |
|----------------|----------------|----------|------------------|
| Circuit-Opening Errors | Count toward failure threshold | Connection timeouts, HTTP 5xx responses, DNS resolution failures | Increment failure counter and evaluate threshold |
| Pass-Through Errors | Do not affect circuit state | HTTP 4xx client errors, authentication failures, validation errors | Record for metrics but exclude from health calculations |
| Configuration Errors | Log warnings but continue operation | Invalid fallback functions, misconfigured thresholds | Use safe defaults and alert operators |

The failure detection system must operate efficiently under high request volumes while maintaining accuracy. Each request outcome must be classified and recorded with minimal latency impact. The classification rules must be configurable per service to accommodate different error semantics across various APIs and protocols.

#### Recovery Testing and Graduation

The half-open state implementation must provide controlled recovery testing that minimizes risk while gathering sufficient evidence of service restoration. The system must support configurable success thresholds that determine when to transition back to normal operation.

The recovery testing process follows this sequence:
1. Upon entering half-open state, allow a limited number of test requests through to the downstream service
2. Monitor test request outcomes with the same error classification rules used in closed state
3. If any test request fails with a circuit-opening error, immediately return to open state and reset the timeout
4. If the configured number of consecutive successful test requests is achieved, transition to closed state and reset all failure counters
5. Maintain metrics on recovery attempts and success rates for operational visibility

> **Critical Implementation Detail:** The half-open state must strictly limit the number of concurrent test requests to prevent overwhelming a recovering service. This requires careful coordination between concurrent request handlers to ensure only the designated number of test requests proceed.

#### Thread Safety and Concurrency

All circuit breaker operations must be thread-safe and maintain consistency under high concurrency without introducing performance bottlenecks. The system will be accessed by multiple goroutines simultaneously, requiring careful synchronization of state transitions and metrics updates.

| Component | Concurrency Requirement | Synchronization Strategy |
|-----------|------------------------|--------------------------|
| State transitions | Atomic state changes with consistent metrics | Single mutex protecting state and transition logic |
| Metrics collection | High-throughput updates with eventual consistency | Atomic counters for frequently updated metrics |
| Sliding window buckets | Consistent bucket rotation with concurrent reads | Read-write mutex allowing concurrent reads during stable periods |
| Configuration updates | Safe runtime reconfiguration without service interruption | Copy-on-write for configuration structs |

The concurrency model must avoid race conditions that could lead to inconsistent state or incorrect failure counting. State transitions must be atomic operations that update both the circuit state and related metrics in a coordinated manner.

### Non-Functional Goals

The circuit breaker system must meet stringent performance, reliability, and observability requirements to be suitable for production microservices environments. These non-functional goals ensure the circuit breaker itself does not become a bottleneck or reliability concern.

#### Performance and Latency Requirements

The circuit breaker must introduce minimal overhead to request processing, particularly in the closed state where most requests should flow through normally. The system must be optimized for the common case of healthy services while still providing efficient protection during failures.

| Performance Metric | Requirement | Measurement Context |
|-------------------|-------------|-------------------|
| Closed state latency overhead | < 1ms p99 added latency | Per-request processing including state check and metrics update |
| Open state rejection latency | < 100μs p99 response time | Immediate rejection without downstream calls |
| Throughput capacity | Support 10,000+ requests/second per circuit | Concurrent request handling with proper synchronization |
| Memory efficiency | < 10MB base memory footprint | Including sliding window data and metrics storage |
| CPU overhead | < 5% additional CPU utilization | During normal operation with healthy downstream services |

The performance requirements must be maintained even under failure scenarios when the circuit breaker is actively protecting against cascading failures. The open state fast-rejection capability is particularly critical for maintaining upstream service responsiveness.

#### Reliability and Fault Tolerance

The circuit breaker implementation itself must be highly reliable and not introduce new failure modes into the system. As a critical infrastructure component responsible for fault tolerance, it must exhibit exceptional operational stability.

| Reliability Aspect | Requirement | Verification Method |
|-------------------|-------------|-------------------|
| Circuit breaker failures | Zero circuit breaker-induced service failures | Comprehensive error handling and graceful degradation |
| State consistency | No inconsistent state under any failure scenario | Race condition testing and state invariant verification |
| Configuration resilience | Safe operation with invalid or missing configuration | Default values and configuration validation |
| Memory stability | No memory leaks during extended operation | Long-running tests with memory profiling |
| Recovery guarantee | Always attempt recovery testing within timeout bounds | Timeout mechanism testing and state transition verification |

The reliability requirements extend to graceful handling of edge cases such as clock changes, configuration updates during runtime, and resource exhaustion scenarios. The circuit breaker must fail safe, defaulting to allowing requests through rather than blocking all traffic if its own logic encounters errors.

#### Observability and Debugging

The system must provide comprehensive visibility into circuit breaker state, decision-making process, and operational metrics to support effective troubleshooting and capacity planning.

| Observability Component | Data Provided | Update Frequency |
|------------------------|---------------|------------------|
| Real-time state metrics | Current state, failure counts, success rates per circuit | Updated on every state transition and request completion |
| Historical trending data | Failure rate trends, state transition frequency, recovery success rates | Sliding window data with configurable retention periods |
| Decision audit trail | Why state transitions occurred, error classification decisions | Logged at appropriate levels for post-incident analysis |
| Configuration visibility | Active thresholds, timeout values, window sizes per service | Available for runtime inspection and validation |
| Performance metrics | Latency distributions, throughput rates, concurrency levels | Real-time metrics suitable for alerting and dashboards |

The observability system must support both real-time monitoring for operational awareness and historical analysis for capacity planning and post-incident investigation. All metrics must be structured for easy integration with standard monitoring and alerting systems.

### Explicit Non-Goals

To maintain focus on core circuit breaker functionality and avoid unnecessary complexity, several related but distinct concerns are explicitly excluded from this implementation. Understanding these boundaries prevents scope creep and helps maintain system simplicity.

#### Service Discovery and Registry

The circuit breaker system will not implement service discovery, service registration, or dynamic service topology management. Service identification and endpoint resolution are considered external concerns that should be handled by dedicated service discovery systems or configuration management.

| Excluded Functionality | Rationale | Alternative Approach |
|----------------------|-----------|-------------------|
| Automatic service discovery | Adds complexity and couples circuit breaker to specific discovery mechanisms | Integrate with existing service discovery via configuration |
| Health check initiation | Overlaps with dedicated health checking systems and monitoring infrastructure | Use external health checks to inform circuit configuration |
| Load balancer integration | Circuit breaker operates at client level, not infrastructure level | Implement as client-side library with load balancer cooperation |
| Dynamic topology updates | Requires distributed consensus and state coordination complexity | Handle topology changes through configuration management systems |

This boundary ensures the circuit breaker can integrate with various service discovery approaches without being tightly coupled to any specific implementation. Services must be pre-configured or identified through external mechanisms.

#### Distributed Coordination and Consensus

Individual circuit breaker instances will operate independently based on their local view of service health. The system will not attempt to coordinate circuit breaker state across multiple service instances or implement distributed consensus for circuit decisions.

| Excluded Capability | Technical Reasoning | Impact and Mitigation |
|--------------------|-------------------|----------------------|
| Cross-instance state sharing | Distributed consensus adds significant complexity and new failure modes | Each instance makes independent decisions based on local observations |
| Global circuit coordination | Requires reliable broadcast mechanisms and conflict resolution | Accept that different instances may have different circuit states temporarily |
| Centralized circuit management | Creates single points of failure and scaling bottlenecks | Manage configuration through standard configuration distribution mechanisms |
| Cluster-wide failure detection | Requires sophisticated distributed algorithms and partition handling | Rely on service mesh or infrastructure-level failure detection for cluster concerns |

This decision prioritizes operational simplicity and fault isolation. While it may result in some inconsistency in circuit states across instances, it eliminates entire classes of distributed system failures that could compromise the reliability benefits the circuit breaker is intended to provide.

#### Advanced Machine Learning and Adaptive Algorithms

The initial implementation will use static, configurable thresholds rather than implementing adaptive threshold adjustment, predictive failure detection, or machine learning-based optimization of circuit parameters.

| Excluded Advanced Feature | Complexity Concerns | Future Extension Path |
|--------------------------|-------------------|----------------------|
| Adaptive threshold tuning | Requires training data collection, model management, and validation infrastructure | Can be added as optional enhancement once basic system proves reliable |
| Predictive failure detection | Needs sophisticated time series analysis and anomaly detection capabilities | Integrate with external ML platforms rather than building into circuit breaker |
| Automatic parameter optimization | Requires extensive instrumentation and feedback loop implementation | Provide rich metrics for external optimization tools |
| Behavioral pattern learning | Adds significant computational overhead and model lifecycle management | Support extensible configuration to accommodate learned parameters |

Static configuration provides predictable behavior that operators can reason about effectively. The metrics and observability capabilities included in scope will provide the foundation for future adaptive enhancements without compromising the core reliability characteristics.

#### Protocol-Specific Advanced Features

While the system will support integration with HTTP and gRPC clients, it will not implement advanced protocol-specific features that go beyond basic request/response pattern handling.

| Protocol Feature | Inclusion Decision | Implementation Approach |
|------------------|-------------------|------------------------|
| HTTP/gRPC basic integration | **Included** - Essential for practical usage | Wrapper and interceptor patterns for transparent integration |
| Streaming protocol support | **Excluded** - Adds complexity in defining request/failure semantics | Focus on request-response patterns, add streaming support in future versions |
| Protocol-specific error handling | **Limited** - Basic HTTP status code classification only | Configurable error classification rules rather than deep protocol integration |
| Advanced routing and load balancing | **Excluded** - Overlaps with service mesh and proxy responsibilities | Integrate at client library level with existing routing infrastructure |
| Message queuing integration | **Excluded** - Fundamentally different failure semantics than synchronous calls | Consider separate circuit breaker implementations optimized for async messaging |

This scope limitation ensures the core circuit breaker logic remains protocol-agnostic while still providing practical integration points for the most common synchronous communication patterns in microservices architectures.

### Implementation Guidance

The goals and non-goals established above directly influence the architectural decisions and implementation approach for the circuit breaker system. This guidance translates the requirements into concrete technical recommendations and starter code structure.

#### Technology Recommendations

| Component Category | Simple Implementation | Production-Ready Option |
|-------------------|----------------------|------------------------|
| State Management | In-memory structs with sync.Mutex | In-memory with atomic operations for hot paths |
| Metrics Storage | Local counters and gauges | Integration with Prometheus or similar TSDB |
| Configuration | Static configuration files (JSON/YAML) | Configuration with runtime updates via etcd or Consul |
| Time Management | Direct time.Now() calls | Pluggable Clock interface for testing and simulation |
| Observability | Standard library logging | Structured logging with OpenTelemetry integration |
| Testing Framework | Standard Go testing with table-driven tests | Property-based testing for state machine verification |

#### Recommended Module Structure

The implementation should follow a clear module structure that separates concerns and facilitates testing:

```
circuit-breaker/
├── cmd/
│   ├── demo/                    ← Example application showing usage
│   │   └── main.go
│   └── benchmark/               ← Performance testing tools
│       └── main.go
├── pkg/
│   ├── circuitbreaker/         ← Core circuit breaker implementation
│   │   ├── circuitbreaker.go   ← Main CircuitBreaker type and Execute method
│   │   ├── state.go            ← State enum and transition logic
│   │   ├── config.go           ← Configuration structures and validation
│   │   └── metrics.go          ← Metrics collection and sliding window
│   ├── integration/            ← Client integration wrappers
│   │   ├── http.go             ← HTTP client integration
│   │   ├── grpc.go             ← gRPC interceptor implementation
│   │   └── registry.go         ← Service registry for per-service circuits
│   └── fallback/               ← Fallback and bulkhead functionality
│       ├── fallback.go         ← Fallback function registry and execution
│       └── bulkhead.go         ← Concurrency limiting and isolation
├── internal/
│   ├── clock/                  ← Time abstraction for testing
│   │   ├── clock.go            ← Clock interface and real implementation
│   │   └── mock.go             ← MockClock for deterministic testing
│   └── testutil/               ← Testing utilities
│       ├── mock_server.go      ← MockServer for integration testing
│       └── scenarios.go        ← Common test scenarios and chaos patterns
└── examples/                   ← Complete working examples
    ├── http-service/           ← HTTP service with circuit breaker
    └── grpc-service/           ← gRPC service with circuit breaker
```

#### Core Configuration Structure

```go
// Package circuitbreaker provides the main configuration types that satisfy
// the functional goals while maintaining simplicity for operators.
package circuitbreaker

import "time"

// Config defines all configurable parameters for circuit breaker behavior.
// This structure directly implements the functional goals around failure
// detection, recovery testing, and performance requirements.
type Config struct {
    // TODO: Add FailureThreshold field (int) - number of failures before opening circuit
    // TODO: Add RecoveryTimeout field (time.Duration) - how long to wait before attempting recovery
    // TODO: Add SuccessThreshold field (int) - consecutive successes needed to close circuit
    // TODO: Add SlidingWindowSize field (time.Duration) - time window for failure rate calculation
    // TODO: Add MaxConcurrentRequests field (int) - bulkhead limit for concurrent requests
    // TODO: Add ErrorClassifier field (func(error) bool) - determines which errors trip circuit
}

// Metrics captures all observability data required by non-functional goals.
type Metrics struct {
    // TODO: Add State field (State) - current circuit breaker state
    // TODO: Add TotalRequests field (uint64) - lifetime request count
    // TODO: Add FailureCount field (uint64) - current failure count in window
    // TODO: Add SuccessCount field (uint64) - current success count in window
    // TODO: Add RejectionCount field (uint64) - requests rejected due to open circuit
    // TODO: Add StateTransitions field (uint64) - number of state changes
    // TODO: Add LastStateChange field (time.Time) - when state last changed
}
```

#### State Machine Foundation

```go
// State represents the three-state circuit breaker state machine that implements
// the core functional goals around failure isolation and recovery testing.
type State int

const (
    // StateClosed allows normal request flow with failure tracking
    StateClosed State = iota
    // StateOpen blocks all requests immediately for fast-fail protection  
    StateOpen
    // StateHalfOpen allows limited test requests for recovery verification
    StateHalfOpen
)

// CircuitBreaker implements the core functionality defined in functional goals.
type CircuitBreaker struct {
    // TODO: Add state field (State) protected by mutex for thread safety
    // TODO: Add config field (*Config) for operational parameters
    // TODO: Add metrics field (*Metrics) for observability requirements
    // TODO: Add clock field (Clock) for testable time operations
    // TODO: Add lastFailureTime field (time.Time) for timeout calculations
    // TODO: Add halfOpenRequests field (int) for recovery test limiting
    // TODO: Add mutex field (sync.RWMutex) for concurrent access protection
}

// Execute is the main entry point that implements all functional goals
// around request routing, failure detection, and recovery testing.
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // TODO 1: Acquire read lock and check current state
    // TODO 2: If StateClosed, execute request and update metrics based on outcome
    // TODO 3: If StateOpen, check if recovery timeout has elapsed
    // TODO 4: If timeout elapsed, transition to StateHalfOpen and continue
    // TODO 5: If still in timeout period, reject immediately for fast-fail goal
    // TODO 6: If StateHalfOpen, check concurrent test request limit
    // TODO 7: Execute test request and handle state transition based on outcome
    // TODO 8: Update all metrics to satisfy observability goals
    // TODO 9: Return appropriate response or error based on circuit decision
}
```

#### Milestone Checkpoints

After implementing the basic goals and structure:

1. **Functional Verification**: Create a simple test that opens a circuit after threshold failures and verifies fast-fail behavior
2. **Performance Verification**: Run benchmark showing <1ms overhead in closed state as specified in performance goals
3. **Reliability Verification**: Run concurrent access tests to verify thread safety requirements are met
4. **Observability Verification**: Confirm all required metrics are collected and accessible for monitoring integration

The implementation should satisfy each category of goals incrementally, with the functional goals forming the foundation for non-functional goal achievement.


## High-Level Architecture

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

The circuit breaker system follows a layered architecture designed to provide transparent fault tolerance for microservices communication. Think of this architecture like a sophisticated traffic management system at a busy intersection - there's a central controller (the circuit breaker core) that monitors traffic flow, intelligent sensors (metrics collectors) that gather real-time data about road conditions, and automated routing systems (client integrations) that direct vehicles along the safest paths without drivers needing to understand the underlying complexity.

![Circuit Breaker System Components](./diagrams/system-components.svg)

This layered approach separates concerns while maintaining loose coupling between components. The core circuit breaker logic remains independent of specific transport protocols, metrics systems can be swapped without affecting circuit behavior, and client integrations provide protocol-specific adaptations without duplicating fault tolerance logic. Each layer builds upon the foundation provided by lower layers, creating a robust and extensible fault tolerance system.

### Component Overview

The circuit breaker system consists of five primary components, each with distinct responsibilities and well-defined interfaces. Understanding these components and their interactions is crucial for implementing a maintainable and extensible circuit breaker system.

#### Core Circuit Breaker Engine

The `CircuitBreaker` component serves as the central coordinator for all fault tolerance decisions. Think of it as the brain of an air traffic control system - it processes incoming requests, evaluates current system health, makes routing decisions, and coordinates with other systems to ensure safe operation. This component maintains the authoritative state machine and enforces the fundamental circuit breaker behavior across all three states.

The circuit breaker engine operates as a thread-safe state machine that can handle thousands of concurrent requests while maintaining consistent state transitions. It encapsulates the core algorithm logic including failure threshold evaluation, timeout management, and recovery testing protocols. The engine delegates metrics collection and sliding window calculations to specialized components while retaining full control over state transition decisions.

| Responsibility | Description | Key Methods |
|---|---|---|
| State Management | Maintains current circuit state and handles transitions | `State()`, `Execute()` |
| Request Coordination | Routes requests based on current state and policies | `Execute(ctx, fn)` |
| Failure Detection | Evaluates request outcomes against configured thresholds | Internal failure tracking |
| Recovery Testing | Manages half-open state probing and success validation | Internal state transitions |
| Configuration Management | Applies runtime configuration changes safely | Configuration updates |

#### Metrics Collection and Analysis System

The metrics collection system functions like a hospital's patient monitoring equipment - it continuously gathers vital signs (request success rates, response times, error patterns) and provides both real-time alerts and historical analysis capabilities. The `Metrics` component aggregates raw request data into actionable intelligence that drives circuit breaker state decisions.

This system employs a sliding window algorithm to provide time-aware failure rate calculations that adapt to changing service conditions. Unlike simple failure counters that can be skewed by historical data, the sliding window approach ensures circuit breaker decisions reflect current service health rather than outdated failure patterns.

| Component | Purpose | Key Data Structures |
|---|---|---|
| `Metrics` | Current state snapshot for observability | Success count, failure count, failure rate |
| `SlidingWindow` | Time-based failure rate calculation | Ring buffer of time buckets |
| `WindowBucket` | Individual time slice metrics | Timestamp, success count, failure count |
| `MetricsCollector` | Aggregation and persistence coordination | Collection intervals, retention policies |

#### Fallback and Bulkhead Coordination

The fallback system operates like an emergency response plan - when primary systems fail, it provides predetermined alternative responses to maintain service continuity. The `FallbackRegistry` manages a catalog of fallback functions organized by service and operation type, ensuring that when circuits open, users receive meaningful responses rather than raw error messages.

The bulkhead pattern implementation provides resource isolation between different downstream services, preventing one failing service from consuming all available resources and affecting other services. Think of this like compartments in a ship's hull - if one compartment floods, the others remain watertight and the ship stays afloat.

| Pattern | Implementation | Resource Management |
|---|---|---|
| Fallback Registry | Service-specific fallback function mapping | Function composition and error handling |
| Bulkhead Isolation | Per-service concurrency limits using semaphores | Resource pool management per service |
| Graceful Degradation | Alternative response generation when circuits open | Response caching and templating |

#### Client Integration Layer

The client integration layer provides transparent circuit breaker functionality without requiring application code changes. Like a sophisticated proxy server, it intercepts outgoing requests, applies circuit breaker logic, and forwards or rejects requests based on current circuit state. This layer handles protocol-specific concerns while delegating fault tolerance decisions to the core circuit breaker engine.

The integration layer maintains a service registry that maps downstream services to their respective circuit breaker instances, ensuring that each service gets independent fault tolerance without interference from other services' failure patterns. This isolation is critical for preventing cascade failures between unrelated services.

| Integration Type | Transport Protocol | Implementation Pattern |
|---|---|---|
| HTTP Client Wrapper | HTTP/HTTPS requests | Decorator pattern around http.Client |
| gRPC Interceptor | gRPC unary and streaming calls | Unary and stream interceptor interfaces |
| Service Registry | Service discovery and circuit mapping | Service identifier to circuit breaker mapping |

#### Configuration and Observability Infrastructure

The configuration system provides dynamic parameter adjustment without service restarts, essential for tuning circuit breaker behavior in production environments. The observability infrastructure exposes circuit breaker state and metrics through standard monitoring interfaces, enabling integration with existing operational tooling.

| Infrastructure Component | Purpose | Key Interfaces |
|---|---|---|
| `Config` | Runtime configuration management | Threshold, timeout, and window parameters |
| `Clock` | Time abstraction for testing | `Now()`, `After(duration)` methods |
| Observability Exports | Metrics and state exposure | Prometheus, logging, dashboard integration |

> **Design Insight:** The layered architecture enables independent testing and development of each component. The core circuit breaker logic can be thoroughly tested with mocked metrics and clock components, while integration layers can be tested with mock circuit breakers, allowing parallel development and comprehensive test coverage.

### Architecture Decision Records

> **Decision: Layered Architecture with Dependency Injection**
> - **Context**: Circuit breaker functionality needs to integrate with multiple transport protocols while maintaining testability and avoiding tight coupling between components.
> - **Options Considered**: Monolithic circuit breaker class, plugin-based architecture, layered architecture with interfaces
> - **Decision**: Layered architecture with dependency injection and interface-based component communication
> - **Rationale**: Layered architecture provides clear separation of concerns, dependency injection enables comprehensive testing with mock implementations, and interfaces allow protocol-specific integrations without core logic changes
> - **Consequences**: Enables parallel development of components, comprehensive testing through mocking, but introduces complexity in component coordination and initialization

| Architecture Option | Testability | Extensibility | Complexity | Chosen? |
|---|---|---|---|---|
| Monolithic Circuit Breaker | Low - tightly coupled | Low - protocol changes require core changes | Low | ❌ |
| Plugin-based Architecture | Medium - plugin boundaries | High - runtime plugin loading | High | ❌ |
| Layered with Dependency Injection | High - mockable interfaces | High - new layers via interfaces | Medium | ✅ |

> **Decision: Per-Service Circuit Isolation**
> - **Context**: Microservices architecture requires independent fault tolerance for each downstream service to prevent failure propagation between unrelated services.
> - **Options Considered**: Single global circuit breaker, per-endpoint circuits, per-service circuits
> - **Decision**: Per-service circuit breaker instances with shared configuration templates
> - **Rationale**: Service-level isolation prevents unrelated service failures from affecting each other while avoiding excessive granularity that could lead to resource overhead and configuration complexity
> - **Consequences**: Enables independent service health management and prevents cascade failures, but requires service identification logic and increases memory usage proportional to service count

### Recommended Module Structure

The module structure follows Go's standard project layout conventions while organizing circuit breaker components into logical packages that mirror the architectural layers. This structure supports both internal development and external consumption as a library.

```
circuitbreaker/
├── cmd/
│   └── demo/                    ← demonstration applications
│       ├── http-client/         ← HTTP client example
│       └── grpc-service/        ← gRPC service example
├── pkg/                         ← public API packages
│   ├── circuitbreaker/          ← main circuit breaker package
│   │   ├── breaker.go           ← CircuitBreaker implementation
│   │   ├── config.go            ← Config structure and validation
│   │   ├── metrics.go           ← Metrics collection and calculation
│   │   ├── state.go             ← State enumeration and transitions
│   │   └── window.go            ← SlidingWindow implementation
│   ├── fallback/                ← fallback pattern implementations
│   │   ├── registry.go          ← FallbackRegistry for function management
│   │   └── strategies.go        ← common fallback strategies
│   ├── bulkhead/                ← bulkhead pattern implementation
│   │   └── semaphore.go         ← semaphore-based resource limiting
│   └── integration/             ← client integration packages
│       ├── http/                ← HTTP client integration
│       │   ├── client.go        ← HTTP client wrapper
│       │   └── middleware.go    ← HTTP middleware functions
│       ├── grpc/                ← gRPC integration
│       │   ├── interceptor.go   ← unary and stream interceptors
│       │   └── client.go        ← gRPC client helper functions
│       └── registry/            ← service registry implementation
│           └── service.go       ← service identification and mapping
├── internal/                    ← private implementation packages
│   ├── clock/                   ← time abstraction for testing
│   │   ├── clock.go             ← Clock interface and real implementation
│   │   └── mock.go              ← MockClock for deterministic testing
│   ├── metrics/                 ← internal metrics collection
│   │   └── collector.go         ← MetricsCollector implementation
│   └── testing/                 ← test utilities and mock implementations
│       ├── mockserver.go        ← MockServer for integration testing
│       └── chaos.go             ← chaos testing framework
├── examples/                    ← comprehensive usage examples
│   ├── basic/                   ← basic circuit breaker usage
│   ├── http-integration/        ← HTTP client integration example
│   ├── grpc-integration/        ← gRPC integration example
│   └── monitoring/              ← observability integration examples
├── docs/                        ← additional documentation
├── go.mod                       ← Go module definition
├── go.sum                       ← dependency checksums
└── README.md                    ← project overview and quickstart
```

#### Package Responsibility Matrix

| Package | Primary Responsibility | External Dependencies | Test Strategy |
|---|---|---|---|
| `pkg/circuitbreaker` | Core circuit breaker logic and state machine | Standard library only | Unit tests with MockClock |
| `pkg/fallback` | Fallback pattern implementations | Core circuit breaker package | Unit tests with mock functions |
| `pkg/bulkhead` | Resource isolation and concurrency limiting | Standard library sync primitives | Unit tests with goroutine testing |
| `pkg/integration/http` | HTTP client circuit breaker integration | net/http package | Integration tests with MockServer |
| `pkg/integration/grpc` | gRPC interceptor implementations | gRPC client libraries | Integration tests with mock gRPC services |
| `internal/clock` | Time abstraction for deterministic testing | Standard library time package | Unit tests with fixed time scenarios |
| `internal/testing` | Test utilities and mock implementations | Testing and HTTP libraries | Self-testing utilities |

#### Import and Visibility Guidelines

The package structure enforces clear boundaries between public APIs and internal implementation details. External consumers should only import packages under `pkg/`, while `internal/` packages remain private to the circuit breaker implementation. This design prevents external dependencies on implementation details and allows internal refactoring without breaking changes.

```go
// External usage - recommended imports
import (
    "github.com/yourorg/circuitbreaker/pkg/circuitbreaker"
    "github.com/yourorg/circuitbreaker/pkg/integration/http"
    "github.com/yourorg/circuitbreaker/pkg/fallback"
)

// Internal usage - only within circuitbreaker module
import (
    "github.com/yourorg/circuitbreaker/internal/clock"
    "github.com/yourorg/circuitbreaker/internal/testing"
)
```

### Integration Points

The circuit breaker system integrates with external systems through well-defined interface contracts that minimize coupling while maximizing functionality. These integration points are designed to work with existing infrastructure components without requiring architectural changes to consuming applications.

#### HTTP Client Integration Architecture

The HTTP integration layer wraps standard `http.Client` instances with circuit breaker functionality using the decorator pattern. This approach preserves the familiar `http.Client` interface while adding transparent fault tolerance. The integration maintains full compatibility with existing HTTP middleware chains and request/response processing pipelines.

![Client Integration Architecture](./diagrams/integration-architecture.svg)

The HTTP integration provides multiple integration approaches to accommodate different application architectures and migration strategies. Applications can choose between decorating existing clients, using middleware functions, or implementing custom transport layers depending on their specific requirements and constraints.

| Integration Method | Use Case | Implementation Complexity | Migration Effort |
|---|---|---|---|
| Client Decoration | Wrapping existing http.Client instances | Low | Minimal - change client initialization |
| Middleware Function | Adding to existing middleware chains | Medium | Low - insert middleware in chain |
| Custom Transport | Fine-grained control over request routing | High | Medium - implement http.RoundTripper |
| Global Client Replacement | Organization-wide circuit breaker adoption | Low | High - replace all http.Client usage |

#### gRPC Interceptor Integration

The gRPC integration leverages gRPC's interceptor mechanism to provide transparent circuit breaker functionality for both unary and streaming RPC calls. Interceptors execute before request transmission, allowing circuit breaker logic to evaluate service health and make routing decisions without modifying application business logic.

The interceptor implementation handles both client-side and server-side scenarios, though circuit breaker functionality primarily applies to client-side calls to downstream services. Server-side integration focuses on metrics collection and observability rather than request blocking, since servers should not circuit-break their own incoming requests.

| gRPC Integration Type | Purpose | Implementation Interface |
|---|---|---|
| Unary Client Interceptor | Circuit breaker for simple request-response calls | `grpc.UnaryClientInterceptor` |
| Stream Client Interceptor | Circuit breaker for streaming RPC calls | `grpc.StreamClientInterceptor` |
| Metrics Server Interceptor | Request metrics collection on server side | `grpc.UnaryServerInterceptor` |

#### Service Discovery and Registry Integration

The service registry integration provides automatic circuit breaker provisioning for discovered services. As new services are discovered through service discovery mechanisms (Consul, etcd, Kubernetes service discovery), corresponding circuit breakers are automatically created with appropriate configuration templates. This automation reduces operational overhead and ensures consistent fault tolerance policies across all services.

Service identification strategies determine how requests are mapped to specific circuit breaker instances. The system supports multiple identification approaches including URL-based routing, header-based service identification, and explicit service name configuration. The chosen strategy affects both circuit isolation granularity and configuration complexity.

| Service Identification Strategy | Granularity | Configuration Complexity | Use Case |
|---|---|---|---|
| Host-based (service.domain.com) | Per-service | Low | Simple microservices with dedicated domains |
| Path-based (/api/v1/users) | Per-endpoint | High | Monolithic services with multiple endpoints |
| Header-based (X-Service-Name) | Configurable | Medium | Service mesh environments with routing headers |
| Explicit Configuration | Manual | Low | Static service configurations |

#### Monitoring and Observability Integration

The observability integration exposes circuit breaker state and metrics through standard monitoring interfaces including Prometheus metrics, structured logging, and health check endpoints. This integration enables circuit breaker monitoring within existing operational workflows without requiring specialized monitoring tools.

Circuit breaker metrics follow standard observability patterns with appropriate labels for service identification, state tracking, and performance measurement. The metrics design supports both real-time alerting on circuit state changes and historical analysis of service reliability patterns.

| Observability Component | Integration Method | Key Metrics Exposed |
|---|---|---|
| Prometheus Metrics | HTTP metrics endpoint | Circuit state, request counts, failure rates |
| Structured Logging | Standard logging interfaces | State transitions, configuration changes |
| Health Checks | HTTP health endpoints | Overall circuit breaker system health |
| Distributed Tracing | OpenTracing/OpenTelemetry | Request routing decisions and timing |
| Dashboard Integration | Grafana/custom dashboards | Visual circuit state and trend analysis |

#### Configuration Management Integration

The configuration management integration supports both static configuration files and dynamic configuration updates through configuration management systems. Dynamic configuration enables runtime tuning of circuit breaker parameters without service restarts, essential for production environments where optimal thresholds may require adjustment based on observed service behavior.

Configuration validation ensures that parameter changes maintain system stability and prevent configuration errors that could disable fault tolerance protections. The validation system checks parameter ranges, dependency relationships, and configuration consistency across related services.

| Configuration Source | Update Method | Validation Level | Use Case |
|---|---|---|---|
| Static Configuration Files | Application startup | Schema validation | Development and testing environments |
| Environment Variables | Application startup | Type and range validation | Container-based deployments |
| Configuration Management APIs | Runtime updates | Full consistency validation | Production environments with dynamic tuning |
| Service Discovery Metadata | Automatic discovery | Template-based validation | Auto-scaling environments |

### Common Pitfalls

⚠️ **Pitfall: Shared Circuit Breaker Instances Across Services**
Creating a single circuit breaker instance and using it for multiple downstream services defeats the isolation principle and can cause failures in one service to incorrectly affect another. For example, if Service A fails frequently but Service B is healthy, a shared circuit breaker might block requests to Service B based on Service A's failures. Always create separate circuit breaker instances for each downstream service, typically using a service registry pattern to manage the circuit breaker lifecycle.

⚠️ **Pitfall: Circuit Breaker Per Request Instead of Per Service**
Implementing circuit breakers at the individual request level rather than the service level creates excessive overhead and prevents effective failure pattern detection. Request-level circuits cannot accumulate sufficient failure history to make meaningful state decisions, and they consume excessive memory. Instead, implement circuits at the service level and route all requests to a given service through the same circuit breaker instance.

⚠️ **Pitfall: Ignoring Service Identification in Integration Layers**
Failing to properly identify which service a request targets can result in requests being routed to the wrong circuit breaker or creating excessive circuit breaker instances. This is particularly problematic in HTTP integrations where service identification might rely on URL parsing or header inspection. Implement clear service identification strategies and validate that requests are consistently routed to the correct circuit breaker instances.

⚠️ **Pitfall: Blocking Service Discovery Integration**
Implementing synchronous service discovery lookups within circuit breaker request processing can introduce latency and potential deadlocks. Service discovery should be asynchronous with circuit breaker instances created proactively rather than on-demand during request processing. Use background processes to monitor service discovery changes and pre-provision circuit breaker instances.

⚠️ **Pitfall: Inadequate Error Handling in Integration Layers**
Integration layers must handle both circuit breaker errors (circuit open, bulkhead full) and underlying transport errors (network failures, timeouts) differently. Mixing these error types can result in incorrect failure attribution and inappropriate fallback behavior. Implement clear error classification in integration layers and ensure that circuit breaker errors trigger appropriate fallback responses while transport errors are handled by circuit breaker failure tracking.

### Implementation Guidance

The implementation follows Go's idioms for interface-based design and dependency injection while maintaining simplicity and performance. The core components use minimal external dependencies, making the circuit breaker suitable for high-performance applications and environments with strict dependency constraints.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Rationale |
|---|---|---|---|
| HTTP Integration | net/http client decoration | Custom http.RoundTripper with middleware | net/http provides sufficient functionality for most use cases |
| gRPC Integration | grpc.UnaryClientInterceptor | Custom connection pool with interceptors | Standard interceptors handle most scenarios effectively |
| Metrics Collection | In-memory counters | Prometheus client library integration | In-memory collection minimizes dependencies and latency |
| Time Management | time.Now() and time.After() | Custom Clock interface for testing | Clock abstraction enables deterministic testing |
| Configuration | Struct-based configuration | Dynamic configuration with validation | Static configuration adequate for initial implementation |
| Service Discovery | Map-based service registry | Integration with Consul/etcd | Simple registry sufficient for most microservice environments |

#### Core Package Structure Implementation

The main circuit breaker package provides the core functionality with minimal dependencies:

```go
// pkg/circuitbreaker/breaker.go
package circuitbreaker

import (
    "context"
    "sync"
    "time"
)

// State represents the current circuit breaker state
type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

// Config holds circuit breaker configuration parameters
type Config struct {
    FailureThreshold    int           // failures before opening circuit
    SuccessThreshold    int           // successes to close from half-open
    Timeout            time.Duration // time to wait in open state
    MaxConcurrentCalls int           // bulkhead limit
}

// Metrics provides observability data for the circuit breaker
type Metrics struct {
    State              State     // current circuit state
    TotalRequests      int64     // total requests processed
    SuccessfulRequests int64     // successful requests
    FailedRequests     int64     // failed requests
    ConsecutiveFailures int      // current consecutive failures
    LastFailureTime    time.Time // timestamp of last failure
    StateChangedTime   time.Time // timestamp of last state change
}

// CircuitBreaker provides fault tolerance for service calls
type CircuitBreaker struct {
    config  Config
    state   State
    metrics Metrics
    clock   Clock  // time abstraction for testing
    mutex   sync.RWMutex
    
    // TODO: Add sliding window for advanced failure rate calculation
    // TODO: Add bulkhead semaphore for concurrency limiting
    // TODO: Add fallback function registry
}

// Clock interface abstracts time operations for testing
type Clock interface {
    Now() time.Time
    After(duration time.Duration) <-chan time.Time
}
```

#### HTTP Integration Package Structure

```go
// pkg/integration/http/client.go
package http

import (
    "net/http"
    "github.com/yourorg/circuitbreaker/pkg/circuitbreaker"
    "github.com/yourorg/circuitbreaker/pkg/integration/registry"
)

// ClientWrapper decorates http.Client with circuit breaker functionality
type ClientWrapper struct {
    client   *http.Client
    registry *registry.ServiceRegistry
    
    // TODO: Add service identification strategy
    // TODO: Add request/response middleware chain
}

// WrapClient creates a circuit breaker enabled HTTP client
func WrapClient(client *http.Client, registry *registry.ServiceRegistry) *ClientWrapper {
    // TODO: Initialize wrapper with client and registry
    // TODO: Set up default service identification strategy
    // TODO: Configure default timeout and retry policies
}

// Do executes HTTP request through circuit breaker
func (c *ClientWrapper) Do(req *http.Request) (*http.Response, error) {
    // TODO: Extract service identifier from request
    // TODO: Get circuit breaker for service from registry
    // TODO: Execute request through circuit breaker
    // TODO: Handle circuit breaker errors vs transport errors
    // TODO: Update metrics based on response
}
```

#### Service Registry Package Structure

```go
// pkg/integration/registry/service.go  
package registry

import (
    "github.com/yourorg/circuitbreaker/pkg/circuitbreaker"
    "sync"
)

// ServiceRegistry manages circuit breakers for discovered services
type ServiceRegistry struct {
    circuits map[string]*circuitbreaker.CircuitBreaker
    config   circuitbreaker.Config // template configuration
    mutex    sync.RWMutex
    
    // TODO: Add service discovery integration
    // TODO: Add circuit breaker lifecycle management
}

// GetOrCreateCircuit returns circuit breaker for service
func (r *ServiceRegistry) GetOrCreateCircuit(serviceID string) *circuitbreaker.CircuitBreaker {
    // TODO: Check if circuit exists in map
    // TODO: Create new circuit with template config if not exists
    // TODO: Handle concurrent creation with proper locking
    // TODO: Register circuit for monitoring and cleanup
}

// ServiceIdentifier defines strategy for extracting service ID
type ServiceIdentifier interface {
    ExtractServiceID(req interface{}) (string, error)
}
```

#### Testing Infrastructure Package

```go
// internal/testing/mockserver.go
package testing

import (
    "net/http"
    "net/http/httptest"
    "time"
)

// MockServer provides controllable HTTP server for testing
type MockServer struct {
    server      *httptest.Server
    failureRate int // percentage of requests that should fail
    latency     time.Duration
    
    // TODO: Add failure pattern simulation
    // TODO: Add request logging and analysis
}

// NewMockServer creates a configurable test server
func NewMockServer() *MockServer {
    // TODO: Initialize test server with handler
    // TODO: Set up configurable failure injection
    // TODO: Add metrics collection for test validation
}

// SetFailureRate configures percentage of requests that fail
func (m *MockServer) SetFailureRate(percentage int) {
    // TODO: Update failure rate configuration
    // TODO: Validate percentage range (0-100)
}

// SetLatency configures response delay
func (m *MockServer) SetLatency(duration time.Duration) {
    // TODO: Update latency configuration
    // TODO: Add jitter for realistic testing
}
```

#### Clock Abstraction for Testing

```go
// internal/clock/clock.go
package clock

import "time"

// Clock interface enables deterministic testing of time-based logic
type Clock interface {
    Now() time.Time
    After(duration time.Duration) <-chan time.Time
}

// RealClock provides actual time operations for production use
type RealClock struct{}

// Now returns current system time
func (c RealClock) Now() time.Time {
    return time.Now()
}

// After returns channel that fires after duration
func (c RealClock) After(duration time.Duration) <-chan time.Time {
    return time.After(duration)
}

// MockClock provides controllable time for testing
type MockClock struct {
    currentTime time.Time
    timers      []*MockTimer
    
    // TODO: Add timer management for deterministic testing
    // TODO: Add time advancement controls
    // TODO: Add timer callback scheduling
}

// AdvanceTime moves mock clock forward by duration
func (m *MockClock) AdvanceTime(duration time.Duration) {
    // TODO: Update current time
    // TODO: Trigger any expired timers
    // TODO: Maintain chronological timer ordering
}
```

#### Milestone Checkpoints

**Milestone 1 Checkpoint - Basic Circuit Breaker:**
After implementing the core circuit breaker state machine, you should be able to run the following validation:

```bash
go test ./pkg/circuitbreaker/... -v
```

Expected behavior:
- Circuit starts in `StateClosed` and processes requests normally
- After configured failure threshold, circuit transitions to `StateOpen`
- During open state, requests fail immediately without calling downstream
- After timeout period, circuit transitions to `StateHalfOpen`  
- Successful requests in half-open state transition back to `StateClosed`
- Failed requests in half-open state transition back to `StateOpen`

**Milestone 2 Checkpoint - Advanced Features:**
After implementing sliding window metrics and fallback functionality:

```bash
go test ./pkg/... -v
go run ./cmd/demo/http-client/main.go
```

Expected behavior:
- Sliding window accurately calculates failure rate over time periods
- Fallback functions execute when circuit is open
- Bulkhead limits concurrent requests per service independently
- Different error types are classified correctly (circuit-opening vs pass-through)

**Milestone 3 Checkpoint - Integration Testing:**
After implementing client integration and chaos testing:

```bash
go test ./pkg/integration/... -v -integration
go run ./examples/http-integration/main.go
```

Expected behavior:
- HTTP client wrapper transparently applies circuit breaker to requests
- Per-service circuits operate independently 
- Chaos testing successfully triggers circuit state changes
- Circuit breaker state is observable through metrics endpoints


## Data Model

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

The data model forms the foundation of our circuit breaker implementation, defining the core structures that represent circuit state, configuration parameters, and observability metrics. Think of the data model as the blueprint of a sophisticated monitoring system - just as a hospital's patient monitoring system needs structured data formats for vital signs, alert thresholds, and historical trends, our circuit breaker requires well-defined data structures to track service health, configure behavior parameters, and maintain operational metrics.

![Circuit Breaker Data Model](./diagrams/data-model.svg)

The data model encompasses four primary categories of information: circuit state representation, configuration parameters that govern circuit behavior, real-time metrics for observability, and sliding window data structures for sophisticated failure rate calculations. Each category serves a distinct purpose in the overall system architecture while maintaining clear relationships and dependencies between components.

### Circuit State Types

The circuit breaker state machine operates through three distinct states, each with specific behavioral characteristics and transition criteria. These states represent the fundamental operating modes that determine how requests are processed and routed through the system.

**State Enumeration and Properties**

The `State` enumeration defines the three possible circuit breaker states with their associated behaviors and characteristics:

| State Name | Numeric Value | Request Behavior | Failure Tracking | Timeout Behavior |
|------------|---------------|------------------|------------------|------------------|
| `StateClosed` | 0 | Passes all requests to downstream service | Tracks failures against threshold | N/A - no timeout active |
| `StateOpen` | 1 | Rejects all requests immediately | Stops tracking failures | Waits for recovery timeout |
| `StateHalfOpen` | 2 | Allows limited test requests through | Tracks test request outcomes | Transitions based on success/failure |

The state enumeration provides type safety and clear semantics for state transitions throughout the circuit breaker lifecycle. Each state encapsulates specific request routing logic and failure handling behavior that aligns with the circuit breaker pattern's fault tolerance objectives.

**State Transition Triggers**

State transitions occur based on specific events and thresholds that indicate changes in downstream service health:

| Current State | Trigger Event | Next State | Required Conditions |
|---------------|---------------|------------|-------------------|
| `StateClosed` | Failure threshold reached | `StateOpen` | Consecutive failures >= `FailureThreshold` |
| `StateClosed` | Success | `StateClosed` | Reset failure counter to zero |
| `StateOpen` | Recovery timeout expired | `StateHalfOpen` | Time since opening >= `RecoveryTimeout` |
| `StateHalfOpen` | Test request succeeds | `StateClosed` | Successful test request completes |
| `StateHalfOpen` | Test request fails | `StateOpen` | Any test request failure detected |

> **Decision: Three-State Model**
> - **Context**: Circuit breakers require different behaviors for normal operation, failure protection, and recovery testing
> - **Options Considered**: Two-state (open/closed), three-state (closed/open/half-open), five-state (with intermediate states)
> - **Decision**: Three-state model with closed, open, and half-open states
> - **Rationale**: Provides essential recovery testing capability without unnecessary complexity. Two states lack recovery probing, while five states add complexity without meaningful behavioral distinctions.
> - **Consequences**: Enables safe recovery testing while maintaining simple state machine logic that's easy to reason about and debug.

**State Persistence and Recovery**

Circuit breaker state information includes both transient operational data and persistent configuration that survives service restarts:

| Data Category | Persistence | Recovery Behavior | Thread Safety |
|---------------|-------------|-------------------|---------------|
| Current state | In-memory only | Always starts `StateClosed` | Atomic operations |
| Failure counters | In-memory only | Reset to zero on restart | Mutex protection |
| Configuration | Persistent/config files | Loaded at startup | Read-only after init |
| Metrics history | Configurable retention | Optional persistence | Concurrent-safe collections |

The decision to maintain circuit state only in memory simplifies the implementation while providing adequate fault tolerance for most microservice scenarios. Services that restart frequently will reset to the closed state, which provides a natural recovery mechanism for transient infrastructure issues.

### Configuration Model

The configuration model defines all tunable parameters that control circuit breaker behavior, failure detection sensitivity, and recovery characteristics. This model enables fine-tuning circuit breaker behavior for different service characteristics and operational requirements.

**Core Configuration Structure**

The `Config` structure encompasses all configurable parameters organized by functional area:

| Field Name | Type | Default Value | Description | Validation Rules |
|------------|------|---------------|-------------|------------------|
| `FailureThreshold` | `int` | 5 | Consecutive failures before opening circuit | Must be >= 1 |
| `RecoveryTimeout` | `time.Duration` | 30s | Time to wait before attempting recovery | Must be > 0 |
| `RequestTimeout` | `time.Duration` | 10s | Maximum time to wait for downstream response | Must be > 0 |
| `MaxConcurrentRequests` | `int` | 100 | Bulkhead limit for concurrent requests | Must be > 0 |
| `SlidingWindowSize` | `time.Duration` | 60s | Time window for failure rate calculation | Must be > 0 |
| `SlidingWindowBuckets` | `int` | 10 | Number of time buckets in sliding window | Must be >= 1 |
| `FailureRateThreshold` | `float64` | 0.5 | Failure rate to trigger circuit opening (0.0-1.0) | Must be 0.0 <= x <= 1.0 |
| `MinimumRequestCount` | `int` | 10 | Minimum requests before failure rate applies | Must be >= 1 |

**Service-Specific Configuration**

The configuration model supports per-service customization through a service registry mapping approach:

| Configuration Scope | Storage Location | Override Priority | Update Mechanism |
|---------------------|------------------|-------------------|-------------------|
| Global defaults | Application config file | Lowest | Requires restart |
| Service-specific | Service registry entries | Medium | Hot reload capable |
| Runtime overrides | Administrative API | Highest | Immediate effect |

This hierarchical configuration approach enables operational flexibility while maintaining predictable behavior defaults. Services with different characteristics (latency, reliability, criticality) can have appropriately tuned circuit breaker parameters without requiring code changes.

**Advanced Configuration Options**

Extended configuration parameters support sophisticated circuit breaker behaviors for complex operational scenarios:

| Feature Area | Configuration Fields | Purpose | Default Behavior |
|-------------|---------------------|---------|------------------|
| Error Classification | `CircuitOpeningErrors`, `PassThroughErrors` | Define which error types trigger circuit state changes | All errors count as failures |
| Fallback Strategy | `FallbackEnabled`, `FallbackTimeout` | Control fallback execution behavior | No fallback configured |
| Metrics Collection | `MetricsEnabled`, `MetricsRetention` | Configure observability data collection | Basic metrics with 1-hour retention |
| Testing Support | `ChaosTestingEnabled`, `MockFailureRate` | Enable controlled failure injection | Disabled in production |

> **Decision: Hierarchical Configuration**
> - **Context**: Different services need different circuit breaker parameters, but maintaining separate configuration files is operationally complex
> - **Options Considered**: Single global config, per-service config files, hierarchical with overrides
> - **Decision**: Hierarchical configuration with global defaults and per-service overrides
> - **Rationale**: Provides operational flexibility for service-specific tuning while maintaining simple defaults for common cases. Reduces configuration maintenance burden.
> - **Consequences**: Requires configuration resolution logic but enables fine-grained tuning without configuration proliferation.

**Configuration Validation and Safety**

Configuration validation ensures that parameter combinations result in sensible circuit breaker behavior:

| Validation Category | Rules | Error Handling | Runtime Behavior |
|---------------------|-------|----------------|------------------|
| Range validation | Thresholds within valid ranges (0-1 for rates, >0 for counts) | Reject invalid config with descriptive error | Use previous valid config |
| Logical consistency | Recovery timeout > request timeout, window size > bucket count | Warn and auto-adjust conflicting values | Log warnings, apply corrections |
| Resource limits | Max concurrent requests within system limits | Validate against available resources | Cap at system maximums |
| Update safety | New config doesn't cause immediate circuit opening | Gradual application of new thresholds | Phased rollout of changes |

The validation system prevents configuration errors that could cause immediate service disruption while providing clear feedback about parameter conflicts and their resolution.

### Metrics and Window Data

The metrics and window data structures enable sophisticated failure rate calculation, observability, and operational monitoring of circuit breaker behavior. These structures capture both real-time operational state and historical trends necessary for informed decision-making.

**Real-Time Metrics Structure**

The `Metrics` structure provides a comprehensive snapshot of current circuit breaker operational state:

| Field Name | Type | Update Frequency | Calculation Method | Reset Conditions |
|------------|------|------------------|--------------------|------------------|
| `RequestCount` | `int64` | Per request | Atomic increment | Sliding window expiration |
| `SuccessCount` | `int64` | Per successful response | Atomic increment | Sliding window expiration |
| `FailureCount` | `int64` | Per failed response | Atomic increment | Sliding window expiration |
| `TimeoutCount` | `int64` | Per timeout | Atomic increment | Sliding window expiration |
| `CircuitBreakerOpenCount` | `int64` | Per state transition | Atomic increment | Never reset |
| `LastFailureTime` | `time.Time` | Per failure | Direct assignment | Manual reset only |
| `LastSuccessTime` | `time.Time` | Per success | Direct assignment | Manual reset only |
| `CurrentState` | `State` | Per state transition | Direct assignment | State machine driven |
| `StateTransitionTime` | `time.Time` | Per state transition | Direct assignment | State machine driven |

The metrics structure uses atomic operations for thread-safe updates while providing consistent snapshots for observability and decision-making purposes.

**Sliding Window Data Model**

The sliding window implementation uses time-based buckets to calculate failure rates over configurable time periods:

| Component | Type | Purpose | Lifecycle |
|-----------|------|---------|-----------|
| `WindowBucket` | `struct` | Individual time slice metrics | Created/destroyed with bucket rotation |
| `SlidingWindow` | `struct` | Bucket collection and management | Long-lived, spans multiple buckets |
| `BucketRing` | `[]WindowBucket` | Circular buffer for efficient bucket rotation | Fixed size, reused buckets |
| `WindowPosition` | `struct` | Current window position tracking | Updated with each bucket rotation |

**Window Bucket Structure**

Individual `WindowBucket` structures capture metrics for specific time slices:

| Field Name | Type | Granularity | Thread Safety | Aggregation Role |
|------------|------|-------------|---------------|-------------------|
| `StartTime` | `time.Time` | Bucket boundary timestamp | Read-only after creation | Window position reference |
| `EndTime` | `time.Time` | Bucket boundary timestamp | Read-only after creation | Window boundary validation |
| `RequestCount` | `int32` | Per-bucket counter | Atomic operations | Sum across active buckets |
| `SuccessCount` | `int32` | Per-bucket counter | Atomic operations | Sum for success rate calculation |
| `FailureCount` | `int32` | Per-bucket counter | Atomic operations | Sum for failure rate calculation |
| `AvgLatency` | `time.Duration` | Weighted average per bucket | Lock-protected | Weighted average across buckets |
| `MaxLatency` | `time.Duration` | Maximum observed latency | Atomic compare-and-swap | Maximum across active buckets |

**Metrics Collection and Aggregation**

The `MetricsCollector` coordinates data collection and aggregation across multiple circuit breaker instances:

| Collection Scope | Aggregation Method | Storage Duration | Export Frequency |
|------------------|-------------------|------------------|------------------|
| Per-circuit metrics | Direct field access | Real-time only | On demand |
| Service-level metrics | Sum across circuit instances | Configurable retention | Every 30 seconds |
| System-level metrics | Sum across all services | 24-hour retention | Every 60 seconds |
| Historical trends | Time-series buckets | 7-day retention | Every 300 seconds |

The metrics collection system balances observability requirements with memory efficiency by using configurable retention periods and aggregation levels.

> **Decision: Time-Based Sliding Windows**
> - **Context**: Simple failure counting doesn't account for request timing and can cause oscillating circuit behavior
> - **Options Considered**: Count-based sliding window, time-based sliding window, exponential moving average
> - **Decision**: Time-based sliding window with configurable bucket count and duration
> - **Rationale**: Provides accurate failure rate calculation over meaningful time periods, handles variable request rates better than count-based approaches, more intuitive than exponential averages for operational tuning.
> - **Consequences**: Requires more memory for bucket storage and CPU for bucket rotation, but provides significantly better circuit stability and predictable behavior.

**Memory Management and Performance**

The window data structures implement efficient memory management to handle high-throughput scenarios:

| Optimization Technique | Implementation | Memory Impact | Performance Benefit |
|------------------------|----------------|---------------|-------------------|
| Ring buffer buckets | Pre-allocated circular array | Fixed memory footprint | No allocation/GC overhead |
| Atomic counters | Lock-free increment operations | Minimal contention overhead | High concurrency performance |
| Bucket pooling | Reuse bucket structures | Reduced GC pressure | Consistent allocation patterns |
| Lazy aggregation | Calculate sums on read | Lower write-path overhead | Optimized for read-heavy workloads |

These optimizations ensure that metrics collection doesn't become a performance bottleneck in high-throughput microservice environments while maintaining accuracy and thread safety.

**Data Consistency and Race Conditions**

The metrics system addresses potential race conditions and consistency issues through careful synchronization design:

| Race Condition Scenario | Protection Mechanism | Consistency Guarantee | Performance Impact |
|-------------------------|---------------------|---------------------|-------------------|
| Concurrent bucket updates | Atomic field operations | Eventually consistent within bucket | Near-zero overhead |
| Bucket rotation timing | Single-writer principle | Consistent bucket boundaries | Write lock during rotation only |
| Metrics snapshot reading | Copy-on-read for aggregates | Point-in-time consistency | Minimal read overhead |
| State transition updates | Compare-and-swap operations | Atomic state changes | Lock-free fast path |

The synchronization design prioritizes performance for the common case (request processing) while ensuring consistency for less frequent operations (bucket rotation, state transitions).

### Common Pitfalls

⚠️ **Pitfall: Inconsistent State and Metrics Synchronization**
Circuit breaker state changes and metrics updates can become inconsistent if not properly synchronized. For example, transitioning from closed to open state while simultaneously updating failure counts can result in metrics that don't reflect the actual state transition timing. Always update metrics atomically with state transitions using compare-and-swap operations or explicit locking.

⚠️ **Pitfall: Configuration Validation Bypass**
Skipping configuration validation can lead to nonsensical parameter combinations like recovery timeouts shorter than request timeouts, or failure thresholds of zero. This causes unpredictable circuit behavior and difficult debugging scenarios. Implement comprehensive validation at configuration load time and reject invalid combinations with clear error messages.

⚠️ **Pitfall: Sliding Window Bucket Boundary Issues**
Incorrectly calculating bucket boundaries can cause requests to be counted in wrong time windows, leading to inaccurate failure rate calculations. Ensure bucket start/end times align precisely with wall clock time and handle clock adjustments gracefully. Use truncated timestamps (e.g., `time.Now().Truncate(bucketDuration)`) for consistent bucket alignment.

⚠️ **Pitfall: Memory Leaks in Metrics Collection**
Accumulating metrics without proper cleanup leads to memory leaks, especially in long-running services with many downstream dependencies. Implement proper metrics cleanup when circuits are removed, use bounded retention periods for historical data, and employ object pooling for frequently allocated metric structures.

⚠️ **Pitfall: Race Conditions in Configuration Updates**
Hot-reloading configuration without proper synchronization can cause race conditions between configuration reads and updates. This can result in partially applied configuration changes or inconsistent behavior across concurrent requests. Use read-write mutexes or atomic pointer swapping for configuration updates.

### Implementation Guidance

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| State Management | Atomic int32 with constants | Typed enum with state machine |
| Configuration | JSON config files | YAML with validation schemas |
| Metrics Storage | In-memory maps with mutexes | Lock-free ring buffers |
| Time Operations | Direct time.Now() calls | Injected Clock interface for testing |
| Validation | Simple range checks | Comprehensive validation with error details |

**Recommended Module Structure**

```
internal/circuitbreaker/
  types.go              ← Core data structures and enums
  config.go             ← Configuration model and validation
  metrics.go            ← Metrics structures and collection
  sliding_window.go     ← Window bucket implementation
  types_test.go         ← Data structure tests
  config_test.go        ← Configuration validation tests
  metrics_test.go       ← Metrics collection tests
```

**Core Data Structure Definitions**

```go
package circuitbreaker

import (
    "sync"
    "sync/atomic"
    "time"
)

// State represents the three possible circuit breaker states
type State int32

const (
    // StateClosed allows normal request flow to downstream services
    StateClosed State = iota
    // StateOpen blocks all requests immediately without calling downstream
    StateOpen
    // StateHalfOpen allows limited test requests to probe service recovery
    StateHalfOpen
)

// String returns human-readable state name for logging and debugging
func (s State) String() string {
    // TODO: Implement string representation for each state
    // Hint: Use switch statement with StateClosed -> "CLOSED", etc.
}

// Config holds all circuit breaker configuration parameters
type Config struct {
    // TODO: Add all configuration fields from the table above
    // Hint: Use appropriate Go types (time.Duration, int, float64)
    // Hint: Add JSON/YAML tags for configuration file parsing
}

// Validate checks configuration parameters for consistency and valid ranges
func (c *Config) Validate() error {
    // TODO: Implement validation rules from configuration validation table
    // TODO: Check range constraints (FailureThreshold >= 1, etc.)
    // TODO: Check logical consistency (RecoveryTimeout > RequestTimeout)
    // TODO: Return descriptive errors for invalid configurations
}

// Metrics provides observability into circuit breaker operational state
type Metrics struct {
    // TODO: Add all metrics fields from the metrics structure table
    // TODO: Use atomic types (int64) for thread-safe counters
    // TODO: Use time.Time for timestamp fields
    // TODO: Use sync.RWMutex for protecting complex updates
}

// Snapshot returns a point-in-time copy of current metrics
func (m *Metrics) Snapshot() *Metrics {
    // TODO: Create metrics copy using atomic loads for counters
    // TODO: Protect timestamp fields with read lock
    // TODO: Return new Metrics instance with copied values
}

// WindowBucket represents metrics for a single time slice
type WindowBucket struct {
    // TODO: Add bucket fields from window bucket structure table
    // TODO: Use atomic int32 for counters (frequent updates)
    // TODO: Use time.Time for bucket boundaries
    // TODO: Use atomic values for latency tracking
}

// SlidingWindow manages time-based failure rate calculation
type SlidingWindow struct {
    buckets    []WindowBucket
    bucketSize time.Duration
    windowSize time.Duration
    position   int32
    mutex      sync.RWMutex
    
    // TODO: Add fields for current window position tracking
    // TODO: Add fields for bucket rotation timing
}

// NewSlidingWindow creates initialized sliding window with specified parameters
func NewSlidingWindow(windowSize time.Duration, bucketCount int) *SlidingWindow {
    // TODO: Calculate bucket duration from window size and bucket count
    // TODO: Initialize bucket ring buffer with specified count
    // TODO: Set up initial bucket with current time boundaries
    // TODO: Return configured SlidingWindow instance
}

// RecordRequest updates the current bucket with request outcome
func (sw *SlidingWindow) RecordRequest(success bool, latency time.Duration) {
    // TODO: Get current time bucket (may require rotation)
    // TODO: Atomically increment appropriate counters (success/failure)
    // TODO: Update latency statistics (average, maximum)
    // TODO: Handle bucket rotation if time boundary crossed
}

// FailureRate calculates current failure rate across active window
func (sw *SlidingWindow) FailureRate() (float64, int64) {
    // TODO: Rotate expired buckets to current time
    // TODO: Sum request counts across all active buckets
    // TODO: Sum failure counts across all active buckets  
    // TODO: Calculate and return failure rate (failures/requests) and total requests
    // TODO: Handle division by zero (no requests case)
}
```

**Configuration Loading and Validation Infrastructure**

```go
// ConfigLoader handles loading and validating configuration from various sources
type ConfigLoader struct {
    defaultConfig *Config
    validators    []ConfigValidator
}

// ConfigValidator defines interface for configuration validation rules
type ConfigValidator interface {
    Validate(config *Config) error
}

// LoadConfig loads configuration with hierarchical override support
func (cl *ConfigLoader) LoadConfig(sources ...ConfigSource) (*Config, error) {
    // TODO: Start with default configuration
    // TODO: Apply configuration sources in priority order (global -> service -> runtime)
    // TODO: Run all registered validators on final configuration
    // TODO: Return validated configuration or detailed error
}

// NewDefaultConfig returns sensible defaults for circuit breaker configuration
func NewDefaultConfig() *Config {
    // TODO: Return Config with all default values from configuration table
    // Hint: Use constants for magic numbers (DefaultFailureThreshold = 5, etc.)
}
```

**Metrics Collection Infrastructure**

```go
// MetricsCollector aggregates metrics across multiple circuit breaker instances
type MetricsCollector struct {
    circuits map[string]*Metrics  // serviceID -> metrics
    mutex    sync.RWMutex
    
    // TODO: Add aggregation state fields
    // TODO: Add export/reporting configuration
}

// RecordRequest updates metrics for successful request
func (mc *MetricsCollector) RecordRequest(serviceID string, latency time.Duration) {
    // TODO: Get or create metrics for service
    // TODO: Atomically increment request and success counters
    // TODO: Update last success timestamp
    // TODO: Record latency in sliding window if enabled
}

// RecordFailure updates metrics for failed request  
func (mc *MetricsCollector) RecordFailure(serviceID string, err error) {
    // TODO: Get or create metrics for service
    // TODO: Atomically increment request and failure counters
    // TODO: Update last failure timestamp
    // TODO: Classify error type for circuit breaker decision
}

// AggregateMetrics returns system-wide metrics summary
func (mc *MetricsCollector) AggregateMetrics() map[string]*Metrics {
    // TODO: Acquire read lock for consistent snapshot
    // TODO: Create snapshot of all service metrics
    // TODO: Calculate aggregate statistics (total requests, overall failure rate)
    // TODO: Return service metrics map with aggregate entry
}
```

**Milestone Checkpoints**

After implementing the data model structures:
1. **Configuration Loading**: Run `go test ./internal/circuitbreaker/config_test.go` - should validate all configuration combinations and reject invalid parameters
2. **Metrics Recording**: Verify atomic counter updates work correctly under concurrent load using `go test -race`
3. **Sliding Window**: Test bucket rotation by advancing mock time - failure rates should reflect only recent time windows
4. **Integration Verification**: Create circuit breaker instance with custom config - should initialize with proper defaults and validate parameters

Expected behavior verification:
- Configuration validation rejects negative thresholds and inconsistent timeout values
- Metrics counters increment atomically without race conditions under concurrent access
- Sliding window buckets rotate at proper time boundaries and maintain accurate failure rates
- State transitions update both operational state and metrics timestamps consistently


## Circuit Breaker State Machine

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features)

### Mental Model: Traffic Light Controller

Think of a circuit breaker as an intelligent traffic light controller at a busy intersection. Just as a traffic light manages vehicle flow to prevent accidents and congestion, a circuit breaker manages request flow to prevent service failures and cascade breakdowns.

In our analogy, the **green light** represents the closed state where traffic flows normally through the intersection. The traffic light monitors conditions continuously—counting vehicles, detecting accidents, and measuring flow rates. When everything operates smoothly, drivers proceed without delays, similar to how requests pass through to downstream services when they're healthy.

The **red light** represents the open state where all traffic stops immediately. When the controller detects dangerous conditions—perhaps an accident blocking the intersection or extreme congestion—it immediately switches to red to prevent more vehicles from entering and making the situation worse. Similarly, when a circuit breaker detects service failures exceeding the configured threshold, it stops all requests from reaching the failing service, preventing cascade failures and resource exhaustion.

The **yellow light** represents the half-open state where the controller cautiously tests whether normal flow can resume. After the red light has been active for a predetermined time, the controller briefly switches to yellow to allow a few test vehicles through. If these test vehicles pass through successfully, the controller returns to green. If problems persist, it immediately returns to red. This mirrors how a circuit breaker in half-open state allows limited test requests to probe whether the downstream service has recovered.

This mental model helps us understand that circuit breakers are not binary on/off switches, but intelligent flow controllers that continuously monitor conditions, make state transitions based on observed behavior, and implement gradual recovery testing to ensure stability.

![Circuit Breaker State Machine](./diagrams/state-machine.svg)

### State Definitions and Behaviors

The circuit breaker implements a three-state finite state machine where each state exhibits distinct request handling behavior, failure tracking logic, and transition criteria. Understanding the precise behavior of each state is crucial for implementing correct circuit breaker logic and avoiding common race conditions.

#### Closed State Behavior

The **closed state** represents normal operation where the circuit breaker acts as a transparent proxy, passing all requests through to the downstream service while continuously monitoring request outcomes. In this state, the circuit breaker maintains an active failure counter that increments with each failed request and resets to zero on successful requests. This reset-on-success behavior ensures that transient failures don't accumulate indefinitely and trigger circuit opening during otherwise healthy operation.

The closed state implements synchronous failure tracking where each request outcome immediately updates the failure counter. When a request succeeds, the circuit breaker not only resets the failure counter but also updates the last success timestamp in the metrics collection. This timestamp serves multiple purposes: it provides observability into service health patterns and helps determine recovery confidence when transitioning from half-open back to closed state.

Request processing in closed state follows a straightforward pass-through pattern with failure detection. The circuit breaker invokes the downstream service, measures response time, and classifies the outcome based on configurable error classification rules. Network timeouts, HTTP 5xx responses, and connection failures typically count as circuit-opening failures, while client errors like HTTP 4xx responses usually don't, since they indicate request problems rather than service health issues.

The transition trigger for leaving closed state is failure threshold breach. When consecutive failures reach the configured `FailureThreshold`, the circuit breaker immediately transitions to open state without attempting additional requests. This immediate transition prevents further resource waste and begins the recovery timeout period.

| Closed State Property | Value | Description |
|----------------------|-------|-------------|
| Request Routing | Pass-through to downstream | All requests forwarded to actual service |
| Failure Tracking | Active increment/reset | Failure count increases on failure, resets on success |
| Success Handling | Reset failure counter | Successful requests reset consecutive failure count to zero |
| Transition Trigger | Failure threshold breach | Consecutive failures >= `FailureThreshold` opens circuit |
| Observability | Full metrics collection | Records latency, success rate, and error details |
| Concurrency | Unlimited (respecting bulkhead) | No artificial request limiting beyond bulkhead constraints |

#### Open State Behavior

The **open state** represents a protective mode where the circuit breaker immediately rejects all incoming requests without attempting to contact the downstream service. This fail-fast behavior serves multiple critical purposes: it prevents resource exhaustion in the calling service, reduces load on the already-struggling downstream service, and provides predictable response times for upstream consumers.

In open state, request rejection happens at the circuit breaker level before any network calls, timeouts, or resource allocation occurs. The circuit breaker returns a predefined error indicating the circuit is open, allowing calling code to immediately execute fallback logic or propagate the failure upstream with minimal latency. This behavior contrasts sharply with the unpredictable timeouts and resource consumption that would occur if requests continued reaching a failing service.

The open state maintains a recovery timer that starts when the circuit first transitions from closed to open. This timer runs independently of request flow—even if no requests arrive during the open period, the timer continues counting down toward the recovery timeout. When the recovery timeout expires, the circuit breaker automatically transitions to half-open state to begin testing service recovery.

Metrics collection continues in open state, but with different characteristics. The circuit breaker records each rejected request as a circuit breaker event rather than a service failure, providing visibility into how many requests the circuit breaker protected during the outage period. This data helps operators understand the impact of circuit breaker activation and tune timeout parameters appropriately.

| Open State Property | Value | Description |
|---------------------|-------|-------------|
| Request Routing | Immediate rejection | All requests fail fast with circuit open error |
| Failure Tracking | Suspended | No failure counting since no downstream calls occur |
| Recovery Timer | Active countdown | Timer started at transition, counts down to recovery timeout |
| Transition Trigger | Recovery timeout expiry | Automatic transition to half-open after `RecoveryTimeout` |
| Observability | Circuit breaker metrics | Records rejection count and circuit open duration |
| Concurrency | Zero downstream calls | No requests reach downstream service |

#### Half-Open State Behavior

The **half-open state** implements cautious recovery testing where the circuit breaker allows a limited number of test requests to reach the downstream service while maintaining the ability to quickly return to open state if failures persist. This state balances the need to detect service recovery with the risk of overwhelming a service that may still be struggling.

The half-open state operates with a success-first policy where even a single successful request can trigger immediate transition back to closed state. This aggressive recovery approach assumes that if the service can handle one request successfully, it has likely recovered and can handle normal traffic volumes. However, implementations may configure a minimum success count requirement for more conservative recovery behavior.

Request limiting in half-open state typically allows only one concurrent request to the downstream service, though this may be configurable. Additional requests arriving while a test request is in flight are usually rejected with the same circuit open error used in open state. This limiting prevents multiple clients from simultaneously overwhelming a service that may be in a fragile recovery state.

Failure handling in half-open state is immediate and decisive. Any failure—whether network error, timeout, or service error—triggers immediate transition back to open state and restarts the recovery timeout period. This behavior ensures that the circuit breaker doesn't repeatedly attempt requests against a service that hasn't actually recovered.

The transition logic from half-open state is binary: success leads to closed state with reset failure counters, while failure leads to open state with a fresh recovery timer. There is no mechanism to remain in half-open state indefinitely—each test request must produce a definitive outcome that drives a state transition.

| Half-Open State Property | Value | Description |
|-------------------------|-------|-------------|
| Request Routing | Limited test requests | Single test request allowed, others rejected |
| Failure Tracking | Immediate transition | Any failure immediately returns to open state |
| Success Handling | Return to closed | Success transitions to closed with reset counters |
| Transition Trigger | First request outcome | Success → closed, failure → open |
| Observability | Recovery attempt metrics | Records recovery test outcomes and timing |
| Concurrency | Single test request | Only one request in flight to downstream service |

### State Transition Logic

The state transition algorithm forms the core of circuit breaker functionality, implementing the decision logic that determines when and how to move between states based on observed service behavior. The algorithm must handle concurrent requests safely while making consistent transition decisions based on failure patterns and timing constraints.

#### Transition Algorithm Steps

The state transition evaluation occurs synchronously with each request outcome in closed and half-open states, and asynchronously via timer expiry in open state. This hybrid approach ensures responsive failure detection while providing predictable recovery behavior independent of traffic patterns.

**Closed State Transition Logic:**

1. **Request Outcome Classification**: When a request completes, the circuit breaker first classifies the outcome using configurable error classification rules. Network failures, timeouts exceeding `RequestTimeout`, and HTTP 5xx responses typically classify as circuit-relevant failures. Client errors like HTTP 4xx responses, authentication failures, or application-specific errors may not count toward circuit opening, depending on configuration.

2. **Failure Counter Update**: For circuit-relevant failures, increment the consecutive failure counter atomically using thread-safe operations. The counter tracks only consecutive failures—any successful request resets it to zero. This consecutive counting approach prevents isolated transient failures during otherwise healthy operation from triggering circuit opening.

3. **Threshold Evaluation**: After updating the failure counter, compare it against the configured `FailureThreshold`. If the counter equals or exceeds the threshold, immediately trigger transition to open state. This comparison must happen atomically with the counter update to prevent race conditions where multiple concurrent requests could all increment the counter and trigger multiple transitions.

4. **Success Counter Reset**: For successful requests, atomically reset the consecutive failure counter to zero and update the last success timestamp. This reset ensures that the circuit breaker doesn't accumulate failures across periods of healthy operation.

**Open State Transition Logic:**

5. **Recovery Timer Management**: When transitioning to open state, start a recovery timer set to expire after `RecoveryTimeout` duration. This timer operates independently of request flow—it continues running even if no requests arrive. Timer implementation should be robust against system clock changes and process restarts.

6. **Timer Expiry Handling**: When the recovery timer expires, automatically transition to half-open state regardless of current request activity. This transition should reset internal state appropriately for recovery testing, including preparing the limited request allowance mechanism.

**Half-Open State Transition Logic:**

7. **Request Limiting Enforcement**: Before processing any request in half-open state, check if the maximum number of test requests are already in flight. If the limit is reached, immediately reject the request with a circuit open error without affecting the circuit state.

8. **Test Request Outcome Evaluation**: When a test request completes, evaluate its outcome using the same classification rules as closed state. However, the threshold for returning to open state is typically much lower—often any single failure triggers the transition.

9. **Recovery Success Handling**: If the test request succeeds, immediately transition to closed state and reset all failure counters. Update the last success timestamp and prepare for normal request processing. Some implementations require multiple consecutive successes before closing the circuit for more conservative recovery.

10. **Recovery Failure Handling**: If the test request fails, immediately transition back to open state and restart the recovery timer with the full `RecoveryTimeout` duration. This restart prevents rapid retry attempts against a service that hasn't actually recovered.

| Current State | Event Type | Condition | Next State | Actions Taken |
|---------------|------------|-----------|------------|---------------|
| Closed | Request Success | Any success | Closed | Reset failure counter, update last success time |
| Closed | Request Failure | Failure count < threshold | Closed | Increment failure counter |
| Closed | Request Failure | Failure count >= threshold | Open | Start recovery timer, record state transition |
| Open | Timer Expiry | Recovery timeout elapsed | Half-Open | Reset test request allowance |
| Open | Request Arrival | Timer still running | Open | Reject immediately with circuit open error |
| Half-Open | Request Success | Test request succeeds | Closed | Reset all counters, record recovery |
| Half-Open | Request Failure | Test request fails | Open | Restart recovery timer, record failed recovery |
| Half-Open | Request Arrival | Test request in flight | Half-Open | Reject with circuit open error |

### Architecture Decision Records

The circuit breaker state machine requires several critical design decisions that significantly impact performance, reliability, and operational behavior. Each decision involves trade-offs between responsiveness, accuracy, resource usage, and complexity.

> **Decision: Consecutive Failure Threshold vs. Failure Rate Threshold**
> - **Context**: Circuit breakers can open based on consecutive failures or failure rate over a time window. Consecutive counting is simpler but may not accurately represent service health during mixed success/failure patterns. Failure rate calculation requires sliding window tracking but provides more nuanced health assessment.
> - **Options Considered**:
>   1. Consecutive failure counting with simple integer threshold
>   2. Failure rate calculation over sliding time window
>   3. Hybrid approach with both consecutive and rate-based thresholds
> - **Decision**: Implement consecutive failure counting for Milestone 1, add sliding window failure rate for Milestone 2
> - **Rationale**: Consecutive failure counting provides immediate feedback when services completely fail, which is the most critical case for preventing cascade failures. Sliding window calculation adds complexity but enables more sophisticated failure detection for services with intermittent issues. Phased implementation allows learning core concepts before adding advanced features.
> - **Consequences**: Simple threshold enables rapid implementation and debugging. Sliding window addition in Milestone 2 requires careful integration with existing state machine logic and may require refactoring state transition conditions.

| Option | Pros | Cons |
|--------|------|------|
| Consecutive failures | Simple implementation, immediate response to total failures | May miss intermittent failure patterns, sensitive to timing |
| Failure rate window | Accurate health assessment, handles intermittent failures | Complex implementation, requires windowing logic |
| Hybrid approach | Covers both failure patterns | Increased complexity, potential for conflicting thresholds |

> **Decision: Synchronous vs. Asynchronous State Transitions**
> - **Context**: State transitions can occur synchronously during request processing or asynchronously via background goroutines. Synchronous transitions provide immediate response but may impact request latency. Asynchronous transitions reduce request processing time but introduce complexity in state consistency and timing.
> - **Options Considered**:
>   1. Fully synchronous transitions during request processing
>   2. Asynchronous transitions via background timer management
>   3. Hybrid with synchronous failure transitions, asynchronous recovery
> - **Decision**: Use hybrid approach—synchronous for failure-based transitions, asynchronous for timer-based recovery
> - **Rationale**: Failure detection requires immediate action to prevent cascade failures, making synchronous transitions essential for closed→open and half-open→open transitions. Recovery timer management (open→half-open) can safely run asynchronously since it doesn't depend on request outcomes and benefits from independent timing control.
> - **Consequences**: Hybrid approach provides optimal responsiveness for failure cases while maintaining clean timer semantics. Requires careful coordination between synchronous request handling and asynchronous timer goroutines.

| Transition Type | Approach | Rationale |
|-----------------|----------|-----------|
| Failure-triggered | Synchronous | Immediate protection, prevents cascade failures |
| Success-triggered | Synchronous | Fast recovery when service demonstrates health |
| Timer-triggered | Asynchronous | Independent of traffic patterns, clean timeout semantics |

> **Decision: Single Test Request vs. Multiple Test Requests in Half-Open State**
> - **Context**: Half-open state can allow one test request or multiple concurrent test requests to probe service recovery. Single request minimizes load on recovering service but may not adequately test capacity. Multiple requests provide better confidence in recovery but risk overwhelming fragile services.
> - **Options Considered**:
>   1. Single test request with immediate state transition
>   2. Multiple test requests with majority success requirement
>   3. Configurable test request count with success ratio threshold
> - **Decision**: Single test request for Milestone 1, configurable count for Milestone 2
> - **Rationale**: Single test request provides simplest implementation and safest recovery behavior for unstable services. Many production circuit breakers use this approach successfully. Configurable test count allows tuning for specific service characteristics in advanced scenarios.
> - **Consequences**: Conservative recovery approach may delay return to normal service but minimizes risk of re-triggering failures. Single request eliminates complex success ratio calculations and race conditions between multiple test requests.

> **Decision: Thread Safety Strategy**
> - **Context**: Circuit breakers must handle concurrent requests safely while maintaining performance. Options include coarse-grained locking, fine-grained locking, lock-free atomic operations, or single-threaded design with channels.
> - **Options Considered**:
>   1. Single mutex protecting all circuit breaker state
>   2. Separate locks for different state components (counters, timers, state)
>   3. Lock-free implementation using atomic operations
>   4. Single-threaded design with goroutine and channels
> - **Decision**: Single mutex for state machine with atomic operations for performance-critical counters
> - **Rationale**: Single mutex provides clear correctness guarantees and simple reasoning about state consistency. Atomic operations for frequently accessed counters (request count, failure count) reduce lock contention during normal operation. This approach balances safety, performance, and implementation complexity.
> - **Consequences**: Coarse-grained locking may create contention under high load but ensures correctness and simplifies debugging. Atomic counters provide good performance for metrics collection while maintaining safety.

### Common Pitfalls

Circuit breaker state machine implementation involves several subtle correctness issues that commonly trip up developers. Understanding these pitfalls and their solutions is essential for building reliable circuit breaker systems.

⚠️ **Pitfall: Race Condition in Failure Counter Updates**

The most common race condition occurs when multiple concurrent requests update the failure counter simultaneously, potentially causing the counter to exceed the threshold multiple times before any request can transition the circuit to open state. This happens when the failure counter increment and threshold check are not atomic operations.

```
// PROBLEMATIC: Non-atomic counter check
failures := atomic.AddInt64(&cb.failureCount, 1)
if failures >= cb.config.FailureThreshold {  // Race: another request might also see threshold breach
    cb.state = StateOpen  // Multiple requests might execute this
}
```

The race condition manifests as multiple goroutines simultaneously detecting threshold breach and attempting to transition to open state. While this doesn't usually cause incorrect final state, it can trigger multiple timer starts, metric updates, or notification callbacks.

**Solution**: Use compare-and-swap operations or protect the entire increment-and-check operation with a mutex to ensure atomic transition detection. The mutex approach is simpler and provides clearer correctness guarantees.

⚠️ **Pitfall: Failure Counter Not Reset on Success**

Many implementations incorrectly treat the failure counter as a cumulative total rather than a consecutive failure counter. This leads to circuit opening even during healthy operation when the total historical failure count exceeds the threshold, regardless of recent success patterns.

The error occurs when success handling only updates metrics without resetting the failure counter, causing transient failures separated by many successful requests to accumulate and eventually trigger circuit opening. This behavior violates the circuit breaker's purpose of detecting sustained failure patterns rather than isolated incidents.

**Solution**: Every successful request must atomically reset the consecutive failure counter to zero. This ensures that the counter only reflects recent consecutive failures rather than historical totals.

⚠️ **Pitfall: Timer Leaks in Open State Transitions**

Timer management bugs commonly occur when transitioning to open state multiple times or when transitioning out of open state before the timer fires. If timers aren't properly stopped and cleaned up, they can accumulate and eventually fire after the circuit has moved to other states, causing unexpected state transitions.

The problem manifests as circuits unexpectedly transitioning to half-open state even when they should be closed, or multiple timers firing and causing rapid state oscillations. Memory leaks can also occur if timer objects accumulate without proper cleanup.

**Solution**: Always stop existing timers before starting new ones, and track timer lifecycle carefully. Use timer cancellation patterns appropriate to your language's concurrency model.

⚠️ **Pitfall: Half-Open State Allowing Unlimited Concurrent Requests**

A common implementation error allows unlimited requests to proceed in half-open state rather than enforcing the test request limit. This defeats the purpose of cautious recovery testing and can overwhelm services that are still struggling to recover.

The issue occurs when developers implement half-open state as simply "allow some requests" without properly tracking and limiting concurrent test requests. Multiple requests then proceed simultaneously, and if they all fail, the circuit receives multiple failure signals that may cause race conditions in state transitions.

**Solution**: Implement proper request limiting in half-open state using semaphores, atomic counters, or similar concurrency control mechanisms. Typically, only one test request should be in flight at any time.

⚠️ **Pitfall: State Machine Doesn't Handle Timer Expiry During Half-Open**

Some implementations fail to properly handle the case where a recovery timer expires while the circuit is in half-open state with a test request in flight. The timer expiry handler may attempt to transition from open to half-open, but the circuit is already in half-open, creating undefined behavior.

This edge case can occur if the half-open test request takes longer than the recovery timeout period, or if multiple timer events fire due to improper timer cleanup. The result may be inconsistent internal state or multiple concurrent test requests.

**Solution**: Timer expiry handlers must check current state before attempting transitions. Only transition from open to half-open—if already in half-open or closed state, the timer should be ignored or cancelled.

⚠️ **Pitfall: Clock Changes Affecting Timer Behavior**

Circuit breaker implementations that use system wall clock time for recovery timers can behave unpredictably when system clocks change due to NTP adjustments, daylight saving time transitions, or manual time changes. Forward clock changes may cause premature recovery attempts, while backward changes may delay recovery indefinitely.

The problem particularly affects implementations that store absolute timestamps and compare against current time, rather than using relative timer mechanisms that are isolated from system clock changes.

**Solution**: Use monotonic clock sources for timer implementation that are unaffected by system clock adjustments. Most languages provide monotonic time functions specifically for measuring elapsed time intervals.

### Implementation Guidance

The circuit breaker state machine requires careful implementation of concurrent state management, timer handling, and atomic counter operations. The following guidance provides a complete foundation for implementing the state machine in Go.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| State Management | Mutex-protected struct | Lock-free atomic state machine |
| Timer Implementation | time.Timer with manual cleanup | Context-based cancellation |
| Counter Operations | sync/atomic package | Custom lock-free counters |
| Metrics Collection | Simple struct with mutex | Ring buffer with atomic operations |
| Configuration | Struct literals | Configuration validation framework |

#### Recommended Module Structure

```
internal/circuitbreaker/
  state.go                 ← State enumeration and String() methods
  circuit_breaker.go       ← Main CircuitBreaker implementation
  circuit_breaker_test.go  ← State machine tests
  config.go                ← Configuration validation and defaults
  metrics.go               ← Metrics collection and aggregation
  clock.go                 ← Clock interface for testability
  mock_clock.go            ← MockClock for deterministic testing
```

This structure separates concerns while keeping the state machine logic centralized in the main `CircuitBreaker` type. The clock abstraction enables deterministic testing of timer-based behavior.

#### Infrastructure Starter Code

**Clock Abstraction** (complete implementation):

```go
// clock.go
package circuitbreaker

import "time"

// Clock provides time operations for circuit breaker timing.
// This interface enables deterministic testing by allowing time control.
type Clock interface {
    Now() time.Time
    After(d time.Duration) <-chan time.Time
}

// RealClock implements Clock using standard time package.
type RealClock struct{}

func (RealClock) Now() time.Time {
    return time.Now()
}

func (RealClock) After(d time.Duration) <-chan time.Time {
    return time.After(d)
}
```

**Mock Clock for Testing** (complete implementation):

```go
// mock_clock.go
package circuitbreaker

import (
    "sync"
    "time"
)

// MockClock provides controllable time for testing circuit breaker timing behavior.
type MockClock struct {
    mu      sync.Mutex
    now     time.Time
    timers  []*mockTimer
}

type mockTimer struct {
    deadline time.Time
    ch       chan time.Time
    fired    bool
}

func NewMockClock(startTime time.Time) *MockClock {
    return &MockClock{now: startTime}
}

func (m *MockClock) Now() time.Time {
    m.mu.Lock()
    defer m.mu.Unlock()
    return m.now
}

func (m *MockClock) After(d time.Duration) <-chan time.Time {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    timer := &mockTimer{
        deadline: m.now.Add(d),
        ch:       make(chan time.Time, 1),
    }
    m.timers = append(m.timers, timer)
    return timer.ch
}

func (m *MockClock) AdvanceTime(d time.Duration) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    m.now = m.now.Add(d)
    
    for _, timer := range m.timers {
        if !timer.fired && !m.now.Before(timer.deadline) {
            timer.fired = true
            select {
            case timer.ch <- m.now:
            default:
            }
        }
    }
}
```

**State Enumeration** (complete implementation):

```go
// state.go
package circuitbreaker

// State represents circuit breaker state machine states.
type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

func (s State) String() string {
    switch s {
    case StateClosed:
        return "CLOSED"
    case StateOpen:
        return "OPEN"
    case StateHalfOpen:
        return "HALF_OPEN"
    default:
        return "UNKNOWN"
    }
}
```

#### Core Logic Skeleton Code

**Main CircuitBreaker Implementation**:

```go
// circuit_breaker.go
package circuitbreaker

import (
    "context"
    "errors"
    "sync"
    "sync/atomic"
    "time"
)

var (
    ErrCircuitOpen = errors.New("circuit breaker is open")
    ErrTimeout     = errors.New("request timeout")
)

// CircuitBreaker implements the three-state circuit breaker pattern.
type CircuitBreaker struct {
    // State management
    mu           sync.RWMutex
    state        State
    clock        Clock
    
    // Configuration
    config *Config
    
    // Counters (use atomic operations)
    failureCount    int64
    requestCount    int64
    successCount    int64
    
    // Timing
    stateTransitionTime time.Time
    recoveryTimer       *time.Timer
    
    // Half-open request limiting
    halfOpenRequests int64
    
    // Metrics
    metrics *Metrics
}

// NewCircuitBreaker creates a circuit breaker with the given configuration.
func NewCircuitBreaker(config *Config) *CircuitBreaker {
    if config == nil {
        config = DefaultConfig()
    }
    
    return &CircuitBreaker{
        state:               StateClosed,
        clock:               RealClock{},
        config:              config,
        stateTransitionTime: time.Now(),
        metrics:             NewMetrics(),
    }
}

// Execute runs the given function through the circuit breaker.
// Returns ErrCircuitOpen if the circuit is open.
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // TODO 1: Check if request should be allowed based on current state
    //         - Closed: allow all requests
    //         - Open: check if recovery timeout has expired
    //         - Half-open: check if test request slot available
    
    // TODO 2: If request not allowed, return ErrCircuitOpen immediately
    
    // TODO 3: Record request start time for latency measurement
    
    // TODO 4: Execute the function with timeout context if configured
    
    // TODO 5: Measure execution time and classify result (success/failure)
    
    // TODO 6: Update internal counters and metrics atomically
    
    // TODO 7: Check if state transition needed based on new counters
    //         - Closed → Open: failure count >= threshold
    //         - Half-open → Closed: test request succeeded  
    //         - Half-open → Open: test request failed
    
    // TODO 8: Return original function result
    
    panic("implement Execute method")
}

// canProceed determines if a request should be allowed in the current state.
func (cb *CircuitBreaker) canProceed() bool {
    // TODO 1: Read current state with appropriate locking
    
    // TODO 2: Handle closed state (always allow)
    
    // TODO 3: Handle open state (check recovery timer)
    //         - If recovery timeout expired, transition to half-open
    //         - Otherwise, deny request
    
    // TODO 4: Handle half-open state (check test request limit)
    //         - If test request already in flight, deny
    //         - Otherwise, allow and mark test request active
    
    panic("implement canProceed method")
}

// recordSuccess handles successful request outcomes.
func (cb *CircuitBreaker) recordSuccess(latency time.Duration) {
    // TODO 1: Update success counters atomically
    
    // TODO 2: Reset failure counter to zero (consecutive failures only)
    
    // TODO 3: Update metrics with timing information
    
    // TODO 4: If in half-open state, transition to closed
    
    // TODO 5: Update last success timestamp
    
    panic("implement recordSuccess method")
}

// recordFailure handles failed request outcomes.
func (cb *CircuitBreaker) recordFailure(err error) {
    // TODO 1: Increment failure counter atomically
    
    // TODO 2: Update failure metrics and timestamps
    
    // TODO 3: Check if failure threshold reached in closed state
    
    // TODO 4: If threshold reached, transition to open state
    
    // TODO 5: If in half-open state, transition back to open
    
    panic("implement recordFailure method")
}

// transitionToOpen changes state to open and starts recovery timer.
func (cb *CircuitBreaker) transitionToOpen() {
    // TODO 1: Acquire write lock for state change
    
    // TODO 2: Set state to StateOpen
    
    // TODO 3: Record state transition time
    
    // TODO 4: Start recovery timer using clock interface
    //         - Timer should fire after config.RecoveryTimeout
    //         - Timer callback should transition to half-open
    
    // TODO 5: Update metrics for state transition
    
    panic("implement transitionToOpen method")
}

// transitionToHalfOpen changes state to half-open for recovery testing.
func (cb *CircuitBreaker) transitionToHalfOpen() {
    // TODO 1: Acquire write lock for state change
    
    // TODO 2: Set state to StateHalfOpen
    
    // TODO 3: Reset half-open request counter
    
    // TODO 4: Record state transition time
    
    // TODO 5: Stop recovery timer if still active
    
    panic("implement transitionToHalfOpen method")
}

// transitionToClosed changes state to closed for normal operation.
func (cb *CircuitBreaker) transitionToClosed() {
    // TODO 1: Acquire write lock for state change
    
    // TODO 2: Set state to StateClosed
    
    // TODO 3: Reset all failure counters
    
    // TODO 4: Record state transition time
    
    // TODO 5: Stop any active timers
    
    panic("implement transitionToClosed method")
}

// State returns the current circuit breaker state thread-safely.
func (cb *CircuitBreaker) State() State {
    cb.mu.RLock()
    defer cb.mu.RUnlock()
    return cb.state
}

// Metrics returns a snapshot of current metrics.
func (cb *CircuitBreaker) Metrics() *Metrics {
    // TODO: Create metrics snapshot with current counter values
    panic("implement Metrics method")
}
```

#### Language-Specific Hints

**Go Concurrency Patterns:**
- Use `sync/atomic` for frequently accessed counters to reduce lock contention
- Prefer `sync.RWMutex` for state access since reads are more common than writes
- Use `time.Timer` with proper cleanup to avoid timer leaks
- Consider `context.WithTimeout` for request-level timeouts
- Use channels carefully in timer callbacks to avoid blocking

**Error Handling:**
- Define package-level error variables for common circuit breaker errors
- Use error wrapping to preserve original errors while adding circuit breaker context
- Consider error classification interfaces for configurable error handling

**Testing Considerations:**
- Always use the `Clock` interface in production code for testability
- Test timer behavior with `MockClock.AdvanceTime()`
- Use table-driven tests for state transition scenarios
- Include race condition tests with `go test -race`

#### Milestone Checkpoint

After implementing the state machine, verify the following behavior:

**Basic State Transitions:**
```bash
go test ./internal/circuitbreaker/ -v
```

Expected test coverage:
- Closed → Open transition on failure threshold
- Open → Half-open transition on timer expiry
- Half-open → Closed transition on successful test request
- Half-open → Open transition on failed test request

**Manual Verification Steps:**
1. Create a circuit breaker with low failure threshold (e.g., 2 failures)
2. Execute successful requests and verify they pass through
3. Execute failing requests and verify circuit opens after threshold
4. Verify requests are rejected immediately when circuit is open
5. Advance time past recovery timeout and verify circuit enters half-open
6. Execute test request and verify appropriate transition

**Signs of Implementation Issues:**
- **Race condition symptoms**: Inconsistent state transitions, panics under concurrent load
- **Timer leak symptoms**: Memory growth over time, multiple timer firings
- **Counter bug symptoms**: Circuit doesn't open despite failures, doesn't close after success
- **Deadlock symptoms**: Requests hang indefinitely, test timeouts

**Debugging Commands:**
```bash
# Run with race detector
go test -race ./internal/circuitbreaker/

# Run with verbose output and specific test
go test -v -run TestStateTransitions ./internal/circuitbreaker/

# Run benchmarks to check performance
go test -bench=. ./internal/circuitbreaker/


## Sliding Window Metrics

> **Milestone(s):** Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

### Mental Model: Hospital Patient Monitoring

Think of a sliding window metrics system as a hospital patient monitoring system. When a patient is connected to a heart monitor, the screen doesn't just show the current heartbeat - it displays a continuous graph of the last several minutes of heart activity. The monitor maintains a rolling window of recent heartbeats, constantly adding new data points while dropping the oldest ones as time progresses.

Just as medical professionals can quickly assess a patient's condition by looking at recent heart rhythm patterns rather than just the current moment, our circuit breaker uses sliding window metrics to assess the health of downstream services based on recent request patterns rather than just counting consecutive failures. If a patient's heart rate spikes for a few seconds but then returns to normal, that's different from a sustained irregular rhythm - similarly, a few failed requests mixed with successes indicates a different service health state than a sustained pattern of failures.

The sliding window continuously tracks request outcomes over time, automatically aging out old data as new requests arrive. This provides a more nuanced and accurate picture of service health, preventing false positives from transient failures while still detecting genuine service degradation patterns.

### Sliding Window Algorithm

The sliding window algorithm divides time into discrete buckets and tracks request metrics within each bucket. As time progresses, the window slides forward by discarding old buckets and creating new ones, maintaining a constant time span of historical data for failure rate calculations.

The core algorithm operates through time-based bucket management. The total window duration is divided into multiple smaller time buckets, each tracking request counts, success counts, failure counts, and latency statistics for requests that arrive during that bucket's time interval. For example, a 60-second window might be divided into 12 buckets of 5 seconds each, providing granular time-based tracking while maintaining reasonable memory usage.

When a new request outcome arrives, the algorithm first determines the current time bucket based on the request timestamp. If the request falls into an existing bucket, the algorithm updates that bucket's counters. If the request arrives after the newest bucket's time range, the algorithm rotates the window by creating a new bucket and potentially discarding the oldest bucket to maintain the fixed window size.

The failure rate calculation examines all active buckets within the sliding window timespan. The algorithm sums the success counts and failure counts across all valid buckets, then calculates the failure rate as the ratio of failures to total requests. Buckets that have aged beyond the window duration are excluded from calculations but may be retained briefly to handle clock skew or out-of-order request processing.

Bucket rotation follows a time-driven approach rather than request-driven rotation. Each bucket has explicit start and end timestamps, and the algorithm creates new buckets at regular intervals based on wall clock time. This ensures consistent time-based behavior regardless of request volume - a service receiving one request per hour and a service receiving thousands of requests per second both maintain accurate time-based windows.

The algorithm handles edge cases through careful timestamp management. When the system starts, it creates an initial bucket for the current time interval. During periods of no requests, buckets may be empty but still consume their time slots, preventing gaps in the timeline. When requests arrive after a period of inactivity, the algorithm may need to create multiple buckets to fill the time gap, ensuring the window accurately represents the passage of time.

| Algorithm Phase | Action | Data Updated | Time Complexity |
|-----------------|--------|--------------|-----------------|
| Bucket Selection | Find or create bucket for request timestamp | WindowBucket position field | O(1) with ring buffer |
| Counter Update | Increment success/failure counts in bucket | RequestCount, SuccessCount, FailureCount | O(1) |
| Window Rotation | Discard expired buckets, create new buckets | SlidingWindow buckets array | O(number of expired buckets) |
| Rate Calculation | Sum counters across active buckets | No data update, read-only | O(number of buckets in window) |
| Latency Tracking | Update average and maximum latency per bucket | AvgLatency, MaxLatency | O(1) with exponential moving average |

### Architecture Decision Records

> **Decision: Time-Based vs Request-Count-Based Windows**
> - **Context**: Circuit breaker needs to assess service health over time, but we must choose between sliding windows based on time duration vs. windows based on number of requests
> - **Options Considered**: 1) Request-count windows (last N requests), 2) Time-based windows (last T seconds), 3) Hybrid approach with both time and count limits
> - **Decision**: Time-based sliding windows with configurable bucket count
> - **Rationale**: Time-based windows provide consistent behavior regardless of request volume. A low-traffic service and high-traffic service both get evaluated over the same time period, preventing bias toward high-volume services. Request-count windows can be misleading - 100 failures in 1 second vs. 100 failures over 10 minutes represent very different service health states.
> - **Consequences**: Enables consistent cross-service health assessment but requires wall-clock time management and bucket rotation logic. Memory usage is predictable and bounded by window duration rather than request volume.

| Option | Pros | Cons |
|--------|------|------|
| Request-count windows | Simple to implement, no time management needed | Unfair to low-traffic services, 100 failures over 1 second vs 1 hour look identical |
| Time-based windows | Fair across different traffic patterns, realistic health assessment | Requires time management, bucket rotation complexity |
| Hybrid approach | Most flexible, can handle various scenarios | Increased complexity, more configuration parameters |

> **Decision: Bucket Count and Duration Trade-off**
> - **Context**: Time-based windows require dividing the window into buckets, but bucket count affects both memory usage and granularity of failure detection
> - **Options Considered**: 1) Few large buckets (6 x 10-second buckets for 60-second window), 2) Many small buckets (60 x 1-second buckets), 3) Configurable bucket size with sensible defaults
> - **Decision**: Default to 10 buckets per window with configurable bucket duration, minimum bucket size of 1 second
> - **Rationale**: 10 buckets provides sufficient granularity to detect failure patterns while keeping memory usage reasonable. Smaller buckets increase precision but consume more memory and CPU for bucket management. Larger buckets reduce precision and may delay failure detection.
> - **Consequences**: Good balance of precision vs. performance, but services with sub-second failure spikes might not be detected immediately. Configurable parameters allow tuning for specific service characteristics.

| Option | Memory Usage | Granularity | Detection Speed | Complexity |
|--------|-------------|-------------|-----------------|------------|
| 6 large buckets | Low | Coarse | Slower | Simple |
| 60 small buckets | High | Fine | Faster | Complex |
| 10 medium buckets | Medium | Good | Good | Moderate |

> **Decision: Ring Buffer vs. Dynamic Array for Bucket Storage**
> - **Context**: Sliding windows need efficient bucket storage that supports rotation as time progresses, with frequent additions and removals of time buckets
> - **Options Considered**: 1) Ring buffer with fixed array size, 2) Dynamic slice/array with append/removal, 3) Linked list with node allocation
> - **Decision**: Ring buffer with fixed-size array and rotating position pointer
> - **Rationale**: Ring buffer provides O(1) bucket access and rotation with predictable memory usage. Dynamic arrays require expensive copy operations when growing/shrinking. Linked lists have pointer overhead and poor cache locality. Ring buffer maps naturally to time-based bucket rotation.
> - **Consequences**: Excellent performance characteristics and predictable memory usage, but requires careful index arithmetic and wraparound handling. Bucket count must be determined at initialization time.

> **Decision: Failure Rate Calculation Precision**
> - **Context**: Failure rates need to be calculated from bucket data, but floating-point precision and integer overflow considerations affect accuracy and performance
> - **Options Considered**: 1) Float64 percentage calculation, 2) Fixed-point arithmetic with integer ratios, 3) Scaled integer percentages
> - **Decision**: Float64 percentage calculation with integer overflow protection
> - **Rationale**: Float64 provides sufficient precision for failure rate percentages while remaining simple to implement and debug. Fixed-point arithmetic adds complexity without significant benefits for this use case. Integer overflow protection prevents wraparound on high-volume services.
> - **Consequences**: Simple and accurate failure rate calculations, but requires careful handling of division by zero and integer overflow scenarios. Performance impact of floating-point arithmetic is negligible compared to I/O operations.

### Common Pitfalls

⚠️ **Pitfall: Bucket Rotation Race Conditions**
Multiple goroutines accessing the sliding window concurrently can create race conditions during bucket rotation. If one goroutine is reading bucket data for failure rate calculation while another goroutine is rotating buckets and updating the position pointer, the reader might access stale or inconsistent bucket data. This manifests as incorrect failure rates or occasional panics from array bounds violations. Use a read-write mutex to protect the entire sliding window structure, with read locks for failure rate calculations and write locks for bucket updates and rotation.

⚠️ **Pitfall: Clock Skew and Out-of-Order Requests**
When requests arrive with timestamps slightly in the past due to clock skew or network delays, they might be assigned to already-rotated buckets, causing the data to be lost or misattributed. This commonly occurs in distributed systems where different services have slight clock differences. Always allow a small grace period (e.g., one bucket duration) for late-arriving requests, and validate timestamps against reasonable bounds before bucket assignment.

⚠️ **Pitfall: Memory Leaks from Unbounded Bucket Creation**
During periods of clock adjustment or system time jumps, the bucket rotation logic might create excessive buckets to "catch up" to the current time, consuming unbounded memory. This can occur when system clocks are adjusted forward by hours or days. Implement maximum bucket creation limits per rotation operation and reasonable bounds checking on timestamp differences to prevent memory exhaustion.

⚠️ **Pitfall: Premature Bucket Expiration**
Aggressively discarding old buckets can lead to windows that are shorter than configured, especially during low-traffic periods. If buckets are discarded immediately when they age past the window duration, brief periods without requests can cause the effective window size to shrink. Maintain buckets for slightly longer than the window duration to ensure consistent window size, and only discard buckets when they're significantly older than the window boundary.

⚠️ **Pitfall: Integer Overflow in High-Volume Services**
Services processing millions of requests per hour can overflow 32-bit counters in window buckets, causing request counts to wrap around and produce incorrect failure rates. This particularly affects long-running services or systems with very high throughput. Use 64-bit integers for all counters (RequestCount, SuccessCount, FailureCount) and implement overflow detection to prevent silent wraparound errors.

⚠️ **Pitfall: Inconsistent Latency Calculations**
Calculating average latency by storing running sums can lead to precision loss and overflow issues. Storing every individual latency value for accurate calculation consumes excessive memory. Use exponential moving averages for latency tracking, which provide good approximations with constant memory usage and naturally weight recent observations more heavily than older ones.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Time Management | Go `time.Time` with manual bucket rotation | Custom `Clock` interface with mock support for testing |
| Bucket Storage | Fixed-size array with manual index management | Ring buffer abstraction with wraparound handling |
| Concurrency Control | Single `sync.RWMutex` for entire sliding window | Per-bucket locks with lock-free bucket selection |
| Metrics Calculation | On-demand calculation during failure rate queries | Background calculation with cached results |

#### Recommended File Structure

```
internal/slidingwindow/
├── sliding_window.go        ← SlidingWindow struct and core logic
├── window_bucket.go         ← WindowBucket struct and bucket operations  
├── metrics_calculator.go    ← Failure rate and latency calculations
├── ring_buffer.go          ← Ring buffer utilities and index management
├── sliding_window_test.go   ← Unit tests with mock clock
└── benchmark_test.go        ← Performance tests for high-volume scenarios
```

#### Infrastructure Starter Code

```go
// ring_buffer.go - Complete ring buffer utilities
package slidingwindow

import "time"

// RingBuffer provides efficient circular buffer operations for time-based buckets.
type RingBuffer struct {
    size     int
    position int
    mask     int // size must be power of 2, mask = size - 1
}

// NewRingBuffer creates a ring buffer with the specified size.
// Size must be a power of 2 for efficient modulo operations.
func NewRingBuffer(size int) *RingBuffer {
    if size&(size-1) != 0 {
        panic("ring buffer size must be power of 2")
    }
    return &RingBuffer{
        size: size,
        mask: size - 1,
    }
}

// CurrentIndex returns the current position in the ring buffer.
func (rb *RingBuffer) CurrentIndex() int {
    return rb.position & rb.mask
}

// AdvancePosition moves the ring buffer position forward by count steps.
func (rb *RingBuffer) AdvancePosition(count int) int {
    rb.position += count
    return rb.CurrentIndex()
}

// IndexForTime calculates the ring buffer index for a specific timestamp
// relative to the bucket duration and window start time.
func (rb *RingBuffer) IndexForTime(timestamp time.Time, windowStart time.Time, bucketDuration time.Duration) int {
    elapsed := timestamp.Sub(windowStart)
    bucketOffset := int(elapsed / bucketDuration)
    return (rb.position + bucketOffset) & rb.mask
}

// metrics_calculator.go - Complete metrics calculation utilities
package slidingwindow

import (
    "math"
    "time"
)

// MetricsSnapshot represents a point-in-time view of sliding window metrics.
type MetricsSnapshot struct {
    TotalRequests   int64
    SuccessRequests int64
    FailureRequests int64
    FailureRate     float64
    AverageLatency  time.Duration
    MaxLatency      time.Duration
    WindowStart     time.Time
    WindowEnd       time.Time
}

// CalculateExponentialMovingAverage updates an EMA with a new sample.
// Alpha should be between 0 and 1, with higher values giving more weight to recent samples.
func CalculateExponentialMovingAverage(currentEMA time.Duration, newSample time.Duration, alpha float64) time.Duration {
    if currentEMA == 0 {
        return newSample // First sample
    }
    
    currentFloat := float64(currentEMA)
    newFloat := float64(newSample)
    updated := alpha*newFloat + (1-alpha)*currentFloat
    
    return time.Duration(updated)
}

// SafeFailureRate calculates failure rate while handling division by zero.
func SafeFailureRate(failures, total int64) float64 {
    if total == 0 {
        return 0.0
    }
    return float64(failures) / float64(total)
}

// DetectCounterOverflow checks if adding a value would cause integer overflow.
func DetectCounterOverflow(current int64, increment int64) bool {
    return current > math.MaxInt64-increment
}
```

#### Core Logic Skeleton Code

```go
// sliding_window.go - Core sliding window implementation skeleton
package slidingwindow

import (
    "sync"
    "time"
)

// SlidingWindow tracks request metrics over a rolling time window using time-based buckets.
type SlidingWindow struct {
    buckets      []WindowBucket  // Ring buffer of time buckets
    bucketSize   time.Duration   // Duration of each individual bucket  
    windowSize   time.Duration   // Total duration of the sliding window
    position     int32           // Current position in ring buffer
    mutex        sync.RWMutex    // Protects concurrent access to buckets
    ringBuffer   *RingBuffer     // Ring buffer management utilities
    windowStart  time.Time       // Start time of the current window
    lastRotation time.Time       // Last time buckets were rotated
}

// NewSlidingWindow creates a new sliding window with the specified parameters.
func NewSlidingWindow(windowSize time.Duration, bucketCount int) *SlidingWindow {
    // TODO 1: Validate windowSize > 0 and bucketCount > 0
    // TODO 2: Calculate bucketSize as windowSize / bucketCount
    // TODO 3: Create ring buffer with bucketCount (ensure power of 2)
    // TODO 4: Initialize buckets slice with empty WindowBucket structs
    // TODO 5: Set windowStart to current time aligned to bucket boundary
    // TODO 6: Return initialized SlidingWindow
    return nil
}

// RecordRequest updates the sliding window with a new request outcome and latency.
func (sw *SlidingWindow) RecordRequest(success bool, latency time.Duration) {
    sw.mutex.Lock()
    defer sw.mutex.Unlock()
    
    now := time.Now()
    
    // TODO 1: Call rotateIfNeeded(now) to ensure current bucket exists
    // TODO 2: Find the appropriate bucket for the current timestamp
    // TODO 3: Increment the bucket's RequestCount (check for overflow)
    // TODO 4: If success is true, increment SuccessCount, otherwise increment FailureCount
    // TODO 5: Update bucket's latency metrics using exponential moving average
    // TODO 6: Update MaxLatency if current latency is higher
    // Hint: Use DetectCounterOverflow before incrementing counters
    // Hint: Use CalculateExponentialMovingAverage for AvgLatency updates
}

// FailureRate calculates the current failure rate across all buckets in the window.
// Returns (failureRate, totalRequests) where failureRate is between 0.0 and 1.0.
func (sw *SlidingWindow) FailureRate() (float64, int64) {
    sw.mutex.RLock()
    defer sw.mutex.RUnlock()
    
    now := time.Now()
    var totalRequests, totalFailures int64
    
    // TODO 1: Iterate through all buckets in the ring buffer
    // TODO 2: For each bucket, check if it falls within the current window timespan
    // TODO 3: Skip buckets that are older than windowSize from current time
    // TODO 4: Sum RequestCount and FailureCount from valid buckets
    // TODO 5: Calculate and return failure rate using SafeFailureRate
    // Hint: A bucket is valid if now.Sub(bucket.StartTime) <= sw.windowSize
    // Hint: Handle empty buckets (StartTime.IsZero()) by skipping them
    
    return 0.0, 0
}

// rotateIfNeeded creates new buckets if enough time has passed since last rotation.
// Must be called with write lock held.
func (sw *SlidingWindow) rotateIfNeeded(now time.Time) {
    // TODO 1: Calculate time elapsed since lastRotation
    // TODO 2: Determine how many new buckets need to be created (elapsed / bucketSize)
    // TODO 3: If no rotation needed (elapsed < bucketSize), return early
    // TODO 4: For each new bucket needed, advance ring buffer position
    // TODO 5: Initialize new bucket with StartTime and EndTime based on position
    // TODO 6: Clear counters and metrics for the new bucket
    // TODO 7: Update lastRotation time to now
    // Hint: Limit maximum buckets created per rotation to prevent memory exhaustion
    // Hint: New bucket StartTime should be aligned to bucket boundaries
}

// getCurrentBucket returns the bucket that should receive new requests at the given time.
// Must be called with lock held.
func (sw *SlidingWindow) getCurrentBucket(timestamp time.Time) *WindowBucket {
    // TODO 1: Calculate which bucket index corresponds to the timestamp
    // TODO 2: Use ring buffer IndexForTime method to find correct bucket
    // TODO 3: Check if the bucket is properly initialized for this timestamp
    // TODO 4: If bucket is stale, reinitialize it with new time boundaries
    // TODO 5: Return pointer to the appropriate bucket
    // Hint: Bucket is stale if its EndTime is before the timestamp
    return nil
}

// Snapshot returns a point-in-time copy of current sliding window metrics.
func (sw *SlidingWindow) Snapshot() *MetricsSnapshot {
    sw.mutex.RLock()
    defer sw.mutex.RUnlock()
    
    // TODO 1: Calculate current failure rate and total requests
    // TODO 2: Find maximum latency across all valid buckets
    // TODO 3: Calculate average latency weighted by request count
    // TODO 4: Determine window start and end times from bucket data
    // TODO 5: Create and return MetricsSnapshot with all calculated values
    // Hint: Weight average latency by each bucket's request count
    // Hint: Window start is the oldest valid bucket's StartTime
    
    return &MetricsSnapshot{}
}
```

#### Language-Specific Hints

**Time Management in Go:**
- Use `time.Now()` for current timestamp, but consider accepting a Clock interface for testability
- `time.Duration` arithmetic: `elapsed := now.Sub(startTime)`, `bucketCount := int(elapsed / bucketDuration)`
- Align timestamps to bucket boundaries: `aligned := timestamp.Truncate(bucketDuration)`
- Use `time.IsZero()` to check for uninitialized time values

**Concurrency with sync.RWMutex:**
- Use `RLock()` for read-only operations like failure rate calculation
- Use `Lock()` for write operations like recording requests or rotating buckets
- Always use `defer` to ensure locks are released: `defer sw.mutex.RUnlock()`
- Keep critical sections as small as possible to minimize lock contention

**Integer Overflow Protection:**
- Use `int64` for all counters to handle high-volume services
- Check overflow before incrementing: `if current > math.MaxInt64 - increment { /* handle overflow */ }`
- Consider using atomic operations for simple counter updates: `atomic.AddInt64(&counter, 1)`

**Ring Buffer Index Arithmetic:**
- Ensure ring buffer size is power of 2 for efficient modulo: `index = position & (size - 1)`
- Handle wraparound explicitly: `nextIndex := (currentIndex + 1) % bufferSize`
- Use bit masking instead of modulo for performance when size is power of 2

#### Milestone Checkpoints

**After implementing SlidingWindow struct and basic RecordRequest:**
```bash
go test -run TestSlidingWindowBasics ./internal/slidingwindow/
```
Expected output: Tests pass for basic request recording and bucket creation. Verify that recording 10 successful requests shows FailureRate() returning (0.0, 10).

**After implementing bucket rotation logic:**
```bash
go test -run TestBucketRotation ./internal/slidingwindow/
```
Expected output: Tests pass for time-based bucket rotation. Manually verify by recording requests, advancing mock clock by one bucket duration, recording more requests, and confirming failure rate includes both time periods.

**After implementing failure rate calculation:**
```bash
go test -run TestFailureRateCalculation ./internal/slidingwindow/
```
Expected output: Tests pass for various failure rate scenarios. Test edge cases: 100% failure rate (1.0, N), 0% failure rate (0.0, N), and empty window (0.0, 0).

**Performance validation:**
```bash
go test -bench=BenchmarkSlidingWindow ./internal/slidingwindow/
```
Expected output: RecordRequest operations should complete in under 1μs, FailureRate calculations under 10μs. Memory allocations should be minimal (< 100 bytes per operation).


## Fallback and Bulkhead System

> **Milestone(s):** Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

### Mental Model: Emergency Backup Systems

Think of the fallback and bulkhead system as the emergency infrastructure in a modern building. When the main elevator breaks down, backup elevators and emergency stairwells ensure people can still move between floors. When the main power grid fails, backup generators automatically kick in to keep critical systems running. Similarly, when a downstream service becomes unavailable, our fallback system provides alternative responses while the bulkhead system prevents resource exhaustion from cascading to other services.

The **fallback mechanism** acts like backup generators that automatically engage when primary power fails. Just as a hospital has multiple backup power sources for different criticality levels (life support gets the most reliable backup, while office lighting might use a simpler system), our fallback system provides different strategies based on service importance. A user profile service might fall back to cached data, while a payment service might gracefully decline with a "try again later" message rather than risk incorrect transactions.

The **bulkhead pattern** works like the watertight compartments in a ship's hull. If one compartment floods, the sealed bulkheads prevent water from spreading to other compartments, keeping the ship afloat. In our system, if one service consumes all available connection threads, bulkheads prevent it from starving other services of resources. Each service gets its own isolated resource pool, ensuring that a runaway service cannot bring down the entire system.

This dual system creates resilience through both **graceful degradation** (fallbacks provide alternative functionality) and **resource isolation** (bulkheads prevent cascade failures). Together, they transform a brittle system that fails completely into a robust system that degrades gracefully under stress.

### Fallback Strategy Types

The fallback system provides multiple strategies for maintaining service availability when circuits are open or services are degraded. Each strategy represents a different trade-off between consistency, performance, and user experience.

**Static Response Fallbacks** represent the simplest fallback strategy, returning predetermined responses when the primary service is unavailable. These work well for non-critical data that can tolerate staleness or default values. A user preferences service might return default theme settings when the preferences database is unavailable, allowing the application to continue functioning with reasonable defaults. The `FallbackRegistry` stores these static responses as functions that require no external dependencies.

| Fallback Type | Use Case | Consistency Level | Performance Impact | Implementation Complexity |
|---------------|----------|-------------------|-------------------|---------------------------|
| Static Response | Default values, error messages | None (predetermined data) | Minimal | Low |
| Cached Response | Previously successful results | Eventually consistent | Low | Medium |
| Alternative Service | Secondary data source | Depends on service | Medium | High |
| Degraded Functionality | Simplified operations | Partial | Variable | Medium |

**Cached Response Fallbacks** provide previously successful responses from a local cache when the circuit is open. This strategy maintains better user experience by serving stale but real data rather than generic defaults. The cache must be populated during healthy periods and requires expiration policies to prevent serving extremely outdated information. Implementation involves intercepting successful responses during closed circuit states and storing them in a local cache with configurable TTL values.

**Alternative Service Fallbacks** route requests to backup services or different endpoints when the primary service fails. This provides the highest consistency but requires maintaining multiple service implementations. A product catalog service might fall back from a fast cache service to a slower database query, or from a personalized recommendation service to a generic trending products service. These fallbacks require careful orchestration to prevent infinite fallback loops.

**Degraded Functionality Fallbacks** provide simplified versions of the requested operation when full functionality is unavailable. Rather than complete failure, the system provides partial results or simplified workflows. A search service might fall back from personalized AI-powered results to basic keyword matching, or a checkout process might fall back from real-time inventory validation to optimistic acceptance with later reconciliation.

The `FallbackRegistry` manages fallback function composition, allowing multiple fallback strategies to be chained together. When the primary service fails, the system attempts the first fallback. If that also fails, it proceeds to the next fallback in the chain. This creates a cascade of graceful degradation rather than immediate total failure.

> **Key Design Insight**: Fallback functions must be more reliable than the primary service they're replacing. A fallback that has the same failure modes as the primary service provides no resilience benefit and may actually increase system complexity without improving reliability.

**Function Composition and Registration** enables dynamic fallback configuration without code changes. Services register multiple fallback functions in priority order, and the circuit breaker system automatically invokes them when needed. The composition pattern allows building complex fallback strategies from simple, testable components.

### Bulkhead Pattern for Isolation

The bulkhead pattern prevents resource exhaustion in one service from affecting other services by providing isolated resource pools. This isolation operates at multiple levels: connection pools, thread pools, memory allocation, and CPU time slicing.

**Connection Pool Isolation** represents the most critical bulkhead implementation, ensuring that each downstream service gets a dedicated pool of connections. Without bulkheads, a slow or hanging service could consume all available connections, preventing requests to healthy services. Each service registered in the `ServiceRegistry` receives its own connection pool with configurable minimum and maximum sizes.

The bulkhead sizing calculation requires careful consideration of service-specific requirements and system-wide resource constraints. A high-volume, low-latency service might require a larger pool than a low-volume administrative service. The total of all pools must not exceed system connection limits, requiring coordination between bulkhead configurations.

| Resource Type | Isolation Method | Configuration Parameters | Monitoring Metrics |
|---------------|------------------|-------------------------|-------------------|
| HTTP Connections | Per-service pools | MinConnections, MaxConnections, IdleTimeout | ActiveConnections, WaitingRequests |
| Goroutines | Semaphore limiting | MaxConcurrentRequests | ActiveGoroutines, QueuedRequests |
| Memory | Request size limits | MaxRequestSize, MaxResponseSize | MemoryUsage, AllocationRate |
| CPU Time | Request timeouts | MaxExecutionTime | ProcessingTime, TimeoutRate |

**Concurrency Bulkheads** use semaphore-based limiting to prevent any single service from consuming all available goroutines or threads. Each service gets a configurable maximum number of concurrent requests it can process. When this limit is reached, additional requests are either queued (with queue size limits) or rejected immediately with a bulkhead capacity error.

The semaphore implementation must handle edge cases carefully. When a service becomes healthy again after being circuit-broken, there may be a surge of queued requests that could overwhelm the recovering service. The bulkhead system coordinates with the circuit breaker to gradually increase allowed concurrency during the half-open state.

**Memory and CPU Bulkheads** prevent resource exhaustion attacks or runaway processes from affecting other services. Memory bulkheads set maximum request and response sizes per service, rejecting oversized payloads before they consume system memory. CPU bulkheads use timeout mechanisms to prevent long-running requests from starving other services of processing time.

**Bulkhead Overflow Handling** defines behavior when resource limits are reached. The system can either queue requests (with bounded queues to prevent memory exhaustion), reject requests immediately with specific error codes, or route requests to fallback handlers. The choice depends on the service's tolerance for latency versus throughput requirements.

> **Critical Design Decision**: Bulkhead rejection should not trigger circuit breaker state changes, as rejections due to resource limits are different from service health issues. The error classification system must distinguish between bulkhead capacity errors and actual service failures.

### Architecture Decision Records

**Decision: Fallback Function Composition Strategy**
- **Context**: Services need multiple levels of fallback (cached data, alternative service, static response) but the circuit breaker shouldn't know about specific fallback implementations. We need a way to compose fallback strategies without tight coupling.
- **Options Considered**: 
  1. Single fallback function per service
  2. Chain of responsibility pattern with ordered fallback list
  3. Strategy pattern with runtime fallback selection
- **Decision**: Chain of responsibility with ordered fallback list
- **Rationale**: Provides maximum flexibility for different degradation strategies while maintaining simple circuit breaker logic. Each fallback can be tested independently, and the composition allows gradual degradation rather than binary success/failure. The ordered list makes fallback behavior predictable and debuggable.
- **Consequences**: Requires careful ordering of fallbacks from most reliable to least reliable. Fallback functions must handle their own timeouts to prevent blocking the fallback chain. Additional complexity in fallback registration and testing, but improved system resilience.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Single fallback | Simple implementation, minimal overhead | No graceful degradation levels, all-or-nothing | No |
| Chain of responsibility | Multiple degradation levels, composable, testable | More complex registration, potential for long chains | Yes |
| Runtime selection | Dynamic fallback choice based on context | Complex decision logic, harder to predict behavior | No |

**Decision: Bulkhead Resource Pool Management**
- **Context**: Each service needs isolated resource pools, but creating separate connection pools for every service could lead to resource waste and configuration complexity. We need to balance isolation with resource efficiency.
- **Options Considered**:
  1. Global shared pool with no isolation
  2. Per-service dedicated pools with fixed sizing
  3. Dynamic pool sizing with minimum guarantees and burst capacity
- **Decision**: Per-service dedicated pools with configurable sizing
- **Rationale**: Provides predictable isolation guarantees without the complexity of dynamic sizing. Fixed pools make capacity planning and monitoring straightforward. Services can be configured based on their specific needs rather than sharing unpredictable resources.
- **Consequences**: Requires careful initial sizing and monitoring to prevent resource waste. May need periodic rebalancing as service usage patterns change. More memory overhead than shared pools, but provides strong isolation guarantees.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Global shared pool | Simple, efficient resource usage | No isolation, cascade failures | No |
| Fixed per-service pools | Predictable isolation, simple configuration | Potential resource waste, requires careful sizing | Yes |
| Dynamic pool sizing | Efficient resource usage, adaptive | Complex implementation, unpredictable behavior | No |

**Decision: Fallback Error Propagation Strategy**
- **Context**: When a fallback function itself fails, we need to decide whether to try the next fallback, fail the entire request, or return a different error type. This affects both system reliability and debugging capability.
- **Options Considered**:
  1. Fail immediately on first fallback failure
  2. Try all fallbacks in sequence until one succeeds or all fail
  3. Classify fallback errors and decide based on error type
- **Decision**: Try all fallbacks in sequence with error classification
- **Rationale**: Maximizes system availability by exhausting all fallback options before failing. Error classification prevents infinite loops when fallbacks have fundamental issues (like configuration errors) that won't be resolved by trying more fallbacks.
- **Consequences**: Longer latency when multiple fallbacks fail, but higher overall availability. Requires careful error classification to avoid masking important failures. Need comprehensive logging to debug fallback chains.

**Decision: Bulkhead Overflow Queue Management**
- **Context**: When a service's bulkhead capacity is reached, incoming requests need handling. We can queue them, reject them immediately, or use hybrid approaches. This affects latency, memory usage, and system stability.
- **Options Considered**:
  1. Unbounded queues that accept all overflow requests
  2. Immediate rejection when capacity is reached
  3. Bounded queues with configurable size and timeout
- **Decision**: Bounded queues with configurable size and timeout
- **Rationale**: Provides a buffer for temporary spikes while preventing memory exhaustion. Configurable queue sizes allow tuning based on service characteristics. Timeouts prevent requests from waiting indefinitely in queues.
- **Consequences**: Requires queue size tuning for each service. Additional complexity in request lifecycle management and timeout handling. Better system stability under load but potential for increased latency during congestion.

### Common Pitfalls

⚠️ **Pitfall: Fallback Functions That Are Less Reliable Than Primary Service**

A common mistake is implementing fallback functions that depend on the same infrastructure as the primary service or have similar failure modes. For example, a fallback that queries a different table in the same database provides no protection against database failures. The fallback chain becomes a series of interdependent failures rather than independent alternatives.

This happens because developers focus on functional alternatives rather than fault isolation. A user service fallback that calls another microservice over the network has similar network, DNS, and infrastructure dependencies as the primary service. When the network partition occurs, both primary and fallback services become unavailable.

**Solution**: Design fallbacks with completely different failure modes. Use local caches, static responses, or alternative infrastructure. Cache-based fallbacks should use local storage, static response fallbacks should require no external dependencies, and alternative service fallbacks should use different networks, data centers, or cloud providers when possible.

⚠️ **Pitfall: Bulkhead Pool Sizes That Don't Account for Connection Lifecycle**

Developers often configure bulkhead pool sizes based on expected request rate without considering connection establishment time, keep-alive behavior, and connection reuse patterns. A service that processes 100 requests per second doesn't necessarily need 100 connections if connections are reused efficiently, but might need more during connection establishment phases.

This miscalculation leads to either resource waste (oversized pools) or unexpected connection exhaustion (undersized pools). The problem is particularly acute with services that have variable response times, where connections may be held longer than expected during slow periods.

**Solution**: Base pool sizing on concurrent connection requirements rather than request rate. Monitor actual connection usage patterns including establishment time, active connection duration, and keep-alive effectiveness. Start with conservative estimates and adjust based on observed metrics rather than theoretical calculations.

⚠️ **Pitfall: Fallback Chain Infinite Loops**

When fallback functions themselves invoke circuit-protected services, circular dependencies can create infinite fallback loops. Service A falls back to Service B, which falls back to Service C, which falls back to Service A. This creates stack overflow, resource exhaustion, or timeout cascades rather than graceful degradation.

The problem is often subtle, occurring through indirect dependencies. A user profile fallback might call a caching service, which itself has a fallback to the original user profile service. The circular dependency isn't obvious in the code but emerges from the service interaction graph.

**Solution**: Implement fallback depth tracking and maximum fallback chain length limits. Each fallback execution should increment a depth counter passed through the request context, and reject requests that exceed configured limits. Draw service dependency graphs including fallback relationships to identify potential cycles before deployment.

⚠️ **Pitfall: Bulkhead Rejection Errors Triggering Circuit State Changes**

A critical mistake is treating bulkhead capacity rejections as service health indicators that should influence circuit breaker state. When bulkhead queues fill up and requests are rejected due to capacity limits, this indicates resource pressure but not necessarily service health issues. Incorrectly counting these as failures can cause premature circuit opening.

This creates a feedback loop where high load causes bulkhead rejections, which are counted as failures, causing the circuit to open, which may cause upstream retries, increasing load and worsening the problem. The circuit breaker system amplifies load problems rather than providing protection.

**Solution**: Implement distinct error types for bulkhead capacity (`ErrBulkheadCapacity`) versus service failures (`ErrServiceUnavailable`). The error classification system should exclude bulkhead errors from circuit breaker failure counts. Monitor bulkhead utilization separately from service health metrics to distinguish between capacity and availability problems.

⚠️ **Pitfall: Fallback Functions Without Their Own Timeouts**

Fallback functions that don't implement their own timeouts can block the entire fallback chain, preventing execution of subsequent fallbacks and causing requests to hang. This is particularly problematic with alternative service fallbacks that make their own network calls without proper timeout configuration.

The issue becomes critical when the primary service is already experiencing latency problems. If fallbacks also become slow, the system degrades to worse performance than simply failing fast. Users experience longer response times than they would with immediate failure.

**Solution**: Every fallback function must implement its own timeout logic, typically shorter than the primary service timeout. Use context cancellation to enforce timeouts and ensure fallback functions respect context deadlines. Fallback timeouts should be aggressive since fallbacks are meant to provide fast alternatives, not slow alternatives.

⚠️ **Pitfall: Memory Leaks in Bulkhead Queue Management**

Improperly managed bulkhead queues can accumulate requests that are never processed or cleaned up, leading to memory leaks and eventual service degradation. This happens when requests are queued but the associated goroutines or cleanup callbacks are not properly managed during service shutdowns or configuration changes.

Queue cleanup is particularly challenging during service restart or reconfiguration scenarios. Queued requests may hold references to resources that should be garbage collected, preventing proper memory management and causing gradual memory exhaustion.

**Solution**: Implement explicit queue lifecycle management with proper cleanup on service shutdown. Use context cancellation to clean up queued requests when bulkhead configurations change. Monitor queue memory usage and implement automatic queue purging when memory thresholds are exceeded. Ensure all queued requests have expiration times to prevent indefinite accumulation.

### Implementation Guidance

![Fallback Execution Flow](./diagrams/fallback-execution.svg)

The fallback and bulkhead system requires careful integration between multiple components to provide both graceful degradation and resource isolation. This implementation guidance provides complete infrastructure code and detailed skeletons for the core learning components.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Semaphore | Go channels (buffered) | `golang.org/x/sync/semaphore` |
| Connection Pooling | `net/http.Transport` with MaxIdleConns | Custom pool with health checking |
| Queue Management | Go channels with timeouts | Priority queues with backpressure |
| Function Registry | Map with RWMutex | Registry with dependency injection |
| Metrics Collection | Atomic counters | Prometheus metrics integration |

**Recommended File Structure:**
```
internal/circuitbreaker/
  fallback/
    registry.go          ← fallback function management
    chain.go            ← fallback chain execution
    types.go            ← fallback function types
    fallback_test.go    ← fallback testing
  bulkhead/
    semaphore.go        ← semaphore-based limiting
    pool.go             ← resource pool management
    config.go           ← bulkhead configuration
    bulkhead_test.go    ← bulkhead testing
  integration/
    client_wrapper.go   ← HTTP client integration
    service_registry.go ← per-service circuit management
```

**Complete Infrastructure Code - Fallback Registry:**

```go
package fallback

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// FallbackFunc represents a fallback function that provides alternative responses
// when the primary service is unavailable.
type FallbackFunc func(ctx context.Context, originalError error) (interface{}, error)

// FallbackChain manages an ordered list of fallback functions for graceful degradation.
type FallbackChain struct {
    functions []FallbackFunc
    maxDepth  int
    timeout   time.Duration
}

// FallbackRegistry manages fallback chains for different services.
type FallbackRegistry struct {
    chains map[string]*FallbackChain
    mutex  sync.RWMutex
}

// NewFallbackRegistry creates a new fallback registry with default configuration.
func NewFallbackRegistry() *FallbackRegistry {
    return &FallbackRegistry{
        chains: make(map[string]*FallbackChain),
    }
}

// RegisterFallback adds a fallback chain for the specified service.
func (r *FallbackRegistry) RegisterFallback(serviceID string, chain *FallbackChain) {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    r.chains[serviceID] = chain
}

// GetFallbackChain retrieves the fallback chain for a service.
func (r *FallbackRegistry) GetFallbackChain(serviceID string) (*FallbackChain, bool) {
    r.mutex.RLock()
    defer r.mutex.RUnlock()
    chain, exists := r.chains[serviceID]
    return chain, exists
}

// NewFallbackChain creates a fallback chain with the specified functions and configuration.
func NewFallbackChain(functions []FallbackFunc, maxDepth int, timeout time.Duration) *FallbackChain {
    return &FallbackChain{
        functions: functions,
        maxDepth:  maxDepth,
        timeout:   timeout,
    }
}

// Execute runs the fallback chain, trying each function until one succeeds or all fail.
func (c *FallbackChain) Execute(ctx context.Context, originalError error) (interface{}, error) {
    // TODO: Implement fallback chain execution (students implement this)
    return nil, fmt.Errorf("fallback chain execution not implemented")
}
```

**Complete Infrastructure Code - Bulkhead Semaphore:**

```go
package bulkhead

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// Semaphore provides concurrency limiting with configurable capacity and timeouts.
type Semaphore struct {
    capacity    int
    permits     chan struct{}
    activeCount int64
    queuedCount int64
    mutex       sync.RWMutex
}

// NewSemaphore creates a semaphore with the specified capacity.
func NewSemaphore(capacity int) *Semaphore {
    return &Semaphore{
        capacity: capacity,
        permits:  make(chan struct{}, capacity),
    }
}

// Acquire attempts to acquire a permit, blocking until available or context timeout.
func (s *Semaphore) Acquire(ctx context.Context) error {
    // Increment queued count for monitoring
    s.incrementQueued()
    defer s.decrementQueued()
    
    select {
    case s.permits <- struct{}{}:
        s.incrementActive()
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

// Release returns a permit to the semaphore.
func (s *Semaphore) Release() {
    select {
    case <-s.permits:
        s.decrementActive()
    default:
        // Should never happen if used correctly
        panic("semaphore release without acquire")
    }
}

// Stats returns current semaphore utilization metrics.
func (s *Semaphore) Stats() (active int64, queued int64, capacity int) {
    s.mutex.RLock()
    defer s.mutex.RUnlock()
    return s.activeCount, s.queuedCount, s.capacity
}

func (s *Semaphore) incrementActive() {
    s.mutex.Lock()
    s.activeCount++
    s.mutex.Unlock()
}

func (s *Semaphore) decrementActive() {
    s.mutex.Lock()
    s.activeCount--
    s.mutex.Unlock()
}

func (s *Semaphore) incrementQueued() {
    s.mutex.Lock()
    s.queuedCount++
    s.mutex.Unlock()
}

func (s *Semaphore) decrementQueued() {
    s.mutex.Lock()
    s.queuedCount--
    s.mutex.Unlock()
}
```

**Core Logic Skeleton - Fallback Chain Execution:**

```go
// Execute runs the fallback chain, trying each function until one succeeds or all fail.
// This is the core learning component that students should implement.
func (c *FallbackChain) Execute(ctx context.Context, originalError error) (interface{}, error) {
    // TODO 1: Check if context already has a fallback depth counter
    // Hint: Use context.Value() with a custom key type
    
    // TODO 2: Extract current fallback depth from context, default to 0
    // Hint: Type assert the context value to int
    
    // TODO 3: Check if current depth exceeds maximum allowed depth
    // Return error if depth limit exceeded to prevent infinite loops
    
    // TODO 4: Increment fallback depth and add to context for nested calls
    // Hint: Use context.WithValue() to pass depth to fallback functions
    
    // TODO 5: Create timeout context for the entire fallback chain
    // Use c.timeout as the maximum time for all fallback attempts
    
    // TODO 6: Iterate through c.functions slice, trying each fallback
    // For each function: create individual timeout context, call function
    
    // TODO 7: For each fallback function call:
    // - Create context with shorter timeout (c.timeout / len(c.functions))
    // - Call function with timeout context and originalError
    // - If successful, return result immediately
    // - If failed, log error and continue to next fallback
    
    // TODO 8: If all fallbacks fail, return comprehensive error
    // Include original error and all fallback errors in error message
    
    // TODO 9: Add metrics recording for fallback execution
    // Record: fallback chain length, successful fallback index, total execution time
    
    // TODO 10: Handle context cancellation gracefully
    // Check ctx.Done() between fallback attempts and return ctx.Err() if cancelled
}
```

**Core Logic Skeleton - Bulkhead Pool Management:**

```go
// BulkheadPool manages resource isolation for a specific service.
type BulkheadPool struct {
    serviceID   string
    semaphore   *Semaphore
    config      BulkheadConfig
    metrics     *BulkheadMetrics
    registry    *ServiceRegistry
}

// BulkheadConfig defines resource limits and behavior for a service bulkhead.
type BulkheadConfig struct {
    MaxConcurrentRequests int
    QueueTimeout         time.Duration
    MaxQueueSize         int
    RejectOnCapacity     bool
}

// Execute runs a function with bulkhead protection, limiting concurrency per service.
func (p *BulkheadPool) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // TODO 1: Record request attempt in metrics
    // Increment total request count for this service's bulkhead
    
    // TODO 2: Try to acquire semaphore permit with context timeout
    // Use p.semaphore.Acquire(ctx) - this may block if capacity is full
    
    // TODO 3: Handle semaphore acquisition failure
    // If acquire fails due to timeout, return ErrBulkheadCapacity
    // This should NOT be counted as a service failure for circuit breaking
    
    // TODO 4: Ensure semaphore release in defer statement
    // Always release permit even if function panics or returns error
    
    // TODO 5: Record active request start time for latency metrics
    // Use time.Now() to capture start time before function execution
    
    // TODO 6: Execute the protected function
    // Call fn() and capture both result and error
    
    // TODO 7: Calculate and record execution latency
    // time.Since(startTime) gives execution duration for metrics
    
    // TODO 8: Update bulkhead metrics based on execution result
    // Record success/failure counts, latency distribution, capacity utilization
    
    // TODO 9: Return original function result and error
    // Don't modify the result - bulkhead is transparent to function outcomes
    
    // TODO 10: Handle panic recovery if function panics
    // Use recover() to catch panics, release semaphore, and re-panic
}
```

**Language-Specific Go Hints:**
- Use `context.WithValue()` for passing fallback depth through call chains
- Implement custom context key types to avoid key collisions: `type fallbackDepthKey struct{}`
- Use `atomic` package for lock-free metrics counters in high-throughput scenarios
- Leverage `defer` statements for resource cleanup even during panics
- Use `select` with `time.After()` for implementing timeouts in fallback chains
- Implement proper error wrapping with `fmt.Errorf("fallback failed: %w", err)` for error chains

**Milestone Checkpoint - Fallback System:**
After implementing the fallback registry and chain execution:
1. Run `go test ./internal/circuitbreaker/fallback/...` - all tests should pass
2. Create a simple HTTP service with primary endpoint that returns 500 errors
3. Register a fallback chain: cached response → static response → error response
4. Send requests and verify fallback chain executes in order when primary fails
5. Check logs show fallback depth tracking and prevent infinite loops
6. Expected behavior: First request fails to primary, tries cache (miss), returns static response

**Milestone Checkpoint - Bulkhead System:**
After implementing bulkhead pools and semaphore management:
1. Run `go test ./internal/circuitbreaker/bulkhead/...` - all tests should pass
2. Create two services with different bulkhead capacities (Service A: 2 concurrent, Service B: 5 concurrent)
3. Send 10 concurrent requests to Service A - expect 2 to proceed, 8 to wait or be rejected
4. Simultaneously send 5 requests to Service B - all should proceed independently
5. Monitor metrics showing per-service capacity utilization and queue depths
6. Expected behavior: Services operate with independent resource pools, no interference


## Client Integration Layer

> **Milestone(s):** Milestone 3 (Integration & Testing)

The client integration layer provides transparent circuit breaker functionality to existing HTTP clients and gRPC services without requiring application code changes. This layer acts as the bridge between the application's network communication logic and the circuit breaker's fault tolerance mechanisms, ensuring that all outbound service calls automatically benefit from failing fast, graceful degradation, and observability.

### Mental Model: Security Checkpoint

Think of the client integration layer as an airport security checkpoint that every passenger must pass through before boarding their flight. Just as security personnel inspect each passenger and their belongings according to established protocols, the client integration layer intercepts every outbound service request and evaluates it against the circuit breaker's health policies.

In this analogy, passengers represent service requests, security personnel represent the circuit breaker logic, and the boarding gates represent downstream services. The security checkpoint operates transparently - passengers follow normal boarding procedures but gain automatic security screening. Similarly, application code makes normal HTTP or gRPC calls but gains automatic fault tolerance.

Just as security checkpoints maintain separate lines for different flight destinations (domestic vs international), the client integration layer maintains separate circuit breakers for different downstream services. Each service gets its own "security line" with independently configured policies and state management. When a particular destination (service) experiences problems, only requests to that destination are affected - other services continue operating normally.

The checkpoint analogy extends to error handling: when security detects a problem, they have escalation procedures (fallback functions) and can redirect passengers to alternative flights (fallback responses). The checkpoint also maintains throughput metrics and can implement capacity controls during peak times (bulkhead pattern).

![Client Integration Architecture](./diagrams/integration-architecture.svg)

This mental model helps understand why the integration layer must be both transparent (passengers don't change their behavior) and comprehensive (every passenger gets screened). The layer intercepts requests at the network boundary, applies circuit breaker policies consistently, and provides observability into service health patterns.

### HTTP Client Integration

HTTP client integration wraps existing `http.Client` instances with circuit breaker functionality using the decorator pattern. This approach preserves the familiar HTTP client interface while adding fault tolerance capabilities that operate transparently behind the scenes.

The `ClientWrapper` structure encapsulates an original HTTP client along with a `ServiceRegistry` that manages circuit breakers for different downstream services. When the wrapped client receives a request, it first determines which service the request targets, retrieves or creates the appropriate circuit breaker, and then routes the request through the circuit breaker's state machine.

| Component | Type | Responsibility |
|-----------|------|----------------|
| `ClientWrapper` | struct | Decorates HTTP client with circuit breaker functionality |
| `ServiceRegistry` | interface | Maps service identifiers to circuit breaker instances |
| `ServiceIdentifier` | interface | Extracts service identity from HTTP requests |
| `RequestInterceptor` | interface | Processes requests before circuit breaker evaluation |
| `ResponseInterceptor` | interface | Processes responses after successful completion |

The HTTP integration follows a well-defined request processing sequence. When application code calls the wrapped client's `Do` method, the wrapper first extracts the service identifier from the request URL, headers, or custom routing logic. This identifier serves as the key for retrieving the appropriate circuit breaker from the service registry.

1. The wrapper receives an `http.Request` from application code through the standard `Do(req) (*http.Response, error)` method
2. It extracts the target service identifier using configurable extraction rules based on hostname, URL path patterns, or custom headers
3. It retrieves or creates a circuit breaker instance for that service from the service registry
4. It evaluates the circuit breaker state to determine if the request should proceed, fail immediately, or execute a fallback
5. For requests that proceed, it executes the original HTTP client's `Do` method and measures the response latency
6. It classifies the response as success or failure based on HTTP status codes and custom error classification rules
7. It records the outcome in the circuit breaker's metrics and updates the state machine accordingly
8. It returns the response to the application code, potentially enriched with circuit breaker metadata in custom headers

> **Decision: Service Identification Strategy**
> - **Context**: HTTP requests need mapping to specific circuit breaker instances for per-service isolation
> - **Options Considered**: 
>   - URL hostname extraction (simple but limited)
>   - URL path prefix matching (flexible but requires configuration)
>   - Custom header-based identification (explicit but requires client changes)
> - **Decision**: Hierarchical identification with hostname as primary and path patterns as secondary
> - **Rationale**: Balances automatic detection with configuration flexibility; most services are hostname-distinguishable
> - **Consequences**: Enables zero-configuration basic usage while supporting complex routing scenarios

| Service ID Strategy | Automatic Detection | Configuration Complexity | Routing Flexibility | Chosen |
|-------------------|-------------------|------------------------|-------------------|---------|
| Hostname Only | High | Low | Low | No |
| Path Patterns | Medium | Medium | High | Yes |
| Custom Headers | Low | Low | High | Fallback |

The wrapper handles HTTP-specific error classification by examining both transport-level errors (connection failures, timeouts) and application-level errors (HTTP status codes). Transport errors typically indicate circuit-opening conditions, while HTTP 4xx errors usually represent client mistakes that shouldn't affect circuit breaker state.

> **Design Insight**: HTTP integration must distinguish between circuit-relevant errors (network failures, 5xx responses) and pass-through errors (4xx client errors). A 404 Not Found shouldn't trip the circuit breaker, but a 503 Service Unavailable should contribute to failure counts.

Error classification follows a layered approach where transport errors (DNS resolution failures, connection timeouts) always count as circuit breaker failures. HTTP response codes undergo additional classification where 5xx server errors contribute to failure metrics, but 4xx client errors are treated as successful circuit interactions with application-level error responses.

| Error Category | HTTP Status Codes | Circuit Impact | Example Scenarios |
|---------------|------------------|----------------|------------------|
| Circuit Failures | 500, 502, 503, 504 | Increment failure count | Service overload, upstream timeouts |
| Pass-through Errors | 400, 401, 403, 404 | No circuit impact | Invalid requests, authorization failures |
| Ambiguous Errors | 429 (rate limiting) | Configurable | May indicate service stress or client behavior |
| Transport Failures | Connection errors | Always circuit failures | DNS failures, network partitions |

### gRPC Interceptor Integration

gRPC integration leverages the interceptor pattern to provide circuit breaker functionality for both unary and streaming RPC calls. Interceptors operate at the gRPC framework level, automatically applying circuit breaker logic to all outbound calls without requiring changes to service definitions or client code.

The `UnaryClientInterceptor` and `StreamClientInterceptor` functions implement the standard gRPC interceptor interfaces, allowing registration during client initialization. These interceptors operate similarly to HTTP middleware but handle gRPC-specific concerns like service method identification, error code classification, and streaming request management.

| Interceptor Type | gRPC Call Pattern | Circuit Breaker Application | Stream Handling |
|-----------------|------------------|---------------------------|-----------------|
| Unary Interceptor | Request-Response | Applied per method call | N/A |
| Stream Interceptor | Streaming calls | Applied at stream establishment | Monitors stream health |

gRPC service identification typically uses the fully qualified method name (package.Service/Method) to determine circuit breaker assignment. This approach provides natural service isolation since gRPC methods are organized by service definitions. The interceptor extracts the service portion of the method name to use as the circuit breaker key.

The unary interceptor follows a straightforward request-response pattern where each RPC call undergoes complete circuit breaker evaluation. The interceptor captures the method name, retrieves the appropriate circuit breaker, evaluates the current state, and either proceeds with the call or returns an immediate error.

1. The gRPC client framework calls the unary interceptor before sending the RPC request
2. The interceptor extracts the service identifier from the method name (e.g., "UserService" from "/api.UserService/GetUser")
3. It retrieves the circuit breaker instance for that service from the service registry
4. It evaluates the circuit state and either proceeds with the call or returns `ErrCircuitOpen`
5. For proceeding calls, it invokes the next interceptor in the chain and measures call latency
6. It examines the gRPC status code to classify the call as success or failure
7. It records the outcome in the circuit breaker metrics and updates state accordingly

> **Decision: gRPC Error Classification**
> - **Context**: gRPC status codes need mapping to circuit breaker failure conditions
> - **Options Considered**:
>   - All non-OK status codes count as failures (overly aggressive)
>   - Only UNAVAILABLE and DEADLINE_EXCEEDED count as failures (too restrictive)
>   - Configurable mapping based on status code categories (flexible but complex)
> - **Decision**: Default mapping with configuration override capability
> - **Rationale**: Provides sensible defaults while allowing service-specific customization
> - **Consequences**: Enables immediate usage with option for fine-tuning per service

| gRPC Status Code | Default Circuit Impact | Rationale | Override Option |
|-----------------|----------------------|-----------|-----------------|
| OK | Success | Successful completion | No |
| CANCELLED | Pass-through | Client-initiated cancellation | Yes |
| INVALID_ARGUMENT | Pass-through | Client error, not service failure | Yes |
| DEADLINE_EXCEEDED | Circuit failure | Service performance problem | No |
| NOT_FOUND | Pass-through | Valid response, resource absent | Yes |
| PERMISSION_DENIED | Pass-through | Authorization issue | Yes |
| RESOURCE_EXHAUSTED | Circuit failure | Service capacity problem | Yes |
| FAILED_PRECONDITION | Pass-through | Business logic constraint | Yes |
| UNAVAILABLE | Circuit failure | Service unavailable | No |
| INTERNAL | Circuit failure | Service implementation error | Yes |

Streaming RPC integration requires additional complexity since streams represent long-lived connections with multiple message exchanges. The stream interceptor applies circuit breaker logic at stream establishment time rather than per message, since breaking an active stream would disrupt ongoing communication.

The streaming interceptor monitors stream health by tracking stream establishment success and any stream-level errors that indicate service problems. Individual message failures within an established stream don't directly trigger circuit breaker state changes, but stream establishment failures and stream-level disconnections contribute to failure metrics.

### Per-Service Circuit Isolation

Per-service circuit isolation ensures that circuit breaker state and configuration remain independent for each downstream service, preventing failure in one service from affecting circuit breaker behavior for other services. This isolation is critical for maintaining service-level fault tolerance in microservices architectures where applications typically depend on multiple downstream services.

The `ServiceRegistry` manages the mapping between service identifiers and circuit breaker instances, creating new instances on demand and maintaining their lifecycle. Each service identifier gets its own `CircuitBreaker` instance with independent state, configuration, and metrics collection.

| Registry Component | Responsibility | Thread Safety | Lifecycle Management |
|-------------------|----------------|---------------|---------------------|
| Service Map | Maps service IDs to circuit breaker instances | Read-write mutex protected | Lazy creation, explicit cleanup |
| Configuration Provider | Supplies per-service configuration | Immutable configs | Reload support |
| Metrics Aggregator | Collects cross-service metrics | Lock-free collection | Periodic snapshot |
| Health Monitor | Tracks service-level health trends | Independent goroutines | Graceful shutdown |

Service isolation extends beyond just maintaining separate state machines to include independent configuration management. Each service can have different failure thresholds, recovery timeouts, and sliding window parameters based on its operational characteristics. High-traffic services might use different thresholds than low-traffic services, and critical services might have more aggressive recovery testing than optional services.

> **Design Insight**: Per-service isolation prevents "guilt by association" where one failing service causes circuit breakers for healthy services to become overly sensitive. Each service's circuit breaker learns its specific failure patterns independently.

The registry implements a lazy initialization pattern where circuit breaker instances are created only when first accessed. This approach avoids pre-allocating resources for services that may not be called during a particular application run, while ensuring consistent behavior when services are accessed.

Configuration inheritance provides a hierarchical approach to service-specific settings. The registry maintains a default configuration that applies to all services, with optional service-specific overrides that modify particular parameters. This design allows operators to set system-wide policies while fine-tuning behavior for specific services.

| Configuration Level | Scope | Override Priority | Example Parameters |
|--------------------|-------|------------------|------------------|
| Global Default | All services | Lowest | Failure threshold: 5, Recovery timeout: 30s |
| Service Family | Related services | Medium | Database services: timeout 10s, API services: timeout 5s |
| Service Specific | Individual service | Highest | payment-service: threshold 2, user-service: threshold 10 |

The service registry handles concurrent access safely by using read-write mutexes that allow multiple goroutines to retrieve existing circuit breakers simultaneously while serializing circuit breaker creation. This approach optimizes for the common case where circuit breakers already exist while ensuring thread-safe initialization of new instances.

### Architecture Decision Records

The client integration layer involves several critical design decisions that significantly impact usability, performance, and maintainability. These decisions balance the competing concerns of transparency, flexibility, and operational simplicity.

> **Decision: Integration Pattern Selection**
> - **Context**: Need to add circuit breaker functionality to existing HTTP and gRPC clients
> - **Options Considered**:
>   - Code generation approach modifying client stubs
>   - Proxy server intercepting network traffic
>   - Decorator pattern wrapping existing clients
> - **Decision**: Decorator pattern with interface preservation
> - **Rationale**: Minimal code changes, preserves existing client interfaces, enables gradual adoption
> - **Consequences**: Requires explicit wrapping but maintains full API compatibility

| Integration Option | Code Changes Required | Performance Overhead | Operational Complexity | Adoption Friction |
|-------------------|----------------------|---------------------|----------------------|------------------|
| Code Generation | High (regenerate stubs) | Lowest | Medium | High |
| Network Proxy | None (transparent) | Medium | High | Low |
| Decorator Pattern | Low (wrap clients) | Low | Low | Medium |

> **Decision: Service Identification Granularity**
> - **Context**: Determine the appropriate level of isolation for circuit breaker instances
> - **Options Considered**:
>   - Per-endpoint granularity (separate circuit per API method)
>   - Per-service granularity (separate circuit per downstream service)
>   - Per-cluster granularity (separate circuit per service cluster)
> - **Decision**: Per-service granularity with optional endpoint-level configuration
> - **Rationale**: Balances isolation effectiveness with resource usage and configuration complexity
> - **Consequences**: Enables service-level fault isolation while keeping circuit breaker count manageable

Per-endpoint granularity would provide the finest isolation but could result in hundreds of circuit breakers for large applications, making configuration and monitoring overwhelming. Per-cluster granularity would be too coarse, allowing failures in one service to affect circuit behavior for other services in the same cluster.

> **Decision: Configuration Management Strategy**
> - **Context**: Support both zero-configuration usage and advanced per-service customization
> - **Options Considered**:
>   - Static configuration files loaded at startup
>   - Dynamic configuration with runtime updates
>   - Hybrid approach with static defaults and dynamic overrides
> - **Decision**: Hybrid configuration with hierarchical inheritance
> - **Rationale**: Enables immediate usage with sensible defaults while supporting operational flexibility
> - **Consequences**: Adds configuration complexity but provides both ease of use and advanced capabilities

| Configuration Strategy | Startup Speed | Runtime Flexibility | Operational Overhead | Learning Curve |
|-----------------------|---------------|-------------------|---------------------|----------------|
| Static Only | Fast | None | Low | Low |
| Dynamic Only | Medium | High | High | High |
| Hybrid Approach | Fast | Medium | Medium | Medium |

> **Decision: Error Classification Approach**
> - **Context**: Determine which errors should contribute to circuit breaker failure counts
> - **Options Considered**:
>   - All errors count as failures (simple but inaccurate)
>   - Transport errors only (too restrictive)
>   - Configurable classification with sensible defaults (flexible but complex)
> - **Decision**: Layered classification with override capability
> - **Rationale**: Provides accurate failure detection out of the box with customization for edge cases
> - **Consequences**: Requires maintaining error classification logic but enables precise failure detection

The layered approach treats transport-level errors (connection failures, DNS resolution failures) as always contributing to circuit breaker metrics, while application-level errors (HTTP status codes, gRPC status codes) undergo additional classification based on whether they indicate service health problems or client usage errors.

### Common Pitfalls

The client integration layer presents several areas where implementation mistakes can significantly impact system behavior. Understanding these pitfalls helps avoid subtle bugs that can compromise fault tolerance effectiveness.

⚠️ **Pitfall: Service Identification Inconsistency**

A common mistake involves inconsistent service identification logic where the same downstream service receives different service identifiers depending on request routing or load balancing. This creates multiple circuit breaker instances for the same logical service, preventing proper failure detection and state management.

For example, if requests to `api.example.com` and `api-prod.example.com` both target the same service but receive different service identifiers, each hostname gets its own circuit breaker. When the service experiences problems, failures get distributed across multiple circuit breakers, preventing any single circuit from reaching its failure threshold.

The fix requires normalizing service identifiers to canonical forms. Implement identifier normalization logic that maps multiple hostnames, IP addresses, or routing variations to a single service identifier. Consider using service discovery information or configuration mappings to ensure consistent identification.

⚠️ **Pitfall: Circuit Breaker Per Request Instance**

Another critical mistake involves creating new circuit breaker instances for each request instead of reusing instances per service. This typically happens when circuit breaker creation is embedded in request handling logic without proper caching or registry usage.

When each request gets its own circuit breaker, failure detection becomes impossible since no single circuit breaker accumulates enough failures to trigger state transitions. Each circuit breaker sees at most one request, preventing the failure threshold from being reached regardless of service health.

The solution requires implementing proper circuit breaker lifecycle management through a service registry. Circuit breakers must be created once per service and reused across all requests to that service. Use the registry pattern to ensure singleton behavior per service identifier.

⚠️ **Pitfall: Thread Safety Violations in Service Registry**

Service registry implementations often contain race conditions when multiple goroutines attempt to create circuit breakers for the same service simultaneously. This can result in multiple circuit breaker instances for the same service or corruption of the service-to-circuit-breaker mapping.

The race condition typically occurs in the "check then create" pattern where code checks if a circuit breaker exists, and if not, creates a new one. Between the check and create operations, another goroutine might create a circuit breaker for the same service, leading to multiple instances.

Proper synchronization requires using mutex protection around the entire check-and-create operation, or implementing atomic operations like compare-and-swap. The registry should use read-write mutexes to allow concurrent reads while protecting writes.

⚠️ **Pitfall: Ignoring Context Cancellation**

HTTP and gRPC clients often ignore context cancellation signals when implementing circuit breaker integration, leading to resource leaks and delayed error reporting. When client code cancels a request context, the circuit breaker integration should respect the cancellation and avoid counting cancelled requests as failures.

Context cancellation represents client-side request abandonment rather than service-side failures. Counting cancelled requests as circuit breaker failures can cause circuits to open inappropriately when clients are impatient or implement aggressive timeouts.

The fix requires checking context cancellation before and during circuit breaker evaluation. If the context is already cancelled, return immediately with context.Canceled error. If cancellation occurs during request execution, classify the result as a cancellation rather than a failure.

⚠️ **Pitfall: Middleware Ordering Dependencies**

When integrating circuit breakers with other middleware (authentication, logging, retry logic), incorrect ordering can cause interference between different middleware components. Circuit breakers should typically operate close to the network transport to capture all failure modes.

For example, placing retry middleware after circuit breaker middleware means that retry attempts get blocked when the circuit is open, preventing the retry logic from helping with transient failures. Conversely, placing circuit breakers after retry middleware means the circuit breaker doesn't see individual failure attempts, only the final retry result.

The correct ordering depends on the desired behavior, but generally follows: authentication → circuit breaker → retry → transport. This ensures authenticated requests get circuit breaker protection, circuit breakers see individual failure attempts for accurate failure rate calculation, and retries can help with transient failures when circuits are closed.

⚠️ **Pitfall: Configuration Hot Reload Race Conditions**

Dynamic configuration updates can introduce race conditions when new configuration is applied while requests are in flight. Configuration changes might affect circuit breaker behavior inconsistently, with some requests using old configuration and others using new configuration.

The race condition is particularly problematic for threshold changes where lowering the failure threshold might immediately trigger circuit opening based on existing failure counts, or raising thresholds might prevent circuits from opening when they should.

Safe configuration updates require coordination with ongoing request processing. Implement configuration versioning where each request uses a consistent configuration snapshot throughout its lifecycle. Consider gradual configuration rollout where changes take effect over multiple evaluation cycles rather than immediately.

### Implementation Guidance

The client integration layer bridges high-level circuit breaker concepts with practical HTTP and gRPC client usage. This implementation focuses on creating decorator patterns that preserve existing client interfaces while adding transparent circuit breaker functionality.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HTTP Client Integration | `net/http` with wrapper struct | `net/http` with custom RoundTripper |
| gRPC Integration | Unary interceptors only | Unary + streaming interceptors |
| Service Registry | In-memory map with mutex | Distributed registry with etcd |
| Configuration Management | YAML files | Dynamic config with hot reload |
| Metrics Collection | Simple counters | Prometheus metrics with labels |

**Recommended File Structure:**

```
circuit-breaker/
  pkg/integration/
    http/
      client_wrapper.go      ← HTTP client decorator
      service_extractor.go   ← Service ID extraction
      error_classifier.go    ← HTTP error classification
      client_wrapper_test.go ← Integration tests
    grpc/
      interceptors.go        ← Unary and stream interceptors
      status_classifier.go   ← gRPC status classification
      interceptors_test.go   ← gRPC integration tests
    registry/
      service_registry.go    ← Circuit breaker instance management
      config_provider.go     ← Per-service configuration
      registry_test.go       ← Registry tests
  examples/
    http_example/
      main.go               ← HTTP client example
    grpc_example/
      main.go               ← gRPC client example
      proto/                ← Example service definitions
```

**Infrastructure Code - Service Registry:**

```go
package registry

import (
    "sync"
    "time"
)

// ServiceRegistry manages circuit breaker instances per service.
// Thread-safe for concurrent access from multiple goroutines.
type ServiceRegistry struct {
    circuits      map[string]*CircuitBreaker
    configs       map[string]*Config
    defaultConfig *Config
    mutex         sync.RWMutex
    metrics       *MetricsCollector
}

// NewServiceRegistry creates a new registry with default configuration.
func NewServiceRegistry(defaultConfig *Config) *ServiceRegistry {
    return &ServiceRegistry{
        circuits:      make(map[string]*CircuitBreaker),
        configs:       make(map[string]*Config),
        defaultConfig: defaultConfig,
        metrics:       NewMetricsCollector(),
    }
}

// GetOrCreateCircuit returns the circuit breaker for a service, creating if needed.
// Thread-safe and ensures only one circuit breaker per service ID.
func (r *ServiceRegistry) GetOrCreateCircuit(serviceID string) *CircuitBreaker {
    // Fast path: read lock for existing circuits
    r.mutex.RLock()
    if circuit, exists := r.circuits[serviceID]; exists {
        r.mutex.RUnlock()
        return circuit
    }
    r.mutex.RUnlock()
    
    // Slow path: write lock for circuit creation
    r.mutex.Lock()
    defer r.mutex.Unlock()
    
    // Double-check pattern: another goroutine might have created it
    if circuit, exists := r.circuits[serviceID]; exists {
        return circuit
    }
    
    // Create new circuit breaker with service-specific config
    config := r.getConfigForService(serviceID)
    circuit := NewCircuitBreaker(config)
    r.circuits[serviceID] = circuit
    
    // Register with metrics collector
    r.metrics.RegisterCircuit(serviceID, circuit)
    
    return circuit
}

// SetServiceConfig updates configuration for a specific service.
func (r *ServiceRegistry) SetServiceConfig(serviceID string, config *Config) {
    r.mutex.Lock()
    defer r.mutex.Unlock()
    r.configs[serviceID] = config
    
    // Update existing circuit if it exists
    if circuit, exists := r.circuits[serviceID]; exists {
        circuit.UpdateConfig(config)
    }
}

func (r *ServiceRegistry) getConfigForService(serviceID string) *Config {
    if config, exists := r.configs[serviceID]; exists {
        return config
    }
    return r.defaultConfig
}
```

**Infrastructure Code - HTTP Client Wrapper:**

```go
package http

import (
    "context"
    "net/http"
    "time"
)

// ClientWrapper decorates an HTTP client with circuit breaker functionality.
// Preserves the standard http.Client interface for drop-in replacement.
type ClientWrapper struct {
    client           *http.Client
    registry         *ServiceRegistry
    serviceExtractor ServiceExtractor
    errorClassifier  ErrorClassifier
    metrics          *MetricsCollector
}

// ServiceExtractor determines service ID from HTTP requests.
type ServiceExtractor interface {
    ExtractServiceID(req *http.Request) string
}

// ErrorClassifier determines if HTTP responses should count as circuit failures.
type ErrorClassifier interface {
    IsCircuitFailure(resp *http.Response, err error) bool
}

// NewClientWrapper creates a circuit breaker enabled HTTP client.
func NewClientWrapper(client *http.Client, registry *ServiceRegistry) *ClientWrapper {
    return &ClientWrapper{
        client:           client,
        registry:         registry,
        serviceExtractor: NewHostnameExtractor(),
        errorClassifier:  NewStandardErrorClassifier(),
        metrics:          NewMetricsCollector(),
    }
}

// Do executes HTTP requests through circuit breaker protection.
// Maintains the same signature as http.Client.Do for compatibility.
func (w *ClientWrapper) Do(req *http.Request) (*http.Response, error) {
    serviceID := w.serviceExtractor.ExtractServiceID(req)
    circuit := w.registry.GetOrCreateCircuit(serviceID)
    
    startTime := time.Now()
    
    // Execute request through circuit breaker
    result, err := circuit.Execute(req.Context(), func(ctx context.Context) (interface{}, error) {
        // Create new request with potentially modified context
        reqWithContext := req.WithContext(ctx)
        return w.client.Do(reqWithContext)
    })
    
    latency := time.Since(startTime)
    
    // Record metrics regardless of outcome
    w.metrics.RecordRequest(serviceID, latency, err == nil)
    
    if err != nil {
        return nil, err
    }
    
    resp := result.(*http.Response)
    
    // Check if response should be classified as circuit failure
    if w.errorClassifier.IsCircuitFailure(resp, nil) {
        // Record as failure but return the response
        circuit.recordFailure(NewHTTPError(resp.StatusCode, resp.Status))
    }
    
    return resp, nil
}
```

**Core Logic Skeleton - Service Extractor:**

```go
// HostnameExtractor extracts service ID from request hostname.
// Implements the ServiceExtractor interface for basic service identification.
type HostnameExtractor struct {
    portStripMode bool
    normalizeFunc func(string) string
}

// ExtractServiceID returns the hostname as the service identifier.
func (e *HostnameExtractor) ExtractServiceID(req *http.Request) string {
    // TODO 1: Get hostname from request URL
    hostname := req.URL.Hostname()
    
    // TODO 2: Handle port stripping if configured
    // Hint: Use strings.Split or net.SplitHostPort
    
    // TODO 3: Apply normalization function if configured
    // Hint: Convert to lowercase, remove common prefixes
    
    // TODO 4: Handle empty hostname (relative URLs)
    // Default to "localhost" or extract from Host header
    
    // TODO 5: Return normalized service identifier
    return "" // Replace with actual implementation
}
```

**Core Logic Skeleton - gRPC Interceptors:**

```go
package grpc

import (
    "context"
    "google.golang.org/grpc"
    "time"
)

// UnaryClientInterceptor creates a circuit breaker interceptor for unary RPCs.
func UnaryClientInterceptor(registry *ServiceRegistry) grpc.UnaryClientInterceptor {
    return func(ctx context.Context, method string, req, reply interface{}, 
                cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
        
        // TODO 1: Extract service ID from method name
        // Hint: Parse "/package.Service/Method" format
        serviceID := extractServiceFromMethod(method)
        
        // TODO 2: Get or create circuit breaker for service
        circuit := registry.GetOrCreateCircuit(serviceID)
        
        // TODO 3: Execute RPC through circuit breaker
        startTime := time.Now()
        _, err := circuit.Execute(ctx, func(ctx context.Context) (interface{}, error) {
            // TODO 4: Call the actual RPC method
            // Hint: Use the invoker parameter with same arguments
            return nil, invoker(ctx, method, req, reply, cc, opts...)
        })
        
        // TODO 5: Record latency and outcome metrics
        latency := time.Since(startTime)
        // Hint: Use gRPC status codes for error classification
        
        return err
    }
}

func extractServiceFromMethod(method string) string {
    // TODO 1: Parse method string format "/package.Service/Method"
    // TODO 2: Extract the Service portion
    // TODO 3: Handle invalid format gracefully
    return "" // Replace with implementation
}
```

**Language-Specific Hints:**

- Use `sync.RWMutex` for service registry to allow concurrent reads with exclusive writes
- Implement `http.RoundTripper` interface for more advanced HTTP client integration
- Use `context.WithTimeout` to enforce request timeouts within circuit breaker execution
- Leverage `google.golang.org/grpc/status` package for gRPC error classification
- Use `sync.Once` for lazy initialization of expensive resources like metrics collectors
- Consider using `sync.Map` for high-concurrency service registries with mostly reads

**Milestone Checkpoint:**

After implementing the client integration layer:

1. **HTTP Integration Test**: Run `go test ./pkg/integration/http/...` - should show all tests passing
2. **gRPC Integration Test**: Run `go test ./pkg/integration/grpc/...` - should show interceptor tests passing  
3. **Manual HTTP Test**: Start example server with `go run examples/http_example/main.go`
   - Make requests to different endpoints
   - Verify separate circuit breakers for different services in metrics output
   - Trigger failures and confirm circuit opening behavior
4. **Service Isolation Test**: Create requests to multiple services and verify independent circuit states
5. **Performance Benchmark**: Run `go test -bench=. ./pkg/integration/...` - wrapper overhead should be <1ms

Expected behavior indicators:
- HTTP requests automatically get circuit breaker protection without code changes
- Different hostnames/services maintain independent circuit breaker states  
- Circuit breaker metrics show per-service failure counts and state transitions
- gRPC calls respect circuit breaker state and record appropriate metrics
- Configuration changes apply to new requests without restart

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| All services share same circuit | Service ID extraction returns constant | Log service IDs from requests | Fix extractor logic |
| Circuits never open despite failures | Error classifier not detecting failures | Check HTTP status/gRPC codes | Update classification rules |
| Memory usage grows over time | Service registry not cleaning up unused circuits | Monitor registry size | Implement circuit cleanup |
| Inconsistent circuit behavior | Race conditions in registry | Run with race detector | Add proper synchronization |
| Configuration not taking effect | Config loaded after circuit creation | Check config loading order | Implement hot reload |


## Interactions and Data Flow

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

The interactions and data flow section reveals how all the circuit breaker components work together to process requests, classify errors, collect metrics, and coordinate state transitions. Understanding these flows is crucial for implementing a robust circuit breaker system that correctly handles the complex orchestration between state management, metrics collection, and recovery testing.

### Mental Model: Airport Security Checkpoint

Think of the circuit breaker request processing flow as an airport security checkpoint system. Just as passengers flow through multiple stations - check-in, security screening, gate access - each with its own rules and decision points, requests flow through the circuit breaker system encountering various checkpoints that determine their fate.

The security checkpoint analogy helps us understand several key concepts:
- **Pre-screening (Bulkhead Check)**: Like checking if the security line is too crowded, the bulkhead pattern ensures we don't overwhelm downstream services
- **Security State (Circuit State)**: Similar to how security levels (normal, elevated, high) affect processing, circuit states determine request handling
- **Screening Process (Request Execution)**: Each request undergoes evaluation, just like passengers go through metal detectors
- **Classification (Error Analysis)**: Security incidents are classified by severity, similar to how we classify errors as circuit-affecting or pass-through
- **Monitoring (Metrics Collection)**: Security checkpoints continuously monitor throughput and incidents, just like our metrics system tracks success and failure rates
- **Recovery Procedures (Half-Open Testing)**: When security issues resolve, checkpoints gradually return to normal operations, mirroring our recovery testing process

### Request Processing Flow

The request processing flow represents the core orchestration logic that determines how each incoming request traverses the circuit breaker system. This flow must handle multiple concurrent requests while maintaining thread safety and ensuring consistent state transitions.

![Request Processing Sequence](./diagrams/request-flow.svg)

The request processing begins when a client calls the `Execute` method on a `CircuitBreaker` instance. The flow follows a sophisticated decision tree that considers current circuit state, bulkhead capacity, error classification rules, and metrics collection requirements.

#### Closed State Request Processing

When the circuit breaker operates in `StateClosed`, requests flow through the normal execution path with full monitoring and metrics collection. The process follows these detailed steps:

1. **Request Arrival and Initial Validation**: The `Execute` method receives a context and function to execute. It immediately checks if the context has already been cancelled or has exceeded its deadline. This early validation prevents unnecessary processing of already-doomed requests.

2. **Bulkhead Capacity Check**: Before proceeding to the downstream service, the system checks if the bulkhead semaphore has available permits. The `Semaphore.Acquire` method attempts to obtain a permit within the context timeout. If no permits are available, the request fails immediately with `ErrBulkheadCapacity`.

3. **Pre-Execution Metrics Recording**: The system records the request attempt in the sliding window metrics. This involves calling `SlidingWindow.rotateIfNeeded` to ensure we're writing to the current time bucket, then incrementing the request count in the appropriate `WindowBucket`.

4. **Downstream Service Invocation**: The protected function executes with the provided context. The system measures execution latency using the `Clock.Now` method at both the start and completion of the operation.

5. **Success Path Processing**: If the function completes without error, the system calls `recordSuccess` which updates both the immediate metrics counters and the sliding window buckets. The success count increments, and the latency measurement is recorded for performance monitoring.

6. **Bulkhead Resource Release**: Regardless of success or failure, the bulkhead semaphore permit is released using `Semaphore.Release` to ensure resource availability for subsequent requests.

#### Open State Request Processing

When the circuit operates in `StateOpen`, the request processing follows a dramatically different path focused on failing fast and providing fallback responses:

1. **Immediate Rejection Check**: Upon entering the `Execute` method, the circuit breaker checks its current state. If the state is `StateOpen`, it immediately proceeds to the failure handling path without attempting downstream service invocation.

2. **Recovery Timer Evaluation**: Before rejecting the request, the system checks if the recovery timeout has elapsed since the transition to open state. This involves comparing the current time from `Clock.Now` with the `stateTransitionTime` plus the configured `RecoveryTimeout`.

3. **Automatic State Transition**: If the recovery timeout has elapsed, the circuit automatically transitions to `StateHalfOpen` using the `transitionToHalfOpen` method. This transition resets the success counter and allows limited recovery testing.

4. **Fallback Execution**: For rejected requests, the system queries the `FallbackRegistry` to find registered fallback functions for the target service. The `FallbackChain.Execute` method processes the fallback sequence, providing alternative responses when possible.

5. **Metrics Recording for Rejections**: Even rejected requests contribute to metrics collection. The system records these as circuit-level rejections, distinct from downstream service failures, providing visibility into circuit breaker activation patterns.

#### Half-Open State Request Processing

The `StateHalfOpen` processing represents the most complex flow, as it must carefully balance recovery testing with protection against premature circuit closure:

1. **Concurrency Control for Testing**: The half-open state maintains a counter of concurrent test requests using atomic operations on `halfOpenRequests`. This ensures only a limited number of requests proceed to the downstream service simultaneously.

2. **Test Request Selection**: The system uses atomic compare-and-swap operations to determine which requests participate in recovery testing. Requests that exceed the configured half-open concurrency limit are immediately rejected with fallback processing.

3. **Critical Recovery Monitoring**: Test requests undergo the same execution and latency monitoring as closed state requests, but their outcomes carry additional weight in state transition decisions. Each success increments the recovery success counter, while failures immediately trigger transition back to open state.

4. **State Transition Decision Logic**: After each test request completion, the system evaluates whether sufficient successful probes have completed to warrant transitioning back to `StateClosed`. This decision considers both the absolute number of successes and the success rate during the testing period.

> **Key Design Insight**: The half-open state represents the most delicate balance in circuit breaker design. It must provide enough testing to validate service recovery while protecting against cascade failures from services that appear healthy but quickly degrade under load.

### Error Classification Logic

Error classification forms the intelligence layer of the circuit breaker system, determining which errors indicate downstream service problems that should affect circuit state versus errors that represent client-side issues or transient network problems.

The `ErrorClassifier` interface provides the pluggable abstraction for error analysis, allowing different services to implement custom classification logic based on their specific error patterns and operational characteristics.

#### Classification Categories and Decision Rules

The error classification system operates on a sophisticated decision tree that analyzes multiple attributes of both the error and the response context:

| Error Category | Circuit Impact | Examples | Classification Logic |
|---|---|---|---|
| Network Connectivity | Circuit Opening | Connection refused, DNS resolution failures, network timeouts | Indicates downstream service unavailability |
| HTTP 5xx Server Errors | Circuit Opening | 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable | Suggests downstream service degradation |
| Timeout Errors | Circuit Opening | Context deadline exceeded during service call | Indicates service responsiveness problems |
| HTTP 4xx Client Errors | Pass-Through | 400 Bad Request, 401 Unauthorized, 404 Not Found | Client error, not service health issue |
| Application-Level Errors | Configurable | Business logic errors, validation failures | Depends on specific application semantics |
| gRPC Status Codes | Mixed | UNAVAILABLE (circuit), INVALID_ARGUMENT (pass-through) | Based on gRPC status code semantics |

The classification logic considers multiple factors beyond just the error type:

**Response Time Analysis**: Even successful requests that exceed expected latency thresholds may be classified as degraded service indicators. The system compares request latency against configurable percentile thresholds to detect service degradation before complete failures occur.

**Error Rate Context**: The classifier considers the current error rate and request volume when making classification decisions. A high error rate may cause the classifier to become more sensitive, treating marginal errors as circuit-affecting to prevent cascade failures.

**Temporal Pattern Recognition**: The system tracks error patterns over time, identifying whether errors are sporadic (likely transient) or sustained (likely indicating service problems). Sustained error patterns increase the likelihood of circuit-opening classification.

#### Custom Classification Implementation

Services implement custom error classification by providing an `ErrorClassifier` that analyzes the specific error patterns relevant to their domain:

```go
type ServiceSpecificClassifier struct {
    timeoutThreshold    time.Duration
    latencyPercentile   float64
    businessErrorCodes  map[string]bool
}
```

The classifier's `IsCircuitFailure` method receives both the response and error, allowing analysis of:
- HTTP status codes and headers
- Response body patterns for application-specific error indicators
- Timing information for latency-based classification
- Context metadata for request classification

> **Design Decision: Pluggable Classification**
> - **Context**: Different services have vastly different error semantics and operational characteristics
> - **Options Considered**: Hard-coded classification rules, configuration-based rules, pluggable interface
> - **Decision**: Pluggable `ErrorClassifier` interface with sensible defaults
> - **Rationale**: Maximum flexibility for service-specific needs while providing working defaults for common cases
> - **Consequences**: Enables precise circuit breaker tuning per service but requires thoughtful classifier implementation

### Metrics Collection Flow

The metrics collection flow orchestrates the continuous gathering, aggregation, and analysis of request performance data that drives circuit breaker state decisions. This flow must handle high-throughput request processing while maintaining accurate metrics for decision-making.

The metrics collection system operates on multiple time scales and aggregation levels, from individual request recording to sliding window analysis and circuit state evaluation.

#### Request-Level Metrics Recording

Every request that flows through the circuit breaker contributes data points to the metrics collection system, regardless of whether it executes against the downstream service or gets rejected by the circuit breaker:

**Success Request Recording**: When a request completes successfully, the `recordSuccess` method updates multiple metrics stores:

1. **Immediate Counters**: Atomic increment of success counters in the `CircuitBreaker` struct and corresponding `Metrics` object
2. **Sliding Window Updates**: Addition of success data point to the current `WindowBucket` with latency measurement
3. **State Evaluation Triggers**: Check if accumulated success count warrants state transition from half-open to closed

**Failure Request Recording**: Failed requests trigger the `recordFailure` method which performs error classification and selective metrics updates:

1. **Error Classification**: The `ErrorClassifier.IsCircuitFailure` method determines if this failure should count toward circuit opening thresholds
2. **Conditional Counter Updates**: Only circuit-affecting failures increment the failure counters and contribute to state transition logic
3. **Comprehensive Logging**: All failures, regardless of classification, are logged for observability and debugging

**Latency Measurement Integration**: Both successful and failed requests contribute latency measurements to sliding window buckets, providing continuous service performance visibility even when the circuit is open.

#### Sliding Window Aggregation Flow

The sliding window metrics system maintains real-time aggregation of request performance data across configurable time periods, providing the foundation for intelligent circuit breaker decision-making:

**Time-Based Bucket Management**: The `SlidingWindow` maintains an array of `WindowBucket` structures, each covering a specific time interval. The `rotateIfNeeded` method continuously evaluates whether new buckets should be created based on the current timestamp:

1. **Bucket Boundary Calculation**: The system calculates bucket boundaries by aligning timestamps to configured bucket intervals
2. **Automatic Rotation**: When the current time exceeds the active bucket's end time, the system creates a new bucket and moves the ring buffer position
3. **Historical Data Retention**: Old buckets outside the configured window size are automatically discarded to prevent memory growth

**Failure Rate Calculation**: The `FailureRate` method aggregates data across all active buckets to compute current failure rates:

1. **Cross-Bucket Aggregation**: Sum request counts and failure counts across all buckets in the sliding window
2. **Rate Computation**: Calculate failure rate as the ratio of failed requests to total requests, with special handling for zero-request scenarios
3. **Confidence Thresholds**: Apply minimum request count thresholds to avoid making state decisions based on insufficient data

**Performance Metrics Aggregation**: Beyond failure rates, the sliding window system tracks latency percentiles, throughput rates, and error distribution patterns that provide comprehensive service health visibility.

#### Circuit State Decision Integration

The metrics collection flow directly drives circuit breaker state transition decisions through continuous evaluation of collected performance data:

**Threshold Evaluation Logic**: The system continuously compares current metrics against configured thresholds:

| Metric | Threshold Type | State Impact | Evaluation Frequency |
|---|---|---|---|
| Failure Rate | Percentage | Open Circuit | Per Request |
| Consecutive Failures | Count | Open Circuit | Per Request |
| Success Count in Half-Open | Count | Close Circuit | Per Request |
| Request Volume | Count | Confidence Gating | Per Request |
| Average Latency | Duration | Optional Circuit Opening | Configurable |

**State Transition Triggering**: When metrics exceed configured thresholds, the system triggers state transitions through dedicated methods:
- `transitionToOpen` when failure thresholds are exceeded
- `transitionToHalfOpen` when recovery timeouts elapse
- `transitionToClosed` when recovery success criteria are met

**Metrics Snapshot Generation**: The `Snapshot` method provides point-in-time metrics copies for external monitoring systems, ensuring consistent data export without impacting request processing performance.

### Recovery Testing Sequence

The recovery testing sequence represents the circuit breaker's intelligence for gradually restoring service to previously failing downstream dependencies. This sequence must balance rapid recovery when services return to health against protection from services that appear recovered but quickly degrade under load.

#### Half-Open State Initialization

The transition from `StateOpen` to `StateHalfOpen` initiates a carefully controlled recovery testing sequence that begins with system state preparation:

**State Transition Preparation**: The `transitionToHalfOpen` method performs several critical initialization steps:

1. **Counter Reset**: The `successCount` and `halfOpenRequests` counters are reset to zero to begin clean recovery measurement
2. **Timestamp Recording**: The `stateTransitionTime` is updated to track how long the circuit remains in half-open state
3. **Timer Cleanup**: Any existing recovery timers are cancelled and cleaned up to prevent resource leaks
4. **Metrics Notification**: The state change is recorded in metrics for observability and debugging

**Concurrency Control Setup**: The half-open state establishes concurrency limits for test requests to prevent overwhelming a potentially fragile recovering service. The system uses atomic operations on `halfOpenRequests` to track concurrent test request count.

#### Test Request Selection and Execution

The half-open state must carefully select which incoming requests participate in recovery testing while providing fallback responses for requests that exceed testing capacity:

**Request Admission Logic**: Each incoming request in half-open state goes through an admission control process:

1. **Atomic Counter Check**: Use atomic compare-and-swap operations to check if the current `halfOpenRequests` count is below the configured limit
2. **Test Request Acceptance**: Requests that pass the concurrency check proceed to downstream service execution with full monitoring
3. **Excess Request Handling**: Requests that exceed the concurrency limit are immediately routed to fallback processing without affecting circuit state

**Enhanced Monitoring for Test Requests**: Recovery test requests receive additional monitoring beyond normal request processing:

1. **Critical Path Latency Tracking**: Test request latencies are compared against baseline performance expectations to detect degraded service recovery
2. **Success/Failure Impact**: Each test request outcome has amplified impact on state transition decisions compared to normal closed-state operation
3. **Timeout Sensitivity**: Test requests may use more aggressive timeout configurations to quickly detect services that appear available but respond slowly

#### Recovery Success Evaluation

The circuit breaker continuously evaluates recovery test outcomes to determine when sufficient evidence exists to transition back to normal operation:

**Success Threshold Logic**: The system maintains configurable success criteria for recovery validation:

| Success Metric | Threshold | Impact | Rationale |
|---|---|---|---|
| Consecutive Successes | Configurable Count | Close Circuit | Demonstrates consistent service availability |
| Success Rate | Percentage | Close Circuit | Validates service reliability under test load |
| Average Latency | Duration Threshold | Close Circuit | Ensures performance meets expectations |
| Test Duration | Time Limit | Force Decision | Prevents indefinite half-open state |

**Progressive Confidence Building**: Rather than requiring just a single successful request, the recovery logic builds confidence through multiple successful interactions:

1. **Initial Success Recording**: The first successful test request demonstrates basic service availability but doesn't immediately close the circuit
2. **Pattern Validation**: Subsequent successful requests validate that the initial success wasn't a fluke and that the service can handle sustained load
3. **Performance Validation**: The system ensures that successful requests also meet latency and throughput expectations before declaring full recovery

#### Recovery Failure Handling

When test requests fail during recovery testing, the system must quickly protect against cascade failures while preserving the ability to retry recovery later:

**Immediate Failure Response**: Any failure during recovery testing triggers rapid protective action:

1. **Instant State Reversion**: Failed recovery test requests immediately trigger `transitionToOpen` to protect upstream services
2. **Timer Reset**: A new recovery timer is started to schedule the next recovery attempt
3. **Failure Analysis**: The failure is analyzed by the error classifier to distinguish between transient issues and genuine service problems

**Adaptive Recovery Timing**: The system may implement exponential backoff for recovery attempts, extending the time between recovery tests when services repeatedly fail recovery testing. This prevents rapid cycling between open and half-open states that can create additional load on struggling services.

> **Critical Recovery Design Principle**: The recovery testing sequence must err on the side of caution. It's better to remain open longer than necessary than to close prematurely and allow cascade failures to propagate through the system.

### Common Pitfalls

⚠️ **Pitfall: Race Conditions in State Transitions**

A common mistake is implementing state transitions without proper synchronization, leading to inconsistent circuit breaker behavior under concurrent load. For example, two simultaneous requests might both detect that failure thresholds are exceeded and both attempt to transition the circuit to open state, potentially causing double timer initialization or inconsistent counter resets.

The fix requires using atomic operations or mutex locking around state transition methods to ensure only one thread can modify circuit state at a time.

⚠️ **Pitfall: Metrics Collection Performance Impact**

Developers often implement metrics collection with expensive operations like holding locks during downstream service calls or performing synchronous disk writes for every request. This can make the circuit breaker slower than the service it's protecting.

The solution involves using atomic counters for hot-path metrics updates, batching metrics writes, and ensuring that metrics collection never blocks request processing.

⚠️ **Pitfall: Half-Open Request Flood**

A dangerous mistake is allowing unlimited concurrent requests during half-open state recovery testing. If a service appears to recover but then gets overwhelmed by a flood of test requests, it can immediately fail again, creating a rapid open-half-open-open cycle.

The fix requires implementing strict concurrency limits on half-open test requests and potentially implementing exponential backoff for recovery attempts.

⚠️ **Pitfall: Error Classification Oversensitivity**

Classifying all errors as circuit-opening failures can cause circuits to open unnecessarily when encountering client errors or transient network issues that don't indicate downstream service problems. This leads to unnecessary service degradation.

The solution involves implementing sophisticated error classification that distinguishes between client errors (4xx HTTP status codes), transient network issues, and genuine service failures (5xx status codes, timeouts).

⚠️ **Pitfall: Sliding Window Memory Leaks**

Implementing sliding windows without proper bucket cleanup can lead to unbounded memory growth as old time buckets accumulate. This is especially problematic in long-running services.

The fix requires implementing proper bucket rotation with automatic cleanup of expired buckets and bounded ring buffer implementations.

### Implementation Guidance

The interactions and data flow implementation requires careful orchestration of concurrent operations, atomic state management, and efficient metrics collection. The following guidance provides the foundation for building a robust and performant circuit breaker system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|---|---|---|
| Concurrency Control | `sync.Mutex` for state protection | `sync.RWMutex` with atomic counters for hot paths |
| Metrics Storage | In-memory maps with mutex protection | Lock-free data structures with atomic operations |
| Time Management | `time.Now()` and `time.Timer` | Pluggable `Clock` interface for testability |
| Error Classification | Simple error type checking | Configurable rules engine with pattern matching |
| Request Context | Basic context cancellation | Rich context with request metadata and tracing |

#### Recommended File Structure

```
internal/circuitbreaker/
  circuit.go              ← Main CircuitBreaker implementation
  metrics.go              ← SlidingWindow and MetricsCollector
  classifier.go           ← ErrorClassifier implementations  
  fallback.go             ← FallbackRegistry and execution logic
  semaphore.go           ← Bulkhead Semaphore implementation
  clock.go               ← Clock interface and implementations
  circuit_test.go        ← Comprehensive integration tests
  testutils/
    mockserver.go        ← HTTP test server with controllable failures
    mockclock.go         ← Deterministic clock for testing
```

#### Core Request Processing Implementation

```go
// Execute processes a request through the circuit breaker with full monitoring.
// This is the main entry point that orchestrates all circuit breaker functionality.
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
    // TODO 1: Acquire bulkhead semaphore permit with context timeout
    //         Return ErrBulkheadCapacity if no permits available
    
    // TODO 2: Check current circuit state and handle each case:
    //         - StateClosed: proceed to normal execution
    //         - StateOpen: check recovery timer and possibly transition to half-open
    //         - StateHalfOpen: implement concurrency-limited recovery testing
    
    // TODO 3: Record request attempt in sliding window metrics
    //         Call rotateIfNeeded() and increment request counter
    
    // TODO 4: Execute the protected function with latency measurement
    //         Capture start time, call fn(), measure duration
    
    // TODO 5: Process the result based on success/failure:
    //         - Success: call recordSuccess() with latency
    //         - Failure: call recordFailure() with error classification
    
    // TODO 6: Evaluate state transition conditions:
    //         - Check failure thresholds for transition to open
    //         - Check success thresholds for transition to closed (in half-open)
    
    // TODO 7: Release bulkhead semaphore permit in defer block
    //         Ensure resource cleanup regardless of execution outcome
    
    // Hint: Use atomic operations for counter updates to avoid lock contention
    // Hint: Implement state transitions as separate methods for clarity
}
```

#### Error Classification Infrastructure

```go
// DefaultErrorClassifier provides sensible defaults for common error patterns.
// This classifier handles HTTP status codes, network errors, and timeout scenarios.
type DefaultErrorClassifier struct {
    circuitStatusCodes map[int]bool  // HTTP codes that should trip circuit
    latencyThreshold   time.Duration // Latency beyond which to consider degraded
}

// IsCircuitFailure determines if an error should contribute to circuit opening.
// This method implements the core intelligence for error classification.
func (c *DefaultErrorClassifier) IsCircuitFailure(resp *http.Response, err error) bool {
    // TODO 1: Handle network-level errors (connection refused, DNS failures)
    //         These typically indicate downstream service unavailability
    
    // TODO 2: Analyze HTTP response status codes if response is available
    //         5xx codes generally indicate service problems
    //         4xx codes usually represent client errors (don't trip circuit)
    
    // TODO 3: Check for timeout errors in context or network layers
    //         Timeouts often indicate service performance degradation
    
    // TODO 4: Evaluate response latency against configured thresholds
    //         Slow but successful responses might indicate degradation
    
    // TODO 5: Return boolean indicating if this error should affect circuit state
    
    // Hint: Use error type assertions to identify specific error types
    // Hint: Consider using error wrapping to preserve original error context
}
```

#### Sliding Window Metrics Implementation

```go
// rotateIfNeeded creates new time buckets when the current bucket expires.
// This method ensures metrics always reflect the current time window.
func (sw *SlidingWindow) rotateIfNeeded(now time.Time) {
    // TODO 1: Calculate the expected bucket for the current timestamp
    //         Align timestamp to bucket boundaries using bucket size
    
    // TODO 2: Check if the current bucket covers the current timestamp
    //         Compare bucket start/end times with current time
    
    // TODO 3: If rotation is needed, advance ring buffer position
    //         Update position using modulo arithmetic for circular buffer
    
    // TODO 4: Initialize new bucket with appropriate time boundaries
    //         Set StartTime, EndTime, and zero all counters
    
    // TODO 5: Clean up old buckets outside the sliding window
    //         Ensure memory usage remains bounded
    
    // Hint: Use mutex protection around bucket modifications
    // Hint: Consider using sync.RWMutex for better read performance
}

// RecordRequest updates the sliding window with request outcome and latency.
// This method feeds all request processing data into the metrics system.
func (sw *SlidingWindow) RecordRequest(success bool, latency time.Duration) {
    // TODO 1: Ensure current bucket exists for timestamp
    //         Call rotateIfNeeded() to handle bucket creation
    
    // TODO 2: Atomically increment request counter in current bucket
    //         Use atomic operations to avoid lock contention
    
    // TODO 3: Update success or failure counter based on outcome
    //         Increment appropriate counter atomically
    
    // TODO 4: Update latency statistics in the bucket
    //         Track average latency and maximum latency
    
    // TODO 5: Trigger any registered metric collection callbacks
    //         Notify external monitoring systems of metric updates
    
    // Hint: Consider using atomic.AddInt64 for counter updates
    // Hint: Implement latency tracking with exponential moving averages
}
```

#### Recovery Testing Logic

```go
// canExecuteInHalfOpen determines if a request should participate in recovery testing.
// This method implements concurrency control for half-open state testing.
func (cb *CircuitBreaker) canExecuteInHalfOpen() bool {
    // TODO 1: Get current count of half-open requests using atomic load
    //         Read halfOpenRequests counter atomically
    
    // TODO 2: Check if current count is below configured limit
    //         Compare against Config.MaxHalfOpenRequests
    
    // TODO 3: If under limit, atomically increment counter and return true
    //         Use compare-and-swap to avoid race conditions
    
    // TODO 4: If at limit, return false to reject request
    //         Request should be handled by fallback system
    
    // Hint: Use atomic.CompareAndSwapInt64 for race-free increment
    // Hint: Implement proper cleanup to decrement counter on completion
}
```

#### Milestone Checkpoints

**Milestone 1 Verification**:
- Run `go test -v ./internal/circuitbreaker/...` 
- Expected: All basic state machine tests pass
- Manual test: Create circuit with failure threshold 3, send 4 failed requests, verify circuit opens
- Check: Circuit should reject subsequent requests immediately with `ErrCircuitOpen`

**Milestone 2 Verification**:
- Test sliding window: Send mix of success/failure over time, verify failure rate calculation
- Test fallback: Configure fallback function, verify it executes when circuit is open
- Expected: Metrics show accurate failure rates, fallback provides alternative responses

**Milestone 3 Verification**:
- Integration test: Wrap HTTP client, make requests to mock server with controlled failures
- Chaos test: Inject network failures, verify circuit opens and recovers appropriately
- Dashboard: Verify circuit states are visible in monitoring interface


## Error Handling and Edge Cases

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

Comprehensive error handling and edge case management form the foundation of a robust circuit breaker system. The circuit breaker must gracefully handle not just the obvious failure scenarios like network timeouts and service errors, but also the subtle edge cases that emerge in distributed systems: clock skew, configuration changes during runtime, concurrent state transitions, and cascading fallback failures.

This section provides a systematic analysis of failure modes, establishes clear error classification rules, addresses complex edge cases involving timing and concurrency, and defines recovery strategies that maintain system stability even under extreme conditions.

### System Failure Modes

Every component in the circuit breaker system can fail in multiple ways, and understanding these failure modes is essential for building resilient error handling. The system must detect failures quickly, classify them correctly, and respond appropriately without introducing additional instability.

#### Circuit Breaker Core Failures

The central `CircuitBreaker` component faces several distinct failure modes, each requiring different detection and recovery strategies. The most critical failure involves **state corruption during concurrent access**, where multiple goroutines attempt to modify the circuit state simultaneously without proper synchronization. This manifests as inconsistent failure counts, incorrect state transitions, or lost metrics updates.

**Thread safety violations** occur when the `failureCount`, `successCount`, or `stateTransitionTime` fields are accessed without proper mutex protection. The system detects this through inconsistent metrics where the sum of success and failure counts doesn't match the total request count, or when state transitions occur without corresponding metric changes.

| Failure Mode | Detection Method | Impact on Circuit Behavior | Recovery Strategy |
|--------------|------------------|---------------------------|------------------|
| State corruption from race conditions | Metrics inconsistency checks, unexpected state values | Incorrect state transitions, lost failure tracking | Reset circuit to closed state with cleared counters |
| Mutex deadlock in state transitions | Request timeout without state change, blocked goroutines | Complete circuit freeze, all requests hang | Circuit breaker replacement with fresh instance |
| Recovery timer failure or leak | Timer not firing after timeout, memory growth | Circuit stuck in open state indefinitely | Manual timer cleanup and state reset |
| Memory overflow from metric accumulation | Unbounded growth in metric storage, OOM errors | System crash, circuit unavailable | Implement metric rotation and bounded storage |
| Configuration reload race conditions | Partial config application, mixed behavior | Inconsistent circuit behavior across requests | Atomic configuration updates with validation |

**Timer management failures** represent another critical failure mode. The `recoveryTimer` may fail to fire due to system clock changes, timer resource exhaustion, or goroutine scheduling issues. When this occurs, the circuit remains permanently in the open state, never transitioning to half-open for recovery testing. The system detects timer failures by monitoring the elapsed time since state transition and comparing it against the configured `RecoveryTimeout`.

**Memory leaks in metrics collection** occur when the `SlidingWindow` continues accumulating `WindowBucket` instances without proper rotation. This happens when the bucket rotation logic fails due to clock issues or when error conditions prevent bucket cleanup. The system monitors memory usage patterns and bucket counts to detect these leaks.

> **Critical Design Insight**: Circuit breaker failures must never cause cascade failures in the application. The circuit breaker should fail open (allowing requests through) rather than fail closed (blocking all traffic) when its internal state becomes unreliable.

#### Sliding Window Metric Failures

The `SlidingWindow` component introduces temporal complexity that creates unique failure modes. **Bucket rotation failures** occur when the time-based bucket management logic encounters clock changes, leap seconds, or extreme system load that prevents timely bucket updates.

**Clock synchronization issues** in distributed systems can cause bucket timestamps to become inconsistent. When different nodes have clock skew, the sliding window calculations produce different failure rates for the same service, leading to inconsistent circuit breaker behavior across instances.

| Failure Mode | Detection Method | Impact on Metrics | Recovery Strategy |
|--------------|------------------|-------------------|------------------|
| Bucket rotation stall | Old buckets persist beyond window size | Stale failure rates, delayed circuit opening | Force bucket rotation with current timestamp |
| Clock skew corruption | Bucket timestamps out of sequence | Incorrect failure rate calculations | Timestamp normalization and bucket rebuild |
| Bucket overflow | Request counts exceed int32 limits | Metric wraparound, negative failure rates | Split high-volume buckets or use larger data types |
| Window position corruption | Ring buffer position invalid | Lost metrics, incorrect bucket access | Rebuild ring buffer with position validation |
| Memory fragmentation | Inefficient bucket storage allocation | Increased memory usage, GC pressure | Implement bucket pooling and reuse |

**Bucket overflow scenarios** arise in high-traffic services where the number of requests within a single bucket exceeds the `int32` storage capacity. This causes metric wraparound, where failure counts become negative, leading to incorrect failure rate calculations that may prevent the circuit from opening when it should.

> **Decision: Bucket Overflow Handling Strategy**
> - **Context**: High-traffic services can generate millions of requests per bucket, exceeding int32 limits
> - **Options Considered**: Increase to int64, split buckets by volume, implement saturating counters
> - **Decision**: Use int64 for request counts with bucket splitting for extreme loads
> - **Rationale**: int64 provides sufficient headroom for most services while bucket splitting handles edge cases without changing core algorithms
> - **Consequences**: Slightly increased memory usage but eliminates overflow-related failures

#### Fallback Chain Failures

The `FallbackRegistry` and `FallbackChain` components introduce recursive failure possibilities where fallback functions themselves fail, potentially creating infinite recursion or stack overflow scenarios. **Fallback depth explosion** occurs when each fallback function triggers additional fallback attempts, exceeding the maximum call stack depth.

**Circular fallback dependencies** create infinite loops where service A's fallback calls service B, whose fallback calls service A. The system must detect these cycles during fallback chain construction and runtime execution.

| Failure Mode | Detection Method | Impact on Request Processing | Recovery Strategy |
|--------------|------------------|------------------------------|------------------|
| Fallback function panic | Recover from panic in fallback execution | Request fails without graceful degradation | Skip failed fallback, continue chain execution |
| Circular fallback dependencies | Cycle detection during chain registration | Infinite recursion, stack overflow | Reject circular chains, maintain acyclic graph |
| Fallback depth explosion | Stack depth monitoring, execution timeout | System resource exhaustion | Enforce maximum fallback depth limit |
| Fallback resource leaks | Monitor goroutines, file descriptors, connections | Resource pool exhaustion | Implement fallback execution timeout and cleanup |
| Fallback configuration corruption | Validate fallback chain integrity | Inconsistent fallback behavior | Rebuild fallback chains from validated configuration |

**Resource leaks in fallback execution** occur when fallback functions create goroutines, open connections, or allocate resources without proper cleanup, especially when fallback execution is cancelled or times out. The system must enforce resource limits and cleanup procedures for all fallback operations.

#### Bulkhead Resource Pool Failures

The `Semaphore` and `BulkheadPool` components manage finite resources and can fail through **resource pool exhaustion**, where all permits are acquired but never released due to goroutine failures or cleanup bugs. This causes new requests to block indefinitely waiting for permits.

**Bulkhead capacity thrashing** occurs when the bulkhead capacity is set too low for normal load, causing excessive request rejections, or too high, failing to provide isolation during overload conditions. The system must monitor permit utilization patterns and rejection rates to detect capacity misconfigurations.

| Failure Mode | Detection Method | Impact on Request Flow | Recovery Strategy |
|--------------|------------------|------------------------|------------------|
| Permit leak | Active permits exceed capacity, permits never decrease | Request blocking, bulkhead starvation | Force permit release, rebuild semaphore |
| Capacity thrashing | High rejection rates or excessive blocking | Poor performance, isolation failure | Dynamic capacity adjustment based on utilization |
| Context cancellation race | Permits acquired after context cancelled | Resource waste, delayed cleanup | Context check before permit acquisition |
| Bulkhead deadlock | Circular permit acquisition dependencies | Complete request blockage | Timeout-based permit acquisition |
| Pool state corruption | Invalid active/queued counts | Incorrect capacity decisions | Reset pool state with permit audit |

### Error Classification Rules

Effective circuit breaker operation requires precise error classification to distinguish between failures that indicate service health problems versus transient issues or client errors that shouldn't affect circuit state. The `ErrorClassifier` component implements these classification rules through a structured decision tree.

#### Primary Error Categories

The error classification system divides errors into four primary categories based on their impact on circuit breaker state and their indication of downstream service health.

**Circuit-Opening Errors** represent genuine service health problems that should contribute to failure counts and potentially trigger circuit state transitions. These include network connectivity failures, service unavailability, timeout errors, and server-side errors that indicate capacity or functionality issues.

**Pass-Through Errors** are client-side or logical errors that don't indicate downstream service health problems. Authentication failures, validation errors, and not-found responses typically fall into this category because they represent correct service behavior rather than service degradation.

| Error Category | Circuit Impact | Failure Count Impact | Examples | Classification Logic |
|----------------|----------------|---------------------|-----------|---------------------|
| Circuit-Opening | Contributes to state transitions | Increments failure counter | Connection timeout, HTTP 5xx, DNS resolution failure | `IsCircuitFailure() returns true` |
| Pass-Through | No circuit state impact | No counter change | HTTP 4xx client errors, authentication failure, validation error | `IsCircuitFailure() returns false` |
| Transient-Retry | May contribute after retry exhaustion | Increments after retry limit | Temporary network glitch, rate limit exceeded | Retry logic before classification |
| Configuration-Error | Immediate escalation | No counter change | Invalid URL, missing credentials, protocol mismatch | Logged as configuration issue |

**Transient-Retry Errors** represent temporary issues that may resolve quickly but could indicate service problems if persistent. The classification system applies retry logic before final classification, contributing to failure counts only after retry exhaustion.

**Configuration-Error** conditions represent client-side configuration problems that prevent proper service communication. These errors are immediately escalated through logging and monitoring but don't affect circuit breaker state since they don't reflect downstream service health.

#### HTTP Response Classification

HTTP responses require nuanced classification based on status codes, response headers, and response content. The classification logic considers both the semantic meaning of status codes and the operational context of the service.

**5xx Server Errors** generally indicate circuit-opening conditions because they represent server-side failures, capacity issues, or service degradation. However, some 5xx responses may be intentional and shouldn't affect circuit state, such as 501 Not Implemented for optional endpoints.

| HTTP Status Range | Default Classification | Override Conditions | Circuit Breaker Action | Rationale |
|------------------|----------------------|-------------------|----------------------|-----------|
| 1xx Informational | Pass-Through | None | No impact | Informational responses, not errors |
| 2xx Success | Success | None | Reset failure count | Successful request completion |
| 3xx Redirection | Pass-Through | Too many redirects → Circuit-Opening | No impact (unless redirect loop) | Client should handle redirections |
| 4xx Client Error | Pass-Through | 408 Timeout, 429 Rate Limit → Circuit-Opening | No impact (except specific codes) | Client-side errors, service functioning correctly |
| 5xx Server Error | Circuit-Opening | 501 Not Implemented → Pass-Through | Increment failure count | Server-side errors indicate service problems |

**4xx Client Errors** typically represent pass-through conditions because they indicate correct server behavior in response to invalid client requests. However, certain 4xx codes like 408 Request Timeout and 429 Too Many Requests may indicate server-side capacity issues and should contribute to circuit-opening decisions.

**Timeout and Connection Errors** consistently represent circuit-opening conditions because they indicate network connectivity issues or server unresponsiveness that affects service availability for all clients.

> **Decision: Rate Limit Error Classification**
> - **Context**: HTTP 429 Too Many Requests can indicate client abuse or server capacity limits
> - **Options Considered**: Always pass-through, always circuit-opening, context-dependent classification
> - **Decision**: Circuit-opening after threshold of rate limit responses within time window
> - **Rationale**: Persistent rate limiting indicates server capacity issues affecting service availability
> - **Consequences**: Circuit breaker may open during legitimate traffic spikes, requiring careful threshold tuning

#### gRPC Error Classification

gRPC errors use structured status codes that provide more semantic information than HTTP status codes, enabling more precise classification decisions. The `ErrorClassifier` maps gRPC status codes to circuit breaker actions based on their indication of service health.

**UNAVAILABLE and DEADLINE_EXCEEDED** consistently indicate circuit-opening conditions because they represent service unavailability or performance degradation that affects client experience. **RESOURCE_EXHAUSTED** also typically indicates circuit-opening conditions as it suggests server capacity issues.

| gRPC Status Code | Circuit Classification | Reasoning | Special Handling |
|-----------------|----------------------|-----------|------------------|
| OK | Success | Request completed successfully | Reset failure counters |
| CANCELLED | Pass-Through | Client-initiated cancellation | No circuit impact |
| UNKNOWN | Circuit-Opening | Unclear error nature, assume service issue | Log for investigation |
| INVALID_ARGUMENT | Pass-Through | Client error, service functioning | No circuit impact |
| DEADLINE_EXCEEDED | Circuit-Opening | Service performance degradation | Increment failure count |
| NOT_FOUND | Pass-Through | Valid service response | No circuit impact |
| ALREADY_EXISTS | Pass-Through | Valid service response | No circuit impact |
| PERMISSION_DENIED | Pass-Through | Valid service response (authorization) | No circuit impact |
| RESOURCE_EXHAUSTED | Circuit-Opening | Server capacity issues | Increment failure count |
| FAILED_PRECONDITION | Pass-Through | Client logic error | No circuit impact |
| ABORTED | Circuit-Opening | Server-side transaction failure | Increment failure count |
| OUT_OF_RANGE | Pass-Through | Client parameter error | No circuit impact |
| UNIMPLEMENTED | Pass-Through | Expected for optional features | No circuit impact |
| INTERNAL | Circuit-Opening | Server internal error | Increment failure count |
| UNAVAILABLE | Circuit-Opening | Service unavailable | Increment failure count |
| DATA_LOSS | Circuit-Opening | Serious server issue | Increment failure count |
| UNAUTHENTICATED | Pass-Through | Valid service response (authentication) | No circuit impact |

**ABORTED status** requires careful consideration because it can indicate either client-side transaction conflicts (pass-through) or server-side resource contention (circuit-opening). The classifier examines error details and frequency to make appropriate decisions.

#### Custom Error Classification Policies

Different services may require custom error classification policies based on their specific failure modes and operational characteristics. The `ErrorClassifier` interface allows per-service customization while maintaining consistent classification logic.

**Domain-Specific Classifications** enable services to define custom error handling for business logic failures that shouldn't affect circuit breaker state. For example, an inventory service may return "out of stock" errors that represent valid business responses rather than service failures.

**Error Message Pattern Matching** allows classification based on error message content when status codes alone don't provide sufficient information. Database connection pool exhaustion messages, for instance, clearly indicate circuit-opening conditions regardless of the wrapper error code.

### Edge Case Scenarios

Distributed systems present numerous edge cases that can disrupt circuit breaker operation in subtle ways. These scenarios often involve timing issues, configuration changes, or race conditions that occur infrequently but can cause significant operational problems when they do occur.

#### Clock and Time-Related Edge Cases

System clock changes present one of the most challenging edge cases for time-dependent circuit breaker logic. **Forward clock jumps** can cause the sliding window to immediately age out all historical data, making recent failure patterns invisible and potentially allowing traffic to unhealthy services.

**Backward clock jumps** create even more complex scenarios where new requests appear to have timestamps in the past relative to existing sliding window buckets. The system must handle these cases without corrupting the temporal ordering that sliding window calculations depend upon.

| Edge Case Scenario | Impact on Circuit Behavior | Detection Method | Mitigation Strategy |
|-------------------|---------------------------|------------------|-------------------|
| Forward clock jump (>1 hour) | Sliding window reset, loss of failure history | Compare bucket timestamps to system monotonic clock | Use monotonic clock for intervals, system clock only for absolute timestamps |
| Backward clock jump | New requests appear in past buckets | Timestamp ordering violations in buckets | Reject requests with past timestamps, rebuild window |
| Leap second adjustment | Bucket boundary calculation errors | Duplicate or missing bucket timestamps | Use leap-second-aware time libraries |
| Timezone changes (DST) | Bucket timing misalignment | Hour-long gaps or duplicates in buckets | Store all times in UTC internally |
| NTP synchronization spike | Sudden time corrections during operation | Large timestamp deltas between consecutive operations | Gradual time adjustment with bounded corrections |

**Leap second handling** requires special consideration because leap seconds can cause duplicate timestamps or missing time ranges that disrupt bucket rotation logic. The system must use leap-second-aware time handling and validate bucket timestamp sequences.

**Network Time Protocol (NTP) synchronization events** can cause sudden time corrections that make recent buckets appear to be from the future or past. The circuit breaker must detect these corrections and adjust its time-based calculations accordingly.

> **Decision: Clock Source Strategy**
> - **Context**: System clock changes can corrupt sliding window time calculations
> - **Options Considered**: System clock only, monotonic clock only, hybrid approach
> - **Decision**: Monotonic clock for intervals and relative timing, system clock for absolute timestamps
> - **Rationale**: Monotonic clocks are not affected by system time adjustments but don't provide wall-clock correlation
> - **Consequences**: Requires clock abstraction layer but eliminates most time-related edge cases

#### Configuration Update Edge Cases

Runtime configuration updates create race conditions where some requests use old configuration while others use new configuration, potentially causing inconsistent circuit breaker behavior. **Partial configuration application** occurs when configuration updates fail halfway through, leaving the system in an inconsistent state.

**Configuration validation during updates** must handle the case where new configuration parameters are invalid but the system must continue operating with existing configuration. The validation process itself can fail, requiring fallback to safe default values.

| Configuration Change Type | Potential Race Condition | Consistency Risk | Resolution Strategy |
|---------------------------|-------------------------|------------------|-------------------|
| Failure threshold change | Some requests use old threshold, others new | Inconsistent circuit opening timing | Atomic configuration swap with validation |
| Recovery timeout update | Timer created with old timeout while new requests use new value | Mixed recovery behavior | Cancel existing timers, recreate with new timeout |
| Sliding window size change | Bucket rotation logic uses mixed window sizes | Corrupted failure rate calculations | Rebuild sliding window with new parameters |
| Service endpoint changes | Requests routed to different circuit breakers | Circuit state inconsistency across endpoints | Endpoint normalization and circuit key management |
| Error classification updates | Mixed error handling for same error types | Inconsistent failure counting | Version configuration changes with request context |

**Service endpoint configuration changes** create particularly complex scenarios where requests to the same logical service may be routed through different circuit breakers if endpoint URLs or service identification logic changes. The system must maintain circuit breaker continuity across configuration updates.

**Fallback function updates** require careful handling because existing requests may still be executing old fallback logic while new requests use updated fallbacks. The system must ensure fallback function compatibility and avoid breaking changes during updates.

#### Concurrent State Transition Edge Cases

Multiple requests arriving simultaneously can trigger concurrent state transitions that create race conditions in circuit breaker logic. **Simultaneous failure threshold breach** occurs when multiple requests fail at exactly the same time, causing multiple goroutines to attempt transitioning the circuit from closed to open.

**Half-open state concurrency** presents unique challenges because the half-open state is designed to allow limited test requests, but determining which requests participate in recovery testing requires careful synchronization to avoid allowing too many test requests.

| Concurrency Scenario | Race Condition Risk | Incorrect Behavior | Synchronization Strategy |
|---------------------|-------------------|-------------------|------------------------|
| Multiple threads hit failure threshold | Multiple open transitions, duplicate timers | Timer resource leaks, inconsistent state | Compare-and-swap for state transitions |
| Half-open request count race | Too many test requests allowed | Recovery testing overload | Atomic counter with bounds checking |
| Success/failure recording race | Lost metric updates, incorrect counts | Inaccurate failure rate calculations | Per-request atomic operations |
| Timer expiration with concurrent requests | State transition during request processing | Requests processed with wrong state | State validation before request execution |
| Circuit breaker registry updates | Service mapping changes during request | Request routed to wrong circuit | Copy-on-write service registry |

**Timer expiration concurrency** creates scenarios where the recovery timer fires to transition from open to half-open while other goroutines are processing requests that should be rejected. The system must validate circuit state at multiple points during request processing.

**Circuit breaker registry updates** can cause requests to be routed to different circuit breaker instances mid-flight if service identification or circuit breaker mapping changes concurrently with request processing.

#### Network and Infrastructure Edge Cases

Network infrastructure failures can create cascading edge cases where circuit breakers throughout the system react to the same underlying issue in ways that amplify the problem. **Split-brain scenarios** occur when network partitions cause different service instances to maintain inconsistent circuit breaker states.

**DNS resolution failures** create edge cases where the same logical service becomes unreachable through different error paths (DNS timeout vs connection refused), potentially affecting error classification and circuit breaker state differently.

| Infrastructure Scenario | Circuit Breaker Impact | Cascade Risk | Containment Strategy |
|------------------------|------------------------|--------------|-------------------|
| Network partition between services | Inconsistent circuit states across instances | Different parts of system have different service availability view | Circuit state synchronization with partition tolerance |
| DNS resolution intermittent failures | Service appears healthy from some clients, failed from others | Partial traffic blocking | DNS result caching with TTL management |
| Load balancer failover | Connection errors during failover window | Circuit opens during normal infrastructure maintenance | Distinguish infrastructure failures from service failures |
| Service mesh configuration changes | Routing changes affect service reachability | Circuit opens due to configuration, not service health | Service mesh integration for configuration awareness |
| Certificate expiration or rotation | SSL/TLS failures classified as service failures | Circuit opens due to infrastructure issue | Certificate-specific error classification |

**Load balancer failover scenarios** cause temporary connection failures that may trigger circuit breaker state transitions even though the underlying service remains healthy. The system must distinguish between infrastructure maintenance and genuine service health issues.

**Certificate rotation edge cases** occur when SSL/TLS certificate updates cause temporary connection failures that shouldn't be classified as service health problems but may trigger circuit breaker responses.

### Recovery and Degradation Strategies

Effective recovery and degradation strategies ensure that circuit breaker systems maintain service availability even when primary failure detection and recovery mechanisms encounter problems. These strategies operate at multiple levels: immediate request-level fallbacks, component-level recovery procedures, and system-level degradation modes.

#### Hierarchical Recovery Strategies

Recovery strategies operate in a hierarchical manner where each level provides increasingly conservative fallback behavior when higher levels fail. **Primary recovery** focuses on restoring normal circuit breaker operation through standard state transitions and metric reset procedures.

**Secondary recovery** activates when primary recovery mechanisms fail, implementing simplified circuit breaker behavior that prioritizes availability over sophisticated failure detection. This mode uses basic timeout-based failure detection instead of sliding window metrics and simplified state transitions.

| Recovery Level | Trigger Conditions | Recovery Actions | Fallback Behavior |
|---------------|-------------------|------------------|-------------------|
| Primary Recovery | Standard circuit opening/closing conditions | Normal state machine transitions, metric calculations | Full circuit breaker functionality |
| Secondary Recovery | Circuit state corruption, metric calculation failures | Simplified failure counting, basic timeout logic | Reduced sophistication, maintained availability |
| Tertiary Recovery | Circuit breaker component failures | Bypass circuit breaker, direct service calls with timeout | No circuit protection, basic timeout only |
| Degraded Operation | System resource exhaustion, configuration corruption | Static configuration, minimal resource usage | Limited functionality, maximum stability |
| Emergency Bypass | Complete circuit breaker system failure | Direct service calls, no failure tracking | No protection, direct service access |

**Tertiary recovery** bypasses circuit breaker logic entirely when the circuit breaker system itself becomes unreliable, falling back to simple timeout-based service calls. This ensures that applications continue functioning even when circuit breaker components fail.

**Degraded operation mode** activates during severe system resource constraints, using static configuration and minimal resource allocation to maintain basic service availability. **Emergency bypass mode** provides the ultimate fallback when all circuit breaker functionality must be disabled.

> **Critical Recovery Principle**: Recovery strategies must never introduce additional failure modes. Each recovery level must be simpler and more reliable than the level above it, eliminating potential failure points rather than adding them.

#### Component-Specific Recovery Procedures

Each major circuit breaker component requires specialized recovery procedures tailored to its specific failure modes and operational requirements. **Circuit breaker state recovery** handles corrupted state transitions, stuck timers, and inconsistent failure counts through systematic state reconstruction.

**Sliding window recovery** addresses bucket corruption, timing issues, and memory leaks through window rebuilding procedures that reconstruct accurate failure rate calculations from available data. When complete reconstruction isn't possible, the system falls back to conservative estimates that err on the side of availability.

| Component | Common Failure Modes | Recovery Procedure | Data Preservation Strategy |
|-----------|---------------------|-------------------|--------------------------|
| `CircuitBreaker` | State corruption, stuck timers | Reset to closed state, clear counters, restart timers | Preserve service configuration, reset operational state |
| `SlidingWindow` | Bucket corruption, timing issues | Rebuild window with current timestamp | Preserve recent bucket data if valid |
| `FallbackRegistry` | Circular dependencies, corrupted chains | Rebuild fallback chains from configuration | Validate and preserve non-circular fallback functions |
| `ServiceRegistry` | Service mapping corruption | Rebuild service-to-circuit mapping | Preserve service configurations, recreate circuit instances |
| `MetricsCollector` | Counter overflow, inconsistent state | Reset counters, restart collection | Archive current metrics, start fresh collection |

**Fallback registry recovery** handles circular dependencies and corrupted fallback chains through systematic chain reconstruction that validates acyclic properties and function availability. When fallback chains cannot be recovered, the system falls back to simple error responses.

**Service registry recovery** rebuilds service-to-circuit-breaker mappings when registry corruption prevents proper service identification. The recovery process preserves service configuration while creating fresh circuit breaker instances with clean operational state.

#### Graceful Degradation Modes

Graceful degradation allows the circuit breaker system to continue providing value even when operating under constrained resources or partial functionality. **Functionality shedding** systematically disables advanced features while maintaining core circuit breaker behavior.

**Resource-constrained operation** adapts circuit breaker behavior for environments with limited memory, CPU, or network resources. This mode uses simplified algorithms that require fewer computational resources while still providing basic failure detection and recovery.

| Degradation Mode | Disabled Features | Retained Features | Resource Savings |
|------------------|------------------|-------------------|------------------|
| Metrics Limited | Sliding window calculations, detailed metrics | Basic failure counting, state transitions | 70% memory reduction |
| Fallback Disabled | Fallback execution, chain processing | Circuit state management, failure detection | 30% CPU reduction |
| Bulkhead Bypassed | Concurrency limiting, resource isolation | Circuit breaker functionality | 50% synchronization overhead reduction |
| Monitoring Reduced | Detailed observability, metric collection | Core circuit operations | 40% instrumentation overhead reduction |
| Configuration Frozen | Dynamic configuration updates | Static configuration operation | Eliminates configuration validation overhead |

**Monitoring-reduced mode** eliminates detailed metric collection and observability features while retaining core circuit breaker functionality. This mode significantly reduces instrumentation overhead in resource-constrained environments.

**Configuration-frozen mode** prevents dynamic configuration updates to eliminate configuration validation and synchronization overhead. The system operates with static configuration that cannot be changed without restart, but maintains full functionality within those constraints.

#### Error Recovery Communication

Recovery procedures must communicate their status and impact to both application code and operational monitoring systems. **Recovery event notifications** inform applications when circuit breaker functionality changes so they can adapt their behavior accordingly.

**Operational visibility into recovery procedures** ensures that system operators understand when degraded modes are active and what functionality is temporarily unavailable. This visibility prevents operators from attempting to use features that have been disabled during recovery.

| Recovery Event Type | Notification Recipients | Information Provided | Expected Response |
|-------------------|------------------------|---------------------|------------------|
| Primary Recovery Initiated | Application logs, metrics | Recovery trigger, expected duration | Continue normal operation |
| Degraded Mode Activated | Operations dashboard, alerts | Disabled features, performance impact | Operational awareness, avoid disabled features |
| Component Recovery Completed | Application metrics | Restored functionality, performance impact | Resume using recovered features |
| Emergency Bypass Activated | Critical alerts, operations | Complete protection loss, direct service access | Immediate operational attention |
| Full Functionality Restored | System logs, metrics | All features operational | Resume normal monitoring |

**Recovery progress tracking** provides visibility into multi-step recovery procedures so operators can understand recovery status and estimated completion time. This tracking helps distinguish between recovery procedures that are progressing normally versus those that have become stuck.

**Recovery rollback procedures** handle cases where recovery attempts fail or cause additional problems. These procedures can abort recovery attempts and restore previous degraded but stable operation rather than continuing with problematic recovery processes.

### Implementation Guidance

The error handling and edge case management system requires careful attention to error classification logic, robust recovery procedures, and comprehensive testing of failure scenarios. The implementation must handle both obvious error cases and subtle edge conditions that emerge in distributed environments.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Classification | Switch statement on error types | Pluggable classifier interface with rule engine |
| Recovery Coordination | Direct function calls | Event-driven recovery with state machine |
| Edge Case Testing | Manual scenario testing | Property-based testing with failure injection |
| Configuration Validation | Schema validation | Multi-stage validation with dependency checking |
| Time Handling | Standard library time package | Monotonic clock abstraction with leap second support |

#### File Structure for Error Handling

```
internal/circuit/
  errors/
    classifier.go              ← Error classification logic
    classifier_test.go         ← Classification rule tests
    recovery.go                ← Component recovery procedures  
    recovery_test.go           ← Recovery procedure tests
    edge_cases.go              ← Edge case handling utilities
    edge_cases_test.go         ← Edge case scenario tests
  testing/
    chaos/
      failure_injector.go      ← Controlled failure injection
      edge_case_scenarios.go   ← Predefined edge case tests
    mocks/
      clock.go                 ← Mock clock for time-based edge cases
      error_generator.go       ← Configurable error generation
```

#### Error Classification Implementation

```go
// ErrorClassifier determines which errors should affect circuit breaker state
type ErrorClassifier interface {
    // IsCircuitFailure returns true if the error should contribute to circuit failure counts
    IsCircuitFailure(response interface{}, err error) bool
    
    // ClassifyError provides detailed error classification for metrics and logging
    ClassifyError(response interface{}, err error) ErrorClassification
}

type ErrorClassification struct {
    Category ErrorCategory
    Severity ErrorSeverity  
    ShouldRetry bool
    CircuitImpact CircuitImpactType
}

// DefaultErrorClassifier implements standard HTTP and gRPC error classification rules
type DefaultErrorClassifier struct {
    // TODO 1: Initialize HTTP status code classification map
    //         Map 5xx codes to CircuitFailure, 4xx codes to PassThrough (except 408, 429)
    // TODO 2: Initialize gRPC status code classification map  
    //         Map UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED to CircuitFailure
    // TODO 3: Set up timeout and connection error detection patterns
    //         Network errors, DNS failures should be CircuitFailure
    httpStatusRules map[int]ErrorCategory
    grpcStatusRules map[string]ErrorCategory
    timeoutPatterns []string
}

func (c *DefaultErrorClassifier) IsCircuitFailure(response interface{}, err error) bool {
    // TODO 1: Check for network-level errors (connection refused, timeout, DNS failure)
    //         These always indicate circuit failures regardless of response
    // TODO 2: For HTTP responses, check status code against httpStatusRules
    //         Return true for 5xx codes (except 501), 408 timeout, 429 rate limit
    // TODO 3: For gRPC responses, extract status code and check against grpcStatusRules
    //         Return true for UNAVAILABLE, DEADLINE_EXCEEDED, INTERNAL, etc.
    // TODO 4: For unknown error types, err on side of circuit impact
    //         Log unknown error types for investigation
}
```

#### Recovery Procedure Framework

```go
// RecoveryManager coordinates component recovery procedures during failures
type RecoveryManager struct {
    // TODO 1: Initialize recovery strategy registry with component-specific procedures
    //         Map component types to their recovery function implementations
    // TODO 2: Set up recovery event notification channels
    //         Components subscribe to recovery events for coordination
    // TODO 3: Configure recovery timeout and retry parameters
    //         Prevent recovery procedures from running indefinitely
    strategies map[ComponentType]RecoveryStrategy
    eventBus   chan RecoveryEvent
    timeouts   map[ComponentType]time.Duration
}

type RecoveryStrategy interface {
    // Recover attempts to restore component to healthy state
    Recover(ctx context.Context, component interface{}) RecoveryResult
    
    // CanRecover determines if recovery is possible for given failure mode  
    CanRecover(failure FailureMode) bool
    
    // EstimateRecoveryTime provides expected recovery duration for planning
    EstimateRecoveryTime(failure FailureMode) time.Duration
}

// CircuitBreakerRecoveryStrategy handles circuit breaker component failures
type CircuitBreakerRecoveryStrategy struct{}

func (s *CircuitBreakerRecoveryStrategy) Recover(ctx context.Context, component interface{}) RecoveryResult {
    // TODO 1: Cast component to *CircuitBreaker and validate type
    //         Return error if component is not circuit breaker instance
    // TODO 2: Assess failure mode by examining circuit state consistency
    //         Check for stuck timers, corrupted counters, invalid state transitions
    // TODO 3: For state corruption: reset to closed state with zero counters
    //         Clear all accumulated failure counts and success counts  
    // TODO 4: For timer issues: cancel existing recovery timer and create new one
    //         Ensure only one recovery timer exists per circuit
    // TODO 5: For metric inconsistencies: reset sliding window and restart collection
    //         Preserve configuration but clear operational state
}
```

#### Edge Case Testing Framework

```go
// EdgeCaseTestSuite provides systematic testing of edge case scenarios
type EdgeCaseTestSuite struct {
    // TODO 1: Initialize mock clock for time-based edge case simulation
    //         Support forward jumps, backward jumps, leap seconds
    // TODO 2: Set up failure injection framework for component failures
    //         Inject failures at precise timing for race condition testing
    // TODO 3: Configure concurrent request generators for race condition testing
    //         Generate high concurrency loads to trigger race conditions
    mockClock     *MockClock
    failureInjector *FailureInjector
    loadGenerator  *ConcurrentLoadGenerator
}

func (suite *EdgeCaseTestSuite) TestClockJumpScenario(t *testing.T) {
    // TODO 1: Create circuit breaker with sliding window using mock clock
    //         Initialize with normal time progression and some request history
    // TODO 2: Generate normal request pattern to establish baseline metrics
    //         Mix of successful and failed requests to create realistic failure rate
    // TODO 3: Simulate forward clock jump of 2 hours
    //         Verify sliding window handles time jump without corruption
    // TODO 4: Send new requests and verify metrics calculation correctness
    //         Ensure old buckets are properly aged out, new buckets created
    // TODO 5: Simulate backward clock jump of 30 minutes
    //         Verify circuit breaker rejects requests with past timestamps
}

func (suite *EdgeCaseTestSuite) TestConcurrentStateTransition(t *testing.T) {  
    // TODO 1: Set up circuit breaker near failure threshold
    //         Configure so next failure will trigger state transition
    // TODO 2: Launch multiple goroutines to send failing requests simultaneously
    //         Use barrier synchronization to ensure concurrent failure processing
    // TODO 3: Verify only one state transition occurs despite concurrent failures
    //         Check that duplicate timers are not created
    // TODO 4: Verify failure count consistency after concurrent updates
    //         Ensure no lost updates or race conditions in counter management
}
```

#### Milestone Checkpoints

After implementing error classification (Milestone 1):
- Run `go test ./internal/circuit/errors/...` - all tests should pass
- Test with mock server returning various HTTP status codes - verify correct classification
- Check that 5xx errors increment failure count while 4xx errors don't affect circuit state
- Verify timeout and connection errors are classified as circuit failures

After implementing recovery procedures (Milestone 2):
- Inject component failures using chaos testing framework
- Verify circuit breakers recover to stable state within configured timeout
- Test degraded operation modes under resource constraints
- Confirm recovery event notifications reach monitoring systems

After implementing edge case handling (Milestone 3):
- Run edge case test suite with clock manipulation scenarios
- Test concurrent request processing under high load
- Verify configuration update scenarios don't corrupt circuit state
- Validate network partition and infrastructure failure handling


## Testing Strategy

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

### Mental Model: Quality Assurance Laboratory

Think of the testing strategy for a circuit breaker system like a comprehensive quality assurance laboratory for medical devices. Just as medical devices undergo rigorous unit testing of individual components (like testing a single sensor in isolation), integration testing with other medical equipment (ensuring the device works with monitors, computers, and other hospital systems), and stress testing under extreme conditions (simulating power outages, network failures, and high-demand scenarios), our circuit breaker system requires the same multi-layered testing approach.

The testing laboratory has different stations: the component testing bench where individual parts are validated in isolation, the integration testing room where components work together under controlled conditions, and the chaos testing chamber where devices face unpredictable real-world scenarios. Each testing station serves a specific purpose and catches different types of defects that other stations might miss.

A comprehensive testing strategy for the circuit breaker system ensures that individual components behave correctly in isolation, work together seamlessly in realistic scenarios, and maintain resilience under chaotic failure conditions. The testing approach spans multiple layers: unit testing validates core logic with precise control over dependencies, integration testing verifies end-to-end behavior with real clients and mock services, chaos testing injects controlled failures to validate resilience properties, and milestone checkpoints provide systematic verification points throughout development.

The testing strategy addresses several critical challenges unique to circuit breaker systems. Circuit breakers are stateful components that transition between states based on temporal events and failure patterns, making them particularly sensitive to timing issues and race conditions. They interact with external services whose behavior cannot be precisely controlled, requiring sophisticated mocking strategies. They must maintain thread safety under high concurrency loads while preserving correctness of state transitions. Finally, they implement complex failure detection and recovery logic that requires systematic validation across numerous failure scenarios.

### Unit Testing Approach

Unit testing forms the foundation of circuit breaker validation by testing individual components in complete isolation with mocked dependencies. The unit testing approach focuses on validating core logic without external dependencies, ensuring deterministic behavior through controlled timing and failure injection, and verifying thread safety through concurrent test scenarios.

The unit testing strategy employs several key principles. **Dependency isolation** ensures that each component under test has all external dependencies mocked or stubbed, allowing precise control over inputs and verification of outputs. **Deterministic timing** replaces wall clock dependencies with a controllable `MockClock` implementation that allows tests to advance time artificially and trigger timeout events predictably. **Failure injection** provides systematic ways to simulate various failure modes and error conditions without relying on actual network failures or service unavailability.

#### Core Component Testing Matrix

The following table outlines the comprehensive testing matrix for core circuit breaker components:

| Component | Test Categories | Mock Dependencies | Key Test Scenarios |
|-----------|----------------|-------------------|-------------------|
| `CircuitBreaker` | State transitions, failure counting, timing | `MockClock`, mock downstream function | Failure threshold triggering, recovery timeout, half-open success/failure |
| `SlidingWindow` | Bucket rotation, failure rate calculation | `MockClock` for time control | Bucket boundaries, failure rate accuracy, concurrent updates |
| `ServiceRegistry` | Circuit creation, configuration management | Mock configuration loader | Per-service isolation, configuration updates, concurrent access |
| `ClientWrapper` | Request routing, error classification | Mock HTTP client, mock registry | Service identification, error classification, fallback execution |
| `FallbackRegistry` | Fallback chain execution | Mock fallback functions | Chain ordering, failure propagation, depth limiting |
| `MetricsCollector` | Metrics aggregation, snapshot creation | None (pure data structure) | Concurrent updates, aggregation accuracy, snapshot consistency |

#### State Machine Testing Strategy

The circuit breaker state machine requires exhaustive testing of all state transitions and edge cases. State machine testing focuses on validating transition triggers, ensuring atomic state changes, and verifying that invalid transitions are properly rejected.

**Decision: State Machine Test Structure**
- **Context**: State machines have complex transition logic that must be validated systematically
- **Options Considered**: Property-based testing, exhaustive state enumeration, scenario-based testing
- **Decision**: Combination of exhaustive enumeration for valid transitions and scenario-based testing for edge cases
- **Rationale**: Exhaustive enumeration catches all valid transitions, while scenario testing handles complex timing and concurrency issues
- **Consequences**: Higher test coverage but requires more test code and longer execution time

The state machine testing employs a systematic approach with transition tables and edge case scenarios:

| Current State | Trigger Event | Expected Next State | Validation Steps |
|---------------|---------------|-------------------|------------------|
| `StateClosed` | Failure threshold reached | `StateOpen` | Verify failure count, timer start, metrics update |
| `StateOpen` | Recovery timeout expires | `StateHalfOpen` | Verify timeout duration, half-open counter reset |
| `StateHalfOpen` | Success in recovery | `StateClosed` | Verify success threshold, counter reset, timer cleanup |
| `StateHalfOpen` | Failure in recovery | `StateOpen` | Verify immediate transition, timer restart |
| `StateClosed` | Below failure threshold | `StateClosed` | Verify counter updates, no state change |
| `StateOpen` | Request attempt | `StateOpen` | Verify immediate rejection, no downstream call |

#### Concurrent Testing Framework

Circuit breakers must maintain correctness under high concurrency loads. The concurrent testing framework validates thread safety by subjecting components to simultaneous operations from multiple goroutines while verifying that invariants are maintained.

The concurrent testing approach uses several strategies. **Race condition detection** employs the Go race detector and systematic concurrent access patterns to identify potential data races. **Invariant checking** validates that component invariants hold true even under concurrent modification, such as ensuring that failure counts never become negative or state transitions follow valid paths. **Load testing** subjects components to high-throughput scenarios that mirror production usage patterns.

| Concurrency Test Type | Target Component | Test Pattern | Invariants Verified |
|----------------------|------------------|--------------|-------------------|
| Concurrent Execute | `CircuitBreaker` | Multiple goroutines calling `Execute` | State consistency, counter accuracy |
| Parallel Metrics Collection | `MetricsCollector` | Concurrent recording and snapshots | No data races, consistent aggregation |
| Simultaneous Configuration Updates | `ServiceRegistry` | Config changes during circuit usage | Configuration consistency, no service interruption |
| Concurrent Window Updates | `SlidingWindow` | Parallel bucket rotations and updates | Bucket integrity, accurate failure rates |

#### Mock Framework Design

Effective unit testing requires sophisticated mocking capabilities that provide both control over component behavior and visibility into component interactions. The mock framework design focuses on controllable failure injection, deterministic timing, and comprehensive interaction tracking.

**MockClock Implementation Strategy**

The `MockClock` serves as the cornerstone of deterministic testing by replacing time-based dependencies with controllable alternatives. The mock clock maintains an internal time state that advances only when explicitly instructed, allowing tests to precisely control timing-dependent behavior.

| MockClock Method | Behavior | Test Usage |
|------------------|----------|------------|
| `Now() time.Time` | Returns controlled current time | Verify timestamp-dependent logic |
| `After(duration) <-chan time.Time` | Returns controllable timer channel | Trigger timeout events on demand |
| `AdvanceTime(duration)` | Moves internal clock forward | Simulate time passage for timeout testing |
| `SetTime(timestamp)` | Sets absolute time | Test specific time-based scenarios |

**MockServer Design for HTTP Testing**

The `MockServer` provides controlled HTTP service simulation for testing client wrapper behavior and error classification logic. The mock server supports configurable failure rates, latency simulation, and response customization.

| MockServer Configuration | Purpose | Test Scenarios |
|-------------------------|---------|----------------|
| `SetFailureRate(percentage)` | Controls random failure rate | Test failure threshold triggering |
| `SetLatency(milliseconds)` | Simulates network delays | Test timeout behavior |
| `SetResponsePattern(pattern)` | Custom response logic | Test error classification |
| `EnableConnectionErrors()` | Simulates network issues | Test connection failure handling |

#### Error Classification Testing

Error classification logic determines which errors should affect circuit breaker state and which should be treated as pass-through errors. This logic requires comprehensive testing across various error types and response patterns.

The error classification testing strategy validates the `DefaultErrorClassifier` implementation across multiple dimensions:

| Error Type | Expected Classification | Test Verification | Circuit Impact |
|------------|------------------------|-------------------|----------------|
| HTTP 500 Server Error | Circuit failure | Should count toward threshold | Yes |
| HTTP 404 Not Found | Pass-through error | Should not affect circuit state | No |
| Connection timeout | Circuit failure | Should trigger failure counting | Yes |
| DNS resolution failure | Circuit failure | Should count as downstream failure | Yes |
| HTTP 400 Bad Request | Pass-through error | Should not affect circuit state | No |
| gRPC UNAVAILABLE | Circuit failure | Should count toward threshold | Yes |
| gRPC INVALID_ARGUMENT | Pass-through error | Should not affect circuit state | No |

#### Common Pitfalls in Unit Testing

⚠️ **Pitfall: Testing with Wall Clock Time**

Many developers attempt to test timing-dependent behavior using actual time delays with `time.Sleep()`. This approach creates flaky tests that depend on system timing and execution speed. Wall clock testing also makes tests slow and unpredictable, particularly in continuous integration environments with variable load.

The correct approach uses `MockClock` injection to control time artificially. Instead of waiting for actual timeouts, tests advance the mock clock and verify that timeout events trigger appropriately. This makes tests both faster and deterministic.

⚠️ **Pitfall: Insufficient Concurrency Testing**

Circuit breakers are inherently concurrent components, but many unit tests only verify single-threaded behavior. This leads to race conditions and data corruption in production environments that don't surface during testing.

Comprehensive concurrency testing requires systematic concurrent access patterns using multiple goroutines and race condition detection tools. Every shared data structure and state transition must be tested under concurrent modification scenarios.

⚠️ **Pitfall: Mocking Too Little vs. Too Much**

Under-mocking leads to unit tests that depend on external services or network conditions, making them slow and unreliable. Over-mocking creates tests that verify mock behavior instead of actual component logic, providing false confidence.

The correct balance mocks external dependencies (HTTP clients, clocks, downstream services) while testing actual component logic. Mock interfaces at system boundaries but avoid mocking internal component interactions within the system under test.

### Integration Testing

Integration testing validates end-to-end circuit breaker behavior by testing components working together with real HTTP/gRPC clients and mock downstream services. Integration testing focuses on realistic request flows, proper service identification, and correct error handling across component boundaries.

The integration testing strategy bridges the gap between unit testing's isolated component validation and production deployment's complex interactions. Integration tests use real HTTP clients and gRPC connections while controlling downstream service behavior through sophisticated mock services. This approach validates that components correctly integrate while maintaining predictable test conditions.

#### Integration Test Architecture

Integration testing employs a layered architecture that mirrors production deployment while maintaining test controllability. The architecture includes real client libraries, actual network communication, and controllable mock services that simulate various downstream behaviors.

**Decision: Real Clients with Mock Services**
- **Context**: Integration testing must validate actual client integration without depending on external services
- **Options Considered**: Mock everything, use real external services, real clients with mock services
- **Decision**: Real HTTP/gRPC clients with sophisticated mock downstream services
- **Rationale**: Real clients catch integration issues while mock services provide controllable failure scenarios
- **Consequences**: More complex test setup but higher confidence in client integration correctness

The integration test architecture components work together to provide comprehensive end-to-end validation:

| Component Layer | Real Implementation | Mock/Controlled Element | Purpose |
|----------------|-------------------|------------------------|---------|
| Client Layer | `http.Client`, gRPC client | None - actual client libraries | Validate real client integration |
| Circuit Breaker Layer | `ClientWrapper`, `ServiceRegistry` | None - actual circuit breaker logic | Test circuit breaker behavior |
| Network Layer | HTTP/gRPC protocols | None - actual network calls | Validate protocol handling |
| Service Layer | Mock HTTP/gRPC servers | Controllable responses and failures | Simulate downstream behavior |

#### Service Identification Testing

Service identification determines which circuit breaker instance handles each request. This logic must correctly extract service identifiers from various request types and route requests to appropriate circuit breaker instances.

Service identification testing validates the `ServiceExtractor` implementations across different protocols and configuration scenarios:

| Request Type | Service Extraction Method | Expected Service ID | Test Verification |
|--------------|---------------------------|-------------------|------------------|
| HTTP with hostname | Extract from URL host | `api.example.com` | Verify correct circuit selection |
| HTTP with path prefix | Extract from URL path | `user-service` | Test path-based routing |
| gRPC with service name | Extract from gRPC metadata | `UserService.GetUser` | Validate gRPC service identification |
| Custom header identification | Extract from HTTP headers | Header value | Test custom identification logic |

#### End-to-End Request Flow Testing

End-to-end request flow testing validates complete request processing from client initiation through circuit breaker evaluation, downstream service interaction, and response processing. These tests verify that all components work together correctly under various conditions.

The request flow testing covers multiple scenarios with detailed validation at each step:

**Successful Request Flow Test**
1. Client initiates HTTP request through `ClientWrapper`
2. `ServiceExtractor` identifies target service from request URL
3. `ServiceRegistry` retrieves appropriate `CircuitBreaker` instance
4. Circuit breaker in `StateClosed` allows request to proceed
5. Request reaches mock downstream service successfully
6. Response returns through circuit breaker to client
7. Circuit breaker records successful request in metrics
8. Client receives successful response

**Circuit Open Request Flow Test**
1. Mock service configured for 100% failure rate
2. Multiple requests trigger failure threshold
3. Circuit breaker transitions to `StateOpen`
4. Subsequent request immediately returns `ErrCircuitOpen`
5. No downstream service call is made
6. Fallback function executes if configured
7. Circuit metrics reflect open state

**Half-Open Recovery Testing**
1. Circuit breaker in `StateOpen` due to previous failures
2. Mock clock advances past recovery timeout
3. Circuit transitions to `StateHalfOpen`
4. First request in half-open state is allowed through
5. Mock service returns success response
6. Circuit breaker records success and transitions to `StateClosed`
7. Subsequent requests flow normally

#### HTTP Client Integration Testing

HTTP client integration testing validates the `ClientWrapper` implementation with real HTTP clients and various request patterns. These tests ensure that circuit breaker functionality transparently integrates with existing HTTP client usage patterns.

The HTTP integration testing framework validates multiple client scenarios:

| Integration Scenario | Test Setup | Validation Points | Expected Behavior |
|---------------------|------------|-------------------|------------------|
| Standard HTTP GET | Real `http.Client` with mock server | Request interception, service ID extraction | Transparent circuit breaker application |
| HTTP POST with body | JSON request body to mock service | Body preservation, response handling | Request content unchanged |
| Custom headers | Authentication headers and custom metadata | Header propagation, metadata preservation | Headers reach downstream service |
| Connection pooling | Multiple requests through same client | Pool behavior, circuit isolation | Connection reuse with circuit protection |
| Request timeout | Client timeout vs circuit timeout | Timeout precedence, error handling | Appropriate timeout behavior |

#### gRPC Interceptor Testing

gRPC interceptor testing validates circuit breaker integration through gRPC unary and streaming interceptors. The testing framework uses real gRPC clients with mock gRPC servers to verify transparent circuit breaker application.

**gRPC Unary Interceptor Testing**

Unary interceptor testing validates single request-response RPC calls through the circuit breaker. The tests verify proper metadata handling, error conversion, and service identification from gRPC method names.

| gRPC Test Scenario | Method Under Test | Validation Points | Expected Outcome |
|-------------------|-------------------|-------------------|------------------|
| Successful unary call | `UserService.GetUser` | Service extraction, response handling | Normal RPC behavior |
| Service unavailable | `UserService.CreateUser` | gRPC status code handling | Circuit failure counting |
| Authentication failure | `UserService.DeleteUser` | Error classification | Pass-through error |
| Request timeout | Long-running RPC call | Timeout vs circuit timeout | Appropriate timeout handling |

**gRPC Streaming Interceptor Testing**

Streaming interceptor testing validates server-side and client-side streaming RPC calls through the circuit breaker. Streaming calls require special handling for circuit state evaluation and error propagation.

The streaming test scenarios validate circuit behavior across stream lifecycles:

| Streaming Type | Test Scenario | Circuit Evaluation Point | Expected Behavior |
|----------------|---------------|-------------------------|------------------|
| Server streaming | Circuit open at stream start | Stream initiation | Immediate stream error |
| Client streaming | Circuit opens during stream | Mid-stream evaluation | Stream termination with circuit error |
| Bidirectional streaming | Circuit recovery during stream | Stream continuation | Stream continues after recovery |

#### Mock Service Framework

The mock service framework provides sophisticated downstream service simulation for integration testing. Mock services support configurable failure patterns, latency simulation, and response customization while maintaining realistic service behavior.

**HTTP Mock Service Capabilities**

The HTTP mock service implementation supports comprehensive failure simulation and response control:

| Mock Service Feature | Configuration Options | Test Usage | Implementation Notes |
|---------------------|----------------------|------------|-------------------|
| Failure rate control | Percentage-based failures | Threshold testing | Random or deterministic failure patterns |
| Latency simulation | Fixed or variable delays | Timeout testing | Configurable delay distributions |
| Response customization | Status codes, headers, body | Error classification testing | Pattern-based response rules |
| Connection simulation | Connection drops, refused connections | Network failure testing | TCP-level failure simulation |

**gRPC Mock Service Implementation**

The gRPC mock service provides equivalent functionality for gRPC-based testing scenarios:

| gRPC Mock Feature | Configuration | Test Application | Technical Implementation |
|------------------|---------------|------------------|------------------------|
| Status code control | gRPC status enumeration | Error classification | Controlled gRPC status returns |
| Metadata handling | Custom metadata responses | Metadata propagation testing | gRPC metadata manipulation |
| Streaming control | Stream interruption patterns | Streaming failure scenarios | Stream lifecycle management |
| Service registration | Dynamic service registration | Multi-service testing | gRPC service descriptor management |

#### Performance Integration Testing

Performance integration testing validates circuit breaker behavior under load conditions that mirror production usage patterns. These tests ensure that circuit breaker overhead remains minimal and that performance degrades gracefully under various load scenarios.

The performance testing framework measures multiple dimensions:

| Performance Metric | Measurement Method | Acceptance Criteria | Test Scenarios |
|-------------------|-------------------|-------------------|----------------|
| Request throughput | Requests per second with/without circuit breaker | <5% overhead in normal operation | High-concurrency load testing |
| Latency overhead | Request latency distribution | <1ms P99 latency increase | Latency measurement across percentiles |
| Memory overhead | Heap allocation and GC pressure | Minimal allocation increase | Memory profiling during load |
| CPU overhead | CPU utilization with circuit breaker active | <2% CPU increase | CPU profiling under load |

#### Common Pitfalls in Integration Testing

⚠️ **Pitfall: Insufficient Mock Service Realism**

Many integration tests use overly simplistic mock services that don't accurately simulate real downstream service behavior. Simple mocks that always return the same response or fail deterministically don't catch issues with request routing, service identification, or error handling.

Realistic mock services should simulate variable latency, intermittent failures, and realistic response patterns. Mock services should also properly handle HTTP semantics like connection pooling, keep-alive behavior, and standard response headers.

⚠️ **Pitfall: Testing Only Happy Path Integration**

Integration tests often focus primarily on successful request flows while neglecting error conditions and edge cases. This leads to insufficient validation of error handling, fallback execution, and recovery behavior.

Comprehensive integration testing requires systematic coverage of failure scenarios, including network failures, timeout conditions, malformed responses, and partial failures. Each error condition should be tested to ensure proper circuit breaker behavior.

⚠️ **Pitfall: Race Conditions in Mock Services**

Mock services that handle concurrent requests without proper synchronization can introduce race conditions that cause test flakiness. Mock service state modifications during test execution can lead to unpredictable test outcomes.

Mock services require proper synchronization for state changes and configuration updates. Test frameworks should provide atomic operations for mock service configuration and clear isolation between test cases.

### Chaos Testing Framework

Chaos testing validates circuit breaker resilience by injecting controlled failures and observing system behavior under adverse conditions. The chaos testing framework systematically explores failure scenarios that are difficult to reproduce through traditional testing approaches, ensuring that the circuit breaker maintains correctness and availability during unpredictable failure conditions.

### Mental Model: Earthquake Simulation Laboratory

Think of chaos testing like an earthquake simulation laboratory for building safety testing. Just as engineers subject buildings to controlled seismic activity to validate structural integrity under extreme conditions, chaos testing subjects the circuit breaker system to controlled failure injection to validate resilience properties. The earthquake simulator can generate various types of ground motion - sudden jolts, rolling waves, sustained shaking - to test different aspects of building response. Similarly, chaos testing injects various failure patterns - network partitions, service crashes, resource exhaustion - to test different aspects of circuit breaker behavior.

The earthquake laboratory has safety measures in place: emergency stops, structural monitoring, and controlled failure boundaries to prevent damage to the test facility. Chaos testing requires similar safeguards: failure injection controls, system monitoring, and containment boundaries to prevent chaos from affecting production systems or causing permanent damage to test environments.

The chaos testing framework focuses on validating emergent system properties that arise from component interactions under stress. While unit and integration testing validate known failure paths with predictable inputs, chaos testing explores unknown failure modes and validates system behavior under conditions that were not explicitly designed for.

#### Chaos Testing Architecture

The chaos testing architecture provides systematic failure injection capabilities with safety controls and comprehensive observability. The architecture separates failure injection mechanisms from system monitoring and safety controls, allowing controlled exploration of failure scenarios while maintaining system observability and safety boundaries.

**Decision: Controlled Chaos with Safety Boundaries**
- **Context**: Chaos testing must validate real failure resilience without causing system damage or test environment instability
- **Options Considered**: Production chaos testing, isolated chaos testing, hybrid approach with safety controls
- **Decision**: Isolated chaos testing with comprehensive safety boundaries and monitoring
- **Rationale**: Provides realistic failure scenarios while maintaining system safety and test reproducibility
- **Consequences**: More complex test infrastructure but safer and more controllable chaos exploration

The chaos testing architecture incorporates multiple layers of control and monitoring:

| Architecture Layer | Components | Responsibilities | Safety Mechanisms |
|-------------------|------------|------------------|------------------|
| Chaos Controller | `FailureInjector`, `ChaosScheduler` | Orchestrate failure scenarios | Emergency stop, failure boundaries |
| Target System | Circuit breaker system under test | Normal operation under chaos | Health monitoring, automatic recovery |
| Monitoring Layer | `MetricsCollector`, `HealthChecker` | System observability during chaos | Automated alerts, safety thresholds |
| Safety Layer | `SafetyController`, `EmergencyStop` | Prevent permanent damage | Automatic intervention, rollback capabilities |

#### Failure Injection Mechanisms

The failure injection framework provides systematic ways to introduce various types of failures into the circuit breaker system. Failure injection operates at multiple levels: network failures, service failures, resource exhaustion, and timing anomalies.

**Network Failure Injection**

Network failure injection simulates various network conditions that affect service communication. The network chaos framework can inject latency, packet loss, connection drops, and bandwidth limitations to test circuit breaker behavior under adverse network conditions.

| Network Failure Type | Injection Mechanism | Test Scenarios | Expected Circuit Behavior |
|---------------------|-------------------|----------------|-------------------------|
| Connection timeout | TCP connection blocking | Slow connection establishment | Timeout detection and circuit opening |
| Request timeout | Response delay injection | Slow service responses | Request timeout handling |
| Connection drops | TCP connection termination | Mid-request connection loss | Connection error classification |
| Intermittent failures | Probabilistic packet loss | Unreliable network conditions | Failure threshold management |
| Network partitions | Route blocking | Service isolation | Service unavailability detection |

**Service Failure Injection**

Service failure injection simulates downstream service problems including crashes, resource exhaustion, and degraded performance. The service chaos framework can introduce various service-level failures while monitoring circuit breaker response.

| Service Failure Pattern | Implementation | Duration Control | Recovery Simulation |
|------------------------|----------------|------------------|-------------------|
| Complete service crash | Process termination | Fixed duration | Automatic restart |
| Memory exhaustion | Resource consumption | Gradual increase | Memory cleanup |
| CPU exhaustion | High CPU load injection | Sustained load | Load reduction |
| Disk space exhaustion | Disk space consumption | Progressive filling | Space cleanup |
| Database connection exhaustion | Connection pool saturation | Connection limit | Connection release |

#### Chaos Test Scenarios

Chaos testing explores complex failure scenarios that combine multiple failure modes and test system behavior under realistic adverse conditions. Each chaos scenario has specific objectives, success criteria, and safety boundaries.

**Cascade Failure Prevention Testing**

Cascade failure prevention testing validates that circuit breakers successfully prevent upstream service degradation when downstream services fail. The test introduces failures in leaf services and verifies that failures don't propagate upstream.

The cascade failure test scenario follows this pattern:
1. **Setup Phase**: Deploy service dependency chain (A → B → C → D)
2. **Baseline Phase**: Establish normal operation metrics for all services
3. **Failure Injection**: Introduce failures in service D (leaf service)
4. **Observation Phase**: Monitor circuit breaker behavior in service C
5. **Propagation Prevention**: Verify that services A and B remain healthy
6. **Recovery Phase**: Restore service D and observe recovery propagation

| Test Metric | Measurement | Success Criteria | Failure Indication |
|-------------|-------------|------------------|-------------------|
| Upstream service latency | P99 response times | Remains within 2x baseline | Significant latency increase |
| Upstream service error rate | Percentage of failed requests | Stays below 5% | Error rate spike |
| Circuit breaker state transitions | State change timing | Appropriate open/close timing | Delayed or missing transitions |
| Recovery propagation time | Time to full recovery | Complete recovery within 60 seconds | Extended recovery time |

**Resource Exhaustion Testing**

Resource exhaustion testing validates circuit breaker behavior when system resources become constrained. The testing framework systematically exhausts different resource types while monitoring circuit breaker performance and correctness.

| Resource Type | Exhaustion Method | Monitoring Points | Expected Behavior |
|---------------|-------------------|-------------------|------------------|
| Memory | Gradual allocation increase | Heap size, GC frequency | Graceful degradation, no crashes |
| File descriptors | Connection opening without closing | FD count, connection errors | Connection limit handling |
| Network connections | Connection pool exhaustion | Pool utilization, wait times | Connection timeout handling |
| CPU capacity | High-load background processes | CPU utilization, request latency | Latency-based failure detection |

**Time-Based Chaos Testing**

Time-based chaos testing validates circuit breaker behavior under various timing anomalies including clock skew, time jumps, and scheduling delays. These conditions can affect timeout calculations, state transition timing, and metrics accuracy.

The time-based chaos scenarios explore temporal edge cases:

| Timing Anomaly | Injection Method | Test Focus | Validation Points |
|----------------|------------------|------------|------------------|
| Clock jumps forward | System clock modification | Timeout calculation accuracy | No premature timeouts |
| Clock jumps backward | System clock modification | Timer handling robustness | No stuck timers |
| High scheduling delays | CPU overload simulation | State transition timing | Robust timing logic |
| Timer resolution limitations | High-frequency operations | Timer accuracy | Acceptable timing precision |

#### Chaos Orchestration Framework

The chaos orchestration framework coordinates complex failure scenarios across multiple system components while maintaining safety controls and comprehensive observability. The orchestration system manages failure injection timing, monitors system health, and provides emergency intervention capabilities.

**Chaos Scenario Definition Language**

The chaos framework uses a declarative scenario definition language that specifies failure patterns, timing, and safety constraints. This allows systematic exploration of failure scenarios while maintaining reproducibility and safety.

```yaml
# Example scenario definition (YAML format for readability)
scenario: "downstream-service-cascade-failure"
duration: "10m"
phases:
  - name: "baseline"
    duration: "2m"
    actions: []
  - name: "failure-injection"
    duration: "5m"
    actions:
      - type: "service-failure"
        target: "user-service"
        failure: "connection-timeout"
        rate: "50%"
  - name: "recovery"
    duration: "3m"
    actions:
      - type: "restore-service"
        target: "user-service"
safety_constraints:
  - max_error_rate: "25%"
  - max_latency_increase: "5x"
  - emergency_stop_conditions:
    - "system_crash"
    - "data_corruption"
```

**Safety Monitoring and Controls**

Safety monitoring provides continuous oversight of system health during chaos testing with automatic intervention capabilities when safety boundaries are exceeded. The safety system operates independently of the chaos injection framework to provide reliable protection.

| Safety Control | Monitoring Method | Intervention Trigger | Automatic Action |
|----------------|------------------|---------------------|------------------|
| System health monitoring | Continuous health checks | Health check failure | Emergency stop and system recovery |
| Error rate monitoring | Request success rate tracking | Error rate above threshold | Gradual failure injection reduction |
| Resource usage monitoring | System resource utilization | Resource exhaustion risk | Resource cleanup and load reduction |
| Data integrity monitoring | Periodic data consistency checks | Data corruption detection | Immediate test termination and restoration |

#### Chaos Metrics and Analysis

Chaos testing generates comprehensive metrics that validate system resilience properties and identify areas for improvement. The metrics collection framework captures both system-level behavior and circuit breaker-specific behavior during chaos scenarios.

**Resilience Metrics Framework**

The resilience metrics framework measures multiple dimensions of system behavior under chaos conditions:

| Resilience Dimension | Key Metrics | Measurement Method | Success Indicators |
|---------------------|-------------|-------------------|-------------------|
| Availability | Service uptime percentage | Continuous health monitoring | >99% uptime during chaos |
| Performance | Latency percentiles under chaos | Request timing measurement | Latency within acceptable bounds |
| Correctness | Data consistency during failures | Data validation checks | No data corruption or loss |
| Recovery | Time to restore normal operation | Recovery time measurement | Recovery within target SLA |

**Circuit Breaker Behavior Analysis**

Chaos testing provides detailed analysis of circuit breaker behavior under various failure conditions. The analysis framework validates that circuit breakers respond appropriately to different failure patterns and maintain system stability.

| Circuit Breaker Metric | Analysis Focus | Expected Patterns | Anomaly Detection |
|------------------------|----------------|------------------|------------------|
| State transition frequency | Stability vs responsiveness | Appropriate sensitivity to failures | Excessive state flapping |
| False positive rate | Unnecessary circuit opening | Minimal false positives | Circuit opening without failures |
| False negative rate | Missed failure detection | Prompt failure detection | Delayed circuit opening |
| Recovery efficiency | Time to restore service | Efficient recovery testing | Slow or failed recovery |

#### Common Pitfalls in Chaos Testing

⚠️ **Pitfall: Chaos Testing in Production Without Safeguards**

Running chaos tests in production environments without proper safety controls can cause actual service outages and customer impact. Many teams attempt to run chaos experiments directly against production systems without adequate monitoring, safety boundaries, or rollback capabilities.

Safe chaos testing requires comprehensive safety mechanisms including automated monitoring, emergency stop capabilities, blast radius controls, and rapid rollback procedures. Chaos testing should progress gradually from isolated environments to production with increasing safety controls at each level.

⚠️ **Pitfall: Unrealistic Failure Scenarios**

Chaos tests that inject unrealistic failure patterns don't provide meaningful validation of system resilience. Simple failure scenarios like immediate service crashes don't reflect the complex, gradual, or intermittent failure patterns that occur in real environments.

Effective chaos testing requires realistic failure models based on actual production failure patterns. Failure injection should simulate gradual degradation, partial failures, and intermittent issues that mirror real-world service behavior.

⚠️ **Pitfall: Insufficient Chaos Test Observability**

Chaos testing without comprehensive monitoring and analysis provides limited insight into system behavior and can miss important resilience issues. Many chaos tests focus on whether the system "survives" without analyzing how well it maintains functionality during failures.

Comprehensive chaos testing requires detailed metrics collection, behavior analysis, and systematic evaluation of resilience properties. Every chaos scenario should include specific success criteria and detailed analysis of system behavior during failure conditions.

### Milestone Checkpoints

Milestone checkpoints provide systematic verification points throughout circuit breaker implementation, ensuring that each development phase meets its acceptance criteria before proceeding to the next milestone. Each checkpoint includes specific verification steps, expected outcomes, and troubleshooting guidance for common issues.

#### Milestone 1: Basic Circuit Breaker Checkpoint

The basic circuit breaker milestone establishes the core state machine functionality with proper state transitions, failure counting, and thread safety. This checkpoint validates fundamental circuit breaker behavior before adding advanced features.

**Verification Steps for Basic Circuit Breaker**

The milestone 1 checkpoint follows a systematic verification process:

1. **State Machine Validation**: Create a circuit breaker with a failure threshold of 3 and verify initial state is `StateClosed`
2. **Failure Counting Test**: Execute 3 failing requests and verify the circuit transitions to `StateOpen`
3. **Open State Behavior**: Attempt additional requests and verify they return `ErrCircuitOpen` immediately
4. **Recovery Timeout Test**: Advance mock clock past recovery timeout and verify transition to `StateHalfOpen`
5. **Half-Open Success Test**: Execute successful request in half-open state and verify transition to `StateClosed`
6. **Half-Open Failure Test**: Execute failing request in half-open state and verify transition back to `StateOpen`
7. **Thread Safety Test**: Run concurrent requests and verify state consistency without race conditions

| Verification Test | Expected Behavior | Pass Criteria | Common Failure Modes |
|------------------|-------------------|---------------|---------------------|
| Initial state | Circuit starts in `StateClosed` | `cb.State() == StateClosed` | Incorrect initialization |
| Failure threshold | Circuit opens after 3 failures | State transitions to `StateOpen` | Failure counting errors |
| Open circuit behavior | Immediate request rejection | Returns `ErrCircuitOpen` | Requests reaching downstream |
| Recovery timeout | Transition to half-open | State becomes `StateHalfOpen` | Timer not triggering |
| Half-open recovery | Successful recovery to closed | State returns to `StateClosed` | Success not resetting counters |
| Concurrent access | No race conditions | Race detector passes | Data races in state management |

**Command Line Verification**

```bash
# Run basic circuit breaker tests
go test -race ./internal/circuitbreaker -run TestBasicCircuitBreaker -v

# Expected output:
# === RUN TestBasicCircuitBreaker
# === RUN TestBasicCircuitBreaker/InitialState
# === RUN TestBasicCircuitBreaker/FailureThreshold
# === RUN TestBasicCircuitBreaker/OpenCircuitBehavior
# === RUN TestBasicCircuitBreaker/RecoveryTimeout
# === RUN TestBasicCircuitBreaker/HalfOpenRecovery
# === RUN TestBasicCircuitBreaker/ConcurrentAccess
# --- PASS: TestBasicCircuitBreaker (0.10s)
# PASS
```

**Manual Verification Steps**

1. Create a simple HTTP server that returns errors 50% of the time
2. Configure circuit breaker with failure threshold of 3 and recovery timeout of 5 seconds
3. Send 10 requests and observe circuit opening after 3 failures
4. Continue sending requests and verify immediate rejections
5. Wait 5 seconds and send another request to trigger recovery testing
6. Send successful requests and verify circuit closing

**Troubleshooting Common Issues**

| Issue Symptom | Likely Cause | Diagnostic Steps | Resolution |
|---------------|--------------|------------------|------------|
| Circuit never opens | Failure counting not working | Check failure threshold logic | Verify failure increment logic |
| Race condition failures | Concurrent state access | Run with race detector | Add proper mutex protection |
| Timer not firing | Mock clock issues | Check timer implementation | Verify mock clock timer handling |
| State transitions incorrect | Logic errors in state machine | Add debug logging | Review transition conditions |

#### Milestone 2: Advanced Features Checkpoint

The advanced features milestone adds sliding window metrics, fallback functionality, and bulkhead isolation. This checkpoint validates sophisticated failure detection and graceful degradation capabilities.

**Verification Steps for Advanced Features**

The milestone 2 checkpoint expands verification to include advanced functionality:

1. **Sliding Window Validation**: Configure 1-minute sliding window and verify failure rate calculation accuracy
2. **Fallback Execution Test**: Configure fallback function and verify execution when circuit is open
3. **Bulkhead Isolation Test**: Configure concurrency limit and verify request limiting behavior
4. **Error Classification Test**: Send different error types and verify proper classification
5. **Per-Service Configuration Test**: Configure multiple services with different parameters and verify isolation
6. **Metrics Collection Test**: Verify comprehensive metrics collection and aggregation
7. **Configuration Management Test**: Update configuration dynamically and verify immediate effect

| Advanced Feature | Test Scenario | Expected Behavior | Validation Method |
|-----------------|---------------|-------------------|------------------|
| Sliding window | Variable failure rates over time | Accurate failure rate calculation | Compare calculated vs expected rates |
| Fallback execution | Circuit open with fallback configured | Fallback function returns alternative response | Verify fallback response received |
| Bulkhead isolation | Concurrent requests exceeding limit | Excess requests rejected with `ErrBulkheadCapacity` | Count rejected requests |
| Error classification | Mix of circuit and pass-through errors | Only circuit errors affect state | Verify state changes appropriately |
| Service isolation | Multiple services with different configs | Independent circuit behavior | Verify no cross-service interference |
| Dynamic configuration | Configuration updates during operation | Immediate parameter changes | Test configuration hot-reload |

**Integration Test Execution**

```bash
# Run advanced features integration tests
go test -race ./internal/integration -run TestAdvancedFeatures -v

# Expected output showing sliding window accuracy:
# === RUN TestAdvancedFeatures
# === RUN TestAdvancedFeatures/SlidingWindowAccuracy
#   sliding_window_test.go:45: Failure rate: 0.333 (expected: 0.333)
# === RUN TestAdvancedFeatures/FallbackExecution
#   fallback_test.go:67: Fallback response: "Service temporarily unavailable"
# === RUN TestAdvancedFeatures/BulkheadIsolation
#   bulkhead_test.go:89: Concurrent requests: 10, rejected: 3
# --- PASS: TestAdvancedFeatures (2.30s)
```

**Performance Benchmark Validation**

Advanced features must maintain acceptable performance characteristics:

```bash
# Run performance benchmarks
go test -bench=BenchmarkCircuitBreaker ./internal/circuitbreaker -benchmem

# Expected performance criteria:
# BenchmarkCircuitBreakerExecute-8      1000000    1500 ns/op     48 B/op    2 allocs/op
# BenchmarkSlidingWindowUpdate-8        2000000     800 ns/op     32 B/op    1 allocs/op
# BenchmarkFallbackExecution-8           500000    3000 ns/op     96 B/op    3 allocs/op
```

Performance acceptance criteria:
- Circuit breaker execution overhead: <2ms P99 latency
- Memory allocation: <100 bytes per request
- Sliding window update: <1ms P99 latency
- Fallback execution: <5ms P99 latency

#### Milestone 3: Integration & Testing Checkpoint

The integration and testing milestone validates complete system integration with HTTP/gRPC clients and comprehensive testing infrastructure. This checkpoint ensures production-ready functionality and operational robustness.

**End-to-End Integration Verification**

The milestone 3 checkpoint validates complete system integration:

1. **HTTP Client Integration**: Deploy circuit breaker with real HTTP client and verify transparent operation
2. **gRPC Interceptor Integration**: Configure gRPC interceptor and verify RPC call protection
3. **Multi-Service Integration**: Deploy multiple services with independent circuit breakers
4. **Chaos Testing Execution**: Run chaos scenarios and verify resilience properties
5. **Monitoring Integration**: Verify metrics export and dashboard functionality
6. **Configuration Management**: Test configuration loading from multiple sources
7. **Operational Procedures**: Validate debugging, monitoring, and maintenance procedures

| Integration Component | Test Scenario | Success Criteria | Verification Method |
|----------------------|---------------|------------------|-------------------|
| HTTP client wrapper | Real HTTP requests through circuit breaker | Transparent integration with existing code | Monitor request interception |
| gRPC interceptor | RPC calls with circuit protection | No changes to existing gRPC code | Verify interceptor activation |
| Service isolation | Multiple services with failures | Independent circuit behavior | Cross-service failure isolation |
| Chaos resilience | Systematic failure injection | System maintains availability | Automated resilience testing |
| Observability | Metrics and logging integration | Complete operational visibility | Dashboard functionality |
| Configuration hot-reload | Dynamic configuration updates | No service restart required | Runtime configuration changes |

**Production Readiness Checklist**

The production readiness checklist validates operational requirements:

| Readiness Category | Requirements | Validation Steps | Documentation |
|-------------------|-------------|------------------|---------------|
| Performance | Meets latency and throughput targets | Load testing and benchmarking | Performance test reports |
| Reliability | Handles failures gracefully | Chaos testing scenarios | Resilience test results |
| Observability | Comprehensive metrics and logging | Monitoring dashboard setup | Operations runbook |
| Security | No security vulnerabilities | Security scanning and review | Security assessment |
| Configuration | Flexible configuration management | Configuration validation testing | Configuration documentation |
| Documentation | Complete operational documentation | Documentation review checklist | User and operator guides |

**Chaos Testing Validation**

```bash
# Execute comprehensive chaos testing suite
go test -timeout=30m ./test/chaos -run TestChaosScenarios -v

# Expected chaos test results:
# === RUN TestChaosScenarios
# === RUN TestChaosScenarios/DownstreamServiceFailure
#   chaos_test.go:123: Circuit opened after 3 failures (2.1s)
#   chaos_test.go:145: Recovery completed successfully (8.7s)
# === RUN TestChaosScenarios/NetworkPartition
#   chaos_test.go:167: Service maintained 99.2% availability
# === RUN TestChaosScenarios/ResourceExhaustion
#   chaos_test.go:189: Graceful degradation under memory pressure
# --- PASS: TestChaosScenarios (25m34s)
```

**Final System Verification**

The final verification ensures complete system functionality:

1. **Deploy test environment** with multiple services and realistic traffic patterns
2. **Execute comprehensive test suite** covering all functional and non-functional requirements
3. **Run extended chaos testing** to validate resilience under prolonged failure conditions
4. **Validate monitoring and alerting** to ensure operational visibility
5. **Perform load testing** to verify performance under production-like conditions
6. **Execute disaster recovery procedures** to validate system recovery capabilities
7. **Complete security assessment** to ensure no vulnerabilities are introduced

**System Health Dashboard Verification**

The system health dashboard should display:
- Current circuit breaker states for all services
- Request success rates and error rates over time
- Response latency percentiles across services
- Circuit breaker state transition history
- Active fallback executions and success rates
- Bulkhead utilization and rejection rates
- System resource utilization metrics

### Implementation Guidance

The testing strategy implementation requires sophisticated tooling and infrastructure to support comprehensive validation across multiple testing dimensions. The implementation focuses on creating reusable testing components, systematic test organization, and automated verification procedures.

#### Technology Recommendations for Testing

| Testing Layer | Simple Option | Advanced Option | Recommended for Learning |
|---------------|---------------|-----------------|-------------------------|
| Unit Testing | Go testing package with testify | Property-based testing with gopter | Go testing + testify for structured assertions |
| Mock Framework | Manual mocks with interfaces | Automated mock generation with gomock | Manual mocks for learning, gomock for productivity |
| Integration Testing | httptest package for HTTP mocking | Testcontainers for real service dependencies | httptest for circuit breaker focus |
| Chaos Testing | Simple failure injection with channels | Dedicated chaos engineering framework | Custom failure injection for understanding |
| Load Testing | Simple goroutine-based load generation | Professional load testing with k6 or Artillery | Goroutine-based for learning fundamentals |
| Time Control | Manual mock clock implementation | Advanced time mocking with clockwork | Manual implementation for deeper understanding |

#### Recommended Testing File Structure

The testing implementation follows a structured organization that separates different testing concerns while maintaining clear relationships between test types and components:

```
project-root/
├── internal/
│   ├── circuitbreaker/
│   │   ├── circuitbreaker.go           ← Core implementation
│   │   ├── circuitbreaker_test.go      ← Unit tests for state machine
│   │   ├── sliding_window_test.go      ← Unit tests for metrics
│   │   ├── testutil/                   ← Testing utilities
│   │   │   ├── mock_clock.go          ← Controllable time for testing
│   │   │   ├── mock_service.go        ← HTTP/gRPC service mocking
│   │   │   └── test_helpers.go        ← Common test setup functions
│   │   └── benchmarks/
│   │       └── performance_test.go     ← Performance benchmarks
│   ├── registry/
│   │   ├── service_registry.go         ← Service management
│   │   ├── service_registry_test.go    ← Unit tests
│   │   └── integration_test.go         ← Cross-component integration
│   └── client/
│       ├── http_wrapper.go             ← HTTP client integration
│       ├── http_wrapper_test.go        ← Unit tests
│       └── grpc_interceptor_test.go    ← gRPC integration tests
├── test/
│   ├── integration/                    ← End-to-end integration tests
│   │   ├── http_integration_test.go    ← Real HTTP client testing
│   │   ├── grpc_integration_test.go    ← Real gRPC client testing
│   │   └── multi_service_test.go       ← Multiple service scenarios
│   ├── chaos/                          ← Chaos engineering tests
│   │   ├── chaos_framework.go          ← Failure injection framework
│   │   ├── network_chaos_test.go       ← Network failure scenarios
│   │   ├── service_chaos_test.go       ← Service failure scenarios
│   │   └── recovery_chaos_test.go      ← Recovery behavior testing
│   └── load/                           ← Performance and load tests
│       ├── load_test.go                ← High-throughput testing
│       ├── stress_test.go              ← Resource exhaustion testing
│       └── endurance_test.go           ← Extended operation testing
├── examples/                           ← Example implementations
│   ├── basic_usage/                    ← Simple circuit breaker usage
│   ├── advanced_features/              ← Sliding window and fallback examples
│   └── integration_examples/           ← HTTP/gRPC integration examples
└── scripts/
    ├── run_tests.sh                    ← Comprehensive test execution
    ├── chaos_test.sh                   ← Chaos testing orchestration
    └── benchmark.sh                    ← Performance testing automation
```

#### Core Testing Infrastructure

**Mock Clock Implementation**

```go
// MockClock provides controllable time for deterministic testing
// This implementation allows tests to control time advancement and timer triggering
type MockClock struct {
    now    time.Time
    timers []*mockTimer
    mu     sync.RWMutex
}

// NewMockClock creates a mock clock starting at the specified time
func NewMockClock(startTime time.Time) *MockClock {
    // TODO 1: Initialize MockClock with starting time
    // TODO 2: Initialize empty timer slice
    // TODO 3: Return configured mock clock
}

// Now returns the current mock time
func (mc *MockClock) Now() time.Time {
    // TODO 1: Acquire read lock for thread safety
    // TODO 2: Return current mock time
    // TODO 3: Release lock
}

// After creates a controllable timer channel
func (mc *MockClock) After(d time.Duration) <-chan time.Time {
    // TODO 1: Acquire write lock for timer manipulation
    // TODO 2: Calculate timer expiration time (now + duration)
    // TODO 3: Create mockTimer with expiration time and result channel
    // TODO 4: Add timer to timers slice
    // TODO 5: Release lock and return result channel
}

// AdvanceTime moves mock clock forward and triggers expired timers
func (mc *MockClock) AdvanceTime(d time.Duration) {
    // TODO 1: Acquire write lock for time advancement
    // TODO 2: Update current time by adding duration
    // TODO 3: Iterate through timers to find expired ones
    // TODO 4: Trigger expired timers by sending time on their channels
    // TODO 5: Remove triggered timers from slice
    // TODO 6: Release lock
}

type mockTimer struct {
    expiresAt time.Time
    ch        chan time.Time
    triggered bool
}
```

**HTTP Mock Service Framework**

```go
// MockServer provides controllable HTTP service simulation for testing
type MockServer struct {
    server      *httptest.Server
    failureRate float64
    latency     time.Duration
    responses   map[string]*http.Response
    mu          sync.RWMutex
}

// NewMockServer creates a configurable mock HTTP server
func NewMockServer() *MockServer {
    // TODO 1: Create MockServer instance
    // TODO 2: Set up HTTP handler with configurable behavior
    // TODO 3: Create httptest.Server with handler
    // TODO 4: Start server and return MockServer
}

// SetFailureRate configures the percentage of requests that should fail
func (ms *MockServer) SetFailureRate(rate float64) {
    // TODO 1: Validate rate is between 0.0 and 1.0
    // TODO 2: Acquire write lock
    // TODO 3: Update failure rate
    // TODO 4: Release lock
}

// SetLatency configures artificial delay for responses
func (ms *MockServer) SetLatency(d time.Duration) {
    // TODO 1: Acquire write lock
    // TODO 2: Update latency setting
    // TODO 3: Release lock
}

// handler processes HTTP requests with configured failure and latency simulation
func (ms *MockServer) handler(w http.ResponseWriter, r *http.Request) {
    // TODO 1: Acquire read lock to read configuration
    // TODO 2: Check if request should fail based on failure rate
    // TODO 3: Apply configured latency delay
    // TODO 4: Generate appropriate response (success or failure)
    // TODO 5: Write response and set appropriate status code
}
```

**Chaos Testing Framework**

```go
// ChaosController orchestrates systematic failure injection with safety controls
type ChaosController struct {
    scenarios    map[string]*ChaosScenario
    running      bool
    safetyLimits *SafetyLimits
    monitoring   *ChaosMonitoring
    mu           sync.RWMutex
}

// ChaosScenario defines a complete chaos testing scenario
type ChaosScenario struct {
    Name           string
    Duration       time.Duration
    FailureActions []FailureAction
    SafetyChecks   []SafetyCheck
    SuccessCriteria []SuccessCriterion
}

// ExecuteScenario runs a chaos scenario with comprehensive monitoring
func (cc *ChaosController) ExecuteScenario(ctx context.Context, scenarioName string) error {
    // TODO 1: Load scenario configuration and validate
    // TODO 2: Initialize monitoring and safety systems
    // TODO 3: Execute baseline measurement phase
    // TODO 4: Begin failure injection according to scenario timeline
    // TODO 5: Continuously monitor safety limits and system health
    // TODO 6: Execute recovery phase and measure recovery metrics
    // TODO 7: Generate comprehensive test report with analysis
}

// FailureInjector provides controlled failure injection capabilities
type FailureInjector struct {
    networkChaos  *NetworkChaosInjector
    serviceChaos  *ServiceChaosInjector
    resourceChaos *ResourceChaosInjector
    activeFailures map[string]*InjectedFailure
    mu            sync.RWMutex
}

// InjectNetworkFailure introduces network-level failures
func (fi *FailureInjector) InjectNetworkFailure(target string, failureType NetworkFailureType, config FailureConfig) error {
    // TODO 1: Validate target and failure type
    // TODO 2: Create network failure injection based on type
    // TODO 3: Apply failure with specified configuration
    // TODO 4: Track active failure for cleanup
    // TODO 5: Return success or configuration error
}
```

#### Milestone Testing Checkpoints

**Milestone 1 Verification Script**

```go
// TestMilestone1BasicCircuitBreaker validates core circuit breaker functionality
func TestMilestone1BasicCircuitBreaker(t *testing.T) {
    // Test setup with mock dependencies
    mockClock := NewMockClock(time.Now())
    config := &Config{
        FailureThreshold: 3,
        RecoveryTimeout: 30 * time.Second,
        RequestTimeout:  5 * time.Second,
    }
    
    cb := NewCircuitBreaker(config, WithClock(mockClock))
    
    t.Run("InitialState", func(t *testing.T) {
        // TODO 1: Verify circuit breaker starts in StateClosed
        // TODO 2: Verify initial metrics are zero
        // TODO 3: Verify configuration is properly set
    })
    
    t.Run("FailureThreshold", func(t *testing.T) {
        // TODO 1: Execute failing function 3 times
        // TODO 2: Verify circuit opens after 3rd failure
        // TODO 3: Verify failure count is accurate
        // TODO 4: Verify state transition time is recorded
    })
    
    t.Run("OpenCircuitBehavior", func(t *testing.T) {
        // TODO 1: Ensure circuit is in StateOpen
        // TODO 2: Execute function and verify ErrCircuitOpen returned
        // TODO 3: Verify downstream function is not called
        // TODO 4: Verify metrics reflect open state
    })
    
    // Additional test cases for half-open recovery and concurrent access...
}
```

**Integration Testing Verification**

```bash
#!/bin/bash
# Integration testing verification script

echo "Running Circuit Breaker Integration Tests..."

# Start mock services for testing
echo "Starting mock HTTP service..."
go run test/integration/mock_service.go &
HTTP_SERVICE_PID=$!

echo "Starting mock gRPC service..."
go run test/integration/mock_grpc_service.go &
GRPC_SERVICE_PID=$!

# Wait for services to start
sleep 2

# Run integration tests
echo "Executing HTTP integration tests..."
go test -v ./test/integration -run TestHTTPIntegration

echo "Executing gRPC integration tests..."
go test -v ./test/integration -run TestGRPCIntegration

echo "Executing multi-service integration tests..."
go test -v ./test/integration -run TestMultiServiceIntegration

# Cleanup
echo "Cleaning up test services..."
kill $HTTP_SERVICE_PID $GRPC_SERVICE_PID

echo "Integration testing complete."
```

#### Debugging and Troubleshooting Tools

| Issue Category | Diagnostic Command | Expected Output | Troubleshooting Steps |
|----------------|-------------------|-----------------|----------------------|
| Unit test failures | `go test -v ./internal/circuitbreaker` | All tests pass with detailed output | Check mock setup, verify test assertions |
| Race conditions | `go test -race ./...` | No race warnings | Add proper locking, review concurrent access |
| Integration failures | `go test -v ./test/integration` | Services communicate properly | Check service startup, network connectivity |
| Performance issues | `go test -bench=. -benchmem` | Acceptable latency and memory usage | Profile memory allocation, optimize hot paths |
| Chaos test failures | `go test -timeout=30m ./test/chaos` | System maintains availability | Review failure injection, check safety limits |

The comprehensive testing strategy ensures that the circuit breaker implementation meets all functional and non-functional requirements while providing confidence in production deployment readiness.


## Debugging Guide

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

The debugging guide provides a systematic approach to identifying, diagnosing, and resolving issues in circuit breaker implementations. Circuit breaker systems involve complex state transitions, concurrent operations, timing-dependent behavior, and distributed system interactions that can create subtle bugs. This guide equips developers with the tools, techniques, and knowledge needed to effectively troubleshoot circuit breaker issues in development, testing, and production environments.

### Common Symptoms and Diagnosis

Understanding how to map observable symptoms to underlying root causes is essential for effective debugging. Circuit breaker issues often manifest in ways that don't immediately point to the circuit breaker itself, making systematic diagnosis critical. The following diagnostic table provides a structured approach to identifying and resolving common circuit breaker problems.

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| **All requests fail immediately with `ErrCircuitOpen`** | Circuit breaker stuck in open state due to incorrect state transitions or failed recovery timer | 1. Check current state via `State()` method<br>2. Verify failure count vs threshold<br>3. Check recovery timer status<br>4. Examine state transition logs | Reset circuit manually or fix timer initialization in `transitionToOpen()` method |
| **Circuit never opens despite failures** | Failure threshold not reached, errors not classified as circuit failures, or concurrent access issues | 1. Verify `failureCount` increments on errors<br>2. Check `ErrorClassifier` rules<br>3. Validate `FailureThreshold` configuration<br>4. Review thread safety of counter updates | Fix error classification logic or add proper locking around `recordFailure()` |
| **Circuit opens and closes rapidly (flapping)** | Recovery timeout too short, failure threshold too low, or network instability | 1. Analyze state transition frequency in metrics<br>2. Check `RecoveryTimeout` value<br>3. Monitor downstream service health<br>4. Review half-open request patterns | Increase recovery timeout, implement exponential backoff, or add jitter to prevent thundering herd |
| **Requests hang indefinitely** | Deadlock in circuit breaker logic, missing timeout configuration, or downstream service not responding | 1. Check for deadlocks using goroutine dumps<br>2. Verify `RequestTimeout` is set<br>3. Monitor downstream service response times<br>4. Check bulkhead semaphore acquisition | Add request timeouts, fix locking order, or implement proper context cancellation |
| **Memory usage grows continuously** | Sliding window buckets not being garbage collected, metrics retention too long, or service registry leaks | 1. Profile memory usage with `go tool pprof`<br>2. Check sliding window bucket rotation<br>3. Verify service registry cleanup<br>4. Monitor metrics collection overhead | Implement bucket cleanup in `SlidingWindow`, add service deregistration, or reduce metrics retention |
| **Circuit doesn't recover from half-open** | Success threshold too high, half-open requests failing, or state transition logic incorrect | 1. Monitor half-open request success rate<br>2. Check `successCount` increments<br>3. Verify success threshold configuration<br>4. Review `transitionToClosed()` conditions | Reduce success threshold, fix success counting, or verify downstream service recovery |
| **Different services affecting each other** | Shared circuit breaker instances, incorrect service identification, or bulkhead misconfiguration | 1. Verify service isolation in `ServiceRegistry`<br>2. Check `ServiceExtractor` logic<br>3. Monitor per-service metrics<br>4. Validate bulkhead configuration | Fix service ID extraction, ensure separate circuit instances, or configure independent bulkheads |
| **Fallback functions never execute** | Circuit not actually open, fallback registration missing, or fallback chain misconfigured | 1. Verify circuit state during fallback scenarios<br>2. Check fallback registration for service<br>3. Test fallback functions independently<br>4. Review fallback execution logs | Register fallbacks properly, fix circuit state detection, or debug fallback chain execution |
| **Performance degradation with circuit breaker** | Excessive locking contention, inefficient metrics collection, or sliding window overhead | 1. Profile CPU usage and lock contention<br>2. Measure metrics collection overhead<br>3. Check sliding window bucket access patterns<br>4. Monitor goroutine count and scheduling | Optimize locking granularity, reduce metrics collection frequency, or implement lock-free counters |
| **Inconsistent behavior across instances** | Clock skew between instances, configuration differences, or race conditions in shared state | 1. Check system clock synchronization<br>2. Compare configuration across instances<br>3. Look for race conditions in state transitions<br>4. Verify distributed coordination logic | Synchronize clocks, standardize configuration, or add proper synchronization primitives |

⚠️ **Pitfall: Debugging Production Issues Without Proper Observability**

Many circuit breaker issues are difficult to diagnose because insufficient logging and metrics were implemented during development. Developers often add minimal logging and then struggle to understand system behavior during production incidents. Always implement comprehensive observability before deploying circuit breakers to production environments.

### Debugging Techniques

Effective circuit breaker debugging requires a combination of static analysis, dynamic inspection, and systematic testing approaches. The following techniques provide developers with practical methods for investigating circuit breaker behavior and identifying the root causes of issues.

**State Inspection and Monitoring**

Circuit breaker state inspection forms the foundation of effective debugging. The `CircuitBreaker` component should expose comprehensive state information through the `Metrics()` method, allowing developers to understand current system behavior and historical patterns. Key inspection points include current state, failure counts, success rates, state transition timestamps, and sliding window contents.

The metrics inspection process involves taking periodic snapshots of circuit breaker state and analyzing patterns over time. During debugging sessions, developers should capture metrics before, during, and after problem scenarios to understand how system behavior changes. The `MetricsCollector` provides aggregated views across all circuit breakers, enabling system-wide analysis and identification of correlated issues across services.

For real-time debugging, developers can implement debugging endpoints that expose detailed circuit breaker information. These endpoints should return current state, recent state transitions, failure patterns, and configuration details. The debugging interface should also support filtering by service ID and time range to focus on specific problem areas.

**Logging Strategy for Circuit Breaker Events**

Strategic logging provides critical visibility into circuit breaker decision-making and state transitions. Effective logging balances information completeness with performance impact, focusing on events that indicate problems or significant state changes. The logging strategy should capture state transitions, threshold breaches, recovery attempts, and error classification decisions.

State transition logging should record the previous state, new state, trigger event, and relevant metrics at transition time. This information enables developers to reconstruct the sequence of events leading to problematic behavior. The log entries should include service ID, timestamp, failure count, success count, and configuration parameters to provide complete context.

Error classification logging helps debug issues where errors aren't being properly categorized as circuit failures or pass-through errors. Each error classification decision should be logged with the error details, classification result, and reasoning. This logging enables developers to validate and tune error classification rules based on actual system behavior.

**Mock Clock and Deterministic Testing**

Time-dependent circuit breaker behavior creates significant debugging challenges, particularly for recovery timeouts and sliding window operations. The `MockClock` interface enables deterministic testing and debugging by providing controllable time advancement. Developers can step through time-dependent scenarios at their own pace and verify expected behavior at each step.

The mock clock debugging technique involves replacing the system clock with a controllable implementation during testing and debugging sessions. This approach enables reproduction of timing-sensitive bugs, validation of timeout behavior, and testing of recovery scenarios without waiting for real time to elapse. The `AdvanceTime()` method allows precise control over time progression and timer triggering.

For debugging sliding window issues, the mock clock enables step-by-step analysis of bucket rotation, failure rate calculation, and window state evolution. Developers can advance time by specific intervals and examine sliding window contents to verify correct bucket management and metric aggregation.

**Concurrent Behavior Analysis**

Circuit breaker concurrent behavior debugging requires specialized techniques to identify race conditions, deadlocks, and resource contention issues. The Go runtime provides powerful tools for analyzing concurrent behavior, including race detection, goroutine analysis, and mutex profiling.

Race condition detection should be enabled during development and testing using the `-race` flag. Race conditions in circuit breaker implementations typically occur around state transitions, counter updates, and sliding window modifications. The race detector identifies specific memory locations where unsynchronized access occurs, enabling targeted fixes.

Deadlock analysis involves examining goroutine dumps to identify blocked goroutines and their waiting conditions. Circuit breaker deadlocks often occur when multiple locks are acquired in different orders or when goroutines wait indefinitely for resources. The `SIGQUIT` signal generates goroutine dumps that show current goroutine states and lock dependencies.

**Service Registry and Configuration Debugging**

Service registry debugging focuses on verifying correct service identification, circuit breaker instance management, and configuration application. Common issues include incorrect service ID extraction, shared circuit breaker instances across services, and configuration inheritance problems.

The service registry debugging process involves logging all service registration events, circuit breaker creation operations, and configuration updates. This logging enables verification that each service gets its own circuit breaker instance and that configuration changes are applied correctly. The debugging interface should expose the current service registry state, including registered services, circuit breaker instances, and active configurations.

Configuration debugging requires systematic verification of parameter inheritance, override application, and validation results. The `ConfigLoader` should log all configuration sources, merge operations, and final applied values. This logging helps identify configuration conflicts and ensures that intended settings are actually applied to circuit breaker instances.

### Logging and Observability

Comprehensive logging and observability form the foundation of effective circuit breaker debugging and operational monitoring. The logging strategy must balance information completeness with performance impact while providing sufficient detail to diagnose complex distributed system issues. Strategic logging points throughout the circuit breaker system enable reconstruction of system behavior and identification of problem patterns.

**Strategic Logging Points**

Circuit breaker logging should focus on decision points, state changes, and error conditions that indicate potential problems. The most critical logging points include state transitions, threshold breaches, error classification decisions, recovery attempts, and configuration changes. Each log entry should include sufficient context to understand the event's significance and relationship to overall system behavior.

State transition logging captures the fundamental circuit breaker behavior changes. When the circuit transitions from closed to open, the log entry should include the service ID, current failure count, configured threshold, recent error details, and timestamp. This information enables developers to understand why the circuit opened and whether the decision was correct based on system conditions.

Request processing logging should be selective to avoid overwhelming log volume while capturing essential decision information. Log entries should be generated for the first request that triggers a state transition, recovery testing requests in half-open state, and requests that encounter unexpected errors during processing. The request logs should include service ID, circuit state, request outcome, and processing latency.

Error classification logging provides visibility into how the `ErrorClassifier` makes decisions about which errors should affect circuit breaker state. Each classification decision should be logged with error details, classification result, and applied rules. This logging enables validation and tuning of error classification logic based on actual system behavior.

**Metrics Collection and Aggregation**

The `MetricsCollector` provides quantitative data about circuit breaker behavior across all services and time periods. Effective metrics collection focuses on key performance indicators that indicate system health, problem patterns, and operational trends. The metrics collection strategy should support both real-time monitoring and historical analysis.

Core metrics include request counts, success rates, failure rates, state transition frequencies, and recovery times for each service and circuit breaker instance. These metrics enable identification of problematic services, validation of threshold configurations, and assessment of system reliability. The metrics should be collected at multiple time granularities to support both immediate problem diagnosis and trend analysis.

Sliding window metrics provide detailed visibility into failure rate calculations and circuit breaker decision-making. The `SlidingWindow` component should expose current window contents, bucket details, and calculated failure rates. This information enables validation of sliding window behavior and tuning of window size and bucket duration parameters.

Aggregated metrics across all circuit breakers provide system-wide visibility and enable identification of correlated issues across services. The aggregation should include total request volumes, overall success rates, active circuit breaker counts by state, and recovery attempt frequencies. These metrics help operations teams understand overall system health and identify widespread issues.

**Observability Integration**

Circuit breaker observability should integrate with existing monitoring and alerting infrastructure to provide seamless operational visibility. The integration should support popular observability platforms and provide standard metrics formats for compatibility with monitoring tools.

Prometheus metrics integration enables collection of circuit breaker metrics in a standard format that integrates with existing monitoring infrastructure. The metrics should include labels for service ID, circuit state, and other relevant dimensions. Standard counter and gauge metrics should track request volumes, success rates, and state transition events.

Distributed tracing integration provides request-level visibility into circuit breaker decisions and their impact on overall request processing. Circuit breaker operations should add trace spans that include circuit state, decision rationale, and processing outcomes. This tracing enables correlation of circuit breaker behavior with downstream service performance and user-visible request outcomes.

Health check integration enables automated monitoring and alerting based on circuit breaker state. The health check endpoints should report overall system health based on circuit breaker states and provide detailed information about problematic services. The health checks should integrate with load balancers and service discovery systems to enable automatic traffic management based on circuit breaker state.

**Alerting and Notification Strategy**

Effective alerting helps operations teams respond quickly to circuit breaker issues while avoiding alert fatigue from normal operational events. The alerting strategy should focus on significant events that require human attention and provide sufficient context for effective response.

Critical alerts should be generated when circuit breakers open unexpectedly, indicating potential downstream service issues. These alerts should include service identification, failure details, affected request volume, and suggested response actions. The alerts should be rate-limited to prevent flooding during widespread outages.

Recovery alerts notify operations teams when circuit breakers successfully recover from open state, indicating that downstream services have returned to normal operation. These notifications help track system recovery progress and validate that recovery procedures are working correctly.

Configuration change alerts ensure that circuit breaker parameter modifications are properly tracked and monitored. These alerts should include before and after configuration values, change initiator, and affected services. Configuration tracking helps correlate behavioral changes with parameter modifications.

### Performance Debugging

Circuit breaker performance debugging focuses on identifying and resolving bottlenecks, resource contention, and inefficiencies that impact system performance. Performance issues in circuit breaker implementations can significantly affect overall application performance, making systematic performance analysis essential for production deployments.

**Profiling Circuit Breaker Operations**

Performance profiling provides quantitative analysis of circuit breaker resource usage and identifies optimization opportunities. The Go runtime provides comprehensive profiling tools that enable analysis of CPU usage, memory allocation, goroutine behavior, and lock contention. Effective profiling requires systematic measurement under realistic load conditions.

CPU profiling reveals where circuit breaker operations spend processing time and identifies inefficient algorithms or excessive computation. Common performance bottlenecks include inefficient sliding window calculations, excessive metrics collection overhead, and complex error classification logic. The CPU profiler shows function-level time distribution and enables identification of optimization targets.

Memory profiling identifies memory allocation patterns, garbage collection pressure, and potential memory leaks in circuit breaker implementations. Common memory issues include excessive metrics retention, sliding window bucket accumulation, and service registry growth. Memory profiling helps optimize data structures and implement appropriate cleanup procedures.

Lock contention profiling reveals synchronization bottlenecks that limit concurrent performance. Circuit breaker implementations often use locks to protect shared state, and excessive contention can significantly impact performance under high concurrency. The mutex profiler identifies hot locks and enables optimization of locking granularity and critical section duration.

**Concurrency and Locking Analysis**

Circuit breaker concurrency performance depends heavily on effective synchronization strategies that balance data consistency with concurrent access performance. Poorly designed locking can create bottlenecks that limit system scalability and responsiveness. Systematic concurrency analysis identifies contention points and optimization opportunities.

Lock granularity analysis examines the scope and duration of critical sections to identify optimization opportunities. Fine-grained locking can reduce contention but increases complexity, while coarse-grained locking simplifies implementation but may limit concurrency. The optimal strategy depends on access patterns and contention levels under realistic load.

Read-write lock usage analysis determines whether read-heavy operations can benefit from `sync.RWMutex` instead of exclusive locks. Circuit breaker state inspection and metrics collection are typically read-heavy operations that can benefit from reader-writer synchronization. However, writer starvation can occur if reads significantly outnumber writes.

Lock-free optimization opportunities involve implementing certain operations without explicit locking using atomic operations or other synchronization primitives. Simple counter updates and state checks can often be implemented with atomic operations, reducing lock contention and improving performance under high concurrency.

**Sliding Window Performance Optimization**

Sliding window implementations can significantly impact circuit breaker performance due to their continuous bucket rotation and metrics calculation requirements. Performance optimization focuses on efficient data structures, optimized calculation algorithms, and appropriate cleanup procedures.

Bucket rotation efficiency affects the overhead of maintaining sliding windows under high request volumes. Inefficient rotation implementations can create performance spikes when buckets are created or rotated. Optimization involves using ring buffer data structures, pre-allocating bucket arrays, and implementing efficient rotation algorithms.

Failure rate calculation optimization reduces the computational overhead of determining circuit breaker state transitions. Naive implementations recalculate failure rates for every request, while optimized implementations cache calculations and update incrementally. The optimization strategy should balance accuracy with computational efficiency.

Memory usage optimization addresses sliding window memory consumption, particularly for systems with many services or long window durations. Optimization techniques include bucket compaction, selective metrics retention, and efficient data structure selection. The goal is to maintain necessary functionality while minimizing memory footprint.

**Bulkhead and Resource Pool Performance**

Bulkhead pattern performance depends on efficient semaphore implementation and appropriate resource pool sizing. Performance issues can arise from excessive blocking, resource pool exhaustion, or inefficient resource management. Systematic analysis identifies optimization opportunities and configuration improvements.

Semaphore contention analysis examines blocking patterns and wait times for resource acquisition. High contention indicates potential resource pool undersizing or inefficient resource usage patterns. The analysis should consider both average and peak load conditions to ensure appropriate capacity planning.

Resource pool utilization analysis determines whether bulkhead configurations are appropriately sized for actual workloads. Underutilized pools waste resources, while overutilized pools create excessive blocking and reduced throughput. The analysis should consider request patterns, processing times, and acceptable latency targets.

Queue management performance affects bulkhead behavior under high load conditions. Inefficient queue implementations can create bottlenecks and unfair resource allocation. Optimization involves selecting appropriate queue data structures and implementing fair scheduling algorithms.

### Implementation Guidance

Circuit breaker debugging requires comprehensive tooling, systematic approaches, and practical techniques for identifying and resolving issues. The implementation guidance provides concrete tools and code structures that enable effective debugging of circuit breaker systems.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Logging Framework | Standard library `log` package with structured JSON | Structured logging with `logrus` or `zap` |
| Metrics Collection | In-memory counters with HTTP exposition | Prometheus client with metric registry |
| Profiling Tools | Built-in `go tool pprof` with HTTP handlers | Continuous profiling with `pprof` and visualization |
| Testing Framework | Standard `testing` package with table tests | `testify` suite with assertion helpers |
| Mock Services | `httptest` package with configurable responses | Docker-based test services with chaos injection |
| Observability Platform | Simple metrics HTTP endpoint | Grafana dashboards with Prometheus backend |

**Recommended File/Module Structure**

```
circuit-breaker/
  internal/
    debug/
      inspector.go           ← state inspection utilities
      profiler.go           ← performance profiling helpers
      mock_clock.go         ← controllable time for testing
      mock_server.go        ← configurable test server
      chaos_injector.go     ← failure injection utilities
    observability/
      metrics_collector.go  ← centralized metrics collection
      logger.go            ← structured logging configuration
      tracing.go           ← distributed tracing integration
      health_checker.go    ← health check endpoints
  cmd/
    debug-server/
      main.go              ← debugging HTTP server
  tools/
    circuit-analyzer/      ← circuit breaker analysis tools
      main.go
  tests/
    integration/
      debug_test.go        ← debugging integration tests
    chaos/
      failure_scenarios_test.go ← chaos testing scenarios
```

**Infrastructure Starter Code**

```go
// Package debug provides comprehensive debugging utilities for circuit breaker systems.
// This package includes state inspection, performance profiling, and testing utilities.
package debug

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "net/http/httptest"
    "sync"
    "time"
)

// CircuitInspector provides comprehensive state inspection for circuit breaker debugging.
// It offers real-time visibility into circuit breaker state, metrics, and configuration.
type CircuitInspector struct {
    registry *ServiceRegistry
    collector *MetricsCollector
    logger    Logger
    mutex     sync.RWMutex
}

// NewCircuitInspector creates a new inspector with registry and collector dependencies.
func NewCircuitInspector(registry *ServiceRegistry, collector *MetricsCollector, logger Logger) *CircuitInspector {
    return &CircuitInspector{
        registry:  registry,
        collector: collector,
        logger:    logger,
    }
}

// InspectCircuit returns comprehensive state information for a specific service.
func (ci *CircuitInspector) InspectCircuit(serviceID string) (*CircuitInspection, error) {
    ci.mutex.RLock()
    defer ci.mutex.RUnlock()
    
    circuit, exists := ci.registry.GetCircuit(serviceID)
    if !exists {
        return nil, fmt.Errorf("circuit not found for service: %s", serviceID)
    }
    
    metrics := ci.collector.GetMetrics(serviceID)
    config := ci.registry.GetConfig(serviceID)
    
    return &CircuitInspection{
        ServiceID:           serviceID,
        CurrentState:        circuit.State(),
        StateTransitionTime: circuit.stateTransitionTime,
        FailureCount:        circuit.failureCount,
        SuccessCount:        circuit.successCount,
        Configuration:       config,
        Metrics:            metrics,
        Timestamp:          time.Now(),
    }, nil
}

// MockClock provides controllable time for deterministic circuit breaker testing.
// It replaces system time with predictable, controllable time advancement.
type MockClock struct {
    now    time.Time
    timers []*mockTimer
    mu     sync.RWMutex
}

// NewMockClock creates a mock clock starting at the specified time.
func NewMockClock(startTime time.Time) *MockClock {
    return &MockClock{
        now:    startTime,
        timers: make([]*mockTimer, 0),
    }
}

// Now returns the current mock time.
func (mc *MockClock) Now() time.Time {
    mc.mu.RLock()
    defer mc.mu.RUnlock()
    return mc.now
}

// After creates a timer channel that triggers after the specified duration.
func (mc *MockClock) After(duration time.Duration) <-chan time.Time {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    timer := &mockTimer{
        expiresAt: mc.now.Add(duration),
        ch:        make(chan time.Time, 1),
        triggered: false,
    }
    
    mc.timers = append(mc.timers, timer)
    return timer.ch
}

// AdvanceTime moves the mock clock forward and triggers expired timers.
func (mc *MockClock) AdvanceTime(duration time.Duration) {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    mc.now = mc.now.Add(duration)
    
    for _, timer := range mc.timers {
        if !timer.triggered && timer.expiresAt.Before(mc.now) || timer.expiresAt.Equal(mc.now) {
            timer.triggered = true
            select {
            case timer.ch <- mc.now:
            default:
            }
        }
    }
}

// MockServer provides configurable HTTP server for testing circuit breaker behavior.
// It supports failure injection, latency simulation, and response customization.
type MockServer struct {
    server      *httptest.Server
    failureRate float64
    latency     time.Duration
    responses   map[string]*http.Response
    mu          sync.RWMutex
}

// NewMockServer creates a new mock server with default successful responses.
func NewMockServer() *MockServer {
    ms := &MockServer{
        failureRate: 0.0,
        latency:     0,
        responses:   make(map[string]*http.Response),
    }
    
    ms.server = httptest.NewServer(http.HandlerFunc(ms.handleRequest))
    return ms
}

// SetFailureRate configures the percentage of requests that should fail (0.0 to 1.0).
func (ms *MockServer) SetFailureRate(rate float64) {
    ms.mu.Lock()
    defer ms.mu.Unlock()
    ms.failureRate = rate
}

// SetLatency configures artificial response delay for all requests.
func (ms *MockServer) SetLatency(duration time.Duration) {
    ms.mu.Lock()
    defer ms.mu.Unlock()
    ms.latency = duration
}

// URL returns the base URL of the mock server.
func (ms *MockServer) URL() string {
    return ms.server.URL
}

// Close shuts down the mock server.
func (ms *MockServer) Close() {
    ms.server.Close()
}

// PerformanceProfiler provides systematic performance analysis for circuit breaker operations.
type PerformanceProfiler struct {
    enabled    bool
    collectors map[string]*performanceCollector
    mutex      sync.RWMutex
}

// NewPerformanceProfiler creates a profiler with specified collection intervals.
func NewPerformanceProfiler() *PerformanceProfiler {
    return &PerformanceProfiler{
        enabled:    false,
        collectors: make(map[string]*performanceCollector),
    }
}

// StartProfiling begins performance data collection for the specified operation.
func (pp *PerformanceProfiler) StartProfiling(operationName string) {
    pp.mutex.Lock()
    defer pp.mutex.Unlock()
    
    if !pp.enabled {
        return
    }
    
    collector := &performanceCollector{
        operationName: operationName,
        startTime:     time.Now(),
        samples:       make([]performanceSample, 0, 1000),
    }
    
    pp.collectors[operationName] = collector
}

// ChaosInjector provides controlled failure injection for testing circuit breaker resilience.
type ChaosInjector struct {
    scenarios      map[string]*ChaosScenario
    activeFailures map[string]*InjectedFailure
    safetyLimits   *SafetyLimits
    mutex          sync.RWMutex
}

// NewChaosInjector creates a chaos injector with safety limits and monitoring.
func NewChaosInjector(safetyLimits *SafetyLimits) *ChaosInjector {
    return &ChaosInjector{
        scenarios:      make(map[string]*ChaosScenario),
        activeFailures: make(map[string]*InjectedFailure),
        safetyLimits:   safetyLimits,
    }
}
```

**Core Logic Skeleton Code**

```go
// DebuggingMiddleware provides comprehensive debugging capabilities for circuit breaker operations.
// It logs decisions, collects metrics, and provides inspection interfaces.
type DebuggingMiddleware struct {
    inspector *CircuitInspector
    profiler  *PerformanceProfiler
    logger    Logger
}

// WrapCircuitBreaker adds debugging capabilities to an existing circuit breaker.
func (dm *DebuggingMiddleware) WrapCircuitBreaker(cb *CircuitBreaker) *DebuggingCircuitBreaker {
    // TODO 1: Create debugging wrapper that implements CircuitBreaker interface
    // TODO 2: Add performance timing around Execute method calls
    // TODO 3: Log all state transitions with full context
    // TODO 4: Collect detailed metrics for each operation
    // TODO 5: Provide inspection methods for debugging
    // Hint: Use decorator pattern to wrap existing circuit breaker
}

// DiagnoseCircuitIssues performs systematic analysis of circuit breaker problems.
func (dm *DebuggingMiddleware) DiagnoseCircuitIssues(serviceID string) *DiagnosisReport {
    // TODO 1: Inspect current circuit breaker state and configuration
    // TODO 2: Analyze recent metrics and failure patterns
    // TODO 3: Check for common configuration issues
    // TODO 4: Validate error classification behavior
    // TODO 5: Generate diagnosis report with recommended fixes
    // Hint: Use symptom-cause-fix mapping table from debugging guide
}

// AnalyzePerformanceBottlenecks identifies performance issues in circuit breaker operations.
func (pp *PerformanceProfiler) AnalyzePerformanceBottlenecks() *PerformanceAnalysis {
    // TODO 1: Collect CPU and memory profiling data
    // TODO 2: Analyze lock contention and goroutine blocking
    // TODO 3: Measure sliding window calculation overhead
    // TODO 4: Profile metrics collection performance impact
    // TODO 5: Generate performance optimization recommendations
    // Hint: Use go tool pprof integration for detailed analysis
}

// ExecuteSystematicTests runs comprehensive debugging test scenarios.
func (ci *ChaosInjector) ExecuteSystematicTests(ctx context.Context, testSuite string) error {
    // TODO 1: Load test scenario configuration from test suite
    // TODO 2: Initialize mock services with configurable failure modes
    // TODO 3: Execute controlled failure injection scenarios
    // TODO 4: Monitor circuit breaker behavior during failures
    // TODO 5: Validate recovery behavior and state transitions
    // Hint: Use safety limits to prevent uncontrolled failure propagation
}
```

**Language-Specific Debugging Hints**

For effective Go circuit breaker debugging:

- Use `go build -race` to detect race conditions in concurrent circuit breaker operations
- Enable mutex profiling with `runtime.SetMutexProfileFraction()` to identify lock contention
- Use `go tool pprof` with HTTP endpoints to analyze CPU and memory usage patterns
- Implement structured logging with `context.Context` propagation for request tracing
- Use `sync/atomic` for high-frequency counters to reduce lock contention
- Enable garbage collection tracing with `GODEBUG=gctrace=1` to identify memory pressure
- Use `testing.Short()` to skip long-running chaos tests during development
- Implement custom `fmt.Stringer` interfaces for readable debugging output

**Milestone Checkpoints**

After implementing debugging capabilities for each milestone:

**Milestone 1 Checkpoint:**
- Run `go test -race ./internal/debug/...` to verify thread-safe debugging utilities
- Start debug server with `go run ./cmd/debug-server/` and verify state inspection endpoints
- Execute `curl http://localhost:8080/debug/circuits` to view all circuit breaker states
- Verify mock clock functionality by running time-dependent test scenarios
- Check that state transition logging captures all circuit breaker state changes

**Milestone 2 Checkpoint:**
- Profile sliding window performance under high request volumes using `go tool pprof`
- Execute chaos scenarios with controlled failure injection and verify circuit behavior
- Verify metrics collection accuracy by comparing collected metrics with expected values
- Test fallback execution logging by triggering circuit open conditions
- Validate bulkhead performance by measuring semaphore acquisition times under load

**Milestone 3 Checkpoint:**
- Execute end-to-end debugging scenarios with HTTP and gRPC client integration
- Verify distributed tracing integration captures circuit breaker decision spans
- Test alerting integration by triggering circuit state changes and monitoring notifications
- Validate chaos testing framework safety limits prevent uncontrolled failure propagation
- Confirm debugging dashboards accurately reflect real-time circuit breaker state

**Debugging Tips Table**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Circuit breaker state transitions not logged | Missing logging in transition methods | Check `transitionToOpen()`, `transitionToClosed()` methods | Add structured logging with service ID, timestamp, and metrics context |
| Performance degradation under high load | Lock contention in hot paths | Profile with `go tool pprof -http :8080 http://localhost:6060/debug/pprof/mutex` | Implement lock-free counters or reduce critical section duration |
| Metrics inconsistency across instances | Clock skew or race conditions | Compare timestamps and check for race conditions with `-race` flag | Synchronize clocks and add proper synchronization primitives |
| Chaos tests causing system instability | Safety limits not enforced | Monitor resource usage and failure propagation during tests | Implement circuit breakers for chaos injection and add safety timeouts |
| Memory leaks in long-running tests | Sliding window buckets not cleaned up | Profile memory with `go tool pprof heap` and check bucket retention | Implement bucket cleanup and add periodic garbage collection |


## Future Extensions

> **Milestone(s):** Beyond Milestone 3 (Integration & Testing) - Advanced enhancements for production-scale deployments

This section explores advanced enhancements that extend the circuit breaker pattern beyond its basic implementation. These extensions address sophisticated production scenarios where static configurations and isolated circuit breakers may not provide optimal resilience. The enhancements focus on three key areas: adaptive behavior based on learned patterns, coordination across distributed instances, and predictive failure detection through advanced analytics.

The extensions represent evolutionary improvements that can be incrementally added to the basic circuit breaker implementation without disrupting existing functionality. Each enhancement addresses specific limitations discovered in real-world deployments and provides measurably better system resilience under varying conditions.

### Adaptive Threshold Adjustment

#### Mental Model: Smart Thermostat Learning

Think of adaptive threshold adjustment like a smart thermostat that learns your daily patterns and adjusts temperature settings automatically. Initially, you set basic rules like "68°F during the day, 65°F at night." Over time, the smart thermostat observes that you're always cold on Mondays (when you work from home), that sunny afternoons make the house warmer, and that your schedule changes seasonally. It adapts its thresholds based on these learned patterns, providing better comfort with less manual intervention.

Similarly, an adaptive circuit breaker learns service behavior patterns over time. Instead of using fixed failure thresholds like "open after 10 consecutive failures," it adjusts thresholds based on observed service characteristics: time-of-day patterns, seasonal load variations, dependency chain effects, and historical recovery times. This creates a more intelligent circuit breaker that provides optimal protection under varying conditions.

#### Service Behavior Pattern Learning

The adaptive threshold system continuously analyzes service behavior to identify patterns that influence optimal threshold settings. The learning system maintains historical data about service performance characteristics, failure patterns, and recovery behaviors across different operational contexts.

Service behavior analysis focuses on several key dimensions that affect threshold optimization. **Temporal patterns** include time-of-day effects where services may be more fragile during peak hours or more resilient during maintenance windows. **Load correlation patterns** examine how failure rates change with request volume, identifying services that become unstable under high load versus those that maintain consistent reliability. **Dependency impact patterns** analyze how failures in downstream services affect upstream circuit breaker effectiveness, potentially requiring different thresholds for different call chains.

The learning system tracks **failure clustering characteristics** to understand whether failures tend to occur in bursts (requiring lower thresholds for faster protection) or as isolated events (allowing higher thresholds to reduce false positives). **Recovery time patterns** help optimize the recovery timeout duration based on observed service restart times, database connection pool recovery, cache warming periods, and other service-specific recovery characteristics.

**Environmental correlation analysis** identifies external factors that influence service reliability, such as deployment events, database maintenance windows, traffic pattern changes, and infrastructure scaling events. This contextual awareness allows the adaptive system to temporarily adjust thresholds during known periods of increased fragility.

> **Decision: Machine Learning Algorithm Selection**
> - **Context**: Need to predict optimal thresholds based on historical service behavior patterns with limited training data and real-time constraints
> - **Options Considered**: Deep neural networks for complex pattern recognition, time series forecasting models for temporal prediction, ensemble methods combining multiple approaches
> - **Decision**: Exponential weighted moving averages with contextual adjustment factors
> - **Rationale**: Provides good adaptation to changing patterns while remaining computationally efficient and interpretable, requires minimal training data, and allows real-time threshold updates
> - **Consequences**: May miss complex non-linear patterns that deep learning could capture, but provides reliable baseline adaptation with clear operational behavior

| Algorithm Option | Training Data Requirements | Computational Overhead | Interpretability | Real-time Performance |
|------------------|---------------------------|------------------------|------------------|---------------------|
| Deep Neural Networks | High (thousands of samples) | High (GPU recommended) | Low (black box) | Medium (inference overhead) |
| Time Series Models | Medium (hundreds of samples) | Medium (statistical computation) | Medium (model parameters) | High (lightweight prediction) |
| Exponential Weighted Averages | Low (dozens of samples) | Low (simple arithmetic) | High (transparent math) | Very High (minimal overhead) |

#### Threshold Optimization Engine

The threshold optimization engine processes learned behavior patterns to calculate optimal circuit breaker parameters for each service. The engine balances multiple competing objectives: minimizing false positives that block healthy requests, minimizing false negatives that allow requests to failing services, and optimizing recovery speed to restore service availability quickly.

The optimization process operates on multiple threshold parameters simultaneously. **Failure count thresholds** determine how many failures trigger state transitions, with optimization considering the trade-off between sensitivity to problems and tolerance for transient failures. **Failure rate thresholds** set percentage-based triggers that adapt to varying request volumes, requiring analysis of typical request patterns and failure distributions. **Recovery timeout durations** balance giving services adequate time to recover against keeping circuits open unnecessarily long.

**Time window adjustments** optimize the sliding window size used for failure rate calculations. Services with rapid recovery patterns benefit from shorter windows that quickly recognize improvement, while services with slow, gradual degradation require longer windows to avoid premature circuit opening. The optimization engine continuously evaluates window effectiveness by measuring how well different window sizes predict actual service health transitions.

The engine implements **multi-objective optimization** using weighted scoring functions that consider operational priorities. Production services handling critical business functions receive optimization weights that favor availability over latency, while internal services may prioritize fast failure detection over avoiding service disruption. **Contextual threshold sets** allow different thresholds during different operational periods, such as stricter thresholds during peak business hours and more relaxed thresholds during maintenance windows.

**Confidence-based adjustment** ensures that threshold changes only occur when the learning system has sufficient evidence to support the modification. New services start with conservative default thresholds and gradually adapt as behavioral data accumulates. Services with inconsistent patterns maintain broader threshold ranges to accommodate variability, while services with predictable patterns can use more precise thresholds for optimal protection.

#### Implementation Strategy

The adaptive threshold implementation requires careful integration with the existing circuit breaker system to provide learning capabilities without compromising performance or reliability. The implementation follows a gradual rollout strategy that maintains backward compatibility while introducing adaptive capabilities incrementally.

**Data collection infrastructure** captures detailed service interaction metrics beyond basic success/failure counts. This includes request latency distributions, error type classifications, temporal context (time of day, day of week), load characteristics (requests per second, concurrent connections), and environmental context (deployment versions, infrastructure changes). The collection system maintains data retention policies that balance learning effectiveness with storage costs.

**Learning pipeline architecture** processes collected metrics to identify behavior patterns and generate threshold recommendations. The pipeline operates asynchronously to avoid impacting request processing performance, using batch processing for historical pattern analysis and stream processing for real-time adaptation signals. **Pattern detection algorithms** identify significant changes in service behavior that warrant threshold adjustments, while **noise filtering** prevents adaptation based on temporary anomalies or insufficient data.

**Threshold update mechanisms** provide safe, gradual changes to circuit breaker parameters. Updates use **staged rollout** approaches where new thresholds are first tested with a small percentage of traffic while monitoring for adverse effects. **Rollback capabilities** automatically revert to previous thresholds if adaptation causes increased failures or degraded service availability. **Override controls** allow operators to disable adaptation for critical services or during operational events where predictable behavior is essential.

**Safety boundaries** prevent adaptive adjustments from creating unsafe configurations. **Minimum and maximum threshold limits** ensure that adaptation cannot create circuits that never open (no protection) or that open too aggressively (service unavailability). **Rate limiting** controls how frequently thresholds can change to prevent oscillation or rapid successive adjustments. **Human approval workflows** require operator confirmation for significant threshold changes that exceed predefined safety bounds.

### Distributed Circuit Coordination

#### Mental Model: Air Traffic Control System

Think of distributed circuit coordination like an air traffic control system managing flights across multiple airports. Individual airport towers (circuit breaker instances) monitor local conditions and make immediate safety decisions for their airspace. However, when weather systems, equipment failures, or traffic congestion affect multiple airports, regional coordination centers share information and coordinate responses across the entire network.

Similarly, individual circuit breaker instances monitor their local service interactions and make immediate protection decisions. But when cascading failures, infrastructure problems, or traffic spikes affect multiple service instances, distributed coordination allows circuit breakers to share state information and coordinate responses across the entire system. This prevents situations where some circuit breakers remain closed while others have opened, leading to uneven load distribution and potential secondary failures.

#### State Synchronization Mechanisms

Distributed circuit coordination requires robust state synchronization mechanisms that balance consistency with performance and fault tolerance. The synchronization system must handle network partitions, clock drift, and varying instance load while maintaining circuit breaker responsiveness.

**Event-driven state sharing** propagates circuit state changes across distributed instances without requiring continuous synchronization overhead. When a circuit breaker transitions to the open state, it broadcasts a state change event containing the service identifier, new state, transition timestamp, and contextual information about the triggering failures. Other instances receive these events and update their local circuit state accordingly, ensuring coordinated protection across the distributed system.

The synchronization mechanism implements **eventual consistency** rather than strong consistency to maintain circuit breaker performance characteristics. Strong consistency would require distributed consensus for every state transition, introducing latency and availability risks that contradict circuit breaker design principles. Eventual consistency allows immediate local decisions while propagating state changes asynchronously, accepting temporary inconsistency in favor of system resilience.

**Conflict resolution strategies** handle situations where multiple circuit breaker instances detect failures simultaneously or where network partitions create divergent state. The system uses **timestamp-based ordering** with logical clocks to determine the authoritative state sequence, ensuring that all instances eventually converge to the same state regardless of event delivery order. **Service health consensus** algorithms aggregate failure signals from multiple instances to distinguish between localized problems and system-wide service degradation.

**State persistence and recovery** mechanisms ensure that coordinated state survives instance restarts and network partitions. Circuit breakers persist their state changes to durable storage and include coordination metadata such as distributed state version numbers and peer instance identifiers. During startup or partition recovery, instances synchronize their state with peers to restore coordination consistency.

> **Decision: Coordination Architecture Pattern**
> - **Context**: Need to coordinate circuit breaker state across distributed instances without introducing single points of failure or excessive latency
> - **Options Considered**: Centralized coordination service, peer-to-peer gossip protocol, event streaming with consensus
> - **Decision**: Hybrid approach with event streaming backbone and peer-to-peer backup
> - **Rationale**: Event streaming provides efficient distribution during normal operation while peer-to-peer gossip ensures coordination during infrastructure failures
> - **Consequences**: Requires dual coordination paths and complexity in conflict resolution, but provides high availability and performance

| Architecture Option | Latency | Availability | Complexity | Infrastructure Dependencies |
|---------------------|---------|--------------|------------|---------------------------|
| Centralized Service | Low | Medium (SPOF) | Low | High (coordination service) |
| Peer-to-Peer Gossip | Medium | High | Medium | Low (network only) |
| Event Streaming | Low | High | Medium | Medium (message broker) |
| Hybrid Approach | Low-Medium | Very High | High | Medium (with fallback) |

#### Global Circuit State Management

Global circuit state management provides a system-wide view of circuit breaker status while maintaining the decentralized decision-making that makes circuit breakers effective. The management system aggregates distributed state information, detects system-wide patterns, and coordinates responses to widespread failures.

**Hierarchical state aggregation** organizes circuit breaker state information across multiple levels: individual circuit instances, service clusters, data centers, and geographical regions. Each aggregation level maintains summary statistics about circuit health, failure rates, and traffic patterns within its scope. This hierarchical organization enables both fine-grained local decisions and coarse-grained global policies without overwhelming individual instances with global state information.

**Global failure pattern detection** identifies system-wide issues that require coordinated responses beyond individual circuit breaker decisions. **Cascade failure detection** recognizes when failures propagate through service dependency chains, triggering coordinated circuit opening to prevent further cascade propagation. **Infrastructure failure correlation** identifies when multiple circuit breakers open simultaneously due to shared infrastructure problems, such as database failures, network partitions, or load balancer issues.

The global state management system implements **distributed circuit policies** that coordinate behavior across multiple instances. **Coordinated recovery testing** ensures that half-open state transitions are staggered across instances to prevent thundering herd effects when services recover. **Load balancing integration** shares circuit state information with load balancers to route traffic away from instances with open circuits, improving overall system throughput and reducing user-visible failures.

**System-wide health scoring** provides aggregate metrics that combine local circuit state with global system context. The health score considers not just individual service reliability, but also dependency chain stability, infrastructure capacity, and traffic distribution patterns. This comprehensive health assessment enables more informed circuit breaker decisions and helps prevent localized failures from causing unnecessary global degradation.

#### Consensus and Conflict Resolution

Distributed circuit breaker coordination requires robust consensus and conflict resolution mechanisms that handle network partitions, clock drift, and varying instance perspectives on service health. The consensus system must maintain circuit breaker performance characteristics while ensuring eventual consistency across all instances.

**Weighted voting consensus** allows circuit breaker instances to participate in distributed state decisions with voting power based on their traffic volume, operational health, and historical accuracy. Instances processing higher request volumes receive greater influence in consensus decisions since they have more comprehensive information about service health. **Geographic and availability zone weighting** ensures that consensus decisions consider infrastructure topology, preventing situations where circuit breakers in a single location dominate decisions affecting global service availability.

The conflict resolution system handles **split-brain scenarios** where network partitions create isolated groups of circuit breaker instances with divergent state. **Quorum-based decisions** require a minimum number of instances to agree on state changes, preventing small groups from making system-wide decisions during partitions. **Partition tolerance strategies** allow isolated instances to continue making local protection decisions while maintaining conservative thresholds until coordination is restored.

**Temporal conflict resolution** addresses situations where clock drift or network delays create ambiguous event ordering. The system implements **vector clocks** to track causal relationships between state changes across instances, ensuring that conflict resolution preserves logical ordering rather than just timestamp ordering. **Event sequence reconciliation** algorithms merge divergent state histories after partition recovery, applying conflict resolution rules to create a consistent global state.

**Authority delegation mechanisms** provide fallback coordination when full consensus cannot be achieved. **Regional coordination leaders** can make authoritative decisions for their geographic area during global coordination failures, while **load-based authority** allows the highest-traffic instances to make emergency coordination decisions when rapid response is required. These delegation mechanisms include automatic reversion to full consensus when normal coordination is restored.

### Advanced Metrics and Analytics

#### Mental Model: Modern Medical Diagnostics

Think of advanced circuit breaker metrics and analytics like modern medical diagnostics that go far beyond basic vital signs. Traditional medical checkups measure obvious indicators like blood pressure, temperature, and heart rate. Advanced diagnostics use continuous monitoring, pattern recognition, and predictive modeling to identify health issues before they become critical problems.

Similarly, basic circuit breaker metrics track obvious indicators like request counts, failure rates, and current state. Advanced analytics implement continuous monitoring of subtle patterns, predictive modeling of service degradation, and early warning systems that identify potential failures before they trigger circuit breaker responses. This shift from reactive to proactive monitoring dramatically improves system resilience by preventing problems rather than just responding to them.

#### Predictive Failure Detection

Predictive failure detection analyzes patterns in service behavior, infrastructure metrics, and external signals to identify potential failures before they manifest as request failures. This proactive approach allows circuit breakers to take preventive action or adjust sensitivity before service degradation affects user experience.

**Leading indicator identification** focuses on metrics that change before service failures occur. **Resource utilization trends** such as gradually increasing memory usage, CPU saturation, or database connection pool exhaustion often precede service failures by minutes or hours. **Latency distribution shifts** where the tail latencies (95th, 99th percentiles) increase significantly while median latency remains stable indicate emerging performance problems that may lead to timeouts and failures.

**Dependency health correlation** analyzes the health of services throughout the dependency chain to predict upstream impact. When database query performance degrades, dependent services may continue functioning normally for several minutes until connection pools exhaust or caches expire. **Error rate acceleration** detection identifies when failure rates are increasing exponentially rather than remaining stable, indicating that the service is entering an unstable state that will likely worsen.

The prediction system implements **anomaly detection algorithms** that identify unusual patterns in service behavior without requiring explicit failure thresholds. **Statistical process control** methods track metrics using control charts that identify when service behavior moves outside normal operational bounds, even if it hasn't reached failure thresholds. **Machine learning-based anomaly detection** models learn normal service behavior patterns and flag deviations that correlate with historical failure periods.

**Time-series forecasting** predicts future service health based on current trends and historical patterns. **Resource exhaustion prediction** uses current consumption rates to estimate when critical resources will reach capacity, allowing proactive circuit breaker sensitivity adjustments. **Seasonal pattern recognition** identifies recurring patterns in service reliability, such as increased failure rates during daily peak traffic or reduced reliability during weekly maintenance windows.

> **Decision: Predictive Algorithm Strategy**
> - **Context**: Need to balance prediction accuracy with computational overhead and false positive rates in production monitoring systems
> - **Options Considered**: Complex ensemble models with multiple algorithms, simple statistical methods with domain knowledge, hybrid approach with tiered complexity
> - **Decision**: Tiered prediction system with statistical methods for real-time prediction and machine learning for deeper analysis
> - **Rationale**: Provides immediate predictive capability with low overhead while enabling sophisticated analysis for capacity planning and optimization
> - **Consequences**: Requires maintaining multiple prediction pipelines and complexity in combining results, but provides both immediate protection and strategic insights

| Prediction Approach | Latency | Accuracy | Computational Cost | False Positive Rate |
|---------------------|---------|----------|-------------------|-------------------|
| Statistical Process Control | Very Low | Medium | Very Low | Medium |
| Time Series Forecasting | Low | High | Low | Low |
| Machine Learning Ensemble | High | Very High | High | Very Low |
| Hybrid Tiered System | Low-Medium | High | Medium | Low |

#### Real-time Service Health Scoring

Real-time service health scoring provides comprehensive, continuously updated assessments of service health that incorporate multiple dimensions of service behavior beyond simple failure rates. The scoring system enables more nuanced circuit breaker decisions and provides operational teams with detailed service health insights.

**Multi-dimensional health metrics** combine various service health indicators into comprehensive health scores. **Performance health** considers request latency distributions, throughput capacity, and response time consistency to assess whether the service is meeting performance expectations. **Reliability health** analyzes failure rates, error type distributions, and failure clustering patterns to evaluate service stability. **Capacity health** examines resource utilization, queue depths, and connection pool usage to assess whether the service is operating within sustainable capacity limits.

The health scoring system implements **weighted composite scoring** that reflects the relative importance of different health dimensions for each service. Critical business services may weight reliability more heavily than performance, while real-time services may prioritize performance consistency. **Contextual weight adjustment** modifies scoring weights based on operational context, such as increasing reliability weights during peak business hours or adjusting performance weights during maintenance windows.

**Dynamic baseline calculation** ensures that health scores reflect relative service health rather than absolute metrics, accommodating services with different normal operating characteristics. **Adaptive threshold learning** adjusts what constitutes "healthy" behavior based on observed service patterns, preventing false alarms for services with naturally variable performance while maintaining sensitivity to actual degradation.

The scoring system provides **health trend analysis** that identifies whether service health is improving, degrading, or stable over time. **Health velocity scoring** measures how rapidly health is changing, identifying services in rapid decline that require immediate attention versus those with gradual degradation that allow time for preventive action. **Comparative health ranking** provides relative health assessments across similar services, helping operations teams prioritize attention and resources.

#### Operational Intelligence Dashboard

The operational intelligence dashboard provides comprehensive visibility into circuit breaker system behavior, service health patterns, and system-wide resilience metrics. The dashboard combines real-time monitoring with analytical insights to support both immediate operational decisions and strategic system improvements.

**System-wide resilience overview** provides executive-level visibility into overall system health and circuit breaker effectiveness. **Cascade failure prevention metrics** track how circuit breakers have prevented failure propagation, measuring both the number of prevented failures and the estimated impact reduction. **Mean time to recovery** analytics show how circuit breaker automation has improved system recovery speed compared to manual intervention. **Availability impact analysis** quantifies the trade-offs between circuit breaker protection and service availability, helping optimize threshold configurations.

**Service-specific deep dive capabilities** allow operations teams to investigate individual service behavior and circuit breaker performance. **Circuit breaker decision audit trails** provide complete histories of state transitions, threshold adjustments, and recovery events with contextual information about triggering conditions. **Failure pattern analysis** visualizes failure distributions, clustering patterns, and correlation with external events to support root cause analysis and prevention strategies.

The dashboard implements **predictive alerting** that combines multiple signals to provide early warning of potential issues. **Degradation trajectory visualization** shows current service health trends and projected future health based on predictive models. **Risk assessment displays** highlight services with elevated failure probability based on dependency health, resource utilization, and historical patterns. **Recommended action guidance** suggests specific operational interventions based on current system state and predicted future conditions.

**Interactive analysis tools** enable operations teams to explore service behavior patterns and test hypothetical scenarios. **Threshold simulation capabilities** allow testing different circuit breaker configurations against historical traffic patterns to optimize settings. **Dependency impact modeling** visualizes how failures in one service would propagate through the system given current circuit breaker configurations. **Capacity planning integration** connects circuit breaker metrics with infrastructure scaling decisions to improve overall system resilience.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Machine Learning Pipeline | Exponential weighted moving averages with Go math libraries | TensorFlow Serving with Go client for complex models |
| Distributed Coordination | Redis pub/sub with Go redis client | Apache Kafka with Confluent Go client for event streaming |
| Time Series Storage | In-memory ring buffers with periodic snapshots | InfluxDB with Go client for comprehensive historical data |
| Predictive Analytics | Statistical process control with native Go | Prometheus with custom metrics for advanced monitoring |
| Real-time Dashboard | HTTP API with JSON responses | WebSocket streaming with real-time chart libraries |

#### Recommended File Structure

```
project-root/
  cmd/
    adaptive-service/main.go          ← adaptive threshold service
    coordinator/main.go               ← distributed coordination service
    analytics-dashboard/main.go       ← dashboard and analytics service
  internal/
    circuitbreaker/                   ← existing circuit breaker core
    adaptive/
      learner.go                      ← behavior pattern learning
      optimizer.go                    ← threshold optimization engine
      predictor.go                    ← predictive failure detection
      adaptive_test.go                ← adaptive system tests
    coordination/
      state_sync.go                   ← distributed state synchronization
      consensus.go                    ← consensus and conflict resolution
      event_stream.go                 ← event streaming coordination
      coordination_test.go            ← coordination system tests
    analytics/
      health_scorer.go                ← real-time health scoring
      predictor.go                    ← predictive analytics
      dashboard_api.go                ← dashboard API endpoints
      analytics_test.go               ← analytics system tests
    monitoring/
      metrics_collector.go            ← enhanced metrics collection
      telemetry.go                    ← telemetry and observability
  configs/
    adaptive-config.yaml              ← adaptive system configuration
    coordination-config.yaml          ← coordination configuration
    analytics-config.yaml             ← analytics configuration
  deployments/
    adaptive-service.yaml             ← Kubernetes deployment specs
    coordination-service.yaml
    analytics-dashboard.yaml
```

#### Core Component Infrastructure

**Adaptive Learning Infrastructure:**

```go
package adaptive

import (
    "context"
    "sync"
    "time"
    "math"
)

// ServiceBehaviorPattern represents learned patterns about service behavior
type ServiceBehaviorPattern struct {
    ServiceID string
    TemporalPatterns map[string]float64  // hour-of-day, day-of-week patterns
    LoadCorrelations map[string]float64  // failure rate vs load correlations
    FailureClustering ClusteringMetrics  // how failures cluster together
    RecoveryCharacteristics RecoveryMetrics
    EnvironmentalFactors map[string]float64
    LastUpdated time.Time
    ConfidenceScore float64
}

// ThresholdOptimizer manages adaptive threshold adjustments
type ThresholdOptimizer struct {
    patterns map[string]*ServiceBehaviorPattern
    configs map[string]*AdaptiveConfig
    mutex sync.RWMutex
    updateChannel chan ThresholdUpdate
}

// Complete infrastructure for pattern learning and threshold optimization
// TODO: Implement behavior pattern collection from circuit breaker events
// TODO: Implement exponential weighted moving average calculations
// TODO: Implement confidence-based threshold adjustment algorithms
// TODO: Implement safety boundary checks for threshold changes
// TODO: Implement gradual rollout mechanisms for new thresholds
```

**Distributed Coordination Infrastructure:**

```go
package coordination

import (
    "context"
    "sync"
    "time"
)

// CoordinationEvent represents state changes distributed across instances
type CoordinationEvent struct {
    ServiceID string
    InstanceID string
    StateChange StateTransition
    Timestamp time.Time
    VectorClock map[string]int64
    ContextData map[string]interface{}
}

// DistributedCircuitCoordinator manages coordination across instances
type DistributedCircuitCoordinator struct {
    localInstanceID string
    peerInstances map[string]*PeerInstance
    eventStream EventStream
    stateManager *GlobalStateManager
    consensusEngine *ConsensusEngine
    mutex sync.RWMutex
}

// Complete infrastructure for distributed coordination
// TODO: Implement event-driven state synchronization
// TODO: Implement weighted voting consensus algorithms
// TODO: Implement partition tolerance and conflict resolution
// TODO: Implement hierarchical state aggregation
// TODO: Implement coordination safety boundaries and rollback
```

**Advanced Analytics Infrastructure:**

```go
package analytics

import (
    "context"
    "time"
    "sync"
)

// ServiceHealthScore represents comprehensive service health assessment
type ServiceHealthScore struct {
    ServiceID string
    OverallScore float64
    PerformanceHealth float64
    ReliabilityHealth float64
    CapacityHealth float64
    TrendDirection string  // "improving", "stable", "degrading"
    TrendVelocity float64
    PredictedHealth map[string]float64  // future health predictions
    CalculatedAt time.Time
}

// PredictiveAnalyzer implements predictive failure detection
type PredictiveAnalyzer struct {
    healthScorers map[string]*HealthScorer
    predictors map[string]*FailurePredictor
    anomalyDetectors map[string]*AnomalyDetector
    alertThresholds map[string]*PredictiveThresholds
    mutex sync.RWMutex
}

// Complete infrastructure for predictive analytics and health scoring
// TODO: Implement multi-dimensional health scoring algorithms
// TODO: Implement time-series forecasting for resource exhaustion
// TODO: Implement anomaly detection with statistical process control
// TODO: Implement leading indicator identification and correlation
// TODO: Implement predictive alerting with recommended actions
```

#### Core Logic Skeletons

**Adaptive Threshold Learning:**

```go
// LearnServiceBehavior analyzes service metrics to identify behavior patterns
// that influence optimal threshold settings
func (l *BehaviorLearner) LearnServiceBehavior(serviceID string, metrics []*ServiceMetrics) *ServiceBehaviorPattern {
    // TODO 1: Extract temporal patterns from metrics (hour-of-day, day-of-week effects)
    // TODO 2: Calculate load correlation coefficients (failure rate vs request volume)
    // TODO 3: Analyze failure clustering using time-series analysis
    // TODO 4: Characterize recovery patterns (time to recovery, success rates)
    // TODO 5: Identify environmental correlation factors (deployments, infrastructure)
    // TODO 6: Calculate pattern confidence scores based on data volume and consistency
    // TODO 7: Update existing patterns using exponential weighted averaging
    // TODO 8: Return updated behavior pattern with metadata
    // Hint: Use sliding time windows to calculate correlations
    // Hint: Apply statistical significance tests to avoid overfitting
}

// OptimizeThresholds calculates optimal circuit breaker parameters based on learned patterns
func (o *ThresholdOptimizer) OptimizeThresholds(serviceID string, pattern *ServiceBehaviorPattern) *OptimizedThresholds {
    // TODO 1: Load current thresholds and historical performance data
    // TODO 2: Calculate optimal failure count threshold using pattern clustering data
    // TODO 3: Optimize failure rate threshold based on temporal and load patterns
    // TODO 4: Adjust recovery timeout based on observed recovery characteristics
    // TODO 5: Optimize sliding window size based on service stability patterns
    // TODO 6: Apply multi-objective optimization balancing availability vs protection
    // TODO 7: Validate optimized thresholds against safety boundaries
    // TODO 8: Calculate confidence scores for threshold recommendations
    // Hint: Use weighted scoring functions for multi-objective optimization
    // Hint: Implement gradual adjustment rather than dramatic threshold changes
}
```

**Distributed State Coordination:**

```go
// SynchronizeCircuitState propagates local state changes to distributed instances
func (c *DistributedCircuitCoordinator) SynchronizeCircuitState(serviceID string, stateChange StateTransition) error {
    // TODO 1: Create coordination event with vector clock timestamp
    // TODO 2: Add local context data about the state change trigger
    // TODO 3: Broadcast event to all peer instances via event stream
    // TODO 4: Update local vector clock and peer tracking
    // TODO 5: Handle broadcast failures with retry and fallback mechanisms
    // TODO 6: Log coordination event for audit and debugging
    // TODO 7: Update global state aggregation with local change
    // TODO 8: Trigger any necessary cascade coordination actions
    // Hint: Use asynchronous broadcasting to avoid blocking local decisions
    // Hint: Implement exponential backoff for failed peer communications
}

// ResolveStateConflict handles conflicting state changes from multiple instances
func (c *ConsensusEngine) ResolveStateConflict(serviceID string, conflicts []*CoordinationEvent) StateTransition {
    // TODO 1: Sort conflicts by vector clock ordering to establish causality
    // TODO 2: Apply weighted voting based on instance traffic and health
    // TODO 3: Consider geographic and availability zone distribution
    // TODO 4: Handle partition scenarios with quorum-based decisions
    // TODO 5: Apply conflict resolution rules (latest state, most severe state)
    // TODO 6: Calculate consensus confidence score
    // TODO 7: Log resolution decision with full context for audit
    // TODO 8: Return authoritative state transition with metadata
    // Hint: Use logical timestamps to handle clock drift between instances
    // Hint: Prefer more restrictive states (open over closed) during conflicts
}
```

**Predictive Analytics Implementation:**

```go
// PredictServiceFailure analyzes current service metrics to predict potential failures
func (p *PredictiveAnalyzer) PredictServiceFailure(serviceID string, currentMetrics *ServiceMetrics) *FailurePrediction {
    // TODO 1: Extract leading indicators from current metrics
    // TODO 2: Compare current patterns against historical failure precursors
    // TODO 3: Analyze dependency chain health for upstream impact prediction
    // TODO 4: Apply anomaly detection algorithms to identify unusual patterns
    // TODO 5: Use time-series forecasting to predict resource exhaustion
    // TODO 6: Calculate failure probability based on multiple prediction models
    // TODO 7: Estimate time-to-failure if degradation continues
    // TODO 8: Generate recommended preventive actions based on prediction
    // Hint: Combine multiple prediction methods for higher confidence
    // Hint: Include uncertainty estimates in predictions for better decision making
}

// CalculateServiceHealthScore computes comprehensive real-time health assessment
func (h *HealthScorer) CalculateServiceHealthScore(serviceID string, metrics *ServiceMetrics) *ServiceHealthScore {
    // TODO 1: Calculate performance health score from latency and throughput
    // TODO 2: Calculate reliability health score from error rates and patterns
    // TODO 3: Calculate capacity health score from resource utilization
    // TODO 4: Apply service-specific weights to different health dimensions
    // TODO 5: Compute overall composite health score with confidence intervals
    // TODO 6: Analyze health trends and calculate trend velocity
    // TODO 7: Generate predicted future health based on current trends
    // TODO 8: Return comprehensive health assessment with metadata
    // Hint: Use percentile-based scoring to handle metric distribution variations
    // Hint: Implement adaptive baselines that learn normal service behavior
}
```

#### Milestone Checkpoints

**Adaptive Threshold System Checkpoint:**
After implementing adaptive threshold learning, verify functionality by running `go test ./internal/adaptive/...` and confirming that:
- Behavior pattern learning identifies temporal and load correlations from historical data
- Threshold optimization produces different recommendations for services with different patterns
- Safety boundaries prevent extreme threshold adjustments
- Gradual rollout mechanisms can be simulated with test data
- Confidence scoring correctly reflects data quality and pattern consistency

**Distributed Coordination Checkpoint:**
After implementing distributed coordination, verify functionality by:
- Starting multiple coordinator instances and confirming state synchronization
- Testing network partition scenarios with partition simulation tools
- Verifying consensus resolution during conflicting state changes
- Confirming event ordering preservation despite network delays
- Testing authority delegation during coordination service failures

**Advanced Analytics Checkpoint:**
After implementing predictive analytics, verify functionality by:
- Running prediction algorithms against historical failure data
- Confirming health score calculations reflect multiple service dimensions
- Testing anomaly detection with injected service degradation
- Verifying dashboard API responses contain comprehensive analytics data
- Confirming predictive alerts trigger appropriately for leading indicators

#### Language-Specific Implementation Notes

**Go-Specific Adaptive Implementation:**
- Use `sync.RWMutex` for concurrent access to learned behavior patterns
- Implement exponential weighted moving averages with `math.Exp()` for efficient pattern updates
- Use `time.Ticker` for periodic threshold optimization runs
- Leverage `context.Context` for cancellable long-running learning operations
- Use `encoding/gob` for efficient pattern serialization and persistence

**Go-Specific Coordination Implementation:**
- Use `sync.Map` for concurrent peer instance tracking
- Implement vector clocks with `map[string]int64` for logical timestamps
- Use buffered channels for asynchronous event broadcasting
- Leverage `net/http` with connection pooling for peer communication
- Use `context.WithTimeout` for peer communication with deadline handling

**Go-Specific Analytics Implementation:**
- Use `container/ring` for efficient circular buffer management in health scoring
- Implement time-series analysis with `gonum.org/v1/gonum` for statistical operations
- Use `encoding/json` with streaming for large analytics data responses
- Leverage `sync.Pool` for efficient metrics object reuse
- Use `time.Duration` for consistent temporal calculations across analytics


## Glossary

> **Milestone(s):** Milestone 1 (Basic Circuit Breaker), Milestone 2 (Advanced Features), Milestone 3 (Integration & Testing)

This glossary provides comprehensive definitions of key technical terms, patterns, and concepts used throughout the circuit breaker system. Each entry includes both conceptual understanding and practical implementation context to help developers navigate the complex landscape of resilient microservices communication.

### Core Circuit Breaker Terms

| Term | Definition | Context |
|------|------------|---------|
| **circuit breaker** | A component that monitors service health and blocks requests to failing services, preventing cascade failures by failing fast instead of waiting for timeouts | The primary pattern implemented in this system, analogous to electrical circuit breakers that protect electrical systems from overload |
| **cascade failure** | A failure mode where one service failure causes upstream service failures, propagating through the system like dominoes | The fundamental problem that circuit breakers solve by isolating failing services |
| **failing fast** | The strategy of rejecting requests immediately instead of waiting for timeout when a service is known to be unhealthy | Reduces latency and resource consumption during service failures |
| **graceful degradation** | Providing simplified functionality when primary service fails, maintaining partial system operation | Implemented through fallback patterns and alternative response strategies |
| **state machine** | Three-state system (closed, open, half-open) managing circuit behavior based on service health | Core component that determines whether requests are allowed through or blocked |
| **failure threshold** | Number of consecutive failures or failure rate percentage before opening circuit | Configurable parameter that controls sensitivity to service health changes |
| **recovery testing** | Limited request testing in half-open state to probe service recovery | Mechanism for transitioning back to normal operation after service restoration |
| **observability** | Metrics and monitoring capabilities for operational visibility into circuit breaker behavior | Essential for debugging, performance tuning, and operational awareness |

### State Management Terms

| Term | Definition | Context |
|------|------------|---------|
| **StateClosed** | Circuit state allowing normal request flow through to downstream service | Default healthy state where all requests pass through normally |
| **StateOpen** | Circuit state blocking all requests immediately without calling downstream service | Protective state activated when failure threshold is exceeded |
| **StateHalfOpen** | Circuit state allowing limited test requests through to probe recovery | Transition state for testing service recovery before full restoration |
| **state transition** | Process of changing from one circuit state to another based on success/failure events | Governed by configured thresholds and timeout parameters |
| **consecutive failures** | Uninterrupted sequence of failed requests used to trigger state transitions | Key metric for determining when to open circuit breaker |
| **recovery timeout** | Duration the circuit remains open before transitioning to half-open for recovery testing | Prevents immediate retries and allows failing service time to recover |
| **thread safety** | Safe concurrent access to circuit breaker state across multiple goroutines | Critical for correct operation in concurrent microservices environments |

### Metrics and Monitoring Terms

| Term | Definition | Context |
|------|------------|---------|
| **sliding window** | Time-based failure rate calculation using rolling time buckets for accurate health assessment | Advanced alternative to simple consecutive failure counting |
| **bucket rotation** | Process of creating new time buckets and discarding old ones as time advances | Maintains sliding window by managing circular buffer of time periods |
| **ring buffer** | Circular array data structure for efficient bucket storage with fixed memory usage | Optimized storage mechanism for time-series metrics data |
| **failure rate** | Percentage of failed requests within a time window used for circuit decisions | More sophisticated health metric than simple failure counting |
| **exponential moving average** | Weighted average giving more weight to recent samples for trend analysis | Statistical technique for smoothing metrics over time |
| **time alignment** | Rounding timestamps to bucket boundaries for consistent time window management | Ensures accurate bucket boundaries regardless of request timing |
| **metrics snapshot** | Point-in-time copy of circuit metrics for consistent reporting | Provides atomic view of circuit state for monitoring systems |

### Advanced Pattern Terms

| Term | Definition | Context |
|------|------------|---------|
| **fallback patterns** | Alternative responses when primary service unavailable, providing graceful degradation | Secondary execution paths when circuit is open |
| **fallback chain** | Ordered sequence of alternative response strategies tried in succession | Hierarchical approach to graceful degradation with multiple backup options |
| **bulkhead isolation** | Separating resources to prevent cascade failures between different services | Resource partitioning pattern that complements circuit breakers |
| **resource pool** | Isolated allocation of connections, threads, or other resources per service | Implementation mechanism for bulkhead pattern |
| **semaphore limiting** | Controlling maximum concurrent operations using permit-based access control | Technique for implementing bulkhead resource limits |
| **service isolation** | Maintaining independent circuit breakers per downstream service | Prevents failures in one service from affecting circuit state of others |
| **bulkhead pattern** | Isolation strategy that partitions resources to prevent total system failure | Named after ship compartments that prevent total flooding |

### Integration and Client Terms

| Term | Definition | Context |
|------|------------|---------|
| **transparent integration** | Adding functionality without changing existing client interfaces | Design goal for seamless adoption of circuit breaker functionality |
| **decorator pattern** | Wrapping existing components with additional functionality while preserving interface | Primary integration pattern for HTTP and gRPC clients |
| **service identification** | Mapping requests to specific circuit breaker instances based on destination | Critical for maintaining separate circuit state per service |
| **service registry** | Central mapping of services to circuit breaker instances and configurations | Management component for multi-service circuit breaker coordination |
| **interceptor pattern** | Framework-level request/response processing for transparent functionality injection | gRPC integration mechanism for circuit breaker application |
| **middleware ordering** | Sequence of request processing components in HTTP/gRPC stacks | Important for correct interaction between circuit breakers and other middleware |
| **client wrapper** | HTTP client decorator that adds circuit breaker functionality | Implementation approach for HTTP client integration |

### Error Handling Terms

| Term | Definition | Context |
|------|------------|---------|
| **error classification** | Determining which errors should affect circuit breaker state vs pass through | Critical for avoiding false circuit opens due to client errors |
| **transient failures** | Temporary errors that may resolve automatically and should trigger circuit breaker | Network timeouts, temporary service overload, connection failures |
| **permanent failures** | Persistent errors that indicate service problems and should open circuit | Service crashes, configuration errors, dependency failures |
| **circuit failure** | Error type that should contribute to failure count and potentially open circuit | Distinguished from client errors or non-service-related failures |
| **pass-through error** | Error that should be returned to client without affecting circuit state | Client validation errors, authentication failures, etc. |
| **failure mode** | Specific way a component can fail and its impact on system behavior | Catalog of potential failures for comprehensive error handling |
| **recovery procedure** | Systematic approach to restore system health after failures | Operational processes for handling various failure scenarios |

### Configuration and Management Terms

| Term | Definition | Context |
|------|------------|---------|
| **configuration inheritance** | Hierarchical configuration with defaults and service-specific overrides | Flexible configuration management supporting both global and per-service settings |
| **adaptive threshold** | Dynamic adjustment of circuit parameters based on service behavior patterns | Advanced feature for automatic optimization of circuit breaker sensitivity |
| **service behavior pattern** | Learned characteristics of service performance and failure modes | Data used for intelligent threshold adjustment and failure prediction |
| **threshold optimization** | Process of tuning circuit parameters for optimal performance | Balance between sensitivity to failures and tolerance for transient issues |
| **configuration validation** | Verification that circuit breaker parameters are valid and consistent | Prevents runtime failures due to invalid configuration values |
| **hot configuration reload** | Updating circuit parameters without service restart | Operational feature for dynamic tuning in production environments |

### Testing and Quality Assurance Terms

| Term | Definition | Context |
|------|------------|---------|
| **chaos testing** | Systematic failure injection to validate system resilience | Testing approach that deliberately introduces failures to verify circuit breaker behavior |
| **failure injection** | Controlled introduction of failures into system components | Technique for testing fault tolerance and recovery mechanisms |
| **mock clock** | Controllable time implementation for deterministic testing | Testing utility that eliminates time-based race conditions in tests |
| **chaos orchestration** | Coordinating complex failure scenarios with safety controls | Framework for running sophisticated resilience tests safely |
| **safety boundaries** | Limits and controls to prevent chaos testing damage | Protective measures that ensure testing doesn't impact production systems |
| **resilience metrics** | Measurements of system behavior under adverse conditions | Quantitative assessment of circuit breaker effectiveness |
| **edge case scenarios** | Unusual conditions that test system robustness | Corner cases like clock changes, configuration updates, race conditions |
| **integration testing** | End-to-end testing with real clients and mock services | Validation of complete circuit breaker functionality in realistic scenarios |
| **milestone checkpoints** | Systematic verification points in development process | Structured approach to validating implementation progress |

### Performance and Optimization Terms

| Term | Definition | Context |
|------|------------|---------|
| **latency overhead** | Additional time introduced by circuit breaker processing | Performance cost of resilience features that must be minimized |
| **memory footprint** | RAM usage by circuit breaker data structures and metrics | Resource consumption consideration for high-traffic services |
| **lock contention** | Competition for shared resources in concurrent environments | Performance bottleneck that affects circuit breaker scalability |
| **hot path optimization** | Minimizing overhead in frequently executed code paths | Performance technique focusing on common request processing scenarios |
| **metrics aggregation** | Combining individual request data into summary statistics | Process for creating useful operational metrics from raw request data |
| **performance profiling** | Systematic measurement of circuit breaker performance characteristics | Analysis technique for identifying optimization opportunities |

### Advanced Features Terms

| Term | Definition | Context |
|------|------------|---------|
| **distributed coordination** | Coordinating circuit breaker state across multiple service instances | Advanced feature for consistent behavior in multi-instance deployments |
| **consensus algorithm** | Agreement mechanism for resolving conflicting state changes across instances | Distributed systems technique for maintaining consistency |
| **predictive failure detection** | Identifying potential failures before they manifest as request failures | Machine learning approach to proactive circuit breaker management |
| **behavioral pattern learning** | Analyzing service behavior to identify patterns affecting optimal thresholds | AI-driven optimization of circuit breaker parameters |
| **leading indicators** | Metrics that change before service failures occur | Early warning signals that can trigger proactive circuit opening |
| **service health scoring** | Comprehensive assessment combining performance, reliability, and capacity metrics | Holistic approach to service health beyond simple success/failure rates |
| **operational intelligence** | Analytical insights supporting both immediate decisions and strategic improvements | Data-driven approach to circuit breaker optimization and service reliability |

### Implementation and Development Terms

| Term | Definition | Context |
|------|------------|---------|
| **race condition** | Timing-dependent bugs in concurrent systems | Common pitfall in circuit breaker implementation requiring careful synchronization |
| **atomic operation** | Indivisible operation that prevents inconsistent intermediate states | Concurrency primitive essential for thread-safe circuit breaker state |
| **mutex locking** | Synchronization mechanism for protecting shared data structures | Primary technique for ensuring thread safety in Go implementations |
| **channel communication** | Go-specific mechanism for coordinating between goroutines | Alternative to mutex-based synchronization for some circuit breaker operations |
| **context propagation** | Passing request context through circuit breaker operations | Go pattern for timeout and cancellation handling |
| **dependency injection** | Design pattern for providing external dependencies to components | Architecture approach for testable circuit breaker implementations |
| **interface abstraction** | Defining contracts for pluggable components | Design technique enabling flexible circuit breaker customization |

### Error Constants and Types

| Term | Definition | Context |
|------|------------|---------|
| **ErrCircuitOpen** | Error returned when circuit breaker is in open state | Standard error type indicating circuit protection is active |
| **ErrTimeout** | Error returned when request exceeds configured timeout | Timeout-related failure that may contribute to circuit breaker state |
| **ErrBulkheadCapacity** | Error when resource pool is at maximum capacity | Bulkhead pattern error indicating resource exhaustion |
| **ErrFallbackDepthExceeded** | Error when fallback chain exceeds maximum depth | Protection against infinite fallback recursion |
| **ErrServiceNotFound** | Error when service ID cannot be determined from request | Configuration or integration error in service identification |
| **ErrConfigurationInvalid** | Error for invalid circuit breaker configuration parameters | Validation error for configuration consistency |

### Monitoring and Debugging Terms

| Term | Definition | Context |
|------|------------|---------|
| **circuit inspection** | Detailed examination of circuit breaker state for debugging | Diagnostic capability for troubleshooting circuit behavior |
| **state transition audit** | Logging and tracking of circuit state changes over time | Observability feature for understanding circuit behavior patterns |
| **performance bottleneck** | Component or operation limiting overall system performance | Optimization target identified through performance analysis |
| **diagnostic report** | Comprehensive analysis of circuit breaker issues and recommendations | Automated troubleshooting output for operational support |
| **operational metrics** | Runtime measurements of circuit breaker behavior and effectiveness | Production monitoring data for service reliability assessment |
| **debugging instrumentation** | Additional logging and metrics for troubleshooting purposes | Development and debugging aids for circuit breaker implementation |

This comprehensive glossary serves as both a reference guide and learning resource, helping developers understand the rich terminology surrounding resilient microservices architecture and circuit breaker pattern implementation. Each term connects to specific implementation concerns and design decisions documented throughout the circuit breaker system design.

### Implementation Guidance

The glossary serves primarily as a reference resource, but understanding the relationship between terms and their implementation is crucial for effective development.

#### A. Terminology Usage Guidelines

| Context | Preferred Terms | Avoid |
|---------|----------------|--------|
| Documentation | circuit breaker, failing fast, graceful degradation | breaker, fast fail, degraded mode |
| Code Comments | state machine, failure threshold, recovery testing | FSM, threshold, recovery |
| Error Messages | cascade failure, service isolation, bulkhead capacity | cascading failure, isolation, capacity |
| Metrics Names | sliding window, failure rate, success count | moving window, error rate, success total |

#### B. Term Categorization for Learning

**Foundation Concepts (Learn First):**
- circuit breaker, cascade failure, failing fast
- state machine, failure threshold, recovery testing
- thread safety, observability

**Core Implementation (Learn Second):**
- sliding window, fallback patterns, bulkhead isolation
- service registry, transparent integration, error classification
- decorator pattern, service identification

**Advanced Features (Learn Third):**
- adaptive threshold, distributed coordination, predictive failure detection
- behavioral pattern learning, chaos testing, operational intelligence

**Operational Concerns (Learn Fourth):**
- configuration inheritance, performance profiling, circuit inspection
- diagnostic report, milestone checkpoints, safety boundaries

#### C. Glossary Integration with Code

The glossary terms map directly to code structures and documentation:

```go
// Terms should appear in code comments and documentation
type CircuitBreaker struct {
    // state represents the current state machine position
    state State
    // failureThreshold defines consecutive failures before opening
    failureThreshold int64
    // slidingWindow tracks failure rate over time
    slidingWindow *SlidingWindow
}
```

#### D. Consistency Checking

When implementing the circuit breaker system, maintain consistency between:
- Variable names and glossary terms
- Error messages and defined error types  
- Documentation language and glossary definitions
- API method names and their conceptual purpose

#### E. Learning Path Integration

The glossary supports the milestone-based learning approach:

**Milestone 1:** Focus on circuit breaker, state machine, failure threshold, recovery testing, thread safety
**Milestone 2:** Add sliding window, fallback patterns, bulkhead isolation, error classification  
**Milestone 3:** Include transparent integration, service registry, chaos testing, observability

This structured approach ensures developers build understanding incrementally while maintaining consistent terminology throughout the implementation process.

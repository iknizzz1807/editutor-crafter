# AUDIT & FIX: serverless-runtime

## CRITIQUE
- **CRIU Complexity Massively Understated**: Milestone 3 lists 'container snapshot and restore for sub-100ms cold starts' as an AC alongside basic Python/Node packaging from M1. CRIU (Checkpoint/Restore In Userspace) requires specific kernel capabilities (CAP_SYS_ADMIN), careful handling of file descriptors, TCP connections, and process state. This is an entire project in itself, not a milestone sub-task. The AC 'sub-100ms cold starts' is also unrealistic without Firecracker-level engineering.
- **Missing Event Trigger/Gateway Milestone**: The project only handles HTTP request routing but never addresses event-driven invocation (S3 events, message queues, timers). A real FaaS platform supports multiple trigger types, and this is architecturally distinct from HTTP routing.
- **No Security Model**: Multi-tenant function execution without an explicit security milestone is dangerous. There's no AC requiring seccomp profiles, capability dropping, or tenant isolation verification. The pitfall 'Allowing functions to escape sandbox through privilege escalation' is mentioned but not enforced.
- **Milestone Ordering Issue**: Cold Start Optimization (M3) should logically come after Request Routing (M4), since you need to understand the request flow to know where warm pool integration fits. Currently the warm pool is built before the routing layer that uses it.
- **Uniform 16-hour Estimates**: Every milestone is exactly 16 hours, which is suspicious. Container isolation (M2) with cgroups/namespaces/seccomp is significantly harder than function packaging (M1).
- **Scale-to-Zero AC Insufficient**: 'Implement scale-to-zero removing all instances after idle timeout' doesn't address the cold start penalty this introduces. The AC should require measuring and reporting cold start latency when scaling from zero.
- **No Observability/Logging Milestone**: A FaaS runtime without function log capture, metrics, and invocation tracking is incomplete. Users need to debug their functions.

## FIXED YAML
```yaml
id: serverless-runtime
name: "Serverless Function Runtime"
description: "Build a Function-as-a-Service runtime with container isolation, cold start optimization, event-driven scaling, and multi-tenant execution."
difficulty: expert
estimated_hours: "85-120"
essence: >
  Function execution lifecycle orchestration through process-level isolation
  (Linux namespaces, cgroups, seccomp), cold start mitigation via container
  reuse pools and pre-warming strategies, event-driven request routing and
  queuing, and dynamic auto-scaling based on concurrency metrics with
  scale-to-zero capability.
why_important: >
  Building this teaches production-grade isolation techniques, performance
  optimization under strict latency constraints, and distributed systems
  patterns for auto-scaling and load balancing—skills directly applicable
  to cloud infrastructure and platform engineering roles.
learning_outcomes:
  - Design function packaging and deployment pipelines for multiple language runtimes
  - Implement container-based process isolation with resource limits and security policies
  - Build cold start optimization through container reuse and pre-warming pools
  - Design request routing with concurrency limits and backpressure
  - Implement event-driven auto-scaling with scale-to-zero capability
  - Build function observability with log capture, metrics, and invocation tracking
skills:
  - Container Isolation (namespaces, cgroups)
  - Cold Start Optimization
  - Auto-Scaling Algorithms
  - Request Routing & Load Balancing
  - Resource Limit Enforcement
  - Function Lifecycle Management
  - Security Sandboxing
tags:
  - cold-start
  - devops
  - expert
  - functions
  - multi-tenancy
  - scaling
  - service
  - containers
architecture_doc: architecture-docs/serverless-runtime/index.md
languages:
  recommended:
    - Go
    - Rust
  also_possible:
    - Java
    - Python
resources:
  - name: "AWS Lambda Under the Hood"
    url: https://aws.amazon.com/blogs/compute/understanding-and-remediating-cold-starts-an-aws-lambda-perspective/
    type: article
  - name: "Firecracker MicroVM"
    url: https://github.com/firecracker-microvm/firecracker
    type: documentation
  - name: "OpenFaaS Documentation"
    url: https://docs.openfaas.com/
    type: documentation
  - name: "Linux Namespaces and Cgroups"
    url: https://man7.org/linux/man-pages/man7/namespaces.7.html
    type: documentation
  - name: "CRIU Checkpoint/Restore"
    url: https://criu.org/Main_Page
    type: documentation
prerequisites:
  - type: skill
    name: "Linux process management (fork, exec, signals)"
  - type: skill
    name: "HTTP server development"
  - type: skill
    name: "Docker basics"
  - type: skill
    name: "Basic networking (TCP, sockets)"
milestones:
  - id: serverless-runtime-m1
    name: "Function Packaging & Registry"
    description: >
      Build the function definition format, code packaging pipeline, and
      artifact storage for deploying user functions.
    acceptance_criteria:
      - "Function definition format specifies handler entry point, runtime (Python/Node/Go), memory limit, timeout, and environment variables"
      - "Code upload API accepts a ZIP or tar.gz archive containing function source and dependency manifest"
      - "Dependency resolution installs declared dependencies (pip requirements.txt, npm package.json, go.mod) into the package"
      - "Packaged artifacts are stored with content-addressable hashing (SHA-256) for deduplication and integrity verification"
      - "Function versioning assigns immutable version IDs; deploying new code creates a new version without affecting the previous one"
      - "Package size is validated against a configurable limit (default 50MB) and rejected if exceeded"
    pitfalls:
      - "Transitive dependency conflicts between function deps and runtime deps—use isolated virtual environments"
      - "Not validating archive contents allows path traversal attacks (zip slip vulnerability)"
      - "Including unnecessary files (node_modules, __pycache__) bloats package size—provide .funcignore support"
      - "Go binaries must be cross-compiled for the execution environment's OS/architecture"
    concepts:
      - Artifact packaging and storage
      - Content-addressable storage
      - Dependency resolution
      - Immutable versioning
    skills:
      - Archive handling and validation
      - Dependency management across languages
      - Storage backend integration
      - API design
    deliverables:
      - "Function definition schema with handler, runtime, resource limits, and environment config"
      - "Code upload and packaging API with archive validation"
      - "Dependency bundling for Python, Node.js, and Go runtimes"
      - "Content-addressed artifact storage with version tracking"
    estimated_hours: "12-16"

  - id: serverless-runtime-m2
    name: "Execution Environment & Isolation"
    description: >
      Build isolated execution environments using Linux namespaces and cgroups
      with enforced resource limits and security policies.
    acceptance_criteria:
      - "Each function invocation runs in an isolated Linux namespace (PID, NET, MNT, UTS) preventing cross-function interference"
      - "Memory limit is enforced via cgroups; exceeding the limit triggers OOM kill of the function process"
      - "CPU limit is enforced via cgroups CPU quota; function cannot consume more than its allocated CPU share"
      - "Function process is terminated after configurable timeout (default 30s) with SIGKILL if SIGTERM is not handled"
      - "Filesystem isolation provides a read-only root with a writable /tmp (tmpfs) that is cleaned after invocation"
      - "Seccomp profile restricts system calls to a safe whitelist preventing privilege escalation"
      - "Function stdout/stderr are captured and forwarded to the logging subsystem"
    pitfalls:
      - "Not dropping all Linux capabilities allows container escape via privilege escalation"
      - "Zombie processes from function forks accumulate if PID namespace init doesn't reap children"
      - "Writable /tmp without size limits allows disk exhaustion—use tmpfs with size cap"
      - "Network namespace without proper setup leaves function with no connectivity or unrestricted access"
    concepts:
      - Linux namespaces (PID, NET, MNT, UTS, USER)
      - Cgroups v2 resource control
      - Seccomp BPF filtering
      - Filesystem overlays
    skills:
      - Linux namespace and cgroup programming
      - Process lifecycle management
      - Security policy enforcement
      - Log stream capture
    deliverables:
      - "Namespace-isolated execution environment with PID, network, mount, and UTS isolation"
      - "Cgroup-based memory and CPU enforcement with configurable limits per function"
      - "Seccomp profile restricting dangerous system calls"
      - "Execution timeout enforcement with graceful SIGTERM followed by SIGKILL"
      - "Log capture forwarding function stdout/stderr to centralized logging"
    estimated_hours: "18-25"

  - id: serverless-runtime-m3
    name: "Request Routing & Concurrency"
    description: >
      Build the HTTP gateway that routes invocation requests to function instances,
      manages concurrency limits, and implements request queuing with backpressure.
    acceptance_criteria:
      - "HTTP gateway accepts POST /invoke/{function_name} requests and routes to the correct function"
      - "Per-function concurrency limit caps the number of simultaneous executions (configurable, default 10)"
      - "Requests exceeding concurrency limit are queued with a bounded queue size (default 100)"
      - "Queue overflow returns HTTP 429 Too Many Requests with Retry-After header"
      - "Request timeout returns HTTP 504 Gateway Timeout if function does not respond within deadline"
      - "Synchronous invocation returns function response directly; asynchronous invocation returns 202 Accepted with a status polling endpoint"
      - "Warm instances are preferred over cold instances when routing requests"
    pitfalls:
      - "Unbounded request queue causes memory exhaustion under sustained load—must enforce queue depth limit"
      - "Routing to a cold instance when warm instances are available increases latency unnecessarily"
      - "Not draining in-flight requests during shutdown causes request loss"
      - "Head-of-line blocking: one slow function blocks the queue for all requests to that function"
    concepts:
      - Reverse proxy routing
      - Bounded queue with backpressure
      - Concurrency semaphores
      - Graceful shutdown and request draining
    skills:
      - HTTP server and reverse proxy
      - Concurrent request management
      - Queue data structure implementation
      - Backpressure signaling
    deliverables:
      - "HTTP gateway with function name-based request routing"
      - "Per-function concurrency limiter using semaphore or token bucket"
      - "Bounded request queue with overflow rejection (HTTP 429)"
      - "Synchronous and asynchronous invocation modes"
      - "Warm instance preference in routing decisions"
    estimated_hours: "14-18"

  - id: serverless-runtime-m4
    name: "Cold Start Optimization"
    description: >
      Minimize function startup latency through container reuse, pre-warming pools,
      and pre-initialized runtime environments.
    acceptance_criteria:
      - "Container reuse keeps warm instances alive after invocation for a configurable keep-alive period (default 5 minutes)"
      - "Warm pool maintains a configurable number of pre-initialized containers per function (default 0, configurable up to N)"
      - "Cold start latency is measured and reported as a metric for every invocation (cold vs. warm classification)"
      - "Pre-initialized environments pre-load the language runtime and function dependencies before the first request"
      - "Warm instance eviction uses LRU policy when total warm instances exceed the global memory budget"
      - "Cold start latency for Python/Node functions is under 2 seconds; warm start latency is under 50ms (excluding function execution time)"
    pitfalls:
      - "Over-allocating warm pool instances wastes memory—budget must be enforced globally across all functions"
      - "Stale warm instances may hold outdated function code after a deployment—must invalidate on version change"
      - "JVM-based runtimes (Java) have disproportionately long cold starts due to class loading—requires specific optimization"
      - "Keep-alive connections in warm instances may hold file descriptors or database connections that leak"
    concepts:
      - Container reuse and lifecycle
      - Pre-warming strategies
      - LRU eviction policies
      - Cold/warm start classification
    skills:
      - Container lifecycle management
      - Memory budgeting and eviction
      - Latency measurement and reporting
      - Cache invalidation on deployment
    deliverables:
      - "Container reuse system keeping instances warm after invocation with configurable TTL"
      - "Pre-warm pool maintaining ready-to-use containers per function"
      - "Cold start metrics tracking per-invocation startup classification and latency"
      - "LRU eviction of warm instances when global memory budget is exceeded"
      - "Version-aware cache invalidation retiring warm instances on function redeployment"
    estimated_hours: "14-20"

  - id: serverless-runtime-m5
    name: "Auto-Scaling & Scale-to-Zero"
    description: >
      Implement demand-driven auto-scaling that adjusts function instance count
      based on concurrency metrics, with scale-to-zero for idle functions.
    acceptance_criteria:
      - "Scale-up triggers when concurrent executions exceed a target threshold (e.g., 80% of concurrency limit)"
      - "Scale-down removes idle instances after a cooldown period (configurable, default 5 minutes) to avoid oscillation"
      - "Scale-to-zero terminates all instances after prolonged inactivity (configurable, default 15 minutes)"
      - "Minimum and maximum instance counts are configurable per function and enforced by the scaler"
      - "Scaling decisions account for cold start latency: pre-scale before hitting the concurrency wall"
      - "Scaling events (up, down, to-zero) are logged with timestamp, trigger reason, and resulting instance count"
      - "Integration test: burst of 100 concurrent requests to a scaled-to-zero function completes within 10 seconds with all requests served"
    pitfalls:
      - "Scaling too aggressively on transient spikes causes instance churn and wasted resources"
      - "Scale-to-zero timeout too short for bursty workloads causes repeated cold starts"
      - "Not accounting for cold start latency in scaling decisions means instances aren't ready when needed"
      - "Rapid scale-down during traffic lull followed by immediate scale-up (flapping)—use hysteresis or cooldown windows"
    concepts:
      - Reactive auto-scaling
      - Concurrency-based scaling metrics
      - Hysteresis and cooldown periods
      - Scale-to-zero lifecycle
    skills:
      - Scaling algorithm design
      - Metrics-driven control loops
      - Cooldown and damping mechanisms
      - Integration testing under load
    deliverables:
      - "Concurrency metrics collector tracking active executions per function"
      - "Scale-up controller adding instances when concurrency threshold is exceeded"
      - "Scale-down controller with cooldown period preventing oscillation"
      - "Scale-to-zero controller terminating all instances after inactivity timeout"
      - "Per-function min/max instance configuration enforcement"
      - "Scaling event log with timestamps, triggers, and instance counts"
    estimated_hours: "14-20"

  - id: serverless-runtime-m6
    name: "Observability & Function Logging"
    description: >
      Build function invocation logging, metrics collection, and a simple
      dashboard for monitoring the runtime and debugging functions.
    acceptance_criteria:
      - "Every invocation is logged with: function name, version, request ID, start time, duration, memory used, and status (success/error/timeout)"
      - "Function stdout/stderr logs are associated with the invocation request ID for correlation"
      - "Metrics are collected: invocations per second, error rate, p50/p95/p99 latency, cold start percentage per function"
      - "Metrics are queryable via an API endpoint returning JSON time-series data"
      - "Error logs include the function's stack trace when the function crashes or times out"
      - "Log retention enforces a configurable maximum age (default 7 days) with automatic cleanup"
    pitfalls:
      - "High-throughput logging to disk becomes a bottleneck—use buffered async writes"
      - "Not correlating logs with request IDs makes debugging multi-invocation issues impossible"
      - "Storing all logs indefinitely exhausts disk—must implement rotation and retention policies"
      - "Metrics cardinality explosion if tracking per-invocation dimensions—aggregate to per-function"
    concepts:
      - Structured logging
      - Request correlation IDs
      - Time-series metrics
      - Log rotation and retention
    skills:
      - Structured log formatting
      - Metrics aggregation
      - Async I/O for logging
      - API design for metrics queries
    deliverables:
      - "Invocation log recorder with structured fields per execution"
      - "Function log collector associating stdout/stderr with request IDs"
      - "Metrics aggregator computing per-function latency percentiles, error rates, and cold start ratio"
      - "Metrics query API returning time-series data for monitoring"
      - "Log retention cleaner removing logs older than configured TTL"
    estimated_hours: "10-15"
```
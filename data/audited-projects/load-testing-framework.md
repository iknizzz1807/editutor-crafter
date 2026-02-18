# AUDIT & FIX: load-testing-framework

## CRITIQUE
- **Coordinated Omission**: This is the most critical technical issue. The original mentions it in M1 pitfalls ('measure time from request creation, not send') but the M3 AC for latency percentiles doesn't require correcting for it. If the system under test is slow and virtual users are waiting for responses before sending the next request (closed-loop), the measured latencies will dramatically understate actual user experience. This must be an explicit AC, not just a pitfall.
- **Resource Monitoring for Workers**: When generating high load, the load generator itself can become the bottleneck (CPU saturation, network socket exhaustion, GC pauses). No AC requires monitoring the generator's own resource usage to detect when results are unreliable because the generator is saturated.
- **Source Port Exhaustion**: At high request rates, ephemeral port exhaustion and TIME_WAIT socket accumulation can throttle the load generator. No mention of connection pooling tuning, SO_REUSEADDR, or local port range configuration.
- **M1 Session management is good but under-specified**: 'Session and cookie management persists authentication tokens' is vague. Should specify cookie jar, redirect following, and header propagation.
- **M2 Worker redistribution is very ambitious**: 'Worker failures are detected and their load is redistributed to remaining healthy workers'—this is a complex distributed systems problem (rebalancing without duplicating load). The AC doesn't specify how to avoid a thundering herd when a worker fails and its VUs are redistributed.
- **M2 missing network latency accounting**: Worker-to-coordinator network latency is added to reported response times if timing is done on the coordinator side. Timing must be done on the worker.
- **M3 HDR vs t-digest confusion**: Both are mentioned but they serve different purposes. HDR histogram is for accurate pre-known-range percentiles; t-digest is for streaming quantiles with unknown distribution. The project should pick one primary approach and explain the tradeoff.
- **Missing test scenario definition**: The 'User scenario definition DSL' deliverable in M1 needs more specificity—what constructs does it support? Sequential steps, conditional branches, loops?
- **Missing HTML report with charts**: M3 says 'Final report is exported in HTML and JSON' but no deliverable explicitly produces the HTML report with embedded charts.

## FIXED YAML
```yaml
id: load-testing-framework
name: Load Testing Framework
description: >-
  Distributed performance testing framework with virtual user simulation,
  coordinator-worker architecture, coordinated-omission-aware latency measurement,
  and real-time metrics streaming.
difficulty: expert
estimated_hours: "45-55"
essence: >-
  Coordinated execution of concurrent virtual users across distributed worker
  nodes with coordinated-omission-corrected latency instrumentation, HDR histogram
  aggregation for accurate percentile calculation, generator resource monitoring
  to detect self-saturation, and real-time streaming of performance metrics under
  sustained high-throughput conditions.
why_important: >-
  Building this teaches production-grade distributed systems patterns, high-performance
  concurrent programming, and the statistical nuances of measuring system performance
  under load—including the coordinated omission problem that invalidates most naive
  benchmarks. These skills are critical for infrastructure and performance engineering.
learning_outcomes:
  - Implement virtual user simulation with realistic think-time distributions and session management
  - Design a coordinator-worker architecture for distributed test execution with failure handling
  - Build coordinated-omission-aware request instrumentation capturing intended vs actual send time
  - Implement HDR histogram aggregation for accurate percentile calculation across distributed workers
  - Monitor generator resource usage (CPU, memory, network) to detect self-saturation
  - Handle ephemeral port exhaustion and connection pool tuning for high-throughput workloads
  - Build real-time metrics dashboards with WebSocket streaming
  - Generate comprehensive HTML and JSON reports with percentile charts and analysis
skills:
  - Go Concurrency Patterns
  - Distributed System Coordination
  - Statistical Performance Analysis
  - HDR Histogram Implementation
  - Coordinated Omission Correction
  - Real-time Data Aggregation
  - WebSocket Communication
  - Worker Pool Architecture
  - High-Performance Instrumentation
  - Network Socket Management
tags:
  - benchmarks
  - distributed
  - expert
  - framework
  - load-testing
  - metrics
  - performance
  - scenarios
  - testing
architecture_doc: architecture-docs/load-testing-framework/index.md
languages:
  recommended:
    - Go
    - Rust
  also_possible:
    - Python
resources:
  - name: Grafana k6 Documentation
    url: https://grafana.com/docs/k6/latest/
    type: documentation
  - name: Locust Official Documentation
    url: https://docs.locust.io/
    type: documentation
  - name: HdrHistogram on GitHub
    url: https://github.com/HdrHistogram/HdrHistogram
    type: tool
  - name: Gil Tene - How NOT to Measure Latency (Coordinated Omission)""
    url: https://www.youtube.com/watch?v=lJ8ydIuPFeU
    type: video
  - name: AWS Distributed Load Testing Architecture
    url: https://aws.amazon.com/solutions/implementations/distributed-load-testing-on-aws/
    type: documentation
prerequisites:
  - type: project
    id: http-server-basic
    name: HTTP Server (Basic)
  - type: skill
    name: Concurrency and threading (goroutines/channels or equivalent)
  - type: skill
    name: HTTP protocol fundamentals
  - type: skill
    name: Basic statistics (mean, median, percentiles)
milestones:
  - id: load-testing-framework-m1
    name: Virtual User Simulation
    description: >-
      Implement virtual users with realistic think times, session management,
      connection pool tuning, and scenario execution.
    acceptance_criteria:
      - "Concurrent virtual users are spawned up to a configurable maximum with a linear ramp-up period (e.g., add N users per second until target is reached)"
      - "Each virtual user executes a defined scenario as a repeating sequence of HTTP requests with configurable iteration count or duration limit"
      - Think time between requests is configurable: fixed delay, uniform random range, or exponential distribution to simulate realistic user pacing
      - "Session management maintains a per-user cookie jar, follows redirects, and propagates authentication headers across requests within a scenario"
      - HTTP connection pooling is configurable: max connections per host, keep-alive timeout, and idle connection cleanup are exposed as tunable parameters
      - "Request timing records both 'intended send time' (scheduled) and 'actual send time' to enable coordinated omission correction in the metrics layer"
      - "Scenario DSL supports sequential steps, named requests with method/URL/headers/body, response assertions (status code, body contains), and variable extraction from responses for use in subsequent requests"
    pitfalls:
      - Coordinated omission: in a closed-loop model, if the server is slow, VUs wait before sending the next request, hiding the true latency experienced by users who would have arrived during the delay. Record intended send time.
      - Connection pooling misconfiguration: using a single connection per VU serializes requests; too many connections exhausts server file descriptors
      - Ephemeral port exhaustion: at >10k req/s, TIME_WAIT sockets accumulate. Configure SO_REUSEADDR and expand the local port range (net.ipv4.ip_local_port_range on Linux)
      - "Not following redirects by default causes authentication flows to fail silently"
      - "Variable extraction failure in one step silently breaks subsequent steps—fail fast with clear error"
    concepts:
      - Open-loop vs closed-loop load generation
      - Poisson arrival rate for realistic traffic patterns
      - HTTP connection pooling and keep-alive
      - Cookie jar and session state management
      - Coordinated omission problem
    skills:
      - Goroutine/thread pool management
      - HTTP client configuration
      - Statistical distributions for think times
      - Scenario DSL design
    deliverables:
      - "Scenario definition DSL supporting sequential steps with method, URL, headers, body, assertions, and variable extraction"
      - "Virtual user spawner with configurable ramp-up rate and maximum concurrency"
      - "Think time simulator with fixed, uniform, and exponential distribution options"
      - "Session handler maintaining per-VU cookie jar, auth headers, and extracted variables"
      - "Connection pool configuration exposing max connections, keep-alive, and idle timeout"
      - "Timing recorder capturing intended send time and actual send time per request"
    estimated_hours: "14-16"

  - id: load-testing-framework-m2
    name: Distributed Workers with Resource Monitoring
    description: >-
      Implement distributed load generation with coordinator-worker architecture,
      synchronized execution, and generator resource monitoring.
    acceptance_criteria:
      - Virtual user load is distributed across connected worker nodes proportionally (configurable: even split or weighted)
      - "Coordinator signals synchronized test start across all workers; test begins within 1 second of the signal on all workers (verified)"
      - "Response timing is measured on the WORKER node, not the coordinator, to avoid network latency contamination"
      - "Metrics from all workers are aggregated using HDR histogram merge into a unified result with correct percentile calculations"
      - "Worker health monitoring detects worker failures via heartbeat (configurable interval, default 5s); coordinator logs the failure and continues the test with remaining workers (load is NOT automatically redistributed to avoid thundering herd)"
      - "Each worker reports its own CPU utilization, memory usage, and network socket count; the coordinator flags results as potentially unreliable if any worker exceeds 80% CPU or runs out of ephemeral ports"
      - "Coordinator gracefully stops all workers on test completion or abort signal; workers drain in-flight requests before shutting down"
    pitfalls:
      - "Measuring timing on the coordinator adds network RTT to every measurement—always measure on the worker"
      - Clock skew between workers: use relative elapsed time from test start rather than absolute timestamps for time-series alignment
      - Automatic load redistribution on worker failure can cause a thundering herd: remaining workers suddenly spike in load—prefer logging the failure and reducing total target load
      - "HDR histogram merge requires all histograms to have the same configuration (value range, significant digits)—validate on worker registration"
      - Generator CPU saturation: if the worker is at 100% CPU, it can't send requests fast enough and reported latencies include generator queuing time
    concepts:
      - Coordinator-worker architecture
      - Clock synchronization and relative timing
      - HDR histogram merging across distributed nodes
      - Generator self-saturation detection
      - Scatter-gather pattern for metric collection
    skills:
      - gRPC or WebSocket for worker coordination
      - Distributed system design
      - Resource monitoring (CPU, memory, network)
      - Histogram aggregation
    deliverables:
      - "Worker agent that connects to coordinator, receives scenario and VU allocation, and executes load"
      - "Coordinator that distributes work, synchronizes start, and collects results"
      - "Worker-side timing collection with per-request latency measurement"
      - "HDR histogram merge aggregating worker histograms into unified percentile results"
      - "Worker resource monitor reporting CPU, memory, and socket usage to coordinator"
      - "Self-saturation detector flagging results when generator resources are exhausted"
    estimated_hours: "14-16"

  - id: load-testing-framework-m3
    name: Real-time Metrics, Reporting & Coordinated Omission Correction
    description: >-
      Implement live metrics dashboard with HDR histogram percentiles,
      coordinated omission correction, and comprehensive report generation.
    acceptance_criteria:
      - "Latency percentiles p50, p90, p95, p99, and p99.9 are calculated using HDR histogram with configurable significant digits (default 3)"
      - Coordinated omission correction is applied: for each late response, the histogram is populated with estimated service times for the requests that WOULD have been sent during the delay period, using the recorded intended-vs-actual send time delta
      - "Both corrected and uncorrected latency percentiles are reported side-by-side so users can see the impact of coordinated omission"
      - "Throughput (requests/second) and error rate (percentage) are computed over 1-second sliding windows and displayed in real-time"
      - "Live dashboard updates via WebSocket at least once per second showing current latency percentiles, throughput, error rate, VU count, and worker health"
      - "Final report is exported in both HTML (with embedded charts) and JSON formats with summary statistics, time-series data, and percentile distribution"
      - Report handles edge cases: zero requests, all errors, and empty time windows produce valid output (not crashes)
    pitfalls:
      - "Reporting average latency hides tail latency—always report percentiles, never just mean"
      - "HDR histogram requires pre-defining the value range (e.g., 1ms to 60s) and significant digits; values outside the range are clipped—choose range carefully"
      - "Coordinated omission correction can significantly increase reported p99/p99.9—this is correct behavior, not a bug. The uncorrected values are the ones that are wrong."
      - "Streaming percentile calculation using naive sorted arrays is O(n log n) per query—HDR histogram gives O(1) percentile lookups"
      - WebSocket connection drops during long tests: implement auto-reconnect with buffered catch-up
    concepts:
      - HDR Histogram for percentile accuracy
      - Coordinated omission correction methodology
      - Time-series windowing and sliding windows
      - WebSocket streaming for live dashboards
    skills:
      - HDR histogram implementation/integration
      - Coordinated omission correction algorithm
      - WebSocket server implementation
      - HTML report generation with charts
    deliverables:
      - "HDR histogram-based percentile calculator with configurable range and significant digits"
      - "Coordinated omission corrector populating estimated missed requests into the histogram"
      - "Sliding-window throughput and error rate calculator updated every second"
      - "WebSocket-based live dashboard showing real-time metrics during test execution"
      - "HTML report generator with embedded latency percentile charts, throughput time-series, and summary table"
      - "JSON report exporter with complete time-series data and summary statistics"
    estimated_hours: "14-16"
```
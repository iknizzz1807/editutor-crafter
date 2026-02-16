# AUDIT & FIX: ad-exchange-engine

## CRITIQUE
- **100GB/s telemetry is physically impossible on a single machine**: The audit is dead right. PCIe 5.0 x16 bandwidth is ~64GB/s, and that's theoretical unidirectional. Main memory bandwidth on a high-end server (8-channel DDR5) is ~300GB/s but that's internal, not network ingestion. 100GB/s over the network requires 800Gbps NICs which don't exist for commodity servers. This must be reframed as a distributed stream processing system with realistic per-node throughput.
- **Missing budget pacing**: Without pacing algorithms (e.g., AIMD-based or probabilistic throttling), campaigns exhaust budgets in seconds during peak traffic and show zero ads for the rest of the day. This is a fundamental ad-tech requirement.
- **Missing creative/ad markup validation**: Ad exchanges must validate HTML/JS ad creatives for malware, malicious scripts, and policy violations before serving. This is a security-critical gap.
- **'False positive rate below 0.01%' is unmeasurable without ground truth**: You need labeled data (known bots vs. known humans) to measure false positive rate. The AC should specify how ground truth is established.
- **C10M gateway is aspirational but vague**: '10 million concurrent connections' requires kernel bypass (DPDK, io_uring, XDP) which is a massive undertaking. The AC mentions io_uring and DPDK but treats them as equivalent choices when they have fundamentally different architectures (kernel bypass vs. async kernel IO).
- **Missing OpenRTB protocol compliance**: Real RTB uses the OpenRTB specification. No mention of bid request/response format standards.
- **Missing impression/click tracking and reconciliation**: The pipeline ends at auction but doesn't track whether ads were actually displayed or clicked.
- **Missing second-price auction mechanics**: The AC says 'auction logic' but doesn't specify first-price vs. second-price vs. modified second-price, which fundamentally changes bidder strategy.
- **Global state milestone is vague about consistency model**: 'Eventually consistent budget tracking' — what's the consistency window? What CRDTs or convergence mechanisms are used?

## FIXED YAML
```yaml
id: ad-exchange-engine
name: Lighthouse Ad-Exchange Engine
description: "Real-time bidding exchange with high-performance networking, sub-10ms auction latency, fraud detection, budget pacing, and multi-region settlement."
difficulty: expert
estimated_hours: "90-130"
essence: >
  Ultra-low-latency network I/O serving OpenRTB bid requests, in-memory second-price
  auctions with cache-aligned data structures, real-time fraud filtering via streaming
  anomaly detection, budget pacing algorithms for even spend distribution, and
  multi-region eventually-consistent financial settlement with audit logging.
architecture_doc: architecture-docs/ad-exchange-engine/index.md
languages:
  recommended:
    - Rust
    - C++
    - Go
  also_possible:
    - Java
resources:
  - type: documentation
    name: OpenRTB 2.6 Specification
    url: https://iabtechlab.com/standards/openrtb/
  - type: article
    name: "io_uring and high-performance networking"
    url: https://unixism.net/loti/
  - type: article
    name: "DPDK Overview"
    url: https://www.dpdk.org/about/
prerequisites:
  - type: skill
    name: Systems programming (memory layout, cache lines, SIMD)
  - type: skill
    name: Networking (TCP/UDP, epoll/io_uring)
  - type: skill
    name: Concurrency (lock-free data structures, atomics)
  - type: skill
    name: Container/process isolation
skills:
  - High-performance networking (io_uring / epoll)
  - Lock-free data structures
  - Cache-aligned memory layout
  - OpenRTB protocol
  - Real-time streaming anomaly detection
  - Budget pacing algorithms
  - Eventually consistent distributed systems
  - SIMD-accelerated data processing
tags:
  - build-from-scratch
  - expert
  - go
  - high-performance
  - networking
  - real-time
  - rust
  - systems
milestones:
  - id: ad-exchange-engine-m1
    name: "High-Performance Network Gateway"
    description: >
      Build a high-performance HTTP server capable of handling hundreds of thousands
      of concurrent connections with minimal latency, using async I/O (io_uring or
      epoll) and zero-copy techniques.
    acceptance_criteria:
      - "HTTP server uses io_uring (Linux) or kqueue (macOS) for async I/O; the choice is documented with rationale for the target platform"
      - "Achieve at least 200,000 requests per second on a single core for a simple echo endpoint, verified by a load test (wrk or similar)"
      - "Support at least 100,000 concurrent keep-alive connections without file descriptor exhaustion (configure ulimits, use connection pooling)"
      - "Request parsing uses zero-copy techniques: HTTP headers are parsed in-place from the read buffer without copying into separate string allocations"
      - "Inter-thread communication uses lock-free ring buffers (SPSC or MPSC) with documented memory ordering guarantees (acquire/release semantics)"
      - "Hot path allocations are minimized: the request-response path for a cached auction uses arena/pool allocation with <10 heap allocations per request, verified by profiling"
      - "Graceful shutdown: in-flight requests complete within a configurable drain timeout before the server exits"
    pitfalls:
      - "io_uring vs DPDK confusion: io_uring is async kernel I/O (still uses the kernel network stack); DPDK is full kernel bypass requiring dedicated NICs and hugepages. They are fundamentally different architectures. Choose one and understand the tradeoffs."
      - "False sharing: lock-free ring buffers with producer/consumer counters on the same cache line cause severe performance degradation. Pad counters to separate cache lines (64 bytes)."
      - "File descriptor limits: Linux defaults to 1024 open FDs. Forgetting to raise ulimits causes silent connection refusal under load."
      - "Memory ordering bugs: using relaxed atomics where acquire/release is needed causes data corruption that only manifests under high contention on specific CPU architectures (ARM more than x86)."
    concepts:
      - Async I/O (io_uring, epoll, kqueue)
      - Lock-free data structures
      - Zero-copy networking
      - Cache-line optimization
    skills:
      - io_uring or epoll programming
      - Lock-free ring buffer implementation
      - Memory allocator design (arena/pool)
      - Performance profiling and benchmarking
    deliverables:
      - Async HTTP server with io_uring/epoll backend
      - Lock-free SPSC/MPSC ring buffer for inter-thread communication
      - Zero-copy HTTP request parser
      - Load test demonstrating 200k+ RPS on single core
      - Profiling report showing hot path allocation count
    estimated_hours: "18-25"

  - id: ad-exchange-engine-m2
    name: "OpenRTB Auction Engine"
    description: >
      Implement the real-time bidding auction engine: parse OpenRTB bid requests,
      fan out to bidders, collect responses, run second-price auction, and return
      the winning ad with sub-10ms p99 latency.
    acceptance_criteria:
      - "Bid request parser handles OpenRTB 2.5/2.6 JSON bid request format, extracting impression objects, device info, user segments, and geo data"
      - "Auction engine fans out bid requests to registered demand-side platform (DSP) endpoints with a configurable per-bidder timeout (default 100ms); slow bidders are excluded from the auction"
      - "Second-price auction: the winning bidder pays the second-highest bid price plus a minimum increment (1 cent), not their own bid price"
      - "Targeting rules filter eligible campaigns using bitset-based evaluation: geo (country/region), device type, user segment membership, and time-of-day targeting are evaluated in <1ms"
      - "Cache-aligned data structures: campaign and targeting data is stored in contiguous, cache-line-aligned arrays for sequential access; benchmark shows >80% L1 cache hit rate during auction"
      - "Ad creative/markup validation: winning ad HTML/JS is sanitized against a whitelist of allowed tags and attributes; known malicious patterns (crypto miners, redirects) are rejected"
      - "P99 auction latency remains under 10ms under sustained load of 50,000 auctions per second, verified by latency histogram"
    pitfalls:
      - "First-price vs second-price: implementing first-price when the protocol expects second-price (or vice versa) fundamentally changes bidder economics. Document the auction type explicitly."
      - "Bidder timeout handling: if you wait for the slowest bidder, all auctions are slow. Use a hard timeout with async response collection; late responses are discarded."
      - "JSON parsing overhead: OpenRTB bid requests can be large (10KB+). Use a streaming JSON parser or pre-allocate parse buffers to avoid per-request allocation."
      - "Ad markup injection: if ad creative HTML is served without sanitization, malicious bidders can inject scripts that steal user data or redirect to phishing sites."
    concepts:
      - OpenRTB protocol
      - Second-price auction mechanics
      - Bitset-based targeting
      - Cache-aware data layout
    skills:
      - OpenRTB JSON parsing
      - Bitset operations for targeting rules
      - Cache-line-aware struct layout
      - Concurrent fan-out with timeout
      - HTML/JS sanitization
    deliverables:
      - OpenRTB bid request parser
      - Bidder fan-out engine with per-bidder timeout
      - Second-price auction implementation
      - Bitset-based targeting rule evaluator
      - Ad creative sanitizer
      - Latency benchmark demonstrating p99 <10ms at 50k QPS
    estimated_hours: "20-28"

  - id: ad-exchange-engine-m3
    name: "Fraud Detection & Budget Pacing"
    description: >
      Implement real-time fraud detection using streaming anomaly detection and
      budget pacing algorithms to ensure even campaign spend distribution.
    acceptance_criteria:
      - "Sliding window anomaly detection identifies IP addresses or device IDs exhibiting abnormal request patterns (e.g., >100 requests/second from a single IP) using count-min sketch or similar probabilistic data structure"
      - "Distributed blacklist: flagged IPs/devices are propagated to all exchange nodes via a shared cache (Redis or gossip protocol) within 5 seconds of detection"
      - "Fraud detection pipeline processes the full request stream inline with <1ms added latency per request; heavy analysis (SIMD-accelerated pattern matching) runs on a dedicated thread pool processing batched data"
      - "Measurable fraud metrics: the system reports detection rate, flagged request count, and estimated false positive rate using a holdout sample of manually-labeled traffic (at least 10,000 labeled samples)"
      - "Budget pacing: each campaign has a daily budget and the pacing algorithm distributes spend evenly across the day using a probabilistic throttle (bid probability = remaining_budget / remaining_time * adjustment_factor)"
      - "Pacing adjusts dynamically: if traffic volume drops, the pacing algorithm increases bid probability to avoid under-delivery; if traffic spikes, it decreases bid probability to avoid exhaustion"
      - "Budget overspend protection: even under race conditions with multiple auction nodes, total spend exceeds the daily budget by no more than 1%, enforced by periodic budget synchronization"
    pitfalls:
      - "100GB/s per node is impossible: even with DPDK, a single 100GbE NIC delivers 12.5GB/s. Design the fraud pipeline as distributed, with each node processing its local traffic share."
      - "False positive measurement without ground truth is meaningless: you must have labeled data (known bots, known humans) to compute false positive rate. Use honeypot traffic and known bot signatures for labeling."
      - "Budget pacing race conditions: two nodes simultaneously checking remaining budget and both deciding to bid can cause overspend. Use optimistic locking with periodic reconciliation, not distributed locks (too slow)."
      - "SIMD portability: AVX-512 instructions are not available on all CPUs (notably AMD before Zen 4). Use runtime feature detection and provide scalar fallbacks."
    concepts:
      - Streaming anomaly detection
      - Probabilistic data structures (count-min sketch, bloom filter)
      - Budget pacing algorithms
      - Distributed cache synchronization
    skills:
      - Count-min sketch implementation
      - SIMD programming with runtime detection
      - Probabilistic throttling algorithms
      - Distributed cache integration
      - Metric collection and analysis
    deliverables:
      - Sliding window anomaly detector with count-min sketch
      - Distributed blacklist with sub-5-second propagation
      - Inline fraud filtering with <1ms overhead
      - Budget pacing engine with probabilistic throttle
      - Budget synchronization across nodes with <1% overspend guarantee
      - False positive measurement framework with labeled holdout set
    estimated_hours: "22-30"

  - id: ad-exchange-engine-m4
    name: "Impression Tracking, Global State & Settlement"
    description: >
      Implement impression/click event tracking, multi-region budget state
      synchronization with eventual consistency, and immutable financial
      settlement ledger.
    acceptance_criteria:
      - "Impression tracking: when an ad is displayed, the client fires an impression beacon; the exchange records the impression with auction ID, winning bid, and timestamp"
      - "Click tracking: click-through URLs are wrapped with a redirect through the exchange for click recording before forwarding to the advertiser's landing page"
      - "Impression-auction reconciliation: a background process matches impressions to auctions, flagging discrepancies (impressions without auctions, auctions without impressions) for investigation"
      - "Multi-region budget state: each region maintains a local budget counter updated optimistically; regions synchronize budget state every N seconds (configurable, default 5s) using a convergent replicated data type (CRDT counter) or central coordinator"
      - "Late-arriving events (impressions received after budget synchronization) are handled with a grace window; events arriving after the grace window are logged but do not affect billing"
      - "Deduplication: duplicate impression or click events (same auction_id + event_type) are detected and discarded using a time-windowed deduplication set"
      - "Immutable financial ledger: all billing events (impressions, clicks, spend) are written to an append-only log with cryptographic chaining (each entry's hash includes the previous entry's hash) for tamper detection"
      - "Regional failover: if a region becomes unavailable, traffic is rerouted to surviving regions; the failed region's uncommitted budget changes are reconciled upon recovery without exceeding campaign budgets by more than the documented tolerance"
    pitfalls:
      - "CRDT counter convergence delay: during the convergence window, total cluster-wide budget tracking is approximate. Document the maximum possible overspend during a partition."
      - "Click fraud via beacon replay: impression/click beacons must include a cryptographic nonce or signed token to prevent replay by bots."
      - "Append-only log corruption detection: if the hash chain is only verified on read, corruption can go undetected for days. Verify chain integrity periodically in a background process."
      - "Failover budget reconciliation: if a region was partitioned and independently spending budget, reconciliation may discover the combined spend exceeds the budget. Define a policy (absorb overspend vs. bill advertiser)."
    concepts:
      - Impression/click tracking pipeline
      - CRDTs and eventual consistency
      - Cryptographic audit logs
      - Regional failover
    skills:
      - Event tracking with beacon URLs
      - CRDT implementation (G-Counter or PN-Counter)
      - Append-only log with hash chaining
      - Multi-region deployment and failover
      - Reconciliation algorithms
    deliverables:
      - Impression beacon endpoint with auction-impression matching
      - Click redirect tracker with recording
      - Multi-region budget synchronization using CRDT counters
      - Late event handling with grace window
      - Deduplication filter for impression/click events
      - Append-only financial ledger with cryptographic hash chain
      - Reconciliation process for post-failover budget alignment
    estimated_hours: "22-30"
```
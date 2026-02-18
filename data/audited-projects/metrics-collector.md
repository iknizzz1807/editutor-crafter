# AUDIT & FIX: metrics-collector

## CRITIQUE
- **Out-of-Order Samples**: Gorilla-style XOR encoding assumes samples arrive in timestamp order. Out-of-order samples break the delta-of-delta encoding, producing incorrect compression or corrupted data. The original has no AC requiring rejection or handling of out-of-order samples.
- **Staleness Handling**: When a metric series stops reporting (service crashes, target removed), the original system has no way to mark the series as stale. Queries continue to return the last known value indefinitely, which is misleading. Prometheus uses a special NaN staleness marker—this must be addressed.
- **Scrape Jitter**: If 100 targets are all scraped on exactly the same 15-second boundary, they all receive simultaneous HTTP requests (thundering herd). Adding random jitter (e.g., +/- 10% of scrape interval) spreads the load. The original has no mention of this.
- **M1 Counter reset handling**: Counter resets on process restart produce a value decrease, but rate() needs to handle this. M1 mentions it as a pitfall but M4 AC should explicitly require it.
- **M2 Scrape interval per target**: The deliverable mentions 'configurable per-target intervals' but the AC doesn't explicitly require per-target scrape configuration. All targets at the same interval is a common but limiting default.
- **M3 '2 bytes per sample'**: This is achievable with Gorilla compression on regular-interval data but the AC doesn't specify the conditions (regular vs irregular timestamps). Compression ratio should be measured, not just claimed.
- **M4 Missing subquery support**: The query engine AC covers basic instant/range queries and aggregation but doesn't mention subqueries (using the result of one query as input to another), which are essential for complex alerting expressions.
- **Missing summary metric type**: M1 mentions counter, gauge, and histogram but not summary (client-side calculated quantiles). While Prometheus recommends histograms over summaries, summaries are part of the exposition format and must be parseable.
- **Missing remote write/read**: No mention of remote write (forwarding metrics to long-term storage) or remote read (federated queries), which are critical for production-scale metrics systems.

## FIXED YAML
```yaml
id: metrics-collector
name: Metrics Collector
description: >-
  Pull-based metrics collection system with Gorilla-style time-series compression,
  staleness handling, scrape jitter, query engine with rate() and counter reset
  detection, and cardinality management.
difficulty: advanced
estimated_hours: "50-60"
essence: >-
  Pull-based metric scraping with jitter to prevent thundering herd, time-series
  storage using delta-of-delta and XOR encoding with out-of-order rejection and
  staleness markers, combined with a declarative query language supporting rate()
  with counter reset detection, aggregation functions, and label-based filtering
  over high-cardinality data streams.
why_important: >-
  Understanding how metrics systems work—from scraping to compression to query
  execution—enables you to design better instrumentation, debug query performance
  issues, and operate metrics infrastructure at scale. These skills are essential
  for SRE, platform engineering, and backend roles.
learning_outcomes:
  - Design metrics data model with counters, gauges, histograms, and summaries with label validation
  - Implement pull-based scraping with jitter, service discovery, and staleness markers
  - Build time-series storage with Gorilla compression and out-of-order rejection
  - Design a PromQL-like query engine with rate(), counter reset detection, and aggregation
  - Handle high-cardinality metrics with configurable series limits
  - Implement staleness detection for disappeared metric series
skills:
  - Time-Series Compression (Gorilla)
  - Service Discovery
  - Pull-Based Scraping with Jitter
  - Query Language Design
  - Delta-of-Delta Encoding
  - XOR Float Encoding
  - Label-Based Indexing
  - Staleness Handling
  - Counter Reset Detection
  - Downsampling Strategies
tags:
  - advanced
  - databases
  - distributed-systems
  - exposition
  - labels
  - observability
  - prometheus
  - scraping
architecture_doc: architecture-docs/metrics-collector/index.md
languages:
  recommended:
    - Go
    - Rust
  also_possible:
    - Python
resources:
  - name: Prometheus Documentation
    url: https://prometheus.io/docs/introduction/overview/
    type: documentation
  - name: Write a TSDB from Scratch
    url: https://nakabonne.dev/posts/write-tsdb-from-scratch/
    type: tutorial
  - name: PromQL Querying Basics
    url: https://prometheus.io/docs/prometheus/latest/querying/basics/
    type: documentation
  - name: Gorilla Time Series Compression (Facebook)
    url: https://www.vldb.org/pvldb/vol8/p1816-teller.pdf
    type: paper
  - name: Prometheus Staleness Handling
    url: https://prometheus.io/docs/prometheus/latest/querying/basics/#staleness
    type: documentation
prerequisites:
  - type: project
    id: time-series-db
  - type: skill
    name: Binary encoding and bit manipulation
  - type: skill
    name: HTTP server and client implementation
milestones:
  - id: metrics-collector-m1
    name: Metrics Data Model with Validation
    description: >-
      Implement the metrics data model with counters, gauges, histograms, and
      summaries, with label validation and cardinality enforcement.
    acceptance_criteria:
      - "Counter type supports only monotonic increment operations; decrement attempts are rejected with an error"
      - "Gauge type supports set, increment, and decrement operations reflecting point-in-time values"
      - "Histogram type buckets observations into immutable configurable boundaries and tracks count and sum; bucket boundaries cannot be changed after creation"
      - "Summary type (client-side quantiles) is parseable from the exposition format even though server-side histograms are preferred"
      - "Labels attach key-value pairs to metrics; label names are validated against pattern [a-zA-Z_][a-zA-Z0-9_]* and label values are arbitrary UTF-8 strings"
      - Metric names are validated: must match [a-zA-Z_:][a-zA-Z0-9_:]* and must not use reserved prefixes (__)
      - Cardinality enforcement: configurable maximum unique label combinations per metric name (default 10,000); exceeding the limit logs a warning and drops new series
      - "Metric metadata (HELP text, TYPE declaration, UNIT) is stored and retrievable alongside metric values"
    pitfalls:
      - "Counter resets on process restart produce a value decrease—consumers must use rate() which detects resets, but the data model should allow storing the reset event"
      - "High cardinality labels (user_id, request_path) create millions of series, exhausting memory—enforce limits at ingestion"
      - "Histogram bucket boundaries are immutable after creation; changing them requires creating a new metric—choose boundaries based on expected value distribution"
      - "Summary quantiles are pre-calculated on the client; they cannot be aggregated across instances—prefer histograms for server-side aggregation"
    concepts:
      - Metric type semantics (counter monotonicity, gauge mutability)
      - Label cardinality and memory impact
      - Histogram vs summary tradeoffs
      - Metric naming conventions
    skills:
      - Data structure design for time series
      - Input validation and constraint enforcement
      - Memory-efficient label storage
      - Type system implementation
    deliverables:
      - "Counter, gauge, histogram, and summary type implementations with correct semantic operations"
      - "Label validator enforcing naming conventions and cardinality limits"
      - "Metric name validator rejecting invalid names and reserved prefixes"
      - "Metric metadata store for HELP, TYPE, and UNIT information"
    estimated_hours: "10-12"

  - id: metrics-collector-m2
    name: Scrape Engine with Jitter and Staleness
    description: >-
      Build a scrape engine that pulls metrics from discovered targets with
      jitter to prevent thundering herd, and marks series as stale when targets
      disappear.
    acceptance_criteria:
      - "Scrape targets are discovered from static config files and at least one dynamic backend (DNS-based service discovery or file-based discovery)"
      - "Metrics are pulled from HTTP endpoints by parsing the Prometheus text exposition format including metric types, label sets, and timestamps"
      - "Scrape scheduling includes random jitter of +/-10% of the scrape interval to prevent all targets from being scraped simultaneously (thundering herd prevention)"
      - "Scrape timeout (configurable per target, default 10s) cancels the HTTP request and marks the target as down for that cycle; 'up' metric is set to 0"
      - "When a target is removed from discovery or fails to respond for a configurable number of consecutive scrapes (default 5), all its series are marked with a staleness marker (special NaN value)"
      - "Staleness markers cause the series to be excluded from query results after the staleness timestamp; lookback window queries do not return stale data"
      - "Per-target scrape interval is configurable; different targets can have different scrape frequencies"
      - Label collision between target labels (e.g., instance, job) and metric-internal labels is resolved: target labels take precedence with a configurable prefix for conflicting metric labels
    pitfalls:
      - Thundering herd: 1000 targets all scraped at exactly T+0, T+15, T+30 creates periodic load spikes—jitter spreads scrapes across the interval
      - Missing staleness markers: a crashed service's last metric value appears 'frozen' in queries indefinitely, misleading dashboards and alerts
      - Network timeout blocking the scrape goroutine: use context.WithTimeout and cancel properly
      - Label collision: target has a metric with label 'instance' and the scrape config also adds 'instance'—one overwrites the other silently without collision handling
    concepts:
      - Pull-based scraping architecture
      - Scrape jitter for load distribution
      - Staleness markers for disappeared series
      - Service discovery integration
      - Label collision resolution
    skills:
      - HTTP client with timeout and context cancellation
      - Prometheus exposition format parsing
      - Service discovery integration
      - Staleness marker implementation
    deliverables:
      - "Target discovery from static config and DNS-based service discovery"
      - "HTTP scraper parsing Prometheus exposition format with type and metadata handling"
      - "Jitter-based scrape scheduler distributing scrape times across the interval"
      - "Staleness marker injection for disappeared targets/series"
      - "Per-target scrape configuration with independent intervals and timeouts"
      - "Label collision resolver with configurable prefix for conflicts"
    estimated_hours: "12-14"

  - id: metrics-collector-m3
    name: Time-Series Storage with Gorilla Compression
    description: >-
      Implement time-series storage with Gorilla-style compression, out-of-order
      sample rejection, write-ahead logging, and retention.
    acceptance_criteria:
      - "Gorilla-style compression encodes timestamps as delta-of-delta and values as XOR of consecutive floats, achieving less than 2 bytes per sample on regular-interval data (measured via benchmark on 100k+ samples)"
      - "Out-of-order samples (timestamp older than the latest sample for that series) are rejected with a logged warning; they are NOT silently accepted as they would corrupt delta encoding"
      - "Write-ahead log (WAL) records all ingested samples before they are written to compressed chunks; on crash recovery, the WAL is replayed to recover data"
      - "WAL checkpointing truncates the WAL after compressed chunks are flushed to disk, preventing unbounded growth"
      - "Retention policy deletes compressed chunks and WAL segments older than the configured retention period (default 15 days)"
      - "Chunk boundaries align to configurable time windows (default 2 hours); concurrent writes to different chunks are serialized per-series using per-series locks"
      - "High-cardinality label combinations are bounded by the cardinality limits from M1; the storage layer does not accept series exceeding the limit"
    pitfalls:
      - Out-of-order samples corrupt delta-of-delta encoding: the delta becomes negative in unexpected ways, producing incorrect decompressed values—reject strictly
      - Compression ratio degrades significantly with irregular scrape intervals: if scrape times vary by seconds, delta-of-delta uses more bits per timestamp—document expected ratios
      - "Concurrent writes to the same series (e.g., from duplicate scrape configs) require per-series locking; global locks kill throughput"
      - Chunk boundary at exact timestamp: a sample at exactly T=chunk_boundary belongs to which chunk? Define consistently (e.g., inclusive start, exclusive end)
      - "WAL replay after crash can be slow for large WAL files—implement incremental checkpointing"
    concepts:
      - Gorilla time-series compression (delta-of-delta, XOR)
      - Out-of-order rejection
      - Write-ahead logging for durability
      - Block-based storage with chunk boundaries
      - Per-series concurrency control
    skills:
      - Bit-level encoding and decoding
      - WAL implementation and recovery
      - File-based chunk management
      - Per-series lock management
    deliverables:
      - "Gorilla compression engine implementing delta-of-delta timestamps and XOR float values"
      - "Out-of-order sample rejection with logging"
      - "Write-ahead log with checkpoint and recovery"
      - "Chunk-based storage organizing compressed data by time windows"
      - "Retention manager deleting chunks older than configured TTL"
      - "Compression benchmark measuring bytes/sample on regular and irregular interval data"
    estimated_hours: "14-16"

  - id: metrics-collector-m4
    name: Query Engine with rate() and Counter Reset Detection
    description: >-
      Build a PromQL-like query engine with instant queries, range queries,
      aggregation, rate() with counter reset detection, and staleness-aware
      lookback.
    acceptance_criteria:
      - "PromQL-like queries support metric name selection, label matchers (=, !=, =~, !~), and time range specification"
      - "Instant queries return the most recent sample within a configurable lookback window (default 5 minutes); samples older than the lookback are not returned (staleness-aware)"
      - "Range queries return all samples within [start, end] at the specified step interval"
      - "Aggregation functions sum, avg, max, min, count, and quantile produce correct results grouped by specified label dimensions with 'by' and 'without' clauses"
      - rate() function calculates per-second rate from counter metrics with counter reset detection: when a counter value decreases (process restart), rate() treats it as a reset and adds the new value to the accumulated total
      - Label matchers support: exact match (=), not-equal (!=), regex match (=~), and negative regex (!~); regex uses RE2 syntax
      - "Query timeout (configurable, default 2 minutes) cancels long-running queries and returns an error; queries scanning more than configurable max series (default 50,000) are rejected upfront"
      - Staleness markers are respected: series with staleness markers are excluded from query results after the staleness timestamp
    pitfalls:
      - rate() without counter reset detection: counter restarts produce a massive negative rate; naive (last-first)/duration gives incorrect results
      - "Regex label matching can be slow (O(n) scan of all series)—use inverted index for exact matches, fall back to regex only for =~ and !~"
      - "High-cardinality queries (e.g., sum by (user_id)) scan millions of series and cause OOM—enforce max series limit per query"
      - Staleness not respected: stale series appear in aggregations with their last known value, skewing results
      - Quantile aggregation across labels: quantile(0.99, ...) over pre-computed quantiles is incorrect; only histogram_quantile over histograms gives correct results
    concepts:
      - Expression parsing and abstract syntax trees
      - Vector and matrix result types
      - Counter reset detection in rate()
      - Staleness-aware lookback
      - Query limits and timeout
    skills:
      - Query language parser (recursive descent or PEG)
      - Time-series math with reset detection
      - Label matcher optimization with inverted index
      - Memory-bounded query execution
    deliverables:
      - "PromQL-style query parser supporting instant queries, range queries, and label matchers"
      - "rate() function with counter reset detection"
      - "Aggregation engine implementing sum, avg, max, min, count, quantile with by/without grouping"
      - "Staleness-aware lookback excluding stale series from results"
      - "Query limiter enforcing max series and timeout per query"
      - "Label matcher optimizer using inverted index for exact matches"
    estimated_hours: "14-16"
```
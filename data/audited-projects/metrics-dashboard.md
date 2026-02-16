# AUDIT & FIX: metrics-dashboard

## CRITIQUE
- **Downsampling/Rollups**: The storage AC mentions 'retention and compaction' but the original M2 AC does not explicitly require downsampling (rollups) for long-term historical views. Without downsampling, querying months of high-resolution data is impossibly slow and expensive. The audit finding is correct.
- **Pull vs Push ambiguity**: M1 mentions 'Prometheus exposition format' and the deliverables mention 'Push and pull model support' but the AC doesn't clearly distinguish or require both. This is architecturally significant—pull-based scraping and push-based ingestion have very different failure modes and backpressure characteristics.
- **Template Variables**: The visualization milestone has no AC for template variables (e.g., $cluster, $namespace) that allow a single dashboard to be reused across environments. This is a fundamental Grafana/dashboard pattern.
- **High Cardinality**: The pitfall 'Too many unique label values' is mentioned but no AC requires cardinality limiting or enforcement. Cardinality explosion is the #1 operational issue with metrics systems.
- **M2 '2 bytes per sample' claim**: The AC says 'millions of data points with efficient write throughput and compression' but doesn't specify a measurable compression target. The essence mentions Gorilla compression but M2 doesn't require implementing it.
- **M3 is a full frontend project**: Building a dashboard with line charts, heatmaps, auto-refresh, sharing, and time range selectors in '6-10 hours' is wildly underestimated for an advanced project. Either scope it down or increase hours.
- **M4 Missing deduplication**: The alerting system AC mentions 'notification channels' but doesn't address alert deduplication across evaluation cycles—without it, a continuously firing alert sends a notification every evaluation interval.
- **Missing recording rules**: No mention of recording rules (pre-computed aggregations stored as new time series) which are essential for query performance at scale.

## FIXED YAML
```yaml
id: metrics-dashboard
name: Metrics & Alerting Dashboard
description: >-
  Time-series metrics system with collection, efficient storage with downsampling,
  a query engine, visualization dashboard, and threshold-based alerting.
difficulty: advanced
estimated_hours: "30-45"
essence: >-
  Time-series data ingestion via pull-based scraping and push-based gateway,
  label-based indexing for multi-dimensional queries with cardinality enforcement,
  block-based storage with Gorilla compression and downsampling for long-term
  retention, threshold-based rule evaluation with state machine transitions and
  notification routing, and dashboard visualization with template variables.
why_important: >-
  Building this teaches production observability patterns that power modern cloud
  infrastructure, combining time-series storage design, query optimization,
  cardinality management, and alerting—skills essential for SRE and backend
  engineering roles.
learning_outcomes:
  - Implement metrics instrumentation using counters, gauges, and histograms with label cardinality enforcement
  - Design time-series storage with Gorilla-style compression, retention policies, and downsampling
  - Build a query engine supporting metric selection, label filtering, aggregation, and rate calculations
  - Develop alerting rules with state machine transitions (pending/firing/resolved) and notification routing
  - Create visualization dashboards with template variables for multi-environment reuse
  - Implement both pull-based scraping and push-based ingestion with clear failure mode documentation
  - Debug high-cardinality metrics issues and implement cardinality limits
skills:
  - Time-Series Databases
  - Metrics Instrumentation
  - Query Language Design
  - Alerting & Notifications
  - Data Visualization
  - Pull-Based Scraping
  - Push-Based Ingestion
  - Service Discovery
  - Cardinality Management
  - Downsampling
tags:
  - advanced
  - devops
  - go
  - grafana
  - python
  - time-series
  - visualization
architecture_doc: architecture-docs/metrics-dashboard/index.md
languages:
  recommended:
    - Go
    - Python
  also_possible:
    - JavaScript
    - Rust
resources:
  - type: documentation
    name: Prometheus Documentation
    url: https://prometheus.io/docs/introduction/overview/
  - type: documentation
    name: Grafana Documentation
    url: https://grafana.com/docs/grafana/latest/
  - type: article
    name: Google SRE Book - Monitoring
    url: https://sre.google/sre-book/monitoring-distributed-systems/
  - type: paper
    name: Gorilla Time-Series Compression (Facebook)
    url: https://www.vldb.org/pvldb/vol8/p1816-teller.pdf
prerequisites:
  - type: skill
    name: HTTP APIs
  - type: skill
    name: Time-series data concepts
  - type: skill
    name: Basic statistics (mean, percentiles)
  - type: skill
    name: Docker
milestones:
  - id: metrics-dashboard-m1
    name: Metrics Collection with Cardinality Enforcement
    description: >-
      Implement metrics collection with counters, gauges, and histograms via
      both pull-based scraping and push-based ingestion, with label cardinality limits.
    acceptance_criteria:
      - "Counter metrics are monotonically increasing cumulative values; increments by negative values are rejected"
      - "Gauge metrics are point-in-time values that can increase, decrease, or be set to an arbitrary value"
      - "Histogram metrics bucket observations into configurable ranges and track count and sum; bucket boundaries are immutable after creation"
      - "Labels attach key-value dimensions to metrics; label names are validated against [a-zA-Z_][a-zA-Z0-9_]* pattern"
      - "Pull-based scraping fetches metrics from configured HTTP endpoints in Prometheus exposition format at configurable intervals"
      - "Push-based ingestion accepts metrics via HTTP POST in JSON format for short-lived jobs that cannot be scraped"
      - "Cardinality enforcement rejects or drops metric series when unique label combinations exceed a configurable limit per metric name (default 10,000)"
      - "Metric operations are thread-safe: concurrent increments from 100 goroutines produce the correct total"
    pitfalls:
      - "Cardinality explosion: a label like 'user_id' or 'request_path' with unbounded values creates millions of series, exhausting memory—enforce cardinality limits"
      - "Counter resets on process restart: consumers must use rate() which handles resets, but raw counter values will show drops"
      - "Histogram bucket boundaries cannot be changed after creation without invalidating existing data—choose boundaries carefully upfront"
      - "Thread safety: naive counter increment without atomic operations or mutexes loses increments under concurrency"
    concepts:
      - Time-series data model
      - Metric types and their semantics
      - Pull vs push collection models
      - Label cardinality and its memory impact
    skills:
      - Concurrent data structures
      - Time-series data modeling
      - HTTP endpoint scraping
      - Exposition format parsing
    deliverables:
      - "Metric type implementations for counter, gauge, and histogram with correct semantic behavior"
      - "Pull-based scraper fetching and parsing Prometheus exposition format from configurable targets"
      - "Push-based HTTP ingestion endpoint accepting JSON-formatted metric data points"
      - "Label validator and cardinality enforcer rejecting metrics exceeding configured limits"
    estimated_hours: "7-10"

  - id: metrics-dashboard-m2
    name: Storage, Downsampling & Querying
    description: >-
      Implement time-series storage with compression, retention policies,
      downsampling for historical data, and a query engine with aggregation.
    acceptance_criteria:
      - "Time-series storage achieves less than 2 bytes per sample using delta-of-delta timestamp encoding and XOR float compression (Gorilla-style)"
      - "Retention policy automatically deletes raw data older than a configurable period (e.g., 15 days)"
      - "Downsampling produces 5-minute resolution rollups (min, max, avg, count, sum) for data older than the raw retention period; rollups are retained for a longer configurable period (e.g., 1 year)"
      - "Range queries return data points within specified start/end timestamps; queries automatically use downsampled data for time ranges exceeding raw retention"
      - "Aggregation functions sum, avg, max, min, rate, and quantile produce correct results grouped by specified label dimensions"
      - "rate() correctly handles counter resets by detecting decreases and adjusting the calculation"
      - "High-cardinality label queries are bounded: queries that would scan more than a configurable number of series (default 50,000) are rejected with an error"
    pitfalls:
      - "Not implementing downsampling: querying 1 year of 15-second resolution data is O(millions) of points per series—downsampling is mandatory for historical views"
      - "Gorilla compression ratio degrades with irregular scrape intervals—document expected compression ratios for regular vs irregular data"
      - "Concurrent writes to the same chunk corrupt data without proper locking—use per-series or per-chunk locks"
      - "rate() across counter resets: if the counter resets mid-window, naive (last-first)/duration gives a negative rate—detect resets and adjust"
      - "Downsampled data loses precision: p99 cannot be accurately computed from 5-minute averages—store raw histograms for percentile queries"
    concepts:
      - Gorilla time-series compression
      - Downsampling and rollup strategies
      - Query planning (raw vs downsampled)
      - Counter reset detection
    skills:
      - Compression algorithm implementation
      - Retention and downsampling pipeline
      - Query engine design
      - Memory-efficient data structures
    deliverables:
      - "Time-series storage engine with Gorilla-style compression achieving <2 bytes/sample"
      - "Retention manager deleting raw data past TTL"
      - "Downsampling pipeline producing 5-minute rollups with min/max/avg/count/sum"
      - "Query engine supporting metric selection, label filtering, range queries, and aggregation functions"
      - "Counter reset detection in rate() calculations"
    estimated_hours: "8-12"

  - id: metrics-dashboard-m3
    name: Visualization Dashboard with Template Variables
    description: >-
      Build a web dashboard for visualizing metrics with template variables,
      auto-refresh, and configurable time ranges.
    acceptance_criteria:
      - "Line charts display time-series data with proper axis labels, legends, and hover tooltips showing exact values"
      - "Template variables (e.g., $cluster, $service) are defined per dashboard and used in panel queries; changing a variable re-queries all panels"
      - "Dashboard layout and panel configuration is saved and loaded from persistent JSON schema; dashboards can be exported/imported"
      - "Auto-refresh updates all panels at configurable intervals (5s, 15s, 30s, 1m, 5m) without full page reload"
      - "Time range selector supports relative ranges (last 1h, 6h, 24h, 7d) and absolute date-time ranges"
      - "Missing data points in sparse time series are handled: charts show gaps rather than interpolating false data"
      - "Chart rendering remains responsive with up to 10,000 data points per panel; downsampling is applied for display if needed"
    pitfalls:
      - "Interpolating missing data points: connecting the dots across gaps in sparse data misleads operators—show gaps explicitly"
      - "Rendering 100,000+ points in the browser causes UI freeze—apply server-side downsampling for display resolution"
      - "Time zone issues: storing timestamps in local time causes confusion across teams—always use UTC internally, convert for display"
      - "Template variable queries that return thousands of values make the dropdown unusable—limit template variable cardinality"
    concepts:
      - Data visualization best practices
      - Template variable substitution in queries
      - Real-time dashboard updates
      - Sparse time-series rendering
    skills:
      - Frontend charting (Chart.js, D3, or similar)
      - WebSocket or polling for live updates
      - JSON schema design for dashboard persistence
      - Time zone handling
    deliverables:
      - "Dashboard configuration system with panels, layout, queries, and template variables in JSON schema"
      - "Chart renderer for line charts with proper axis labels, legends, and gap handling"
      - "Template variable system with dropdown selectors that re-query all panels on change"
      - "Auto-refresh mechanism with configurable intervals"
      - "Time range selector with relative and absolute range support"
    estimated_hours: "7-10"

  - id: metrics-dashboard-m4
    name: Alerting System with State Machine
    description: >-
      Implement alerting rules with state machine transitions, deduplication,
      and notification routing.
    acceptance_criteria:
      - "Alert rules define a metric query, comparison operator, threshold value, for-duration, and evaluation interval"
      - "Alert states transition correctly: inactive -> pending (condition met but for-duration not elapsed) -> firing (for-duration elapsed) -> resolved (condition no longer met)"
      - "Firing alerts send a notification on first transition to firing; repeated evaluations while firing do NOT send duplicate notifications unless repeat_interval has elapsed"
      - "Resolved notifications are sent when a firing alert transitions to resolved"
      - "Notification channels (email, Slack webhook, generic webhook) are configurable per alert rule with customizable message templates"
      - "Alert silencing suppresses notifications for alerts matching specified label matchers during a configurable time window"
      - "Alert evaluation state is persisted: a process restart does not reset pending durations or lose firing state"
    pitfalls:
      - "Alert fatigue: too many alerts with low thresholds—require for-duration to filter transient spikes"
      - "Flapping alerts: metric oscillating around threshold causes rapid firing/resolved cycles—implement hysteresis (different thresholds for firing and resolving)"
      - "Missing resolved notifications: operators don't know the incident is over—always send resolved notifications"
      - "State loss on restart: pending durations reset, causing delayed re-firing—persist alert state"
      - "Deduplication failure: sending a notification every evaluation interval (e.g., every 15s) floods channels"
    concepts:
      - Alert state machine (inactive/pending/firing/resolved)
      - Notification deduplication and repeat intervals
      - Alert silencing
      - State persistence
    skills:
      - State machine implementation
      - Notification delivery systems
      - Template rendering for alert messages
      - Persistent state management
    deliverables:
      - "Alert rule definition system with metric query, threshold, for-duration, and evaluation interval"
      - "Alert state machine with correct transitions and persistence across restarts"
      - "Notification dispatcher with deduplication, repeat_interval enforcement, and resolved notifications"
      - "Notification channel integrations for email, Slack, and webhook with customizable templates"
      - "Alert silencing mechanism suppressing notifications for matching labels during maintenance windows"
    estimated_hours: "8-12"

```
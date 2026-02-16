# AUDIT & FIX: distributed-tracing

## CRITIQUE
- The project is generally well-structured and technically sound. The milestones follow a logical progression: context propagation -> span recording -> collection/storage -> query/visualization.
- M1 AC 'Generate globally unique trace and span IDs' doesn't specify the format. W3C requires 16-byte (128-bit) trace-id and 8-byte (64-bit) span-id in hexadecimal. Should be explicit.
- M1 deliverable says 'Baggage items for cross-service key-value metadata propagation' but no AC tests baggage propagation. This is a gap.
- M2 AC 'Start and end spans with precise timing' — doesn't define precision. The essence says 'nanosecond-precision' but the AC doesn't enforce it.
- M2 has no AC for batch export, which is mentioned in concepts. The span export mechanism is crucial — are spans exported synchronously (blocking) or asynchronously (batched)?
- M3 AC mentions both 'Head-based and tail-based sampling' but tail-based sampling is extremely complex (requires holding all spans of a trace in memory until a decision can be made). This is underspecified.
- M3 doesn't specify a concrete storage backend. 'Time-series database' is vague — could be ClickHouse, Cassandra, Elasticsearch, or even SQLite. The AC should be storage-agnostic but specify the query patterns that must be efficient.
- M4 AC 'Service dependency graph shows directed edges' — good, but no specification for how to handle missing spans (partial traces).
- M4 doesn't specify how latency percentiles are computed (approximate vs exact, time window, sample size).
- No mention of span limits (max attributes, max events per span) which is important for production systems.
- No mention of trace/span relationship beyond parent-child (e.g., links for batch operations).
- The estimated hours sum (40-58) is reasonable for the scope.

## FIXED YAML
```yaml
id: distributed-tracing
name: Distributed Tracing System
description: >-
  Build a distributed tracing system to track requests across microservices
  with context propagation, span recording, collection, storage, and
  visualization.
difficulty: advanced
estimated_hours: 55
essence: >-
  Causal operation correlation through hierarchical span recording with
  high-precision timing, persistent trace identifier propagation across service
  boundaries via W3C Trace Context headers, and efficient storage/retrieval of
  trace data for reconstructing distributed request flows and diagnosing
  latency.
why_important: >-
  Building a tracing system teaches the full observability stack from
  instrumentation to storage to visualization, critical skills for debugging
  production microservices where requests span dozens of services and failures
  cascade unpredictably.
learning_outcomes:
  - Implement W3C Trace Context propagation across service boundaries
  - Design hierarchical span data structures with parent-child relationships and timing
  - Build a high-throughput collector service that batches and persists trace data
  - Implement trace storage with efficient query patterns for trace ID lookup and time-range searches
  - Design sampling strategies (head-based and tail-based) for volume reduction
  - Build timeline visualization reconstructing request flows from span data
  - Implement baggage propagation for cross-cutting concerns
skills:
  - Context Propagation
  - Span Instrumentation
  - High-Throughput Data Ingestion
  - Time-Series Storage
  - Sampling Strategies
  - Trace Visualization
  - Microservices Observability
tags:
  - advanced
  - context-propagation
  - distributed
  - sampling
  - spans
  - observability
architecture_doc: architecture-docs/distributed-tracing/index.md
languages:
  recommended:
    - Go
    - Java
  also_possible:
    - Rust
    - Python
resources:
  - name: OpenTelemetry
    url: https://opentelemetry.io/docs/
    type: documentation
  - name: Jaeger Architecture
    url: https://www.jaegertracing.io/docs/architecture/
    type: documentation
  - name: W3C Trace Context Specification
    url: https://www.w3.org/TR/trace-context/
    type: documentation
prerequisites:
  - type: skill
    name: Microservices architecture
  - type: skill
    name: HTTP and gRPC networking
  - type: skill
    name: Data storage and indexing
milestones:
  - id: distributed-tracing-m1
    name: Trace Context & Propagation
    description: >-
      Implement trace and span ID generation, W3C Trace Context propagation
      across HTTP and gRPC boundaries, and baggage propagation for cross-cutting
      metadata.
    estimated_hours: 12
    concepts:
      - "W3C Trace Context: traceparent header format (version-trace_id-span_id-trace_flags)"
      - "Trace ID: 16-byte (128-bit) random identifier encoded as 32 hex chars"
      - "Span ID: 8-byte (64-bit) random identifier encoded as 16 hex chars"
      - "Context propagation: inject on outgoing requests, extract on incoming requests"
      - "Baggage: key-value pairs propagated across all downstream services"
    skills:
      - W3C Trace Context header parsing and generation
      - Cryptographically random ID generation
      - HTTP and gRPC interceptor/middleware patterns
      - Async context propagation
    acceptance_criteria:
      - "Trace IDs are generated as 128-bit (16-byte) cryptographically random values encoded as 32 lowercase hex characters; span IDs are 64-bit (8-byte) encoded as 16 hex chars"
      - "Outgoing HTTP requests inject a valid W3C traceparent header (format: 00-{trace_id}-{span_id}-{trace_flags}) containing the current trace and span context"
      - "Incoming HTTP requests extract the traceparent header; if valid, the trace continues with the received trace_id; if absent or malformed, a new trace_id is generated"
      - "gRPC metadata propagation injects and extracts trace context from gRPC metadata keys, enabling tracing across gRPC service calls"
      - "Parent-child span relationships are correctly established: a span created within the scope of another span records the outer span's ID as its parent_span_id"
      - "Baggage items (arbitrary key-value pairs) are propagated to all downstream services via the W3C tracestate or baggage header and are accessible in each service's span context"
      - "Malformed traceparent headers (wrong length, invalid hex, unsupported version) are handled gracefully by starting a new trace rather than crashing"
    pitfalls:
      - Generating a new trace ID when an existing valid one should be continued
      - Not propagating trace context into async operations (background jobs, message queues)
      - Invalid traceparent parsing causing crashes instead of graceful fallback
      - Baggage items with unbounded size consuming excessive header bandwidth
      - Using sequential IDs instead of random, leaking information about trace volume
    deliverables:
      - Trace ID generator (128-bit random, 32 hex chars)
      - Span ID generator (64-bit random, 16 hex chars)
      - W3C traceparent header injector and extractor
      - gRPC metadata trace context injector and extractor
      - Parent-child span relationship tracking
      - Baggage propagation via W3C baggage header
      - Graceful handling of malformed trace context

  - id: distributed-tracing-m2
    name: Span Recording & Export
    description: >-
      Record span data with precise timing, attributes, events, errors, and
      asynchronous batch export to the collector.
    estimated_hours: 12
    concepts:
      - "Span lifecycle: start (record timestamp) -> add attributes/events -> end (record timestamp, compute duration)"
      - "Span attributes: key-value pairs for filtering (http.method, http.status_code, db.system)"
      - "Span events: timestamped log entries within a span's lifetime"
      - "Batch export: buffer completed spans in memory, flush to collector periodically or when buffer is full"
      - "Span limits: maximum number of attributes, events, and links per span"
    skills:
      - Span lifecycle management
      - High-precision timestamp recording
      - Non-blocking asynchronous export
      - Memory-bounded buffer management
    acceptance_criteria:
      - "Spans record start and end timestamps with at least microsecond precision; duration is computed as end_time - start_time"
      - "Spans support string, integer, float, and boolean attribute values; a configurable limit (default 128) caps the number of attributes per span"
      - "Span events are recorded with a name, timestamp, and optional attributes; a configurable limit (default 128) caps events per span"
      - "Errors are recorded on spans with status=ERROR, exception type, message, and stack trace as attributes"
      - "Span status is set to OK on successful completion and ERROR on failure, with an optional status message"
      - "Completed spans are buffered in memory and exported asynchronously in batches (configurable batch size default 512, flush interval default 5 seconds) without blocking the calling thread"
      - "Export failures are retried with exponential backoff up to 3 attempts; permanently failed spans are dropped and counted in a dropped_spans metric"
      - "Calling span.End() more than once is a no-op; forgetting to call End() is detected and warned after a configurable timeout (default 5 minutes)"
    pitfalls:
      - Span End() never called causing memory leak from unbounded span accumulation
      - Too many attributes or events per span causing large payloads and high storage cost
      - High-cardinality attribute values (user IDs as attribute values on every span) causing storage explosion
      - Blocking on export channel when collector is slow, causing request latency increase
      - Losing spans silently when export buffer overflows without any metric or log
    deliverables:
      - Span start and end recording with microsecond-precision timestamps
      - Span attribute storage with configurable limit
      - Span event recording with timestamp and attributes
      - Error recording with exception details and stack trace
      - Span status (OK, ERROR) with status message
      - Asynchronous batch exporter with configurable batch size and flush interval
      - Export retry with exponential backoff and drop counting
      - Unclosed span detection with warning

  - id: distributed-tracing-m3
    name: Collector & Storage
    description: >-
      Build a collector service that receives spans via HTTP/gRPC, applies
      sampling, enriches span data, and persists to a queryable storage
      backend.
    estimated_hours: 16
    concepts:
      - "Collector pipeline: receive -> sample -> enrich -> store"
      - "Head-based sampling: decide at trace start whether to sample (probabilistic)"
      - "Tail-based sampling: decide after seeing all spans (keep errors, slow traces) — requires holding trace in memory"
      - "Storage indexing: by trace_id (primary), by service+operation+time (search), by duration (latency queries)"
      - Clock skew detection and adjustment for cross-service span timing
    skills:
      - High-throughput HTTP/gRPC ingestion
      - Probabilistic and deterministic sampling
      - Storage backend design for time-series trace data
      - Batch write optimization
    acceptance_criteria:
      - "Collector exposes HTTP POST /v1/traces and gRPC SpanExporter.Export endpoints accepting batches of spans in JSON or protobuf format"
      - "Head-based probabilistic sampling is configurable (e.g., sample 10% of traces) and is deterministic based on trace_id so all spans of a sampled trace are kept"
      - "Tail-based sampling holds trace spans in memory for a configurable window (default 30 seconds) and retains complete traces that contain error spans or exceed a latency threshold"
      - "Span enrichment adds service metadata (service name, version, host) from collector configuration before storage"
      - "Storage supports efficient queries by: (1) trace_id lookup returning all spans in <100ms, (2) service + operation + time range search, (3) duration-based filtering for slow trace discovery"
      - "Batch writes to storage aggregate spans and flush at configurable intervals to reduce write amplification"
      - "Collector handles ingestion spikes gracefully with bounded in-memory queue; backpressure returns 429 to exporters when queue is full"
    pitfalls:
      - Tail-based sampling holding too many traces in memory causing OOM under high load
      - Head-based sampling losing important error traces (should always sample errors)
      - Clock skew between services making span timing look invalid (parent shorter than child)
      - Hot partition in storage when sharding by trace_id (popular traces)
      - Not batching storage writes causing excessive I/O under high ingestion rate
    deliverables:
      - Span ingestion API (HTTP and gRPC endpoints)
      - Head-based probabilistic sampler (configurable rate, deterministic by trace_id)
      - Tail-based sampler with in-memory trace buffer and error/latency-based retention
      - Span enrichment pipeline adding service metadata
      - Storage backend with indexes for trace_id, service+operation+time, and duration
      - Batch write processor for efficient storage
      - Backpressure mechanism returning 429 when ingestion queue is full

  - id: distributed-tracing-m4
    name: Query & Visualization
    description: >-
      Build query APIs and visualization for trace search, timeline display,
      service dependency graphs, and latency analysis.
    estimated_hours: 15
    concepts:
      - "Trace reconstruction: assemble all spans of a trace into parent-child tree"
      - "Service dependency graph: infer edges from spans where service A calls service B"
      - "Latency percentiles: p50, p95, p99 computed from span durations over time windows"
      - "Partial traces: handle missing spans gracefully (show gaps in timeline)"
    skills:
      - Time-range indexed querying
      - Tree reconstruction from flat span list
      - Service dependency graph construction
      - Percentile computation
    acceptance_criteria:
      - "Trace search API supports filtering by service name, operation name, tag key-value pairs, min/max duration, and time range; all queries MUST include a time range to prevent unbounded scans"
      - "Trace detail API returns all spans for a given trace_id assembled into a parent-child tree; orphan spans (missing parent) are shown as root-level entries with a visual indicator"
      - "Trace timeline visualization displays spans as horizontal bars positioned by start time with width proportional to duration; parent-child hierarchy is shown via indentation or nesting"
      - "Service dependency graph API returns directed edges (caller_service -> callee_service) with request count and average latency, derived from observed span data over a configurable time window"
      - "Latency percentile API computes p50, p95, and p99 durations for a given service + operation over a time window; computation handles at least 100K spans efficiently"
      - "Search results are paginated (max 100 traces per page) and sorted by timestamp descending by default"
    pitfalls:
      - Query without time range scanning entire storage causing timeout or OOM
      - Deep trace hierarchies (>100 levels) causing stack overflow in recursive tree building (use iterative approach)
      - Timezone display issues in UI showing timestamps in wrong timezone
      - Missing spans creating disconnected subtrees in the visualization
      - Latency percentile computation being inaccurate on small sample sizes
    deliverables:
      - Trace search API with service, operation, tag, duration, and time range filters
      - Trace detail API assembling spans into parent-child tree with orphan handling
      - Trace timeline visualization with hierarchical span display
      - Service dependency graph API with edge weights (count, avg latency)
      - Latency percentile analysis (p50, p95, p99) per service and operation
      - Paginated search results with configurable sort order

```
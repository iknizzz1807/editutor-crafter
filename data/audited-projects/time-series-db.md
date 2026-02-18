# AUDIT & FIX: time-series-db

## CRITIQUE
- **Logical Gap (Confirmed - Tag Index):** The original mentions 'Tag-based filtering and indexing using inverted index' as a deliverable in M3 (Query Engine) but has no dedicated milestone for building the tag/label inverted index. Without this, every tag filter query requires a full scan of all series. This is architecturally critical—TSDB queries are almost always filtered by tags (e.g., 'host=web-01 AND region=us-east'). It must be its own milestone.
- **Technical Inaccuracy (Confirmed - XOR compression):** The original essence says 'XOR for float values' which is correct, but M1 deliverables use 'Gorilla float compression' as if it applies to all values. Gorilla/XOR compression is specific to IEEE 754 floating-point values. Integer metrics, boolean values, and string annotations require different encoding. The project should distinguish between timestamp compression (delta-of-delta), float compression (XOR/Gorilla), and integer compression (delta + varint).
- **Estimated Hours (180):** This is extremely high for 5 milestones. Either the scope is too large per milestone or the milestones should be split. With 6 milestones at more realistic scope, ~120 hours is achievable.
- **M1 Storage Engine:** Tries to do too much—TSM tree, delta-of-delta, Gorilla, dictionary encoding, block index, and mmap all in one milestone at 36 hours. This is a mini-project within a project.
- **M2 Write Path:** 'Lock-free ring buffers for concurrent writes' is mentioned as a concept but never validated in any AC. The AC just says 'buffer writes in memory'.
- **M5 Query Language:** Lists 6 deliverables including SQL-like language, Flux-style pipeline, HTTP write API, query API, Prometheus compatibility, and Grafana compatibility. This is absurdly overscoped.
- **Missing: Series Cardinality Management:** High cardinality series (millions of unique label combinations) is the #1 operational problem in TSDBs. No milestone addresses detection or mitigation.

## FIXED YAML
```yaml
id: time-series-db
name: Time-Series Database
description: >-
  Optimized for time-stamped data with specialized compression, inverted
  tag index, write-ahead logging, time-range queries, aggregations,
  and retention management.
difficulty: advanced
estimated_hours: 130
essence: >-
  Time-partitioned columnar block storage using delta-of-delta compression for
  timestamps and XOR (Gorilla) compression for float values, combined with an
  inverted index for tag-based series lookup, write-ahead logging for ingestion
  durability, and time-range scan optimization with pushdown aggregation and
  block-level min/max pruning.
why_important: >-
  Building this teaches production database engineering including write-optimized
  storage, specialized compression algorithms, inverted index design, and query
  optimization—skills directly applicable to understanding systems like
  Prometheus, InfluxDB, and TimescaleDB.
learning_outcomes:
  - Implement columnar block storage with time-partitioned data organization
  - Build delta-of-delta timestamp compression and Gorilla XOR float compression
  - Design an inverted index for tag-based series lookup and filtering
  - Implement a high-throughput write path with WAL and in-memory buffering
  - Build time-range query execution with block pruning and pushdown aggregation
  - Design retention policies with automatic expiration and background compaction
skills:
  - Time-series data modeling
  - Columnar storage
  - Delta-of-delta compression
  - XOR/Gorilla float compression
  - Inverted index for tags
  - Write-ahead logging
  - Retention and compaction
  - Query optimization
tags:
  - advanced
  - aggregations
  - compression
  - databases
  - downsampling
  - retention
  - time-series
architecture_doc: architecture-docs/time-series-db/index.md
languages:
  recommended:
    - Go
    - Rust
    - C++
  also_possible:
    - Java
resources:
  - name: "Gorilla: Facebook's TSDB Paper"
    url: https://www.vldb.org/pvldb/vol8/p1816-teller.pdf
    type: paper
  - name: Write a TSDB Engine from Scratch
    url: https://nakabonne.dev/posts/write-tsdb-from-scratch/
    type: tutorial
  - name: InfluxDB Documentation
    url: https://docs.influxdata.com/
    type: documentation
  - name: Prometheus TSDB Design
    url: https://fabxc.org/tsdb/
    type: article
  - name: LSM Trees Explained
    url: https://medium.com/@dwivedi.ankit21/lsm-trees-the-go-to-data-structure-for-databases-search-engines-and-more-c3a48fa469d2
    type: article
prerequisites:
  - type: skill
    name: Database engine basics (B-tree or LSM tree)
  - type: skill
    name: Binary encoding and compression
  - type: skill
    name: File I/O and memory-mapped files
milestones:
  - id: time-series-db-m1
    name: Storage Engine & Compression
    description: >-
      Build time-partitioned columnar block storage with specialized compression
      for timestamps and float values.
    acceptance_criteria:
      - Data is organized into time-partitioned blocks, each covering a fixed time range (e.g., 2 hours)
      - Within each block, timestamps and values are stored in separate columnar arrays
      - Delta-of-delta encoding compresses timestamps: for regular-interval data (e.g., 15s scrape), achieves <1 bit per timestamp on average
      - XOR (Gorilla) compression encodes float values by XOR-ing consecutive values and storing only the changed bits, achieving <2 bits per value for slowly-changing metrics
      - Integer values use delta + variable-byte encoding (distinct from Gorilla which is float-specific)
      - Block index stores min/max timestamp per block for efficient time-range pruning
      - Storage size for 1M data points of a single float metric with 15s interval is under 2MB (measured)
      - Blocks are read-only once sealed; only the active (current time window) block accepts writes
    pitfalls:
      - Applying Gorilla XOR compression to non-float data (integers, strings) produces poor compression—use type-specific encoding
      - Delta-of-delta assumes mostly regular timestamps; highly irregular timestamps compress poorly—measure and document
      - Block time range too large wastes memory; too small creates too many files—tune based on write rate
      - Not aligning block sizes with filesystem page sizes causes read amplification
      - Block sealing must be atomic—a partially sealed block is corrupt
    concepts:
      - Time-partitioned blocks organize data by time range for efficient pruning
      - Columnar layout separates timestamps from values for compression and vectorized processing
      - Delta-of-delta exploits regularity in timestamp intervals
      - Gorilla XOR compression exploits small differences between consecutive float values
      - Block index with min/max timestamps enables time-range pruning
    skills:
      - Columnar storage design
      - Delta-of-delta encoding
      - Gorilla XOR float compression
      - Block management
    deliverables:
      - Time-partitioned block storage with configurable time range per block
      - Columnar arrays for timestamps and values within each block
      - Delta-of-delta timestamp compression
      - Gorilla XOR float compression
      - Delta + varint integer compression
      - Block index with min/max timestamps
      - Compression ratio benchmark for 1M float data points
    estimated_hours: 24

  - id: time-series-db-m2
    name: Tag Index (Inverted Index for Labels)
    description: >-
      Build an inverted index mapping tag key-value pairs to series IDs,
      enabling efficient tag-based series lookup and filtering.
    acceptance_criteria:
      - Each time series is identified by a unique series ID derived from its sorted set of tag key-value pairs
      - Inverted index maps each tag pair (e.g., host=web-01) to a sorted posting list of series IDs
      - Multi-tag queries (e.g., host=web-01 AND region=us-east) are resolved by intersecting posting lists
      - Tag value lookup (e.g., all values for tag 'host') returns distinct values in O(k) where k is the number of values
      - Series cardinality tracking reports the total number of unique series, with a warning threshold for high cardinality (configurable, default 1M series)
      - Index supports label matching with equality (=), not-equal (!=), and regex (~=) operators
      - Index is persisted to disk and loaded on startup; new series are incrementally added without full rebuild
    pitfalls:
      - High cardinality tags (e.g., user_id as a tag) cause the inverted index to explode in size—enforce cardinality limits and warn users
      - Regex matching on tag values requires full scan of the tag value set—optimize common patterns (prefix, suffix) and fall back to regex for complex patterns
      - Posting list intersection must use the smallest list first (sorted by size) for efficiency
      - Tag key ordering in series identification must be canonical (sorted) to prevent duplicate series for the same label set
      - Index corruption on crash must be recoverable—rebuild from WAL or block metadata
    concepts:
      - Inverted index maps tag key-value pairs to sets of series IDs
      - Posting list intersection resolves multi-tag queries efficiently
      - Series cardinality is the number of unique tag combinations
      - High cardinality causes memory and query performance problems
    skills:
      - Inverted index design
      - Posting list operations (intersection, union)
      - Cardinality monitoring
      - Regex matching optimization
    deliverables:
      - Series ID generation from canonical sorted tag set
      - Inverted index mapping tag pairs to sorted posting lists
      - Posting list intersection for multi-tag queries
      - Tag value enumeration per key
      - Regex tag matching support
      - Series cardinality tracking and high-cardinality warning
      - Index persistence and incremental update
    estimated_hours: 20

  - id: time-series-db-m3
    name: Write Path
    description: >-
      High-throughput ingestion with WAL-backed durability and in-memory
      buffering.
    acceptance_criteria:
      - Every write is appended to a write-ahead log (WAL) and fsync'd before being acknowledged to the client
      - In-memory buffer (memtable/head block) accumulates writes for the current time window
      - Memtable flushes to a sealed on-disk block when the time window closes or memory threshold is reached
      - WAL segments are rotated and old segments are deleted after their data has been flushed to blocks
      - Out-of-order writes (late-arriving data) are accepted into the memtable if they fall within the current time window; data outside the window is rejected with an error or buffered for compaction
      - Write throughput sustains at least 100,000 data points per second on a single core (measured via benchmark)
      - Backpressure mechanism rejects or throttles writes when the memtable is near capacity, returning a clear error to the client
    pitfalls:
      - Not fsyncing the WAL before acknowledging writes risks data loss on crash—this is a hard durability requirement
      - WAL segments must be rotated and cleaned up after flush; unbounded WAL growth exhausts disk
      - Blocking writes when the memtable is full (instead of signaling backpressure) causes cascading client timeouts
      - Out-of-order data outside the current time window is common in distributed systems (clock skew, delayed delivery)—rejecting it loses data; accepting it complicates compaction
      - Memory usage must be bounded; unbounded buffering from high-cardinality series causes OOM
    concepts:
      - Write-ahead log ensures durability before acknowledgment
      - Memtable buffers writes in memory for batch flush to disk
      - WAL rotation and cleanup after flush prevents disk exhaustion
      - Backpressure signals overload to clients before OOM
    skills:
      - WAL implementation
      - Memory management
      - Write batching
      - Backpressure design
    deliverables:
      - Write-ahead log with fsync before acknowledgment
      - In-memory buffer (memtable) for current time window
      - Memtable flush to sealed on-disk block
      - WAL rotation and cleanup after flush
      - Out-of-order write handling within current window
      - Backpressure mechanism for write throttling
      - Write throughput benchmark (target: 100K points/sec)
    estimated_hours: 22

  - id: time-series-db-m4
    name: Query Engine
    description: >-
      Efficient time-range queries with block pruning, tag filtering,
      and aggregation functions.
    acceptance_criteria:
      - Time-range query specifies start and end timestamps; only blocks overlapping the range are read (block pruning via min/max index)
      - Tag filter predicates are resolved via the inverted index BEFORE scanning blocks—only matching series are scanned
      - Aggregation functions (sum, avg, min, max, count) produce correct results over the selected time range and series
      - Windowed aggregation (GROUP BY time bucket) partitions results into fixed-width time intervals (e.g., 5m, 1h)
      - Downsampling returns lower-resolution data by aggregating raw points into larger time buckets
      - Query over 1M data points with a tag filter on 1 of 1000 series completes in under 100ms (measured)
      - Gaps in time-series data (missing points) are handled correctly by aggregation functions (e.g., avg skips gaps, count excludes gaps)
    pitfalls:
      - Not pushing tag filters to the index layer forces scanning all series and filtering in the query layer—orders of magnitude slower
      - Loading entire time ranges into memory before filtering causes OOM for large queries—stream results
      - AVG aggregation must track both sum and count; computing it incrementally avoids a second pass
      - Gaps in data must not be interpolated silently—report gaps or fill with explicit null values, never fabricate data
      - Windowed aggregation boundary alignment (e.g., does the 5m window start at : 00 or at query start?) must be documented
    concepts:
      - Block pruning skips blocks outside the query time range using min/max index
      - Tag filter pushdown resolves matching series via inverted index before block scan
      - Streaming query execution processes blocks sequentially without buffering entire result
      - Windowed aggregation partitions time range into fixed-width buckets
    skills:
      - Query planning with pushdown
      - Aggregation computation
      - Streaming result processing
      - Time-window alignment
    deliverables:
      - Time-range block pruning using block min/max index
      - Tag filter resolution via inverted index before block scan
      - Aggregation functions (sum, avg, min, max, count)
      - Windowed aggregation with configurable time bucket width
      - Streaming query execution for large result sets
      - Query latency benchmark with tag filter on large dataset
    estimated_hours: 24

  - id: time-series-db-m5
    name: Retention & Compaction
    description: >-
      Automatic data lifecycle management with time-based retention,
      block compaction, and optional downsampling.
    acceptance_criteria:
      - TTL-based retention automatically deletes blocks whose max timestamp is older than the configured retention period
      - Retention enforcement runs as a background task at configurable intervals (default 1 hour)
      - Block compaction merges small or overlapping blocks into larger optimized blocks, improving query performance
      - Compaction runs in the background without blocking writes or queries (verified by measuring write/query latency during compaction)
      - Continuous downsampling pre-aggregates old data into lower-resolution blocks (e.g., 1m raw → 1h downsampled), configurable per measurement
      - Disk space reclamation is verified: after retention deletes blocks, disk usage decreases by the expected amount
      - Compaction and retention are throttled to limit I/O impact during peak query hours
    pitfalls:
      - Running compaction during peak query load degrades latency—schedule during off-peak or throttle I/O
      - Downsampling must preserve statistical properties; storing only averages loses min/max range information—store multiple aggregates
      - Deleting blocks before compaction of overlapping data completes causes data loss
      - Disk space monitoring is essential; retention enforcement that runs too infrequently lets disk fill up
      - Compacting very large blocks requires proportional temporary disk space—monitor free space before starting
    concepts:
      - TTL-based retention deletes data blocks older than configured age
      - Background compaction merges blocks for query efficiency
      - Downsampling reduces resolution while preserving aggregate statistics
      - I/O throttling limits background task impact on foreground operations
    skills:
      - Retention policy enforcement
      - Background compaction design
      - Downsampling algorithms
      - I/O scheduling and throttling
    deliverables:
      - TTL-based retention with configurable retention period per measurement
      - Background retention task with configurable interval
      - Block compaction merging small/overlapping blocks
      - Continuous downsampling pre-aggregating old data
      - I/O throttling for background tasks
      - Disk space reclamation verification
    estimated_hours: 20

  - id: time-series-db-m6
    name: Query Language & API
    description: >-
      Implement a query language for time-series queries and expose write/query
      endpoints via HTTP API.
    acceptance_criteria:
      - Query language supports SELECT with time range (WHERE time > X AND time < Y), tag filters, and aggregation functions
      - GROUP BY time(interval) syntax supports tumbling time windows with configurable bucket width
      - FILL clause specifies gap-filling strategy (null, previous, linear, or explicit value)
      - HTTP write API accepts data points in a line protocol format (measurement,tag1=v1,tag2=v2 field1=value1 timestamp)
      - HTTP query API accepts query strings and returns results in JSON format with column headers and typed values
      - Query validation rejects unbounded queries (missing time range) with a clear error message
      - Rate and derivative functions compute per-second rates from counter metrics (handling counter resets)
    pitfalls:
      - Allowing unbounded queries (no time range predicate) can scan the entire database and OOM—reject or limit them
      - Timezone handling: store and query in UTC; convert to user timezone only at display time
      - Rate function must detect counter resets (value decreasing) and handle them by assuming a wrap-around
      - Line protocol parsing must handle escaped characters in tag values and field values
      - JSON result format must distinguish between null (gap) and 0 (actual value)
    concepts:
      - SQL-like query language with time-series extensions (time grouping, fill, rate)
      - Line protocol is a compact text format for time-series data ingestion
      - Rate and derivative functions convert cumulative counters to per-second rates
      - Gap-filling strategies handle missing data points in query results
    skills:
      - Query language parsing
      - HTTP API design
      - Line protocol parsing
      - Rate computation
    deliverables:
      - Query language parser with time range, tag filter, aggregation, and GROUP BY time support
      - FILL clause for gap-filling strategies
      - Rate and derivative functions for counter metrics
      - HTTP write API accepting line protocol format
      - HTTP query API returning JSON results
      - Query validation rejecting unbounded queries
    estimated_hours: 20
```
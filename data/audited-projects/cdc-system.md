# AUDIT & FIX: cdc-system

## CRITIQUE
- **Critical: No Initial Snapshot Milestone**: CDC systems cannot start from a transaction log alone if the database already has data. You need a consistent point-in-time snapshot of existing data before you can begin streaming changes. Debezium, for example, performs a snapshot phase first. This is a fundamental architectural requirement that the project completely omits.
- **Exactly-Once Claimed but Not Delivered**: The essence claims 'exactly-once delivery semantics' but Milestone 2 only implements 'at-least-once' (resume from offset). True exactly-once requires either: (a) idempotent consumers using primary key deduplication, or (b) transactional outbox pattern, or (c) Kafka transactions with read-committed consumers. The ACs must be corrected to either achieve exactly-once or honestly describe at-least-once with idempotent consumers.
- **Milestone Hour Estimates Uniformly Wrong**: All three milestones are 16.5 hours. Log parsing is harder than schema evolution. The distribution should be weighted toward M1 and M2.
- **Binary Log Format Versioning Not AC'd**: Pitfall mentions 'binary log formats change between versions' but no AC requires handling version detection or format negotiation.
- **DDL Handling Conflated with Schema Evolution**: Milestone 3 mentions DDL events in pitfalls but DDL (ALTER TABLE) detection should be in M1 (log parsing) since DDL events appear in the transaction log, not in the schema registry.
- **No Backfill/Resync Capability**: After initial snapshot, sometimes consumers need to re-snapshot a table (e.g., after a bug). This is related to the initial snapshot omission.
- **Consumer Offset Commit Semantics Unclear**: M2 pitfall says 'offset commit after processing prevents duplicates' — this is misleading. Committing offset after processing prevents data loss (at-least-once); committing before processing prevents duplicates but risks data loss (at-most-once). Neither achieves exactly-once alone.
- **No Mention of Logical Replication Protocol**: For PostgreSQL, CDC typically uses logical replication (pg_logical or pgoutput plugin), not raw WAL parsing. Raw WAL is physical and doesn't contain logical row data. This is a significant technical inaccuracy.

## FIXED YAML
```yaml
id: cdc-system
name: "Change Data Capture"
description: >-
  Build a change data capture system that takes a consistent initial snapshot,
  then streams row-level changes from database transaction logs to downstream
  consumers with ordering guarantees and schema evolution support.
difficulty: expert
estimated_hours: "50-65"
essence: >-
  Consistent initial snapshot capture followed by logical replication log parsing
  (PostgreSQL logical decoding or MySQL binlog) to extract row-level change
  events, with at-least-once delivery and idempotent consumer patterns to achieve
  effectively-once processing semantics, plus distributed offset tracking for
  ordered, resumable replication.
why_important: >-
  Building this teaches you low-level database internals, distributed systems
  delivery guarantee patterns, and event-driven architecture — foundational
  skills for data infrastructure engineering at companies building real-time
  data platforms.
learning_outcomes:
  - Implement consistent initial snapshot capture with concurrent change tracking
  - Parse logical replication streams (PostgreSQL logical decoding / MySQL binlog) to extract change events
  - Implement at-least-once delivery with idempotent consumer patterns for effectively-once semantics
  - Design offset tracking and checkpointing for consumer failure recovery
  - Build schema registry integration for backward and forward compatibility
  - Handle DDL changes (ALTER TABLE) detected in the replication stream
  - Implement partition-level ordering guarantees in distributed event streams
  - Debug replication lag and backpressure in streaming pipelines
skills:
  - Logical Replication Log Parsing
  - Event Streaming Architecture
  - Delivery Guarantee Semantics
  - Schema Evolution
  - Distributed Systems
  - Idempotency Patterns
  - Offset Management
  - Snapshot Consistency
tags:
  - cdc
  - change-capture
  - database
  - debezium
  - events
  - expert
  - replication
  - streaming
architecture_doc: architecture-docs/cdc-system/index.md
languages:
  recommended:
    - Java
    - Go
    - Python
  also_possible: []
resources:
  - name: Debezium Documentation""
    url: https://debezium.io/documentation/reference/stable/index.html
    type: documentation
  - name: Kafka Delivery Semantics""
    url: https://docs.confluent.io/kafka/design/delivery-semantics.html
    type: documentation
  - name: PostgreSQL Logical Decoding""
    url: https://www.postgresql.org/docs/current/logicaldecoding.html
    type: documentation
  - name: MySQL Binary Log""
    url: https://dev.mysql.com/doc/refman/8.0/en/binary-log.html
    type: documentation
  - name: Change Data Capture Overview""
    url: https://www.confluent.io/learn/change-data-capture/
    type: tutorial
prerequisites:
  - type: project
    id: build-sqlite
  - type: skill
    name: SQL and database administration basics
  - type: skill
    name: Message broker concepts (Kafka or similar)
milestones:
  - id: cdc-system-m1
    name: "Initial Snapshot & Log Position Capture"
    description: >-
      Implement consistent initial snapshot of existing table data with a recorded
      log position, so that subsequent log streaming picks up exactly where the
      snapshot left off without missing or duplicating rows.
    estimated_hours: "12-16"
    concepts:
      - Consistent point-in-time snapshot
      - Transaction isolation for snapshot consistency (REPEATABLE READ or SERIALIZABLE)
      - Log position (LSN/binlog position) capture at snapshot time
      - Chunked snapshot for large tables
      - Snapshot vs streaming mode transition
    skills:
      - Database transaction isolation levels
      - Bulk data export techniques
      - Log position tracking
      - Chunked/paginated data reading
    acceptance_criteria:
      - Snapshot captures all existing rows for configured tables at a consistent point-in-time using a single transaction with REPEATABLE READ isolation (PostgreSQL) or FLUSH TABLES WITH READ LOCK + consistent snapshot (MySQL)
      - Log position (PostgreSQL LSN or MySQL binlog filename+position) is recorded at the start of the snapshot transaction
      - Snapshot data is exported in configurable chunk sizes (default 10,000 rows) to avoid OOM on large tables
      - Each snapshot chunk is published as a batch of INSERT-type change events with a 'snapshot' flag distinguishing them from live changes
      - After snapshot completes, the system transitions to streaming mode starting from the recorded log position with zero gap and zero overlap
      - Snapshot progress is checkpointed per-table so that a crash during snapshot resumes from the last completed chunk, not from the beginning
      - Tables with no primary key are handled with a warning — snapshot works but subsequent CDC ordering guarantees are weaker
      - Re-snapshot of a single table can be triggered manually without affecting other tables' streaming
    pitfalls:
      - Snapshot without a consistent log position creates a gap where changes during the snapshot are lost
      - Long-running snapshot transactions hold locks or consume replication slots; chunk and checkpoint to minimize duration
      - MySQL's FLUSH TABLES WITH READ LOCK requires RELOAD privilege and blocks all writes globally; use LOCK TABLES for per-table locking where possible
      - Tables without primary keys cannot guarantee row-level ordering in subsequent CDC; warn loudly
      - Snapshot of a 1TB table takes hours; without chunked checkpointing, any crash restarts from zero
    deliverables:
      - Snapshot coordinator managing per-table snapshot state and progress
      - Consistent snapshot reader using transaction isolation with log position capture
      - Chunked data export with configurable batch size and checkpoint
      - Snapshot-to-streaming transition logic starting log parsing from captured position
      - Re-snapshot trigger for individual tables

  - id: cdc-system-m2
    name: "Log Parsing & Change Event Streaming"
    description: >-
      Parse database logical replication stream to extract row-level change events
      and publish them to a message broker with ordering guarantees and at-least-once
      delivery. Implement idempotent consumer patterns for effectively-once semantics.
    estimated_hours: "18-22"
    concepts:
      - Logical replication (PostgreSQL pgoutput/wal2json, MySQL binlog row-based)
      - Change event structure (operation type, before/after images, metadata)
      - Partition key selection for ordering guarantees
      - At-least-once delivery with idempotent consumers
      - Consumer offset management and checkpointing
      - Backpressure handling
    skills:
      - Logical replication protocol interaction
      - Event serialization (JSON, Avro)
      - Message broker integration (Kafka)
      - Idempotent upsert patterns
      - Offset/LSN tracking
    acceptance_criteria:
      - Parser connects to PostgreSQL logical replication slot (using pgoutput or wal2json plugin) or MySQL binlog (row-based format) and reads change entries
      - INSERT, UPDATE, and DELETE operations are correctly extracted; UPDATE events include both before-image and after-image row values
      - DDL events (ALTER TABLE, DROP TABLE) are detected in the log stream and published as schema change events
      - Change events are published to Kafka topics partitioned by table name, with partition key set to the row's primary key for per-row ordering
      - Consumer receives every change event at least once even after transient broker or consumer failures (at-least-once delivery)
      - Idempotent consumer pattern is implemented — consumers use primary key + event LSN/offset to deduplicate, achieving effectively-once processing
      - Source connector commits LSN/binlog position to durable storage only after events are confirmed published to the broker
      - Consumer lag is monitored; an alert fires when lag exceeds a configurable threshold (default 10,000 events or 5 minutes)
      - Backpressure is applied — if the broker is unavailable or slow, the log reader pauses rather than buffering unboundedly in memory
      - Transaction boundaries are preserved — all changes within a single database transaction are published atomically (as a batch or with transaction markers)
    pitfalls:
      - PostgreSQL logical decoding requires a replication slot; orphaned slots prevent WAL cleanup and fill disk
      - MySQL binlog must be in ROW format (not STATEMENT or MIXED) for row-level change extraction
      - Committing offset before processing risks data loss (at-most-once); committing after processing risks duplicates (at-least-once); idempotent consumers resolve this
      - Partition by primary key, not just table name — partitioning only by table serializes all changes for large tables
      - Large transactions (e.g., bulk UPDATE of 1M rows) produce huge event batches; implement chunking or streaming within transactions
      - Replication slot on PostgreSQL must be actively consumed; falling behind causes WAL accumulation that can fill disk
    deliverables:
      - Logical replication connector for PostgreSQL (pgoutput/wal2json) or MySQL (binlog)
      - Change event model with operation type, table, primary key, before/after images, LSN, and transaction ID
      - DDL event detector for ALTER TABLE and DROP TABLE
      - Kafka producer with per-primary-key partitioning and at-least-once delivery
      - Idempotent consumer with primary-key + LSN deduplication
      - LSN/offset checkpoint manager with durable persistence
      - Backpressure handler pausing log reading when downstream is slow
      - Consumer lag monitor with configurable alert threshold

  - id: cdc-system-m3
    name: "Schema Evolution & Compatibility"
    description: >-
      Handle schema changes detected in the replication stream without breaking
      consumers. Integrate with a schema registry for versioned schemas and
      compatibility checking.
    estimated_hours: "12-16"
    concepts:
      - Schema versioning and registry (Confluent Schema Registry or custom)
      - Backward, forward, and full compatibility modes
      - Avro or Protobuf schema evolution rules
      - DDL-triggered schema updates
      - Consumer deserialization with schema evolution
    skills:
      - Schema registry API integration
      - Compatibility validation
      - Schema versioning and migration
      - Serialization format design
    acceptance_criteria:
      - Schema registry stores versioned schema for each table; each schema version is assigned a unique version ID
      - When a DDL event (ALTER TABLE) is detected in the replication stream, the new schema is registered in the schema registry
      - Column additions with nullable defaults pass backward compatibility check and are registered as a new version
      - Column removal or type narrowing (e.g., BIGINT→INT) fails forward compatibility validation and produces an error requiring manual resolution
      - Change events include the schema version ID; consumers fetch the correct schema version for deserialization
      - Consumers can deserialize events written with older schema versions using backward-compatible reader schema (e.g., Avro schema evolution)
      - Schema change notification events are published to a dedicated topic so consumers can update their deserialization logic proactively
      - Schema registry is not a single point of failure — schemas are cached locally by producers and consumers with configurable TTL
      - Breaking schema changes (column type change, column removal) trigger an alert and optionally pause the pipeline pending human review
    pitfalls:
      - DDL events require special handling — a schema change mid-transaction means events before and after the DDL use different schemas
      - Schema registry caching without TTL causes consumers to use stale schemas indefinitely
      - Backward compatibility is the most important mode for consumers reading historical data; forward compatibility matters more for producers
      - Type narrowing (BIGINT→INT) can cause data loss during deserialization; always flag as breaking
      - Avro requires default values for new fields in backward-compatible evolution; forgetting defaults breaks deserialization
    deliverables:
      - Schema registry with versioned storage, version ID assignment, and REST API
      - DDL event handler registering new schema versions on ALTER TABLE
      - Compatibility checker validating backward, forward, and full compatibility
      - Schema-aware serializer embedding version ID in change events
      - Consumer-side schema evolution handling with cached schema lookup
      - Breaking change detector with alerting and optional pipeline pause

  - id: cdc-system-m4
    name: "Operational Resilience & Monitoring"
    description: >-
      Implement end-to-end operational capabilities: re-snapshot triggers,
      replication slot management, lag monitoring, and health checks for
      production deployment.
    estimated_hours: "8-11"
    concepts:
      - Replication slot lifecycle management
      - End-to-end latency measurement
      - Health check endpoints
      - Graceful shutdown and restart
    skills:
      - Operational monitoring design
      - Health check implementation
      - Replication slot management
      - Metrics collection and export
    acceptance_criteria:
      - Replication slot is created on startup if absent and dropped on graceful shutdown to prevent WAL accumulation
      - End-to-end latency is measured from database commit timestamp to consumer processing timestamp; metric is exported for monitoring
      - Health check endpoint reports status of database connection, replication stream, broker connection, and schema registry
      - Graceful shutdown flushes in-flight events, commits final offset, and releases replication slot
      - Crash recovery restarts from last checkpointed LSN/offset and verifies consistency with snapshot state
      - Per-table metrics track events_published, events_consumed, errors, and current lag
      - Operational runbook documents common failure scenarios (replication slot full, broker unavailable, schema incompatibility) with resolution steps
    pitfalls:
      - Orphaned replication slots in PostgreSQL prevent WAL cleanup and will fill disk within hours on write-heavy databases
      - End-to-end latency measurement requires synchronized clocks between database and consumer; use database-generated timestamps, not wall clock
      - Health check must distinguish between degraded (high lag but processing) and failed (stopped processing) states
    deliverables:
      - Replication slot lifecycle manager (create, monitor, drop)
      - End-to-end latency metric exporter
      - Health check HTTP endpoint with component-level status
      - Graceful shutdown handler with flush and offset commit
      - Per-table operational metrics dashboard or export
      - Operational runbook for common failure scenarios
```
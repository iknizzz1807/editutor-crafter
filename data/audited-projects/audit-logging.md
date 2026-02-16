# AUDIT & FIX: audit-logging

## CRITIQUE
- The essence claims 'Merkle tree indexing for efficient O(log N) verification proofs' but no milestone implements a Merkle tree. Milestone 2 only implements linear hash chaining, which provides O(N) verification (must recompute the entire chain from the tampered point). This is a significant discrepancy between the stated architecture and the implementation.
- Missing: Log sanitization / PII redaction. The M1 pitfall mentions 'PII in audit logs violates GDPR right to erasure' but provides no acceptance criteria or deliverable for handling this. This is especially problematic for an immutable (append-only) log: you cannot delete PII after writing it. The solution is to prevent PII from entering the log in the first place, or to use tokenization.
- Milestone 1 skills mention 'PII detection and redaction techniques' but no AC requires it.
- Milestone 3 AC says 'sub-second response times' but provides no data volume context. Sub-second on 100 events is trivial; sub-second on 100 million events requires careful index design.
- Milestone 2 AC mentions 'monotonically increasing sequence number' and 'hash chain continuity across segment boundaries' but doesn't specify how the first entry of a new segment references the last entry of the previous segment.
- Milestone 3 deliverable mentions 'PDF format' export but PDF generation is a significant non-trivial effort that distracts from the core audit logging learning. Should be optional or removed.
- No mention of access control on the audit log itself — who can read audit events? Audit logs often contain sensitive operational data.
- The project's connection to log-aggregator prerequisite is unclear. An audit log has different requirements (immutability, legal compliance) than an aggregation pipeline.

## FIXED YAML
```yaml
id: audit-logging
name: Audit Logging System
description: >-
  Build an immutable audit trail with cryptographic hash chaining for
  tamper detection, PII sanitization, efficient querying, and compliance
  export capabilities.
difficulty: intermediate
estimated_hours: "30-40"
essence: >-
  Append-only storage with cryptographic hash chaining (each entry includes
  hash of previous entry) for tamper detection, PII sanitization to prevent
  sensitive data from entering the immutable log, indexed querying for
  efficient time-range and actor-based retrieval, and compliance-ready
  export for SOC2/HIPAA/GDPR audit submissions.
why_important: >-
  Audit logs are legally required for compliance frameworks (SOC 2, HIPAA,
  GDPR, PCI-DSS). Understanding immutable logging, tamper detection, and
  PII handling teaches security engineering principles applicable to any
  production system handling sensitive data.
learning_outcomes:
  - Design immutable append-only log storage with hash chain integrity
  - Implement PII sanitization to prevent sensitive data in immutable logs
  - Build cryptographic hash chains for tamper detection
  - Implement efficient querying with indexed time-range and actor filters
  - Handle retention and archival policies for compliance requirements
  - Design audit event schemas with extensible metadata
  - Build compliance export in standard formats
  - Understand access control requirements for audit data
skills:
  - Append-only Data Structures
  - Hash Chain Cryptography
  - PII Detection and Sanitization
  - Event Schema Design
  - Retention Policy Management
  - Compliance Reporting
  - Immutable Storage Patterns
  - Audit Query Optimization
tags:
  - backend
  - compliance
  - immutable
  - intermediate
  - retention
  - security
architecture_doc: architecture-docs/audit-logging/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible: []
resources:
  - name: "Write-Ahead Log Pattern"
    url: https://martinfowler.com/articles/patterns-of-distributed-systems/write-ahead-log.html
    type: article
  - name: "Efficient Tamper-Evident Logging (Crosby & Wallach)"
    url: https://static.usenix.org/event/sec09/tech/full_papers/crosby.pdf
    type: paper
  - name: "SOC 2 Compliance Requirements"
    url: https://www.venn.com/learn/soc2-compliance/
    type: documentation
prerequisites:
  - type: skill
    name: Database fundamentals (SQL or key-value stores)
  - type: skill
    name: Cryptographic hash functions (SHA-256)
  - type: skill
    name: HTTP API design
milestones:
  - id: audit-logging-m1
    name: "Audit Event Schema and PII Sanitization"
    description: >-
      Design the audit event schema with required fields, extensible
      metadata, and a PII sanitization layer that prevents sensitive
      data from entering the immutable log.
    acceptance_criteria:
      - >-
        Event schema defines required fields: actor (who), action (what),
        resource (on what), timestamp (when, UTC), outcome (success/failure),
        and correlation_id (for distributed tracing).
      - >-
        Custom metadata fields (key-value pairs) can be attached to any
        event for domain-specific context without modifying the core schema.
      - >-
        Request context (client IP address, user-agent, request ID) is
        automatically captured and included in each event.
      - >-
        Events are validated against the schema at creation time. Events
        missing required fields are rejected with a descriptive error.
      - >-
        PII sanitization layer processes all events before storage:
        configurable rules detect and redact or tokenize sensitive fields
        (email addresses, phone numbers, SSNs, etc.).
      - >-
        Sanitization is applied to both core fields and custom metadata,
        using pattern matching (regex) and field-name-based rules.
      - >-
        Sanitized values are replaced with tokens or masked values
        (e.g., 'user@example.com' → 'REDACTED-EMAIL-abc123') that
        allow correlation without exposing PII.
      - >-
        Schema versioning: each event includes a schema_version field.
        Older schema versions can be read and interpreted by newer code.
    pitfalls:
      - >-
        PII in immutable logs: once PII is written to an append-only log,
        GDPR right-to-erasure compliance becomes impossible without
        destroying the entire log. Sanitize BEFORE writing, not after.
      - >-
        Over-redaction: redacting too aggressively makes logs useless for
        investigation. Use tokenization (reversible with a separate key)
        instead of irreversible masking when possible.
      - >-
        Missing actor context: audit events without proper actor
        identification (just 'user_id: 42' instead of role, IP, session)
        are insufficient for security investigation.
      - >-
        Schema evolution: adding new required fields breaks compatibility
        with old events. New fields should be optional or have defaults.
    concepts:
      - Structured audit event modeling
      - PII detection and sanitization strategies
      - Tokenization vs. masking vs. encryption
      - Schema versioning and evolution
      - GDPR compliance for immutable stores
    skills:
      - Data modeling for compliance
      - PII detection (regex, field-name rules)
      - Tokenization implementation
      - Schema versioning
    deliverables:
      - Audit event schema with required fields and extensible metadata
      - Event validation rejecting incomplete events
      - PII sanitization layer with configurable detection rules
      - Tokenization for reversible PII masking
      - Schema versioning support
      - Test suite for schema validation and PII sanitization
    estimated_hours: "10-12"

  - id: audit-logging-m2
    name: "Immutable Append-Only Storage with Hash Chain"
    description: >-
      Implement append-only storage with cryptographic hash chaining
      for tamper detection, monotonic ordering, and chain verification.
    acceptance_criteria:
      - >-
        Events are appended to storage with a monotonically increasing
        sequence number assigned atomically at write time. No gaps or
        duplicates in sequence numbers.
      - >-
        Each event's integrity hash includes: the previous event's hash,
        the sequence number, and the serialized event data. The first
        event uses a known genesis hash.
      - >-
        Hash chain verification: given any contiguous range of events,
        recomputing hashes from the range start and comparing to stored
        hashes detects any modification, insertion, or deletion.
      - >-
        Full chain verification from genesis to current head completes
        and reports the first corrupted entry (if any) with its sequence
        number.
      - >-
        Log rotation: when a segment reaches a configurable size (e.g.,
        100MB or 1M events), a new segment is started. The first entry
        of the new segment includes the final hash of the previous
        segment, maintaining chain continuity.
      - >-
        Storage is append-only: the API provides no update or delete
        operations. Attempting to modify a stored event returns an
        error.
      - >-
        At-rest encryption: stored events are encrypted using AES-256
        (or equivalent) with a configurable key.
      - >-
        Concurrent append safety: multiple writers can append events
        simultaneously without producing duplicate sequence numbers or
        broken hash chains (using transactions or atomic operations).
    pitfalls:
      - >-
        Hash chain verification is O(N) for N events (must recompute
        from the point of suspected tampering to the end). For large
        logs, full verification is slow. Consider periodic checkpoints
        or signed chain heads to enable partial verification.
      - >-
        Hash chain breaks on out-of-order insertion: sequence numbers
        and hash chain computation must be strictly serialized. Use
        a single-writer or transactional append to ensure ordering.
      - >-
        Backup and restore must preserve the complete hash chain.
        Restoring a partial backup breaks chain verification. Always
        restore to segment boundaries.
      - >-
        Log rotation segment boundary: if the cross-segment link
        (previous segment's final hash) is lost, the entire chain
        after the rotation point becomes unverifiable.
    concepts:
      - Cryptographic hash chains for tamper evidence
      - Append-only storage semantics
      - Monotonic sequence numbers and ordering
      - Log rotation with chain continuity
      - O(N) verification cost and mitigation strategies
    skills:
      - Hash chain implementation
      - Append-only storage design
      - Concurrent write safety
      - At-rest encryption
    deliverables:
      - Append-only event store with no update/delete operations
      - Cryptographic hash chain linking each event to its predecessor
      - Chain verification function detecting tampering with corrupted entry identification
      - Log rotation with cross-segment hash chain continuity
      - At-rest encryption for stored events
      - Concurrent append safety with atomic sequence number assignment
    estimated_hours: "10-12"

  - id: audit-logging-m3
    name: "Indexed Querying, Retention, and Compliance Export"
    description: >-
      Build efficient audit log querying with database indexes, retention
      policies for archival, and compliance export in standard formats.
    acceptance_criteria:
      - >-
        Database indexes on timestamp, actor_id, resource_id, and action
        enable filtered queries. Time-range query on 1 million events
        returns results in under 500ms.
      - >-
        Query API supports filtering by: time range (start/end),
        actor (ID or name), resource (ID or type), action, and outcome.
        Filters are composable (AND logic).
      - >-
        Cursor-based pagination returns results in configurable page
        sizes (default 100, max 1000) with a stable cursor that handles
        concurrent appends without skipping or duplicating results.
      - >-
        Export produces correctly formatted CSV and JSON files. CSV
        includes headers and properly escaped fields. JSON uses
        newline-delimited JSON (NDJSON) for streaming.
      - >-
        Large exports (>100K events) use streaming output to avoid
        loading all results into memory simultaneously.
      - >-
        Retention policy: events older than a configurable age (e.g.,
        7 years for SOC2) are archived to cold storage (compressed
        files). Archived events are removed from the primary store but
        remain queryable through the archive interface.
      - >-
        Audit log access control: only users with an 'audit-reader' role
        can query audit events. All access to the audit log is itself
        audit-logged (meta-auditing).
      - >-
        Summary report generation: count of events by actor, action,
        and time period for compliance review dashboards.
    pitfalls:
      - >-
        Full-text search without an index causes table scans on large
        datasets. Use database full-text indexing (PostgreSQL tsvector,
        Elasticsearch) for keyword search.
      - >-
        Large exports exhausting memory: use streaming generators (Python
        yield, Go channels) with backpressure to limit memory usage.
      - >-
        Timezone inconsistency: store all timestamps in UTC. Query
        parameters accept timezone-aware input and convert to UTC
        internally. Export includes UTC timestamps with ISO 8601 format.
      - >-
        Retention policy must preserve hash chain integrity: archiving
        old events must keep the chain-linking hashes (or checkpoint
        hashes) so the remaining chain can still be verified.
      - >-
        Access control on audit logs: if anyone can read audit logs, the
        logs themselves become a data exfiltration vector. Audit log
        access must be restricted and logged.
    concepts:
      - Database indexing for audit log queries
      - Cursor-based pagination
      - Streaming export for large datasets
      - Retention and archival policies
      - Access control for audit data
    skills:
      - Database index design
      - Streaming export implementation
      - Retention policy enforcement
      - Access control implementation
      - Compliance report generation
    deliverables:
      - Indexed query engine with composite filters (time, actor, resource, action)
      - Cursor-based pagination with stable ordering
      - Streaming CSV and NDJSON export for large result sets
      - Retention policy with configurable age-based archival
      - Access control restricting audit log reads to authorized roles
      - Summary report generator for compliance dashboards
      - Meta-auditing: audit log access is itself logged
    estimated_hours: "10-14"
```
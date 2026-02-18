# AUDIT & FIX: feature-flags

## CRITIQUE
- **Logical Gap (Confirmed - Default Fallbacks):** If the evaluation engine fails, the configuration file is corrupted, or the flag store is unreachable, there is no defined behavior. The system must have hardcoded defaults that are returned when evaluation fails — this is a safety-critical feature.
- **Logical Gap (Confirmed - Targeting Rule Hierarchy):** No mention of rule priority/precedence. In production systems, the order is typically: Kill switch override > User ID targeting > Segment/Group targeting > Percentage rollout > Default value. Without this hierarchy, evaluation is non-deterministic.
- **Estimated Hours Mismatch:** 35 hours total with 11.5 hours per milestone. 3 * 11.5 = 34.5. This is suspiciously uniform — real milestones have different complexity levels. M1 (evaluation engine) and M3 (A/B testing with statistics) are dramatically different in scope.
- **M3 Scope Explosion:** A/B testing with statistical significance, hypothesis testing, sample ratio mismatch detection, and sequential testing is an entire project by itself. Cramming it into one milestone of a feature flag project is unrealistic.
- **Missing CRUD:** There is no milestone for the flag management API (create, update, delete flags via HTTP). M1 assumes flags exist but never defines how they are created.
- **Missing Audit Trail:** Feature flag changes in production systems must be auditable (who changed what, when). This is completely absent.
- **Pitfall Quality:** 'Circular prerequisites cause infinite loop' is good. 'Rule priority ties cause non-deterministic evaluation' is good. But M2's 'Large flag payloads slow down evaluation' is not really a pitfall of SSE — it's a payload design issue.
- **Missing: Flag Lifecycle States:** No mention of flag states (active, archived, stale). In production, flags that are on for everyone for 6 months should be cleaned up.

## FIXED YAML
```yaml
id: feature-flags
name: Feature Flag System
description: >-
  Build a feature flag system with rule-based evaluation, targeting
  hierarchy, real-time updates via SSE, default fallbacks, and basic
  experiment tracking.
difficulty: intermediate
estimated_hours: "30-40"
essence: >-
  Rule-based flag evaluation with a defined targeting hierarchy (kill
  switch > user override > segment > percentage rollout > default),
  consistent hashing for stable percentage bucketing, SSE-based push
  updates with local cache and fallback defaults, and exposure logging
  for experiment analysis.
why_important: >-
  Feature flags enable safe deployments, gradual rollouts, and A/B
  testing. Understanding the evaluation hierarchy, fallback mechanisms,
  and real-time update infrastructure teaches patterns used in every
  major feature flag platform (LaunchDarkly, Unleash, Flagsmith).
learning_outcomes:
  - Design flag evaluation with a defined targeting rule hierarchy
  - Implement percentage rollouts using consistent hashing for stable user bucketing
  - Build a flag management CRUD API with audit logging
  - Implement real-time flag updates via SSE with local cache and fallback defaults
  - Handle evaluation engine failures with safe default values
  - Track flag exposures for basic experiment analysis
  - Detect circular flag dependencies using graph traversal
  - Design flag lifecycle states (active, archived, stale detection)
skills:
  - Flag evaluation logic with rule precedence
  - Targeting rules engine
  - SSE streaming
  - Local caching with fallback defaults
  - Consistent hashing for user bucketing
  - CRUD API with audit trail
  - Exposure logging
  - Graph cycle detection
tags:
  - architecture
  - backend
  - devops
  - intermediate
  - rollout
  - targeting
  - toggles
architecture_doc: architecture-docs/feature-flags/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible:
    - JavaScript
    - Rust
resources:
  - name: OpenFeature Specification
    url: https://openfeature.dev/
    type: documentation
  - name: Unleash Feature Flag Best Practices
    url: https://docs.getunleash.io/topics/feature-flags/feature-flag-best-practices
    type: documentation
  - name: Flagsmith Real-Time Flag Updates
    url: https://docs.flagsmith.com/advanced-use/real-time-flags
    type: documentation
  - name: SSE vs WebSockets Comparison
    url: https://softwaremill.com/sse-vs-websockets-comparing-real-time-communication-protocols/
    type: article
prerequisites:
  - type: skill
    name: REST API design
  - type: skill
    name: HTTP server implementation
  - type: skill
    name: JSON data modeling
milestones:
  - id: feature-flags-m1
    name: Flag Management API and Storage
    description: >-
      Build the CRUD API for creating, reading, updating, and deleting
      feature flags. Each flag has a key, type (boolean/string/number/JSON),
      default value, targeting rules, and lifecycle state. All changes
      are audit-logged.
    acceptance_criteria:
      - POST /flags creates a flag with: key (unique string), type (boolean|string|number|json), default_value, description, and optional targeting rules
      - "GET /flags/{key} returns the full flag definition including rules and current state"
      - "PUT /flags/{key} updates flag definition; changes are audit-logged with timestamp, user, old value, new value"
      - "DELETE /flags/{key} soft-deletes (archives) the flag; it is excluded from evaluation but retained for audit"
      - "GET /flags returns paginated list of all flags with filtering by state (active, archived)"
      - Flag key uniqueness enforced: duplicate key creation returns 409 Conflict
      - "Audit log queryable via GET /flags/{key}/audit returning chronological change history"
    pitfalls:
      - Hard-deleting flags: clients with cached references to deleted flags crash. Use soft-delete with archival.
      - No audit trail: in production, flag changes cause incidents. Without audit logs, root cause analysis is impossible.
      - Not validating default_value against flag type: a boolean flag with default_value 'hello' silently breaks evaluation
      - Allowing flag key changes: renaming a key breaks all SDKs referencing the old key. Keys should be immutable.
    concepts:
      - CRUD API design for configuration management
      - Soft delete and archival
      - Audit logging for compliance and debugging
      - Immutable identifiers
    skills:
      - REST API implementation
      - Data validation and type checking
      - Audit log design
      - Pagination and filtering
    deliverables:
      - Flag CRUD API (create, read, update, archive)
      - Flag storage with type validation
      - Audit log recording all flag mutations
      - Paginated flag listing with state filters
    estimated_hours: "6-8"

  - id: feature-flags-m2
    name: Flag Evaluation Engine with Targeting Hierarchy
    description: >-
      Build the core evaluation engine that resolves a flag value for a
      given user context. Implement a strict targeting hierarchy:
      kill switch > user ID override > segment rules > percentage
      rollout > default value. Include safe fallback defaults.
    acceptance_criteria:
      - "evaluate(flag_key, user_context) returns {value, variant_key, reason} where reason explains which rule matched"
      - Evaluation hierarchy (checked in order, first match wins): 1) Kill switch (flag disabled → default), 2) User ID override list, 3) Segment/attribute rules (AND/OR conditions), 4) Percentage rollout, 5) Default value
      - Percentage rollout uses consistent hashing: hash(flag_key + user_id) % 100 determines bucket. Same user always gets same variant for a given flag.
      - Segment rules support attribute conditions: equals, not_equals, contains, regex, in_list on user context attributes
      - "Circular flag dependencies (flag A depends on flag B which depends on flag A) are detected at creation time and rejected with 400 error"
      - "If evaluation throws any exception (missing flag, corrupt rules, unexpected type), the hardcoded fallback default is returned with reason='ERROR_FALLBACK'"
      - "Evaluation latency is < 1ms for flags with up to 10 rules (verified by benchmark)"
    pitfalls:
      - Non-deterministic rule evaluation: without strict priority ordering, the same request can get different results depending on rule iteration order
      - Inconsistent hashing causing user flip-flop: if the hash function changes or the modulo base changes, users switch variants. Use a stable hash like MurmurHash3.
      - No fallback on error: if the flag store is corrupted or unreachable, evaluate() throws an exception instead of returning a safe default. This crashes the application.
      - Circular dependency infinite loop: flag A's rule references flag B, flag B's rule references flag A. Without cycle detection at write time, evaluation loops forever.
    concepts:
      - Targeting rule hierarchy with strict precedence
      - Consistent hashing for stable percentage bucketing
      - Safe fallback defaults on evaluation failure
      - Dependency graph cycle detection (DFS/topological sort)
    skills:
      - Rule engine implementation with priority ordering
      - Consistent hash function (MurmurHash3 or equivalent)
      - Boolean expression evaluation (AND/OR/NOT)
      - Graph cycle detection
    deliverables:
      - Evaluation engine with strict targeting hierarchy
      - Consistent hashing for percentage rollout stability
      - Attribute-based segment rule evaluator
      - Error fallback returning safe defaults with ERROR_FALLBACK reason
    estimated_hours: "8-10"

  - id: feature-flags-m3
    name: Real-Time Flag Updates with SSE and Local Cache
    description: >-
      Implement real-time flag updates using Server-Sent Events (SSE).
      SDKs maintain a local cache of flag values and receive push
      updates when flags change. If the SSE connection fails, the SDK
      falls back to the local cache and periodically polls.
    acceptance_criteria:
      - GET /stream returns an SSE event stream. Each event contains: event type (flag_updated, flag_archived), flag key, new flag definition, and monotonic event ID
      - "When a flag is created/updated/archived via the CRUD API, an SSE event is broadcast to all connected clients within 2 seconds"
      - "SDK maintains a local in-memory cache of all flag values. evaluate() reads from cache, not the network."
      - "On SSE reconnect, SDK sends Last-Event-ID header and receives all missed events (catch-up)"
      - "If SSE connection is down for > 30s, SDK falls back to periodic polling (configurable interval, default 60s) until SSE reconnects"
      - "SSE keepalive comments sent every 15s to prevent proxy/firewall connection timeout"
      - "SDK reconnection uses exponential backoff with jitter (base 1s, max 30s) to prevent thundering herd"
      - "At least 200 concurrent SSE connections supported without degrading API response latency by more than 20%"
    pitfalls:
      - Thundering herd on SSE reconnect: if 1000 SDKs lose connection simultaneously and all reconnect at once, the server is overwhelmed. Exponential backoff with jitter is mandatory.
      - Stale cache served indefinitely: if SSE is down and polling also fails, the cache TTL should eventually trigger a 'stale cache' warning in evaluation reasons
      - Not sending SSE keepalive: intermediate proxies (nginx, AWS ALB) close idle connections after 60s
      - Full flag payloads in every SSE event: for large flag sets, send only the changed flag, not the entire flag collection
    concepts:
      - Server-Sent Events protocol
      - Local cache with push invalidation
      - Fallback from push to poll
      - Exponential backoff with jitter
    skills:
      - SSE server implementation
      - Client SDK cache design
      - Connection resilience (backoff, reconnect, fallback)
      - Event ID tracking for catch-up
    deliverables:
      - SSE endpoint broadcasting flag change events
      - SDK with local cache and SSE consumer
      - Fallback polling when SSE is unavailable
      - Reconnection with backoff, jitter, and catch-up
    estimated_hours: "8-10"

  - id: feature-flags-m4
    name: Exposure Logging and Basic Experiment Tracking
    description: >-
      Track flag evaluations (exposures) for analytics. Support basic
      A/B experiment definition with variant assignment tracking.
      Statistical analysis is out of scope — focus on correct exposure
      logging and assignment consistency.
    acceptance_criteria:
      - Every flag evaluation logs an exposure event: {flag_key, variant_key, user_id, timestamp, reason} to a configurable sink (file, HTTP endpoint, or in-memory buffer)
      - "Exposure events are batched and flushed periodically (default every 10s or 100 events) to reduce I/O overhead"
      - Experiment mode can be enabled per flag: marks the flag as an active experiment with defined variant keys and traffic allocation percentages
      - User-to-variant assignment is persistent: once a user is assigned to a variant, they always get the same variant (consistent hashing ensures this)
      - "GET /flags/{key}/exposures returns aggregated exposure counts per variant (total evaluations, unique users)"
      - Sample ratio mismatch detection: if actual variant distribution deviates from configured allocation by > 5%, a warning is logged
    pitfalls:
      - Logging every evaluation synchronously: blocks the hot path. Buffer and flush asynchronously.
      - Not deduplicating exposures: if a user is evaluated 100 times per minute, the exposure log grows 100x. Deduplicate by (user_id, flag_key, variant) per flush window.
      - Sample ratio mismatch going undetected: a bug in the hashing function or targeting rules causes uneven distribution, invalidating any experiment conclusions
      - Exposures logged for non-experiment flags: only log exposures for flags marked as experiments, or provide a filter, to avoid log explosion
    concepts:
      - Exposure logging and analytics
      - Consistent user-to-variant assignment
      - Sample ratio mismatch detection
      - Asynchronous event batching
    skills:
      - Event batching and async flushing
      - Aggregation queries
      - Statistical ratio validation
      - Analytics pipeline design
    deliverables:
      - Exposure event logger with async batching
      - Experiment mode per flag with variant allocation
      - Exposure aggregation API (counts per variant)
      - Sample ratio mismatch detector with warning logs
    estimated_hours: "6-8"
```
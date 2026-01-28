#!/usr/bin/env python3
"""Add acceptance_criteria for remaining milestones with correct names."""

import yaml
from pathlib import Path

ACCEPTANCE_CRITERIA_2 = {
    "alerting-system": {
        "Alert Rule Evaluation": [
            "Evaluate PromQL-like expressions against metrics",
            "Support comparison thresholds (>, <, ==)",
            "Handle for-duration (pending state)",
            "Return firing/resolved status"
        ],
        "Alert Grouping": [
            "Group alerts by configurable labels",
            "Reduce notification noise with aggregation",
            "Support group_wait and group_interval",
            "Handle group key generation"
        ],
        "Silencing & Inhibition": [
            "Silence alerts matching specific labels",
            "Support time-based silence windows",
            "Inhibit alerts based on other firing alerts",
            "Handle silence expiration"
        ]
    },

    "apm-system": {
        "Service Map": [
            "Build service dependency graph from traces",
            "Calculate inter-service call metrics",
            "Detect service topology changes",
            "Visualize service relationships"
        ],
        "Trace Sampling": [
            "Implement head-based sampling (probability)",
            "Support tail-based sampling (on interesting traces)",
            "Configure sampling rates per service",
            "Ensure sampled traces include all spans"
        ]
    },

    "audit-logging": {
        "Audit Event Model": [
            "Define schema: actor, action, resource, timestamp, outcome",
            "Support custom metadata fields",
            "Include request context (IP, user agent)",
            "Validate events against schema"
        ],
        "Immutable Storage with Hash Chain": [
            "Append events to log with sequence number",
            "Generate hash chain (each entry hashes previous)",
            "Detect tampering via hash verification",
            "Support log rotation preserving chain"
        ],
        "Audit Query & Export": [
            "Query by actor, action, time range",
            "Support pagination for large results",
            "Export in compliance formats (CSV, JSON)",
            "Generate audit reports"
        ]
    },

    "cdc-system": {
        "Log Parsing & Change Events": [
            "Parse PostgreSQL WAL or MySQL binlog",
            "Extract INSERT/UPDATE/DELETE operations",
            "Include before/after row values",
            "Handle transaction boundaries"
        ],
        "Event Streaming & Delivery": [
            "Publish events to Kafka/message queue",
            "Guarantee at-least-once delivery",
            "Support exactly-once with deduplication",
            "Handle consumer lag"
        ],
        "Schema Evolution & Compatibility": [
            "Track schema versions for tables",
            "Handle column additions/removals",
            "Support schema registry integration",
            "Maintain backward compatibility"
        ]
    },

    "cdn-implementation": {
        "Edge Cache Implementation": [
            "Cache responses based on URL and headers",
            "Implement LRU/LFU eviction",
            "Handle cache variants (Accept-Encoding)",
            "Support conditional requests (ETag)"
        ],
        "Origin Shield & Request Collapsing": [
            "Collapse concurrent requests for same resource",
            "Implement origin shield layer",
            "Reduce origin load during cache miss storms",
            "Support stale-while-revalidate"
        ]
    },

    "chaos-engineering": {
        "Fault Injection Framework": [
            "Inject network latency to services",
            "Simulate packet loss/corruption",
            "Kill processes/containers",
            "Exhaust resources (CPU, memory, disk)"
        ],
        "Experiment Orchestration": [
            "Define experiments with targets and faults",
            "Schedule experiments with duration",
            "Implement safety abort conditions",
            "Track experiment state"
        ],
        "GameDay Automation": [
            "Run multiple experiments in sequence",
            "Monitor system metrics during experiments",
            "Auto-rollback on safety threshold breach",
            "Generate experiment reports"
        ]
    },

    "ci-cd-pipeline": {
        "Pipeline Definition Parser": [
            "Parse YAML pipeline definitions",
            "Validate stage and step structure",
            "Support parallel and sequential steps",
            "Handle conditional execution (if/when)"
        ],
        "Job Executor": [
            "Execute steps in isolated containers",
            "Stream logs in real-time",
            "Handle step timeouts",
            "Support retries on failure"
        ],
        "Artifact Management": [
            "Upload artifacts from steps",
            "Download artifacts in dependent steps",
            "Support artifact expiration",
            "Handle large artifacts efficiently"
        ],
        "Deployment Strategies": [
            "Implement rolling deployment",
            "Support blue-green deployment",
            "Handle canary releases",
            "Implement manual approval gates"
        ]
    },

    "collaborative-editor": {
        "Operation-based CRDT": [
            "Implement RGA (Replicated Growable Array)",
            "Handle concurrent insert operations",
            "Ensure convergence across replicas",
            "Support delete operations"
        ],
        "Cursor Presence": [
            "Track cursor positions per user",
            "Broadcast cursor updates in real-time",
            "Display user name/color at cursor",
            "Handle cursor position after edits"
        ],
        "Operational Transformation": [
            "Transform operations against concurrent ops",
            "Maintain document consistency",
            "Handle operation priority/ordering",
            "Support intention preservation"
        ],
        "Undo/Redo with Collaboration": [
            "Track operation history per user",
            "Implement selective undo (own operations only)",
            "Handle undo of merged operations",
            "Support redo after undo"
        ]
    },

    "container-runtime": {
        "Process Isolation with Namespaces": [
            "Create PID namespace (process sees itself as PID 1)",
            "Create mount namespace (isolated filesystem)",
            "Create network namespace (isolated network)",
            "Create UTS namespace (isolated hostname)"
        ],
        "Resource Limits with Cgroups": [
            "Set memory limits (cgroups v2 memory.max)",
            "Set CPU limits (cpu.max quota)",
            "Monitor resource usage",
            "Handle OOM situations"
        ],
        "Overlay Filesystem": [
            "Mount overlay filesystem for container root",
            "Handle multiple image layers",
            "Support copy-on-write semantics",
            "Clean up layers on container removal"
        ],
        "Container Networking": [
            "Create veth pair for container network",
            "Set up bridge networking",
            "Implement port mapping (NAT)",
            "Handle container DNS resolution"
        ]
    },

    "etl-pipeline": {
        "Pipeline DAG Definition": [
            "Define tasks with dependencies in config",
            "Parse DAG from YAML/Python",
            "Validate no cycles in DAG",
            "Support parameterized tasks"
        ],
        "Data Extraction & Loading": [
            "Extract from databases (SQL connectors)",
            "Extract from APIs (HTTP clients)",
            "Load to target with batching",
            "Handle schema mapping"
        ],
        "Data Transformations": [
            "Support SQL-based transformations",
            "Support Python UDF transformations",
            "Handle null values and type conversions",
            "Implement data validation rules"
        ],
        "Pipeline Orchestration & Monitoring": [
            "Schedule pipeline runs (cron)",
            "Track run status and timing",
            "Handle failures with alerts",
            "Provide run history and logs"
        ]
    },

    "event-sourcing": {
        "Aggregate & Event Sourcing": [
            "Load aggregate from event stream",
            "Apply events to rebuild state",
            "Emit new events from commands",
            "Handle command validation"
        ]
    },

    "feature-flags": {
        "Flag Evaluation Engine": [
            "Evaluate flag rules with user context",
            "Support percentage rollouts (consistent hashing)",
            "Handle complex rule conditions",
            "Return variant with reason"
        ],
        "Real-time Flag Updates": [
            "Push flag changes to SDKs",
            "Support streaming (SSE/WebSocket)",
            "Handle SDK reconnection",
            "Ensure consistency across instances"
        ],
        "Flag Analytics & Experiments": [
            "Track flag evaluations",
            "Support A/B experiment metrics",
            "Calculate statistical significance",
            "Generate experiment reports"
        ]
    },

    "file-upload-service": {
        "Chunked Upload Protocol": [
            "Initialize multipart upload",
            "Accept chunk uploads with part numbers",
            "Track chunk completion status",
            "Complete upload by assembling chunks"
        ],
        "Storage Abstraction": [
            "Abstract storage interface (local, S3, GCS)",
            "Handle storage credentials",
            "Generate signed download URLs",
            "Support storage migration"
        ],
        "Virus Scanning & Validation": [
            "Scan uploads with ClamAV or similar",
            "Validate file types (magic bytes)",
            "Enforce size limits",
            "Quarantine suspicious files"
        ]
    },

    "infrastructure-as-code": {
        "Configuration Parser": [
            "Parse HCL/YAML configuration files",
            "Support variable interpolation",
            "Handle includes/modules",
            "Validate configuration syntax"
        ],
        "Dependency Graph & Planning": [
            "Build resource dependency graph",
            "Generate execution plan (create, update, delete)",
            "Show plan diff before apply",
            "Support targeted apply"
        ],
        "Provider Abstraction": [
            "Define provider interface",
            "Implement CRUD for resource types",
            "Handle provider authentication",
            "Support provider state refresh"
        ]
    },

    "job-scheduler": {
        "Cron Expression Parser": [
            "Parse standard cron expressions",
            "Calculate next run times",
            "Support extended cron (seconds, @daily)",
            "Handle timezone conversions"
        ],
        "Job Queue with Priorities": [
            "Enqueue jobs with priority levels",
            "Support delayed job execution",
            "Handle job deduplication",
            "Implement job TTL"
        ],
        "Worker Coordination": [
            "Distribute jobs to available workers",
            "Implement job locking (prevent duplicate runs)",
            "Handle worker heartbeat/health",
            "Support graceful worker shutdown"
        ]
    },

    "load-testing-framework": {
        "Virtual User Simulation": [
            "Spawn concurrent virtual users",
            "Execute user scenarios (request sequences)",
            "Support think time between requests",
            "Handle session/cookie management"
        ],
        "Distributed Workers": [
            "Distribute load across worker nodes",
            "Coordinate test start/stop",
            "Aggregate metrics from workers",
            "Handle worker failures"
        ],
        "Real-time Metrics & Reporting": [
            "Calculate latency percentiles in real-time",
            "Track throughput and error rates",
            "Generate live dashboard",
            "Export final report (HTML, JSON)"
        ]
    },

    "log-aggregator": {
        "Log Index": [
            "Build inverted index for log messages",
            "Index structured fields (level, service)",
            "Support time-based partitioning",
            "Handle index compaction"
        ],
        "Log Query Engine": [
            "Support full-text search in messages",
            "Filter by structured fields",
            "Support regex patterns",
            "Implement query pagination"
        ]
    },

    "media-processing": {
        "Image Processing": [
            "Resize images to multiple sizes",
            "Convert between formats (JPEG, PNG, WebP)",
            "Generate thumbnails",
            "Handle EXIF metadata"
        ],
        "Video Transcoding": [
            "Transcode to multiple formats/codecs",
            "Generate adaptive bitrate variants",
            "Extract video thumbnails",
            "Support HLS/DASH output"
        ],
        "Processing Queue & Progress": [
            "Queue media processing jobs",
            "Track processing progress",
            "Handle failures with retries",
            "Notify on completion"
        ]
    },

    "metrics-collector": {
        "Metrics Data Model": [
            "Define metric types (counter, gauge, histogram)",
            "Support labels/dimensions",
            "Handle metric metadata",
            "Validate metric names"
        ],
        "Scrape Engine": [
            "Discover scrape targets",
            "Pull metrics from endpoints",
            "Handle scrape timeouts",
            "Support service discovery"
        ],
        "Time Series Storage": [
            "Store time series efficiently",
            "Implement compression (Gorilla)",
            "Support retention policies",
            "Handle high cardinality"
        ]
    },

    "multiplayer-game-server": {
        "Game Loop & Tick System": [
            "Run fixed-rate game loop (e.g., 60 ticks/sec)",
            "Process input from all players",
            "Update game state each tick",
            "Handle tick timing consistency"
        ],
        "Client Prediction & Reconciliation": [
            "Apply inputs locally on client",
            "Compare with server authoritative state",
            "Reconcile discrepancies smoothly",
            "Handle prediction errors gracefully"
        ],
        "Lag Compensation": [
            "Rewind server state for hit detection",
            "Account for client latency",
            "Handle interpolation between states",
            "Balance fairness across latencies"
        ],
        "State Synchronization": [
            "Serialize game state efficiently",
            "Send delta updates to reduce bandwidth",
            "Handle state snapshot for reconnection",
            "Support client interpolation"
        ]
    },

    "notification-service": {
        "Channel Abstraction & Routing": [
            "Define channel interface (email, SMS, push)",
            "Route notifications to appropriate channels",
            "Handle channel-specific formatting",
            "Support fallback channels"
        ],
        "User Preferences & Unsubscribe": [
            "Store notification preferences per user",
            "Handle unsubscribe requests",
            "Support notification categories",
            "Respect quiet hours"
        ],
        "Delivery Tracking & Analytics": [
            "Track delivery status (sent, delivered, failed)",
            "Record open/click events",
            "Calculate delivery metrics",
            "Alert on delivery issues"
        ]
    },

    "oauth2-provider": {
        "Client Registration & Authorization Endpoint": [
            "Register OAuth clients with credentials",
            "Display authorization consent page",
            "Generate authorization codes",
            "Support redirect URI validation"
        ],
        "Token Endpoint & JWT Generation": [
            "Exchange code for access/refresh tokens",
            "Generate signed JWTs",
            "Support client_credentials grant",
            "Handle token expiration"
        ],
        "Token Introspection & Revocation": [
            "Implement introspection endpoint (RFC 7662)",
            "Support token revocation (RFC 7009)",
            "Validate tokens for resource servers",
            "Handle revoked token checking"
        ],
        "UserInfo Endpoint & Consent Management": [
            "Return user claims at userinfo endpoint",
            "Store user consent decisions",
            "Support consent revocation",
            "Handle scope-based claims"
        ]
    },

    "payment-gateway": {
        "Payment Intent & Idempotency": [
            "Create payment intents with idempotency key",
            "Track payment state transitions",
            "Handle duplicate requests safely",
            "Support payment metadata"
        ],
        "Payment Processing & 3DS": [
            "Process card payments via provider",
            "Implement 3D Secure authentication",
            "Handle authentication redirects",
            "Support multiple payment methods"
        ],
        "Refunds & Disputes": [
            "Process full and partial refunds",
            "Track refund status",
            "Handle chargeback notifications",
            "Support dispute evidence submission"
        ],
        "Webhook Reconciliation": [
            "Process payment webhooks",
            "Update payment status from webhooks",
            "Handle webhook signature verification",
            "Reconcile payments with expected state"
        ]
    },

    "rate-limiter-distributed": {
        "Rate Limiting Algorithms": [
            "Implement sliding window counter",
            "Support token bucket algorithm",
            "Handle window boundary transitions",
            "Configure rate and window size"
        ],
        "Multi-tier Rate Limiting": [
            "Support per-user and per-IP limits",
            "Implement global rate limits",
            "Handle burst allowance",
            "Support different limits per endpoint"
        ]
    },

    "rbac-system": {
        "Role & Permission Model": [
            "Define roles with named permissions",
            "Support role hierarchy (inheritance)",
            "Assign roles to users",
            "Handle role templates"
        ],
        "ABAC Policy Engine": [
            "Define attribute-based policies",
            "Evaluate policies with context attributes",
            "Support policy combining (AND/OR)",
            "Handle policy priority/ordering"
        ],
        "Resource-Based & Multi-tenancy": [
            "Support permissions per resource",
            "Handle tenant isolation",
            "Implement resource ownership",
            "Support cross-tenant admin access"
        ],
        "Audit Logging & Policy Testing": [
            "Log all access decisions",
            "Support policy simulation/testing",
            "Track permission changes",
            "Generate access reports"
        ]
    },

    "saga-orchestrator": {
        "Saga Orchestrator Engine": [
            "Execute saga steps in sequence",
            "Track step completion status",
            "Trigger compensation on failure",
            "Handle step timeouts"
        ],
        "Saga State Persistence": [
            "Persist saga execution state",
            "Support saga recovery after crash",
            "Track compensation progress",
            "Handle idempotent step execution"
        ]
    },

    "search-engine": {
        "TF-IDF & BM25 Ranking": [
            "Calculate term frequency per document",
            "Calculate inverse document frequency",
            "Implement BM25 scoring formula",
            "Rank search results by score"
        ],
        "Fuzzy Matching & Autocomplete": [
            "Implement Levenshtein distance matching",
            "Support prefix-based autocomplete",
            "Handle typo tolerance",
            "Rank suggestions by relevance"
        ],
        "Query Parser & Filters": [
            "Parse boolean queries (AND, OR, NOT)",
            "Support phrase queries",
            "Handle field-specific filters",
            "Support numeric range queries"
        ]
    },

    "secret-management": {
        "Encrypted Secret Storage": [
            "Store secrets encrypted at rest",
            "Use AES-256-GCM encryption",
            "Support secret versioning",
            "Handle encryption key rotation"
        ],
        "Access Policies & Authentication": [
            "Define access policies for secrets",
            "Authenticate clients (token, TLS cert)",
            "Authorize access based on policies",
            "Log all access attempts"
        ],
        "Dynamic Secrets": [
            "Generate database credentials on demand",
            "Handle credential rotation",
            "Revoke credentials on lease expiration",
            "Support multiple secret backends"
        ],
        "Unsealing & High Availability": [
            "Implement Shamir's secret sharing for master key",
            "Support multiple unseal keys",
            "Handle HA with leader election",
            "Support auto-unseal with cloud KMS"
        ]
    },

    "service-mesh": {
        "Traffic Interception": [
            "Intercept inbound/outbound traffic via iptables",
            "Handle transparent proxying",
            "Parse HTTP/gRPC protocols",
            "Support pass-through for unknown protocols"
        ],
        "Service Discovery Integration": [
            "Integrate with Kubernetes/Consul",
            "Handle endpoint updates",
            "Support DNS-based discovery",
            "Cache service endpoints"
        ],
        "mTLS and Certificate Management": [
            "Generate certificates for services",
            "Enforce mutual TLS between services",
            "Handle certificate rotation",
            "Verify service identity"
        ],
        "Load Balancing Algorithms": [
            "Implement round-robin distribution",
            "Support least-connections",
            "Handle weighted distribution",
            "Implement health-aware routing"
        ]
    },

    "session-management": {
        "Secure Session Creation & Storage": [
            "Generate cryptographically secure session IDs",
            "Store session data server-side",
            "Support multiple storage backends",
            "Handle session expiration"
        ],
        "Cookie Security & Transport": [
            "Set HttpOnly, Secure, SameSite flags",
            "Handle cookie encryption",
            "Support cookie-less sessions (header-based)",
            "Implement CSRF protection"
        ],
        "Multi-Device & Concurrent Sessions": [
            "Track sessions per user across devices",
            "Support session listing/revocation",
            "Handle concurrent session limits",
            "Implement session activity tracking"
        ]
    },

    "subscription-billing": {
        "Plans & Pricing": [
            "Define subscription plans with tiers",
            "Support multiple pricing models (flat, tiered)",
            "Handle plan versioning",
            "Support trial periods"
        ],
        "Proration & Plan Changes": [
            "Calculate prorated charges on upgrade",
            "Calculate credits on downgrade",
            "Handle mid-cycle plan changes",
            "Support immediate vs end-of-cycle changes"
        ],
        "Usage-Based Billing": [
            "Track usage events per subscription",
            "Aggregate usage for billing period",
            "Calculate charges based on usage tiers",
            "Handle usage overage"
        ]
    },

    "time-series-db": {
        "Storage Engine": [
            "Design time series storage format",
            "Implement column-oriented storage",
            "Handle high write throughput",
            "Support efficient range queries"
        ],
        "Write Path": [
            "Buffer writes in memory (WAL)",
            "Batch writes to storage",
            "Handle out-of-order data",
            "Implement write acknowledgment"
        ],
        "Retention & Compaction": [
            "Enforce data TTL",
            "Implement storage compaction",
            "Support downsampling for old data",
            "Handle storage reclamation"
        ],
        "Query Language & API": [
            "Implement time series query language",
            "Support aggregation functions",
            "Handle GROUP BY time intervals",
            "Expose REST/gRPC API"
        ]
    },

    "webhook-delivery": {
        "Webhook Registration & Security": [
            "Store webhook endpoints with secrets",
            "Generate HMAC signatures for payloads",
            "Validate endpoint ownership",
            "Handle secret rotation"
        ],
        "Delivery Queue & Retry Logic": [
            "Queue webhook events for delivery",
            "Implement exponential backoff retries",
            "Handle delivery timeouts",
            "Track delivery attempts"
        ],
        "Circuit Breaker & Rate Limiting": [
            "Implement circuit breaker per endpoint",
            "Disable endpoints after repeated failures",
            "Rate limit delivery attempts",
            "Support endpoint recovery"
        ],
        "Event Log & Replay": [
            "Store event delivery history",
            "Support manual event replay",
            "Provide delivery logs per webhook",
            "Handle replay with deduplication"
        ]
    },

    "websocket-server": {
        "HTTP Upgrade Handshake": [
            "Parse HTTP upgrade request",
            "Validate Sec-WebSocket-Key header",
            "Generate Sec-WebSocket-Accept response",
            "Complete protocol upgrade"
        ],
        "Frame Parsing": [
            "Parse WebSocket frame header",
            "Handle continuation frames",
            "Unmask client frames",
            "Support text and binary frames"
        ],
        "Connection Management": [
            "Track active connections",
            "Handle connection state (OPEN, CLOSING, CLOSED)",
            "Implement clean close handshake",
            "Support connection metadata"
        ],
        "Ping/Pong Heartbeat": [
            "Send periodic ping frames",
            "Handle pong responses",
            "Detect dead connections",
            "Configure heartbeat interval"
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    expert_projects = data.get('expert_projects', {})
    updated = 0

    for project_id, criteria_by_milestone in ACCEPTANCE_CRITERIA_2.items():
        if project_id not in expert_projects:
            print(f"Project not found: {project_id}")
            continue

        project = expert_projects[project_id]
        milestones = project.get('milestones', [])

        for milestone in milestones:
            milestone_name = milestone.get('name', '')

            if milestone_name in criteria_by_milestone:
                if not milestone.get('acceptance_criteria'):
                    milestone['acceptance_criteria'] = criteria_by_milestone[milestone_name]
                    updated += 1

    data['expert_projects'] = expert_projects

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"Updated: {updated} milestones")


if __name__ == "__main__":
    main()

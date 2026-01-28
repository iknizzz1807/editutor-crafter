#!/usr/bin/env python3
"""Add acceptance_criteria to milestones that are missing them."""

import yaml
from pathlib import Path

# Acceptance criteria for each project's milestones
# Format: {project_id: {milestone_name: [criteria1, criteria2, ...]}}

ACCEPTANCE_CRITERIA = {
    "alerting-system": {
        "Alert Rules Engine": [
            "Define alert rules with threshold conditions",
            "Support comparison operators (>, <, ==, !=, >=, <=)",
            "Allow multiple conditions with AND/OR logic",
            "Evaluate rules against incoming metrics"
        ],
        "Alert State Machine": [
            "Track alert state transitions (OK -> PENDING -> FIRING -> RESOLVED)",
            "Implement configurable pending period before firing",
            "Handle flapping detection (rapid state changes)",
            "Persist alert state for crash recovery"
        ],
        "Notification Routing": [
            "Route alerts to different channels based on severity",
            "Support multiple notification channels (email, Slack, webhook)",
            "Implement notification grouping by alert name/labels",
            "Handle notification rate limiting"
        ],
        "Escalation & Silencing": [
            "Implement escalation policies (notify manager after X minutes)",
            "Support maintenance windows (silence alerts during period)",
            "Allow manual alert acknowledgment",
            "Track escalation history for auditing"
        ]
    },

    "apm-system": {
        "Trace Collection": [
            "Receive spans via HTTP/gRPC endpoint",
            "Parse OpenTelemetry trace format",
            "Store spans with trace ID indexing",
            "Handle high-volume span ingestion (1000+ spans/sec)"
        ],
        "Trace Assembly": [
            "Reconstruct full traces from spans",
            "Build span tree with parent-child relationships",
            "Calculate trace duration and critical path",
            "Handle missing/orphan spans gracefully"
        ],
        "Performance Metrics": [
            "Calculate service latency percentiles (p50, p95, p99)",
            "Track error rates by service/endpoint",
            "Compute throughput (requests/second)",
            "Detect performance anomalies vs baseline"
        ]
    },

    "audit-logging": {
        "Audit Event Schema": [
            "Define schema with who/what/when/where/outcome fields",
            "Include correlation ID for request tracing",
            "Support custom metadata fields",
            "Validate events against schema before storage"
        ],
        "Immutable Storage": [
            "Write events to append-only log",
            "Generate cryptographic hash chain for integrity",
            "Prevent modification of existing records",
            "Support log rotation without breaking hash chain"
        ],
        "Query & Compliance": [
            "Search by actor, action, resource, time range",
            "Generate compliance reports (who accessed what)",
            "Support retention policies (auto-archive old logs)",
            "Export in standard formats (CSV, JSON)"
        ]
    },

    "background-job-processor": {
        "Job Queue Core": [
            "Enqueue jobs with payload to Redis list",
            "Support multiple named queues with priorities",
            "Implement job serialization/deserialization",
            "Handle large payloads (>1MB) appropriately"
        ],
        "Worker Process": [
            "Poll queue and process jobs",
            "Support configurable concurrency (multiple workers)",
            "Handle graceful shutdown (finish current job)",
            "Report worker heartbeat for monitoring"
        ],
        "Retry & Error Handling": [
            "Implement exponential backoff retry (1s, 2s, 4s...)",
            "Move failed jobs to dead letter queue after max retries",
            "Store error details with failed jobs",
            "Allow manual retry of dead jobs"
        ],
        "Scheduling & Cron": [
            "Schedule jobs for future execution (delay)",
            "Support recurring jobs with cron syntax",
            "Prevent duplicate scheduled jobs",
            "Handle scheduler crash recovery"
        ],
        "Monitoring & Dashboard": [
            "Track queue depth, processing rate, error rate",
            "Web dashboard showing job status",
            "Alert on queue backlog or high error rate",
            "Job search by ID, status, time range"
        ]
    },

    "build-graphql-engine": {
        "GraphQL Parser": [
            "Parse GraphQL query string to AST",
            "Support queries, mutations, subscriptions",
            "Handle fragments and variables",
            "Provide meaningful parse errors with line numbers"
        ],
        "Schema & Type System": [
            "Define GraphQL types (Object, Scalar, Enum, Interface, Union)",
            "Build schema from type definitions",
            "Validate schema for correctness",
            "Support custom scalar types"
        ],
        "Query Execution": [
            "Execute queries against schema",
            "Resolve fields with resolver functions",
            "Handle nested field resolution",
            "Implement proper null handling per spec"
        ],
        "Database Schema Reflection": [
            "Introspect database tables and columns",
            "Auto-generate GraphQL types from tables",
            "Map database types to GraphQL types",
            "Generate relationship fields from foreign keys"
        ],
        "Query to SQL Compilation": [
            "Compile GraphQL query to SQL",
            "Optimize with JOIN for relationships",
            "Prevent N+1 queries",
            "Support filtering, pagination, ordering"
        ]
    },

    "cdc-system": {
        "WAL/Binlog Reader": [
            "Connect to database replication stream",
            "Parse PostgreSQL WAL or MySQL binlog format",
            "Track position/offset for resume",
            "Handle schema changes gracefully"
        ],
        "Change Event Processing": [
            "Convert raw changes to structured events",
            "Include before/after values for updates",
            "Preserve transaction boundaries",
            "Filter by table/schema"
        ],
        "Event Publishing": [
            "Publish events to message queue (Kafka/Redis)",
            "Guarantee at-least-once delivery",
            "Support exactly-once with deduplication",
            "Handle backpressure from slow consumers"
        ]
    },

    "cdn-implementation": {
        "Edge Caching": [
            "Cache responses based on Cache-Control headers",
            "Implement cache key generation (URL, headers, query)",
            "Support cache variants (Accept-Encoding, etc.)",
            "Handle conditional requests (If-None-Match)"
        ],
        "Cache Invalidation": [
            "Purge specific URLs from cache",
            "Support wildcard/prefix purge",
            "Propagate invalidation to all edge nodes",
            "Track invalidation for debugging"
        ],
        "Origin Shielding": [
            "Collapse multiple requests for same resource",
            "Implement origin connection pooling",
            "Handle origin failures with stale-if-error",
            "Load balance across multiple origins"
        ]
    },

    "chaos-engineering": {
        "Fault Injection": [
            "Inject network latency to specific services",
            "Drop/corrupt packets between services",
            "Simulate service crashes",
            "Fill disk space or exhaust memory"
        ],
        "Experiment Definition": [
            "Define experiments with target, fault type, duration",
            "Support blast radius limits (% of instances)",
            "Schedule experiments for specific times",
            "Implement safety abort conditions"
        ],
        "Steady State Monitoring": [
            "Define steady state hypothesis (metrics thresholds)",
            "Monitor metrics during experiment",
            "Automatically abort if safety threshold breached",
            "Generate experiment report with results"
        ]
    },

    "ci-cd-pipeline": {
        "Pipeline Definition": [
            "Define pipeline stages in YAML format",
            "Support sequential and parallel steps",
            "Handle step dependencies",
            "Support conditional execution (only on main branch)"
        ],
        "Execution Engine": [
            "Execute pipeline steps in containers",
            "Pass artifacts between steps",
            "Stream logs in real-time",
            "Handle step timeouts and retries"
        ],
        "Source Integration": [
            "Trigger pipelines on git push/PR",
            "Clone repository into workspace",
            "Checkout specific commit/branch",
            "Support webhook triggers"
        ],
        "Deployment": [
            "Deploy to staging/production environments",
            "Support rolling and blue-green deployments",
            "Implement manual approval gates",
            "Handle deployment rollback"
        ]
    },

    "collaborative-editor": {
        "CRDT Implementation": [
            "Implement text CRDT (RGA or similar)",
            "Handle concurrent insertions at same position",
            "Ensure eventual consistency across clients",
            "Support offline editing and merge"
        ],
        "Real-time Sync": [
            "WebSocket connection for real-time updates",
            "Broadcast operations to all connected clients",
            "Handle client reconnection and state sync",
            "Optimize bandwidth with operation batching"
        ],
        "Cursor & Selection": [
            "Show other users' cursor positions",
            "Display user names/colors for identification",
            "Update cursor positions in real-time",
            "Handle selection ranges"
        ],
        "Version History": [
            "Store document snapshots periodically",
            "Allow viewing historical versions",
            "Support reverting to previous version",
            "Track changes with author attribution"
        ]
    },

    "container-runtime": {
        "Image Management": [
            "Pull container images from registry",
            "Unpack OCI image layers",
            "Implement layer caching",
            "Verify image signatures/digests"
        ],
        "Container Lifecycle": [
            "Create container from image",
            "Start/stop/restart containers",
            "Execute commands inside running container",
            "Remove containers and cleanup resources"
        ],
        "Namespace Setup": [
            "Create pid, mount, network, uts namespaces",
            "Setup root filesystem with pivot_root",
            "Configure network interface in container",
            "Apply cgroup resource limits"
        ],
        "OCI Compliance": [
            "Read OCI runtime specification config.json",
            "Support OCI container lifecycle hooks",
            "Implement standard OCI interface",
            "Pass OCI conformance tests"
        ]
    },

    "data-quality-framework": {
        "Expectation Engine": [
            "Define data quality expectations (not_null, unique, range)",
            "Evaluate expectations against datasets",
            "Support custom expectation definitions",
            "Return detailed validation results"
        ],
        "Data Profiling": [
            "Calculate column statistics (min, max, mean, nulls)",
            "Detect data types automatically",
            "Generate distribution histograms",
            "Identify potential data quality issues"
        ],
        "Anomaly Detection": [
            "Detect statistical anomalies (Z-score, IQR)",
            "Track metric trends over time",
            "Alert on sudden distribution changes",
            "Handle seasonality in time series"
        ],
        "Data Contracts": [
            "Define schema contracts in YAML",
            "Version contracts with semantic versioning",
            "Validate data against contract",
            "Track contract violations over time"
        ]
    },

    "etl-pipeline": {
        "Extract": [
            "Connect to multiple data sources (DB, API, files)",
            "Handle pagination for API sources",
            "Support incremental extraction (since last run)",
            "Track extraction metadata (row count, timing)"
        ],
        "Transform": [
            "Apply data transformations (map, filter, aggregate)",
            "Support schema mapping and type conversion",
            "Handle null values and data cleaning",
            "Implement transformation chaining"
        ],
        "Load": [
            "Write to target data stores",
            "Support upsert (insert or update)",
            "Handle batch loading for performance",
            "Implement transactional loading"
        ],
        "Orchestration": [
            "Schedule ETL jobs (cron or interval)",
            "Track job runs with status and timing",
            "Handle job failures and retries",
            "Support job dependencies"
        ]
    },

    "event-sourcing": {
        "Event Store": [
            "Append events to stream",
            "Read events for a stream in order",
            "Support optimistic concurrency (expected version)",
            "Handle high write throughput"
        ],
        "Projections": [
            "Build read models from events",
            "Support multiple projections per stream",
            "Handle projection rebuild from scratch",
            "Track projection position in stream"
        ],
        "Snapshots": [
            "Create aggregate snapshots periodically",
            "Load snapshot + events since snapshot",
            "Handle snapshot versioning (schema changes)",
            "Automatic snapshot creation after N events"
        ],
        "Subscriptions": [
            "Subscribe to event streams",
            "Deliver events to subscribers in real-time",
            "Support catch-up subscriptions",
            "Handle subscriber failures and retries"
        ]
    },

    "feature-flags": {
        "Flag Definition": [
            "Create boolean flags with default value",
            "Support flag variants (A/B/C)",
            "Group flags by environment",
            "Track flag metadata (owner, description)"
        ],
        "Targeting Rules": [
            "Target flags by user attributes (id, role, country)",
            "Support percentage rollout",
            "Implement AND/OR rule conditions",
            "Handle default value when no rules match"
        ],
        "SDK & Evaluation": [
            "Client SDK that evaluates flags locally",
            "Cache flag configuration in SDK",
            "Support real-time flag updates",
            "Handle offline mode gracefully"
        ]
    },

    "file-upload-service": {
        "Chunked Upload": [
            "Split large files into chunks (e.g., 5MB)",
            "Upload chunks in parallel",
            "Track chunk completion status",
            "Reassemble chunks on server"
        ],
        "Resumable Uploads": [
            "Store upload progress persistently",
            "Resume from last successful chunk",
            "Handle upload timeout gracefully",
            "Support upload expiration"
        ],
        "Validation & Security": [
            "Validate file type by magic bytes (not extension)",
            "Scan uploads for malware (ClamAV or similar)",
            "Enforce file size limits",
            "Generate secure download URLs"
        ]
    },

    "filesystem": {
        "Block Layer": [
            "Read/write fixed-size blocks to disk/file",
            "Implement block allocation bitmap",
            "Handle block caching for performance",
            "Support multiple block sizes"
        ],
        "Inode Management": [
            "Store file metadata in inode (size, permissions, timestamps)",
            "Allocate and free inodes",
            "Support direct and indirect block pointers",
            "Handle inode table on disk"
        ],
        "Directory Operations": [
            "Store directory entries (name -> inode mapping)",
            "Support directory creation and deletion",
            "Implement path resolution (a/b/c)",
            "Handle . and .. entries"
        ],
        "File Operations": [
            "Create, read, write, truncate files",
            "Support file seeking",
            "Handle file holes (sparse files)",
            "Update timestamps on operations"
        ],
        "FUSE Interface": [
            "Mount filesystem via FUSE",
            "Implement FUSE callbacks (getattr, read, write)",
            "Handle concurrent access",
            "Support unmount and cleanup"
        ]
    },

    "fuzzer": {
        "Target Execution": [
            "Execute target program with test input",
            "Capture exit code and crash signals",
            "Handle target timeout",
            "Support different input delivery methods (stdin, file, argv)"
        ],
        "Coverage Tracking": [
            "Instrument target for code coverage (llvm-cov or similar)",
            "Track basic block / edge coverage",
            "Detect new coverage from inputs",
            "Build coverage bitmap"
        ],
        "Mutation Engine": [
            "Implement bit flip mutations",
            "Support byte insertion/deletion",
            "Implement arithmetic mutations (add/sub)",
            "Use coverage feedback to guide mutations"
        ],
        "Corpus Management": [
            "Store inputs that increase coverage",
            "Minimize corpus (remove redundant inputs)",
            "Support initial seed corpus",
            "Export crash-inducing inputs"
        ],
        "Fuzzing Loop": [
            "Pick input from corpus",
            "Mutate and execute",
            "Track coverage and update corpus",
            "Report crashes with reproduction steps"
        ]
    },

    "gitops-deployment": {
        "Git Repository Sync": [
            "Clone Git repository",
            "Poll for changes periodically",
            "Support webhook triggers for immediate sync",
            "Track current synced commit SHA"
        ],
        "Manifest Generation": [
            "Support plain Kubernetes YAML",
            "Support Kustomize overlays",
            "Support Helm chart rendering",
            "Validate manifests before apply"
        ],
        "Sync & Reconciliation": [
            "Compare desired state (git) with actual state (cluster)",
            "Apply diffs to reach desired state",
            "Handle resource creation, update, deletion",
            "Support sync hooks (pre/post)"
        ],
        "Health Assessment": [
            "Monitor deployed resource health",
            "Detect degraded/progressing/healthy states",
            "Track sync errors and warnings",
            "Support custom health checks"
        ],
        "Rollback & History": [
            "Store deployment history",
            "Support rollback to previous version",
            "Show diff between versions",
            "Track who deployed what when"
        ]
    },

    "graphql-server": {
        "Schema & Type System": [
            "Define types, queries, mutations in SDL",
            "Support custom scalar types",
            "Implement interfaces and unions",
            "Validate schema at startup"
        ],
        "Resolvers & Data Fetching": [
            "Implement resolver functions per field",
            "Pass context to resolvers",
            "Handle resolver errors gracefully",
            "Support async resolvers"
        ],
        "DataLoader & N+1 Prevention": [
            "Batch database requests with DataLoader",
            "Cache results within single request",
            "Clear cache between requests",
            "Show N+1 queries eliminated in logs"
        ],
        "Subscriptions": [
            "WebSocket connection for subscriptions",
            "Publish events to subscribers",
            "Handle subscription filtering",
            "Clean up on client disconnect"
        ]
    },

    "infrastructure-as-code": {
        "Resource Definition": [
            "Define resources in HCL/YAML format",
            "Support resource dependencies",
            "Handle resource attributes and outputs",
            "Validate configuration syntax"
        ],
        "Provider Interface": [
            "Abstract provider interface for different clouds",
            "Implement CRUD operations per resource type",
            "Handle provider authentication",
            "Support multiple providers in single config"
        ],
        "State Management": [
            "Store current infrastructure state",
            "Lock state during operations",
            "Support state backends (local, remote)",
            "Handle state corruption recovery"
        ],
        "Plan & Apply": [
            "Generate execution plan (create, update, delete)",
            "Show diff of planned changes",
            "Apply changes in correct order",
            "Support dry-run mode"
        ]
    },

    "job-scheduler": {
        "Job Definition": [
            "Define jobs with command and schedule",
            "Support cron schedule syntax",
            "Handle job dependencies",
            "Store job metadata (owner, description)"
        ],
        "Distributed Execution": [
            "Distribute jobs across worker nodes",
            "Implement job locking (prevent duplicate runs)",
            "Handle worker failures",
            "Support job routing to specific workers"
        ],
        "Monitoring": [
            "Track job execution history",
            "Alert on job failures",
            "Provide job execution logs",
            "Dashboard showing upcoming and past runs"
        ]
    },

    "kubernetes-operator": {
        "Custom Resource Definition": [
            "Define CRD schema with OpenAPI validation",
            "Support spec and status subresources",
            "Implement printer columns for kubectl",
            "Handle CRD versioning"
        ],
        "Controller Setup": [
            "Watch custom resource events",
            "Use informers for efficient caching",
            "Handle connection errors and reconnection",
            "Support leader election for HA"
        ],
        "Reconciliation Loop": [
            "Compare desired state with actual state",
            "Create/update/delete owned resources",
            "Update status with current state",
            "Handle reconciliation errors gracefully"
        ],
        "Webhooks": [
            "Implement validating admission webhook",
            "Implement mutating webhook for defaults",
            "Handle webhook TLS certificates",
            "Return meaningful validation errors"
        ],
        "Testing & Deployment": [
            "Unit tests with fake client",
            "Integration tests with envtest",
            "Build container image for operator",
            "Deploy with proper RBAC permissions"
        ]
    },

    "ledger-system": {
        "Account & Entry Model": [
            "Design account types (asset, liability, equity, income, expense)",
            "Store journal entries with debit and credit lines",
            "Enforce debit = credit for each entry",
            "Support multi-currency accounts"
        ],
        "Transaction Recording": [
            "Create journal entries atomically",
            "Validate account types for entry lines",
            "Generate unique transaction IDs",
            "Support transaction metadata"
        ],
        "Balance Calculation": [
            "Calculate account balance efficiently",
            "Support balance at specific date",
            "Implement running balance updates",
            "Handle large transaction volumes"
        ],
        "Audit Trail": [
            "Log all changes with timestamp and actor",
            "Prevent modification of posted entries",
            "Support entry reversal (not deletion)",
            "Generate audit reports"
        ],
        "Financial Reports": [
            "Generate balance sheet",
            "Generate income statement",
            "Support trial balance report",
            "Export reports in standard formats"
        ]
    },

    "llm-finetuning-pipeline": {
        "Dataset Preparation": [
            "Load and validate training data",
            "Convert to instruction-response format",
            "Split into train/validation sets",
            "Handle data augmentation"
        ],
        "LoRA Configuration": [
            "Identify target modules for LoRA",
            "Configure rank and alpha parameters",
            "Set up trainable adapters",
            "Verify parameter count reduction"
        ],
        "QLoRA & Quantization": [
            "Load model in 4-bit precision",
            "Configure quantization parameters",
            "Handle mixed precision training",
            "Monitor memory usage"
        ],
        "Training Loop": [
            "Implement training loop with gradient accumulation",
            "Track training loss",
            "Save checkpoints periodically",
            "Support early stopping"
        ],
        "Evaluation & Merging": [
            "Evaluate on held-out set",
            "Compare metrics before/after fine-tuning",
            "Merge LoRA weights into base model",
            "Export merged model for inference"
        ]
    },

    "load-testing-framework": {
        "Test Definition": [
            "Define test scenarios in code/config",
            "Support request sequences",
            "Handle request parameterization",
            "Support think time between requests"
        ],
        "Load Generation": [
            "Generate concurrent virtual users",
            "Support ramp-up patterns",
            "Distribute load across workers",
            "Handle rate limiting"
        ],
        "Metrics Collection": [
            "Collect response times per request",
            "Calculate percentiles (p50, p95, p99)",
            "Track throughput and error rate",
            "Generate real-time metrics"
        ]
    },

    "lock-free-structures": {
        "Atomic Operations": [
            "Implement compare-and-swap wrapper",
            "Understand memory ordering (relaxed, acquire, release, seq_cst)",
            "Handle ABA problem awareness",
            "Demonstrate atomic counter"
        ],
        "Lock-free Stack": [
            "Implement Treiber stack with CAS",
            "Handle concurrent push/pop",
            "Ensure linearizability",
            "Benchmark against mutex-based stack"
        ],
        "Lock-free Queue": [
            "Implement Michael-Scott queue",
            "Handle dummy node",
            "Ensure FIFO ordering",
            "Handle empty queue correctly"
        ],
        "Hazard Pointers": [
            "Implement hazard pointer scheme",
            "Track in-use pointers",
            "Safe memory reclamation",
            "Handle thread exit cleanup"
        ],
        "Lock-free Hash Map": [
            "Implement lock-free hash map",
            "Handle bucket resizing",
            "Support concurrent read/write",
            "Benchmark throughput vs locks"
        ]
    },

    "log-aggregator": {
        "Log Ingestion": [
            "Receive logs via HTTP/TCP/UDP",
            "Parse multiple log formats (JSON, syslog, custom)",
            "Handle high ingestion rate",
            "Buffer logs during downstream outage"
        ],
        "Storage": [
            "Store logs with timestamp indexing",
            "Implement log compression",
            "Support retention policies",
            "Handle storage rotation"
        ],
        "Search": [
            "Full-text search in logs",
            "Filter by time range, level, source",
            "Support regex patterns",
            "Return results in time order"
        ]
    },

    "media-processing": {
        "Transcode Pipeline": [
            "Accept video upload in various formats",
            "Convert to target format/codec (H.264, H.265)",
            "Generate multiple quality levels",
            "Track transcode progress"
        ],
        "Thumbnail Generation": [
            "Extract frames at intervals",
            "Generate sprite sheets for scrubbing",
            "Support custom thumbnail times",
            "Handle various aspect ratios"
        ],
        "CDN Integration": [
            "Upload processed media to CDN/S3",
            "Generate signed URLs for access",
            "Support HLS manifest generation",
            "Handle CDN cache invalidation"
        ]
    },

    "metrics-collector": {
        "Metric Ingestion": [
            "Receive metrics in Prometheus format",
            "Support metric types (counter, gauge, histogram)",
            "Handle high cardinality labels",
            "Validate metric names and labels"
        ],
        "Storage": [
            "Store time-series data efficiently",
            "Implement data compression",
            "Support downsampling for old data",
            "Handle storage rotation"
        ],
        "Query Engine": [
            "Support PromQL-like queries",
            "Implement aggregation functions (sum, avg, max)",
            "Support range queries",
            "Handle label matching"
        ],
        "Alerting Integration": [
            "Define alert rules on metrics",
            "Evaluate rules periodically",
            "Send alerts to configured receivers",
            "Support alert silencing"
        ]
    },

    "ml-model-serving": {
        "Model Loading & Inference": [
            "Load models from different frameworks (PyTorch, TensorFlow, ONNX)",
            "Run inference with input validation",
            "Handle model warmup",
            "Support GPU inference"
        ],
        "Request Batching": [
            "Batch incoming requests for efficiency",
            "Configure batch size and timeout",
            "Handle partial batch execution",
            "Track batching metrics"
        ],
        "Model Versioning": [
            "Store multiple model versions",
            "Support version-specific inference",
            "Handle model hot-swapping",
            "Track version metadata"
        ],
        "A/B Testing & Canary": [
            "Split traffic between model versions",
            "Configure traffic percentages",
            "Track metrics per version",
            "Support gradual rollout"
        ],
        "Monitoring & Observability": [
            "Track inference latency",
            "Monitor prediction distribution",
            "Detect data/model drift",
            "Alert on anomalies"
        ]
    },

    "mlops-platform": {
        "Experiment Tracking": [
            "Log experiment parameters",
            "Track metrics over training",
            "Store artifacts (models, plots)",
            "Compare experiments"
        ],
        "Model Registry": [
            "Register trained models with metadata",
            "Version models with stage transitions",
            "Store model lineage (data, code)",
            "Support model search and discovery"
        ],
        "Training Pipeline": [
            "Define training steps in DAG",
            "Execute on compute infrastructure",
            "Track resource usage",
            "Support distributed training"
        ],
        "Model Deployment": [
            "Deploy model as API endpoint",
            "Support canary deployments",
            "Handle autoscaling",
            "Integrate with inference servers"
        ],
        "Model Monitoring": [
            "Track prediction metrics in production",
            "Detect data drift",
            "Alert on model degradation",
            "Support A/B testing analysis"
        ]
    },

    "multi-tenant-saas": {
        "Tenant Data Model": [
            "Add tenant_id to all data tables",
            "Implement tenant context in requests",
            "Support tenant-specific configurations",
            "Handle tenant creation/deletion"
        ],
        "Request Context & Isolation": [
            "Extract tenant from request (subdomain, header, JWT)",
            "Inject tenant context into database queries",
            "Prevent cross-tenant data access",
            "Log tenant context for auditing"
        ],
        "Row-Level Security": [
            "Implement RLS policies in database",
            "Test isolation with multiple tenants",
            "Handle admin/superuser access",
            "Verify no data leakage in queries"
        ],
        "Tenant Customization": [
            "Support per-tenant feature flags",
            "Allow tenant branding (logo, colors)",
            "Handle tenant-specific integrations",
            "Manage tenant settings UI"
        ],
        "Usage Tracking & Billing": [
            "Track usage metrics per tenant",
            "Implement usage-based billing",
            "Generate invoices",
            "Handle quota enforcement"
        ]
    },

    "multiplayer-game-server": {
        "Game State Management": [
            "Store authoritative game state on server",
            "Handle state updates from clients",
            "Implement game loop tick rate",
            "Support state snapshots"
        ],
        "Network Synchronization": [
            "Broadcast state to connected clients",
            "Implement delta compression",
            "Handle client interpolation/extrapolation",
            "Support lag compensation"
        ],
        "Player Management": [
            "Handle player join/leave",
            "Implement matchmaking",
            "Support lobbies and rooms",
            "Handle player disconnection"
        ],
        "Anti-cheat": [
            "Validate client inputs server-side",
            "Detect speed/teleport hacks",
            "Implement rate limiting",
            "Log suspicious behavior"
        ]
    },

    "notification-service": {
        "Channel Abstraction": [
            "Support multiple channels (email, SMS, push)",
            "Abstract channel interface",
            "Handle channel-specific formatting",
            "Support channel preferences per user"
        ],
        "Template System": [
            "Define notification templates",
            "Support variable substitution",
            "Handle localization",
            "Preview templates before sending"
        ],
        "Delivery Pipeline": [
            "Queue notifications for delivery",
            "Handle retries on failure",
            "Track delivery status",
            "Respect rate limits"
        ],
        "User Preferences": [
            "Store notification preferences per user",
            "Support opt-out/unsubscribe",
            "Handle quiet hours",
            "Aggregate frequent notifications"
        ]
    },

    "oauth2-provider": {
        "Authorization Endpoint": [
            "Handle authorization code flow",
            "Support PKCE for public clients",
            "Generate authorization codes",
            "Handle user consent UI"
        ],
        "Token Endpoint": [
            "Exchange code for tokens",
            "Issue access and refresh tokens",
            "Support client credentials flow",
            "Handle token refresh"
        ],
        "Token Validation": [
            "Validate JWT tokens",
            "Support token introspection endpoint",
            "Handle token revocation",
            "Check scopes and claims"
        ],
        "Client Management": [
            "Register OAuth clients",
            "Store client credentials securely",
            "Support redirect URI validation",
            "Handle client types (confidential, public)"
        ]
    },

    "order-matching-engine": {
        "Order Book Data Structure": [
            "Implement price-level ordered structure",
            "Support bid and ask sides",
            "Handle price-time priority",
            "Efficient insertion and removal"
        ],
        "Order Operations": [
            "Handle limit order placement",
            "Support market orders",
            "Implement order cancellation",
            "Handle order modification"
        ],
        "Matching Engine": [
            "Match orders by price-time priority",
            "Generate trade executions",
            "Handle partial fills",
            "Update order book after match"
        ],
        "Concurrency & Performance": [
            "Handle concurrent order submissions",
            "Achieve low latency (<1ms per order)",
            "Support high throughput (10K+ orders/sec)",
            "Implement lock-free where possible"
        ],
        "Market Data & API": [
            "Publish order book snapshots",
            "Stream trade executions",
            "Provide REST/WebSocket API",
            "Handle market data subscriptions"
        ]
    },

    "payment-gateway": {
        "Payment Processing": [
            "Create payment intents",
            "Handle card tokenization",
            "Process payments through provider",
            "Support multiple currencies"
        ],
        "Transaction State": [
            "Track payment states (pending, success, failed)",
            "Implement idempotency keys",
            "Handle payment webhooks",
            "Support refunds"
        ],
        "Fraud Prevention": [
            "Implement basic fraud checks",
            "Support 3D Secure",
            "Rate limit by IP/card",
            "Flag suspicious transactions"
        ],
        "PCI Compliance": [
            "Never store raw card numbers",
            "Use tokenization for card data",
            "Implement proper logging (no PAN)",
            "Secure data transmission (TLS)"
        ]
    },

    "profiler": {
        "Stack Sampling": [
            "Sample call stacks at regular intervals",
            "Capture both user and kernel stacks",
            "Handle different sampling rates",
            "Support process and thread targeting"
        ],
        "Symbol Resolution": [
            "Load symbol tables from binaries",
            "Resolve addresses to function names",
            "Handle DWARF debug info",
            "Support JIT symbol resolution"
        ],
        "Flame Graph Generation": [
            "Aggregate samples by call stack",
            "Generate SVG flame graph",
            "Support interactive zoom/search",
            "Handle inverted flame graphs"
        ],
        "Memory Profiling": [
            "Track heap allocations",
            "Identify allocation call sites",
            "Detect memory leaks",
            "Generate allocation flame graph"
        ]
    },

    "rate-limiter-distributed": {
        "Algorithm Implementation": [
            "Implement sliding window counter",
            "Store counters in Redis",
            "Handle clock skew between nodes",
            "Support different time windows"
        ],
        "Coordination": [
            "Share rate limit state across nodes",
            "Handle Redis connection failures",
            "Support local fallback",
            "Implement eventual consistency"
        ]
    },

    "rbac-system": {
        "Role Definition": [
            "Define roles with permissions",
            "Support role hierarchy",
            "Handle role assignment to users",
            "Implement role templates"
        ],
        "Permission Checks": [
            "Evaluate access for user/resource/action",
            "Support resource-level permissions",
            "Cache permission evaluations",
            "Handle permission inheritance"
        ],
        "Policy Engine": [
            "Define policies in declarative format",
            "Support ABAC attributes",
            "Evaluate policies efficiently",
            "Log access decisions"
        ],
        "Admin Interface": [
            "UI for role management",
            "Audit role changes",
            "Support bulk user assignment",
            "Generate access reports"
        ]
    },

    "reverse-proxy": {
        "HTTP Proxy Core": [
            "Accept HTTP requests",
            "Forward to backend servers",
            "Return backend response to client",
            "Handle HTTP/1.1 and HTTP/2"
        ],
        "Load Balancing": [
            "Implement round-robin distribution",
            "Support weighted distribution",
            "Health check backends",
            "Remove unhealthy backends from pool"
        ],
        "Connection Pooling": [
            "Maintain persistent connections to backends",
            "Implement connection pool per backend",
            "Handle connection timeouts",
            "Support connection limits"
        ],
        "Caching": [
            "Cache responses based on headers",
            "Implement cache key generation",
            "Support cache invalidation",
            "Respect Cache-Control directives"
        ],
        "SSL Termination": [
            "Terminate TLS at proxy",
            "Support SNI for multiple domains",
            "Forward to backends over HTTP",
            "Handle certificate reload"
        ]
    },

    "saga-orchestrator": {
        "Saga Definition": [
            "Define saga steps with compensating actions",
            "Support step dependencies",
            "Handle step timeout configuration",
            "Store saga definitions"
        ],
        "Execution Engine": [
            "Execute saga steps in order",
            "Track execution state",
            "Handle step failures",
            "Trigger compensation on failure"
        ],
        "Compensation": [
            "Execute compensating actions in reverse order",
            "Handle compensation failures",
            "Support retry for compensation",
            "Log compensation status"
        ]
    },

    "search-engine": {
        "Inverted Index": [
            "Build inverted index from documents",
            "Support index updates (add, delete)",
            "Handle large vocabularies",
            "Implement index compression"
        ],
        "Query Processing": [
            "Parse search queries",
            "Support boolean operators (AND, OR, NOT)",
            "Handle phrase queries",
            "Support fuzzy matching"
        ],
        "Ranking": [
            "Implement TF-IDF scoring",
            "Support BM25 ranking",
            "Handle field boosting",
            "Return ranked results"
        ],
        "Query Optimization": [
            "Skip list optimization",
            "Query plan optimization",
            "Cache frequent queries",
            "Handle pagination efficiently"
        ]
    },

    "secret-management": {
        "Secret Storage": [
            "Store secrets encrypted at rest",
            "Support multiple secret types (string, file, cert)",
            "Implement secret versioning",
            "Handle secret rotation"
        ],
        "Access Control": [
            "Define access policies for secrets",
            "Authenticate clients",
            "Log all access attempts",
            "Support temporary credentials"
        ],
        "Encryption": [
            "Use strong encryption (AES-256-GCM)",
            "Implement key hierarchy (master key, data keys)",
            "Support key rotation",
            "Handle key escrow/recovery"
        ],
        "Integration": [
            "Provide SDK for applications",
            "Support environment variable injection",
            "Implement Kubernetes integration",
            "Handle dynamic secret generation"
        ]
    },

    "serverless-runtime": {
        "Function Packaging": [
            "Package function code with dependencies",
            "Store function packages",
            "Support multiple runtimes (Python, Node, Go)",
            "Handle function versioning"
        ],
        "Execution Environment": [
            "Create isolated execution environment",
            "Inject function handler",
            "Enforce resource limits (memory, CPU)",
            "Handle function timeout"
        ],
        "Cold Start Optimization": [
            "Pre-warm execution environments",
            "Implement snapshot/restore",
            "Cache warm instances",
            "Track cold start metrics"
        ],
        "Request Routing": [
            "Route requests to function instances",
            "Handle concurrent requests",
            "Implement request queuing",
            "Support synchronous and async invocation"
        ],
        "Auto-Scaling": [
            "Scale instances based on request rate",
            "Implement scale-to-zero",
            "Handle scaling limits",
            "Track scaling metrics"
        ]
    },

    "service-mesh": {
        "Sidecar Proxy": [
            "Intercept inbound/outbound traffic",
            "Handle protocol parsing (HTTP, gRPC)",
            "Implement connection pooling",
            "Support transparent proxy mode"
        ],
        "Service Discovery": [
            "Integrate with service registry",
            "Handle endpoint updates",
            "Support DNS-based discovery",
            "Cache service endpoints"
        ],
        "Traffic Management": [
            "Implement load balancing",
            "Support traffic splitting",
            "Handle retries and timeouts",
            "Implement circuit breaking"
        ],
        "Observability": [
            "Generate distributed traces",
            "Collect metrics per service",
            "Export to observability backends",
            "Support access logging"
        ]
    },

    "session-management": {
        "Session Creation": [
            "Generate secure session ID",
            "Store session data server-side",
            "Set secure cookie attributes (HttpOnly, Secure, SameSite)",
            "Handle session binding to client"
        ],
        "Session Storage": [
            "Support multiple backends (memory, Redis, DB)",
            "Implement session expiration",
            "Handle session cleanup",
            "Support distributed sessions"
        ],
        "Security": [
            "Prevent session fixation",
            "Regenerate session ID on auth state change",
            "Implement CSRF protection",
            "Detect session hijacking"
        ]
    },

    "stream-processing-engine": {
        "Stream Abstraction": [
            "Define stream data structure",
            "Support operators (map, filter, flatMap)",
            "Handle operator chaining",
            "Implement parallel processing"
        ],
        "Windowing": [
            "Support tumbling windows",
            "Support sliding windows",
            "Support session windows",
            "Handle window triggers"
        ],
        "Event Time & Watermarks": [
            "Track event time vs processing time",
            "Generate watermarks",
            "Handle late events",
            "Support allowed lateness"
        ],
        "Stateful Processing": [
            "Maintain operator state",
            "Implement checkpointing",
            "Support state recovery",
            "Handle state migration"
        ],
        "Exactly-Once Semantics": [
            "Implement idempotent sinks",
            "Support transactional writes",
            "Handle checkpoint barriers",
            "Verify no duplicates on recovery"
        ]
    },

    "subscription-billing": {
        "Plan Management": [
            "Define subscription plans with pricing",
            "Support tiered and usage-based pricing",
            "Handle plan versioning",
            "Support promotional pricing"
        ],
        "Subscription Lifecycle": [
            "Create subscriptions for customers",
            "Handle upgrades and downgrades",
            "Process cancellations",
            "Handle subscription pausing"
        ],
        "Invoice Generation": [
            "Generate invoices at billing cycle",
            "Calculate prorated charges",
            "Handle credits and discounts",
            "Generate PDF invoices"
        ],
        "Dunning Management": [
            "Retry failed payments",
            "Send payment failure notifications",
            "Handle payment grace period",
            "Process subscription cancellation on non-payment"
        ]
    },

    "time-series-db": {
        "Data Model": [
            "Define time series with timestamp and values",
            "Support tags/labels for series",
            "Handle high cardinality efficiently",
            "Implement data types (float, int, bool)"
        ],
        "Ingestion": [
            "Accept time series data points",
            "Batch writes for performance",
            "Handle out-of-order data",
            "Support multiple write protocols"
        ],
        "Compression": [
            "Implement timestamp delta encoding",
            "Compress values (Gorilla or similar)",
            "Track compression ratios",
            "Handle decompression on read"
        ],
        "Query Engine": [
            "Support time range queries",
            "Implement aggregation functions",
            "Support downsampling",
            "Handle complex filter expressions"
        ],
        "Retention": [
            "Implement TTL for data",
            "Support retention policies",
            "Handle data compaction",
            "Manage storage lifecycle"
        ]
    },

    "vector-database": {
        "Vector Storage": [
            "Store vectors with metadata",
            "Support multiple vector dimensions",
            "Handle vector normalization",
            "Implement efficient storage layout"
        ],
        "Distance Metrics": [
            "Implement cosine similarity",
            "Support Euclidean distance",
            "Support dot product",
            "Handle metric-specific optimizations"
        ],
        "Brute Force Search": [
            "Implement exact nearest neighbor search",
            "Support top-k queries",
            "Handle large datasets efficiently",
            "Use as baseline for ANN accuracy"
        ],
        "HNSW Index": [
            "Implement HNSW graph construction",
            "Support configurable M and ef parameters",
            "Handle incremental inserts",
            "Achieve sub-linear search complexity"
        ],
        "Query API & Server": [
            "REST/gRPC API for vector operations",
            "Support batch queries",
            "Handle metadata filtering",
            "Implement hybrid search (vector + keyword)"
        ]
    },

    "webhook-delivery": {
        "Webhook Registration": [
            "Store webhook endpoints per customer",
            "Validate endpoint URLs",
            "Support event type filtering",
            "Generate webhook secrets for signing"
        ],
        "Event Dispatch": [
            "Queue events for delivery",
            "Sign payloads with HMAC",
            "Include timestamp for replay prevention",
            "Respect delivery order"
        ],
        "Retry & Reliability": [
            "Retry failed deliveries with backoff",
            "Handle endpoint timeouts",
            "Disable endpoints after repeated failures",
            "Log delivery attempts"
        ],
        "Monitoring": [
            "Track delivery success rate",
            "Provide delivery logs per webhook",
            "Alert on high failure rates",
            "Support manual retry"
        ]
    },

    "websocket-server": {
        "Connection Handling": [
            "Handle WebSocket upgrade request",
            "Maintain connection state",
            "Implement ping/pong heartbeat",
            "Handle graceful disconnect"
        ],
        "Message Processing": [
            "Parse incoming messages",
            "Support text and binary frames",
            "Handle message fragmentation",
            "Validate message format"
        ],
        "Broadcasting": [
            "Publish messages to multiple clients",
            "Support rooms/channels",
            "Handle subscription management",
            "Implement efficient fan-out"
        ],
        "Scalability": [
            "Support multiple server instances",
            "Use pub/sub for cross-instance messaging",
            "Handle connection migration",
            "Track connection metrics"
        ]
    },

    "workflow-orchestrator": {
        "DAG Definition": [
            "Define tasks with dependencies",
            "Parse DAG from config file",
            "Validate DAG has no cycles",
            "Support task parameters"
        ],
        "Scheduler": [
            "Trigger DAG runs based on schedule",
            "Support cron schedules",
            "Handle manual triggers",
            "Track scheduled vs actual run time"
        ],
        "Task Execution": [
            "Execute ready tasks (dependencies met)",
            "Handle task success/failure",
            "Support task retries",
            "Implement task timeout"
        ],
        "Worker & Executor": [
            "Distribute tasks to workers",
            "Support different executors (local, container)",
            "Handle worker failures",
            "Track task assignments"
        ],
        "Web UI & Monitoring": [
            "Display DAG graph",
            "Show run history",
            "View task logs",
            "Trigger manual runs"
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    expert_projects = data.get('expert_projects', {})
    updated = 0
    skipped = 0

    for project_id, criteria_by_milestone in ACCEPTANCE_CRITERIA.items():
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
                else:
                    skipped += 1
            else:
                print(f"  Milestone not found in criteria: {project_id}/{milestone_name}")

    data['expert_projects'] = expert_projects

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nUpdated: {updated} milestones")
    print(f"Skipped (already had criteria): {skipped}")


if __name__ == "__main__":
    main()

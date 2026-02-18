# AUDIT & FIX: build-ci-system

## CRITIQUE
- **Missing workspace cleanup and artifact retention**: The audit is correct. Without cleanup policies, disk space exhaustion is a certainty in any real CI system. No milestone addresses this at all.
- **Missing caching layer**: Build caching (dependency caches, Docker layer caching) is THE primary performance optimization in CI. Its absence makes the project unrealistic and incomplete.
- **Environment variable persistence between steps**: The AC says steps run sequentially as shell commands, but doesn't address whether each step is a separate `docker exec`, a single shell session, or separate containers. In real CI systems (GitHub Actions, GitLab CI), each step shares the same filesystem but may or may not share environment variables. This ambiguity is a critical design gap.
- **Secret masking is underspecified**: M2 says 'without leaking to logs' but doesn't specify the mechanism. Real CI systems scan output streams for known secret values and replace them with `***`. This is non-trivial and needs its own AC.
- **Missing git clone step**: No milestone describes how the source code actually gets into the container. In real CI, there's typically a checkout step or the workspace is mounted. This is fundamental.
- **Webhook replay protection missing**: M3 mentions 'replay attacks' as a pitfall but there's no AC requiring idempotency keys or timestamp-based replay rejection.
- **Dashboard milestone (M4) is too broad**: Combining real-time log streaming (WebSocket), build history, DAG visualization, and SVG badge generation in one milestone is unrealistic for the estimated hours.
- **Missing cancellation**: No mechanism to cancel a running pipeline. This is essential for any CI system.
- **Priority scheduling AC is ambitious but underspecified**: 'Critical pipelines processed before lower-priority jobs' needs a concrete mechanism (priority queue, separate queues, preemption?).
- **Pitfalls are too terse throughout**: 'Container cleanup on failure' doesn't explain the specific trap (orphaned containers, dangling volumes, zombie processes).

## FIXED YAML
```yaml
id: build-ci-system
name: Build Your Own CI System
description: "A pipeline executor with container isolation, webhook triggers, caching, real-time logs, and a web dashboard."
difficulty: expert
estimated_hours: "60-90"
essence: >
  Event-driven orchestration of isolated container workloads through webhook-triggered
  job queuing, YAML DSL parsing with schema validation, container-based execution
  with shared workspace and environment propagation, build caching, and distributed
  execution with real-time log streaming and state management.
why_important: >
  Building a CI system exposes you to production-grade distributed systems architecture,
  container orchestration, event-driven design, and the full DevOps lifecycle—skills
  fundamental to modern infrastructure engineering.
learning_outcomes:
  - Implement YAML schema validation and DAG construction for pipeline configuration
  - Design container-based job isolation with workspace mounting and environment propagation
  - Build webhook receivers with HMAC signature verification and replay protection
  - Implement job queues with priority scheduling and distributed task delivery
  - Design real-time log streaming from containers to web clients with secret masking
  - Build idempotent job executors with retry logic, timeout handling, and cancellation
  - Implement artifact storage, build caching, and retention policies
  - Design workspace cleanup to prevent disk exhaustion on workers
skills:
  - Container Orchestration
  - Message Queue Systems
  - Webhook Processing
  - YAML Schema Validation
  - Distributed Job Scheduling
  - Real-time Log Streaming
  - Event-Driven Architecture
  - Process Isolation
  - Build Caching Strategies
tags:
  - agents
  - artifacts
  - build-from-scratch
  - caching
  - expert
  - go
  - pipelines
  - python
  - rust
  - triggers
architecture_doc: architecture-docs/build-ci-system/index.md
languages:
  recommended:
    - Go
    - Python
    - Rust
  also_possible:
    - JavaScript
    - Java
resources:
  - type: article
    name: How GitHub Actions Works
    url: https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions
  - type: article
    name: Building a CI Server
    url: https://blog.boot.dev/education/build-ci-cd-server/
  - type: documentation
    name: Jenkins Architecture
    url: https://www.jenkins.io/doc/developer/architecture/
prerequisites:
  - type: skill
    name: Docker/containers (docker run, docker exec, volume mounts)
  - type: skill
    name: Git hooks/webhooks
  - type: skill
    name: Process management
  - type: skill
    name: Queue systems (conceptual)
milestones:
  - id: build-ci-system-m1
    name: "Pipeline Configuration Parser"
    description: >
      Parse YAML pipeline definitions into a validated DAG of stages and jobs, with
      environment variable substitution, conditional execution, and matrix expansion.
    acceptance_criteria:
      - "YAML pipeline files are parsed into a structured pipeline object with stages, jobs, and steps; invalid YAML produces clear schema validation errors with line numbers"
      - "Job dependency declarations are validated as a DAG; circular dependencies are detected and rejected with a descriptive error message naming the cycle"
      - Environment variables are substituted using ${VAR} and $VAR syntax in step commands, with support for default values (${VAR: -default}) and error on undefined (${VAR:?error})
      - Conditional expressions (if: branch == 'main', if: event == 'pull_request') are parsed and evaluated, controlling whether a step or job is included in the execution plan
      - "Matrix builds expand axis definitions into the cartesian product of job configurations; 'exclude' rules filter out specific invalid combinations before execution"
      - "A 'services' block defines sidecar containers (e.g., postgres, redis) that run alongside the job container on the same network"
    pitfalls:
      - Circular dependencies: a naive topological sort will infinite loop or silently produce wrong order. Use Kahn's algorithm with explicit cycle detection.
      - Matrix combinatorial explosion: a 5x5x5 matrix generates 125 jobs. Implement a hard cap and warn the user.
      - YAML type coercion: YAML silently converts 'on' to boolean True and '3.10' to float 3.1. Use safe_load and explicit string types for branch names and version numbers.
      - Environment variable injection attacks: a step command containing ${USER_INPUT} with malicious content (e.g., '; rm -rf /') enables command injection. Sanitize or quote substitutions.
    concepts:
      - YAML parsing and schema validation
      - DAG construction and topological sort
      - Template substitution
      - Conditional expression evaluation
    skills:
      - YAML schema design and validation (JSON Schema or custom)
      - Directed acyclic graph algorithms
      - Expression parser for conditionals
      - Cartesian product generation with filtering
    deliverables:
      - YAML pipeline parser with schema validation and clear error messages
      - DAG builder with cycle detection for job dependencies
      - Environment variable substitution engine with default and error modes
      - Matrix expansion engine with exclude rules
      - Conditional expression evaluator for if-clauses
    estimated_hours: "10-14"

  - id: build-ci-system-m2
    name: "Job Execution Engine"
    description: >
      Execute pipeline jobs in isolated Docker containers with workspace mounting,
      environment propagation between steps, secret masking, and git checkout.
    acceptance_criteria:
      - "Each job runs in a fresh Docker container using the image specified in the pipeline config; the container is guaranteed to be removed after execution (pass or fail) to prevent resource leaks"
      - "A git clone/checkout step populates the workspace with the repository at the triggered commit SHA before any user-defined steps run"
      - "Steps within a single job execute sequentially inside the same container, sharing the same filesystem (workspace) and able to propagate environment variables to subsequent steps via a defined mechanism (e.g., writing to a shared env file)"
      - The job fails immediately on the first non-zero exit code from a step, unless the step is marked with 'continue-on-error: true'
      - Secret values injected as environment variables are masked in all log output: any occurrence of a secret value in stdout/stderr is replaced with '***' before storage or streaming
      - "Each step's stdout and stderr are captured with timestamps and stored for later retrieval; output exceeding a configurable limit (default 10MB per step) is truncated with a warning"
      - "Jobs that exceed a configurable timeout are killed (SIGTERM, then SIGKILL after grace period) and marked as TIMED_OUT"
      - "Sidecar service containers (defined in the 'services' block) start before the job and are torn down after it completes"
    pitfalls:
      - Container cleanup: if the process crashes between container creation and cleanup registration, orphaned containers accumulate. Use Docker labels and a periodic reaper process.
      - Environment propagation: each 'docker exec' call starts a new shell, losing exports from previous steps. Use a shared .env file that each step sources, or run all steps in a single shell script.
      - Secret masking in binary output: scanning for secret strings in potentially binary stdout is fragile. Only mask in text mode and skip binary artifact streams.
      - Timeout race condition: if SIGTERM is sent right as a step completes, the exit code may be ambiguous. Record the timeout event explicitly rather than relying on exit codes alone.
    concepts:
      - Container isolation and lifecycle
      - Workspace management
      - Environment propagation
      - Secret masking
    skills:
      - Docker API (container create, start, exec, cp, rm)
      - Process lifecycle management and signal handling
      - Stream processing for log capture and masking
      - Resource limit enforcement (timeout, memory, disk)
    deliverables:
      - Docker container lifecycle manager (create, run steps, cleanup)
      - Git checkout step populating the workspace
      - Sequential step runner with environment propagation and exit code handling
      - Secret masking filter on log output streams
      - Timeout enforcement with graceful shutdown
    estimated_hours: "14-20"

  - id: build-ci-system-m3
    name: "Caching, Artifacts & Workspace Management"
    description: >
      Implement build caching for dependency restoration, artifact collection and
      storage, and workspace/resource cleanup with retention policies.
    acceptance_criteria:
      - "Cache keys are computed from a user-defined expression (e.g., hash of lockfile); on cache hit, the cached directory is restored into the workspace before steps run, saving build time"
      - "On cache miss, the specified paths are archived after the job completes and stored with the computed cache key for future runs"
      - "Artifacts specified by glob patterns are collected from the job workspace after all steps complete and uploaded to persistent storage with metadata (job ID, timestamp, size)"
      - "Artifacts from upstream jobs can be downloaded into downstream jobs, enabling cross-job data passing within a pipeline"
      - "Retention policy automatically deletes artifacts and caches older than a configurable age (default 30 days) via a background cleanup job"
      - "Workspace directories on workers are cleaned up after job completion; a periodic garbage collector removes any orphaned workspaces older than 24 hours"
      - "Storage usage metrics (total artifact size, cache hit rate, disk usage per worker) are exposed via an API endpoint"
    pitfalls:
      - Cache poisoning: a malicious branch can write a bad cache that poisons builds on the main branch. Scope cache keys by branch or use content-addressable storage.
      - Glob pattern edge cases: '**/*.jar' may follow symlinks or match inside node_modules, producing multi-GB artifacts. Set a max artifact size.
      - Race condition in cache writes: two concurrent jobs with the same cache key may write simultaneously, corrupting the cache. Use atomic rename or locking.
      - Disk exhaustion: without the cleanup job running, a busy CI system fills disks within days. Make cleanup a first-class concern, not an afterthought.
    concepts:
      - Content-addressable caching
      - Artifact lifecycle management
      - Retention policies
      - Resource cleanup
    skills:
      - Cache key computation and content hashing
      - File archiving (tar, zip) and extraction
      - Background job scheduling for cleanup
      - Storage abstraction (local, S3-compatible)
    deliverables:
      - Build cache system with key computation, save, and restore operations
      - Artifact collection from glob patterns with upload to persistent storage
      - Cross-job artifact download for downstream dependencies
      - Retention policy enforcer with configurable age-based cleanup
      - Worker workspace garbage collector
    estimated_hours: "10-14"

  - id: build-ci-system-m4
    name: "Webhook Triggers & Job Queue"
    description: >
      Handle incoming webhooks with signature verification and replay protection,
      queue pipeline runs with priority scheduling, and distribute work to a pool
      of concurrent workers.
    acceptance_criteria:
      - "Webhook receiver validates HMAC-SHA256 signatures on incoming payloads using the configured secret; requests with invalid or missing signatures are rejected with 401"
      - "Replay protection rejects webhook deliveries with timestamps older than 5 minutes or with a previously-seen delivery ID (idempotency)"
      - "Webhook handler parses GitHub push, pull_request, and tag event payloads and matches them against pipeline trigger configurations to determine which pipelines to enqueue"
      - "Job queue stores pending pipeline runs durably (database or Redis) with at-least-once delivery semantics; a crashed worker's job is re-queued after a visibility timeout"
      - "Worker pool dequeues and executes pipeline jobs with configurable maximum concurrency; jobs exceeding the limit wait in the queue"
      - Priority scheduling: jobs can be assigned priority levels (e.g., production deploy > feature branch); higher-priority jobs are dequeued before lower-priority ones even if enqueued later
      - Pipeline cancellation API: an API call or new push to the same branch cancels any in-progress or queued pipeline for the previous commit on that branch
      - Rate limiting: a configurable maximum number of pipeline triggers per repository per minute prevents webhook floods from overwhelming the system
    pitfalls:
      - Queue starvation: if high-priority jobs are continuously enqueued, low-priority jobs may never execute. Implement aging that gradually increases priority over time.
      - "At-least-once delivery means jobs may run twice if a worker crashes after completion but before acknowledgment. Make pipeline execution idempotent or use fencing tokens."
      - Webhook secret rotation: changing the secret invalidates all in-flight webhook deliveries. Support multiple active secrets during rotation periods.
      - Cancellation race: cancelling a job that is between steps requires signaling the execution engine; if the engine doesn't poll for cancellation, the job runs to completion.
    concepts:
      - HMAC signature verification
      - Work queue with priority and visibility timeout
      - Idempotency and replay protection
      - Distributed worker pool
    skills:
      - HMAC-SHA256 signature verification
      - Durable queue implementation (Redis BRPOPLPUSH or database polling)
      - Worker pool management with concurrency limits
      - Rate limiting algorithms (token bucket or sliding window)
    deliverables:
      - Webhook receiver with HMAC verification and replay protection
      - Pipeline trigger matcher mapping events to pipeline configurations
      - Priority job queue with durable storage and at-least-once delivery
      - Worker pool with configurable concurrency and job cancellation
      - Rate limiter for webhook ingestion
    estimated_hours: "12-16"

  - id: build-ci-system-m5
    name: "Web Dashboard & Real-time Logs"
    description: >
      Build a web dashboard showing build history, real-time log streaming,
      pipeline DAG visualization, and embeddable status badges.
    acceptance_criteria:
      - "Build list view shows recent pipeline runs with status (running/passed/failed/cancelled), trigger type, branch, commit SHA, and wall-clock duration; paginated for large histories"
      - Real-time log streaming: clicking on a running job displays live log output in the browser within 2 seconds of it being produced, using WebSocket or Server-Sent Events
      - "Completed job logs are served from storage with support for searching within log output and jumping to specific steps"
      - "Pipeline DAG visualization renders stages and jobs as a graph with directed edges showing dependencies; node color reflects job status (green/red/gray/spinning)"
      - "Build status badges return SVG images (pass/fail/running) at a stable URL suitable for embedding in README files; badges are served with appropriate Cache-Control headers (short TTL)"
      - "Artifact download page lists collected artifacts for each pipeline run with file names, sizes, and download links"
      - Manual pipeline trigger: a button or API allows triggering a pipeline run for a specific branch/commit without a webhook event
    pitfalls:
      - WebSocket connection management: reconnecting after network disruption must resume from the last received log offset, not from the beginning, to avoid duplicates or gaps.
      - Large log rendering: naively appending millions of log lines to the DOM causes the browser to freeze. Use virtualized scrolling (render only visible lines).
      - Badge caching: CDN or browser caching badges for too long shows stale status. Use Cache-Control: no-cache or short max-age with ETag validation.
      - SVG injection: if badge text includes user-controlled data (branch names), it must be XML-escaped to prevent SVG injection attacks.
    concepts:
      - Real-time streaming protocols
      - DAG visualization
      - SVG generation
      - Pagination and virtual scrolling
    skills:
      - WebSocket or SSE implementation
      - Frontend rendering with virtual scrolling
      - SVG generation and XML escaping
      - Cursor-based pagination
      - REST API design
    deliverables:
      - Build history page with status, trigger, branch, duration, and pagination
      - Real-time log streaming viewer with reconnection and offset tracking
      - Pipeline DAG visualization with live status updates
      - SVG status badges with proper caching headers
      - Artifact download page
      - Manual pipeline trigger endpoint
    estimated_hours: "12-18"
```
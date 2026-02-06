# Distributed Job Scheduler: Design Document


## Overview

A distributed job scheduler that executes tasks across multiple workers using cron expressions for timing, with priorities, retries, and fault-tolerant coordination. The key architectural challenge is maintaining consistency and preventing duplicate execution while ensuring high availability and fair job distribution across a dynamic cluster of workers.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** This section provides foundational understanding for all three milestones by establishing the need for distributed job scheduling and the technical challenges that inform our architectural decisions.

### Problem Overview

Think of distributed job scheduling as managing a city's public transportation system. Just as a city needs buses to run on time, reach every neighborhood, and continue operating even when some vehicles break down, modern applications need **scheduled tasks** to execute reliably across multiple machines, handle varying workloads, and maintain service availability despite individual server failures.

In a traditional single-server environment, job scheduling is straightforward—you run a cron daemon on one machine, and it executes tasks at predetermined times. This works perfectly for simple scenarios, like a personal blog that generates a daily summary or a small business that backs up its database nightly. However, as systems scale and reliability requirements increase, this centralized approach reveals critical limitations.

Consider an e-commerce platform that needs to process millions of scheduled tasks daily: sending abandoned cart reminders, generating analytics reports, processing subscription renewals, cleaning up temporary files, and synchronizing inventory across multiple warehouses. A single-server scheduler becomes a **single point of failure**—if that machine crashes, all scheduled work stops. Even worse, the machine might become overwhelmed by the sheer volume of tasks, leading to delays, missed schedules, and cascading failures throughout the system.

The distributed job scheduler we're building addresses these challenges by spreading the work across multiple **worker nodes**, much like how a transit system uses multiple buses to serve a city. When one bus breaks down, others continue running their routes. When passenger demand increases in certain areas, the system can deploy additional buses to those routes. Similarly, our scheduler can distribute jobs across healthy workers, automatically reassign work from failed nodes, and scale capacity by adding more worker machines.

The core insight is that **reliable scheduling at scale requires coordination**. Just as buses need a central dispatch system to prevent multiple buses from serving the same route simultaneously while ensuring no routes go unserved, distributed workers need coordination mechanisms to prevent duplicate job execution while guaranteeing that every scheduled task eventually runs.

However, this coordination introduces complexity that doesn't exist in single-server systems. We must solve fundamental distributed systems problems: How do multiple workers agree on who should execute a specific job? How do we detect when a worker has failed and reassign its work? How do we ensure that a job runs exactly once, even when network partitions occur or workers crash mid-execution? These challenges form the technical foundation for our entire design.

### Existing Approaches

The landscape of job scheduling solutions reflects different trade-offs between simplicity, reliability, and scalability. Understanding these existing approaches helps us appreciate why we need a custom distributed scheduler and informs our architectural decisions.

| Approach | Description | Strengths | Limitations |
|----------|-------------|-----------|-------------|
| **Centralized Schedulers** | Single coordinator manages all scheduling (cron, Jenkins, Airflow single-node) | Simple to reason about, strong consistency guarantees, easy debugging | Single point of failure, limited scalability, resource bottleneck |
| **Message Queue Systems** | Jobs queued in brokers, workers poll for tasks (RabbitMQ, AWS SQS + Lambda) | High throughput, natural load distribution, proven reliability | Limited scheduling flexibility, complex delay mechanisms, external dependencies |
| **Database-Driven Polling** | Jobs stored in DB tables, workers query for pending tasks | Simple implementation, leverages existing infrastructure, transaction support | Database becomes bottleneck, polling inefficiency, lock contention |
| **Distributed Coordination** | Multiple schedulers coordinate via consensus (Kubernetes CronJobs, distributed Quartz) | High availability, horizontal scalability, fault tolerance | Complex implementation, consensus overhead, split-brain risks |

**Centralized schedulers** like traditional cron or single-node Airflow offer the strongest consistency guarantees because there's only one decision-maker. When you schedule a job to run "every day at 2 AM," you can be absolutely certain it won't run multiple times because only one scheduler exists. These systems are also easier to debug—you have one log file, one configuration, and one point of control. However, they fail catastrophically when the central coordinator goes down, and they cannot scale beyond the resources of a single machine.

**Message queue approaches** excel at handling high-throughput job processing by naturally distributing work across multiple consumer workers. Systems like RabbitMQ with delayed message plugins or AWS SQS with visibility timeouts provide reliable job delivery guarantees. However, they struggle with complex scheduling requirements. Implementing a job that runs "every weekday at 9 AM except holidays" requires external scheduling logic to enqueue messages at the right times, and advanced features like job dependencies or conditional execution become cumbersome.

**Database-driven polling** represents a common middle-ground approach where jobs are stored as database records with scheduling metadata, and workers periodically query for pending tasks. This leverages existing database infrastructure and provides ACID transaction support for job state transitions. However, the database becomes a bottleneck as worker counts increase, and frequent polling creates unnecessary load. Additionally, implementing fair job distribution and preventing race conditions requires careful lock management.

**Distributed coordination approaches** like Kubernetes CronJobs or clustered Quartz schedulers provide both high availability and scalability by running multiple scheduler instances that coordinate through consensus protocols. These systems can survive individual node failures and distribute load across multiple coordinators. However, they introduce significant complexity in consensus algorithms, leader election, and split-brain prevention.

> **Design Insight**: Each approach optimizes for different priorities. Centralized systems prioritize correctness and simplicity. Message queues prioritize throughput and reliability. Database polling prioritizes implementation simplicity. Distributed coordination prioritizes availability and scalability. Our design must explicitly choose which properties matter most for our use case.

Our distributed job scheduler combines elements from multiple approaches. We use **distributed coordination** for high availability and scalability, but implement our own lightweight consensus mechanism tailored to scheduling workloads rather than adopting heavyweight protocols like Raft. We incorporate **message queue patterns** for reliable job delivery between coordinators and workers, but maintain scheduling logic within the scheduler rather than pushing complexity to external systems. We leverage **database storage** for job persistence and atomic state transitions, but minimize polling overhead through event-driven notifications.

> **Decision: Hybrid Architecture with Custom Coordination**
> - **Context**: Existing solutions either sacrifice availability for simplicity or introduce unnecessary complexity for general-purpose consensus
> - **Options Considered**: 
>   1. Pure message queue with external scheduling (simple but limited scheduling flexibility)
>   2. Database polling with advisory locks (simple but poor scaling characteristics)  
>   3. Full Raft consensus for all scheduling decisions (robust but heavyweight for our use case)
>   4. Custom coordination protocol optimized for scheduling workloads
> - **Decision**: Hybrid architecture with custom coordination protocol
> - **Rationale**: Scheduling workloads have specific characteristics (time-based triggers, priority ordering, delayed execution) that allow for simpler coordination than general-purpose consensus while still maintaining availability and consistency
> - **Consequences**: We gain scheduling-optimized performance and can implement exactly-once execution guarantees, but must implement and maintain our own coordination logic

### Technical Challenges

Building a distributed job scheduler requires solving three fundamental distributed systems problems: **consensus**, **failure detection**, and **exactly-once execution**. These challenges are interconnected—solving one affects how we approach the others—and their solutions form the core of our architectural design.

#### Consensus: Who Decides What Jobs Run When?

The consensus problem in distributed scheduling is like coordinating traffic at a busy intersection without traffic lights. Multiple scheduler nodes might simultaneously decide that a job scheduled for "2:00 PM" should execute now, leading to duplicate execution. Alternatively, in trying to avoid duplicates, all nodes might assume another node will handle the job, resulting in missed executions.

Traditional consensus algorithms like Raft or PBFT provide strong consistency guarantees by ensuring all nodes agree on a single sequence of operations. However, these algorithms are designed for general-purpose state machine replication and carry significant overhead for our scheduling use case. Consider a job scheduled to run every hour—full consensus for each execution decision would require multiple round-trips between nodes, potentially taking longer than the job execution itself.

Our scheduling workload has specific characteristics that allow for more efficient coordination:

1. **Time-based triggers**: Most decisions are triggered by time passage rather than external events
2. **Predictable scheduling**: Next execution times can be calculated in advance using cron expressions
3. **Priority ordering**: Jobs have explicit priority levels that provide natural ordering
4. **Delayed execution**: Jobs can wait in queues without immediate execution requirements

These characteristics enable a **leader-based coordination model** where one scheduler node acts as the primary coordinator for job scheduling decisions, while other nodes remain in standby mode ready to take over during failures. This approach reduces coordination overhead during normal operation while maintaining availability through leader election during failures.

| Consensus Challenge | Traditional Approach | Our Scheduling-Optimized Approach |
|---------------------|---------------------|-----------------------------------|
| **Job Execution Decisions** | Full consensus on each job execution | Leader decides, followers standby with leader election |
| **Clock Synchronization** | Complex vector clocks or logical timestamps | NTP synchronization with tolerance windows |
| **Schedule Conflicts** | Distributed lock acquisition per job | Centralized scheduling with atomic job claiming |
| **Split-Brain Prevention** | Quorum-based decisions requiring majority | Lease-based leadership with automatic expiration |

#### Failure Detection: When Has a Worker Actually Failed?

Failure detection in distributed systems is notoriously difficult because there's no reliable way to distinguish between a slow worker and a failed worker. Network partitions can make healthy workers appear dead, while zombie processes can continue sending heartbeats despite being unable to process jobs. This is known as the **failure detection impossibility** in asynchronous distributed systems.

In our job scheduler context, failure detection mistakes have serious consequences:

- **False positives** (marking healthy workers as failed) lead to unnecessary job reassignment and potential duplicate execution
- **False negatives** (not detecting actual failures) result in jobs being assigned to dead workers, causing missed executions and system-wide delays

Consider a worker executing a long-running data analysis job that takes 45 minutes to complete. If our failure detection timeout is set to 30 minutes, we'll incorrectly mark the worker as failed and potentially start duplicate execution on another worker. Conversely, if we set the timeout to 90 minutes to accommodate long jobs, actual worker failures won't be detected quickly enough, leaving jobs stuck in limbo.

Our approach uses a **multi-layered failure detection strategy** that combines multiple signals:

1. **Heartbeat timeouts**: Workers send periodic "I'm alive" messages with configurable timeout thresholds
2. **Job progress updates**: Workers report intermediate progress on long-running jobs to prove they're actively processing
3. **Lease-based job ownership**: Jobs are assigned with time-bounded leases that automatically expire, enabling recovery without explicit failure detection
4. **Graceful shutdown signaling**: Workers announce their intention to shut down, allowing clean job handoff

| Failure Detection Layer | Purpose | Timeout | False Positive Risk | False Negative Risk |
|-------------------------|---------|---------|-------------------|-------------------|
| **Heartbeat Messages** | Basic liveness detection | 30 seconds | Network partitions | Process zombies |
| **Job Progress Updates** | Active processing verification | 5 minutes | Long-running jobs | Infinite loops |
| **Lease Expiration** | Automatic job recovery | Job-specific | Resource constraints | Slow processing |
| **Graceful Shutdown** | Clean worker termination | 60 seconds | None | Sudden crashes |

#### Exactly-Once Execution: The Distributed Scheduling Holy Grail

Exactly-once execution is the most challenging problem in distributed job scheduling. Unlike at-least-once delivery (where duplicates are acceptable) or at-most-once delivery (where job loss is acceptable), exactly-once requires that every scheduled job runs precisely once, despite worker failures, network partitions, and coordinator crashes.

The fundamental challenge is that network failures can occur at any point during job execution, making it impossible to distinguish between these scenarios:

1. **Job never started**: The worker received the job assignment but crashed before beginning execution
2. **Job started but didn't finish**: The worker began execution but crashed mid-way through processing
3. **Job finished but acknowledgment was lost**: The worker completed the job successfully but the network failed before it could report completion
4. **Job finished and reported**: The worker completed the job and successfully reported completion, but a delayed retry caused duplicate assignment

Each scenario requires different recovery actions, but they're indistinguishable from the coordinator's perspective. Traditional approaches use **idempotency** (making duplicate executions safe) or **transactional semantics** (rolling back partial executions), but these place significant burden on job implementations.

Our scheduler implements exactly-once execution through a combination of techniques:

**Job Leasing with Fencing Tokens**: Each job assignment includes a unique, monotonically increasing fencing token. Workers must present valid tokens when reporting job results, preventing race conditions where slow workers report results after their jobs have been reassigned.

**Execution State Tracking**: Jobs progress through explicit states (`PENDING` → `CLAIMED` → `EXECUTING` → `COMPLETED`), with each transition recorded atomically in persistent storage. This provides a clear audit trail and enables precise recovery logic.

**Idempotency Key Deduplication**: Jobs include client-provided idempotency keys that prevent duplicate submissions. Even if the same cron schedule fires multiple times due to coordinator failures, duplicate jobs are silently ignored.

**Timeout-Based Recovery**: Jobs that remain in intermediate states (`CLAIMED`, `EXECUTING`) beyond their lease timeouts are automatically made available for reassignment, ensuring forward progress even when workers fail silently.

| Exactly-Once Challenge | Problem | Our Solution | Trade-offs |
|-------------------------|---------|--------------|------------|
| **Duplicate Submission** | Same cron schedule triggers multiple times | Idempotency key deduplication | Requires deterministic key generation |
| **Worker Failure Mid-Execution** | Unknown whether job completed before crash | Atomic state transitions + timeouts | Jobs must be idempotent or resumable |
| **Network Partition During Reporting** | Coordinator can't distinguish completion from failure | Fencing tokens prevent stale reports | Adds complexity to job result handling |
| **Coordinator Failure During Assignment** | Job assignment state may be lost or duplicated | Persistent job state + recovery scanning | Requires durable storage for all state |

⚠️ **Pitfall: Assuming Network Reliability**

A common mistake is designing the scheduler assuming that network operations either succeed completely or fail immediately. In reality, network operations can experience arbitrary delays, partial failures, or reordering. For example, a job assignment message might be delayed for several minutes due to network congestion, arriving after the job has already been reassigned to another worker due to timeout. This can lead to duplicate execution unless the system uses fencing tokens or similar disambiguation mechanisms.

The solution is to design all inter-component communication as potentially unreliable and implement disambiguation mechanisms (like fencing tokens) that allow recipients to determine whether received messages are current or stale.

> **Critical Design Principle**: Our distributed scheduler must degrade gracefully under various failure conditions. When consensus is impossible, we prefer conservative decisions (potentially missing a single job execution) over dangerous ones (definitely executing jobs multiple times). This bias toward safety over liveness reflects the reality that most scheduled jobs are periodic—missing one execution of an hourly job is recoverable, while duplicate executions often cause irreversible side effects.

These three technical challenges—consensus, failure detection, and exactly-once execution—drive every major architectural decision in our distributed job scheduler. The solutions we choose for each problem must work together cohesively, as optimizing for one often creates constraints for the others.

### Implementation Guidance

This foundational section doesn't include implementation code, but understanding the technical challenges above informs the technology choices we'll make throughout the project.

#### Technology Recommendations for Distributed Coordination

| Component | Simple Option | Advanced Option | Trade-offs |
|-----------|---------------|-----------------|------------|
| **Consensus/Coordination** | Redis with Lua scripts for atomic operations | etcd with Raft consensus for leader election | Redis simpler but less fault-tolerant; etcd more complex but industry-proven |
| **Job Persistence** | PostgreSQL with advisory locks | Redis Streams with consumer groups | PostgreSQL stronger consistency; Redis higher performance |
| **Worker Communication** | HTTP REST with polling | gRPC with streaming for real-time updates | REST simpler debugging; gRPC better performance |
| **Time Synchronization** | Basic system clock comparison | NTP daemon with drift monitoring | System clocks sufficient for most use cases; NTP required for tight timing |

#### Key Architectural Decisions to Make

As we progress through the implementation milestones, we'll need to make several critical decisions that balance the trade-offs identified in this section:

**Consistency vs. Availability**: When network partitions occur, do we prioritize ensuring no duplicate job execution (consistency) or ensuring jobs continue to be scheduled (availability)? Our design leans toward consistency for safety.

**Coordination Complexity**: Do we implement full consensus for all decisions, or use simpler leader-based coordination with the risk of brief unavailability during failover? We choose leader-based coordination for performance.

**Failure Detection Sensitivity**: Do we use aggressive timeouts that quickly detect failures but risk false positives, or conservative timeouts that reduce false positives but slow failure recovery? We use configurable timeouts that can be tuned per deployment.

**Storage Dependencies**: Do we require external coordination services like etcd, or can we build coordination on top of simpler storage like Redis or PostgreSQL? We support multiple storage backends with different consistency guarantees.

These decisions will become concrete as we implement each milestone, starting with the cron expression parser that must handle timezone-aware scheduling across distributed nodes.


## Goals and Non-Goals

> **Milestone(s):** This section establishes the scope and requirements for all three milestones by defining what the distributed job scheduler must accomplish and explicitly excluding complex enterprise features.

Think of defining project goals like planning a city's public transportation system. You need to be crystal clear about what routes you'll serve (functional goals), how many passengers you can handle and how fast they'll get there (non-functional goals), and what services you explicitly won't provide like door-to-door limousine service (non-goals). Without this clarity, you'll either build something that doesn't meet user needs or get overwhelmed trying to solve every possible problem at once.

The goals for our distributed job scheduler emerge from real-world requirements where organizations need to execute recurring tasks reliably across multiple machines. These goals directly map to our three milestones: parsing scheduling expressions, managing job priorities and queuing, and coordinating work across a cluster of machines. Equally important are the non-goals that keep this project feasible for implementation while still providing substantial learning value.

### Functional Goals

The functional goals define what the distributed job scheduler must actually do for users. These capabilities form the core user-facing behavior that determines whether the system succeeds or fails in practice.

**Cron Expression Scheduling** forms the foundation of time-based job execution. The scheduler must parse standard five-field cron expressions (`minute hour day-of-month month day-of-week`) and calculate the next execution time with perfect accuracy. This includes handling wildcards (`*`), ranges (`1-5`), step values (`*/15`), and specific lists (`1,3,5`). The system should also support common shorthand aliases like `@daily`, `@hourly`, and `@weekly` for user convenience.

> **Decision: Standard Five-Field Cron Support**
> - **Context**: Multiple cron formats exist, from traditional five fields to extended six-field versions with seconds
> - **Options Considered**: Five-field only, six-field with seconds, full Quartz-style expressions
> - **Decision**: Support standard five-field cron with common aliases
> - **Rationale**: Five-field cron covers 95% of real-world scheduling needs while keeping parsing complexity manageable for a learning project
> - **Consequences**: Users get familiar, well-documented syntax, but cannot schedule sub-minute tasks

| Cron Feature | Supported | Example | Description |
|--------------|-----------|---------|-------------|
| Wildcards | Yes | `* * * * *` | Every minute |
| Specific Values | Yes | `30 14 * * 1` | 2:30 PM every Monday |
| Ranges | Yes | `0 9-17 * * 1-5` | Every hour during business days |
| Step Values | Yes | `*/15 * * * *` | Every 15 minutes |
| Lists | Yes | `0 8,12,18 * * *` | 8 AM, noon, and 6 PM daily |
| Aliases | Yes | `@daily`, `@weekly` | Common scheduling shortcuts |

**Priority-Based Job Queuing** ensures that critical jobs execute before less important ones. The system must accept jobs with numeric priority levels (higher numbers indicate higher priority) and guarantee that a higher-priority job never waits behind a lower-priority job when workers are available. This requires a priority queue implementation that can handle thousands of pending jobs efficiently.

The queuing system must also support **delayed execution** where jobs submitted now don't become eligible for execution until a future timestamp. This enables both one-time scheduled jobs and recurring jobs generated from cron expressions. Jobs should remain invisible to workers until their scheduled time arrives.

**Deduplication** prevents the same job from being queued multiple times within a configurable time window. Each job submission includes an idempotency key, and the system silently ignores submissions with duplicate keys that haven't expired. This protects against client retry logic accidentally flooding the queue with identical work.

| Queue Operation | Behavior | Guarantees |
|----------------|----------|------------|
| Enqueue | Accept job with priority and schedule time | Atomic insertion with deduplication check |
| Dequeue | Return highest-priority ready job | Strict priority ordering among ready jobs |
| Delayed Jobs | Hold jobs until scheduled time | Jobs become visible exactly at schedule time |
| Deduplication | Reject jobs with duplicate idempotency keys | Configurable deduplication window (e.g., 1 hour) |

**Multi-Worker Coordination** distributes job execution across a cluster of worker machines while preventing duplicate execution. The system must implement distributed locking so that exactly one worker can claim each job. Workers use the `claimJob()` operation to atomically reserve a job for execution, receiving a unique fencing token that prevents stale operations from affecting job state.

The coordination layer tracks worker health through periodic `heartbeat()` signals. When a worker fails to send heartbeats within the configured timeout, the system automatically reassigns that worker's in-progress jobs to healthy workers. This ensures jobs complete even when individual machines crash or become unresponsive.

**Fault Tolerance** keeps the scheduler operating despite individual component failures. The system must detect worker failures through missed heartbeats and reassign their jobs to healthy workers. Job state persists in durable storage (Redis or similar) so that scheduler restarts don't lose pending work. The leader election mechanism ensures exactly one coordinator node manages job assignment even if multiple scheduler instances are running.

> **Decision: At-Least-Once Job Execution**
> - **Context**: Distributed systems can guarantee at-most-once or at-least-once execution, but not exactly-once without significant complexity
> - **Options Considered**: At-most-once (jobs may be lost), at-least-once (jobs may run twice), exactly-once (complex)
> - **Decision**: Provide at-least-once execution with idempotency key support
> - **Rationale**: Most job types can be made idempotent, and losing jobs is worse than occasionally running them twice
> - **Consequences**: Job implementations must handle duplicate execution gracefully, but no jobs are lost due to failures

| Fault Tolerance Feature | Recovery Time | Detection Method |
|------------------------|---------------|------------------|
| Worker Failure | 30-60 seconds | Missed heartbeat timeout |
| Coordinator Failure | 10-30 seconds | Leader election timeout |
| Storage Failure | Manual intervention | Connection errors and timeouts |
| Network Partition | Variable | Split-brain detection via consensus |

### Non-Functional Goals

The non-functional goals establish performance, scalability, and reliability targets that make the scheduler suitable for production-like workloads while remaining implementable as a learning project.

**Performance Targets** ensure the scheduler can handle realistic workloads without becoming a bottleneck. The system should support at least 1,000 jobs per minute throughput for job submissions and completions. Cron expression parsing should complete in under 1 millisecond per expression. Job queue operations (enqueue/dequeue) should complete in under 10 milliseconds at the 95th percentile when the queue contains up to 10,000 pending jobs.

**Scalability Requirements** allow the system to grow with organizational needs. The scheduler must support at least 10 concurrent worker nodes without performance degradation. The job queue should handle up to 100,000 pending jobs while maintaining sub-second dequeue times. Adding new workers should be as simple as starting a new process with the coordinator's address.

| Scalability Dimension | Target | Measurement Method |
|--------------------|---------|-------------------|
| Concurrent Workers | 10-50 nodes | Worker registration count |
| Pending Jobs | 100,000 jobs | Queue depth metrics |
| Job Throughput | 1,000 jobs/minute | Completed jobs per time window |
| Cron Parsing | <1ms per expression | Microbenchmark timing |
| Queue Operations | <10ms p95 latency | Operation timing histograms |

**Reliability Standards** ensure the scheduler operates dependably in production environments. The system should achieve 99.9% uptime, meaning no more than 8.76 hours of downtime per year. No jobs should be lost due to single-point failures like individual worker crashes or temporary network partitions. The system should gracefully degrade by continuing to execute jobs even if some workers become unavailable.

**Resource Efficiency** keeps the scheduler lightweight enough to run alongside other services. Memory usage should remain under 512 MB for a coordinator managing 10,000 jobs and 20 workers. CPU usage should stay under 25% on a modest server (2-4 cores) during normal operations. Network bandwidth should be minimal, with heartbeats and coordination messages consuming less than 1 MB/minute per worker.

> The key insight for non-functional goals is that they must be measurable and achievable within the project constraints. Setting targets like "unlimited scalability" or "microsecond latency" would make the project impossible to complete, while setting no targets at all would result in a system that doesn't work well enough to demonstrate the concepts effectively.

**Operational Simplicity** ensures the scheduler can be deployed and maintained without extensive infrastructure. The system should start with a single configuration file and require no external dependencies beyond Redis for job storage. Logs should provide enough information to diagnose common problems without requiring specialized monitoring tools. The entire system should be deployable on a developer's laptop for testing and development.

### Explicit Non-Goals

The non-goals establish firm boundaries on what this scheduler will NOT include, preventing scope creep while acknowledging that these features exist in enterprise scheduling systems.

**Job Dependencies and Workflows** are explicitly excluded because they add significant complexity to both the data model and execution engine. Features like "run Job B only after Job A completes successfully" or "run these five jobs in parallel, then run the final job" require dependency graph resolution, deadlock detection, and complex state management. While valuable in production systems, these features would triple the implementation effort without providing proportional learning value about the core distributed systems concepts.

**Resource-Aware Scheduling** that considers CPU, memory, or disk constraints is not included. The scheduler treats all jobs as equivalent resource consumers and doesn't consider whether a worker has sufficient capacity for a specific job type. Real production schedulers often include features like "only run this job on machines with 8+ GB RAM" or "don't run more than 2 CPU-intensive jobs per worker simultaneously." These features require resource modeling, capacity planning, and bin-packing algorithms that would shift focus away from distributed coordination.

| Excluded Feature Category | Specific Examples | Why Excluded |
|---------------------------|------------------|--------------|
| Job Dependencies | Workflow orchestration, dependency graphs | Complex state management distracts from core distributed systems learning |
| Resource Constraints | Memory limits, CPU quotas, disk space checks | Requires sophisticated resource modeling beyond project scope |
| Advanced Security | Authentication, authorization, job isolation | Security implementation would dominate development time |
| Multi-Datacenter | Cross-region replication, geo-distributed coordination | Network complexity exceeds single-datacenter learning goals |
| Job Artifacts | Input/output file management, result storage | File handling distracts from scheduling and coordination logic |

**Authentication and Authorization** are omitted to focus on the distributed systems challenges rather than security implementation. The scheduler assumes all job submissions are authorized and all workers are trusted. Production systems would include features like API keys, role-based access control, and job owner isolation, but implementing these would consume significant development time without advancing understanding of distributed coordination, consensus algorithms, or failure recovery.

**Advanced Monitoring and Metrics** beyond basic logging are not included. While production schedulers provide detailed dashboards showing job success rates, worker utilization, queue depth over time, and performance histograms, building these interfaces would shift effort away from the core distributed systems implementation. The system will log enough information for debugging, but won't include web dashboards or metrics exporters.

> **Decision: Focus on Core Distributed Systems Concepts**
> - **Context**: Distributed job schedulers in production include dozens of enterprise features for security, monitoring, resource management, and workflow orchestration
> - **Options Considered**: Include subset of enterprise features, build minimal viable scheduler, create full-featured system
> - **Decision**: Implement only features essential for demonstrating distributed coordination, consensus, and fault tolerance
> - **Rationale**: The learning value comes from solving distributed systems problems, not from building user interfaces or security layers
> - **Consequences**: The resulting scheduler demonstrates core concepts clearly but would need additional features for production use

**Multi-Datacenter Deployment** and geographic distribution are explicitly out of scope. The scheduler assumes all components (coordinators, workers, and storage) operate within a single datacenter with reliable, low-latency network connectivity. Cross-datacenter replication, split-brain resolution across WAN links, and geographic failover introduce network partition scenarios that are beyond the project's educational goals.

**Job Result Storage and Artifact Management** are not included. The scheduler tracks whether jobs completed successfully or failed, but doesn't store job output, intermediate files, or result artifacts. Workers report completion status through `reportCompletion()`, but any job-specific data management is the responsibility of the job implementation itself, not the scheduler infrastructure.

**Dynamic Job Modification** capabilities like updating cron expressions for running schedules, changing job priorities after submission, or canceling queued jobs are omitted. Once a job enters the system, it follows its original schedule and priority until completion. While useful for operational flexibility, these features require complex state transitions and conflict resolution that would complicate the core coordination logic.

The explicit non-goals serve as a contract with implementers: these boundaries won't expand during development. When facing implementation challenges, the solution should work within these constraints rather than adding excluded features as "quick fixes." This discipline ensures the project remains focused on its educational objectives while building a system sophisticated enough to demonstrate real distributed systems principles.

### Implementation Guidance

This section provides practical direction for implementing the functional and non-functional goals using Go as the primary development language.

**Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Job Queue Storage | Redis with sorted sets | Redis with custom Lua scripts |
| Worker Coordination | etcd for leader election | Custom Raft implementation |
| Cron Parsing | `github.com/robfig/cron/v3` | Custom parser with extended syntax |
| Configuration | YAML files with `gopkg.in/yaml.v3` | etcd-based dynamic configuration |
| Logging | Standard library `log/slog` | Structured logging with `logrus` |
| HTTP API | `net/http` with `gorilla/mux` | gRPC with Protocol Buffers |
| Metrics | Basic counters in memory | Prometheus metrics export |
| Testing | Standard `testing` package | Testcontainers for integration tests |

**Recommended Project Structure:**

```
distributed-job-scheduler/
├── cmd/
│   ├── coordinator/
│   │   └── main.go              ← coordinator service entry point
│   └── worker/
│       └── main.go              ← worker service entry point
├── internal/
│   ├── coordinator/
│   │   ├── coordinator.go       ← leader election and job assignment
│   │   ├── coordinator_test.go
│   │   └── heartbeat.go         ← worker health tracking
│   ├── cron/
│   │   ├── parser.go           ← cron expression parsing (Milestone 1)
│   │   ├── parser_test.go
│   │   └── schedule.go         ← next execution time calculation
│   ├── queue/
│   │   ├── priority_queue.go   ← job priority queue (Milestone 2)
│   │   ├── priority_queue_test.go
│   │   ├── deduplication.go    ← idempotency key handling
│   │   └── delayed_jobs.go     ← scheduled job visibility
│   ├── worker/
│   │   ├── worker.go           ← job execution and heartbeat (Milestone 3)
│   │   ├── worker_test.go
│   │   └── job_executor.go     ← job execution engine
│   └── models/
│       ├── job.go              ← Job, Worker data structures
│       ├── cron.go             ← CronExpression struct
│       └── constants.go        ← PENDING, EXECUTING, etc.
├── pkg/
│   ├── api/
│   │   └── client.go           ← HTTP client for job submission
│   └── storage/
│       └── redis.go            ← Redis connection and operations
├── configs/
│   ├── coordinator.yaml        ← coordinator configuration
│   └── worker.yaml             ← worker configuration
├── scripts/
│   ├── start-coordinator.sh    ← development startup scripts
│   └── start-worker.sh
├── docker/
│   ├── docker-compose.yml      ← local development environment
│   └── Dockerfile              ← service container image
└── docs/
    └── api.md                  ← HTTP API documentation
```

**Core Data Structures (Complete Definitions):**

```go
package models

import (
    "time"
)

// Job represents a scheduled task with priority and execution metadata
type Job struct {
    ID              string            `json:"id" redis:"id"`
    Name            string            `json:"name" redis:"name"`
    CronExpression  string            `json:"cron_expression" redis:"cron_expression"`
    Priority        int               `json:"priority" redis:"priority"`
    Payload         map[string]string `json:"payload" redis:"payload"`
    IdempotencyKey  string            `json:"idempotency_key" redis:"idempotency_key"`
    State           JobState          `json:"state" redis:"state"`
    WorkerID        string            `json:"worker_id,omitempty" redis:"worker_id"`
    FencingToken    string            `json:"fencing_token,omitempty" redis:"fencing_token"`
    ScheduledAt     time.Time         `json:"scheduled_at" redis:"scheduled_at"`
    ClaimedAt       *time.Time        `json:"claimed_at,omitempty" redis:"claimed_at"`
    CompletedAt     *time.Time        `json:"completed_at,omitempty" redis:"completed_at"`
    RetryCount      int               `json:"retry_count" redis:"retry_count"`
    MaxRetries      int               `json:"max_retries" redis:"max_retries"`
    CreatedAt       time.Time         `json:"created_at" redis:"created_at"`
    UpdatedAt       time.Time         `json:"updated_at" redis:"updated_at"`
}

// Worker represents a node that executes jobs with capacity tracking
type Worker struct {
    ID              string            `json:"id" redis:"id"`
    Address         string            `json:"address" redis:"address"`
    Capacity        int               `json:"capacity" redis:"capacity"`
    CurrentJobs     int               `json:"current_jobs" redis:"current_jobs"`
    Capabilities    []string          `json:"capabilities" redis:"capabilities"`
    LastHeartbeat   time.Time         `json:"last_heartbeat" redis:"last_heartbeat"`
    State           WorkerState       `json:"state" redis:"state"`
    StartedAt       time.Time         `json:"started_at" redis:"started_at"`
    Metadata        map[string]string `json:"metadata" redis:"metadata"`
}

// CronExpression holds parsed cron schedule with next execution calculation
type CronExpression struct {
    Original    string        `json:"original"`
    Minutes     []int         `json:"minutes"`     // 0-59
    Hours       []int         `json:"hours"`       // 0-23
    DaysOfMonth []int         `json:"days_month"`  // 1-31
    Months      []int         `json:"months"`      // 1-12
    DaysOfWeek  []int         `json:"days_week"`   // 0-6 (Sunday=0)
    Timezone    *time.Location `json:"timezone"`
}

// JobState constants for job lifecycle
type JobState string

const (
    PENDING   JobState = "PENDING"   // job queued, waiting for worker
    CLAIMED   JobState = "CLAIMED"   // job assigned to worker, not yet started
    EXECUTING JobState = "EXECUTING" // job currently running on worker
    COMPLETED JobState = "COMPLETED" // job finished successfully
    FAILED    JobState = "FAILED"    // job failed after all retries
)

// WorkerState constants for worker lifecycle
type WorkerState string

const (
    AVAILABLE   WorkerState = "AVAILABLE"   // worker ready to accept jobs
    BUSY        WorkerState = "BUSY"        // worker at capacity
    UNAVAILABLE WorkerState = "UNAVAILABLE" // worker failed or shutdown
)
```

**Configuration Management (Complete Implementation):**

```go
package config

import (
    "fmt"
    "gopkg.in/yaml.v3"
    "os"
    "time"
)

type CoordinatorConfig struct {
    Server struct {
        Port         int           `yaml:"port"`
        ReadTimeout  time.Duration `yaml:"read_timeout"`
        WriteTimeout time.Duration `yaml:"write_timeout"`
    } `yaml:"server"`
    
    Redis struct {
        Address  string `yaml:"address"`
        Password string `yaml:"password"`
        Database int    `yaml:"database"`
    } `yaml:"redis"`
    
    Coordinator struct {
        HeartbeatTimeout time.Duration `yaml:"heartbeat_timeout"`
        ElectionTimeout  time.Duration `yaml:"election_timeout"`
        JobScanInterval  time.Duration `yaml:"job_scan_interval"`
    } `yaml:"coordinator"`
}

type WorkerConfig struct {
    Worker struct {
        ID           string        `yaml:"id"`
        Capacity     int           `yaml:"capacity"`
        Capabilities []string      `yaml:"capabilities"`
        HeartbeatInterval time.Duration `yaml:"heartbeat_interval"`
    } `yaml:"worker"`
    
    Coordinator struct {
        Address string `yaml:"address"`
    } `yaml:"coordinator"`
    
    Redis struct {
        Address  string `yaml:"address"`
        Password string `yaml:"password"`
        Database int    `yaml:"database"`
    } `yaml:"redis"`
}

func LoadCoordinatorConfig(path string) (*CoordinatorConfig, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("reading config file: %w", err)
    }
    
    var config CoordinatorConfig
    if err := yaml.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parsing config YAML: %w", err)
    }
    
    return &config, nil
}

func LoadWorkerConfig(path string) (*WorkerConfig, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("reading config file: %w", err)
    }
    
    var config WorkerConfig
    if err := yaml.Unmarshal(data, &config); err != nil {
        return nil, fmt.Errorf("parsing config YAML: %w", err)
    }
    
    return &config, nil
}
```

**HTTP API Infrastructure (Complete Implementation):**

```go
package api

import (
    "encoding/json"
    "net/http"
    "time"
    "github.com/gorilla/mux"
    "distributed-job-scheduler/internal/models"
)

type JobSubmissionRequest struct {
    Name           string            `json:"name"`
    CronExpression string            `json:"cron_expression"`
    Priority       int               `json:"priority"`
    Payload        map[string]string `json:"payload"`
    IdempotencyKey string            `json:"idempotency_key"`
    MaxRetries     int               `json:"max_retries"`
}

type JobSubmissionResponse struct {
    JobID     string    `json:"job_id"`
    Status    string    `json:"status"`
    NextRun   time.Time `json:"next_run,omitempty"`
    Message   string    `json:"message,omitempty"`
}

type ErrorResponse struct {
    Error   string `json:"error"`
    Code    string `json:"code,omitempty"`
    Details string `json:"details,omitempty"`
}

// JobSubmissionHandler handles HTTP requests for job submission
func JobSubmissionHandler(jobQueue JobQueue) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // TODO: Parse JSON request body into JobSubmissionRequest
        // TODO: Validate cron expression using cron parser
        // TODO: Generate unique job ID and fencing token
        // TODO: Create Job struct with PENDING state
        // TODO: Submit job to priority queue with deduplication check
        // TODO: Calculate next execution time from cron expression
        // TODO: Return JobSubmissionResponse with job details
        // TODO: Handle errors with appropriate HTTP status codes
    }
}

// WorkerHeartbeatHandler accepts heartbeat signals from workers
func WorkerHeartbeatHandler(coordinator Coordinator) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // TODO: Extract worker ID from URL path or headers
        // TODO: Parse heartbeat payload with current job count and capabilities
        // TODO: Update worker's last heartbeat timestamp in coordination store
        // TODO: Return current job assignments or coordination commands
        // TODO: Handle new worker registration if ID not found
    }
}

// SetupRoutes configures HTTP routes for the coordinator API
func SetupRoutes(jobQueue JobQueue, coordinator Coordinator) *mux.Router {
    r := mux.NewRouter()
    
    // Job management endpoints
    r.HandleFunc("/api/v1/jobs", JobSubmissionHandler(jobQueue)).Methods("POST")
    r.HandleFunc("/api/v1/jobs", ListJobsHandler(jobQueue)).Methods("GET")
    r.HandleFunc("/api/v1/jobs/{id}", GetJobHandler(jobQueue)).Methods("GET")
    
    // Worker coordination endpoints
    r.HandleFunc("/api/v1/workers/heartbeat", WorkerHeartbeatHandler(coordinator)).Methods("POST")
    r.HandleFunc("/api/v1/workers", ListWorkersHandler(coordinator)).Methods("GET")
    
    return r
}

// Helper function for JSON responses
func WriteJSONResponse(w http.ResponseWriter, status int, data interface{}) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(data)
}

// Helper function for error responses
func WriteErrorResponse(w http.ResponseWriter, status int, message string) {
    response := ErrorResponse{
        Error: message,
        Code:  http.StatusText(status),
    }
    WriteJSONResponse(w, status, response)
}
```

**Redis Storage Interface (Complete Implementation):**

```go
package storage

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    "github.com/go-redis/redis/v8"
    "distributed-job-scheduler/internal/models"
)

type RedisClient struct {
    client *redis.Client
    ctx    context.Context
}

func NewRedisClient(addr, password string, db int) *RedisClient {
    rdb := redis.NewClient(&redis.Options{
        Addr:     addr,
        Password: password,
        DB:       db,
    })
    
    return &RedisClient{
        client: rdb,
        ctx:    context.Background(),
    }
}

// Job storage operations
func (r *RedisClient) StoreJob(job *models.Job) error {
    jobData, err := json.Marshal(job)
    if err != nil {
        return fmt.Errorf("marshaling job: %w", err)
    }
    
    // Store job data
    jobKey := fmt.Sprintf("job:%s", job.ID)
    if err := r.client.Set(r.ctx, jobKey, jobData, 0).Err(); err != nil {
        return fmt.Errorf("storing job data: %w", err)
    }
    
    // Add to priority queue with score = (priority << 32) | scheduled_timestamp
    score := float64(job.Priority<<32) + float64(job.ScheduledAt.Unix())
    queueKey := "job_queue"
    if err := r.client.ZAdd(r.ctx, queueKey, &redis.Z{
        Score:  score,
        Member: job.ID,
    }).Err(); err != nil {
        return fmt.Errorf("adding to priority queue: %w", err)
    }
    
    return nil
}

func (r *RedisClient) GetJob(jobID string) (*models.Job, error) {
    jobKey := fmt.Sprintf("job:%s", jobID)
    data, err := r.client.Get(r.ctx, jobKey).Result()
    if err != nil {
        return nil, fmt.Errorf("getting job data: %w", err)
    }
    
    var job models.Job
    if err := json.Unmarshal([]byte(data), &job); err != nil {
        return nil, fmt.Errorf("unmarshaling job: %w", err)
    }
    
    return &job, nil
}

// Worker registration and heartbeat operations
func (r *RedisClient) RegisterWorker(worker *models.Worker) error {
    workerData, err := json.Marshal(worker)
    if err != nil {
        return fmt.Errorf("marshaling worker: %w", err)
    }
    
    workerKey := fmt.Sprintf("worker:%s", worker.ID)
    if err := r.client.Set(r.ctx, workerKey, workerData, 0).Err(); err != nil {
        return fmt.Errorf("storing worker data: %w", err)
    }
    
    // Add to active workers set
    activeWorkersKey := "active_workers"
    if err := r.client.SAdd(r.ctx, activeWorkersKey, worker.ID).Err(); err != nil {
        return fmt.Errorf("adding to active workers: %w", err)
    }
    
    return nil
}

func (r *RedisClient) UpdateWorkerHeartbeat(workerID string, timestamp time.Time) error {
    workerKey := fmt.Sprintf("worker:%s", workerID)
    
    // Update heartbeat timestamp using hash field
    if err := r.client.HSet(r.ctx, workerKey, "last_heartbeat", timestamp.Unix()).Err(); err != nil {
        return fmt.Errorf("updating heartbeat: %w", err)
    }
    
    return nil
}

// Atomic job claiming operation using Lua script
const claimJobScript = `
local job_id = ARGV[1]
local worker_id = ARGV[2]
local fencing_token = ARGV[3]
local current_time = ARGV[4]

local job_key = "job:" .. job_id
local job_data = redis.call("GET", job_key)

if not job_data then
    return {err = "Job not found"}
end

local job = cjson.decode(job_data)
if job.state ~= "PENDING" then
    return {err = "Job not in PENDING state"}
end

-- Update job state to CLAIMED
job.state = "CLAIMED"
job.worker_id = worker_id
job.fencing_token = fencing_token
job.claimed_at = current_time

local updated_data = cjson.encode(job)
redis.call("SET", job_key, updated_data)

-- Remove from general queue, add to worker's queue
redis.call("ZREM", "job_queue", job_id)
redis.call("LPUSH", "worker_jobs:" .. worker_id, job_id)

return {ok = "Job claimed successfully"}
`

func (r *RedisClient) ClaimJob(jobID, workerID, fencingToken string) error {
    currentTime := time.Now().Unix()
    
    result := r.client.Eval(r.ctx, claimJobScript, []string{}, 
        jobID, workerID, fencingToken, currentTime)
    
    if result.Err() != nil {
        return fmt.Errorf("claiming job: %w", result.Err())
    }
    
    return nil
}
```

**Milestone Verification Checkpoints:**

After implementing each milestone, verify correct behavior using these specific tests:

**Milestone 1 Checkpoint - Cron Parser:**
```bash
# Run cron parser tests
go test ./internal/cron/... -v

# Manual verification
go run cmd/test-cron/main.go "0 9 * * 1-5"
# Expected output: Next execution at 2024-01-08 09:00:00 UTC (next weekday at 9 AM)

go run cmd/test-cron/main.go "*/15 * * * *"
# Expected output: Next execution within 15 minutes of current time
```

**Milestone 2 Checkpoint - Priority Queue:**
```bash
# Run queue tests
go test ./internal/queue/... -v

# Manual verification with Redis CLI
redis-cli ZADD test_queue 100 job1 200 job2 150 job3
redis-cli ZREVRANGE test_queue 0 -1 WITHSCORES
# Expected: job2 (score 200), job3 (score 150), job1 (score 100)
```

**Milestone 3 Checkpoint - Worker Coordination:**
```bash
# Start coordinator
go run cmd/coordinator/main.go -config configs/coordinator.yaml

# Start worker in another terminal
go run cmd/worker/main.go -config configs/worker.yaml

# Submit test job via HTTP
curl -X POST http://localhost:8080/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"test","cron_expression":"* * * * *","priority":100,"payload":{"command":"echo hello"}}'

# Expected: Job appears in worker logs within 60 seconds
```

**Language-Specific Implementation Tips:**

1. **Time Handling**: Use `time.Time` in UTC for all internal calculations, convert to local timezone only for display
2. **Concurrency**: Use `sync.RWMutex` for protecting shared data structures like worker registry
3. **Context Cancellation**: Pass `context.Context` through all long-running operations for graceful shutdown
4. **Error Wrapping**: Use `fmt.Errorf("operation: %w", err)` for error context without losing original error
5. **JSON Tags**: Include both `json` and `redis` struct tags for serialization to different stores
6. **Configuration**: Use `gopkg.in/yaml.v3` for human-readable config files
7. **Testing**: Use `testcontainers-go` for integration tests that need real Redis instances
8. **Logging**: Use structured logging with fields: `slog.Info("job claimed", "jobID", job.ID, "workerID", worker.ID)`


## High-Level Architecture

> **Milestone(s):** This section provides the architectural foundation for all three milestones by establishing the core components and their interactions that enable cron expression parsing (Milestone 1), priority job queuing (Milestone 2), and worker coordination (Milestone 3).

The distributed job scheduler architecture can be understood through a mental model of a sophisticated **orchestra with multiple conductors and musicians**. Just as an orchestra needs sheet music (cron expressions), a conductor to coordinate timing (scheduler service), musicians with different capabilities (workers), and a system to distribute music sheets fairly (job queue), our distributed scheduler coordinates the execution of jobs across multiple machines while maintaining consistency and fault tolerance.

The architecture consists of four primary layers that work together to provide reliable, distributed job scheduling. The **coordination layer** acts as the nervous system, managing leader election and maintaining cluster state. The **scheduler service** serves as the brain, parsing cron expressions and making job assignment decisions. The **job queue** functions as the memory, storing pending jobs with priorities and timing information. Finally, the **worker pool** represents the muscles, executing the actual job payloads while reporting status back to the coordination layer.

### System Components

The distributed job scheduler is composed of several core components, each with distinct responsibilities that collectively enable fault-tolerant job execution across a cluster of machines.

![System Architecture Overview](./diagrams/system-architecture.svg)

#### Scheduler Service Components

The **Scheduler Service** acts as the central orchestrator, responsible for transforming cron expressions into executable jobs and coordinating their distribution to workers. This service embodies the "conductor" role in our orchestra analogy, ensuring that each job plays at precisely the right time.

| Component | Responsibility | Key Operations | Failure Impact |
|-----------|---------------|----------------|-----------------|
| Cron Engine | Parse cron expressions and calculate next execution times | `parseExpression()`, `calculateNext()`, `validateTimezone()` | Jobs stop being scheduled until recovery |
| Job Scheduler | Create job instances from recurring schedules | `createJobFromSchedule()`, `applyPriority()`, `setIdempotencyKey()` | New job instances not created, existing jobs continue |
| Assignment Manager | Distribute jobs to available workers based on capacity | `selectWorker()`, `assignJob()`, `trackAssignments()` | Jobs queue up but don't execute until recovery |
| Heartbeat Monitor | Track worker health and detect failures | `processHeartbeat()`, `detectFailures()`, `triggerRecovery()` | Failed workers not detected, jobs may be stuck |

The Scheduler Service maintains several critical data structures to coordinate job execution:

| Data Structure | Purpose | Key Fields | Update Frequency |
|----------------|---------|------------|------------------|
| Active Schedules Map | Track all registered cron schedules | `scheduleID`, `cronExpression`, `nextRun`, `lastRun` | Every job creation |
| Worker Registry | Maintain current worker status and capacity | `workerID`, `capacity`, `currentLoad`, `capabilities` | Every heartbeat (30s) |
| Job Assignments | Track which worker owns which job | `jobID`, `workerID`, `claimedAt`, `fencingToken` | Every job claim/completion |
| Failed Workers Set | Workers pending cleanup and job recovery | `workerID`, `failedAt`, `assignedJobs` | On heartbeat timeout |

> **Design Insight:** The Scheduler Service is designed to be stateless except for in-memory caches. All persistent state lives in the coordination layer (etcd) or job queue (Redis), allowing multiple scheduler instances to run simultaneously with leader election determining which instance actively makes scheduling decisions.

#### Job Queue Infrastructure

The **Job Queue** serves as the system's durable memory, implementing a sophisticated priority queue with delayed execution capabilities. Think of it as a **hospital emergency room triage system** - jobs arrive with different priority levels, some must wait until a specific time (delayed execution), and the system ensures the most critical jobs are handled first while preventing duplicate treatment of the same patient.

| Queue Component | Purpose | Implementation | Scalability Limit |
|-----------------|---------|----------------|-------------------|
| Priority Heap | Order jobs by priority and scheduled time | Redis sorted sets with score-based ordering | ~10M jobs per Redis instance |
| Delayed Job Timer | Hold jobs until their execution time arrives | Redis key expiration notifications | Limited by Redis memory |
| Deduplication Store | Prevent duplicate job submissions | Redis hash with idempotency keys | ~100M unique keys |
| Dead Letter Queue | Store permanently failed jobs for analysis | Separate Redis list with TTL | Manual cleanup required |

The queue implements several sophisticated algorithms to ensure fair job distribution:

1. **Priority Resolution**: When multiple jobs have the same priority level, the queue uses scheduled time as a secondary sort key, ensuring first-in-first-out behavior within priority bands.

2. **Delayed Visibility**: Jobs with future scheduled times are stored in a separate "delayed" namespace and moved to the active queue through Redis key expiration events, providing precise timing control.

3. **Atomic Job Claims**: Workers claim jobs using Redis Lua scripts that atomically check job availability, update job state to `CLAIMED`, and set a fencing token to prevent duplicate processing.

4. **Deduplication Windows**: The system maintains idempotency keys for a configurable time window (default 24 hours), allowing clients to safely retry job submissions without creating duplicates.

> **Architecture Decision: Redis vs Database for Job Queue**
> - **Context**: Need to choose storage backend for job queue with priority ordering and atomic operations
> - **Options Considered**: 
>   - PostgreSQL with indexed priority columns
>   - Redis with sorted sets and Lua scripts  
>   - In-memory Go data structures with clustering
> - **Decision**: Redis with sorted sets
> - **Rationale**: Redis sorted sets provide O(log N) priority ordering natively, Lua scripts enable atomic multi-operation commands, and pub/sub supports real-time job notifications. PostgreSQL would require complex locking for atomic job claims, while in-memory structures lose durability.
> - **Consequences**: Introduces Redis as a dependency but provides superior performance for queue operations. Limits job metadata size due to Redis memory constraints.

#### Worker Pool Management

The **Worker Pool** represents the distributed execution layer, where actual job processing occurs. Workers function like **specialized craftspeople in a guild** - each worker registers their capabilities, maintains their tools (execution environment), and communicates regularly with the guild master (coordinator) about their availability and current projects.

Workers operate through a well-defined lifecycle that ensures reliable job execution:

| Worker State | Description | Valid Transitions | Trigger Conditions |
|--------------|-------------|-------------------|-------------------|
| `REGISTERING` | Initial state during startup | → `AVAILABLE`, → `UNAVAILABLE` | Successful/failed coordinator registration |
| `AVAILABLE` | Ready to accept new jobs | → `BUSY`, → `UNAVAILABLE` | Job assignment or failure detection |
| `BUSY` | At capacity, cannot accept more jobs | → `AVAILABLE`, → `UNAVAILABLE` | Job completion or failure |
| `UNAVAILABLE` | Failed or shutting down | → `AVAILABLE` (recovery only) | Heartbeat timeout or explicit shutdown |

Each worker maintains comprehensive metadata to support intelligent job assignment:

| Worker Metadata | Type | Purpose | Update Trigger |
|-----------------|------|---------|----------------|
| `ID` | string | Unique worker identifier across cluster | At registration |
| `Address` | string | Network endpoint for direct communication | At registration |
| `Capacity` | int | Maximum concurrent jobs this worker can handle | At registration, capacity changes |
| `CurrentJobs` | int | Number of jobs currently executing | Job claim/completion |
| `Capabilities` | []string | Job types this worker can execute | At registration |
| `LastHeartbeat` | time.Time | Most recent health check timestamp | Every heartbeat interval |
| `Metadata` | map[string]string | Custom worker attributes for job matching | At registration, periodic updates |

Workers implement a robust heartbeat mechanism that serves multiple purposes beyond simple health checking:

1. **Capacity Reporting**: Each heartbeat includes current job count and available capacity, enabling intelligent load balancing.

2. **Capability Updates**: Workers can dynamically advertise new capabilities or remove support for job types during runtime.

3. **Graceful Shutdown Signaling**: Workers use heartbeat metadata to indicate planned shutdown, allowing coordinators to stop assigning new jobs while existing jobs complete.

4. **Performance Metrics**: Heartbeats carry job execution statistics that help coordinators optimize future job assignments.

> **Common Pitfall: Heartbeat Frequency vs Job Duration**
> 
> ⚠️ **Pitfall: Setting heartbeat timeout shorter than maximum job duration**
> 
> If heartbeat timeout is 60 seconds but jobs can run for 10 minutes, the coordinator will incorrectly mark workers as failed while they're executing long jobs. This causes job reassignment and potential duplicate execution.
> 
> **Solution**: Set heartbeat timeout to at least 2x the maximum expected job duration, or implement job-specific timeout extensions where workers can request longer heartbeat intervals for long-running jobs.

#### Coordination Layer Architecture

The **Coordination Layer** provides the distributed systems infrastructure that enables multiple scheduler instances and workers to operate as a unified cluster. This layer implements consensus protocols and distributed locking mechanisms that prevent the chaos that would result from multiple coordinators making conflicting decisions simultaneously.

The coordination layer is built around **etcd** as the primary consensus store, chosen for its strong consistency guarantees and proven reliability in production distributed systems:

| Coordination Service | Purpose | Data Stored | Consistency Requirements |
|---------------------|---------|-------------|-------------------------|
| Leader Election | Ensure single active scheduler | Current leader ID, lease expiration | Strong consistency, linearizable reads |
| Worker Discovery | Maintain cluster member registry | Worker metadata, health status | Eventually consistent |
| Configuration Management | Distribute cluster-wide settings | Cron schedules, retry policies | Strong consistency for updates |
| Distributed Locking | Coordinate job recovery operations | Lock owner, expiration time | Strong consistency, exclusive access |

The leader election algorithm follows a lease-based approach that prevents split-brain scenarios:

1. **Candidate Announcement**: Scheduler instances announce candidacy by attempting to create a leader key in etcd with a TTL lease.

2. **Lease Renewal**: The current leader continuously renews its lease every 10 seconds (with 30-second TTL) to maintain leadership.

3. **Leadership Transfer**: When a leader fails to renew its lease, the key expires and other candidates immediately compete for leadership.

4. **Graceful Handoff**: During planned shutdown, leaders can explicitly transfer leadership by deleting their key and allowing immediate re-election.

> **Architecture Decision: etcd vs Consul vs Zookeeper for Coordination**
> - **Context**: Need distributed coordination service for leader election and configuration management
> - **Options Considered**:
>   - etcd with Raft consensus
>   - Consul with Raft consensus  
>   - Apache Zookeeper with ZAB protocol
> - **Decision**: etcd
> - **Rationale**: etcd provides simpler HTTP/gRPC APIs compared to Zookeeper's complex protocol, has excellent Go client libraries, and offers strong consistency guarantees. Consul focuses more on service discovery and less on general-purpose coordination. etcd is proven in Kubernetes and other large-scale systems.
> - **Consequences**: Requires etcd cluster deployment and management, but provides reliable foundation for all coordination needs with excellent operational tooling.

### Deployment Model

The distributed job scheduler is designed for deployment across multiple machines in a cluster configuration, with each component type running on dedicated or shared infrastructure based on operational requirements and scale.

#### Physical Deployment Topology

The recommended deployment follows a **three-tier architecture** that separates coordination, scheduling, and execution concerns while maintaining high availability and fault tolerance:

| Tier | Components | Minimum Nodes | Recommended Sizing | Network Requirements |
|------|------------|---------------|-------------------|---------------------|
| Coordination Tier | etcd cluster, Redis cluster | 3 etcd, 3 Redis | 2 CPU, 4GB RAM per node | Low latency, high bandwidth between nodes |
| Control Tier | Scheduler services | 2 active/standby | 4 CPU, 8GB RAM per node | Low latency to coordination tier |
| Worker Tier | Worker processes | Variable | Based on job requirements | Moderate latency acceptable |

**Coordination Tier Deployment**: The coordination tier forms the backbone of cluster consensus and must be deployed with careful attention to fault tolerance. The etcd cluster should span multiple availability zones with odd numbers of nodes (3 or 5) to maintain quorum during failures. Redis deployment can follow either clustering or master/replica patterns depending on job throughput requirements.

**Control Tier Deployment**: Scheduler services run in active/standby configuration with leader election determining which instance actively schedules jobs. Multiple scheduler instances can run simultaneously, but only the elected leader performs job creation and assignment. This design allows for instant failover when the current leader fails.

**Worker Tier Deployment**: Workers can be deployed flexibly based on job execution requirements. CPU-intensive jobs might require dedicated worker nodes, while I/O-bound jobs could share nodes with other services. Workers automatically register with the coordination tier on startup and can be added or removed dynamically without affecting system operation.

#### Network Communication Patterns

The scheduler implements multiple communication patterns optimized for different types of cluster coordination:

| Communication Type | Protocol | Frequency | Failure Handling |
|-------------------|----------|-----------|------------------|
| Scheduler → etcd | gRPC | Continuous (leader election) | Exponential backoff, multiple endpoints |
| Scheduler → Redis | Redis protocol | Per job operation | Connection pooling, failover |
| Worker → Scheduler | HTTP/JSON | 30-second heartbeats | Retry with jitter, graceful degradation |
| Scheduler → Worker | HTTP/JSON | Job assignments only | Timeout and reassignment |

The system is designed to gracefully handle network partitions and intermittent connectivity issues. Workers continue executing assigned jobs even during network partitions, reporting completion when connectivity resumes. Schedulers detect worker failures through heartbeat timeouts and reassign jobs to healthy workers.

#### Scaling and Resource Planning

Resource planning should consider both steady-state operation and peak load scenarios:

**Scheduler Service Scaling**: A single scheduler instance can typically handle 10,000+ active schedules and coordinate 100+ workers. The primary bottleneck is usually etcd throughput for leader election heartbeats and worker registration updates.

**Job Queue Scaling**: Redis memory requirements scale linearly with queue depth. Estimate 1KB per queued job for metadata, with additional memory for deduplication tracking. A single Redis instance can typically handle 100,000+ queued jobs.

**Worker Scaling**: Workers scale horizontally without limits. Each worker's capacity depends on job types - CPU-bound jobs might allow 1-2 concurrent executions per core, while I/O-bound jobs could handle 10-20 concurrent executions.

> **Deployment Insight**: The scheduler is designed as a "shared-nothing" architecture where all persistent state lives in external stores (etcd, Redis). This enables sophisticated deployment patterns like blue/green deployments, rolling updates, and cross-datacenter replication without complex data migration procedures.

### Recommended File Structure

The Go module organization follows domain-driven design principles, separating concerns by functional area while maintaining clear dependency boundaries between components. This structure supports the three-milestone development approach by organizing code into coherent packages that can be developed and tested independently.

```
distributed-job-scheduler/
├── cmd/                              # Application entry points
│   ├── scheduler/                    # Scheduler service binary
│   │   └── main.go                   # Service initialization and configuration
│   ├── worker/                       # Worker service binary  
│   │   └── main.go                   # Worker startup and registration
│   └── cli/                          # Administrative command-line tools
│       └── main.go                   # Job submission, cluster status
├── internal/                         # Private application packages
│   ├── coordinator/                  # Milestone 3: Worker coordination
│   │   ├── coordinator.go            # Leader election and job assignment
│   │   ├── heartbeat.go             # Worker health monitoring
│   │   ├── recovery.go              # Failed job recovery logic
│   │   └── coordinator_test.go      # Coordination integration tests
│   ├── cron/                        # Milestone 1: Cron expression parsing
│   │   ├── parser.go                # Cron syntax parsing and validation
│   │   ├── calculator.go            # Next execution time calculation
│   │   ├── timezone.go              # Timezone handling and DST logic
│   │   └── parser_test.go           # Comprehensive parsing test suite
│   ├── queue/                       # Milestone 2: Priority job queue
│   │   ├── priority_queue.go        # Priority-based job ordering
│   │   ├── delayed_queue.go         # Scheduled job timing management
│   │   ├── deduplication.go         # Idempotency key handling
│   │   ├── redis_backend.go         # Redis storage implementation
│   │   └── queue_test.go            # Queue operation tests
│   ├── worker/                      # Worker execution engine
│   │   ├── worker.go                # Job execution and lifecycle management
│   │   ├── executor.go              # Job payload processing interface
│   │   ├── heartbeat_client.go      # Coordinator communication
│   │   └── worker_test.go           # Worker behavior tests
│   ├── models/                      # Shared data structures
│   │   ├── job.go                   # Job struct and state management
│   │   ├── worker.go                # Worker struct and capability tracking
│   │   ├── schedule.go              # Schedule definitions and metadata
│   │   └── cron_expression.go       # Parsed cron expression representation
│   └── storage/                     # Storage layer abstractions
│       ├── etcd/                    # etcd coordination client
│       │   ├── client.go            # Connection management and operations
│       │   ├── leader_election.go   # Leader election implementation
│       │   └── watcher.go           # Configuration change notifications
│       ├── redis/                   # Redis queue client
│       │   ├── client.go            # Connection pooling and commands
│       │   ├── lua_scripts.go       # Atomic operation scripts
│       │   └── pubsub.go            # Job notification handling
│       └── interfaces.go            # Storage interface definitions
├── pkg/                             # Public API packages
│   ├── api/                         # HTTP REST API definitions
│   │   ├── handlers/                # Request handlers for job management
│   │   ├── middleware/              # Authentication, logging, metrics
│   │   └── types/                   # API request/response structures
│   ├── client/                      # Go client library for job submission
│   │   ├── scheduler_client.go      # High-level client interface
│   │   └── retry.go                 # Client-side retry and backoff
│   └── metrics/                     # Observability and monitoring
│       ├── prometheus.go            # Prometheus metrics integration
│       └── tracing.go               # Distributed tracing support
├── configs/                         # Configuration files and templates
│   ├── scheduler.yaml               # Scheduler service configuration
│   ├── worker.yaml                  # Worker service configuration
│   └── docker-compose.yml           # Local development environment
├── deployments/                     # Deployment manifests and scripts
│   ├── kubernetes/                  # Kubernetes YAML manifests
│   ├── docker/                      # Dockerfile for containerization
│   └── terraform/                   # Infrastructure as code
├── docs/                           # Additional documentation
├── scripts/                        # Development and deployment scripts
└── tests/                          # Integration and end-to-end tests
    ├── integration/                # Cross-component integration tests
    ├── e2e/                       # Full system end-to-end tests
    └── fixtures/                  # Test data and mock configurations
```

#### Package Dependency Guidelines

The module structure enforces clear dependency boundaries that support milestone-based development:

| Package | Can Import | Cannot Import | Reasoning |
|---------|------------|---------------|-----------|
| `internal/cron` | `internal/models` only | Other internal packages | Pure parsing logic, no external dependencies |
| `internal/queue` | `internal/models`, `internal/storage/redis` | `internal/coordinator`, `internal/worker` | Queue operations independent of coordination |
| `internal/coordinator` | All internal packages | None | Top-level orchestration needs access to all components |
| `internal/worker` | `internal/models`, `pkg/client` | `internal/coordinator` | Workers should not directly depend on coordination logic |

#### Development Workflow Organization

The file structure supports a natural development progression aligned with the project milestones:

**Milestone 1 Development Focus**: Developers start with `internal/cron` and `internal/models/cron_expression.go`, building and testing cron parsing logic in isolation. The clear separation allows comprehensive testing without external dependencies.

**Milestone 2 Development Focus**: Queue implementation in `internal/queue` can proceed independently, with Redis backend abstracted through interfaces in `internal/storage`. Mock implementations support testing without Redis infrastructure.

**Milestone 3 Development Focus**: Coordinator logic in `internal/coordinator` integrates all previous components, with worker coordination building on established queue and parsing functionality.

> **File Structure Insight**: The separation of `internal/` and `pkg/` follows Go conventions where `internal/` packages cannot be imported by external projects, while `pkg/` provides stable public APIs. This structure supports future open-source distribution while protecting internal implementation details from external dependencies.

The `cmd/` directory structure allows building separate binaries for different deployment scenarios - monolithic deployments can run scheduler and worker in the same process, while distributed deployments can deploy them separately. The shared `internal/` packages support both patterns without code duplication.

### Implementation Guidance

This section provides concrete technology choices and starter code to help junior developers translate the architectural design into working Go code.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Trade-offs |
|-----------|---------------|-----------------|------------|
| HTTP Framework | `net/http` with `gorilla/mux` | `gin-gonic/gin` with middleware | Standard library vs performance optimized |
| Configuration | `gopkg.in/yaml.v3` | `spf13/viper` with env overrides | Simple YAML vs dynamic configuration |
| Logging | `log/slog` (Go 1.21+) | `sirupsen/logrus` with hooks | Built-in structured logging vs ecosystem |
| Metrics | `prometheus/client_golang` | `prometheus/client_golang` + Grafana | Industry standard, no simpler alternative |
| etcd Client | `go.etcd.io/etcd/clientv3` | Same with custom retry wrapper | Official client, wrap for convenience |
| Redis Client | `go-redis/redis/v9` | `go-redis/redis/v9` with cluster | Mature library, cluster support when needed |
| Testing | `testing` + `testify/assert` | `testify/suite` + `testcontainers` | Simple assertions vs full test suites |

#### Infrastructure Starter Code

**etcd Client Wrapper** (Complete implementation for coordination layer):

```go
// internal/storage/etcd/client.go
package etcd

import (
    "context"
    "time"
    
    clientv3 "go.etcd.io/etcd/client/v3"
    "go.etcd.io/etcd/client/v3/concurrency"
)

// Client wraps etcd operations with retry logic and connection management
type Client struct {
    client   *clientv3.Client
    session  *concurrency.Session
    timeout  time.Duration
}

// NewClient creates a new etcd client with recommended settings
func NewClient(endpoints []string) (*Client, error) {
    client, err := clientv3.New(clientv3.Config{
        Endpoints:   endpoints,
        DialTimeout: 5 * time.Second,
    })
    if err != nil {
        return nil, err
    }
    
    session, err := concurrency.NewSession(client, concurrency.WithTTL(30))
    if err != nil {
        client.Close()
        return nil, err
    }
    
    return &Client{
        client:  client,
        session: session,
        timeout: 10 * time.Second,
    }, nil
}

// Campaign attempts to become leader for the given election key
func (c *Client) Campaign(ctx context.Context, election string) error {
    e := concurrency.NewElection(c.session, election)
    return e.Campaign(ctx, c.session.Lease())
}

// IsLeader checks if this client currently holds leadership
func (c *Client) IsLeader(ctx context.Context, election string) (bool, error) {
    e := concurrency.NewElection(c.session, election)
    resp, err := e.Leader(ctx)
    if err != nil {
        return false, err
    }
    return string(resp.Kvs[0].Value) == string(c.session.Lease()), nil
}

// Close releases all resources
func (c *Client) Close() error {
    c.session.Close()
    return c.client.Close()
}
```

**Redis Queue Backend** (Complete implementation for job queue):

```go
// internal/storage/redis/client.go
package redis

import (
    "context"
    "encoding/json"
    "time"
    
    "github.com/redis/go-redis/v9"
)

// Client wraps Redis operations for job queue management
type Client struct {
    rdb *redis.Client
}

// NewClient creates Redis client with connection pooling
func NewClient(addr, password string, db int) *Client {
    rdb := redis.NewClient(&redis.Options{
        Addr:     addr,
        Password: password,
        DB:       db,
        PoolSize: 10,
    })
    
    return &Client{rdb: rdb}
}

// EnqueueJob adds job to priority queue with deduplication
func (c *Client) EnqueueJob(ctx context.Context, job interface{}, priority float64, idempotencyKey string) error {
    // Lua script for atomic enqueue with deduplication
    script := redis.NewScript(`
        local key = KEYS[1]
        local dedup_key = KEYS[2] 
        local job_data = ARGV[1]
        local priority = ARGV[2]
        local idempotency = ARGV[3]
        
        if redis.call('EXISTS', dedup_key) == 1 then
            return 0  -- Job already exists
        end
        
        redis.call('ZADD', key, priority, job_data)
        redis.call('SETEX', dedup_key, 86400, idempotency)  -- 24 hour dedup window
        return 1
    `)
    
    jobData, err := json.Marshal(job)
    if err != nil {
        return err
    }
    
    _, err = script.Run(ctx, c.rdb, 
        []string{"jobs:pending", "dedup:" + idempotencyKey},
        string(jobData), priority, idempotencyKey,
    ).Result()
    
    return err
}

// DequeueJob atomically claims highest priority job
func (c *Client) DequeueJob(ctx context.Context, workerID string) ([]byte, error) {
    script := redis.NewScript(`
        local pending_key = KEYS[1]
        local claimed_key = KEYS[2]
        local worker_id = ARGV[1]
        
        local job = redis.call('ZPOPMIN', pending_key)
        if #job == 0 then
            return nil  -- No jobs available
        end
        
        local job_data = job[1]
        redis.call('HSET', claimed_key, job_data, worker_id)
        return job_data
    `)
    
    result, err := script.Run(ctx, c.rdb,
        []string{"jobs:pending", "jobs:claimed"},
        workerID,
    ).Result()
    
    if err != nil || result == nil {
        return nil, err
    }
    
    return []byte(result.(string)), nil
}

// Close releases Redis connection
func (c *Client) Close() error {
    return c.rdb.Close()
}
```

#### Core Logic Skeleton Code

**Scheduler Service Main Logic** (Skeleton for Milestone 3):

```go
// internal/coordinator/coordinator.go
package coordinator

import (
    "context"
    "sync"
    "time"
    
    "your-project/internal/models"
    "your-project/internal/storage/etcd" 
    "your-project/internal/storage/redis"
)

// Coordinator manages job scheduling and worker coordination
type Coordinator struct {
    etcdClient  *etcd.Client
    redisClient *redis.Client
    workers     map[string]*models.Worker
    isLeader    bool
    mu          sync.RWMutex
}

// NewCoordinator creates a new coordinator instance
func NewCoordinator(etcdClient *etcd.Client, redisClient *redis.Client) *Coordinator {
    return &Coordinator{
        etcdClient:  etcdClient,
        redisClient: redisClient,
        workers:     make(map[string]*models.Worker),
    }
}

// Start begins coordinator operations with leader election
func (c *Coordinator) Start(ctx context.Context) error {
    // TODO 1: Start leader election campaign in background goroutine
    // TODO 2: Start worker heartbeat monitoring loop
    // TODO 3: Start job assignment loop (only when leader)
    // TODO 4: Start job recovery scanner for failed workers
    // Hint: Use ctx.Done() to gracefully shutdown all goroutines
}

// ProcessWorkerHeartbeat handles incoming worker health signals
func (c *Coordinator) ProcessWorkerHeartbeat(workerID string, metadata models.Worker) error {
    // TODO 1: Validate worker metadata (capacity, capabilities)
    // TODO 2: Update worker's LastHeartbeat timestamp
    // TODO 3: If worker was marked UNAVAILABLE, transition to AVAILABLE
    // TODO 4: Update worker's current job count and capacity
    // Hint: Use c.mu.Lock() for thread-safe worker map updates
}

// AssignJobToWorker selects optimal worker and assigns job
func (c *Coordinator) AssignJobToWorker(job *models.Job) error {
    // TODO 1: Get list of AVAILABLE workers with matching capabilities
    // TODO 2: Select worker with lowest current load (currentJobs/capacity ratio)
    // TODO 3: Atomically claim job in Redis with worker ID and fencing token
    // TODO 4: Update worker's CurrentJobs count and state if at capacity
    // Hint: Use atomic operations to prevent race conditions in job claims
}

// RecoverFailedWorkerJobs reassigns jobs from unresponsive workers
func (c *Coordinator) RecoverFailedWorkerJobs(workerID string) error {
    // TODO 1: Query Redis for all jobs claimed by this worker
    // TODO 2: For each job, check if it's still executing (ping worker)
    // TODO 3: Reset job state from CLAIMED back to PENDING
    // TODO 4: Clear worker assignment and increment retry count
    // TODO 5: Remove worker from active registry
    // Hint: Use Lua scripts for atomic job state transitions
}
```

**Worker Service Core Logic** (Skeleton for all milestones):

```go
// internal/worker/worker.go  
package worker

import (
    "context"
    "time"
    
    "your-project/internal/models"
    "your-project/pkg/client"
)

// Worker represents a job execution node in the cluster
type Worker struct {
    id           string
    capacity     int
    capabilities []string
    currentJobs  int
    client       *client.SchedulerClient
    executors    map[string]JobExecutor
}

// JobExecutor defines interface for job type handlers
type JobExecutor interface {
    Execute(ctx context.Context, payload map[string]string) error
    GetTimeout() time.Duration
}

// Start begins worker operations with coordinator registration
func (w *Worker) Start(ctx context.Context) error {
    // TODO 1: Register worker with coordinator (POST /workers)
    // TODO 2: Start heartbeat goroutine (every 30 seconds)
    // TODO 3: Start job polling loop 
    // TODO 4: Handle graceful shutdown on ctx.Done()
    // Hint: Use sync.WaitGroup to coordinate goroutine shutdown
}

// pollForJobs continuously requests jobs from coordinator
func (w *Worker) pollForJobs(ctx context.Context) {
    // TODO 1: Call coordinator's claimJob() API endpoint
    // TODO 2: If job received, validate payload and find appropriate executor
    // TODO 3: Execute job in separate goroutine with timeout context
    // TODO 4: Report completion or failure back to coordinator
    // TODO 5: Handle job retry logic for transient failures
    // Hint: Respect worker capacity - don't claim more jobs than can handle
}

// executeJob runs job payload with timeout and error handling
func (w *Worker) executeJob(ctx context.Context, job *models.Job) error {
    // TODO 1: Find executor for job type from capabilities
    // TODO 2: Create timeout context based on job or executor limits
    // TODO 3: Call executor.Execute() with job payload
    // TODO 4: Handle context cancellation and timeout errors
    // TODO 5: Update job metrics (execution time, success/failure)
    // Hint: Use defer to ensure completion reporting even if job panics
}

// sendHeartbeat reports worker status to coordinator
func (w *Worker) sendHeartbeat() error {
    // TODO 1: Collect current worker metrics (currentJobs, capacity)
    // TODO 2: Send heartbeat to coordinator (POST /heartbeat)
    // TODO 3: Handle coordinator response (job assignments, shutdown signals)
    // TODO 4: Update worker state based on coordinator instructions
    // Hint: Include worker capabilities in heartbeat for dynamic job routing
}
```

#### File Structure Commands

To set up the recommended directory structure:

```bash
# Create main directory structure
mkdir -p cmd/{scheduler,worker,cli}
mkdir -p internal/{coordinator,cron,queue,worker,models,storage/{etcd,redis}}
mkdir -p pkg/{api/{handlers,middleware,types},client,metrics}
mkdir -p configs deployments/{kubernetes,docker,terraform}
mkdir -p tests/{integration,e2e,fixtures}

# Initialize Go module
go mod init distributed-job-scheduler

# Add essential dependencies
go get go.etcd.io/etcd/client/v3@latest
go get github.com/redis/go-redis/v9@latest  
go get github.com/gorilla/mux@latest
go get github.com/stretchr/testify@latest
go get gopkg.in/yaml.v3@latest
```

#### Milestone Checkpoints

**After Milestone 1 (Cron Parser)**:
```bash
# Test cron parsing functionality
cd internal/cron && go test -v
# Expected: All cron expression tests pass
# Manual verification: Create simple main.go that parses "@daily" and prints next 5 execution times
```

**After Milestone 2 (Priority Queue)**:
```bash  
# Test queue operations
cd internal/queue && go test -v
# Expected: Priority ordering, deduplication, delayed execution tests pass
# Manual verification: Start Redis, enqueue jobs with different priorities, verify dequeue order
```

**After Milestone 3 (Worker Coordination)**:
```bash
# Integration test with all components
go test ./tests/integration/... -v
# Expected: Multi-worker coordination, leader election, job recovery tests pass
# Manual verification: Start coordinator + 2 workers, kill one worker, verify job reassignment
```

#### Language-Specific Go Hints

- **Graceful Shutdown**: Use `context.Context` with `signal.NotifyContext()` for clean service shutdown
- **Atomic Operations**: Use `sync/atomic` package for counters, avoid mutex overhead for simple increments  
- **Time Handling**: Always use `time.UTC()` for cron calculations, convert to local time only for display
- **Error Wrapping**: Use `fmt.Errorf("operation failed: %w", err)` to maintain error chains
- **JSON Marshaling**: Implement custom `MarshalJSON()`/`UnmarshalJSON()` for complex types like `CronExpression`
- **Testing**: Use `testify/require` for test setup, `testify/assert` for verification - `require` stops on failure, `assert` continues
- **Redis Lua Scripts**: Store scripts as constants and use `redis.NewScript()` for atomic multi-key operations
- **HTTP Clients**: Set reasonable timeouts (`http.Client{Timeout: 30 * time.Second}`) to prevent hanging requests


## Data Model

> **Milestone(s):** This section defines the core data structures that underpin all three milestones - Job entities for Milestone 1's cron scheduling, Worker entities for Milestone 3's coordination, and Schedule models that tie everything together.

The data model serves as the foundation for our distributed job scheduler, defining how jobs, workers, and schedules are represented, stored, and related to each other. Think of the data model as the blueprint for a city's infrastructure - it defines the roads (relationships), buildings (entities), and addressing system (identifiers) that allow all the traffic (data flow) to move efficiently and safely.

A well-designed data model in a distributed system must balance several competing concerns: consistency across multiple nodes, performance under high load, and flexibility to support complex scheduling scenarios. Our model must also handle the temporal nature of scheduled jobs, where the same logical job may exist in multiple states across time, and the dynamic nature of worker nodes that can join, leave, or fail at any moment.

![Data Model Relationships](./diagrams/data-model.svg)

The core entities in our system form a triangle of relationships: Jobs represent work to be done, Workers represent compute capacity to execute that work, and Schedules represent the timing patterns that govern when jobs should run. Each entity has its own lifecycle, state management needs, and consistency requirements that we must carefully design.

### Job Definition

The `Job` entity represents a unit of work to be executed by the distributed scheduler. Think of a job as a train ticket - it contains all the information needed to identify the passenger (job payload), determine the route (scheduling information), track the journey (execution state), and ensure the ticket isn't used twice (idempotency).

Jobs in our system have a complex lifecycle that spans from creation through multiple execution attempts to final completion or failure. Unlike simple message queue systems where messages are consumed once, scheduled jobs may need to execute repeatedly on a cron schedule, require retry logic for transient failures, and maintain state across multiple worker nodes and coordinator failures.

| Field Name | Type | Description |
|------------|------|-------------|
| ID | string | Unique identifier for the job, typically a UUID to ensure global uniqueness across the distributed system |
| Name | string | Human-readable name for the job, used for monitoring and debugging purposes |
| CronExpression | string | Cron expression defining when this job should execute (e.g., "0 9 * * MON" for 9 AM every Monday) |
| Priority | int | Numeric priority where higher values indicate higher priority (0 = lowest, 100 = highest) |
| Payload | map[string]string | Key-value pairs containing the actual work parameters that workers need to execute the job |
| IdempotencyKey | string | Client-provided key to prevent duplicate job submissions; jobs with the same key are deduplicated |
| State | JobState | Current execution state of the job (PENDING, CLAIMED, EXECUTING, COMPLETED, FAILED) |
| WorkerID | string | Identifier of the worker currently assigned to execute this job; empty if not yet claimed |
| FencingToken | string | Unique token that prevents stale worker reports from affecting completed jobs |
| ScheduledAt | time.Time | When this job instance should execute; calculated from cron expression for recurring jobs |
| ClaimedAt | *time.Time | Timestamp when a worker claimed this job; nil if not yet claimed |
| CompletedAt | *time.Time | Timestamp when job execution finished (successfully or failed); nil if still in progress |
| RetryCount | int | Number of times this job has been retried after failure |
| MaxRetries | int | Maximum number of retry attempts before marking the job as permanently failed |
| CreatedAt | time.Time | Timestamp when this job was first created in the system |
| UpdatedAt | time.Time | Timestamp of the last modification to this job record |

The job state machine is central to ensuring exactly-once execution in our distributed environment. Each state transition represents a coordination point between the scheduler and workers, with careful attention to failure scenarios that could leave jobs in inconsistent states.

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| PENDING | Worker calls `claimJob()` | CLAIMED | Set WorkerID, ClaimedAt timestamp, generate FencingToken |
| CLAIMED | Worker starts execution | EXECUTING | Update state, maintain heartbeat timeout |
| EXECUTING | Worker calls `reportCompletion()` with success | COMPLETED | Set CompletedAt, clear WorkerID, schedule next instance if recurring |
| EXECUTING | Worker calls `reportCompletion()` with failure | FAILED or PENDING | Increment RetryCount, reset for retry or mark failed if MaxRetries exceeded |
| CLAIMED | Claim timeout expires | PENDING | Clear WorkerID and ClaimedAt to allow re-assignment |
| EXECUTING | Worker heartbeat timeout | PENDING | Clear WorkerID, increment RetryCount, log recovery action |

> **Decision: Fencing Token Strategy**
> - **Context**: Workers can crash, recover, or experience network delays that cause them to report completion for jobs that have already been reassigned to other workers
> - **Options Considered**: Monotonic sequence numbers, UUID-based tokens, timestamp-based tokens
> - **Decision**: UUID-based fencing tokens generated when jobs are claimed
> - **Rationale**: UUIDs provide uniqueness without requiring coordination between nodes, unlike sequence numbers that need centralized generation. They're also immune to clock skew issues that affect timestamp-based approaches
> - **Consequences**: Enables safe job recovery and prevents duplicate execution reports, but adds storage overhead and complexity to worker reporting logic

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Sequence Numbers | Simple ordering, small storage | Requires coordinated counter, single point of failure | No |
| UUID Tokens | No coordination needed, globally unique | Larger storage, no natural ordering | **Yes** |
| Timestamps | Natural ordering, human readable | Clock skew issues, not guaranteed unique | No |

The idempotency key mechanism prevents clients from accidentally submitting the same job multiple times due to network retries or application bugs. When a job is submitted with an idempotency key that already exists, the scheduler returns the existing job instead of creating a duplicate. This is similar to how payment systems prevent double-charging when a customer clicks "submit" multiple times.

The payload structure as a string map provides flexibility for different job types while maintaining simplicity for serialization and storage. More complex payload types can be JSON-encoded into string values, allowing the scheduler to remain agnostic about job content while still supporting rich data structures.

> The critical insight for job lifecycle management is that state transitions must be atomic and fenced. A job that transitions from EXECUTING to COMPLETED must never transition back to PENDING due to a delayed worker report, which is why fencing tokens are essential for correctness.

⚠️ **Pitfall: Race Condition Between Claim and Timeout**
When a worker claims a job but experiences a delay before starting execution, the coordinator might timeout the claim and reassign the job to another worker. Without proper fencing, both workers could execute the same job. The fencing token ensures that only the worker with the current token can successfully report completion, preventing this race condition.

⚠️ **Pitfall: Infinite Retry Loops**
Jobs that fail due to permanent conditions (bad payload data, missing dependencies) will consume resources indefinitely if retry logic doesn't distinguish between transient and permanent failures. The MaxRetries field provides a circuit breaker, but job implementations should also return non-retryable error types for permanent failures.

### Worker Model

The `Worker` entity represents a compute node capable of executing jobs in our distributed scheduler. Think of workers as delivery trucks in a logistics network - each truck has a capacity (how many packages it can carry), capabilities (refrigerated, oversized, hazardous materials), and a current location and status that the dispatch center needs to track for efficient job assignment.

Workers in our system are dynamic entities that can join and leave the cluster at any time. Unlike static partitioning schemes, our model supports elastic scaling where workers can be added during peak load periods and removed when demand decreases. This requires careful state management to ensure that work isn't lost when workers disappear unexpectedly.

| Field Name | Type | Description |
|------------|------|-------------|
| ID | string | Unique identifier for the worker, typically combining hostname and process ID for easy debugging |
| Address | string | Network address where the worker can be reached for job assignment and health checks (host:port format) |
| Capacity | int | Maximum number of concurrent jobs this worker can execute, based on CPU cores and memory |
| CurrentJobs | int | Number of jobs currently being executed by this worker |
| Capabilities | []string | List of job types or features this worker supports (e.g., ["gpu", "large-memory", "secure-enclave"]) |
| LastHeartbeat | time.Time | Timestamp of the most recent heartbeat signal from this worker |
| State | WorkerState | Current operational state of the worker (AVAILABLE, BUSY, UNAVAILABLE) |
| StartedAt | time.Time | When this worker first registered with the scheduler |
| Metadata | map[string]string | Additional worker-specific information like version, region, instance type |

The worker capacity model allows for load balancing based on actual resource utilization rather than simple round-robin assignment. Workers with higher capacity can accept more jobs, while workers approaching their limits are avoided for new assignments. This prevents overloading individual nodes while maximizing cluster utilization.

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| AVAILABLE | Job assigned and CurrentJobs >= Capacity | BUSY | Stop offering jobs to this worker |
| BUSY | Job completed and CurrentJobs < Capacity | AVAILABLE | Resume offering jobs to this worker |
| AVAILABLE/BUSY | Heartbeat timeout exceeded | UNAVAILABLE | Reassign all jobs, remove from assignment pool |
| UNAVAILABLE | Worker re-registers with `heartbeat()` | AVAILABLE | Reset LastHeartbeat, set CurrentJobs to 0 |

Worker capabilities enable job affinity and constraint-based scheduling. Jobs can specify required capabilities (e.g., GPU access, specific software versions), and the scheduler ensures they're only assigned to compatible workers. This is essential for workloads that need specialized hardware or have security requirements.

> **Decision: Heartbeat-Based Health Monitoring**
> - **Context**: The scheduler needs to detect worker failures quickly to reassign jobs, but network partitions and temporary slowdowns shouldn't trigger false positives
> - **Options Considered**: Pull-based health checks, push-based heartbeats, hybrid approach with both
> - **Decision**: Push-based heartbeat mechanism where workers periodically call `heartbeat()`
> - **Rationale**: Push-based heartbeats reduce coordinator overhead since workers contact the scheduler rather than the scheduler polling every worker. Workers can include status updates in heartbeat messages, providing richer information than simple ping responses
> - **Consequences**: Enables fast failure detection and efficient resource utilization tracking, but requires careful timeout tuning to balance false positives against detection latency

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Pull-based Health Checks | Coordinator controls timing, works through firewalls | High coordinator overhead, delayed updates | No |
| Push-based Heartbeats | Low coordinator overhead, real-time updates | Requires worker initiative, complex timeout handling | **Yes** |
| Hybrid Approach | Best of both worlds | Complex implementation, potential conflicts | No |

The metadata field allows workers to provide context that helps with debugging and operational monitoring. For example, workers can report their software version, AWS instance type, or geographic region. This information doesn't affect scheduling logic but is invaluable for troubleshooting performance issues or ensuring compliance with data locality requirements.

Worker address management handles the networking complexity of reaching workers for job assignment. In containerized environments, workers might have dynamic IP addresses or run behind load balancers. The address field accommodates these scenarios while providing a stable endpoint for coordinator communication.

> The key insight for worker lifecycle management is that workers are ephemeral resources that can disappear without notice, so the scheduler must be designed with failure as the default case rather than an exception. All worker state must be recoverable, and job assignments must include timeout mechanisms.

⚠️ **Pitfall: Heartbeat Timeout During Long Jobs**
Workers executing long-running jobs might miss heartbeat deadlines not because they've failed, but because they're busy processing. This can cause the coordinator to incorrectly mark them as unavailable and reassign their jobs. The heartbeat mechanism should be implemented as a separate goroutine that continues signaling even while jobs are executing.

⚠️ **Pitfall: Worker State Inconsistency**
If a worker's CurrentJobs count becomes inconsistent with reality (due to bugs or missed completion reports), it might permanently remain in BUSY state even when idle. Periodic reconciliation between the worker's actual job count and the coordinator's records prevents this deadlock scenario.

### Schedule Model

The `Schedule` model represents the temporal patterns that govern when jobs execute in our distributed scheduler. Think of schedules as the conductor's score in an orchestra - they define not just when each instrument (job) should play, but also how the timing coordinates with other schedules to create a harmonious system-wide performance.

Schedules in our system bridge the gap between human-readable cron expressions and the precise timing calculations needed by the distributed coordinator. They handle the complexity of timezone conversions, daylight saving time transitions, and edge cases like February 29th or "the 31st of every month" in months that only have 30 days.

| Field Name | Type | Description |
|------------|------|-------------|
| Original | string | The exact cron expression as provided by the user, preserved for debugging and display |
| Minutes | []int | Parsed minute values (0-59); empty slice means "every minute", single value means specific minute |
| Hours | []int | Parsed hour values (0-23); supports ranges like 9-17 for business hours |
| DaysOfMonth | []int | Parsed day-of-month values (1-31); handles month boundaries and leap years |
| Months | []int | Parsed month values (1-12); supports seasonal scheduling patterns |
| DaysOfWeek | []int | Parsed day-of-week values (0-6, Sunday=0); handles weekly recurring patterns |
| Timezone | *time.Location | Time zone for interpreting the cron expression; nil defaults to UTC |

The parsed field arrays enable efficient next-time calculation without repeatedly parsing the cron expression string. Each array contains the specific values when execution should occur, making the calculation algorithm a matter of finding the next valid combination across all dimensions.

Cron expression parsing handles several complex scenarios that naive implementations often miss. The interaction between day-of-month and day-of-week fields follows cron's "OR" semantics - a job scheduled for "15 * * * MON" runs both on the 15th of every month AND every Monday, not just Mondays that fall on the 15th.

> **Decision: Pre-parsed Field Arrays vs. Runtime Parsing**
> - **Context**: Next execution time calculation happens frequently as the scheduler evaluates which jobs are ready to run
> - **Options Considered**: Parse cron expression on every calculation, parse once and store parsed format, hybrid caching approach
> - **Decision**: Parse cron expressions once during job creation and store the parsed field arrays
> - **Rationale**: Parsing is computationally expensive and involves string manipulation and regex matching. Pre-parsing moves this cost to job creation time, which happens much less frequently than schedule evaluation
> - **Consequences**: Faster schedule evaluation and reduced CPU overhead during peak scheduling periods, but increases storage requirements and complicates cron expression updates

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Runtime Parsing | Low memory usage, handles dynamic changes | High CPU overhead, repeated work | No |
| Pre-parsed Storage | Fast evaluation, one-time parsing cost | Higher storage, complex updates | **Yes** |
| Hybrid Caching | Balance of speed and memory | Complex cache invalidation, potential consistency issues | No |

Timezone handling is one of the most complex aspects of schedule management. A job scheduled for "9 AM daily" in New York should execute at different UTC times throughout the year due to daylight saving time transitions. The Timezone field enables accurate local time interpretation while the coordinator operates on UTC internally.

The next execution time calculation algorithm works by finding the earliest future time that satisfies all the cron field constraints simultaneously:

1. **Start with the current time** rounded up to the next minute boundary (since cron expressions have minute-level granularity)
2. **Iterate through years** starting from the current year, but limit the search to prevent infinite loops for impossible expressions
3. **For each valid year, iterate through the months** specified in the Months array
4. **For each valid month, calculate candidate days** that satisfy either the DaysOfMonth constraint OR the DaysOfWeek constraint (cron's OR semantics)
5. **For each candidate day, iterate through the hours** specified in the Hours array
6. **For each valid hour, iterate through the minutes** specified in the Minutes array
7. **Return the first combination** that represents a time after the current time
8. **Handle timezone conversion** if the schedule specifies a non-UTC timezone

The algorithm must handle several edge cases that can cause infinite loops or incorrect results:

- **Impossible dates**: "31 * * 2 *" (February 31st) should never match any date
- **Leap year handling**: "29 * * 2 *" should only match in leap years
- **Daylight saving transitions**: 2:30 AM might not exist on spring-forward days
- **Year boundaries**: Ensure the search eventually terminates for expressions that have no valid future dates

> The critical insight for schedule calculation is that cron expressions define constraints rather than sequences. Unlike simple interval timers, cron schedules require constraint satisfaction across multiple temporal dimensions, making the calculation algorithm fundamentally a search problem.

Timezone support requires careful coordination between the schedule representation and the execution coordinator. The CronExpression stores timezone information and handles local-to-UTC conversion, but the rest of the system operates purely in UTC to avoid consistency issues across distributed nodes that might be running in different timezones.

⚠️ **Pitfall: Daylight Saving Time Transitions**
During spring-forward transitions, times like 2:30 AM simply don't exist in local time zones that observe daylight saving time. A naive implementation might either throw an error or calculate an incorrect UTC time. The correct approach is to detect these non-existent times and advance to the next valid time slot.

⚠️ **Pitfall: Day-of-Month vs Day-of-Week Confusion**
Many developers expect cron's day-of-month and day-of-week fields to work with AND semantics (both must be satisfied), but the actual cron standard uses OR semantics (either can be satisfied). A job scheduled for "0 9 15 * MON" runs at 9 AM both on the 15th of every month AND every Monday, not just Mondays that fall on the 15th.

⚠️ **Pitfall: Infinite Loop on Impossible Expressions**
Cron expressions like "0 0 30 2 *" (February 30th) will never have a valid execution time. The calculation algorithm must detect these impossible expressions and either reject them during parsing or terminate the search after a reasonable number of iterations to prevent infinite loops.

### Implementation Guidance

The data model implementation serves as the foundation for all three milestones, providing the type definitions and persistence layer that enable cron parsing, job queuing, and worker coordination. Think of this as building the database schema and entity classes that everything else will depend on.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Data Storage | JSON files with file locking | Redis with persistence enabled |
| Serialization | `encoding/json` with struct tags | Protocol Buffers with code generation |
| Time Handling | `time.Time` with UTC normalization | `time.Time` with timezone-aware helpers |
| Validation | Manual field validation | Struct validation library (go-playground/validator) |

**B. Recommended File Structure:**

```
project-root/
  internal/model/
    job.go              ← Job struct and methods
    worker.go           ← Worker struct and methods  
    schedule.go         ← CronExpression parsing and calculation
    storage.go          ← Data persistence interface
    types.go            ← Enums and constants
  internal/storage/
    redis.go            ← Redis storage implementation
    memory.go           ← In-memory storage for testing
  cmd/scheduler/
    main.go             ← Entry point
```

**C. Complete Infrastructure Starter Code:**

```go
// internal/model/types.go
package model

// JobState represents the current execution state of a job
type JobState int

const (
    PENDING JobState = iota
    CLAIMED
    EXECUTING  
    COMPLETED
    FAILED
)

func (s JobState) String() string {
    switch s {
    case PENDING:
        return "pending"
    case CLAIMED:
        return "claimed"
    case EXECUTING:
        return "executing"
    case COMPLETED:
        return "completed"
    case FAILED:
        return "failed"
    default:
        return "unknown"
    }
}

// WorkerState represents the current operational state of a worker
type WorkerState int

const (
    AVAILABLE WorkerState = iota
    BUSY
    UNAVAILABLE
)

func (s WorkerState) String() string {
    switch s {
    case AVAILABLE:
        return "available"
    case BUSY:
        return "busy"
    case UNAVAILABLE:
        return "unavailable"
    default:
        return "unknown"
    }
}

// Storage interface for persisting scheduler entities
type Storage interface {
    // Job operations
    CreateJob(job *Job) error
    GetJob(id string) (*Job, error)
    UpdateJob(job *Job) error
    ListJobs(state JobState) ([]*Job, error)
    
    // Worker operations  
    RegisterWorker(worker *Worker) error
    GetWorker(id string) (*Worker, error)
    UpdateWorker(worker *Worker) error
    ListWorkers(state WorkerState) ([]*Worker, error)
    
    // Cleanup operations
    DeleteJob(id string) error
    UnregisterWorker(id string) error
}
```

```go
// internal/storage/memory.go
package storage

import (
    "fmt"
    "sync"
    "github.com/yourorg/scheduler/internal/model"
)

// MemoryStorage provides in-memory storage for testing
type MemoryStorage struct {
    mu      sync.RWMutex
    jobs    map[string]*model.Job
    workers map[string]*model.Worker
}

func NewMemoryStorage() *MemoryStorage {
    return &MemoryStorage{
        jobs:    make(map[string]*model.Job),
        workers: make(map[string]*model.Worker),
    }
}

func (s *MemoryStorage) CreateJob(job *model.Job) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if _, exists := s.jobs[job.ID]; exists {
        return fmt.Errorf("job %s already exists", job.ID)
    }
    
    // Deep copy to prevent external modifications
    jobCopy := *job
    s.jobs[job.ID] = &jobCopy
    return nil
}

func (s *MemoryStorage) GetJob(id string) (*model.Job, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    job, exists := s.jobs[id]
    if !exists {
        return nil, fmt.Errorf("job %s not found", id)
    }
    
    // Deep copy to prevent external modifications
    jobCopy := *job
    return &jobCopy, nil
}

func (s *MemoryStorage) UpdateJob(job *model.Job) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    
    if _, exists := s.jobs[job.ID]; !exists {
        return fmt.Errorf("job %s not found", job.ID)
    }
    
    // Deep copy to prevent external modifications
    jobCopy := *job
    s.jobs[job.ID] = &jobCopy
    return nil
}

func (s *MemoryStorage) ListJobs(state model.JobState) ([]*model.Job, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    
    var jobs []*model.Job
    for _, job := range s.jobs {
        if job.State == state {
            jobCopy := *job
            jobs = append(jobs, &jobCopy)
        }
    }
    
    return jobs, nil
}

// Implement remaining Storage interface methods for workers...
// (Similar pattern to job methods)
```

**D. Core Logic Skeleton Code:**

```go
// internal/model/job.go
package model

import (
    "time"
    "github.com/google/uuid"
)

// Job represents a unit of work to be executed by the distributed scheduler
type Job struct {
    ID              string            `json:"id"`
    Name            string            `json:"name"`
    CronExpression  string            `json:"cron_expression"`
    Priority        int               `json:"priority"`
    Payload         map[string]string `json:"payload"`
    IdempotencyKey  string            `json:"idempotency_key"`
    State           JobState          `json:"state"`
    WorkerID        string            `json:"worker_id,omitempty"`
    FencingToken    string            `json:"fencing_token,omitempty"`
    ScheduledAt     time.Time         `json:"scheduled_at"`
    ClaimedAt       *time.Time        `json:"claimed_at,omitempty"`
    CompletedAt     *time.Time        `json:"completed_at,omitempty"`
    RetryCount      int               `json:"retry_count"`
    MaxRetries      int               `json:"max_retries"`
    CreatedAt       time.Time         `json:"created_at"`
    UpdatedAt       time.Time         `json:"updated_at"`
}

// NewJob creates a new job with default values and validation
func NewJob(name, cronExpr string, priority int, payload map[string]string) (*Job, error) {
    // TODO 1: Validate cron expression using CronExpression.Parse()
    // TODO 2: Generate unique ID using uuid.New().String()
    // TODO 3: Calculate initial ScheduledAt time from cron expression
    // TODO 4: Set CreatedAt and UpdatedAt to current time
    // TODO 5: Initialize State to PENDING, RetryCount to 0
    // TODO 6: Set MaxRetries to default value (e.g., 3)
    // Hint: Use time.Now().UTC() for consistent timezone handling
}

// ClaimForWorker atomically claims this job for the specified worker
func (j *Job) ClaimForWorker(workerID string) error {
    // TODO 1: Check if job is in PENDING state, return error if not
    // TODO 2: Generate new fencing token using uuid.New().String()
    // TODO 3: Set WorkerID, FencingToken, and ClaimedAt timestamp
    // TODO 4: Transition State to CLAIMED
    // TODO 5: Update the UpdatedAt timestamp
    // Hint: Caller must persist the job after claiming
}

// MarkExecuting transitions job to executing state with validation
func (j *Job) MarkExecuting(workerID, fencingToken string) error {
    // TODO 1: Validate that WorkerID matches the claimer
    // TODO 2: Validate that fencingToken matches current token  
    // TODO 3: Check that current State is CLAIMED
    // TODO 4: Transition State to EXECUTING
    // TODO 5: Update the UpdatedAt timestamp
}

// ReportCompletion handles job completion or failure reporting
func (j *Job) ReportCompletion(workerID, fencingToken string, success bool, errorMsg string) error {
    // TODO 1: Validate workerID and fencingToken match current assignment
    // TODO 2: Check that current State is EXECUTING
    // TODO 3: Set CompletedAt timestamp
    // TODO 4: If success, transition to COMPLETED state
    // TODO 5: If failure and retries available, increment RetryCount and reset to PENDING
    // TODO 6: If failure and no retries left, transition to FAILED state
    // TODO 7: Clear WorkerID and FencingToken
    // TODO 8: Update the UpdatedAt timestamp
    // Hint: Store errorMsg in Metadata field for debugging
}
```

```go
// internal/model/worker.go  
package model

import (
    "time"
)

// Worker represents a compute node capable of executing jobs
type Worker struct {
    ID            string            `json:"id"`
    Address       string            `json:"address"`
    Capacity      int               `json:"capacity"`
    CurrentJobs   int               `json:"current_jobs"`
    Capabilities  []string          `json:"capabilities"`
    LastHeartbeat time.Time         `json:"last_heartbeat"`
    State         WorkerState       `json:"state"`
    StartedAt     time.Time         `json:"started_at"`
    Metadata      map[string]string `json:"metadata"`
}

// NewWorker creates a new worker with validation
func NewWorker(id, address string, capacity int, capabilities []string) (*Worker, error) {
    // TODO 1: Validate that ID is non-empty and unique format
    // TODO 2: Validate that address is valid host:port format
    // TODO 3: Validate that capacity is positive integer
    // TODO 4: Set State to AVAILABLE and CurrentJobs to 0
    // TODO 5: Set StartedAt and LastHeartbeat to current time
    // TODO 6: Initialize Metadata map and copy capabilities slice
    // Hint: Use net.ResolveTCPAddr() to validate address format
}

// UpdateHeartbeat records a heartbeat and updates worker state
func (w *Worker) UpdateHeartbeat(currentJobs int) {
    // TODO 1: Update LastHeartbeat to current time
    // TODO 2: Update CurrentJobs to the reported value
    // TODO 3: Update State based on capacity: AVAILABLE if under capacity, BUSY if at/over capacity
    // TODO 4: Validate that currentJobs is non-negative and not greater than realistic limits
    // Hint: Consider adding validation for currentJobs > Capacity (might indicate bug)
}

// IsHealthy checks if worker is responding within timeout
func (w *Worker) IsHealthy(timeout time.Duration) bool {
    // TODO 1: Calculate time since LastHeartbeat
    // TODO 2: Return true if within timeout and State is not UNAVAILABLE
    // TODO 3: Return false if timeout exceeded or worker marked unavailable
}

// CanAcceptJob checks if worker can accept a new job
func (w *Worker) CanAcceptJob(requiredCapabilities []string) bool {
    // TODO 1: Check if State is AVAILABLE (not BUSY or UNAVAILABLE)
    // TODO 2: Check if CurrentJobs < Capacity
    // TODO 3: Check if worker has all required capabilities
    // TODO 4: Return true only if all conditions are met
    // Hint: Use string comparison to check capability matching
}
```

```go
// internal/model/schedule.go
package model

import (
    "time"
    "strings"
)

// CronExpression represents a parsed cron expression with timezone support
type CronExpression struct {
    Original     string         `json:"original"`
    Minutes      []int          `json:"minutes"`
    Hours        []int          `json:"hours"`  
    DaysOfMonth  []int          `json:"days_of_month"`
    Months       []int          `json:"months"`
    DaysOfWeek   []int          `json:"days_of_week"`
    Timezone     *time.Location `json:"timezone,omitempty"`
}

// ParseCronExpression parses a cron expression string into structured format
func ParseCronExpression(expr string) (*CronExpression, error) {
    // TODO 1: Split expression into fields (space-separated)
    // TODO 2: Validate field count (5 or 6 fields supported)
    // TODO 3: Parse each field using parseField() helper
    // TODO 4: Handle special expressions like @daily, @hourly
    // TODO 5: Set timezone to UTC if not specified
    // TODO 6: Store original expression for debugging
    // Hint: Use strings.Fields() to split on whitespace
}

// NextExecutionTime calculates when this cron expression should next run
func (c *CronExpression) NextExecutionTime(from time.Time) (time.Time, error) {
    // TODO 1: Convert from time to the cron expression's timezone
    // TODO 2: Round up to next minute boundary (cron has minute precision)
    // TODO 3: Iterate through years, months, days to find next valid time
    // TODO 4: Check both day-of-month and day-of-week constraints (OR semantics)
    // TODO 5: Handle edge cases like February 29, month boundaries
    // TODO 6: Convert result back to UTC before returning
    // TODO 7: Return error if no valid time found within reasonable range (2 years)
    // Hint: Use time.Date() to construct candidate times and check validity
}

// parseField parses a single cron field (minutes, hours, etc.) into integer slice
func parseField(field string, min, max int) ([]int, error) {
    // TODO 1: Handle wildcard "*" - return all values in range
    // TODO 2: Handle step values like "*/15" or "2-10/3"
    // TODO 3: Handle ranges like "9-17" 
    // TODO 4: Handle lists like "1,15,30"
    // TODO 5: Handle single values like "5"
    // TODO 6: Validate all values are within min-max range
    // TODO 7: Sort and deduplicate the result slice
    // Hint: Use strconv.Atoi() for string to int conversion
}
```

**E. Language-Specific Hints:**

- **Time Handling**: Always use `time.Now().UTC()` for consistency across distributed nodes. Use `time.LoadLocation()` for timezone parsing in cron expressions.
- **JSON Serialization**: Add `json` struct tags with `omitempty` for optional fields like `ClaimedAt *time.Time`.
- **UUID Generation**: Use `github.com/google/uuid` package for generating job IDs and fencing tokens.
- **Validation**: Consider using `github.com/go-playground/validator/v10` for struct validation with tags like `validate:"required,min=1,max=100"`.
- **Concurrency**: Use `sync.RWMutex` for storage implementations to allow concurrent reads while protecting writes.

**F. Milestone Checkpoint:**

After implementing the data model:
- Run `go test ./internal/model/...` - should pass basic struct creation and validation tests
- Create a job with `NewJob()` and verify all fields are populated correctly
- Parse a simple cron expression like `"0 9 * * MON"` and verify the parsed fields match expected values
- Test worker heartbeat updates and verify state transitions between AVAILABLE and BUSY
- Verify that job state transitions follow the documented state machine (can't go from COMPLETED back to EXECUTING)

Expected output when testing cron parsing:
```
Expression: "0 9 * * 1"
Minutes: [0]
Hours: [9] 
DaysOfMonth: [1,2,3,...,31]
Months: [1,2,3,...,12]
DaysOfWeek: [1]
Next execution: 2024-01-08 09:00:00 UTC (next Monday at 9 AM)
```

Signs that something is wrong:
- Job IDs are not unique across multiple calls to `NewJob()`
- Cron parsing returns incorrect field arrays (e.g., `*` doesn't expand to full range)
- Worker state doesn't change when `CurrentJobs` exceeds `Capacity`
- Time calculations are inconsistent across different system timezones


## Cron Expression Parser

> **Milestone(s):** Milestone 1 - Cron Expression Parser. This section implements the core scheduling logic that parses cron expressions and calculates next execution times with timezone support.

### Mental Model: Understanding cron as a calendar pattern matching system

Think of cron expressions as a sophisticated calendar filter that answers the question: "When should this job run next?" Imagine you're a personal assistant managing a complex schedule for an executive. Instead of writing down specific dates and times, you create rules like "every Monday at 9 AM" or "the 15th of every month at noon." Cron expressions work exactly the same way - they define patterns that match against the calendar to determine when jobs should execute.

The power of cron lies in its ability to express complex recurring patterns with simple field-based notation. Each field acts like a filter that constrains when execution can occur. When all five (or six) filters align - meaning the current time matches all specified constraints - the job becomes eligible for execution. This pattern-matching approach makes cron expressions incredibly flexible while remaining human-readable.

Consider the analogy of a combination lock. Just as a combination lock only opens when all tumblers align to the correct positions, a cron job only executes when the current time matches all field constraints simultaneously. The minute field must match, AND the hour field must match, AND the day constraints must be satisfied, and so on. This intersection-based logic is the fundamental principle underlying cron scheduling.

However, unlike simple calendar entries, cron expressions must handle edge cases that don't occur in human scheduling. What happens when you schedule a job for the 31st of every month, but February only has 28 days? How do you handle timezone transitions during daylight saving time? These complexities make cron parsing more challenging than it initially appears, requiring careful consideration of calendar arithmetic and timezone handling.

The `CronExpression` structure captures this pattern-matching concept by breaking down the textual cron expression into structured field lists. Instead of repeatedly parsing the string "0 9 * * 1", the parser converts it once into a format where minutes=[0], hours=[9], and daysOfWeek=[1], making subsequent time calculations much more efficient.

### Parsing Algorithm: Field-by-field validation and range expansion for cron expressions

The cron parsing algorithm transforms a textual cron expression into a structured representation that enables efficient time calculations. This process involves tokenization, field validation, and range expansion to handle the various syntactic features that cron expressions support.

> **Decision: Five-field vs Six-field Cron Support**
> - **Context**: Standard Unix cron uses five fields (minute, hour, day-of-month, month, day-of-week), but many modern schedulers support a six-field format that includes seconds.
> - **Options Considered**: Support only five-field format for simplicity, support only six-field format for precision, support both formats with auto-detection.
> - **Decision**: Support both formats with auto-detection based on field count.
> - **Rationale**: This provides maximum compatibility with existing cron expressions while enabling sub-minute scheduling precision when needed. Auto-detection eliminates the need for users to specify the format explicitly.
> - **Consequences**: Parsing logic becomes slightly more complex, but the scheduler can handle both traditional Unix cron expressions and modern high-precision scheduling requirements.

| Format Type | Field Count | Fields | Use Case |
|-------------|-------------|---------|----------|
| Traditional | 5 | minute, hour, day-of-month, month, day-of-week | Standard Unix cron compatibility |
| Extended | 6 | second, minute, hour, day-of-month, month, day-of-week | Sub-minute scheduling precision |

The parsing process follows a systematic approach that validates each field and expands shorthand notations into explicit value lists. The algorithm begins by tokenizing the input string on whitespace boundaries, then determines the format based on the number of fields detected. This auto-detection prevents users from having to specify the format explicitly while ensuring backward compatibility with existing cron expressions.

**Field Parsing Algorithm:**

1. **Tokenization**: Split the input string on whitespace to extract individual field values, handling multiple consecutive spaces and tab characters gracefully.

2. **Format Detection**: Count the number of fields to determine whether this is a five-field or six-field expression, adjusting field positions accordingly.

3. **Shorthand Expansion**: Check for special shorthand expressions like @yearly, @monthly, @daily, @hourly, and @reboot, converting them to their equivalent standard field representations.

4. **Field-by-Field Processing**: For each field position, validate the syntax and expand ranges, lists, and step values into explicit integer lists.

5. **Range Validation**: Ensure all expanded values fall within valid ranges for each field type (0-59 for minutes, 1-31 for day-of-month, etc.).

6. **Semantic Validation**: Check for semantic errors like invalid day-of-month values for specific months or conflicting day-of-week specifications.

Each field supports multiple syntactic forms that must be parsed and normalized into integer lists. The wildcard character (*) represents all valid values for that field. Ranges (1-5) specify inclusive spans of values. Lists (1,3,5) enumerate specific values explicitly. Step values (*/5 or 2-10/3) generate arithmetic progressions within specified ranges.

| Syntax Type | Example | Expansion | Description |
|-------------|---------|-----------|-------------|
| Wildcard | * | All valid values | Matches every possible value for this field |
| Specific Value | 15 | [15] | Matches exactly one value |
| Range | 1-5 | [1,2,3,4,5] | Matches all values in inclusive range |
| List | 1,3,5 | [1,3,5] | Matches explicitly enumerated values |
| Step (wildcard) | */5 | [0,5,10,15,...] | Every 5th value starting from minimum |
| Step (range) | 10-20/3 | [10,13,16,19] | Every 3rd value within range |

The range expansion logic handles edge cases carefully. When processing step values, the algorithm starts from the range minimum (or field minimum for wildcards) and increments by the step size until exceeding the range maximum. This ensures consistent behavior regardless of whether the step divides evenly into the range size.

Field validation enforces both syntactic and semantic constraints. Syntactic validation ensures proper format (no invalid characters, correct range syntax), while semantic validation checks domain-specific rules (no 31st day in February, valid timezone names). The validator maintains separate error lists for each category to provide detailed feedback about parsing failures.

> The critical insight in cron parsing is that expansion happens once at parse time, not at every time calculation. Converting "*/5" to [0,5,10,15,20,25,30,35,40,45,50,55] during parsing makes subsequent time calculations much faster, since they only need to check membership in pre-computed lists.

**Common Edge Cases in Parsing:**

The parser must handle several edge cases that frequently cause issues in cron implementations. Day-of-month and day-of-week interaction follows Unix cron semantics where both constraints are OR'ed together rather than AND'ed. This means "0 9 15 * 1" runs at 9 AM on the 15th of every month OR every Monday, not just on Mondays that fall on the 15th.

Timezone specifications can appear as suffixes to cron expressions in extended formats. The parser must extract timezone information and validate it against the system's timezone database before storing it in the `CronExpression` structure. Invalid timezone names should cause parsing failures with clear error messages.

Range boundaries require careful validation. Day-of-month ranges that include 31 are valid during parsing but may not match certain months during execution. The parser accepts these ranges but the time calculation algorithm must handle month-specific limitations appropriately.

| Field | Valid Range | Special Cases | Common Errors |
|-------|-------------|---------------|---------------|
| Second | 0-59 | N/A | Using 60 for leap seconds |
| Minute | 0-59 | N/A | Using 60 instead of 0 |
| Hour | 0-23 | N/A | Using 24 instead of 0 |
| Day-of-Month | 1-31 | Month-dependent validity | Using 0, assuming all months have 31 days |
| Month | 1-12 | N/A | Using 0-11 instead of 1-12 |
| Day-of-Week | 0-7 | Both 0 and 7 represent Sunday | Confusion about Sunday representation |

### Next Time Calculation: Algorithm for finding the next valid execution time from current timestamp

The next execution time calculation is the most complex part of cron processing because it must handle calendar arithmetic, timezone transitions, and the interaction between day-of-month and day-of-week constraints. The algorithm starts from the current timestamp and incrementally advances through time units until finding a moment that satisfies all field constraints simultaneously.

Think of this process like solving a multi-dimensional constraint satisfaction problem. Each cron field represents a constraint that the target timestamp must satisfy. The algorithm works by advancing time in the smallest possible increments - typically one minute for five-field expressions or one second for six-field expressions - and checking whether the current time satisfies all constraints.

> **Decision: Incremental vs Mathematical Next Time Calculation**
> - **Context**: There are two approaches to finding the next execution time - incrementally advancing time units until constraints are satisfied, or mathematically calculating the next valid time directly.
> - **Options Considered**: Pure incremental approach (simple but potentially slow), pure mathematical approach (fast but complex), hybrid approach that uses math for simple cases and incremental for complex ones.
> - **Decision**: Implement a hybrid approach that uses mathematical shortcuts for simple patterns but falls back to incremental advancement for complex constraints.
> - **Rationale**: Simple patterns like "every hour" or "daily at 3 AM" can be calculated directly using modular arithmetic, providing excellent performance. Complex patterns involving multiple constraints or day-of-week/day-of-month interactions require incremental checking to handle edge cases correctly.
> - **Consequences**: Provides optimal performance for common scheduling patterns while maintaining correctness for complex expressions. Code complexity is moderate since the mathematical shortcuts are optional optimizations.

**Next Time Calculation Algorithm:**

1. **Normalization**: Convert the current timestamp to the target timezone and truncate to the appropriate precision (minute or second boundary based on expression format).

2. **Constraint Checking**: Evaluate whether the current time satisfies all field constraints, returning immediately if it matches exactly.

3. **Increment Strategy**: Choose the appropriate time increment based on which constraints are failing - advance by seconds, minutes, hours, days, or months as appropriate.

4. **Calendar Advancement**: Increment the time by the chosen amount, handling month boundaries, leap years, and other calendar complexities correctly.

5. **Constraint Re-evaluation**: Check the new timestamp against all field constraints, repeating the process until a matching time is found.

6. **Infinite Loop Protection**: Implement safeguards to prevent infinite loops when no valid execution time exists within a reasonable future window.

The algorithm optimizes performance by choosing the largest possible time increment at each step. If the current hour doesn't match the hour constraint, there's no point in checking individual minutes - the algorithm can jump directly to the next valid hour. This coarse-grained advancement significantly reduces the number of iterations required.

| Failed Constraint | Increment Strategy | Next Check Point | Optimization Benefit |
|-------------------|-------------------|------------------|---------------------|
| Second | +1 second | Next second | Minimal (baseline increment) |
| Minute | Advance to next valid minute | Start of target minute | 60x faster than second-by-second |
| Hour | Advance to next valid hour | Start of target hour | 3600x faster than second-by-second |
| Day-of-Month | Advance to next valid day | Start of target day | Up to 86400x faster |
| Month | Advance to next valid month | Start of target month | Up to 2.6M x faster |

**Day-of-Week and Day-of-Month Interaction:**

The most complex aspect of cron time calculation is handling the interaction between day-of-month and day-of-week constraints when both are specified (neither is a wildcard). Unix cron semantics dictate that this creates an OR condition - the job runs if EITHER constraint is satisfied, not only when BOTH are satisfied simultaneously.

This means the expression "0 9 15 * 1" (9 AM on the 15th or on Mondays) has two separate series of execution times that must be computed and merged. The algorithm handles this by calculating next execution times for each constraint independently, then selecting the earliest result.

Consider a concrete example: if today is Wednesday, March 10th, and we're looking for the next execution of "0 9 15 * 1":
- Next 15th: March 15th at 9 AM (5 days from now)
- Next Monday: March 14th at 9 AM (4 days from now)  
- Result: March 14th at 9 AM (earlier of the two)

**Calendar Arithmetic and Edge Cases:**

The algorithm must handle numerous calendar edge cases that don't occur in typical application development. Month lengths vary (28, 29, 30, or 31 days), requiring leap year calculations for February. When advancing from January 31st to February, the algorithm must recognize that February 31st doesn't exist and adjust accordingly.

Timezone transitions during daylight saving time create additional complexity. When clocks "spring forward," certain times don't exist (2:30 AM might be skipped entirely). When clocks "fall back," certain times occur twice. The algorithm must handle these transitions gracefully, typically by selecting the first occurrence of ambiguous times and skipping non-existent times.

| Edge Case | Scenario | Algorithm Behavior | Example |
|-----------|----------|-------------------|---------|
| Missing day | Job scheduled for 31st, current month has 30 days | Advance to next month with sufficient days | April 31st → May 31st |
| Leap year | February 29th in non-leap year | Skip to next valid occurrence | Feb 29th 2023 → Feb 29th 2024 |
| DST spring forward | 2:30 AM doesn't exist | Skip to next valid time after transition | 2:30 AM → 3:30 AM |
| DST fall back | 2:30 AM occurs twice | Use first occurrence only | 2:30 AM (standard time) |
| Year boundary | December → January transition | Handle year increment correctly | Dec 31st → Jan 1st (next year) |

**Performance Optimization Strategies:**

For frequently-used cron expressions, the scheduler can implement caching strategies that pre-compute next execution times. Simple patterns like "every hour" or "daily at midnight" benefit from mathematical shortcuts that avoid iterative time advancement entirely.

The algorithm maintains an upper bound on search iterations to prevent infinite loops when no valid execution time exists. This can occur with malformed expressions or edge cases involving leap years and month boundaries. After exceeding the iteration limit, the algorithm returns an error rather than consuming infinite CPU time.

For expressions with large gaps between executions (like "every February 29th"), the algorithm uses month-level advancement to avoid checking individual days unnecessarily. This optimization is particularly important for expressions that match infrequently, where naive day-by-day advancement would be prohibitively expensive.

### Timezone Handling: UTC normalization and daylight saving time considerations

Timezone handling in distributed job schedulers requires careful consideration of where timezone conversions occur and how daylight saving time transitions are managed. The fundamental principle is to store all timestamps in UTC internally while supporting timezone-aware scheduling for user convenience.

Think of timezone handling like international conference call scheduling. Participants specify their local times ("let's meet at 3 PM my time"), but the calendar system converts everything to a common reference (UTC) for storage and comparison. When displaying times back to users, the system converts from UTC to their local timezone. This normalization approach prevents confusion and ensures consistent behavior across different scheduler nodes.

> **Decision: Timezone Storage and Conversion Strategy**
> - **Context**: Cron expressions can specify timezone information, but the scheduler must coordinate across multiple nodes that may be in different timezones.
> - **Options Considered**: Store all times in local timezone (simple but brittle), store all times in UTC (consistent but requires conversion), store times with timezone information (flexible but complex).
> - **Decision**: Store all absolute timestamps in UTC, but preserve timezone information in cron expressions for calculation purposes.
> - **Rationale**: UTC storage ensures consistency across scheduler nodes regardless of their local timezone configuration. Preserving timezone information in cron expressions allows proper handling of daylight saving time transitions and user-friendly scheduling.
> - **Consequences**: Requires timezone conversion at job scheduling time and display time, but provides robust behavior during timezone transitions and distributed deployment.

The `CronExpression` structure includes a `Timezone` field that specifies the timezone for interpreting the cron schedule. When this field is nil, the expression uses UTC. When specified, the timezone affects how the cron expression is evaluated against wall-clock time in that timezone, while the calculated execution times are still converted to UTC for storage.

**UTC Normalization Process:**

1. **Parse Timezone**: Extract timezone information from the cron expression or use system default if not specified.

2. **Calculate in Local Time**: Perform next time calculation in the specified timezone to handle daylight saving transitions correctly.

3. **Convert to UTC**: Transform the calculated local time to UTC for storage in job queues and execution tracking.

4. **Validate Conversion**: Ensure the conversion is unambiguous (handles DST transition edge cases).

5. **Store UTC Timestamp**: Persist the UTC timestamp while retaining timezone information for future calculations.

This normalization process ensures that jobs scheduled in different timezones execute at the correct absolute times, even when scheduler nodes are distributed across multiple geographic regions.

**Daylight Saving Time Transition Handling:**

Daylight saving time transitions create two types of temporal anomalies that cron schedulers must handle gracefully. During "spring forward" transitions, certain times don't exist (clocks jump from 1:59 AM directly to 3:00 AM). During "fall back" transitions, certain times occur twice (2:30 AM happens once in daylight time, then again in standard time).

For jobs scheduled during non-existent times, the algorithm applies a forward adjustment strategy. If a job is scheduled for 2:30 AM during a spring forward transition, it executes at the next valid time (typically 3:30 AM after the transition). This ensures jobs don't get permanently stuck waiting for a time that will never occur.

| Transition Type | Problem | Algorithm Behavior | Example |
|-----------------|---------|-------------------|---------|
| Spring Forward | Scheduled time doesn't exist | Execute at next valid time | 2:30 AM → 3:30 AM |
| Fall Back | Scheduled time occurs twice | Execute only during first occurrence | 2:30 AM (DST), skip 2:30 AM (Standard) |
| Timezone Change | Historical timezone rules | Use timezone rules active at execution time | Job scheduled before rule change |

For jobs scheduled during ambiguous times (times that occur twice), the scheduler executes the job only during the first occurrence. This prevents duplicate execution while maintaining predictable behavior. The algorithm achieves this by checking whether the calculated local time, when converted back to UTC, produces the expected UTC timestamp.

**Cross-Timezone Coordination:**

In distributed deployments where scheduler nodes run in different timezones, UTC normalization becomes critical for preventing duplicate job execution. Consider a scenario where one scheduler node runs in New York and another in Los Angeles. Without UTC normalization, both nodes might claim responsibility for executing a job scheduled for "3 AM Eastern Time" because their local time calculations would differ.

The solution is to perform all coordination and locking operations using UTC timestamps. When a scheduler node calculates that a job should execute at "3 AM Eastern Time," it converts this to a UTC timestamp (8 AM UTC during standard time, 7 AM UTC during daylight time) and uses this UTC value for all distributed coordination operations.

> The key insight for timezone handling is that scheduling calculations must happen in the target timezone to respect local rules like daylight saving time, but all coordination and storage must use UTC to ensure consistency across distributed nodes.

**Common Timezone Pitfalls:**

⚠️ **Pitfall: Storing Local Timestamps**  
Storing execution times in local timezone format makes the scheduler vulnerable to system timezone changes and creates inconsistencies in distributed deployments. Always convert calculated execution times to UTC before storage.

⚠️ **Pitfall: Ignoring DST Transitions**  
Calculating next execution times without considering daylight saving transitions can result in jobs running at unexpected times or not running at all. Always perform time calculations in the target timezone before converting to UTC.

⚠️ **Pitfall: Assuming Fixed UTC Offsets**  
Timezone offsets change throughout the year due to daylight saving time. Never hard-code UTC offsets; always use proper timezone libraries that handle historical and future timezone rule changes.

⚠️ **Pitfall: Invalid Timezone Names**  
Accepting arbitrary timezone names without validation can cause runtime errors during time calculations. Validate timezone names against the system's timezone database during cron expression parsing.

The scheduler should maintain a configurable policy for handling timezone-related errors. Options include failing the job submission, defaulting to UTC, or using the system's local timezone. The choice depends on whether the scheduler prioritizes safety (fail fast) or availability (best-effort execution).

| Error Condition | Conservative Behavior | Permissive Behavior | Recommended |
|-----------------|----------------------|---------------------|-------------|
| Invalid timezone name | Reject job submission | Default to UTC | Conservative |
| Ambiguous DST time | Reject job submission | Use first occurrence | Permissive |
| Non-existent DST time | Reject job submission | Adjust to next valid time | Permissive |
| Timezone rule changes | Require job resubmission | Apply new rules automatically | Depends on use case |

### Implementation Guidance

The cron expression parser implementation requires careful attention to timezone handling, calendar arithmetic, and performance optimization. Go's time package provides excellent support for timezone-aware calculations, making it the ideal choice for implementing robust cron functionality.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Cron Parsing | Custom parser with string splitting | Third-party library like robfig/cron |
| Time Calculations | Go's time package with manual arithmetic | Specialized calendar library |
| Timezone Data | Go's built-in time/tzdata | External timezone database |
| Performance | Naive incremental time advancement | Optimized mathematical shortcuts |

**Recommended File Structure:**

```
internal/cron/
  parser.go              ← CronExpression parsing logic
  parser_test.go         ← Comprehensive parsing tests
  calculator.go          ← Next time calculation algorithms
  calculator_test.go     ← Time calculation test cases
  timezone.go            ← Timezone handling utilities
  timezone_test.go       ← DST transition test cases
  expression.go          ← CronExpression data structures
  errors.go              ← Cron-specific error types
```

**Core Data Structures:**

```go
// CronExpression represents a parsed cron expression with timezone support.
// Fields contain expanded integer lists for efficient time matching.
type CronExpression struct {
    // Original contains the unparsed cron expression string for debugging
    Original string
    
    // Time field constraints as expanded integer lists
    Seconds      []int  // 0-59, nil for 5-field expressions
    Minutes      []int  // 0-59
    Hours        []int  // 0-23
    DaysOfMonth  []int  // 1-31
    Months       []int  // 1-12
    DaysOfWeek   []int  // 0-7 (both 0 and 7 represent Sunday)
    
    // Timezone specifies the timezone for schedule evaluation
    // nil means UTC
    Timezone *time.Location
}

// ParseResult contains parsing outcome with detailed error information
type ParseResult struct {
    Expression *CronExpression
    Errors     []ParseError
    Warnings   []string
}

// ParseError provides specific information about parsing failures
type ParseError struct {
    Field    string  // Which field caused the error
    Value    string  // The problematic value
    Position int     // Character position in original string
    Message  string  // Human-readable error description
}
```

**Cron Expression Parser Implementation:**

```go
// ParseCronExpression parses a cron expression string into structured format.
// Supports both 5-field (minute hour day month weekday) and 6-field 
// (second minute hour day month weekday) formats with automatic detection.
func ParseCronExpression(expr string) (*CronExpression, error) {
    // TODO 1: Trim whitespace and validate non-empty input
    
    // TODO 2: Check for shorthand expressions (@yearly, @monthly, @daily, etc.)
    // and convert to standard field format
    
    // TODO 3: Split expression on whitespace to extract individual fields
    
    // TODO 4: Determine format (5-field vs 6-field) based on field count
    
    // TODO 5: Parse each field according to its position and constraints
    // Use parseField() helper for each field type
    
    // TODO 6: Validate semantic constraints (day-of-month vs month compatibility)
    
    // TODO 7: Extract timezone information if present (typically as suffix)
    
    // TODO 8: Create and populate CronExpression struct with parsed values
    
    return nil, nil // Replace with actual implementation
}

// parseField parses a single cron field and returns the expanded integer list.
// Handles wildcards (*), ranges (1-5), lists (1,3,5), and steps (*/2, 1-10/3).
func parseField(field string, min, max int) ([]int, error) {
    // TODO 1: Handle wildcard (*) - return all values in range
    
    // TODO 2: Split on commas to handle lists (1,3,5,7-10)
    
    // TODO 3: For each list item, check for step syntax (value/step)
    
    // TODO 4: Parse ranges (start-end) and individual values
    
    // TODO 5: Expand step values within ranges
    
    // TODO 6: Validate all values are within min/max bounds
    
    // TODO 7: Sort and deduplicate the final value list
    
    return nil, nil // Replace with actual implementation
}
```

**Next Time Calculation Implementation:**

```go
// NextExecutionTime calculates the next time this cron expression should execute
// after the given timestamp. Returns time in UTC regardless of expression timezone.
func (c *CronExpression) NextExecutionTime(after time.Time) (time.Time, error) {
    // TODO 1: Convert 'after' timestamp to expression timezone for calculation
    
    // TODO 2: Truncate to appropriate precision (second or minute boundary)
    
    // TODO 3: Check if current time already matches all constraints
    
    // TODO 4: Implement main calculation loop with increment strategy:
    //   - Determine which constraint is failing
    //   - Choose appropriate time increment (second, minute, hour, day, month)
    //   - Advance time by chosen increment
    //   - Re-check all constraints
    
    // TODO 5: Handle day-of-week and day-of-month interaction (OR logic)
    
    // TODO 6: Implement infinite loop protection (max iterations limit)
    
    // TODO 7: Convert final result back to UTC before returning
    
    return time.Time{}, nil // Replace with actual implementation
}

// matchesConstraints checks if the given timestamp satisfies all cron field constraints.
// Time should be provided in the expression's timezone for accurate evaluation.
func (c *CronExpression) matchesConstraints(t time.Time) bool {
    // TODO 1: Extract time components (second, minute, hour, day, month, weekday)
    
    // TODO 2: Check each field constraint using contains() helper
    
    // TODO 3: Handle special day-of-week/day-of-month logic:
    //   - If both are specified (not wildcards), use OR logic
    //   - If only one is specified, use normal AND logic
    
    // TODO 4: Return true only if all applicable constraints are satisfied
    
    return false // Replace with actual implementation
}

// contains checks if a value exists in an integer slice (field constraint list)
func contains(slice []int, value int) bool {
    for _, v := range slice {
        if v == value {
            return true
        }
    }
    return false
}
```

**Timezone Handling Utilities:**

```go
// parseTimezone extracts timezone information from a cron expression.
// Supports timezone suffixes like "0 9 * * * America/New_York"
func parseTimezone(expr string) (*time.Location, string, error) {
    // TODO 1: Check for timezone suffix patterns (common timezone name formats)
    
    // TODO 2: Validate timezone name against time.LoadLocation
    
    // TODO 3: Return parsed timezone and expression with timezone removed
    
    return time.UTC, expr, nil // Replace with actual implementation
}

// convertToTimezone safely converts a UTC timestamp to the specified timezone,
// handling DST transitions and invalid times appropriately.
func convertToTimezone(utc time.Time, tz *time.Location) time.Time {
    if tz == nil {
        return utc
    }
    return utc.In(tz)
}

// convertToUTC converts a timezone-aware timestamp to UTC, handling
// ambiguous times during DST transitions consistently.
func convertToUTC(local time.Time) time.Time {
    return local.UTC()
}
```

**Milestone Checkpoint:**

After implementing the cron expression parser, you should be able to:

1. **Parse Valid Expressions**: `ParseCronExpression("0 9 * * 1")` returns a valid CronExpression with Minutes=[0], Hours=[9], DaysOfWeek=[1].

2. **Handle Extended Syntax**: `ParseCronExpression("30 0 9 * * 1-5")` correctly parses the 6-field format with seconds.

3. **Calculate Next Times**: For expression "0 9 * * *" at timestamp 2024-01-15 10:00:00 UTC, NextExecutionTime returns 2024-01-16 09:00:00 UTC.

4. **Manage Timezones**: Expression "0 9 * * * America/New_York" correctly adjusts for EST/EDT transitions.

**Test Commands:**
```bash
go test ./internal/cron/... -v
go test ./internal/cron/... -run TestDSTTransitions
go test ./internal/cron/... -run TestComplexExpressions
```

**Expected Behavior:**
- All parsing tests pass, including edge cases like February 29th and invalid timezone names
- Next time calculations handle month boundaries and leap years correctly  
- DST transition tests verify proper handling of spring forward and fall back scenarios
- Performance benchmarks show sub-millisecond parsing times for typical expressions

**Common Implementation Issues:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|----------------|-----|
| Jobs run twice during DST | Ambiguous time handling | Check logs during DST transition | Use first occurrence only |
| Jobs missing on 31st | Month boundary logic error | Test with expressions like "0 0 31 * *" | Skip months without target day |
| Parser panics on invalid input | Missing input validation | Test with malformed expressions | Add comprehensive input validation |
| Wrong timezone conversions | Hard-coded UTC offsets | Test around DST transitions | Use time.Location for all conversions |

![Cron Expression Parsing](./diagrams/cron-parsing-flow.svg)


## Priority Job Queue

> **Milestone(s):** Milestone 2 - Job Queue with Priorities. This section implements a distributed priority queue system that orders jobs by priority levels, supports delayed execution for scheduled jobs, and prevents duplicate submissions through idempotency keys.

### Mental Model: Priority Queue as a Hospital Triage System

Think of the priority job queue as a hospital emergency room triage system. When patients arrive, they don't get treated in first-come-first-served order. Instead, a triage nurse evaluates each patient and assigns a priority level based on urgency - heart attack patients go straight to the front, while someone with a minor cut waits longer. The triage system also handles scheduled appointments (delayed execution) and prevents the same patient from being registered multiple times if they're brought in by both an ambulance and a family member (deduplication).

The priority job queue works similarly. Jobs arrive with different priority levels, and the system ensures that high-priority jobs get executed before low-priority ones, even if they arrived later. Some jobs are "scheduled appointments" that shouldn't be processed until a specific time, like a daily backup that runs at 3 AM. The deduplication mechanism prevents the same job from being queued multiple times if a client retries a submission or if multiple systems try to schedule the same recurring task.

This mental model helps understand why we need more than a simple FIFO queue. Real-world job scheduling requires sophisticated ordering, timing controls, and duplicate prevention - just like a hospital needs more than a simple waiting line to handle the complexity of medical care prioritization.

### Priority Mechanism: Numeric Priority Levels and Heap-Based Ordering

The priority mechanism forms the core ordering logic of our job queue. Jobs are assigned numeric priority values where lower numbers indicate higher priority - similar to airline boarding groups where Group 1 boards before Group 5. This convention aligns with common computer science practice where priority 0 represents the highest urgency.

![Priority Queue Structure](./diagrams/job-priority-queue.svg)

The priority system uses five standard levels, though the implementation supports any integer value:

| Priority Level | Numeric Value | Description | Example Use Cases |
|----------------|---------------|-------------|-------------------|
| Critical | 0 | System-critical operations that must execute immediately | Security patches, critical alerts, system recovery |
| High | 100 | Important business operations with tight deadlines | Payment processing, customer notifications, data backups |
| Normal | 500 | Standard business operations with moderate timing requirements | Report generation, data synchronization, maintenance tasks |
| Low | 1000 | Background operations that can be delayed | Log cleanup, cache warming, analytics processing |
| Bulk | 2000 | Large batch operations that should yield to other work | Data imports, bulk exports, archive operations |

> **Decision: Numeric Priority with Lower-is-Higher Convention**
> - **Context**: Need a priority system that supports both predefined levels and custom priorities for specialized use cases
> - **Options Considered**: 
>   - Enum-based priorities (CRITICAL, HIGH, NORMAL, LOW)
>   - Numeric priorities with higher-is-higher (1-10 scale)
>   - Numeric priorities with lower-is-higher (0-N scale)
> - **Decision**: Numeric priorities with lower-is-higher convention
> - **Rationale**: Numeric values allow infinite granularity for custom priorities between standard levels. Lower-is-higher convention matches Unix process priorities and min-heap data structures, reducing cognitive overhead for systems programmers.
> - **Consequences**: Enables priority 150 jobs to run between High (100) and Normal (500). Requires documentation to clarify that 0 is highest priority, not lowest.

The queue implementation uses a min-heap data structure to maintain priority ordering efficiently. When jobs are inserted, the heap automatically positions them based on their priority value, ensuring that `Pop()` operations always return the highest-priority job in O(log n) time.

**Priority Comparison Algorithm:**

The system compares jobs using a multi-level comparison that considers both priority and timing:

1. **Primary Comparison**: Compare numeric priority values (lower wins)
2. **Secondary Comparison**: If priorities are equal, compare scheduled execution times (earlier wins)
3. **Tertiary Comparison**: If both priority and timing are equal, compare creation timestamps (older wins)
4. **Final Tiebreaker**: If all other fields are equal, compare job IDs lexicographically for deterministic ordering

This comparison ensures that within each priority level, jobs execute in temporal order, providing predictable behavior for jobs with identical priorities.

**Priority Inheritance and Boosting:**

The system supports priority boosting for jobs that have been waiting beyond acceptable thresholds. This prevents priority inversion where high-priority work depends on completion of lower-priority tasks:

| Boost Condition | Wait Time Threshold | Priority Adjustment | Rationale |
|------------------|---------------------|-------------------|-----------|
| Age-based boost | 30 minutes | Increase priority by 50 points | Prevents starvation of normal-priority jobs |
| Dependency boost | Immediate | Inherit dependent job's priority | Ensures dependency chains don't create bottlenecks |
| Resource boost | 15 minutes | Increase priority by 25 points | Prioritizes jobs holding scarce resources |

### Delayed Execution: Visibility Timeout Pattern for Scheduled Jobs

Delayed execution enables jobs to remain invisible in the queue until their designated execution time arrives. This mechanism supports both one-time scheduled jobs ("run this backup at 3 AM tomorrow") and recurring jobs managed by the cron expression parser ("run this report every Monday at 9 AM").

The implementation uses the **visibility timeout pattern** borrowed from message queue systems like Amazon SQS. Jobs exist in the queue data structure but remain invisible to workers until their `ScheduledAt` timestamp passes. This approach provides several advantages over alternative designs:

> **Decision: Visibility Timeout Pattern vs Separate Timer Service**
> - **Context**: Need to store jobs that shouldn't execute until a future time without requiring separate infrastructure
> - **Options Considered**: 
>   - External timer service that inserts jobs at scheduled time
>   - Separate "delayed jobs" storage with timer-based promotion
>   - Visibility timeout within single queue structure
> - **Decision**: Visibility timeout within single queue structure
> - **Rationale**: Single queue eliminates coordination complexity between timer service and queue. Visibility timeout pattern is well-understood from message queues. Redis native support reduces implementation complexity.
> - **Consequences**: Requires queue scanning to find eligible jobs, but Redis sorted sets make this efficient. Eliminates race conditions between timer service and queue operations.

**Delayed Job States and Transitions:**

Jobs progress through distinct visibility states as they approach their execution time:

| State | Description | Worker Visibility | Queue Location |
|-------|-------------|------------------|----------------|
| Scheduled | Job exists but execution time is future | Invisible | Delayed jobs sorted set |
| Eligible | Execution time has passed | Visible | Priority queue heap |
| Claimed | Worker has claimed job for processing | Invisible to other workers | Worker's active jobs set |
| Executing | Worker is actively processing job | Invisible | Worker's executing jobs set |

**Visibility Promotion Algorithm:**

The queue periodically scans for delayed jobs whose execution time has arrived and promotes them to visible status:

1. **Scan Trigger**: Every 30 seconds, or when queue becomes empty, or on explicit promotion request
2. **Time Range Query**: Query delayed jobs sorted set for entries with `ScheduledAt <= current_time`
3. **Batch Promotion**: Move up to 100 eligible jobs from delayed set to priority queue in single transaction
4. **Priority Insertion**: Insert each promoted job into priority queue heap based on its priority value
5. **Atomic Cleanup**: Remove promoted jobs from delayed set using Redis pipeline for atomicity

The batch promotion strategy balances responsiveness with efficiency. Promoting too few jobs creates unnecessary scanning overhead, while promoting too many jobs at once can cause latency spikes.

**Scheduling Precision and Drift:**

The visibility timeout pattern provides **eventual consistency** for job scheduling rather than real-time precision. Jobs become eligible within the scan interval (30 seconds by default) of their scheduled time:

| Precision Requirement | Scan Interval | Use Case Suitability |
|----------------------|---------------|---------------------|
| ±30 seconds | 30 seconds (default) | Most business operations, reports, backups |
| ±5 seconds | 5 seconds | Time-sensitive notifications, monitoring alerts |
| ±1 second | 1 second | Real-time processing, financial transactions |

> The system prioritizes operational simplicity over microsecond precision. For use cases requiring sub-second scheduling accuracy, consider dedicated real-time job scheduling systems rather than distributed queue-based approaches.

### Deduplication Strategy: Idempotency Keys and Hash-Based Detection

Deduplication prevents the same logical job from being queued multiple times, which commonly occurs when clients retry failed submissions, when multiple systems attempt to schedule the same recurring job, or when network issues cause duplicate messages. The system uses **idempotency keys** provided by job submitters combined with hash-based detection for automatic duplicate prevention.

**Idempotency Key Design:**

Each job submission includes an `IdempotencyKey` field that uniquely identifies the logical operation. When clients submit jobs with identical idempotency keys, only the first submission creates a queue entry - subsequent submissions return the original job information without creating duplicates.

| Idempotency Scope | Key Format | Example | Deduplication Window |
|------------------|------------|---------|---------------------|
| Global unique | UUID v4 | `550e8400-e29b-41d4-a716-446655440000` | Permanent |
| Operation-based | `operation:params:timestamp` | `backup:user-db:20240315` | 24 hours |
| Client-scoped | `client-id:sequence` | `payment-service:12345` | 1 hour |
| Content-based | SHA-256 of normalized payload | `sha256:a1b2c3...` | 1 hour |

The deduplication window determines how long the system remembers previous submissions. Permanent deduplication works for globally unique operations, while time-bounded deduplication suits operations that legitimately repeat after certain intervals.

**Hash-Based Automatic Detection:**

Beyond explicit idempotency keys, the system automatically detects duplicates by computing content hashes of job payloads. This catches duplicates even when clients don't provide idempotency keys or use different keys for identical operations.

The content hash includes normalized versions of key job fields:

1. **Normalized Job Name**: Converted to lowercase, whitespace trimmed
2. **Sorted Payload Keys**: Payload map keys sorted alphabetically for consistent hashing
3. **Canonical Cron Expression**: Cron expressions parsed and reformatted in standard form
4. **Priority Level**: Included to distinguish identical payloads with different priorities

**Deduplication Storage and Cleanup:**

The system maintains deduplication state in Redis using multiple data structures for efficient lookups and automatic cleanup:

| Storage Structure | Purpose | Key Pattern | Expiration |
|------------------|---------|-------------|------------|
| Idempotency map | Explicit key tracking | `dedup:key:{idempotency-key}` | Configurable (1-24 hours) |
| Content hash set | Automatic duplicate detection | `dedup:hash:{content-hash}` | 1 hour default |
| Job reference map | Links dedup entries to job IDs | `dedup:job:{job-id}` | Same as job TTL |

> **Decision: Redis-Based Deduplication vs In-Memory Cache**
> - **Context**: Need persistent deduplication that survives service restarts and works across multiple queue service instances
> - **Options Considered**: 
>   - In-memory LRU cache with configurable size
>   - Redis with TTL-based automatic cleanup
>   - Database table with periodic cleanup job
> - **Decision**: Redis with TTL-based automatic cleanup  
> - **Rationale**: Redis TTL provides automatic cleanup without manual intervention. Shared state across service instances prevents duplicates even during deployments. Redis performance characteristics suit high-frequency deduplication checks.
> - **Consequences**: Requires Redis infrastructure dependency. Network latency for every deduplication check. Memory usage grows with deduplication window size.

**Deduplication Algorithm:**

The complete deduplication check follows this sequence:

1. **Extract Idempotency Key**: Check if job submission includes explicit `IdempotencyKey`
2. **Check Explicit Deduplication**: Query Redis for existing entry with same idempotency key
3. **Early Return**: If idempotency key matches, return existing job ID without creating duplicate
4. **Compute Content Hash**: Calculate normalized hash of job name, payload, cron expression, and priority
5. **Check Content Deduplication**: Query Redis for existing entry with same content hash
6. **Hash Match Handling**: If content hash matches, compare full job details to confirm true duplicate vs hash collision
7. **Create New Entry**: If no duplicates found, create new job and store both idempotency key and content hash mappings
8. **Atomic Transaction**: Use Redis MULTI/EXEC to ensure deduplication state and job creation happen atomically

**Edge Cases and Collision Handling:**

The deduplication system handles several edge cases that can cause incorrect behavior:

⚠️ **Pitfall: Hash Collisions Causing False Duplicates**
Hash collisions can cause the system to incorrectly identify distinct jobs as duplicates. While SHA-256 collisions are extremely rare, they can occur with crafted inputs or in high-volume systems processing billions of jobs.

**Detection**: Jobs with different payloads are rejected as duplicates during submission.

**Fix**: When content hashes match, perform full job comparison including all payload fields. Only treat as duplicate if complete job specifications match exactly. Log hash collisions for monitoring.

⚠️ **Pitfall: Race Conditions During Concurrent Submission**
Multiple clients submitting identical jobs simultaneously can create duplicates if deduplication checks happen before job creation completes.

**Detection**: Multiple jobs exist with identical idempotency keys or content hashes.

**Fix**: Use Redis transactions (MULTI/EXEC) to make deduplication check and job creation atomic. Implement exponential backoff retry for clients when deduplication conflicts occur.

⚠️ **Pitfall: Stale Deduplication Entries After Job Deletion**
When jobs are deleted or expire, their deduplication entries may remain, preventing legitimate resubmission of the same operation.

**Detection**: Job submissions rejected for non-existent jobs referenced in deduplication entries.

**Fix**: Link deduplication entries to job TTL using Redis key expiration. Clean up orphaned deduplication entries during periodic maintenance.

### Implementation Guidance

This subsection provides the Go-specific implementation details for building the priority job queue with Redis backing store and atomic operations.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Queue Storage | Redis with Go-Redis client | Redis Cluster with sentinel failover |
| Priority Queue | Go container/heap interface | Custom heap with batch operations |
| Serialization | JSON with encoding/json | MessagePack with vmihailenco/msgpack |
| Time Handling | time.Time with UTC normalization | Dedicated time library like jinzhu/now |
| Deduplication | SHA-256 with crypto/sha256 | xxHash for performance-critical paths |
| Atomic Operations | Redis transactions (MULTI/EXEC) | Redis Lua scripts for complex operations |

**Recommended File Structure:**

```
internal/queue/
  queue.go                    ← PriorityQueue interface and Redis implementation
  queue_test.go              ← Unit tests for queue operations
  job.go                     ← Job struct and methods (NewJob, validation)
  job_test.go                ← Job creation and state transition tests
  deduplication.go           ← Idempotency and hash-based duplicate detection
  deduplication_test.go      ← Deduplication logic tests
  heap.go                    ← Go heap.Interface implementation for priorities
  redis_client.go            ← Redis connection management and health checks
  redis_scripts.go           ← Lua scripts for atomic Redis operations
  types.go                   ← JobState enum, configuration structs
internal/storage/
  interface.go               ← Storage interface for queue persistence
  redis_storage.go           ← Redis-specific storage implementation
cmd/queue-service/
  main.go                    ← Queue service entry point
  config.go                  ← Configuration loading and validation
```

**Infrastructure Starter Code:**

**Redis Client Setup (`internal/queue/redis_client.go`):**

```go
package queue

import (
    "context"
    "fmt"
    "time"

    "github.com/go-redis/redis/v8"
)

// RedisConfig holds Redis connection configuration
type RedisConfig struct {
    Address         string        `json:"address"`
    Password        string        `json:"password"`
    Database        int           `json:"database"`
    MaxRetries      int           `json:"max_retries"`
    PoolSize        int           `json:"pool_size"`
    ConnectTimeout  time.Duration `json:"connect_timeout"`
    ReadTimeout     time.Duration `json:"read_timeout"`
    WriteTimeout    time.Duration `json:"write_timeout"`
}

// NewRedisClient creates configured Redis client with health checking
func NewRedisClient(config RedisConfig) (*redis.Client, error) {
    client := redis.NewClient(&redis.Options{
        Addr:         config.Address,
        Password:     config.Password,
        DB:           config.Database,
        MaxRetries:   config.MaxRetries,
        PoolSize:     config.PoolSize,
        DialTimeout:  config.ConnectTimeout,
        ReadTimeout:  config.ReadTimeout,
        WriteTimeout: config.WriteTimeout,
    })

    // Test connection
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := client.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("failed to connect to Redis: %w", err)
    }

    return client, nil
}

// HealthCheck verifies Redis connectivity
func (c *RedisClient) HealthCheck(ctx context.Context) error {
    return c.client.Ping(ctx).Err()
}

// Close gracefully shuts down Redis connection
func (c *RedisClient) Close() error {
    return c.client.Close()
}
```

**Job State Management (`internal/queue/types.go`):**

```go
package queue

import (
    "fmt"
    "time"
)

// JobState represents current execution state of a job
type JobState string

const (
    PENDING   JobState = "pending"
    CLAIMED   JobState = "claimed"
    EXECUTING JobState = "executing"
    COMPLETED JobState = "completed"
    FAILED    JobState = "failed"
)

// String implements Stringer interface for JobState
func (js JobState) String() string {
    return string(js)
}

// IsValid checks if job state is one of defined values
func (js JobState) IsValid() bool {
    switch js {
    case PENDING, CLAIMED, EXECUTING, COMPLETED, FAILED:
        return true
    default:
        return false
    }
}

// CanTransitionTo checks if state transition is valid
func (js JobState) CanTransitionTo(next JobState) bool {
    transitions := map[JobState][]JobState{
        PENDING:   {CLAIMED, FAILED},
        CLAIMED:   {EXECUTING, PENDING, FAILED},
        EXECUTING: {COMPLETED, FAILED, PENDING},
        COMPLETED: {},  // Terminal state
        FAILED:    {},  // Terminal state
    }
    
    allowed, exists := transitions[js]
    if !exists {
        return false
    }
    
    for _, allowedNext := range allowed {
        if allowedNext == next {
            return true
        }
    }
    return false
}

// QueueConfig holds priority queue configuration
type QueueConfig struct {
    ScanInterval        time.Duration `json:"scan_interval"`
    BatchPromotionSize  int           `json:"batch_promotion_size"`
    DefaultJobTTL       time.Duration `json:"default_job_ttl"`
    DeduplicationWindow time.Duration `json:"deduplication_window"`
    MaxRetries          int           `json:"max_retries"`
    PriorityBoostAge    time.Duration `json:"priority_boost_age"`
    PriorityBoostAmount int           `json:"priority_boost_amount"`
}

// DefaultQueueConfig returns sensible defaults
func DefaultQueueConfig() QueueConfig {
    return QueueConfig{
        ScanInterval:        30 * time.Second,
        BatchPromotionSize:  100,
        DefaultJobTTL:       24 * time.Hour,
        DeduplicationWindow: 1 * time.Hour,
        MaxRetries:          3,
        PriorityBoostAge:    30 * time.Minute,
        PriorityBoostAmount: 50,
    }
}
```

**Core Logic Skeleton Code:**

**Priority Queue Interface (`internal/queue/queue.go`):**

```go
package queue

import (
    "context"
    "time"
)

// PriorityQueue defines interface for priority-based job queue
type PriorityQueue interface {
    // SubmitJob adds job to queue with deduplication check
    SubmitJob(ctx context.Context, job *Job) (*Job, error)
    
    // ClaimJob atomically assigns highest priority job to worker
    ClaimJob(ctx context.Context, workerID string) (*Job, error)
    
    // CompleteJob marks job as successfully completed
    CompleteJob(ctx context.Context, jobID string, result map[string]string) error
    
    // FailJob marks job as failed and handles retry logic
    FailJob(ctx context.Context, jobID string, reason string) error
    
    // PromoteDelayedJobs moves eligible delayed jobs to active queue
    PromoteDelayedJobs(ctx context.Context) (int, error)
    
    // GetQueueStats returns current queue statistics
    GetQueueStats(ctx context.Context) (QueueStats, error)
}

// SubmitJob adds new job to queue with comprehensive deduplication
func (pq *RedisPriorityQueue) SubmitJob(ctx context.Context, job *Job) (*Job, error) {
    // TODO 1: Validate job fields - check required fields, validate cron expression, verify priority range
    // TODO 2: Check idempotency key deduplication - query Redis for existing job with same IdempotencyKey
    // TODO 3: If idempotency match found, return existing job without creating duplicate
    // TODO 4: Compute content hash for automatic deduplication - normalize payload, hash job content
    // TODO 5: Check content hash deduplication - query Redis for jobs with same content hash
    // TODO 6: If content hash matches, compare full job details to handle hash collisions
    // TODO 7: Create new job with generated ID, timestamps, and initial PENDING state
    // TODO 8: Store job in appropriate queue - delayed queue if ScheduledAt is future, priority queue if immediate
    // TODO 9: Create deduplication entries atomically using Redis transaction
    // TODO 10: Return created job with populated metadata
    // Hint: Use Redis MULTI/EXEC for atomic job creation and deduplication storage
}

// ClaimJob atomically assigns highest priority available job to worker
func (pq *RedisPriorityQueue) ClaimJob(ctx context.Context, workerID string) (*Job, error) {
    // TODO 1: Promote any eligible delayed jobs to active queue
    // TODO 2: Get highest priority job from priority queue (lowest numeric priority value)
    // TODO 3: Check if job is already claimed or expired - validate job state and TTL
    // TODO 4: Atomically claim job for worker using Redis transaction with job state transition
    // TODO 5: Set job ClaimedAt timestamp and WorkerID, transition state to CLAIMED
    // TODO 6: Generate unique fencing token to prevent stale worker operations
    // TODO 7: Add job to worker's active jobs set for tracking
    // TODO 8: Set visibility timeout for job claim to handle worker failures
    // TODO 9: Return claimed job with worker metadata populated
    // Hint: Use Redis Lua script for atomic priority queue pop and claim operations
}
```

**Deduplication Implementation (`internal/queue/deduplication.go`):**

```go
package queue

import (
    "context"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "sort"
    "strings"
    "time"
)

// DeduplicationChecker handles job duplicate detection
type DeduplicationChecker struct {
    client *redis.Client
    config QueueConfig
}

// CheckDuplicate performs comprehensive deduplication check
func (dc *DeduplicationChecker) CheckDuplicate(ctx context.Context, job *Job) (*Job, bool, error) {
    // TODO 1: Check explicit idempotency key if provided by client
    // TODO 2: Query Redis for existing deduplication entry with same IdempotencyKey
    // TODO 3: If idempotency match found, retrieve and return existing job details
    // TODO 4: Compute normalized content hash of job payload and metadata
    // TODO 5: Query Redis for existing jobs with same content hash
    // TODO 6: For content hash matches, perform full job comparison to handle hash collisions
    // TODO 7: Return existing job if true duplicate found, nil if unique job
    // Hint: Use Redis pipelining for efficient multiple key lookups
}

// computeContentHash creates deterministic hash of job content
func (dc *DeduplicationChecker) computeContentHash(job *Job) string {
    // TODO 1: Create normalized representation of job for consistent hashing
    // TODO 2: Sort payload map keys alphabetically for deterministic ordering
    // TODO 3: Include job name (normalized), cron expression (canonical), priority level
    // TODO 4: Serialize normalized job data to JSON with consistent field ordering
    // TODO 5: Compute SHA-256 hash of serialized data and return hex encoding
    // Hint: Use json.Marshal with sorted maps for consistent serialization
}

// StoreDeduplicationEntry creates deduplication tracking entries
func (dc *DeduplicationChecker) StoreDeduplicationEntry(ctx context.Context, job *Job) error {
    // TODO 1: Create idempotency key mapping if IdempotencyKey is provided
    // TODO 2: Create content hash mapping for automatic duplicate detection
    // TODO 3: Set appropriate TTL on deduplication entries based on configuration
    // TODO 4: Use Redis transaction to create all deduplication entries atomically
    // TODO 5: Handle Redis errors and retry transient failures
    // Hint: Use Redis SETEX for TTL-based automatic cleanup of deduplication entries
}
```

**Heap Implementation for Priority Ordering (`internal/queue/heap.go`):**

```go
package queue

import (
    "container/heap"
    "time"
)

// JobHeap implements heap.Interface for priority-ordered jobs
type JobHeap []*Job

// Len returns number of jobs in heap
func (jh JobHeap) Len() int {
    return len(jh)
}

// Less compares two jobs for priority ordering (lower priority number = higher priority)
func (jh JobHeap) Less(i, j int) bool {
    // TODO 1: Compare priority values - lower numeric priority means higher priority
    // TODO 2: If priorities equal, compare scheduled execution times - earlier time wins
    // TODO 3: If priority and time equal, compare creation timestamps - older jobs first
    // TODO 4: Final tiebreaker using job IDs for deterministic ordering
    // Hint: Return true if job i should be popped before job j
}

// Swap exchanges two jobs in heap
func (jh JobHeap) Swap(i, j int) {
    jh[i], jh[j] = jh[j], jh[i]
}

// Push adds job to heap (called by heap.Push)
func (jh *JobHeap) Push(x interface{}) {
    *jh = append(*jh, x.(*Job))
}

// Pop removes highest priority job from heap (called by heap.Pop)
func (jh *JobHeap) Pop() interface{} {
    old := *jh
    n := len(old)
    job := old[n-1]
    *jh = old[0 : n-1]
    return job
}

// NewJobHeap creates initialized job heap
func NewJobHeap() *JobHeap {
    jh := &JobHeap{}
    heap.Init(jh)
    return jh
}
```

**Redis Lua Scripts for Atomic Operations (`internal/queue/redis_scripts.go`):**

```go
package queue

// Lua script for atomic job claim operation
const claimJobScript = `
local priority_queue_key = KEYS[1]
local worker_jobs_key = KEYS[2]
local job_key = KEYS[3]
local worker_id = ARGV[1]
local claim_timeout = ARGV[2]
local current_time = ARGV[3]

-- Get highest priority job from priority queue
local job_data = redis.call('ZPOPMIN', priority_queue_key, 1)
if #job_data == 0 then
    return nil  -- No jobs available
end

local job_id = job_data[1]
local job_info = redis.call('HGETALL', job_key .. job_id)

-- Check if job is still claimable
local job_state = job_info['state'] or 'unknown'
if job_state ~= 'pending' then
    return {'error', 'job_not_claimable', job_state}
end

-- Atomically claim job for worker
redis.call('HSET', job_key .. job_id, 
    'state', 'claimed',
    'worker_id', worker_id,
    'claimed_at', current_time)

-- Add to worker's active jobs
redis.call('SADD', worker_jobs_key, job_id)
redis.call('EXPIRE', worker_jobs_key, claim_timeout)

return {job_id, job_info}
`

// Additional Lua scripts for batch promotion, cleanup, etc.
const promoteDelayedJobsScript = `
-- Script for promoting delayed jobs to active queue
-- TODO: Implement batch delayed job promotion logic
`
```

**Language-Specific Hints:**

- Use `go-redis/redis/v8` for Redis client with context support and connection pooling
- Implement `container/heap` interface for efficient priority queue operations in memory
- Use `crypto/sha256` for content hashing with hex encoding for Redis key compatibility  
- Handle Redis pipeline operations for efficient batch deduplication checks
- Use `time.Time.UTC()` for consistent timestamp normalization across timezones
- Implement exponential backoff with `time.Sleep()` for Redis retry logic
- Use Redis transactions (`MULTI/EXEC`) for atomic job state transitions
- Consider Redis Lua scripts for complex atomic operations that require multiple commands

**Milestone Checkpoint:**

After implementing the priority queue foundation:

1. **Unit Test Verification**: Run `go test ./internal/queue/...` - all tests should pass
2. **Priority Ordering Test**: Submit jobs with priorities 100, 500, 100, 200 - claim order should be 100, 100, 200, 500  
3. **Deduplication Test**: Submit identical jobs with same idempotency key - only first submission should create queue entry
4. **Delayed Execution Test**: Submit job with `ScheduledAt` 30 seconds in future - job should not be claimable until promotion scan runs
5. **Redis Persistence Test**: Restart queue service - submitted jobs should still exist in Redis and be claimable

**Expected Behavior Verification:**
- Queue service starts without errors and connects to Redis successfully
- Job submission via REST API returns job ID and confirmation
- Duplicate job submissions return original job ID instead of creating new entries
- Priority ordering is maintained across service restarts
- Delayed jobs become claimable after their scheduled time passes

**Debugging Signs:**
- Jobs claimed out of priority order → Check heap Less() implementation and Redis ZPOPMIN usage
- Duplicate jobs created despite idempotency keys → Verify atomic Redis transactions and key formats  
- Delayed jobs never become eligible → Check promotion scan interval and Redis key expiration
- Memory leaks in long-running tests → Ensure proper Redis connection cleanup and heap memory management


## Worker Coordination

> **Milestone(s):** Milestone 3 - Worker Coordination. This section implements distributed worker management that enables fault-tolerant job execution across multiple worker nodes through leader election, heartbeat monitoring, and automatic job recovery when workers fail.

The worker coordination layer transforms our job scheduler from a single-node system into a truly distributed platform capable of scaling across multiple machines while maintaining consistency and fault tolerance. This coordination system ensures that jobs are fairly distributed across available workers, prevents duplicate execution through distributed locking, and automatically recovers from worker failures without losing work.

### Mental Model: Worker coordination as an orchestra with conductor and failure recovery

Think of worker coordination like a symphony orchestra performing a complex musical piece. The **conductor** (coordinator node) stands at the front, directing the performance by assigning musical parts to different musicians (workers) based on their instruments and current availability. Each musician must periodically make eye contact with the conductor (heartbeat) to receive cues and confirm they're still playing their assigned part.

When a violinist suddenly becomes ill and leaves mid-performance, the conductor notices immediately because they stopped responding to cues. The conductor then quickly reassigns that violinist's remaining musical phrases to other available violinists who have the same instrument and capacity. The audience (job submitters) never notices the disruption because the music continues seamlessly.

The critical insight is that coordination requires both **active leadership** (one conductor making assignment decisions) and **continuous monitoring** (heartbeats to detect failures). Without the conductor, musicians would play over each other chaotically. Without heartbeats, the conductor couldn't detect when musicians fail and need replacement.

Just as an orchestra has backup conductors ready to step in if the primary conductor collapses, our distributed system uses **leader election** to ensure exactly one coordinator is making decisions at any time, with other coordinator candidates ready to take over instantly upon failure.

![Worker State Machine](./diagrams/worker-state-machine.svg)

### Leader Election: Raft-like consensus for scheduler leadership and split-brain prevention

**Leader election** is the process by which multiple coordinator nodes agree on a single leader responsible for job assignment and worker management. This prevents the catastrophic **split-brain** scenario where multiple coordinators simultaneously make conflicting job assignments, leading to duplicate execution or worker thrashing.

![Leader Election Process](./diagrams/leader-election-flow.svg)

> **Decision: etcd-based Leader Election with Lease Mechanism**
> - **Context**: Multiple coordinator nodes need to agree on a single leader for job assignment decisions, with fast failover when the leader crashes
> - **Options Considered**: Database-based locking, Redis-based election, etcd lease-based election
> - **Decision**: etcd lease-based leader election with automatic renewal
> - **Rationale**: etcd provides strong consistency guarantees through Raft consensus, built-in lease expiration for fast failure detection, and watch mechanisms for immediate leadership change notifications
> - **Consequences**: Requires etcd as additional infrastructure dependency, but provides robust consensus with sub-second failover times

| Election Component | Purpose | Behavior |
|------------------|---------|----------|
| `Leadership Lease` | Time-bounded leadership claim | Coordinator holds renewable lease; leadership expires if not renewed |
| `Election Key` | Distributed lock for leadership | Single etcd key that only one coordinator can hold at a time |
| `Candidate Registration` | Discovery of potential leaders | All coordinator nodes register as candidates with their addresses |
| `Lease Renewal` | Maintain active leadership | Leader continuously renews lease every heartbeat interval |
| `Watch Mechanism` | Fast failure detection | Non-leader candidates watch election key for leadership changes |

The leader election algorithm follows these steps:

1. **Candidate Registration**: Each coordinator node attempts to create an etcd lease with a TTL of 30 seconds and writes its node ID and address to the leadership election key with that lease
2. **Leadership Determination**: The first node to successfully write to the election key becomes the leader; subsequent attempts fail due to key already existing
3. **Leadership Announcement**: The new leader broadcasts its leadership to all workers and begins accepting job assignment responsibilities
4. **Lease Renewal Loop**: The leader continuously renews its lease every 10 seconds (one-third of TTL) to maintain leadership
5. **Failure Detection**: If the leader fails to renew within the TTL window, etcd automatically deletes the election key, triggering immediate re-election
6. **Candidate Notification**: All candidate coordinators watch the election key; when it's deleted, they immediately attempt to claim leadership
7. **Split-Brain Prevention**: etcd's strong consistency ensures only one node can hold the election key, making simultaneous leadership impossible

> The critical insight is that **lease expiration** provides both failure detection and automatic cleanup. A crashed leader doesn't need to explicitly release leadership - the lease simply expires, triggering immediate failover.

**Leadership Responsibilities:**

| Responsibility | Leader Action | Non-Leader Action |
|---------------|---------------|------------------|
| Job Assignment | Assigns jobs from queue to available workers | Rejects assignment requests, returns leader address |
| Worker Health Monitoring | Processes heartbeats, detects failed workers | Ignores heartbeat messages |
| Job Recovery | Reassigns jobs from failed workers | Does not perform recovery operations |
| Schedule Management | Evaluates cron expressions, creates scheduled jobs | Passive; waits for leadership |
| Worker Registration | Accepts new worker registrations | Redirects to current leader |

#### Common Pitfalls

⚠️ **Pitfall: Lease TTL Too Short Causes Leadership Thrashing**
Setting the lease TTL too aggressively (e.g., 5 seconds) can cause false failovers during temporary network hiccups or garbage collection pauses. The leader temporarily cannot renew its lease, loses leadership, then immediately reclaims it, causing workers to experience multiple leadership changes.

**Solution**: Use a lease TTL of 30 seconds with renewal every 10 seconds, providing tolerance for temporary failures while maintaining sub-30-second failover times.

⚠️ **Pitfall: Stale Leadership Claims After Network Partition**
A coordinator that was partitioned from etcd but can still communicate with workers might continue acting as leader, unaware that its lease expired and another node claimed leadership.

**Solution**: Implement **fencing tokens** - the leader includes a monotonically increasing token in all job assignments. Workers reject assignments with tokens lower than the highest they've seen, preventing stale leaders from making assignments.

### Worker Registration: Dynamic worker discovery, capacity reporting, and capability matching

**Worker registration** enables dynamic scaling by allowing new workers to join the cluster and existing workers to report their current capacity and capabilities. This creates a live inventory that the coordinator uses for intelligent job assignment decisions.

The `Worker` data model captures all information needed for coordination:

| Field | Type | Description |
|-------|------|-------------|
| `ID` | `string` | Unique worker identifier (typically hostname + UUID) |
| `Address` | `string` | Network address for direct communication (host:port) |
| `Capacity` | `int` | Maximum number of concurrent jobs this worker can execute |
| `CurrentJobs` | `int` | Number of jobs currently executing on this worker |
| `Capabilities` | `[]string` | Tags indicating job types this worker can handle |
| `LastHeartbeat` | `time.Time` | Timestamp of most recent heartbeat received |
| `State` | `WorkerState` | Current operational state of the worker |
| `StartedAt` | `time.Time` | When this worker initially registered |
| `Metadata` | `map[string]string` | Additional worker-specific information |

**Worker States:**

| State | Meaning | Transition Triggers |
|-------|---------|-------------------|
| `AVAILABLE` | Ready to accept new job assignments | Heartbeat received, job completed |
| `BUSY` | At capacity, cannot accept new jobs | `CurrentJobs >= Capacity` |
| `UNAVAILABLE` | Failed or gracefully shutting down | Heartbeat timeout, explicit shutdown |

**Registration Process:**

1. **Initial Registration**: Worker sends registration request to coordinator with its capabilities and capacity
2. **Capability Validation**: Coordinator validates that worker capabilities match known job types
3. **Worker ID Assignment**: Coordinator assigns unique ID and records worker in coordination storage
4. **Heartbeat Initiation**: Worker begins sending periodic heartbeats to maintain registration
5. **Job Eligibility**: Worker becomes eligible for job assignments once registration is confirmed

> **Decision: Pull-Based Job Assignment vs Push-Based Distribution**
> - **Context**: Workers need to receive job assignments, but network failures can cause assignments to be lost
> - **Options Considered**: Coordinator pushes jobs to workers, workers poll coordinator for jobs, hybrid push-pull with acknowledgment
> - **Decision**: Workers poll coordinator for job assignments (pull-based)
> - **Rationale**: Pull-based assignment is more fault-tolerant because workers control timing and can retry failed requests. Push-based requires complex acknowledgment and retry logic when workers are temporarily unreachable.
> - **Consequences**: Slightly higher latency for job assignment, but much simpler failure handling and worker autonomy

**Capability Matching:**

Jobs include a `RequiredCapabilities` field that must be matched against worker capabilities for assignment eligibility:

| Matching Rule | Example | Behavior |
|---------------|---------|----------|
| Exact Match | Job requires `["python", "gpu"]`, Worker has `["python", "gpu", "docker"]` | ✅ Worker eligible |
| Missing Capability | Job requires `["python", "gpu"]`, Worker has `["python", "docker"]` | ❌ Worker not eligible |
| Subset Match | Job requires `["python"]`, Worker has `["python", "gpu", "docker"]` | ✅ Worker eligible |
| No Requirements | Job requires `[]`, Worker has any capabilities | ✅ Worker eligible |

![Job Execution Sequence](./diagrams/job-execution-sequence.svg)

### Heartbeat Mechanism: Failure detection through periodic health checks and timeout handling

The **heartbeat mechanism** provides continuous liveness monitoring that enables rapid failure detection and automatic job recovery. Workers send periodic heartbeat messages to prove they're operational and can continue executing assigned jobs.

**Heartbeat Message Structure:**

| Field | Type | Description |
|-------|------|-------------|
| `WorkerID` | `string` | Unique identifier of the sending worker |
| `Timestamp` | `time.Time` | When this heartbeat was generated |
| `CurrentJobs` | `int` | Number of jobs currently executing |
| `LoadAverage` | `float64` | System load indicator (optional) |
| `FreeMemory` | `int64` | Available memory in bytes (optional) |
| `Status` | `string` | Worker status message or health indicator |

**Heartbeat Timing Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Heartbeat Interval | 15 seconds | Frequent enough for quick failure detection, infrequent enough to avoid network overhead |
| Failure Threshold | 45 seconds | Three missed heartbeats before declaring worker failed |
| Grace Period | 60 seconds | Additional time for job completion during graceful shutdown |

The coordinator processes heartbeats through this algorithm:

1. **Heartbeat Reception**: Coordinator receives heartbeat message from worker
2. **Worker Lookup**: Verify worker ID exists in registered worker table
3. **Timestamp Update**: Update `LastHeartbeat` field with current time
4. **State Transition**: If worker was `UNAVAILABLE` due to timeout, transition back to `AVAILABLE`
5. **Load Tracking**: Update `CurrentJobs` count for capacity management
6. **Response Generation**: Send acknowledgment with current coordinator fencing token

**Failure Detection Process:**

1. **Timeout Scanning**: Coordinator runs background task every 10 seconds scanning all registered workers
2. **Staleness Check**: Calculate time since last heartbeat for each worker
3. **Failure Declaration**: Workers with heartbeats older than 45 seconds are marked `UNAVAILABLE`
4. **Job Recovery Trigger**: Failed workers trigger immediate job reassignment process
5. **Cleanup Scheduling**: Failed workers are removed from registry after 24 hours of inactivity

> The failure detection algorithm must balance **false positive rate** (healthy workers marked failed due to network delays) against **detection latency** (time to detect actual failures). Our 45-second threshold tolerates temporary network issues while enabling recovery within one minute.

**Graceful Shutdown Protocol:**

When workers need to shut down (deployment updates, maintenance), they follow this graceful shutdown sequence:

1. **Shutdown Signal**: Worker receives SIGTERM or shutdown command
2. **Stop Accepting Jobs**: Worker stops polling coordinator for new job assignments
3. **Completion Wait**: Worker continues executing current jobs with configurable timeout
4. **Status Broadcast**: Worker sends heartbeat with status "SHUTTING_DOWN"
5. **Final Heartbeat**: After job completion, worker sends final heartbeat with status "OFFLINE"
6. **Cleanup**: Coordinator removes worker from active registry

| Worker Method | Parameters | Returns | Description |
|---------------|------------|---------|-------------|
| `heartbeat()` | `status string, metrics map[string]float64` | `ack HeartbeatAck, err error` | Sends liveness signal with current status |
| `CanAcceptJob()` | `capabilities []string` | `bool` | Checks if worker can handle job with given requirements |
| `UpdateHeartbeat()` | `timestamp time.Time, status string` | `error` | Records heartbeat and updates worker state |
| `IsHealthy()` | `timeout time.Duration` | `bool` | Checks if worker heartbeat is within timeout window |

### Job Recovery: Detecting failed workers and reassigning their in-progress jobs

**Job recovery** ensures that work is never lost when workers fail unexpectedly. When a worker becomes unavailable, all jobs it was executing must be identified and reassigned to healthy workers to maintain system reliability.

The recovery process addresses several complex scenarios where workers fail at different stages of job execution:

| Failure Scenario | Job State | Recovery Action |
|-----------------|-----------|-----------------|
| Worker fails immediately after claiming job | `CLAIMED` | Reset to `PENDING`, make available for reassignment |
| Worker fails during job execution | `EXECUTING` | Reset to `PENDING`, increment retry count |
| Worker fails after completion but before reporting | `EXECUTING` | Check for side effects, potentially duplicate execution |
| Worker recovers before timeout | Any state | Cancel recovery, allow worker to continue |

**Recovery Algorithm:**

1. **Failure Detection**: Coordinator identifies worker as failed due to heartbeat timeout
2. **Job Query**: Query storage for all jobs with `WorkerID` matching failed worker
3. **State Analysis**: Examine each job's current state and execution progress
4. **Fencing Check**: Verify job hasn't completed using external side effect detection
5. **Job Reset**: Reset job state to `PENDING` and clear worker assignment
6. **Retry Logic**: Increment `RetryCount` and check against `MaxRetries` limit
7. **Reassignment**: Make jobs eligible for assignment to healthy workers
8. **Monitoring**: Log recovery actions for operational visibility

> **Decision: Optimistic Recovery vs Pessimistic Job Fencing**
> - **Context**: Workers might complete jobs after being declared failed, leading to duplicate execution if jobs are immediately reassigned
> - **Options Considered**: Immediate reassignment (optimistic), wait period before reassignment (pessimistic), external completion checking
> - **Decision**: Optimistic recovery with retry limits and idempotency requirements
> - **Rationale**: Most job failures occur before significant progress, and jobs should be designed for idempotent execution anyway. Waiting introduces unnecessary latency for legitimate failures.
> - **Consequences**: Possible duplicate execution in edge cases, but much faster recovery for genuine failures

**Fencing Token Mechanism:**

To prevent **stale worker reports** where a failed worker later reports job completion, the system uses fencing tokens:

| Component | Token Usage | Behavior |
|-----------|-------------|----------|
| Job Assignment | Coordinator assigns incrementing token with job | Token written to job record in storage |
| Worker Execution | Worker includes token in all status updates | Token proves worker has legitimate claim |
| Completion Reporting | Worker provides token when reporting results | Coordinator rejects reports with wrong token |
| Recovery Process | New assignment gets new token | Previous worker's reports become invalid |

**Recovery State Machine:**

| Current Job State | Recovery Action | Next State | Reason |
|------------------|----------------|------------|---------|
| `CLAIMED` | Reset assignment | `PENDING` | Worker failed before starting execution |
| `EXECUTING` | Check retry count | `PENDING` or `FAILED` | Worker failed during execution |
| `COMPLETING` | External verification | `COMPLETED` or `PENDING` | Worker may have finished before failure |

![Failure Recovery Sequence](./diagrams/failure-recovery-sequence.svg)

**Retry Logic During Recovery:**

The recovery process must decide whether failed jobs should be retried or marked as permanently failed:

```
if job.RetryCount >= job.MaxRetries:
    job.State = FAILED
    job.FailureReason = "Max retries exceeded after worker failure"
else:
    job.State = PENDING
    job.RetryCount += 1
    job.WorkerID = ""
    job.FencingToken = generateNewToken()
```

**Recovery Metrics and Monitoring:**

| Metric | Purpose | Alert Threshold |
|--------|---------|----------------|
| Jobs Recovered Per Hour | Track system stability | > 100/hour indicates worker instability |
| Recovery Latency | Time from failure to reassignment | > 2 minutes indicates coordinator issues |
| Duplicate Executions | Count jobs run multiple times | > 1% indicates timing problems |
| Recovery Success Rate | Percentage of recovered jobs that complete | < 90% indicates job design issues |

#### Common Pitfalls

⚠️ **Pitfall: Race Condition Between Recovery and Worker Return**
A worker experiences temporary network partition, coordinator declares it failed and reassigns its jobs, then worker reconnects and tries to report completion of already-reassigned jobs.

**Solution**: Use strict fencing token validation. When jobs are reassigned during recovery, they receive new fencing tokens. The original worker's completion reports include the old token and are rejected.

⚠️ **Pitfall: Infinite Recovery Loop for Consistently Failing Jobs**
Jobs that fail due to invalid input or missing dependencies get repeatedly reassigned to different workers, all of which fail for the same reason, creating an endless recovery cycle.

**Solution**: Implement exponential backoff for job reassignment after recovery. Jobs that fail multiple times across different workers should be delayed longer between retry attempts and eventually moved to a dead letter queue for manual inspection.

### Implementation Guidance

The worker coordination system requires careful orchestration of multiple Go services with strong consistency guarantees and efficient failure detection. The following implementation provides a production-ready foundation for distributed job scheduling.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Coordination Store | etcd v3 client | etcd with embedded proxy |
| Leader Election | etcd lease-based | Raft implementation |
| Inter-node Communication | HTTP REST + JSON | gRPC with Protocol Buffers |
| Configuration Management | YAML files + env vars | etcd-based dynamic config |
| Health Monitoring | HTTP health endpoints | Prometheus metrics + alerts |
| Logging | structured JSON logs | Distributed tracing with Jaeger |

**Recommended File Structure:**

```
internal/coordinator/
  coordinator.go           ← main coordinator logic and leader election
  coordinator_test.go      ← coordinator unit tests
  worker_manager.go        ← worker registration and heartbeat handling
  worker_manager_test.go   ← worker management tests
  job_recovery.go          ← failed worker job recovery logic
  job_recovery_test.go     ← recovery logic tests
  election.go              ← etcd-based leader election
  election_test.go         ← leader election tests
  fencing.go               ← token generation and validation
  fencing_test.go          ← fencing mechanism tests

internal/worker/
  worker.go                ← worker node implementation
  worker_test.go           ← worker unit tests
  heartbeat.go             ← heartbeat and health reporting
  heartbeat_test.go        ← heartbeat mechanism tests
  executor.go              ← job execution logic
  executor_test.go         ← execution tests

internal/etcd/
  client.go                ← etcd client wrapper with retries
  client_test.go           ← etcd integration tests
  lease.go                 ← lease management utilities
  watch.go                 ← etcd watch helper functions

pkg/coordination/
  types.go                 ← shared types for coordination messages
  constants.go             ← timing constants and configuration
```

**Core Coordinator Infrastructure (COMPLETE):**

```go
package coordinator

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "sync"
    "time"

    clientv3 "go.etcd.io/etcd/client/v3"
    "go.etcd.io/etcd/client/v3/concurrency"
)

// ETCDConfig holds etcd connection configuration
type ETCDConfig struct {
    Endpoints   []string      `yaml:"endpoints"`
    DialTimeout time.Duration `yaml:"dial_timeout"`
    Username    string        `yaml:"username"`
    Password    string        `yaml:"password"`
}

// CoordinatorConfig holds coordinator configuration
type CoordinatorConfig struct {
    NodeID              string        `yaml:"node_id"`
    ElectionPrefix      string        `yaml:"election_prefix"`
    LeaseTimeout        time.Duration `yaml:"lease_timeout"`
    HeartbeatInterval   time.Duration `yaml:"heartbeat_interval"`
    FailureThreshold    time.Duration `yaml:"failure_threshold"`
    RecoveryInterval    time.Duration `yaml:"recovery_interval"`
}

// Coordinator manages distributed job scheduling coordination
type Coordinator struct {
    config       CoordinatorConfig
    etcdClient   *clientv3.Client
    session      *concurrency.Session
    election     *concurrency.Election
    
    // Leadership state
    isLeader     bool
    leaderMux    sync.RWMutex
    fencingToken int64
    
    // Worker management
    workers      map[string]*Worker
    workerMux    sync.RWMutex
    
    // Background tasks
    ctx          context.Context
    cancel       context.CancelFunc
    wg           sync.WaitGroup
}

// NewCoordinator creates a new coordinator instance with etcd backend
func NewCoordinator(config CoordinatorConfig, etcdConfig ETCDConfig) (*Coordinator, error) {
    etcdClient, err := clientv3.New(clientv3.Config{
        Endpoints:   etcdConfig.Endpoints,
        DialTimeout: etcdConfig.DialTimeout,
        Username:    etcdConfig.Username,
        Password:    etcdConfig.Password,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to connect to etcd: %w", err)
    }

    session, err := concurrency.NewSession(etcdClient, 
        concurrency.WithTTL(int(config.LeaseTimeout.Seconds())))
    if err != nil {
        return nil, fmt.Errorf("failed to create etcd session: %w", err)
    }

    ctx, cancel := context.WithCancel(context.Background())
    
    return &Coordinator{
        config:     config,
        etcdClient: etcdClient,
        session:    session,
        election:   concurrency.NewElection(session, config.ElectionPrefix),
        workers:    make(map[string]*Worker),
        ctx:        ctx,
        cancel:     cancel,
    }, nil
}

// Start begins coordinator operations including leader election
func (c *Coordinator) Start() error {
    log.Printf("Starting coordinator %s", c.config.NodeID)
    
    // Start leader election
    c.wg.Add(1)
    go c.runLeaderElection()
    
    // Start worker monitoring
    c.wg.Add(1)
    go c.runWorkerMonitoring()
    
    // Start job recovery
    c.wg.Add(1)
    go c.runJobRecovery()
    
    return nil
}

// Stop gracefully shuts down coordinator
func (c *Coordinator) Stop() error {
    log.Printf("Stopping coordinator %s", c.config.NodeID)
    
    c.cancel()
    c.wg.Wait()
    
    if c.session != nil {
        c.session.Close()
    }
    
    if c.etcdClient != nil {
        c.etcdClient.Close()
    }
    
    return nil
}

// IsLeader returns current leadership status
func (c *Coordinator) IsLeader() bool {
    c.leaderMux.RLock()
    defer c.leaderMux.RUnlock()
    return c.isLeader
}

// GetFencingToken returns current fencing token for job assignments
func (c *Coordinator) GetFencingToken() int64 {
    c.leaderMux.RLock()
    defer c.leaderMux.RUnlock()
    return c.fencingToken
}
```

**Worker Heartbeat Infrastructure (COMPLETE):**

```go
package worker

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

// WorkerConfig holds worker configuration
type WorkerConfig struct {
    ID                 string            `yaml:"id"`
    Address            string            `yaml:"address"`
    Capacity           int               `yaml:"capacity"`
    Capabilities       []string          `yaml:"capabilities"`
    CoordinatorAddress string            `yaml:"coordinator_address"`
    HeartbeatInterval  time.Duration     `yaml:"heartbeat_interval"`
    ShutdownTimeout    time.Duration     `yaml:"shutdown_timeout"`
    Metadata           map[string]string `yaml:"metadata"`
}

// Worker represents a job execution node
type Worker struct {
    config        WorkerConfig
    httpClient    *http.Client
    
    // Job execution
    currentJobs   map[string]*Job
    jobMux        sync.RWMutex
    
    // Lifecycle management
    ctx           context.Context
    cancel        context.CancelFunc
    wg            sync.WaitGroup
    isShuttingDown bool
    shutdownMux   sync.RWMutex
}

// NewWorker creates a new worker instance
func NewWorker(config WorkerConfig) *Worker {
    ctx, cancel := context.WithCancel(context.Background())
    
    return &Worker{
        config: config,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
        currentJobs: make(map[string]*Job),
        ctx:         ctx,
        cancel:      cancel,
    }
}

// Start begins worker operations including registration and heartbeating
func (w *Worker) Start() error {
    log.Printf("Starting worker %s", w.config.ID)
    
    // Register with coordinator
    if err := w.registerWithCoordinator(); err != nil {
        return fmt.Errorf("failed to register: %w", err)
    }
    
    // Start heartbeat loop
    w.wg.Add(1)
    go w.runHeartbeatLoop()
    
    // Start job polling loop
    w.wg.Add(1)
    go w.runJobPollingLoop()
    
    return nil
}

// Stop gracefully shuts down worker
func (w *Worker) Stop() error {
    log.Printf("Stopping worker %s", w.config.ID)
    
    w.shutdownMux.Lock()
    w.isShuttingDown = true
    w.shutdownMux.Unlock()
    
    // Cancel context to stop background loops
    w.cancel()
    
    // Wait for current jobs to complete with timeout
    done := make(chan struct{})
    go func() {
        w.wg.Wait()
        close(done)
    }()
    
    select {
    case <-done:
        log.Printf("Worker %s stopped gracefully", w.config.ID)
    case <-time.After(w.config.ShutdownTimeout):
        log.Printf("Worker %s shutdown timeout, forcing stop", w.config.ID)
    }
    
    return nil
}

// HeartbeatMessage represents worker heartbeat data
type HeartbeatMessage struct {
    WorkerID     string            `json:"worker_id"`
    Timestamp    time.Time         `json:"timestamp"`
    CurrentJobs  int               `json:"current_jobs"`
    Status       string            `json:"status"`
    Metadata     map[string]string `json:"metadata"`
}

// registerWithCoordinator sends initial registration request
func (w *Worker) registerWithCoordinator() error {
    registrationData := map[string]interface{}{
        "worker_id":    w.config.ID,
        "address":      w.config.Address,
        "capacity":     w.config.Capacity,
        "capabilities": w.config.Capabilities,
        "metadata":     w.config.Metadata,
    }
    
    // Implementation continues with HTTP request to coordinator...
    return nil
}

// GetCurrentJobCount returns number of jobs currently executing
func (w *Worker) GetCurrentJobCount() int {
    w.jobMux.RLock()
    defer w.jobMux.RUnlock()
    return len(w.currentJobs)
}

// CanAcceptJob checks if worker can handle job with given requirements
func (w *Worker) CanAcceptJob(requiredCapabilities []string) bool {
    w.shutdownMux.RLock()
    defer w.shutdownMux.RUnlock()
    
    if w.isShuttingDown {
        return false
    }
    
    if w.GetCurrentJobCount() >= w.config.Capacity {
        return false
    }
    
    // Check capability matching
    workerCaps := make(map[string]bool)
    for _, cap := range w.config.Capabilities {
        workerCaps[cap] = true
    }
    
    for _, required := range requiredCapabilities {
        if !workerCaps[required] {
            return false
        }
    }
    
    return true
}
```

**Core Logic Skeleton for Leader Election:**

```go
// runLeaderElection handles leader election and maintains leadership
func (c *Coordinator) runLeaderElection() {
    defer c.wg.Done()
    
    for {
        select {
        case <-c.ctx.Done():
            return
        default:
            // TODO 1: Campaign for leadership using etcd election
            // Use: c.election.Campaign(c.ctx, c.config.NodeID)
            
            // TODO 2: If campaign succeeds, set leadership state and generate new fencing token
            // Set: c.isLeader = true, c.fencingToken = time.Now().UnixNano()
            
            // TODO 3: Start leadership maintenance loop
            // Monitor election key and renew session lease
            
            // TODO 4: Handle leadership loss detection
            // Watch for election key changes or session expiration
            
            // TODO 5: Clean up leadership state when stepping down
            // Set: c.isLeader = false, clear cached state
        }
    }
}

// runWorkerMonitoring periodically checks worker health and removes failed workers
func (c *Coordinator) runWorkerMonitoring() {
    defer c.wg.Done()
    
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-c.ctx.Done():
            return
        case <-ticker.C:
            // TODO 1: Check if this node is the leader
            // Skip monitoring if not leader to avoid conflicts
            
            // TODO 2: Iterate through all registered workers
            // Get worker list from c.workers map with proper locking
            
            // TODO 3: Check each worker's last heartbeat timestamp
            // Compare time.Since(worker.LastHeartbeat) against c.config.FailureThreshold
            
            // TODO 4: Mark workers as UNAVAILABLE if heartbeat is stale
            // Update worker.State and log failure detection
            
            // TODO 5: Trigger job recovery for newly failed workers
            // Call job recovery process for workers that just became unavailable
        }
    }
}

// ProcessWorkerHeartbeat handles incoming heartbeat from worker
func (c *Coordinator) ProcessWorkerHeartbeat(heartbeat *HeartbeatMessage) error {
    // TODO 1: Validate heartbeat message fields
    // Check that WorkerID is non-empty and timestamp is recent
    
    // TODO 2: Acquire worker registry lock
    // Use c.workerMux.Lock() for exclusive access
    
    // TODO 3: Look up worker in registry
    // Find worker by heartbeat.WorkerID in c.workers map
    
    // TODO 4: Update worker heartbeat timestamp and status
    // Set LastHeartbeat = time.Now(), update CurrentJobs count
    
    // TODO 5: Transition worker state if it was previously UNAVAILABLE
    // If worker.State == UNAVAILABLE, set to AVAILABLE
    
    // TODO 6: Return heartbeat acknowledgment with current fencing token
    // Include c.GetFencingToken() in response for worker validation
    
    return nil
}
```

**Core Logic Skeleton for Job Recovery:**

```go
// runJobRecovery periodically checks for jobs from failed workers and reassigns them
func (c *Coordinator) runJobRecovery() {
    defer c.wg.Done()
    
    ticker := time.NewTicker(c.config.RecoveryInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-c.ctx.Done():
            return
        case <-ticker.C:
            // TODO 1: Check if this node is the leader
            // Only leader should perform job recovery
            
            // TODO 2: Find all jobs assigned to UNAVAILABLE workers
            // Query storage for jobs where WorkerID matches failed workers
            
            // TODO 3: Group jobs by their current state
            // Handle CLAIMED, EXECUTING, and COMPLETING states differently
            
            // TODO 4: Reset job assignments and increment retry counts
            // Clear WorkerID, generate new FencingToken, increment RetryCount
            
            // TODO 5: Check retry limits and mark permanently failed jobs
            // If RetryCount >= MaxRetries, set State = FAILED
            
            // TODO 6: Make recovered jobs available for reassignment
            // Set State = PENDING for jobs that can be retried
        }
    }
}

// RecoverJobsFromWorker handles job recovery when a specific worker fails
func (c *Coordinator) RecoverJobsFromWorker(workerID string) error {
    // TODO 1: Query storage for all jobs assigned to this worker
    // Use storage query: WHERE WorkerID = workerID AND State IN (CLAIMED, EXECUTING)
    
    // TODO 2: For each job, determine recovery action based on current state
    // CLAIMED jobs can be immediately reset, EXECUTING jobs need retry logic
    
    // TODO 3: Generate new fencing tokens for recovered jobs
    // Use time.Now().UnixNano() + job sequence for uniqueness
    
    // TODO 4: Update job records with recovery information
    // Clear WorkerID, set new FencingToken, increment RetryCount
    
    // TODO 5: Log recovery actions for operational monitoring
    // Include workerID, jobID, old state, new state in log messages
    
    // TODO 6: Update job queue to make recovered jobs available
    // Call priority queue methods to reinsert jobs for assignment
    
    return nil
}

// ValidateFencingToken checks if worker has authority to report on job
func (c *Coordinator) ValidateFencingToken(jobID string, workerID string, token string) (bool, error) {
    // TODO 1: Look up current job record in storage
    // Get job by jobID and check current WorkerID and FencingToken
    
    // TODO 2: Verify worker ID matches job assignment
    // Return false if job.WorkerID != workerID
    
    // TODO 3: Compare provided token with stored token
    // Return false if job.FencingToken != token
    
    // TODO 4: Check if job is in a state that accepts worker reports
    // Only CLAIMED and EXECUTING jobs should accept status updates
    
    // TODO 5: Return validation result
    // true = worker has authority, false = stale or invalid token
    
    return false, nil
}
```

**Language-Specific Implementation Hints:**

- **etcd Client**: Use `go.etcd.io/etcd/client/v3` for robust etcd integration with automatic retries and connection pooling
- **Context Cancellation**: Use `context.WithCancel()` for graceful shutdown coordination across all goroutines
- **Atomic Operations**: Use `sync.RWMutex` for worker registry access and `sync.atomic` for fencing token generation
- **Time Handling**: Always use `time.Now().UTC()` for consistent timestamps across distributed nodes
- **Error Wrapping**: Use `fmt.Errorf("context: %w", err)` for error chain preservation in distributed debugging
- **Structured Logging**: Use `log/slog` or similar for JSON-structured logs with correlation IDs across services

**Milestone Checkpoint:**

After implementing worker coordination, verify the system with these tests:

1. **Leader Election Test**: `go run cmd/coordinator/main.go` on three separate machines - exactly one should claim leadership, others should remain candidates
2. **Worker Registration**: Start workers with different capabilities - they should appear in coordinator logs as registered and healthy
3. **Failure Detection**: Kill a worker process - coordinator should detect failure within 60 seconds and log worker state change
4. **Job Recovery**: Submit jobs, kill workers mid-execution - jobs should be reassigned to healthy workers within recovery interval
5. **Graceful Shutdown**: Send SIGTERM to worker - it should complete current jobs before exiting, coordinator should clean up registration

Expected behavior: Workers register successfully, heartbeats maintain registration, leader election produces exactly one leader, failed workers trigger job recovery within 2 minutes, graceful shutdown completes current work.


## Interactions and Data Flow

> **Milestone(s):** This section synthesizes all three milestones by defining the communication patterns and message flows that connect the cron parser (Milestone 1), priority queue (Milestone 2), and worker coordination (Milestone 3) into a cohesive distributed system.

### Mental Model: Orchestra Performance

Think of the distributed job scheduler's interactions like a symphony orchestra performance. The **coordinator** acts as the conductor, maintaining the tempo and ensuring all musicians (workers) play their parts harmoniously. **Job submission** is like sheet music being distributed to the orchestra - each piece has timing requirements (cron expressions), priority levels (solo vs. background), and specific instructions (job payload). The **execution flow** resembles the actual performance where musicians claim their parts, play them according to the conductor's timing, and signal completion. **Coordination messages** are the conductor's gestures and the musicians' acknowledgments that keep everyone synchronized, detect when someone misses their cue, and recover gracefully from mistakes.

Just as an orchestra must handle musicians arriving late, instruments breaking, or the conductor temporarily stepping away, our distributed scheduler must gracefully manage worker failures, network partitions, and leadership changes while ensuring the "performance" (job execution) continues without missing critical notes.

### Job Submission Flow

The job submission flow represents the journey from external job creation through validation, scheduling, and queue insertion. This flow must handle duplicate detection, priority assignment, and delayed scheduling while maintaining consistency across the distributed system.

#### Submission Process Overview

The job submission process begins when an external client creates a job request and continues through multiple validation and enrichment stages before the job becomes eligible for worker assignment. The process involves three primary actors: the **client** (job submitter), the **scheduler service** (API gateway and validation), and the **priority queue** (storage and ordering).

> **Decision: Synchronous vs Asynchronous Submission API**
> - **Context**: Clients need feedback about job acceptance, but synchronous processing could create bottlenecks under high load
> - **Options Considered**: Fully synchronous processing, fire-and-forget async, async with acknowledgment
> - **Decision**: Synchronous validation and deduplication with asynchronous scheduling
> - **Rationale**: Clients get immediate feedback about validation errors and duplicates, but scheduling decisions happen asynchronously to prevent blocking
> - **Consequences**: Enables immediate error handling while maintaining high throughput; requires careful state management for partially processed jobs

The submission flow follows this sequence of operations:

1. **Client Request Validation**: The scheduler service validates the incoming job request against schema requirements, checking that required fields are present and properly formatted.

2. **Cron Expression Parsing**: The system parses and validates the cron expression using the parser from Milestone 1, calculating the initial `ScheduledAt` timestamp for the job's first execution.

3. **Idempotency Check**: The deduplication system checks if a job with the same `IdempotencyKey` already exists, preventing duplicate submissions from causing redundant work.

4. **Job Enrichment**: The system enriches the job with metadata including creation timestamps, generated unique IDs, and computed priority scores.

5. **Queue Insertion**: The priority queue atomically inserts the job, positioning it correctly based on priority and scheduled execution time.

6. **Response Generation**: The scheduler service returns confirmation to the client with the job ID and next execution time.

#### Submission Message Formats

| Message Type | Direction | Format | Description |
|--------------|-----------|---------|-------------|
| JobSubmissionRequest | Client → Scheduler | JSON over HTTP | Contains job definition, cron expression, and metadata |
| JobSubmissionResponse | Scheduler → Client | JSON over HTTP | Returns job ID, validation status, and next execution time |
| QueueInsertionMessage | Scheduler → Queue | Internal struct | Enriched job data for priority queue insertion |
| DeduplicationQuery | Scheduler → Storage | Redis command | Idempotency key lookup for duplicate detection |

The `JobSubmissionRequest` message contains all information necessary to create and schedule a job:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| Name | string | Yes | Human-readable job identifier |
| CronExpression | string | Yes | Valid cron timing specification |
| Payload | map[string]string | Yes | Job-specific execution parameters |
| Priority | int | No | Priority level (default: 0) |
| IdempotencyKey | string | No | Client-provided deduplication key |
| MaxRetries | int | No | Maximum retry attempts (default: 3) |
| Capabilities | []string | No | Required worker capabilities |

#### Deduplication Strategy Details

The job submission flow implements sophisticated deduplication to prevent clients from accidentally creating duplicate work. The system uses a multi-layered approach combining idempotency keys and content hashing.

> **Decision: Idempotency Key vs Content Hash for Deduplication**
> - **Context**: Need to prevent duplicate jobs while supporting both intentional and accidental resubmission scenarios
> - **Options Considered**: Client-provided keys only, automatic content hashing, hybrid approach
> - **Decision**: Primary idempotency keys with fallback content hashing
> - **Rationale**: Gives clients explicit control while providing automatic protection; content hashing catches unintentional duplicates
> - **Consequences**: More complex deduplication logic but better user experience and data integrity

| Deduplication Method | When Applied | Scope | Duration |
|---------------------|--------------|-------|----------|
| Idempotency Key | Client provides explicit key | Exact key match | Configurable TTL |
| Content Hash | Automatic for all jobs | Identical payload + schedule | 24 hours default |
| Name + Schedule | Fallback protection | Same name and cron pattern | Until job completes |

When deduplication detects an existing job, the system's response depends on the existing job's state:

| Existing Job State | Response Action | HTTP Status | Message to Client |
|-------------------|----------------|-------------|-------------------|
| PENDING or CLAIMED | Return existing job | 200 OK | Job already exists with ID |
| EXECUTING | Return existing job | 200 OK | Job currently executing |
| COMPLETED | Create new job | 201 Created | Previous job completed, created new |
| FAILED | Create new job | 201 Created | Previous job failed, created retry |

#### Error Handling in Submission Flow

The submission flow must gracefully handle various error conditions while providing clear feedback to clients about what went wrong and how to fix it.

| Error Condition | Detection Point | Recovery Action | Client Response |
|----------------|-----------------|-----------------|-----------------|
| Invalid cron expression | Parsing stage | Reject with detailed error | 400 Bad Request with parse details |
| Missing required fields | Validation stage | Reject with field list | 400 Bad Request with missing fields |
| Storage unavailable | Queue insertion | Retry with backoff | 503 Service Unavailable |
| Duplicate job detected | Deduplication check | Return existing job | 200 OK with existing job ID |
| Priority out of range | Validation stage | Clamp to valid range | 200 OK with adjusted priority |

#### Common Pitfalls in Job Submission

⚠️ **Pitfall: Race Condition in Deduplication**
During high-frequency submissions, two identical jobs might pass the deduplication check simultaneously if the check and insertion aren't atomic. This occurs because the deduplication query and queue insertion happen in separate operations, creating a window where duplicate jobs can both appear to be unique.

The fix involves using atomic compare-and-set operations or database transactions to ensure the deduplication check and job insertion happen atomically. In Redis, this means using a Lua script that performs both operations as a single atomic unit.

⚠️ **Pitfall: Timezone Confusion in Scheduled Jobs**
Clients often submit cron expressions assuming local timezone interpretation, but the scheduler stores everything in UTC. This causes jobs to run at unexpected times, especially around daylight saving time transitions.

The solution requires explicit timezone handling in the submission API, either requiring clients to specify timezones explicitly or documenting that all times are treated as UTC. The cron parser should normalize all scheduled times to UTC immediately upon submission.

### Job Execution Flow

The job execution flow orchestrates the complex dance between coordinators and workers to ensure jobs run exactly once at the correct time with proper failure handling. This flow represents the core operational behavior of the distributed scheduler.

![System Architecture Overview](./diagrams/system-architecture.svg)

![Job Execution Sequence](./diagrams/job-execution-sequence.svg)

#### Execution Process Overview

Job execution involves four distinct phases: **job readiness detection**, **worker assignment**, **execution monitoring**, and **completion handling**. Each phase has specific responsibilities and failure modes that must be handled gracefully to maintain system reliability.

The execution process operates continuously across multiple components:

1. **Promotion Phase**: The scheduler periodically scans for delayed jobs that have become eligible for execution, moving them from the delayed queue to the active priority queue.

2. **Assignment Phase**: Available workers poll the coordinator for work, and the coordinator atomically assigns the highest-priority job to the most suitable worker.

3. **Execution Phase**: The assigned worker executes the job while sending periodic heartbeats to maintain its lease on the work.

4. **Reporting Phase**: The worker reports execution results back to the coordinator, which updates job state and triggers retry logic if necessary.

> **Decision: Pull vs Push Job Assignment Model**
> - **Context**: Workers need job assignments, but the coordinator must maintain control over job distribution and prevent duplicate assignments
> - **Options Considered**: Coordinator pushes jobs to workers, workers pull jobs from coordinator, hybrid approach with notifications
> - **Decision**: Worker-initiated pull model with coordinator-managed assignment
> - **Rationale**: Pull model gives workers control over their capacity while preventing coordinator from overwhelming failed workers; atomic assignment prevents duplicates
> - **Consequences**: Requires worker polling but provides better load balancing and failure isolation; adds latency but improves reliability

#### Job State Machine During Execution

Jobs transition through multiple states during the execution flow, with specific rules governing valid transitions and the actions triggered by each state change.

| Current State | Triggering Event | Next State | Actions Taken | Responsible Component |
|---------------|------------------|------------|---------------|----------------------|
| PENDING | Promotion timer fires | PENDING | Move to active queue | Scheduler service |
| PENDING | Worker claims job | CLAIMED | Set WorkerID, ClaimedAt | Priority queue |
| CLAIMED | Worker starts execution | EXECUTING | Set FencingToken | Worker |
| EXECUTING | Worker reports success | COMPLETED | Set CompletedAt, cleanup | Coordinator |
| EXECUTING | Worker reports failure | FAILED | Increment RetryCount | Coordinator |
| FAILED | Retry available | PENDING | Reset state, new schedule | Coordinator |
| FAILED | Max retries exceeded | FAILED | Final cleanup | Coordinator |
| CLAIMED/EXECUTING | Worker heartbeat timeout | PENDING | Clear assignment | Coordinator |

#### Worker Job Claiming Process

The job claiming process represents the critical handoff from the priority queue to an available worker. This process must be atomic to prevent duplicate assignments while being efficient enough to support high-throughput job processing.

The claiming process follows these detailed steps:

1. **Worker Availability Check**: The worker evaluates its current capacity, checking that `CurrentJobs < Capacity` and that it possesses any required capabilities for available jobs.

2. **Claim Request Submission**: The worker sends a claim request to the coordinator containing its worker ID, current load information, and capability list.

3. **Coordinator Job Selection**: The coordinator queries the priority queue for the highest-priority job that matches the worker's capabilities and hasn't exceeded its retry limits.

4. **Atomic Assignment**: The coordinator uses a compare-and-swap operation to atomically assign the job to the worker, setting the `WorkerID` and generating a unique `FencingToken`.

5. **Lease Establishment**: The system records the claim time and establishes a lease timeout, after which the job becomes eligible for reassignment if no progress is reported.

6. **Response Transmission**: The coordinator returns the claimed job to the worker along with the fencing token that the worker must include in all subsequent operations on this job.

#### Fencing Token Mechanism

The fencing token system prevents race conditions and ensures that only the legitimately assigned worker can report results for a job, even in the presence of network delays and worker failures.

> **Decision: Fencing Token Implementation Strategy**
> - **Context**: Workers might report results after their lease has expired and the job has been reassigned to another worker
> - **Options Considered**: Monotonic sequence numbers, timestamp-based tokens, UUID-based tokens
> - **Decision**: Monotonic sequence numbers with coordinator-managed generation
> - **Rationale**: Sequence numbers provide clear ordering for conflict resolution; coordinator generation ensures uniqueness and prevents conflicts
> - **Consequences**: Requires coordinator state for token generation but provides strongest consistency guarantees

| Token Component | Purpose | Generation Strategy | Validation Method |
|----------------|---------|-------------------|-------------------|
| Job ID | Identifies specific job | Client or system generated | Database lookup |
| Worker ID | Identifies assigned worker | Worker registration | Active worker registry |
| Sequence Number | Prevents stale reports | Monotonic increment | Compare with stored value |
| Assignment Time | Establishes lease validity | Coordinator timestamp | Compare with current time |

The fencing token validation process works as follows:

1. **Token Extraction**: The coordinator extracts the fencing token from the worker's completion report
2. **Job State Verification**: The system verifies that the job exists and is in a reportable state (`CLAIMED` or `EXECUTING`)
3. **Worker Authority Check**: The system confirms that the reporting worker matches the assigned worker ID
4. **Sequence Validation**: The system verifies that the sequence number matches the most recent assignment token
5. **Lease Validity Check**: The system confirms that the assignment hasn't expired based on the heartbeat timeout

#### Execution Monitoring and Heartbeats

During job execution, the worker must maintain regular communication with the coordinator to prove it's making progress and hasn't failed. This monitoring system enables rapid detection of worker failures and prevents jobs from being lost.

| Heartbeat Component | Purpose | Frequency | Timeout Action |
|--------------------|---------|-----------|----------------|
| Worker Liveness | Proves worker is responsive | Every 30 seconds | Mark worker UNAVAILABLE |
| Job Progress | Shows execution is proceeding | Every 60 seconds | Reassign job to new worker |
| Capacity Update | Reports current job load | With each heartbeat | Update job assignment eligibility |
| Capability Status | Reports available features | On change or every 5 minutes | Update job matching criteria |

The heartbeat message format includes comprehensive worker state information:

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| WorkerID | string | Unique worker identifier | Yes |
| Timestamp | time.Time | When heartbeat was generated | Yes |
| CurrentJobs | int | Number of jobs currently executing | Yes |
| ExecutingJobIDs | []string | List of job IDs in progress | Yes |
| Status | string | Worker operational status | Yes |
| Capabilities | []string | Currently available worker capabilities | No |
| Metadata | map[string]string | Additional worker information | No |

#### Retry Logic and Failure Handling

When jobs fail or workers become unresponsive, the scheduler implements sophisticated retry logic that balances rapid recovery with system stability.

> **Decision: Exponential Backoff vs Fixed Interval Retries**
> - **Context**: Failed jobs need rescheduling, but immediate retries might overwhelm struggling resources
> - **Options Considered**: Immediate retry, fixed delay, exponential backoff, jittered exponential backoff
> - **Decision**: Jittered exponential backoff with maximum delay cap
> - **Rationale**: Exponential backoff reduces load on struggling resources; jitter prevents thundering herd; cap prevents indefinite delays
> - **Consequences**: More complex scheduling logic but better system stability and recovery characteristics

| Retry Attempt | Delay Calculation | Jitter Range | Maximum Delay |
|---------------|------------------|--------------|---------------|
| 1 | 2^1 = 2 seconds | ±50% (1-3 seconds) | 2 minutes |
| 2 | 2^2 = 4 seconds | ±50% (2-6 seconds) | 2 minutes |
| 3 | 2^3 = 8 seconds | ±50% (4-12 seconds) | 2 minutes |
| 4+ | 2^4 = 16 seconds | ±50% (8-24 seconds) | 2 minutes |

The retry system categorizes failures into different types with distinct handling strategies:

| Failure Type | Cause | Retry Strategy | Special Handling |
|-------------|-------|----------------|------------------|
| Transient Network | Connection timeout | Immediate retry | None |
| Resource Exhaustion | Out of memory/disk | Exponential backoff | Route to different worker |
| Logic Error | Invalid job payload | No retry | Send to dead letter queue |
| Worker Crash | Process termination | Exponential backoff | Reassign to healthy worker |
| Coordinator Failure | Leadership change | Wait for leader election | Resume after new leader elected |

#### Common Pitfalls in Job Execution

⚠️ **Pitfall: Duplicate Execution During Worker Recovery**
When a worker fails during job execution, the coordinator reassigns the job to a new worker. However, if the original worker recovers and completes the job, both workers might report completion, leading to duplicate processing or inconsistent state.

This occurs because network partitions can make a worker appear failed when it's actually still processing. The worker continues execution while the coordinator reassigns the job elsewhere. The fencing token mechanism prevents this by ensuring only the worker with the current token can successfully report results.

⚠️ **Pitfall: Heartbeat Timeout During Long-Running Jobs**
Jobs that take longer than the heartbeat timeout will be reassigned to new workers even though the original worker is still making progress. This creates wasteful duplicate work and can cause resource contention.

The solution involves implementing job progress heartbeats separate from worker liveness heartbeats. Workers executing long jobs should send periodic progress updates that extend the job lease without requiring full job completion.

⚠️ **Pitfall: Lost Jobs During Coordinator Failover**
When the coordinator fails during job assignment, jobs might be claimed by workers but not properly recorded in the persistent store. These jobs exist only in coordinator memory and are lost during failover.

Prevention requires ensuring all job state changes are persisted before sending responses to workers. The job assignment must be a transaction that atomically updates both the job state and the worker assignment records.

### Coordination Messages

The coordination message system enables distributed operation by facilitating leader election, worker registration, failure detection, and job assignment across the scheduler cluster. These messages form the nervous system of the distributed scheduler.

#### Message Categories and Purposes

Coordination messages fall into four primary categories, each serving a specific aspect of distributed system management:

| Category | Purpose | Frequency | Reliability Requirements |
|----------|---------|-----------|-------------------------|
| Leadership | Leader election and status | On leadership change | Strong consistency required |
| Registration | Worker join/leave cluster | On worker lifecycle events | At-least-once delivery |
| Heartbeat | Failure detection and health monitoring | Every 30-60 seconds | Best-effort with timeout detection |
| Assignment | Job distribution and result reporting | Per job execution | Exactly-once with fencing |

#### Leader Election Messages

Leader election messages coordinate the selection and maintenance of a single coordinator node responsible for job assignment and worker management. The election process must handle network partitions and prevent split-brain scenarios.

> **Decision: Raft vs ETCD vs Custom Leader Election**
> - **Context**: Need robust leader election that handles network partitions and prevents split-brain scenarios
> - **Options Considered**: Custom implementation, Raft consensus library, ETCD-based election
> - **Decision**: ETCD-based leader election with lease mechanism
> - **Rationale**: ETCD provides battle-tested consensus with lease-based leadership that automatically handles failures; reduces implementation complexity
> - **Consequences**: External dependency on ETCD but significantly more reliable than custom implementation; automatic failover and split-brain prevention

| Message Type | Direction | Trigger | Contents |
|-------------|-----------|---------|----------|
| CandidateAnnouncement | Node → ETCD | Leadership vacancy | Node ID, capabilities, priority |
| LeadershipClaim | ETCD → Nodes | Election completion | Leader node ID, lease expiration |
| LeadershipRenewal | Leader → ETCD | Lease refresh | Current leader ID, health status |
| LeadershipTransfer | Leader → ETCD | Graceful shutdown | New leader suggestion |

The leader election process follows this sequence:

1. **Vacancy Detection**: Nodes detect leadership vacancy through ETCD watch mechanisms or lease expiration notifications
2. **Candidate Registration**: Eligible nodes register as candidates in ETCD with their capabilities and priority scores
3. **Election Resolution**: ETCD's consensus mechanism selects the leader based on predefined criteria (lowest node ID wins in case of ties)
4. **Leadership Establishment**: The elected leader establishes a lease and begins accepting job assignment responsibilities
5. **Ongoing Maintenance**: The leader periodically renews its lease while other nodes monitor for leadership changes

#### Worker Registration Messages

Worker registration messages handle the dynamic addition and removal of worker nodes from the cluster, enabling elastic scaling and graceful shutdown procedures.

| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| WorkerID | string | Unique worker identifier | Must be DNS-safe, 3-63 characters |
| Address | string | Network address for communication | Must be reachable from coordinator |
| Capabilities | []string | List of supported job types | Each capability must match known types |
| Capacity | int | Maximum concurrent jobs | Must be positive integer |
| Metadata | map[string]string | Additional worker information | Optional key-value pairs |
| StartedAt | time.Time | Worker startup timestamp | Must be recent (within 5 minutes) |

The worker registration process ensures new workers can join the cluster and contribute to job processing:

1. **Initial Registration**: New workers send registration messages to the current leader with their capabilities and capacity
2. **Validation and Acceptance**: The coordinator validates worker information and adds the worker to the active registry
3. **Capability Matching**: The system updates job assignment algorithms to consider the new worker's capabilities
4. **Health Monitoring**: The coordinator begins expecting regular heartbeats from the newly registered worker
5. **Job Assignment Eligibility**: The worker becomes eligible to receive job assignments based on its capacity and capabilities

#### Heartbeat Message Protocol

The heartbeat protocol provides the foundation for failure detection and system health monitoring across the distributed cluster.

| Message Component | Size | Purpose | Update Frequency |
|------------------|------|---------|------------------|
| WorkerID | 16 bytes | Worker identification | Never changes |
| SequenceNumber | 8 bytes | Message ordering | Increments with each heartbeat |
| Timestamp | 8 bytes | Generation time | Current time for each message |
| JobStatus | Variable | Current job information | Updates when jobs start/complete |
| ResourceUsage | 32 bytes | CPU, memory, disk usage | Updated each heartbeat |

The heartbeat message format provides comprehensive worker state information:

```
HeartbeatMessage {
    WorkerID: string
    SequenceNumber: int64
    Timestamp: time.Time
    Status: WorkerState
    CurrentJobs: int
    ExecutingJobs: []JobProgress
    ResourceUsage: ResourceMetrics
    Capabilities: []string
}

JobProgress {
    JobID: string
    StartedAt: time.Time
    Progress: float64  // 0.0 to 1.0
    EstimatedCompletion: time.Time
}

ResourceMetrics {
    CPUPercent: float64
    MemoryPercent: float64
    DiskPercent: float64
    NetworkBytesPerSecond: int64
}
```

#### Job Assignment Messages

Job assignment messages coordinate the handoff of work from the priority queue to available workers, including all necessary context for successful execution.

> **Decision: Message Format for Job Assignment**
> - **Context**: Job assignment must include complete execution context while minimizing network overhead
> - **Options Considered**: Full job serialization, reference-based with lookup, hybrid approach
> - **Decision**: Full job serialization with compressed payload
> - **Rationale**: Eliminates coordination dependencies during execution; worker has all necessary information locally
> - **Consequences**: Larger message size but better fault tolerance and simpler worker implementation

| Assignment Component | Purpose | Size Estimate | Compression Strategy |
|--------------------|---------|---------------|---------------------|
| Job Metadata | Identification and scheduling | 200 bytes | None (small, structured) |
| Execution Payload | Job-specific parameters | Variable | gzip compression |
| Worker Instructions | Execution hints and requirements | 100 bytes | None |
| Fencing Information | Authority and lease details | 64 bytes | None |

The complete job assignment message structure:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| JobID | string | Yes | Unique job identifier |
| FencingToken | string | Yes | Authority token for this assignment |
| WorkerID | string | Yes | Target worker identifier |
| JobDefinition | Job | Yes | Complete job specification |
| ExecutionHints | map[string]string | No | Performance and resource hints |
| LeaseExpiration | time.Time | Yes | When assignment expires without progress |
| RetryContext | RetryInfo | No | Previous attempt information |

The job assignment process ensures reliable work distribution:

1. **Assignment Preparation**: Coordinator selects highest-priority job and most suitable available worker
2. **Token Generation**: System generates unique fencing token linking job, worker, and assignment time
3. **Atomic Assignment**: Database transaction atomically assigns job and records assignment details
4. **Message Transmission**: Complete job assignment message sent to target worker
5. **Acknowledgment Handling**: Worker confirms receipt and begins execution preparation
6. **Lease Monitoring**: Coordinator tracks assignment lease and prepares for timeout handling

#### Message Reliability and Ordering

The coordination message system must handle network failures, message loss, and out-of-order delivery while maintaining system correctness.

| Message Type | Delivery Guarantee | Ordering Requirement | Timeout Handling |
|-------------|-------------------|---------------------|------------------|
| Leadership | Exactly-once via ETCD | Total ordering required | ETCD manages timeouts |
| Registration | At-least-once | No ordering requirement | Retry with exponential backoff |
| Heartbeat | Best-effort | Sequence number ordering | Timeout triggers failure detection |
| Assignment | Exactly-once | Per-job ordering only | Reassignment after timeout |

#### Common Pitfalls in Coordination Messages

⚠️ **Pitfall: Message Amplification During Network Partitions**
During network partitions, nodes might repeatedly retry coordination messages, creating message storms when connectivity is restored. This can overwhelm the coordination service and delay recovery.

The solution involves implementing jittered exponential backoff for all coordination messages and rate limiting to prevent message storms. Additionally, nodes should use circuit breaker patterns to temporarily stop sending messages when the coordination service appears unavailable.

⚠️ **Pitfall: Stale Leadership Information**
Workers might continue sending messages to a coordinator that has lost leadership, wasting resources and potentially missing important updates from the new leader.

Prevention requires implementing leadership change notifications through ETCD watches, allowing workers to immediately redirect their coordination messages to the current leader. Workers should also include leadership generation numbers in messages to detect stale leadership.

⚠️ **Pitfall: Heartbeat Message Buildup**
If the coordinator becomes temporarily unresponsive, heartbeat messages can accumulate in network buffers or message queues. When the coordinator recovers, processing this backlog can create false failure detections and unnecessary job reassignments.

The fix involves implementing heartbeat message deduplication based on sequence numbers and timestamps. The coordinator should process only the most recent heartbeat from each worker and discard older messages to get an accurate view of current worker state.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Message Transport | HTTP REST with JSON (net/http) | gRPC with Protocol Buffers |
| Message Queuing | Direct HTTP calls with retries | Redis pub/sub or Apache Kafka |
| Coordination Store | Redis with atomic operations | ETCD with watches and leases |
| Serialization | JSON encoding/decoding | Protocol Buffers or MessagePack |
| Load Balancing | Round-robin with health checks | Weighted least-connections |

#### Recommended File Structure

```
internal/coordination/
  coordinator.go           ← main coordinator logic
  coordinator_test.go      ← coordinator unit tests
  messages.go             ← message type definitions
  election.go             ← leader election implementation
  heartbeat.go            ← heartbeat handling logic
  assignment.go           ← job assignment coordination
  
internal/transport/
  http_client.go          ← HTTP client wrapper
  http_server.go          ← HTTP server with routing
  grpc_client.go          ← gRPC client (optional)
  grpc_server.go          ← gRPC server (optional)
  
internal/messaging/
  publisher.go            ← message publishing interface
  subscriber.go           ← message subscription interface
  redis_transport.go      ← Redis-based message transport
  
cmd/coordinator/
  main.go                ← coordinator service entry point
  
cmd/worker/
  main.go                ← worker service entry point
```

#### Core Message Handling Infrastructure

```go
// Message transport abstraction for testing and flexibility
type MessageTransport interface {
    SendMessage(ctx context.Context, target string, message Message) error
    ReceiveMessages(ctx context.Context, handler MessageHandler) error
    Subscribe(ctx context.Context, topic string, handler MessageHandler) error
    Close() error
}

// Base message interface implemented by all coordination messages
type Message interface {
    Type() MessageType
    Target() string
    Payload() []byte
    Validate() error
}

// HTTP-based message transport implementation
type HTTPTransport struct {
    client    *http.Client
    server    *http.Server
    handlers  map[MessageType]MessageHandler
    logger    *log.Logger
}

func NewHTTPTransport(port int) *HTTPTransport {
    return &HTTPTransport{
        client: &http.Client{
            Timeout: 10 * time.Second,
            Transport: &http.Transport{
                MaxIdleConns:       100,
                IdleConnTimeout:    90 * time.Second,
                DisableCompression: false,
            },
        },
        handlers: make(map[MessageType]MessageHandler),
        logger:   log.New(os.Stdout, "[TRANSPORT] ", log.LstdFlags),
    }
}

func (t *HTTPTransport) SendMessage(ctx context.Context, target string, message Message) error {
    payload := bytes.NewBuffer(message.Payload())
    
    req, err := http.NewRequestWithContext(ctx, "POST", target, payload)
    if err != nil {
        return fmt.Errorf("failed to create request: %w", err)
    }
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Message-Type", string(message.Type()))
    
    resp, err := t.client.Do(req)
    if err != nil {
        return fmt.Errorf("failed to send message: %w", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode >= 400 {
        body, _ := io.ReadAll(resp.Body)
        return fmt.Errorf("message rejected: status=%d, body=%s", resp.StatusCode, string(body))
    }
    
    return nil
}

// Message handler function type
type MessageHandler func(ctx context.Context, message Message) error

// Message routing and validation
func (t *HTTPTransport) handleHTTPMessage(w http.ResponseWriter, r *http.Request) {
    msgType := MessageType(r.Header.Get("Message-Type"))
    handler, exists := t.handlers[msgType]
    if !exists {
        http.Error(w, "Unknown message type", http.StatusBadRequest)
        return
    }
    
    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Failed to read body", http.StatusBadRequest)
        return
    }
    
    message, err := DeserializeMessage(msgType, body)
    if err != nil {
        http.Error(w, fmt.Sprintf("Failed to deserialize: %v", err), http.StatusBadRequest)
        return
    }
    
    if err := message.Validate(); err != nil {
        http.Error(w, fmt.Sprintf("Invalid message: %v", err), http.StatusBadRequest)
        return
    }
    
    ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
    defer cancel()
    
    if err := handler(ctx, message); err != nil {
        t.logger.Printf("Handler error: %v", err)
        http.Error(w, "Processing failed", http.StatusInternalServerError)
        return
    }
    
    w.WriteHeader(http.StatusOK)
}
```

#### Job Assignment Message Implementation

```go
// Complete job assignment message structure
type JobAssignmentMessage struct {
    MessageID      string                 `json:"message_id"`
    JobID          string                 `json:"job_id"`
    WorkerID       string                 `json:"worker_id"`
    FencingToken   string                 `json:"fencing_token"`
    Job            *Job                   `json:"job"`
    ExecutionHints map[string]string      `json:"execution_hints,omitempty"`
    LeaseExpiration time.Time             `json:"lease_expiration"`
    RetryContext   *RetryInfo             `json:"retry_context,omitempty"`
    Timestamp      time.Time              `json:"timestamp"`
}

func (m *JobAssignmentMessage) Type() MessageType {
    return MessageTypeJobAssignment
}

func (m *JobAssignmentMessage) Target() string {
    return m.WorkerID  // Message is targeted at specific worker
}

func (m *JobAssignmentMessage) Payload() []byte {
    data, _ := json.Marshal(m)  // Handle error in production
    return data
}

func (m *JobAssignmentMessage) Validate() error {
    if m.JobID == "" {
        return errors.New("job_id is required")
    }
    if m.WorkerID == "" {
        return errors.New("worker_id is required")
    }
    if m.FencingToken == "" {
        return errors.New("fencing_token is required")
    }
    if m.Job == nil {
        return errors.New("job definition is required")
    }
    if m.LeaseExpiration.Before(time.Now()) {
        return errors.New("lease_expiration must be in the future")
    }
    return nil
}

// Job assignment coordination logic
type JobAssigner struct {
    transport    MessageTransport
    queue        PriorityQueue
    workers      WorkerRegistry
    tokenGen     FencingTokenGenerator
    logger       *log.Logger
}

func (a *JobAssigner) AssignJobToWorker(ctx context.Context, workerID string) error {
    // TODO 1: Query priority queue for highest-priority job matching worker capabilities
    // TODO 2: Generate unique fencing token for this assignment
    // TODO 3: Atomically claim job and update state to CLAIMED
    // TODO 4: Create job assignment message with complete job context
    // TODO 5: Send assignment message to target worker
    // TODO 6: Record assignment in persistent storage for failure recovery
    // TODO 7: Start lease expiration timer for automatic reassignment
    // Hint: Use database transactions to ensure atomic job claiming
    // Hint: Include retry context if this job has failed before
    
    return nil
}
```

#### Heartbeat Protocol Implementation

```go
// Comprehensive heartbeat message structure
type HeartbeatMessage struct {
    WorkerID        string             `json:"worker_id"`
    SequenceNumber  int64              `json:"sequence_number"`
    Timestamp       time.Time          `json:"timestamp"`
    Status          WorkerState        `json:"status"`
    CurrentJobs     int                `json:"current_jobs"`
    ExecutingJobs   []JobProgress      `json:"executing_jobs"`
    ResourceUsage   ResourceMetrics    `json:"resource_usage"`
    Capabilities    []string           `json:"capabilities"`
    Metadata        map[string]string  `json:"metadata,omitempty"`
}

type JobProgress struct {
    JobID               string     `json:"job_id"`
    StartedAt           time.Time  `json:"started_at"`
    Progress            float64    `json:"progress"`          // 0.0 to 1.0
    EstimatedCompletion time.Time  `json:"estimated_completion"`
    Status              string     `json:"status"`
}

type ResourceMetrics struct {
    CPUPercent            float64 `json:"cpu_percent"`
    MemoryPercent         float64 `json:"memory_percent"`
    DiskPercent           float64 `json:"disk_percent"`
    NetworkBytesPerSecond int64   `json:"network_bytes_per_second"`
}

func (m *HeartbeatMessage) Type() MessageType {
    return MessageTypeHeartbeat
}

func (m *HeartbeatMessage) Target() string {
    return "coordinator"  // Always sent to current coordinator
}

func (m *HeartbeatMessage) Payload() []byte {
    data, _ := json.Marshal(m)
    return data
}

func (m *HeartbeatMessage) Validate() error {
    if m.WorkerID == "" {
        return errors.New("worker_id is required")
    }
    if m.SequenceNumber <= 0 {
        return errors.New("sequence_number must be positive")
    }
    if time.Since(m.Timestamp) > time.Minute {
        return errors.New("timestamp is too old")
    }
    if m.CurrentJobs < 0 {
        return errors.New("current_jobs cannot be negative")
    }
    return nil
}

// Heartbeat handling on coordinator side
type HeartbeatProcessor struct {
    workers     WorkerRegistry
    jobRecovery JobRecoveryService
    logger      *log.Logger
    mutex       sync.RWMutex
}

func (p *HeartbeatProcessor) ProcessHeartbeat(ctx context.Context, message Message) error {
    heartbeat, ok := message.(*HeartbeatMessage)
    if !ok {
        return errors.New("invalid message type for heartbeat processor")
    }
    
    // TODO 1: Validate heartbeat message format and freshness
    // TODO 2: Look up worker in registry and check sequence number ordering
    // TODO 3: Update worker state with current capacity and job information
    // TODO 4: Reset worker failure detection timer
    // TODO 5: Update job progress information for executing jobs
    // TODO 6: Check for any jobs that need progress updates
    // TODO 7: Respond with any pending assignments or commands for worker
    // Hint: Use atomic operations when updating worker state
    // Hint: Log sequence number gaps as potential message loss
    
    return nil
}
```

#### Message Flow Orchestration

```go
// End-to-end job execution flow coordinator
type ExecutionFlowCoordinator struct {
    transport   MessageTransport
    queue       PriorityQueue
    workers     WorkerRegistry
    storage     Storage
    config      CoordinatorConfig
    logger      *log.Logger
}

func (c *ExecutionFlowCoordinator) StartJobExecutionFlow(ctx context.Context) {
    // TODO 1: Start goroutine for job promotion (delayed -> active queue)
    // TODO 2: Start goroutine for worker heartbeat monitoring
    // TODO 3: Start goroutine for job assignment processing
    // TODO 4: Start goroutine for completion report handling
    // TODO 5: Set up signal handling for graceful shutdown
    // Hint: Use context cancellation to coordinate shutdown
    // Hint: Implement proper error handling and restart logic for each goroutine
}

func (c *ExecutionFlowCoordinator) handleJobCompletion(ctx context.Context, message Message) error {
    completion, ok := message.(*JobCompletionMessage)
    if !ok {
        return errors.New("invalid message type")
    }
    
    // TODO 1: Validate fencing token to ensure worker authority
    // TODO 2: Update job state based on completion result (success/failure)
    // TODO 3: Handle retry logic if job failed and retries remain
    // TODO 4: Clean up worker assignment and update capacity
    // TODO 5: Trigger next scheduled execution for recurring jobs
    // TODO 6: Send completion notification to job submission client
    // Hint: Use database transactions for state updates
    // Hint: Implement dead letter queue for jobs that exhaust retries
    
    return nil
}
```

#### Milestone Checkpoints

**Checkpoint 1: Basic Message Transport**
- Run: `go test ./internal/transport/... -v`
- Expected: All message serialization and HTTP transport tests pass
- Manual verification: Start coordinator and worker, send test heartbeat message
- Success indicator: Worker successfully registers and sends first heartbeat

**Checkpoint 2: Job Assignment Flow**
- Run: `go test ./internal/coordination/... -v -run TestJobAssignment`
- Expected: Job assignment messages are properly formatted and delivered
- Manual verification: Submit test job, verify worker receives assignment message
- Success indicator: Job transitions from PENDING to CLAIMED state

**Checkpoint 3: End-to-End Execution**
- Run: `go run cmd/coordinator/main.go & go run cmd/worker/main.go`
- Expected: Complete job submission through execution and completion reporting
- Manual verification: Submit job via API, monitor logs for execution flow
- Success indicator: Job completes successfully and reports back to coordinator


## Error Handling and Edge Cases

> **Milestone(s):** This section covers error handling patterns that apply across all three milestones - cron parsing failures (Milestone 1), queue operation failures (Milestone 2), and worker coordination failures (Milestone 3). The distributed nature of the system requires sophisticated failure detection and recovery mechanisms to maintain consistency and availability.

The distributed job scheduler operates in an environment where failures are inevitable rather than exceptional. Like a city's emergency response system that must function despite individual component failures, our scheduler requires comprehensive error handling that prevents cascading failures and ensures graceful degradation. This section explores the failure modes inherent in distributed systems and establishes recovery strategies that maintain operational consistency while maximizing availability.

### Mental Model: Emergency Response Coordination

Think of error handling in a distributed job scheduler as coordinating emergency services across a city. Just as fire departments, police, and hospitals must maintain service during equipment failures, network outages, and staff shortages, our scheduler must continue operating when workers crash, networks partition, or coordination services become unavailable. The key insight is that **failure is not binary** - systems fail partially, intermittently, and in complex combinations that require nuanced responses rather than simple retry logic.

Emergency services use several principles that apply directly to our distributed scheduler: **redundancy** (multiple fire stations), **failure detection** (911 dispatch monitoring), **resource reallocation** (sending backup units when primary responds are unavailable), and **graceful degradation** (reduced service levels rather than complete shutdown). Our error handling strategy implements these same patterns through worker redundancy, heartbeat monitoring, job reassignment, and priority-based service reduction.

### Failure Modes

Distributed systems exhibit failure patterns that rarely occur in single-node applications. Understanding these failure modes enables us to design detection and recovery mechanisms that maintain system integrity despite component failures.

> **Decision: Failure Classification Taxonomy**
> - **Context**: Distributed job scheduler experiences various failure types requiring different detection and recovery strategies
> - **Options Considered**: Binary fail/success model, Component-level categorization, Impact-based classification
> - **Decision**: Multi-dimensional failure taxonomy based on scope, duration, and detectability
> - **Rationale**: Enables targeted recovery strategies and prevents over-engineering simple failure cases
> - **Consequences**: More complex error handling logic but much better system resilience and recovery times

#### Network Partition Failures

Network partitions represent the most challenging failure mode because they create **split-brain scenarios** where different parts of the system have inconsistent views of reality. Unlike clean failures where components stop responding entirely, partitions create zones of connectivity where nodes within a zone can communicate but cannot reach nodes in other zones.

| Partition Scenario | Symptoms | Impact | Detection Method |
|-------------------|----------|--------|------------------|
| Coordinator-Worker Partition | Workers cannot send heartbeats; coordinator sees mass worker failure | Job assignment stops; workers continue executing claimed jobs | Heartbeat timeout; workers cannot reach coordination service |
| Inter-Worker Partition | Workers isolated from each other but connected to coordinator | Load balancing becomes ineffective; some workers overloaded | Job claim patterns show uneven distribution |
| Coordination Service Partition | Cannot access etcd/Redis cluster | All coordination stops; system freezes | Connection timeouts to backing services |
| Client-Scheduler Partition | Job submission requests fail | New jobs cannot be submitted | HTTP/gRPC connection failures |

The **Byzantine nature** of network partitions means that different components may observe different failure patterns simultaneously. A coordinator might lose connectivity to half its workers while maintaining connectivity to the coordination service, creating a scenario where the coordinator attempts to reassign jobs from healthy workers it cannot communicate with.

**Partition detection** relies on timeout-based mechanisms combined with **fencing tokens** to prevent stale operations. When a coordinator loses communication with a worker, it cannot immediately determine whether the worker failed or a network partition occurred. The coordinator must wait for a timeout period before declaring the worker failed and reassigning its jobs. However, if the worker is healthy but partitioned, it may complete its current job and attempt to report completion after the coordinator has already reassigned the job to another worker.

#### Worker Crash Failures

Worker failures manifest in several distinct patterns, each requiring different detection and recovery strategies. Unlike network partitions where the worker may still be processing jobs, crash failures represent complete worker unavailability.

| Failure Type | Characteristics | Detection Latency | Recovery Strategy |
|--------------|-----------------|-------------------|-------------------|
| Hard Crash | Process terminates immediately | Next heartbeat interval | Immediate job reassignment |
| Graceful Shutdown | Worker completes current jobs before stopping | Worker sends shutdown signal | Allow completion, then reassign remaining |
| Resource Exhaustion | Worker becomes unresponsive due to memory/CPU limits | Heartbeat timeout or health check failure | Restart worker, reassign jobs |
| Dependency Failure | External service unavailable (database, API) | Job execution failures | Retry with backoff, possibly reassign |

**Crash detection** combines multiple signals to distinguish between different failure types. The primary mechanism is **heartbeat timeout**, where workers must send periodic liveness signals to the coordinator. However, heartbeat timeout alone cannot distinguish between a crashed worker and a temporarily overloaded worker that cannot send heartbeats promptly.

The coordinator implements **graduated failure detection** that escalates response based on failure duration:

1. **Initial timeout (30 seconds)**: Mark worker as potentially unavailable, stop assigning new jobs
2. **Extended timeout (2 minutes)**: Attempt direct health check to worker endpoint
3. **Failure confirmation (5 minutes)**: Declare worker failed, begin job reassignment process
4. **Cleanup timeout (10 minutes)**: Remove worker from active pool, clean up metadata

This graduated approach prevents **flapping**, where temporary network hiccups cause repeated job reassignments that waste resources and potentially violate exactly-once execution guarantees.

#### Coordination Service Outages

The coordination service (etcd or Redis cluster) represents a **single point of failure** for cluster-wide operations despite being internally distributed. While etcd and Redis provide their own fault tolerance, they can still become unavailable from the scheduler's perspective due to network issues, resource exhaustion, or configuration problems.

| Service | Outage Impact | Mitigation Strategy | Recovery Process |
|---------|---------------|-------------------|------------------|
| etcd Leader Election | Cannot elect new coordinator; current coordinator continues | Local coordinator state caching | Retry election with exponential backoff |
| etcd Job Metadata | Cannot update job states | Write-ahead logging to local storage | Replay logs when connectivity restored |
| Redis Priority Queue | Cannot enqueue/dequeue jobs | Local job buffer with overflow to disk | Drain buffer to Redis when available |
| Redis Deduplication | Cannot prevent duplicate submissions | Accept duplicates temporarily | Post-recovery deduplication scan |

**Coordination service failures** require the scheduler to operate in **degraded mode** rather than stopping entirely. The system maintains local state and implements **write-ahead logging** to ensure that state changes during the outage can be replayed when connectivity is restored.

The most critical challenge is **preventing split-brain scenarios** during coordination service outages. If the current coordinator leader loses connectivity to etcd but workers can still reach etcd, a new coordinator may be elected while the original coordinator continues operating. This scenario requires **fencing tokens** that increment with each leader election, ensuring that operations from stale coordinators are rejected.

⚠️ **Pitfall: Coordination Service Timeout Configuration**
Setting coordination service timeouts too aggressively causes false positives where temporary network delays trigger unnecessary failover procedures. However, setting timeouts too conservatively delays failure detection and prolongs service degradation. The timeout values must account for network latency variance, service processing time, and the cost of false positives versus false negatives. A common mistake is using the same timeout values for different operation types - leader election can tolerate longer timeouts than job state updates.

### Retry Strategies

Retry logic in distributed systems requires sophisticated strategies that account for failure types, system load, and consistency requirements. Unlike simple HTTP retries, our job scheduler must implement **differentiated retry strategies** that prevent retry storms while ensuring forward progress.

> **Decision: Retry Strategy Framework**
> - **Context**: Different failure types require different retry approaches to balance reliability with system stability
> - **Options Considered**: Universal exponential backoff, Per-operation retry logic, Adaptive retry with feedback
> - **Decision**: Hierarchical retry framework with operation-specific strategies and system-wide backpressure
> - **Rationale**: Provides fine-grained control while preventing retry amplification and cascading failures
> - **Consequences**: More complex implementation but much better system stability under load

#### Exponential Backoff Implementation

**Exponential backoff** forms the foundation of our retry strategy but requires careful tuning to prevent **retry amplification** where multiple components retrying simultaneously create artificial load spikes that prevent recovery.

| Retry Scenario | Initial Delay | Backoff Factor | Max Delay | Max Attempts | Jitter |
|----------------|---------------|----------------|-----------|--------------|--------|
| Heartbeat Transmission | 1 second | 2.0 | 30 seconds | 5 | ±20% |
| Job State Update | 100ms | 1.5 | 10 seconds | 8 | ±25% |
| Worker Registration | 5 seconds | 2.0 | 5 minutes | 3 | ±30% |
| Coordination Service Connection | 2 seconds | 1.8 | 2 minutes | 10 | ±15% |

The **jitter component** prevents **thundering herd** scenarios where multiple workers retry simultaneously after a shared dependency recovers. Each retry delay includes random variation: `actual_delay = base_delay * (1 + jitter * random(-1, 1))`.

**Backoff calculation** follows the formula: `delay = min(initial_delay * (backoff_factor ^ attempt), max_delay)`. However, our implementation includes **circuit breaker** logic that suspends retries when failure rates exceed thresholds, preventing resources from being consumed by operations that are unlikely to succeed.

The retry framework implements **differentiated strategies** based on operation criticality:

1. **Critical operations** (job completion reports): Aggressive retries with local persistence and manual intervention escalation
2. **Important operations** (heartbeats): Moderate retries with graceful degradation when max attempts exceeded
3. **Best-effort operations** (metrics reporting): Limited retries with silent failure after max attempts
4. **Background operations** (cleanup tasks): Infrequent retries with extended backoff periods

#### Maximum Attempts and Escalation

**Maximum retry attempts** prevent infinite retry loops while ensuring that temporary failures do not cause permanent data loss. However, different operation types require different escalation strategies when max attempts are exceeded.

| Operation Type | Max Attempts | Escalation Strategy | Manual Intervention Required |
|----------------|--------------|-------------------|----------------------------|
| Job Completion Report | 15 | Write to dead letter queue | Yes - job state reconciliation |
| Worker Heartbeat | 5 | Mark worker as failed | No - automatic recovery |
| Job Assignment | 8 | Return job to queue | No - automatic retry |
| Leader Election | 3 | Enter follower mode | Possibly - configuration check |

**Escalation procedures** ensure that operations that cannot succeed through retries alone are handled appropriately rather than being lost. The dead letter queue serves as a **last resort repository** for operations that require manual intervention or extended retry periods.

The system implements **retry budgets** that limit the total number of retries across all operations within a time window. This prevents scenarios where retry storms consume all available resources and prevent successful operations from completing. Each operation type has an allocated retry budget, and when the budget is exhausted, operations fail fast rather than queuing for later retry.

**Exponential backoff state** is maintained per operation type and target to ensure that repeated failures to the same destination use progressively longer delays. However, successful operations reset the backoff state, allowing rapid recovery when services become available.

#### Dead Letter Queue Handling

The **dead letter queue** serves as a repository for operations that cannot be completed through normal retry mechanisms. Unlike simple failure logging, the dead letter queue enables **recovery workflows** that can resolve issues and replay failed operations.

Dead letter queue entries include comprehensive context for later resolution:

| Field | Purpose | Example Value |
|-------|---------|---------------|
| Original Operation | The operation that failed | JobCompletionReport |
| Failure Reason | Why retries were exhausted | "Coordinator unreachable after 15 attempts" |
| Original Timestamp | When operation was first attempted | 2024-01-15T10:30:00Z |
| Context Data | Information needed for replay | Job ID, Worker ID, Execution Result |
| Retry History | Previous attempt timestamps and errors | [attempt1: timeout, attempt2: connection refused] |
| Manual Review Flag | Whether human intervention is required | true |

**Dead letter processing** includes both automated and manual resolution paths:

1. **Automated recovery**: Periodic replay of dead letter entries when target services become available
2. **Batch reconciliation**: Comparing dead letter entries against current system state to identify resolved operations
3. **Manual intervention workflows**: Administrative interfaces for resolving operations that require human judgment
4. **Data consistency checks**: Validation that replaying dead letter operations will not violate system invariants

The dead letter queue implements **aging policies** that automatically archive entries older than configurable thresholds. However, critical operations like job completion reports are never automatically discarded, ensuring that job execution results are not lost even during extended outages.

⚠️ **Pitfall: Retry Strategy Coordination**
Implementing retry logic independently in each component can create **retry amplification** where failures cascade through the system as each component retries failed operations. For example, if the coordination service becomes temporarily unavailable, all workers may simultaneously retry heartbeats, creating a load spike when the service recovers that prevents successful recovery. Retry strategies must be coordinated across components with shared circuit breakers and backpressure mechanisms.

### Consistency Guarantees

Distributed job scheduling must balance **consistency** with **availability** under the constraints of the CAP theorem. Our scheduler provides configurable consistency guarantees that can be tuned based on application requirements and operational priorities.

> **Decision: Consistency Model Selection**
> - **Context**: Job execution requires balancing exactly-once guarantees with system availability during failures
> - **Options Considered**: Exactly-once execution, At-least-once execution, At-most-once execution
> - **Decision**: Configurable consistency with at-least-once as default and exactly-once as optional mode
> - **Rationale**: Most applications can handle duplicate job execution better than missing job execution
> - **Consequences**: Simpler failure handling but requires idempotent job design

#### At-Least-Once vs Exactly-Once Execution

**At-least-once execution** guarantees that every submitted job will be executed at least once, but jobs may be executed multiple times due to failures and retries. **Exactly-once execution** guarantees that every job is executed exactly one time, but achieving this guarantee requires more complex coordination that can impact availability.

| Consistency Model | Guarantee | Implementation Complexity | Failure Recovery Time | Use Cases |
|-------------------|-----------|---------------------------|----------------------|-----------|
| At-least-once | Jobs never lost, may duplicate | Low | Fast (seconds) | Idempotent operations, data processing |
| Exactly-once | Jobs executed precisely once | High | Slow (minutes) | Financial transactions, state mutations |
| At-most-once | Jobs never duplicate, may be lost | Medium | Medium (30 seconds) | Best-effort notifications, logging |

**At-least-once implementation** uses **optimistic execution** where workers claim jobs and begin execution immediately, reporting completion when finished. If a worker fails during execution, the job is reassigned to another worker after a timeout period. This approach prioritizes availability and fast recovery at the cost of potential duplicate execution.

**Exactly-once implementation** requires **two-phase commit** protocols where workers must confirm job completion before the job is marked as completed system-wide. This involves:

1. **Execution phase**: Worker claims job and executes it locally
2. **Preparation phase**: Worker reports execution completion but job remains in "completing" state
3. **Commit phase**: Coordinator confirms no other worker has claimed the job and marks it as completed
4. **Cleanup phase**: All replicas acknowledge the completion and remove local job state

The exactly-once protocol introduces **blocking periods** where job completion is delayed while the coordinator verifies that no other worker has also executed the job. During network partitions or coordinator failures, jobs may remain in the "completing" state until consistency can be verified.

#### Idempotency Requirements

**Idempotency** enables at-least-once execution patterns by ensuring that executing a job multiple times produces the same result as executing it once. However, achieving idempotency requires careful job design and system support for **idempotency tracking**.

Job idempotency is implemented through several mechanisms:

| Mechanism | Scope | Implementation | Example |
|-----------|-------|----------------|---------|
| Idempotency Key | Per Job | Client-provided unique identifier | UUID generated at job submission |
| Content Hash | Per Execution | Hash of normalized job payload | SHA-256 of sorted job parameters |
| External System State | Per Side Effect | Query before mutation | Check if database record already exists |
| Operation Sequencing | Per Workflow | Monotonic sequence numbers | Process events in timestamp order |

**Idempotency key enforcement** prevents duplicate job submissions when clients retry job creation requests. The scheduler maintains a deduplication cache that maps idempotency keys to job identifiers, ensuring that submitting the same job multiple times results in a single job execution.

**Content-based deduplication** prevents duplicate jobs even when different idempotency keys are used. This addresses scenarios where multiple systems submit identical jobs independently, such as multiple monitoring systems detecting the same failure condition.

The scheduler provides **idempotency validation helpers** that jobs can use to check whether their side effects have already been applied:

1. **State query interfaces**: Check external system state before applying mutations
2. **Operation logging**: Record intended operations before applying them
3. **Result caching**: Store operation results to avoid recomputation
4. **Conditional execution**: Use compare-and-swap operations for state mutations

⚠️ **Pitfall: False Idempotency Assumptions**
Many operations that appear idempotent actually have hidden side effects that violate idempotency. For example, sending an email notification may be considered idempotent because sending the same email twice is acceptable, but if the email system generates unique tracking identifiers or charges per message, duplicate execution creates unintended consequences. Jobs must be carefully designed to handle duplicate execution at the system level, not just at the application logic level.

#### Fencing Tokens and Split-Brain Prevention

**Fencing tokens** provide a mechanism to prevent **split-brain scenarios** where multiple coordinators or workers believe they have authority over the same resources. Fencing tokens are monotonically increasing identifiers that accompany all operations, allowing the system to reject stale operations from previous epochs.

| Token Type | Scope | Generation | Validation |
|------------|--------|------------|------------|
| Coordinator Token | Cluster-wide | Incremented at each leader election | Coordination service validates before writes |
| Worker Token | Per Worker | Generated at worker registration | Coordinator validates before job assignment |
| Job Token | Per Job | Generated at job assignment | Worker validates before state updates |
| Operation Token | Per Operation | Generated at operation start | Target validates before applying changes |

**Coordinator fencing** prevents scenarios where a partitioned coordinator continues operating after a new coordinator has been elected. Each coordinator operation includes the current coordinator token, and the coordination service rejects operations with stale tokens. This ensures that only the current leader can modify cluster state.

**Worker fencing** prevents workers from reporting results for jobs that have been reassigned to other workers. When a coordinator reassigns a job due to worker failure, it generates a new job token. The original worker's completion report includes the old token and is rejected, preventing duplicate completion reporting.

**Fencing token implementation** requires atomic operations in the coordination service:

1. **Token generation**: Atomic increment operation that returns new token value
2. **Token validation**: Compare-and-swap operation that succeeds only if token is current
3. **Token refresh**: Periodic token update to maintain operation authority
4. **Token revocation**: Explicit invalidation of tokens during failure scenarios

The fencing protocol ensures **monotonic consistency** where the system never moves backward in time or accepts operations from previous epochs. However, fencing introduces **operational overhead** as every distributed operation must include token validation.

Fencing tokens are **persisted durably** in the coordination service to survive coordinator restarts and network partitions. During recovery scenarios, the new coordinator queries the coordination service to obtain the current token value and increments it to establish new epoch authority.

> **Critical Design Insight**: Fencing tokens represent a fundamental trade-off between consistency and performance. Every operation becomes more expensive due to token validation, but this cost is essential for preventing data corruption during failure scenarios. Systems that skip fencing to improve performance inevitably encounter split-brain scenarios that require complex manual recovery procedures.

### Implementation Guidance

The error handling and retry mechanisms require careful integration with Go's context cancellation and timeout systems to provide responsive failure detection without resource leaks.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Retry Framework | Manual exponential backoff with time.Sleep | github.com/cenkalti/backoff library |
| Circuit Breaker | Counter-based with mutex | github.com/sony/gobreaker |
| Timeout Management | context.WithTimeout per operation | Hierarchical timeout trees |
| Error Classification | Custom error types with Is/As | github.com/pkg/errors wrapping |
| Metrics Collection | Prometheus counters/histograms | OpenTelemetry with distributed tracing |

#### Recommended File Structure

```
internal/
  errors/
    retry.go              ← exponential backoff implementation
    circuit_breaker.go    ← circuit breaker pattern
    dead_letter.go        ← dead letter queue management
    errors.go             ← custom error types
  fencing/
    tokens.go             ← fencing token generation/validation
    coordinator.go        ← coordinator fencing logic
    worker.go            ← worker fencing implementation
  coordination/
    failure_detector.go   ← heartbeat and failure detection
    recovery.go          ← job recovery and reassignment
    split_brain.go       ← split-brain prevention
```

#### Infrastructure Starter Code

```go
// Package errors provides retry and error handling infrastructure
package errors

import (
    "context"
    "fmt"
    "math"
    "math/rand"
    "time"
)

// RetryConfig defines retry behavior for different operation types
type RetryConfig struct {
    InitialDelay   time.Duration
    BackoffFactor  float64
    MaxDelay       time.Duration
    MaxAttempts    int
    JitterPercent  float64
}

// DefaultRetryConfigs provides sensible retry configurations
var DefaultRetryConfigs = map[string]RetryConfig{
    "heartbeat": {
        InitialDelay:  time.Second,
        BackoffFactor: 2.0,
        MaxDelay:      30 * time.Second,
        MaxAttempts:   5,
        JitterPercent: 0.2,
    },
    "job_state": {
        InitialDelay:  100 * time.Millisecond,
        BackoffFactor: 1.5,
        MaxDelay:      10 * time.Second,
        MaxAttempts:   8,
        JitterPercent: 0.25,
    },
}

// RetryableError indicates an operation can be retried
type RetryableError struct {
    Err       error
    Temporary bool
    Backoff   time.Duration
}

func (e *RetryableError) Error() string {
    return fmt.Sprintf("retryable error (backoff=%v): %v", e.Backoff, e.Err)
}

// IsRetryable returns true if error should be retried
func IsRetryable(err error) bool {
    var retryableErr *RetryableError
    return errors.As(err, &retryableErr)
}

// CircuitBreaker prevents cascade failures during sustained error conditions
type CircuitBreaker struct {
    mu          sync.Mutex
    state       CircuitState
    failures    int
    lastFailure time.Time
    config      CircuitConfig
}

type CircuitState int

const (
    CircuitClosed CircuitState = iota
    CircuitOpen
    CircuitHalfOpen
)

type CircuitConfig struct {
    FailureThreshold int
    RecoveryTimeout  time.Duration
    TestRequests     int
}

// Execute runs operation with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, operation func() error) error {
    // TODO 1: Check circuit state and return fast failure if open
    // TODO 2: Execute operation and record success/failure
    // TODO 3: Update circuit state based on recent failure patterns
    // TODO 4: Return appropriate error with circuit state information
    return nil
}

// DeadLetterQueue stores operations that failed all retry attempts
type DeadLetterQueue struct {
    storage    Storage
    maxAge     time.Duration
    maxEntries int
}

type DeadLetterEntry struct {
    ID              string
    OriginalOp      string
    FailureReason   string
    Context         map[string]string
    RetryHistory    []RetryAttempt
    CreatedAt       time.Time
    RequiresManual  bool
}

type RetryAttempt struct {
    Timestamp time.Time
    Error     string
    Duration  time.Duration
}

// Add stores failed operation for later resolution
func (dlq *DeadLetterQueue) Add(ctx context.Context, entry *DeadLetterEntry) error {
    // TODO 1: Validate entry has required fields
    // TODO 2: Store entry in persistent storage
    // TODO 3: Check if cleanup is needed for old entries
    // TODO 4: Emit metrics for dead letter queue depth
    return nil
}
```

#### Core Logic Skeleton Code

```go
// RetryWithBackoff executes operation with exponential backoff retry logic
func RetryWithBackoff(ctx context.Context, config RetryConfig, operation func() error) error {
    // TODO 1: Initialize attempt counter and current delay
    // TODO 2: Loop until max attempts reached or context cancelled
    // TODO 3: Execute operation and check if error is retryable
    // TODO 4: Calculate next delay with jitter and backoff factor
    // TODO 5: Sleep for calculated delay or return on context cancellation
    // TODO 6: Return last error if all attempts exhausted
    return nil
}

// ValidateFencingToken checks if operation token is current and valid
func (f *FencingManager) ValidateFencingToken(ctx context.Context, tokenType string, token int64) error {
    // TODO 1: Query coordination service for current token value
    // TODO 2: Compare provided token with current token
    // TODO 3: Return specific error if token is stale
    // TODO 4: Update token validation metrics
    return nil
}

// DetectWorkerFailure monitors heartbeats and detects failed workers
func (fd *FailureDetector) DetectWorkerFailure(ctx context.Context, workerID string) error {
    // TODO 1: Check time since last heartbeat for worker
    // TODO 2: Compare against graduated timeout thresholds
    // TODO 3: Attempt direct health check if initial timeout exceeded
    // TODO 4: Mark worker as failed if all checks fail
    // TODO 5: Trigger job recovery process for failed worker
    return nil
}

// RecoverJobsFromFailedWorker reassigns jobs when worker becomes unavailable
func (r *JobRecovery) RecoverJobsFromFailedWorker(ctx context.Context, workerID string, reason string) error {
    // TODO 1: Query all jobs currently assigned to failed worker
    // TODO 2: Determine which jobs were executing vs pending
    // TODO 3: Reset job states to allow reassignment
    // TODO 4: Add jobs back to priority queue with appropriate priority
    // TODO 5: Log recovery actions and update metrics
    return nil
}
```

#### Language-Specific Hints

- Use `context.WithTimeout()` for operation-level timeouts and `context.WithCancel()` for graceful shutdown
- Implement custom error types with `errors.Is()` and `errors.As()` support for error classification
- Use `sync.Mutex` for circuit breaker state but consider `sync.RWMutex` for high-read scenarios
- Leverage `time.NewTimer()` for retry delays to avoid blocking goroutines during backoff periods
- Use `atomic` package for counters and flags that are accessed from multiple goroutines
- Implement graceful shutdown with `sync.WaitGroup` to ensure in-flight operations complete

#### Milestone Checkpoints

**After implementing retry framework:**
- Run `go test ./internal/errors/...` - all retry tests should pass
- Test circuit breaker behavior: rapid failures should open circuit, recovery should close it
- Verify jitter prevents thundering herd: multiple concurrent retries should have different delays

**After implementing fencing tokens:**
- Start coordinator, verify token increments with each leadership change
- Simulate split-brain: old coordinator operations should be rejected with stale token errors
- Test job assignment fencing: worker cannot complete job after reassignment

**After implementing failure detection:**
- Stop worker heartbeats, verify coordinator detects failure within expected timeout
- Test graduated failure detection: temporary slowness should not trigger immediate failure
- Verify job recovery: jobs from failed worker should be reassigned to healthy workers

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Jobs executed multiple times | Worker failure during execution, job reassigned | Check job execution logs for same job ID from multiple workers | Implement idempotency keys, verify fencing tokens |
| Retry storms overloading services | All components retrying simultaneously | Check retry metrics for spikes after service recovery | Add jitter to retry delays, implement circuit breakers |
| Split-brain job assignments | Network partition with multiple active coordinators | Check coordination service logs for multiple leaders | Verify fencing token implementation, check leader election |
| Dead letter queue growing rapidly | Systemic failure with all retry attempts failing | Analyze failure patterns in dead letter entries | Fix underlying service issues, tune retry parameters |
| Workers marked failed but still healthy | Heartbeat timeouts too aggressive | Compare heartbeat intervals with timeout thresholds | Increase timeout values, check network latency patterns |

The error handling implementation should focus on **fail-fast for unrecoverable errors** and **intelligent retry for transient failures**. The key is distinguishing between these categories correctly and implementing backpressure mechanisms that prevent retry storms from overwhelming recovering services.


## Testing Strategy

> **Milestone(s):** This section covers testing approaches that verify the correct implementation across all three milestones - unit tests for isolated component validation (cron parsing, queue operations, coordination logic), integration tests for multi-component scenarios, and milestone-specific verification checkpoints that ensure proper functionality at each development stage.

Building a comprehensive testing strategy for a distributed job scheduler is like designing quality assurance for a city's emergency response system. Just as emergency services must be tested at multiple levels - individual responder training, department coordination drills, and full-scale disaster simulations - our distributed scheduler requires unit tests for component isolation, integration tests for system-wide behavior, and milestone checkpoints for development validation. The complexity arises from testing distributed systems where timing, concurrency, and failure modes create non-deterministic behavior that traditional testing approaches struggle to handle.

The testing strategy operates at three distinct levels, each addressing different aspects of system correctness. Unit testing focuses on algorithmic correctness and component isolation, ensuring individual pieces work correctly in controlled environments. Integration testing validates system-wide behavior under realistic conditions including network partitions, worker failures, and concurrent job execution. Milestone checkpoints provide development feedback loops that verify expected functionality at each implementation stage, preventing architectural drift and catching integration issues early.

### Unit Testing

Think of unit testing for a distributed scheduler like testing individual instruments in an orchestra before the full performance. Each component must demonstrate perfect execution in isolation before we can trust it to perform correctly when coordinated with other components under the unpredictable conditions of distributed operation.

Unit testing in our distributed job scheduler focuses on three primary domains: **cron expression parsing and scheduling logic**, **priority queue operations and deduplication mechanisms**, and **coordination algorithms including leader election and heartbeat processing**. Each domain presents unique challenges that require specialized testing approaches to ensure correctness under all edge cases.

#### Cron Expression Parsing Tests

The cron expression parser represents pure algorithmic logic that transforms textual time patterns into structured execution schedules. Unit testing this component requires comprehensive coverage of cron syntax variations, timezone handling edge cases, and calendar arithmetic correctness across different temporal boundaries.

**Cron Field Parsing Tests** validate the `parseField()` method's ability to correctly interpret individual cron field syntax including wildcards, ranges, step values, and shorthand aliases. These tests use table-driven approaches to verify parsing accuracy across the full spectrum of valid and invalid inputs.

| Test Category | Input Examples | Expected Behavior | Edge Cases |
|---------------|----------------|-------------------|-------------|
| Wildcard Parsing | `*`, `*/5`, `*/15` | Expands to full range with step intervals | Step values that don't divide evenly |
| Range Parsing | `1-5`, `10-20`, `MON-FRI` | Expands to inclusive integer sequences | Cross-boundary ranges like `23-1` |
| List Parsing | `1,3,5`, `MON,WED,FRI` | Expands to explicit value lists | Duplicate values and unsorted lists |
| Step Parsing | `*/2`, `1-10/3`, `MON-FRI/2` | Applies step intervals to base ranges | Step larger than range span |
| Alias Parsing | `@daily`, `@hourly`, `@weekly` | Converts to standard five-field format | Invalid aliases and malformed syntax |

**Next Execution Time Tests** verify the `NextExecutionTime()` method's calendar arithmetic across temporal boundaries including month transitions, leap years, and daylight saving time adjustments. These tests require comprehensive coverage of edge cases that occur at calendar boundaries.

| Boundary Type | Test Scenarios | Validation Points | Common Failures |
|---------------|----------------|-------------------|-----------------|
| Month Boundaries | Jobs scheduled for day 31 in February | Correctly skips to next valid month | Infinite loops on impossible dates |
| Leap Year Handling | February 29 scheduling in non-leap years | Proper year advancement logic | Incorrect leap year detection |
| DST Transitions | Jobs scheduled during spring-forward/fall-back | Maintains consistent UTC scheduling | Duplicate or skipped executions |
| Year Boundaries | December to January transitions | Correct year increment behavior | Off-by-one errors in year calculation |
| Week Boundaries | Day-of-week constraints across month ends | Proper week calculation logic | Incorrect week number arithmetic |

**Timezone Conversion Tests** validate the timezone normalization logic that ensures consistent UTC storage while supporting local timezone interpretation. These tests verify correct conversion behavior across timezone boundaries and DST transitions.

```
Test Matrix: Timezone Conversion Validation
- Standard Time Zones: EST, PST, GMT, UTC+5
- DST Transitions: Spring forward, Fall back scenarios  
- Edge Cases: Invalid timezones, missing location data
- Boundary Conditions: Midnight crossings, leap seconds
```

#### Priority Queue Operation Tests

Priority queue testing validates the core job ordering and atomic operation logic that ensures correct job distribution under concurrent access patterns. The priority queue must maintain strict ordering guarantees while supporting delayed execution and deduplication features.

**Priority Ordering Tests** verify the `JobHeap` implementation maintains correct priority ordering under all insertion and removal operations. These tests validate both the heap property maintenance and the tie-breaking logic for jobs with identical priorities.

| Test Scenario | Setup | Operation | Expected Result |
|---------------|-------|-----------|-----------------|
| Basic Priority Order | Jobs with priorities 1, 5, 3, 2 | Pop all jobs | Retrieved in order: 1, 2, 3, 5 |
| Identical Priority | Multiple jobs with priority 3 | Pop with FIFO tie-breaking | Oldest job retrieved first |
| Mixed Priorities | Interleaved high/low priority jobs | Random insertions and removals | Always returns highest priority |
| Empty Queue | No jobs present | Pop operation | Returns nil with appropriate error |
| Single Job | Queue with one job | Pop followed by another pop | First succeeds, second returns empty |

**Delayed Execution Tests** validate the visibility timeout pattern implementation that holds jobs until their scheduled execution time. These tests verify that delayed jobs remain invisible to `ClaimJob()` operations until promotion time arrives.

| Timing Scenario | Job Schedule | Current Time | Visibility Status |
|-----------------|--------------|--------------|-------------------|
| Future Scheduled | 2024-01-01 12:00 | 2024-01-01 11:30 | Hidden from claim operations |
| Exact Time | 2024-01-01 12:00 | 2024-01-01 12:00 | Becomes visible immediately |
| Past Due | 2024-01-01 12:00 | 2024-01-01 12:30 | Visible and claimable |
| Promotion Batch | 100 delayed jobs | Time advancement | Batch promotion efficiency |

**Deduplication Logic Tests** verify the `CheckDuplicate()` method correctly identifies duplicate job submissions using both idempotency keys and content hashing. These tests ensure no duplicate jobs enter the execution pipeline while avoiding false positive deduplication.

| Deduplication Type | Primary Key | Secondary Check | Test Validation |
|--------------------|-------------|-----------------|-----------------|
| Idempotency Key | Client-provided unique key | Exact key match | Same key prevents duplicate submission |
| Content Hash | Hash of normalized payload | Deterministic hash comparison | Identical content detected regardless of key |
| Key Collision | Different jobs, same key | Payload comparison | Content differs, both jobs accepted |
| Hash Collision | Different payloads, same hash | Full payload comparison | Rare hash collision handled correctly |

⚠️ **Pitfall: Non-Deterministic Hash Calculation**
A common testing mistake is creating content hashes that vary based on map iteration order or pointer values. The `computeContentHash()` function must produce identical hashes for semantically identical job payloads, requiring field normalization and sorted serialization. Test this by creating the same job payload multiple times and verifying hash consistency.

#### Coordination Logic Tests

Coordination logic testing validates the distributed consensus and failure detection algorithms that enable fault-tolerant job execution across multiple worker nodes. These tests focus on algorithmic correctness rather than network behavior, using mock coordination backends to ensure deterministic test execution.

**Leader Election Tests** verify the consensus algorithm implementation that ensures exactly one coordinator node assumes leadership at any given time. These tests use controlled scenarios to validate election correctness under various failure and recovery patterns.

| Election Scenario | Node Configuration | Expected Outcome | Validation Points |
|-------------------|-------------------|------------------|-------------------|
| Clean Election | 3 healthy nodes, no previous leader | Single winner elected | All nodes agree on leader |
| Leadership Change | Current leader fails, 2 remaining nodes | New leader elected promptly | No split-brain condition |
| Network Partition | 3 nodes split into 2+1 partitions | Majority partition maintains leadership | Minority partition steps down |
| Simultaneous Startup | All nodes start concurrently | Deterministic leader selection | Election completes in bounded time |

**Heartbeat Processing Tests** validate the failure detection logic that monitors worker health and triggers job recovery when workers become unresponsive. These tests verify correct timeout handling and state transitions.

| Heartbeat Pattern | Worker Behavior | Expected Response | Recovery Actions |
|-------------------|------------------|-------------------|------------------|
| Regular Heartbeats | Heartbeat every 10 seconds | Worker marked healthy | No recovery needed |
| Missed Heartbeat | Skip one heartbeat cycle | Worker marked suspect | Grace period activated |
| Extended Silence | No heartbeat for 60 seconds | Worker marked failed | Job recovery initiated |
| Recovery Heartbeat | Heartbeat after failure | Worker re-registered | Previous jobs reassigned |

**Fencing Token Tests** verify the token-based mechanism that prevents stale worker reports from corrupting job state. These tests validate that only the current job assignee can report completion or failure status.

| Token Scenario | Job Assignment Token | Report Token | Result |
|-----------------|---------------------|--------------|--------|
| Valid Token | Token 12345 | Token 12345 | Report accepted |
| Stale Token | Token 12345 | Token 12344 | Report rejected |
| No Token | Token assigned | No token provided | Report rejected |
| Token Mismatch | Token for Job A | Token for Job B | Report rejected |

> **Critical Testing Insight**: Unit tests for coordination logic must focus on algorithmic correctness rather than distributed system behavior. Use dependency injection and mock backends to create deterministic test scenarios that validate the decision-making logic without the complexity of actual network communication or timing dependencies.

### Integration Testing

Integration testing for distributed systems is like conducting full-scale emergency response drills where all departments must coordinate under realistic stress conditions. Unlike unit tests that isolate individual components, integration tests validate system-wide behavior including timing dependencies, network communication, and failure recovery patterns that only emerge when components interact in realistic deployment scenarios.

Integration testing operates across three primary dimensions: **multi-worker job distribution scenarios**, **failure injection and recovery validation**, and **end-to-end job execution flows**. Each dimension addresses different aspects of distributed system correctness, ensuring the scheduler maintains consistency and availability despite the inherent challenges of distributed operation.

#### Multi-Worker Coordination Scenarios

Multi-worker scenarios validate the job distribution and coordination algorithms under realistic deployment conditions where multiple worker nodes compete for jobs while maintaining system consistency. These scenarios test the interaction between job queuing, worker registration, and coordination messaging.

**Load Distribution Tests** verify that jobs are distributed fairly across available workers based on their capacity and current load. These tests validate both the initial assignment algorithm and the rebalancing behavior when worker availability changes.

| Test Scenario | Worker Configuration | Job Load | Expected Distribution |
|---------------|---------------------|----------|----------------------|
| Equal Capacity | 3 workers, capacity 10 each | 15 jobs submitted | 5 jobs per worker |
| Mixed Capacity | Workers with capacity 5, 10, 15 | 30 jobs submitted | Proportional to capacity |
| Dynamic Scaling | Start 2 workers, add 2 more | Continuous job stream | Load rebalances to new workers |
| Worker Removal | 4 workers, remove 2 gracefully | Jobs in progress | Completing jobs finish, new jobs redistribute |

**Concurrent Job Claiming Tests** validate the atomic job assignment mechanism prevents duplicate execution when multiple workers simultaneously attempt to claim available jobs. These tests stress-test the `ClaimJob()` operation under high concurrency.

```
Concurrency Test Pattern:
1. Submit 100 jobs to priority queue
2. Start 10 workers simultaneously  
3. Each worker attempts to claim jobs concurrently
4. Verify exactly 100 jobs are claimed (no duplicates)
5. Verify no jobs are claimed by multiple workers
6. Validate fencing token uniqueness across claims
```

**Capability Matching Tests** verify that jobs requiring specific worker capabilities are assigned only to workers that advertise those capabilities. These tests validate the constraint satisfaction logic in the job assignment algorithm.

| Job Requirements | Available Workers | Assignment Expectation | Validation |
|------------------|-------------------|------------------------|------------|
| `["python", "gpu"]` | Worker A: `["python"]`, Worker B: `["python", "gpu"]` | Assigned to Worker B | Capability constraints satisfied |
| `["database"]` | No workers with database capability | Job remains unassigned | Waits for capable worker |
| `[]` (no requirements) | Any available worker | Assigned to any worker | No capability restrictions |

#### Failure Injection and Recovery

Failure injection testing validates the system's resilience to various failure modes that occur in production distributed environments. These tests systematically introduce failures and verify that the system detects, isolates, and recovers from failures while maintaining data consistency.

**Worker Failure Scenarios** test the system's ability to detect unresponsive workers and reassign their jobs to healthy workers without data loss or duplicate execution.

| Failure Type | Injection Method | Expected Detection | Recovery Validation |
|--------------|------------------|-------------------|-------------------|
| Sudden Worker Death | Kill worker process | Missed heartbeat detection | Jobs reassigned within timeout |
| Network Partition | Block worker network access | Heartbeat timeout | Jobs recovered, worker isolated |
| Slow Worker | Introduce artificial delays | Performance degradation detection | Load rebalanced to faster workers |
| Resource Exhaustion | Consume worker memory/CPU | Resource monitoring alerts | Worker marked unavailable |

**Coordinator Failure Tests** validate the leader election and failover mechanisms that ensure continuous system operation despite coordinator node failures.

```
Coordinator Failover Test Sequence:
1. Start 3-node coordinator cluster with elected leader
2. Submit jobs and verify normal operation
3. Kill current leader node abruptly
4. Verify new leader election completes within SLA
5. Confirm job scheduling continues without interruption
6. Validate no jobs are lost or duplicated during transition
```

**Split-Brain Prevention Tests** verify the coordination system prevents multiple nodes from simultaneously believing they are the leader, which could result in conflicting job assignments and data corruption.

| Network Partition | Partition A | Partition B | Expected Behavior |
|-------------------|-------------|-------------|-------------------|
| 3-node cluster, 2+1 split | 2 nodes | 1 node | Majority partition maintains leadership |
| 5-node cluster, 2+3 split | 2 nodes | 3 nodes | Majority partition (3 nodes) leads |
| 2-node cluster partition | 1 node | 1 node | Both step down, no leader until reunion |

**Data Consistency Tests** validate that concurrent operations and failures don't corrupt the job state or create impossible system states like jobs assigned to multiple workers simultaneously.

| Consistency Scenario | Operation Pattern | Validation |
|----------------------|-------------------|------------|
| Concurrent Job Updates | Multiple workers report on same job | Last valid update wins, fencing prevents conflicts |
| Worker Failure During Execution | Worker dies mid-job execution | Job marked for retry, not marked complete |
| Coordinator Failure During Assignment | Leader dies during job assignment | Assignment either completes fully or rolls back |

> **Integration Testing Philosophy**: Integration tests should focus on emergent behaviors that only appear when components interact under realistic conditions. Unlike unit tests that verify algorithmic correctness, integration tests validate system properties like consistency, availability, and partition tolerance that define distributed system correctness.

#### End-to-End Job Execution Flow

End-to-end testing validates complete job lifecycle flows from initial cron schedule evaluation through final execution completion. These tests ensure all system components integrate correctly to deliver the promised scheduling and execution guarantees.

**Complete Job Lifecycle Tests** trace individual jobs through the entire execution pipeline, validating state transitions and ensuring no jobs are lost or duplicated during normal operation.

```
End-to-End Flow Validation:
1. Cron timer fires for scheduled job
2. Job inserted into priority queue with correct priority
3. Available worker claims job atomically
4. Worker executes job payload successfully  
5. Worker reports completion with valid fencing token
6. Job marked complete and archived
7. Next cron execution calculated and scheduled
```

**Retry and Failure Handling Tests** verify the system correctly handles job execution failures and applies the configured retry policy including exponential backoff and maximum retry limits.

| Job Failure Type | Retry Configuration | Expected Behavior |
|-------------------|-------------------|-------------------|
| Transient Failure | Max retries: 3, exponential backoff | Job retried with increasing delays |
| Permanent Failure | Max retries: 3, all attempts fail | Job moved to dead letter queue |
| Worker Failure | Job in progress when worker dies | Job reassigned to different worker |
| Timeout Failure | Job exceeds execution timeout | Job killed and retried on different worker |

**Scheduling Accuracy Tests** validate that recurring jobs are scheduled and executed according to their cron expressions with acceptable timing precision despite system load and failures.

| Cron Expression | Expected Execution Pattern | Tolerance | Validation Method |
|-----------------|---------------------------|-----------|-------------------|
| `0 */5 * * *` | Every 5 minutes | ±30 seconds | Compare actual vs expected execution times |
| `0 0 * * MON` | Every Monday at midnight | ±2 minutes | Verify weekly execution pattern |
| `0 0 1 * *` | First day of each month | ±5 minutes | Confirm monthly boundary handling |

⚠️ **Pitfall: Integration Test Environment Consistency**
Integration tests often fail intermittently due to timing dependencies and resource contention. Ensure test environments provide consistent resource allocation, use deterministic timing where possible, and implement proper test isolation to prevent tests from interfering with each other. Consider using containerized test environments to ensure reproducible conditions.

### Milestone Checkpoints

Milestone checkpoints provide structured validation points that ensure correct implementation at each development stage. These checkpoints serve as both progress indicators and regression prevention mechanisms, catching integration issues early before they compound into complex debugging scenarios.

Think of milestone checkpoints like progressive flight training evaluations - a pilot must demonstrate mastery of basic instruments before advancing to navigation, then to weather handling, and finally to emergency procedures. Each checkpoint validates not just new functionality but also ensures previous capabilities remain intact under the expanded system complexity.

#### Milestone 1: Cron Expression Parser Validation

The first milestone checkpoint validates the cron expression parsing and next execution time calculation logic that forms the foundation for all scheduling operations.

**Functional Verification Tests** ensure the parser correctly handles the full spectrum of cron expression syntax while providing accurate next execution time calculations.

| Validation Category | Test Commands | Expected Behavior | Success Criteria |
|---------------------|---------------|-------------------|-------------------|
| Basic Parsing | `go test ./internal/cron/...` | All parser unit tests pass | 100% test coverage on core parsing logic |
| Expression Validation | Parse sample expressions | Valid expressions accepted, invalid rejected | Clear error messages for malformed input |
| Next Time Calculation | Calculate execution times | Accurate future timestamps returned | Timing precision within 1-second tolerance |
| Timezone Support | Test across multiple timezones | Correct UTC normalization | DST transitions handled properly |

**Parser Stress Testing** validates performance and correctness under high-volume parsing operations that simulate production workloads.

```
Stress Test Validation:
- Parse 10,000 varied cron expressions
- Measure parsing performance (target: <1ms per expression)
- Verify memory usage remains bounded
- Confirm no memory leaks during batch processing
```

**Edge Case Coverage** ensures the parser handles calendar arithmetic edge cases that commonly cause scheduling failures in production systems.

| Edge Case Category | Specific Tests | Validation Points |
|--------------------|----------------|-------------------|
| Month Boundaries | Feb 31, Apr 31 expressions | Correctly skips to next valid date |
| Leap Year Handling | Feb 29 in non-leap years | Proper year advancement |
| DST Transitions | 2 AM during spring forward | No duplicate or missed executions |
| Year Rollover | Dec 31 to Jan 1 transitions | Correct year increment |

#### Milestone 2: Priority Queue System Validation

The second milestone checkpoint validates the priority queue implementation including delayed execution, deduplication, and atomic job operations.

**Queue Operation Verification** tests the core priority queue functionality under both single-threaded and concurrent access patterns.

| Test Category | Validation Command | Success Criteria | Performance Target |
|---------------|-------------------|------------------|-------------------|
| Priority Ordering | `go test ./internal/queue/priority_test.go` | Jobs dequeued in strict priority order | O(log n) insertion/removal |
| Delayed Jobs | `go test ./internal/queue/delay_test.go` | Scheduled jobs remain invisible until promotion | Sub-second promotion accuracy |
| Deduplication | `go test ./internal/queue/dedup_test.go` | Duplicate submissions prevented | Hash calculation <100μs |
| Concurrent Access | `go test -race ./internal/queue/...` | No race conditions detected | Thread-safe under load |

**Redis Integration Testing** validates the persistent queue backend handles Redis connection failures and ensures data durability across service restarts.

```
Redis Integration Validation:
1. Start Redis and queue service
2. Submit 1000 jobs with mixed priorities
3. Stop queue service (graceful shutdown)
4. Restart queue service 
5. Verify all jobs recovered with correct priorities
6. Test Redis failover scenarios
```

**Deduplication Accuracy Tests** verify the duplicate detection logic correctly identifies duplicate submissions while avoiding false positives that would reject legitimate jobs.

| Deduplication Scenario | Input Jobs | Expected Outcome | Validation |
|------------------------|------------|------------------|------------|
| Identical Payloads | Same job submitted twice | Only one job queued | Content hash match detected |
| Same Idempotency Key | Different payloads, same key | Both jobs rejected after first | Key-based deduplication |
| Hash Collision Simulation | Crafted payloads with same hash | Jobs accepted, full comparison performed | Rare collision handled |

#### Milestone 3: Worker Coordination Validation

The third milestone checkpoint validates the complete distributed coordination system including leader election, worker registration, and job recovery mechanisms.

**Coordination System Testing** verifies the distributed consensus and worker management functionality operates correctly across multiple nodes.

| Coordination Feature | Test Approach | Success Criteria | Fault Tolerance |
|---------------------|---------------|------------------|-----------------|
| Leader Election | Multi-node cluster startup | Single leader elected | Recovery from leader failure |
| Worker Registration | Dynamic worker join/leave | Workers tracked correctly | Graceful shutdown handling |
| Job Assignment | Multi-worker job distribution | Fair load balancing | Failed worker job recovery |
| Heartbeat Monitoring | Worker health tracking | Failure detection within SLA | False positive prevention |

**Multi-Node Integration Tests** validate system behavior across realistic deployment scenarios with multiple coordinator and worker nodes.

```
Multi-Node Test Scenario:
1. Deploy 3-node coordinator cluster
2. Register 5 worker nodes with varying capacity
3. Submit continuous job stream (100 jobs/minute)
4. Introduce random worker failures
5. Verify job completion rate >99%
6. Validate no duplicate job executions
7. Confirm system stability over 1-hour test
```

**Failure Recovery Validation** tests the system's ability to maintain operation and data consistency during various failure scenarios.

| Failure Type | Recovery Test | Validation Metrics |
|--------------|---------------|--------------------|
| Worker Node Failure | Kill worker during job execution | Jobs reassigned within 60 seconds |
| Coordinator Failure | Kill leader node | New leader elected within 30 seconds |
| Network Partition | Isolate subset of nodes | Majority partition continues operation |
| Database Failure | Redis/etcd unavailability | Graceful degradation, recovery on reconnect |

**End-to-End System Validation** confirms the complete system delivers the promised distributed job scheduling functionality with acceptable performance and reliability characteristics.

```
Complete System Checkpoint:
✓ Cron jobs scheduled with <5 second accuracy
✓ Priority ordering maintained under load
✓ Worker failures handled without job loss  
✓ Leader election completes in <30 seconds
✓ System throughput >1000 jobs/hour/worker
✓ Memory usage bounded under continuous operation
✓ No race conditions detected in stress testing
```

> **Milestone Checkpoint Philosophy**: Each checkpoint should validate not just new functionality but also ensure previous milestone capabilities remain intact. This regression prevention approach catches integration issues early and provides confidence for continued development on a solid foundation.

### Implementation Guidance

The testing strategy implementation requires careful orchestration of test environments, mock services, and validation frameworks that can handle the complexity of distributed system testing while maintaining deterministic and reproducible results.

**A. Testing Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Unit Testing | Go's built-in testing package | Testify with assertions and mocking |
| Integration Testing | Docker Compose test environments | Kubernetes test namespaces |
| Mock Services | Hand-written mocks | Gomock generated mocks |
| Time Control | Fixed time.Now() override | Clockwork time manipulation |
| Redis Testing | Miniredis embedded server | Real Redis with test database |
| etcd Testing | Embedded etcd test server | Real etcd cluster in containers |

**B. Testing File Structure:**
```
project-root/
  internal/
    cron/
      parser.go
      parser_test.go           ← Unit tests for cron parsing
      integration_test.go      ← Cron timing integration tests
    queue/
      priority.go
      priority_test.go         ← Unit tests for queue operations
      redis_test.go           ← Redis integration tests
    coordinator/
      coordinator.go
      coordinator_test.go      ← Unit tests for coordination logic
      election_test.go        ← Leader election integration tests
  test/
    integration/
      multi_worker_test.go     ← Multi-node integration scenarios
      failure_injection_test.go ← Failure recovery tests
      e2e_test.go             ← Complete system validation
    fixtures/
      test_jobs.json          ← Sample job definitions
      cron_expressions.txt    ← Test cron patterns
    mocks/
      mock_storage.go         ← Generated storage mocks
      mock_transport.go       ← Generated transport mocks
  scripts/
    test_runner.sh            ← Automated test execution
    setup_test_env.sh         ← Test environment setup
```

**C. Core Testing Infrastructure (Complete Implementation):**

```go
// test/infrastructure/test_harness.go
package infrastructure

import (
    "context"
    "testing"
    "time"
    "github.com/stretchr/testify/require"
    "github.com/alicebob/miniredis/v2"
    "go.etcd.io/etcd/server/v3/embed"
)

// TestHarness provides complete testing infrastructure for distributed scheduler tests
type TestHarness struct {
    t        *testing.T
    redis    *miniredis.Miniredis
    etcd     *embed.Etcd
    cleanup  []func()
    timeCtrl *TimeController
}

// NewTestHarness creates a complete testing environment with Redis, etcd, and time control
func NewTestHarness(t *testing.T) *TestHarness {
    redis, err := miniredis.Run()
    require.NoError(t, err)
    
    etcdConfig := embed.NewConfig()
    etcdConfig.Dir = t.TempDir()
    etcdConfig.LogLevel = "error"
    etcdConfig.Logger = "zap"
    
    etcd, err := embed.StartEtcd(etcdConfig)
    require.NoError(t, err)
    
    harness := &TestHarness{
        t:        t,
        redis:    redis,
        etcd:     etcd,
        timeCtrl: NewTimeController(),
        cleanup:  make([]func(), 0),
    }
    
    harness.cleanup = append(harness.cleanup, func() {
        redis.Close()
        etcd.Close()
    })
    
    return harness
}

// TimeController provides deterministic time control for testing
type TimeController struct {
    currentTime time.Time
    callbacks   []func(time.Time)
}

func NewTimeController() *TimeController {
    return &TimeController{
        currentTime: time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC),
        callbacks:   make([]func(time.Time), 0),
    }
}

func (tc *TimeController) Now() time.Time {
    return tc.currentTime
}

func (tc *TimeController) Advance(duration time.Duration) {
    tc.currentTime = tc.currentTime.Add(duration)
    for _, callback := range tc.callbacks {
        callback(tc.currentTime)
    }
}

// Cleanup releases all test resources
func (h *TestHarness) Cleanup() {
    for i := len(h.cleanup) - 1; i >= 0; i-- {
        h.cleanup[i]()
    }
}
```

**D. Unit Test Skeleton Templates:**

```go
// internal/cron/parser_test.go - Cron Parser Unit Tests
package cron

import (
    "testing"
    "time"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestParseCronExpression(t *testing.T) {
    tests := []struct {
        name        string
        expression  string
        expectValid bool
        expectError string
    }{
        // TODO 1: Add test cases for valid cron expressions
        // TODO 2: Add test cases for invalid field values  
        // TODO 3: Add test cases for malformed syntax
        // TODO 4: Add test cases for timezone expressions
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // TODO 5: Call ParseCronExpression with test input
            // TODO 6: Validate expected success/failure
            // TODO 7: Check error message format if applicable
            // TODO 8: Verify parsed field values for valid expressions
        })
    }
}

func TestNextExecutionTime(t *testing.T) {
    // TODO 1: Create test cron expressions with known patterns
    // TODO 2: Set up controlled time scenarios  
    // TODO 3: Calculate expected next execution times manually
    // TODO 4: Validate NextExecutionTime returns correct results
    // TODO 5: Test edge cases like month boundaries and DST
    // TODO 6: Verify timezone conversion accuracy
}
```

```go
// internal/queue/priority_test.go - Priority Queue Unit Tests  
package queue

func TestJobHeapOrdering(t *testing.T) {
    heap := NewJobHeap()
    
    // TODO 1: Create jobs with different priority values
    // TODO 2: Insert jobs in random order
    // TODO 3: Pop all jobs and verify strict priority ordering
    // TODO 4: Test tie-breaking for identical priorities
    // TODO 5: Validate heap property maintained during operations
}

func TestDelayedJobVisibility(t *testing.T) {
    queue := NewPriorityQueue(testRedisConfig())
    
    // TODO 1: Submit jobs with future ScheduledAt times
    // TODO 2: Attempt ClaimJob before scheduled time
    // TODO 3: Verify jobs remain invisible 
    // TODO 4: Advance time to scheduled execution
    // TODO 5: Verify jobs become claimable after promotion
}

func TestDeduplication(t *testing.T) {
    queue := NewPriorityQueue(testRedisConfig())
    
    // TODO 1: Create jobs with identical idempotency keys
    // TODO 2: Submit duplicate jobs concurrently
    // TODO 3: Verify only one job accepted
    // TODO 4: Test content-based deduplication
    // TODO 5: Validate hash collision handling
}
```

**E. Integration Test Framework:**

```go
// test/integration/multi_worker_test.go
package integration

import (
    "context"
    "testing"
    "time"
    "sync"
    "github.com/stretchr/testify/require"
)

func TestMultiWorkerJobDistribution(t *testing.T) {
    harness := infrastructure.NewTestHarness(t)
    defer harness.Cleanup()
    
    // TODO 1: Start coordinator cluster with test harness
    // TODO 2: Register multiple workers with different capacities
    // TODO 3: Submit batch of jobs with mixed priorities
    // TODO 4: Verify jobs distributed proportionally to capacity
    // TODO 5: Check no jobs claimed by multiple workers
    // TODO 6: Validate all jobs eventually complete
}

func TestWorkerFailureRecovery(t *testing.T) {
    harness := infrastructure.NewTestHarness(t)
    defer harness.Cleanup()
    
    // TODO 1: Start system with multiple workers
    // TODO 2: Assign jobs to workers
    // TODO 3: Simulate worker failure (kill process)
    // TODO 4: Verify failed worker jobs are reassigned
    // TODO 5: Confirm no duplicate execution occurs
    // TODO 6: Check system continues normal operation
}
```

**F. Milestone Checkpoint Validation:**

```bash
#!/bin/bash
# scripts/milestone_checkpoint.sh

# Milestone 1: Cron Parser Validation
echo "=== Milestone 1: Cron Expression Parser ==="
go test -v ./internal/cron/... || exit 1
go test -race ./internal/cron/... || exit 1

# Validate specific parser capabilities
echo "Testing cron expression edge cases..."
go run ./cmd/cron-validator/ < test/fixtures/cron_expressions.txt || exit 1

# Milestone 2: Priority Queue Validation  
echo "=== Milestone 2: Priority Queue System ==="
go test -v ./internal/queue/... || exit 1

# Start Redis for integration testing
docker run -d --name test-redis -p 6379:6379 redis:alpine
sleep 2

go test -v ./test/integration/queue_test.go || exit 1
docker stop test-redis && docker rm test-redis

# Milestone 3: Worker Coordination Validation
echo "=== Milestone 3: Worker Coordination ==="
go test -v ./internal/coordinator/... || exit 1
go test -v ./test/integration/coordination_test.go || exit 1

# End-to-End System Validation
echo "=== Complete System Validation ==="
./scripts/setup_test_env.sh
go test -v -timeout=5m ./test/integration/e2e_test.go || exit 1

echo "All milestone checkpoints passed successfully!"
```

**G. Debugging Integration Test Failures:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Tests pass individually, fail in suite | Resource cleanup issues | Check for leaked connections/processes | Add proper cleanup in teardown |
| Intermittent integration failures | Race conditions or timing issues | Add deterministic timing controls | Use TimeController for predictable timing |
| "Connection refused" errors | Test infrastructure not ready | Services starting too quickly | Add readiness checks before tests |
| Memory leaks in long tests | Resources not released | Profile memory usage | Ensure all contexts cancelled, connections closed |
| etcd election timeouts | Cluster formation issues | Check etcd logs for split-brain | Configure proper cluster membership |

This comprehensive testing strategy ensures reliable validation of the distributed job scheduler across all implementation phases, providing both development feedback and production confidence through systematic verification of system behavior under normal and failure conditions.


## Debugging Guide

> **Milestone(s):** This section covers debugging techniques that apply across all three milestones - diagnosing cron parsing and scheduling issues (Milestone 1), troubleshooting queue operation failures and priority conflicts (Milestone 2), and resolving worker coordination and distributed consensus problems (Milestone 3).

Debugging a distributed job scheduler presents unique challenges that combine the complexity of distributed systems with the time-sensitive nature of scheduled execution. Think of debugging a distributed scheduler like diagnosing problems in a city's emergency response system - you need to understand not just individual component failures, but how those failures cascade through the entire coordination network, potentially causing missed emergencies or duplicate responses across multiple stations.

The debugging process requires systematic observation of three interconnected layers: the scheduling logic that determines when jobs should run, the coordination protocols that distribute work across workers, and the execution monitoring that tracks job completion. Unlike debugging a single-process application where you can step through code linearly, distributed scheduler debugging requires correlating events across multiple nodes, understanding timing relationships, and distinguishing between genuine failures and expected distributed systems behavior like temporary network partitions.

This guide provides structured approaches to identify, diagnose, and resolve the most common categories of distributed scheduler problems. Each troubleshooting workflow follows a hypothesis-driven approach that systematically eliminates potential causes while gathering evidence about the actual failure mode.

### Common Symptoms

Understanding the observable symptoms of distributed scheduler problems enables rapid problem classification and appropriate diagnostic approaches. Each symptom category corresponds to failures in specific system layers, allowing targeted investigation rather than broad system exploration.

**Duplicate Job Execution Symptoms**

The most critical symptom category involves jobs executing multiple times despite exactly-once execution guarantees. This manifests in several observable patterns that indicate different underlying failure modes.

Multiple completion reports for the same job represent the classic duplicate execution symptom. The system logs show the same job ID completing successfully on different workers within a short time window, often with identical or conflicting results. This typically indicates failures in the fencing token validation mechanism or race conditions during job claiming.

Idempotency key violations occur when jobs with identical `IdempotencyKey` values execute multiple times despite deduplication logic. The symptom appears as duplicate database records, double-charged transactions, or redundant external API calls that should have been prevented by the deduplication system. This suggests failures in the `DeduplicationChecker` or inconsistent deduplication state across coordinator nodes.

Worker claim conflicts manifest as multiple workers believing they own the same job simultaneously. The logs show different worker IDs reporting progress or completion for identical job IDs, often accompanied by fencing token validation errors when workers attempt to report completion. This indicates failures in the atomic job claiming mechanism or coordinator split-brain scenarios.

Retry amplification creates a cascade of duplicate executions when failed jobs retry across multiple workers simultaneously. The symptom appears as exponentially increasing execution attempts for jobs that should follow controlled retry policies, often overwhelming downstream systems with duplicate requests. This suggests coordination failures during worker crash scenarios or incorrect retry state management.

| Duplicate Symptom | Observable Evidence | Likely Root Cause | Investigation Priority |
|------------------|-------------------|------------------|---------------------|
| Multiple completion reports | Same job ID completing on different workers | Fencing token validation failure | High - affects data integrity |
| Idempotency violations | Same IdempotencyKey executes multiple times | DeduplicationChecker failure | Critical - bypasses client guarantees |
| Worker claim conflicts | Multiple workers reporting same job progress | Atomic claiming race condition | High - indicates coordination breakdown |
| Retry amplification | Exponential retry attempts across workers | Failed worker job recovery logic | Medium - self-limiting but wasteful |

**Missed Schedule Symptoms**

Missed schedules represent the opposite failure mode where jobs fail to execute when expected, violating timing guarantees and potentially causing business impact through delayed processing.

Cron schedule drift occurs when recurring jobs gradually shift their execution times away from the expected cron schedule. The symptom manifests as jobs that should run every hour at :00 minutes instead executing at :03, then :07, then :12, creating an accumulating delay. This typically indicates problems in next execution time calculation or clock synchronization across coordinator nodes.

Delayed job promotion failures cause jobs with future execution times to remain invisible past their scheduled execution time. The symptom appears as jobs stuck in delayed state while their `ScheduledAt` time has passed, often accompanied by gaps in execution logs during specific time periods. This suggests failures in the `PromoteDelayedJobs` background process.

Priority queue starvation occurs when high-priority jobs prevent lower-priority jobs from executing, even when worker capacity exists. The symptom manifests as continuously growing queues of lower-priority jobs while only high-priority jobs execute, eventually leading to timeout failures for starved jobs. This indicates problems in priority queue balancing or worker assignment algorithms.

Timezone calculation errors cause jobs to execute at incorrect absolute times, particularly during daylight saving time transitions. The symptom appears as jobs executing one hour early or late relative to local business hours, or failing to execute during the "spring forward" hour that doesn't exist. This suggests failures in UTC normalization or timezone conversion logic.

Worker capacity exhaustion prevents job execution when all workers reach their capacity limits simultaneously. The symptom manifests as growing job queues despite healthy workers, often correlating with long-running job execution times or worker resource constraints. This indicates insufficient worker scaling or improper capacity management.

| Missed Schedule Symptom | Observable Evidence | Likely Root Cause | Investigation Priority |
|------------------------|-------------------|------------------|---------------------|
| Cron schedule drift | Gradual execution time delays | Next execution calculation errors | Medium - affects long-term reliability |
| Delayed job promotion failures | Jobs stuck past ScheduledAt time | PromoteDelayedJobs process failure | High - breaks delayed execution |
| Priority queue starvation | Lower priority jobs never execute | Priority balancing algorithm failure | Medium - affects fairness guarantees |
| Timezone calculation errors | Jobs execute at wrong local times | UTC conversion or DST handling bugs | High - affects business SLA compliance |
| Worker capacity exhaustion | Growing queues despite healthy workers | Capacity management or scaling issues | Medium - affects throughput scaling |

**Worker Coordination Failures**

Worker coordination failures disrupt the distributed consensus mechanisms that enable fault-tolerant job distribution, often cascading into both duplicate execution and missed schedule problems.

Leader election split-brain scenarios occur when multiple coordinator nodes simultaneously believe they hold leadership authority. The symptom manifests as conflicting job assignments, duplicate scheduling decisions, and workers receiving contradictory instructions from different coordinator nodes. The logs show multiple nodes reporting leadership status and issuing fencing tokens with overlapping ranges.

Heartbeat timeout cascades happen when network issues cause multiple workers to appear failed simultaneously, triggering mass job reassignment that overwhelms the remaining workers. The symptom appears as sudden spikes in job reassignment activity followed by worker overload, often correlating with network connectivity issues or coordination service degradation.

Worker registration failures prevent new workers from joining the cluster or cause existing workers to lose their registration intermittently. The symptom manifests as workers that appear healthy locally but don't receive job assignments, often accompanied by authentication or connectivity errors when attempting to register with the coordinator.

Job recovery loops occur when failed worker detection triggers job reassignment, but the reassignment process itself fails, creating repeated attempts to recover the same jobs. The symptom appears as continuous job state transitions between `CLAIMED` and `PENDING` without successful execution, often accompanied by worker state thrashing.

Fencing token validation errors prevent workers from completing jobs due to stale or invalid authorization tokens. The symptom manifests as successful job execution followed by completion report rejection, causing jobs to remain in `EXECUTING` state until timeout. This typically indicates coordination service connectivity issues or token generation failures.

| Coordination Failure Symptom | Observable Evidence | Likely Root Cause | Investigation Priority |
|----------------------------|-------------------|------------------|---------------------|
| Leader election split-brain | Multiple coordinators claiming leadership | Consensus algorithm failure or network partition | Critical - affects all system operations |
| Heartbeat timeout cascades | Mass worker failure detection | Network issues or coordination service overload | High - triggers unnecessary job reassignment |
| Worker registration failures | Healthy workers not receiving job assignments | Authentication or connectivity problems | Medium - affects cluster capacity |
| Job recovery loops | Jobs stuck cycling between CLAIMED and PENDING | Job reassignment logic failures | High - creates resource waste and blocks execution |
| Fencing token validation errors | Job completion reports rejected despite success | Stale tokens or coordinator connectivity issues | High - causes jobs to appear failed incorrectly |

### Diagnostic Tools

Effective diagnosis of distributed scheduler problems requires coordinated observation across multiple system components, with particular emphasis on correlating events across time and node boundaries. The diagnostic strategy combines structured logging, quantitative metrics, and distributed tracing to build a comprehensive picture of system behavior during both normal operation and failure scenarios.

**Logging Strategies**

Structured logging provides the foundation for distributed scheduler diagnosis by creating searchable, correlatable records of system events across all components. The logging strategy must balance information richness with performance impact while ensuring consistent data formats across all system boundaries.

Each log entry includes mandatory context fields that enable correlation across distributed components: `component` identifies the logging service (coordinator, worker, queue), `node_id` specifies the physical machine, `correlation_id` links related operations across services, `timestamp` provides nanosecond-precision timing, and `log_level` indicates severity. Additional fields like `job_id`, `worker_id`, and `fencing_token` provide domain-specific context for scheduler operations.

The coordinator logging strategy focuses on decision points and state transitions that affect job distribution and worker coordination. Leadership election events log candidate announcements, vote outcomes, and leadership transitions with detailed reasoning. Job assignment decisions include worker selection criteria, capacity calculations, and assignment success or failure reasons. Worker health monitoring logs heartbeat reception, timeout detection, and failure recovery initiation with precise timing information.

Worker logging emphasizes job execution lifecycle and coordination protocol participation. Job claim attempts log the claiming worker ID, job priority comparison results, and atomic claim success or failure with detailed error information. Job execution logging captures start times, progress checkpoints, completion status, and any execution errors with full stack traces. Heartbeat transmission logs include coordinator connectivity status and any heartbeat rejection reasons.

Queue operation logging tracks job flow through the priority queue system with emphasis on state transitions and timing. Job submission logs include deduplication check results, priority assignment reasoning, and queue insertion confirmation. Job promotion from delayed to active status logs the promotion trigger time and any promotion failures. Priority ordering logs capture job comparison decisions during queue operations.

| Log Level | Coordinator Events | Worker Events | Queue Events | Required Fields |
|-----------|-------------------|---------------|--------------|-----------------|
| ERROR | Leadership loss, worker failure detection | Job execution failures, heartbeat rejections | Queue corruption, deduplication failures | component, node_id, correlation_id, timestamp, error_details |
| WARN | Slow heartbeat responses, job reassignments | Long-running jobs, capacity warnings | Priority queue imbalance, delayed promotion lag | component, node_id, correlation_id, timestamp, warning_reason |
| INFO | Leadership elections, job assignments | Job claims, execution completions | Job submissions, priority promotions | component, node_id, correlation_id, timestamp, operation_result |
| DEBUG | Heartbeat processing, fencing token generation | Job progress updates, worker registrations | Deduplication checks, priority comparisons | component, node_id, correlation_id, timestamp, detailed_state |

**Metrics Collection**

Quantitative metrics provide real-time insight into system health and performance characteristics, enabling both automated alerting and trend analysis for capacity planning. The metrics strategy emphasizes leading indicators that predict problems before they cause observable failures.

Job execution metrics track the core business functionality of the scheduler system. Job submission rate measures incoming work load with breakdown by priority levels and job types. Job completion rate tracks successful execution with timing distributions to identify performance degradation. Job failure rate monitors error conditions with categorization by failure type (timeout, execution error, coordination failure) to identify systemic issues. Job queue depth measures pending work backlog with priority-level breakdown to identify starvation or overload conditions.

Worker coordination metrics monitor the health of the distributed consensus mechanisms. Active worker count tracks cluster capacity with breakdown by worker capabilities and current load. Heartbeat latency measures coordination responsiveness with percentile distributions to identify coordination service degradation. Leader election frequency tracks consensus stability - frequent elections indicate network or leadership failures. Job assignment latency measures the time from job submission to worker claim, indicating coordination efficiency.

Queue operation metrics provide insight into the priority queue performance and correctness. Deduplication hit rate measures how frequently duplicate jobs are submitted, indicating client behavior patterns. Delayed job promotion latency tracks the accuracy of scheduled execution timing. Priority queue operation latency measures the performance of queue insertions and claims under load.

System-level metrics monitor the underlying infrastructure health. Coordination service response times track etcd or Redis performance that underpins consensus operations. Network partition detection counts coordination connectivity failures between nodes. Memory usage tracks queue size growth and potential memory leaks in long-running processes.

| Metric Category | Metric Name | Aggregation | Alert Thresholds | Business Impact |
|-----------------|-------------|-------------|------------------|-----------------|
| Job Execution | job_submission_rate | Per second by priority | > 1000/sec (capacity warning) | Indicates incoming load trends |
| Job Execution | job_completion_latency_p99 | 99th percentile in milliseconds | > 30000ms (30 second warning) | Affects SLA compliance |
| Job Execution | job_failure_rate | Percentage over 5-minute window | > 5% (critical alert) | Direct business impact |
| Worker Coordination | active_worker_count | Current count | < 2 workers (critical alert) | Affects fault tolerance |
| Worker Coordination | heartbeat_latency_p95 | 95th percentile in milliseconds | > 5000ms (coordination degradation) | Predicts coordination failures |
| Worker Coordination | leader_election_frequency | Elections per hour | > 1/hour (stability warning) | Indicates consensus instability |
| Queue Operations | deduplication_hit_rate | Percentage over 1-minute window | > 50% (client behavior warning) | Indicates client retry storms |
| Queue Operations | delayed_promotion_lag | Seconds behind scheduled time | > 60s (timing accuracy warning) | Affects schedule reliability |

**Distributed Tracing Approaches**

Distributed tracing reconstructs the complete execution flow of individual jobs across multiple system components, providing causal relationships between events that are difficult to establish through logs and metrics alone. The tracing strategy focuses on critical execution paths that cross service boundaries.

Job execution traces begin with job submission and follow the complete lifecycle through queue insertion, delayed promotion, worker assignment, execution, and completion reporting. Each trace span represents a logical operation within a single component, while the overall trace shows the end-to-end execution flow with precise timing information and failure points.

The job submission trace span captures the initial request processing including payload validation, deduplication checking, priority assignment, and queue insertion. The trace includes custom attributes for job priority, scheduled execution time, and deduplication results that enable filtering and analysis of specific job patterns.

Queue operation spans track the job through priority queue management including delayed job promotion, priority comparison during claims, and atomic job assignment. The trace attributes include queue depth at operation time, priority comparison results, and worker selection criteria that influenced assignment decisions.

Worker execution spans capture the actual job processing including claim acquisition, execution startup, progress reporting, and completion status. The span attributes include worker capacity information, execution environment details, and any error conditions that occurred during processing.

Coordination protocol spans track the distributed consensus operations that enable worker coordination including heartbeat processing, leader election participation, and job recovery operations. These spans often cross multiple service boundaries as coordination decisions propagate through the cluster.

Trace sampling strategies balance information completeness with system performance impact. All failed operations generate traces regardless of sampling rate to ensure failure diagnosis information is always available. Successful operations use adaptive sampling that increases trace collection during periods of high error rates or performance degradation.

| Trace Type | Span Hierarchy | Key Attributes | Sampling Rate | Retention Period |
|------------|-----------------|----------------|---------------|------------------|
| Job Execution | submission → queue_insert → promotion → assignment → execution → completion | job_id, priority, worker_id, execution_time | 1% normal, 100% on failures | 7 days |
| Worker Coordination | heartbeat → health_check → assignment_eligibility → job_claim | worker_id, capacity, capabilities, claim_result | 5% normal, 100% on coordination failures | 3 days |
| Queue Operations | priority_comparison → deduplication_check → atomic_insert → delayed_promotion | queue_depth, priority_order, dedup_result, promotion_timing | 10% normal, 100% on queue errors | 5 days |
| Coordination Protocol | leader_election → consensus_vote → leadership_transition → authority_validation | node_id, vote_result, leadership_status, authority_scope | 100% always | 14 days |

### Troubleshooting Workflows

Systematic troubleshooting workflows provide structured approaches to diagnose and resolve the most common categories of distributed scheduler failures. Each workflow follows a hypothesis-driven methodology that systematically eliminates potential causes while gathering evidence about the actual failure mode.

**Network Partition Diagnosis**

Network partitions represent one of the most challenging failure modes in distributed scheduler systems, creating scenarios where different parts of the cluster have inconsistent views of system state. The partition diagnosis workflow systematically identifies partition boundaries and determines appropriate recovery strategies.

The initial partition detection phase examines coordination service connectivity patterns across all cluster nodes. Begin by checking etcd or Redis connectivity from each coordinator and worker node, looking for patterns where specific node groups can communicate internally but cannot reach other groups. Review network routing tables and firewall configurations to identify infrastructure changes that might have created connectivity barriers.

Coordinator split-brain investigation focuses on leadership status across the cluster. Query the leadership election state from each coordinator node to identify whether multiple nodes believe they hold leadership authority. Check the fencing token generation logs to see if multiple coordinators are issuing overlapping token ranges, which indicates split-brain scenarios. Review recent leadership election logs for unusual patterns like rapid leadership transitions or vote timeouts.

Worker isolation analysis determines which workers can communicate with which coordinators during the partition. Check worker heartbeat logs to identify patterns where workers successfully send heartbeats to some coordinators but receive timeout errors from others. Review job assignment patterns to see if specific workers are receiving assignments from multiple coordinator nodes simultaneously.

Job state consistency verification examines whether the same jobs have different states on different sides of the partition. Query job state from multiple coordinator nodes and compare results, looking for jobs that appear as `EXECUTING` on one coordinator but `PENDING` on another. Check for duplicate job assignments where the same job ID is claimed by workers on different sides of the partition.

Recovery strategy determination depends on the partition scope and duration. For brief network hiccups lasting less than heartbeat timeout intervals, the system should self-recover as connectivity resumes. For sustained partitions lasting longer than coordination timeouts, manual intervention may be required to resolve state inconsistencies and prevent duplicate job execution during partition recovery.

| Partition Symptom | Investigation Steps | Evidence to Collect | Recovery Action |
|------------------|-------------------|-------------------|-----------------|
| Multiple leaders detected | Check leadership status on all coordinators | Leadership election logs, fencing token ranges | Stop all coordinators, restart with clean election |
| Worker heartbeat split | Verify worker connectivity to each coordinator | Heartbeat success/failure patterns by coordinator | Identify majority partition, isolate minority nodes |
| Duplicate job assignments | Compare job states across coordinators | Job state differences, worker claim conflicts | Cancel assignments from minority partition coordinators |
| Coordination service split | Test etcd/Redis connectivity from each node | Connection success patterns, error messages | Restore network connectivity, restart coordination service |

**Timing Issue Analysis**

Timing-related problems in distributed schedulers often involve subtle interactions between clock synchronization, timezone handling, and distributed coordination that create intermittent failures difficult to reproduce consistently.

Clock skew detection examines time differences across cluster nodes that can cause coordination failures or incorrect scheduling decisions. Begin by comparing system clock values across all coordinator and worker nodes, looking for differences larger than a few seconds. Check NTP synchronization status and identify any nodes that have lost time synchronization or are using different time sources.

Cron expression evaluation verification tests the next execution time calculation logic against known timezone and daylight saving time scenarios. Create test cases with specific cron expressions and verify that next execution times are calculated correctly across DST transitions, leap years, and month boundaries. Pay particular attention to expressions using day-of-week and day-of-month combinations that can create complex interaction patterns.

Delayed job promotion timing analysis examines the accuracy of scheduled job execution by comparing actual execution times with intended schedule times. Review the `PromoteDelayedJobs` execution logs to identify patterns where promotion occurs significantly before or after the intended execution time. Check for systematic delays that might indicate performance problems in the promotion process.

Coordination timeout investigation focuses on distributed operation timing that affects consensus and job assignment. Review heartbeat timeout patterns to identify whether timeouts occur due to network latency, coordinator overload, or genuine worker failures. Examine job claim timeouts that might indicate coordination service performance problems or excessive contention during high-load periods.

Race condition identification requires careful analysis of timing-sensitive operations that might behave differently under varying load or network conditions. Focus on atomic job claiming operations where multiple workers might attempt to claim the same job simultaneously. Review fencing token validation timing to identify scenarios where token expiration occurs between job claim and completion reporting.

| Timing Issue Category | Diagnostic Approach | Tools and Techniques | Resolution Strategy |
|----------------------|-------------------|-------------------|-------------------|
| Clock skew problems | Compare system times across all nodes | chrony/ntpd status, manual time comparison | Implement NTP synchronization, add clock skew detection |
| Cron calculation errors | Test edge cases with known correct results | Unit tests for DST transitions, month boundaries | Fix timezone handling logic, add comprehensive test coverage |
| Delayed promotion lag | Compare intended vs actual execution times | PromoteDelayedJobs timing logs, execution latency metrics | Optimize promotion process, adjust promotion interval |
| Coordination timeouts | Analyze heartbeat and consensus timing | Coordination service performance metrics, network latency | Tune timeout values, improve coordination service performance |
| Job claim race conditions | Review atomic operation timing | Concurrent claim attempt logs, success/failure patterns | Implement stronger consistency guarantees, add retry logic |

**Race Condition Investigation**

Race conditions in distributed schedulers typically occur during concurrent operations on shared state, creating timing-dependent failures that may only manifest under specific load or network conditions.

Job claiming race condition analysis focuses on scenarios where multiple workers attempt to claim the same job simultaneously. Begin by reviewing job claim attempt logs during high-concurrency periods, looking for patterns where multiple workers report successful claim operations for the same job ID. Examine the atomic operation implementation in the underlying storage system to verify that compare-and-swap operations are functioning correctly.

Deduplication race condition investigation examines scenarios where identical jobs bypass deduplication logic due to timing windows in the duplicate detection process. Review job submission logs for jobs with identical `IdempotencyKey` values that both result in successful queue insertion. Check the deduplication hash table consistency and verify that hash computation produces identical results for equivalent job payloads.

Worker registration race conditions occur when multiple workers attempt to register with the same ID simultaneously or when worker state updates conflict during rapid heartbeat processing. Examine worker registration logs for duplicate worker IDs or workers that appear to register successfully but don't receive job assignments. Review heartbeat processing for workers that report successful heartbeat transmission but show as unavailable in coordinator state.

Coordination state race conditions involve conflicts during leader election or job assignment operations where distributed state updates occur in different orders across cluster nodes. Review leadership election logs for scenarios where vote counting produces inconsistent results or where multiple nodes claim victory simultaneously. Examine job assignment state for jobs that appear assigned to multiple workers or workers that report assignment conflicts.

Recovery operation race conditions can occur when failed worker detection triggers job reassignment while the original worker is still processing the job successfully. Review job recovery logs for scenarios where job reassignment occurs followed by completion reports from the original worker. Check fencing token validation for jobs that complete successfully but have their completion reports rejected due to stale authorization.

| Race Condition Type | Detection Method | Evidence Patterns | Mitigation Approach |
|-------------------|-----------------|------------------|-------------------|
| Job claim conflicts | Multiple workers claiming same job | Duplicate successful claim logs for same job ID | Implement stronger atomic operations, add claim verification |
| Deduplication bypass | Identical IdempotencyKey jobs both execute | Same IdempotencyKey with multiple queue insertions | Add deduplication transaction scope, implement retries |
| Worker registration conflicts | Duplicate worker IDs or state inconsistency | Worker registration success with no job assignments | Serialize worker registration, add registration verification |
| Coordination state conflicts | Inconsistent leadership or assignment state | Multiple leaders or conflicting job assignments | Implement coordination transaction boundaries |
| Recovery operation conflicts | Job reassignment during successful execution | Job completion after reassignment triggers | Strengthen fencing token validation, add recovery coordination |

⚠️ **Pitfall: Log Analysis Without Correlation**

A common debugging mistake involves analyzing logs from individual components without correlating events across the distributed system boundaries. For example, investigating duplicate job execution by only examining worker logs misses the coordination failures that enabled the duplication. Always gather logs from all relevant components (coordinator, worker, queue) for the same time period and correlate events using `correlation_id` and `job_id` fields to understand the complete failure sequence.

⚠️ **Pitfall: Metric Alert Threshold Sensitivity**

Setting metric alert thresholds too sensitively creates alert fatigue while setting them too loosely misses genuine problems. Distributed scheduler metrics often show natural variation due to job patterns and network conditions. Establish baseline behavior during normal operation before setting alert thresholds, and use percentage-based thresholds rather than absolute values to account for system scaling.

⚠️ **Pitfall: Race Condition Reproduction Attempts**

Attempting to reproduce race conditions by manually triggering concurrent operations often fails because the timing conditions that created the original race may not be replicable in testing environments. Instead, focus on identifying the shared state and atomic operation boundaries where races can occur, then implement stronger consistency guarantees rather than trying to reproduce the exact race scenario.

### Implementation Guidance

Implementing effective debugging capabilities requires building observability into the distributed scheduler from the ground up, rather than adding diagnostic tools as an afterthought. The debugging implementation focuses on structured data collection, efficient storage and retrieval, and automated analysis tools that accelerate problem diagnosis.

**Technology Recommendations**

| Component | Simple Option | Advanced Option | Trade-offs |
|-----------|---------------|-----------------|------------|
| Structured Logging | Go standard log/slog with JSON formatting | ELK Stack (Elasticsearch, Logstash, Kibana) | Simple: easy setup, limited search. Advanced: powerful search, complex infrastructure |
| Metrics Collection | Prometheus with Go prometheus client | Datadog or New Relic with custom dashboards | Simple: open source, self-hosted. Advanced: managed service, better UI |
| Distributed Tracing | Jaeger with OpenTelemetry Go SDK | Zipkin with custom span correlation | Simple: CNCF standard, good community. Advanced: better performance analysis |
| Time Series Database | Prometheus for metrics storage | InfluxDB with Grafana for visualization | Simple: integrated with collection. Advanced: better time series performance |
| Log Aggregation | File-based logging with logrotate | Fluentd with Elasticsearch backend | Simple: no external dependencies. Advanced: centralized searchable logs |

**Recommended File Structure**

```
internal/
  debugging/
    logger.go              ← Structured logging configuration
    metrics.go             ← Metrics collection and registration  
    tracing.go             ← Distributed tracing setup
    correlation.go         ← Correlation ID management
    diagnostics.go         ← Health check and diagnostic endpoints
    debug_handler.go       ← Debug information HTTP endpoints
  monitoring/
    alerts.go              ← Metric alert definitions
    dashboards.go          ← Dashboard configuration
    healthcheck.go         ← System health assessment
cmd/
  debug-cli/
    main.go                ← Command-line debugging utility
    job_inspector.go       ← Job state inspection commands
    worker_inspector.go    ← Worker state inspection commands
    trace_analyzer.go      ← Trace analysis utilities
tools/
  log-analyzer/
    main.go                ← Log analysis and correlation tool
    pattern_detector.go    ← Common failure pattern detection
  metrics-dashboard/
    grafana-config.json    ← Pre-built dashboard definitions
    alert-rules.yaml       ← Prometheus alert rule definitions
```

**Infrastructure Starter Code**

Complete logging infrastructure with structured output and correlation support:

```go
// internal/debugging/logger.go
package debugging

import (
    "context"
    "log/slog"
    "os"
    "time"
)

type ContextKey string

const CorrelationIDKey ContextKey = "correlation_id"

// CorrelatedLogger provides structured logging with automatic correlation ID injection
type CorrelatedLogger struct {
    logger *slog.Logger
    component string
    nodeID string
}

// NewCorrelatedLogger creates a structured logger with component identification
func NewCorrelatedLogger(component, nodeID string) *CorrelatedLogger {
    handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelDebug,
        AddSource: true,
    })
    
    logger := slog.New(handler).With(
        slog.String("component", component),
        slog.String("node_id", nodeID),
        slog.Time("timestamp", time.Now()),
    )
    
    return &CorrelatedLogger{
        logger: logger,
        component: component,
        nodeID: nodeID,
    }
}

// WithContext extracts correlation ID from context and adds it to log entry
func (l *CorrelatedLogger) WithContext(ctx context.Context) *slog.Logger {
    correlationID, ok := ctx.Value(CorrelationIDKey).(string)
    if !ok {
        correlationID = "unknown"
    }
    
    return l.logger.With(slog.String("correlation_id", correlationID))
}

// JobOperation logs job-related operations with standardized fields
func (l *CorrelatedLogger) JobOperation(ctx context.Context, level slog.Level, operation string, jobID string, attrs ...slog.Attr) {
    logger := l.WithContext(ctx).With(
        slog.String("operation", operation),
        slog.String("job_id", jobID),
    )
    
    logger.LogAttrs(ctx, level, operation, attrs...)
}

// WorkerOperation logs worker-related operations with standardized fields  
func (l *CorrelatedLogger) WorkerOperation(ctx context.Context, level slog.Level, operation string, workerID string, attrs ...slog.Attr) {
    logger := l.WithContext(ctx).With(
        slog.String("operation", operation),
        slog.String("worker_id", workerID),
    )
    
    logger.LogAttrs(ctx, level, operation, attrs...)
}

// CoordinationOperation logs coordination protocol events
func (l *CorrelatedLogger) CoordinationOperation(ctx context.Context, level slog.Level, operation string, attrs ...slog.Attr) {
    logger := l.WithContext(ctx).With(
        slog.String("operation", operation),
        slog.String("protocol", "coordination"),
    )
    
    logger.LogAttrs(ctx, level, operation, attrs...)
}
```

Complete metrics collection infrastructure with Prometheus integration:

```go
// internal/debugging/metrics.go
package debugging

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "time"
)

// SchedulerMetrics provides comprehensive metrics collection for distributed scheduler
type SchedulerMetrics struct {
    // Job execution metrics
    JobSubmissionRate *prometheus.CounterVec
    JobCompletionLatency *prometheus.HistogramVec
    JobFailureRate *prometheus.CounterVec
    JobQueueDepth *prometheus.GaugeVec
    
    // Worker coordination metrics  
    ActiveWorkerCount prometheus.Gauge
    HeartbeatLatency *prometheus.HistogramVec
    LeaderElectionCount prometheus.Counter
    JobAssignmentLatency prometheus.Histogram
    
    // Queue operation metrics
    DeduplicationHitRate *prometheus.CounterVec
    DelayedPromotionLag prometheus.Histogram
    QueueOperationLatency *prometheus.HistogramVec
    
    // System health metrics
    CoordinationServiceLatency *prometheus.HistogramVec
    NetworkPartitionCount prometheus.Counter
    MemoryUsage prometheus.Gauge
}

// NewSchedulerMetrics creates and registers all scheduler metrics with Prometheus
func NewSchedulerMetrics(component string) *SchedulerMetrics {
    return &SchedulerMetrics{
        JobSubmissionRate: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "scheduler_job_submissions_total",
                Help: "Total number of job submissions by priority level",
            },
            []string{"component", "priority", "job_type"},
        ),
        
        JobCompletionLatency: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "scheduler_job_completion_duration_seconds",
                Help: "Job completion latency distribution",
                Buckets: []float64{0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0},
            },
            []string{"component", "priority", "success"},
        ),
        
        JobFailureRate: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "scheduler_job_failures_total", 
                Help: "Total job failures by failure type",
            },
            []string{"component", "failure_type", "retriable"},
        ),
        
        JobQueueDepth: promauto.NewGaugeVec(
            prometheus.GaugeOpts{
                Name: "scheduler_queue_depth",
                Help: "Current job queue depth by priority level",
            },
            []string{"component", "priority", "state"},
        ),
        
        ActiveWorkerCount: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "scheduler_active_workers",
                Help: "Number of currently active workers",
            },
        ),
        
        HeartbeatLatency: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "scheduler_heartbeat_latency_seconds",
                Help: "Worker heartbeat response time distribution", 
                Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0},
            },
            []string{"component", "worker_id", "success"},
        ),
        
        LeaderElectionCount: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "scheduler_leader_elections_total",
                Help: "Total number of leader election events",
            },
        ),
        
        JobAssignmentLatency: promauto.NewHistogram(
            prometheus.HistogramOpts{
                Name: "scheduler_job_assignment_duration_seconds",
                Help: "Time from job submission to worker assignment",
                Buckets: []float64{0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0},
            },
        ),
        
        DeduplicationHitRate: promauto.NewCounterVec(
            prometheus.CounterOpts{
                Name: "scheduler_deduplication_checks_total",
                Help: "Deduplication check results",
            },
            []string{"component", "result"},
        ),
        
        DelayedPromotionLag: promauto.NewHistogram(
            prometheus.HistogramOpts{
                Name: "scheduler_delayed_promotion_lag_seconds", 
                Help: "Delay between scheduled time and actual job promotion",
                Buckets: []float64{0, 1, 5, 10, 30, 60, 300},
            },
        ),
        
        QueueOperationLatency: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "scheduler_queue_operation_duration_seconds",
                Help: "Queue operation latency by operation type",
                Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
            },
            []string{"component", "operation"},
        ),
        
        CoordinationServiceLatency: promauto.NewHistogramVec(
            prometheus.HistogramOpts{
                Name: "scheduler_coordination_service_duration_seconds",
                Help: "Coordination service response time distribution",
                Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
            },
            []string{"component", "service", "operation"},
        ),
        
        NetworkPartitionCount: promauto.NewCounter(
            prometheus.CounterOpts{
                Name: "scheduler_network_partitions_total",
                Help: "Total number of detected network partition events",
            },
        ),
        
        MemoryUsage: promauto.NewGauge(
            prometheus.GaugeOpts{
                Name: "scheduler_memory_usage_bytes",
                Help: "Current memory usage in bytes",
            },
        ),
    }
}

// RecordJobSubmission records a job submission event with priority and type labels
func (m *SchedulerMetrics) RecordJobSubmission(component, priority, jobType string) {
    m.JobSubmissionRate.WithLabelValues(component, priority, jobType).Inc()
}

// RecordJobCompletion records job completion timing with success indicator
func (m *SchedulerMetrics) RecordJobCompletion(component, priority string, duration time.Duration, success bool) {
    successLabel := "false"
    if success {
        successLabel = "true"
    }
    m.JobCompletionLatency.WithLabelValues(component, priority, successLabel).Observe(duration.Seconds())
}

// RecordJobFailure records job failure with categorization
func (m *SchedulerMetrics) RecordJobFailure(component, failureType string, retriable bool) {
    retriableLabel := "false"  
    if retriable {
        retriableLabel = "true"
    }
    m.JobFailureRate.WithLabelValues(component, failureType, retriableLabel).Inc()
}

// UpdateQueueDepth updates the current queue depth for a specific priority level
func (m *SchedulerMetrics) UpdateQueueDepth(component, priority, state string, depth float64) {
    m.JobQueueDepth.WithLabelValues(component, priority, state).Set(depth)
}

// RecordHeartbeat records worker heartbeat timing and success
func (m *SchedulerMetrics) RecordHeartbeat(component, workerID string, latency time.Duration, success bool) {
    successLabel := "false"
    if success {
        successLabel = "true"  
    }
    m.HeartbeatLatency.WithLabelValues(component, workerID, successLabel).Observe(latency.Seconds())
}
```

**Core Debugging Utilities**

Debug information collection and analysis utilities:

```go
// internal/debugging/diagnostics.go
package debugging

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "runtime"
    "time"
)

// DiagnosticCollector gathers system diagnostic information for debugging
type DiagnosticCollector struct {
    component string
    nodeID    string
    startTime time.Time
}

// SystemDiagnostics contains comprehensive system state for debugging
type SystemDiagnostics struct {
    Component    string            `json:"component"`
    NodeID       string            `json:"node_id"`
    Timestamp    time.Time         `json:"timestamp"`
    Uptime       time.Duration     `json:"uptime"`
    MemoryStats  runtime.MemStats  `json:"memory_stats"`
    GoRoutines   int               `json:"goroutines"`
    SystemInfo   SystemInfo        `json:"system_info"`
    CustomState  map[string]interface{} `json:"custom_state"`
}

// SystemInfo contains basic system identification information
type SystemInfo struct {
    GoVersion   string `json:"go_version"`
    OS          string `json:"os"`
    Arch        string `json:"arch"`
    CPUs        int    `json:"cpus"`
}

// NewDiagnosticCollector creates diagnostic collector for specified component
func NewDiagnosticCollector(component, nodeID string) *DiagnosticCollector {
    return &DiagnosticCollector{
        component: component,
        nodeID:    nodeID,
        startTime: time.Now(),
    }
}

// CollectDiagnostics gathers comprehensive system diagnostic information
func (d *DiagnosticCollector) CollectDiagnostics(customState map[string]interface{}) *SystemDiagnostics {
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    
    return &SystemDiagnostics{
        Component:   d.component,
        NodeID:      d.nodeID,
        Timestamp:   time.Now(),
        Uptime:      time.Since(d.startTime),
        MemoryStats: memStats,
        GoRoutines:  runtime.NumGoroutine(),
        SystemInfo: SystemInfo{
            GoVersion: runtime.Version(),
            OS:        runtime.GOOS,
            Arch:      runtime.GOARCH,
            CPUs:      runtime.NumCPU(),
        },
        CustomState: customState,
    }
}

// DebugHandler provides HTTP endpoint for diagnostic information retrieval
type DebugHandler struct {
    collector *DiagnosticCollector
    stateProvider func() map[string]interface{}
}

// NewDebugHandler creates HTTP handler for debug endpoints
func NewDebugHandler(collector *DiagnosticCollector, stateProvider func() map[string]interface{}) *DebugHandler {
    return &DebugHandler{
        collector: collector,
        stateProvider: stateProvider,
    }
}

// ServeHTTP handles diagnostic information requests
func (h *DebugHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    // TODO 1: Extract diagnostic request type from URL path (/debug/system, /debug/state, etc.)
    // TODO 2: Call appropriate diagnostic collection method based on request type
    // TODO 3: Collect custom state from stateProvider function
    // TODO 4: Marshal diagnostic information to JSON response
    // TODO 5: Set appropriate HTTP headers (Content-Type: application/json)
    // TODO 6: Write JSON response with proper error handling
    // Hint: Use h.collector.CollectDiagnostics(h.stateProvider()) for full diagnostics
}

// LogAnalyzer provides utilities for parsing and analyzing structured logs
type LogAnalyzer struct {
    correlationIndex map[string][]LogEntry
    timeIndex        []LogEntry
}

// LogEntry represents a parsed structured log entry
type LogEntry struct {
    Timestamp     time.Time              `json:"timestamp"`
    Component     string                 `json:"component"`
    NodeID        string                 `json:"node_id"`
    CorrelationID string                 `json:"correlation_id"`
    Level         string                 `json:"level"`
    Operation     string                 `json:"operation,omitempty"`
    JobID         string                 `json:"job_id,omitempty"`
    WorkerID      string                 `json:"worker_id,omitempty"`
    Message       string                 `json:"message"`
    Attributes    map[string]interface{} `json:"attributes"`
}

// NewLogAnalyzer creates log analysis utility with indexing capability
func NewLogAnalyzer() *LogAnalyzer {
    return &LogAnalyzer{
        correlationIndex: make(map[string][]LogEntry),
        timeIndex:        make([]LogEntry, 0),
    }
}

// ParseLogLine parses structured JSON log entry and adds to indexes
func (a *LogAnalyzer) ParseLogLine(line string) error {
    // TODO 1: Parse JSON log line into LogEntry struct
    // TODO 2: Add entry to correlationIndex using correlation_id as key
    // TODO 3: Insert entry into timeIndex maintaining chronological order
    // TODO 4: Return parsing errors with context about malformed fields
    // Hint: Use json.Unmarshal() for parsing and sort.Search() for time index insertion
    return nil
}

// GetCorrelatedEntries retrieves all log entries for specific correlation ID
func (a *LogAnalyzer) GetCorrelatedEntries(correlationID string) []LogEntry {
    // TODO 1: Look up correlation ID in correlationIndex
    // TODO 2: Return slice of log entries in chronological order
    // TODO 3: Return empty slice if correlation ID not found
    // Hint: Sort entries by timestamp before returning
    return nil
}

// AnalyzeFailurePattern identifies common failure patterns in log sequences  
func (a *LogAnalyzer) AnalyzeFailurePattern(entries []LogEntry) FailurePattern {
    // TODO 1: Scan entries for error-level log messages
    // TODO 2: Identify timing patterns (rapid retries, timeout sequences)
    // TODO 3: Correlate component failure sequences (coordinator -> worker failures)
    // TODO 4: Classify failure type based on observed patterns
    // TODO 5: Return structured failure analysis with root cause hypothesis
    return FailurePattern{}
}

// FailurePattern represents analysis results for a failure sequence
type FailurePattern struct {
    FailureType   string    `json:"failure_type"`
    RootCause     string    `json:"root_cause"`
    ComponentPath []string  `json:"component_path"` 
    Duration      time.Duration `json:"duration"`
    Symptoms      []string  `json:"symptoms"`
    Recommendation string   `json:"recommendation"`
}
```

**Milestone Checkpoints**

After implementing the debugging infrastructure, verify the following behavior:

1. **Structured Logging Verification**: Start all scheduler components and submit test jobs. Check that logs contain all required fields (component, node_id, correlation_id, timestamp) and can be filtered by correlation ID. Verify that job operations include job_id fields and worker operations include worker_id fields.

2. **Metrics Collection Validation**: Access the metrics endpoint (typically `/metrics`) and verify that all defined metrics appear with appropriate labels. Submit jobs and confirm that job submission and completion metrics increment correctly. Monitor queue depth metrics during job processing.

3. **Diagnostic Endpoint Testing**: Access the debug endpoints (`/debug/system`) and verify that complete diagnostic information returns in JSON format. Check that memory statistics, goroutine counts, and custom state information appear correctly.

4. **Correlation Analysis Testing**: Generate correlated log entries by submitting jobs and verify that the log analyzer can retrieve all entries for a specific correlation ID. Test failure pattern analysis with intentionally failed jobs.

**Debugging Tips**

| Symptom | Likely Cause | Diagnostic Command | Fix Approach |
|---------|--------------|-------------------|--------------|
| Metrics not updating | Prometheus registration failure | Check `/metrics` endpoint for metric presence | Verify metric registration in component initialization |
| Correlation IDs missing | Context propagation failure | Grep logs for "correlation_id":"unknown" | Add correlation ID to all context.Context values |
| Log entries missing fields | Logger configuration error | Check log JSON structure with `jq` tool | Verify structured logger initialization with required fields |
| Debug endpoint returns errors | State provider function panic | Check component logs during debug request | Add error handling in state provider function |
| Trace spans not appearing | Tracing backend connectivity | Check Jaeger UI for trace presence | Verify tracing configuration and network connectivity |


## Future Extensions

> **Milestone(s):** This section explores potential enhancements that build upon the completed implementation of all three milestones - extending the cron parser with workflow dependencies (Milestone 1), adding advanced queue features like resource awareness (Milestone 2), and scaling the worker coordination beyond single-cluster deployments (Milestone 3).

The distributed job scheduler established through the three core milestones provides a solid foundation for reliable, fault-tolerant job execution. However, real-world production systems often require capabilities beyond basic scheduling, prioritization, and coordination. This section outlines significant enhancements that transform the scheduler from a general-purpose task executor into a comprehensive workflow orchestration platform capable of handling complex enterprise workloads across multiple data centers.

Think of the current scheduler as a well-organized city bus system - it reliably transports passengers (jobs) from origin to destination using fixed routes (cron schedules) with good capacity management (priorities) and fault tolerance (worker coordination). The future extensions described here are like evolving from a bus system to a comprehensive transportation network that includes subways, trains, and planes - adding route dependencies, real-time capacity optimization, and inter-city connections.

These extensions fall into three categories that build naturally upon the existing architecture. **Advanced Scheduling** capabilities transform the scheduler from executing independent jobs to orchestrating complex workflows with dependencies and resource constraints. **Operational Features** provide the visibility and control mechanisms necessary for production deployments at scale. **Scalability Improvements** enable the scheduler to operate across multiple data centers and handle workloads that exceed single-cluster capacity.

> The key architectural insight is that these extensions maintain backward compatibility with the existing three-milestone implementation while adding new layers of functionality. Existing jobs continue to work unchanged, but new capabilities become available through extended APIs and configuration options.

Each extension category addresses different aspects of production readiness. Advanced scheduling tackles the complexity of real-world workflows where jobs have interdependencies and compete for limited resources. Operational features provide the observability and administrative capabilities that operations teams require for managing large-scale deployments. Scalability improvements address the fundamental limits of single-cluster architectures by enabling geographic distribution and horizontal scaling beyond individual data center capacity.

### Advanced Scheduling

The current scheduler executes jobs independently based on cron expressions and priorities. Advanced scheduling extends this model to handle **job dependencies**, **resource-aware scheduling**, and **workflow orchestration** - transforming the system from a task executor into a comprehensive workflow engine.

Think of job dependencies like a restaurant kitchen during dinner service. The salad station can't plate dishes until the grill finishes the protein, the dessert chef waits for dinner orders to complete, and dishwashing depends on completed courses returning to the kitchen. Similarly, data processing workflows often require strict ordering - the ETL job must complete before the reporting job runs, and the backup job should wait until all database operations finish.

#### Job Dependencies and Workflow DAGs

Job dependencies introduce directed acyclic graph (DAG) scheduling where jobs specify prerequisite jobs that must complete successfully before execution begins. This extends the current `Job` structure to include dependency relationships and success criteria.

**Extended Job Definition for Dependencies:**

| Field | Type | Description |
|-------|------|-------------|
| Dependencies | []JobDependency | List of prerequisite jobs that must complete before this job can execute |
| DependencyMode | DependencyMode | How to handle dependency completion (ALL_SUCCESS, ANY_SUCCESS, ALL_COMPLETE) |
| DependencyTimeout | time.Duration | Maximum time to wait for dependencies before marking job as failed |
| WorkflowID | string | Optional identifier grouping related jobs into a workflow unit |
| WorkflowPosition | int | Job's position in workflow execution order (for visualization) |

**JobDependency Structure:**

| Field | Type | Description |
|-------|------|-------------|
| JobID | string | Identifier of the prerequisite job |
| JobName | string | Human-readable name of the prerequisite job for workflow visualization |
| RequiredState | JobState | Required completion state (COMPLETED for success, FAILED for failure trigger) |
| DataPassing | map[string]string | Output data from prerequisite job to pass as input to dependent job |
| TimeoutAction | TimeoutAction | Action when dependency doesn't complete within timeout (FAIL, SKIP, PROCEED) |

The dependency resolution algorithm operates through a **dependency graph evaluator** that maintains workflow state and triggers job execution when prerequisites complete:

1. **Dependency Registration**: When jobs with dependencies are submitted, the scheduler builds an in-memory dependency graph linking prerequisite jobs to their dependents
2. **Completion Monitoring**: As jobs complete, the dependency evaluator checks all dependent jobs to see if their prerequisites are now satisfied
3. **Eligibility Promotion**: Jobs with satisfied dependencies are promoted from a "waiting" state to the normal priority queue for worker assignment
4. **Failure Propagation**: When a prerequisite job fails, dependent jobs can either fail immediately, skip execution, or proceed based on their configured `TimeoutAction`
5. **Cycle Detection**: The scheduler validates that dependency relationships form a DAG and rejects job submissions that would create cycles

> **Decision: Dependency Storage Strategy**
> - **Context**: Job dependencies need persistent storage to survive coordinator restarts and enable dependency resolution across time
> - **Options Considered**: In-memory only, database relations, graph database, Redis graph structures
> - **Decision**: Redis with graph-like operations using sets for incoming/outgoing edges per job
> - **Rationale**: Leverages existing Redis infrastructure, provides atomic operations for dependency updates, enables efficient graph traversal queries
> - **Consequences**: Dependency relationships persist across restarts, but complex graph queries are less efficient than dedicated graph databases

#### Resource-Aware Scheduling

The current scheduler assigns jobs to workers based solely on availability and capability matching. Resource-aware scheduling extends this to consider **memory requirements**, **CPU constraints**, **storage needs**, and **exclusive resource access** when making assignment decisions.

Think of resource-aware scheduling like an airport gate assignment system. Small regional jets can use any gate, but wide-body aircraft require gates with jetbridges capable of handling their size. Similarly, some gates have ground power connections needed for electric aircraft, while others provide fuel access for conventional planes. The scheduler must match aircraft requirements with gate capabilities while maximizing utilization.

**Extended Worker Resource Model:**

| Field | Type | Description |
|-------|------|-------------|
| AvailableMemoryMB | int64 | Current available memory in megabytes for job execution |
| AvailableCPUCores | float64 | Available CPU cores (fractional for partial core allocation) |
| AvailableStorageGB | int64 | Available temporary storage in gigabytes |
| ExclusiveResources | []string | List of exclusive resources this worker can provide (GPU types, license slots) |
| ResourceLimits | ResourceLimits | Maximum resource levels this worker can provide |
| CurrentAllocations | map[string]ResourceAllocation | Resources currently allocated to running jobs |

**Job Resource Requirements:**

| Field | Type | Description |
|-------|------|-------------|
| RequiredMemoryMB | int64 | Minimum memory required for successful job execution |
| RequiredCPUCores | float64 | Minimum CPU cores needed (fractional allocation supported) |
| RequiredStorageGB | int64 | Temporary storage space needed during execution |
| ExclusiveResource | string | Exclusive resource required (empty string if none needed) |
| ResourceReservation | time.Duration | How long to hold resources before job starts (for delayed jobs) |
| AffinityRules | []AffinityRule | Preferences for worker selection based on resource characteristics |

The resource-aware assignment algorithm extends the current job claiming process with resource validation and allocation tracking:

1. **Resource Requirement Analysis**: When a job becomes eligible for assignment, the scheduler extracts its resource requirements and determines compatible workers
2. **Worker Capacity Filtering**: The scheduler filters available workers to only those with sufficient resources to meet the job's requirements
3. **Affinity Evaluation**: Among capable workers, affinity rules are evaluated to prefer workers with characteristics like same datacenter, SSD storage, or specific CPU architectures
4. **Resource Reservation**: When a job is assigned to a worker, the scheduler reserves the required resources in its tracking system to prevent over-allocation
5. **Dynamic Rebalancing**: If high-priority jobs require resources currently allocated to lower-priority work, the scheduler can preempt lower-priority jobs
6. **Resource Release**: When jobs complete, their allocated resources are immediately returned to the worker's available pool

#### Workflow Orchestration Engine

Building upon job dependencies and resource awareness, workflow orchestration provides high-level primitives for defining and executing complex multi-job workflows with **conditional execution**, **parallel branches**, **loops**, and **error handling**.

Think of workflow orchestration like a sophisticated manufacturing assembly line. Raw materials enter the line, pass through multiple stations that can operate in parallel or sequence, with quality control checkpoints that can redirect work to rework stations or alternate paths. The entire process is coordinated by a central control system that monitors progress and handles disruptions.

**Workflow Definition Structure:**

| Field | Type | Description |
|-------|------|-------------|
| WorkflowID | string | Unique identifier for this workflow template |
| Name | string | Human-readable workflow name for administrative interfaces |
| Steps | []WorkflowStep | Ordered list of steps defining the workflow execution path |
| GlobalVariables | map[string]string | Variables available to all steps in the workflow |
| ErrorHandling | ErrorHandlingPolicy | How to handle step failures (ABORT, CONTINUE, RETRY_STEP) |
| MaxExecutionTime | time.Duration | Total timeout for entire workflow execution |
| CronSchedule | string | Optional cron expression for recurring workflow execution |

**WorkflowStep Definition:**

| Field | Type | Description |
|-------|------|-------------|
| StepID | string | Unique identifier within the workflow |
| StepType | StepType | Type of step (JOB, CONDITION, PARALLEL, LOOP, WAIT) |
| JobTemplate | *Job | Job definition for JOB-type steps |
| Condition | string | Boolean expression for CONDITION-type steps |
| ParallelBranches | [][]WorkflowStep | Parallel execution branches for PARALLEL-type steps |
| LoopCondition | string | Loop continuation condition for LOOP-type steps |
| WaitDuration | time.Duration | Fixed wait duration for WAIT-type steps |
| OnSuccess | []string | Next step IDs to execute on successful completion |
| OnFailure | []string | Next step IDs to execute on failure |
| RetryPolicy | StepRetryPolicy | Step-specific retry configuration |

The workflow orchestration engine operates as a state machine that tracks workflow execution progress and coordinates step transitions:

1. **Workflow Instantiation**: When a workflow is triggered (by cron schedule or API call), the engine creates a workflow execution instance with unique execution ID
2. **Step Resolution**: The engine identifies the first step(s) to execute based on the workflow definition and begins execution
3. **Job Generation**: For JOB-type steps, the engine creates actual `Job` instances from templates, substituting workflow variables and context
4. **Condition Evaluation**: CONDITION-type steps evaluate boolean expressions against workflow variables and job outputs to determine execution paths
5. **Parallel Coordination**: PARALLEL-type steps spawn multiple execution branches that run concurrently, with the engine tracking completion of all branches
6. **State Persistence**: Workflow execution state is persisted to enable recovery from coordinator failures during long-running workflows
7. **Completion Detection**: The workflow completes when all steps finish successfully or when an unrecoverable error occurs based on the error handling policy

> **Decision: Workflow Expression Language**
> - **Context**: Conditional steps and loop conditions require expression evaluation within workflow context
> - **Options Considered**: JavaScript V8 engine, Go template syntax, custom domain-specific language, JSONPath expressions
> - **Decision**: Go template syntax with custom functions for job output access and variable manipulation
> - **Rationale**: Leverages Go's built-in template engine, provides familiar syntax for Go developers, enables safe expression evaluation without full scripting language security concerns
> - **Consequences**: Expressions are limited to Go template capabilities but execution is fast and secure, with no risk of arbitrary code execution

**Common Pitfalls in Advanced Scheduling:**

⚠️ **Pitfall: Circular Dependency Creation**  
Allowing users to create job dependencies without cycle detection leads to workflows that can never execute because jobs wait for each other in a cycle. This manifests as jobs permanently stuck in "waiting for dependencies" state. Implement topological sort validation during workflow submission and reject any workflow definition that contains cycles.

⚠️ **Pitfall: Resource Over-Commitment**  
Reserving resources for delayed jobs without considering time-based availability leads to resource starvation where workers appear full but aren't actually executing work. Implement time-aware resource reservations that only hold resources close to job execution time, with automatic reservation release if jobs don't start within expected windows.

⚠️ **Pitfall: Workflow State Explosion**  
Storing complete workflow execution state for every step and variable update creates excessive storage overhead and slow state operations. Implement incremental state updates that only persist state changes, with periodic state compaction to merge incremental updates into consolidated checkpoints.

### Operational Features

Production distributed systems require comprehensive observability, administrative controls, and historical analysis capabilities. The operational features category adds **metrics and monitoring**, **job execution history**, **administrative interfaces**, and **audit logging** to transform the scheduler from a development tool into an enterprise-ready platform.

Think of operational features like the instrumentation and controls in an airline's operations center. Dispatchers need real-time visibility into flight status, weather conditions, and aircraft maintenance schedules. They require historical data to analyze patterns and optimize routes. Emergency procedures and administrative controls enable rapid response to disruptions. Similarly, a production job scheduler needs comprehensive operational capabilities to ensure reliable service delivery.

#### Comprehensive Metrics and Monitoring

The current scheduler includes basic Prometheus metrics for job execution and worker coordination. Comprehensive monitoring extends this with **detailed performance metrics**, **business-level indicators**, **alerting rules**, and **dashboard templates** that provide complete visibility into scheduler health and performance.

**Enhanced Metrics Collection:**

| Metric Category | Metrics | Purpose |
|-----------------|---------|---------|
| Job Lifecycle | job_submission_rate, job_completion_latency, job_success_rate, job_retry_frequency | Track job processing health and identify performance bottlenecks |
| Queue Dynamics | queue_depth_by_priority, queue_wait_time, job_aging_histogram, deduplication_rate | Monitor queue behavior and capacity planning needs |
| Worker Performance | worker_utilization, worker_job_throughput, worker_failure_rate, worker_resource_efficiency | Assess worker health and identify problematic nodes |
| Coordination Health | leader_election_frequency, heartbeat_latency, split_brain_events, consensus_latency | Monitor distributed coordination stability and network issues |
| Resource Utilization | memory_allocation_efficiency, cpu_utilization_distribution, storage_usage_patterns, exclusive_resource_contention | Track resource usage patterns for capacity planning |
| Workflow Execution | workflow_completion_rate, workflow_step_latency, workflow_branch_parallelism, workflow_failure_causes | Monitor complex workflow execution and identify optimization opportunities |

**Business-Level Indicators:**

Beyond technical metrics, operational deployments require business-level indicators that translate system behavior into business impact measurements:

| Business Indicator | Calculation | Business Impact |
|--------------------|-------------|-----------------|
| Schedule Adherence Rate | (Jobs started within SLA window) / (Total scheduled jobs) | Measures reliability of time-sensitive business processes |
| Critical Job Success Rate | (Critical priority jobs completed successfully) / (Total critical jobs) | Tracks success of highest-impact business operations |
| Resource ROI | (Successful job compute hours) / (Total provisioned compute hours) | Measures efficiency of infrastructure investment |
| Workflow SLA Compliance | (Workflows completed within SLA) / (Total workflows) | Tracks end-to-end business process reliability |
| Dependency Chain Efficiency | Average workflow completion time / Sum of individual job times | Measures overhead of workflow coordination vs direct execution |

**Alerting Framework:**

The monitoring system includes a comprehensive alerting framework with **graduated severity levels**, **smart alert correlation**, and **automated runbook integration**:

1. **Threshold-Based Alerts**: Traditional alerts based on metric thresholds (queue depth exceeds capacity, worker failure rate above normal)
2. **Anomaly Detection**: Machine learning-based alerts that identify unusual patterns in job execution, resource usage, or coordination behavior
3. **Correlation Rules**: Logic that groups related alerts to prevent alert storms during widespread issues (network partition affecting multiple workers)
4. **Escalation Policies**: Graduated response based on alert severity and duration (page on-call engineer for critical failures, create ticket for performance degradation)
5. **Runbook Integration**: Automatic attachment of relevant troubleshooting procedures and diagnostic commands to alert notifications

#### Job Execution History and Analytics

The current scheduler focuses on active job execution without preserving detailed historical information. Comprehensive job history provides **execution analytics**, **performance trending**, **failure pattern analysis**, and **capacity planning data** through long-term data retention and analysis capabilities.

**Job Execution Record:**

| Field | Type | Description |
|-------|------|-------------|
| ExecutionID | string | Unique identifier for this specific job execution attempt |
| JobID | string | Identifier of the job definition that was executed |
| WorkflowID | string | Optional workflow identifier if job was part of workflow execution |
| SubmissionTime | time.Time | When the job was originally submitted to the scheduler |
| ScheduledTime | time.Time | When the job was scheduled to begin execution based on cron expression |
| ClaimTime | time.Time | When a worker successfully claimed the job for execution |
| StartTime | time.Time | When job execution actually began on the worker |
| CompletionTime | time.Time | When job execution finished (successfully or with failure) |
| ExecutionDuration | time.Duration | Total time spent executing the job |
| QueueWaitTime | time.Duration | Time spent waiting in queue before worker assignment |
| ClaimToStartDelay | time.Duration | Delay between worker claim and actual execution start |
| WorkerID | string | Identifier of the worker that executed the job |
| ResourcesUsed | ResourceUtilization | Actual resource consumption during execution |
| ExitCode | int | Job process exit code |
| ExecutionLogs | string | Captured standard output and error from job execution |
| RetryAttempts | []RetryRecord | Details of any retry attempts before final completion |
| Dependencies | []DependencyResolution | How job dependencies were resolved for this execution |

**Performance Analytics Engine:**

The analytics engine processes historical execution records to provide insights for optimization and troubleshooting:

1. **Trend Analysis**: Identifies patterns in job execution times, failure rates, and resource usage over time to detect performance degradation or improvement
2. **Capacity Planning**: Analyzes historical queue depths, worker utilization, and execution patterns to recommend infrastructure scaling decisions
3. **Failure Root Cause Analysis**: Correlates job failures with system conditions (worker health, resource availability, network issues) to identify common failure causes
4. **Schedule Optimization**: Recommends schedule adjustments based on actual execution patterns to reduce resource contention and improve completion rates
5. **Worker Performance Profiling**: Identifies workers with consistently poor performance or resource efficiency to guide maintenance decisions

**Data Retention and Archival:**

Long-term job history requires careful data management to balance analytical value with storage costs:

| Retention Period | Data Granularity | Storage Location | Access Pattern |
|------------------|------------------|------------------|----------------|
| Last 7 days | Full execution records with logs | Hot storage (SSD) | Real-time dashboards, immediate troubleshooting |
| 8-90 days | Execution records without logs | Warm storage (HDD) | Performance analysis, capacity planning |
| 91 days - 1 year | Aggregated daily/hourly summaries | Cold storage (object store) | Long-term trending, compliance reporting |
| 1+ years | Monthly aggregate summaries | Archive storage | Historical analysis, audit requirements |

#### Administrative Interfaces

Production schedulers require comprehensive administrative capabilities for **job management**, **worker administration**, **system configuration**, and **emergency operations**. The administrative interface provides both web-based dashboards and programmatic APIs for operational teams.

**Web-Based Administrative Dashboard:**

The dashboard provides comprehensive visibility and control through several specialized views:

1. **System Overview Dashboard**: Real-time system health, active job counts, worker status, and critical alerts in a single view
2. **Job Management Interface**: Search, filter, and manage jobs with capabilities to pause, resume, cancel, or manually trigger job execution
3. **Worker Administration Panel**: View worker health, resource utilization, and administrative actions like graceful shutdown or resource limit adjustments
4. **Workflow Visualization**: Graphical representation of workflow definitions and execution progress with interactive dependency graphs
5. **Performance Analytics Dashboard**: Historical performance trends, capacity utilization reports, and optimization recommendations
6. **Configuration Management**: Administrative interface for scheduler settings, worker policies, and system parameters

**Administrative API Endpoints:**

| Endpoint Category | Operations | Purpose |
|-------------------|------------|---------|
| Job Administration | List, search, pause, resume, cancel, retry, manual trigger | Operational control over job execution |
| Worker Management | List workers, drain worker, remove worker, update capacity | Worker lifecycle and capacity management |
| System Configuration | Get/set scheduler config, update cron schedules, modify priorities | Runtime configuration adjustments |
| Emergency Operations | Emergency stop, bulk job cancellation, system maintenance mode | Crisis response and maintenance procedures |
| Audit and Reporting | Execution reports, performance summaries, compliance exports | Operational reporting and audit trails |

**Role-Based Access Control:**

Administrative interfaces require sophisticated access control to ensure appropriate separation of duties:

| Role | Permissions | Typical Use Cases |
|------|-------------|-------------------|
| Viewer | Read-only access to dashboards and job status | Developers checking job status, managers reviewing performance |
| Operator | Job pause/resume/cancel, worker drain operations | On-call engineers responding to alerts, scheduled maintenance |
| Administrator | Full system configuration, worker management, emergency operations | System administrators, DevOps team leads |
| Auditor | Read-only access to all data including sensitive logs and configurations | Compliance audits, security reviews |

#### Comprehensive Audit Logging

Enterprise deployments require detailed audit trails for **compliance requirements**, **security monitoring**, and **operational analysis**. Comprehensive audit logging captures all administrative actions, system events, and security-relevant activities with tamper-evident storage.

**Audit Event Categories:**

| Category | Events Logged | Security Relevance |
|----------|---------------|-------------------|
| Administrative Actions | Job creation/modification/deletion, worker management, configuration changes | High - tracks all system modifications |
| Authentication/Authorization | User login/logout, permission checks, access denials | Critical - security monitoring and compliance |
| Job Execution | Job starts, completions, failures, retry attempts | Medium - operational analysis and debugging |
| System Events | Leader elections, worker registrations, coordinator failovers | Medium - system health and coordination monitoring |
| Data Access | Job payload access, log retrieval, configuration queries | Low - data access patterns and usage analysis |

**Audit Record Structure:**

| Field | Type | Description |
|-------|------|-------------|
| AuditID | string | Unique identifier for this audit event |
| Timestamp | time.Time | Precise timestamp when event occurred |
| EventType | AuditEventType | Category and specific type of audited event |
| ActorID | string | Identity of user or system component that triggered the event |
| ActorType | ActorType | Whether actor is human user, system service, or external API client |
| TargetResource | string | Resource that was affected by the audited action |
| Action | string | Specific action taken (CREATE, UPDATE, DELETE, ACCESS) |
| Outcome | AuditOutcome | Whether the action succeeded, failed, or was denied |
| SourceIP | string | Network address where the action originated |
| UserAgent | string | Client identifier for API or web interface actions |
| SessionID | string | Session identifier for correlated actions within same user session |
| Changes | map[string]AuditChange | Before/after values for modification actions |
| Context | map[string]string | Additional context relevant to the specific event |

**Tamper-Evident Storage:**

Audit logs require protection against modification to maintain their evidentiary value:

1. **Cryptographic Hashing**: Each audit record includes a hash of its content and the hash of the previous record, creating a blockchain-like chain
2. **Append-Only Storage**: Audit logs are stored in append-only files or database tables that prevent modification of existing records
3. **External Backup**: Regular encrypted backups of audit logs to external storage systems controlled by separate administrative domains
4. **Integrity Verification**: Automated processes verify the hash chain integrity and alert on any tampering attempts
5. **Retention Enforcement**: Automated retention policies prevent premature deletion while ensuring compliance with data retention regulations

### Scalability Improvements

The current three-milestone implementation operates effectively within a single cluster or data center. Production deployments often require capabilities that exceed single-cluster capacity or span multiple geographic regions. Scalability improvements enable **horizontal sharding**, **multi-datacenter deployment**, and **performance optimizations** that support enterprise-scale workloads.

Think of scalability improvements like expanding from a single factory to a global manufacturing network. The single factory has excellent coordination and efficiency, but geographic expansion requires regional facilities, supply chain coordination, and quality standardization across locations. Similarly, scaling the job scheduler beyond single-cluster deployment requires careful design of data partitioning, cross-region coordination, and performance optimization.

#### Horizontal Sharding Architecture

Single-cluster deployments face fundamental limits in job throughput, worker capacity, and coordination overhead. Horizontal sharding distributes jobs across multiple independent scheduler clusters based on **sharding keys**, with a **shard coordinator** managing job distribution and cross-shard queries.

**Sharding Strategy Design:**

The sharding architecture divides the job namespace across multiple clusters using configurable sharding keys:

| Sharding Key | Distribution Strategy | Use Cases |
|--------------|----------------------|-----------|
| Job Name Prefix | Hash-based distribution by job name prefix | Application-based isolation (billing-*, analytics-*, etc.) |
| Cron Schedule | Time-based distribution by execution frequency | Separate high-frequency and batch workloads |
| Priority Level | Priority-based shard assignment | Dedicated clusters for critical vs. routine work |
| Resource Requirements | Resource-based distribution by job resource needs | Specialized clusters for CPU-intensive vs. memory-intensive jobs |
| Tenant ID | Multi-tenant isolation with dedicated shards | Enterprise deployments with strict tenant isolation |

**Shard Coordinator Architecture:**

The shard coordinator operates as a lightweight routing layer that distributes jobs to appropriate shards without becoming a bottleneck:

| Component | Responsibility | Scalability Characteristics |
|-----------|---------------|---------------------------|
| Routing Service | Determines target shard for job submissions | Stateless, horizontally scalable, caches shard mappings |
| Shard Registry | Maintains shard health and capacity information | Replicated across multiple registry nodes, eventually consistent |
| Cross-Shard Query Engine | Aggregates queries across multiple shards | Parallel query execution, result merging, timeout handling |
| Rebalancing Coordinator | Manages shard capacity and job migration | Low-frequency operations, careful coordination to prevent disruption |

**Job Distribution Algorithm:**

The shard coordinator uses a consistent hashing algorithm to ensure stable job assignment even as shards are added or removed:

1. **Shard Key Extraction**: Extract the configured sharding key value from incoming job submissions
2. **Hash Calculation**: Compute a consistent hash of the shard key using a stable hash function (SHA-256)
3. **Shard Selection**: Map the hash value to a specific shard using consistent hashing ring or modulo arithmetic
4. **Health Check**: Verify the selected shard is healthy and accepting jobs; if not, select the next healthy shard in the ring
5. **Job Submission**: Forward the job to the selected shard's job submission API with original metadata preserved
6. **Response Handling**: Return the shard's response to the original client, including any shard-specific job identifiers

> **Decision: Cross-Shard Dependency Handling**
> - **Context**: Jobs with dependencies may be distributed across different shards, requiring coordination for dependency resolution
> - **Options Considered**: Prohibit cross-shard dependencies, centralized dependency tracker, distributed dependency resolution, job co-location
> - **Decision**: Implement job co-location where jobs with dependencies are assigned to the same shard as their prerequisites
> - **Rationale**: Maintains dependency resolution performance and simplifies coordination logic by avoiding cross-shard communication
> - **Consequences**: May create load imbalances if dependency chains are concentrated, but eliminates complex distributed dependency protocols

#### Multi-Datacenter Deployment

Enterprise deployments often require job execution across multiple geographic regions for **disaster recovery**, **latency optimization**, and **regulatory compliance**. Multi-datacenter deployment extends the scheduler to operate across regions while maintaining consistency and handling network partitions gracefully.

**Datacenter Topology Models:**

| Deployment Model | Characteristics | Coordination Requirements |
|------------------|-----------------|--------------------------|
| Active-Passive | Single active datacenter, passive backup for disaster recovery | Minimal coordination, fast failover, potential data loss during transition |
| Active-Active Regional | Each datacenter handles jobs for its geographic region | Moderate coordination for cross-region workflows and shared schedules |
| Global Active-Active | Jobs distributed globally based on resource availability | High coordination overhead, complex consensus, maximum availability |
| Federated | Independent schedulers with optional job sharing | Minimal coordination, loose coupling, manual intervention for cross-datacenter workflows |

**Network Partition Tolerance:**

Multi-datacenter deployments must handle network partitions gracefully without losing job execution capability:

1. **Partition Detection**: Monitor inter-datacenter connectivity using multiple network paths and consensus protocols to identify partitions
2. **Split-Brain Prevention**: Use external consensus systems (like managed etcd clusters) or quorum-based decisions to prevent multiple datacenters claiming leadership
3. **Graceful Degradation**: Continue processing region-local jobs during partitions while deferring cross-region workflows until connectivity restores
4. **State Reconciliation**: Automatically reconcile job states and execution history when partitions heal, handling conflicts through timestamp ordering or manual review
5. **Emergency Procedures**: Provide administrative controls to manually override partition detection for emergency operations

**Cross-Datacenter Job Scheduling:**

Some jobs require execution in specific datacenters due to data locality, regulatory requirements, or resource availability:

| Scheduling Strategy | Implementation | Trade-offs |
|---------------------|----------------|------------|
| Datacenter Affinity | Jobs specify preferred datacenter in metadata | Simple implementation, may create load imbalances |
| Data Locality Optimization | Schedule jobs in datacenter closest to required data | Optimal performance, requires data location tracking |
| Regulatory Compliance | Enforce jobs execute only in compliant regions | Strict compliance, potential capacity constraints |
| Load Balancing | Dynamically assign jobs based on datacenter capacity | Optimal resource utilization, complex coordination |

#### Performance Optimizations

High-throughput deployments require performance optimizations that minimize coordination overhead, reduce latency, and maximize resource utilization efficiency. These optimizations focus on **coordination protocol improvements**, **caching strategies**, and **batching optimizations**.

**Coordination Protocol Enhancements:**

The current leader election and worker coordination protocols can be optimized for higher scale:

1. **Hierarchical Leadership**: Implement regional leaders that coordinate with a global leader, reducing coordination overhead for local operations
2. **Lease-Based Coordination**: Use longer lease periods with more sophisticated renewal protocols to reduce heartbeat frequency without sacrificing failure detection speed
3. **Bulk Operations**: Batch multiple coordination operations (job assignments, status updates) into single messages to reduce network overhead
4. **Protocol Pipelining**: Allow multiple outstanding coordination requests to improve throughput in high-latency network environments
5. **Adaptive Timeouts**: Dynamically adjust coordination timeouts based on network conditions and system load to optimize for both responsiveness and stability

**Intelligent Caching Strategies:**

Caching reduces load on core coordination services and improves response times for frequently accessed data:

| Cache Type | Cached Data | Invalidation Strategy |
|------------|-------------|----------------------|
| Job Metadata Cache | Job definitions, cron expressions, priorities | TTL-based with immediate invalidation on updates |
| Worker Capability Cache | Worker resources, capabilities, health status | Heartbeat-driven updates with fallback TTL |
| Cron Calculation Cache | Next execution times for common cron expressions | Long TTL with timezone-aware invalidation |
| Dependency Graph Cache | Job dependency relationships for active workflows | Version-based invalidation when dependencies change |
| Shard Routing Cache | Shard assignment mappings for sharded deployments | Consistent hash-based with health-driven updates |

**Batching and Bulk Operations:**

Single-operation APIs create coordination overhead that limits throughput. Batching operations improves efficiency:

1. **Bulk Job Submission**: Accept multiple jobs in single API calls with transactional semantics for all-or-nothing submission
2. **Batch Worker Updates**: Aggregate multiple worker status updates into periodic batch operations to reduce coordination frequency
3. **Bulk Query Operations**: Support queries that return multiple jobs or workers in single requests to reduce API round-trips
4. **Streaming Updates**: Provide streaming APIs for real-time job status updates instead of polling-based approaches
5. **Lazy Evaluation**: Defer expensive operations like dependency resolution until actually needed rather than computing eagerly

**Common Pitfalls in Scalability Improvements:**

⚠️ **Pitfall: Inconsistent Sharding During Rebalancing**  
Rebalancing shards while jobs are in flight can lead to job assignments to incorrect shards or lost jobs during migration. Implement careful coordination protocols that ensure all in-flight jobs complete on their original shards before migrating future jobs to rebalanced shards.

⚠️ **Pitfall: Cross-Datacenter Clock Skew**  
Cron scheduling across datacenters with significant clock skew causes jobs to execute at unexpected times or miss schedules entirely. Implement NTP synchronization monitoring and validate that clock skew remains within acceptable bounds (typically under 1 second) across all scheduler nodes.

⚠️ **Pitfall: Cache Invalidation Storms**  
Popular job patterns or mass configuration updates can trigger cache invalidation storms that overload coordination services during cache refill. Implement staggered cache invalidation with exponential backoff and circuit breakers around cache refresh operations.

### Implementation Guidance

The future extensions described above represent significant enhancements to the core scheduler implementation. While these features go beyond the scope of the three-milestone project, understanding their architectural patterns and implementation approaches provides valuable insight into production system evolution.

**Technology Recommendations for Extensions:**

| Extension Category | Recommended Technologies | Rationale |
|-------------------|-------------------------|-----------|
| Job Dependencies | Redis Graph, Neo4j, or PostgreSQL with recursive queries | Graph relationships require efficient traversal and cycle detection |
| Resource Scheduling | Kubernetes-style resource quotas, cgroups integration | Proven patterns for resource management and isolation |
| Multi-Datacenter | etcd clusters, Consul Connect, or custom Raft implementation | Mature distributed consensus systems with partition tolerance |
| Metrics/Monitoring | Prometheus + Grafana, DataDog, or New Relic | Established monitoring ecosystems with alerting and dashboards |
| Workflow Orchestration | Temporal, Apache Airflow patterns, or custom state machines | Mature workflow engines provide proven orchestration patterns |

**File Structure for Extensions:**

```
project-root/
  cmd/
    scheduler/              ← main scheduler service
    shard-coordinator/      ← sharding coordinator service
    admin-dashboard/        ← administrative web interface
  internal/
    scheduler/              ← core scheduler from milestones
    extensions/
      dependencies/         ← job dependency resolution
        graph.go           ← dependency graph operations
        resolver.go        ← dependency resolution engine
      resources/           ← resource-aware scheduling
        allocator.go       ← resource allocation tracking
        scheduler.go       ← resource-aware job assignment
      workflows/           ← workflow orchestration
        engine.go          ← workflow execution engine
        definition.go      ← workflow definition parsing
      sharding/            ← horizontal sharding
        coordinator.go     ← shard coordination logic
        hasher.go          ← consistent hashing implementation
      monitoring/          ← enhanced monitoring
        collector.go       ← comprehensive metrics collection
        analytics.go       ← historical data analysis
      admin/               ← administrative interfaces
        api.go             ← administrative REST API
        dashboard.go       ← web dashboard handlers
  web/                     ← dashboard static assets
    dashboard/
      index.html           ← main dashboard page
      workflow-viz.js      ← workflow visualization
  deployments/
    k8s/                   ← Kubernetes deployment manifests
    docker-compose/        ← Docker Compose for development
```

**Extension Implementation Priorities:**

For teams considering implementing these extensions, the recommended priority order balances implementation complexity with operational value:

1. **Enhanced Monitoring and Metrics** - Essential for production deployment, builds on existing Prometheus integration
2. **Job Execution History** - Provides immediate operational value with moderate implementation complexity
3. **Administrative Interfaces** - Critical for operational teams, can start with simple REST APIs and evolve to full dashboards  
4. **Resource-Aware Scheduling** - High value for resource optimization, extends existing worker coordination patterns
5. **Job Dependencies** - Complex but enables significant workflow capabilities, requires careful dependency resolution design
6. **Horizontal Sharding** - Needed only for very high scale, complex distributed systems challenges
7. **Multi-Datacenter Deployment** - Specialized requirement for global deployments, highest complexity implementation
8. **Workflow Orchestration** - Builds on dependencies, provides highest-level abstraction for complex workflows

**Integration Testing for Extensions:**

Extensions require comprehensive integration testing beyond the unit testing approaches outlined in the main testing strategy:

1. **Multi-Node Testing**: Test extension behavior across multiple coordinator and worker nodes to verify distributed behavior
2. **Failure Injection**: Systematically test network partitions, node failures, and resource exhaustion scenarios  
3. **Load Testing**: Validate performance characteristics under production-level job submission and execution rates
4. **Cross-Version Testing**: Ensure extensions maintain backward compatibility with jobs and workflows created by older versions
5. **End-to-End Workflows**: Test complete business workflows that exercise multiple extension features in combination

**Milestone Checkpoints for Extensions:**

Each extension category represents a significant development milestone with specific verification criteria:

- **Advanced Scheduling Milestone**: Successfully execute a workflow with job dependencies, resource constraints, and conditional steps
- **Operational Features Milestone**: Deploy monitoring dashboards, execute administrative operations through web interface, demonstrate audit trail completeness  
- **Scalability Milestone**: Demonstrate job execution across multiple shards or datacenters with consistent behavior and partition tolerance

These extensions transform the distributed job scheduler from a reliable task execution system into a comprehensive enterprise workflow platform. While the three core milestones provide a solid foundation for most use cases, these extensions enable the scheduler to handle the complex requirements of large-scale production deployments across diverse organizational and technical environments.


## Glossary

> **Milestone(s):** This section provides essential terminology definitions that apply across all three milestones - cron scheduling terminology for Milestone 1, priority queue and deduplication concepts for Milestone 2, and distributed systems coordination terms for Milestone 3.

This glossary defines the key terminology used throughout the distributed job scheduler design document. Understanding these terms is crucial for implementing the system correctly and communicating effectively about distributed scheduling concepts. The terms are organized by domain area, starting with fundamental distributed systems concepts, then moving to scheduling-specific terminology, and finally covering operational and debugging concepts.

### Distributed Systems Core Concepts

**Exactly-once execution** - A guarantee that each job runs precisely one time, never skipped and never duplicated. This is the strongest consistency guarantee in distributed scheduling but requires careful coordination mechanisms. In practice, most systems achieve at-least-once execution with idempotent job design rather than true exactly-once semantics due to the complexity of distributed consensus.

**At-least-once execution** - A weaker guarantee where jobs may run multiple times but are never lost or skipped. This is more practical to implement in distributed systems because it only requires durable job queuing and retry logic, not distributed coordination to prevent duplicates. Jobs must be designed to handle duplicate execution gracefully.

**Leader election** - The process of selecting a single coordinator node from multiple candidates in a distributed system. The leader is responsible for making decisions that require global coordination, such as job assignment and worker failure detection. Leader election prevents split-brain scenarios where multiple nodes attempt to coordinate simultaneously.

**Split-brain** - A failure condition where network partitions cause multiple coordinator nodes to believe they are the leader simultaneously. This can result in duplicate job execution, conflicting worker assignments, and data corruption. Prevention requires consensus protocols and fencing mechanisms to ensure only one active leader.

**Fencing token** - A unique, monotonically increasing identifier that prevents stale operations from interfering with current system state. When a worker claims a job, it receives a fencing token. Later operations must present this token to prove they are authorized and current. Outdated tokens are rejected, preventing race conditions during worker failures.

**Network partition** - A failure mode where network connectivity is lost between subsets of nodes, effectively splitting the cluster into isolated groups. Each partition may continue operating independently, potentially causing inconsistent state. Partition tolerance requires careful design of consensus algorithms and data consistency mechanisms.

**Consensus algorithms** - Distributed protocols that enable multiple nodes to agree on a single value or decision despite failures and network issues. Examples include Raft, Paxos, and Byzantine Fault Tolerance. Consensus is essential for leader election, cluster membership, and ensuring consistent job scheduling decisions across coordinators.

**Distributed locking** - Mechanisms that ensure only one node can access a shared resource at a time, even across network boundaries. In job scheduling, distributed locks prevent multiple workers from claiming the same job simultaneously. Implementation typically uses external coordination services like etcd or Redis with lease-based timeouts.

### Job Scheduling and Cron Concepts

**Cron expression** - A time-based job scheduler pattern using five or six fields (minute, hour, day-of-month, month, day-of-week, optionally seconds) to specify when jobs should execute. Each field can contain specific values, ranges, lists, step values, or wildcards. For example, `0 */2 * * *` means "every two hours at minute 0".

**Field constraint** - Individual time component filters within a cron expression that determine when execution is allowed. For minute field `10,20,30`, the constraint allows execution only when the current minute is exactly 10, 20, or 30. All field constraints must be satisfied simultaneously for execution to occur.

**Next execution time** - The future timestamp when a cron job should run next, calculated by finding the earliest time after the current moment that satisfies all field constraints. This requires calendar arithmetic to handle month boundaries, leap years, and timezone transitions correctly.

**Pattern matching** - The process of checking whether a current timestamp satisfies all field constraints in a cron expression. This involves expanding ranges and step values into explicit lists, then testing if each time component (minute, hour, etc.) appears in the corresponding constraint list.

**Range expansion** - Converting cron syntax shortcuts like `1-5` (range) or `*/15` (step values) into explicit lists of valid values. Range `1-5` becomes `[1, 2, 3, 4, 5]`, while `*/15` in the minute field becomes `[0, 15, 30, 45]`. This simplifies pattern matching logic.

**Calendar arithmetic** - Mathematical operations involving dates and time periods that must account for irregular calendar rules like varying month lengths, leap years, and daylight saving time transitions. Simple timestamp addition fails across month boundaries; proper date libraries handle these complexities.

**Timezone normalization** - Converting all timestamps to UTC (Coordinated Universal Time) for consistent storage and comparison, then converting back to local timezone for display or cron evaluation. This prevents scheduling errors when coordinators and workers operate in different timezones.

**Daylight saving time (DST)** - Seasonal time adjustments that create temporal anomalies twice yearly. During "spring forward", 2:30 AM doesn't exist. During "fall back", 1:30 AM occurs twice. Cron scheduling must handle these transitions by skipping or duplicating execution appropriately.

**DST transition** - The specific moments when clocks change between standard and daylight saving time. Jobs scheduled during transition periods may be skipped (spring forward) or run twice (fall back) unless the scheduler implements special handling for these edge cases.

### Priority Queue and Job Management

**Priority queue** - A data structure that serves the highest-priority element first, regardless of insertion order. In job scheduling, higher-priority jobs are dequeued before lower-priority ones. Implementation typically uses a min-heap or max-heap depending on whether lower or higher numbers indicate priority.

**Deduplication** - Prevention of duplicate job submissions using idempotency keys or content hashes. When a job is submitted with the same deduplication identifier as an existing job, the system returns the existing job instead of creating a duplicate. This prevents accidental double-execution from client retry logic.

**Idempotency key** - A client-provided unique identifier that ensures multiple submissions of the same job create only one execution. The key should be deterministic based on job content and intent. For example, `"user-123-daily-report-2024-01-15"` ensures only one daily report per user per day.

**Content hash** - A deterministic hash (like SHA-256) computed from normalized job payload content. Used for deduplication when no explicit idempotency key is provided. The hash must be computed from a canonical representation to ensure identical jobs produce identical hashes regardless of field ordering.

**Delayed execution** - Jobs that exist in the system but remain invisible and ineligible for worker assignment until a specified future time. Implementation uses the visibility timeout pattern where jobs are stored with a "not_before" timestamp and promoted to the active queue when that time arrives.

**Visibility timeout pattern** - A queuing mechanism where messages exist but remain invisible to consumers until a timeout expires. For job scheduling, this enables delayed execution by setting visibility timeout to the scheduled execution time. Jobs become visible and claimable only when ready to run.

**Atomic operations** - Database or data structure operations that complete entirely or not at all, with no partial states visible to concurrent operations. Critical for job claiming to prevent race conditions where multiple workers attempt to claim the same job simultaneously.

**Priority inversion** - A condition where high-priority jobs are blocked by lower-priority work, either through resource contention or implementation bugs. In job queues, this can occur if priority ordering is not strictly maintained or if workers become blocked on low-priority long-running tasks.

### Worker Coordination and Fault Tolerance

**Worker coordination** - The process of managing job distribution across multiple worker nodes, including worker registration, capability matching, load balancing, and failure detection. Coordination ensures work is distributed efficiently while maintaining fault tolerance through redundancy.

**Heartbeat** - A periodic liveness signal sent from workers to coordinators indicating the worker is healthy and available for job assignment. Missed heartbeats trigger failure detection. Heartbeat messages typically include current job count, resource utilization, and capability information.

**Graceful shutdown** - A controlled worker termination process that completes currently executing jobs before stopping, rather than abruptly terminating and losing work. This prevents job loss during planned maintenance or scaling operations.

**Capability matching** - Ensuring workers can handle specific job types by comparing required job capabilities against worker-provided capabilities. For example, jobs requiring GPU processing are only assigned to workers advertising GPU capability. This prevents job failures from resource mismatches.

**Fault tolerance** - The system's ability to continue operating correctly despite node failures, network issues, or other faults. Achieved through redundancy, replication, failure detection, and recovery mechanisms. In job scheduling, fault tolerance ensures jobs complete even when individual workers fail.

**Failure detection** - Mechanisms for identifying when system components have stopped functioning correctly. Common approaches include heartbeat timeouts, health check failures, and monitoring error rates. Accurate failure detection is crucial for triggering recovery procedures promptly.

**Job recovery** - The process of reassigning jobs from failed workers to healthy workers. Recovery must handle jobs in various states: claimed but not started, currently executing, and completed but not reported. Proper recovery prevents job loss while avoiding duplicate execution.

**Message amplification** - A failure mode where message retries create exponentially increasing load, potentially overwhelming the system. Can occur when multiple components retry simultaneously or when retry delays are not properly randomized. Mitigation requires exponential backoff with jitter.

**Thundering herd** - A scenario where many workers simultaneously attempt to claim jobs or reconnect after a shared dependency (like the coordinator) recovers from failure. This can overwhelm the recovering component. Prevention uses randomized delays and gradual reconnection.

### Error Handling and Reliability

**Exponential backoff** - A retry strategy where the delay between attempts increases exponentially (e.g., 1s, 2s, 4s, 8s) to reduce load on failing systems and increase the chance of recovery. Often combined with jitter (random variation) to prevent synchronized retry storms across multiple clients.

**Circuit breaker** - A reliability pattern that prevents cascade failures by temporarily stopping calls to a failing service. The circuit breaker monitors error rates and "opens" (blocks requests) when failures exceed a threshold, allowing the downstream service time to recover.

**Dead letter queue** - A repository for operations that failed all retry attempts and require manual intervention. Dead letter entries include the original operation, failure history, and diagnostic context. This prevents permanent job loss while allowing operators to investigate and potentially resubmit failed work.

**Retry amplification** - A failure mode where multiple system layers retry the same operation simultaneously, creating exponential load increase. For example, if both the client and server retry a failed job submission, the total retry attempts become multiplicative rather than additive.

**Jitter** - Random variation added to retry delays to prevent synchronized behavior across multiple clients. Without jitter, all clients retry at exactly the same intervals, creating periodic load spikes. Random jitter spreads the load over time for better system stability.

**Idempotency** - A property where executing the same operation multiple times produces the same result as executing it once. Essential for reliable distributed systems because network failures can make it unclear whether operations succeeded, requiring safe retry mechanisms.

**Graduated failure detection** - An escalating response strategy where the system's reaction intensifies based on failure duration and severity. Short transient failures might trigger simple retries, while sustained failures escalate to circuit breaker activation and alerting.

### Testing and Development

**Unit testing** - Testing individual components (functions, classes, modules) in isolation using mocked dependencies. For the job scheduler, this includes testing cron parsing logic, priority queue operations, and worker state transitions independently of external systems.

**Integration testing** - Testing component interactions under realistic conditions with actual dependencies like Redis, etcd, or databases. This verifies that components work together correctly and can handle real network delays, connection failures, and concurrent access patterns.

**End-to-end testing** - Complete system validation that tests the entire job execution flow from submission through completion. This includes submitting jobs, verifying they execute on workers, handling failures, and confirming final state matches expectations.

**Failure injection** - Systematic introduction of failures into the system during testing to verify resilience mechanisms work correctly. Examples include network partitions, process crashes, disk failures, and resource exhaustion. Also called chaos engineering or fault injection.

**Race conditions** - Timing-dependent bugs in concurrent code where the outcome depends on the relative timing of operations across threads or processes. Common in distributed systems due to network delays. Prevention requires careful synchronization and atomic operations.

**Deterministic testing** - Creating predictable test behavior by controlling sources of randomness like timestamps, network delays, and thread scheduling. This makes tests repeatable and debuggable by eliminating timing-dependent failures.

**Test harness** - Infrastructure providing consistent test environments including mock services, controlled time, isolated databases, and cleanup mechanisms. The harness handles setup and teardown, allowing tests to focus on business logic verification.

**Milestone checkpoints** - Structured validation points at each development stage that verify expected functionality is working before proceeding. Checkpoints include automated tests, manual verification steps, and integration smoke tests.

### Observability and Debugging

**Structured logging** - Logging with consistent data formats (typically JSON) that enable programmatic analysis, filtering, and correlation. Each log entry includes standard fields like timestamp, component, correlation ID, and operation name, plus operation-specific details.

**Correlation ID** - A unique identifier that links related operations across distributed system boundaries. When a job is submitted, the same correlation ID appears in logs from the coordinator, worker, and queue components, enabling end-to-end request tracing.

**Distributed tracing** - A technique for tracking request flow across multiple service boundaries by propagating trace context and recording timing spans. Each operation adds a span with start time, duration, and metadata, creating a complete timeline of distributed request execution.

**Metrics collection** - Systematic gathering of quantitative system performance data like job completion rates, queue depths, worker utilization, and error rates. Metrics are typically time-series data stored in systems like Prometheus for alerting and dashboards.

**Log aggregation** - Centralized collection and indexing of log entries from multiple distributed components. Aggregation enables searching across all system logs simultaneously and correlating events that span multiple services.

**Failure pattern analysis** - Systematic identification of common failure sequences in distributed systems by analyzing log patterns, metrics anomalies, and error correlations. Helps identify root causes and prevent recurring issues through improved error handling.

**Observability** - The ability to understand system internal state through external outputs (logs, metrics, traces). High observability enables rapid troubleshooting and performance optimization by providing visibility into system behavior during both normal and failure conditions.

**Alert threshold** - Metric value boundaries that trigger automated notifications when crossed. Thresholds must balance sensitivity (catching real issues quickly) with specificity (avoiding false alarms). Common examples include error rate percentages and response time percentiles.

### Future Extensions and Advanced Concepts

**Job dependencies** - Prerequisites that must be satisfied before a job can execute, such as completion of other jobs or availability of required data. Dependencies create directed acyclic graphs (DAGs) of job relationships and require topological sorting for execution order.

**Resource-aware scheduling** - Job assignment that considers memory, CPU, storage, and other resource requirements against worker capacity. This prevents resource exhaustion and improves overall system efficiency by matching job needs with worker capabilities.

**Workflow orchestration** - Coordinated execution of multi-job workflows with conditional logic, parallel branches, loops, and error handling. More complex than simple job dependencies, workflows require state machines and execution engines to manage complex business processes.

**Horizontal sharding** - Distributing jobs across multiple independent scheduler clusters based on characteristics like tenant ID, job type, or hash key. Sharding improves scalability by reducing coordination overhead and enabling independent scaling of different workload types.

**Multi-datacenter deployment** - Operating the scheduler across multiple geographic regions for disaster recovery and latency optimization. Requires careful handling of clock skew, network partitions, and data consistency across wide-area networks.

**Audit logging** - Detailed recording of administrative actions, configuration changes, and security events for compliance and security purposes. Audit logs are typically immutable and include who performed actions, when, what changed, and why.

This comprehensive glossary provides the foundation for understanding distributed job scheduler concepts and implementing the system correctly. The terminology spans from fundamental distributed systems concepts through specific scheduling algorithms to operational concerns, supporting both learning and professional communication about the system.

### Implementation Guidance

The terminology in this glossary should be used consistently throughout code comments, documentation, and team communication. Here are practical guidelines for applying these terms in implementation:

**Code Documentation Standards**

When writing code comments and documentation, use these terms precisely and consistently. For example, always refer to "exactly-once execution" rather than mixing terms like "exactly-once delivery" or "once-only processing". This consistency helps team members communicate clearly and reduces misunderstandings during code reviews and debugging sessions.

**Variable and Function Naming**

Incorporate terminology into identifier names to make code self-documenting. Use `idempotencyKey` rather than generic names like `dedupId`. Name functions like `detectWorkerFailure()` and `performLeaderElection()` to clearly indicate their purpose using standard distributed systems terminology.

**Error Messages and Logging**

Include proper terminology in error messages and log entries to aid debugging. Instead of "job failed", use "job exceeded maximum retry attempts and moved to dead letter queue". This provides operators with precise information about system state and required actions.

**Team Communication Guidelines**

| Term Category | Usage Guideline | Example |
|---------------|-----------------|---------|
| Consistency Levels | Always specify exactly-once vs at-least-once when discussing guarantees | "This implementation provides at-least-once execution with idempotent job design" |
| Failure Modes | Use precise terms for different types of failures | "We're seeing split-brain behavior, not general leader election failure" |
| Timing Concepts | Distinguish between different time-related operations | "The DST transition caused cron schedule skipping, not general timezone normalization issues" |
| System States | Use exact state names from data model | "Worker is in UNAVAILABLE state, not just 'down' or 'broken'" |

**Documentation Cross-References**

When writing design documents or architectural decision records, reference this glossary to ensure consistent terminology. Include brief definitions for complex terms when first introduced in new contexts, with references back to the full glossary definitions for complete understanding.

This implementation guidance ensures the terminology serves its purpose of enabling clear, precise communication about the distributed job scheduler system throughout the development lifecycle.

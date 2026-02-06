# Background Job Processor: Design Document


## Overview

This document designs a distributed background job processing system similar to Celery/Sidekiq, enabling asynchronous execution of tasks with scheduling, retries, and monitoring. The key architectural challenge is ensuring reliable, ordered task execution across multiple worker processes while maintaining fault tolerance, performance, and observability in a distributed environment.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


> **Milestone(s):** This section provides foundational understanding for the entire project, establishing the core problems and mental models that all subsequent milestones will address.

## Context and Problem Statement

At the heart of modern web applications lies a fundamental tension: user-facing requests must complete quickly (typically under 100-500 milliseconds) to provide responsive interfaces, while many necessary business operations are inherently slow or unpredictable. These operations—sending emails, processing uploaded images, generating reports, updating search indexes, synchronizing data with external APIs—can take seconds, minutes, or even hours to complete. Attempting to execute such work synchronously within request-response cycles creates intolerable user delays, timeout errors, and brittle systems that fail under load.

This section establishes why we need a dedicated background job processing system, explores why naive approaches fail at scale, and examines the architectural patterns that successful systems like Celery and Sidekiq employ. We begin with an intuitive analogy to build a mental model, then formalize the technical problem, and finally survey existing approaches to understand the trade-offs our design must navigate.

### Mental Model: The Restaurant Kitchen Ticket System

Imagine a busy restaurant during dinner service. Customers (web requests) arrive and place orders (tasks) with the server (web application). Some orders are simple and quick—a glass of water (simple database query). Others are complex and time-consuming—a well-done steak with customized sides, sauce modifications, and allergy considerations (resource-intensive background work).

**Naive Approach (Synchronous Kitchen):** The server takes the order, walks to the kitchen, stands by the grill, prepares the steak themselves, plates it, and then returns to the table 25 minutes later. During this time, the server cannot attend to other customers, the kitchen is cluttered with servers trying to cook, and customers wait endlessly. This is analogous to a web server process blocking on a long-running task within the HTTP request handler—terrible utilization of resources and terrible customer experience.

**Background Job System (Kitchen Ticket System):** Instead, the server writes the order on a ticket (job) and places it in a queue (the ticket rail). The kitchen staff (worker processes) continuously monitor the ticket rail, grab the next ticket (dequeue a job), prepare the order (execute the task), and place the finished dish in the pickup area (store results). Meanwhile, the server is free to attend to other customers immediately after submitting the ticket. The ticket system provides:

- **Decoupling:** Servers and cooks work independently at their own paces.
- **Prioritization:** Urgent orders (fire tables, VIP customers) can jump the queue.
- **Specialization:** Different stations (grill, sauté, pastry) handle specific ticket types.
- **Resilience:** If a cook gets sick (worker crash), another can pick up their tickets.
- **Visibility:** The chef can monitor ticket volume (dashboard) and reassign staff dynamically.

This mental model maps directly to our technical system:

| Restaurant Concept | Technical Equivalent |
|-------------------|----------------------|
| Customer Order | Task/Job to be performed |
| Server | Web application/API server |
| Ticket | Job definition (serialized payload) |
| Ticket Rail | Job queue (Redis list/stream) |
| Kitchen Staff | Worker processes |
| Cook Stations | Queue names/priorities |
| Head Chef | Monitoring dashboard |
| "86" Board (out of stock) | Dead letter queue |

The critical insight is that **asynchronous processing via queues decouples the act of requesting work from the act of executing it**, enabling each component to operate at its optimal scale and failure independently.

### The Asynchronous Task Execution Problem

Formally, the asynchronous task execution problem consists of four interconnected challenges that must be solved simultaneously:

1. **Reliable Work Dispatch:** How do producers (web servers) reliably submit work units (jobs) such that no work is lost, even during system failures, while maintaining acceptable latency for the submission operation itself?

2. **Scalable Work Consumption:** How do consumers (workers) efficiently retrieve and execute jobs with configurable concurrency, ensuring公平分配 across workers, handling varying workloads, and preventing single workers from becoming bottlenecks?

3. **Fault Tolerance and Recovery:** When jobs fail (due to transient errors, resource constraints, or bugs), how does the system automatically retry them with intelligent backoff, and when retries are exhausted, how does it preserve failed jobs for inspection without blocking the queue?

4. **Observability and Control:** How do operators monitor queue health, job progress, and system performance in real-time, and how can they intervene (pause, retry, cancel) when necessary?

Naive approaches fail systematically as scale increases:

| Naive Approach | Failure Mode | Consequence |
|----------------|--------------|-------------|
| **In-process threading** | One failing job crashes the entire worker process. No persistence across restarts. | Lost jobs, requires manual intervention. |
| **Database-backed queues** (simple `jobs` table) | Polling causes high database load. Transactional locks create contention. | Database becomes bottleneck, poor performance at scale. |
| **Direct RPC/HTTP calls to workers** | Tight coupling between producers and consumers. Worker failure causes producer failures. | Cascading failures, requires complex circuit breakers. |
| **Cron jobs for recurring tasks** | No retry logic, poor error handling, duplicate execution if jobs run longer than interval. | Missed jobs, duplicated work, no visibility into failures. |
| **Local filesystem queues** | Not accessible across multiple servers. Disk corruption loses all jobs. | Cannot scale horizontally, single point of failure. |

The core technical requirements that emerge are:

- **Durability:** Jobs must survive process crashes and system restarts.
- **Atomicity:** Job state transitions (enqueued → processing → completed) must occur atomically to prevent double-processing or lost jobs.
- **Scalability:** The system must handle increasing job volumes by adding workers without reconfiguration.
- **Isolation:** A misbehaving job should not affect other jobs or the system itself.
- **Observability:** Every job's status, timing, and errors must be trackable.
- **Configurability:** Retry policies, priorities, and scheduling must be adjustable per job type.

Our design must address these requirements while remaining simple enough to implement, understand, and operate.

### Existing Approaches and Trade-offs

Several established solutions and patterns exist, each with distinct architectural choices and trade-offs. Understanding these helps inform our design decisions.

**1. Dedicated Job Queue Systems (Celery, Sidekiq, RQ)**

These are full-featured frameworks that implement the complete pattern we're building.

| System | Language | Broker | Architecture | Key Trade-offs |
|--------|----------|---------|--------------|----------------|
| **Celery** | Python | Redis, RabbitMQ, etc. | Distributed task queue with worker processes, beat scheduler | Extremely flexible but complex configuration. "Batteries included" leads to large dependency footprint. |
| **Sidekiq** | Ruby | Redis | Multi-threaded workers within processes, emphasis on simplicity | Ruby-centric, requires Redis. Excellent performance but less language-agnostic. |
| **RQ (Redis Queue)** | Python | Redis | Simple, minimal design. Single-threaded workers. | Easy to understand but limited features. No native scheduling, simpler retry logic. |

**Common Architectural Patterns:**

- **Broker-Centric Design:** All components communicate through a central broker (Redis, RabbitMQ). This provides a clear decoupling point but makes the broker a potential single point of failure and performance bottleneck.
- **Process-per-Worker vs Thread-per-Worker:** Celery uses prefork process pools for isolation; Sidekiq uses threads within processes for memory efficiency. This is a fundamental trade-off between isolation and resource usage.
- **Polling vs Event-Driven Workers:** Most workers poll the broker for jobs (BRPOP, basic.get). Advanced systems use event-driven protocols (AMQP's basic.consume) for lower latency but increased complexity.
- **Embedded vs External Monitoring:** Some systems include web dashboards; others rely on external monitoring (Prometheus, Grafana).

**2. Database as Queue**

Using a relational database table as a job queue is a common alternative, with two primary patterns:

| Pattern | Implementation | Pros | Cons |
|---------|---------------|------|------|
| **Poll-based** | Workers SELECT FOR UPDATE SKIP LOCKED on a `jobs` table. | Leverages existing infrastructure, ACID guarantees. | High database load, scaling challenges, table bloat. |
| **Notify-based** | PostgreSQL LISTEN/NOTIFY with SKIP LOCKED. | Reduced polling overhead. | Database connection overhead, still consumes DB resources. |

> **Design Insight:** While tempting for simplicity, database-backed queues typically become the primary scalability bottleneck in growing systems. They couple job processing throughput to database capacity, which is expensive and difficult to scale horizontally compared to purpose-built brokers like Redis.

**3. Message Brokers (RabbitMQ, Kafka, AWS SQS)**

These are general-purpose message brokers that can be used as job queues.

| Broker | Model | Job Queue Suitability |
|--------|-------|------------------------|
| **RabbitMQ** | AMQP, exchanges/queues | Excellent for job queues: persistence, acknowledgments, dead letter exchanges. Requires more operational knowledge. |
| **Kafka** | Log-based, partitioned streams | Better for event streaming. Job processing possible but lacks built-in per-message retry. Overkill for simple tasks. |
| **AWS SQS** | Cloud queue service | Fully managed, simple API. Limited features (max 15-minute visibility timeout, no native scheduling). Vendor lock-in. |

**4. Cloud-Native Services (AWS Lambda, Google Cloud Tasks)**

Serverless functions and managed task services represent a different architectural approach.

| Service | Model | Trade-offs |
|---------|-------|------------|
| **AWS Lambda** | Event-driven functions | No infrastructure to manage, automatic scaling. Cold starts, limited execution time (15 min), vendor lock-in, debugging complexity. |
| **Google Cloud Tasks** | Managed task queue | Fully managed, integrates with App Engine/Cloud Run. Vendor-specific, less control over underlying infrastructure. |

**Key Architectural Decisions Our Design Must Address:**

Based on these existing approaches, we identify several pivotal decisions:

> **Decision: Broker Selection**
> - **Context:** We need a durable, high-performance data store for job queues that supports multiple data structures (lists, sorted sets, hashes) and provides atomic operations.
> - **Options Considered:** 
>   1. **Relational Database (PostgreSQL):** Familiar, ACID guarantees, but poor performance under high throughput.
>   2. **Redis:** In-memory data structures with persistence options, extremely fast, supports needed operations (LPUSH, BRPOP, ZADD).
>   3. **RabbitMQ:** Purpose-built message broker with advanced routing, but additional operational complexity.
> - **Decision:** Use Redis as the primary broker.
> - **Rationale:** Redis provides the necessary data structures with atomic operations, offers excellent performance, is widely available in cloud environments, and has a simple operational model. The project prerequisites already include Redis basics.
> - **Consequences:** System performance is tied to Redis availability and memory capacity. Requires Redis persistence configuration for durability.

> **Decision: Worker Concurrency Model**
> - **Context:** Workers must execute multiple jobs concurrently to utilize multi-core systems efficiently while maintaining fault isolation.
> - **Options Considered:**
>   1. **Process Pool (prefork):** Each job runs in a separate process. Maximum isolation, but higher memory overhead.
>   2. **Thread Pool:** Each job runs in a thread within a single process. Lower memory overhead, but one misbehaving job can affect others.
>   3. **Green Threads/Async:** Single-threaded event loop with cooperative multitasking. Efficient for I/O-bound tasks but requires all job code to be async.
> - **Decision:** Use a process pool for maximum isolation.
> - **Rationale:** Since jobs can be arbitrary Python code (possibly buggy or resource-intensive), process isolation prevents one job from crashing the entire worker. This matches Celery's approach and provides the cleanest failure boundaries.
> - **Consequences:** Higher memory usage, inter-process communication overhead for job state updates.

> **Decision: Job Persistence Format**
> - **Context:** Jobs must be serialized for storage in Redis and deserialized by workers.
> - **Options Considered:**
>   1. **JSON:** Human-readable, language-agnostic, but verbose and limited in data types.
>   2. **MessagePack:** Binary, compact, preserves some type information, faster than JSON.
>   3. **Pickle:** Python-specific, can serialize almost anything, but security risks with untrusted data.
> - **Decision:** Use JSON for primary serialization with type-preserving extensions.
> - **Rationale:** JSON is universally supported, debuggable (can inspect in Redis CLI), and safe. We'll add custom encoding for Python-specific types (datetime, Decimal) as needed. This balances simplicity with practicality.
> - **Consequences:** Slightly larger payload size than binary formats, requires custom encoders/decoders for special types.

| Comparison: Serialization Formats |
|-----------------------------------|
| Format | Pros | Cons | Our Choice? |
| JSON | Human-readable, standard, safe | No binary data support, verbose | ✅ Primary choice |
| MessagePack | Compact, fast, binary-safe | Less human-readable, extra dependency | ❌ Optional optimization |
| Pickle | Handles Python objects natively | Security risks, Python-only | ❌ Too dangerous |

These decisions shape the architecture we'll detail in subsequent sections. The system we're designing occupies a middle ground: more feature-complete than RQ but simpler than Celery, with explicit trade-offs chosen for educational clarity and practical utility.

### Implementation Guidance

> **Note:** This section provides preliminary implementation guidance to set up the foundational concepts. Detailed code for each component will appear in subsequent sections.

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Broker** | Redis (single instance) | Redis Cluster for horizontal scaling |
| **Serialization** | JSON with `json` module | MessagePack with `msgpack` package |
| **Process Management** | `multiprocessing.Pool` | `concurrent.futures.ProcessPoolExecutor` |
| **HTTP Dashboard** | Flask + Jinja2 templates | FastAPI + WebSocket for real-time updates |
| **Testing** | `unittest` with `fakeredis` | `pytest` with Redis fixtures |
| **Packaging** | Single module | Package with `setup.py`/`pyproject.toml` |

#### B. Recommended File/Module Structure

```
background-job-processor/
├── pyproject.toml              # Project dependencies and metadata
├── README.md
├── src/
│   └── jobqueue/              # Main package
│       ├── __init__.py
│       ├── exceptions.py      # Custom exceptions (JobError, QueueFullError, etc.)
│       ├── models.py          # Core data models (Job, Queue, Worker, etc.)
│       ├── serialization.py   # JSON encoding/decoding with custom extensions
│       ├── redis_client.py    # Redis connection pooling and wrapper
│       ├── queue_manager.py   # Milestone 1: Enqueue, priority, queue management
│       ├── worker.py          # Milestone 2: Worker main loop, job execution
│       ├── retry_manager.py   # Milestone 3: Exponential backoff, dead letter queue
│       ├── scheduler.py       # Milestone 4: Delayed jobs, cron scheduling
│       ├── monitor.py         # Milestone 5: Metrics collection, dashboard backend
│       └── web/
│           ├── __init__.py
│           ├── app.py         # Flask/FastAPI application
│           ├── templates/     # Jinja2 templates for dashboard
│           └── static/        # CSS, JavaScript
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_queue_manager.py
│   └── ...
└── scripts/
    ├── worker-cli.py          # CLI to start workers
    └── enqueue-job.py         # CLI to enqueue jobs manually
```

#### C. Infrastructure Starter Code: Redis Client Wrapper

```python
# src/jobqueue/redis_client.py
"""
Redis connection management with connection pooling and error handling.
"""
import redis
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class RedisClient:
    """Wrapper around Redis client with connection pooling and retry logic."""
    
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None
    
    def __init__(self, url: str = "redis://localhost:6379/0", **kwargs):
        """
        Initialize Redis connection.
        
        Args:
            url: Redis connection URL (redis://[:password]@host:port/db)
            **kwargs: Additional arguments passed to redis.Redis()
        """
        if RedisClient._instance is not None:
            raise RuntimeError("Use RedisClient.get_instance() to get singleton")
        
        self.url = url
        self.connection_kwargs = kwargs
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection with retry logic."""
        try:
            self._client = redis.Redis.from_url(
                self.url,
                decode_responses=True,  # Automatically decode bytes to str
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                **self.connection_kwargs
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.url}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis at {self.url}: {e}")
            raise
    
    @classmethod
    def get_instance(cls, url: Optional[str] = None, **kwargs) -> 'RedisClient':
        """Get singleton RedisClient instance."""
        if cls._instance is None:
            if url is None:
                url = "redis://localhost:6379/0"
            cls._instance = cls(url, **kwargs)
        return cls._instance
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get the underlying Redis client instance."""
        if cls._instance is None:
            cls.get_instance()
        if cls._instance._client is None:
            cls._instance._connect()
        return cls._instance._client
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """
        Execute a Redis command with error handling.
        
        Args:
            command: Redis command name (e.g., 'lpush', 'zadd')
            *args: Command arguments
            **kwargs: Additional options
            
        Returns:
            Command result
            
        Raises:
            redis.RedisError: If Redis operation fails
        """
        client = self.get_client()
        try:
            method = getattr(client, command)
            return method(*args, **kwargs)
        except redis.RedisError as e:
            logger.error(f"Redis command {command} failed: {e}")
            raise
    
    def pipeline(self):
        """Return a Redis pipeline for atomic operations."""
        return self.get_client().pipeline()

# Convenience function for easy access
def get_redis():
    """Get the Redis client instance."""
    return RedisClient.get_client()
```

#### D. Core Logic Skeleton: Job Model

```python
# src/jobqueue/models.py
"""
Core data models for the job queue system.
"""
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Status of a job through its lifecycle."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTER = "dead_letter"

class Job:
    """
    Represents a unit of work to be processed asynchronously.
    
    This is the core data structure that gets serialized and stored in Redis.
    """
    
    def __init__(
        self,
        job_type: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        job_id: Optional[str] = None,
        queue: str = "default",
        priority: int = 0,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        created_at: Optional[datetime] = None,
        **extra_metadata
    ):
        """
        Initialize a job.
        
        Args:
            job_type: String identifier for the handler function/class
            args: Positional arguments for the job handler
            kwargs: Keyword arguments for the job handler
            job_id: Unique identifier (auto-generated if None)
            queue: Target queue name
            priority: Higher priority jobs are processed first
            max_retries: Maximum number of retry attempts before giving up
            timeout_seconds: Maximum execution time in seconds
            created_at: Job creation timestamp (auto-generated if None)
            **extra_metadata: Additional metadata stored with the job
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.job_type = job_type
        self.args = args or []
        self.kwargs = kwargs or {}
        self.queue = queue
        self.priority = priority
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.created_at = created_at or datetime.utcnow()
        self.metadata = extra_metadata
        
        # Runtime state (not serialized)
        self.status = JobStatus.PENDING
        self.attempts = 0
        self.errors: List[Dict[str, Any]] = []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert job to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        data = {
            'job_id': self.job_id,
            'job_type': self.job_type,
            'args': self.args,
            'kwargs': self.kwargs,
            'queue': self.queue,
            'priority': self.priority,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata,
            'attempts': self.attempts,
            'errors': self.errors,
            'status': self.status.value,
        }
        # Only include runtime fields if they exist
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.result is not None:
            data['result'] = self.result
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """
        Reconstruct a job from dictionary representation.
        
        Args:
            data: Dictionary from Redis storage
            
        Returns:
            Reconstructed Job instance
        """
        # TODO 1: Extract core fields from data dict
        # TODO 2: Parse datetime strings back to datetime objects
        # TODO 3: Reconstruct job with all fields
        # TODO 4: Set runtime state fields (status, attempts, errors, etc.)
        # TODO 5: Return the reconstructed job instance
        # Hint: Use datetime.fromisoformat() for parsing
        pass
    
    def serialize(self) -> str:
        """
        Serialize job to JSON string for storage in Redis.
        
        Returns:
            JSON string representation
        """
        # TODO 1: Convert job to dictionary using to_dict()
        # TODO 2: Serialize dictionary to JSON string
        # TODO 3: Validate that serialized size is under 1MB limit
        # TODO 4: If over limit, raise PayloadTooLargeError
        # TODO 5: Return the JSON string
        pass
    
    @classmethod
    def deserialize(cls, data: str) -> 'Job':
        """
        Deserialize job from JSON string.
        
        Args:
            data: JSON string from Redis
            
        Returns:
            Deserialized Job instance
        """
        # TODO 1: Parse JSON string to dictionary
        # TODO 2: Reconstruct job using from_dict()
        # TODO 3: Return the job instance
        pass
    
    def record_error(self, error: Exception) -> None:
        """
        Record an error that occurred during job execution.
        
        Args:
            error: Exception that was raised
        """
        error_info = {
            'type': error.__class__.__name__,
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'attempt': self.attempts,
        }
        # TODO: Optionally add stack trace in debug mode
        self.errors.append(error_info)
    
    def should_retry(self) -> bool:
        """
        Determine if job should be retried based on attempts and max_retries.
        
        Returns:
            True if job should be retried, False otherwise
        """
        # TODO 1: Check if attempts < max_retries
        # TODO 2: Return True if should retry, False otherwise
        pass
```

#### E. Language-Specific Hints: Python

- **Use `datetime.utcnow()`** instead of `datetime.now()` for timestamps to avoid timezone issues across servers.
- **JSON serialization of datetime:** Use ISO format strings (`datetime.isoformat()`), which are both human-readable and parseable.
- **Redis connections:** Always use `decode_responses=True` when creating Redis client to avoid byte-string handling.
- **Connection pooling:** Let `redis.Redis` handle connection pooling internally; create one client instance and reuse it.
- **Process vs Thread:** Use `multiprocessing` for worker pools to isolate job crashes, but be aware that objects passed to worker processes must be picklable.
- **Graceful shutdown:** Catch `KeyboardInterrupt` and `SystemExit` in worker main loop, and use `signal.signal()` to handle SIGTERM.

#### F. Milestone Checkpoint: After Section Completion

After understanding this section (but before implementing Milestone 1), you should be able to:

1. **Explain in your own words** why background job processing is necessary and the failures of naive approaches.
2. **Draw the restaurant ticket system analogy** and map each component to the technical system.
3. **Articulate the trade-offs** between Redis, database-backed, and message broker queues.
4. **Run the Redis starter code** successfully:

```bash
# Start Redis (if not already running)
docker run -d -p 6379:6379 redis:7-alpine

# Test the Redis connection
python -c "from src.jobqueue.redis_client import RedisClient; r = RedisClient.get_instance(); print(r.execute('ping'))"
# Should print: True
```

5. **Create a Job instance** and inspect its dictionary representation:

```python
from src.jobqueue.models import Job

job = Job(
    job_type="send_email",
    args=["user@example.com", "Welcome!"],
    kwargs={"template": "welcome"},
    queue="emails",
    priority=1,
    max_retries=3
)

print(job.job_id)  # Should show a UUID
print(job.to_dict())  # Should show dictionary with all fields
```

**Signs something is wrong:**
- Redis connection fails: Check if Redis is running and the URL is correct.
- Job serialization fails: Ensure all data types are JSON-serializable (no datetime objects in args/kwargs directly).
- Import errors: Verify your Python path includes the `src` directory or install the package in development mode with `pip install -e .`.

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "ModuleNotFoundError: No module named 'jobqueue'" | Python cannot find the package | Run `python -c "import sys; print(sys.path)"` to see Python path | Install with `pip install -e .` or set `PYTHONPATH` |
| Redis connection timeout | Redis not running or wrong port | Run `redis-cli ping` from terminal | Start Redis: `docker run -d -p 6379:6379 redis` |
| "datetime is not JSON serializable" | Direct datetime objects in job data | Check job args/kwargs for datetime objects | Use ISO format strings or extend JSON encoder |
| Job ID collisions | Using simple incrementing IDs | Check ID generation logic | Use UUIDs (uuid4) or ULIDs for distributed uniqueness |
| Cannot pickle local object | Passing unpicklable objects to multiprocessing | Check what's in job args/kwargs | Ensure all arguments are picklable (no lambdas, local functions) |


## Goals and Non-Goals

> **Milestone(s):** This section establishes the fundamental requirements and boundaries that apply across all milestones, providing the foundation for architectural decisions made throughout the entire system design. These goals guide implementation priorities and trade-offs from Milestone 1 through Milestone 5.

### Functional Requirements (Goals)

Think of our job processor as an **industrial factory** that must reliably transform raw materials (job requests) into finished products (completed work). The factory must operate 24/7, handle different product lines with varying priorities, recover from equipment failures, and provide supervisors with real-time visibility into operations. These functional requirements represent the core capabilities our factory must deliver to be commercially viable.

| Category | Requirement | Description | Implementation Impact |
|----------|-------------|-------------|----------------------|
| **Job Management** | Enqueue jobs with metadata | Jobs must be enqueued with JSON-serialized payloads containing job type, arguments, priority, retry limits, and custom metadata. Each job receives a unique ID (UUID/ULID) for tracking. | Foundation for all job operations; requires robust serialization and validation (Milestone 1). |
| | Multiple priority queues | Support at least 3-5 named queues with configurable priority weights. Higher-priority queues are polled more frequently by workers while maintaining fairness. | Impacts worker polling algorithm and requires weighted queue selection logic (Milestone 1). |
| | Queue inspection | APIs to check queue length, peek at pending jobs, list all queues, and retrieve job status by ID. Essential for operational visibility. | Adds monitoring endpoints and Redis query patterns (Milestones 1, 5). |
| **Processing** | FIFO job execution | Within the same priority level, jobs execute in first-in-first-out order. Priority weights determine which queue's jobs are processed next. | Guides Redis data structure selection (list vs. sorted set) and dequeue algorithms (Milestone 1). |
| | Concurrent execution | Single worker process must handle multiple jobs concurrently using thread/process pools. Configurable concurrency level (default: CPU count). | Affects worker architecture, resource isolation, and coordination (Milestone 2). |
| | Graceful shutdown | Workers must complete current jobs before exiting when receiving SIGTERM/SIGINT. No partial state or orphaned jobs after shutdown. | Requires signal handlers and job state persistence (Milestone 2). |
| **Reliability** | Automatic retries with backoff | Failed jobs automatically retry with exponential backoff (1s, 2s, 4s... up to max retries). Configurable jitter prevents thundering herd. | Requires retry scheduler, delayed execution mechanism, and attempt tracking (Milestone 3). |
| | Dead letter queue | Jobs exhausting max retries move to a dedicated dead letter queue with full error history. Can be manually retried or inspected. | Creates new storage location and requires error serialization (Milestone 3). |
| | Worker heartbeat & failure detection | Workers report liveness at regular intervals (e.g., every 30s). Stale workers (no heartbeat for 2x interval) are detected and their jobs returned to queue. | Requires periodic background tasks and stale worker cleanup (Milestone 2). |
| **Scheduling** | Delayed execution | Jobs can be scheduled for future execution at specific timestamps. Must remain invisible to workers until scheduled time arrives. | Requires time-based data structure (Redis sorted set) and scheduler process (Milestone 4). |
| | Recurring jobs (cron) | Define jobs that run on schedule using standard cron syntax (5-field: minute, hour, day, month, weekday). Unique constraints prevent duplicate execution. | Requires cron parser, schedule storage, and catch-up logic for missed runs (Milestone 4). |
| **Monitoring** | Real-time dashboard | Web dashboard showing queue depths, active workers, processing rates, error rates, and recent job history. Updates without page refresh. | Requires WebSocket/SSE, metrics aggregation, and UI components (Milestone 5). |
| | Job search & filtering | Search jobs by ID, status, queue, time range, or job type. Paginated results for large datasets. | Creates query interface over job history data (Milestone 5). |
| | Alerting | Configurable alerts when queue backlog exceeds threshold or error rate spikes above limit. Notify via webhook or email. | Requires background monitoring and notification system (Milestone 5). |
| **Operational** | Payload validation | Reject jobs with payloads >1MB at enqueue time. Validate JSON structure and required fields. | Prevents memory issues and malformed job processing (Milestone 1). |
| | Job timeout enforcement | Kill jobs exceeding their configured timeout (default: 30 minutes). Prevent worker starvation by long-running jobs. | Requires subprocess management and timeout monitoring (Milestone 2). |
| | Cross-language compatibility | Job payloads serializable/deserializable across Python, Go, and Rust clients. Use standard formats (JSON/msgpack). | Influences serialization format choice and type system design (Milestone 1). |

> **Key Insight:** The progression from Milestone 1 to 5 represents a journey from core job mechanics (enqueue/dequeue) through reliability features (retries), time-based operations (scheduling), and finally operational excellence (monitoring). Each milestone builds on the previous one—you cannot implement retries without a working worker system, nor scheduling without reliable queue infrastructure.

#### Job Lifecycle Requirements

The system must support the complete job lifecycle shown in the state machine diagram:

![Job State Machine](./diagrams/job-state-machine.svg)

1. **Creation → Pending:** Jobs enter the system via `enqueue()` and await worker pickup.
2. **Pending → Active:** Worker claims job via atomic dequeue operation.
3. **Active → Completed:** Job executes successfully, result stored.
4. **Active → Failed:** Job execution throws exception, error recorded.
5. **Failed → Retry Scheduled:** If retries remain, job scheduled for future retry with backoff.
6. **Retry Scheduled → Pending:** Scheduler moves job back to queue after backoff delay.
7. **Failed → Dead Letter:** Max retries exhausted, job moved to dead letter queue.
8. **Dead Letter → Pending (manual):** Admin manually retries job from dashboard.

Each transition must be atomic and leave the system in a consistent state, even during failures.

### Non-Functional Requirements

Consider these as the **factory's operating standards**—not what it produces, but how well it operates. A factory might produce excellent widgets but if it consumes excessive power, breaks down frequently, or requires specialized operators, it's not commercially viable. These requirements ensure our job processor operates efficiently, reliably, and maintainably in production environments.

| Category | Requirement | Metric | Rationale |
|----------|-------------|--------|-----------|
| **Performance** | Enqueue latency | ≤10ms P99 for single job enqueue | Producers shouldn't block waiting for job acceptance. |
| | Dequeue latency | ≤50ms P99 for worker to get next job | Workers should spend minimal time waiting for work. |
| | Throughput | ≥1000 jobs/sec per worker process | Handle moderate workloads on modest hardware. |
| | Memory efficiency | ≤50MB baseline per worker process | Enable running multiple workers on same machine. |
| **Reliability** | Durability | Job persistence ≥99.9% after successful enqueue | Jobs shouldn't disappear once accepted. |
| | Availability | System operational ≥99.5% of time | Allow for maintenance and occasional failures. |
| | Data consistency | No job loss during graceful shutdown | Critical for user trust in the system. |
| | Failure recovery | ≤5 minutes to recover from Redis restart | Quick recovery from infrastructure issues. |
| **Scalability** | Horizontal scaling | Linear throughput increase with added workers | Support growing workloads by adding resources. |
| | Queue isolation | One misbehaving queue doesn't block others | Fault isolation between different job types. |
| | Redis connection efficiency | ≤100 connections per worker pool | Prevent overwhelming Redis with connections. |
| **Operational** | Monitoring granularity | Metrics updated every 30 seconds | Balance real-time visibility with system load. |
| | Dashboard responsiveness | ≤2 second page load for 10k job history | Operators need quick access to system state. |
| | Configuration simplicity | ≤5 required configuration parameters | Reduce deployment friction and errors. |
| | Documentation completeness | All public APIs documented with examples | Enable adoption and reduce support burden. |
| **Security** | Input validation | All user input validated before processing | Prevent injection attacks and malformed data. |
| | Dashboard authentication | Optional basic auth for web dashboard | Protect operational interface in production. |
| | Job payload encryption | Support for encrypted payloads (optional) | Handle sensitive job data requirements. |
| | Redis security | Support Redis ACL and TLS connections | Enterprise deployment requirements. |

> **Design Principle:** These non-functional requirements create concrete constraints for architectural decisions. For example, the ≤10ms enqueue latency requirement rules out complex validation that requires external service calls, while the ≥99.9% durability requirement necessitates synchronous Redis writes (or WAL) for critical operations.

#### Performance Trade-off Decisions

The system makes deliberate trade-offs between competing non-functional requirements:

| Trade-off | Choice | Rationale | Consequence |
|-----------|--------|-----------|-------------|
| Latency vs. Durability | **Durability with fsync every 1s** | Job loss is worse than slightly higher latency for background jobs. | Enqueue operations may block during Redis persistence. |
| Memory vs. Functionality | **Keep 10k most recent job history** | Most debugging uses recent jobs; older history can be archived. | Dashboard job search limited to recent history without external storage. |
| Real-time vs. Load | **30-second metric aggregation** | Real-time updates aren't critical for background job monitoring. | Dashboard shows near-real-time but not exact current state. |
| Simplicity vs. Features | **Polling scheduler vs. event-based** | Simpler to implement and debug; acceptable for scheduled jobs. | Scheduler consumes CPU even when no jobs are due. |

### Explicit Non-Goals

Just as important as what we build is what we **deliberately choose not to build**. A factory producing automobiles shouldn't also try to refine gasoline—it should focus on its core competency and integrate with specialists. These non-goals prevent scope creep, keep the implementation focused, and acknowledge that some problems are better solved by other systems or future extensions.

| Non-Goal | Explanation | Alternative/Workaround |
|----------|-------------|------------------------|
| **Job dependency graphs** | No support for complex workflows where job B depends on job A's completion. | Chain jobs by having job A enqueue job B upon completion, or use external workflow engine. |
| **Exactly-once execution** | Cannot guarantee jobs execute exactly once; at-least-once is the best guarantee. | Design jobs to be idempotent; include idempotency keys in job metadata for deduplication. |
| **Job migration during upgrades** | No automatic migration of in-flight jobs during code deployments. | Drain queues before deployment or use versioned job handlers that can process multiple formats. |
| **Multi-broker support** | Only Redis is supported as the broker; no pluggable broker architecture. | Use Redis Sentinel/Cluster for HA; future extension could abstract broker interface. |
| **Built-in rate limiting** | No per-user or per-job-type rate limiting in the core. | Implement in job handlers or use Redis-backed rate limiter before enqueueing jobs. |
| **Job prioritization within queue** | All jobs in same queue are FIFO; no sub-prioritization within queue. | Use multiple queues with different priorities or add timestamp-based ordering within queue. |
| **Long-term job archival** | Job history older than configured retention is deleted, not archived. | External system can consume job completion events and archive to cold storage. |
| **Built-in user management** | Dashboard has simple auth or none; no RBAC or multi-user permission system. | Place dashboard behind reverse proxy with authentication for production use. |
| **Geographic job routing** | Jobs cannot be routed to workers in specific regions/datacenters. | Use separate Redis instances per region or implement routing in producer logic. |
| **Real-time job streaming** | Cannot subscribe to job events via streaming APIs (though events are emitted). | Use Redis pub/sub or add webhook callbacks as extension. |
| **Job pausing/resuming** | Cannot pause a specific job's execution once started. | Implement checkpointing within job handler code for long-running jobs. |
| **Automatic scaling of workers** | No auto-scaling based on queue depth; manual worker management required. | Use container orchestration (K8s HPA) that monitors queue metrics to scale workers. |
| **Built-in job versioning** | No automatic handling of breaking changes to job payload formats. | Use version field in job metadata and handler version compatibility logic. |
| **Transactional enqueue** | Cannot atomically enqueue multiple jobs across queues. | Use Redis pipelines for best-effort atomicity or implement compensating transactions. |

> **Architectural Philosophy:** By explicitly stating these non-goals, we acknowledge common requests but defer them to keep the initial implementation manageable. The architecture should be extensible enough to add these features later without major redesign. For example, while we don't build job dependency graphs, the system's webhook events could feed into an external workflow orchestrator.

#### Boundary Decisions with Rationale

For each major non-goal, we made explicit architectural decisions:

**Decision: No Built-in Job Dependencies**
- **Context:** Many background job systems need to chain jobs (A → B → C) or create complex workflows.
- **Options Considered:** 
  1. **Built-in DAG engine:** Complex, requires persistent graph storage and complex scheduling.
  2. **Simple chaining:** Job A enqueues job B on completion—simple but creates tight coupling.
  3. **External workflow system:** Keep job processor simple, integrate with specialized workflow engines.
- **Decision:** Option 3—no built-in dependencies, but emit completion events for external integration.
- **Rationale:** Job dependencies belong in a workflow layer, not the core job execution engine. Most users can implement simple chaining in job handlers, while complex workflows need specialized tools (Airflow, Temporal) that our system can feed into.
- **Consequences:** Users must implement chaining manually or use external system; system remains simple and focused.

**Decision: At-Least-Once Semantics Only**
- **Context:** Exactly-once execution requires distributed transactions and persistent deduplication.
- **Options Considered:**
  1. **Exactly-once with idempotency keys:** Complex, requires persistent store of processed IDs.
  2. **At-least-once with idempotent jobs:** Simpler, pushes idempotency responsibility to job handlers.
  3. **At-most-once:** Unacceptable for reliable job processing.
- **Decision:** Option 2—guarantee jobs execute at least once, possibly multiple times during failures.
- **Rationale:** Exactly-once semantics are extremely difficult in distributed systems and often require application-level idempotency anyway. By embracing at-least-once, we simplify the core system dramatically while providing the tools (idempotency keys, job metadata) for handlers to implement idempotency.
- **Consequences:** Job handlers must be idempotent; system cannot guarantee no duplicates during network partitions.

**Decision: Single Broker (Redis) Architecture**
- **Context:** Production systems might want to use different brokers (RabbitMQ, Kafka, SQS) for different needs.
- **Options Considered:**
  1. **Pluggable broker interface:** Abstract broker operations behind interface, support multiple backends.
  2. **Redis-only with extension points:** Optimize for Redis, allow extensions for other brokers.
  3. **Multi-broker from day one:** Support 2-3 popular brokers in initial implementation.
- **Decision:** Option 2—Redis-only core with clear abstraction boundaries for future broker support.
- **Rationale:** Redis provides all needed functionality with excellent performance and reliability. Supporting multiple brokers initially would triple implementation complexity. By keeping clean internal interfaces, we can add other brokers later without breaking existing Redis users.
- **Consequences:** Users locked into Redis; migration to other brokers requires code changes later.

### Implementation Guidance

> **Implementation Note:** While this section doesn't directly translate to code, the goals and non-goals defined here should guide every implementation decision. Before writing any code, ensure your design addresses each functional requirement while respecting the non-goals.

#### Technology Recommendations

| Component | Simple Option (Beginner-Friendly) | Advanced Option (Production-Ready) |
|-----------|-----------------------------------|-----------------------------------|
| **Serialization** | JSON with Python's `json` module | MessagePack (`msgpack`) for better performance and binary data |
| **Redis Client** | `redis-py` with connection pooling | `redis-py` with client-side sharding for cluster support |
| **Concurrency** | `concurrent.futures.ThreadPoolExecutor` | `multiprocessing.Pool` with process isolation for CPU-bound jobs |
| **Web Dashboard** | Flask + Jinja2 templates with auto-refresh | FastAPI + WebSocket for real-time updates + React frontend |
| **Cron Parsing** | `croniter` library for schedule calculation | Custom parser supporting extended cron syntax with timezones |
| **Metrics** | Custom Redis counters updated after job completion | Prometheus client library + Grafana for advanced visualization |

#### Recommended Configuration Structure

Create a configuration system that balances simplicity (≤5 required parameters) with flexibility:

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class QueueConfig:
    name: str
    priority: int = 1  # Higher = more frequent polling
    max_length: Optional[int] = None  # Optional queue size limit

@dataclass
class WorkerConfig:
    queues: List[str]  # Queues this worker polls
    concurrency: int = 4  # Number of concurrent job executions
    heartbeat_interval: int = 30  # Seconds between heartbeats
    job_timeout: int = 1800  # Default job timeout in seconds (30 min)

@dataclass
class RedisConfig:
    url: str = "redis://localhost:6379/0"
    max_connections: int = 100
    socket_timeout: int = 5
    retry_on_timeout: bool = True

@dataclass
class SystemConfig:
    redis: RedisConfig
    queues: List[QueueConfig]
    workers: List[WorkerConfig]
    
    # Non-functional defaults
    max_payload_size: int = 1024 * 1024  # 1MB
    job_history_size: int = 10000  # Keep last 10k jobs
    retry_base_delay: int = 1  # Base delay in seconds for exponential backoff
    retry_max_attempts: int = 5  # Default max retries
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Load configuration from environment variables with defaults."""
        return cls(
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
            ),
            queues=[
                QueueConfig(name="default", priority=1),
                QueueConfig(name="high", priority=3),
                QueueConfig(name="low", priority=1),
            ],
            workers=[
                WorkerConfig(
                    queues=["default", "high", "low"],
                    concurrency=int(os.getenv("WORKER_CONCURRENCY", "4"))
                )
            ]
        )
```

#### Milestone Goal Mapping

As you implement each milestone, verify your work against these specific goal checkpoints:

| Milestone | Must-Have Goal | How to Verify |
|-----------|----------------|---------------|
| **1** | Jobs enqueue and dequeue in FIFO order | Enqueue 10 jobs, dequeue 10 jobs - verify same order. |
| | Payloads >1MB rejected | Try enqueuing 2MB payload - get validation error. |
| **2** | Worker processes jobs concurrently | Enqueue 5 jobs that each sleep 1s - all complete in ~1s with concurrency=5. |
| | Graceful shutdown on SIGTERM | Send SIGTERM to worker during job execution - job completes before exit. |
| **3** | Exponential backoff with jitter | Fail job 3 times - verify retry delays are ~1s, ~2s, ~4s ± jitter. |
| | Dead letter queue after max retries | Fail job beyond max retries - find it in dead letter queue. |
| **4** | Delayed job execution | Schedule job for 5s future - verify it executes ~5s after enqueue. |
| | Cron job re-enqueues | Create cron job for every minute - see it enqueue automatically each minute. |
| **5** | Real-time dashboard updates | Open dashboard, enqueue job - see queue depth update without refresh. |
| | Job search by criteria | Enqueue jobs with different metadata - search finds them by attributes. |

#### Common Implementation Pitfalls to Avoid

While implementing to meet these goals, watch for these specific pitfalls:

⚠️ **Pitfall: Over-engineering for edge cases**
- **Description:** Trying to handle every possible failure mode or feature request from the start.
- **Why it's wrong:** Violates YAGNI principle, dramatically increases complexity, delays delivery of core functionality.
- **Fix:** Implement only the requirements from the table above. Add extensibility points for future enhancements but don't implement them now.

⚠️ **Pitfall: Ignoring non-functional requirements**
- **Description:** Building functionally correct system that's too slow, uses too much memory, or can't scale.
- **Why it's wrong:** System may work in development but fail in production under load.
- **Fix:** Measure performance early—profile enqueue/dequeue latency, memory usage of workers, Redis connection counts.

⚠️ **Pitfall: Violating explicit non-goals**
- **Description:** Adding "just one small feature" that's explicitly listed as a non-goal.
- **Why it's wrong:** Creates scope creep, introduces complexity for features users shouldn't depend on.
- **Fix:** When tempted to add a feature, check if it's in the non-goals table. If yes, document it as a future extension but don't implement now.

⚠️ **Pitfall: Hardcoding instead of configuring**
- **Description:** Embedding queue names, timeouts, retry counts in code instead of configuration.
- **Why it's wrong:** Requires code changes for operational adjustments, limits deployment flexibility.
- **Fix:** Use the configuration pattern shown above for all tunable parameters.

---


## High-Level Architecture

> **Milestone(s):** This section provides the architectural foundation for all five milestones, defining the core components, their responsibilities, and how they interact. It establishes the blueprint that each subsequent milestone's detailed component design will follow and extend.

This section presents the **bird's-eye view** of the Background Job Processor system. Before diving into individual components like queues or workers, it's essential to understand the complete picture—how all parts fit together into a cohesive whole. The architecture is designed around a central **broker** (Redis) that acts as the system's nervous system, connecting all components while maintaining loose coupling and enabling horizontal scaling. We'll explore this through three lenses: the static view of components and their responsibilities, the dynamic view of how they communicate, and the structural view of how code should be organized.

### System Components and Responsibilities

Think of the system as a **distributed factory** with specialized departments. Each department has a specific role, operates independently, but communicates through a central dispatch office (Redis). This separation of concerns allows each component to be developed, deployed, and scaled independently.

![System Component Overview](./diagrams/system-component.svg)

The system comprises seven core components, each with distinct responsibilities:

| Component | Primary Responsibility | Key Data It Owns/Maintains | Interaction Partners |
|-----------|------------------------|----------------------------|----------------------|
| **Producer** | Creates and submits jobs for asynchronous execution | Local job definitions and arguments (not persisted centrally) | Queue Manager (via client library) |
| **Queue Manager** | Validates, prioritizes, and stores incoming jobs into appropriate queues | Queue configurations, job serialization/deserialization logic | Producer, Redis (broker), Worker |
| **Worker** | Fetches jobs from queues and executes the corresponding task logic | Worker process state, job execution environment, thread/process pool | Queue Manager (via Redis), Job Handlers, Monitoring |
| **Scheduler** | Manages delayed and recurring jobs based on cron expressions | Schedule definitions, next execution timestamps, uniqueness constraints | Redis (for scheduled sets), Queue Manager |
| **Retry Manager** | Handles failed jobs with exponential backoff and manages dead letter queue | Retry policies, backoff calculations, error history | Worker (on failure), Redis (for retry sets) |
| **Monitor/Dashboard** | Provides real-time visibility into system health and job status | Aggregated metrics, job history, alert configurations | All components (via Redis and direct reporting) |
| **Redis (Broker)** | Serves as the central data store for all job and system state | Job queues, scheduled sets, retry sets, worker heartbeats, metrics | All other components |

Let's examine each component in detail:

**Producer** is any application code that needs to perform work asynchronously. It's not a separate service but a **client library** integrated into your application. The producer's sole responsibility is to define a job (type, arguments, metadata) and hand it off to the Queue Manager. It doesn't wait for job completion—this is the essence of asynchronous execution.

**Queue Manager** acts as the **post office sorting facility** for jobs. When a producer submits a job, the Queue Manager validates the payload size (rejecting oversized jobs), assigns a unique job ID (ULID for time-ordered uniqueness), serializes the job to JSON, and determines which queue it belongs to based on the job type and priority configuration. It then uses Redis list operations to atomically enqueue the job. The Queue Manager exposes a clean API (`enqueue`, `bulk_enqueue`, `queue_stats`) that producers call.

**Worker** is the **factory assembly line** that actually does the work. Each worker process runs an infinite loop that polls Redis for available jobs across multiple queues, respecting priority weights. When it obtains a job, it deserializes it, looks up the corresponding job handler (registered during worker startup), and executes it within a configurable timeout. The worker manages the job's lifecycle state transitions (PENDING → ACTIVE → COMPLETED/FAILED) and reports heartbeats to Redis so the monitoring system knows it's alive.

**Scheduler** functions like a **calendar app with recurring events**. It runs as a separate background process that continuously polls Redis sorted sets for jobs scheduled for future execution. For delayed jobs (run once at a specific time), it simply moves them to the appropriate work queue when their timestamp arrives. For recurring jobs defined by cron expressions, it calculates the next execution time after each run and re-enqueues the job. The scheduler also handles timezone conversions and ensures unique jobs aren't duplicated.

**Retry Manager** implements a **customer service escalation process**. When a job fails, the worker consults the Retry Manager to determine what should happen next. The Retry Manager examines the job's attempt count, applies an exponential backoff algorithm (with optional jitter) to calculate the next retry delay, and stores the job in a Redis sorted set keyed by its retry timestamp. A separate polling process (or the main scheduler) later moves retry jobs back to work queues. Jobs that exceed their maximum retry count are moved to the dead letter queue for manual inspection.

**Monitor/Dashboard** serves as the **air traffic control tower** for the entire system. It collects metrics from Redis (queue lengths, worker heartbeats) and from job completion events, aggregates them, and presents them in a real-time web dashboard. It can trigger alerts when queues exceed thresholds or error rates spike. The dashboard also provides administrative controls to manually retry failed jobs or inspect job details.

**Redis** is the **central nervous system** connecting all components. We use different Redis data structures for different purposes:
- **Lists** for FIFO job queues (LPUSH/RPOP)
- **Sorted Sets** for scheduled and retry jobs (ZADD/ZRANGEBYSCORE)
- **Hashes** for job details and worker heartbeats (HSET/HGETALL)
- **Streams** for job completion events and metrics (XADD/XREAD)
- **Sets** for tracking active workers and queues (SADD/SMEMBERS)

> **Design Insight:** Redis serves as both the queue broker AND the system of record for job state. This simplifies the architecture (no separate database needed) but requires careful Redis data design to avoid bottlenecks. All components are stateless except for Redis—this enables horizontal scaling of workers and producers.

Now let's examine the key interfaces between these components:

**Queue Manager Interface Table:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `enqueue` | `job_type: str`, `args: List`, `kwargs: Dict`, `queue: str`, `priority: int`, `max_retries: int`, `timeout_seconds: int`, `metadata: Dict` | `job_id: str` | Validates job payload, generates job ID, serializes job, and pushes to appropriate Redis queue |
| `bulk_enqueue` | `jobs: List[Dict]` (each dict contains same fields as `enqueue` parameters) | `List[str]` (job IDs) | Atomically enqueues multiple jobs in a single Redis transaction |
| `queue_stats` | `queue_names: Optional[List[str]]` | `Dict[str, Dict]` with keys: `pending_count`, `active_count`, `scheduled_count`, `retry_count` | Returns current metrics for specified queues (or all queues) |
| `peek` | `queue: str`, `count: int` | `List[Job]` | Returns next `count` jobs from queue without removing them (for inspection) |

**Worker Interface Table:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `register_handler` | `job_type: str`, `handler: Callable` | `None` | Maps a job type string to a Python function that will execute jobs of that type |
| `start` | `config: WorkerConfig` | `None` (blocks until shutdown) | Starts the worker main loop, polling queues and executing jobs |
| `stop` | `graceful: bool = True` | `None` | Signals the worker to stop (immediately if `graceful=False`, else after current job completes) |
| `pause` | `queue: Optional[str]` | `None` | Temporarily stops processing jobs from specified queue (or all queues) |
| `resume` | `queue: Optional[str]` | `None` | Resumes processing from a paused queue |

**Scheduler Interface Table:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `schedule` | `job: Job`, `run_at: datetime` | `schedule_id: str` | Schedules a job for one-time execution at `run_at` |
| `schedule_cron` | `job: Job`, `cron_expression: str`, `timezone: Optional[str]` | `schedule_id: str` | Schedules a recurring job using cron expression |
| `unschedule` | `schedule_id: str` | `bool` (success) | Removes a scheduled job (both one-time and recurring) |
| `list_schedules` | `job_type: Optional[str]` | `List[Dict]` | Returns all scheduled jobs, optionally filtered by job type |

**Retry Manager Interface Table:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `schedule_retry` | `job: Job`, `error: Exception` | `retry_at: datetime` | Calculates next retry time using exponential backoff, stores job in retry set, records error |
| `get_dead_letter_jobs` | `queue: Optional[str]`, `limit: int` | `List[Job]` | Retrieves jobs from dead letter queue for manual inspection/retry |
| `retry_dead_letter_job` | `job_id: str` | `bool` (success) | Moves a job from dead letter queue back to work queue for immediate retry |
| `purge_dead_letter_jobs` | `older_than: Optional[datetime]` | `int` (count purged) | Permanently removes jobs from dead letter queue, optionally filtered by age |

**Monitor Interface Table:**
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_queue_metrics` | `time_range: str = "1h"` | `Dict[str, Any]` | Returns queue depths, processing rates, error rates for all queues over time range |
| `get_worker_status` | `worker_id: Optional[str]` | `List[Dict]` or `Dict` | Returns status of all workers or specific worker (ID, current job, last heartbeat) |
| `search_jobs` | `filters: Dict` (job_id, status, queue, time_range) | `List[Job]` | Searches job history with pagination and filtering |
| `trigger_alert` | `alert_type: str`, `details: Dict` | `None` | Triggers an alert (logs, webhook, etc.) when system conditions warrant |

### Communication Patterns and Data Flow

The components communicate through **two primary patterns**: **broker-mediated messaging** (via Redis) for job data flow, and **direct API calls** for management operations. This hybrid approach balances decoupling with efficiency.

**Primary Communication Channels:**

1. **Redis as Message Broker**: All job-related data flows through Redis using atomic operations:
   - **Producer → Queue Manager → Redis**: Jobs are LPUSHed to list keys like `queue:{queue_name}`
   - **Redis → Worker**: Workers BRPOP jobs from multiple queues
   - **Worker → Retry Manager → Redis**: Failed jobs are ZADDed to sorted set `retry:{queue_name}` with score = retry timestamp
   - **Scheduler → Redis**: Due jobs are moved from `scheduled` sorted sets to work queues
   - **All → Monitor**: Components write metrics/events to Redis streams/hashes for monitoring

2. **Direct HTTP/RPC for Management**: The dashboard and administrative tools communicate directly with components via REST APIs or RPC:
   - **Dashboard → Monitor API**: Real-time metrics polling
   - **Dashboard → Queue Manager**: Manual job enqueueing/ inspection
   - **Dashboard → Retry Manager**: Dead letter queue operations

**Data Flow Through the System:**

Let's trace a job through the complete system with three scenarios: immediate execution, scheduled execution, and failed execution with retries.

**Scenario 1: Immediate Job Execution (Happy Path)**

1. **Producer** creates a `Job` object with `job_type="process_image"`, `args=["image.jpg"]`, `queue="high_priority"`.
2. **Producer** calls `QueueManager.enqueue(job)`.
3. **Queue Manager** validates the job (size < 1MB), generates a ULID job ID, serializes the job to JSON, and executes `LPUSH queue:high_priority <serialized_job>`.
4. **Worker** (continuously polling with `BRPOP queue:high_priority queue:low_priority ...`) receives the serialized job from Redis.
5. **Worker** deserializes the job, looks up the registered handler for `process_image`, and executes it within a timeout.
6. **Worker** on success: Updates job status to `COMPLETED`, stores result in Redis hash `job:{job_id}`, emits completion event to Redis stream `events:completed`.
7. **Monitor** reads from `events:completed` stream, updates real-time metrics, and makes job result available via dashboard.

**Scenario 2: Scheduled Recurring Job**

1. **Producer** calls `Scheduler.schedule_cron(job, "0 * * * *")` (hourly job).
2. **Scheduler** calculates next run time (e.g., 14:00 today), stores job in Redis sorted set `scheduled:cron` with score = 14:00 timestamp.
3. **Scheduler Process** (separate daemon) polls `scheduled:cron` every 30 seconds for jobs with score <= current time.
4. At 14:00, scheduler finds the job, moves it to work queue via `QueueManager.enqueue(job)`, calculates next run time (15:00), updates the score in `scheduled:cron`.
5. Job flows through normal execution path (Scenario 1).

**Scenario 3: Failed Job with Retries**

1. **Worker** executes a job, but it raises `ConnectionError`.
2. **Worker** catches the exception, calls `RetryManager.schedule_retry(job, error)`.
3. **Retry Manager** increments job's `attempts` count, calculates retry delay using exponential backoff (e.g., 2^attempt seconds), stores job in Redis sorted set `retry:{queue_name}` with score = current time + delay.
4. **Retry Poller** (could be same as scheduler) periodically checks retry sets for due jobs (score <= current time), moves them back to work queues.
5. If job fails again, process repeats with longer delay (4s, 8s, 16s...).
6. After exceeding `max_retries` (default 5), job is moved to dead letter queue (Redis list `dead:{queue_name}`) with full error history.

**Component Interaction Patterns Table:**
| Interaction | Pattern | Technology | Data Format | Reliability Guarantee |
|-------------|---------|------------|-------------|------------------------|
| Job Enqueue | Fire-and-forget | Redis LPUSH | JSON | At-most-once (may be lost if Redis crashes before persistence) |
| Job Dequeue | Blocking poll | Redis BRPOP | JSON | At-least-once (job may be processed multiple times if worker crashes after dequeue but before processing) |
| Scheduled Jobs | Periodic poll | Redis ZRANGEBYSCORE | JSON | At-least-once (may execute multiple times if scheduler crashes after enqueue but before updating score) |
| Worker Heartbeats | Periodic write | Redis HSET | JSON | Best-effort (stale workers detected via timeout) |
| Metrics Collection | Write to stream | Redis XADD | JSON | Best-effort (occasional loss acceptable) |
| Dashboard Updates | Polling + WebSocket | HTTP + Redis pub/sub | JSON | Best-effort |

> **Design Insight:** The system guarantees **at-least-once delivery** for jobs, which is appropriate for most background tasks (idempotent by design). For truly once-only processing, jobs must include their own idempotency logic using unique keys in the application domain.

**Data Flow State Transitions:**

A job moves through various Redis data structures during its lifecycle:

```
Job Created → queue:{name} LIST (PENDING)
            → (Worker fetches) → job:{id} HASH (ACTIVE)
            → (Success) → events:completed STREAM (COMPLETED)
            → (Failure) → retry:{queue} ZSET (RETRY_SCHEDULED)
            → (After backoff) → queue:{name} LIST again
            → (Max retries) → dead:{queue} LIST (DEAD_LETTER)
```

**Concurrency and Scaling Considerations:**

- **Multiple Workers** can poll the same queue simultaneously—Redis ensures each job goes to only one worker via atomic BRPOP.
- **Queue Priority** is implemented by having workers poll multiple queues with weighted round-robin (more BRPOP calls on high-priority queues).
- **Horizontal Scaling** is achieved by adding more worker processes or even worker hosts—all connect to the same Redis instance.
- **Redis Bottlenecks** may occur at very high throughput (>10K jobs/sec); solutions include Redis clustering or sharding queues across instances.

**Failure Handling in Communication:**

- **Redis Connection Loss**: All components implement exponential backoff reconnection logic with circuit breaker pattern.
- **Worker Crash During Job**: Jobs remain in ACTIVE state; a cleanup process periodically moves stale active jobs back to PENDING queue.
- **Network Partition**: Split-brain scenarios are mitigated by favoring availability—jobs continue processing on both sides of partition, requiring application-level idempotency.

### Recommended File and Module Structure

Organizing code clearly from the start prevents "big ball of mud" architecture. The recommended structure follows **clean architecture** principles: core domain logic independent of frameworks, with clear separation between components.

**Project Root Structure:**
```
background_job_processor/
├── pyproject.toml                    # Python project config and dependencies
├── README.md
├── .env.example                      # Example environment variables
├── docker-compose.yml                # For local Redis and dashboard
├── scripts/                          # Utility scripts
│   ├── start_worker.py
│   ├── start_scheduler.py
│   └── seed_test_jobs.py
├── src/                              # Main source code (Python package)
│   └── job_processor/
│       ├── __init__.py
│       ├── core/                     # Domain models and interfaces
│       │   ├── __init__.py
│       │   ├── job.py               # Job class and serialization
│       │   ├── exceptions.py        # Custom exceptions
│       │   └── types.py             # Type definitions and enums
│       ├── redis_client/             # Redis abstraction layer
│       │   ├── __init__.py
│       │   ├── client.py            # RedisClient wrapper
│       │   ├── connection_pool.py
│       │   └── errors.py
│       ├── queue/                    # Milestone 1: Queue Manager
│       │   ├── __init__.py
│       │   ├── manager.py           # QueueManager class
│       │   ├── priority.py          # Priority calculation logic
│       │   ├── serialization.py     # Job serialization/validation
│       │   └── tests/
│       ├── worker/                   # Milestone 2: Worker Process
│       │   ├── __init__.py
│       │   ├── worker.py            # Worker main class
│       │   ├── handler_registry.py  # Job type to handler mapping
│       │   ├── heartbeat.py         # Worker health reporting
│       │   ├── pool.py              # Thread/process pool for concurrency
│       │   └── tests/
│       ├── retry/                    # Milestone 3: Retry System
│       │   ├── __init__.py
│       │   ├── manager.py           # RetryManager class
│       │   ├── backoff.py           # Exponential backoff calculators
│       │   ├── dead_letter.py       # Dead letter queue operations
│       │   └── tests/
│       ├── scheduler/                # Milestone 4: Scheduling
│       │   ├── __init__.py
│       │   ├── scheduler.py         # Scheduler main class
│       │   ├── cron.py              # Cron expression parser
│       │   ├── timezone.py          # Timezone handling utilities
│       │   └── tests/
│       ├── monitoring/               # Milestone 5: Monitoring
│       │   ├── __init__.py
│       │   ├── metrics.py           # Metrics collection and aggregation
│       │   ├── dashboard/           # Web dashboard (FastAPI/Flask)
│       │   │   ├── app.py
│       │   │   ├── routes.py
│       │   │   ├── static/
│       │   │   └── templates/
│       │   ├── alerts.py            # Alerting logic
│       │   └── tests/
│       ├── config/                   # Configuration management
│       │   ├── __init__.py
│       │   ├── settings.py          # SystemConfig and friends
│       │   └── validation.py
│       └── utils/                    # Shared utilities
│           ├── __init__.py
│           ├── ulid_generator.py
│           ├── signal_handlers.py
│           └── validation.py
└── tests/                            # Integration and end-to-end tests
    ├── conftest.py
    ├── test_integration.py
    └── fixtures/
```

**Module Dependencies Flow:** 
```
core/ (no external dependencies)
  ↓
redis_client/ (depends on core, redis-py)
  ↓
queue/ (depends on core, redis_client)
  ↓
worker/ (depends on core, redis_client, queue)
  ↓
retry/ (depends on core, redis_client, queue)
  ↓
scheduler/ (depends on core, redis_client, queue)
  ↓
monitoring/ (depends on core, redis_client, all other modules)
```

**Key Design Decisions in Module Structure:**

> **Decision: Layered Architecture vs Vertical Slicing**
> - **Context**: We need to organize code for a complex system with multiple interrelated components. Two common approaches are layered architecture (separating concerns like data access, business logic, presentation) and vertical slicing (organizing by feature/bounded context).
> - **Options Considered**:
>   1. **Pure Layered Architecture**: `data_access/`, `business_logic/`, `api/` layers
>   2. **Vertical Slices by Component**: `queue/`, `worker/`, `scheduler/` as self-contained modules
>   3. **Hybrid Approach**: Component-based with shared core layer
> - **Decision**: Hybrid approach (option 3) as shown in the structure above
> - **Rationale**: Each component (queue, worker, scheduler) has clear boundaries and can be developed/tested independently (vertical slicing), while shared concerns like Redis client, configuration, and core domain models are extracted to common layers to avoid duplication. This balances modularity with DRY principles.
> - **Consequences**: Components are loosely coupled and can be optional (e.g., you could use just the queue and worker without scheduler). However, there's some complexity in managing cross-component dependencies.

**Component Interface Contracts:**

Each component module exposes a clean public API while hiding implementation details:

- **`queue` module**: Exports `QueueManager` class, `QueueConfig`, `enqueue_job` helper function
- **`worker` module**: Exports `Worker`, `WorkerConfig`, `register_job_handler` decorator
- **`scheduler` module**: Exports `Scheduler`, `CronSchedule`, `schedule_job` helper
- **`retry` module**: Exports `RetryManager`, `ExponentialBackoff`, `DeadLetterQueue`
- **`monitoring` module**: Exports `MetricsCollector`, `DashboardApp`, `setup_monitoring`

**Configuration Management Strategy:**

All components share a central `SystemConfig` object loaded from environment variables with sensible defaults. Each component section (queue, worker) has its own config class that can be overridden individually.

```python
# Example configuration loading
config = SystemConfig.from_env()
queue_manager = QueueManager(config.redis, config.queues)
worker = Worker(config.workers[0], config.redis)
```

**Deployment View:**

- **Development**: Single Redis instance, workers and scheduler run as separate processes on same machine
- **Production**: Redis cluster or managed Redis service, workers distributed across multiple machines/containers, scheduler as redundant active-passive pair
- **Container Orchestration**: Each component (worker, scheduler, dashboard) in its own Docker container, scaled independently

**Module Responsibility Table:**
| Module | Primary Classes | Key Responsibilities | Depends On |
|--------|----------------|---------------------|------------|
| `core` | `Job`, `JobStatus` | Define core domain models and interfaces | None (pure Python) |
| `redis_client` | `RedisClient` | Abstract Redis operations, connection pooling, error handling | `core`, `redis-py` |
| `queue` | `QueueManager` | Job validation, serialization, enqueueing with priority | `core`, `redis_client` |
| `worker` | `Worker`, `HandlerRegistry` | Polling queues, executing jobs, concurrency management | `core`, `redis_client`, `queue` |
| `retry` | `RetryManager`, `ExponentialBackoff` | Managing retry logic, backoff calculations, dead letter queue | `core`, `redis_client`, `queue` |
| `scheduler` | `Scheduler`, `CronParser` | Scheduling delayed/recurring jobs, timezone handling | `core`, `redis_client`, `queue` |
| `monitoring` | `MetricsCollector`, `DashboardApp` | Collecting metrics, providing web UI, alerting | All other modules |

> **Design Insight:** This modular structure enables incremental implementation aligned with the project milestones. You can build and test the `queue` module (Milestone 1) independently before implementing `worker` (Milestone 2), with each milestone adding a new module to the system.

### Implementation Guidance

**Technology Recommendations Table:**
| Component | Simple Option | Advanced Option | Rationale for Choice |
|-----------|---------------|-----------------|----------------------|
| **Redis Client** | `redis-py` with connection pooling | `hiredis` parser for performance + `redis-py` | `redis-py` is standard, well-maintained, and sufficient for most workloads |
| **Web Dashboard** | FastAPI + Jinja2 templates + HTMX for dynamic updates | Separate React frontend with FastAPI backend | FastAPI provides automatic OpenAPI docs, async support; HTMX keeps complexity low |
| **Job Serialization** | JSON with `orjson` for speed | MessagePack for smaller payloads | JSON is universal, debuggable; `orjson` is 2-3x faster than stdlib `json` |
| **Concurrency Model** | `concurrent.futures.ThreadPoolExecutor` | `multiprocessing.Pool` for CPU-bound tasks | Threads are fine for I/O-bound jobs; processes for CPU-bound but with serialization overhead |
| **Configuration** | Python-dotenv + Pydantic models | ConfigMap + environment variables (K8s) | Pydantic provides validation with clear error messages |
| **Testing** | pytest with `fakeredis` for unit tests | Docker-based integration tests with real Redis | `fakeredis` is fast for unit tests; real Redis for integration |

**Recommended File Structure Implementation:**

Here's the minimal structure to begin with (expanding as milestones progress):

```bash
mkdir -p background_job_processor/src/job_processor/{core,redis_client,queue,worker,retry,scheduler,monitoring,config,utils}
touch background_job_processor/src/job_processor/__init__.py
```

**Infrastructure Starter Code:**

Create these foundational files first—they're prerequisites for all components:

**1. Core Domain Models (`src/job_processor/core/job.py`):**
```python
"""Core Job model and serialization."""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class JobStatus(Enum):
    """Enumeration of possible job states."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTER = "dead_letter"


@dataclass
class Job:
    """Represents a unit of work to be processed asynchronously."""
    job_id: str
    job_type: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    queue: str = "default"
    priority: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for date_field in ['created_at', 'started_at', 'completed_at']:
            if data[date_field]:
                data[date_field] = data[date_field].isoformat() if isinstance(data[date_field], datetime) else data[date_field]
        # Convert status enum to string
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Reconstruct job from dictionary."""
        # Convert string dates back to datetime objects
        date_fields = ['created_at', 'started_at', 'completed_at']
        for field_name in date_fields:
            if field_name in data and data[field_name] and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
            elif field_name in data and not data[field_name]:
                data[field_name] = None
        
        # Convert status string back to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = JobStatus(data['status'])
        
        return cls(**data)
    
    def serialize(self) -> str:
        """Serialize job to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Job':
        """Deserialize job from JSON string."""
        return cls.from_dict(json.loads(data))
    
    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during job execution."""
        import traceback
        self.errors.append({
            'type': error.__class__.__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'occurred_at': datetime.utcnow().isoformat(),
            'attempt': self.attempts
        })
    
    def should_retry(self) -> bool:
        """Determine if job should be retried."""
        return self.attempts < self.max_retries and self.status != JobStatus.DEAD_LETTER
```

**2. Redis Client Abstraction (`src/job_processor/redis_client/client.py`):**
```python
"""Redis client wrapper with connection pooling and error handling."""
import logging
from typing import Any, Dict, Optional
import redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


class RedisClient:
    """Singleton Redis client wrapper with connection pooling."""
    
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None
    
    def __init__(self, url: str, **connection_kwargs):
        """Initialize Redis client (private, use get_instance)."""
        self.url = url
        self.connection_kwargs = connection_kwargs
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to Redis."""
        try:
            self._client = redis.from_url(
                self.url,
                **self.connection_kwargs,
                decode_responses=True  # Auto-decode bytes to str
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.url}")
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    @classmethod
    def get_instance(cls, url: Optional[str] = None, **kwargs) -> 'RedisClient':
        """Get singleton RedisClient instance."""
        if cls._instance is None:
            if url is None:
                url = "redis://localhost:6379/0"
            cls._instance = cls(url, **kwargs)
        return cls._instance
    
    def get_client(self) -> redis.Redis:
        """Get the underlying Redis client instance."""
        if self._client is None:
            self.connect()
        return self._client
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute a Redis command with error handling."""
        try:
            client = self.get_client()
            method = getattr(client, command)
            return method(*args, **kwargs)
        except RedisError as e:
            logger.error(f"Redis command failed: {command} {args}: {e}")
            # Re-raise for caller to handle
            raise
    
    def pipeline(self) -> redis.Pipeline:
        """Return a Redis pipeline for atomic operations."""
        return self.get_client().pipeline()
    
    def health_check(self) -> bool:
        """Check if Redis is responsive."""
        try:
            return self.get_client().ping()
        except RedisError:
            return False
```

**3. Configuration Management (`src/job_processor/config/settings.py`):**
```python
"""Configuration management with Pydantic models."""
import os
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class RedisConfig(BaseModel):
    """Redis connection configuration."""
    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=10, ge=1)
    socket_timeout: int = Field(default=5, ge=1)
    retry_on_timeout: bool = Field(default=True)


class QueueConfig(BaseModel):
    """Queue configuration."""
    name: str
    priority: int = Field(default=1, ge=0)
    max_length: Optional[int] = Field(default=None, ge=1)
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v or ':' in v:
            raise ValueError('Queue name cannot be empty or contain colons')
        return v


class WorkerConfig(BaseModel):
    """Worker process configuration."""
    queues: List[str] = Field(default_factory=lambda: ["default"])
    concurrency: int = Field(default=4, ge=1)
    heartbeat_interval: int = Field(default=30, ge=5)
    job_timeout: int = Field(default=300, ge=1)


class SystemConfig(BaseModel):
    """Main system configuration."""
    redis: RedisConfig = Field(default_factory=RedisConfig)
    queues: List[QueueConfig] = Field(default_factory=lambda: [QueueConfig(name="default")])
    workers: List[WorkerConfig] = Field(default_factory=lambda: [WorkerConfig()])
    max_payload_size: int = Field(default=1048576, ge=1024)  # 1MB default
    job_history_size: int = Field(default=1000, ge=0)  # How many completed jobs to keep
    retry_base_delay: int = Field(default=1, ge=1)  # Base delay in seconds for exponential backoff
    retry_max_attempts: int = Field(default=5, ge=0)
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load configuration from environment variables with defaults."""
        return cls(
            redis=RedisConfig(
                url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
                socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
                retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
            ),
            max_payload_size=int(os.getenv("MAX_PAYLOAD_SIZE", "1048576")),
            job_history_size=int(os.getenv("JOB_HISTORY_SIZE", "1000")),
            retry_base_delay=int(os.getenv("RETRY_BASE_DELAY", "1")),
            retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "5"))
        )
```

**Core Logic Skeleton for Queue Manager (`src/job_processor/queue/manager.py`):**
```python
"""Queue Manager - handles job enqueueing with validation and priority."""
import logging
from typing import Dict, List, Optional, Any
from uuid import uuid4

from ..core.job import Job, JobStatus
from ..redis_client.client import RedisClient
from ..config.settings import QueueConfig, SystemConfig


logger = logging.getLogger(__name__)


class QueueManager:
    """Manages job queues with validation, serialization, and priority handling."""
    
    def __init__(self, redis_client: RedisClient, queue_configs: List[QueueConfig]):
        """Initialize QueueManager with Redis client and queue configurations."""
        self.redis = redis_client
        self.queues = {q.name: q for q in queue_configs}
        # TODO: Initialize any required Redis data structures on startup
        # TODO: Validate that queue names don't conflict with Redis key patterns
    
    def enqueue(
        self,
        job_type: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        queue: str = "default",
        priority: int = 0,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enqueue a job for asynchronous execution.
        
        Args:
            job_type: String identifier for the job handler
            args: Positional arguments for the job handler
            kwargs: Keyword arguments for the job handler
            queue: Name of the queue to enqueue to
            priority: Priority level (higher = more important)
            max_retries: Maximum number of retry attempts
            timeout_seconds: Maximum execution time in seconds
            metadata: Additional metadata for the job
            
        Returns:
            job_id: Unique identifier for the enqueued job
            
        Raises:
            ValueError: If queue doesn't exist or payload is too large
            RedisError: If Redis operation fails
        """
        # TODO 1: Validate that the queue exists in configuration
        # TODO 2: Create Job object with generated ULID/UUID job_id
        # TODO 3: Validate payload size (args + kwargs + metadata) doesn't exceed max_payload_size
        # TODO 4: Serialize job to JSON string
        # TODO 5: Use Redis pipeline for atomic operations:
        #   - LPUSH to queue:queue_name with serialized job
        #   - HSET to job:job_id with job metadata
        #   - SADD to queues:active with queue name (for monitoring)
        # TODO 6: Return job_id
        pass
    
    def bulk_enqueue(self, jobs: List[Dict[str, Any]]) -> List[str]:
        """
        Atomically enqueue multiple jobs.
        
        Args:
            jobs: List of job dictionaries (same format as enqueue parameters)
            
        Returns:
            List of job IDs in the same order as input jobs
        """
        # TODO 1: Validate all jobs before any Redis operations
        # TODO 2: Create pipeline for atomic bulk insert
        # TODO 3: For each job: validate, create Job object, serialize, add to pipeline
        # TODO 4: Execute pipeline
        # TODO 5: Return list of job IDs
        pass
    
    def queue_stats(self, queue_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Get statistics for specified queues (or all queues).
        
        Returns dictionary with queue names as keys and dicts containing:
        - pending_count: Jobs waiting in queue
        - active_count: Jobs currently being processed
        - scheduled_count: Jobs scheduled for future execution
        - retry_count: Jobs waiting for retry
        - dead_letter_count: Jobs in dead letter queue
        """
        # TODO 1: If queue_names is None, get all configured queue names
        # TODO 2: For each queue, use Redis LLEN for pending count
        # TODO 3: For each queue, use Redis SCARD for active jobs (from job:active:{queue})
        # TODO 4: For each queue, use Redis ZCARD for scheduled and retry counts
        # TODO 5: For each queue, use Redis LLEN for dead letter queue
        # TODO 6: Return aggregated dictionary
        pass
    
    def peek(self, queue: str, count: int = 10) -> List[Job]:
        """
        Peek at next jobs in queue without removing them.
        
        Args:
            queue: Queue name
            count: Maximum number of jobs to return
            
        Returns:
            List of Job objects (deserialized)
        """
        # TODO 1: Validate queue exists
        # TODO 2: Use Redis LRANGE to get first 'count' items from queue
        # TODO 3: Deserialize each item to Job object
        # TODO 4: Return list of Jobs
        pass
```

**Language-Specific Hints (Python):**
- Use `orjson` instead of standard `json` for 2-3x faster serialization (install via `pip install orjson`)
- For ULID generation, use `ulid-py` package: `from ulid import ULID; job_id = str(ULID())`
- Use `redis-py` version 4.0+ for proper async support and connection management
- Implement connection pooling via `redis.ConnectionPool` to reuse connections across components
- Use Python's `signal` module for graceful shutdown: `signal.signal(signal.SIGTERM, shutdown_handler)`
- For cron parsing, consider `croniter` library instead of implementing your own parser

**Milestone Checkpoint for Architecture Foundation:**
After setting up the basic structure and core files, verify your foundation:
1. **Run validation**: `python -c "from src.job_processor.core.job import Job; from src.job_processor.config.settings import SystemConfig; print('Core imports successful')"`
2. **Test Redis connection**: Create a simple test script that instantiates `RedisClient` and calls `health_check()`
3. **Verify serialization roundtrip**: Create a test that creates a `Job`, serializes it, deserializes it, and compares fields
4. **Expected outcome**: All imports succeed, Redis connects (if running locally), and job serialization preserves all data types including datetimes.

**Common Setup Issues:**
- **Redis not running**: Install Redis via `brew install redis` (macOS) or `apt install redis-server` (Linux), then `redis-server`
- **Python path issues**: Install package in development mode: `pip install -e .` from project root
- **Missing dependencies**: Create `requirements.txt` with: `redis>=4.5, pydantic>=2.0, python-dotenv>=1.0, orjson>=3.9`


## Data Model

> **Milestone(s):** This section establishes the foundational data structures for all five milestones. The data model defines how jobs are represented, stored, and transmitted between components, forming the backbone of the entire background job processing system.

The data model is the DNA of our background job processor—it defines how information is structured, stored, and flows through the system. Think of it as the **central filing system** in a busy government office. Each job is like a case file that moves through different departments (queues, workers, retry systems). The filing system must be organized so any clerk (system component) can quickly find the current status of any case, understand its history, and know what to do next. A poorly designed filing system leads to lost cases, duplicated work, and confusion about responsibilities.

This section defines three layers of data organization: the **persistent storage** in Redis (how data is stored on disk/network), the **in-memory objects** (how data is structured while being processed), and the **serialization format** (how data moves between these two layers). Each layer serves distinct purposes with different constraints—Redis storage prioritizes atomic operations and durability, in-memory objects prioritize type safety and business logic, and serialization prioritizes cross-language compatibility and space efficiency.

### Redis Data Structures and Keys

Redis serves as our **central bulletin board and filing cabinet**—a shared space where all components post messages and look up information. Unlike a traditional relational database, Redis is a key-value store with specialized data structures optimized for different access patterns. Our design uses multiple Redis data structures, each chosen for specific operational characteristics that match how we need to access the data.

The key naming convention follows a clear pattern: `{system}:{component}:{identifier}`. This colon-separated hierarchy enables Redis key scanning operations, logical grouping, and automatic key expiration management. All keys use lowercase with underscores for readability.

#### Primary Job Storage

Jobs in the active processing pipeline are stored in Redis lists, which provide O(1) push/pop operations and maintain FIFO ordering:

| Key Pattern | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `job_queue:{queue_name}` | List | Primary job queue storing serialized `Job` objects in FIFO order. Workers pop from the left, enqueuers push to the right. | `job_queue:default`, `job_queue:emails` |
| `job_active` | Set | Set of job IDs currently being processed by workers. Used for crash recovery to detect orphaned jobs. | Contains: `"job_01HXYZ...", "job_01HABC..."` |
| `job_retry_schedule` | Sorted Set | Delayed retry queue with execution timestamp as score. Jobs are moved back to main queue when their scheduled time arrives. | Score: 1678901234.56, Value: serialized job |

Each queue list contains JSON-serialized `Job` objects. The list structure naturally supports FIFO semantics with `LPUSH` (add to end) and `BRPOP` (remove from beginning with blocking). The separate `job_active` set allows us to track which jobs are currently being processed without modifying the original queue entry, enabling crash recovery.

#### Job Metadata and State Tracking

Beyond the queue itself, we maintain comprehensive metadata about each job's lifecycle:

| Key Pattern | Data Type | Description | Expiration |
|-------------|-----------|-------------|------------|
| `job:{job_id}` | Hash | Complete job state including arguments, status, timestamps, and error history. Serves as the system of record. | 7 days after completion |
| `job_status:{job_id}` | String | Current job status as `JobStatus` enum value. Optimized for quick status checks without loading full job. | Same as parent job |
| `job_result:{job_id}` | String | Serialized job result for completed jobs. Stored separately to avoid loading large results during status checks. | 7 days after completion |

The `job:{job_id}` hash contains all mutable job state fields. Storing this separately from the queue payload allows us to update job status and metadata without modifying the serialized job in the queue (which is immutable once enqueued). The hash structure provides O(1) access to individual fields, which is efficient for partial updates.

#### System Management and Monitoring

Operational data enables queue management, worker coordination, and monitoring:

| Key Pattern | Data Type | Description | Purpose |
|-------------|-----------|-------------|---------|
| `worker:heartbeat:{worker_id}` | String | Timestamp of last worker heartbeat. Used to detect dead workers. | Worker liveness detection |
| `worker:current_job:{worker_id}` | String | Job ID currently being processed by worker. Maps worker to active job. | Crash recovery correlation |
| `queue:config:{queue_name}` | Hash | Queue configuration including priority and maximum length. | Runtime queue management |
| `schedule:recurring` | Hash | Recurring job definitions with cron expressions and job templates. | Scheduler configuration |
| `schedule:due` | Sorted Set | Scheduled jobs with execution timestamp as score. | Time-based scheduling |
| `dead_letter:{queue_name}` | List | Jobs that exceeded maximum retry attempts. Requires manual intervention. | Error analysis and manual retry |
| `metrics:queue:{queue_name}:depth` | String | Current queue depth (updated periodically). | Dashboard monitoring |
| `metrics:job:{job_type}:processed` | Counter | Number of jobs processed (incremented atomically). | Performance analytics |

The heartbeat system uses simple string keys with timestamps. When a worker updates its heartbeat, it also sets an automatic expiration (TTL) slightly longer than the heartbeat interval. If the key expires, monitoring systems know the worker has died. This pattern avoids the need for a separate cleanup process.

#### Indexes and Reverse Lookups

To support operational queries ("find all jobs of type X", "find all failed jobs in queue Y"), we maintain secondary indexes:

| Key Pattern | Data Type | Description | Maintenance |
|-------------|-----------|-------------|-------------|
| `index:job_type:{job_type}:{status}` | Set | Set of job IDs for a given job type and status. Enables quick filtering. | Updated on status changes |
| `index:queue:{queue_name}:{status}` | Set | Set of job IDs for a given queue and status. | Updated on status changes |
| `index:time:enqueued:{date}` | Sorted Set | Job IDs indexed by enqueue timestamp. Enables time-range queries. | Score = timestamp |

These indexes are updated atomically with the main job status changes using Redis transactions (`MULTI/EXEC`). While Redis doesn't have automatic indexing like relational databases, this manual indexing pattern enables the query capabilities needed for the monitoring dashboard without expensive full scans.

> **Design Insight:** We separate **job data** (in hashes) from **job placement** (in lists/sorted sets). This is analogous to separating a library's book catalog (metadata about books) from the shelves where books are physically placed (ordered collection). The catalog contains complete information about each book, while the shelf position determines when it will be accessed. This separation allows us to update a job's metadata (like recording an error) without moving it between queues.

#### Redis Key Design ADR

> **Decision: Hierarchical Key Structure with Colon Separators**
> - **Context**: Redis has a flat key namespace, but we need to organize thousands of keys for different purposes (jobs, workers, queues, metrics). We need a structure that supports pattern matching for cleanup operations, logical grouping for debugging, and clear ownership.
> - **Options Considered**:
>   1. Flat names with prefixes: `job_12345`, `worker_abc`
>   2. Hierarchical with colons: `job:12345`, `worker:abc:heartbeat`
>   3. Hierarchical with dots: `job.12345`, `worker.abc.heartbeat`
> - **Decision**: Use hierarchical colon-separated keys (`system:component:id:subcomponent`).
> - **Rationale**: Colons are Redis' conventional hierarchy separator (used in Redis Cluster hash tags). The `KEYS` and `SCAN` commands work naturally with pattern matching (`job:*` finds all job keys). This structure also matches common Redis monitoring tools' expectations. Dots are less conventional in Redis and might conflict with some Redis modules' naming schemes.
> - **Consequences**: Keys are slightly longer but self-documenting. We can delete all keys for a component with `DEL job:*` (though in production we'd use `SCAN` for large datasets). The hierarchy makes debugging easier when inspecting Redis with `redis-cli`.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Flat prefixes | Shorter keys, slightly less memory | No hierarchy, harder to pattern match, mixes different key types | No |
| Colon hierarchy | Standard Redis convention, excellent pattern matching, self-documenting | Slightly longer keys | **Yes** |
| Dot hierarchy | Familiar from domain names, hierarchical | Not standard in Redis, might conflict with modules | No |

### In-Memory Object Types and Schemas

While Redis stores serialized data, our application logic works with rich, typed objects in memory. These objects encapsulate business logic, validation rules, and behavioral methods. Think of them as the **active case files** on a clerk's desk—annotated, with sticky notes, calculations in the margins, and decision logic applied. The in-memory objects transform raw data into intelligent entities that know how to validate themselves, calculate retry delays, and transition between states.

#### Core Job Object

The `Job` class is the central entity representing a unit of work. It follows a **builder pattern** where required fields are set at construction and optional metadata accumulates during execution:

| Field | Type | Description | Immutable? |
|-------|------|-------------|------------|
| `job_id` | `str` | Unique identifier (ULID format). Provides natural time-based ordering. | Yes |
| `job_type` | `str` | Job handler type (e.g., "send_email", "process_image"). Maps to handler class. | Yes |
| `args` | `List[Any]` | Positional arguments for the job handler. | Yes |
| `kwargs` | `Dict[str, Any]` | Keyword arguments for the job handler. | Yes |
| `queue` | `str` | Target queue name (e.g., "default", "high_priority"). | Yes |
| `priority` | `int` | Relative priority within queue (higher = more important). Default: 0. | Yes |
| `max_retries` | `int` | Maximum retry attempts before moving to dead letter queue. Default: 3. | Yes |
| `timeout_seconds` | `int` | Maximum execution time before job is forcibly terminated. Default: 1800. | Yes |
| `created_at` | `datetime` | Job creation timestamp (UTC). | Yes |
| `metadata` | `Dict[str, Any]` | Custom key-value pairs for tracing, feature flags, etc. | No |
| `status` | `JobStatus` | Current lifecycle state (enum). | No |
| `attempts` | `int` | Number of execution attempts so far (starts at 0). | No |
| `errors` | `List[Dict]` | Error history with timestamps, exception details, and stack traces. | No |
| `started_at` | `Optional[datetime]` | When current execution attempt began. `None` if not started. | No |
| `completed_at` | `Optional[datetime]` | When job reached terminal state (completed/failed/dead). `None` if pending. | No |
| `result` | `Optional[Any]` | Job execution result (serializable). Only set for `COMPLETED` jobs. | No |

The `job_id` uses ULID (Universally Unique Lexicographically Sortable Identifier) which provides both uniqueness and time-based ordering. Unlike UUIDv4, ULIDs are sortable by creation time, making time-range queries efficient. The `metadata` field is a flexible bag for cross-cutting concerns like tracing IDs, tenant identifiers, or feature flags—it's mutable because these can be added during processing.

#### Job Status State Machine

The `JobStatus` enum defines the legal states a job can occupy, forming a strict state machine (illustrated in ![Job State Machine](./diagrams/job-state-machine.svg)):

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| `PENDING` | Worker dequeues job | `ACTIVE` | Set `started_at` to current time, increment `attempts` |
| `ACTIVE` | Job completes successfully | `COMPLETED` | Set `completed_at`, store result, clear `started_at` |
| `ACTIVE` | Job fails with retries remaining | `RETRY_SCHEDULED` | Record error, calculate next retry time, schedule in sorted set |
| `ACTIVE` | Job fails with no retries left | `DEAD_LETTER` | Record error, move to dead letter queue, set `completed_at` |
| `RETRY_SCHEDULED` | Retry timer expires | `PENDING` | Move from retry sorted set to main queue |
| `RETRY_SCHEDULED` | Manual retry requested | `PENDING` | Reset error count, move to main queue immediately |
| `DEAD_LETTER` | Manual retry requested | `PENDING` | Reset error count and attempts, move to main queue |
| Any state | Job cancelled | `FAILED` | Record cancellation error, set `completed_at` |

The state machine ensures jobs progress through well-defined paths. Terminal states (`COMPLETED`, `FAILED`, `DEAD_LETTER`) are final—jobs in these states are only kept for history and monitoring. The `FAILED` state is used for immediate failures (like validation errors) that shouldn't retry, while `DEAD_LETTER` is for exhausted retries.

#### Configuration Objects

System configuration is represented as typed objects rather than loose dictionaries, enabling validation and IDE support:

**QueueConfig** defines the behavior of a named queue:
| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `name` | `str` | Queue identifier (must be alphanumeric with underscores). | Required |
| `priority` | `int` | Weight for weighted round-robin polling (higher = more frequent). | 1 |
| `max_length` | `Optional[int]` | Maximum jobs allowed in queue. `None` means unlimited. | `None` |

**WorkerConfig** controls worker process behavior:
| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `queues` | `List[str]` | List of queue names to poll (in priority order). | `["default"]` |
| `concurrency` | `int` | Maximum parallel job executions (threads/processes). | 4 |
| `heartbeat_interval` | `int` | Seconds between heartbeat updates to Redis. | 30 |
| `job_timeout` | `int` | Default job timeout (seconds) if not specified per job. | 1800 |

**RedisConfig** encapsulates Redis connection details:
| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `url` | `str` | Redis connection URL (e.g., "redis://localhost:6379/0"). | Required |
| `max_connections` | `int` | Connection pool maximum size. | 20 |
| `socket_timeout` | `int` | Socket timeout in seconds. | 5 |
| `retry_on_timeout` | `bool` | Automatically retry on connection timeout. | `True` |

**SystemConfig** is the root configuration container:
| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `redis` | `RedisConfig` | Redis connection configuration. | Required |
| `queues` | `List[QueueConfig]` | All queue definitions in the system. | `[]` |
| `workers` | `List[WorkerConfig]` | Worker process configurations. | `[]` |
| `max_payload_size` | `int` | Maximum job payload size in bytes. | 1,048,576 |
| `job_history_size` | `int` | Number of completed jobs to retain in history. | 10,000 |
| `retry_base_delay` | `int` | Base delay in seconds for exponential backoff. | 1 |
| `retry_max_attempts` | `int` | Global default maximum retry attempts. | 3 |

#### Redis Client Wrapper

The `RedisClient` class provides a thread-safe, connection-pooled interface to Redis with automatic error handling and retry logic:

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Redis connection URL (read-only). |
| `connection_kwargs` | `Dict[str, Any]` | Additional connection parameters (read-only). |
| `_client` | `Optional[redis.Redis]` | Private Redis client instance (lazy-initialized). |

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_instance` | `url: Optional[str] = None, **kwargs` | `RedisClient` | Singleton factory method ensuring single connection pool per configuration. |
| `get_client` | None | `redis.Redis` | Returns underlying Redis client (initializes if needed). |
| `execute` | `command: str, *args, **kwargs` | `Any` | Executes Redis command with automatic connection retry and error logging. |
| `pipeline` | None | `redis.Pipeline` | Returns Redis pipeline for atomic multi-command execution. |

The singleton pattern ensures all parts of the application share the same connection pool, preventing connection exhaustion. The `execute` method wraps all Redis operations with consistent error handling—transient network errors trigger retries with exponential backoff, while logical errors (like wrong command syntax) raise immediate exceptions.

#### Object Relationships

The data model relationships form a clear ownership hierarchy (as shown in ![Data Model Relationships](./diagrams/data-model-relationships.svg)):

1. **SystemConfig owns RedisConfig, multiple QueueConfigs, and multiple WorkerConfigs**
2. **Each Job references a QueueConfig by name**
3. **Each WorkerConfig references multiple QueueConfigs by name**
4. **RedisClient is a singleton service used by all components**
5. **Job objects are created by producers, stored in Redis, loaded by workers, and updated throughout lifecycle**

This structure minimizes coupling—components only need references to what they directly use. For example, a worker only needs its `WorkerConfig` and a `RedisClient` instance; it doesn't need the full `SystemConfig`.

### Job Serialization and Encoding

Serialization is the **language translator** between our Python objects and Redis storage. It must preserve all data types, handle versioning, and be efficient for both small and large payloads. We face the classic tension between human readability (JSON) and efficiency (binary formats). Our design chooses JSON as the primary format with strict schema validation, augmented with custom encoders for special types like datetimes.

#### Serialization Format ADR

> **Decision: JSON with Custom Type Encoders**
> - **Context**: Jobs contain diverse data types (strings, numbers, lists, dicts, datetimes, optional values). We need a serialization format that is human-readable for debugging, widely supported across languages (in case we add Go or Rust workers later), and efficient enough for typical job sizes (< 1MB).
> - **Options Considered**:
>   1. **JSON**: Human-readable, universal support, but limited type system (no datetime, no binary data)
>   2. **MessagePack**: Binary, compact, preserves some type hints, but less human-readable
>   3. **Protocol Buffers**: Strong typing, versioning, compact binary, but requires schema compilation
> - **Decision**: Use JSON with custom encoders/decoders for extended types (datetime, bytes as base64).
> - **Rationale**: Debuggability is critical during development—engineers need to inspect queues with `redis-cli` and understand the data. JSON's ubiquity means any language can eventually consume our queues. The 1MB payload limit means space efficiency is secondary. We can always add MessagePack as an optimization later while keeping JSON as a fallback.
> - **Consequences**: Slightly larger payloads than binary formats. Datetimes are serialized to ISO 8601 strings. Binary data must be base64-encoded. We must implement robust custom encoders/decoders.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| JSON | Human-readable, universal, no schema needed, easy debugging | Larger size, limited types, slower parsing than binary | **Primary** |
| MessagePack | Compact, preserves some types, fast parsing | Not human-readable, less universal support | Fallback option |
| Protocol Buffers | Strong typing, versioning, compact | Schema management overhead, compilation step | No |

#### Serialization Schema

The JSON schema for serialized jobs follows a strict structure with versioning for future evolution:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["v", "job_id", "job_type", "args", "kwargs", "queue"],
  "properties": {
    "v": {"type": "integer", "const": 1},
    "job_id": {"type": "string", "pattern": "^[0-9A-Z]{26}$"},
    "job_type": {"type": "string", "maxLength": 100},
    "args": {"type": "array", "maxItems": 100},
    "kwargs": {"type": "object", "maxProperties": 50},
    "queue": {"type": "string", "maxLength": 50},
    "priority": {"type": "integer", "minimum": -100, "maximum": 100},
    "max_retries": {"type": "integer", "minimum": 0, "maximum": 100},
    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
    "created_at": {"type": "string", "format": "date-time"},
    "metadata": {"type": "object", "additionalProperties": true},
    "status": {"enum": ["PENDING", "ACTIVE", "COMPLETED", "FAILED", "RETRY_SCHEDULED", "DEAD_LETTER"]},
    "attempts": {"type": "integer", "minimum": 0},
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["timestamp", "exception", "message"],
        "properties": {
          "timestamp": {"type": "string", "format": "date-time"},
          "exception": {"type": "string"},
          "message": {"type": "string"},
          "traceback": {"type": "string"}
        }
      }
    },
    "started_at": {"type": ["string", "null"], "format": "date-time"},
    "completed_at": {"type": ["string", "null"], "format": "date-time"},
    "result": {}
  },
  "additionalProperties": false
}
```

The `v` field (version) allows future schema migrations—if we need to change the format, we can detect version 1 and apply migration logic. All fields have reasonable size limits to prevent abuse (e.g., `args` limited to 100 items, `kwargs` to 50 properties).

#### Type Encoding Rules

Special Python types require custom encoding rules to survive JSON serialization:

| Python Type | JSON Representation | Notes |
|-------------|---------------------|-------|
| `datetime` | ISO 8601 string with timezone (e.g., "2023-01-15T12:30:45.123456Z") | Always UTC, microsecond precision |
| `bytes` | Base64-encoded string | Used sparingly; large binaries should be stored elsewhere |
| `Decimal` | String representation | Preserves exact decimal values without float rounding |
| `set` | Array | Order is not preserved (sets are unordered) |
| `frozenset` | Array | Same as set |
| `tuple` | Array | Preserved as list on deserialization (JSON has no tuple type) |
| `Enum` | String value (`.value` property) | Requires Enum to have string values |
| `UUID` | Canonical string representation (8-4-4-4-12) | Lowercase, no braces |
| `None` | `null` | Standard JSON null |

The serialization system uses a custom JSON encoder class that extends `json.JSONEncoder`, overriding the `default` method to handle these special types. During deserialization, we use a custom object hook that recognizes encoded types by structure (e.g., a dict with `{"__type__": "datetime", "value": "..."}`) or by convention (ISO datetime strings can be auto-detected by regex).

#### Serialization Methods

The `Job` class provides symmetric serialization methods that handle the complete round-trip:

| Method | Parameters | Returns | Algorithm |
|--------|------------|---------|-----------|
| `to_dict` | None | `Dict[str, Any]` | 1. Create dict with all job fields 2. Convert special types using encoding rules 3. Add version field `"v": 1` |
| `from_dict` | `data: Dict[str, Any]` | `Job` | 1. Validate required fields exist 2. Check version compatibility 3. Decode special types using decoding rules 4. Return new Job instance |
| `serialize` | None | `str` | 1. Call `to_dict()` 2. Convert dict to JSON string with compact separation 3. Validate size against `max_payload_size` |
| `deserialize` | `data: str` | `Job` | 1. Parse JSON string to dict 2. Call `from_dict()` with parsed data |
| `record_error` | `error: Exception` | `None` | 1. Create error dict with timestamp, exception class name, message, and traceback 2. Append to `errors` list 3. Update status if needed |
| `should_retry` | None | `bool` | 1. Return `True` if status is `FAILED` and `attempts < max_retries` 2. Return `False` otherwise |

The `serialize` method includes size validation—if the serialized JSON exceeds `max_payload_size` (default 1MB), it raises a `JobTooLargeError` before attempting to store in Redis. This prevents queue clogging with oversized jobs.

#### Common Pitfalls in Data Modeling

⚠️ **Pitfall: Storing unserializable objects in job arguments**
- **Description**: Passing database connections, file handles, or complex ORM objects in `args` or `kwargs`.
- **Why it's wrong**: These objects can't be serialized to JSON. The job fails during enqueue with cryptic serialization errors, or worse, serializes to something meaningless that causes runtime failures.
- **Fix**: Only pass primitive types, dicts, lists, or simple data objects. If you need database access, pass record IDs and fetch fresh connections in the job handler.

⚠️ **Pitfall: Modifying serialized job in queue**
- **Description**: Trying to update a job's properties after it's been enqueued by modifying the JSON in the Redis list.
- **Why it's wrong**: The job in the queue is immutable by design. Multiple workers might have already read the serialized data, and updates won't be reflected. The correct place for mutable state is the `job:{id}` hash.
- **Fix**: Store mutable state in the job hash, not in the queue payload. Use atomic updates to the hash when job state changes.

⚠️ **Pitfall: Forgetting timezone awareness**
- **Description**: Storing datetimes as naive (timezone-unaware) objects or using local system time.
- **Why it's wrong**: Workers running in different timezones will interpret scheduled times differently. Daylight saving transitions cause duplicates or gaps.
- **Fix**: Always use UTC for storage. Convert to local time only for display. Use `datetime.utcnow()` not `datetime.now()`.

⚠️ **Pitfall: Inconsistent ULID generation**
- **Description**: Generating ULIDs on different machines without synchronized clocks or using random components incorrectly.
- **Why it's wrong**: ULIDs lose their time-sortable property if clocks are skewed. Jobs might appear out of order.
- **Fix**: Use a ULID library that handles clock synchronization (monotonic counter for same-millisecond collisions). Ensure system clocks are synchronized with NTP.

⚠️ **Pitfall: Not planning for schema evolution**
- **Description**: Serializing jobs without version field, assuming the schema will never change.
- **Why it's wrong**: When you need to add a new field or change encoding, existing jobs in queues become unreadable.
- **Fix**: Include version field (`"v": 1`) in all serialized jobs. Write migration code that can upgrade old versions on deserialization.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| JSON Serialization | Python's built-in `json` module with custom encoder | `orjson` for 10x faster serialization (C-optimized) |
| ULID Generation | `python-ulid` library (pure Python) | `ulid-py` with C extensions for performance |
| Redis Client | `redis-py` with connection pooling | `redis-py` with client-side caching enabled |
| Configuration | `pydantic` for validation and type checking | Custom YAML/TOML loader with environment variable override |
| Date/Time | `pendulum` for timezone-aware operations | Python 3.9+ `zoneinfo` with `pytz` compatibility layer |

#### Recommended File/Module Structure

```
background_jobs/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── models.py           # QueueConfig, WorkerConfig, RedisConfig, SystemConfig
│   └── loader.py           # SystemConfig.from_env(), from_yaml()
├── models/
│   ├── __init__.py
│   ├── job.py              # Job class with serialization methods
│   ├── job_status.py       # JobStatus enum
│   └── errors.py           # Custom exceptions (JobTooLargeError, etc.)
├── storage/
│   ├── __init__.py
│   ├── redis_client.py     # RedisClient singleton wrapper
│   ├── serialization.py    # Custom JSON encoder/decoder for special types
│   └── keys.py             # Key generation utilities (e.g., job_key(), queue_key())
└── utils/
    ├── __init__.py
    ├── ulid_generator.py   # ULID generation with monotonic counter
    └── validation.py       # Payload size validation, queue name validation
```

#### Infrastructure Starter Code

**Complete Redis Client Wrapper** (`storage/redis_client.py`):
```python
import redis
from typing import Optional, Any, Dict
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class RedisClient:
    """Thread-safe Redis client singleton with connection pooling."""
    
    _instances: Dict[str, 'RedisClient'] = {}
    _lock: Lock = Lock()
    
    def __init__(self, url: str, **connection_kwargs):
        self.url = url
        self.connection_kwargs = connection_kwargs
        self._client: Optional[redis.Redis] = None
        
    @classmethod
    def get_instance(cls, url: Optional[str] = None, **kwargs) -> 'RedisClient':
        """Get singleton instance for given URL and connection parameters."""
        if url is None:
            url = "redis://localhost:6379/0"
            
        instance_key = f"{url}:{str(kwargs)}"
        
        with cls._lock:
            if instance_key not in cls._instances:
                cls._instances[instance_key] = cls(url, **kwargs)
            return cls._instances[instance_key]
    
    def get_client(self) -> redis.Redis:
        """Lazy initialization of Redis client with connection pool."""
        if self._client is None:
            self._client = redis.from_url(
                self.url,
                **self.connection_kwargs,
                decode_responses=True,  # Auto-decode bytes to str
                health_check_interval=30,
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.url}")
        return self._client
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command with automatic retry on connection errors."""
        client = self.get_client()
        try:
            # Get the method from client (e.g., client.get, client.set)
            method = getattr(client, command)
            return method(*args, **kwargs)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection error: {e}, retrying...")
            # Reset client to force reconnection
            self._client = None
            client = self.get_client()
            method = getattr(client, command)
            return method(*args, **kwargs)
    
    def pipeline(self) -> redis.Pipeline:
        """Return a Redis pipeline for atomic operations."""
        client = self.get_client()
        return client.pipeline()
    
    def close(self):
        """Close Redis connection (called during graceful shutdown)."""
        if self._client:
            self._client.close()
            self._client = None
```

**Complete Custom JSON Encoder/Decoder** (`storage/serialization.py`):
```python
import json
import base64
from datetime import datetime, date, time
from decimal import Decimal
from uuid import UUID
from enum import Enum
from typing import Any, Dict

class JobEncoder(json.JSONEncoder):
    """Custom JSON encoder for Job serialization."""
    
    def default(self, obj: Any) -> Any:
        # Handle datetime objects (convert to ISO format with timezone)
        if isinstance(obj, datetime):
            # Ensure datetime is timezone-aware (UTC)
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.isoformat()
        
        # Handle date objects
        if isinstance(obj, date):
            return obj.isoformat()
        
        # Handle bytes (base64 encode)
        if isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "value": base64.b64encode(obj).decode('ascii')
            }
        
        # Handle Decimal (string representation)
        if isinstance(obj, Decimal):
            return str(obj)
        
        # Handle UUID
        if isinstance(obj, UUID):
            return str(obj)
        
        # Handle Enum (store value)
        if isinstance(obj, Enum):
            return obj.value
        
        # Let base class handle other types (will raise TypeError)
        return super().default(obj)

def job_object_hook(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Custom object hook for JSON decoding to restore special types."""
    # Check for encoded bytes
    if "__type__" in dct and dct["__type__"] == "bytes":
        return base64.b64decode(dct["value"])
    
    # Auto-detect ISO datetime strings (naive or with timezone)
    for key, value in dct.items():
        if isinstance(value, str):
            # Try to parse as datetime (ISO format)
            try:
                # Check if it looks like an ISO datetime
                if "T" in value and ("-" in value or ":" in value):
                    # Parse with datetime.fromisoformat (Python 3.7+)
                    # Note: This handles timezone-aware strings in Python 3.11+
                    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    dct[key] = dt
            except (ValueError, AttributeError):
                pass
    
    return dct

def serialize_job(job_dict: Dict[str, Any]) -> str:
    """Serialize job dictionary to JSON string with custom encoding."""
    return json.dumps(job_dict, cls=JobEncoder, separators=(',', ':'))

def deserialize_job(json_str: str) -> Dict[str, Any]:
    """Deserialize JSON string to job dictionary with custom decoding."""
    return json.loads(json_str, object_hook=job_object_hook)
```

#### Core Logic Skeleton Code

**Job Class with Serialization Methods** (`models/job.py`):
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from enum import Enum
from .job_status import JobStatus
from storage.serialization import serialize_job, deserialize_job

class Job:
    """Core job representation with serialization capabilities."""
    
    def __init__(
        self,
        job_id: str,
        job_type: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        queue: str,
        priority: int = 0,
        max_retries: int = 3,
        timeout_seconds: int = 1800,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status: JobStatus = JobStatus.PENDING,
        attempts: int = 0,
        errors: Optional[List[Dict[str, Any]]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        result: Optional[Any] = None,
    ):
        # TODO 1: Validate required fields (job_id, job_type, queue)
        # TODO 2: Set created_at to UTC now if not provided
        # TODO 3: Initialize metadata as empty dict if None
        # TODO 4: Initialize errors as empty list if None
        # TODO 5: Store all parameters as instance attributes
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        # TODO 1: Create dictionary with all instance attributes
        # TODO 2: Convert datetime objects using ISO format
        # TODO 3: Add version field "v": 1 for schema evolution
        # TODO 4: Handle Optional fields (skip if None or use null)
        # TODO 5: Return dictionary ready for JSON serialization
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Reconstruct job from dictionary."""
        # TODO 1: Validate required fields exist in data
        # TODO 2: Check version compatibility (data.get("v") == 1)
        # TODO 3: Convert string dates back to datetime objects
        # TODO 4: Create and return new Job instance with loaded data
        # TODO 5: Handle missing optional fields with defaults
        pass
    
    def serialize(self) -> str:
        """Serialize job to JSON string."""
        # TODO 1: Call to_dict() to get dictionary representation
        # TODO 2: Use serialize_job() helper to convert to JSON string
        # TODO 3: Validate size against max_payload_size (1MB default)
        # TODO 4: Raise JobTooLargeError if payload exceeds limit
        # TODO 5: Return the JSON string
        pass
    
    @classmethod
    def deserialize(cls, data: str) -> 'Job':
        """Deserialize job from JSON string."""
        # TODO 1: Use deserialize_job() helper to parse JSON string
        # TODO 2: Call from_dict() with the parsed dictionary
        # TODO 3: Return the reconstructed Job instance
        # TODO 4: Handle JSON parsing errors with descriptive exceptions
        pass
    
    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during job execution."""
        # TODO 1: Create error dictionary with keys:
        #   - timestamp: current UTC datetime
        #   - exception: error class name (type(error).__name__)
        #   - message: str(error)
        #   - traceback: formatted traceback string
        # TODO 2: Append error dictionary to self.errors list
        # TODO 3: Update self.status based on retry logic
        # TODO 4: Increment self.attempts counter
        pass
    
    def should_retry(self) -> bool:
        """Determine if job should be retried."""
        # TODO 1: Check if status is FAILED (not DEAD_LETTER)
        # TODO 2: Compare attempts to max_retries
        # TODO 3: Return True if attempts < max_retries
        # TODO 4: Return False otherwise
        pass
    
    def calculate_next_retry(self, base_delay: int = 1) -> datetime:
        """Calculate next retry time using exponential backoff."""
        # TODO 1: Compute delay = base_delay * (2 ** (self.attempts - 1))
        # TODO 2: Add optional jitter (±10% of delay)
        # TODO 3: Return current time + delay as datetime
        pass
```

**Configuration Models** (`config/models.py`):
```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class QueueConfig:
    """Configuration for a named queue."""
    name: str
    priority: int = 1
    max_length: Optional[int] = None
    
    def __post_init__(self):
        # TODO 1: Validate queue name (alphanumeric + underscores)
        # TODO 2: Validate priority is non-negative integer
        # TODO 3: Validate max_length is positive if not None
        pass

@dataclass
class WorkerConfig:
    """Configuration for a worker process."""
    queues: List[str] = field(default_factory=lambda: ["default"])
    concurrency: int = 4
    heartbeat_interval: int = 30
    job_timeout: int = 1800
    
    def __post_init__(self):
        # TODO 1: Validate queues list is non-empty
        # TODO 2: Validate concurrency is positive integer
        # TODO 3: Validate heartbeat_interval is between 5 and 300 seconds
        # TODO 4: Validate job_timeout is positive integer
        pass

@dataclass  
class RedisConfig:
    """Redis connection configuration."""
    url: str
    max_connections: int = 20
    socket_timeout: int = 5
    retry_on_timeout: bool = True
    
    def __post_init__(self):
        # TODO 1: Validate URL format (starts with redis://)
        # TODO 2: Validate max_connections is positive
        # TODO 3: Validate socket_timeout is positive
        pass

@dataclass
class SystemConfig:
    """Root system configuration."""
    redis: RedisConfig
    queues: List[QueueConfig] = field(default_factory=list)
    workers: List[WorkerConfig] = field(default_factory=list)
    max_payload_size: int = 1_048_576  # 1MB
    job_history_size: int = 10_000
    retry_base_delay: int = 1
    retry_max_attempts: int = 3
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load configuration from environment variables with defaults."""
        # TODO 1: Read REDIS_URL from environment (default: redis://localhost:6379/0)
        # TODO 2: Parse queue definitions from QUEUES environment variable
        # TODO 3: Parse worker configurations from WORKERS environment variable
        # TODO 4: Create RedisConfig instance
        # TODO 5: Create QueueConfig instances from parsed data
        # TODO 6: Create WorkerConfig instances from parsed data
        # TODO 7: Return SystemConfig instance with all components
        pass
```

#### Language-Specific Hints

1. **Use Python 3.8+** for dataclasses with `field(default_factory=...)` support
2. **For ULIDs**, install `python-ulid` package: `pip install python-ulid`
3. **Use `datetime.timezone.utc`** for timezone-aware UTC timestamps (Python 3.2+)
4. **For JSON serialization**, `json.dumps()` with `default` parameter handles custom types
5. **Use `__post_init__` method** in dataclasses for validation logic
6. **Implement `__hash__` and `__eq__`** for Job if you need to store in sets/dicts
7. **Use `functools.lru_cache`** for expensive computations like cron expression parsing
8. **For thread safety**, use `threading.Lock` in singleton patterns

#### Milestone Checkpoint

After implementing the data model (Milestone 1 foundation), run this validation script:

```bash
# Test job serialization round-trip
python -m pytest tests/test_job_serialization.py -v

# Expected output:
# test_job_to_dict_roundtrip ... PASSED
# test_job_serialize_deserialize ... PASSED  
# test_job_with_special_types ... PASSED
# test_job_size_validation ... PASSED
```

Manual verification steps:
1. Create a Job with various data types (string, int, list, dict, datetime)
2. Call `job.serialize()` - should return JSON string
3. Call `Job.deserialize()` on that string - should return identical Job
4. Check that `job.job_id` is a valid ULID (26 characters, alphanumeric)
5. Verify that datetimes are stored in ISO format with timezone indicator
6. Attempt to serialize a 2MB payload - should raise `JobTooLargeError`

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| JSON serialization fails with "TypeError: Object of type datetime is not JSON serializable" | Forgot to use custom encoder or passed datetime directly to `json.dumps()` | Check if you're using `JobEncoder` or calling `serialize()` method | Use `job.serialize()` instead of `json.dumps(job)` |
| Job appears in queue but workers can't process it | Job schema changed or version mismatch | Check Redis: `redis-cli LRANGE job_queue:default 0 0` to see raw JSON | Ensure all components use same schema version |
| ULIDs not sorting chronologically | Clock skew between machines or incorrect ULID generation | Compare ULID timestamps with system clock | Use NTP for clock sync, ensure ULID lib uses monotonic counter |
| Datetimes showing wrong timezone | Naive datetime objects or mixing UTC/local time | Check `created_at` field in serialized JSON for timezone indicator | Always use `datetime.now(timezone.utc)` not `datetime.utcnow()` |
| Redis connections exhausted | Creating new RedisClient instance per job instead of singleton | Check Redis `CLIENT LIST` command - too many connections | Use `RedisClient.get_instance()` singleton pattern |


## Component Design: Job Queue Core

> **Milestone(s):** Milestone 1: Job Queue Core. This section provides the complete design for the foundational queueing system that enables producers to enqueue jobs and workers to retrieve them. It establishes the core job representation, queue management, and priority handling that all other components build upon.

### Mental Model: Post Office Sorting Facility

Imagine a large, well-organized post office that handles packages (jobs) from senders (producers) to recipients (job handlers). This mental model helps visualize the core queueing concepts:

- **Mail Collection Point (Enqueue Interface):** Senders bring packages to the counter. Each package has a destination address (queue name) and a priority sticker (priority). The clerk (Queue Manager) validates the package isn't too large or improperly addressed, then stamps it with a unique tracking number (job ID) and places it in the correct sorting bin.

- **Sorting Bins (Queues in Redis):** The post office has multiple wall-mounted bins, each labeled with a destination city (queue name). Some bins are marked "Express" (high priority) while others are "Standard" (normal priority). Within each bin, packages are stacked in order of arrival (FIFO). The facility has a policy that bins can only hold a certain number of packages to prevent overloading.

- **Sorting Machine (Priority Algorithm):** A sorting machine (worker polling algorithm) continuously checks bins for packages to process. It doesn't just check bins in random order—it has a weighted schedule: it checks the "Express" bins twice as often as "Standard" bins. This ensures urgent packages move faster without completely starving regular ones.

- **Package Manifest (Job Serialization):** Each package contains a detailed manifest (serialized job payload) listing the contents (arguments), handling instructions (job type), and special requirements (timeout, retries). The manifest uses a standardized format (JSON) that anyone in the postal system can read, ensuring packages can be processed by any worker station.

This mental model emphasizes several critical design aspects: **validation at entry** prevents problems downstream, **standardized packaging** enables interoperability, **organized storage** ensures efficient retrieval, and **intelligent sorting** respects priority without starvation. The post office doesn't process packages itself—it merely stores and routes them to available workers, just as our queue core doesn't execute jobs but manages their lifecycle until pickup.

### Queue Manager Interface

The Queue Manager is the central orchestration component that mediates all job enqueue operations. It provides a clean, abstract interface for producers while handling the complex details of validation, serialization, and atomic Redis operations. Think of it as the "front desk" of our post office—all interactions with the queue system go through this interface.

The Queue Manager's primary responsibilities are:
1. **Job Validation:** Ensuring jobs meet size and schema constraints before acceptance
2. **Serialization:** Converting in-memory `Job` objects to a storage-ready format
3. **Atomic Enqueue:** Safely adding jobs to Redis with proper priority handling
4. **Queue Inspection:** Providing visibility into queue state without modifying it
5. **Configuration Management:** Applying queue-specific rules (max length, priority)

The complete interface is defined in the following table:

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `enqueue_job` | `job: Job` | `str` (job_id) | Validates, serializes, and atomically adds a job to the appropriate Redis queue. Returns the unique job ID. Raises `ValidationError` for oversized/malformed jobs, `QueueFullError` if queue at capacity. |
| `bulk_enqueue` | `jobs: List[Job]` | `List[str]` (job_ids) | Enqueues multiple jobs in a single atomic transaction using Redis pipeline. Returns list of job IDs in same order. If any job fails validation, the entire operation rolls back. |
| `get_queue_length` | `queue_name: str` | `int` | Returns the current number of pending jobs in the specified queue. Counts only jobs in the main pending queue (not active, retry, or dead letter). |
| `peek_queue` | `queue_name: str`, `count: int = 10` | `List[Job]` | Retrieves up to `count` jobs from the front of the queue without removing them. Useful for debugging and monitoring. Jobs are deserialized back to `Job` objects. |
| `list_queues` | `include_internal: bool = False` | `List[str]` | Returns names of all configured queues. When `include_internal` is True, also returns system queues (retry, dead letter, scheduler). |
| `delete_job` | `job_id: str`, `queue_name: str` | `bool` | Removes a specific job from a queue if it exists. Used for manual job management. Returns True if job was found and removed. |
| `get_job_by_id` | `job_id: str` | `Optional[Job]` | Retrieves a job from Redis by its ID, regardless of which queue or state it's in. Returns None if job doesn't exist. |
| `validate_job` | `job: Job` | `None` | Internal method that checks job size (against `max_payload_size`), required fields, and serializability. Raises `ValidationError` with specific reason. |

The Queue Manager maintains an internal mapping of queue configurations (`QueueConfig` objects) that define each queue's behavior. When a job is enqueued, the manager:

1. Validates the job payload size and structure
2. Generates a unique job ID (ULID for time-ordered uniqueness)
3. Sets the job's initial status to `PENDING` and timestamps
4. Serializes the job to JSON format
5. Atomically pushes the serialized job to the appropriate Redis list
6. Updates queue metrics for monitoring

This interface deliberately exposes only queue management operations—not worker operations like dequeue, which are handled by the Worker component. This separation of concerns ensures the queue manager remains focused on producer-side concerns while workers handle consumer-side logic.

### Enqueue and Priority Algorithm

The algorithm for enqueuing jobs with priority support is more nuanced than a simple `LPUSH` to a Redis list. Our system supports multiple named queues with configurable priority weights, which means workers should poll higher-priority queues more frequently. However, we must maintain strict FIFO ordering *within* each queue. The algorithm achieves this through a combination of Redis operations and weighted polling logic in workers.

#### Step-by-Step Enqueue Algorithm

When a producer calls `enqueue_job(job)`:

1. **Validation Phase:**
   - Check if job payload size exceeds `SystemConfig.max_payload_size` (default 1MB)
   - Verify all required `Job` fields are present: `job_type`, `args`, `kwargs`, `queue`
   - Ensure the target queue exists in configuration (or create with default priority)
   - If validation fails, raise `ValidationError` with specific message

2. **Job Preparation:**
   - Generate job ID using ULID (Universally Unique Lexicographically Sortable Identifier)
     - ULID provides time-ordered uniqueness: `01J5XZR0W3F6Y2A4B8C9D10E11` (26 chars)
     - First 10 characters are timestamp (milliseconds since epoch in base32)
     - Remaining 16 characters are random bytes, ensuring uniqueness even at same millisecond
   - Set `job.job_id` to generated ULID
   - Set `job.status` to `PENDING`
   - Set `job.created_at` to current UTC datetime
   - Initialize `job.attempts = 0`, `job.errors = []`
   - Set `job.started_at`, `job.completed_at`, `job.result` to `None`

3. **Serialization:**
   - Convert `Job` object to dictionary via `Job.to_dict()`
   - Convert datetime objects to ISO 8601 strings for JSON compatibility
   - Serialize dictionary to JSON string via `Job.serialize()`
   - Example serialized job:
     ```json
     {
       "job_id": "01J5XZR0W3F6Y2A4B8C9D10E11",
       "job_type": "process_image",
       "args": ["photo.jpg", {"width": 800, "height": 600}],
       "kwargs": {"quality": 90},
       "queue": "image_processing",
       "priority": 2,
       "max_retries": 3,
       "timeout_seconds": 300,
       "created_at": "2024-01-15T10:30:00Z",
       "metadata": {"user_id": 12345, "batch_id": "batch-001"},
       "status": "PENDING",
       "attempts": 0,
       "errors": [],
       "started_at": null,
       "completed_at": null,
       "result": null
     }
     ```

4. **Atomic Enqueue to Redis:**
   - Use Redis pipeline for atomic operation sequence:
     1. `HSET job:{job_id} {serialized_job}` - Store job in Redis hash for later retrieval
     2. `EXPIRE job:{job_id} {job_ttl}` - Set TTL (e.g., 7 days) to prevent memory bloat
     3. `LPUSH queue:{queue_name} {job_id}` - Add job ID to the queue's list
     4. `INCR queue:{queue_name}:count` - Increment queue length metric
   - If any step fails, the entire pipeline is rolled back
   - Return the generated job ID to the producer

#### Priority Weighted Polling Algorithm (Worker-Side)

While the enqueue algorithm handles producer-side operations, the priority system requires special consideration for how workers poll queues. The Queue Manager doesn't implement this directly, but it influences the algorithm through queue configuration. Here's how priority-weighted polling works:

1. **Queue Configuration Example:**
   ```python
   queues = [
       QueueConfig(name="critical", priority=3),    # High priority
       QueueConfig(name="default", priority=2),     # Medium priority  
       QueueConfig(name="low", priority=1),         # Low priority
       QueueConfig(name="reports", priority=1)      # Also low priority
   ]
   ```

2. **Worker Polling Logic:**
   - Workers maintain a list of queues sorted by priority weight
   - They calculate polling frequency: `poll_count = priority_weight / total_weight`
   - For the above configuration:
     - Total weight = 3 + 2 + 1 + 1 = 7
     - "critical" gets 3/7 ≈ 43% of polls
     - "default" gets 2/7 ≈ 29% of polls
     - "low" and "reports" each get 1/7 ≈ 14% of polls
   - Workers use Redis `BRPOP` with timeout, cycling through queues according to these weights

This approach ensures higher-priority queues are checked more frequently while maintaining FIFO within each queue. It's a trade-off: true priority queuing (where a high-priority job jumps ahead of all lower-priority jobs) would require a single sorted data structure, but that's more complex and reduces throughput. Our approach provides good enough priority handling for most use cases while maintaining simplicity and performance.

> **Key Insight:** We separate job storage (Redis hash) from queue ordering (Redis list). The list contains only job IDs, not full job data. This reduces memory usage when moving jobs between queues (e.g., to retry queue) and allows efficient job lookup by ID. The trade-off is an extra Redis operation to fetch job data when a worker dequeues a job ID.

### ADR: Job Storage Mechanism

> **Decision: Separate Job Storage from Queue Ordering**
>
> **Context:** We need to store complete job data (arguments, metadata, status history) while also maintaining queue ordering for efficient worker polling. Jobs may move between multiple queues during their lifecycle (pending → active → retry → dead letter), and we need efficient access to job data by ID for monitoring and management.
>
> **Options Considered:**
> 1. **Store full job data in queue lists:** Each Redis list element contains the complete serialized job.
> 2. **Separate storage with ID pointers:** Store jobs in Redis hashes keyed by job ID, with queue lists containing only IDs.
> 3. **Hybrid with metadata in lists:** Store minimal metadata in lists with full data in separate storage.
>
> **Decision:** Option 2 - Separate storage with ID pointers. Store complete job data in Redis hashes (`job:{job_id}`) and queue job IDs in Redis lists (`queue:{queue_name}`).
>
> **Rationale:** This separation provides multiple benefits: (1) Moving jobs between queues is cheap—just move the ID, not the full payload; (2) Job lookup by ID is O(1) via hash get; (3) We can store additional job metadata in the hash without affecting queue operations; (4) Serialized job data is stored only once, reducing memory usage; (5) It aligns with Redis best practices for modeling relationships.
>
> **Consequences:** Positive: Efficient queue operations and job management. Negative: Requires two Redis operations to dequeue (get ID from list, then get data from hash). This is acceptable because the extra operation's cost is minimal compared to job execution time, and we can mitigate it with pipelining.

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| Full data in lists | Single operation to dequeue, simpler implementation | Moving jobs between queues requires re-serializing and copying data, memory inefficient with large payloads, no efficient job-by-ID lookup | Too inefficient for job lifecycle management |
| **Separate storage with IDs** | **Efficient job movement, O(1) job lookup, memory efficient, flexible metadata** | **Requires two operations to dequeue** | **Chosen - benefits outweigh minor performance cost** |
| Hybrid (metadata in lists) | Balances lookup efficiency with queue performance | Complex serialization/deserialization logic, still requires separate storage for full job data | Adds complexity without clear benefits over option 2 |

### ADR: Queue Priority Strategy

> **Decision: Weighted Round-Robin Polling**
>
> **Context:** We need to support multiple queues with different priority levels. High-priority jobs should be processed sooner than low-priority ones, but we must avoid complete starvation of low-priority queues. The system should be predictable and configurable.
>
> **Options Considered:**
> 1. **Strict priority polling:** Always check highest-priority queue first, only check lower queues when higher ones are empty.
> 2. **Weighted random selection:** Randomly select queue based on weights each poll.
> 3. **Weighted round-robin:** Systematically cycle through queues, visiting higher-priority queues more frequently.
> 4. **Multiple Redis sorted sets:** Single data structure with priority scores, requiring atomic ZADD and ZRANGE operations.
>
> **Decision:** Option 3 - Weighted round-robin polling implemented in workers.
>
> **Rationale:** Strict priority causes starvation. Weighted random lacks predictability. Multiple sorted sets add Redis complexity and atomicity challenges. Weighted round-robin provides predictable, configurable priority handling without starvation. It's simple to implement and understand: if queue A has priority 3 and queue B has priority 1, workers will check A three times for every one time they check B. This gives priority jobs 3x more polling attention while still guaranteeing B gets processed eventually.
>
> **Consequences:** Positive: Predictable priority handling, no starvation, simple implementation. Negative: Not true priority queuing (a high-priority job enqueued after low-priority jobs in a different queue might still wait). This is acceptable for most background job scenarios where approximate priority is sufficient.

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| Strict priority polling | Highest priority jobs always processed first | Complete starvation of low-priority queues, poor resource utilization | Unacceptable starvation risk |
| Weighted random selection | Simple to implement, probabilistic priority | Unpredictable timing, can still starve low-priority queues by chance | Lack of predictability problematic |
| **Weighted round-robin** | **Predictable, configurable, no starvation, simple** | **Not true priority (FIFO within queue only)** | **Chosen - best balance of fairness and priority** |
| Multiple Redis sorted sets | True priority across all jobs, single atomic operation | Complex implementation, Redis contention, harder to debug | Over-engineering for most use cases |

### Common Pitfalls in Queue Implementation

⚠️ **Pitfall: Not Validating Payload Size Before Redis Operations**

**Description:** Checking payload size only after serialization or during Redis operations. If a 5MB payload is serialized to JSON (making it even larger) and only rejected when Redis fails with "OOM" error or slow performance.

**Why It's Wrong:** Wastes CPU on serialization, creates unclear error messages, may cause partial failures in bulk operations, and could trigger Redis memory issues before validation catches it.

**How to Fix:** Validate payload size *before* any serialization. Calculate approximate JSON size by summing string lengths of arguments + fixed overhead. Implement in `QueueManager.validate_job()`:
```python
estimated_size = len(json.dumps(job.args)) + len(json.dumps(job.kwargs)) + 500  # metadata overhead
if estimated_size > config.max_payload_size:
    raise ValidationError(f"Job payload exceeds {config.max_payload_size} bytes limit")
```

---

⚠️ **Pitfall: Forgetting TTL on Job Data**

**Description:** Storing jobs in Redis without expiration time, assuming they'll always be deleted after processing. However, jobs can get stuck (worker crashes, bugs) or monitoring queries may keep references, causing indefinite memory growth.

**Why It's Wrong:** Redis memory fills up over time, causing evictions or crashes. Production incidents often trace back to forgotten TTLs.

**How to Fix:** Always set TTL when storing jobs. Use a conservative value (e.g., 7 days) that exceeds maximum possible job lifetime. In Redis pipeline:
```python
pipeline.hset(f"job:{job_id}", mapping=serialized_job)
pipeline.expire(f"job:{job_id}", 604800)  # 7 days in seconds
```

---

⚠️ **Pitfall: Non-Atomic Multi-Step Queue Operations**

**Description:** Performing separate Redis commands for storing job data and adding to queue list. If system crashes between commands, job data may be stored but not enqueued (orphaned) or enqueued without data (worker will fail).

**Why It's Wrong:** Creates inconsistent state that's hard to detect and recover from. Orphaned jobs waste memory; missing data causes worker crashes.

**How to Fix:** Use Redis pipelines or transactions for all multi-step operations. The enqueue process should be atomic:
```python
with redis.pipeline() as pipe:
    pipe.hset(f"job:{job_id}", mapping=serialized_job)
    pipe.expire(f"job:{job_id}", ttl)
    pipe.lpush(f"queue:{queue_name}", job_id)
    pipe.execute()  # All commands happen atomically
```

---

⚠️ **Pitfall: Using Local Time Instead of UTC**

**Description:** Storing timestamps in local timezone or using `datetime.now()` without timezone. When deployed across regions or during daylight saving transitions, job ordering and scheduling become inconsistent.

**Why It's Wrong:** Causes incorrect job ordering (timezone differences), scheduling errors (DST shifts), and debugging nightmares when comparing timestamps from different servers.

**How to Fix:** Always use UTC for all timestamps. In Python:
```python
from datetime import datetime, timezone
job.created_at = datetime.now(timezone.utc)
```
Serialize as ISO 8601 with Z suffix: `"2024-01-15T10:30:00Z"`.

---

⚠️ **Pitfall: Blocking Operations Without Timeouts**

**Description:** Using Redis `BLPOP` without timeout or with very long timeout in workers. This prevents graceful shutdown (worker hangs waiting for job when SIGTERM arrives) and hides connectivity issues.

**Why It's Wrong:** Graceful shutdown fails, forcing SIGKILL. Network partitions cause all workers to hang indefinitely. System becomes unresponsive.

**How to Fix:** Use reasonable timeouts (1-5 seconds) on blocking Redis operations. Implement heartbeat checking in worker main loop. For graceful shutdown, use interruptible waits or check shutdown flag between polls.

---

⚠️ **Pitfall: Serializing Non-Serializable Objects**

**Description:** Allowing job arguments to contain database connections, file handles, or custom objects without proper serialization. JSON serialization fails with cryptic errors.

**Why It's Wrong:** Jobs fail immediately on enqueue with unclear errors. Requires producers to understand serialization details. Limits what can be passed as job arguments.

**How to Fix:** Provide clear documentation and validation. Use JSON-serializable types only: strings, numbers, booleans, lists, dicts. For complex objects, pass identifiers (database IDs, file paths) and have the job handler reconstruct them. Implement validation in `Job.to_dict()` that checks each argument.

### Implementation Guidance for Queue Core

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option | Recommendation |
|-----------|---------------|-----------------|----------------|
| Redis Client | `redis-py` (official client) | `hiredis` parser for performance | `redis-py` with connection pooling - simple, robust, well-supported |
| Serialization | JSON (built-in) | MessagePack or CBOR | JSON - human-readable, debuggable, good enough for most payloads |
| ID Generation | UUID4 (random) | ULID (time-ordered) | **ULID** - provides time-based sorting, collision resistant, readable |
| Validation | Manual size calculation | Pydantic models with validation | Manual + JSON schema check - keeps dependencies minimal |
| Connection Handling | Simple client per operation | Connection pool with health checks | **Connection pool** - essential for production performance |

#### B. Recommended File/Module Structure

```
background_job_processor/
├── pyproject.toml                    # Project dependencies and metadata
├── README.md
├── src/
│   └── job_processor/
│       ├── __init__.py
│       ├── core/                      # Core queue infrastructure
│       │   ├── __init__.py
│       │   ├── job.py                 # Job class definition and serialization
│       │   ├── redis_client.py        # RedisClient wrapper with pooling
│       │   ├── queue_manager.py       # QueueManager implementation
│       │   └── config.py              # SystemConfig, QueueConfig, etc.
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── ulid_generator.py      # ULID generation utility
│       │   └── validation.py          # Payload validation helpers
│       └── cli.py                     # Command-line interface for testing
├── tests/
│   ├── __init__.py
│   └── test_queue_core.py             # Tests for Milestone 1
└── examples/
    └── basic_enqueue.py               # Example usage of queue system
```

#### C. Infrastructure Starter Code

**Complete Redis Client Wrapper (`redis_client.py`):**
```python
"""Redis client wrapper with connection pooling and error handling."""
import redis
from typing import Optional, Any, Dict
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class RedisClient:
    """Singleton Redis client with connection pooling.
    
    This wrapper provides a consistent interface for Redis operations with
    proper connection management, error handling, and logging.
    """
    
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None
    
    def __init__(self, url: str = "redis://localhost:6379/0", **connection_kwargs):
        """Initialize Redis client (private, use get_instance).
        
        Args:
            url: Redis connection URL (redis://, rediss://, unix://)
            **connection_kwargs: Additional connection parameters passed to redis.Redis
        """
        if RedisClient._instance is not None:
            raise RuntimeError("Use RedisClient.get_instance() to get singleton")
        
        self.url = url
        self.connection_kwargs = connection_kwargs
        self._client = None
        
    @classmethod
    def get_instance(cls, url: Optional[str] = None, **kwargs) -> 'RedisClient':
        """Get singleton RedisClient instance.
        
        Args:
            url: Optional Redis URL (uses default if None on first call)
            **kwargs: Connection parameters
        
        Returns:
            Singleton RedisClient instance
        """
        if cls._instance is None:
            url = url or "redis://localhost:6379/0"
            cls._instance = cls(url, **kwargs)
            cls._instance._connect()
        return cls._instance
    
    def _connect(self) -> None:
        """Establish Redis connection with pooling."""
        try:
            # Parse URL and set up connection pool
            pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=self.connection_kwargs.get('max_connections', 10),
                socket_timeout=self.connection_kwargs.get('socket_timeout', 5),
                retry_on_timeout=self.connection_kwargs.get('retry_on_timeout', True),
                decode_responses=True  # Automatically decode responses to strings
            )
            self._client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self.url}")
            
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get_client(self) -> redis.Redis:
        """Get the underlying Redis client instance.
        
        Returns:
            redis.Redis client instance
        
        Raises:
            RuntimeError: If client is not connected
        """
        if self._client is None:
            raise RuntimeError("Redis client not connected")
        return self._client
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute a Redis command with error handling.
        
        Args:
            command: Redis command name (e.g., 'hset', 'lpush')
            *args: Command arguments
            **kwargs: Additional options
        
        Returns:
            Command result
        
        Raises:
            redis.RedisError: On Redis operation failure
        """
        client = self.get_client()
        try:
            # Get the method from redis.Redis instance
            method = getattr(client, command.lower())
            return method(*args, **kwargs)
        except redis.RedisError as e:
            logger.error(f"Redis command failed: {command} {args}: {e}")
            raise
    
    def pipeline(self) -> redis.Pipeline:
        """Return a Redis pipeline for atomic operations.
        
        Returns:
            redis.Pipeline instance for executing multiple commands atomically
        """
        client = self.get_client()
        return client.pipeline()
    
    @contextmanager
    def transaction(self):
        """Context manager for Redis transaction (MULTI/EXEC).
        
        Usage:
            with redis_client.transaction() as pipe:
                pipe.set('key1', 'value1')
                pipe.incr('key2')
        """
        pipe = self.pipeline()
        try:
            yield pipe
            pipe.execute()
        except redis.RedisError as e:
            pipe.reset()
            raise
    
    def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")
```

**Complete ULID Generator Utility (`utils/ulid_generator.py`):**
```python
"""ULID generation for time-ordered unique job IDs."""
import time
import os
import random
from datetime import datetime, timezone

# Base32 encoding alphabet (Crockford's)
BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
BASE32_LEN = len(BASE32)


def ulid() -> str:
    """Generate a ULID string.
    
    ULID format: 01J5XZR0W3F6Y2A4B8C9D10E11
    - First 10 chars: Timestamp in milliseconds (base32 encoded)
    - Last 16 chars: Random bytes (base32 encoded)
    
    Returns:
        26-character ULID string
    """
    # Get current time in milliseconds
    timestamp = int(time.time() * 1000)
    
    # Encode timestamp (48 bits -> 10 base32 chars)
    timestamp_chars = []
    for i in range(10):
        timestamp_chars.append(BASE32[(timestamp >> (5 * (9 - i))) & 0x1F])
    
    # Generate random bytes (80 bits -> 16 base32 chars)
    random_bytes = bytearray(os.urandom(10))
    
    # Ensure randomness doesn't exceed 80 bits
    random_bytes[0] &= 0xFF  # Clear top bits to keep within 80 bits
    
    random_chars = []
    # Process 5-bit chunks from the 80 random bits
    random_bits = int.from_bytes(random_bytes, 'big')
    for i in range(16):
        random_chars.append(BASE32[(random_bits >> (5 * (15 - i))) & 0x1F])
    
    return ''.join(timestamp_chars + random_chars)


def ulid_to_datetime(ulid_str: str) -> datetime:
    """Extract timestamp from ULID.
    
    Args:
        ulid_str: ULID string
    
    Returns:
        datetime object in UTC
    """
    if len(ulid_str) != 26:
        raise ValueError(f"Invalid ULID length: {ulid_str}")
    
    timestamp_part = ulid_str[:10]
    timestamp = 0
    
    # Decode base32 timestamp
    for char in timestamp_part:
        timestamp = timestamp * 32 + BASE32.index(char)
    
    # Convert milliseconds to seconds
    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)


# Test the implementation
if __name__ == "__main__":
    # Generate and test a few ULIDs
    for _ in range(5):
        u = ulid()
        dt = ulid_to_datetime(u)
        print(f"ULID: {u} -> Time: {dt.isoformat()}")
```

#### D. Core Logic Skeleton Code

**Job Class with Serialization (`core/job.py`):**
```python
"""Job class definition with serialization methods."""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import uuid


class JobStatus(Enum):
    """Job lifecycle status states."""
    PENDING = "pending"          # Waiting in queue
    ACTIVE = "active"            # Currently being processed
    COMPLETED = "completed"      # Finished successfully
    FAILED = "failed"            # Finished with error
    RETRY_SCHEDULED = "retry_scheduled"  # Failed and scheduled for retry
    DEAD_LETTER = "dead_letter"  # Permanently failed after max retries


class Job:
    """Represents a unit of work to be processed asynchronously.
    
    This is the core data structure that flows through the entire system.
    """
    
    def __init__(
        self,
        job_type: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        queue: str = "default",
        priority: int = 1,
        max_retries: int = 3,
        timeout_seconds: int = 300,
        metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        status: JobStatus = JobStatus.PENDING,
        attempts: int = 0,
        errors: Optional[List[Dict[str, Any]]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        result: Optional[Any] = None,
    ):
        """Initialize a new Job.
        
        Args:
            job_type: Identifier for the handler function/class
            args: Positional arguments for the job handler
            kwargs: Keyword arguments for the job handler
            queue: Target queue name
            priority: Queue priority weight (higher = more important)
            max_retries: Maximum retry attempts before giving up
            timeout_seconds: Maximum execution time in seconds
            metadata: Arbitrary key-value data for monitoring
            job_id: Unique identifier (auto-generated if None)
            created_at: Creation timestamp (auto-set if None)
            status: Current job status
            attempts: Number of execution attempts so far
            errors: List of error records from previous failures
            started_at: When job execution began
            completed_at: When job finished (success or failure)
            result: Job execution result (on success)
        """
        self.job_id = job_id or str(uuid.uuid4())  # Will replace with ULID later
        self.job_type = job_type
        self.args = args or []
        self.kwargs = kwargs or {}
        self.queue = queue
        self.priority = priority
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.status = status
        self.attempts = attempts
        self.errors = errors or []
        self.started_at = started_at
        self.completed_at = completed_at
        self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization.
        
        Returns:
            Dictionary representation of the job
        """
        # TODO 1: Convert datetime objects to ISO format strings
        # Hint: Use .isoformat() for datetime objects, check if value is datetime
        
        # TODO 2: Convert JobStatus enum to string value
        # Hint: Access .value property of the enum
        
        # TODO 3: Ensure all values are JSON-serializable
        # Hint: Recursively check lists and dicts, raise TypeError for non-serializable values
        
        # TODO 4: Return complete dictionary with all fields
        # Structure should match the Job constructor parameters
        
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Reconstruct job from dictionary.
        
        Args:
            data: Dictionary representation from to_dict()
        
        Returns:
            Job instance
        """
        # TODO 1: Parse ISO format strings back to datetime objects
        # Hint: datetime.fromisoformat() for ISO strings
        
        # TODO 2: Convert status string back to JobStatus enum
        # Hint: JobStatus(value) to reconstruct
        
        # TODO 3: Handle optional fields that might be missing in older serializations
        # Hint: Use .get() with default values
        
        # TODO 4: Create and return Job instance with parsed data
        # Pass all fields to the constructor
        
        pass
    
    def serialize(self) -> str:
        """Serialize job to JSON string.
        
        Returns:
            JSON string representation
        """
        # TODO 1: Convert to dictionary using to_dict()
        # TODO 2: Serialize to JSON with default=str for any non-serializable values
        # TODO 3: Return the JSON string
        pass
    
    @classmethod
    def deserialize(cls, data: str) -> 'Job':
        """Deserialize job from JSON string.
        
        Args:
            data: JSON string from serialize()
        
        Returns:
            Job instance
        """
        # TODO 1: Parse JSON string to dictionary
        # TODO 2: Convert from dictionary using from_dict()
        # TODO 3: Return the Job instance
        pass
    
    def record_error(self, error: Exception) -> None:
        """Record an error that occurred during job execution.
        
        Args:
            error: Exception that was raised
        """
        # TODO 1: Create error dictionary with keys:
        # - 'type': error class name (type(error).__name__)
        # - 'message': str(error)
        # - 'traceback': formatted traceback string (use traceback.format_exc())
        # - 'timestamp': current UTC time in ISO format
        
        # TODO 2: Append error dict to self.errors list
        
        # TODO 3: Increment self.attempts counter
        
        pass
    
    def should_retry(self) -> bool:
        """Determine if job should be retried.
        
        Returns:
            True if job has retries remaining and hasn't exceeded max_retries
        """
        # TODO 1: Check if attempts < max_retries
        # TODO 2: Return True if more retries available, False otherwise
        pass
```

**Queue Manager Implementation (`core/queue_manager.py`):**
```python
"""Queue manager for enqueuing and managing jobs."""
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .job import Job, JobStatus
from .redis_client import RedisClient
from .config import QueueConfig, SystemConfig
from utils.ulid_generator import ulid

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when job validation fails."""
    pass


class QueueFullError(Exception):
    """Raised when queue is at maximum capacity."""
    pass


class QueueManager:
    """Manages job enqueue operations and queue inspection."""
    
    def __init__(self, config: SystemConfig):
        """Initialize QueueManager with configuration.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.redis = RedisClient.get_instance(
            url=config.redis.url,
            max_connections=config.redis.max_connections,
            socket_timeout=config.redis.socket_timeout,
            retry_on_timeout=config.redis.retry_on_timeout
        )
        
        # Build queue config map for quick lookup
        self.queue_configs: Dict[str, QueueConfig] = {
            qc.name: qc for qc in config.queues
        }
    
    def enqueue_job(self, job: Job) -> str:
        """Validates, serializes, and atomically adds a job to the appropriate Redis queue.
        
        Args:
            job: Job instance to enqueue
        
        Returns:
            job_id: Unique identifier for the enqueued job
        
        Raises:
            ValidationError: If job fails validation
            QueueFullError: If queue is at maximum capacity
        """
        # TODO 1: Validate the job using _validate_job() method
        # TODO 2: Generate ULID for job_id if not already set
        # TODO 3: Set job status to PENDING and created_at timestamp
        # TODO 4: Get queue config for the target queue
        # TODO 5: Check if queue is at capacity (if max_length is set)
        # TODO 6: Serialize job to JSON using job.serialize()
        # TODO 7: Use Redis pipeline for atomic operations:
        #   a. Store job data in hash: HSET job:{job_id} {serialized_job}
        #   b. Set TTL on job hash: EXPIRE job:{job_id} {job_ttl}
        #   c. Add job ID to queue list: LPUSH queue:{queue_name} {job_id}
        #   d. Increment queue length metric: INCR queue:{queue_name}:count
        # TODO 8: Log successful enqueue with job_id and queue
        # TODO 9: Return job_id
        
        pass
    
    def bulk_enqueue(self, jobs: List[Job]) -> List[str]:
        """Enqueues multiple jobs in a single atomic transaction.
        
        Args:
            jobs: List of Job instances to enqueue
        
        Returns:
            List of job IDs in same order as input jobs
        
        Raises:
            ValidationError: If any job fails validation (entire batch rolls back)
        """
        # TODO 1: Validate all jobs first (fail fast)
        # TODO 2: Generate job IDs for all jobs missing them
        # TODO 3: Prepare all jobs (status, timestamps)
        # TODO 4: Group jobs by queue for efficient Redis operations
        # TODO 5: Use Redis pipeline for atomic batch:
        #   For each job:
        #     - HSET job:{job_id} {serialized_job}
        #     - EXPIRE job:{job_id} {job_ttl}
        #   For each queue with jobs:
        #     - LPUSH queue:{queue_name} [job_id1, job_id2, ...]
        #     - INCRBY queue:{queue_name}:count {count}
        # TODO 6: Return list of job IDs
        
        pass
    
    def get_queue_length(self, queue_name: str) -> int:
        """Returns the current number of pending jobs in the specified queue.
        
        Args:
            queue_name: Name of the queue
        
        Returns:
            Number of pending jobs
        """
        # TODO 1: Use Redis LLEN command on queue:{queue_name} key
        # TODO 2: Return the length as integer
        
        pass
    
    def peek_queue(self, queue_name: str, count: int = 10) -> List[Job]:
        """Retrieves jobs from the front of the queue without removing them.
        
        Args:
            queue_name: Name of the queue
            count: Maximum number of jobs to peek
        
        Returns:
            List of Job instances (deserialized)
        """
        # TODO 1: Use Redis LRANGE to get job IDs from start of list
        # TODO 2: For each job ID, fetch job data from hash
        # TODO 3: Deserialize job data to Job objects
        # TODO 4: Return list of Job instances
        
        pass
    
    def list_queues(self, include_internal: bool = False) -> List[str]:
        """Returns names of all configured queues.
        
        Args:
            include_internal: If True, include system queues
        
        Returns:
            List of queue names
        """
        # TODO 1: Get all queue names from self.queue_configs
        # TODO 2: If include_internal is True, add system queue names:
        #   - retry:scheduled (sorted set)
        #   - dead:letter (list)
        #   - scheduler:queued (list)
        # TODO 3: Return sorted list of queue names
        
        pass
    
    def delete_job(self, job_id: str, queue_name: str) -> bool:
        """Removes a specific job from a queue if it exists.
        
        Args:
            job_id: Job identifier
            queue_name: Queue to remove job from
        
        Returns:
            True if job was found and removed
        """
        # TODO 1: Use Redis LREM to remove job_id from queue:{queue_name} list
        # TODO 2: If removed, decrement queue:{queue_name}:count metric
        # TODO 3: Return True if count > 0, False otherwise
        
        pass
    
    def get_job_by_id(self, job_id: str) -> Optional[Job]:
        """Retrieves a job from Redis by its ID.
        
        Args:
            job_id: Job identifier
        
        Returns:
            Job instance if found, None otherwise
        """
        # TODO 1: Use Redis HGET to fetch job data from job:{job_id} hash
        # TODO 2: If data exists, deserialize to Job object
        # TODO 3: Return Job instance or None
        
        pass
    
    def _validate_job(self, job: Job) -> None:
        """Internal method to validate job before enqueue.
        
        Args:
            job: Job to validate
        
        Raises:
            ValidationError: If validation fails
        """
        # TODO 1: Check if job payload size exceeds max_payload_size
        # Hint: Estimate size using len(json.dumps(job.args)) + len(json.dumps(job.kwargs))
        
        # TODO 2: Validate required fields: job_type, queue
        
        # TODO 3: Check that args and kwargs are JSON-serializable
        # Hint: Try json.dumps on each and catch TypeError
        
        # TODO 4: Ensure queue exists in config (or use default)
        
        # TODO 5: Validate priority is positive integer
        
        pass
```

#### E. Language-Specific Hints

**Python-Specific Implementation Tips:**

1. **JSON Serialization of Datetimes:** Use a custom JSON encoder for datetime objects:
   ```python
   import json
   from datetime import datetime
   
   class DateTimeEncoder(json.JSONEncoder):
       def default(self, obj):
           if isinstance(obj, datetime):
               return obj.isoformat()
           return super().default(obj)
   
   # Usage
   json_str = json.dumps(job_dict, cls=DateTimeEncoder)
   ```

2. **Redis Connection Pooling:** Configure connection pooling properly:
   ```python
   pool = redis.ConnectionPool.from_url(
       'redis://localhost:6379/0',
       max_connections=10,
       socket_timeout=5,
       socket_connect_timeout=5,
       retry_on_timeout=True,
       health_check_interval=30
   )
   ```

3. **Atomic Operations with Pipelines:** Always use pipelines for multi-step Redis operations:
   ```python
   with redis_client.pipeline() as pipe:
       pipe.multi()  # Start transaction
       pipe.hset(f"job:{job_id}", mapping=job_data)
       pipe.expire(f"job:{job_id}", 604800)
       pipe.lpush(f"queue:{queue_name}", job_id)
       pipe.execute()  # Atomic execution
   ```

4. **Size Estimation Without Full Serialization:** Estimate JSON size efficiently:
   ```python
   def estimate_json_size(obj):
       """Estimate JSON size without serializing entire object."""
       if isinstance(obj, (str, bytes)):
           return len(obj)
       elif isinstance(obj, (int, float)):
           return 10  # Approximate
       elif isinstance(obj, (list, tuple)):
           return sum(estimate_json_size(item) for item in obj) + 2
       elif isinstance(obj, dict):
           size = 2
           for k, v in obj.items():
               size += estimate_json_size(k) + estimate_json_size(v) + 1
           return size
       return 50  # Conservative estimate for unknown types
   ```

5. **ULID Implementation:** Use the `ulid-py` library for production:
   ```python
   # pip install ulid-py
   from ulid import ULID
   job_id = str(ULID())  # Generates time-ordered ULID
   ```

#### F. Milestone Checkpoint

After implementing the Job Queue Core (Milestone 1), verify your implementation with these checkpoints:

**Checkpoint 1: Job Serialization Round-Trip**
```bash
python -m pytest tests/test_queue_core.py::test_job_serialization_roundtrip -v
```
**Expected Output:** Test passes with no errors. A job serialized and deserialized should have identical field values (timestamps within millisecond tolerance).

**Checkpoint 2: Enqueue and Retrieve**
```bash
# Start Redis locally first: redis-server
python examples/basic_enqueue.py
```
**Example script output:**
```
Enqueued job with ID: 01J5XZR0W3F6Y2A4B8C9D10E11
Queue 'default' length: 1
Peeked job: Job(job_id='01J5XZR0W3F6Y2A4B8C9D10E11', job_type='test_task', ...)
```

**Checkpoint 3: Payload Size Validation**
```bash
python -m pytest tests/test_queue_core.py::test_oversized_payload_rejected -v
```
**Expected:** Test passes with `ValidationError` raised for payload > 1MB.

**Checkpoint 4: Queue Priority Configuration**
```bash
python examples/queue_priority_test.py
```
**Verify:** The script should show that high-priority queues receive more frequent polling attention (simulated by checking dequeue order over many iterations).

**Signs Something Is Wrong:**
- **Redis connection errors:** Check Redis is running and connection URL is correct
- **Serialization errors:** Ensure all job arguments are JSON-serializable (no custom objects)
- **Missing jobs after enqueue:** Verify pipeline operations are atomic and all execute successfully
- **ULIDs not time-ordered:** Check ULID generation uses milliseconds, not seconds
- **Queue length mismatch:** Ensure `LPUSH` and `INCR` operations happen in same pipeline

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Job disappears after enqueue | Pipeline not executing or Redis transaction rolled back | Check Redis logs for errors, verify `pipe.execute()` is called | Ensure all pipeline operations are followed by `execute()` |
| Job data exists but not in queue | Orphaned job - stored in hash but not added to list | Use `redis-cli KEYS "job:*"` to find jobs, check if ID in queue list | Use atomic pipeline for both operations |
| Queue length metric wrong | Race condition in INCR/DECR operations | Monitor `queue:{name}:count` vs actual LLEN | Use Redis transactions or Lua scripts for atomic increments |
| Serialization fails with TypeError | Non-JSON-serializable arguments in job | Add validation before serialization, log argument types | Convert objects to serializable forms (IDs, paths, strings) |
| ULIDs not monotonic | Using UUID instead of time-based ID | Check ULID generation uses current time in milliseconds | Implement proper ULID or use `ulid-py` library |
| Jobs enqueued to non-existent queue | Queue config not loaded or mismatch | Check `queue_configs` map in QueueManager | Ensure queue exists in config or create default config |
| Redis memory growing indefinitely | Missing TTL on job hashes | Check `job:{id}` keys with `TTL` command | Always set TTL when storing jobs |

---


## Component Design: Worker Process

> **Milestone(s):** Milestone 2: Worker Process. This section provides the complete design for worker processes that fetch jobs from Redis queues, execute them with proper isolation, handle failures gracefully, and report their liveness for monitoring.

### Mental Model: Factory Assembly Line Worker

Imagine a factory assembly line with multiple stations. Each **worker** is a skilled operator who:
1. **Waits at a designated station** (queue) for work items to arrive
2. **Receives a work ticket** (job) describing exactly what needs to be done
3. **Performs the operation** using the right tools (handler functions)
4. **Marks the ticket** as complete, moves it to the finished pile, or escalates it if something goes wrong
5. **Reports their status** regularly to the floor manager to confirm they're still operational

Just like factory workers have different specializations (some handle painting, others assembly), our workers can be configured to monitor specific queues for different job types. They work concurrently within their station, can be told to finish their current task before going home (graceful shutdown), and have safety mechanisms to stop working if a task takes too long or becomes dangerous (timeouts and exception handling).

This mental model emphasizes key worker characteristics: **specialization by queue**, **reliable task execution**, **concurrent operation within limits**, and **constant status reporting** for system health monitoring.

### Worker Interface and Configuration

The worker component exposes a clean interface for starting, stopping, and monitoring worker processes. Each worker instance is configured through a `WorkerConfig` object that determines its behavior.

**Worker Configuration Data Structure:**

| Field | Type | Description |
|-------|------|-------------|
| `queues` | `List[str]` | List of queue names this worker should monitor (e.g., `["email", "reports", "default"]`) |
| `concurrency` | `int` | Maximum number of jobs this worker can process simultaneously (default: 1) |
| `heartbeat_interval` | `int` | Seconds between heartbeats written to Redis (default: 30) |
| `job_timeout` | `int` | Maximum seconds a job can run before being forcibly terminated (default: 300) |
| `worker_id` | `str` | Unique identifier for this worker instance (auto-generated if not provided) |
| `hostname` | `str` | Host machine name for monitoring (auto-detected) |
| `pid` | `int` | Process ID for monitoring (auto-detected) |

**Worker Public Interface:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `start()` | None | None | Starts the worker main loop, begins polling queues and processing jobs |
| `stop(graceful=True)` | `graceful: bool` | None | Stops the worker; if `graceful=True`, completes current job before stopping |
| `pause()` | None | None | Temporarily stops fetching new jobs but continues processing current ones |
| `resume()` | None | None | Resumes fetching new jobs after being paused |
| `get_status()` | None | `Dict[str, Any]` | Returns current worker status: idle/active, current job, queue depths, etc. |
| `register_handler(job_type, handler_func)` | `job_type: str`, `handler_func: Callable` | None | Registers a function to handle a specific job type |

**Worker Internal State Tracking:**

Each worker maintains internal state that evolves throughout its lifecycle:

| State | Description | Triggers |
|-------|-------------|----------|
| `STARTING` | Worker is initializing connections and loading handlers | Worker instantiation |
| `IDLE` | Worker is waiting for jobs (blocking on queue reads) | After startup or job completion |
| `PROCESSING` | Worker is actively executing a job | Job dequeued and execution started |
| `PAUSED` | Worker is not fetching new jobs but may complete current job | Manual pause or system overload |
| `SHUTTING_DOWN` | Worker is finishing current job before exiting | Stop signal received |
| `STOPPED` | Worker has completely terminated | All cleanup complete |

These states are captured in the worker state machine diagram:

![Worker Process State Machine](./diagrams/worker-state-machine.svg)

### Worker Main Loop Algorithm

The worker's core logic follows a deterministic main loop that orchestrates job fetching, execution, and status reporting. Below is the step-by-step algorithm executed by each worker process:

1. **Initialization Phase**
   1. Generate a unique worker ID using ULID format (timestamp + randomness for lexicographic sorting)
   2. Establish connection to Redis using the `RedisClient.get_instance()` singleton
   3. Load job handler registry from application code (functions decorated with `@job_handler`)
   4. Set up signal handlers for SIGTERM (graceful shutdown) and SIGINT (immediate shutdown)
   5. Initialize worker status in Redis with current timestamp and `STARTING` state
   6. Start heartbeat thread that updates Redis every `heartbeat_interval` seconds

2. **Main Processing Loop**
   1. **State Check**: If worker state is `SHUTTING_DOWN`, break loop and proceed to shutdown sequence
   2. **Pause Check**: If worker state is `PAUSED`, sleep for 1 second and continue loop
   3. **Queue Selection**: Determine which queue to poll next using priority-weighted round-robin
      - Calculate total priority weight: sum of all configured queue priorities
      - Generate random number between 0 and total weight
      - Iterate through queues, subtracting each queue's priority until random number ≤ 0
      - Select that queue for this polling cycle
   4. **Job Fetching**: Attempt to fetch a job from the selected queue using `BRPOPLPUSH`
      - Use `BRPOPLPUSH source_queue processing_queue timeout=1`
      - If timeout occurs with no job, continue to next iteration
      - If job retrieved, parse JSON into `Job` object using `Job.deserialize()`
   5. **Job Processing Setup**:
      - Update job status to `ACTIVE` and set `started_at` timestamp in Redis
      - Update worker status to `PROCESSING` with current job ID
      - Start execution timeout timer based on `job.timeout_seconds`
   6. **Job Execution**:
      - Look up handler function for `job.job_type` in registry
      - If no handler found, mark job as failed with "No handler registered" error
      - Execute handler with `job.args` and `job.kwargs` in isolated context (thread/process)
      - Wait for completion or timeout
   7. **Post-Execution Handling**:
      - **If successful**: Update job status to `COMPLETED`, store result, increment metrics
      - **If failed with exception**: Call `job.record_error()`, check `job.should_retry()`
        - If retries remain: schedule retry with exponential backoff (Milestone 3)
        - If no retries remain: move to dead letter queue
      - **If timeout**: Record timeout error, treat as failure with retry logic
   8. **Cleanup**: Remove job from processing queue, update worker status to `IDLE`

3. **Shutdown Sequence**
   1. Set worker state to `SHUTTING_DOWN` in Redis
   2. Wait for current job to complete (up to graceful shutdown timeout)
   3. If job doesn't complete in timeout, force termination and move job back to queue
   4. Close Redis connection
   5. Stop heartbeat thread
   6. Set final worker state to `STOPPED`

The complete flow of a successful job execution is shown in the sequence diagram:

![Happy Path Job Execution Sequence](./diagrams/happy-path-sequence.svg)

### ADR: Concurrency Model (Process vs Thread)

> **Decision: Thread Pool with Process Isolation for CPU-bound Tasks**
>
> - **Context**: Workers need to execute multiple jobs concurrently to maximize throughput. The system must handle both I/O-bound tasks (API calls, database queries) and CPU-bound tasks (image processing, data analysis). We need isolation to prevent one failing job from crashing the entire worker.
>
> - **Options Considered**:
>   1. **Single-threaded worker**: Simple but poor resource utilization
>   2. **Thread pool**: Good for I/O-bound tasks, shares memory space (risky)
>   3. **Process pool**: Full isolation, higher overhead for job data transfer
>   4. **Hybrid approach**: Thread pool for I/O tasks with optional process isolation for CPU tasks
>
> - **Decision**: Implement thread pool for general execution with optional process-based isolation for CPU-bound job types.
>
> - **Rationale**: Most background jobs in web applications are I/O-bound (sending emails, calling APIs, database operations). Threads provide excellent concurrency for these with minimal overhead. However, for CPU-intensive tasks or unreliable third-party code, process isolation prevents memory leaks and crashes from affecting other jobs. By making isolation configurable per job type, we get the best of both worlds.
>
> - **Consequences**: 
>   - Enables efficient handling of mixed workloads
>   - Adds complexity to job data serialization for process boundaries
>   - Requires careful management of process pool size to avoid resource exhaustion
>   - Provides safer execution environment for untrusted job code

**Concurrency Model Comparison:**

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Single-threaded | Simple, no race conditions | Poor CPU utilization, blocks on I/O | No |
| Thread pool | Good I/O concurrency, shared memory | One job can crash entire worker, GIL limits CPU tasks | **Partially** (default) |
| Process pool | Full isolation, bypasses GIL | Higher memory, serialization overhead | **Partially** (optional) |
| Hybrid approach | Best of both worlds, flexible per job type | Most complex implementation | **Yes** |

### ADR: Dequeue Strategy and Reliability

> **Decision: BRPOPLPUSH with Processing Queue for Reliable Dequeue**
>
> - **Context**: When a worker fetches a job from Redis, the job must be removed from the main queue to prevent duplicate processing. However, if the worker crashes during execution, the job should not be lost forever. We need atomic dequeue with crash recovery.
>
> - **Options Considered**:
>   1. **RPOP then delete**: Simple but can lose jobs if worker crashes between RPOP and processing
>   2. **BRPOP with timeout**: Blocking pop but job disappears from Redis during execution
>   3. **BRPOPLPUSH**: Atomically moves job to processing queue, visible during execution
>   4. **Redis Streams with consumer groups**: Modern approach with built-in acknowledgment
>
> - **Decision**: Use `BRPOPLPUSH` to move jobs from main queue to a processing queue, with periodic health checks to recover orphaned jobs.
>
> - **Rationale**: `BRPOPLPUSH` provides atomic move operation in Redis, ensuring the job is never in "limbo" between queues. While Redis Streams are more feature-rich, they require Redis 5.0+ and add complexity for our FIFO queue needs. The processing queue acts as a visibility mechanism: jobs being processed remain in Redis so we can detect crashed workers and recover their jobs.
>
> - **Consequences**:
>   - Requires maintaining processing queues per worker or worker group
>   - Adds recovery logic to check for stale jobs in processing queues
>   - Provides at-least-once delivery semantics (jobs may be processed twice if worker crashes after processing but before cleanup)
>   - Compatible with older Redis versions

**Dequeue Strategy Comparison:**

| Option | Reliability | Complexity | Performance | Chosen? |
|--------|-------------|------------|-------------|---------|
| RPOP then delete | Low (job loss on crash) | Simple | High | No |
| BRPOP | Medium (job invisible during processing) | Simple | High | No |
| BRPOPLPUSH | High (visible in processing queue) | Moderate | High | **Yes** |
| Redis Streams | High (built-in features) | High | Medium | No |

### Common Pitfalls in Worker Implementation

⚠️ **Pitfall: Blocking Forever on Queue Read**
- **Description**: Using `BRPOP` with timeout=0 or `BLPOP` without timeout causes worker to block indefinitely, preventing graceful shutdown.
- **Why it's wrong**: The worker cannot respond to shutdown signals while blocked in Redis I/O, requiring SIGKILL to terminate.
- **Fix**: Use `BRPOPLPUSH` with a reasonable timeout (1-5 seconds) and check shutdown flags between polls.

⚠️ **Pitfall: Not Isoling Job Execution**
- **Description**: Running job handlers directly in the worker's main thread allows a job exception to crash the entire worker process.
- **Why it's wrong**: A bug in one job handler terminates all concurrent jobs and requires worker restart.
- **Fix**: Execute each job in a separate thread or process with proper exception boundaries.

⚠️ **Pitfall: Forgetting Heartbeat Updates**
- **Description**: Worker starts heartbeat thread but doesn't handle Redis connection failures, causing stale heartbeats.
- **Why it's wrong**: Monitoring system marks active workers as dead, triggering unnecessary job recovery.
- **Fix**: Implement retry logic in heartbeat updates and validate Redis connection before each update.

⚠️ **Pitfall: Ignoring Memory Leaks in Long-running Workers**
- **Description**: Worker processes thousands of jobs without restarting, gradually accumulating memory from uncollectable Python references.
- **Why it's wrong**: Worker memory grows unbounded, eventually causing OOM kills or performance degradation.
- **Fix**: Implement soft memory limits that trigger graceful shutdown after processing N jobs or reaching memory threshold.

⚠️ **Pitfall: Incomplete Job State Transition**
- **Description**: Worker updates job status to `ACTIVE` but crashes before setting `COMPLETED` or `FAILED`, leaving job in ambiguous state.
- **Why it's wrong**: Recovery system cannot determine if job needs to be retried or was successful.
- **Fix**: Use Redis transactions (MULTI/EXEC) to atomically update both job status and worker state.

### Implementation Guidance for Worker

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Concurrency | `concurrent.futures.ThreadPoolExecutor` | `multiprocessing.Pool` with configurable isolation |
| Signal Handling | `signal.signal()` for basic SIGTERM | `asyncio` event loop with signal integration |
| Timeout Management | `threading.Timer` for job timeouts | Resource-aware timeout with CPU time monitoring |
| Job Isolation | Basic try/except in thread | Subprocess with resource limits (CPU, memory) |

**B. Recommended File/Module Structure:**

```
background_job_processor/
├── worker/
│   ├── __init__.py
│   ├── worker.py              # Main Worker class implementation
│   ├── config.py              # WorkerConfig and related configuration
│   ├── handlers.py            # Job handler registry and decorator
│   ├── concurrency.py         # Thread/process pool management
│   ├── heartbeat.py           # Heartbeat thread implementation
│   └── signals.py             # Signal handling utilities
├── models/
│   └── job.py                 # Job class (from Milestone 1)
├── redis_client.py            # RedisClient (from Milestone 1)
└── config/
    └── system_config.py       # SystemConfig (from Milestone 1)
```

**C. Infrastructure Starter Code:**

**Heartbeat Thread Implementation** (complete working code):

```python
# worker/heartbeat.py
import threading
import time
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class WorkerHeartbeat:
    """Thread that periodically updates worker liveness in Redis."""
    
    def __init__(self, worker_id: str, redis_client, interval: int = 30):
        self.worker_id = worker_id
        self.redis_client = redis_client
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_successful = None
        
    def start(self):
        """Start the heartbeat thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Heartbeat thread already running")
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{self.worker_id}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Heartbeat started for worker {self.worker_id}")
        
    def stop(self):
        """Stop the heartbeat thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            logger.info(f"Heartbeat stopped for worker {self.worker_id}")
            
    def _run(self):
        """Main heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                # Update heartbeat in Redis with current timestamp
                now = datetime.utcnow().isoformat()
                key = f"worker:heartbeat:{self.worker_id}"
                self.redis_client.execute("SET", key, now, "EX", self.interval * 3)
                self._last_successful = now
                
                # Also update worker status if available
                status_key = f"worker:status:{self.worker_id}"
                current_status = self.redis_client.execute("GET", status_key)
                if current_status:
                    self.redis_client.execute("EXPIRE", status_key, self.interval * 3)
                    
            except Exception as e:
                logger.error(f"Heartbeat update failed: {e}")
                
            # Sleep for interval, but check stop event more frequently
            for _ in range(self.interval * 10):  # Check every 0.1 seconds
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
                
    def is_healthy(self) -> bool:
        """Check if heartbeat is updating successfully."""
        if not self._last_successful:
            return False
            
        # Check if last successful update was within 2 intervals
        try:
            last_time = datetime.fromisoformat(self._last_successful)
            cutoff = datetime.utcnow().timestamp() - (self.interval * 2)
            return last_time.timestamp() > cutoff
        except (ValueError, TypeError):
            return False
```

**D. Core Logic Skeleton Code:**

**Worker Main Loop Method** (with TODOs matching algorithm steps):

```python
# worker/worker.py
import signal
import time
import json
import logging
from typing import Dict, Callable, Optional, List
from datetime import datetime

from models.job import Job, JobStatus
from redis_client import RedisClient
from worker.config import WorkerConfig
from worker.heartbeat import WorkerHeartbeat
from worker.concurrency import JobExecutor

logger = logging.getLogger(__name__)

class Worker:
    """Main worker class that processes jobs from Redis queues."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id = config.worker_id or self._generate_worker_id()
        self.redis = RedisClient.get_instance()
        self.handlers: Dict[str, Callable] = {}
        self.heartbeat: Optional[WorkerHeartbeat] = None
        self.executor: Optional[JobExecutor] = None
        self.state = "STARTING"
        self.current_job: Optional[Job] = None
        self._shutdown_requested = False
        self._paused = False
        
    def start(self):
        """Start the worker main loop."""
        # TODO 1: Set up signal handlers for SIGTERM and SIGINT
        #   - SIGTERM should trigger graceful shutdown
        #   - SIGINT should trigger immediate shutdown
        
        # TODO 2: Initialize Redis connection using self.redis.get_client()
        
        # TODO 3: Start heartbeat thread with self.config.heartbeat_interval
        
        # TODO 4: Initialize job executor (thread/process pool) with self.config.concurrency
        
        # TODO 5: Set worker state to "IDLE" in Redis
        
        # TODO 6: Enter main processing loop (call self._run_loop())
        
        # TODO 7: After loop exits, perform cleanup in self._shutdown()
        
        pass
        
    def _run_loop(self):
        """Main worker processing loop."""
        while not self._shutdown_requested:
            try:
                # TODO 1: Check if worker is paused - if so, sleep briefly and continue
                
                # TODO 2: Select queue to poll using priority-weighted algorithm
                #   - Get queue priorities from configuration
                #   - Calculate weighted random selection
                #   - Choose queue for this iteration
                
                # TODO 3: Attempt to fetch job from selected queue using BRPOPLPUSH
                #   - Use timeout of 1 second to allow shutdown checking
                #   - If no job, continue to next iteration
                #   - If job received, parse JSON into Job object
                
                # TODO 4: Update job status to ACTIVE and record start time
                #   - Use Redis pipeline for atomic updates
                #   - Set job.started_at = current time
                #   - Update worker state to PROCESSING with job ID
                
                # TODO 5: Execute job using self.executor.submit()
                #   - Look up handler function for job.job_type
                #   - Submit to executor with timeout from job.timeout_seconds
                #   - Handle case where no handler is registered
                
                # TODO 6: Wait for job completion with timeout
                #   - If successful: update job status to COMPLETED
                #   - If failed: call job.record_error() and handle retry logic
                #   - If timeout: record timeout error and handle as failure
                
                # TODO 7: Clean up processing queue entry
                #   - Remove job from processing queue
                #   - Update worker state back to IDLE
                #   - Increment success/failure metrics
                
            except Exception as e:
                logger.exception(f"Unexpected error in worker loop: {e}")
                # Brief sleep to prevent tight error loop
                time.sleep(1)
                
    def _handle_job_result(self, job: Job, result, error: Optional[Exception]):
        """Process the result of a job execution."""
        # TODO 1: If error is None (success), update job status to COMPLETED
        #   - Set job.completed_at = current time
        #   - Store result in job.result (if serializable)
        #   - Update success metrics
        
        # TODO 2: If error occurred, call job.record_error(error)
        #   - Check job.should_retry() to determine next action
        #   - If retries remain: schedule retry with exponential backoff (Milestone 3)
        #   - If no retries: move to dead letter queue (Milestone 3)
        
        # TODO 3: Update job in Redis with final status and any result/error
        #   - Use Redis pipeline for atomic updates
        #   - Set appropriate TTL based on job status
        
        pass
        
    def stop(self, graceful: bool = True):
        """Request worker shutdown."""
        # TODO 1: Set self._shutdown_requested = True
        
        # TODO 2: If graceful=True, set state to SHUTTING_DOWN and wait for current job
        #   - Wait up to job.timeout_seconds for current job to complete
        #   - If timeout, force terminate and requeue job
        
        # TODO 3: If graceful=False, cancel current job immediately
        #   - Use executor.shutdown(wait=False)
        #   - Move current job back to main queue
        
        # TODO 4: Stop heartbeat thread
        
        # TODO 5: Close Redis connection
        
        pass
```

**E. Language-Specific Hints:**

- **Signal Handling**: Use `signal.signal(signal.SIGTERM, handler)` to catch shutdown signals. Set a flag in the handler rather than exiting immediately.
- **Thread Safety**: Use `threading.Lock` for shared state between main thread and heartbeat thread.
- **Job Timeouts**: Use `concurrent.futures.TimeoutError` with `executor.submit()` and `future.result(timeout=...)`.
- **Graceful Shutdown**: Implement `__enter__` and `__exit__` methods on Worker class to use as context manager.
- **Logging**: Use structured logging with `logging.JSONFormatter` for production monitoring.

**F. Milestone Checkpoint:**

After implementing the Worker component, verify functionality with these steps:

1. **Start a worker**:
   ```python
   from worker import Worker
   from worker.config import WorkerConfig
   
   config = WorkerConfig(
       queues=["default"],
       concurrency=2,
       heartbeat_interval=30,
       job_timeout=300
   )
   
   worker = Worker(config)
   worker.start()  # Runs in foreground, Ctrl+C to stop
   ```

2. **Test job execution**:
   - Enqueue a test job using QueueManager from Milestone 1
   - Observe worker logs: should show "Processing job X", "Job completed successfully"
   - Check Redis: job status should transition PENDING → ACTIVE → COMPLETED

3. **Verify graceful shutdown**:
   - Send SIGTERM to worker process: `kill -TERM <pid>`
   - Worker should finish current job before exiting
   - Confirm "Shutting down gracefully" appears in logs

4. **Check heartbeat**:
   - Monitor Redis key `worker:heartbeat:<worker_id>`
   - Value should update every `heartbeat_interval` seconds
   - If worker dies, key should expire after 3×interval

**Expected Output Signs:**
- ✅ Worker connects to Redis and starts heartbeat
- ✅ Worker fetches jobs from configured queues
- ✅ Jobs execute with registered handlers
- ✅ Worker responds to SIGTERM by finishing current job
- ❌ If jobs fail silently, check exception handling in job wrapper
- ❌ If worker hangs on shutdown, check BRPOPLPUSH timeout value


## Component Design: Retry & Error Handling

> **Milestone(s):** Milestone 3: Retry & Error Handling. This section provides the complete design for automatic retry mechanisms with exponential backoff and dead letter queue functionality, enabling resilient job processing in the face of transient failures.

### Mental Model: Customer Service Escalation Process

Think of the retry system as a customer service escalation process in a large organization. When a customer request (job) fails initially:

1. **Initial attempt** → Frontline support handles the request (first execution attempt)
2. **Failure detection** → The request is marked as requiring follow-up (error is recorded)
3. **Escalation timing** → The request is scheduled for a retry after a calculated delay (escalation timer)
4. **Escalation levels** → Each retry goes to a more experienced specialist (exponential backoff)
5. **Maximum escalation** → After exhausting all escalation levels, the case goes to a special investigations team (dead letter queue)
6. **Case review** → The investigations team can manually review, fix, and retry or archive the case (manual intervention)

This mental model helps understand why we need progressive delays between retries (to allow temporary issues to resolve), why we track error history (to understand the failure pattern), and why we need a dead letter queue (for cases requiring human intervention). The system ensures transient failures (network blips, temporary resource constraints) are automatically resolved while permanent failures (bad data, broken integrations) are surfaced for manual handling.

### Retry Manager Interface

The `RetryManager` orchestrates the entire retry lifecycle, from detecting job failures to scheduling retries and managing the dead letter queue. It works closely with the `Worker` component (which detects failures) and the `Scheduler` component (which polls for scheduled retries).

| Component | Responsibility | Key Methods | Collaborates With |
|-----------|----------------|-------------|-------------------|
| `RetryManager` | Manages retry logic and state | `schedule_retry`, `handle_failure`, `move_to_dead_letter`, `retry_dead_letter_job` | `Worker`, `QueueManager`, `Scheduler` |
| `BackoffCalculator` | Computes retry delays | `calculate_delay`, `add_jitter` | `RetryManager` |
| `DeadLetterQueue` | Stores permanently failed jobs | `store`, `retrieve`, `list`, `delete` | `RetryManager`, `QueueManager` |

The `RetryManager` provides these key interfaces:

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `handle_job_failure` | `job: Job`, `error: Exception` | `None` | Called by `Worker` when a job fails. Records the error, checks retry eligibility, and either schedules a retry or moves to dead letter queue. |
| `schedule_retry` | `job: Job`, `delay_seconds: int` | `str` (job ID) | Schedules a job for retry after specified delay. Returns the job ID for tracking. |
| `move_to_dead_letter` | `job: Job` | `bool` | Moves a job that has exhausted retries to the dead letter queue. Returns success status. |
| `retry_dead_letter_job` | `job_id: str` | `Optional[Job]` | Removes a job from the dead letter queue and re-enqueues it for immediate retry. Returns the re-enqueued job or `None` if not found. |
| `get_retry_stats` | `job_type: Optional[str]` | `Dict[str, Any]` | Returns retry statistics (success rate, average attempts, common errors) for all jobs or filtered by job type. |
| `cleanup_expired_retries` | `max_age_hours: int` | `int` (count removed) | Removes old retry records from Redis to prevent memory bloat. Returns number of records removed. |

The retry process integrates with the job state machine (referenced in ![Job State Machine](./diagrams/job-state-machine.svg)):

1. **Job execution fails** → `Worker` calls `RetryManager.handle_job_failure(job, error)`
2. **Error recording** → Job's `errors` list is updated with exception details
3. **Retry check** → If `job.attempts < job.max_retries`, schedule retry; otherwise move to dead letter
4. **Retry scheduling** → Job status changes to `RETRY_SCHEDULED` and is added to Redis sorted set with execution timestamp
5. **Retry execution** → `Scheduler` polls sorted set, finds due retries, re-enqueues to appropriate queue
6. **Dead letter handling** → Jobs in `DEAD_LETTER` status can be manually inspected and retried via dashboard

### Retry Scheduling and Backoff Algorithm

The retry scheduling algorithm ensures failed jobs are retried with increasing delays to avoid overwhelming recovering systems while maintaining fairness across different job types. Here's the complete algorithm executed by `RetryManager.handle_job_failure`:

1. **Record error details**:
   - Extract exception class name, message, and formatted stack trace
   - Create error dictionary with timestamp and attempt number
   - Append to `job.errors` list (capped at last 10 errors to prevent unbounded growth)

2. **Increment attempt counter**:
   - Increment `job.attempts` by 1
   - Update `job.started_at = None` and `job.completed_at = None` to reset execution timing

3. **Check retry eligibility**:
   - If `job.attempts > job.max_retries`: proceed to dead letter queue
   - If job has `no_retry` metadata flag: proceed to dead letter queue
   - If error is in permanent failure list (configurable): proceed to dead letter queue
   - Otherwise: schedule retry

4. **Calculate retry delay** (exponential backoff with jitter):
   - Base formula: `delay = base_delay * (2 ^ (attempt - 1))`
   - Apply optional jitter: `delay_with_jitter = delay * (1 ± random_jitter_factor)`
   - Enforce maximum delay cap: `final_delay = min(delay_with_jitter, max_delay)`
   - Add minimum floor: `final_delay = max(final_delay, min_delay)`

5. **Schedule retry execution**:
   - Calculate retry timestamp: `retry_at = current_time + final_delay`
   - Set job status to `RETRY_SCHEDULED`
   - Store job in Redis sorted set with score = retry_at (Unix timestamp)
   - Also store job in Redis hash for quick retrieval by job ID

6. **Move to dead letter queue** (if retries exhausted):
   - Set job status to `DEAD_LETTER`
   - Store job in dead letter Redis hash (keyed by job ID)
   - Also add to dead letter list for chronological access
   - Optionally send notification/alert about permanent failure

The retry scheduling uses Redis sorted sets for time-ordered execution:

| Redis Key Pattern | Data Type | Purpose | Example Key |
|-------------------|-----------|---------|-------------|
| `retry:scheduled:{queue}` | Sorted Set (ZSET) | Stores job IDs with retry timestamp as score | `retry:scheduled:email` |
| `retry:jobs:{job_id}` | Hash (HASH) | Stores full job data for scheduled retries | `retry:jobs:01H5XYZABCDEF` |
| `dead:letter:{queue}` | List (LIST) | Chronological list of dead letter job IDs | `dead:letter:email` |
| `dead:jobs:{job_id}` | Hash (HASH) | Stores full job data for dead letter jobs | `dead:jobs:01H5XYZABCDEF` |

Here's a concrete walk-through: Consider a `send_email` job with `max_retries=3` and `base_delay=1` second that fails due to a temporary SMTP connection error.

1. **Attempt 1 fails at 10:00:00** → Error recorded, attempts=1, delay = 1 * 2^(1-1) = 1 second
2. **Scheduled retry at 10:00:01** → Job added to sorted set with score=10:00:01 (Unix timestamp)
3. **Attempt 2 fails at 10:00:01** → Error recorded, attempts=2, delay = 1 * 2^(2-1) = 2 seconds  
4. **Scheduled retry at 10:00:03** → Job added to sorted set with score=10:00:03
5. **Attempt 3 fails at 10:00:03** → Error recorded, attempts=3, delay = 1 * 2^(3-1) = 4 seconds
6. **Scheduled retry at 10:00:07** → Job added to sorted set with score=10:00:07
7. **Attempt 4 (final) fails at 10:00:07** → attempts=4 > max_retries=3 → Move to dead letter queue

> **Key Insight**: The exponential backoff algorithm (`delay = base * 2^(attempt-1)`) creates a geometric progression that quickly increases delays while remaining computationally simple. Adding jitter (±10-20%) prevents retry stampedes where many jobs retry simultaneously after a system outage.

### ADR: Backoff Algorithm Selection

> **Decision: Exponential Backoff with Configurable Jitter**

**Context**: When jobs fail due to transient errors (network timeouts, temporary resource constraints), immediate retries can exacerbate the problem by creating retry storms that overwhelm recovering systems. We need a backoff algorithm that spaces retries further apart with each attempt while providing some randomness to avoid synchronization.

**Options Considered**:

| Option | Pros | Cons |
|--------|------|------|
| **Fixed delay** (constant interval) | Simple to implement and understand | Doesn't adapt to failure severity; can cause persistent overload |
| **Linear backoff** (delay = base * attempt) | Gradually increases delay | May increase too slowly for serious outages; predictable pattern |
| **Exponential backoff** (delay = base * 2^(attempt-1)) | Quickly reduces retry pressure; standard in distributed systems | Can create long waits for users; may exceed reasonable timeouts |
| **Fibonacci backoff** (delay = base * fib(attempt)) | More gradual than exponential; less extreme waits | More complex; less standardized in industry |
| **Randomized exponential** (exponential + jitter) | Prevents synchronization; standard AWS/Google approach | Slightly more complex than pure exponential |

**Decision**: We selected **Exponential backoff with configurable jitter** as the default algorithm.

**Rationale**:
1. **Industry standard**: AWS, Google Cloud, and major message queues use exponential backoff, making it familiar to developers
2. **Mathematical properties**: Doubling delay each attempt provides O(2^n) growth which quickly reduces retry pressure during outages
3. **Configurable**: Base delay and jitter factor can be tuned per job type (sensitive jobs can use smaller base, batch jobs larger)
4. **Jitter prevents synchronization**: Without jitter, all jobs that failed at the same time retry simultaneously, potentially causing cascading failures. Adding ±10-25% random jitter spreads retries.
5. **Predictable worst-case**: Maximum wait time can be calculated as `sum(base * 2^i for i in 0..max_retries-1)` which helps set SLAs

**Consequences**:
- **Positive**: System naturally backs off during outages, reducing load on recovering services
- **Positive**: Jitter prevents thundering herd problems during recovery
- **Negative**: Users of synchronous wrappers may experience long waits (mitigated by setting reasonable max_retries)
- **Negative**: Jobs with tight deadlines may need custom retry strategies (supported via hooks)

The algorithm is implemented in a `BackoffCalculator` class with these configurable parameters:

| Parameter | Default | Description | Rationale |
|-----------|---------|-------------|-----------|
| `base_delay_seconds` | 1 | Initial delay for first retry | 1 second allows quick recovery from transient blips |
| `max_delay_seconds` | 3600 | Maximum delay between retries | 1 hour prevents excessively long delays |
| `jitter_factor` | 0.1 | Random variation (±10%) | 10% provides good desynchronization without excessive unpredictability |
| `min_delay_seconds` | 0.1 | Minimum delay (floor) | 100ms prevents near-instant retries |

### ADR: Dead Letter Queue Design

> **Decision: Dual Storage Dead Letter Queue with Manual Intervention**

**Context**: When jobs exhaust all retry attempts, we need to preserve them for investigation rather than silently discarding them. The dead letter queue (DLQ) must store failed jobs with complete error history while supporting manual review, retry, or deletion. The design must balance storage efficiency with queryability.

**Options Considered**:

| Option | Pros | Cons |
|--------|------|------|
| **Redis List only** | Simple FIFO storage; easy to pop oldest jobs | No random access by job ID; limited metadata querying |
| **Redis Hash only** | Fast job retrieval by ID; supports partial updates | No chronological ordering; hard to list all jobs |
| **Redis Sorted Set** | Chronological ordering by failure time; range queries | Duplicate storage of job data; more complex |
| **Dual storage (List + Hash)** | Chronological list + random access; best of both | Double storage overhead; need to keep synchronized |
| **External database** | Unlimited storage; rich querying | Additional infrastructure; operational complexity |

**Decision**: We selected **Dual storage using Redis List and Hash** for the dead letter queue.

**Rationale**:
1. **Chronological access needed**: Operators want to see "most recent failures" first, which Redis List provides via `LRANGE`
2. **Random access needed**: When investigating a specific job ID from logs/alerts, Hash provides O(1) lookup
3. **Redis already available**: No new infrastructure dependencies
4. **Controlled storage**: We can implement automatic expiration/TTL to prevent unlimited growth
5. **Atomic synchronization**: Redis transactions can keep List and Hash in sync atomically

**Consequences**:
- **Positive**: Both chronological browsing and direct job lookup are efficient
- **Positive**: Leverages existing Redis infrastructure without new dependencies  
- **Negative**: 2x storage overhead for DLQ jobs (mitigated by compression and TTL)
- **Negative**: Manual cleanup required for List/Hash synchronization (handled in cleanup jobs)

The dead letter queue implementation stores each job in two places:

1. **Chronological list**: `dead:letter:{queue}` (Redis List) contains job IDs in insertion order
2. **Job data store**: `dead:jobs:{job_id}` (Redis Hash) contains full serialized job with error history

When moving a job to DLQ, this atomic operation occurs:
```redis
MULTI
HSET dead:jobs:{job_id} data {serialized_job} timestamp {failure_time}
RPUSH dead:letter:{queue} {job_id}
EXPIRE dead:jobs:{job_id} {dlq_ttl}
EXPIRE dead:letter:{queue} {dlq_ttl}
EXEC
```

> **Design Principle**: The dead letter queue is a diagnostic tool, not a permanent archive. Jobs automatically expire after a configurable TTL (default 30 days) to prevent unbounded Redis memory growth. Important failures should be alerted on and investigated promptly.

### Common Pitfalls in Retry Implementation

⚠️ **Pitfall: Infinite retry loops without limits**
- **Description**: Forgetting to enforce `max_retries` or setting it too high (or infinite)
- **Why it's wrong**: Jobs with permanent failures (bad data, broken integrations) retry forever, wasting resources and hiding real problems
- **How to fix**: Always set a reasonable default `max_retries` (e.g., 3-5) and validate that `job.attempts <= job.max_retries` before scheduling retry

⚠️ **Pitfall: Retrying non-idempotent operations without safeguards**
- **Description**: Automatically retrying jobs that have side effects (charging credit cards, sending emails) can cause duplicate charges/spam
- **Why it's wrong**: Business logic executes multiple times for what should be a single operation
- **How to fix**: Either (1) mark non-idempotent jobs with `no_retry` metadata flag, or (2) implement idempotency keys in job handlers that check for previous execution

⚠️ **Pitfall: Losing error context between retries**
- **Description**: Only storing the latest error instead of full error history
- **Why it's wrong**: Operators can't see if the failure mode changed (network → database → validation), which is crucial for debugging
- **How to fix**: Append each error with timestamp, exception class, message, and stack trace to `job.errors` list

⚠️ **Pitfall: Blocking retries on Redis sorted set polling**
- **Description**: Using `ZRANGEBYSCORE` without pagination on large retry sets can block Redis
- **Why it's wrong**: Large numbers of scheduled retries cause long-running Redis commands that delay other operations
- **How to fix**: Process retries in batches (e.g., 100 at a time) and use `ZRANGEBYSCORE ... LIMIT` for pagination

⚠️ **Pitfall: Not handling clock skew in distributed retry scheduling**
- **Description**: Assuming all workers and scheduler have synchronized clocks when comparing retry timestamps
- **Why it's wrong**: Jobs may retry too early or too late if system clocks differ
- **How to fix**: Use Redis server time (`TIME` command) for retry scheduling, not local system time

⚠️ **Pitfall: Dead letter queue without size limits or expiration**
- **Description**: Storing all failed jobs indefinitely in Redis
- **Why it's wrong**: Redis memory fills up, causing evictions or crashes
- **How to fix**: Implement automatic cleanup based on age (TTL) and/or maximum queue size, with configurable limits

### Implementation Guidance for Retry System

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Retry Storage | Redis sorted sets + hashes | Redis Streams with consumer groups |
| Error Serialization | JSON with Python's `traceback.format_exc()` | Structured logging with error codes |
| Backoff Algorithm | Exponential with fixed jitter | Adaptive backoff based on error type |
| Dead Letter Queue | Redis List + Hash with TTL | External database (PostgreSQL) for long-term storage |

**B. Recommended File/Module Structure**

```
background-job-processor/
├── src/
│   ├── job_processor/
│   │   ├── __init__.py
│   │   ├── job.py                    # Job class definition
│   │   ├── redis_client.py           # RedisClient wrapper
│   │   ├── queue_manager.py          # QueueManager implementation
│   │   ├── worker.py                 # Worker implementation  
│   │   ├── retry/                    # ← NEW: Retry subsystem
│   │   │   ├── __init__.py
│   │   │   ├── manager.py            # RetryManager class
│   │   │   ├── backoff.py            # BackoffCalculator class
│   │   │   ├── dead_letter.py        # DeadLetterQueue class
│   │   │   └── errors.py             # Retry-specific exceptions
│   │   ├── scheduler.py              # Scheduler (for retry polling)
│   │   └── utils/
│   │       ├── serialization.py
│   │       └── time_utils.py
├── tests/
│   ├── test_retry_manager.py
│   ├── test_backoff.py
│   └── test_dead_letter.py
└── pyproject.toml
```

**C. Infrastructure Starter Code**

```python
# src/job_processor/retry/backoff.py
"""Exponential backoff calculator with jitter."""
import random
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff with jitter."""
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 3600.0  # 1 hour
    min_delay_seconds: float = 0.1     # 100ms minimum
    jitter_factor: float = 0.1         # ±10% randomness
    max_attempts: int = 5


class BackoffCalculator:
    """Calculates retry delays using exponential backoff with jitter."""
    
    def __init__(self, config: Optional[BackoffConfig] = None):
        self.config = config or BackoffConfig()
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay for given attempt number.
        
        Args:
            attempt: Current attempt number (1-based, where 1 is first retry)
            
        Returns:
            Delay in seconds before next retry attempt
        """
        if attempt < 1:
            attempt = 1
            
        # Exponential backoff: base * 2^(attempt-1)
        exponential_delay = self.config.base_delay_seconds * (2 ** (attempt - 1))
        
        # Apply jitter: add random variation ±jitter_factor
        if self.config.jitter_factor > 0:
            jitter = 1 + random.uniform(
                -self.config.jitter_factor, 
                self.config.jitter_factor
            )
            exponential_delay *= jitter
        
        # Enforce minimum and maximum bounds
        bounded_delay = max(self.config.min_delay_seconds, 
                           min(exponential_delay, self.config.max_delay_seconds))
        
        return bounded_delay
    
    def calculate_next_retry_time(self, attempt: int) -> float:
        """
        Calculate absolute timestamp for next retry attempt.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            Unix timestamp (seconds since epoch) for next retry
        """
        delay = self.calculate_delay(attempt)
        return time.time() + delay
```

**D. Core Logic Skeleton Code**

```python
# src/job_processor/retry/manager.py
"""Retry manager for handling job failures and scheduling retries."""
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..job import Job, JobStatus
from ..redis_client import RedisClient
from .backoff import BackoffCalculator, BackoffConfig
from .dead_letter import DeadLetterQueue


class RetryManager:
    """Manages retry logic for failed jobs."""
    
    def __init__(self, redis_client: RedisClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize retry manager.
        
        Args:
            redis_client: Redis client for storing retry state
            config: Optional configuration dictionary
        """
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Configuration with defaults
        self.config = {
            'base_delay': config.get('base_delay', 1.0),
            'max_delay': config.get('max_delay', 3600.0),
            'max_attempts': config.get('max_attempts', 5),
            'jitter_factor': config.get('jitter_factor', 0.1),
            'retry_key_prefix': config.get('retry_key_prefix', 'retry:'),
            'dead_letter_key_prefix': config.get('dead_letter_key_prefix', 'dead:'),
        }
        
        self.backoff_calculator = BackoffCalculator(
            BackoffConfig(
                base_delay_seconds=self.config['base_delay'],
                max_delay_seconds=self.config['max_delay'],
                max_attempts=self.config['max_attempts'],
                jitter_factor=self.config['jitter_factor']
            )
        )
        
        self.dead_letter_queue = DeadLetterQueue(
            redis_client,
            key_prefix=self.config['dead_letter_key_prefix']
        )
    
    def handle_job_failure(self, job: Job, error: Exception) -> None:
        """
        Handle a job failure by scheduling retry or moving to dead letter queue.
        
        Called by Worker when a job execution fails.
        
        Args:
            job: The failed job instance
            error: Exception that caused the failure
            
        Returns:
            None
        """
        # TODO 1: Record the error in job.errors list using job.record_error(error)
        # Hint: job.record_error() should capture exception class, message, and stack trace
        
        # TODO 2: Increment job.attempts counter
        
        # TODO 3: Check if job should be retried (job.should_retry() method)
        # Consider: max_retries, no_retry metadata flag, permanent error types
        
        # TODO 4: If job should be retried:
        #   - Calculate retry delay using self.backoff_calculator.calculate_delay(job.attempts)
        #   - Call self.schedule_retry(job, delay_seconds)
        #   - Update job.status = JobStatus.RETRY_SCHEDULED
        
        # TODO 5: If job should NOT be retried (retries exhausted or permanent error):
        #   - Call self.move_to_dead_letter(job)
        #   - Update job.status = JobStatus.DEAD_LETTER
        #   - Log warning about permanent failure
        
        # TODO 6: Store updated job state in Redis (job.to_dict() and HSET)
        # Use Redis pipeline for atomic operations
        
        pass
    
    def schedule_retry(self, job: Job, delay_seconds: float) -> str:
        """
        Schedule a job for retry after specified delay.
        
        Args:
            job: Job to retry
            delay_seconds: Delay before retry (in seconds)
            
        Returns:
            Job ID of the scheduled retry
        """
        # TODO 1: Calculate retry timestamp: current_time + delay_seconds
        # Use time.time() for current Unix timestamp
        
        # TODO 2: Create retry key names:
        #   retry_scheduled_key = f"{self.config['retry_key_prefix']}scheduled:{job.queue}"
        #   retry_job_key = f"{self.config['retry_key_prefix']}jobs:{job.job_id}"
        
        # TODO 3: Use Redis pipeline for atomic operations:
        #   - ZADD retry_scheduled_key {retry_timestamp} {job.job_id}
        #   - HSET retry_job_key data {job.serialize()}
        #   - EXPIRE retry_job_key {delay_seconds * 2} (TTL slightly longer than retry delay)
        
        # TODO 4: Execute pipeline and return job.job_id
        
        # TODO 5: Log retry scheduling at INFO level
        pass
    
    def move_to_dead_letter(self, job: Job) -> bool:
        """
        Move a job to the dead letter queue.
        
        Args:
            job: Job that has exhausted retries
            
        Returns:
            True if successful, False otherwise
        """
        # TODO 1: Call self.dead_letter_queue.store(job)
        # The DeadLetterQueue class should handle dual storage (List + Hash)
        
        # TODO 2: Remove job from any retry scheduled sets
        # Use ZREM on retry_scheduled_key
        
        # TODO 3: Return success status
        pass
    
    def get_due_retries(self, queue_name: str, max_count: int = 100) -> List[Job]:
        """
        Retrieve jobs whose retry time has arrived.
        
        Called by Scheduler component.
        
        Args:
            queue_name: Queue to check for due retries
            max_count: Maximum number of jobs to return at once
            
        Returns:
            List of Job instances ready for retry
        """
        # TODO 1: Get current timestamp using time.time()
        
        # TODO 2: Construct retry scheduled key: f"{self.config['retry_key_prefix']}scheduled:{queue_name}"
        
        # TODO 3: Use ZRANGEBYSCORE with scores 0 to current_timestamp, LIMIT 0 max_count
        # This gets job IDs whose retry time has arrived
        
        # TODO 4: For each job_id retrieved:
        #   - Get job data from HGET retry:jobs:{job_id}
        #   - Deserialize into Job object using Job.deserialize()
        #   - Remove from retry scheduled set (ZREM)
        #   - Delete retry job hash (HDEL)
        # Use pipeline for atomic multi-get and cleanup
        
        # TODO 5: Return list of Job objects
        pass
```

**E. Language-Specific Hints**

1. **Error serialization**: Use Python's `traceback.format_exception()` to capture full stack traces:
   ```python
   import traceback
   error_info = {
       'type': type(error).__name__,
       'message': str(error),
       'traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
       'timestamp': datetime.utcnow().isoformat()
   }
   ```

2. **Redis atomic operations**: Always use pipelines for multi-step Redis operations:
   ```python
   pipe = self.redis.get_client().pipeline()
   pipe.zadd('retry:scheduled:email', {job_id: retry_timestamp})
   pipe.hset(f'retry:jobs:{job_id}', 'data', job.serialize())
   pipe.expire(f'retry:jobs:{job_id}', int(delay_seconds * 2))
   pipe.execute()  # All commands execute atomically
   ```

3. **Configuration management**: Use Python's `dataclasses` for type-safe configuration:
   ```python
   from dataclasses import dataclass, field
   from typing import Dict, Any
   
   @dataclass
   class RetryConfig:
       base_delay: float = 1.0
       max_delay: float = 3600.0
       max_attempts: int = 5
       jitter_factor: float = 0.1
       per_job_type_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
   ```

**F. Milestone Checkpoint**

After implementing the retry system, verify functionality with these tests:

1. **Basic retry flow**: 
   ```python
   # Enqueue a job that will fail
   job = Job(job_type='failing_task', max_retries=3)
   job_id = queue_manager.enqueue_job(job)
   
   # Worker processes and fails the job
   worker.start()
   # Job should move to retry scheduled state
   
   # Check Redis for retry scheduling
   retries = retry_manager.get_due_retries('default')
   # Should be empty initially (not yet due)
   
   # Wait for retry delay, then check again
   time.sleep(2)  # Wait for first retry (1s base delay + jitter)
   retries = retry_manager.get_due_retries('default')
   # Should contain the job now
   ```

2. **Dead letter queue test**:
   ```python
   # Create job with max_retries=0 that fails
   job = Job(job_type='permanent_failure', max_retries=0)
   job_id = queue_manager.enqueue_job(job)
   
   # Worker processes and fails
   worker.process_one_job()  # Should move directly to DLQ
   
   # Check dead letter queue
   dlq_jobs = dead_letter_queue.list_jobs('default', limit=10)
   # Should contain the failed job with DEAD_LETTER status
   
   # Manual retry from DLQ
   retried_job = retry_manager.retry_dead_letter_job(job_id)
   # Should return Job object and remove from DLQ
   ```

3. **Error history verification**:
   ```python
   # Process a job that fails multiple times
   job = get_job_from_redis(job_id)
   print(f"Error count: {len(job.errors)}")  # Should equal attempts
   for i, error in enumerate(job.errors):
       print(f"Attempt {i+1}: {error['type']} - {error['message']}")
   # Should show full error history across all attempts
   ```

**Expected output signs**:
- Jobs with transient failures automatically retry with increasing delays
- After final failure, jobs appear in dead letter queue with complete error history
- Redis memory usage remains bounded (TTL working)
- No infinite retry loops occur

**Diagnose problems**:
- **Symptom**: Jobs retry immediately without delay → Check `BackoffCalculator.calculate_delay()` logic
- **Symptom**: Jobs disappear after final failure (not in DLQ) → Check `move_to_dead_letter()` implementation
- **Symptom**: Redis memory growing unbounded → Check TTL/EXPIRE commands on retry and DLQ keys
- **Symptom**: Error history missing stack traces → Check `Job.record_error()` implementation

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Jobs retry forever | `max_retries` not enforced or too high | Check job.attempts vs job.max_retries in logs | Implement retry limit check in `handle_job_failure()` |
| All retries happen simultaneously | Missing jitter in backoff algorithm | Check retry timestamps in Redis sorted set | Add jitter factor to `BackoffCalculator` |
| Error history shows only last error | Errors overwritten instead of appended | Inspect `job.errors` list after multiple failures | Ensure `job.errors.append()` not `job.errors = ` |
| Dead letter queue grows indefinitely | Missing TTL/expiration | Check Redis keys with `TTL dead:jobs:*` | Add EXPIRE commands when storing in DLQ |
| Retries happen at wrong times | Clock skew between systems | Compare Redis TIME with system time | Use Redis server time for scheduling |
| Manual retry from DLQ fails | Job not removed from DLQ lists | Check both List and Hash storage after retry | Ensure atomic removal from both data structures |


## Component Design: Scheduling & Cron

> **Milestone(s):** Milestone 4: Scheduling & Cron. This section provides the complete design for delayed and recurring job scheduling using cron expressions, enabling jobs to be executed at specific future times or on recurring schedules with unique constraint enforcement and timezone handling.

### Mental Model: Calendar App with Recurring Events

Think of the scheduler as a sophisticated calendar application with recurring event capabilities. Just as a calendar app stores events with specific dates, times, and recurrence rules (like "every Monday at 9 AM"), then notifies you when each event occurs, the scheduler stores jobs with execution times and recurrence patterns, then "notifies" the system by enqueuing them at the right moment.

Imagine setting up a weekly team meeting in your calendar:
1. **Initial Setup**: You create an event titled "Weekly Sync" with start time "next Monday 10:00 AM" and recurrence "weekly"
2. **Storage**: The calendar stores this as a recurring event definition with the original start time and recurrence rule
3. **Notification**: Every week, your calendar checks which events are due and sends you a notification
4. **Uniqueness**: If you try to create the same event twice for the same time slot, the calendar prevents duplicates
5. **Missed Events**: If your phone was off during an event time, the calendar shows it as "missed" when you turn it back on

The scheduler operates similarly:
- **Delayed Jobs**: Like one-time calendar events ("dentist appointment next Tuesday at 3 PM")
- **Recurring Jobs**: Like repeating events ("team meeting every Monday at 10 AM")
- **Notification**: Instead of alerting you, it enqueues jobs into worker queues
- **Missed Jobs**: If the scheduler is down, it detects missed schedules and enqueues overdue jobs on restart

This mental model helps understand the core responsibilities: storing schedule definitions, calculating when jobs are due, ensuring uniqueness, and handling timezone complications.

### Scheduler Interface

The scheduler component provides a clean API for scheduling jobs for future execution, whether as one-time delayed jobs or recurring jobs based on cron expressions. It operates as a standalone process that periodically checks for due jobs and enqueues them into the appropriate worker queues.

**Primary Responsibilities:**
1. Store and manage schedule definitions for delayed and recurring jobs
2. Periodically poll for jobs whose execution time has arrived
3. Evaluate cron expressions to calculate next execution times
4. Enforce uniqueness constraints to prevent duplicate job enqueueing
5. Handle timezone conversions consistently
6. Recover from scheduler downtime by catching up on missed jobs

**Core Data Structures:**

| Type Name | Fields | Description |
|-----------|--------|-------------|
| `Schedule` | `schedule_id str`, `job_type str`, `args List[Any]`, `kwargs Dict[str, Any]`, `queue str`, `cron_expression Optional[str]`, `run_at Optional[datetime]`, `timezone str`, `enabled bool`, `unique_key Optional[str]`, `unique_window_seconds int`, `created_at datetime`, `last_enqueued_at Optional[datetime]`, `next_run_at Optional[datetime]`, `metadata Dict[str, Any]` | Defines a scheduled job, either one-time (`run_at`) or recurring (`cron_expression`). The `unique_key` prevents duplicate enqueueing within the `unique_window_seconds`. |
| `ScheduledJob` | `schedule_id str`, `job_id str`, `scheduled_for datetime`, `enqueued_at Optional[datetime]`, `status ScheduleStatus` | Represents a specific scheduled execution instance, tracking when a job from a schedule should run and whether it has been enqueued. |
| `ScheduleStatus` | `PENDING`, `ENQUEUED`, `SKIPPED`, `ERROR` | Status of individual scheduled job instances. `SKIPPED` occurs when uniqueness constraint prevents enqueueing. |

**Scheduler Interface Methods:**

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `schedule_delayed_job` | `job Job`, `run_at datetime` | `str` (schedule_id) | Schedule a job for one-time execution at `run_at`. Validates that `run_at` is in the future. |
| `schedule_recurring_job` | `job Job`, `cron_expression str`, `timezone str = "UTC"` | `str` (schedule_id) | Schedule a job for recurring execution based on cron expression. Validates cron syntax and calculates initial `next_run_at`. |
| `unschedule_job` | `schedule_id str` | `bool` | Remove a schedule by ID, preventing future executions. Returns `True` if schedule existed and was removed. |
| `pause_schedule` | `schedule_id str` | `bool` | Temporarily disable a schedule without removing it. Future polling will skip this schedule. |
| `resume_schedule` | `schedule_id str` | `bool` | Re-enable a paused schedule. Next run time is recalculated from current time. |
| `get_schedule` | `schedule_id str` | `Optional[Schedule]` | Retrieve schedule definition by ID, including current status and next run time. |
| `list_schedules` | `enabled_only bool = False` | `List[Schedule]` | List all schedules, optionally filtered to only enabled ones. |
| `update_schedule` | `schedule_id str`, `updates Dict[str, Any]` | `bool` | Update specific fields of a schedule (e.g., cron expression, timezone). Recalculates `next_run_at` if cron or timezone changes. |
| `enqueue_due_jobs` | `max_jobs int = 100` | `int` (jobs enqueued) | Main polling method: finds due jobs, enqueues them to worker queues, updates schedule state. Returns count of jobs successfully enqueued. |

**Schedule Storage Strategy:**

The scheduler uses two primary Redis data structures:
1. **Schedule Definitions Hash**: `schedules:{schedule_id}` stores the full `Schedule` object as a hash for quick lookup and updates
2. **Schedule Timeline Sorted Set**: `schedules:timeline` uses `next_run_at` timestamps as scores and `schedule_id` as members for efficient polling of due schedules

**Example Schedule Creation Flow:**
1. Client calls `schedule_recurring_job(job, "0 9 * * 1", "America/New_York")` (every Monday at 9 AM Eastern)
2. Scheduler validates cron expression, parses timezone, calculates next Monday 9 AM in that timezone
3. Creates `Schedule` object with `schedule_id`, stores in Redis hash
4. Adds `schedule_id` to `schedules:timeline` sorted set with score = next run timestamp
5. Returns `schedule_id` to client for future reference

### Scheduler Polling and Cron Evaluation

The scheduler operates on a polling model where it periodically checks for due schedules and enqueues them. This approach balances simplicity with reliability, avoiding complex event-driven architectures while ensuring no schedules are missed during brief downtimes.

**Polling Algorithm (Numbered Steps):**
1. **Initialize Polling Loop**: The scheduler starts with configuration defining `polling_interval_seconds` (default: 60 seconds) and `batch_size` (default: 100 jobs per poll)
2. **Calculate Polling Window**: Determine current time in UTC (`now_utc`) and calculate `poll_until = now_utc + polling_interval_seconds` to catch jobs due within the next polling interval
3. **Retrieve Due Schedule IDs**: Use Redis `ZRANGEBYSCORE` on `schedules:timeline` with score range `0` to `poll_until` to get schedule IDs due for execution
4. **Process Each Due Schedule**:
   1. Fetch full schedule definition from Redis hash
   2. Validate schedule is still `enabled` (skip if paused)
   3. Apply uniqueness check using `unique_key` and `unique_window_seconds`
   4. Create `Job` instance from schedule's `job_type`, `args`, `kwargs`, and `queue`
   5. Generate `job_id` using ULID for chronological sorting
   6. Use `QueueManager.enqueue_job()` to enqueue job to appropriate worker queue
   7. Record `ScheduledJob` instance tracking this execution
   8. Update schedule's `last_enqueued_at` to current time
5. **Calculate Next Run for Recurring Jobs**: For schedules with cron expressions, calculate next execution time using current time as reference, update `next_run_at`, and update score in timeline sorted set
6. **Cleanup One-Time Delayed Jobs**: For schedules with only `run_at` (no cron), remove from timeline and delete schedule definition after enqueueing
7. **Handle Errors Gracefully**: If any schedule fails during processing (e.g., queue full, Redis error), log error, increment error counter, but continue processing other schedules
8. **Sleep Until Next Poll**: Wait `polling_interval_seconds` before repeating, but adjust for processing time (e.g., if processing took 15 seconds, sleep only 45 seconds)

**Cron Expression Evaluation:**

The system uses standard 5-field cron expressions (minute, hour, day of month, month, day of week) with the following parsing and evaluation logic:

| Field | Allowed Values | Special Characters | Notes |
|-------|----------------|-------------------|-------|
| Minute | 0-59 | `*` , `-` , `/` , `,` | `*/5` = every 5 minutes |
| Hour | 0-23 | `*` , `-` , `/` , `,` | `0-8` = hours 0 through 8 |
| Day of Month | 1-31 | `*` , `-` , `/` , `,` , `?` , `L` | `L` = last day of month |
| Month | 1-12 or JAN-DEC | `*` , `-` , `/` , `,` | Case-insensitive month names |
| Day of Week | 0-7 or SUN-SAT | `*` , `-` , `/` , `,` , `?` , `L` , `#` | 0 and 7 = Sunday, `2#3` = third Tuesday |

**Next Run Calculation Algorithm:**
1. **Start Reference Time**: Use current time in schedule's timezone, or `last_enqueued_at` if calculating next run after successful enqueue
2. **Increment to Next Candidate**: Add 1 minute to reference time to ensure we don't re-enqueue for same minute
3. **Iterate Through Fields**: Check each cron field against candidate time:
   1. If minute doesn't match, increment minutes until match found
   2. If hour doesn't match, increment hours (reset minutes to 0)
   3. If day of month doesn't match, increment days (reset hours/minutes)
   4. If month doesn't match, increment months (reset days/hours/minutes)
   5. If day of week doesn't match, increment days (could require checking multiple days)
4. **Handle Edge Cases**: 
   - 31st day in months with only 30 days → roll to 1st of next month
   - February 29th in non-leap years → roll to March 1st
   - Last day of month (`L`) → calculate actual last day for that month/year
5. **Return Valid Time**: First time that matches all cron fields, converted back to UTC for storage

**Uniqueness Constraint Implementation:**

To prevent duplicate enqueueing of the same recurring job (e.g., if scheduler polls twice in same minute), the system implements a window-based uniqueness check:

1. **Generate Unique Key**: Combine schedule ID with execution period identifier (e.g., `"email_report:2024-03-25-09:00"` for hourly job at 9 AM on March 25)
2. **Check Redis Lock**: Use Redis `SET key value NX EX window_seconds` as atomic check-and-set
3. **If Key Exists**: Job already enqueued within uniqueness window → skip enqueueing, mark as `SKIPPED`
4. **If Key Doesn't Exist**: Proceed with enqueueing, set key with TTL = `unique_window_seconds`

**Example Polling Cycle:**
- Current time: 2024-03-25 09:01:30 UTC
- Polling interval: 60 seconds
- Due schedules in timeline with scores ≤ 2024-03-25 09:02:30 (now + 60s)
- Found: `schedule:report:daily` with cron `0 9 * * *` (9 AM daily)
- Schedule timezone: America/New_York (UTC-4 currently)
- Check: 9 AM Eastern = 13:00 UTC, not due yet → skip
- Found: `schedule:cleanup:hourly` with cron `0 * * * *` (top of every hour)
- Check: 09:00 UTC just passed, within polling window → enqueue job
- Calculate next run: next hour at 10:00 UTC
- Update timeline score to 10:00 UTC timestamp

![Scheduler Polling Algorithm Flowchart](./diagrams/scheduler-polling-flowchart.svg)

### ADR: Scheduler Architecture (Polling vs Event)

> **Decision: Polling-Based Scheduler Architecture**
> - **Context**: We need to execute jobs at specific future times or on recurring schedules. The scheduler must be reliable, handle missed schedules during downtime, and not overwhelm Redis with constant queries. We considered event-driven approaches using Redis keyspace notifications or separate timer services.
> - **Options Considered**:
>   1. **Polling Architecture**: Scheduler process periodically queries Redis for due schedules
>   2. **Event-Driven with Redis Keyspace Notifications**: Use Redis `PSUBSCRIBE` to notifications when sorted set scores pass thresholds
>   3. **External Timer Service**: Use dedicated timer service like RabbitMQ's delayed message exchange or Redis Streams with consumer groups
> - **Decision**: Implement polling architecture with configurable interval (default 60 seconds)
> - **Rationale**: 
>   1. **Simplicity**: Polling is straightforward to implement, debug, and reason about
>   2. **Reliability**: Missed polls (scheduler restart) are automatically caught on next poll using timeline scores
>   3. **Resource Efficiency**: 60-second intervals generate minimal Redis load (one ZRANGEBYSCORE per poll)
>   4. **Precision Acceptable**: Most scheduled jobs (daily reports, hourly cleanups) don't require sub-minute precision
>   5. **Avoids Complex Dependencies**: Redis keyspace notifications require additional configuration and have delivery guarantees that complicate reliability
> - **Consequences**:
>   - **Positive**: Simple implementation, easy to test, handles scheduler restarts gracefully
>   - **Negative**: Jobs may be enqueued up to `polling_interval_seconds` late (worst-case)
>   - **Mitigation**: Use shorter intervals (e.g., 10 seconds) for time-sensitive jobs, though this increases Redis load

**Comparison of Scheduler Architecture Options:**

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Polling** | Simple implementation, self-contained, handles downtime naturally, tunable precision vs load trade-off | Jobs can be delayed up to polling interval, constant Redis queries even when no jobs due | **Chosen** - Best balance of simplicity and reliability for this use case |
| **Redis Keyspace Notifications** | Event-driven, near-instantaneous triggering, reduces polling load | Complex setup (notify-keyspace-events), notifications can be lost if client disconnected, requires persistent connection | Risk of lost notifications undermines reliability; added complexity not justified |
| **Redis Streams Consumer Groups** | Built-in Redis reliability features, consumer tracking, message persistence | Complex implementation, requires managing consumer groups, overkill for scheduling | Scheduling doesn't need full message queue semantics; polling is simpler |
| **External Timer Service** | Potentially more precise, dedicated service for timing | Additional dependency, operational complexity, network latency | Violates YAGNI; polling meets requirements without new dependencies |

**Polling Interval Trade-off Analysis:**

| Interval | Precision | Redis Load (QPS) | Missed Schedule Risk | Recommended For |
|----------|-----------|------------------|----------------------|-----------------|
| 60 seconds | ±60 seconds | 0.017 queries/sec | Very low | Default: daily/hourly jobs, system maintenance |
| 30 seconds | ±30 seconds | 0.033 queries/sec | Low | Moderate precision needs: cache warming, periodic sync |
| 10 seconds | ±10 seconds | 0.1 queries/sec | Moderate | Higher precision: near-real-time batch processing |
| 1 second | ±1 second | 1 query/sec | High | Real-time scheduling (rarely needed) |

> **Design Insight**: The polling interval represents a classic trade-off between timeliness and resource usage. By making it configurable per scheduler instance, we allow different deployment profiles: a primary scheduler with 60-second intervals for most jobs, and a secondary scheduler with 10-second intervals for time-sensitive jobs, both reading from the same schedule store.

### ADR: Timezone Handling Strategy

> **Decision: Store All Times in UTC, Evaluate Cron in Schedule Timezone**
> - **Context**: Scheduled jobs often need to run at specific local times (e.g., "9 AM in New York" for morning reports). We must handle timezones correctly, including daylight saving time transitions, without introducing ambiguity or complexity in storage and comparison.
> - **Options Considered**:
>   1. **UTC-Only**: Store all times in UTC, require users to convert cron expressions to UTC
>   2. **Schedule Timezone with UTC Storage**: Store next run time in UTC but calculate it using schedule's timezone
>   3. **Local Time Storage**: Store times in local timezone, convert during comparisons
>   4. **Timezone-Aware Cron**: Extend cron syntax with timezone field (e.g., `0 9 * * * America/New_York`)
> - **Decision**: Store `next_run_at` in UTC, but evaluate cron expressions in the schedule's configured timezone
> - **Rationale**:
>   1. **Comparison Simplicity**: All timeline comparisons in Redis use UTC timestamps, avoiding timezone conversions during polling
>   2. **Daylight Saving Handling**: Using Python's `pytz` or `zoneinfo` libraries handles DST transitions automatically when calculating next run
>   3. **Storage Consistency**: UTC storage ensures consistent sorting and comparison in Redis sorted sets
>   4. **User Expectation**: Users specify schedules in local time (e.g., "9 AM Eastern"), not UTC equivalents which change with DST
> - **Consequences**:
>   - **Positive**: Clean separation of concerns, handles DST correctly, consistent storage format
>   - **Negative**: Requires timezone database (pytz/zoneinfo), extra conversion step during cron evaluation
>   - **Edge Cases**: Ambiguous times during DST fall-back (2 AM occurs twice) must be handled by choosing first occurrence

**Timezone Implementation Details:**

1. **Schedule Configuration**: Each `Schedule` has `timezone` field (string like "America/New_York")
2. **Cron Evaluation**:
   - Load timezone object using `zoneinfo.ZoneInfo(timezone)` (Python 3.9+) or `pytz.timezone(timezone)`
   - Convert current reference time from UTC to schedule timezone
   - Calculate next run in schedule timezone
   - Convert next run back to UTC for storage
3. **UTC Storage**: All Redis timeline scores and `run_at` fields are UNIX timestamps (seconds since epoch) or ISO 8601 strings ending with "Z"
4. **Daylight Saving Transitions**:
   - Spring forward (1 AM → 3 AM): Jobs scheduled for 2:30 AM will run at 3:30 AM on transition day
   - Fall back (2 AM → 1 AM): Jobs scheduled for 1:30 AM will run twice (once before, once after transition)

**Example: Daily Report at 9 AM Eastern Through DST Transition:**

| Date | Eastern Time (EDT/EST) | UTC Equivalent | Notes |
|------|------------------------|----------------|-------|
| March 10, 2024 | 09:00 EDT | 13:00 UTC | Before DST (Eastern Daylight Time) |
| March 11, 2024 | 09:00 EDT | 13:00 UTC | DST begins at 2 AM → 3 AM |
| November 3, 2024 | 09:00 EST | 14:00 UTC | DST ends at 2 AM → 1 AM |
| November 4, 2024 | 09:00 EST | 14:00 UTC | After DST (Eastern Standard Time) |

The scheduler correctly handles this by:
1. Storing `next_run_at` as UTC timestamp (13:00 or 14:00 UTC)
2. When calculating next run on Nov 3, 2024:
   - Current time: 2024-11-03 13:00 UTC (just ran)
   - Convert to Eastern: 2024-11-03 09:00 EDT
   - Add 1 day for daily schedule: 2024-11-04 09:00 EST (note timezone change)
   - Convert to UTC: 2024-11-04 14:00 UTC
   - Store `next_run_at` = 14:00 UTC

**Comparison of Timezone Strategies:**

| Strategy | Storage Format | DST Handling | User Experience | Implementation Complexity |
|----------|----------------|--------------|-----------------|---------------------------|
| **UTC-Only** | All times UTC | User must handle | Poor: users calculate UTC offsets | Simple but error-prone for users |
| **UTC Storage, TZ Evaluation** | Times UTC, TZ metadata | Automatic via timezone libs | Good: users specify local time | Moderate: conversion logic |
| **Local Time Storage** | Times with timezone | Complex comparisons | Good: intuitive storage | High: comparison logic complex |
| **Timezone-Aware Cron** | Extended cron syntax | Built into syntax | Best: explicit in expression | High: custom parser needed |

> **Design Insight**: Timezone handling is a classic "store in UTC, display in local" pattern adapted for scheduling. By performing timezone conversion only during next-run calculation (not during storage or comparison), we maintain simple UTC-based polling while respecting local time requirements.

### Common Pitfalls in Scheduler Implementation

⚠️ **Pitfall: Naïve Cron Evaluation Without Timezone Consideration**
- **Description**: Evaluating cron expressions using server local time instead of schedule's configured timezone, causing jobs to run at wrong times when server timezone differs from schedule timezone.
- **Why It's Wrong**: A server in UTC running a "9 AM Eastern" report would enqueue it at 9 AM UTC (5 AM Eastern), four hours early. This violates user expectations and can break time-sensitive operations.
- **Fix**: Always convert reference time to schedule's timezone before cron evaluation, then convert result back to UTC for storage.

⚠️ **Pitfall: Polling Without Catch-up Logic**
- **Description**: Scheduler only checks for jobs with `next_run_at <= now()`, missing schedules that were due during scheduler downtime.
- **Why It's Wrong**: If scheduler restarts after being down for 2 hours, hourly jobs scheduled during that period would be skipped entirely, breaking recurring job guarantees.
- **Fix**: Always poll with range `0` to `now() + polling_interval`, not just `now()`. For additional safety, implement a "catch-up" mode on scheduler startup that processes all schedules with `next_run_at < now()`.

⚠️ **Pitfall: Infinite Loop on Invalid Cron Expressions**
- **Description**: When calculating next run time for invalid cron like `0 0 31 2 *` (February 31st), naïve algorithms may get stuck trying to find a matching date.
- **Why It's Wrong**: Scheduler process hangs indefinitely or crashes when encountering malformed schedules, affecting all scheduled jobs.
- **Fix**: Implement maximum iteration limits (e.g., 100,000 minutes ≈ 70 days) when searching for next matching time. Validate cron expressions when schedules are created, rejecting obviously invalid ones.

⚠️ **Pitfall: Race Conditions in Uniqueness Checking**
- **Description**: Two scheduler instances polling simultaneously might both pass uniqueness check and enqueue the same job, creating duplicates.
- **Why It's Wrong**: Duplicate jobs waste resources and may cause application-level issues if jobs aren't idempotent.
- **Fix**: Use Redis `SET key value NX EX window` for atomic check-and-set. The first scheduler to set the key wins; others find key exists and skip enqueueing.

⚠️ **Pitfall: Not Handling DST Transition Ambiguity**
- **Description**: During fall-back transition (2 AM → 1 AM), 1:30 AM occurs twice. Naïve scheduling might pick wrong occurrence or skip entirely.
- **Why It's Wrong**: Jobs might run at wrong time (1:30 AM before transition instead of after) or be skipped on transition day.
- **Fix**: Use `fold` parameter in Python's `datetime` or `pytz` `localize` with `is_dst` parameter to handle ambiguous times consistently (typically choose first occurrence).

⚠️ **Pitfall: Stale Schedule Definitions**
- **Description**: Schedules remain in Redis after being deleted or disabled, causing continuous errors during polling.
- **Why It's Wrong**: Polling loop constantly encounters missing schedule data, filling logs with errors and potentially affecting performance.
- **Fix**: Implement stale schedule cleanup: when fetching schedule fails (missing hash), remove it from timeline sorted set. Add health check metric for orphaned timeline entries.

⚠️ **Pitfall: Blocking Polling Loop**
- **Description**: Processing many due schedules synchronously causes polling loop to exceed interval, delaying subsequent polls.
- **Why It's Wrong**: Scheduler falls behind real time, causing increasing latencies for all scheduled jobs.
- **Fix**: Process schedules in bounded batches (e.g., 100 per poll). Use async processing or worker pool for enqueue operations. Monitor average processing time vs polling interval.

### Implementation Guidance for Scheduler

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Cron Parsing** | `croniter` library (pip install croniter) | Custom parser using regex and datetime math |
| **Timezone Handling** | Python 3.9+ `zoneinfo` (standard library) | `pytz` library (more comprehensive historical data) |
| **Polling Loop** | `while True` with `time.sleep()` | `asyncio` with async Redis client for concurrent processing |
| **Unique Key Generation** | Redis `SET NX EX` atomic operations | Redis Lua script for complex uniqueness logic |
| **Schedule Storage** | Redis Hashes for schedules, Sorted Set for timeline | RedisJSON module for nested schedule objects |

**B. Recommended File/Module Structure:**

```
background-job-processor/
├── scheduler/                    # Scheduler component
│   ├── __init__.py
│   ├── scheduler.py             # Main Scheduler class
│   ├── cron.py                  # Cron expression parser/evaluator
│   ├── timezone.py              # Timezone handling utilities
│   ├── models.py                # Schedule, ScheduledJob classes
│   └── cli.py                   # Command-line interface
├── core/                        # Shared core components
│   ├── job.py                   # Job class (shared)
│   ├── queue_manager.py         # QueueManager (shared)
│   └── redis_client.py          # RedisClient (shared)
└── config/
    └── settings.py              # Configuration management
```

**C. Infrastructure Starter Code (COMPLETE, ready to use):**

```python
# scheduler/timezone.py
"""Timezone handling utilities for schedule evaluation."""
import datetime
from typing import Optional
from zoneinfo import ZoneInfo

# Fallback for Python < 3.9
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


def ensure_timezone(dt: datetime.datetime, timezone: str) -> datetime.datetime:
    """Ensure datetime has timezone, converting naive to specified timezone.
    
    Args:
        dt: Datetime (naive or aware)
        timezone: IANA timezone string (e.g., "America/New_York")
    
    Returns:
        Timezone-aware datetime in specified timezone
    
    Raises:
        ValueError: If timezone string is invalid
    """
    if dt.tzinfo is None:
        # Naive datetime, assume it's in the specified timezone
        tz = ZoneInfo(timezone)
        return dt.replace(tzinfo=tz)
    else:
        # Already timezone-aware, convert to target timezone
        tz = ZoneInfo(timezone)
        return dt.astimezone(tz)


def to_utc(dt: datetime.datetime) -> datetime.datetime:
    """Convert timezone-aware datetime to UTC.
    
    Args:
        dt: Timezone-aware datetime
    
    Returns:
        UTC datetime
    
    Raises:
        ValueError: If dt is naive (no timezone)
    """
    if dt.tzinfo is None:
        raise ValueError("Cannot convert naive datetime to UTC")
    return dt.astimezone(datetime.timezone.utc)


def from_utc(utc_dt: datetime.datetime, timezone: str) -> datetime.datetime:
    """Convert UTC datetime to specified timezone.
    
    Args:
        utc_dt: UTC datetime (must be timezone-aware with UTC)
        timezone: Target IANA timezone string
    
    Returns:
        Datetime in target timezone
    """
    if utc_dt.tzinfo != datetime.timezone.utc:
        raise ValueError(f"Expected UTC datetime, got {utc_dt.tzinfo}")
    
    target_tz = ZoneInfo(timezone)
    return utc_dt.astimezone(target_tz)


def next_dst_transition(timezone: str, after_dt: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
    """Find next DST transition after given datetime.
    
    Args:
        timezone: IANA timezone string
        after_dt: Datetime to search after (defaults to now)
    
    Returns:
        Datetime of next DST transition, or None if no upcoming transitions
    """
    tz = ZoneInfo(timezone)
    after = after_dt or datetime.datetime.now(datetime.timezone.utc)
    
    # Check transitions up to 2 years in future
    end = after + datetime.timedelta(days=730)
    
    # Get transitions from zoneinfo
    # Note: This is a simplified implementation
    # Full implementation would use pytz for historical transitions
    transitions = getattr(tz, '_transitions', [])
    
    for trans_time, trans_info in transitions:
        trans_dt = datetime.datetime.fromtimestamp(trans_time, tz=datetime.timezone.utc)
        if after < trans_dt < end:
            return trans_dt
    
    return None
```

```python
# scheduler/models.py
"""Data models for schedule definitions."""
import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.job import Job


class ScheduleStatus(Enum):
    """Status of a scheduled job instance."""
    PENDING = "pending"
    ENQUEUED = "enqueued"
    SKIPPED = "skipped"  # Due to uniqueness constraint
    ERROR = "error"


@dataclass
class Schedule:
    """Definition of a scheduled job (recurring or one-time)."""
    schedule_id: str = field(default_factory=lambda: f"schedule_{uuid.uuid4().hex}")
    job_type: str = ""
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    queue: str = "default"
    
    # Scheduling configuration
    cron_expression: Optional[str] = None  # e.g., "0 9 * * *"
    run_at: Optional[datetime.datetime] = None  # For one-time delayed jobs
    
    # Timezone handling
    timezone: str = "UTC"
    
    # State
    enabled: bool = True
    unique_key: Optional[str] = None  # For uniqueness constraint
    unique_window_seconds: int = 300  # 5 minutes default uniqueness window
    
    # Metadata
    created_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    last_enqueued_at: Optional[datetime.datetime] = None
    next_run_at: Optional[datetime.datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule to dictionary for serialization."""
        return {
            'schedule_id': self.schedule_id,
            'job_type': self.job_type,
            'args': self.args,
            'kwargs': self.kwargs,
            'queue': self.queue,
            'cron_expression': self.cron_expression,
            'run_at': self.run_at.isoformat() if self.run_at else None,
            'timezone': self.timezone,
            'enabled': self.enabled,
            'unique_key': self.unique_key,
            'unique_window_seconds': self.unique_window_seconds,
            'created_at': self.created_at.isoformat(),
            'last_enqueued_at': self.last_enqueued_at.isoformat() if self.last_enqueued_at else None,
            'next_run_at': self.next_run_at.isoformat() if self.next_run_at else None,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schedule':
        """Reconstruct schedule from dictionary."""
        # Parse datetime fields
        run_at = None
        if data.get('run_at'):
            run_at = datetime.datetime.fromisoformat(data['run_at'])
        
        last_enqueued_at = None
        if data.get('last_enqueued_at'):
            last_enqueued_at = datetime.datetime.fromisoformat(data['last_enqueued_at'])
        
        next_run_at = None
        if data.get('next_run_at'):
            next_run_at = datetime.datetime.fromisoformat(data['next_run_at'])
        
        created_at = datetime.datetime.fromisoformat(data['created_at'])
        
        return cls(
            schedule_id=data['schedule_id'],
            job_type=data['job_type'],
            args=data['args'],
            kwargs=data['kwargs'],
            queue=data['queue'],
            cron_expression=data.get('cron_expression'),
            run_at=run_at,
            timezone=data.get('timezone', 'UTC'),
            enabled=data.get('enabled', True),
            unique_key=data.get('unique_key'),
            unique_window_seconds=data.get('unique_window_seconds', 300),
            created_at=created_at,
            last_enqueued_at=last_enqueued_at,
            next_run_at=next_run_at,
            metadata=data.get('metadata', {}),
        )
    
    def create_job(self) -> Job:
        """Create a Job instance from this schedule."""
        from core.job import Job, JobStatus
        
        # Generate job_id with ULID for chronological sorting
        # Using uuid as fallback
        import ulid
        job_id = str(ulid.ULID())
        
        return Job(
            job_id=job_id,
            job_type=self.job_type,
            args=self.args,
            kwargs=self.kwargs,
            queue=self.queue,
            priority=0,  # Default priority
            max_retries=3,  # Default retries
            timeout_seconds=300,  # Default timeout
            created_at=datetime.datetime.utcnow(),
            metadata={
                **self.metadata,
                'schedule_id': self.schedule_id,
                'scheduled_for': self.next_run_at.isoformat() if self.next_run_at else None,
            },
            status=JobStatus.PENDING,
            attempts=0,
            errors=[],
            started_at=None,
            completed_at=None,
            result=None,
        )


@dataclass
class ScheduledJob:
    """Instance of a scheduled job execution."""
    schedule_id: str
    job_id: str
    scheduled_for: datetime.datetime
    enqueued_at: Optional[datetime.datetime] = None
    status: ScheduleStatus = ScheduleStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'schedule_id': self.schedule_id,
            'job_id': self.job_id,
            'scheduled_for': self.scheduled_for.isoformat(),
            'enqueued_at': self.enqueued_at.isoformat() if self.enqueued_at else None,
            'status': self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledJob':
        """Reconstruct from dictionary."""
        scheduled_for = datetime.datetime.fromisoformat(data['scheduled_for'])
        
        enqueued_at = None
        if data.get('enqueued_at'):
            enqueued_at = datetime.datetime.fromisoformat(data['enqueued_at'])
        
        status = ScheduleStatus(data['status'])
        
        return cls(
            schedule_id=data['schedule_id'],
            job_id=data['job_id'],
            scheduled_for=scheduled_for,
            enqueued_at=enqueued_at,
            status=status,
        )
```

**D. Core Logic Skeleton Code (signature + TODOs only):**

```python
# scheduler/scheduler.py
"""Main scheduler implementation."""
import datetime
import logging
import time
from typing import List, Optional, Tuple

from croniter import croniter

from core.job import Job
from core.queue_manager import QueueManager
from core.redis_client import RedisClient
from scheduler.models import Schedule, ScheduledJob, ScheduleStatus
from scheduler.timezone import ensure_timezone, to_utc, from_utc


class Scheduler:
    """Main scheduler class for delayed and recurring job scheduling."""
    
    def __init__(self, redis_client: RedisClient, queue_manager: QueueManager,
                 polling_interval_seconds: int = 60, batch_size: int = 100):
        """Initialize scheduler.
        
        Args:
            redis_client: Redis client for schedule storage
            queue_manager: Queue manager for enqueuing jobs
            polling_interval_seconds: How often to poll for due schedules
            batch_size: Maximum schedules to process per poll
        """
        self.redis = redis_client
        self.queue_manager = queue_manager
        self.polling_interval = polling_interval_seconds
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self._running = False
        
        # Redis key patterns
        self.schedule_key = "schedule:{schedule_id}"
        self.timeline_key = "schedules:timeline"
        self.uniqueness_key = "schedule:unique:{unique_key}"
        self.scheduled_jobs_key = "schedule:jobs:{schedule_id}"
    
    def schedule_delayed_job(self, job: Job, run_at: datetime.datetime) -> str:
        """Schedule a job for one-time delayed execution.
        
        Args:
            job: Job to schedule
            run_at: When to enqueue the job (must be in future)
        
        Returns:
            Schedule ID for future reference
        
        Raises:
            ValueError: If run_at is in the past
        """
        # TODO 1: Validate run_at is in the future (compare to UTC now)
        # TODO 2: Create Schedule object with run_at (no cron_expression)
        # TODO 3: Generate unique schedule_id
        # TODO 4: Calculate next_run_at (same as run_at)
        # TODO 5: Store schedule in Redis hash (schedule_key)
        # TODO 6: Add to timeline sorted set with score = next_run_at timestamp
        # TODO 7: Return schedule_id
        
        # HINT: Use to_utc() to convert run_at to UTC for storage
        # HINT: Use redis.pipeline() for atomic schedule creation + timeline addition
    
    def schedule_recurring_job(self, job: Job, cron_expression: str,
                               timezone: str = "UTC") -> str:
        """Schedule a job for recurring execution based on cron expression.
        
        Args:
            job: Job to schedule
            cron_expression: Cron expression (e.g., "0 9 * * *")
            timezone: IANA timezone for schedule evaluation
        
        Returns:
            Schedule ID for future reference
        
        Raises:
            ValueError: If cron expression is invalid
        """
        # TODO 1: Validate cron expression using croniter
        # TODO 2: Create Schedule object with cron_expression and timezone
        # TODO 3: Generate unique schedule_id
        # TODO 4: Calculate initial next_run_at using calculate_next_run()
        # TODO 5: Store schedule in Redis hash
        # TODO 6: Add to timeline sorted set with score = next_run_at timestamp
        # TODO 7: Return schedule_id
        
        # HINT: Use self._calculate_next_run() for initial calculation
        # HINT: Consider using current time in schedule's timezone as reference
    
    def _calculate_next_run(self, schedule: Schedule,
                           reference_time: Optional[datetime.datetime] = None) -> datetime.datetime:
        """Calculate next run time for a schedule.
        
        Args:
            schedule: Schedule definition
            reference_time: Time to calculate from (defaults to now)
        
        Returns:
            Next run time in UTC
        
        Raises:
            ValueError: If schedule has neither cron nor run_at
        """
        # TODO 1: If schedule has run_at (one-time), return run_at (already in UTC)
        # TODO 2: If schedule has cron_expression:
        #   a. Determine reference time (use last_enqueued_at or now if None)
        #   b. Convert reference time to schedule's timezone
        #   c. Use croniter to get next matching time in schedule's timezone
        #   d. Handle edge cases: maximum iterations, invalid dates
        #   e. Convert result back to UTC
        # TODO 3: Return UTC datetime
        
        # HINT: For cron, add 1 minute to reference time to avoid re-enqueuing immediately
        # HINT: Use croniter.croniter.get_next(datetime) for next run calculation
        # HINT: Wrap in try/except for croniter errors
    
    def _enqueue_scheduled_job(self, schedule: Schedule,
                              scheduled_for: datetime.datetime) -> Tuple[bool, Optional[str]]:
        """Enqueue a job from a schedule.
        
        Args:
            schedule: Schedule definition
            scheduled_for: When this job was scheduled to run
        
        Returns:
            Tuple of (success, job_id or error message)
        """
        # TODO 1: Check uniqueness constraint if schedule.unique_key is set
        # TODO 2: If unique_key exists and within window, return (False, "skipped")
        # TODO 3: Create Job instance from schedule
        # TODO 4: Enqueue job using queue_manager.enqueue_job()
        # TODO 5: If successful:
        #   a. Record ScheduledJob instance
        #   b. Update schedule.last_enqueued_at
        #   c. Set uniqueness key with TTL if needed
        # TODO 6: Return (success, job_id or error)
        
        # HINT: For uniqueness check: redis.set(key, 1, nx=True, ex=window)
        # HINT: Use pipeline for atomic updates
    
    def _poll_due_schedules(self) -> int:
        """Poll for and process due schedules.
        
        Returns:
            Number of jobs successfully enqueued
        """
        # TODO 1: Get current UTC time
        # TODO 2: Calculate cutoff = now + polling_interval (for catch-up)
        # TODO 3: Use ZRANGEBYSCORE to get schedule_ids with score <= cutoff
        # TODO 4: For each schedule_id (up to batch_size):
        #   a. Fetch schedule from Redis hash
        #   b. Skip if schedule is None or not enabled
        #   c. Process schedule using _process_schedule()
        # TODO 5: Return count of successfully enqueued jobs
        
        # HINT: Use redis.pipeline() for batch operations
        # HINT: Handle missing schedules (remove from timeline)
    
    def _process_schedule(self, schedule: Schedule) -> bool:
        """Process a single due schedule.
        
        Args:
            schedule: Schedule to process
        
        Returns:
            True if job was enqueued, False otherwise
        """
        # TODO 1: Determine scheduled_for time (from timeline score or schedule.next_run_at)
        # TODO 2: Attempt to enqueue job using _enqueue_scheduled_job()
        # TODO 3: If successful or skipped due to uniqueness:
        #   a. Calculate next run time for recurring jobs
        #   b. Update schedule.next_run_at
        #   c. Update timeline sorted set with new score
        # TODO 4: If one-time job (no cron), remove from storage
        # TODO 5: Return True if job was enqueued
        
        # HINT: For next run calculation, use schedule's last_enqueued_at or now as reference
        # HINT: Use _calculate_next_run() for recurring jobs
    
    def start(self) -> None:
        """Start the scheduler polling loop."""
        # TODO 1: Set _running = True
        # TODO 2: Log scheduler startup with configuration
        # TODO 3: While _running:
        #   a. Record loop start time
        #   b. Call _poll_due_schedules()
        #   c. Calculate sleep time = polling_interval - processing_time
        #   d. Sleep max(0, sleep_time)
        # TODO 4: Log scheduler shutdown
        
        # HINT: Use time.monotonic() for precise timing
        # HINT: Catch KeyboardInterrupt for graceful shutdown
    
    def stop(self) -> None:
        """Stop the scheduler polling loop."""
        # TODO 1: Set _running = False
        # TODO 2: Log scheduler stopping
    
    def calculate_next_runs_for_all_schedules(self) -> int:
        """Recalculate next_run_at for all schedules (e.g., after timezone change).
        
        Returns:
            Number of schedules updated
        """
        # TODO 1: Get all schedule_ids from timeline or scan
        # TODO 2: For each schedule:
        #   a. Fetch schedule
        #   b. Calculate new next_run_at
        #   c. Update timeline score
        # TODO 3: Return count updated
        
        # HINT: Useful for maintenance after DST rules change
```

**E. Language-Specific Hints:**

- **Cron Parsing**: Use `croniter` library: `pip install croniter`. It handles complex cron expressions including ranges, steps, and month/day names.
- **Timezone Handling**: Python 3.9+ has `zoneinfo` in standard library. For older Python, use `backports.zoneinfo` or `pytz`. Avoid `datetime.replace(tzinfo=)` for pytz timezones - use `localize()` instead.
- **UTC Timestamps**: Store all timestamps as UNIX timestamps (float) or ISO 8601 strings ending with "Z" in Redis. Use `datetime.datetime.now(datetime.timezone.utc)` for current UTC time.
- **Redis Sorted Sets**: Use `ZADD`, `ZRANGEBYSCORE`, and `ZREM` for timeline management. Scores should be UNIX timestamps (seconds since epoch).
- **Atomic Operations**: Use Redis pipelines or Lua scripts for operations requiring multiple commands (e.g., check uniqueness, enqueue job, update schedule).
- **ULID Generation**: Use `ulid-py` library for chronological job IDs: `pip install ulid-py`. ULIDs are sortable by creation time unlike UUIDs.

**F. Milestone Checkpoint:**

After implementing the scheduler:

1. **Test Basic Scheduling**:
   ```bash
   # Start Redis
   redis-server
   
   # In Python shell
   python -c "
   from scheduler.scheduler import Scheduler
   from core.job import Job
   from core.queue_manager import QueueManager
   from core.redis_client import RedisClient
   import datetime
   
   redis = RedisClient.get_instance('redis://localhost:6379')
   queue_manager = QueueManager(SystemConfig.from_env())
   scheduler = Scheduler(redis, queue_manager, polling_interval_seconds=10)
   
   # Schedule a job for 30 seconds from now
   job = Job(job_type='test_task', args=['hello'])
   run_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=30)
   schedule_id = scheduler.schedule_delayed_job(job, run_at)
   print(f'Scheduled job with ID: {schedule_id}')
   
   # Start scheduler in background thread
   import threading
   thread = threading.Thread(target=scheduler.start)
   thread.start()
   
   # Wait 45 seconds, then check if job was enqueued
   import time
   time.sleep(45)
   queue_len = queue_manager.get_queue_length('default')
   print(f'Jobs in queue: {queue_len}')  # Should be 1
   
   scheduler.stop()
   thread.join()
   "
   ```

2. **Test Recurring Jobs**:
   ```bash
   # Schedule a job to run every minute
   python -c "
   from scheduler.scheduler import Scheduler
   from core.job import Job
   # ... setup as above
   
   job = Job(job_type='heartbeat', args=[])
   schedule_id = scheduler.schedule_recurring_job(job, '* * * * *', 'UTC')
   print(f'Recurring schedule ID: {schedule_id}')
   
   # Start scheduler, wait 2.5 minutes
   # Should see 2-3 jobs enqueued (depending on exact timing)
   "

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Scheduled jobs never run** | Timeline sorted set empty or wrong scores | Check `ZRANGE schedules:timeline 0 -1 WITHSCORES` in Redis | Ensure `next_run_at` is in UTC when added to sorted set |
| **Jobs run at wrong time** | Timezone conversion error | Compare schedule's `timezone` field with actual calculation | Use `ensure_timezone()` and `to_utc()` consistently |
| **Duplicate jobs enqueued** | Uniqueness constraint not working | Check Redis for `schedule:unique:*` keys with `TTL` | Use `SET NX EX` atomically, not separate `GET` + `SET` |
| **Scheduler uses 100% CPU** | Polling loop with no sleep | Add logging to measure loop iteration time | Ensure `time.sleep()` accounts for processing time |
| **Missed jobs after restart** | Timeline scores in past not processed | Check if `ZRANGEBYSCORE` uses `now() + interval` not just `now()` | Always poll with range `0` to `now + interval` |
| **Cron job runs twice in same minute** | Uniqueness window too short | Check `unique_window_seconds` vs cron frequency | Increase window to cover entire polling interval |

```

---


## Component Design: Monitoring & Dashboard
> **Milestone(s):** Milestone 5: Monitoring & Dashboard. This section provides the complete design for real-time monitoring, web dashboard, and alerting systems that provide observability into job processing, worker health, and system performance. This corresponds to the final milestone, building upon all previous components to create comprehensive operational visibility.

### Mental Model: Air Traffic Control Dashboard

Think of the monitoring system as an **air traffic control dashboard** for your distributed job processing system. Air traffic controllers monitor:

1. **Active Flights (Jobs)**: Where they are (which worker), their status (taking off, cruising, landing), and estimated completion time
2. **Runway Capacity (Queues)**: How many flights are waiting to take off, and which runways are most congested
3. **Weather Conditions (System Health)**: Any storms (errors) approaching, visibility issues (latency), or equipment problems
4. **Historical Patterns**: Which routes have the most turbulence (error-prone jobs), peak traffic hours, and operational bottlenecks

Just as air traffic controllers need real-time visibility without overwhelming detail, our monitoring system must:
- **Aggregate** high-level metrics while allowing drill-down to specific incidents
- **Alert** on anomalies without generating noise
- **Preserve context** so when something goes wrong, you can trace the entire journey of a job
- **Scale** to handle thousands of jobs without impacting the performance of the actual job processing

This mental model emphasizes that monitoring is not just passive observation but active system management—knowing when to re-route traffic (reprioritize queues), allocate more resources (scale workers), or investigate incidents (debug failing jobs).

### Monitoring API and Dashboard Interface

The monitoring system consists of two primary interfaces: a **programmatic API** for automation and integration, and a **web dashboard** for human operators. Both share the same underlying data model but present it differently.

#### Core Monitoring Data Types

| Type Name | Fields | Description |
|-----------|--------|-------------|
| `MetricPoint` | `timestamp datetime`, `name str`, `value float`, `labels Dict[str, str]` | Single timestamped measurement with key-value labels for dimensionality |
| `JobSummary` | `job_id str`, `job_type str`, `queue str`, `status JobStatus`, `enqueued_at datetime`, `started_at Optional[datetime]`, `completed_at Optional[datetime]`, `duration_seconds Optional[float]`, `attempts int`, `error_count int` | Aggregated job execution data for dashboard display |
| `WorkerStatus` | `worker_id str`, `hostname str`, `pid int`, `state str`, `current_job Optional[JobSummary]`, `queues List[str]`, `last_heartbeat datetime`, `processed_count int`, `failed_count int`, `uptime_seconds float` | Current status of a worker process |
| `QueueMetrics` | `queue_name str`, `pending_count int`, `active_count int`, `completed_count int`, `failed_count int`, `retry_scheduled_count int`, `dead_letter_count int`, `enqueue_rate float`, `process_rate float`, `error_rate float`, `avg_process_time float`, `oldest_job_age Optional[float]` | Comprehensive metrics for a single queue |
| `SystemAlert` | `alert_id str`, `severity str` (CRITICAL, WARNING, INFO), `source str`, `message str`, `condition str`, `triggered_at datetime`, `acknowledged_at Optional[datetime]`, `resolved_at Optional[datetime]`, `metadata Dict[str, Any]` | Alert generated by monitoring rules |
| `DashboardConfig` | `refresh_interval int`, `time_range str` (1h, 24h, 7d), `queue_filters List[str]`, `worker_filters List[str]`, `job_type_filters List[str]`, `metric_aggregation str` (avg, sum, p95, p99) | User dashboard configuration |

![Monitoring Data Collection Architecture](./diagrams/monitoring-architecture.svg)

#### Monitoring API Endpoints

The programmatic API provides RESTful endpoints for external monitoring systems (Prometheus, Datadog) and automation scripts:

| Method Name | HTTP Endpoint | Parameters | Returns | Description |
|-------------|---------------|------------|---------|-------------|
| `get_system_metrics()` | `GET /api/metrics/system` | `time_range str = "1h"`, `aggregation str = "avg"` | `Dict[str, List[MetricPoint]]` | System-wide metrics: memory usage, Redis connections, active workers |
| `get_queue_metrics()` | `GET /api/metrics/queues` | `queue_names Optional[List[str]]`, `time_range str = "1h"` | `List[QueueMetrics]` | Current metrics for specified queues (all if none specified) |
| `get_worker_status()` | `GET /api/workers` | `active_only bool = True` | `List[WorkerStatus]` | Status of all workers (including inactive if `active_only=False`) |
| `get_job_history()` | `GET /api/jobs` | `job_id Optional[str]`, `queue Optional[str]`, `status Optional[JobStatus]`, `job_type Optional[str]`, `limit int = 100`, `offset int = 0`, `time_range Optional[str]` | `Dict[str, Any]` (with `jobs List[JobSummary]`, `total int`) | Search jobs with filtering and pagination |
| `get_dead_letter_jobs()` | `GET /api/dead-letter` | `queue Optional[str]`, `job_type Optional[str]`, `limit int = 50` | `List[Job]` | Jobs in dead letter queue for manual intervention |
| `retry_dead_letter_job()` | `POST /api/dead-letter/{job_id}/retry` | `job_id str` (path), `queue Optional[str]` | `Dict[str, Any]` (with `success bool`, `new_job_id Optional[str]`) | Retry a specific dead letter job in its original queue |
| `delete_dead_letter_job()` | `DELETE /api/dead-letter/{job_id}` | `job_id str` (path) | `Dict[str, Any]` (with `success bool`) | Permanently remove a job from dead letter queue |
| `get_alerts()` | `GET /api/alerts` | `severity Optional[str]`, `acknowledged bool = False`, `limit int = 50` | `List[SystemAlert]` | Get current active alerts |
| `acknowledge_alert()` | `POST /api/alerts/{alert_id}/acknowledge` | `alert_id str` (path) | `Dict[str, Any]` (with `success bool`) | Mark an alert as acknowledged by an operator |
| `create_alert_rule()` | `POST /api/alert-rules` | `rule_config Dict[str, Any]` | `Dict[str, Any]` (with `rule_id str`) | Create a new alerting rule (admin only) |
| `get_metrics_stream()` | `GET /api/metrics/stream` | - | `StreamingResponse` (Server-Sent Events) | Real-time stream of metric updates for dashboard |

#### Web Dashboard Interface

The web dashboard provides human-readable visualization through these key views:

| View Name | Purpose | Key Visualizations |
|-----------|---------|-------------------|
| **System Overview** | High-level system health | - Queue depth gauges<br>- Worker status grid<br>- Error rate sparklines<br>- System metrics time-series |
| **Queue Detail** | Drill-down into specific queue | - Job status distribution pie chart<br>- Processing rate over time<br>- Top error-producing job types<br>- Oldest pending jobs list |
| **Worker Dashboard** | Worker process monitoring | - Worker state heatmap<br>- CPU/memory usage per worker<br>- Current job execution time<br>- Worker failure history |
| **Job Explorer** | Search and inspect jobs | - Filterable job table with pagination<br>- Job execution timeline<br>- Error stack trace viewer<br>- Manual retry controls |
| **Dead Letter Queue** | Manage permanently failed jobs | - DLQ job table with error details<br>- Bulk retry/delete operations<br>- Failure pattern analysis |
| **Alert Center** | Alert management | - Active alert dashboard<br>- Alert history with filters<br>- Alert rule configuration |
| **System Configuration** | Runtime configuration view | - Queue configuration table<br>- Worker pool settings<br>- Redis connection status |

> **Design Insight:** The dashboard prioritizes **operational clarity** over completeness. Each view answers specific questions operators have: "Why is my queue backing up?", "Which worker is stuck?", "What's failing and why?" The API supports automation while the dashboard supports human intuition.

### Metrics Collection and Aggregation

Metrics flow through a three-stage pipeline: **collection → aggregation → storage**. This design ensures monitoring doesn't block job processing while providing accurate, real-time visibility.

#### Collection Strategy

Metrics are collected at these key points in the system:

| Collection Point | What's Collected | How It's Collected |
|------------------|------------------|-------------------|
| **Job Enqueue** | Queue length increment, job type counter, payload size | `QueueManager.enqueue_job()` emits Redis Stream event |
| **Job Start** | Job transition to ACTIVE, worker assignment, queue length decrement | `Worker` emits event when beginning job execution |
| **Job Completion** | Success/failure status, execution duration, attempt count | `Worker` wraps job execution with timing and status tracking |
| **Job Failure** | Error details, retry scheduling, backoff delay | `RetryManager.handle_job_failure()` records error context |
| **Worker Heartbeat** | Worker liveness, current job, processed counts | `WorkerHeartbeat` periodically updates Redis hash |
| **Scheduler Activity** | Scheduled job enqueue, missed schedules, cron evaluation | `Scheduler._enqueue_scheduled_job()` emits events |
| **System Resources** | Redis memory usage, connection count, CPU via OS | External agent collects via Redis INFO and psutil |

#### Aggregation Algorithm

Raw events are aggregated into 10-second buckets to reduce storage and query load:

1. **Event Emission**: Components emit events to Redis Stream `monitoring:events` with JSON payload:
   ```json
   {
     "timestamp": "2024-01-15T10:30:45.123Z",
     "type": "job_completed",
     "queue": "emails",
     "job_type": "send_welcome",
     "worker_id": "worker-abc123",
     "duration_ms": 125.5,
     "success": true
   }
   ```

2. **Background Aggregator**: A lightweight aggregator process runs every 10 seconds:
   ```
   1. Reads all events from `monitoring:events` stream since last run
   2. Groups events by (metric_name, queue, job_type, worker_id) tuple
   3. Calculates aggregates for each group:
      - Count → sum of occurrences
      - Duration → average, p95, p99, min, max
      - Error rate → failed_count / total_count
   4. Stores aggregated metrics to Redis Sorted Set with timestamp score
   5. Truncates old raw events beyond retention window
   ```

3. **Metric Retention**:
   - Raw events: 1 hour retention (for debugging)
   - 10-second aggregates: 7 days retention
   - 1-minute aggregates (rolled up from 10-second): 30 days retention
   - 1-hour aggregates: 90 days retention

#### Key Performance Indicators (KPIs)

The system tracks these essential KPIs for operational health:

| KPI Name | Calculation | Alert Threshold | Purpose |
|----------|-------------|-----------------|---------|
| **Queue Depth** | `LLEN queue:{name}` | > 1000 pending jobs | Detect processing backlog |
| **Processing Rate** | Jobs completed per second (10s window) | < 10% of expected rate | Detect worker slowdown |
| **Error Rate** | Failed jobs / total jobs (5m window) | > 5% | Detect systemic failures |
| **Worker Health** | Workers with heartbeat < 60s ago / total workers | < 80% healthy | Detect worker process failures |
| **Job Duration P99** | 99th percentile job duration (by job_type) | > 2x historical baseline | Detect performance degradation |
| **Redis Memory Usage** | `used_memory` / `maxmemory` | > 85% | Prevent Redis OOM |
| **Schedule Miss Rate** | Missed schedules / total schedules (1h window) | > 1% | Detect scheduler issues |

> **Design Insight:** Monitoring should be **cheap enough to always be on**. By aggregating metrics and using Redis' efficient data structures, we avoid the common pitfall where monitoring itself becomes a performance bottleneck. The 10-second aggregation strikes a balance between real-time responsiveness and system load.

### ADR: Real-time Update Strategy

> **Decision: Server-Sent Events (SSE) over WebSockets for Real-time Dashboard Updates**
> 
> - **Context**: The dashboard needs near real-time updates (queue depths, worker status, job completion) without requiring page refreshes. We need a lightweight, scalable solution that integrates well with our existing HTTP API and doesn't add significant complexity.
> 
> - **Options Considered**:
>   1. **Polling (HTTP GET every few seconds)**: Simple but inefficient, creates constant load regardless of change frequency
>   2. **WebSockets**: Full-duplex, low-latency, but requires connection management and protocol handling
>   3. **Server-Sent Events (SSE)**: HTTP-based, unidirectional from server to client, automatic reconnection, simple API
>   4. **Redis Pub/Sub with bridge**: Push updates via Redis publish, bridge to HTTP layer
> 
> - **Decision**: Use **Server-Sent Events (SSE)** for real-time dashboard updates.
> 
> - **Rationale**:
>   - **Simplicity**: SSE works over standard HTTP, requires no special protocol handling on server or client
>   - **Automatic reconnection**: Built-in reconnection with last-event-id tracking
>   - **Efficient**: Server pushes updates only when metrics change, no constant polling
>   - **Compatibility**: Works with existing authentication/authorization middleware
>   - **Resource friendly**: Single HTTP connection per dashboard client vs. bidirectional WebSocket overhead
>   - **Graceful degradation**: If SSE connection drops, dashboard falls back to periodic polling
> 
> - **Consequences**:
>   - ✅ Dashboard gets sub-second updates without constant polling
>   - ✅ Simple implementation using standard HTTP libraries
>   - ✅ Automatic reconnection handles network issues gracefully
>   - ✅ Works behind load balancers and proxies with standard HTTP
>   - ❌ Unidirectional only (server → client), but dashboard doesn't need client→server updates
>   - ❌ Browser limits of 6 concurrent HTTP connections per domain (mitigated by using single SSE stream for all metrics)

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Polling** | - Extremely simple<br>- Works everywhere<br>- No connection state | - High latency (poll interval)<br>- Constant load even when idle<br>- Wasted bandwidth | Creates unnecessary load; poor real-time experience |
| **WebSockets** | - Full duplex<br>- Lowest latency<br>- Rich protocol options | - Complex connection management<br>- Requires protocol upgrade<br>- Overkill for unidirectional updates | Complexity outweighs benefits for our use case |
| **Server-Sent Events** | - HTTP-based, simple<br>- Automatic reconnection<br>- Efficient push model | - Unidirectional only<br>- Less control than WebSockets | **CHOSEN**: Perfect fit for dashboard metrics streaming |
| **Redis Pub/Sub Bridge** | - Leverages existing Redis<br>- Decouples producers from consumers | - Requires bridge process<br>- Additional moving parts | Adds complexity without significant advantage |

### ADR: Job History Storage Strategy

> **Decision: Two-tiered storage with Redis for recent history and optional archival to external storage**
> 
> - **Context**: Job execution history is valuable for debugging and analytics but can grow unbounded. We need to store enough history for operational debugging (hours/days) while providing an optional path for longer-term retention without overwhelming Redis memory.
> 
> - **Options Considered**:
>   1. **Redis-only with TTL**: Store all job history in Redis with expiration (e.g., 7 days TTL)
>   2. **External database primary**: Write all job history directly to PostgreSQL/MySQL
>   3. **Two-tiered with Redis cache**: Recent history in Redis (fast access), older history in external DB
>   4. **Hybrid with configurable backend**: Plugable storage backends (Redis, SQL, filesystem)
> 
> - **Decision**: Implement **two-tiered storage** with Redis for recent history (configurable retention) and optional archival to external database for long-term retention.
> 
> - **Rationale**:
>   - **Performance**: Redis provides sub-millisecond access to recent job history for dashboard
>   - **Operational simplicity**: Most debugging needs recent history (last few hours/days)
>   - **Memory control**: TTL prevents unbounded Redis growth
>   - **Flexibility**: Optional external archival for compliance or analytics without impacting core system
>   - **Progressive enhancement**: Start with Redis-only, add archival later if needed (YAGNI principle)
> 
> - **Consequences**:
>   - ✅ Fast access to recent job history for dashboard and debugging
>   - ✅ Configurable Redis memory usage via TTL settings
>   - ✅ Optional external archival without core system dependency
>   - ✅ Simple initial implementation (Redis only)
>   - ❌ Historical queries beyond TTL require external archival setup
>   - ❌ Two data stores to manage if archival enabled

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Redis-only with TTL** | - Maximum performance<br>- Simple architecture<br>- No external dependencies | - History limited by TTL<br>- Redis memory pressure<br>- No long-term analytics | **CHOSEN** as base: Simple and fast for operational needs |
| **External DB primary** | - Unlimited history<br>- Rich query capabilities<br>- Persistent storage | - Slower access<br>- Additional dependency<br>- Operational complexity | Overkill for core operational dashboard needs |
| **Two-tiered with cache** | - Best of both worlds<br>- Fast recent access + long-term storage | - Complexity of synchronization<br>- Cache invalidation challenges | **CHOSEN** as extensible: Can add archival later |
| **Plugable backends** | - Maximum flexibility<br>- Choose storage per deployment | - Highest complexity<br>- Interface abstraction overhead | Premature optimization; start simple, extend if needed |

#### Job History Storage Implementation

```plaintext
Redis Data Structures for Job History:

1. `job:history:{job_id}` → Hash with complete job execution record
   - Fields: job_data (JSON), status, started_at, completed_at, duration_ms, attempts, errors (JSON)
   - TTL: 7 days (configurable)

2. `job:history:index:by_queue:{queue_name}` → Sorted Set
   - Score: completed_at timestamp (or enqueued_at if not completed)
   - Member: job_id
   - Used for: "Show recent jobs in queue X"
   - TTL: 7 days (via periodic cleanup of expired jobs)

3. `job:history:index:by_type:{job_type}` → Sorted Set
   - Score: completed_at timestamp
   - Member: job_id
   - Used for: "Show recent jobs of type Y"
   - TTL: 7 days

4. `job:history:index:by_status:{status}` → Sorted Set
   - Score: timestamp of status change
   - Member: job_id
   - Used for: "Show recent failed jobs"
   - TTL: 7 days for COMPLETED, 30 days for FAILED/DEAD_LETTER

Archival Process (optional):
1. Background process scans jobs nearing TTL expiration
2. Serializes job history to JSON
3. Pushes to configurable sink (PostgreSQL, S3, Elasticsearch)
4. Removes from Redis after successful archival
```

### Common Pitfalls in Monitoring Implementation

⚠️ **Pitfall 1: Monitoring blocks job processing**
- **What happens**: Adding metric collection directly in the hot path of job execution (e.g., synchronous writes to Redis during job start/complete)
- **Why it's wrong**: Adds latency to job processing; under high load, monitoring can become the bottleneck
- **How to fix**: Use asynchronous event emission (Redis Streams) or batch writes. Workers should emit events to a local buffer that's flushed periodically.

⚠️ **Pitfall 2: Storing unlimited job history in Redis**
- **What happens**: Every job execution creates permanent records in Redis, causing memory to grow without bound
- **Why it's wrong**: Eventually Redis hits memory limits, causing eviction or crashes; performance degrades as datasets grow
- **How to fix**: Implement TTL on all job history keys (7-30 days). For longer retention, add optional archival to external storage.

⚠️ **Pitfall 3: Dashboard queries blocking worker operations**
- **What happens**: Complex dashboard queries (e.g., "show all jobs from last month") run expensive Redis operations that block worker access
- **Why it's wrong**: Monitoring should be observational only; it should never impact system performance
- **How to fix**: Use read replicas for dashboard queries, implement query timeouts, and denormalize data for common queries.

⚠️ **Pitfall 4: Alert fatigue from noisy thresholds**
- **What happens**: Setting alert thresholds too sensitively (e.g., alert on every single job failure) creates alert spam that operators ignore
- **Why it's wrong**: Important alerts get lost in the noise; operators develop "alert blindness"
- **How to fix**: Use rate-based alerting (error rate > 5% over 5 minutes), implement alert deduplication, and tier alerts by severity.

⚠️ **Pitfall 5: Exposing sensitive job data in dashboard**
- **What happens**: Job arguments containing PII, passwords, or tokens are displayed in dashboard without sanitization
- **Why it's wrong**: Security breach; sensitive data exposed to anyone with dashboard access
- **How to fix**: Implement job argument sanitization filters, mask sensitive fields (e.g., `password: ***`), and control dashboard access with authentication.

⚠️ **Pitfall 6: Hardcoded metric collection intervals**
- **What happens**: Metric collection runs at fixed intervals (e.g., every 10 seconds) regardless of system load
- **Why it's wrong**: Under low load, this wastes resources; under high load, it may not capture enough detail
- **How to fix**: Implement adaptive sampling: increase frequency during high activity, decrease during quiet periods.

### Implementation Guidance for Monitoring

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Web Framework** | Flask + Flask-SSE | FastAPI with async SSE support |
| **Frontend Dashboard** | Vanilla JavaScript + Chart.js | React/Vue.js with D3.js for custom visualizations |
| **Real-time Updates** | Server-Sent Events (SSE) | WebSockets with Socket.IO fallback |
| **Metrics Storage** | Redis TimeSeries module | Redis + PostgreSQL for long-term archival |
| **Alerting Engine** | Custom rule evaluator | Integration with Prometheus Alertmanager |
| **Charting Library** | Chart.js (simple, good defaults) | Apache ECharts (more customizable) |

#### B. Recommended File/Module Structure

```
background-job-processor/
├── src/
│   ├── job_processor/
│   │   ├── __init__.py
│   │   ├── core/                   # Milestone 1-2
│   │   │   ├── job.py
│   │   │   ├── queue_manager.py
│   │   │   └── worker.py
│   │   ├── retry/                  # Milestone 3
│   │   │   ├── retry_manager.py
│   │   │   └── backoff_calculator.py
│   │   ├── scheduler/              # Milestone 4
│   │   │   ├── scheduler.py
│   │   │   └── cron_parser.py
│   │   ├── monitoring/             # Milestone 5 (THIS SECTION)
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py          # Metric collection and aggregation
│   │   │   ├── events.py           # Event emission and handling
│   │   │   ├── dashboard_api.py    # REST API endpoints
│   │   │   ├── dashboard_ui.py     # Web dashboard (HTML/JS)
│   │   │   ├── alerting.py         # Alert rule evaluation
│   │   │   ├── history_store.py    # Job history storage
│   │   │   └── sse_stream.py       # Server-Sent Events streaming
│   │   └── utils/
│   │       ├── redis_client.py
│   │       └── config.py
├── tests/
│   └── monitoring/
│       ├── test_metrics.py
│       ├── test_dashboard_api.py
│       └── test_alerting.py
├── static/                         # Dashboard static assets
│   ├── css/
│   ├── js/
│   └── charts/
├── templates/                      # Dashboard HTML templates
│   └── dashboard.html
└── requirements.txt
```

#### C. Infrastructure Starter Code

**Complete metrics aggregation service (`monitoring/aggregator.py`):**

```python
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import redis

class MetricsAggregator:
    """Background service that aggregates raw monitoring events into metrics."""
    
    def __init__(self, redis_client: redis.Redis, interval_seconds: int = 10):
        self.redis = redis_client
        self.interval = interval_seconds
        self._running = False
        self._thread = None
        
    def start(self):
        """Start the aggregation thread."""
        self._running = True
        self._thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop the aggregation thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
            
    def _aggregation_loop(self):
        """Main aggregation loop."""
        while self._running:
            try:
                self._run_aggregation()
            except Exception as e:
                # Log but continue - monitoring should not crash the system
                print(f"Aggregation error: {e}")
            time.sleep(self.interval)
            
    def _run_aggregation(self):
        """Execute a single aggregation cycle."""
        # Get all events since last run
        last_run_key = "monitoring:aggregator:last_run"
        last_run = self.redis.get(last_run_key)
        start_id = last_run.decode() if last_run else "0-0"
        
        # Read events from stream
        stream_key = "monitoring:events"
        events = self.redis.xrange(stream_key, min=start_id, max="+", count=1000)
        
        if not events:
            return
            
        # Process events by type and labels
        metrics_by_key = defaultdict(list)
        for event_id, event_data in events:
            event_type = event_data.get(b"type", b"unknown").decode()
            labels_json = event_data.get(b"labels", b"{}")
            labels = json.loads(labels_json.decode())
            value = float(event_data.get(b"value", b"0"))
            timestamp = event_data.get(b"timestamp", b"").decode()
            
            # Create metric key from labels
            label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
            metric_key = f"{event_type}:{':'.join(label_parts)}"
            metrics_by_key[metric_key].append((timestamp, value))
            
        # Calculate aggregates
        bucket_time = datetime.utcnow().replace(second=0, microsecond=0)
        bucket_timestamp = int(bucket_time.timestamp())
        
        for metric_key, values in metrics_by_key.items():
            if not values:
                continue
                
            # Calculate aggregates
            values_only = [v for _, v in values]
            count = len(values_only)
            avg_val = sum(values_only) / count if count > 0 else 0
            sorted_vals = sorted(values_only)
            p95 = sorted_vals[int(count * 0.95)] if count > 0 else 0
            
            # Store aggregated metric
            metric_hash = {
                "count": str(count),
                "avg": str(avg_val),
                "p95": str(p95),
                "min": str(min(values_only)),
                "max": str(max(values_only))
            }
            
            agg_key = f"monitoring:metrics:{metric_key}:{bucket_timestamp}"
            self.redis.hmset(agg_key, metric_hash)
            self.redis.expire(agg_key, 604800)  # 7 days TTL
            
        # Update last run pointer
        last_event_id = events[-1][0] if events else start_id
        self.redis.set(last_run_key, last_event_id)
        
        # Clean up old raw events (keep last hour only)
        one_hour_ago = int((datetime.utcnow() - timedelta(hours=1)).timestamp() * 1000)
        self.redis.xtrim(stream_key, minid=one_hour_ago)
```

#### D. Core Logic Skeleton Code

**Dashboard API endpoint for real-time metrics stream (`monitoring/dashboard_api.py`):**

```python
import json
import time
from flask import Flask, Response, request
import redis

app = Flask(__name__)

@app.route('/api/metrics/stream')
def metrics_stream():
    """Server-Sent Events stream for real-time dashboard updates."""
    
    def generate_events():
        # TODO 1: Get Redis client from application context
        redis_client = None  # Implement connection pooling
        
        # TODO 2: Get last event ID from request header for reconnection
        last_id = request.headers.get('Last-Event-ID', '0-0')
        
        # TODO 3: Subscribe to monitoring events stream
        stream_key = 'monitoring:events'
        
        while True:
            try:
                # TODO 4: Read new events since last_id (blocking with timeout)
                # Use redis.xread with count=10, block=5000ms
                events = []  # Implement stream reading
                
                for stream, event_list in events:
                    for event_id, event_data in event_list:
                        # TODO 5: Filter events based on dashboard filters
                        # Get filters from request query params
                        queue_filter = request.args.get('queue')
                        job_type_filter = request.args.get('job_type')
                        
                        # TODO 6: Format event for SSE (event: type, data: JSON)
                        event_type = event_data.get(b'type', b'metric').decode()
                        event_data_json = json.dumps({
                            'id': event_id,
                            'type': event_type,
                            'data': {k.decode(): v.decode() for k, v in event_data.items()},
                            'timestamp': time.time()
                        })
                        
                        # TODO 7: Yield SSE formatted data
                        yield f"event: {event_type}\ndata: {event_data_json}\n\n"
                        
                        # Update last_id for reconnection
                        last_id = event_id
                
                # TODO 8: Send heartbeat comment every 30 seconds to keep connection alive
                # yield ": heartbeat\n\n"
                
            except Exception as e:
                # TODO 9: Log error and yield error event
                error_data = json.dumps({'error': str(e)})
                yield f"event: error\ndata: {error_data}\n\n"
                break
    
    # TODO 10: Return SSE response with appropriate headers
    return Response(
        generate_events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable nginx buffering
        }
    )

# TODO 11: Implement other API endpoints from the API table
# - GET /api/metrics/system
# - GET /api/metrics/queues
# - GET /api/workers
# - GET /api/jobs
# - GET /api/dead-letter
# - POST /api/dead-letter/{job_id}/retry
# - DELETE /api/dead-letter/{job_id}
# - GET /api/alerts
# - POST /api/alerts/{alert_id}/acknowledge
# - POST /api/alert-rules
```

**Alert rule evaluator (`monitoring/alerting.py`):**

```python
from typing import Dict, List, Any, Optional
from datetime import datetime
import redis

class AlertRule:
    """Represents an alert rule with condition and thresholds."""
    
    def __init__(self, rule_id: str, config: Dict[str, Any]):
        self.rule_id = rule_id
        self.name = config.get('name', 'Unnamed Rule')
        self.severity = config.get('severity', 'WARNING')  # CRITICAL, WARNING, INFO
        self.condition = config.get('condition', '')  # e.g., "error_rate > 0.05"
        self.window_seconds = config.get('window_seconds', 300)  # 5 minutes
        self.cooldown_seconds = config.get('cooldown_seconds', 300)  # 5 minutes
        self.enabled = config.get('enabled', True)
        self.labels = config.get('labels', {})  # e.g., {"queue": "emails", "job_type": "send_welcome"}
        
    def evaluate(self, metrics_store) -> bool:
        """Evaluate rule against current metrics."""
        # TODO 1: Parse condition string into evaluable expression
        # Example: "error_rate > 0.05 AND queue_depth > 1000"
        
        # TODO 2: Fetch relevant metrics for the time window
        # Use metrics_store.get_metrics(window_seconds=self.window_seconds, labels=self.labels)
        
        # TODO 3: Calculate aggregated values from metrics
        # For rate-based conditions, calculate rate over window
        
        # TODO 4: Evaluate condition against calculated values
        # Use safe_eval or ast parsing for condition evaluation
        
        # TODO 5: Return True if condition is met
        return False
        
    def should_alert(self, last_alert_time: Optional[datetime]) -> bool:
        """Check if alert should fire considering cooldown period."""
        if not self.enabled:
            return False
            
        if last_alert_time is None:
            return True
            
        # Check cooldown period
        seconds_since_last = (datetime.utcnow() - last_alert_time).total_seconds()
        return seconds_since_last >= self.cooldown_seconds

class AlertManager:
    """Manages alert rules and firing alerts."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.rules: Dict[str, AlertRule] = {}
        self.rule_key_prefix = "monitoring:alert_rules:"
        
    def load_rules(self):
        """Load all alert rules from Redis."""
        # TODO 1: Scan for all rule keys with prefix
        # Use redis.scan_iter(self.rule_key_prefix + "*")
        
        # TODO 2: For each key, load JSON config and create AlertRule
        
        # TODO 3: Store in self.rules dictionary
        
    def evaluate_all_rules(self):
        """Evaluate all enabled alert rules."""
        # TODO 1: Get last alert times for each rule from Redis
        
        # TODO 2: For each rule, evaluate condition
        
        # TODO 3: If condition met and cooldown passed, fire alert
        
        # TODO 4: Create SystemAlert object and store to Redis
        
        # TODO 5: Update last alert time for rule
        
    def fire_alert(self, rule: AlertRule, condition_value: float):
        """Fire an alert and store it."""
        # TODO 1: Generate unique alert_id (ULID)
        
        # TODO 2: Create SystemAlert object with all fields
        
        # TODO 3: Store to Redis Sorted Set (by timestamp) and Hash (full data)
        
        # TODO 4: Optionally send notification (email, webhook, etc.)
        
        # TODO 5: Return alert_id
        return "alert_123"
```

#### E. Language-Specific Hints

**Python Implementation Tips:**

1. **Use Flask for simple dashboard**: Flask + Flask-SSE provides a straightforward SSE implementation. For production, use `gevent` or `gunicorn` with async workers.

2. **Redis connection management**: Use connection pooling for dashboard queries to avoid overwhelming Redis with connections:
   ```python
   import redis
   from redis.connection import ConnectionPool
   
   pool = ConnectionPool.from_url('redis://localhost:6379', max_connections=20)
   redis_client = redis.Redis(connection_pool=pool)
   ```

3. **SSE implementation**: Flask doesn't natively support SSE well; use a queue-based approach:
   ```python
   from flask import Response
   import queue
   import json
   
   class MessageAnnouncer:
       def __init__(self):
           self.listeners = []
       
       def listen(self):
           q = queue.Queue(maxsize=5)
           self.listeners.append(q)
           return q
       
       def announce(self, msg):
           for i in reversed(range(len(self.listeners))):
               try:
                   self.listeners[i].put_nowait(msg)
               except queue.Full:
                   del self.listeners[i]
   
   announcer = MessageAnnouncer()
   ```

4. **Async metrics collection**: Use background threads for aggregation, not async/await, since Redis operations are blocking.

5. **Job argument sanitization**: Before displaying job arguments in dashboard, filter sensitive fields:
   ```python
   SENSITIVE_FIELDS = {'password', 'token', 'secret', 'key', 'authorization'}
   
   def sanitize_job_args(args_dict):
       sanitized = {}
       for key, value in args_dict.items():
           if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
               sanitized[key] = '***REDACTED***'
           else:
               sanitized[key] = value
       return sanitized
   ```

#### F. Milestone Checkpoint

**After implementing Monitoring & Dashboard, verify:**

1. **Dashboard loads**: Start the web server and navigate to `http://localhost:8080/dashboard`. You should see:
   - System overview with queue depths
   - Worker status table
   - Real-time metrics updating every few seconds

2. **API endpoints work**: Test API with curl:
   ```bash
   # Get queue metrics
   curl http://localhost:8080/api/metrics/queues
   
   # Get worker status
   curl http://localhost:8080/api/workers
   
   # Stream real-time events (SSE)
   curl -N http://localhost:8080/api/metrics/stream
   ```

3. **Metrics collection**: Enqueue a few jobs and verify they appear in:
   - Dashboard queue depth indicators
   - Job history table (after completion)
   - Real-time event stream (if connected via SSE)

4. **Alerting test**: Set up a simple alert rule (error rate > 0%) and trigger a job failure:
   ```python
   # Create alert rule via API
   import requests
   rule = {
       "name": "Test Error Alert",
       "severity": "WARNING",
       "condition": "error_rate > 0",
       "window_seconds": 60,
       "labels": {"queue": "default"}
   }
   requests.post("http://localhost:8080/api/alert-rules", json=rule)
   ```
   
   Check that alerts appear in `/api/alerts` endpoint after triggering a job failure.

**Signs something is wrong:**
- Dashboard shows "No data" or "Connection lost" → Check Redis connection and event emission
- Real-time updates not working → Check SSE endpoint and browser console for errors
- High Redis memory usage → Verify TTL is set on history keys and aggregation is running
- Dashboard loads slowly → Check for expensive Redis queries (use `SLOWLOG` command)

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Dashboard shows "Disconnected"** | SSE connection failing | Check browser console for network errors; verify `/api/metrics/stream` endpoint returns proper SSE headers | Ensure response has `text/event-stream` MIME type and proper headers |
| **Real-time updates delayed by minutes** | Event aggregation not running | Check if `MetricsAggregator` process is running; look for events in `monitoring:events` stream | Start aggregator service; check for exceptions in aggregator logs |
| **Redis memory growing rapidly** | Job history TTL not set | Check Redis keys with `keys job:history:*`; verify TTL with `ttl keyname` | Ensure all history keys have TTL; implement periodic cleanup job |
| **Alert rules not firing** | Rule evaluation failing | Check alert manager logs; test rule condition manually with current metrics | Debug rule evaluation logic; verify metrics exist for rule labels |
| **Dashboard queries timing out** | Expensive Redis operations | Use `redis-cli --stat` to see operations; check `SLOWLOG GET 10` | Add query limits; implement caching for expensive queries; use read replica |
| **Job arguments visible in dashboard** | Sanitization not applied | Check dashboard display of job details; look for sensitive fields | Implement argument sanitization in Job.to_dict() or dashboard serialization |

---


## Interactions and Data Flow

> **Milestone(s):** This section integrates concepts and components from all five milestones, showing how they work together to handle key operational scenarios: normal job flow, error recovery, and scheduled job execution. Understanding these interactions is critical for implementing a cohesive system.

This section details the end-to-end behavior of the Background Job Processor through three critical scenarios: the successful processing of a job from submission to completion, the recovery flow when a worker crashes mid-execution, and the lifecycle of a scheduled recurring job. These scenarios demonstrate how the system's components—`QueueManager`, `Worker`, `RetryManager`, `Scheduler`, and monitoring subsystems—orchestrate data flow and state transitions to deliver reliable asynchronous processing.

### Happy Path: Job Submission to Completion

**Mental Model: Assembly Line in a Factory**
Imagine a manufacturing plant. An order (job) is submitted to the front office (producer). The office validates the order, stamps it with a unique ID, and places it on the correct conveyor belt (queue) based on its type. A worker on the factory floor picks up the order, performs the required assembly steps (handler execution), marks it as completed, and places the finished product in the shipping area (result storage). The entire process is tracked in a central log (monitoring system), allowing managers to see the order's progress in real-time.

This scenario begins with a producer submitting a job and ends with the job's successful completion, involving all core components in their normal operational states.

#### Step-by-Step Flow

The following numbered sequence details the journey of a single job, referencing the ![System Component Overview](./diagrams/system-component.svg) and ![Happy Path Job Execution Sequence](./diagrams/happy-path-sequence.svg) diagrams.

1.  **Job Creation and Submission:** The producer application creates a `Job` instance, populating its required fields (`job_type`, `args`, `queue`) and optional metadata. It then calls `QueueManager.enqueue_job(job)`.

2.  **Validation and Serialization:** Inside `QueueManager.enqueue_job`, the following occurs:
    *   `QueueManager._validate_job(job)` is called. It checks that the payload size is under `SystemConfig.max_payload_size`, that the specified queue exists in `QueueManager.queue_configs`, and that the job's `job_type` is a non-empty string. If validation fails, a `ValidationError` is raised to the producer.
    *   The job's `status` is set to `PENDING`, `job_id` is assigned a ULID, and `created_at` is set to the current UTC timestamp.
    *   `Job.serialize()` is invoked, converting the job's dictionary representation (from `Job.to_dict()`) into a JSON string.

3.  **Atomic Enqueue to Redis:** The `QueueManager` uses a Redis pipeline (via `RedisClient.pipeline()`) to execute the following commands atomically:
    *   `HSET` to store the serialized job under a key like `job:{job_id}`. This creates a persistent record.
    *   `EXPIRE` is set on the job key using a TTL (e.g., 7 days) to prevent indefinite memory bloat.
    *   `LPUSH` to add the `job_id` to the appropriate Redis list key, e.g., `queue:{queue_name}`. This action makes the job visible to workers.
    *   `INCR` on a metric key like `stats:queue:{queue_name}:enqueued` for monitoring.
    *   If the queue has a `QueueConfig.max_length`, the pipeline also checks `LLEN` and may raise a `QueueFullError` before the `LPUSH`.
    *   The pipeline is executed. Upon success, `enqueue_job` returns the generated `job_id` to the producer.

4.  **Worker Polling and Dequeue:** A `Worker` process, configured to listen on the relevant queue, is in its main loop. It uses the priority-weighted polling algorithm to select a queue and then calls `BRPOPLPUSH` (or a similar reliable dequeue pattern) with a timeout. This command atomically moves the `job_id` from the main `queue:{queue_name}` list to a temporary `queue:{queue_name}:processing` list, providing at-least-once delivery semantics. The worker receives the `job_id`.

5.  **Job Materialization and Dispatch:** The worker uses `QueueManager.get_job_by_id(job_id)` to fetch the full job data from the `job:{job_id}` hash. It then calls `Job.deserialize()` to recreate the `Job` object in memory. The worker updates the job's `status` to `ACTIVE` and `started_at` to the current time, persisting this change back to Redis. It then looks up the job's `job_type` in its internal `Worker.handlers` registry to find the appropriate handler function.

6.  **Job Execution:** The worker invokes the registered handler function, passing the job's `args` and `kwargs`. The handler executes the business logic (e.g., sending an email, processing an image). The worker's `WorkerHeartbeat` thread continues to periodically update a key like `worker:heartbeat:{worker_id}` in Redis, indicating liveness.

7.  **Success Handling and Cleanup:** Upon successful handler completion:
    *   The worker updates the job's `status` to `COMPLETED`, sets `completed_at`, and optionally stores the handler's return value in `result`.
    *   The job's updated state is persisted to the `job:{job_id}` hash in Redis.
    *   The worker removes the `job_id` from the `queue:{queue_name}:processing` list (e.g., via `LREM`), completing the reliable dequeue cycle.
    *   Metric keys are incremented (e.g., `stats:queue:{queue_name}:completed`).

8.  **Monitoring Update:** The `MetricsAggregator`, running periodically, observes the changes in job state and queue lengths. It updates aggregated metrics (like `QueueMetrics.process_rate`). If the dashboard is open, it may receive these updates via Server-Sent Events (SSE), refreshing the display to show the job's completion and updated queue statistics.

The following table summarizes the key Redis commands and state changes during this flow:

| Step | Primary Redis Command(s) | Job State Transition | Key Data Structures Updated |
| :--- | :--- | :--- | :--- |
| 1-3 | `HSET job:{id} ...`, `EXPIRE`, `LPUSH queue:{name} {id}` | None → `PENDING` | `job:{id}` (Hash), `queue:{name}` (List) |
| 4 | `BRPOPLPUSH queue:{name} queue:{name}:processing` | `PENDING` → (Implicitly Active) | `queue:{name}` (List), `queue:{name}:processing` (List) |
| 5-6 | `HGET job:{id}`, `HSET job:{id} status ACTIVE` | `PENDING` → `ACTIVE` | `job:{id}` (Hash), `worker:heartbeat:{id}` (String) |
| 7 | `HSET job:{id} status COMPLETED ...`, `LREM queue:{name}:processing {id}` | `ACTIVE` → `COMPLETED` | `job:{id}` (Hash), `queue:{name}:processing` (List) |

### Error Recovery Flow: Worker Crash and Retry

**Mental Model: Paramedic Response and Hospital Triage**
When a factory worker (the `Worker` process) suddenly faints (crashes), the system's safety protocols activate. First, a supervisor (the heartbeat monitor) notices the worker's missing check-ins and marks them as inactive. Another worker then safely retrieves the half-assembled product (the job) from the worker's station (the processing queue) and assesses the damage. If the product can be fixed, it's sent to a repair station (the retry queue) with a note indicating the problem and a scheduled repair time. If it's beyond repair after several attempts, it's sent to the quality control failure analysis bin (the dead letter queue) for engineering review.

This scenario highlights the system's fault tolerance, demonstrating how jobs are not lost when a `Worker` fails unexpectedly and how the retry mechanism with exponential backoff manages transient failures.

#### Step-by-Step Flow

The sequence, also illustrated in ![Retry with Exponential Backoff Flow](./diagrams/retry-flow-sequence.svg), begins after step 6 of the Happy Path, assuming the worker crashes during job execution.

1.  **Worker Crash:** During the execution of a handler (step 6), the `Worker` process suffers a fatal error (e.g., a segmentation fault, `SIGKILL`, or host failure) and terminates immediately. It does not reach the success handling logic (step 7).

2.  **Stale Heartbeat Detection:** The worker's `WorkerHeartbeat` thread also stops, ceasing updates to the `worker:heartbeat:{worker_id}` key in Redis. The `WorkerHeartbeat` key has a TTL slightly longer than the heartbeat interval (e.g., interval=30s, TTL=45s). After the TTL expires, the key disappears. The monitoring system's `AlertManager` or dashboard, which periodically scans for worker keys, now identifies this worker as "stale" or "dead."

3.  **Job Re-queue via Processing Queue:** Because the job was dequeued using `BRPOPLPUSH` (or a similar pattern), its `job_id` remains in the `queue:{queue_name}:processing` list. A separate "janitor" process (which can be part of the `Worker` startup routine or a dedicated maintenance script) periodically scans all `queue:*:processing` lists. For each `job_id` found, it checks if the associated worker (which can be tracked by storing the `worker_id` in the processing list or alongside the job) is still alive (by checking the heartbeat key). If the worker is dead, the janitor process moves the `job_id` back from the `queue:{queue_name}:processing` list to the main `queue:{queue_name}` list using `RPOPLPUSH` or a similar atomic operation. The job is now `PENDING` again and available for any healthy worker.

4.  **Worker Retry and Failure:** A healthy worker picks up the job (returning to step 4). It loads the job, sees its `status` is `PENDING` (or `ACTIVE` with a very old `started_at`), and begins execution. This time, the handler fails due to a transient error (e.g., a network timeout, a deadlock). The worker catches the exception.

5.  **Error Recording and Retry Check:** The worker calls `Job.record_error(error)`, which appends a structured error dictionary (containing exception class, message, and traceback) to `Job.errors` and increments `Job.attempts`. It then calls `Job.should_retry()` which compares `Job.attempts` against `Job.max_retries`. If retries are allowed, the worker proceeds to the retry flow.

6.  **Retry Scheduling:** The worker calls `RetryManager.handle_job_failure(job, error)`. This method:
    *   Calculates the next retry delay using `BackoffCalculator.calculate_delay(Job.attempts)`. For attempt 1, this might be 1 second; for attempt 2, 2 seconds, etc., plus optional jitter.
    *   Computes an absolute timestamp for the retry: `current_time + delay`.
    *   Updates the job's `status` to `RETRY_SCHEDULED`.
    *   Uses `ZADD` to add the `job_id` with a score equal to the retry timestamp to a Redis sorted set key, e.g., `retry_queue:{queue_name}`.
    *   Persists the updated job (with the new error and attempt count) to the `job:{job_id}` hash.
    *   Removes the `job_id` from the processing queue.

7.  **Retry Polling and Re-enqueue:** The `RetryManager` (or a dedicated thread within each `Worker`) periodically polls the retry sorted sets using `ZRANGEBYSCORE` with a score range of `0` to `current_timestamp`. For each due `job_id`, it retrieves the full job, changes its `status` back to `PENDING`, and calls `QueueManager.enqueue_job(job)` to place it back on its original main queue. The job ID is removed from the sorted set via `ZREM`.

8.  **Eventual Success or Final Failure:** The job is retried by workers (looping through steps 4-7). If it eventually succeeds, the happy path completes. If `Job.attempts` exceeds `Job.max_retries`, `Job.should_retry()` returns `False`.

9.  **Dead Letter Queue (DLQ) Placement:** In the case of final failure, `RetryManager.handle_job_failure` calls `RetryManager.move_to_dead_letter(job)`. This method:
    *   Sets the job's `status` to `DEAD_LETTER`.
    *   Persists the final state to `job:{job_id}`.
    *   Adds the `job_id` to a dedicated list or set key, e.g., `dead_letter_queue`.
    *   Triggers a high-priority alert (e.g., `CRITICAL` severity) via the `AlertManager`.
    *   The job is now out of the processing loop and awaits manual intervention via the dashboard's `retry_dead_letter_job` or `delete_dead_letter_job` APIs.

| Failure Scenario | Detection Mechanism | Automatic Recovery Action | Manual Intervention Required? |
| :--- | :--- | :--- | :--- |
| Worker Process Crash | Heartbeat TTL expiry | Job moved from processing queue back to main queue via janitor process. | No, unless the janitor process itself fails. |
| Transient Handler Error (e.g., network timeout) | Exception caught by worker wrapper. | Job is scheduled for retry with exponential backoff. | No. |
| Persistent Handler Error (e.g., bug in code) | Max retries exhausted. | Job moved to Dead Letter Queue (DLQ). | **Yes.** An operator must inspect the error and decide to retry or delete. |
| Redis Connection Loss | Redis client exception, heartbeat fails. | Worker may crash/restart; jobs remain in processing queue until janitor cleans up. | No, assuming worker process manager (e.g., systemd) restarts it. |

> **Design Insight:** The combination of a **processing queue** (for crash safety) and a **retry sorted set** (for scheduled retries) is crucial. The processing queue ensures no job is ever truly "in flight" without a paper trail in Redis, enabling recovery from worker crashes. The sorted set provides efficient scheduling without requiring a separate timer thread for each failed job.

#### Common Pitfalls in Error Recovery

⚠️ **Pitfall: Busy-Waiting on Dead Workers**
**Mistake:** The janitor process continuously polls the processing queues in a tight loop, consuming CPU and Redis resources even when no workers have crashed.
**Why it's wrong:** This wastes resources and can exacerbate problems during system stress.
**Fix:** Implement an exponential backoff in the janitor's polling loop (e.g., sleep for 5 seconds, then 10, up to a max of 60 seconds). Additionally, use Redis's `KEYS` or `SCAN` command sparingly to find processing queues, and consider publishing events to a Pub/Sub channel when a worker stops heartbeating to trigger immediate cleanup.

⚠️ **Pitfall: Losing Error Context on Retry**
**Mistake:** Overwriting the job's error field on each retry, so only the last error is preserved.
**Why it's wrong:** Debugging becomes difficult because you cannot see the history of failures that led to the dead letter state.
**Fix:** Use `Job.record_error(error)`, which *appends* to the `Job.errors` list. Ensure the serialization format (e.g., JSON) supports a list of complex error objects.

⚠️ **Pitfall: Retry Storm on Permanent Failure**
**Mistake:** A job with a permanent error (e.g., "FileNotFound") is continuously retried, wasting resources and clogging the queue until `max_retries` is reached.
**Why it's wrong:** It delays the processing of other jobs and creates unnecessary load.
**Fix:** Implement **retry filters** or allow handlers to raise specific exception types (e.g., `ImmediateFailError`) that bypass the retry logic and go directly to the DLQ.

### Scheduled Job: Cron to Execution

**Mental Model: Calendar App with Recurring Events**
Think of a sophisticated calendar application. You create a recurring event "Team Stand-up" scheduled for "Every weekday at 9 AM UTC." The calendar service doesn't create all future events at once. Instead, it constantly looks at the current time and the rules for all recurring events. When it detects that the current time matches a rule's "next scheduled time," it creates a new calendar entry (enqueues a job) for that specific occurrence. It then immediately calculates the *next* occurrence after this one and waits. If the service is down for maintenance at 9 AM, upon restart, it checks which events it missed and creates them immediately (catch-up logic).

This scenario involves the `Scheduler` component, demonstrating how future-dated and recurring jobs are managed, enqueued, and eventually processed.

#### Step-by-Step Flow

1.  **Schedule Definition:** An administrator or application calls `schedule_recurring_job(job, cron_expression, timezone='UTC')`. This function creates a `Schedule` object, assigns a `schedule_id`, and calculates the initial `next_run_at` time using `_calculate_next_run(schedule, reference_time=now)`. The `Schedule` is persisted to a Redis hash (e.g., `schedule:{schedule_id}`). The `Schedule.status` is set to `PENDING` for its future runs.

2.  **Scheduler Polling Loop:** The `Scheduler` process runs `_poll_due_schedules()` in a loop, sleeping for `polling_interval_seconds` between iterations. During each poll:
    *   It fetches all `Schedule` objects from Redis where `enabled=True`.
    *   For each schedule, it checks if `Schedule.next_run_at <= current_time`. If not, it skips.
    *   For due schedules, it calls `_process_schedule(schedule)`.

3.  **Processing a Due Schedule:** `_process_schedule` performs the following atomic operations (typically within a Redis pipeline/WATCH to handle concurrent schedulers):
    *   **Uniqueness Check:** If `Schedule.unique_key` is set, it uses Redis's `SET key value NX EX window` command to claim a lock for the `unique_window_seconds`. If the lock cannot be acquired (because a duplicate was enqueued recently), the schedule's status for this run is marked as `SKIPPED`, `last_enqueued_at` is updated, and the next run time is recalculated.
    *   **Job Creation and Enqueue:** If unique (or not required), it creates a concrete `Job` instance from the template in the `Schedule`. The job's `queue`, `job_type`, `args`, `kwargs`, and metadata are copied. The job's `metadata` field may be augmented with the `schedule_id` and the `scheduled_for` timestamp.
    *   `_enqueue_scheduled_job` is called, which uses `QueueManager.enqueue_job` to push the job onto the designated worker queue.
    *   **Schedule Update:** Upon successful enqueue, the schedule's `last_enqueued_at` is set to the current time, and `status` for this run is set to `ENQUEUED`. `Schedule.next_run_at` is recalculated based on the `cron_expression` and `timezone`, using `_calculate_next_run(schedule, reference_time=now)`.

4.  **Catch-up Logic on Scheduler Start:** When the `Scheduler` starts (e.g., after a deployment or crash), it runs `calculate_next_runs_for_all_schedules()`. This method ensures `next_run_at` is set correctly for all schedules. More importantly, if `Scheduler` detects that a schedule's `next_run_at` is in the past (meaning it missed runs while down), it will immediately process those missed schedules, enqueuing jobs for each missed occurrence up to a sensible limit (to prevent a thundering herd). This is a critical feature for reliability.

5.  **Job Execution:** From this point, the journey of the enqueued job is identical to the **Happy Path**. Workers pick it up (step 4 onward) and execute it. The job's metadata retains its origin as a scheduled job, which can be useful for monitoring and debugging.

6.  **Timezone and Daylight Saving Time (DST) Handling:** The `_calculate_next_run` function uses a timezone-aware datetime library (like `pytz` or `zoneinfo`). When calculating the next run for a schedule in a timezone that observes DST, the function correctly handles ambiguous or non-existent times (e.g., 2:30 AM during a "spring forward" transition). A common strategy is to default to the later valid time during an ambiguous period and to skip non-existent times by advancing to the next valid interval.

| Schedule Event | Scheduler Action | Redis Command Example | Outcome |
| :--- | :--- | :--- | :--- |
| New recurring schedule created. | Calculate `next_run_at`, store schedule. | `HSET schedule:{id} ...` | Schedule is dormant until its first due time. |
| Poll finds due schedule. | Check uniqueness, create job, enqueue, update schedule. | Pipeline: `SET NX`, `HSET job:{jid}`, `LPUSH queue:...`, `HSET schedule:{sid}` | Job is placed on worker queue. Schedule is updated for next run. |
| Scheduler restarts after crash. | Recalculate next runs, process missed schedules. | `ZSCAN` for past-due `next_run_at` scores, then process. | Missed jobs are enqueued, ensuring no schedule is permanently skipped. |
| Admin pauses a schedule. | Set `Schedule.enabled = False`. | `HSET schedule:{id} enabled 0` | Scheduler skips this schedule in future polls. |

#### Architecture Decision Record: Scheduler Catch-up Strategy

> **Decision: Limited Catch-up with Skip Option**
> - **Context:** When the Scheduler process is down (for maintenance, crash, etc.), scheduled jobs will be missed. Upon restart, we must decide how to handle these missed occurrences.
> - **Options Considered:**
>     1.  **Enqueue All Missed Jobs:** Calculate every missed execution time since the last successful run and enqueue a job for each.
>     2.  **Enqueue Only the Most Recent Missed Job:** Enqueue a single job for the most recent missed occurrence.
>     3.  **Skip All Missed Jobs:** Do not enqueue any missed jobs; simply resume the schedule from the next future time.
>     4.  **Configurable, Limited Catch-up:** Allow configuration to enqueue up to N missed jobs, with an option to skip the rest.
> - **Decision:** Option 4 (Configurable, Limited Catch-up).
> - **Rationale:** Option 1 can cause a dangerous "thundering herd" problem, instantly flooding the system with a backlog of jobs which may overwhelm workers and downstream services. Option 2 loses data (missed executions). Option 3 may be unacceptable for critical schedules (e.g., daily reports). Option 4 provides safety with a cap (e.g., max 10 catch-up jobs) and allows operators to tune based on the job's idempotency and importance.
> - **Consequences:** The system is protected from self-inflicted denial-of-service on restart. Operators must understand the catch-up limit for their critical schedules. Some missed jobs beyond the limit will be permanently skipped, which may require manual intervention for critical business processes.

| Option | Pros | Cons | Why Not Chosen |
| :--- | :--- | :--- | :--- |
| Enqueue All | No data loss, simple. | Risk of thundering herd, system overload. | Too dangerous for automated recovery. |
| Enqueue Most Recent | Handles short outages, simple. | Loses history of multiple missed runs. | Unacceptable for schedules with high frequency or long outages. |
| Skip All | Safe, no overload. | Potentially unacceptable data loss for business. | Lacks flexibility for important jobs. |
| **Configurable, Limited Catch-up** | **Safe, flexible, tunable per schedule.** | **Adds configuration complexity.** | **Chosen for optimal safety and control.** |

### Implementation Guidance

This section provides concrete code skeletons and structure to implement the interaction flows described above.

#### A. Technology Recommendations Table

| Interaction Scenario | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| Reliable Job Dequeue (Worker) | Use `BRPOPLPUSH` with a processing queue. | Use Redis Streams (`XREADGROUP`, `XACK`) for more sophisticated consumer groups and pending entry management. |
| Retry Scheduling | Use a Redis Sorted Set (`ZSET`) with retry timestamp as score. | Use a dedicated time-series database or a priority queue service for extremely high-scale retry scheduling. |
| Scheduler Polling | Simple `time.sleep()` loop in the main thread. | Use `asyncio` with async Redis client for non-blocking polling of multiple schedules. |
| Janitor Process for Crashed Workers | A separate script run by a cron job every minute. | Integrate the cleanup logic into the `Worker`'s startup routine and use Redis Pub/Sub to broadcast worker deaths for immediate cleanup. |

#### B. Recommended File/Module Structure

Add the following files to manage the interaction flows and recovery processes:

```
background_job_processor/
├── core/
│   ├── __init__.py
│   ├── job.py                  # Job class definition
│   ├── queue_manager.py        # QueueManager class
│   ├── worker.py               # Worker class, main loop
│   ├── retry_manager.py        # RetryManager class
│   ├── scheduler.py            # Scheduler class
│   └── janitor.py              # Cleanup process for stalled jobs
├── models/
│   ├── __init__.py
│   ├── schedule.py             # Schedule, ScheduledJob classes
│   └── metrics.py              # MetricPoint, QueueMetrics, etc.
├── web/
│   ├── __init__.py
│   └── dashboard.py            # API endpoints for dashboard
├── utils/
│   ├── __init__.py
│   ├── redis_client.py         # RedisClient wrapper
│   └── backoff.py              # BackoffCalculator
└── scripts/
    ├── run_worker.py           # Worker entry point
    ├── run_scheduler.py        # Scheduler entry point
    └── run_janitor.py          # Janitor script (can be cron)
```

#### C. Infrastructure Starter Code

**`utils/redis_client.py` - A robust Redis client wrapper:**
```python
import redis
import logging
from typing import Optional, Any
from functools import wraps

class RedisClient:
    """Singleton wrapper for Redis client with connection pooling and error handling."""
    
    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None
    
    def __init__(self, url: str, **connection_kwargs):
        # ... (initialization as per naming conventions)
    
    @classmethod
    def get_instance(cls, url: Optional[str] = None, **kwargs) -> 'RedisClient':
        if cls._instance is None:
            if url is None:
                raise ValueError("URL required for first RedisClient initialization")
            cls._instance = cls(url, **kwargs)
        return cls._instance
    
    def get_client(self) -> redis.Redis:
        if self._client is None or not self.ping():
            self._reconnect()
        return self._client
    
    def ping(self) -> bool:
        try:
            return self._client is not None and self._client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            return False
    
    def _reconnect(self):
        # ... reconnection logic with retry backoff
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute a Redis command with automatic retry on connection errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = self.get_client()
                method = getattr(client, command.lower())
                return method(*args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                logging.warning(f"Redis connection error on {command}, retry {attempt+1}: {e}")
                self._reconnect()
            except redis.RedisError as e:
                logging.error(f"Redis error on {command}: {e}")
                raise
    
    def pipeline(self) -> redis.Pipeline:
        """Return a pipeline for atomic operations."""
        return self.get_client().pipeline()
```

#### D. Core Logic Skeleton Code

**`core/janitor.py` - Cleanup process for crashed workers:**
```python
import time
import logging
from typing import List
from .redis_client import RedisClient
from core.queue_manager import QueueManager

class Janitor:
    """Cleans up jobs stuck in processing queues from dead workers."""
    
    def __init__(self, redis_client: RedisClient, scan_interval: int = 30):
        self.redis = redis_client
        self.scan_interval = scan_interval
        self.running = False
        
    def _get_stale_processing_queues(self) -> List[str]:
        """Find all queue:processing lists where the associated worker is dead."""
        # TODO 1: Use SCAN to find all keys matching pattern 'queue:*:processing'
        # TODO 2: For each processing queue key, extract the worker_id (might be stored in a separate key or in the list's metadata)
        # TODO 3: Check if the worker's heartbeat key (worker:heartbeat:{id}) exists and hasn't expired
        # TODO 4: If worker is dead, add the processing queue key to the result list
        # TODO 5: Return the list of stale processing queue keys
        pass
    
    def _requeue_stale_jobs(self, processing_queue_key: str) -> int:
        """Move all jobs from a stale processing queue back to its main queue."""
        # TODO 1: Extract the main queue name from the processing queue key (e.g., 'queue:email:processing' -> 'queue:email')
        # TODO 2: In a loop, use RPOPLPUSH to move jobs from processing queue back to main queue atomically
        # TODO 3: Count the number of jobs moved
        # TODO 4: Log a warning for each job moved
        # TODO 5: Return the count of requeued jobs
        pass
    
    def run_once(self) -> int:
        """Execute a single cleanup cycle. Returns number of jobs requeued."""
        total_requeued = 0
        stale_queues = self._get_stale_processing_queues()
        for queue_key in stale_queues:
            count = self._requeue_stale_jobs(queue_key)
            total_requeued += count
        return total_requeued
    
    def start(self):
        """Start the janitor's main loop."""
        self.running = True
        logging.info("Janitor started.")
        while self.running:
            try:
                requeued = self.run_once()
                if requeued > 0:
                    logging.warning(f"Janitor requeued {requeued} stale jobs.")
            except Exception as e:
                logging.error(f"Janitor error: {e}", exc_info=True)
            time.sleep(self.scan_interval)
    
    def stop(self):
        self.running = False
```

**`core/retry_manager.py` - Handling retry scheduling:**
```python
import time
from datetime import datetime
from typing import List, Optional
from .job import Job, JobStatus
from .redis_client import RedisClient
from utils.backoff import BackoffCalculator

class RetryManager:
    
    def handle_job_failure(self, job: Job, error: Exception) -> None:
        """Main entry point: decide retry or dead letter."""
        # TODO 1: Call job.record_error(error) to update error list and attempt count
        # TODO 2: Call job.should_retry() to check if max_retries not exceeded
        # TODO 3: If should_retry is True:
        #   - Calculate delay using self.backoff_calculator.calculate_delay(job.attempts)
        #   - Call self.schedule_retry(job, delay)
        # TODO 4: If should_retry is False:
        #   - Call self.move_to_dead_letter(job)
        #   - Log a critical error
        # TODO 5: Update the job's status in Redis to RETRY_SCHEDULED or DEAD_LETTER
        pass
    
    def schedule_retry(self, job: Job, delay_seconds: float) -> str:
        """Schedule a job for retry after a delay."""
        # TODO 1: Calculate the absolute retry timestamp: current_time + delay_seconds
        # TODO 2: Use Redis ZADD to add job.job_id to sorted set key 'retry_queue:{job.queue}' with score = retry_timestamp
        # TODO 3: Return the job_id
        pass
    
    def get_due_retries(self, queue_name: str, max_count: int = 100) -> List[Job]:
        """Get jobs whose retry time has arrived."""
        # TODO 1: Get current timestamp
        # TODO 2: Use ZRANGEBYSCORE with scores 0 to current_timestamp to get job_ids due for retry
        # TODO 3: For each job_id, fetch the full Job from Redis using QueueManager.get_job_by_id
        # TODO 4: Return the list of Job objects
        pass
```

#### E. Language-Specific Hints (Python)

*   **Atomic Operations:** Always use Redis pipelines (`self.redis.pipeline()`) for operations that must be atomic, like enqueuing a job (store + push) or moving a job between queues. Use `WATCH` for optimistic concurrency control when updating schedules.
*   **Time and Timezones:** Use `datetime.datetime.now(datetime.timezone.utc)` for all timestamps stored in Redis. For cron schedule evaluation with timezones, use the `pytz` or `zoneinfo` (Python 3.9+) libraries. Be mindful of ambiguous times during DST transitions.
*   **Graceful Shutdown:** Use `signal.signal(signal.SIGTERM, handler)` in your `Worker` and `Scheduler` main loops to catch termination signals. Set a flag (`_shutdown_requested`) and break out of the loop cleanly after finishing the current unit of work.
*   **Connection Management:** Use the provided `RedisClient` wrapper, which includes connection pooling and automatic retry on transient network errors. Configure `max_connections` in the pool to match your concurrency level.

#### F. Milestone Checkpoint for Interactions

After implementing the core components (Milestones 1-4), you should be able to run the following integration test:

1.  **Start Infrastructure:** Start Redis. Run one worker (`python scripts/run_worker.py --queues default`). Run the scheduler (`python scripts/run_scheduler.py`).
2.  **Test Happy Path:** Use a Python shell to enqueue a simple job that prints a message. Verify the worker prints the message and the job status in Redis becomes `COMPLETED`.
3.  **Test Worker Crash Recovery:** Enqueue a long-running job (e.g., `time.sleep(30)`). While it's running, kill the worker process (Ctrl+C or `kill -9`). Wait for the heartbeat TTL (e.g., 45 seconds). Start the worker again. Observe that the killed job is re-queued and eventually completes.
4.  **Test Retry Logic:** Enqueue a job that fails with an exception and has `max_retries=3`. Watch the logs and Redis sorted set (`ZRANGE retry_queue:default 0 -1 WITHSCORES`). You should see the job ID scheduled, then retried, until it finally moves to the dead letter queue after 3 attempts.
5.  **Test Scheduled Job:** Create a recurring schedule that runs every minute. Observe the scheduler logs and the main queue length. You should see a new job appear in the queue approximately every minute.

**Signs of Success:** Jobs complete reliably despite failures. The dashboard shows accurate queue depths, worker status, and job history. No jobs are lost when workers or the scheduler restart.

**Common Debugging Issues:**
*   **Jobs disappearing:** Likely a bug in the reliable dequeue pattern (`BRPOPLPUSH`). Check that jobs are being moved to the processing queue and removed only after successful completion.
*   **Retries not happening:** Ensure the `RetryManager.get_due_retries` is being called periodically (e.g., by the worker before polling for new jobs) and that the retry timestamps in the sorted set are correct.
*   **Scheduler not enqueuing jobs:** Check the scheduler's polling loop logs. Verify cron expressions are valid and that `next_run_at` is being calculated correctly with the proper timezone.


## Error Handling and Edge Cases

> **Milestone(s):** This section spans all five milestones, addressing cross-cutting concerns of failure detection, recovery, and edge case management that are critical for building a production-ready distributed system.

Distributed systems operate in an inherently unreliable environment where failures are not exceptional but expected. A background job processor must be designed with the fundamental assumption that network connections will drop, processes will crash, clocks will drift, and data will become corrupted. This section systematically catalogs the failure modes the system must handle, details how to detect them, outlines recovery strategies, and provides specific solutions for challenging edge cases. Unlike simpler applications where failures might be catastrophic, our design treats failures as normal operational states that trigger well-defined recovery paths, ensuring the system maintains job durability, processing guarantees, and operational visibility even under adverse conditions.

### Failure Modes and Detection

> **Mental Model: The Hospital Emergency Room Triage System**  
> Imagine an emergency room where patients arrive with various conditions. The triage nurse must quickly assess each patient's vital signs, classify the severity of their condition, and route them to the appropriate treatment area. Some issues are immediately life-threatening (cardiac arrest), some are urgent but not critical (broken bones), and some are minor (small cuts). Similarly, our system must continuously monitor its own "vital signs"—network connectivity, process health, data integrity—and classify failures by severity to apply the correct intervention. Detection mechanisms act as the triage nurses, constantly checking pulses and responding to alarms.

The system must detect failures across multiple dimensions: infrastructure (Redis, network), processes (workers, scheduler), data (corruption, validation), and temporal (timeouts, clock skew). Each failure mode requires specific detection strategies that balance sensitivity (catching real failures) with specificity (avoiding false alarms). The following table categorizes the primary failure modes, their symptoms, and detection mechanisms:

| Failure Mode | Symptoms & Indicators | Detection Mechanism | Severity Class |
|--------------|-----------------------|---------------------|----------------|
| **Redis Connection Failure** | TCP connection timeout, connection reset, "Connection refused" errors, inability to ping Redis | Periodic health checks (e.g., `PING` command every 5 seconds), monitoring connection pool exhaustion, tracking consecutive command failures | Critical (system cannot function without broker) |
| **Network Partition** | Workers can connect to Redis but cannot communicate with each other or external services; some commands succeed while others timeout | Heartbeat cross-checks between workers, external service health checks integrated into job execution, monitoring for asymmetric failure patterns | Critical (can lead to split-brain scenarios) |
| **Worker Process Crash** | Worker heartbeat stops updating, processing queue jobs remain stuck in `ACTIVE` state beyond timeout, OS process disappears | `WorkerHeartbeat` timestamp exceeds `heartbeat_interval * 3`, `Janitor` process scans for stale `processing` queues, monitoring process PID existence | High (jobs become orphaned) |
| **Worker Graceful Shutdown Stall** | Worker receives SIGTERM but doesn't exit within graceful shutdown timeout, hangs during job cleanup | Internal shutdown timer in `Worker.stop()` method, monitoring for `Worker.state` stuck in `SHUTTING_DOWN` beyond timeout | Medium (blocks deployment rotations) |
| **Job Handler Timeout** | Job execution exceeds `Job.timeout_seconds`, worker becomes unresponsive to heartbeats during execution | Per-job timeout thread/process monitoring, watchdog timer that sends SIGKILL to runaway job handlers | Medium (resource leak, blocks queue) |
| **Job Data Corruption** | `Job.deserialize()` raises decoding errors, missing required fields, invalid data types | Validation in `QueueManager._validate_job()` during enqueue, checksum verification (CRC32) stored with serialized data, schema validation on deserialize | High (job lost, cannot be processed) |
| **Redis Memory Exhaustion** | Redis returns OOM errors, `used_memory` approaches `maxmemory`, key evictions occur despite `noeviction` policy | Monitoring Redis metrics via `INFO memory`, alerting on memory usage >80%, tracking eviction count for `noeviction` policy | Critical (data loss possible) |
| **Clock Skew** | Scheduled jobs fire at wrong times, retry delays miscalculated, heartbeat timestamps inconsistent across servers | NTP monitoring, comparing system time to Redis server time (`TIME` command), detecting scheduler runs that are significantly early/late | Medium (timing-sensitive operations fail) |
| **Deadlock in Worker Concurrency** | Worker's thread/process pool stops making progress, all workers idle but jobs remain in queue, CPU usage drops to zero | Monitoring job completion rate dropping to zero while queue non-empty, watchdog thread that samples executor thread states | High (silent failure, hard to detect) |
| **Queue Overflow** | `QueueConfig.max_length` exceeded, `QueueFullError` exceptions, enqueue operations fail | Monitoring queue lengths vs. configured limits, alerting on `QueueFullError` frequency, tracking producer backpressure | Medium (job submission blocked) |
| **Silent Data Corruption in Redis** | Job data retrieved from Redis differs from what was stored, bit flips due to hardware issues | Store CRC32 checksum with each serialized job, verify on retrieval, Redis `DEBUG DIGEST` command for entire dataset (expensive) | Critical (data integrity compromised) |

> **Architecture Decision: Multi-Layered Failure Detection**
> **Context**: We need to detect failures promptly without overwhelming the system with monitoring overhead or generating excessive false positives. Different components have different failure characteristics requiring tailored detection approaches.
> **Options Considered**:
> 1. **Centralized Health Service**: A dedicated service periodically probes all components and aggregates status.
> 2. **Decentralized Self-Reporting**: Each component reports its own health via heartbeets and logs, with consumers interpreting.
> 3. **Hybrid Approach**: Components self-report basic health (heartbeats) while dedicated monitors perform deeper diagnostic checks.
> **Decision**: Hybrid approach with component heartbeats for liveness and dedicated `Janitor` and `MetricsAggregator` for deeper health checks.
> **Rationale**: Heartbeats provide cheap, continuous liveness signals with low latency. Dedicated diagnostic processes can perform more expensive checks (like stale job detection) without impacting primary data paths. This balances detection speed with system overhead.
> **Consequences**: Requires implementing both heartbeat mechanisms and background maintenance processes. Failure detection latency depends on configuration: heartbeat intervals for quick detection vs. janitor scan intervals for orphaned job cleanup.

Detection mechanisms are implemented across multiple components:
- **Heartbeat System**: `WorkerHeartbeat` updates a Redis key with current timestamp at `heartbeat_interval`. The dashboard and alerting systems monitor these keys for staleness (timestamp > interval × 3).
- **Janitor Process**: `Janitor` periodically scans all `processing` queues (e.g., `queue:processing:{worker_id}`) and checks if the associated worker's heartbeat is current. If not, it assumes the worker crashed and re-queues the jobs.
- **Metric-Based Alerts**: `AlertManager` evaluates rules against metrics like queue growth rate, error rate, and worker count, firing alerts when thresholds are breached.
- **Timeout Watchdogs**: Each `Worker` runs a watchdog timer per job that kills the job handler if it exceeds `Job.timeout_seconds`. The scheduler has its own timeout for the polling loop to detect stalls.
- **Connection Health Checks**: `RedisClient` implements circuit breaker patterns, tracking consecutive failures and entering "open" state to prevent cascading failures during Redis outages.

### Recovery and Compensation Strategies

> **Mental Model: The Automobile Insurance Claims Process**  
> When an accident occurs, insurance companies don't just fix the car—they follow a structured process: assess damage, determine fault, arrange repairs, provide rental cars, and handle medical bills. Each type of damage has a corresponding recovery procedure. Similarly, our system must have specific recovery strategies for each failure mode, not just restarting components but compensating for lost work, preserving data integrity, and restoring normal operation with minimal manual intervention.

Once a failure is detected, the system must execute appropriate recovery actions. The strategy depends on the failure type, component affected, and current system state. Recovery aims to restore system functionality while preserving job durability guarantees (at-least-once or at-most-once semantics) and maintaining data consistency. The following table maps failure modes to recovery strategies:

| Failure Mode | Immediate Recovery Action | Compensation Strategy | Data Consistency Impact |
|--------------|---------------------------|----------------------|-------------------------|
| **Redis Connection Failure** | Exponential backoff reconnect, fail-fast on enqueue operations with clear error to producers | Buffer jobs in memory (with size limit) for later retry, enter degraded mode (pause scheduling, slow heartbeats) | Jobs may be lost if memory buffer overflows before Redis recovery. At-least-once semantics compromised during outage. |
| **Worker Process Crash** | `Janitor` detects stale `processing` queue and moves jobs back to main queue with `Job.attempts` incremented | Worker supervisor (systemd, kubernetes) restarts worker process, `WorkerHeartbeat` cleanup removes orphaned heartbeat key | Job may be partially executed (non-idempotent operations require careful design). At-least-once semantics preserved. |
| **Job Handler Timeout** | Watchdog sends SIGKILL to job handler process/thread, job marked as `FAILED` and passed to `RetryManager` | Retry with exponential backoff (if job idempotent), alert for manual inspection if timeout persistent for job type | If job not idempotent, duplicate side effects possible on retry. System maintains forward progress. |
| **Job Data Corruption** | Job moved to dead letter queue with `VALIDATION_ERROR` flag, error details stored | Notification to operators via alert, manual inspection required to fix or discard job | Single job lost, but system continues processing other jobs. Dead letter queue provides audit trail. |
| **Redis Memory Exhaustion** | Switch to `allkeys-lru` eviction policy temporarily, emergency purge of old job history, alert operators | Scale Redis memory, restart Redis with larger `maxmemory`, drain queues to secondary storage | Possible job loss if eviction occurs. System prioritizes recent jobs over history. |
| **Clock Skew** | Alert operators to fix NTP synchronization, scheduler uses Redis time (`TIME` command) as authoritative source | Scheduler runs catch-up logic for missed scheduled jobs when clock corrects, recalculates retry timestamps | Scheduled jobs may fire late or early. Recovery ensures all jobs eventually enqueued. |
| **Deadlock in Worker Concurrency** | Watchdog thread detects stall and kills entire worker process, triggering worker crash recovery | Reduce concurrency level, analyze job handlers for locking issues, implement thread dumps for diagnosis | All jobs currently processing are failed and retried. Temporary throughput reduction. |
| **Queue Overflow** | Reject new jobs with `QueueFullError`, trigger scale-up alerts for workers, increase `max_length` temporarily | Auto-scale worker capacity, implement backpressure to producers, add overflow queue (secondary Redis instance) | Producers must handle rejection gracefully. System protects itself from overload. |
| **Network Partition** | Continue processing jobs that don't require external services, pause enqueue of jobs requiring partitioned resources | When partition heals, reconciliation process compares job states across partitions, manual intervention may be needed | Split-brain may cause duplicate processing. Idempotency keys and deterministic job IDs reduce risk. |

> **Design Principle: Graceful Degradation Over Catastrophic Failure**  
> When components fail, the system should degrade functionality predictably rather than collapsing entirely. For example, during Redis outages, workers can continue processing jobs already in memory while the queue manager buffers new jobs locally (up to a limit). The dashboard can display cached metrics when Redis is unavailable. This principle ensures the system remains partially operational during partial failures.

Recovery strategies are implemented through coordinated actions across components:

1. **Automatic Retry with Exponential Backoff**: For transient failures (network timeouts, temporary resource unavailability), the `RetryManager` schedules retries with increasing delays using `BackoffCalculator.calculate_delay()`. This prevents overwhelming recovering systems with immediate retry storms.

2. **Orphaned Job Recovery via Janitor**: The `Janitor` process runs periodically (e.g., every 30 seconds) and executes:
   - `Janitor._get_stale_processing_queues()`: Scans Redis for all `queue:processing:{worker_id}` keys
   - Checks if the corresponding worker heartbeat key (`worker:heartbeat:{worker_id}`) has expired
   - For each stale processing queue, calls `Janitor._requeue_stale_jobs()` to:
     1. Atomically move all jobs from the processing queue back to the main queue using `RPOPLPUSH` in a pipeline
     2. Increment the `Job.attempts` count for each job
     3. Update `Job.status` from `ACTIVE` to `PENDING`
     4. Log the recovery event for auditing

3. **Circuit Breaker for External Dependencies**: The `RedisClient` implements a circuit breaker pattern that tracks consecutive connection failures. After a threshold (e.g., 5 failures), the circuit opens and subsequent commands immediately fail fast without attempting network calls. After a reset timeout (e.g., 30 seconds), the circuit enters half-open state, allowing a test command through; if successful, the circuit closes and normal operation resumes.

4. **Catch-Up Logic for Scheduler**: When the `Scheduler` starts or detects significant clock corrections, it calls `Scheduler.calculate_next_runs_for_all_schedules()` to recalculate `Schedule.next_run_at` for all schedules. It then runs `Scheduler.enqueue_due_jobs()` with an expanded time window (e.g., previous 24 hours) to enqueue any jobs missed during downtime. The uniqueness window (`Schedule.unique_window_seconds`) prevents duplicate enqueues for the same schedule period.

5. **Dead Letter Queue Manual Intervention**: Jobs that exhaust retries or fail validation are moved to the dead letter queue (`dead:letter:{queue_name}`) where they persist for manual inspection. The dashboard provides operations to:
   - View error details and stack traces
   - Retry the job (moves back to main queue with reset attempts)
   - Delete the job permanently
   - Download job payload for debugging

### Specific Edge Cases and Solutions

> **Mental Model: The "Corner Cases" Playbook in Sports**  
> Professional sports teams prepare for unusual game situations: what if the ball hits a bird mid-flight? What if a player's jersey gets torn? They have predetermined rules and procedures. Similarly, our system must anticipate rare but possible edge cases and have explicit solutions codified rather than leaving behavior undefined.

Edge cases are scenarios that occur infrequently but can cause system instability, data loss, or undefined behavior if not handled explicitly. These often involve timing issues, race conditions, or unexpected combinations of normal events. The following table documents specific edge cases and their solutions:

| Edge Case Scenario | Description | Potential Impact | Solution |
|-------------------|-------------|------------------|----------|
| **Worker Crash During Job Acknowledgement** | Worker successfully processes job but crashes after business logic completes but before removing job from processing queue. | Job could be re-queued by `Janitor` and executed twice (duplicate processing). | Use atomic "process and acknowledge" operations: store job result in Redis atomically with removal from processing queue via `RedisClient.pipeline()` multi-exec. |
| **Clock Jump Forward/Backward** | System time changes abruptly due to NTP correction or manual clock adjustment. | Scheduled jobs may fire at wrong times, retry delays miscalculated, heartbeat comparisons invalid. | Use Redis `TIME` command as monotonic clock source for all scheduling decisions. For heartbeats, use system uptime (`time.monotonic()` in Python) for intervals, wall clock for absolute times. |
| **Simultaneous Worker Startup with Same ID** | Two workers accidentally configured with same `worker_id` (e.g., duplicated container image). | Both update same heartbeat key, making liveness detection unreliable; both might process same jobs. | Generate unique `worker_id` automatically: `hostname:pid:timestamp` (e.g., `web01:12345:1678901234`). Include random component in heartbeat key to detect collisions. |
| **Job Payload Contains Non-Serializable Objects** | Producer passes a database connection, file handle, or lambda function in job arguments. | `Job.serialize()` fails or produces corrupted serialization that cannot be deserialized later. | `QueueManager._validate_job()` performs deep validation using `json.dumps()` test before enqueue. Reject with `ValidationError` describing the problematic field. |
| **Redis Failover During Transaction** | Redis primary fails after a multi-exec transaction starts but before it completes. | Partial transaction applied, leaving data inconsistent (e.g., job enqueued but job metadata not stored). | Use Redis Cluster with proper failover configuration (minimizes window). Implement idempotent operations where possible. Add reconciliation checks that scan for inconsistencies. |
| **Queue Priority Starvation** | High-priority queue constantly has jobs, preventing lower-priority queues from ever being polled. | Low-priority jobs never get processed, effectively creating a denial of service for certain job types. | Implement weighted round-robin with dynamic adjustment: track time since last poll per queue and temporarily boost priority of starved queues. |
| **Zombie Jobs in Processing Queue** | Job marked as `ACTIVE` but no worker is processing it (worker died without cleanup). | Job stuck forever, consuming memory, not retried. | `Janitor` scans all processing queues, not just those with heartbeat keys. Any job in processing queue for longer than `job_timeout * 2` is automatically re-queued. |
| **Cron Expression Matching Daylight Saving Transition** | Cron schedule like "30 2 * * *" (2:30 AM daily) on a day when clocks spring forward from 2:00 to 3:00. | Job may be skipped (no 2:30 AM exists) or run twice (when falling back). | Use UTC for all cron evaluations, or use timezone-aware scheduling with `pytz` library that handles DST transitions correctly (skips or doubles as appropriate). |
| **Retry Storm After Service Recovery** | External service is down, causing many jobs to fail and schedule retries. When service recovers, all retries fire simultaneously. | Thundering herd overwhelms recovering service, causing it to fail again. | Add jitter to retry delays (`BackoffCalculator` with `jitter_factor`). Implement congestion control: gradually increase retry concurrency after service recovery detected. |
| **Memory Bloat from Job History** | Job history and metrics accumulate indefinitely in Redis, consuming all memory. | Redis crashes or evicts current job data, causing system failure. | Implement TTL on all job metadata keys (e.g., 7 days). `Janitor` process periodically purges old records. Move old data to cold storage (S3, database) before deletion. |
| **Two Workers Simultaneously Poll Same Queue** | Both workers call `BRPOPLPUSH` at same instant, both get different jobs (normal). But race condition could cause both to get same job in some edge cases. | Same job processed twice. | Redis `BRPOPLPUSH` is atomic by design, but to be extra safe, use `RPOPLPUSH` with transaction and check for duplicate job IDs in processing queue before execution. |
| **Job Depends on Deleted External Resource** | Job references a database record ID that gets deleted before job executes. | Job fails with "record not found" error, retries won't help. | Implement early validation in job handler: check resource existence, if missing, mark job as `COMPLETED` with warning (or move to dead letter with `SKIPPED` reason). |
| **Scheduler Process Deadlock** | Scheduler holds Redis connection while waiting for queue manager, queue manager waiting for scheduler (circular dependency). | All scheduled jobs stop being enqueued, but system appears healthy (no crashes). | Use connection pooling with timeout, implement watchdog thread that monitors scheduler loop progress, break circular dependencies with async patterns. |
| **ULID Collision** | Extremely rare case where two jobs generate same ULID (1 in 2^128 probability). | One job overwrites another in Redis storage, causing data loss. | Fall back to check-and-regenerate: after generating ULID, check `EXISTS` in Redis, if exists, append random suffix. Log collision as critical event. |

> **Key Insight: Edge Cases Are Inevitable, Not Theoretical**  
> In distributed systems running at scale, statistically improbable events occur regularly. The "one in a million" chance happens daily when processing millions of jobs. Therefore, every edge case in this table must be treated as a mandatory requirement, not an optional enhancement.

Several edge cases require particular attention in implementation:

**Clock Skew and Time Handling**: The system uses a hybrid time strategy:
- **Redis as Authoritative Clock**: For all scheduled job times and retry timestamps, use `RedisClient.execute('TIME')` which returns server time unaffected by client clock skew.
- **Monotonic Clocks for Intervals**: For heartbeat intervals and job timeouts, use `time.monotonic()` (Python) or `process.hrtime()` (Node.js) which are unaffected by system time adjustments.
- **Timezone-Aware Scheduling**: Store `Schedule.timezone` as IANA timezone string (e.g., "America/New_York") and use `pytz` or equivalent to convert UTC to local time for cron evaluation, handling DST transitions correctly.

**Idempotency and Duplicate Processing**: To handle potential duplicate job execution (from retries, worker crashes, or network partitions):
- **Idempotency Keys**: Allow producers to pass an `idempotency_key` in `Job.metadata`. Store this in Redis with TTL (e.g., 24 hours). Before processing job, check if key exists; if yes, skip execution and return previous result.
- **Deterministic Job IDs**: For recurring jobs, generate job ID from schedule ID plus execution timestamp (e.g., `{schedule_id}:{yyyymmddhhmm}`) to prevent duplicate enqueues for same schedule period.
- **Transactional State Updates**: Use Redis pipelines to atomically update job status, store results, and remove from processing queue in a single operation.

**Memory Management and Cleanup**: To prevent Redis memory exhaustion:
- **TTL on All Keys**: Set appropriate TTLs: job metadata (7 days), worker heartbeats (2×heartbeat_interval), metrics (30 days), processing queues (job_timeout×3).
- **Janitor Cleanup**: `Janitor` also removes expired keys beyond their TTL using `SCAN` and `DEL` operations in batches to avoid blocking Redis.
- **Job History Archival**: For jobs older than TTL, move to cold storage (compress and store in S3) before deletion, maintaining audit trail without bloating Redis.

**Network Partition Handling (Split-Brain)**: In a network partition where workers can't reach Redis but can reach each other:
- **Continue Processing Local Jobs**: Workers can complete jobs already fetched into memory, storing results locally.
- **Buffer Results for Reconciliation**: When partition heals, workers push buffered results to Redis with conflict resolution (last-write-wins with timestamp).
- **Fencing Tokens**: Use Redis `INCR` on a fencing token key to detect stale workers after partition; workers with lower token values are considered stale and stop processing.

### Implementation Guidance

**A. Technology Recommendations for Error Handling**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Circuit Breaker | Manual failure counting with simple timeout | `tenacity` library with retry patterns or `pybreaker` for state machine implementation |
| Connection Pooling | `redis-py` default connection pool | `redis-py` with `ConnectionPool` and `HealthCheck` enabled, custom retry logic |
| Watchdog Timer | `threading.Timer` per job execution | `concurrent.futures` with `as_completed` and timeout, process-based isolation with `multiprocessing` |
| Time Handling | `datetime.utcnow()` for timestamps, `time.monotonic()` for intervals | `pendulum` or `arrow` for timezone-aware operations, Redis `TIME` command for authoritative time |
| Health Checks | Simple endpoint returning 200 OK | Comprehensive health check verifying Redis connectivity, queue depths, worker status |

**B. Recommended File/Module Structure**

```
background_job_processor/
├── src/
│   ├── job_processor/
│   │   ├── __init__.py
│   │   ├── exceptions.py           # Custom exceptions: ValidationError, QueueFullError, etc.
│   │   ├── circuit_breaker.py      # Circuit breaker pattern implementation
│   │   ├── janitor.py              # Janitor process for cleanup
│   │   ├── time_utils.py           # Time handling utilities, monotonic clocks
│   │   └── reconciliation.py       # Reconciliation logic for network partitions
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── watchdog.py             # Job timeout watchdog
│   │   └── heartbeat.py            # WorkerHeartbeat implementation
│   └── redis/
│       ├── __init__.py
│       ├── client.py               # RedisClient with circuit breaker
│       └── health.py               # Redis health checks
└── tests/
    └── test_error_handling.py      # Tests for failure scenarios
```

**C. Infrastructure Starter Code**

```python
# src/job_processor/exceptions.py
"""Custom exceptions for the job processor."""

class JobProcessorError(Exception):
    """Base exception for all job processor errors."""
    pass

class ValidationError(JobProcessorError):
    """Raised when job validation fails."""
    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(f"{message} (field: {field})" if field else message)

class QueueFullError(JobProcessorError):
    """Raised when queue is at maximum capacity."""
    def __init__(self, queue_name: str, max_length: int):
        super().__init__(f"Queue '{queue_name}' is full (max: {max_length})")

class WorkerTimeoutError(JobProcessorError):
    """Raised when a worker operation times out."""
    pass

class RedisConnectionError(JobProcessorError):
    """Raised when Redis connection fails."""
    pass

# src/job_processor/circuit_breaker.py
"""Circuit breaker pattern implementation."""

import time
from enum import Enum
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker that tracks failures and opens circuit after threshold."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: tuple = (Exception,)
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_state_change = time.monotonic()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.monotonic() - self.last_state_change > self.recovery_timeout:
                logger.info(f"Circuit {self.name} entering HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = time.monotonic()
            else:
                raise RedisConnectionError(f"Circuit {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            # Success - reset failure count if we were in HALF_OPEN
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit {self.name} reset to CLOSED after successful call")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result
        except self.expected_exceptions as e:
            self._record_failure()
            raise
    
    def _record_failure(self):
        """Record a failure and update circuit state."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.failure_threshold):
            logger.warning(f"Circuit {self.name} opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit {self.name} re-opening after HALF_OPEN failure")
            self.state = CircuitState.OPEN
            self.last_state_change = time.monotonic()
```

**D. Core Logic Skeleton Code**

```python
# src/job_processor/janitor.py
class Janitor:
    """Maintenance process that cleans up stale jobs and expired data."""
    
    def __init__(self, redis_client: RedisClient, scan_interval: int = 30):
        self.redis = redis_client
        self.scan_interval = scan_interval
        self.running = False
        self._thread = None
    
    def _get_stale_processing_queues(self) -> List[str]:
        """
        Find all queue:processing lists where the associated worker is dead.
        
        Returns:
            List of Redis keys for stale processing queues.
        
        TODO 1: Use SCAN to find all keys matching pattern 'queue:processing:*'
        TODO 2: For each key, extract worker_id from the key name
        TODO 3: Check if worker heartbeat key exists and is recent
        TODO 4: If heartbeat missing or stale, add processing queue key to results
        TODO 5: Also check for jobs stuck in processing queue longer than 2×timeout
        """
        pass
    
    def _requeue_stale_jobs(self, processing_queue_key: str) -> int:
        """
        Move all jobs from a stale processing queue back to its main queue.
        
        Args:
            processing_queue_key: Redis key of the stale processing queue
            
        Returns:
            Number of jobs requeued.
        
        TODO 1: Parse main queue name from processing_queue_key pattern
        TODO 2: Use Redis pipeline to atomically:
            a. Get all jobs from processing queue (LRANGE)
            b. Delete processing queue (DEL)
            c. For each job:
                - Deserialize job
                - Increment job.attempts
                - Update job.status to PENDING
                - Re-serialize job
                - LPUSH to main queue
        TODO 3: Log requeue operation with job count and worker ID
        TODO 4: Return number of jobs requeued
        """
        pass
    
    def _cleanup_expired_keys(self) -> int:
        """
        Remove expired keys from Redis to prevent memory bloat.
        
        Returns:
            Number of keys deleted.
        
        TODO 1: Scan for job metadata keys (pattern 'job:*:metadata')
        TODO 2: Check TTL for each key, if TTL < 0 (expired), add to delete list
        TODO 3: Scan for old heartbeat keys (pattern 'worker:heartbeat:*')
        TODO 4: Check last update time, if older than 3×interval, add to delete list
        TODO 5: Batch delete keys in groups of 100 to avoid blocking Redis
        """
        pass
    
    def run_once(self) -> int:
        """
        Execute a single cleanup cycle.
        
        Returns:
            Total number of jobs requeued plus keys deleted.
        
        TODO 1: Call _get_stale_processing_queues() to get stale queues
        TODO 2: For each stale queue, call _requeue_stale_jobs()
        TODO 3: Call _cleanup_expired_keys()
        TODO 4: Sum and return total operations performed
        """
        pass
    
    def start(self) -> None:
        """Start the janitor's main loop in a background thread."""
        # TODO 1: Set self.running = True
        # TODO 2: Create threading.Thread target=_run_loop
        # TODO 3: Start thread, store in self._thread
        # TODO 4: Log janitor startup
        pass
    
    def stop(self) -> None:
        """Stop the janitor."""
        # TODO 1: Set self.running = False
        # TODO 2: If thread exists, wait for it to join with timeout
        # TODO 3: Log janitor shutdown
        pass
    
    def _run_loop(self) -> None:
        """Main loop that runs cleanup periodically."""
        # TODO 1: While self.running is True:
        # TODO 2: Call self.run_once()
        # TODO 3: Sleep for self.scan_interval seconds
        # TODO 4: Catch and log exceptions (don't let loop die on error)
        pass

# src/redis/client.py
class RedisClient:
    """Redis client wrapper with circuit breaker and error handling."""
    
    def execute(self, command: str, *args, **kwargs) -> Any:
        """
        Execute a Redis command with error handling.
        
        TODO 1: Get circuit breaker for this Redis instance
        TODO 2: Wrap _execute_command with circuit_breaker.call()
        TODO 3: Handle specific Redis errors:
            - ConnectionError: increment circuit breaker, raise RedisConnectionError
            - TimeoutError: increment circuit breaker, retry with backoff
            - ResponseError: log and raise as-is (application error)
        TODO 4: On success, return result
        """
        pass
    
    def pipeline(self) -> 'RedisPipeline':
        """
        Return a Redis pipeline for atomic operations.
        
        TODO 1: Create pipeline from underlying redis client
        TODO 2: Wrap pipeline execution with circuit breaker
        TODO 3: Implement context manager support (with statement)
        TODO 4: Ensure pipeline executes atomically even with failures
        """
        pass
```

**E. Language-Specific Hints**

1. **Python's `time.monotonic()`** is crucial for intervals unaffected by system clock changes. Use it for heartbeat intervals and job timeouts.

2. **Use `contextlib` context managers** for resource cleanup (Redis connections, file handles) to ensure resources are released even during exceptions.

3. **`tenacity` library** provides advanced retry patterns with exponential backoff, jitter, and conditional retrying that can replace manual retry logic.

4. **`pytz` or `zoneinfo` (Python 3.9+)** for timezone-aware scheduling. Always convert to UTC before storing in Redis.

5. **`asyncio` timeout context** for implementing watchdogs: `async with asyncio.timeout(seconds): await job_handler()`.

6. **Redis `SCRIPT EXISTS` and `EVALSHA`** for executing Lua scripts atomically, useful for complex recovery operations.

**F. Milestone Checkpoint for Error Handling**

After implementing error handling mechanisms, verify the system behaves correctly under failure conditions:

1. **Test Redis Connection Failure**:
   ```bash
   # Start system normally, then stop Redis
   $ sudo systemctl stop redis
   # Observe circuit breaker opening after 5 failed commands
   # Check logs for "Circuit redis is OPEN" messages
   # Restart Redis and verify system automatically recovers
   ```

2. **Test Worker Crash Recovery**:
   ```bash
   # Start worker processing jobs
   # Send SIGKILL to worker process
   # Wait 30 seconds for janitor scan
   # Verify orphaned jobs are re-queued with incremented attempt count
   # Check dashboard shows job as "recovered" not "lost"
   ```

3. **Test Clock Skew Handling**:
   ```bash
   # Set system clock forward 1 hour
   $ sudo date -s "+1 hour"
   # Observe scheduler logs using Redis TIME command, not system time
   # Verify scheduled jobs still fire at correct UTC time
   # Reset clock and verify catch-up logic enqueues missed jobs
   ```

**G. Debugging Tips for Error Scenarios**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Jobs disappearing from queue | Worker crash during processing, janitor not running | Check `processing` queues in Redis (`KEYS queue:processing:*`). Look for stuck jobs. Verify janitor process is running and scanning. | Start janitor, manually move jobs back with `RPOPLPUSH`. |
| All workers stuck idle | Deadlock in concurrency pool, all threads blocked | Check worker logs for timeout messages. Use thread dump (`import sys; sys._current_frames()`). Monitor CPU usage. | Reduce concurrency, implement watchdog that kills stuck workers, review job handlers for locking issues. |
| Scheduled jobs never fire | Scheduler process dead, cron expression error, timezone mismatch | Check scheduler logs for errors. Verify `Schedule.next_run_at` in Redis. Test cron expression with online validator. | Restart scheduler, fix cron expression, ensure timezone configuration matches schedule. |
| Redis memory constantly increasing | No TTL on keys, job history not cleaned, memory leak | Run `redis-cli info memory`. Check key TTLs with `TTL keyname`. Look for keys without expiration. | Implement TTL on all job keys, run janitor cleanup, increase Redis memory or implement LRU eviction. |
| Duplicate job execution | Worker crash after processing but before acknowledgment, network partition | Check job attempt count in metadata. Look for same job ID in both results and queue. Check for split-brain scenario logs. | Implement atomic processing+acknowledgment, use idempotency keys, ensure proper network partition handling. |
| Exponential backoff not working | Retry delay calculation error, jitter factor too large | Check `RetryManager` logs for scheduled retry times. Verify `BackoffCalculator.calculate_delay()` output. | Debug backoff calculation, ensure retry jobs are being enqueued to correct sorted set with proper score. |


## Testing Strategy

> **Milestone(s):** This section spans all five milestones, providing a comprehensive verification approach that ensures each component works correctly in isolation and as part of the integrated system.

A robust testing strategy is the linchpin of reliable distributed systems. Unlike monolithic applications where failures manifest predictably, distributed job processing systems exhibit emergent behaviors: race conditions surface only under specific timing, retry mechanisms can amplify errors exponentially, and resource leaks accumulate invisibly until a catastrophic failure. This section establishes a systematic approach to verification that moves beyond naive unit testing to encompass stateful integration testing, fault injection, and real-world scenario simulation.

### Testing Philosophy and Approach

**Mental Model: The Building Inspector's Checklist** — Think of testing this system as inspecting a newly constructed building. Unit tests are like checking individual materials (bricks, beams, wiring) for quality. Integration tests verify that systems work together (plumbing connects to fixtures, electrical circuits power lights). End-to-end tests simulate actual occupancy (people living in the building, using all systems simultaneously). Stress tests are like inviting a thousand people to a party to see if the floor holds. Each inspection type catches different classes of defects that would otherwise remain hidden until catastrophic failure.

Our testing philosophy embraces three core principles:

1. **Determinism in a Non-Deterministic World**: Distributed systems are inherently non-deterministic due to timing variations, network delays, and concurrent operations. Our tests must either eliminate non-determinism through careful isolation or embrace it through statistical verification of probabilistic guarantees.

2. **Stateful Verification Over Mock-Heavy Isolation**: While mocking external dependencies (Redis) simplifies unit tests, it creates false confidence. Redis operations have subtle atomicity and persistence characteristics that mocks often overlook. We favor integration tests with a real Redis instance in a controlled environment, supplemented by unit tests for pure business logic.

3. **Fault Injection as First-Class Testing**: The system's reliability emerges from how it handles failures, not how it behaves in perfect conditions. We systematically inject failures at component boundaries (network timeouts, Redis unavailability, process crashes) to verify recovery mechanisms work as designed.

The testing pyramid for this system has a unique shape due to its distributed nature:

![Testing Pyramid](./diagrams/system-component.svg)

**Testing Layers and Their Purposes**

| Layer | Scope | Key Techniques | Test Environment | Frequency |
|-------|-------|----------------|------------------|-----------|
| **Unit Tests** | Individual functions, classes, algorithms | Mock external dependencies, parameterized tests, property-based testing | Local development machine, CI pipeline | On every change |
| **Integration Tests** | Interactions between 2-3 components (e.g., QueueManager + Redis) | Real Redis instance, containerized dependencies, transactional rollback | CI pipeline with Docker Compose | On every change |
| **End-to-End Tests** | Full system workflows across all components | Multiple processes, real network communication, failure injection | Dedicated test environment | Daily or per milestone |
| **Chaos Tests** | System behavior under injected failures | Network partitioning, process killing, Redis failure simulation | Isolated test cluster | Weekly or major releases |
| **Performance Tests** | System behavior under load | Load generation, metric collection, scalability verification | Production-like environment | Monthly or architectural changes |

**Test Environment Architecture**

To support this multi-layered approach, we establish a standardized test environment with the following characteristics:

1. **Redis Isolation**: Each test class spins up a dedicated Redis instance on a unique port using Docker or Redis's built-in `--port` option. This prevents test interference and allows parallel test execution.

2. **Transactional Rollback**: Integration tests wrap operations in Redis transactions or use the `FLUSHDB` command between tests to ensure clean state. For end-to-end tests, we use separate Redis databases selected via the `SELECT` command.

3. **Time Control**: Scheduling and retry tests require precise control over time. We implement a mockable clock abstraction that allows tests to simulate time jumps, freeze time at specific moments, or accelerate time for testing long delays.

4. **Process Management**: End-to-end tests that involve multiple worker processes use Python's `multiprocessing` module with careful cleanup to prevent orphaned processes. Each test starts fresh worker processes and verifies they terminate correctly.

> **Key Insight**: The most valuable tests for distributed systems are **integration tests that verify distributed invariants** — properties that must hold true across multiple components despite concurrent operations. Examples include "every job enqueued is eventually processed exactly once" or "the sum of jobs across all queues equals the total jobs in the system."

**ADR: Testing Strategy for Redis Dependencies**

> **Decision: Real Redis in Integration Tests with Clean State Isolation**
> - **Context**: Redis operations have subtle behaviors (atomicity, expiration, transaction semantics) that are difficult to mock accurately. Mock-based tests provide false confidence, while requiring a Redis server adds complexity to the test environment.
> - **Options Considered**:
>   1. **Pure Mocking**: Mock all Redis calls with an in-memory simulation
>   2. **Fake Redis Server**: Use a lightweight Redis-compatible server like `fakeredis`
>   3. **Real Redis with Isolation**: Run actual Redis server with isolated databases/ports
> - **Decision**: Use real Redis server with isolated state for integration tests, supplemented by mocking for unit tests of non-Redis logic.
> - **Rationale**: Only real Redis guarantees correct behavior of atomic operations, Lua scripting, and persistence semantics. The risk of mock inaccuracy outweighs the setup complexity, especially with containerization making Redis trivial to run. We mitigate complexity through reusable test fixtures.
> - **Consequences**: Tests require Redis installation or Docker, increasing CI setup complexity slightly. Tests run slower than with mocks but catch subtle bugs that mocks would miss. We maintain a fast unit test suite for business logic to keep overall test time reasonable.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Pure Mocking | Fast, no external dependencies | Misses Redis-specific behaviors, false confidence | No |
| Fake Redis Server | Redis-like API, faster than real Redis | May differ from real Redis in edge cases | No |
| Real Redis with Isolation | Accurate behavior, tests real interactions | Requires Redis installation, slower tests | **Yes** |

**Common Testing Patterns**

We employ several recurring patterns across test suites:

1. **Round-Trip Verification**: Enqueue a job, dequeue it, verify payload preservation — the most basic integration test that catches serialization bugs.

2. **Concurrent Assertion**: Start multiple workers simultaneously, enqueue jobs, verify all jobs are processed exactly once despite concurrent access.

3. **State Machine Validation**: For each job, verify it progresses through the correct states (`PENDING` → `ACTIVE` → `COMPLETED`) and that invalid transitions are prevented.

4. **Timeout and Cancellation**: Verify that jobs timeout correctly, workers handle SIGTERM gracefully, and scheduled jobs can be canceled mid-execution.

5. **Recovery Simulation**: Simulate worker crashes by killing processes mid-job, then verify the janitor process re-queues the orphaned jobs.

### Milestone Implementation Checkpoints

Each milestone includes specific verification checkpoints that validate core functionality. These checkpoints serve as progress indicators and quality gates before proceeding to the next milestone.

**Milestone 1: Job Queue Core**

> **Mental Model: The Conveyor Belt Factory Acceptance Test** — Imagine testing a newly installed conveyor belt system. You'd verify packages can be placed on the belt (enqueue), move along it in order (FIFO), be inspected without removal (peek), and be redirected to different lanes (multiple queues). You'd also check safety mechanisms like weight limits (payload size validation) and emergency stops (queue deletion).

| Checkpoint | Test Description | Verification Method | Expected Outcome |
|------------|------------------|---------------------|------------------|
| **CP1.1** | Job serialization round-trip | Unit test: `Job.serialize()` → `Job.deserialize()` | Original job equals deserialized job with all fields preserved |
| **CP1.2** | Basic enqueue/dequeue FIFO order | Integration test with Redis: Enqueue 3 jobs, dequeue 3 times | Jobs retrieved in same order as enqueued |
| **CP1.3** | Multiple queue support | Integration test: Create 2 queues, enqueue jobs to each | Jobs only retrieved from queue they were enqueued to |
| **CP1.4** | Payload size validation | Unit test: Attempt to enqueue job > 1MB | `ValidationError` raised before Redis interaction |
| **CP1.5** | Queue inspection APIs | Integration test: Enqueue jobs, call `QueueManager.peek_queue()` and `get_queue_length()` | Peek returns correct jobs without removal, length matches count |
| **CP1.6** | Job ID uniqueness | Integration test: Enqueue 1000 jobs concurrently | All job IDs are unique (no collisions) |
| **CP1.7** | Atomic bulk enqueue | Integration test: Enqueue 10 jobs via `bulk_enqueue()`, kill Redis mid-operation | Either all jobs enqueued or none (no partial state) |

**Implementation Verification for Milestone 1**

After implementing Milestone 1, run the following verification suite:

1. **Unit Test Suite**: Execute all unit tests for `Job` class serialization, validation logic, and `QueueManager` business logic (excluding Redis operations).
   
2. **Integration Test Suite**: Run the Redis-dependent tests which require a running Redis instance. These tests should pass with a clean Redis installation.

3. **Manual Verification Script**: Execute a simple end-to-end workflow:
   ```bash
   # Start Redis locally
   redis-server --port 6379
   
   # Run the verification script
   python tests/milestone1_verification.py
   ```

The verification script should:
- Enqueue 100 jobs with varying payloads
- Verify all jobs are stored in Redis with correct metadata
- Peek at the queue to verify jobs are in FIFO order
- Attempt to enqueue an oversized payload and confirm it's rejected
- Display queue metrics showing correct job counts

**Common Testing Pitfalls for Milestone 1**

⚠️ **Pitfall: Testing with Production Redis**
- **Description**: Accidentally running tests against a production Redis instance, corrupting real job data.
- **Why Wrong**: Tests should never affect production data. This can cause data loss and service disruption.
- **Fix**: Always use isolated Redis instances (different port or database) with automatic cleanup. Implement safety checks that refuse to connect to production Redis based on hostname or explicit test flag.

⚠️ **Pitfall: Assuming Serialization is Trivial**
- **Description**: Writing naive `json.dumps()`/`json.loads()` tests that miss edge cases like datetime objects or custom classes.
- **Why Wrong**: Job payloads often contain complex Python types that don't serialize to JSON by default.
- **Fix**: Use parameterized tests with diverse payload types: strings, numbers, lists, dicts, datetimes, Decimal, and custom objects with `to_dict()` methods. Verify round-trip equivalence, not just string equality.

**Milestone 2: Worker Process**

> **Mental Model: The Restaurant Kitchen Staff Evaluation** — When evaluating kitchen staff, you'd verify each cook can: 1) Receive orders from the ticket rail (poll queues), 2) Prepare dishes according to recipes (job handlers), 3) Work on multiple orders simultaneously (concurrency), 4) Clean up properly when closing (graceful shutdown), and 5) Signal when overwhelmed (heartbeat failure).

| Checkpoint | Test Description | Verification Method | Expected Outcome |
|------------|------------------|---------------------|------------------|
| **CP2.1** | Worker polling from multiple queues | Integration test: Start worker, enqueue jobs to different priority queues | Worker processes jobs from higher-priority queues first |
| **CP2.2** | Job handler dispatch | Unit test: Register multiple handlers, dispatch jobs | Correct handler called with correct arguments |
| **CP2.3** | Graceful shutdown | Integration test: Send SIGTERM to worker while job is processing | Worker completes current job before exiting, leaves other jobs in queue |
| **CP2.4** | Worker heartbeat | Integration test: Start worker, kill process abruptly, wait | Heartbeat key expires, stale worker detection triggers |
| **CP2.5** | Concurrent job processing | Integration test: Worker with concurrency=3, enqueue 5 long-running jobs | 3 jobs process simultaneously, then remaining 2 |
| **CP2.6** | Job timeout enforcement | Integration test: Enqueue job with 1-second timeout that sleeps for 5 seconds | Job killed after 1 second, marked as failed with timeout error |
| **CP2.7** | Worker pause/resume | Integration test: Pause worker, enqueue jobs, resume worker | No jobs processed while paused, all jobs processed after resume |

**Implementation Verification for Milestone 2**

After implementing Milestone 2, run the following verification:

1. **Worker Lifecycle Test**: Start a worker, enqueue jobs, verify processing, then stop the worker gracefully.
   
2. **Concurrency Stress Test**: Start worker with concurrency=5, enqueue 100 quick jobs, verify all processed correctly with no race conditions.

3. **Failure Recovery Test**: Start worker processing a long job, kill the worker process abruptly (SIGKILL), then start a janitor process to verify orphaned job detection and re-queuing.

The verification should produce metrics showing:
- All jobs processed exactly once
- No jobs stuck in processing state after worker shutdown
- Heartbeat updates occurring at correct intervals
- Timeout errors properly recorded for long-running jobs

**Milestone 3: Retry & Error Handling**

> **Mental Model: The Customer Service Escalation Protocol Test** — Testing a customer service escalation system involves: 1) Verifying initial complaints are logged (job failure recording), 2) Ensuring follow-ups happen at increasing intervals (exponential backoff), 3) Confirming unresolved issues escalate to management (dead letter queue), and 4) Validating managers can review and re-open cases (manual retry).

| Checkpoint | Test Description | Verification Method | Expected Outcome |
|------------|------------------|---------------------|------------------|
| **CP3.1** | Exponential backoff calculation | Unit test: `BackoffCalculator.calculate_delay()` with various attempt numbers | Delays follow 2^attempt pattern within configured bounds |
| **CP3.2** | Retry scheduling | Integration test: Fail a job, verify it appears in Redis sorted set with correct score | Job scheduled for retry at correct future timestamp |
| **CP3.3** | Retry limit enforcement | Integration test: Job with max_retries=3 fails repeatedly | After 3 failures, job moves to dead letter queue, not retried again |
| **CP3.4** | Error serialization | Unit test: Job fails with exception, `Job.record_error()` called | Error details (type, message, traceback) stored in job metadata |
| **CP3.5** | Dead letter queue storage | Integration test: Move job to dead letter queue, retrieve via API | Job retrievable with all error history intact |
| **CP3.6** | Manual retry from dead letter | Integration test: Retry job from dead letter queue | Job re-enqueued with fresh attempt counter, processes successfully |
| **CP3.7** | Jitter in backoff | Statistical test: Calculate 1000 retry delays with jitter enabled | Delays vary within ±jitter_factor range, not perfectly predictable |

**Implementation Verification for Milestone 3**

Retry system verification requires time manipulation to avoid actual waiting. We use a mock clock to accelerate time:

1. **Retry Flow Test**: Configure a job to fail, verify it's scheduled for retry, advance mock clock to retry time, verify job executes again.

2. **Dead Letter Integration Test**: Exhaust retries for a job, verify it moves to dead letter queue, retrieve it via API, manually retry it, verify success.

3. **Error Context Preservation Test**: Fail a job with a complex exception containing nested attributes, verify all error context is preserved through serialization and available in dead letter queue.

Key metrics to verify:
- Retry delays follow exponential sequence with jitter
- No jobs exceed their max_retries limit
- Error count increments correctly with each failure
- Dead letter queue grows only when retries exhausted

**Milestone 4: Scheduling & Cron**

> **Mental Model: The Calendar Alarm System Validation** — Testing a calendar alarm system involves: 1) Setting one-time alarms (delayed jobs), 2) Configuring recurring meetings (cron jobs), 3) Ensuring timezone conversions work correctly, 4) Preventing duplicate alerts for the same event (uniqueness constraints), and 5) Verifying missed alarms fire on system restart (catch-up logic).

| Checkpoint | Test Description | Verification Method | Expected Outcome |
|------------|------------------|---------------------|------------------|
| **CP4.1** | Delayed job execution | Integration test with time mock: Schedule job for future, advance clock | Job enqueues only when scheduled time arrives, not before |
| **CP4.2** | Cron expression parsing | Unit test: Parse various cron strings, calculate next run times | Next run times match expected schedule |
| **CP4.3** | Recurring job execution | Integration test: Schedule cron job, advance clock through multiple periods | Job enqueues at each scheduled interval |
| **CP4.4** | Timezone handling | Integration test: Schedule job in non-UTC timezone, verify enqueue times | Job enqueues according to specified timezone, not system time |
| **CP4.5** | Uniqueness constraints | Integration test: Schedule unique job, attempt to enqueue duplicate within window | Duplicate job skipped (status=SKIPPED), not enqueued |
| **CP4.6** | Scheduler catch-up | Integration test: Disable scheduler, miss scheduled time, re-enable scheduler | Missed job enqueued immediately on scheduler restart |
| **CP4.7** | Schedule pause/resume | Integration test: Pause schedule, advance time, resume schedule | No jobs enqueued while paused, resumes correctly |

**Implementation Verification for Milestone 4**

Scheduler verification requires careful time manipulation and Redis state inspection:

1. **Cron Schedule Test**: Schedule a job with cron expression "*/2 * * * *" (every 2 minutes), advance mock clock 10 minutes, verify job enqueues 5 times.

2. **Timezone Boundary Test**: Schedule a job for 2:30 AM in "America/New_York" timezone, verify correct UTC conversion accounting for daylight saving time if applicable.

3. **Uniqueness Window Test**: Schedule unique job with 60-second window, enqueue it, attempt immediate re-enqueue, verify second attempt is skipped.

The verification should confirm:
- Scheduled jobs appear in Redis sorted set with correct timestamps
- Cron expressions calculate next run times accurately
- Timezone conversions handle daylight saving transitions
- Unique constraints prevent duplicates within specified windows

**Milestone 5: Monitoring & Dashboard**

> **Mental Model: The Air Traffic Control System Certification** — Certifying an air traffic control dashboard involves: 1) Verifying all aircraft appear on radar (real-time updates), 2) Ensuring alert thresholds trigger warnings (alerting), 3) Validating historical flight data is searchable (job history), 4) Confirming operators can intervene when needed (manual controls), and 5) Testing system doesn't overload under heavy traffic (dashboard performance).

| Checkpoint | Test Description | Verification Method | Expected Outcome |
|------------|------------------|---------------------|------------------|
| **CP5.1** | Real-time metrics updates | Integration test: Enqueue/process jobs while dashboard connected via WebSocket/SSE | Dashboard updates within 1 second of state change |
| **CP5.2** | Queue metrics calculation | Unit test: Simulate job flows, calculate `QueueMetrics` | All counts and rates computed correctly |
| **CP5.3** | Alert rule evaluation | Integration test: Configure alert rule, trigger condition | Alert fires when threshold exceeded, respects cooldown |
| **CP5.4** | Job history search | Integration test: Create jobs with various attributes, search with filters | Only matching jobs returned, pagination works |
| **CP5.5** | Dashboard API rate limiting | Load test: Generate many concurrent dashboard requests | Requests throttled, Redis not overwhelmed |
| **CP5.6** | Manual job controls | Integration test: Retry/delete dead letter jobs via dashboard API | Operations succeed, system state updates accordingly |
| **CP5.7** | Metrics aggregation | Integration test: Run aggregator over sample data | Historical metrics computed, stored efficiently |

**Implementation Verification for Milestone 5**

Dashboard verification involves both API testing and UI simulation:

1. **Real-time Flow Test**: Start dashboard, enqueue jobs via API, verify WebSocket/SSE stream shows updates within expected latency.

2. **Alerting Integration Test**: Configure alert for queue depth > 10, enqueue 11 jobs, verify alert triggers, acknowledge alert, verify alert state updates.

3. **Search Performance Test**: Insert 10,000 jobs into history, search with complex filters, verify response time < 500ms with pagination.

Verification should produce:
- Dashboard displaying correct queue depths and worker status
- Alert history showing triggered alerts with correct metadata
- Job search returning accurate results with pagination
- Manual retry operations successfully re-queuing dead letter jobs

### Integration and End-to-End Tests

Integration tests verify interactions between components, while end-to-end tests simulate real-world usage scenarios across the entire system. These tests catch emergent behaviors that unit tests cannot.

**Integration Test Architecture**

Our integration test suite follows a layered approach:

1. **Component Integration Tests**: Test interactions between 2-3 components (e.g., QueueManager + Worker + Redis).

2. **Workflow Integration Tests**: Test complete workflows (enqueue → process → retry → dead letter) across multiple components.

3. **Failure Scenario Tests**: Inject failures at specific points (network timeout, Redis crash, worker SIGKILL) and verify recovery.

4. **Concurrency Tests**: Run multiple producers and consumers simultaneously to detect race conditions.

**Key Integration Test Scenarios**

| Test Scenario | Components Involved | Key Verification | Failure Injection Points |
|---------------|---------------------|------------------|--------------------------|
| **Happy Path Flow** | Producer → QueueManager → Redis → Worker → Result Store | Job processed exactly once, result stored | None (baseline verification) |
| **Worker Crash Recovery** | Producer → Worker (crashed) → Janitor → New Worker | Orphaned job re-queued, processed exactly once | Kill worker mid-job execution |
| **Redis Connection Loss** | All components with Redis circuit breaker | Operations fail fast, resume after Redis recovery | Simulate Redis network partition |
| **Concurrent Priority Queues** | Multiple producers → QueueManager → Multiple workers | Higher-priority jobs processed first despite concurrency | Vary job durations and arrival times |
| **Scheduler Catch-up** | Scheduler (disabled) → Time advance → Scheduler (enabled) | Missed jobs enqueued immediately, future jobs scheduled correctly | Disable scheduler during scheduled execution time |
| **Dead Letter Manual Retry** | Failed job → DeadLetterQueue → Dashboard API → QueueManager → Worker | Manually retried job succeeds, removed from dead letter | None (manual operation verification) |

**End-to-End Test Environment**

End-to-end tests require a complete running system. We use Docker Compose to orchestrate:

```yaml
# test-environment/docker-compose.yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"  # Different port to avoid conflict
  
  worker-1:
    build: .
    command: ["worker", "--queues", "high,low", "--concurrency", "2"]
    depends_on: [redis]
  
  worker-2:
    build: .
    command: ["worker", "--queues", "medium", "--concurrency", "1"]
    depends_on: [redis]
  
  scheduler:
    build: .
    command: ["scheduler", "--polling-interval", "5"]
    depends_on: [redis]
  
  dashboard:
    build: .
    command: ["dashboard", "--port", "8080"]
    ports:
      - "8080:8080"
    depends_on: [redis]
```

**End-to-End Test Execution Pattern**

Each end-to-end test follows this pattern:

1. **Setup**: Start all services via Docker Compose, wait for health checks
2. **Execution**: Perform test actions via API calls or CLI
3. **Verification**: Query dashboard APIs, inspect Redis state, verify outcomes
4. **Teardown**: Capture logs, stop services, clean up resources

**Sample End-to-End Test: Scheduled Job with Retry**

This test verifies the complete lifecycle of a scheduled job that fails and retries:

1. **Schedule a recurring job** via scheduler API to run every minute
2. **Advance system time** by 5 minutes (using time mock in test environment)
3. **Verify** 5 job instances were enqueued
4. **Configure first 2 executions to fail** by registering a failing handler
5. **Verify** failed jobs are scheduled for retry with exponential backoff
6. **Advance time** to retry times
7. **Configure retries to succeed** by changing handler
8. **Verify** all jobs eventually complete successfully
9. **Check dashboard** shows correct metrics: 5 jobs completed, 2 initial failures, 3 retries

**Concurrency and Race Condition Testing**

Distributed systems exhibit subtle race conditions. We employ specific techniques to detect them:

| Technique | Implementation | What It Detects |
|-----------|----------------|-----------------|
| **Deterministic Simulation** | Use controlled thread scheduling with barriers | Race conditions in multi-threaded code |
| **Statistical Assertion** | Run test 1000 times, verify invariant always holds | Probabilistic race conditions |
| **Stress Testing** | Run system under heavy load with random delays | Resource leaks, deadlocks |
| **Linearizability Checking** | Record operation history, verify appears sequential | Concurrent data structure bugs |

**Example: Testing Job Uniqueness Under Concurrency**

```python
# Pseudocode for testing uniqueness constraint race condition
def test_unique_job_concurrent_enqueue():
    # Schedule a unique job with 60-second window
    schedule_id = scheduler.schedule_recurring_job(
        job_type="unique_task",
        cron_expression="* * * * *",
        unique_key="task_1",
        unique_window_seconds=60
    )
    
    # Simulate concurrent scheduler invocations (e.g., multiple scheduler processes)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scheduler._process_schedule, schedule) 
                  for _ in range(5)]
        results = [f.result() for f in futures]
    
    # Only one should succeed in enqueuing the job
    assert sum(1 for success, _ in results if success) == 1
```

**Performance and Load Testing**

While not part of the core milestones, performance testing validates the architecture scales as expected:

1. **Throughput Test**: Measure jobs processed per second with varying worker counts
2. **Latency Test**: Measure enqueue-to-completion time under different loads
3. **Redis Memory Test**: Monitor Redis memory growth with long-running job history
4. **Dashboard Scalability Test**: Verify dashboard remains responsive with 10k+ jobs in history

**Continuous Integration Pipeline**

The CI pipeline executes tests in this order:

```mermaid
graph LR
    A[Code Commit] --> B[Unit Tests<br/>Fast, no Redis]
    B --> C[Integration Tests<br/>With Redis, slower]
    C --> D[End-to-End Tests<br/>Docker Compose, slowest]
    D --> E[Performance Tests<br/>Optional, on schedule]
    E --> F[Release Candidate]
```

**Test Data Management**

We maintain realistic test data for comprehensive testing:

| Data Set | Purpose | Characteristics |
|----------|---------|-----------------|
| **Job Payload Variants** | Test serialization edge cases | Large payloads, nested structures, binary data, special characters |
| **Cron Expression Suite** | Test scheduler correctness | Simple intervals, complex ranges, timezone transitions, invalid expressions |
| **Error Type Catalog** | Test error handling | Network errors, application exceptions, timeout exceptions, memory errors |
| **Load Profiles** | Performance testing | Steady load, burst load, increasing load, spike load |

**Test Maintenance Strategy**

Tests require maintenance as the system evolves:

1. **Flaky Test Detection**: CI tracks test failure rates, automatically quarantines flaky tests
2. **Test Data Generation**: Use property-based testing to generate edge cases automatically
3. **Golden Master Testing**: For complex outputs (dashboard UI), maintain "golden master" snapshots
4. **Test Coverage Tracking**: Monitor coverage of critical paths (error handling, recovery logic)

> **Key Principle**: The most valuable tests are those that fail when the system behaves incorrectly but pass when it behaves correctly. This seems obvious but is often violated by tests that pass for the wrong reasons (e.g., mocks that don't simulate real behavior) or fail for irrelevant reasons (e.g., timing sensitivities in test environment).

### Implementation Guidance

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Testing Framework | `pytest` with `pytest-asyncio` | `pytest` with custom distributed test runner |
| Redis for Tests | Docker Redis container | Redis cluster for distributed testing |
| Mock Clock | `freezegun` library | Custom time abstraction with dependency injection |
| Concurrency Testing | `threading`/`multiprocessing` | `asyncio` with controlled event loop scheduling |
| Dashboard Testing | `requests` for API testing | `playwright` for full browser automation |
| Load Testing | Locust.io for HTTP load testing | Custom job enqueue load generator |

**B. Recommended File/Module Structure**

```
project-root/
├── tests/                           # All test code
│   ├── unit/                        # Unit tests (no external dependencies)
│   │   ├── test_job.py              # Job class tests
│   │   ├── test_backoff_calculator.py
│   │   └── test_cron_parser.py
│   ├── integration/                 # Integration tests (with Redis)
│   │   ├── conftest.py              # pytest fixtures (Redis setup)
│   │   ├── test_queue_manager.py    # QueueManager + Redis tests
│   │   ├── test_worker_integration.py
│   │   └── test_retry_manager.py
│   ├── e2e/                         # End-to-end tests
│   │   ├── docker-compose.yaml      # Test environment
│   │   ├── test_happy_path.py       # Full system workflow
│   │   ├── test_failure_recovery.py
│   │   └── test_scheduler_e2e.py
│   ├── fixtures/                    # Test data fixtures
│   │   ├── job_payloads.json
│   │   ├── cron_expressions.yaml
│   │   └── error_catalog.py
│   └── utils/                       # Test utilities
│       ├── redis_helpers.py         # Redis test helpers
│       ├── time_helpers.py          # Mock time utilities
│       └── process_helpers.py       # Worker process management
├── src/                             # Main source code
└── scripts/
    └── run_tests.sh                 # Test execution script
```

**C. Infrastructure Starter Code**

Here's a complete, reusable test infrastructure for Redis integration tests:

```python
# tests/utils/redis_helpers.py
"""
Redis test helpers for isolated, parallel test execution.
"""
import redis
import random
import threading
from contextlib import contextmanager
from typing import Optional, Generator

class RedisTestContext:
    """Manages isolated Redis instances for parallel testing."""
    
    def __init__(self, port: int = 6379, db: int = 0):
        self.port = port
        self.db = db
        self._client: Optional[redis.Redis] = None
        self._lock = threading.Lock()
        
    @property
    def client(self) -> redis.Redis:
        """Lazy initialization of Redis client."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = redis.Redis(
                        host='localhost',
                        port=self.port,
                        db=self.db,
                        decode_responses=True
                    )
                    # Ping to ensure connection works
                    self._client.ping()
        return self._client
    
    def flush(self) -> None:
        """Clear all data from the current database."""
        if self._client:
            self._client.flushdb()
    
    def get_unique_key(self, prefix: str = "test") -> str:
        """Generate a unique Redis key for tests."""
        return f"{prefix}:{threading.get_ident()}:{random.randint(0, 1000000)}"
    
    @contextmanager
    def pipeline(self) -> Generator[redis.Pipeline, None, None]:
        """Context manager for Redis pipeline with automatic execution."""
        pipe = self.client.pipeline()
        try:
            yield pipe
            pipe.execute()
        except Exception:
            pipe.reset()
            raise
    
    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

# Global test context (can be overridden in tests)
_test_context: Optional[RedisTestContext] = None

def get_test_context() -> RedisTestContext:
    """Get or create the global test context."""
    global _test_context
    if _test_context is None:
        # Use random DB to allow parallel test execution
        _test_context = RedisTestContext(db=random.randint(10, 15))
    return _test_context

@contextmanager
def fresh_redis_context():
    """Context manager that provides a fresh Redis context and cleans up."""
    context = RedisTestContext(db=random.randint(100, 200))
    try:
        yield context
    finally:
        context.flush()
        context.close()
```

**D. Core Test Skeleton Code**

Here's a template for writing integration tests with proper setup/teardown:

```python
# tests/integration/test_queue_manager.py
"""
Integration tests for QueueManager with real Redis.
"""
import pytest
from datetime import datetime
from typing import List

from src.queue_manager import QueueManager
from src.job import Job, JobStatus
from tests.utils.redis_helpers import fresh_redis_context

class TestQueueManagerIntegration:
    """Test QueueManager with Redis integration."""
    
    @pytest.fixture
    def queue_manager(self):
        """Create a QueueManager with fresh Redis context."""
        with fresh_redis_context() as redis_context:
            # Create system config for testing
            config = SystemConfig(
                redis=RedisConfig(url=f"redis://localhost:{redis_context.port}/{redis_context.db}"),
                queues=[QueueConfig(name="test-queue", priority=1)],
                workers=[],
                max_payload_size=1024*1024,  # 1MB
                job_history_size=1000,
                retry_base_delay=1,
                retry_max_attempts=3
            )
            manager = QueueManager(config=config)
            yield manager
            # Cleanup happens automatically via context manager
    
    def test_enqueue_and_dequeue_fifo(self, queue_manager):
        """Test basic FIFO ordering of jobs."""
        # TODO 1: Create 3 test jobs with sequential IDs
        # TODO 2: Enqueue all jobs using queue_manager.enqueue_job()
        # TODO 3: Use Redis BRPOP directly to dequeue jobs in order
        # TODO 4: Verify jobs are returned in same order as enqueued
        # TODO 5: Check job metadata is preserved through round-trip
        pass
    
    def test_bulk_enqueue_atomicity(self, queue_manager):
        """Test that bulk enqueue is atomic (all or nothing)."""
        # TODO 1: Create 10 test jobs
        # TODO 2: Start a Redis transaction to monitor operations
        # TODO 3: Call queue_manager.bulk_enqueue() with the jobs
        # TODO 4: Simulate Redis failure mid-operation (mock redis.execute)
        # TODO 5: Verify either all jobs enqueued or none (check queue length)
        # TODO 6: Clean up test jobs if any were enqueued
        pass
    
    def test_queue_priority_weighted_polling(self, queue_manager):
        """Test that workers poll higher-priority queues more frequently."""
        # TODO 1: Configure multiple queues with different priorities
        # TODO 2: Enqueue equal numbers of jobs to each queue
        # TODO 3: Simulate worker polling using queue_manager's internal method
        # TODO 4: Record which queue is polled each time
        # TODO 5: Verify higher-priority queues are polled more often
        # TODO 6: Statistical test: run 1000 polls, check distribution matches weights
        pass
    
    def test_oversized_payload_rejection(self, queue_manager):
        """Test that jobs larger than max_payload_size are rejected."""
        # TODO 1: Create a job with payload exceeding max_payload_size
        # TODO 2: Attempt to enqueue the job
        # TODO 3: Verify ValidationError is raised
        # TODO 4: Verify no job was actually enqueued to Redis
        # TODO 5: Test boundary case: payload exactly at limit should succeed
        pass
    
    def test_queue_inspection_apis(self, queue_manager):
        """Test peek_queue and get_queue_length APIs."""
        # TODO 1: Enqueue 5 jobs to a test queue
        # TODO 2: Call queue_manager.get_queue_length() - should return 5
        # TODO 3: Call queue_manager.peek_queue(count=3) - should return first 3 jobs
        # TODO 4: Verify peek doesn't remove jobs (call get_queue_length again)
        # TODO 5: Dequeue one job, verify length becomes 4
        # TODO 6: Test edge cases: peek with count larger than queue, peek empty queue
        pass
```

**E. Language-Specific Hints for Python Testing**

1. **Use `pytest` Fixtures for Dependency Injection**:
   ```python
   @pytest.fixture
   def redis_client():
       client = redis.Redis()
       yield client
       client.flushdb()
       client.close()
   ```

2. **Mock Time with `freezegun`**:
   ```python
   from freezegun import freeze_time
   
   @freeze_time("2024-01-01 12:00:00")
   def test_scheduled_job():
       # Time is frozen at Jan 1, 2024
       schedule_job(run_at=datetime(2024, 1, 1, 12, 5, 0))
       # Advance time 5 minutes
       with freeze_time("2024-01-01 12:05:00"):
           process_due_jobs()  # Job should execute
   ```

3. **Test Concurrent Operations with `threading`**:
   ```python
   import threading
   
   def test_concurrent_enqueue():
       results = []
       lock = threading.Lock()
       
       def enqueue_job():
           job_id = queue_manager.enqueue_job(job)
           with lock:
               results.append(job_id)
       
       threads = [threading.Thread(target=enqueue_job) for _ in range(10)]
       for t in threads: t.start()
       for t in threads: t.join()
       
       assert len(set(results)) == 10  # All job IDs unique
   ```

4. **Use `pytest.mark.parametrize` for Comprehensive Input Testing**:
   ```python
   @pytest.mark.parametrize("cron_expression,expected_next_run", [
       ("*/15 * * * *", "2024-01-01 12:15:00"),
       ("0 2 * * *", "2024-01-02 02:00:00"),
       ("0 0 1 * *", "2024-02-01 00:00:00"),
   ])
   def test_cron_expressions(cron_expression, expected_next_run):
       # Test each cron expression
       pass
   ```

**F. Milestone Checkpoint Verification**

After completing each milestone, run these verification commands:

```bash
# Milestone 1: Job Queue Core
pytest tests/unit/test_job.py -v
pytest tests/integration/test_queue_manager.py -v
python scripts/verify_milestone1.py

# Milestone 2: Worker Process  
pytest tests/integration/test_worker_integration.py -v
python scripts/start_worker_test.py --run-verification

# Milestone 3: Retry & Error Handling
pytest tests/unit/test_backoff_calculator.py -v
pytest tests/integration/test_retry_manager.py -v
python scripts/test_retry_flow.py

# Milestone 4: Scheduling & Cron
pytest tests/unit/test_cron_parser.py -v
pytest tests/integration/test_scheduler.py -v
python scripts/test_scheduled_jobs.py --time-mock

# Milestone 5: Monitoring & Dashboard
pytest tests/integration/test_dashboard_api.py -v
python scripts/load_test_dashboard.py --users 10 --duration 30
```

**Expected Outcomes for Milestone Verification:**

| Milestone | Test Command | Success Indicators | Failure Symptoms |
|-----------|--------------|-------------------|------------------|
| **1** | `pytest tests/integration/test_queue_manager.py` | All tests pass, Redis keys created correctly | Jobs lost, serialization errors, queue priority not working |
| **2** | `python scripts/start_worker_test.py` | Worker processes jobs, heartbeats update, graceful shutdown works | Jobs stuck in processing, orphaned workers, timeout not enforced |
| **3** | `python scripts/test_retry_flow.py` | Failed jobs retry with increasing delays, dead letter queue receives exhausted jobs | Infinite retry loops, error context lost, manual retry fails |
| **4** | `python scripts/test_scheduled_jobs.py` | Jobs enqueue at correct times, cron expressions evaluated correctly, uniqueness constraints work | Timezone issues, missed schedules, duplicate jobs enqueued |
| **5** | `python scripts/load_test_dashboard.py` | Dashboard updates in real-time, alerts trigger correctly, search returns accurate results | Dashboard sluggish under load, alerts fire incorrectly, memory leaks |

**G. Debugging Tips for Common Test Failures**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Tests pass locally but fail in CI** | Redis version differences, timezone issues, or race conditions | Check Redis logs in CI, add debug logging to tests, run tests with `--tb=short` | Use same Redis version in CI, fix timezone assumptions, add retries for timing-sensitive tests |
| **Integration tests leave Redis data between tests** | Improper test isolation, missing `flushdb()` in teardown | Check Redis keys after test runs: `redis-cli keys '*'` | Use `fresh_redis_context()` fixture or ensure each test cleans up |
| **Worker tests hang indefinitely** | Deadlock in worker shutdown, missing timeout in test | Add debug logging to worker state machine, use test timeout decorator | Ensure worker threads/processes properly join, add test timeouts |
| **Time-sensitive tests are flaky** | System clock resolution, test execution time variability | Log actual vs expected timestamps, measure test execution time | Use mock time, add tolerance to timestamp comparisons |
| **Concurrent tests fail randomly** | Race condition not caught in single-threaded tests | Run test 100 times, add `pytest-repeat`, use thread sanitizer | Add proper synchronization, use atomic operations, fix race condition |

**Test Coverage Goals**

Aim for these coverage targets (measured via `pytest-cov`):

| Component | Statement Coverage | Branch Coverage | Critical Path Coverage |
|-----------|-------------------|-----------------|------------------------|
| `Job` class | 100% | 100% | 100% (serialization round-trip) |
| `QueueManager` | 95% | 90% | 100% (atomic operations) |
| `Worker` | 90% | 85% | 100% (graceful shutdown) |
| `RetryManager` | 95% | 90% | 100% (dead letter transition) |
| `Scheduler` | 90% | 85% | 100% (cron evaluation) |
| `Dashboard API` | 85% | 80% | 100% (alert triggering) |

> **Final Testing Principle**: Write tests that fail when the system is broken and pass when it's working. This seems obvious but requires careful thought about what "broken" means for each component. A test that passes because mocks are too permissive is worse than no test at all — it provides false confidence.


## Debugging Guide

> **Milestone(s):** This section spans all five milestones, providing systematic diagnostic procedures for common implementation issues that arise during development and operation. The debugging guide transforms vague symptoms into specific causes and actionable fixes.

Building a distributed background job processor involves coordinating multiple processes, network calls, and persistent state—a perfect environment for subtle bugs to emerge. When jobs aren't processing, workers are crashing, or retries aren't happening, systematic debugging is essential. This guide provides concrete procedures for identifying and resolving the most common issues, organized by symptom pattern and component.

Think of debugging this system as being a **detective investigating a crime scene**. You arrive at the scene (a production issue), examine the evidence (logs, Redis data, metrics), look for patterns and anomalies, reconstruct timelines, and identify the perpetrator (the bug). Each piece of evidence tells part of the story, and your job is to connect them logically.

### Common Bugs and Their Fixes

The following table categorizes common implementation bugs by their observable symptoms, underlying causes, and specific fixes. Each entry follows the pattern: you observe a symptom, investigate potential causes, and apply targeted corrections.

| Symptom | Likely Cause | Investigation Steps | Fix |
|---------|--------------|---------------------|-----|
| **Jobs stuck in queue, workers idle** | 1. Workers polling wrong queue names<br>2. Queue priority weights misconfigured<br>3. Redis connection failures in workers<br>4. Workers paused or in shutdown state | 1. Check worker logs for connection errors<br>2. Verify `WorkerConfig.queues` matches queue names<br>3. Use `redis-cli LLEN queue:{name}` to confirm jobs exist<br>4. Check worker heartbeat status in Redis | 1. Ensure queue names in config match those used for enqueue<br>2. Reconfigure workers with correct queue list<br>3. Implement connection retry logic in `RedisClient`<br>4. Restart workers with `SIGTERM` then fresh start |
| **Jobs disappearing without execution** | 1. Jobs moved to processing queue but worker crashed<br>2. `BRPOPLPUSH` without backup processing queue cleanup<br>3. Job validation failing silently during deserialization<br>4. Race condition in multi-worker dequeuing | 1. Check `queue:{name}:processing` lists in Redis<br>2. Verify Janitor process is running<br>3. Examine worker crash logs for segmentation faults<br>4. Check for `ValidationError` in enqueue logs | 1. Implement and run `Janitor` to requeue stale processing jobs<br>2. Use atomic operations for dequeue with proper error handling<br>3. Add detailed logging in `Job.deserialize()`<br>4. Use Redis transactions with `WATCH` for critical operations |
| **Exponential retries happening too fast or not at all** | 1. Incorrect `attempt` counter in retry logic<br>2. Base delay miscalculation (seconds vs milliseconds)<br>3. Retry sorted set score using wrong timestamp format<br>4. Retry scheduler not polling frequently enough | 1. Inspect `job:{id}:attempts` key in Redis<br>2. Check `retry:{queue}` sorted set scores with `ZRANGE`<br>3. Verify `BackoffCalculator.calculate_delay()` implementation<br>4. Check scheduler polling interval vs retry delays | 1. Ensure `Job.attempts` increments atomically on each failure<br>2. Use Unix timestamps (seconds) for sorted set scores<br>3. Debug formula: `base_delay * (2 ** (attempt - 1))`<br>4. Configure scheduler to poll more frequently than minimum retry delay |
| **Scheduled jobs not enqueuing at expected times** | 1. Timezone mismatch between scheduler and cron expression<br>2. `next_run_at` calculation error for cron expressions<br>3. Uniqueness constraint preventing enqueue<br>4. Scheduler process stopped or crashing | 1. Compare scheduler timezone with schedule timezone field<br>2. Check `schedule:{id}` hash for `next_run_at` value<br>3. Look for `SKIPPED` status in schedule history<br>4. Verify scheduler heartbeat in Redis | 1. Use UTC consistently or store timezone with each schedule<br>2. Test cron parsing with `croniter` library<br>3. Adjust `unique_window_seconds` or disable uniqueness<br>4. Implement scheduler supervisor with automatic restart |
| **Memory usage growing unbounded in Redis** | 1. Missing TTL on job data after completion<br>2. Dead letter queue never purged<br>3. Job history accumulating without archival<br>4. Large payloads stored without size limits | 1. Run `redis-cli INFO memory` to check memory usage<br>2. Scan for keys without TTL: `redis-cli --scan | while read k; do echo "$k: $(redis-cli TTL $k)"; done`<br>3. Check `job:{id}:result` keys still present<br>4. Verify `max_payload_size` enforcement | 1. Set TTL on all temporary keys (job results, heartbeats)<br>2. Implement Janitor cleanup of old dead letter jobs<br>3. Add job history archival to external storage<br>4. Enforce payload size validation in `QueueManager._validate_job()` |
| **Worker crashes without logging errors** | 1. Unhandled exception in job handler<br>2. Memory leak in long-running worker process<br>3. Signal handler conflicts with threading<br>4. Job timeout killing process instead of just thread | 1. Check system logs (`dmesg`, `journalctl`) for OOM killer<br>2. Add `try...except` with logging in worker main loop<br>3. Test with reduced concurrency<br>4. Monitor Python GC with `gc.get_count()` | 1. Wrap job execution in comprehensive error handler<br>2. Implement process-based isolation for memory-intensive jobs<br>3. Use `signal.signal()` carefully, preserve original handlers<br>4. Use thread/process timeout rather than process signal |
| **Dashboard shows stale or incorrect metrics** | 1. Metrics aggregation interval too long<br>2. Race condition between metric collection and job state updates<br>3. Redis memory eviction deleting metric keys<br>4. WebSocket/SSE connection dropping silently | 1. Compare dashboard metrics with direct Redis queries<br>2. Check `metrics:queue:{name}:depth` key updates<br>3. Verify `MetricsAggregator` thread is running<br>4. Inspect browser console for WebSocket errors | 1. Reduce aggregation interval for near-real-time updates<br>2. Use atomic Redis operations for metric updates<br>3. Configure Redis maxmemory policy to `volatile-lru`<br>4. Implement WebSocket heartbeat and automatic reconnection |
| **Duplicate execution of the same job** | 1. Worker crash after processing but before acknowledgment<br>2. Race condition in unique job constraint<br>3. Network partition causing split-brain worker behavior<br>4. Retry of non-idempotent operation without deduplication | 1. Check for duplicate job IDs in completed jobs<br>2. Examine processing queue for same job multiple times<br>3. Look for network outage timestamps in logs<br>4. Verify idempotency key usage | 1. Implement idempotency keys for critical operations<br>2. Use Redis `SETNX` for unique job locks<br>3. Add partition tolerance logic with fencing tokens<br>4. Ensure job handlers are idempotent when possible |
| **Priority queue ordering not respected** | 1. Worker polling algorithm checking queues in fixed order<br>2. Priority weights incorrectly normalized<br>3. High-priority jobs stuck behind long-running low-priority jobs<br>4. Queue starvation due to constant high-priority influx | 1. Trace worker polling sequence in debug logs<br>2. Verify `QueueConfig.priority` values are positive integers<br>3. Check if worker concurrency is saturated<br>4. Monitor queue depths for imbalance | 1. Implement weighted random queue selection algorithm<br>2. Use priority values to calculate polling probabilities<br>3. Add preemption for critical high-priority jobs<br>4. Implement separate worker pools for different priority tiers |
| **Graceful shutdown not working** | 1. Signal handler not properly registered<br>2. Worker waiting indefinitely on blocking Redis call<br>3. Current job not interruptible<br>4. Thread pool not shutting down correctly | 1. Test with `kill -TERM <pid>` and observe behavior<br>2. Check if worker is stuck in `BRPOP` with no timeout<br>3. Verify job timeout configuration<br>4. Examine thread states during shutdown | 1. Set timeout on blocking Redis operations<br>2. Implement cooperative cancellation for long-running jobs<br>3. Use `ThreadPoolExecutor` with `shutdown(wait=True)`<br>4. Add shutdown flag checked in main loop |

> **Debugging Principle:** Always start with the simplest explanation. When jobs aren't processing, check that workers are actually running and connected to Redis before investigating complex race conditions. Use the scientific method: form a hypothesis, test it with a specific investigation, then refine.

#### State Transition Anomalies

Jobs and workers follow specific state machines (as shown in ![Job State Machine](./diagrams/job-state-machine.svg) and ![Worker Process State Machine](./diagrams/worker-state-machine.svg)). When states don't transition as expected, use this diagnostic table:

| Anomaly | Expected Transition | Actual State | Investigation | Resolution |
|---------|-------------------|--------------|---------------|------------|
| **Job stuck in ACTIVE** | ACTIVE → COMPLETED or FAILED | Remains ACTIVE for hours | 1. Check worker processing the job<br>2. Look for timeout configuration<br>3. Check for deadlock in job handler | 1. Implement job timeout with forced termination<br>2. Add deadlock detection in worker<br>3. Manual move to FAILED via dashboard |
| **Worker stuck in ShuttingDown** | ShuttingDown → Stopped | Worker hangs during shutdown | 1. Check if current job has cleanup phase<br>2. Look for thread/child process not exiting<br>3. Check for finally blocks with infinite loops | 1. Add shutdown timeout to force exit<br>2. Implement interrupt handling for job threads<br>3. Use process-based isolation for clean termination |
| **Job jumps from PENDING to DEAD_LETTER** | PENDING → ACTIVE → FAILED → RETRY_SCHEDULED → ... → DEAD_LETTER | Direct transition | 1. Check job `max_retries = 0`<br>2. Look for `RetryManager` bypass on specific errors<br>3. Check for manual move to dead letter | 1. Validate `max_retries` configuration<br>2. Review retry filter logic<br>3. Add audit log for manual operations |

### Diagnostic Tools and Techniques

Effective debugging requires the right tools and methodologies. This section outlines both built-in diagnostic capabilities and external tools that reveal system internals.

#### Built-in Diagnostic Endpoints

The monitoring dashboard (Milestone 5) should include diagnostic endpoints that expose internal state without requiring direct Redis access. These endpoints are invaluable for production debugging.

| Endpoint | Purpose | Data Format | Investigation Use |
|----------|---------|-------------|-------------------|
| `GET /api/debug/queue/{name}/contents` | Inspect jobs in a queue (including processing) | `{ "pending": [Job], "processing": [Job], "retry_scheduled": [Job] }` | Find specific stuck jobs, verify job ordering |
| `GET /api/debug/worker/{id}/current_state` | Detailed worker internal state | `Worker` object with private fields | See what job a worker is executing, its thread states |
| `GET /api/debug/job/{id}/trace` | Complete job lifecycle trace | `{ "events": [{"timestamp": "...", "state": "...", "detail": "..."}] }` | Reconstruct job history across retries and workers |
| `POST /api/debug/job/{id}/inspect` | Force re-evaluation of job state | `{ "current_redis_state": {...}, "suggested_actions": [...] }` | Diagnose corrupted job state, suggest repair actions |
| `GET /api/debug/redis/keys` | Scan Redis keys with pattern | `{ "pattern": "...", "keys": [...], "ttls": {...} }` | Find orphaned keys, check TTL expiration |
| `GET /api/debug/metrics/raw` | Raw metrics before aggregation | `{ "timestamp": "...", "counters": {...}, "gauges": {...} }` | Verify metric collection vs aggregation discrepancies |

#### Logging Strategy for Debugging

Strategic logging transforms opaque failures into diagnosable events. Implement these logging practices:

1. **Structured JSON Logging**: Each log entry includes:
   - `component`: `"queue_manager"`, `"worker"`, `"retry_manager"`, etc.
   - `job_id`: When relevant, the job identifier
   - `worker_id`: When relevant, the worker identifier
   - `queue`: Queue name involved
   - `event`: Specific event type (e.g., `"job_enqueued"`, `"job_failed"`, `"retry_scheduled"`)
   - `timestamp`: High-resolution timestamp
   - `level`: Log level (DEBUG, INFO, WARNING, ERROR)

2. **Correlation IDs**: Pass a `correlation_id` through the entire job lifecycle, from enqueue through all retries to completion. This allows tracing a job's journey across multiple log entries and components.

3. **Debug-Level Verbosity**: Enable detailed debug logging for specific components without overwhelming production logs. Example debug events:
   - Queue polling decisions (which queue was selected and why)
   - Redis command execution and timing
   - State machine transitions
   - Lock acquisition and release

4. **Health Check Logging**: Workers and schedulers should log periodic health status including:
   - Memory usage (`psutil.Process().memory_info().rss`)
   - Queue depths being monitored
   - Uptime and processed job counts
   - Redis connection latency

#### External Diagnostic Tools

| Tool | Purpose | Command Examples | Interpretation |
|------|---------|------------------|----------------|
| **redis-cli** | Direct Redis inspection | `redis-cli -h host -p port`<br>`INFO stats` for operations<br>`SLOWLOG GET 10` for slow commands | See command counts, connected clients, memory usage, slow operations |
| **netstat/ss** | Network connection diagnosis | `netstat -tulpn \| grep 6379`<br>`ss -t state established '( dport = :6379 )'` | Verify worker connections to Redis, check for connection leaks |
| **strace/dtrace** | System call tracing | `strace -p <worker_pid> -f -e poll,select,read,write`<br>`sudo dtrace -n 'syscall::read:entry /pid == 12345/ { printf("%s", arg0); }'` | See if process is blocked on I/O, identify hanging system calls |
| **py-spy** (Python) | CPU profiling and sampling | `py-spy top --pid <worker_pid>`<br>`py-spy dump --pid <worker_pid>` | Identify CPU hotspots in worker code, view real-time stack traces |
| **memory-profiler** (Python) | Memory usage tracking | Add `@profile` decorator to functions, run with `mprof` | Find memory leaks in job handlers or queue processing |
| **tcpdump** | Network traffic analysis | `sudo tcpdump -i any port 6379 -w redis.pcap`<br>`tcpflow -c port 6379` | Debug Redis protocol issues, inspect actual commands sent |

#### Interactive Debugging Session Protocol

When faced with a complex bug, follow this structured protocol:

1. **Reproduce the Issue**: 
   - Can you reproduce it consistently? If not, what are the conditions?
   - Create a minimal test case that triggers the bug.

2. **Gather Baseline Information**:
   - System time synchronization (`ntpq -p` or `timedatectl`)
   - Redis version and configuration (`redis-cli INFO server`)
   - Worker configurations and versions
   - Recent deployments or configuration changes

3. **Enable Enhanced Logging**:
   - Temporarily increase log level to DEBUG for affected components
   - Add specific trace logs around the suspected area
   - Capture Redis MONITOR output briefly (caution: performance impact)

4. **Isolate the Component**:
   - Does the issue occur with a single worker vs multiple?
   - Does it affect all queues or just one?
   - Is it time-dependent (only at certain times)?

5. **Check Data Consistency**:
   - Verify Redis data structures match expected schemas
   - Check for orphaned keys without corresponding metadata
   - Validate serialization/deserialization round-trips

6. **Form and Test Hypotheses**:
   - List possible causes in order of likelihood
   - Design tests to eliminate each hypothesis
   - Use the dashboard's debug endpoints to validate

> **Debugging Insight:** Time-related bugs (scheduling, retries, timeouts) often stem from clock skew between systems. Always verify time synchronization between workers, schedulers, and Redis servers. Use NTP and consider using Redis time (`TIME` command) as a reference rather than system time.

### Redis Data Inspection Guide

Redis stores the entire system state. Knowing how to interpret this data is crucial for debugging. This section provides a comprehensive guide to Redis key patterns and how to inspect them.

#### Key Naming Patterns

The system uses consistent key naming patterns. Understanding these patterns allows you to reconstruct system state from raw Redis data.

| Key Pattern | Data Type | Purpose | Example |
|-------------|-----------|---------|---------|
| `queue:{name}` | List | Primary job queue (FIFO) | `queue:default`, `queue:high_priority` |
| `queue:{name}:processing` | List | Jobs currently being processed (temporary) | `queue:default:processing` |
| `job:{id}` | Hash | Job metadata and payload | `job:01HMAK5V2ZJX4Q6W8P3B0D9F7G` |
| `job:{id}:result` | String (JSON) | Job execution result | `job:01HMAK5V2ZJX4Q6W8P3B0D9F7G:result` |
| `job:{id}:errors` | List | Error history for failed job attempts | `job:01HMAK5V2ZJX4Q6W8P3B0D9F7G:errors` |
| `retry:{queue}` | Sorted Set | Scheduled retries (score = execution timestamp) | `retry:default`, `retry:email` |
| `dead_letter:{queue}` | List | Permanently failed jobs | `dead_letter:default` |
| `schedule:{id}` | Hash | Schedule definition | `schedule:daily_report` |
| `schedule:{id}:history` | List | Schedule execution history | `schedule:daily_report:history` |
| `worker:{id}` | Hash | Worker heartbeat and status | `worker:host123-pid4567` |
| `worker:{id}:jobs` | List | Recent jobs processed by worker | `worker:host123-pid4567:jobs` |
| `metrics:queue:{name}:depth` | String | Current queue depth (updated periodically) | `metrics:queue:default:depth` |
| `metrics:queue:{name}:throughput` | Sorted Set | Timestamped throughput samples | `metrics:queue:default:throughput` |
| `alert:{id}` | Hash | Active alert information | `alert:queue_depth_high_01HMAK5V2Z` |
| `unique:{key}` | String | Unique job constraint lock | `unique:daily_report_2024_01_15` |
| `config:{component}` | Hash | Runtime configuration | `config:system`, `config:alert_rules` |

#### Inspection Commands and Scripts

Use these Redis commands to inspect system health and diagnose issues:

**Basic Queue Inspection:**
```bash
# Check queue lengths
redis-cli LLEN queue:default
redis-cli LLEN queue:high_priority

# Peek at next jobs in queue (without removing)
redis-cli LRANGE queue:default 0 4 | while read line; do echo "$line" | python3 -m json.tool; done

# Check processing queue for stuck jobs
redis-cli LLEN queue:default:processing
redis-cli LRANGE queue:default:processing 0 -1

# Check retry schedule
redis-cli ZRANGE retry:default 0 -1 WITHSCORES
```

**Job State Investigation:**
```bash
# Find all jobs in a particular state
redis-cli --scan --pattern 'job:*' | while read key; do 
  status=$(redis-cli HGET $key status);
  if [ "$status" = '"ACTIVE"' ]; then
    echo "Active job: $key";
    redis-cli HGET $key started_at;
  fi;
done

# Get complete job details
redis-cli HGETALL job:01HMAK5V2ZJX4Q6W8P3B0D9F7G

# Check job errors
redis-cli LRANGE job:01HMAK5V2ZJX4Q6W8P3B0D9F7G:errors 0 -1
```

**Worker Health Check:**
```bash
# List all active workers
redis-cli --scan --pattern 'worker:*' | grep -v ':jobs$' | while read key; do
  heartbeat=$(redis-cli HGET $key last_heartbeat);
  echo "$key: $heartbeat";
done

# Check for stale workers (heartbeat > 60 seconds ago)
current_time=$(date +%s)
redis-cli --scan --pattern 'worker:*' | grep -v ':jobs$' | while read key; do
  heartbeat=$(redis-cli HGET $key last_heartbeat);
  if [ ! -z "$heartbeat" ]; then
    age=$((current_time - heartbeat));
    if [ $age -gt 60 ]; then
      echo "STALE WORKER: $key (age: ${age}s)";
    fi;
  fi;
done
```

**Scheduler State Verification:**
```bash
# Check all schedules
redis-cli --scan --pattern 'schedule:*' | grep -v ':history$' | while read key; do
  next_run=$(redis-cli HGET $key next_run_at);
  echo "$key: $next_run";
done

# Check schedule history
redis-cli LRANGE schedule:daily_report:history 0 -1
```

**Memory Usage Analysis:**
```bash
# Check Redis memory info
redis-cli INFO memory

# Find largest keys
redis-cli --scan --pattern '*' | while read key; do
  size=$(redis-cli MEMORY USAGE $key);
  echo "$size $key";
done | sort -n | tail -20

# Check key TTLs
redis-cli --scan --pattern 'job:*:result' | while read key; do
  ttl=$(redis-cli TTL $key);
  if [ $ttl -eq -1 ]; then
    echo "NO TTL: $key";
  fi;
done
```

#### Common Redis Data Anomalies

| Anomaly | Detection Method | Root Cause | Repair Action |
|---------|------------------|------------|---------------|
| **Orphaned processing jobs** | `LLEN queue:*:processing` > 0 with no corresponding active worker | Worker crash during job execution | Run `Janitor.run_once()` or manually move jobs: `redis-cli RPOPLPUSH queue:default:processing queue:default` |
| **Retry scores in past** | `ZRANGE retry:default 0 0 WITHSCORES` shows score < current time | Scheduler not running or too slow | Manually enqueue: `redis-cli ZRANGE retry:default 0 0` to get job ID, then requeue |
| **Duplicate unique keys** | `redis-cli GET unique:{key}` exists but schedule not executed | Race condition in uniqueness check | Delete key: `redis-cli DEL unique:{key}` and manually enqueue job |
| **Job hash missing fields** | `redis-cli HGETALL job:{id}` shows incomplete field set | Serialization bug or partial update | Reconstruct from backup or mark as failed: `redis-cli HSET job:{id} status '"FAILED"'` |
| **Metrics keys without TTL** | `redis-cli TTL metrics:queue:default:depth` returns -1 | Metrics aggregation not setting expiration | Set TTL: `redis-cli EXPIRE metrics:queue:default:depth 3600` |
| **Zombie worker entries** | `worker:{id}` exists but process is dead | Worker crash without cleanup | Remove entry: `redis-cli DEL worker:{id} worker:{id}:jobs` |

#### Redis Lua Scripts for Complex Diagnostics

For complex diagnostics, use Redis Lua scripts to atomically check conditions:

**Script to find jobs stuck in processing with dead workers:**
```lua
-- find_stale_processing.lua
local processing_keys = redis.call('KEYS', 'queue:*:processing')
local results = {}
local current_time = tonumber(redis.call('TIME')[1])

for _, key in ipairs(processing_keys) do
  local queue_name = string.match(key, 'queue:(.+):processing')
  local worker_key = 'worker:active:' .. queue_name  -- hypothetical worker mapping
  
  -- Check if worker for this queue is alive
  local worker_alive = redis.call('EXISTS', worker_key)
  
  if worker_alive == 0 then
    local job_count = redis.call('LLEN', key)
    if job_count > 0 then
      table.insert(results, {queue_name, job_count, key})
    end
  end
end

return results
```

**Script to validate job state consistency:**
```lua
-- validate_job_state.lua
local job_id = ARGV[1]
local job_key = 'job:' .. job_id
local job_status = redis.call('HGET', job_key, 'status')

local inconsistencies = {}

-- Check if job is ACTIVE but not in any processing queue
if job_status == '"ACTIVE"' then
  local found = false
  local processing_keys = redis.call('KEYS', 'queue:*:processing')
  
  for _, key in ipairs(processing_keys) do
    local items = redis.call('LRANGE', key, 0, -1)
    for _, item in ipairs(items) do
      if string.find(item, job_id) then
        found = true
        break
      end
    end
    if found then break end
  end
  
  if not found then
    table.insert(inconsistencies, 'Job marked ACTIVE but not in any processing queue')
  end
end

-- Check if job is PENDING but not in any main queue
if job_status == '"PENDING"' then
  -- Similar logic checking main queues
end

return inconsistencies
```

> **Redis Inspection Principle:** Redis is the single source of truth for the system. When in doubt, examine Redis data directly. The dashboard and logs are derived views that might have their own bugs or lag. Always validate against Redis state.

### Implementation Guidance

This section provides concrete tools and code to implement the debugging capabilities described above.

#### Technology Recommendations

| Diagnostic Need | Simple Option | Advanced Option |
|-----------------|---------------|-----------------|
| Log aggregation | Structured JSON to files + `tail -f` | Elasticsearch + Logstash + Kibana (ELK) |
| Metrics collection | Redis time-series with periodic aggregation | Prometheus + Grafana for real-time metrics |
| Distributed tracing | Correlation IDs in logs | OpenTelemetry with Jaeger/Zipkin |
| Interactive debugging | Python debugger (pdb) | Remote debugging with debugpy |
| Performance profiling | cProfile for CPU, memory-profiler for memory | Py-Spy for sampling, Scalene for comprehensive |
| Alerting | Simple threshold checks in scheduler | Prometheus Alertmanager with complex rules |

#### Debug Endpoint Implementation

Create a dedicated debug module with these endpoints (protected in production):

```python
# debug_api.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import json
import redis
from datetime import datetime, timedelta
import asyncio

from .models import (
    Job, JobStatus, Worker, QueueMetrics, SystemConfig,
    QueueManager, WorkerConfig, RedisClient
)
from .janitor import Janitor

router = APIRouter(prefix="/api/debug", tags=["debug"])

@router.get("/queue/{queue_name}/contents")
async def get_queue_contents(
    queue_name: str,
    include_processing: bool = True,
    include_retry: bool = False,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Inspect all jobs in a queue, including processing and retry queues.
    Useful for diagnosing stuck jobs.
    """
    redis_client = RedisClient.get_instance()
    
    # Get pending jobs
    pending_key = f"queue:{queue_name}"
    pending_jobs_raw = redis_client.lrange(pending_key, 0, limit - 1)
    pending_jobs = []
    for job_data in pending_jobs_raw:
        try:
            job = Job.deserialize(job_data.decode('utf-8'))
            pending_jobs.append(job.to_dict())
        except Exception as e:
            pending_jobs.append({"error": str(e), "raw": job_data[:100]})
    
    result = {"pending": pending_jobs}
    
    # Get processing jobs
    if include_processing:
        processing_key = f"queue:{queue_name}:processing"
        processing_jobs_raw = redis_client.lrange(processing_key, 0, limit - 1)
        processing_jobs = []
        for job_data in processing_jobs_raw:
            try:
                job = Job.deserialize(job_data.decode('utf-8'))
                processing_jobs.append(job.to_dict())
            except Exception as e:
                processing_jobs.append({"error": str(e), "raw": job_data[:100]})
        result["processing"] = processing_jobs
    
    # Get retry scheduled jobs
    if include_retry:
        retry_key = f"retry:{queue_name}"
        retry_jobs = redis_client.zrange(
            retry_key, 0, limit - 1, withscores=True
        )
        retry_list = []
        for job_data, score in retry_jobs:
            try:
                job = Job.deserialize(job_data.decode('utf-8'))
                retry_list.append({
                    "job": job.to_dict(),
                    "scheduled_for": datetime.fromtimestamp(score)
                })
            except Exception as e:
                retry_list.append({"error": str(e), "raw": job_data[:100]})
        result["retry_scheduled"] = retry_list
    
    return result

@router.get("/job/{job_id}/trace")
async def get_job_trace(job_id: str) -> Dict[str, Any]:
    """
    Reconstruct the complete lifecycle of a job.
    """
    redis_client = RedisClient.get_instance()
    
    # Get job metadata
    job_key = f"job:{job_id}"
    job_data = redis_client.hgetall(job_key)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Decode byte strings
    job_data_decoded = {
        k.decode('utf-8'): v.decode('utf-8') 
        for k, v in job_data.items()
    }
    
    # Get error history
    errors_key = f"job:{job_id}:errors"
    errors = redis_client.lrange(errors_key, 0, -1)
    error_history = []
    for error_json in errors:
        try:
            error_history.append(json.loads(error_json.decode('utf-8')))
        except:
            error_history.append({"raw": error_json.decode('utf-8', errors='replace')})
    
    # Get result if exists
    result_key = f"job:{job_id}:result"
    result = redis_client.get(result_key)
    result_data = None
    if result:
        try:
            result_data = json.loads(result.decode('utf-8'))
        except:
            result_data = {"raw": result.decode('utf-8', errors='replace')}
    
    # Reconstruct timeline from metadata
    timeline = []
    
    # Created event
    if 'created_at' in job_data_decoded:
        timeline.append({
            "timestamp": job_data_decoded['created_at'],
            "event": "created",
            "queue": job_data_decoded.get('queue', 'unknown')
        })
    
    # Started event  
    if 'started_at' in job_data_decoded:
        timeline.append({
            "timestamp": job_data_decoded['started_at'],
            "event": "started",
            "status": "ACTIVE"
        })
    
    # Error events
    for i, error in enumerate(error_history):
        timeline.append({
            "timestamp": error.get('timestamp', 'unknown'),
            "event": f"failed_attempt_{i+1}",
            "error": error.get('error_type', 'Unknown'),
            "message": error.get('message', '')[:100]
        })
    
    # Completed event
    if 'completed_at' in job_data_decoded:
        timeline.append({
            "timestamp": job_data_decoded['completed_at'],
            "event": "completed",
            "status": job_data_decoded.get('status', 'unknown')
        })
    
    # Sort timeline
    timeline.sort(key=lambda x: x.get('timestamp', ''))
    
    return {
        "job_id": job_id,
        "metadata": job_data_decoded,
        "error_history": error_history,
        "result": result_data,
        "timeline": timeline
    }

@router.post("/janitor/run")
async def run_janitor(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Manually trigger the janitor process to clean up stale jobs.
    Returns immediately, runs in background.
    """
    async def cleanup_task():
        janitor = Janitor(RedisClient.get_instance())
        requeued = janitor.run_once()
        return requeued
    
    task = asyncio.create_task(cleanup_task())
    background_tasks.add_task(task)
    
    return {
        "status": "started",
        "message": "Janitor cleanup running in background"
    }

@router.get("/redis/keys")
async def scan_redis_keys(
    pattern: str = "*",
    limit: int = 100,
    with_ttl: bool = False
) -> Dict[str, Any]:
    """
    Scan Redis keys matching pattern. Use with caution in production.
    """
    redis_client = RedisClient.get_instance()
    
    # Use SCAN instead of KEYS for production safety
    cursor = 0
    keys = []
    ttls = {}
    
    while True:
        cursor, batch = redis_client.scan(
            cursor=cursor, match=pattern, count=100
        )
        keys.extend([k.decode('utf-8') for k in batch])
        
        if with_ttl:
            for key in batch:
                ttl = redis_client.ttl(key)
                ttls[key.decode('utf-8')] = ttl
        
        if cursor == 0 or len(keys) >= limit:
            break
    
    keys = keys[:limit]
    
    return {
        "pattern": pattern,
        "count": len(keys),
        "keys": keys,
        "ttls": ttls if with_ttl else None
    }

@router.post("/worker/{worker_id}/signal")
async def signal_worker(
    worker_id: str,
    signal: str = "status"
) -> Dict[str, Any]:
    """
    Send a signal to a worker (status, pause, resume, shutdown).
    In production, this would require authentication and authorization.
    """
    if signal not in ["status", "pause", "resume", "shutdown"]:
        raise HTTPException(status_code=400, detail="Invalid signal")
    
    redis_client = RedisClient.get_instance()
    
    # Store signal in Redis for worker to poll
    signal_key = f"worker:{worker_id}:signal"
    redis_client.setex(signal_key, 30, signal)
    
    return {
        "worker_id": worker_id,
        "signal": signal,
        "sent_at": datetime.utcnow().isoformat()
    }
```

#### Diagnostic Worker Implementation

Create a diagnostic worker that can be run alongside normal workers to monitor system health:

```python
# diagnostic_worker.py
import time
import json
import psutil
import redis
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
import threading

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_rss: int
    redis_memory_used: int
    redis_connected_clients: int
    queue_depths: Dict[str, int]
    worker_count: int
    schedule_count: int
    
class DiagnosticWorker:
    """
    Runs periodic system diagnostics and logs anomalies.
    Not a job worker, but a monitoring agent.
    """
    def __init__(self, redis_client: redis.Redis, interval: int = 60):
        self.redis = redis_client
        self.interval = interval
        self.running = False
        self.thread = None
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.Process().memory_info()
        
        # Redis metrics
        redis_info = self.redis.info()
        
        # Queue depths
        queue_depths = {}
        queue_patterns = self.redis.keys("queue:*")
        for key in queue_patterns:
            if b":processing" not in key and b":retry" not in key:
                queue_name = key.decode('utf-8').replace("queue:", "")
                depth = self.redis.llen(key)
                queue_depths[queue_name] = depth
        
        # Worker count
        worker_keys = self.redis.keys("worker:*")
        worker_count = len([k for k in worker_keys if b":jobs" not in k])
        
        # Schedule count
        schedule_keys = self.redis.keys("schedule:*")
        schedule_count = len([k for k in schedule_keys if b":history" not in k])
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_rss=memory_info.rss,
            redis_memory_used=redis_info.get('used_memory', 0),
            redis_connected_clients=redis_info.get('connected_clients', 0),
            queue_depths=queue_depths,
            worker_count=worker_count,
            schedule_count=schedule_count
        )
    
    def check_anomalies(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check for system anomalies based on metrics."""
        anomalies = []
        
        # High memory usage
        if metrics.redis_memory_used > 500 * 1024 * 1024:  # 500MB
            anomalies.append({
                "type": "high_redis_memory",
                "severity": "warning",
                "value": metrics.redis_memory_used,
                "threshold": 500 * 1024 * 1024
            })
        
        # Queue depth anomalies
        for queue_name, depth in metrics.queue_depths.items():
            if depth > 1000:
                anomalies.append({
                    "type": "high_queue_depth",
                    "severity": "warning",
                    "queue": queue_name,
                    "depth": depth
                })
        
        # No workers running
        if metrics.worker_count == 0:
            anomalies.append({
                "type": "no_workers",
                "severity": "critical",
                "message": "No active workers detected"
            })
        
        # Redis connection issues
        if metrics.redis_connected_clients == 0:
            anomalies.append({
                "type": "redis_no_clients",
                "severity": "critical",
                "message": "Redis shows no connected clients"
            })
        
        return anomalies
    
    def run_diagnostics(self):
        """Run one diagnostic cycle."""
        try:
            metrics = self.collect_metrics()
            anomalies = self.check_anomalies(metrics)
            
            # Log metrics
            self.redis.rpush(
                "diagnostics:metrics",
                json.dumps({
                    "timestamp": metrics.timestamp.isoformat(),
                    "metrics": {
                        "cpu_percent": metrics.cpu_percent,
                        "memory_rss": metrics.memory_rss,
                        "redis_memory": metrics.redis_memory_used,
                        "worker_count": metrics.worker_count
                    }
                })
            )
            
            # Log anomalies
            if anomalies:
                for anomaly in anomalies:
                    self.redis.rpush(
                        "diagnostics:anomalies",
                        json.dumps({
                            "timestamp": datetime.utcnow().isoformat(),
                            "anomaly": anomaly
                        })
                    )
                    # Also log to regular logs
                    print(f"ANOMALY: {anomaly}")
            
            # Set TTL on diagnostics keys to prevent unbounded growth
            self.redis.expire("diagnostics:metrics", 7 * 24 * 3600)  # 7 days
            self.redis.expire("diagnostics:anomalies", 7 * 24 * 3600)
            
        except Exception as e:
            print(f"Diagnostic error: {e}")
    
    def start(self):
        """Start the diagnostic worker in a background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def _run_loop(self):
        """Main diagnostic loop."""
        while self.running:
            self.run_diagnostics()
            time.sleep(self.interval)
    
    def stop(self):
        """Stop the diagnostic worker."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
```

#### Debugging Helper Scripts

Create command-line scripts for common debugging tasks:

```python
#!/usr/bin/env python3
# scripts/debug_queue.py
"""
Command-line tool to inspect queue state.
"""
import argparse
import json
import sys
from typing import List, Dict, Any

from job_processor.queue_manager import QueueManager
from job_processor.redis_client import RedisClient

def inspect_queue(queue_name: str, limit: int = 10):
    """Inspect jobs in a queue."""
    redis = RedisClient.get_instance()
    queue_key = f"queue:{queue_name}"
    
    print(f"=== Queue: {queue_name} ===")
    print(f"Length: {redis.llen(queue_key)}")
    print("\nNext jobs:")
    
    jobs = redis.lrange(queue_key, 0, limit - 1)
    for i, job_data in enumerate(jobs):
        try:
            job_json = json.loads(job_data.decode('utf-8'))
            print(f"{i+1}. {job_json.get('job_id', 'unknown')}")
            print(f"   Type: {job_json.get('job_type', 'unknown')}")
            print(f"   Args: {job_json.get('args', [])[:3]}")
            if len(job_json.get('args', [])) > 3:
                print(f"   ... and {len(job_json['args']) - 3} more")
        except json.JSONDecodeError:
            print(f"{i+1}. INVALID JSON: {job_data[:100]}...")
        print()

def find_stuck_jobs(age_hours: int = 1):
    """Find jobs that have been active for too long."""
    redis = RedisClient.get_instance()
    
    # Scan for all job keys
    cursor = 0
    stuck_jobs = []
    
    while True:
        cursor, keys = redis.scan(cursor, match="job:*", count=100)
        
        for key in keys:
            # Skip subkeys like job:{id}:errors
            if b":" in key and key.count(b":") > 1:
                continue
                
            job_data = redis.hgetall(key)
            if not job_data:
                continue
                
            # Decode
            decoded = {}
            for k, v in job_data.items():
                try:
                    decoded[k.decode('utf-8')] = v.decode('utf-8')
                except:
                    decoded[k.decode('utf-8', errors='replace')] = str(v)
            
            # Check if ACTIVE for too long
            if decoded.get('status') == '"ACTIVE"':
                started_at = decoded.get('started_at')
                if started_at:
                    try:
                        # Parse timestamp - might be string in JSON format
                        if started_at.startswith('"'):
                            started_at = started_at[1:-1]
                        # TODO: Parse datetime and compare with age_hours
                        # For now just report
                        stuck_jobs.append({
                            'job_id': decoded.get('job_id', 'unknown'),
                            'started_at': started_at,
                            'queue': decoded.get('queue', 'unknown')
                        })
                    except:
                        pass
        
        if cursor == 0:
            break
    
    if stuck_jobs:
        print(f"Found {len(stuck_jobs)} potentially stuck jobs:")
        for job in stuck_jobs:
            print(f"  {job['job_id']} in {job['queue']} since {job['started_at']}")
    else:
        print("No stuck jobs found.")

def cleanup_processing_queues():
    """Clean up jobs stuck in processing queues."""
    redis = RedisClient.get_instance()
    
    # Find all processing queues
    cursor = 0
    processing_queues = []
    
    while True:
        cursor, keys = redis.scan(cursor, match="queue:*:processing", count=50)
        processing_queues.extend([k.decode('utf-8') for k in keys])
        
        if cursor == 0:
            break
    
    total_moved = 0
    for queue_key in processing_queues:
        # Extract main queue name
        main_queue = queue_key.replace(":processing", "")
        
        # Move all jobs back to main queue
        while True:
            job_data = redis.rpoplpush(queue_key, main_queue)
            if not job_data:
                break
            total_moved += 1
        
        print(f"Cleaned {queue_key}")
    
    print(f"Total jobs requeued: {total_moved}")

def main():
    parser = argparse.ArgumentParser(description="Debug job queue system")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a queue')
    inspect_parser.add_argument('queue', help='Queue name')
    inspect_parser.add_argument('--limit', type=int, default=10, help='Number of jobs to show')
    
    # stuck command
    stuck_parser = subparsers.add_parser('stuck', help='Find stuck jobs')
    stuck_parser.add_argument('--age-hours', type=int, default=1, help='Age threshold in hours')
    
    # cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup processing queues')
    
    args = parser.parse_args()
    
    if args.command == 'inspect':
        inspect_queue(args.queue, args.limit)
    elif args.command == 'stuck':
        find_stuck_jobs(args.age_hours)
    elif args.command == 'cleanup':
        cleanup_processing_queues()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

#### Language-Specific Debugging Hints

**Python-Specific Debugging:**

1. **Async/Await Deadlocks**: Use `asyncio.all_tasks()` to see all running tasks when debugging hangs in async workers.

2. **Memory Leaks**: Use `tracemalloc` to track memory allocations:
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... run operations ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')
   for stat in top_stats[:10]:
       print(stat)
   ```

3. **Thread Dumps**: For hung workers, send `SIGUSR1` to get a Python traceback of all threads:
   ```python
   import signal
   import traceback
   import sys
   
   def debug(sig, frame):
       """Interrupt running process and provide a python prompt for debugging."""
       import pdb
       pdb.Pdb().set_trace(frame)
   
   signal.signal(signal.SIGUSR1, debug)  # Register handler
   ```

4. **Pickle Serialization Issues**: When debugging job serialization, test round-trips:
   ```python
   import pickle
   job_dict = job.to_dict()
   serialized = pickle.dumps(job_dict)
   deserialized = pickle.loads(serialized)
   assert job_dict == deserialized  # Verify no data loss
   ```

#### Milestone Debugging Checkpoints

After implementing each milestone, verify these debugging capabilities work:

**Milestone 1 (Queue Core) Checkpoint:**
- Run `python scripts/debug_queue.py inspect default` - should show enqueued jobs
- Enqueue a job >1MB - should get validation error in logs
- Kill Redis temporarily during enqueue - should see connection error handling

**Milestone 2 (Worker) Checkpoint:**
- Start worker, send `SIGTERM` - should complete current job before exiting
- Check `redis-cli HGETALL worker:{id}` - should see heartbeat updating
- Kill worker mid-job - janitor should move job back to queue

**Milestone 3 (Retry) Checkpoint:**
- Create job that always fails - watch retry count increment in job metadata
- Check `redis-cli ZRANGE retry:default 0 -1 WITHSCORES` - should see scheduled retries
- Exhaust retries - job should move to dead letter queue

**Milestone 4 (Scheduler) Checkpoint:**
- Create schedule for 1 minute in future - watch it enqueue at correct time
- Check `redis-cli HGETALL schedule:{id}` - `next_run_at` should update
- Change system timezone - schedule should adjust correctly

**Milestone 5 (Monitoring) Checkpoint:**
- Access `/api/debug/queue/default/contents` - should see queue contents
- Check WebSocket connection to metrics stream - should receive updates
- Trigger alert condition - should see alert in dashboard

> **Implementation Insight:** Build debugging capabilities alongside features, not as an afterthought. Each component should expose its internal state for inspection. The time spent building debug tools pays back multifold when diagnosing production issues.


## Future Extensions

> **Milestone(s):** This section explores potential enhancements beyond the five core milestones. It demonstrates how the architecture's design decisions enable natural evolution while identifying boundaries where significant rework would be required.

A well-architected system should be both complete in its current form and extensible for future needs. This section examines enhancements that could be built upon the current background job processor, categorized by the level of architectural change required. The core design—with its clear separation of concerns, well-defined interfaces, and Redis-based coordination—enables numerous extensions without fundamental rework, adhering to the **YAGNI** principle while maintaining forward compatibility.

### Architecture-Enabled Extensions

These enhancements leverage existing architectural patterns, data structures, and component boundaries. They require minimal to moderate development effort and don't necessitate breaking changes to core interfaces or data models.

| Extension | Description | Implementation Approach | Architectural Enablers |
|-----------|-------------|------------------------|------------------------|
| **Rate Limiting** | Control the maximum number of jobs processed per unit time per queue, job type, or worker group to prevent resource exhaustion. | Extend `QueueManager` with token bucket or sliding window counters stored in Redis. Workers check limits before processing jobs. | Redis atomic operations for counters, existing queue isolation, configurable worker polling. |
| **Job Dependencies** | Allow jobs to specify prerequisite jobs that must complete successfully before dependent jobs can start. | Add `dependencies` field to `Job` storing list of prerequisite job IDs. New `DependencyManager` component tracks completion and enqueues dependents. | Job lifecycle tracking in Redis, existing event flow from completion to enqueue. |
| **Multi-Language Worker Support** | Enable workers written in languages other than Python (Go, Rust, JavaScript) to process jobs from the same queues. | Standardize job serialization format (JSON/msgpack) and Redis key structures. Create client libraries in each language implementing the same protocol. | Language-agnostic Redis data model, explicit job serialization format, documented key naming conventions. |
| **Priority Queues with Preemption** | Allow higher-priority jobs to interrupt lower-priority jobs already being processed. | Extend `Worker` to support job suspension/resumption or termination with requeue. Add preemption signal handling in job execution wrapper. | Job state machine with explicit `ACTIVE` state, worker heartbeat for liveness, configurable timeout handling. |
| **Job Result Caching** | Cache job results for idempotent operations to avoid redundant computation for identical inputs. | Add optional `cache_key` and `cache_ttl` fields to `Job`. `QueueManager` checks Redis cache before enqueueing, returns cached result if present. | Redis as shared cache, job serialization includes all input parameters, existing result storage mechanism. |
| **Job Chaining & Pipelines** | Define sequences of jobs where each job's output becomes the next job's input, with error handling across the chain. | New `JobChain` object with list of job definitions and error policy. `ChainManager` monitors execution and propagates data between jobs. | Job metadata for correlation, error handling infrastructure, job registry for dynamic job type resolution. |
| **Dynamic Queue Configuration** | Add, remove, or modify queue configurations at runtime without restarting workers or scheduler. | Extend `QueueManager` to watch Redis for configuration changes and adjust polling weights. Workers subscribe to configuration updates via Redis Pub/Sub. | Configuration stored in Redis, worker polling strategy decoupled from static config, heartbeat mechanism for dynamic updates. |
| **Job Tagging & Metadata Search** | Attach arbitrary tags to jobs and enable searching/filtering by tags in the dashboard and APIs. | Add `tags: Dict[str, str]` field to `Job`. Index tags in Redis sorted sets for efficient querying. Extend dashboard filters. | Flexible job metadata field, Redis sorted set pattern for indexing, dashboard filtering architecture. |
| **Worker Resource Profiles** | Assign workers to specific resource pools (CPU-intensive, memory-heavy, GPU) and route jobs accordingly. | Add `resource_profile` field to `Job` and `Worker`. Extend queue selection logic to match profiles. New resource-specific queues. | Multiple named queues with priorities, worker queue subscription model, configurable routing rules. |
| **Delayed Job Cancellation** | Allow producers to cancel or modify delayed/scheduled jobs before they execute. | Store delayed jobs in Redis with cancellation tokens. Add `cancel_job` API that marks jobs as cancelled in sorted set. | Delayed jobs stored in Redis sorted sets with job IDs, scheduler polling with batch retrieval. |

#### Detailed Analysis: Rate Limiting Extension

> **Decision: Token Bucket Rate Limiter per Queue**
> - **Context**: Certain job types (e.g., external API calls, database-intensive operations) must be throttled to prevent overwhelming downstream systems or exhausting shared resources.
> - **Options Considered**:
>   1. **Fixed Window Counter**: Simple count per time window, but suffers from boundary effects allowing bursts.
>   2. **Sliding Window Log**: Accurate but memory-intensive, storing timestamps for each request.
>   3. **Token Bucket**: Smooths rate, allows bursting up to bucket capacity, computationally efficient.
> - **Decision**: Implement token bucket algorithm using Redis Lua scripts for atomic operations.
> - **Rationale**: Token bucket provides predictable smoothing with burst capacity, matches typical rate limiting requirements for external APIs, and Redis Lua scripts ensure atomicity across multiple workers checking limits concurrently.
> - **Consequences**: Adds small overhead to each job dequeue operation, requires Redis Lua support, provides configurable burst capacity and refill rate.

| Rate Limiting Implementation Comparison |
|-----------------------------------------|
| **Option** | **Pros** | **Cons** | **Suitability** |
| Fixed Window Counter | Simple implementation, low memory usage | Boundary bursts allow 2x limit, less accurate | Low-precision limits where bursts acceptable |
| Sliding Window Log | Accurate, no boundary bursts | High memory usage (stores all timestamps), slower | When precise limits are critical and traffic low |
| Token Bucket | Smooth rate, allows controlled bursts, efficient | Slightly more complex, requires bucket size parameter | Most API rate limiting scenarios (chosen) |

**Implementation Walkthrough**:
1. **Configuration Enhancement**: Add `rate_limit` field to `QueueConfig` with `requests_per_second` and `burst_capacity` parameters.
2. **Redis Storage**: Use key pattern `ratelimit:{queue_name}` storing bucket state (tokens, last_update) in a Redis hash.
3. **Atomic Check Script**: Lua script that:
   1. Calculates time elapsed since last update
   2. Adds tokens based on elapsed time × refill rate (capped at burst capacity)
   3. If tokens ≥ 1, decrements tokens by 1 and returns `ALLOWED`
   4. Otherwise returns `DENIED` with time until next token available
4. **Worker Integration**: Before processing a job from a rate-limited queue, worker executes the Lua script. If `DENIED`, worker sleeps for the suggested duration before polling again.
5. **Dashboard Visibility**: Add rate limit status to queue metrics showing current token count, limit status, and throttled worker count.

**Concrete Example**: A webhook queue limited to 10 requests/second with burst capacity of 20. When 25 jobs arrive simultaneously, the first 20 process immediately (burst), then subsequent jobs throttle to 10/second. The token bucket refills at 10 tokens/second, ensuring long-term average compliance.

#### Detailed Analysis: Job Dependencies Extension

> **Decision: Directed Acyclic Graph (DAG) Dependency Model**
> - **Context**: Complex workflows require sequencing where job B cannot start until job A completes successfully, enabling multi-step data pipelines and business processes.
> - **Options Considered**:
>   1. **Linear Chains**: Simple sequential dependencies only (A→B→C), no branching or parallelism.
>   2. **DAG with Explicit Graph Definition**: Full dependency graph defined upfront, allowing parallel execution of independent branches.
>   3. **Dynamic Dependencies**: Jobs can spawn new dependencies at runtime, more flexible but harder to reason about.
> - **Decision**: Implement DAG model with upfront graph definition stored as job metadata.
> - **Rationale**: DAGs capture most real-world workflows (branching, parallelism), allow static validation of cycles, and enable optimal scheduling of independent branches. Upfront definition simplifies implementation and debugging.
> - **Consequences**: Requires graph validation to prevent cycles, adds complexity to job completion handling, needs visualization in dashboard.

**Implementation Walkthrough**:
1. **Job Enhancement**: Add `dependencies: List[str]` field to `Job` storing prerequisite job IDs, and `dependent_jobs: List[str]` for reverse links.
2. **Dependency Manager**: New component that:
   - Validates job graphs are acyclic at enqueue time
   - Maintains Redis sets tracking pending dependencies for each job
   - Listens for job completion events via Redis Pub/Sub
   - When job completes, removes it from dependency sets of dependent jobs
   - When all dependencies satisfied for a job, enqueues it via `QueueManager`
3. **Atomic Graph Operations**: Use Redis transactions to atomically update dependency state and enqueue jobs when ready.
4. **Failure Handling**: If a job fails, dependent jobs can be configured to either (a) not run, (b) run with error placeholder, or (c) run after manual intervention.
5. **Dashboard Visualization**: Display dependency graphs with status color coding, allow drilling into specific job chains.

**Concrete Example**: Data pipeline with jobs: `fetch_data` → [`clean_data`, `validate_schema`] → `merge_results`. `clean_data` and `validate_schema` run in parallel after `fetch_data` completes, both must complete before `merge_results` starts. The dependency manager ensures this ordering automatically.

### Extensions Requiring Major Changes

These enhancements would require significant architectural modifications, breaking changes to interfaces, or fundamental shifts in system assumptions. They represent directions for major version upgrades or entirely new system variants.

| Extension | Description | Required Architectural Changes | Rationale for Major Change |
|-----------|-------------|--------------------------------|----------------------------|
| **Distributed Transactions Across Jobs** | Ensure atomicity across multiple job executions (all succeed or all roll back) with two-phase commit protocol. | New transaction coordinator, WAL for transaction state, participant job interface for prepare/commit/rollback, recovery mechanisms for coordinator failures. | Changes core job execution model from independent to coordinated, requires persistent transaction state, adds complexity to job handlers. |
| **Multi-Broker Support** | Support alternative brokers beyond Redis (RabbitMQ, Kafka, PostgreSQL) with pluggable backend adapters. | Abstract broker interface, adapter pattern for each backend, migration of data models between backends, feature compatibility matrix. | Current architecture deeply coupled to Redis semantics (sorted sets, blocking pops, Lua scripts). Abstracting requires rethinking atomic operations and data structures. |
| **Exactly-Once Processing Guarantees** | Ensure each job is processed exactly once despite worker failures, network partitions, or duplicate enqueues. | Idempotency keys with distributed locking, deterministic job ID generation, transaction log for dequeue operations, coordinated checkpointing. | Conflicts with current at-least-once semantics with retries, requires fundamental changes to dequeue/requeue logic and job deduplication at storage layer. |
| **Geographic Distribution with Cross-Region Replication** | Deploy job processors across multiple regions with automatic failover and geographic job routing. | Active-active Redis replication with conflict resolution, latency-based routing, cross-region job migration, distributed lock coordination. | Assumes single Redis instance; multi-region requires consensus on job ownership, clock synchronization, and replication lag handling. |
| **Stream Processing Mode** | Process continuous streams of data with windowing, aggregation, and real-time analytics alongside batch jobs. | New stream processing engine, windowed state storage, watermarking for out-of-order events, integration with job scheduler for batch/stream hybrid. | Changes from discrete job model to continuous processing model, requires different programming model, state management, and monitoring. |
| **Machine Learning Model Serving Integration** | Specialized support for ML model inference jobs with GPU allocation, model versioning, and inference batching. | GPU-aware scheduling, model registry integration, adaptive batching based on queue depth, performance profiling and auto-scaling. | Requires specialized hardware awareness, performance optimization trade-offs different from CPU jobs, tight coupling with ML infrastructure. |
| **Workflow Engine with Human Tasks** | Extend job dependencies to include human approval steps, forms, and long-running processes (days/weeks). | Human task queue, user assignment and notification, form rendering, long-term state persistence, suspension/resumption with context. | Time scale mismatch (seconds vs days), requires user interface integration, authentication/authorization, different state persistence requirements. |
| **Serverless Execution Backend** | Execute jobs on serverless platforms (AWS Lambda, Google Cloud Functions) instead of persistent worker processes. | Job packaging for serverless environments, cold start optimization, cost-based scheduling, execution environment isolation. | Fundamental shift from persistent workers to ephemeral execution, changes scaling model, monitoring, and error recovery approaches. |

#### Detailed Analysis: Exactly-Once Processing Guarantees

> **Decision: Distributed Idempotency with Deterministic Job Chaining**
> - **Context**: Financial and audit-sensitive workloads require stronger guarantees than at-least-once delivery, preventing duplicate processing that could cause double charges or inconsistent state.
> - **Architectural Challenges**:
>   1. **Dequeue Atomicity**: Current `BRPOPLPUSH` moves job to processing queue but worker crash before completion leaves job in limbo (janitor recovers but could cause duplicate processing).
>   2. **Retry Duplication**: Exponential backoff retries create multiple execution attempts of the same logical job.
>   3. **Producer Duplication**: Network timeouts might cause producers to enqueue the same job twice.
> - **Required Changes**:
>   - **Idempotency Keys**: Require producers to supply idempotency key with each job, stored and checked before processing.
>   - **Deterministic Job IDs**: Generate job IDs as hash of job content + idempotency key to enable deduplication at storage layer.
>   - **Two-Phase Dequeue**: Worker first acquires distributed lock on job ID, processes, marks as committed in Redis transaction, then releases lock.
>   - **Intent Logging**: Write "processing started" record before execution begins, with recovery scanning for orphaned intents.
> - **Impact**: Significant performance overhead (distributed locks, additional Redis writes), more complex failure recovery, incompatible with current simple FIFO processing model.

**Migration Path Considerations**:
1. **Phase 1**: Add optional idempotency key support alongside current semantics for gradual migration.
2. **Phase 2**: Implement exactly-once processing as separate queue type with different guarantees.
3. **Phase 3**: Deprecate at-least-once queues over multiple releases, providing migration tools.

#### Detailed Analysis: Multi-Broker Support

> **Decision: Pluggable Broker Adapter Interface**
> - **Context**: Organizations have existing infrastructure investments (Kafka for event streaming, PostgreSQL for transactional consistency) and want to leverage them as job brokers without maintaining separate Redis clusters.
> - **Architectural Challenges**:
>   1. **Redis-Centric Data Structures**: Sorted sets for retries, lists for queues, pub/sub for signals—each has different semantics in other systems.
>   2. **Atomic Operations**: Redis Lua scripts and pipelines provide atomicity that must be reimplemented for each backend.
>   3. **Performance Characteristics**: Different latency, throughput, and consistency trade-offs affect system design.
> - **Required Changes**:
>   - **Broker Abstraction Layer**: Define `Broker` interface with methods for enqueue, dequeue, sorted set operations, pub/sub, and atomic transactions.
>   - **Adapter Implementations**: Create `RedisBroker`, `KafkaBroker`, `PostgreSQLBroker` each implementing the interface with backend-specific optimizations.
>   - **Feature Flagging**: Some features (e.g., priority queue weighting) may not be available in all backends—requires graceful degradation.
>   - **Migration Tools**: Utilities to transfer job state between brokers during migration.
> - **Impact**: Doubles code complexity, requires maintaining multiple backend implementations, potentially limits use of advanced Redis-specific features.

**Feature Compatibility Matrix**:
| Feature | Redis | Kafka | PostgreSQL | Notes |
|---------|-------|-------|------------|-------|
| FIFO Queues | ✅ Lists | ✅ Topics (partition ordering) | ✅ Table with sequence | Kafka requires single partition per queue for strict FIFO |
| Delayed Jobs | ✅ Sorted Sets | ⚠️ Requires separate scheduler | ✅ UPDATE with WHERE scheduled_at | |
| Priority Queues | ✅ Multiple lists with weighted polling | ❌ No native support | ✅ WITH RECURSIVE query | |
| Exactly-Once Processing | ⚠️ With careful design | ✅ With transactional producer | ✅ Native transactions | |
| Horizontal Scaling | ✅ Native | ✅ Native | ⚠️ Connection pooling limits | |

> **Key Insight**: The architecture's clear separation between job logic and queue management enables these extensions, but the choice of Redis as the broker creates both capabilities and constraints. Extensions that align with Redis's strengths (in-memory data structures, atomic operations) are natural; those requiring different consistency models or storage patterns demand architectural evolution.

### Common Pitfalls in Extension Implementation

⚠️ **Pitfall: Over-Engineering for Hypothetical Use Cases**
- **Description**: Implementing complex extensions (like distributed transactions) before any concrete requirement exists, based on "what if" scenarios.
- **Why It's Wrong**: Increases system complexity, introduces bugs in unused code paths, delays delivery of core functionality, violates YAGNI principle.
- **Fix**: Implement only extensions with proven user demand. Use the architecture's separation of concerns to keep extension points clear but unimplemented until needed.

⚠️ **Pitfall: Breaking Backward Compatibility**
- **Description**: Adding new features that require existing job definitions or worker code to be modified, forcing coordinated deployments.
- **Why It's Wrong**: Causes production outages during rolling updates, requires complex migration procedures, frustrates users.
- **Fix**: Design extensions with backward compatibility: additive fields only, default values for new requirements, feature flags to enable gradually, dual-read/write patterns during transitions.

⚠️ **Pitfall: Neglecting Performance Impact**
- **Description**: Adding rate limiting or dependency tracking without considering the additional Redis operations per job, causing latency spikes under load.
- **Why It's Wrong**: Turns a high-performance job processor into a bottleneck, defeats purpose of async processing.
- **Fix**: Profile extensions under load, use Redis pipelining, consider approximate algorithms (e.g., probabilistic rate limiting), provide benchmarks for each extension.

⚠️ **Pitfall: Creating Monolithic Components**
- **Description**: Building extensions as tightly coupled additions to core components rather than separate services or plugins.
- **Why It's Wrong**: Makes system harder to understand, test, and maintain; increases risk of regressions.
- **Fix**: Implement extensions as separate processes or threads with well-defined interfaces: dependency manager as separate microservice, rate limiter as Redis-side Lua scripts, dashboard extensions as plugin architecture.

### Implementation Guidance

While extensions are beyond the core milestones, their implementation builds upon the established patterns. Here's guidance for implementing the most valuable architecture-enabled extensions.

#### Technology Recommendations for Extensions

| Extension | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Rate Limiting | Redis token bucket with Lua script | Distributed rate limiter with Redis Cluster support, adaptive limits based on downstream health |
| Job Dependencies | Linear chains with Redis sets for tracking | Full DAG with graph database backend (Neo4j), visual workflow editor |
| Multi-Language Support | JSON serialization with language-specific client libraries | gRPC-based job protocol with code generation, shared memory for job payloads |
| Priority Preemption | Job termination with requeue at front of queue | Job suspension/resumption with checkpointing, container snapshotting for long jobs |

#### Recommended File/Module Structure for Extensions

```
background_job_processor/
├── core/                    # Existing core components
│   ├── queue_manager.py
│   ├── worker.py
│   └── ...
├── extensions/              # New extensions directory
│   ├── rate_limiting/
│   │   ├── __init__.py
│   │   ├── token_bucket.py
│   │   ├── lua_scripts/    # Redis Lua scripts
│   │   └── tests/
│   ├── dependencies/
│   │   ├── __init__.py
│   │   ├── dependency_manager.py
│   │   ├── graph_validator.py
│   │   └── tests/
│   └── multi_lang/
│       ├── __init__.py
│       ├── protocol.py     # Serialization protocol spec
│       ├── go_client/      # Go client library
│       └── rust_client/    # Rust client library
└── dashboard_extensions/   # UI extensions
    ├── rate_limit_panel.py
    └── dependency_graph.py
```

#### Infrastructure Starter Code: Rate Limiting Lua Script

```python
# extensions/rate_limiting/lua_scripts/token_bucket.lua
-- KEYS[1]: rate limit key (e.g., "ratelimit:webhooks")
-- ARGV[1]: current time in milliseconds
-- ARGV[2]: refill rate (tokens per millisecond)
-- ARGV[3]: burst capacity (max tokens)
-- Returns: ALLOWED or DENIED:wait_time_ms

local key = KEYS[1]
local now = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(bucket[1]) or capacity
local last_update = tonumber(bucket[2]) or now

-- Calculate refill based on time elapsed
local time_passed = math.max(0, now - last_update)
local refill_amount = time_passed * refill_rate
tokens = math.min(capacity, tokens + refill_amount)

local result
if tokens >= 1 then
    tokens = tokens - 1
    result = "ALLOWED"
else
    -- Calculate time until next token
    local wait_time = math.ceil((1 - tokens) / refill_rate)
    result = "DENIED:" .. wait_time
end

-- Update bucket state
redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
redis.call('EXPIRE', key, math.ceil(capacity / refill_rate / 1000) * 2)  -- Auto-expire idle buckets

return result
```

```python
# extensions/rate_limiting/token_bucket.py
"""Token bucket rate limiter implementation."""
import time
import redis
from typing import Optional, Tuple

class TokenBucketRateLimiter:
    """Rate limiter using token bucket algorithm with Redis Lua script."""
    
    def __init__(self, redis_client: redis.Redis, script_path: str = "token_bucket.lua"):
        self.redis = redis_client
        # Load Lua script
        with open(script_path, 'r') as f:
            self.lua_script = self.redis.register_script(f.read())
    
    def check_limit(self, queue_name: str, requests_per_second: float, 
                   burst_capacity: int) -> Tuple[bool, Optional[float]]:
        """
        Check if job can be processed based on rate limit.
        
        Args:
            queue_name: Name of queue to check
            requests_per_second: Refill rate in tokens per second
            burst_capacity: Maximum tokens bucket can hold
            
        Returns:
            Tuple of (allowed, wait_time_seconds). wait_time is None if allowed.
        """
        key = f"ratelimit:{queue_name}"
        now_ms = int(time.time() * 1000)
        refill_per_ms = requests_per_second / 1000.0
        
        result = self.lua_script(
            keys=[key],
            args=[now_ms, refill_per_ms, burst_capacity]
        )
        
        if result.startswith("ALLOWED"):
            return True, None
        else:
            # Result format: "DENIED:wait_time_ms"
            wait_time_ms = int(result.split(":")[1])
            return False, wait_time_ms / 1000.0
```

#### Core Logic Skeleton: Dependency Manager

```python
# extensions/dependencies/dependency_manager.py
"""Manager for job dependencies and DAG execution."""
from typing import List, Dict, Set, Optional
from datetime import datetime
import json
from core.queue_manager import QueueManager
from core.job import Job

class DependencyManager:
    """Manages job dependencies using Redis for state tracking."""
    
    def __init__(self, redis_client, queue_manager: QueueManager):
        self.redis = redis_client
        self.queue_manager = queue_manager
        self.pubsub = self.redis.pubsub()
        
    def register_job_with_dependencies(self, job: Job, 
                                      dependencies: List[str]) -> str:
        """
        Register a job that depends on other jobs completing.
        
        Args:
            job: Job to register (will not be enqueued until dependencies met)
            dependencies: List of job IDs that must complete first
            
        Returns:
            Job ID of registered job
            
        Raises:
            ValidationError: If dependency graph would create a cycle
        """
        # TODO 1: Validate job has unique ID (generate if not)
        # TODO 2: Check for cycles in dependency graph using Redis sets
        # TODO 3: Store job in Redis with status PENDING_DEPENDENCIES
        # TODO 4: For each dependency, add this job to the dependent_jobs set
        # TODO 5: For each dependency, check if already COMPLETED
        #         - If so, decrement pending count for this job
        # TODO 6: If all dependencies already satisfied, enqueue job immediately
        # TODO 7: Subscribe to job completion events for dependencies
        pass
    
    def _handle_job_completion(self, job_id: str, status: str) -> None:
        """
        Handle completion of a job and notify dependents.
        
        Args:
            job_id: ID of completed job
            status: COMPLETED or FAILED (affects dependent jobs)
        """
        # TODO 1: Retrieve list of dependent job IDs from Redis
        # TODO 2: For each dependent job:
        #   - Decrement pending dependency count in Redis
        #   - If count reaches zero and job is not CANCELLED:
        #        - Retrieve job from Redis
        #        - Enqueue via queue_manager
        #        - Update job status to PENDING
        # TODO 3: If status is FAILED and dependent jobs should not run:
        #   - Mark dependent jobs as CANCELLED
        #   - Optionally notify via error channel
        pass
    
    def start_listening(self) -> None:
        """Start listening for job completion events."""
        # TODO 1: Subscribe to job completion channel via Redis Pub/Sub
        # TODO 2: Start thread to process messages
        # TODO 3: For each completion message, call _handle_job_completion
        pass
    
    def cancel_job_chain(self, job_id: str) -> None:
        """
        Cancel a job and all its dependent jobs recursively.
        
        Args:
            job_id: ID of job to cancel
        """
        # TODO 1: Mark job as CANCELLED in Redis
        # TODO 2: Get dependent jobs recursively
        # TODO 3: Mark all as CANCELLED
        # TODO 4: Clean up dependency tracking sets
        pass
```

#### Language-Specific Hints for Extensions

**Python-Specific Tips**:
- Use `asyncio` for dependency manager event listening to avoid blocking threads.
- For rate limiting Lua scripts, use `redis.register_script()` which automatically handles script caching.
- Implement extensions as context managers for proper resource cleanup: `with RateLimiter(queue) as limiter:`.
- Use Python's `typing` module extensively for extension interfaces to improve IDE support and catch errors early.

**Multi-Language Protocol Considerations**:
1. **Serialization**: Use JSON for simplicity or msgpack for efficiency. Ensure all languages handle the same data types consistently (especially datetime).
2. **Error Handling**: Define standard error response format with code, message, and optional details.
3. **Versioning**: Include protocol version in job payload to handle backward compatibility.
4. **Connection Management**: Each client library should implement connection pooling and automatic reconnection.

#### Milestone Checkpoint for Rate Limiting Extension

After implementing rate limiting:

1. **Test Command**: `python -m pytest extensions/rate_limiting/tests/ -v`
2. **Expected Output**: Tests showing token bucket refill, burst handling, and denial with wait times.
3. **Manual Verification**:
   ```python
   # Start worker with rate-limited queue
   limiter = TokenBucketRateLimiter(redis_client)
   # Enqueue 30 jobs to queue limited to 10/sec with burst 20
   # First 20 should process immediately, next 10 at ~1/second
   # Monitor dashboard to see throttling in action
   ```
4. **Signs of Success**: Dashboard shows queue processing rate capped at limit, worker logs show `Waiting 0.1s for rate limit` messages.
5. **Troubleshooting**:
   - If jobs process faster than limit: Check Lua script loading, ensure time is in milliseconds.
   - If Redis CPU spikes: Rate limit checks are too frequent; increase worker polling interval for rate-limited queues.
   - If bursts not working: Verify burst capacity parameter is being passed correctly to Lua script.

#### Debugging Tips for Dependency Extensions

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| Dependent jobs never run | Parent job completion not triggering | Check Redis Pub/Sub subscription, verify `_handle_job_completion` is called | Ensure dependency manager is started, check message format in completion channel |
| Circular dependency deadlock | Graph validation missed a cycle | Run `validate_dag()` on job registration, check Redis for cycles | Implement Kahn's algorithm for cycle detection before accepting job |
| Memory leak with many dependencies | Redis sets growing unbounded | Monitor Redis memory usage, check for orphaned dependency sets | Implement janitor process to clean up completed dependency chains |
| Race condition: job runs before dependencies met | Concurrent completion events | Use Redis transactions (`MULTI/EXEC`) for dependency count updates | Implement optimistic locking with version stamps on dependency counts |

> **Architectural Insight**: The most valuable extensions are those that leverage the existing Redis infrastructure without changing core job execution semantics. Rate limiting and dependencies enhance the system's capabilities while maintaining its simplicity and performance characteristics. More radical extensions (exactly-once processing, multi-broker) essentially create a different system with different trade-offs—these might be better implemented as separate projects that reuse concepts but not code.


## Glossary

> **Milestone(s):** This section spans all five milestones, providing a definitive reference for the technical terms, domain vocabulary, and acronyms used throughout the design document. A shared vocabulary ensures consistent understanding across the team and reduces ambiguity in technical discussions.

### Terms and Definitions

This glossary defines all key terms used in the Background Job Processor system, organized alphabetically. Each term includes a concise definition and its relevance to the system design.

| Term | Definition | Context in System |
|------|------------|-------------------|
| **Active Job** | A job that has been dequeued by a worker and is currently being executed. Represented by the `JobStatus.ACTIVE` status. | Workers transition jobs from `PENDING` to `ACTIVE` when they begin processing. Active jobs appear in the processing queue until completion. |
| **Adaptive Sampling** | Adjusting the frequency of metric collection based on system load to balance observability with performance overhead. | The `MetricsAggregator` may increase sampling during high activity and decrease during idle periods to optimize Redis usage. |
| **Alert Cooldown** | A minimum time period that must pass between consecutive alerts for the same rule to prevent notification spam. | Configured in `AlertRule.cooldown_seconds` to prevent **alert fatigue** when a condition remains unstable. |
| **Alert Fatigue** | The problem where operators begin ignoring alerts due to excessive volume or frequency, reducing the effectiveness of monitoring. | Addressed through **alert cooldown** periods, severity-based filtering, and intelligent aggregation of similar alerts. |
| **Architecture Decision Record (ADR)** | A documented decision capturing the context, options considered, and rationale for a specific architectural choice. | Used throughout this document to explain design choices like Redis data structures, concurrency models, and retry algorithms. |
| **Atomic Pipeline** | A Redis feature that allows multiple commands to be executed as a single atomic operation, ensuring consistency even with concurrent access. | Used by `QueueManager.bulk_enqueue()` and `QueueManager.delete_job()` to prevent race conditions between workers. |
| **Backoff Calculator** | A component (`BackoffCalculator`) that computes the delay between retry attempts using an exponential backoff algorithm with optional jitter. | Called by `RetryManager.handle_job_failure()` to determine when to retry failed jobs. |
| **Blocking Queue Read** | A Redis operation like `BRPOP` that waits (blocks) for items to become available in a queue rather than returning immediately. | Used by workers in their main loop to avoid busy-waiting and reduce Redis load. |
| **Broker** | The central message store that connects job producers and consumers. In our design, Redis serves as the broker. | All job data flows through the Redis broker, which provides persistence, atomic operations, and pub/sub capabilities. |
| **Catch-Up Logic** | A mechanism that processes jobs missed during scheduler downtime by detecting overdue scheduled jobs and enqueuing them immediately. | Implemented in `Scheduler._poll_due_schedules()` which compares current time with `Schedule.next_run_at` to identify missed executions. |
| **Chaos Testing** | Testing system behavior under intentionally injected failures (network partitions, process crashes, etc.) to verify resilience. | Recommended for validating the system's error handling and recovery mechanisms in production-like environments. |
| **Circuit Breaker** | A design pattern (`CircuitBreaker`) that fails fast when a service is unavailable, preventing cascading failures and allowing time for recovery. | Can be applied to external API calls within job handlers to prevent workers from getting stuck on repeatedly failing operations. |
| **Circuit State** | The current state of a circuit breaker: `CLOSED` (normal operation), `OPEN` (failing fast), or `HALF_OPEN` (testing recovery). | Tracked by the `CircuitBreaker.state` field and transitions based on failure thresholds and recovery timeouts. |
| **Completed Job** | A job that has finished execution successfully. Represented by the `JobStatus.COMPLETED` status. | Moved from active to completed storage with execution metrics, available for dashboard queries until archived. |
| **Concurrency** | The number of jobs a worker can process simultaneously, configured via `WorkerConfig.concurrency`. | Implemented using thread or process pools within a worker to handle multiple jobs in parallel without blocking the main loop. |
| **Correlation ID** | A unique identifier passed through the entire job lifecycle for tracing and debugging purposes. | Can be stored in `Job.metadata` to link job execution logs across workers, retries, and external systems. |
| **Cron Expression** | A time-based schedule expression using five fields (minute, hour, day of month, month, day of week) to define recurring execution times. | Parsed by the scheduler to calculate `Schedule.next_run_at` for recurring jobs. Supports standard Unix cron syntax. |
| **DAG (Directed Acyclic Graph)** | A dependency model without cycles, where jobs depend on the completion of other jobs in a directed, non-circular manner. | Future extension: `DependencyManager` could use DAGs to manage complex job workflows with dependencies. |
| **Dashboard Config** | Configuration (`DashboardConfig`) for the web dashboard controlling display filters, refresh intervals, and metric aggregation settings. | Used by the monitoring API to customize dashboard views without changing code. |
| **Daylight Saving Time** | Seasonal time adjustment where clocks are set forward or backward by one hour, affecting local time calculations. | The scheduler handles DST transitions by evaluating cron expressions in the configured timezone using aware datetime objects. |
| **Dead Letter Queue** | Storage (`DeadLetterQueue`) for jobs that have exhausted all retry attempts, allowing manual inspection and potential retry. | Implemented as a Redis sorted set keyed by job ID, with error history preserved for debugging. |
| **Dead Letter Job** | A job that has permanently failed after exceeding its maximum retry count. Represented by the `JobStatus.DEAD_LETTER` status. | Moved to the dead letter queue by `RetryManager.move_to_dead_letter()` and accessible via the dashboard for manual handling. |
| **Diagnostic Worker** | A background process (`DiagnosticWorker`) that periodically collects system health metrics and checks for anomalies. | Runs alongside the main components to provide proactive monitoring and early warning of system issues. |
| **Directed Acyclic Graph** | See **DAG**. | |
| **End-to-End Tests** | Tests that simulate real user scenarios across the entire system, verifying integration between all components. | Example: Enqueue a job → worker processes it → verify result storage → check dashboard metrics update. |
| **Exponential Backoff** | A retry algorithm that doubles the wait time between consecutive retry attempts to prevent overloading systems. | Implemented in `BackoffCalculator.calculate_delay()` using formula: `base_delay * 2^(attempt-1) ± jitter`. |
| **Failed Job** | A job that has finished execution with an error but may still be retried. Represented by the `JobStatus.FAILED` status. | After failure, the job is either scheduled for retry or moved to the dead letter queue based on retry count. |
| **FIFO (First-In-First-Out)** | An ordering principle where the first element added to a queue is the first to be removed. | Default queue behavior implemented using Redis `LPUSH` (add to end) and `BRPOP` (remove from front). |
| **Flaky Test** | A test that passes and fails intermittently without code changes, often due to timing issues, race conditions, or external dependencies. | Mitigated by using deterministic `mock clock` for time-dependent tests and proper isolation of Redis test instances. |
| **Graceful Shutdown** | A shutdown process that allows the current operation to complete before exiting, rather than terminating abruptly. | Workers implement graceful shutdown by catching `SIGTERM`, setting `_shutdown_requested`, and finishing the current job before exiting. |
| **Heartbeat** | A regular status update published by workers to indicate they are alive and functioning. | Implemented by `WorkerHeartbeat` which periodically writes worker status to Redis with a TTL; missing heartbeats indicate crashed workers. |
| **Idempotent** | A property where repeated execution of an operation produces the same result as a single execution. | Critical for job handlers that may be retried; designers should ensure jobs are idempotent where possible. |
| **Idempotency Key** | A unique identifier used to prevent duplicate processing of the same logical operation. | Future extension: Could be stored in `Job.metadata` to deduplicate identical jobs enqueued multiple times. |
| **Integration Tests** | Tests that verify interactions between multiple components with real dependencies (like Redis). | Use `RedisTestContext` to spin up isolated Redis instances for testing queue operations, worker coordination, and retry logic. |
| **Janitor Process** | A maintenance process (`Janitor`) that cleans up stale jobs from processing queues when workers crash unexpectedly. | Periodically scans for jobs in processing queues whose worker heartbeat has expired and requeues them. |
| **Job** | The unit of work to be processed asynchronously, represented by the `Job` class with fields for type, arguments, metadata, and status. | The fundamental data entity that flows through the system from enqueue to completion or failure. |
| **Job Handler** | A function registered with a worker to execute the business logic for a specific job type. | Registered via `Worker.register_handler()`; invoked by the worker when it dequeues a job of the matching type. |
| **Job History Archival** | The process of moving old job data from Redis to external storage (like a database) to prevent unbounded memory growth. | Implemented by `Janitor._cleanup_expired_keys()` which removes completed jobs after a configurable retention period. |
| **Job Status** | The current state of a job, represented by the `JobStatus` enum: `PENDING`, `ACTIVE`, `COMPLETED`, `FAILED`, `RETRY_SCHEDULED`, `DEAD_LETTER`. | Tracked throughout the job lifecycle and used for filtering in the dashboard and determining next actions. |
| **Job Summary** | A condensed view of job information (`JobSummary`) used for display in the dashboard, containing key fields without full payload. | Created from `Job` instances for efficient dashboard queries that don't require the complete job serialization. |
| **Jitter** | Random variation added to retry delays to prevent synchronization of multiple retrying jobs (thundering herd). | Applied by `BackoffCalculator` using a configurable `jitter_factor` that adds ± random percentage to calculated delays. |
| **Lua Script** | Redis server-side scripting language that allows atomic execution of multiple commands on the Redis server. | Used for complex atomic operations like `TokenBucketRateLimiter.check_limit()` that require multiple Redis commands. |
| **Metric Aggregation** | The process of combining raw job events into statistical summaries (counts, rates, averages) for monitoring. | Performed by `MetricsAggregator` which periodically reads Redis streams and computes aggregated metrics for the dashboard. |
| **Metric Point** | A single metric data point (`MetricPoint`) with timestamp, name, value, and labels for dimensional analysis. | The basic unit of metric storage; aggregated into time series for dashboard charts and alert evaluation. |
| **Mock Clock** | A simulated time control used in tests to manipulate and advance time without waiting for real clock progression. | Essential for testing retry delays, scheduler cron evaluation, and time-based job expiration without real-world delays. |
| **Pending Job** | A job that has been enqueued but not yet picked up by a worker. Represented by the `JobStatus.PENDING` status. | Stored in Redis lists (queues) waiting for worker consumption. Dashboard shows pending count per queue. |
| **Polling Interval** | The time period between scheduler checks for due jobs, configured via `Scheduler.polling_interval_seconds`. | Balances schedule accuracy with Redis load; shorter intervals increase timeliness but increase Redis queries. |
| **Priority Weighted Polling** | A worker polling strategy that checks higher-priority queues more frequently than lower-priority ones based on configured weights. | Implemented by workers using algorithm that selects queues probabilistically based on `QueueConfig.priority` values. |
| **Processing Queue** | A temporary Redis list holding jobs currently being executed by a worker, used for reliability during worker crashes. | When a worker dequeues a job, it moves it from the main queue to a processing queue; on completion, removes it. |
| **Property-Based Testing** | Testing by generating random inputs that must satisfy certain properties, rather than testing specific predefined cases. | Useful for validating job serialization/deserialization invariants and queue operation correctness across diverse inputs. |
| **Protocol Versioning** | A technique for maintaining backward compatibility in APIs by including version identifiers in messages or endpoints. | Future extension: Could be added to job serialization format to allow evolution of `Job` fields without breaking existing jobs. |
| **Queue** | A named channel for jobs with configurable priority, implemented as a Redis list with associated metadata. | Defined by `QueueConfig`; workers subscribe to specific queues and poll them according to priority weights. |
| **Queue Config** | Configuration (`QueueConfig`) for a named queue including its priority and optional maximum length. | Loaded into `QueueManager.queue_configs` to control queue behavior and worker polling strategy. |
| **Queue Full Error** | An exception (`QueueFullError`) raised when attempting to enqueue a job to a queue that has reached its maximum capacity. | Prevented by `QueueManager._validate_job()` checking queue length against `QueueConfig.max_length` before enqueue. |
| **Queue Manager** | The component (`QueueManager`) responsible for validating, enqueuing, and managing jobs in Redis queues. | Provides the primary API for producers to submit jobs and inspect queue state. |
| **Queue Metrics** | Performance metrics for a specific queue (`QueueMetrics`) including depth, processing rate, error rate, and age statistics. | Computed by `MetricsAggregator` and displayed in the dashboard's queue overview section. |
| **Race Condition** | A bug where the outcome depends on the timing of concurrent operations, often leading to inconsistent state. | Mitigated throughout the system using Redis atomic operations, pipelines, and Lua scripts for critical sections. |
| **Real-Time Dashboard** | A web interface that updates without page refresh, showing current system state using WebSocket or SSE connections. | Implemented with Server-Sent Events (SSE) streaming metric updates to the browser as they're collected. |
| **Redis Client** | A wrapper (`RedisClient`) around the Redis connection that provides error handling, connection pooling, and command execution. | Used by all components to interact with Redis; ensures consistent connection management and error recovery. |
| **Redis Config** | Configuration (`RedisConfig`) for Redis connection parameters: URL, timeouts, retry behavior, and connection pool size. | Loaded into `SystemConfig.redis` and used to initialize `RedisClient` instances. |
| **Redis Streams** | A Redis data type that provides an append-only log structure with consumer groups for reliable message processing. | Used for job event logging (completions, failures) that feed into the `MetricsAggregator` for real-time metrics. |
| **Redis Test Context** | A test utility (`RedisTestContext`) that manages isolated Redis instances for integration testing. | Provides clean Redis databases for each test case and ensures proper cleanup after tests complete. |
| **Retry Filter** | Logic to determine if a specific error type should bypass retries and move directly to the dead letter queue. | Future extension: Could be implemented as a configurable mapping of exception types to retry policies in `RetryManager`. |
| **Retry Manager** | The component (`RetryManager`) that handles job failures, schedules retries with exponential backoff, and manages the dead letter queue. | Centralized error handling logic invoked by workers when jobs fail during execution. |
| **Retry Scheduled Job** | A job that has failed and been scheduled for retry at a future time. Represented by the `JobStatus.RETRY_SCHEDULED` status. | Stored in Redis sorted set with execution timestamp as score; periodically moved back to main queue by scheduler. |
| **Scheduler** | The component (`Scheduler`) that enqueues jobs for future or recurring execution based on cron expressions or specific timestamps. | Runs as a separate process that polls for due schedules and enqueues jobs into the appropriate worker queues. |
| **Schedule Status** | The current state of a scheduled job execution: `PENDING`, `ENQUEUED`, `SKIPPED`, or `ERROR`. | Tracked in `ScheduledJob.status` to monitor scheduler operation and detect issues with schedule execution. |
| **Sensitive Data Redaction** | The process of masking confidential information (like passwords, tokens) in job logs and dashboard displays. | Implemented in dashboard API by filtering sensitive fields from `Job.metadata` before returning to clients. |
| **Server-Sent Events (SSE)** | An HTTP-based technology for server-to-client real-time updates, simpler than WebSockets for unidirectional data flow. | Used by the dashboard to stream real-time metric updates from server to browser without polling. |
| **SIGTERM** | A Unix signal sent to a process to request graceful termination (as opposed to SIGKILL which forces immediate termination). | Workers catch SIGTERM to initiate graceful shutdown, allowing current job to complete before exiting. |
| **Split-Brain** | A scenario in distributed systems where a network partition causes components to operate independently, potentially leading to data inconsistency. | Mitigated in our design by using Redis as a single source of truth; network partition would prevent all components from operating. |
| **Statistical Assertion** | A test verification that holds true over many random trials, rather than for a single specific execution. | Used in property-based testing to validate system properties like "no job is lost" across many random job sequences. |
| **System Alert** | A notification of a system condition requiring attention (`SystemAlert`), with severity, message, and timestamps. | Generated by `AlertManager` when alert rules evaluate to true, displayed in dashboard and potentially sent via external channels. |
| **System Config** | The root configuration object (`SystemConfig`) containing all subsystem configurations: Redis, queues, workers, and global settings. | Loaded from environment variables via `SystemConfig.from_env()` and passed to all components during initialization. |
| **System Metrics** | Comprehensive system health metrics (`SystemMetrics`) including CPU, memory, Redis usage, queue depths, and worker counts. | Collected by `DiagnosticWorker` for system-level monitoring beyond job-specific metrics. |
| **Test Fixture** | Reusable setup and teardown code for tests that establishes a consistent initial state for test execution. | Provided by `RedisTestContext` which sets up and tears down Redis instances for integration tests. |
| **Thundering Herd** | A problem where many retried jobs simultaneously become ready and overwhelm the system when they retry at the same time. | Prevented by adding **jitter** to retry delays, staggering retry times across different jobs. |
| **Timezone** | A geographical region's standard time, used for evaluating cron expressions in the correct local context. | Configured per schedule via `Schedule.timezone`; cron expressions are evaluated in this timezone rather than UTC only. |
| **Token Bucket** | A rate limiting algorithm that allows bursts up to a capacity while maintaining a steady average rate over time. | Implemented by `TokenBucketRateLimiter` for future extension to limit job enqueue rates per producer. |
| **TTL (Time-To-Live)** | An expiration time set on Redis keys to automatically remove data after a period, preventing memory bloat. | Applied to worker heartbeats, metric data, and completed job records to ensure Redis memory doesn't grow unbounded. |
| **Two-Tiered Storage** | An architecture with fast cache (Redis) for recent data and slow storage (database) for historical data. | Future extension: Older job history could be moved from Redis to a database while keeping recent data in Redis for fast access. |
| **ULID (Universally Unique Lexicographically Sortable Identifier)** | An identifier format that combines timestamp (lexicographically sortable) with randomness for uniqueness. | Alternative to UUID for job IDs; provides natural chronological ordering when sorted lexicographically. |
| **Uniqueness Window** | A time period during which duplicate enqueueing of the same scheduled job is prevented by the scheduler. | Configured via `Schedule.unique_window_seconds`; prevents multiple instances of the same recurring job within the window. |
| **Validation Error** | An exception (`ValidationError`) raised when job validation fails, such as payload exceeding size limits or missing required fields. | Thrown by `QueueManager._validate_job()` before enqueue to prevent malformed jobs from entering the system. |
| **Worker** | A process (`Worker`) that fetches jobs from queues and executes them using registered handlers, with configurable concurrency. | The core processing unit of the system; multiple workers can run simultaneously for horizontal scaling. |
| **Worker Config** | Configuration (`WorkerConfig`) for a worker process: subscribed queues, concurrency, heartbeat interval, and job timeout. | Passed to `Worker` instance at startup; determines which queues the worker polls and how many jobs it processes concurrently. |
| **Worker Heartbeat** | A component (`WorkerHeartbeat`) that periodically updates worker liveness in Redis, allowing detection of crashed workers. | Runs in a separate thread within each worker process; updates a Redis key with TTL; missing updates indicate worker failure. |
| **Worker Status** | Current status of a worker (`WorkerStatus`): active/inactive, current job, processed counts, and last heartbeat time. | Computed by the monitoring API from worker heartbeats and active job tracking; displayed in the dashboard's worker view. |
| **YAGNI (You Ain't Gonna Need It)** | A principle of not implementing features until they are actually needed, to avoid unnecessary complexity. | Applied throughout the design: features like job dependencies, rate limiting, and multi-language support are marked as future extensions. |
| **Zombie Job** | A job stuck in a processing queue with no active worker processing it, typically because the worker crashed. | Cleaned up by the **janitor process** which detects stale processing queues and requeues their jobs. |
```

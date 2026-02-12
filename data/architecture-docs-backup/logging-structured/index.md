# Structured Logging System: Design Document


## Overview

A production-grade structured logging system that provides thread-safe, multi-level logging with JSON output and distributed request tracing. The key architectural challenge is designing a flexible handler dispatch system that maintains performance while supporting context propagation across async boundaries.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** This section provides foundational context for all milestones by establishing why structured logging is essential and what challenges it solves in production environments.

### The Airport Control Tower Analogy

Think of traditional logging in software systems like early aviation communication — pilots would occasionally radio in their status using informal, unstructured messages: "Approaching from the southeast, low on fuel, weather looks rough." This worked when there were only a few planes in the sky, but as air traffic grew exponentially, this ad-hoc communication created chaos. Controllers couldn't quickly filter messages by urgency, correlate related communications from the same flight, or efficiently route information to the right teams.

Modern air traffic control solved this through **structured communication protocols**. Every radio transmission follows a standardized format: aircraft identification, location coordinates, altitude, heading, fuel status, and request type — all in predictable fields that controllers can instantly parse, filter, and route. A controller can immediately spot all "low fuel" situations, trace the complete journey of flight AA1234, or correlate weather reports with affected routes.

This transformation mirrors exactly what happens when software systems evolve from traditional logging to structured logging. In small applications with a single developer, informal log messages work fine: "User login failed", "Database connection lost", "Processing order 12345". But in production environments with hundreds of services, millions of requests, and distributed teams, these unstructured logs become as chaotic as those early aviation radio calls.

**Structured logging** transforms each log entry into a standardized "flight plan" with consistent fields: timestamp, severity level, service name, request ID, user ID, operation type, and contextual metadata. Operations teams can instantly filter by error severity, trace all actions for a specific user request, correlate events across multiple services, and route alerts to the appropriate team — just like air traffic controllers managing complex airspace.

The analogy extends to **correlation IDs** (like flight numbers that persist across different control zones), **log levels** (like priority codes for emergency vs. routine communications), and **context propagation** (like flight plans that follow aircraft across multiple control centers). Without this structure, debugging production issues becomes like searching for a specific airplane in busy airspace using only fragments of overheard conversations.

### Traditional vs Structured Logging

The fundamental difference between traditional and structured logging lies in **queryability** and **context preservation**. Traditional logging treats each log message as an opaque string, while structured logging treats it as a data record with well-defined fields that can be indexed, filtered, and aggregated.

| Aspect | Traditional String Logs | Structured JSON Logs | Impact on Operations |
|--------|------------------------|---------------------|---------------------|
| **Format** | `"2024-01-15 ERROR: Login failed for user john_doe"` | `{"timestamp": "2024-01-15T10:30:00Z", "level": "ERROR", "event": "login_failed", "user_id": "john_doe", "ip_address": "192.168.1.100"}` | Structured logs enable precise queries like "all login failures from IP range X" |
| **Parsing** | Requires regex patterns, brittle text parsing | Direct JSON deserialization into typed objects | 10x faster log ingestion, no parsing errors |
| **Searchability** | Full-text search only: `grep "Login failed" logfile` | Field-specific queries: `level:ERROR AND event:login_failed` | Precise filtering reduces noise from millions of log entries |
| **Context Linking** | Manual correlation using scattered string fragments | Automatic correlation via request_id field | Complete request trace across 20+ microservices |
| **Alerting Rules** | Complex regex patterns: `/ERROR.*database.*timeout/` | Simple field matching: `level:ERROR AND component:database AND error_type:timeout` | Fewer false positives, faster alert setup |
| **Aggregation** | Impossible without custom parsing scripts | Built-in: `GROUP BY error_type`, `COUNT by user_id` | Real-time dashboards showing error trends |
| **Multi-line Handling** | Stack traces break log parsers, require special handling | Stack trace stored as structured field, never breaks parsing | Reliable log shipping even with complex Java exceptions |
| **Performance** | String concatenation, formatting overhead | Direct object serialization, minimal string operations | 50% reduction in logging CPU overhead |
| **Schema Evolution** | Adding fields breaks existing parsers | New fields automatically available, backward compatible | Zero-downtime log format improvements |
| **Debugging Workflow** | `grep` → `awk` → manual correlation → spreadsheet analysis | Direct database queries → automatic correlation → interactive dashboards | 5 minutes instead of 2 hours to diagnose issues |

The transition from traditional to structured logging parallels the evolution from manually parsing log files to treating logs as a **time-series database**. Each log entry becomes a data point with dimensions (service, user, operation) and metrics (response time, error rate, resource usage) that can be aggregated, visualized, and analyzed in real-time.

> **Critical Insight**: The value of structured logging grows exponentially with system complexity. A monolithic application with 100 daily requests might not benefit significantly, but a microservices architecture handling 10,000 requests per second across 50 services becomes impossible to debug without structured, correlated logs.

### Production Environment Challenges

Production logging systems face four critical challenges that simple print statements or basic file logging cannot address: **thread safety**, **performance at scale**, **distributed correlation**, and **operational reliability**. Each challenge requires specific architectural decisions that shape the entire logging system design.

#### Thread Safety and Concurrent Access

In production environments, multiple threads simultaneously generate log messages, creating **race conditions** that can corrupt output, interleave partial messages, or cause deadlocks. The challenge extends beyond simple file writing to include context management, formatter state, and handler dispatch.

| Challenge | Manifestation | Technical Requirement |
|-----------|---------------|----------------------|
| **Interleaved Output** | Two threads writing "User login" and "Database query" produce garbled output: `"User Datalogbase quin"` | Atomic write operations with message-level locking |
| **Corrupted JSON** | Concurrent JSON serialization creates invalid output: `{"level":"INFO""level":"ERROR"}` | Thread-safe formatter instances or per-thread serialization buffers |
| **Context Confusion** | Thread A's user_id appears in Thread B's log messages | Thread-local context storage with proper isolation boundaries |
| **Handler State** | File handles, network connections shared unsafely between threads | Synchronized handler operations or thread-safe I/O abstractions |
| **Deadlock Risk** | Complex locking hierarchies between logger, handlers, and formatters | Careful lock ordering and non-blocking dispatch patterns |

The thread safety challenge is compounded by **async/await patterns** in modern applications. When an async task yields control, its logging context must be preserved and restored correctly, even when the task resumes on a different thread. This requires sophisticated context propagation mechanisms that go beyond simple thread-local storage.

#### Performance at Scale

Production logging systems must handle thousands of log messages per second while adding minimal overhead to application performance. The challenge involves both **computational efficiency** (CPU cycles for formatting and serialization) and **I/O efficiency** (disk writes and network transmission).

| Scale Factor | Performance Requirement | Design Implication |
|--------------|------------------------|-------------------|
| **Message Volume** | 10,000+ messages/second per service | Non-blocking handler dispatch, async I/O operations |
| **Context Size** | 50+ fields per message (user, request, trace data) | Efficient context merging, lazy field evaluation |
| **Handler Count** | 5+ destinations per message (file, stdout, metrics, remote) | Parallel handler execution, failure isolation |
| **Memory Pressure** | Minimal allocation in hot logging path | Object pooling, pre-allocated buffers, zero-copy serialization |
| **Latency Sensitivity** | <1ms added latency to business logic | Background processing, batched network operations |

The performance challenge creates tension between **completeness** and **speed**. Adding rich context improves debugging capability but increases serialization cost. The solution requires **lazy evaluation** strategies where expensive operations (like stack trace capture or large object serialization) only occur when the log level threshold is met.

> **Design Principle**: The hot path for logging (from application call to handler dispatch) must remain synchronous and fast, while expensive operations (network transmission, disk syncing, complex formatting) should be asynchronous and non-blocking.

#### Distributed Correlation Across Services

Modern applications span multiple services, each generating independent log streams. The critical challenge is **correlating related events** across service boundaries to reconstruct complete user request flows or diagnose cross-service failures.

| Correlation Scenario | Technical Challenge | Required Infrastructure |
|---------------------|-------------------|------------------------|
| **User Request Trace** | HTTP request flows through 8 services, each logs independently | Correlation ID propagation via HTTP headers and message queues |
| **Database Transaction** | Single transaction touches 3 services, fails in service 2 | Transaction ID preservation across service calls and rollbacks |
| **Background Job Chain** | Job spawns 5 child jobs across different workers | Parent-child job ID relationships with hierarchical context |
| **Error Cascade** | Timeout in service A causes errors in services B, C, D | Temporal correlation with root cause identification |
| **Performance Investigation** | Slow response affects multiple upstream services | Request timing correlation with service dependency mapping |

The correlation challenge requires **context propagation protocols** that work across different transport mechanisms (HTTP headers, message queue properties, gRPC metadata) and programming languages. The logging system must automatically extract correlation IDs from incoming requests and inject them into outgoing requests without requiring explicit developer intervention.

#### Log Aggregation and Operational Requirements

Production environments generate terabytes of log data daily, requiring **reliable collection**, **efficient transport**, and **fault-tolerant storage**. The logging system must handle network failures, disk space exhaustion, and downstream service outages without losing critical diagnostic information.

| Operational Challenge | Failure Mode | Recovery Mechanism |
|-----------------------|--------------|-------------------|
| **Network Partition** | Remote log collector becomes unreachable for 30 minutes | Local buffering with disk spillover, automatic retry with exponential backoff |
| **Disk Space Exhaustion** | Log files consume all available disk space | Automatic log rotation, compression, oldest-first deletion with retention policies |
| **Handler Failure** | Elasticsearch cluster goes down, losing audit logs | Multi-destination dispatch with independent failure domains |
| **Configuration Errors** | Invalid JSON formatter crashes logging pipeline | Graceful degradation to plain text output, error isolation per handler |
| **Memory Pressure** | Log buffering consumes excessive memory during bursts | Backpressure mechanisms, emergency log level elevation, buffer size limits |

The aggregation challenge extends to **log format evolution**. Production systems must support gradual rollouts of new log fields without breaking existing parsing infrastructure. This requires **schema compatibility strategies** and **version negotiation protocols** between log producers and consumers.

> **Operational Reality**: In production environments, the logging system itself becomes a critical piece of infrastructure that must be monitored, alerted on, and maintained. Logger failures can mask application issues, making robust error handling and self-monitoring essential design requirements.

These production challenges drive the architectural decisions throughout the logging system design. **Thread safety** requirements shape the locking strategy and context isolation mechanisms. **Performance constraints** determine the handler dispatch architecture and serialization approaches. **Correlation needs** influence the context propagation design and metadata extraction patterns. **Operational requirements** drive error handling strategies and configuration management approaches.

The next sections will show how each challenge translates into specific design decisions, with Architecture Decision Records (ADRs) documenting the rationale behind each choice and the trade-offs involved.

### Implementation Guidance

This implementation guidance provides practical foundations for building the structured logging system, including technology recommendations and starter infrastructure that supports the design principles established above.

#### A. Technology Recommendations

| Component | Simple Option | Advanced Option | Rationale |
|-----------|---------------|-----------------|-----------|
| **JSON Serialization** | `json` standard library | `orjson` or `ujson` for performance | Standard library provides reliability; advanced options needed for >1000 msgs/sec |
| **Thread Safety** | `threading.Lock()` with context managers | `concurrent.futures` with thread pools | Simple locks work for basic scenarios; thread pools handle async dispatch better |
| **Context Storage** | `threading.local()` for thread contexts | `contextvars` for async-compatible contexts | Thread-local sufficient for sync apps; contextvars required for asyncio applications |
| **File I/O** | Built-in `open()` with manual rotation | `logging.handlers.RotatingFileHandler` adapted | Manual control for learning; existing handlers provide production-ready features |
| **Network Transport** | `requests` library for HTTP endpoints | `aiohttp` for async network operations | Synchronous requests for simple cases; async required for non-blocking performance |
| **Configuration** | Python dictionaries and environment variables | `pydantic` for validated configuration schemas | Dict-based config sufficient for core functionality; validation prevents runtime errors |

#### B. Recommended File/Module Structure

Organize the logging system with clear separation of concerns, making each component testable and maintainable:

```
structured_logging/
├── __init__.py                    # Public API exports
├── core/
│   ├── __init__.py
│   ├── logger.py                  # Logger class and hierarchy management
│   ├── levels.py                  # Log level definitions and filtering
│   ├── records.py                 # LogRecord data structure and factory
│   └── registry.py                # Logger registry and configuration
├── handlers/
│   ├── __init__.py
│   ├── base.py                    # Abstract handler interface
│   ├── console.py                 # Stdout/stderr output handlers
│   ├── file.py                    # File-based output with rotation
│   └── remote.py                  # HTTP/network destination handlers
├── formatters/
│   ├── __init__.py
│   ├── base.py                    # Abstract formatter interface
│   ├── json_formatter.py          # Single-line JSON output
│   └── pretty_formatter.py        # Human-readable colored output
├── context/
│   ├── __init__.py
│   ├── manager.py                 # Context storage and propagation
│   ├── correlation.py             # Correlation ID generation and tracking
│   └── middleware.py              # Request context extraction
├── utils/
│   ├── __init__.py
│   ├── serialization.py           # Safe JSON serialization utilities
│   └── threading_utils.py         # Thread-safe helper functions
└── tests/
    ├── unit/                      # Component-specific tests
    ├── integration/               # Multi-component scenarios
    └── performance/               # Scalability and benchmark tests
```

#### C. Infrastructure Starter Code

**Thread-Safe Utilities (`utils/threading_utils.py`):**

```python
import threading
import time
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager

class ThreadSafeCounter:
    """Thread-safe counter for correlation ID generation."""
    
    def __init__(self, start: int = 0):
        self._value = start
        self._lock = threading.Lock()
    
    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

class ThreadSafeDict:
    """Thread-safe dictionary wrapper for shared logger registry."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
    
    def keys(self):
        with self._lock:
            return list(self._data.keys())

@contextmanager
def timeout_context(seconds: float):
    """Context manager for operations that must complete within timeout."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if elapsed > seconds:
            print(f"Warning: Operation took {elapsed:.3f}s (timeout: {seconds}s)")

def safe_call(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """Safely call function, returning default on any exception."""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default
```

**JSON Serialization Utilities (`utils/serialization.py`):**

```python
import json
import datetime
import uuid
from typing import Any, Dict, Set
from decimal import Decimal

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serializable types safely."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, datetime.date):
            return obj.isoformat()
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            # Custom objects - serialize their __dict__
            return obj.__dict__
        elif isinstance(obj, Exception):
            # Exception objects - capture essential info
            return {
                'type': obj.__class__.__name__,
                'message': str(obj),
                'args': obj.args
            }
        else:
            # Fallback to string representation
            return str(obj)

def safe_serialize(data: Dict[str, Any], max_depth: int = 10) -> str:
    """
    Safely serialize dictionary to JSON, handling circular references
    and limiting depth to prevent infinite recursion.
    """
    def clean_dict(obj: Any, depth: int = 0, seen: Set[id] = None) -> Any:
        if seen is None:
            seen = set()
        
        if depth > max_depth:
            return f"<max_depth_exceeded:{type(obj).__name__}>"
        
        if id(obj) in seen:
            return f"<circular_reference:{type(obj).__name__}>"
        
        if isinstance(obj, dict):
            seen.add(id(obj))
            result = {}
            for k, v in obj.items():
                try:
                    result[str(k)] = clean_dict(v, depth + 1, seen.copy())
                except Exception:
                    result[str(k)] = f"<serialization_error:{type(v).__name__}>"
            return result
        elif isinstance(obj, (list, tuple)):
            seen.add(id(obj))
            return [clean_dict(item, depth + 1, seen.copy()) for item in obj]
        else:
            return obj
    
    cleaned_data = clean_dict(data)
    return json.dumps(cleaned_data, cls=SafeJSONEncoder, separators=(',', ':'))

def estimate_serialized_size(data: Dict[str, Any]) -> int:
    """Estimate JSON size without full serialization (for performance)."""
    # Quick size estimation for memory management
    size = 2  # Opening and closing braces
    for key, value in data.items():
        size += len(str(key)) + 4  # Key + quotes + colon + comma
        if isinstance(value, str):
            size += len(value) + 2  # String + quotes
        elif isinstance(value, (int, float)):
            size += len(str(value))
        elif isinstance(value, dict):
            size += estimate_serialized_size(value)
        else:
            size += 20  # Rough estimate for other types
    return size
```

#### D. Core Logic Skeleton Code

**Base Logger Class (`core/logger.py`):**

```python
from typing import Any, Dict, List, Optional, Union
from threading import Lock
import time

class Logger:
    """
    Core logger class that handles message filtering, context enrichment,
    and dispatch to multiple handlers.
    """
    
    def __init__(self, name: str, level: int = 20, parent: Optional['Logger'] = None):
        self.name = name
        self.level = level
        self.parent = parent
        self.handlers: List['Handler'] = []
        self.children: Dict[str, 'Logger'] = {}
        self._lock = Lock()
    
    def log(self, level: int, message: str, **context) -> None:
        """
        Main logging method that processes and dispatches log records.
        
        Args:
            level: Numeric log level (DEBUG=10, INFO=20, etc.)
            message: Human-readable log message
            **context: Additional key-value pairs to include in log record
        """
        # TODO 1: Check if this log level meets the minimum threshold
        #   Compare 'level' parameter against self.level
        #   If level < self.level, return early (message filtered out)
        
        # TODO 2: Create LogRecord object with all required fields
        #   Include: timestamp, level, message, logger_name, context
        #   Use time.time() for timestamp, convert to ISO format
        
        # TODO 3: Enrich record with context from parent loggers
        #   Walk up parent chain, merging context from each level
        #   Child context should override parent context for same keys
        
        # TODO 4: Enrich record with thread-local context
        #   Get current thread's context from ContextManager
        #   Merge with record context (record context takes precedence)
        
        # TODO 5: Dispatch record to all handlers (thread-safely)
        #   Acquire self._lock before accessing self.handlers
        #   Call handle() method on each handler
        #   Continue dispatching even if some handlers fail
        
        # TODO 6: If no handlers at this level, propagate to parent
        #   Check if self.handlers is empty and self.parent exists
        #   Call self.parent.handle_record(record) to propagate up
        
        pass  # Implementation goes here
    
    def debug(self, message: str, **context) -> None:
        """Log DEBUG level message."""
        # TODO: Call self.log() with DEBUG level (value: 10)
        pass
    
    def info(self, message: str, **context) -> None:
        """Log INFO level message.""" 
        # TODO: Call self.log() with INFO level (value: 20)
        pass
    
    def warn(self, message: str, **context) -> None:
        """Log WARN level message."""
        # TODO: Call self.log() with WARN level (value: 30)
        pass
    
    def error(self, message: str, **context) -> None:
        """Log ERROR level message."""
        # TODO: Call self.log() with ERROR level (value: 40)
        pass
    
    def fatal(self, message: str, **context) -> None:
        """Log FATAL level message."""
        # TODO: Call self.log() with FATAL level (value: 50)
        pass
```

#### E. Language-Specific Hints

**Python-Specific Implementation Tips:**

- **Thread Safety**: Use `threading.Lock()` with `with` statements for automatic lock management: `with self._lock: # critical section`
- **Context Variables**: For async compatibility, prefer `contextvars.ContextVar` over `threading.local()` when targeting Python 3.7+
- **JSON Performance**: The standard `json` library is sufficient for most cases. Use `separators=(',', ':')` to eliminate whitespace in output
- **Time Formatting**: Use `datetime.utcnow().isoformat() + 'Z'` for ISO 8601 timestamps with UTC timezone
- **Exception Handling**: Wrap handler calls in `try/except` blocks to prevent one failing handler from stopping others
- **Type Hints**: Use `typing` module extensively for better IDE support: `from typing import Dict, List, Optional, Any`
- **Environment Variables**: Use `os.environ.get('LOG_LEVEL', 'INFO')` for configuration with sensible defaults

**Performance Considerations:**

- **String Formatting**: Use f-strings for message formatting: `f"User {user_id} performed {action}"` (fastest in Python 3.6+)
- **Dictionary Merging**: Use `{**parent_context, **child_context}` for efficient context merging
- **Level Checking**: Implement early return for filtered messages to avoid expensive context gathering
- **Object Pooling**: For high-throughput scenarios, consider reusing LogRecord objects to reduce garbage collection pressure

#### F. Milestone Checkpoint

After implementing the basic logging foundation, verify correct behavior with these checkpoints:

**Checkpoint 1: Basic Logging Works**
```bash
python -c "
from structured_logging import get_logger
logger = get_logger('test')
logger.info('Hello, structured logging!', user_id=123)
"
```
Expected output (to stdout):
```json
{"timestamp":"2024-01-15T10:30:00.123Z","level":"INFO","logger":"test","message":"Hello, structured logging!","user_id":123}
```

**Checkpoint 2: Level Filtering Works**
```python
logger = get_logger('test')
logger.set_level('WARN')  # Should filter out DEBUG and INFO
logger.debug('This should not appear')
logger.info('This should not appear') 
logger.warn('This should appear')
```
Expected: Only the WARN message appears in output.

**Checkpoint 3: Thread Safety Works**
```python
import threading
logger = get_logger('thread_test')

def log_worker(worker_id):
    for i in range(100):
        logger.info(f'Message {i}', worker_id=worker_id)

threads = [threading.Thread(target=log_worker, args=[i]) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```
Expected: All 500 messages appear as valid JSON (no interleaved/corrupted output).

**Signs Something Is Wrong:**

| Symptom | Likely Cause | What to Check |
|---------|--------------|---------------|
| No output appears | Handler not attached or level too high | Verify handler registration and log level settings |
| Garbled JSON output | Thread safety issues | Check that handler writes are atomic and properly locked |
| Missing context fields | Context not properly merged | Verify context propagation from parents and thread-local storage |
| Performance degradation | Blocking I/O in handlers | Move file/network operations to background threads |
| Import errors | Circular dependencies | Review module import structure, use delayed imports if needed |

This foundation provides the infrastructure needed to implement the core logging system while maintaining the thread safety, performance, and reliability requirements established in the problem statement.


## Goals and Non-Goals

> **Milestone(s):** This section defines the scope and success criteria for all three milestones, establishing clear boundaries for what the logging system must accomplish and what it deliberately excludes.

Before diving into technical requirements, it's essential to establish a clear mental model for what we're building. Think of our structured logging system as **the nervous system of a distributed application**. Just as your nervous system carries signals from every part of your body to your brain, providing context about what's happening and allowing you to respond appropriately, our logging system carries structured information from every component of our application to monitoring systems, providing the observability needed to understand, debug, and optimize system behavior.

Unlike traditional logging, which is like having a collection of disconnected sensors that each speak their own language, structured logging creates a unified communication protocol. Every log message follows the same format, carries consistent metadata, and can be correlated with related messages across the entire system. This transforms debugging from archaeological excavation through text files into surgical precision with queryable, contextual data.

The challenge in defining goals for a logging system lies in balancing comprehensiveness with focus. Logging touches every aspect of application development—from local debugging to production monitoring, from performance analysis to security auditing. We must be precise about what problems we're solving while explicitly acknowledging what we're not building to avoid scope creep and maintain architectural clarity.

### Functional Requirements

The functional requirements define **what** our logging system must do—the concrete behaviors and capabilities that users will interact with directly. These requirements map directly to our three milestones and form the acceptance criteria for successful implementation.

**Log Level Management and Filtering** forms the foundation of our system. The logging system must support five distinct log levels with numeric ordering: `DEBUG` (10), `INFO` (20), `WARN` (30), `ERROR` (40), and `FATAL` (50). This hierarchical system allows developers to control the verbosity of their applications by setting a minimum log level threshold. When a logger is configured with level `WARN`, it suppresses all `DEBUG` and `INFO` messages while allowing `WARN`, `ERROR`, and `FATAL` messages to pass through.

The system must support **runtime configuration changes** without requiring application restarts. A production service running with `INFO` level should be able to dynamically switch to `DEBUG` level when troubleshooting issues, then return to `INFO` level to reduce log volume. This capability is crucial for production debugging scenarios where restarting the application would eliminate the problematic state we're trying to diagnose.

| Log Level | Numeric Value | Purpose | Typical Volume |
|-----------|---------------|---------|----------------|
| `DEBUG` | 10 | Detailed diagnostic information for development | Very High |
| `INFO` | 20 | General application flow and state changes | Moderate |
| `WARN` | 30 | Potentially problematic situations | Low |
| `ERROR` | 40 | Error conditions that don't stop the application | Very Low |
| `FATAL` | 50 | Critical errors that may cause application termination | Extremely Rare |

**Structured JSON Output** transforms traditional string-based logging into queryable, machine-readable data. Every log record must serialize as valid, single-line JSON containing consistent field names and data types. This structured format enables powerful querying capabilities in log aggregation systems—instead of searching for text patterns, operators can filter by exact field values, perform numeric comparisons on timestamps, and aggregate data across multiple dimensions.

The JSON output must include these mandatory fields in every log record:

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `timestamp` | string | ISO 8601 formatted timestamp with millisecond precision | "2024-01-15T14:30:25.123Z" |
| `level` | string | Human-readable log level name | "ERROR" |
| `message` | string | Primary log message content | "Database connection failed" |
| `logger_name` | string | Hierarchical name identifying the logger | "service.database.connection" |
| `context` | object | Key-value pairs providing additional metadata | {"user_id": "12345", "request_id": "req_abc123"} |

**Multi-Destination Handler Dispatch** enables log records to flow simultaneously to multiple output destinations. A single log statement might write to local stdout for immediate developer feedback, append to a rotating file for local persistence, and transmit to a remote log collector for centralized aggregation. Each destination operates independently—a failure in one handler must not affect others or block the logging operation.

The handler dispatch mechanism must support these core handler types:

| Handler Type | Purpose | Failure Behavior | Configuration |
|-------------|---------|------------------|---------------|
| Console Handler | Real-time output to stdout/stderr | Log to stderr and continue | Format selection (JSON/pretty) |
| File Handler | Local file persistence with rotation | Create fallback file and continue | File path, rotation size, retention |
| Network Handler | Remote transmission to log collectors | Buffer locally and retry | Endpoint URL, timeout, retry policy |
| Memory Handler | In-process buffering for testing | Drop oldest entries when full | Buffer size, overflow behavior |

**Context Propagation and Correlation** addresses the critical challenge of connecting related log entries across distributed operations. When a user request spans multiple services, functions, and async operations, the logging system must automatically carry contextual information that allows operators to trace the complete request flow.

The system must implement **correlation ID injection**, where a unique identifier is automatically attached to every log record within a request scope. This correlation ID propagates through function calls, async operations, and even across network boundaries when properly configured. Additionally, the context system must support arbitrary key-value pairs that provide additional metadata—user IDs, session tokens, feature flags, or any other contextual information that aids in debugging and analysis.

Context propagation must handle these scenarios:

| Scenario | Context Behavior | Implementation Challenge |
|----------|------------------|--------------------------|
| Synchronous function calls | Context inherits from caller | Thread-local storage management |
| Asynchronous operations | Context copies to new task | Async context boundary preservation |
| Child logger creation | Context inherits from parent | Logger hierarchy traversal |
| Request boundary crossing | Context resets or inherits from headers | HTTP middleware integration |

> **Design Insight**: The correlation ID system transforms distributed debugging from impossible puzzle-solving into systematic investigation. Instead of manually correlating timestamps across multiple log files, operators can query for a single correlation ID and see the complete request flow.

### Non-Functional Requirements

Non-functional requirements define **how well** our system must perform—the quality attributes and constraints that ensure the logging system operates reliably in production environments without becoming a bottleneck or source of problems itself.

**Performance Requirements** demand that logging operations impose minimal overhead on the host application. The primary performance constraint is **latency**—individual log statements must complete in microseconds, not milliseconds. Applications often generate hundreds or thousands of log entries per second, and each logging operation occurs in the critical path of business logic.

The system must achieve these performance targets:

| Metric | Target | Measurement Method | Degradation Impact |
|--------|--------|--------------------|-------------------|
| Log statement latency | < 50 microseconds (P99) | Benchmark with no I/O handlers | Slows application request processing |
| Memory allocation | < 1KB per log record | Memory profiling during load testing | Increases garbage collection pressure |
| CPU overhead | < 2% of application CPU | Performance profiling under sustained load | Reduces application throughput |
| Handler dispatch | < 100 microseconds (P99) | End-to-end logging pipeline timing | Blocks application threads |

To achieve these targets, the system must avoid blocking I/O operations in the critical path. File writes, network transmissions, and other potentially slow operations must occur asynchronously or in background threads. The log record creation and dispatch mechanism must minimize memory allocations and avoid expensive serialization operations until absolutely necessary.

**Thread Safety Requirements** ensure correct behavior when multiple threads simultaneously invoke logging operations. Modern applications extensively use concurrent programming, and the logging system must handle simultaneous log statements without data corruption, race conditions, or deadlocks.

The thread safety model must protect these shared resources:

| Shared Resource | Concurrency Pattern | Protection Mechanism | Performance Impact |
|----------------|-------------------|---------------------|-------------------|
| Logger configuration | Read-heavy, rare writes | Reader-writer locks | Minimal—most operations read-only |
| Log record dispatch | High-frequency writes | Lock-free queues or fine-grained locking | Must not serialize all logging operations |
| Context storage | Thread-local access | Thread-local storage with inheritance | No contention between threads |
| Handler state | Handler-specific access patterns | Handler-managed synchronization | Varies by handler implementation |

**Configurability Requirements** enable the logging system to adapt to different environments and operational needs without code changes. The same application binary must support development logging (verbose, pretty-printed) and production logging (structured, filtered) through configuration alone.

Configuration must support these operational scenarios:

| Environment | Log Level | Output Format | Destinations | Context Fields |
|-------------|-----------|---------------|--------------|----------------|
| Development | `DEBUG` | Pretty-printed JSON with colors | Console only | All available context |
| Testing | `WARN` | Single-line JSON | Memory buffer | Minimal context for speed |
| Staging | `INFO` | Single-line JSON | File + Remote collector | Full production context |
| Production | `INFO` | Single-line JSON | Remote collector only | Security-filtered context |

**Async Compatibility Requirements** ensure the logging system functions correctly in applications using modern async/await programming patterns. Python's asyncio, JavaScript's Promise-based code, and similar patterns create execution contexts that traditional thread-local storage cannot handle properly.

The system must preserve logging context across these async boundaries:

| Async Pattern | Context Challenge | Required Behavior |
|--------------|-------------------|-------------------|
| `async def` function calls | Context inheritance from caller | Context automatically flows to async functions |
| `await` operations | Context preservation across yield points | Context remains available after await resumes |
| Task creation (`asyncio.create_task`) | Context copying to new task | New task inherits current context |
| Concurrent execution (`asyncio.gather`) | Context isolation between tasks | Each task maintains independent context |

> **Architecture Decision: Async Context Strategy**
> - **Context**: Modern applications heavily use async/await patterns, but traditional thread-local storage fails across await boundaries
> - **Options Considered**: 
>   1. Ignore async compatibility—require manual context passing
>   2. Use language-specific async context mechanisms
>   3. Build custom context propagation system
> - **Decision**: Use language-specific async context mechanisms (Python's `contextvars`, etc.)
> - **Rationale**: Language async context systems are specifically designed for this problem and integrate seamlessly with async runtimes
> - **Consequences**: Simpler implementation and better performance, but requires different implementation strategies per language

### Explicit Non-Goals

Clearly defining what our logging system does **not** do is crucial for maintaining focus and preventing scope creep. These non-goals represent important functionality that could theoretically be added but would fundamentally change the nature and complexity of our system.

**Log Storage and Persistence Management** falls outside our scope. Our logging system produces structured log records and dispatches them to configured destinations, but it does not implement log storage backends, retention policies, or data lifecycle management. We provide file handlers that can write to local files, but we do not build distributed storage systems, implement log rotation policies, or manage disk space consumption.

This boundary means our system interfaces with but does not replace dedicated log management solutions:

| Storage Concern | Our Responsibility | External System Responsibility |
|----------------|-------------------|-------------------------------|
| Log record format | Produce structured JSON records | Store and index records efficiently |
| Local file writing | Write records to specified files | Implement file rotation and compression |
| Network transmission | Send records to remote endpoints | Receive, route, and persist records |
| Data retention | Continue producing records | Implement age-based deletion policies |

**Search and Analytics Capabilities** represent a completely different problem domain. While our structured JSON output enables powerful searching and analysis, we do not build query engines, indexing systems, or analytical dashboards. Our role ends when log records reach their configured destinations—whether those destinations provide search capabilities is beyond our scope.

**Real-Time Alerting and Monitoring** requires complex event processing, threshold management, and notification systems. Although our log records contain the data needed for alerting (error rates, performance metrics, business events), we do not implement alert rules, notification channels, or monitoring dashboards. We focus on reliable log record production and dispatch.

**Log Sampling and Rate Limiting** represents advanced operational features that could compromise our core reliability guarantees. While production systems often need these capabilities to manage log volume and costs, implementing them requires complex policy engines and could introduce scenarios where critical log records are dropped. We choose to generate all requested log records and leave volume management to downstream systems.

**Dynamic Schema Management** involves automatically detecting and managing changes in log record structure over time. While our JSON output naturally accommodates additional fields, we do not implement schema registries, version management, or compatibility checking between different log record formats.

**Security and Access Control** for log content remains the responsibility of storage and analysis systems. We do not implement field-level encryption, access control policies, or data masking capabilities. Our context filtering allows removing sensitive fields before serialization, but comprehensive security requires purpose-built security infrastructure.

> **Design Principle**: Single Responsibility
> Our logging system excels at one thing: reliably producing structured, contextual log records and dispatching them to configured destinations. By explicitly avoiding adjacent problem domains, we can focus on making our core functionality extremely reliable, performant, and easy to integrate.

These non-goals create clear integration points where our logging system connects with specialized external systems:

| Integration Point | Our Output | External System Input | Example Tools |
|------------------|------------|----------------------|---------------|
| Log Aggregation | JSON records via network handlers | Structured log ingestion APIs | Elasticsearch, Splunk, Datadog |
| File Management | Raw log files via file handlers | File monitoring and rotation | logrotate, fluentd, filebeat |
| Monitoring | Structured records with metrics | Log-based metric extraction | Prometheus, Grafana, custom dashboards |
| Alerting | Error/warning records with context | Event pattern recognition | PagerDuty, AlertManager, custom systems |

⚠️ **Pitfall: Scope Creep Through "Simple" Features**
Learners often want to add "just a simple search feature" or "basic alerting" to their logging system. These features seem simple but require fundamentally different architectural patterns (indexing, query processing, event detection) that would dominate the system design. Resist the temptation to build a "logging platform"—focus on building an excellent logging library that integrates with existing platforms.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| JSON Serialization | `json` standard library with custom encoder | `orjson` or `ujson` for performance |
| Thread Safety | `threading.Lock` with context managers | `queue.Queue` for lock-free dispatch |
| Async Context | `contextvars` module (Python 3.7+) | Custom context propagation framework |
| Configuration | Environment variables + dataclasses | YAML/TOML files with validation |
| Network Handlers | `urllib3` or `requests` for HTTP | `aiohttp` for async network operations |
| File Operations | Built-in `open()` with manual rotation | `logging.handlers.RotatingFileHandler` adaptation |

**B. Recommended File/Module Structure:**

```
structured_logging/
├── __init__.py                  ← Public API exports
├── core/
│   ├── __init__.py
│   ├── logger.py               ← Logger hierarchy and main API
│   ├── record.py               ← LogRecord data structure
│   ├── levels.py               ← Log level constants and utilities
│   └── context.py              ← Context propagation system
├── handlers/
│   ├── __init__.py
│   ├── base.py                 ← Abstract Handler base class
│   ├── console.py              ← Console/stdout handler
│   ├── file.py                 ← File writing handler
│   └── network.py              ← HTTP/network handlers
├── formatters/
│   ├── __init__.py
│   ├── json_formatter.py       ← JSON serialization
│   ├── pretty_formatter.py     ← Development-friendly output
│   └── registry.py             ← Formatter plugin system
├── utils/
│   ├── __init__.py
│   ├── serialization.py        ← Safe JSON encoding utilities
│   ├── threading.py            ← Thread-safe data structures
│   └── async_utils.py          ← Async context bridge utilities
└── tests/
    ├── test_core/
    ├── test_handlers/
    ├── test_formatters/
    └── integration/
```

**C. Infrastructure Starter Code (COMPLETE):**

```python
# utils/threading.py - Complete thread-safe utilities
import threading
from typing import Any, Dict, Optional

class ThreadSafeDict:
    """Thread-safe dictionary wrapper for shared logger configuration."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested access
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._data.update(updates)
    
    def copy(self) -> Dict[str, Any]:
        with self._lock:
            return self._data.copy()


class ThreadSafeCounter:
    """Thread-safe counter for generating correlation IDs."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
```

```python
# utils/serialization.py - Complete JSON serialization utilities
import json
import sys
from typing import Any, Dict, Set
from decimal import Decimal
from datetime import datetime, date

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable types safely."""
    
    def default(self, obj: Any) -> Any:
        # Handle common non-serializable types
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return f"<{obj.__class__.__name__} object>"
        elif callable(obj):
            return f"<function {getattr(obj, '__name__', 'unknown')}>"
        else:
            return f"<{type(obj).__name__}>"

def safe_serialize(data: Dict[str, Any], max_depth: int = 10) -> str:
    """Safely serialize dictionary to JSON with circular reference protection."""
    try:
        return json.dumps(data, cls=SafeJSONEncoder, separators=(',', ':'))
    except (TypeError, ValueError, RecursionError):
        # Fallback: create safe representation
        safe_data = _make_serializable(data, max_depth, set())
        return json.dumps(safe_data, separators=(',', ':'))

def _make_serializable(obj: Any, max_depth: int, seen_objects: Set[int]) -> Any:
    """Recursively make object serializable, handling circular references."""
    if max_depth <= 0:
        return "<max_depth_reached>"
    
    obj_id = id(obj)
    if obj_id in seen_objects:
        return "<circular_reference>"
    
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        seen_objects.add(obj_id)
        result = {
            str(k): _make_serializable(v, max_depth - 1, seen_objects.copy())
            for k, v in obj.items()
        }
        seen_objects.remove(obj_id)
        return result
    elif isinstance(obj, (list, tuple)):
        seen_objects.add(obj_id)
        result = [
            _make_serializable(item, max_depth - 1, seen_objects.copy())
            for item in obj
        ]
        seen_objects.remove(obj_id)
        return result
    else:
        return SafeJSONEncoder().default(obj)

def estimate_serialized_size(data: Dict[str, Any]) -> int:
    """Estimate JSON size without full serialization for performance."""
    # Quick estimation based on data structure
    size = 2  # Opening and closing braces
    for key, value in data.items():
        size += len(str(key)) + 3  # Key + quotes + colon
        if isinstance(value, str):
            size += len(value) + 2  # Value + quotes
        elif isinstance(value, (int, float)):
            size += len(str(value))
        elif isinstance(value, dict):
            size += estimate_serialized_size(value)
        else:
            size += 20  # Conservative estimate for other types
        size += 1  # Comma separator
    return size
```

**D. Core Logic Skeleton Code:**

```python
# core/levels.py - Log level management (SKELETON)
from typing import Dict

# Log level constants
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
FATAL = 50

LEVEL_NAMES: Dict[int, str] = {
    DEBUG: "DEBUG",
    INFO: "INFO", 
    WARN: "WARN",
    ERROR: "ERROR",
    FATAL: "FATAL"
}

def should_log(message_level: int, configured_level: int) -> bool:
    """Determine if a message should be logged based on level filtering."""
    # TODO 1: Compare message_level with configured_level
    # TODO 2: Return True if message_level >= configured_level
    # TODO 3: Return False otherwise (message should be filtered out)
    # Hint: Higher numeric values represent more severe log levels
    pass

def level_name_to_value(level_name: str) -> int:
    """Convert string level name to numeric value."""
    # TODO 1: Create reverse mapping from LEVEL_NAMES
    # TODO 2: Look up level_name in reverse mapping
    # TODO 3: Return numeric value or raise ValueError for invalid names
    # TODO 4: Handle case-insensitive matching (convert to uppercase)
    pass
```

```python
# core/record.py - Log record data structure (SKELETON)
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class LogRecord:
    """Structured log record containing all message data and metadata."""
    
    def __init__(self, level: int, message: str, logger_name: str, 
                 context: Optional[Dict[str, Any]] = None):
        # TODO 1: Store the provided parameters as instance attributes
        # TODO 2: Generate timestamp using datetime.utcnow().isoformat()
        # TODO 3: Initialize context as empty dict if None provided
        # TODO 4: Ensure context is a copy to prevent external mutation
        # Hint: Use dict(context) or context.copy() to create defensive copy
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary for JSON serialization."""
        # TODO 1: Create dictionary with timestamp, level, message, logger_name
        # TODO 2: Add context fields to the dictionary
        # TODO 3: Convert numeric level to string using LEVEL_NAMES
        # TODO 4: Return complete dictionary ready for JSON serialization
        # Hint: Use LEVEL_NAMES from levels.py to convert level number to string
        pass
    
    def add_context(self, key: str, value: Any) -> None:
        """Add a context field to this log record."""
        # TODO 1: Add key-value pair to context dictionary
        # TODO 2: Handle the case where key already exists (overwrite vs. error?)
        # Design decision: Should duplicate keys overwrite or raise an error?
        pass
```

**E. Language-Specific Hints:**

- **Thread Safety**: Use `threading.RLock()` (reentrant locks) for logger hierarchy access since child loggers may need to access parent configuration during the same operation
- **Async Context**: Import `contextvars` for Python 3.7+ async context propagation—use `contextvars.ContextVar` to store correlation IDs and context data
- **JSON Performance**: The built-in `json` module is sufficient for learning, but consider `orjson` for production use if serialization becomes a bottleneck
- **Memory Management**: Use `__slots__` in `LogRecord` class to reduce memory overhead when creating many log records
- **Exception Handling**: Use `safe_call()` pattern for handler dispatch—if one handler fails, continue with remaining handlers
- **File I/O**: Use `os.fsync()` after writing critical log records to ensure durability, but be aware this impacts performance

**F. Milestone Checkpoint:**

After implementing the goals and requirements defined in this section:

**What to verify after Milestone 1 (Logger Core):**
```bash
python -c "
from structured_logging import Logger
logger = Logger('test', level=INFO)
logger.info('Test message', user_id=12345)
logger.debug('Debug message')  # Should be filtered out
"
```

Expected behavior:
- INFO message appears with JSON format including timestamp, level, message, and user_id context
- DEBUG message does not appear (filtered by level)
- No exceptions or thread safety issues when called from multiple threads

**What to verify after Milestone 2 (Structured Output):**
```bash
python -c "
import json
from structured_logging import Logger
logger = Logger('test')
logger.info('Test message', request_id='req_123')
# Verify output is valid JSON by parsing it
"
```

Expected behavior:
- Output is valid single-line JSON that can be parsed
- All required fields present: timestamp, level, message, logger_name, context
- Context fields properly nested under 'context' key

**What to verify after Milestone 3 (Context & Correlation):**
```bash
python -c "
from structured_logging import Logger, with_correlation_id
logger = Logger('test')
with with_correlation_id():
    logger.info('Message 1')
    logger.info('Message 2')
"
```

Expected behavior:
- Both messages contain the same correlation_id in their context
- Correlation ID is automatically generated and unique per context scope
- Context properly propagates through nested function calls

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|----------------|-----|
| Log messages not appearing | Level filtering too restrictive | Check logger level vs. message level | Set logger level to DEBUG temporarily |
| JSON serialization errors | Non-serializable objects in context | Try serializing context manually | Use SafeJSONEncoder or filter context values |
| Context not propagating | Thread-local storage issues | Print context at each function level | Use contextvars instead of threading.local |
| Performance degradation | Blocking I/O in handlers | Profile handler execution times | Move I/O operations to background threads |
| Race conditions in output | Multiple threads writing simultaneously | Look for interleaved output characters | Add proper locking in handlers |


## High-Level Architecture

> **Milestone(s):** This section provides the architectural foundation for all three milestones by defining the core components, their relationships, and the data flow that enables structured logging with context propagation.

### Component Overview: Core Components: Logger, LogRecord, Handlers, Formatters, and Context Manager

Think of the structured logging system as a **broadcasting studio with multiple output channels**. Just as a news studio takes raw content, formats it for different audiences (TV, radio, web), and broadcasts simultaneously to multiple destinations, our logging system takes application events, structures them consistently, and dispatches them to various outputs like console, files, and remote collectors. The studio has specialized equipment for each step: cameras capture content, editors format it, and transmitters send it to different channels. Similarly, our logging system has specialized components for capturing, formatting, and dispatching log messages.

The structured logging architecture consists of five core components that work together to provide a complete logging solution. Each component has a distinct responsibility and well-defined interfaces that enable flexible composition and extensibility.

#### Logger Component

The `Logger` serves as the primary interface between application code and the logging infrastructure. Think of it as the **control panel** in our broadcasting studio analogy - it's where producers decide what content gets recorded, at what quality level, and which channels receive the broadcast. Each logger maintains its own identity and configuration while participating in a hierarchical organization that enables inheritance and scoped behavior.

Every logger instance maintains several critical pieces of state that determine its behavior and capabilities:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Hierarchical identifier (e.g., "app.database.connection") that determines logger's position in the tree |
| `level` | `int` | Minimum severity threshold - messages below this level are filtered out before processing |
| `parent` | `Logger` | Reference to parent logger for configuration inheritance and context propagation |
| `handlers` | `List[Handler]` | Collection of output destinations that will receive log records from this logger |
| `children` | `Dict[str, Logger]` | Map of child loggers created under this logger's namespace |

The logger hierarchy enables powerful inheritance patterns where child loggers automatically inherit their parent's configuration, handlers, and context fields while allowing selective overrides for specialized behavior. For example, a database connection logger might inherit the application's general configuration but add specialized handlers for database-specific monitoring.

#### LogRecord Structure

The `LogRecord` represents a single log event in its structured form. This is the fundamental data unit that flows through the entire logging pipeline. Think of it as a **standardized news report template** that ensures every piece of information contains the same essential elements regardless of who created it or where it's going.

The LogRecord structure captures all essential information about a logging event in a consistent, queryable format:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `str` | ISO 8601 formatted timestamp indicating when the log event occurred |
| `level` | `int` | Numeric severity level (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50) |
| `message` | `str` | Human-readable description of the event or situation being logged |
| `logger_name` | `str` | Fully qualified name of the logger that created this record |
| `context` | `Dict[str, Any]` | Key-value pairs providing additional structured data about the event |

The context dictionary is where the "structured" aspect of structured logging becomes most apparent. Instead of embedding variable data into string messages, applications attach meaningful key-value pairs that can be queried, filtered, and aggregated by log analysis tools. For example, rather than logging "User john@example.com failed login attempt from IP 192.168.1.100", the system would log a message "User login failed" with context fields `{"user_email": "john@example.com", "client_ip": "192.168.1.100", "login_attempt": "failed"}`.

#### Handler Dispatch System

Handlers are the **output channels** in our broadcasting analogy. Each handler knows how to deliver log records to a specific destination, whether that's stdout for development debugging, rotating files for long-term storage, or remote collectors for centralized analysis. The handler system enables simultaneous dispatch to multiple destinations without requiring application code to manage the complexity.

The abstract `Handler` interface defines the contract that all output destinations must implement:

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `emit` | `record: LogRecord` | `None` | Process and output a single log record to this handler's destination |
| `flush` | `None` | `None` | Ensure all buffered records are written to the underlying destination |
| `close` | `None` | `None` | Release resources and perform cleanup when handler is no longer needed |
| `set_formatter` | `formatter: Formatter` | `None` | Configure the formatter used to render log records for output |

Each handler implementation specializes the abstract interface for its specific destination. A file handler manages file rotation, buffering, and disk I/O. A remote handler manages network connections, retry logic, and serialization for transmission. A console handler manages terminal output, color coding, and user-friendly formatting.

The dispatch mechanism routes each log record to all configured handlers simultaneously. This enables powerful patterns like writing structured JSON to files for analysis while simultaneously displaying pretty-printed, colorized output to the developer console during debugging.

#### Formatter Plugin System

Formatters transform `LogRecord` objects into the final output representation. Think of formatters as **template engines** that take structured data and render it into specific output formats optimized for different consumers. The plugin system allows applications to register custom formatters and select them by name, enabling consistent formatting across different parts of an application.

The formatter interface provides the contract for all output renderers:

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `format` | `record: LogRecord` | `str` | Convert a log record into formatted string output ready for display or transmission |
| `configure` | `options: Dict[str, Any]` | `None` | Apply configuration options specific to this formatter implementation |

Common formatter implementations include the JSON formatter for structured output, the pretty-print formatter for human-readable console output, and custom formatters that integrate with specific monitoring or analysis tools. The plugin system allows applications to register formatters by name and configure different handlers to use different formatters based on their intended audience.

#### Context Manager

The Context Manager handles the complex task of **context propagation** - ensuring that relevant contextual information flows through nested function calls and across async boundaries without requiring explicit parameter passing. Think of it as the **continuity coordinator** in our broadcasting studio, ensuring that related segments maintain consistent information and branding elements throughout the production.

The context system maintains thread-local storage and manages inheritance patterns that allow contextual information to flow naturally through application execution:

| Component | Purpose | Key Behaviors |
|-----------|---------|---------------|
| Thread-Local Storage | Maintains context state per execution thread | Isolates context between concurrent requests, automatically inherits parent context |
| Correlation ID Generator | Creates unique identifiers for request tracing | Generates UUIDs, injects into log context, propagates across service boundaries |
| Async Context Bridge | Preserves context across async/await boundaries | Captures context at task creation, restores context during task execution |
| Context Inheritance | Manages parent-child context relationships | Child contexts inherit parent fields, can override or add new fields |

The context manager ensures that once a correlation ID or other contextual information is established (typically at request boundaries), it automatically appears in every log record created within that execution scope, even deep within nested function calls or across async task switches.

> **Design Insight**: The separation of concerns between Logger (message creation), Handler (output dispatch), Formatter (rendering), and Context Manager (state propagation) creates a highly flexible system where each aspect can be configured and extended independently. This modularity is essential for accommodating the diverse requirements of different deployment environments and use cases.

![System Architecture Overview](./diagrams/system-architecture.svg)

### Data Flow Pipeline: How log messages flow from application code through formatters to output destinations

Understanding the complete journey of a log message from application code to final output destinations reveals how the components work together to provide structured logging capabilities. This pipeline demonstrates the transformation and enrichment that occurs at each stage, showing how raw application events become structured, contextualized, and properly formatted log records.

#### Stage 1: Message Creation and Level Filtering

The logging pipeline begins when application code calls one of the logging methods (`debug`, `info`, `warn`, `error`, or `fatal`). The first critical decision point occurs immediately: **level filtering**. Think of this as a **security checkpoint** that only allows messages of sufficient importance to proceed through the expensive processing pipeline.

The level filtering process follows this sequence:

1. Application code calls `logger.info("User authenticated", user_id="12345", session_id="abc-def")`
2. Logger retrieves its configured minimum level (inherited from parent if not explicitly set)
3. The `should_log` function compares the message level against the configured threshold
4. If the message level is below the threshold, processing stops immediately - no LogRecord is created, no handlers are called, no formatting occurs
5. If the message passes the level check, processing continues to record creation

This early filtering is crucial for performance. In production systems, DEBUG level messages might be completely disabled, and the filtering prevents any overhead from creating LogRecord objects or processing context for messages that will never be output.

#### Stage 2: LogRecord Construction and Context Enrichment

Once a message passes level filtering, the system creates a `LogRecord` object and enriches it with contextual information. This stage transforms a simple message and parameters into a fully structured log event with all relevant metadata.

The record construction process involves several enrichment steps:

1. **Timestamp Generation**: The system captures the current time with microsecond precision and formats it according to the configured timestamp format (typically ISO 8601)
2. **Context Aggregation**: The context manager retrieves all active context fields from thread-local storage, including correlation IDs, user information, and request metadata
3. **Logger Context Inheritance**: Any context fields attached to the logger or its parents are merged with the active context
4. **Parameter Integration**: Keyword arguments passed to the logging method are merged into the context dictionary
5. **Record Assembly**: All components are combined into a complete LogRecord with timestamp, level, message, logger name, and enriched context

The context enrichment stage is where structured logging shows its power. A simple log call might result in a LogRecord with dozens of contextual fields that provide rich information about the application state when the event occurred.

#### Stage 3: Handler Dispatch and Parallel Processing

With a complete LogRecord created, the system enters the handler dispatch phase. This stage demonstrates the **fan-out pattern** where a single log event triggers parallel processing to multiple output destinations. Think of this as a **broadcasting transmitter** that simultaneously sends the same content to radio, television, and internet streams.

The dispatch mechanism follows this parallel processing pattern:

1. The logger iterates through its list of configured handlers
2. For each handler, the system makes an independent `emit` call with the LogRecord
3. Each handler processes the record according to its specific requirements:
   - Console handlers might check if output should be colorized
   - File handlers might check if log rotation is needed
   - Remote handlers might check network connectivity and queue records if offline
4. Handler failures are isolated - if one handler fails, others continue processing
5. Each handler applies its configured formatter to render the LogRecord into final output format

This parallel processing enables powerful deployment patterns. During development, logs might go to a colorized console for immediate feedback. In staging, the same logs might go to both files for debugging and a remote collector for integration testing. In production, logs might go to high-performance local files, metrics aggregators, and centralized logging infrastructure simultaneously.

#### Stage 4: Formatting and Serialization

Each handler applies its configured formatter to transform the structured LogRecord into the appropriate output representation. This stage handles the complex task of serialization while preserving the structured nature of the data and handling edge cases like circular references or non-serializable objects.

The formatting process addresses several challenges:

1. **Serialization Safety**: The `safe_serialize` function handles objects that can't be directly serialized to JSON, converting them to string representations or filtering them out entirely
2. **Circular Reference Detection**: The serializer detects and breaks circular references to prevent infinite recursion during JSON generation
3. **Size Estimation**: For performance, the system can estimate serialized size without full serialization to decide whether to truncate large context objects
4. **Field Ordering**: JSON output uses consistent field ordering to improve readability and enable text-based diffing of log files
5. **Pretty Printing**: Development formatters add indentation, color coding, and human-friendly formatting while preserving the underlying structured data

Different handlers might apply different formatters to the same LogRecord. A file handler might use compact, single-line JSON for efficient storage and parsing, while a console handler might use pretty-printed, colorized output for developer readability.

#### Stage 5: Output Delivery and Error Recovery

The final stage handles actual output delivery to the configured destinations. This stage must handle various failure modes gracefully while maintaining the reliability that production logging systems require. Think of this as the **final mile delivery** where packages must reach their destinations despite traffic, weather, or infrastructure problems.

Output delivery involves several reliability mechanisms:

1. **Buffering Strategy**: Handlers may buffer records for efficiency, but must balance performance with the risk of losing logs during crashes
2. **Failure Recovery**: When output operations fail (disk full, network unreachable, permissions denied), handlers must decide whether to retry, queue for later, or drop records
3. **Resource Management**: File handles, network connections, and buffers must be managed carefully to prevent resource leaks
4. **Graceful Degradation**: When primary output destinations fail, the system should attempt to log error information to alternative destinations rather than crashing the application

The error recovery mechanisms ensure that logging problems don't become application problems. If a remote logging service becomes unavailable, the application continues running and attempts to log the issue to local files or console output.

> **Design Insight**: The pipeline architecture enables **loose coupling** between stages. Application code doesn't need to know about formatters, formatters don't need to know about output destinations, and handlers don't need to know about context propagation. This separation allows each stage to be optimized, configured, and extended independently.

The complete data flow pipeline demonstrates how structured logging transforms simple application events into rich, contextualized, and properly formatted log records delivered to multiple destinations with reliability and performance guarantees.

![Log Processing Data Flow](./diagrams/log-processing-flow.svg)

### Recommended Module Structure: File organization and package layout for clean separation of concerns

A well-organized module structure is crucial for maintaining clean separation of concerns and enabling teams to work on different aspects of the logging system independently. Think of the module structure as **organizing a professional kitchen** - each station has specific responsibilities, specialized tools, and clear interfaces with other stations, allowing multiple chefs to work efficiently without interfering with each other.

The recommended structure follows the principle of **bounded contexts** where each module encapsulates related functionality and exposes minimal, well-defined interfaces to other modules. This organization supports both the learning progression through the three milestones and the maintainability requirements of production systems.

#### Root Package Structure

The top-level package organization reflects the major architectural boundaries and provides clear entry points for different aspects of the system:

```
structured_logging/
├── __init__.py                    # Public API exports
├── logger.py                      # Core Logger and LogRecord classes
├── handlers/                      # Output destination implementations
│   ├── __init__.py               # Handler base class and common utilities
│   ├── console.py                # Console/stdout handler
│   ├── file.py                   # File-based handlers with rotation
│   └── remote.py                 # Network-based handlers
├── formatters/                    # Output format implementations
│   ├── __init__.py               # Formatter base class and registry
│   ├── json.py                   # JSON formatter with serialization safety
│   └── pretty.py                 # Human-readable console formatter
├── context/                       # Context propagation and correlation
│   ├── __init__.py               # Context manager and public interfaces
│   ├── storage.py                # Thread-local context storage
│   ├── correlation.py            # Correlation ID generation and injection
│   └── async_bridge.py           # Async context preservation
├── utils/                         # Shared utilities and helpers
│   ├── __init__.py               # Common utility exports
│   ├── serialization.py          # Safe JSON serialization functions
│   ├── thread_safety.py          # Thread-safe data structures
│   └── validation.py             # Input validation and sanitization
├── config/                        # Configuration management
│   ├── __init__.py               # Configuration loading and validation
│   └── schema.py                 # Configuration schema definitions
└── tests/                         # Comprehensive test suite
    ├── unit/                      # Unit tests for individual components
    ├── integration/               # Integration tests across components
    └── performance/               # Performance and load testing
```

#### Core Module Responsibilities

Each module has clearly defined responsibilities that align with the architectural components and enable independent development and testing.

**Logger Module (`logger.py`)**
This module contains the central `Logger` class and `LogRecord` data structure that form the heart of the logging system. It should be the most stable module since other components depend on its interfaces:

| Class/Function | Purpose | Key Dependencies |
|----------------|---------|------------------|
| `Logger` | Main logging interface with hierarchy support | Handlers, Context Manager, Level utilities |
| `LogRecord` | Structured log event representation | Serialization utilities, Timestamp formatting |
| `LoggerFactory` | Factory for creating and managing logger instances | Configuration system, Logger hierarchy |
| `get_logger` | Primary entry point for obtaining logger instances | LoggerFactory, Thread-safe caching |

**Handlers Module (`handlers/`)**
The handlers package encapsulates all output destination logic. Each handler type gets its own module to enable independent development and specialized dependencies:

| Module | Handler Types | Key Considerations |
|---------|---------------|-------------------|
| `console.py` | Stdout, stderr handlers | Color coding, terminal detection, Unicode support |
| `file.py` | File, rotating file handlers | File locking, atomic writes, disk space monitoring |
| `remote.py` | HTTP, TCP, UDP handlers | Connection pooling, retry logic, serialization formats |

The handler base class in `handlers/__init__.py` provides common functionality like formatter management, error handling patterns, and resource cleanup protocols that all handler implementations can inherit.

**Formatters Module (`formatters/`)**
The formatters package handles the transformation from structured `LogRecord` objects to formatted output strings. This separation allows the same log data to be rendered differently for various audiences:

| Module | Formatter Types | Output Characteristics |
|---------|----------------|------------------------|
| `json.py` | Compact JSON, pretty JSON | Machine-readable, consistent field ordering, size optimization |
| `pretty.py` | Console, colored console | Human-readable, color coding, selective field display |

The formatter registry in `formatters/__init__.py` enables dynamic formatter selection and supports custom formatter registration for specialized output requirements.

**Context Module (`context/`)**
The context package manages the complex requirements of context propagation across different execution models (synchronous, threaded, async). This is separated into multiple modules due to the different challenges each execution model presents:

| Module | Responsibilities | Key Challenges |
|---------|-----------------|----------------|
| `storage.py` | Thread-local context storage | Thread isolation, inheritance patterns, memory cleanup |
| `correlation.py` | Correlation ID management | UUID generation, injection points, cross-service propagation |
| `async_bridge.py` | Async context preservation | Context copying, task boundaries, coroutine lifecycle |

#### Milestone-Aligned Development Structure

The module structure supports incremental development through the three project milestones, allowing learners to build functionality progressively without requiring massive refactoring between milestones:

**Milestone 1 Structure (Logger Core)**
Initial implementation focuses on the core modules with basic implementations:
- `logger.py`: Complete Logger and LogRecord implementation
- `handlers/__init__.py`: Base Handler class and console handler
- `utils/thread_safety.py`: Thread-safe data structures
- `utils/validation.py`: Level validation and basic input sanitization

**Milestone 2 Extensions (Structured Output)**
Milestone 2 adds formatter infrastructure without changing existing modules:
- `formatters/`: Complete formatter package with JSON and pretty-print implementations
- `utils/serialization.py`: Safe JSON serialization with circular reference protection
- Enhanced handlers with formatter integration

**Milestone 3 Extensions (Context & Correlation)**
Milestone 3 adds the context package while maintaining backward compatibility:
- `context/`: Complete context propagation package
- Integration points in `logger.py` for context injection
- Extended handler and formatter support for correlation IDs

#### Testing Structure Alignment

The testing structure mirrors the module organization and supports both component-level testing and cross-cutting integration scenarios:

```
tests/
├── unit/
│   ├── test_logger.py             # Logger class behavior, hierarchy, level filtering
│   ├── test_handlers/             # Handler-specific unit tests
│   │   ├── test_console.py        # Console output, color coding, error handling
│   │   ├── test_file.py           # File operations, rotation, locking
│   │   └── test_remote.py         # Network operations, retry, serialization
│   ├── test_formatters/           # Formatter unit tests
│   │   ├── test_json.py           # JSON serialization, field ordering, safety
│   │   └── test_pretty.py         # Pretty printing, color codes, truncation
│   └── test_context/              # Context propagation unit tests
│       ├── test_storage.py        # Thread-local storage, inheritance
│       ├── test_correlation.py    # ID generation, injection
│       └── test_async_bridge.py   # Async context preservation
├── integration/
│   ├── test_end_to_end.py         # Complete logging pipeline tests
│   ├── test_multi_handler.py     # Multiple handler coordination
│   ├── test_context_flow.py      # Context propagation across components
│   └── test_async_logging.py     # Async/await context preservation
└── performance/
    ├── test_throughput.py         # Message processing performance
    ├── test_memory_usage.py       # Memory footprint and leak detection
    └── test_concurrent_logging.py # Multi-threaded performance and safety
```

> **Design Insight**: The module structure balances **cohesion** (related functionality grouped together) with **loose coupling** (minimal dependencies between modules). Each module can be understood, tested, and modified independently while contributing to the overall system functionality. This structure particularly benefits learning environments where students can focus on one architectural layer at a time.

#### Import Strategy and Public APIs

The package design uses a clear import strategy that exposes high-level APIs while keeping implementation details internal:

**Public API (`__init__.py`)**
The root package exposes only the essential interfaces that application code needs:

```python
# Public exports - these are the stable API
from .logger import get_logger, Logger, LogRecord
from .context import with_context, correlation_id
from .config import configure_logging

# Internal modules should not be imported directly by applications
```

**Internal Module Imports**
Internal modules use relative imports and depend only on the interfaces they need:

```python
# handlers/console.py
from ..formatters import get_formatter
from ..utils.thread_safety import ThreadSafeCounter
from .base import Handler  # Relative import within handlers package
```

This import strategy ensures that refactoring internal implementation doesn't break application code and makes the learning progression clearer by highlighting the distinction between public APIs and implementation details.

The module structure provides a solid foundation for building the structured logging system incrementally while maintaining clean separation of concerns and supporting both learning objectives and production requirements.

### Implementation Guidance

The implementation of the structured logging system requires careful attention to Python-specific patterns, thread safety considerations, and performance optimizations. This guidance provides complete starter code for infrastructure components and detailed skeletons for core learning components.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| JSON Serialization | `json` module with custom encoder | `orjson` or `ujson` for high performance |
| Thread Safety | `threading.RLock` and `threading.local` | `concurrent.futures` with thread pools |
| File Operations | `open()` with manual rotation | `logging.handlers.RotatingFileHandler` base |
| Network Handlers | `urllib3` for HTTP requests | `asyncio` with `aiohttp` for async |
| Correlation IDs | `uuid.uuid4()` generation | `contextvars` for async context propagation |
| Configuration | Dictionary-based config | `pydantic` models for validation |

#### Complete Infrastructure Starter Code

**Thread-Safe Data Structures (`utils/thread_safety.py`)**
```python
"""Thread-safe data structures for concurrent logging operations."""
import threading
from typing import Any, Dict, Optional


class ThreadSafeCounter:
    """Thread-safe counter for handler statistics and correlation ID generation."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value


class ThreadSafeDict:
    """Thread-safe dictionary for shared logger registry and context storage."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RWMutex()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with read lock."""
        with self._lock.reader_lock():
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value with write lock."""
        with self._lock.writer_lock():
            self._data[key] = value
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update multiple values atomically."""
        with self._lock.writer_lock():
            self._data.update(other)
```

**Safe JSON Serialization (`utils/serialization.py`)**
```python
"""Safe JSON serialization handling non-serializable objects and circular references."""
import json
import sys
from typing import Any, Dict, Set
from decimal import Decimal
from datetime import datetime, date


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that safely handles non-serializable types."""
    
    def __init__(self, max_depth: int = 10, max_size: int = 1024 * 1024):
        super().__init__(ensure_ascii=False, separators=(',', ':'))
        self.max_depth = max_depth
        self.max_size = max_size
        self._visited: Set[int] = set()
    
    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to string representations."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return f"<{type(obj).__name__} object>"
        else:
            return str(obj)


def safe_serialize(data: Dict[str, Any], max_depth: int = 10) -> str:
    """
    Safely serialize dictionary to JSON string with circular reference protection.
    
    Args:
        data: Dictionary to serialize
        max_depth: Maximum nesting depth to prevent infinite recursion
        
    Returns:
        JSON string representation
        
    Raises:
        ValueError: If data cannot be serialized even with safety measures
    """
    try:
        return json.dumps(data, cls=SafeJSONEncoder, max_depth=max_depth)
    except (TypeError, ValueError, RecursionError) as e:
        # Fallback: serialize with string conversion
        safe_data = {k: str(v) for k, v in data.items()}
        return json.dumps(safe_data, ensure_ascii=False)


def estimate_serialized_size(data: Dict[str, Any]) -> int:
    """
    Estimate serialized JSON size without full serialization.
    
    This is faster than full serialization for size checking.
    """
    # Simple heuristic: sum of key lengths + estimated value lengths
    size = 2  # Opening and closing braces
    for key, value in data.items():
        size += len(str(key)) + 4  # Key + quotes and colon
        if isinstance(value, str):
            size += len(value) + 2  # String + quotes
        elif isinstance(value, (int, float)):
            size += len(str(value))
        elif isinstance(value, (list, dict)):
            size += 50  # Rough estimate for collections
        else:
            size += 20  # Default estimate for other types
    return size
```

**Safe Function Execution (`utils/validation.py`)**
```python
"""Validation utilities and safe function execution for error handling."""
import functools
import traceback
from typing import Any, Callable, Optional


def safe_call(func: Callable, *args, default: Any = None, **kwargs) -> Any:
    """
    Safely execute function with exception handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default return value if function fails
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value on exception
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # In a production system, you might want to log this error
        # But we need to be careful not to create recursive logging
        print(f"Safe call failed for {func.__name__}: {e}", file=sys.stderr)
        return default


def should_log(message_level: int, configured_level: int) -> bool:
    """
    Determine if message should be logged based on level filtering.
    
    Args:
        message_level: Severity level of the message
        configured_level: Minimum level configured for the logger
        
    Returns:
        True if message should be processed, False if it should be filtered
    """
    return message_level >= configured_level
```

#### Core Logic Skeletons

**Logger Implementation (`logger.py`)**
```python
"""Core Logger and LogRecord classes - implement these according to the milestone requirements."""
import threading
import time
from typing import Any, Dict, List, Optional
from .utils.validation import should_log, safe_call
from .context import get_current_context


# Log level constants - use these exact values
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
FATAL = 50


class LogRecord:
    """Structured log record containing all information about a logging event."""
    
    def __init__(self, timestamp: str, level: int, message: str, 
                 logger_name: str, context: Dict[str, Any]):
        self.timestamp = timestamp
        self.level = level
        self.message = message
        self.logger_name = logger_name
        self.context = context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary for JSON serialization."""
        # TODO 1: Create base dictionary with timestamp, level, message, logger_name
        # TODO 2: Add context fields to the dictionary
        # TODO 3: Ensure consistent field ordering for predictable output
        # TODO 4: Handle any non-serializable values in context
        pass


class Logger:
    """
    Main logging interface with hierarchy support and handler dispatch.
    
    This class implements the core logging functionality including level filtering,
    context enrichment, and handler dispatch to multiple output destinations.
    """
    
    def __init__(self, name: str, level: int = INFO, parent: Optional['Logger'] = None):
        self.name = name
        self.level = level
        self.parent = parent
        self.handlers: List[Handler] = []
        self.children: Dict[str, Logger] = {}
        self._lock = threading.RLock()
    
    def log(self, level: int, message: str, **context) -> None:
        """
        Main logging entry point with filtering and dispatch.
        
        Args:
            level: Severity level of the message
            message: Human-readable description of the event
            **context: Additional key-value pairs for structured data
        """
        # TODO 1: Check if message should be logged using should_log function
        # TODO 2: Get current timestamp in ISO 8601 format
        # TODO 3: Retrieve current context from context manager and merge with **context
        # TODO 4: Create LogRecord object with all information
        # TODO 5: Dispatch record to all configured handlers
        # TODO 6: If no handlers on this logger, propagate to parent logger
        # Hint: Use threading.RLock to ensure thread safety during handler iteration
        pass
    
    def debug(self, message: str, **context) -> None:
        """Log message at DEBUG level."""
        # TODO: Call self.log with DEBUG level
        pass
    
    def info(self, message: str, **context) -> None:
        """Log message at INFO level."""
        # TODO: Call self.log with INFO level
        pass
    
    def warn(self, message: str, **context) -> None:
        """Log message at WARN level."""
        # TODO: Call self.log with WARN level
        pass
    
    def error(self, message: str, **context) -> None:
        """Log message at ERROR level."""
        # TODO: Call self.log with ERROR level
        pass
    
    def fatal(self, message: str, **context) -> None:
        """Log message at FATAL level."""
        # TODO: Call self.log with FATAL level
        pass
    
    def add_handler(self, handler) -> None:
        """Add handler to this logger's dispatch list."""
        # TODO 1: Acquire lock for thread safety
        # TODO 2: Add handler to handlers list if not already present
        # TODO 3: Initialize handler if needed
        pass
    
    def get_child(self, name: str) -> 'Logger':
        """Get or create child logger with inherited configuration."""
        # TODO 1: Check if child already exists in self.children
        # TODO 2: If not, create new Logger with self as parent
        # TODO 3: Inherit level and handlers from parent
        # TODO 4: Store in children dictionary and return
        # Hint: Child logger name should be f"{self.name}.{name}"
        pass


# Global logger registry
_logger_registry: Dict[str, Logger] = {}
_registry_lock = threading.RLock()


def get_logger(name: str) -> Logger:
    """
    Get or create logger instance with the given name.
    
    Args:
        name: Hierarchical logger name (e.g., "app.database.connection")
        
    Returns:
        Logger instance for the given name
    """
    # TODO 1: Check if logger already exists in registry
    # TODO 2: If not, create logger hierarchy based on name (split by '.')
    # TODO 3: Ensure parent loggers are created first
    # TODO 4: Store new loggers in registry
    # TODO 5: Return requested logger
    # Hint: Use _registry_lock for thread safety
    pass
```

**Handler Base Class (`handlers/__init__.py`)**
```python
"""Base handler class and common handler utilities."""
from abc import ABC, abstractmethod
from typing import Optional
from ..logger import LogRecord


class Handler(ABC):
    """Abstract base class for all log handlers."""
    
    def __init__(self):
        self.formatter: Optional['Formatter'] = None
    
    @abstractmethod
    def emit(self, record: LogRecord) -> None:
        """
        Process and output a single log record.
        
        Args:
            record: LogRecord to be processed and output
        """
        pass
    
    def flush(self) -> None:
        """Ensure all buffered records are written to destination."""
        # Default implementation does nothing - override if handler buffers
        pass
    
    def close(self) -> None:
        """Release resources and perform cleanup."""
        # Default implementation does nothing - override if handler has resources
        pass
    
    def set_formatter(self, formatter) -> None:
        """Configure formatter for this handler."""
        self.formatter = formatter
    
    def format_record(self, record: LogRecord) -> str:
        """Format record using configured formatter."""
        if self.formatter:
            return self.formatter.format(record)
        else:
            # Default formatting if no formatter configured
            return f"[{record.timestamp}] {record.level}: {record.message}"
```

#### File Structure Setup

Create the recommended directory structure with these commands:

```bash
mkdir -p structured_logging/{handlers,formatters,context,utils,config,tests/{unit,integration,performance}}
touch structured_logging/__init__.py
touch structured_logging/{logger,handlers,formatters,context,utils,config}/__init__.py
```

#### Milestone Checkpoints

**Milestone 1 Checkpoint - Logger Core**
After implementing the core logger functionality:

```bash
# Test basic logging functionality
python -c "
from structured_logging import get_logger
logger = get_logger('test')
logger.info('Test message', user_id=123)
logger.error('Error occurred', error_code='E001')
"
```

Expected behavior:
- Messages should appear on console (if console handler configured)
- DEBUG messages should be filtered out when logger level is INFO or higher
- Context fields (user_id, error_code) should be included in output
- Multiple threads calling logger methods should not corrupt output

**Milestone 2 Checkpoint - Structured Output**
After implementing JSON formatting:

```python
# Test JSON output formatting
from structured_logging import get_logger
from structured_logging.formatters.json import JSONFormatter
from structured_logging.handlers.console import ConsoleHandler

logger = get_logger('test')
handler = ConsoleHandler()
handler.set_formatter(JSONFormatter())
logger.add_handler(handler)

logger.info('User action', action='login', user_id=123, success=True)
```

Expected output (single line JSON):
```json
{"timestamp":"2024-01-15T10:30:45.123456Z","level":20,"message":"User action","logger_name":"test","action":"login","user_id":123,"success":true}
```

**Milestone 3 Checkpoint - Context & Correlation**
After implementing context propagation:

```python
# Test context propagation
from structured_logging import get_logger, with_context
from structured_logging.context import correlation_id

logger = get_logger('test')

with with_context(request_id=correlation_id(), user_id=456):
    logger.info('Processing request')
    
    def nested_function():
        logger.info('Inside nested function')  # Should inherit context
    
    nested_function()
```

Expected behavior:
- Both log messages should contain the same request_id and user_id
- Context should propagate through nested function calls without explicit passing
- Each request should have a unique correlation ID

#### Language-Specific Hints

- Use `threading.RLock()` instead of `Lock()` for reentrant locking in logger hierarchy
- `datetime.utcnow().isoformat() + 'Z'` provides ISO 8601 timestamps
- `**dict1, **dict2` syntax merges dictionaries in Python 3.5+
- `threading.local()` provides thread-local storage for context propagation
- `contextvars` module (Python 3.7+) handles async context better than threading.local
- `sys.stderr.write()` avoids recursive logging when reporting logging system errors
- Use `json.dumps(separators=(',', ':'))` for compact JSON output
- `functools.wraps` preserves function metadata in decorators for context managers

#### Common Debugging Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No log output appears | No handlers configured or wrong level | Add console handler, check logger.level vs message level |
| Thread safety errors | Missing locks around shared data | Use threading.RLock around handler lists and logger registry |
| Context not propagating | Thread-local storage not inherited | Implement proper context copying in nested calls |
| JSON serialization errors | Non-serializable objects in context | Use SafeJSONEncoder and safe_serialize function |
| Performance degradation | Synchronous I/O in handlers | Implement async handlers or background thread dispatch |


## Data Model

> **Milestone(s):** This section defines the foundational data structures used across all three milestones, with LogRecord and LogLevel supporting Milestone 1 (Logger Core), structured output requirements for Milestone 2 (Structured Output), and Context objects enabling Milestone 3 (Context & Correlation).

The data model forms the backbone of our structured logging system, defining how log information is represented, organized, and processed throughout the entire logging pipeline. Think of the data model as the **blueprint for a filing system** in a busy law firm - every piece of information has a specific place, format, and relationship to other pieces, ensuring that when thousands of documents (log records) flow through the system daily, they can be efficiently organized, searched, and correlated.

Just as a law firm needs consistent document formats to handle cases involving multiple lawyers, departments, and time periods, our logging system needs consistent data structures to handle log messages from multiple threads, services, and request contexts. The data model ensures that whether a log record originates from a background task, an HTTP request handler, or an async coroutine, it follows the same structure and can be processed uniformly by formatters and handlers.

![Data Model Relationships](./diagrams/data-model-relationships.svg)

The three core data structures - `LogRecord`, `LogLevel`, and `Context` - work together to provide a complete logging solution. The `LogRecord` serves as the central container for all log information, the `LogLevel` provides a hierarchy for filtering and prioritization, and the `Context` enables correlation and enrichment across related operations.

> **Design Principle**: Every piece of log data has a well-defined structure and type, enabling consistent processing, serialization, and querying regardless of the log's origin or destination.

### LogRecord Structure

The `LogRecord` represents a single log entry in our system, containing all the information necessary to understand what happened, when it happened, where it happened, and under what circumstances. Think of a `LogRecord` as a **standardized incident report form** that emergency responders fill out - it has specific fields for date, time, location, severity, description, and context, ensuring that no matter who creates the report or where it's processed, all the essential information is captured in a consistent format.

The `LogRecord` structure balances completeness with performance, including only the fields necessary for effective debugging and monitoring while remaining lightweight enough for high-volume logging scenarios.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `str` | ISO 8601 formatted timestamp indicating when the log record was created, with microsecond precision |
| `level` | `int` | Numeric log level (10-50) indicating the severity and importance of the message |
| `message` | `str` | Human-readable description of the logged event, formatted with any provided arguments |
| `logger_name` | `str` | Hierarchical name of the logger that created this record (e.g., "app.database.connection") |
| `context` | `Dict[str, Any]` | Key-value pairs providing additional structured information about the logged event |

The `timestamp` field uses ISO 8601 format (e.g., "2023-10-15T14:30:45.123456Z") to ensure consistent parsing across different systems and time zones. This format is chosen over Unix timestamps because it's human-readable in log files while remaining machine-parseable for aggregation systems.

The `level` field stores the numeric representation of the log level rather than the string name for efficient filtering operations. Converting "DEBUG" string comparisons to integer comparisons (10 < 20) provides significant performance benefits when processing high volumes of log messages.

The `logger_name` captures the complete hierarchical path of the logger, enabling log aggregation and filtering by component or subsystem. For example, "app.web.auth.login" indicates this log came from the login component of the authentication subsystem of the web layer.

> **Decision: Flat LogRecord Structure**
> - **Context**: We need to balance query flexibility with serialization performance and cross-language compatibility.
> - **Options Considered**: 
>   1. Nested object structure with separate metadata objects
>   2. Flat structure with all fields at the top level
>   3. Extensible structure with custom field registration
> - **Decision**: Flat structure with fixed core fields and flexible context dictionary
> - **Rationale**: Flat structures serialize faster to JSON, are easier to query in log aggregation systems, and avoid the complexity of nested object validation while still providing extensibility through the context field
> - **Consequences**: Enables fast JSON serialization and simple log queries, but requires discipline to avoid context field name conflicts

The `context` field serves as the extensibility point for the `LogRecord`, allowing arbitrary structured data to be attached to log entries without modifying the core structure. This field supports the correlation and enrichment patterns needed in distributed systems.

**Context Field Usage Patterns:**

| Usage Pattern | Example Keys | Description |
|---------------|--------------|-------------|
| Request Correlation | `request_id`, `trace_id`, `span_id` | Identifiers linking related log entries across service boundaries |
| User Context | `user_id`, `session_id`, `organization_id` | Identity information for access correlation and debugging |
| Performance Metrics | `duration_ms`, `query_time`, `cache_hit` | Timing and performance data for monitoring and optimization |
| Error Details | `error_code`, `stack_trace`, `retry_count` | Structured error information for debugging and alerting |
| Business Context | `order_id`, `product_id`, `workflow_step` | Domain-specific identifiers for business logic debugging |

The `LogRecord` design emphasizes **immutability** - once created, a log record should not be modified. This prevents race conditions in multi-threaded environments and ensures that log records remain consistent as they flow through the handler dispatch system.

### Log Level Hierarchy

The log level hierarchy provides a standardized way to categorize log messages by importance and urgency, enabling runtime filtering and appropriate routing of messages to different output destinations. Think of log levels as a **hospital triage system** - just as medical staff categorize patients by urgency (critical, urgent, standard, routine) to ensure the most important cases receive immediate attention, log levels categorize messages so that critical errors are never missed while verbose debug information can be filtered out in production.

The five-level hierarchy balances granularity with simplicity, providing enough categories to enable meaningful filtering without creating confusion about which level to choose for a given message.

| Level Name | Numeric Value | Usage Guidelines | Production Visibility |
|------------|---------------|------------------|----------------------|
| `DEBUG` | 10 | Detailed diagnostic information for development troubleshooting | Typically filtered out |
| `INFO` | 20 | General application flow and business logic milestones | Selectively logged |
| `WARN` | 30 | Warning conditions that don't prevent operation but may indicate problems | Always logged |
| `ERROR` | 40 | Error conditions that prevent specific operations but don't crash the application | Always logged and monitored |
| `FATAL` | 50 | Critical errors that may cause application shutdown or data corruption | Always logged, triggers alerts |

The numeric values are spaced by 10 to allow for future intermediate levels if needed, following the convention established by systems like Python's logging module. This spacing provides flexibility while maintaining the core five-level structure.

**Level Filtering Behavior:**

The log level filtering follows a simple rule: messages are processed only if their level is greater than or equal to the configured minimum level. For example, a logger configured at `INFO` level (20) will process `INFO`, `WARN`, `ERROR`, and `FATAL` messages while filtering out `DEBUG` messages.

| Configured Level | DEBUG (10) | INFO (20) | WARN (30) | ERROR (40) | FATAL (50) |
|------------------|------------|-----------|-----------|-----------|------------|
| DEBUG (10) | ✓ Process | ✓ Process | ✓ Process | ✓ Process | ✓ Process |
| INFO (20) | ✗ Filter | ✓ Process | ✓ Process | ✓ Process | ✓ Process |
| WARN (30) | ✗ Filter | ✗ Filter | ✓ Process | ✓ Process | ✓ Process |
| ERROR (40) | ✗ Filter | ✗ Filter | ✗ Filter | ✓ Process | ✓ Process |
| FATAL (50) | ✗ Filter | ✗ Filter | ✗ Filter | ✗ Filter | ✓ Process |

> **Decision: Numeric Log Levels with Fixed Hierarchy**
> - **Context**: We need efficient runtime filtering and clear severity ordering for both humans and machines.
> - **Options Considered**: 
>   1. String-based levels with runtime string comparison
>   2. Numeric levels with fixed integer values
>   3. Configurable custom level hierarchies
> - **Decision**: Fixed numeric hierarchy with standard five levels
> - **Rationale**: Integer comparison (`message_level >= configured_level`) is faster than string comparison, fixed hierarchy prevents configuration errors and ensures consistent behavior across environments, five levels provide sufficient granularity without decision paralysis
> - **Consequences**: Enables fast filtering operations and consistent cross-team usage, but prevents custom level definitions for specialized use cases

The level hierarchy supports **inheritance** in logger hierarchies - child loggers inherit the minimum level from their parent unless explicitly overridden. This enables configuration patterns where an entire subsystem's logging level can be controlled by setting the parent logger's level.

**Runtime Level Changes:**

Log levels can be modified at runtime without requiring application restart, enabling dynamic debugging and troubleshooting. This capability is essential for production environments where restarting applications to enable debug logging is not feasible.

The level change operation must be thread-safe and atomic to prevent inconsistent filtering behavior. Changes take effect immediately for new log messages, but do not affect log records already in the processing pipeline.

### Context Data Model

The context data model enables **correlation and enrichment** of log records with structured metadata that flows through the application execution context. Think of the context as a **digital breadcrumb trail** that follows a request or operation through the system - as the request moves from HTTP parsing to database queries to external API calls, each component can add relevant information to the trail while benefiting from information added by previous components.

Context propagation is essential for debugging distributed systems and understanding the relationships between log entries that may be separated by time, threads, or even service boundaries.

**Context Structure and Semantics:**

| Aspect | Implementation | Description |
|---------|----------------|-------------|
| Storage | `Dict[str, Any]` | Key-value pairs with string keys and arbitrary JSON-serializable values |
| Inheritance | Parent → Child | Child contexts inherit all key-value pairs from their parent context |
| Scope | Thread/Task Local | Context is automatically scoped to the current execution thread or async task |
| Lifecycle | Request Bounded | Context typically created at request start and cleaned up at request end |
| Propagation | Automatic | Context flows through function calls without explicit parameter passing |

The context data model supports several inheritance and propagation patterns that enable both automatic context flow and explicit context management.

**Context Inheritance Patterns:**

| Pattern | Use Case | Behavior | Example |
|---------|----------|----------|---------|
| Automatic Inheritance | HTTP request processing | Child operations inherit request ID and user context | Request handler → Database call → Cache lookup |
| Explicit Context Addition | Component-specific metadata | Components add their own context while preserving parent context | Authentication adds user ID, database adds query time |
| Context Isolation | Background tasks | New context created for background operations | Async task spawned with empty context |
| Context Override | Error handling | Specific fields can be overridden in child contexts | Retry logic overrides attempt count |

> **Decision: Thread-Local Context Storage with Async Bridge**
> - **Context**: We need context to propagate automatically through function calls while supporting both synchronous and asynchronous execution models.
> - **Options Considered**: 
>   1. Explicit context parameter passing through all function calls
>   2. Thread-local storage for synchronous code only
>   3. Thread-local storage with async context variable bridge
> - **Decision**: Thread-local storage with automatic async context preservation
> - **Rationale**: Thread-local storage eliminates the need to modify all function signatures while providing automatic propagation, async context variables ensure context survives task switches and coroutine boundaries, combination supports both sync and async patterns without code changes
> - **Consequences**: Enables seamless context propagation and cleaner function signatures, but requires careful lifecycle management to prevent context leaks

**Context Lifecycle Management:**

Context objects follow a clear lifecycle to prevent memory leaks and ensure proper isolation between different operations:

1. **Creation**: New context created at operation boundary (HTTP request, background task, etc.)
2. **Population**: Initial context fields added (request ID, user information, etc.)
3. **Inheritance**: Child operations inherit context and may add additional fields
4. **Propagation**: Context flows through synchronous calls via thread-local storage
5. **Async Bridge**: Context preserved across async/await boundaries using context variables
6. **Cleanup**: Context cleared at operation completion to prevent memory leaks

**Context Field Naming Conventions:**

To prevent field name conflicts and enable consistent querying, context fields follow standardized naming patterns:

| Field Category | Naming Pattern | Examples | Purpose |
|----------------|----------------|----------|----------|
| Correlation IDs | `{scope}_id` | `request_id`, `trace_id`, `span_id` | Linking related log entries |
| User Context | `user_{attribute}` | `user_id`, `user_role`, `user_org` | Identity and authorization context |
| Performance | `{metric}_{unit}` | `duration_ms`, `memory_mb`, `cpu_percent` | Performance monitoring data |
| Business Data | `{domain}_{entity}` | `order_id`, `product_sku`, `workflow_step` | Business logic context |
| Component Info | `{component}_{detail}` | `db_query`, `api_endpoint`, `cache_key` | Component-specific information |

**Context Serialization and Size Limits:**

Context data must remain JSON-serializable to support structured output formats. The context system includes size estimation and circular reference detection to prevent serialization failures and excessive memory usage.

| Constraint | Limit | Behavior When Exceeded |
|------------|--------|------------------------|
| Maximum Fields | 50 | Additional fields logged as warning, oldest fields preserved |
| Maximum Field Size | 1KB per field | Large fields truncated with indication of truncation |
| Maximum Total Size | 10KB per context | Context pruned starting with newest non-correlation fields |
| Serialization Depth | 10 levels | Deep nested objects flattened or truncated |
| Circular References | Not allowed | Detected and replaced with reference marker |

The context data model enables powerful correlation and debugging capabilities while maintaining performance and reliability constraints necessary for production logging systems.

### Implementation Guidance

This subsection provides concrete implementation patterns and starter code for the core data structures, focusing on thread safety, serialization, and performance considerations that junior developers commonly struggle with.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Timestamp Generation | `datetime.utcnow().isoformat()` | `time.time_ns()` with custom ISO formatter |
| JSON Serialization | `json.dumps()` with custom encoder | `orjson` or `ujson` for performance |
| Thread-Local Storage | `threading.local()` | `contextvars` for async support |
| Context Size Estimation | `len(json.dumps())` | Custom recursive size calculator |

**Recommended Module Structure:**

```
structured_logging/
  core/
    __init__.py           ← exports LogRecord, LogLevel constants
    log_record.py         ← LogRecord class and factory functions
    log_level.py          ← log level constants and filtering utilities
    context.py            ← context management and propagation
  utils/
    serialization.py      ← safe JSON encoding and size estimation
    thread_safety.py      ← thread-safe data structures
  tests/
    test_log_record.py    ← LogRecord creation and serialization tests
    test_context.py       ← context propagation and inheritance tests
```

**Infrastructure Starter Code:**

Complete implementation of thread-safe utilities and serialization helpers:

```python
# utils/serialization.py
import json
import sys
from typing import Any, Dict
from decimal import Decimal
from datetime import datetime, date

class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types safely."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return f"<{obj.__class__.__name__} object>"
        elif hasattr(obj, '__str__'):
            return str(obj)
        else:
            return f"<non-serializable: {type(obj).__name__}>"

def safe_serialize(data: Dict[str, Any], max_depth: int = 10) -> str:
    """Safely serialize dictionary to JSON with circular reference protection."""
    def _check_depth(obj, current_depth=0):
        if current_depth >= max_depth:
            return f"<max_depth_exceeded: {max_depth}>"
        
        if isinstance(obj, dict):
            return {k: _check_depth(v, current_depth + 1) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_check_depth(item, current_depth + 1) for item in obj]
        else:
            return obj
    
    try:
        safe_data = _check_depth(data)
        return json.dumps(safe_data, cls=SafeJSONEncoder, separators=(',', ':'))
    except (TypeError, ValueError) as e:
        return json.dumps({"serialization_error": str(e)}, separators=(',', ':'))

def estimate_serialized_size(data: Dict[str, Any]) -> int:
    """Estimate JSON size without full serialization for performance."""
    # TODO 1: Implement recursive size estimation
    # TODO 2: Handle strings, numbers, booleans with known sizes  
    # TODO 3: Estimate dict/list overhead (brackets, commas, quotes)
    # TODO 4: Return estimated byte count
    # Hint: Use sys.getsizeof() as baseline, add JSON syntax overhead
    pass

# utils/thread_safety.py
import threading
from typing import Any, Dict

class ThreadSafeDict:
    """Thread-safe dictionary wrapper for shared logging context."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Re-entrant lock for nested access
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
    
    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            self._data.update(other)
    
    def copy(self) -> Dict[str, Any]:
        with self._lock:
            return self._data.copy()
    
    def clear(self) -> None:
        with self._lock:
            self._data.clear()

class ThreadSafeCounter:
    """Thread-safe counter for generating unique IDs."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
```

**Core Logic Skeleton Code:**

```python
# core/log_level.py
DEBUG = 10
INFO = 20  
WARN = 30
ERROR = 40
FATAL = 50

# Level name mappings for display
LEVEL_NAMES = {
    DEBUG: "DEBUG",
    INFO: "INFO", 
    WARN: "WARN",
    ERROR: "ERROR",
    FATAL: "FATAL"
}

def should_log(message_level: int, configured_level: int) -> bool:
    """Determine if message should be logged based on level filtering."""
    # TODO 1: Compare message_level against configured_level
    # TODO 2: Return True if message_level >= configured_level
    # TODO 3: Handle invalid level values (default to allowing the message)
    # Hint: Simple integer comparison, but validate inputs first
    pass

# core/log_record.py
from datetime import datetime
from typing import Dict, Any
from .log_level import LEVEL_NAMES

class LogRecord:
    """Immutable log record containing all information for a single log entry."""
    
    def __init__(self, timestamp: str, level: int, message: str, 
                 logger_name: str, context: Dict[str, Any]):
        # TODO 1: Store all parameters as instance attributes
        # TODO 2: Make context a copy to ensure immutability
        # TODO 3: Validate that level is a known log level value
        # TODO 4: Ensure timestamp is ISO 8601 format string
        # Hint: Use context.copy() to prevent external modification
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary for JSON serialization."""
        # TODO 1: Create dictionary with all core fields
        # TODO 2: Add level_name field for human readability  
        # TODO 3: Merge in context fields at top level
        # TODO 4: Return complete dictionary ready for JSON serialization
        # Hint: Use LEVEL_NAMES mapping to get level_name from numeric level
        pass
    
    @classmethod
    def create(cls, level: int, message: str, logger_name: str, 
               context: Dict[str, Any] = None) -> 'LogRecord':
        """Factory method to create LogRecord with current timestamp."""
        # TODO 1: Generate ISO 8601 timestamp string
        # TODO 2: Handle None context by providing empty dict
        # TODO 3: Create and return LogRecord instance
        # TODO 4: Consider timezone handling for timestamp
        # Hint: Use datetime.utcnow().isoformat() + 'Z' for UTC timestamp
        pass

# core/context.py
import threading
import contextvars
from typing import Dict, Any, Optional

# Context variable for async context preservation
_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar('log_context', default={})

# Thread-local storage for sync context
_thread_local = threading.local()

class LoggingContext:
    """Manages logging context with thread-local and async support."""
    
    @staticmethod
    def get_current() -> Dict[str, Any]:
        """Get current logging context, trying async context first."""
        # TODO 1: Try to get context from contextvars (for async)
        # TODO 2: Fall back to thread-local storage (for sync)
        # TODO 3: Return empty dict if no context is set
        # TODO 4: Handle AttributeError from thread-local access
        # Hint: Use try/except for both contextvar and thread-local access
        pass
    
    @staticmethod  
    def set_current(context: Dict[str, Any]) -> None:
        """Set current logging context in both async and sync storage."""
        # TODO 1: Set context in contextvars for async compatibility
        # TODO 2: Set context in thread-local storage for sync compatibility  
        # TODO 3: Handle case where context is None
        # TODO 4: Ensure context is copied to prevent external modification
        # Hint: Set both _context_var and _thread_local.context
        pass
    
    @staticmethod
    def add_fields(**fields: Any) -> None:
        """Add fields to current context without replacing existing context."""
        # TODO 1: Get current context dictionary
        # TODO 2: Create new context with existing fields plus new fields
        # TODO 3: Set the updated context as current
        # TODO 4: Handle case where no current context exists
        # Hint: Use get_current(), update with new fields, call set_current()
        pass
    
    @staticmethod
    def clear() -> None:
        """Clear current logging context."""
        # TODO 1: Clear contextvars context
        # TODO 2: Clear thread-local context  
        # TODO 3: Handle AttributeError if thread-local not initialized
        # TODO 4: Ensure both storage mechanisms are cleared
        # Hint: Set empty dict in both storage locations
        pass
```

**Language-Specific Implementation Hints:**

- **Thread Safety**: Use `threading.RLock()` instead of `Lock()` for context management to allow re-entrant access within the same thread
- **Async Context**: Python's `contextvars` automatically handles async context preservation across `await` boundaries
- **JSON Serialization**: The `json` module's `separators=(',', ':')` parameter eliminates whitespace for compact output
- **Immutability**: Use `context.copy()` in LogRecord constructor to prevent external modification of context data
- **Performance**: Consider `__slots__` on LogRecord class to reduce memory overhead for high-volume logging

**Milestone Checkpoint:**

After implementing the data model:

1. **Run Unit Tests**: `python -m pytest tests/test_log_record.py tests/test_context.py -v`
2. **Expected Output**: All tests pass, demonstrating LogRecord creation, serialization, and context propagation
3. **Manual Verification**:
   ```python
   from core.log_record import LogRecord
   from core.context import LoggingContext
   
   # Test LogRecord creation and serialization
   record = LogRecord.create(INFO, "Test message", "test.logger", {"user_id": 123})
   print(record.to_dict())  # Should show structured dictionary
   
   # Test context propagation
   LoggingContext.set_current({"request_id": "abc123"})
   context = LoggingContext.get_current()
   print(context)  # Should show {"request_id": "abc123"}
   ```

4. **Signs of Problems**:
   - **Serialization failures**: Check SafeJSONEncoder implementation and circular reference handling  
   - **Context not propagating**: Verify both contextvars and thread-local storage are being set
   - **Thread safety issues**: Add logging to identify race conditions in concurrent tests
   - **Memory leaks**: Ensure context.clear() is called and contexts are properly scoped

The data model implementation provides the foundation for all subsequent logging functionality, ensuring consistent data representation and thread-safe operations across the entire system.


## Logger Core Design

> **Milestone(s):** This section primarily addresses Milestone 1 (Logger Core) by defining the central logging infrastructure, hierarchy system, level filtering, thread safety mechanisms, and handler dispatch patterns that form the foundation for all subsequent structured logging capabilities.

### Mental Model: The News Organization

Think of the logging system as a large news organization with multiple departments and bureaus. The **logger hierarchy** works like the organizational chart — the main newsroom (`root` logger) oversees regional bureaus (`child` loggers), which in turn manage local reporters (`leaf` loggers). Each level inherits editorial standards and distribution channels from above, but can add their own specific context and formatting.

**Log level filtering** operates like editorial decisions about what stories make it to print. A DEBUG message is like a reporter's personal note — interesting for development but filtered out in production. An ERROR message is like breaking news that must reach every output channel immediately. The **handler dispatch system** functions like the news distribution network — the same story (log record) gets formatted appropriately and sent to newspapers (console), archives (files), and wire services (remote collectors) simultaneously.

This mental model helps understand why thread safety matters (multiple reporters filing stories concurrently), why context propagation is essential (maintaining story lineage), and why handler failure recovery is critical (ensuring important news reaches at least one destination).

![System Architecture Overview](./diagrams/system-architecture.svg)

### Logger Hierarchy System

The logger hierarchy provides a tree-structured namespace that mirrors application architecture while enabling configuration inheritance and context flow. This design allows fine-grained control over logging behavior at different application layers without requiring explicit configuration at every level.

#### Hierarchy Structure and Naming

The logger hierarchy follows a dot-separated naming convention that reflects application structure. The root logger sits at the top with an empty name, while child loggers use qualified names like `web`, `web.auth`, `web.auth.jwt`, and `database.connection.pool`. This naming strategy provides intuitive organization and enables powerful inheritance patterns.

| Logger Name | Parent | Level Inherited | Handlers Inherited | Context Inherited |
|-------------|--------|-----------------|-------------------|-------------------|
| `` (root) | None | INFO (default) | [ConsoleHandler] | {} |
| `web` | root | INFO | [ConsoleHandler] | {service: "web"} |
| `web.auth` | web | INFO | [ConsoleHandler] | {service: "web", component: "auth"} |
| `web.auth.jwt` | web.auth | DEBUG (overridden) | [ConsoleHandler, FileHandler] | {service: "web", component: "auth", module: "jwt"} |
| `database` | root | WARN (overridden) | [ConsoleHandler, RemoteHandler] | {service: "database"} |

Each logger maintains references to its parent and children, creating a bidirectional tree structure that supports both top-down configuration propagation and bottom-up context enrichment. The `Logger` type encapsulates this relationship:

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Dot-separated qualified name identifying this logger's position in hierarchy |
| `level` | int | Minimum log level for this logger (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50) |
| `parent` | Logger | Reference to parent logger for inheritance, None only for root logger |
| `handlers` | List[Handler] | Output destinations specific to this logger, combined with inherited handlers |
| `children` | Dict[str, Logger] | Map of child logger names to Logger instances for hierarchy navigation |
| `context` | Dict[str, Any] | Key-value pairs automatically attached to all log records from this logger |
| `propagate` | bool | Whether log records should bubble up to parent handlers (default True) |

#### Configuration Inheritance Behavior

Configuration inheritance flows down the hierarchy, with child loggers inheriting level, handlers, and context from their parents while maintaining the ability to override or extend these settings. This inheritance mechanism reduces configuration duplication while preserving flexibility for special cases.

**Level inheritance** follows a "most specific wins" pattern. If a logger doesn't have an explicitly configured level, it inherits from its parent, walking up the hierarchy until it finds a configured level or reaches the root (which defaults to INFO). This approach ensures predictable behavior while minimizing configuration overhead.

**Handler inheritance** combines parent and child handlers unless explicitly disabled. When a log record is generated, it gets dispatched to the logger's own handlers plus all inherited handlers from parent loggers. The `propagate` flag controls whether records bubble up to parent handlers, enabling scenarios where sensitive components log only to specific destinations.

**Context inheritance** merges parent context fields with child-specific fields, with child values taking precedence for duplicate keys. This creates a natural layering where service-level context (like `service: "web"`) gets automatically combined with component-level context (like `component: "auth"`) and request-specific context (like `request_id: "abc123"`).

> **Design Insight**: The inheritance model balances convenience with control. Most loggers inherit sensible defaults, reducing boilerplate configuration, while performance-critical or security-sensitive components can override settings precisely where needed.

#### Logger Factory and Lifecycle

The logger hierarchy uses a factory pattern to ensure singleton behavior and proper parent-child linking. The `get_logger(name)` function serves as the single entry point for logger creation, maintaining a global registry to prevent duplicate instances and ensure consistent hierarchy structure.

**Logger Creation Process**:
1. Parse the requested logger name to identify hierarchy path components
2. Walk the hierarchy from root to target, creating any missing intermediate loggers
3. Link each new logger to its parent and register it in the parent's children dictionary
4. Initialize the new logger with inherited configuration from its parent
5. Register the logger in the global name-to-instance mapping for future lookups
6. Return the logger instance (creating or retrieving existing)

The factory maintains thread safety through a global lock during logger creation, ensuring that concurrent requests for the same logger name return the same instance without race conditions. Once created, loggers are immutable in their hierarchy relationships but mutable in their configuration to support runtime level changes.

> **Architecture Decision: Singleton Logger Instances**
> - **Context**: Multiple parts of an application may request loggers with the same name
> - **Options Considered**: 
>   1. Create new logger instances for each request
>   2. Use singleton pattern with global registry
>   3. Require explicit logger instance passing
> - **Decision**: Singleton pattern with global registry
> - **Rationale**: Ensures consistent configuration and context across all usage sites, eliminates need for dependency injection of logger instances, simplifies logger hierarchy management
> - **Consequences**: Global state requires thread safety, enables convenient logger access but reduces testability, configuration changes affect all code using the same logger name

### Log Level Filtering

Log level filtering provides the primary mechanism for controlling logging verbosity in different environments. The filtering system must be efficient enough for hot code paths while supporting runtime reconfiguration without application restart.

#### Level Hierarchy and Numeric Ordering

The log level system uses numeric values to enable efficient comparison and filtering. Higher numbers indicate higher severity, making the filtering logic a simple numeric comparison rather than complex string matching or lookup operations.

| Level Name | Numeric Value | Typical Usage | Production Visibility |
|------------|---------------|---------------|----------------------|
| `DEBUG` | 10 | Detailed diagnostic information | Hidden |
| `INFO` | 20 | General application flow | Visible |
| `WARN` | 30 | Unusual but handled conditions | Visible |
| `ERROR` | 40 | Error conditions requiring attention | Always visible |
| `FATAL` | 50 | Critical errors causing termination | Always visible |

The `should_log(message_level, configured_level)` function implements the core filtering logic with a simple comparison: `message_level >= configured_level`. This design ensures that setting a logger to WARN level will include WARN, ERROR, and FATAL messages while filtering out DEBUG and INFO messages.

#### Efficient Level Checking

The level filtering system optimizes for the common case where messages are filtered out. The `log()` method performs level checking before any expensive operations like string formatting, context gathering, or handler dispatch. This early-exit pattern prevents performance degradation when verbose logging is enabled in hot code paths.

**Level Check Optimization Strategy**:
1. Check message level against logger's configured level using simple integer comparison
2. Exit immediately if message should be filtered, avoiding all downstream processing
3. Only proceed with LogRecord creation, formatting, and handler dispatch for messages that pass filtering
4. Cache the logger's effective level to avoid walking parent hierarchy on every message

The effective level calculation walks up the logger hierarchy to find the nearest explicitly configured level, but this traversal only occurs when the logger's level changes, not on every log message. The result gets cached in the logger instance until configuration changes invalidate the cache.

#### Runtime Level Reconfiguration

The logging system supports runtime level changes without application restart, enabling dynamic debugging and troubleshooting in production environments. Level changes propagate through the hierarchy using a cache invalidation mechanism that ensures consistent behavior across all loggers.

| Operation | Thread Safety | Propagation | Performance Impact |
|-----------|---------------|-------------|-------------------|
| `set_level(level)` | Write lock on logger | Invalidates child caches | O(1) for single logger |
| `get_effective_level()` | Read lock on hierarchy | Walks up parent chain | O(depth) cached result |
| `should_log(level)` | Read-only comparison | No propagation needed | O(1) cached lookup |

**Level Change Propagation**:
1. Acquire write lock on target logger to prevent concurrent modifications
2. Update logger's configured level and invalidate its cached effective level
3. Walk all descendant loggers and invalidate their cached effective levels
4. Release locks and allow normal logging operations to resume
5. Next log operation on each affected logger recalculates effective level as needed

> **Design Insight**: Runtime level changes are intentionally rare operations that can afford more expensive propagation logic. The common case of level checking during normal logging remains highly optimized with cached effective levels.

![Logger Hierarchy Structure](./diagrams/logger-hierarchy.svg)

### Thread Safety Design

The logging system must handle concurrent access from multiple threads without corrupting output, losing messages, or introducing race conditions. The thread safety design balances correctness with performance, ensuring that logging operations remain fast in single-threaded scenarios while providing strong guarantees in multi-threaded environments.

#### Locking Strategy and Granularity

The thread safety design employs a multi-level locking strategy with different granularity for different operations. This approach minimizes contention while ensuring data consistency across concurrent operations.

| Component | Lock Type | Granularity | Protected Operations |
|-----------|-----------|-------------|---------------------|
| Logger Registry | RWLock | Global | Logger creation, hierarchy modification |
| Logger Instance | RWLock | Per-logger | Level changes, handler modification |
| Handler Collection | RWLock | Per-logger | Handler list modification |
| Log Record Creation | No lock | Thread-local | Timestamp, context capture |
| Handler Dispatch | Read lock | Per-handler | Output writing, formatting |

**Hierarchical Locking Protocol**:
1. Global registry lock protects logger creation and hierarchy modifications
2. Per-logger locks protect configuration changes like level and handler updates
3. Handler dispatch uses read locks to allow concurrent logging while preventing configuration changes
4. Lock ordering follows hierarchy depth to prevent deadlocks (parent before child)
5. Operations acquire minimal lock scope and release as quickly as possible

The locking protocol ensures that common operations (logging messages) require only read locks or no locks, while rare operations (configuration changes, logger creation) use write locks for consistency. This design maximizes concurrency for the typical workload while providing strong consistency guarantees.

#### Handler Dispatch Concurrency

Handler dispatch presents unique concurrency challenges because multiple threads may attempt to write to the same output destination simultaneously. The design provides thread safety at the handler level while allowing parallel dispatch to different handlers.

**Concurrent Dispatch Strategy**:
1. Each handler maintains its own internal synchronization for thread-safe writing
2. Multiple threads can dispatch to different handlers simultaneously
3. Handlers implement internal buffering and atomic write operations
4. Handler failure in one thread doesn't affect logging success in other threads
5. Error handling and retry logic operate independently per handler per thread

The `Handler` interface contract specifies that implementations must be thread-safe, allowing the logger to dispatch to multiple handlers concurrently without additional synchronization. This design pushes the thread safety responsibility to the component best positioned to handle it efficiently.

#### Memory Consistency and Visibility

Thread safety extends beyond locks to include memory consistency guarantees. The logging system ensures that configuration changes made by one thread become visible to other threads promptly and consistently.

| Memory Concern | Solution | Guarantee |
|----------------|----------|-----------|
| Level changes | Write barriers in setters | New levels visible before lock release |
| Handler modifications | Copy-on-write collections | Atomic handler list updates |
| Context propagation | Thread-local storage | Per-thread context isolation |
| Logger hierarchy | Immutable parent references | Safe concurrent hierarchy traversal |
| Cached effective levels | Volatile flags for invalidation | Cache invalidation visible across threads |

The memory consistency design uses a combination of proper locking, volatile flags for cache invalidation, and copy-on-write collections for handler lists. This approach ensures that all threads observe consistent state without requiring global synchronization for read-heavy operations.

> **Architecture Decision: Fine-Grained Locking vs. Global Synchronization**
> - **Context**: Logging occurs frequently and from many threads, requiring high-performance thread safety
> - **Options Considered**:
>   1. Single global lock for all logging operations
>   2. Per-logger locks with hierarchical ordering
>   3. Lock-free algorithms with atomic operations
> - **Decision**: Per-logger locks with read-write separation
> - **Rationale**: Global lock would serialize all logging operations hurting performance, lock-free algorithms are complex and error-prone, per-logger locks allow maximum concurrency while maintaining correctness
> - **Consequences**: Enables high-throughput concurrent logging, requires careful lock ordering to prevent deadlocks, adds complexity to configuration change operations

![Log Processing Data Flow](./diagrams/log-processing-flow.svg)

### Handler Dispatch Mechanism

The handler dispatch system routes log records to multiple output destinations while providing error isolation, failure recovery, and performance optimization. This multi-destination approach enables comprehensive logging strategies where the same message appears in local files, remote collectors, and console output simultaneously.

#### Multi-Destination Routing

Handler dispatch implements a fan-out pattern where each log record gets sent to all configured handlers for the logger and its ancestors (unless propagation is disabled). This design ensures comprehensive coverage while allowing different handlers to apply their own filtering, formatting, and delivery mechanisms.

**Dispatch Routing Process**:
1. Collect all applicable handlers from logger and ancestor chain
2. Create immutable LogRecord with timestamp, level, message, and context
3. Dispatch record to each handler concurrently (when possible)
4. Aggregate results from all handler operations
5. Handle failures according to configured error policies
6. Return success if at least one handler succeeded (or all failed)

The handler collection process walks up the logger hierarchy, gathering handlers from each ancestor until it reaches the root or encounters a logger with `propagate=False`. This creates a comprehensive handler list that respects hierarchy while allowing fine-grained control over propagation.

| Handler Type | Purpose | Concurrency | Failure Tolerance |
|--------------|---------|-------------|------------------|
| ConsoleHandler | Developer feedback | Thread-safe writes | Never fails |
| FileHandler | Local persistence | Buffered writes | Fails on disk full |
| RemoteHandler | Centralized logging | Async network calls | Fails on network issues |
| SyslogHandler | System integration | UDP fire-and-forget | Network failures ignored |

#### Error Isolation and Recovery

Handler failures must not prevent other handlers from receiving log records or cause the logging operation to fail completely. The dispatch system implements error isolation to ensure that a failed file handler doesn't prevent console output or remote logging from succeeding.

**Error Isolation Strategy**:
1. Each handler dispatch occurs within its own exception boundary
2. Handler failures get logged to a separate error logger (avoiding recursion)
3. Failed handlers can be marked as temporarily disabled with exponential backoff
4. Dispatch continues to remaining handlers even after individual failures
5. Success threshold determines overall operation success (e.g., "at least one handler succeeded")

The error isolation mechanism prevents cascading failures while providing visibility into handler health. A special error logger handles handler failures without using the same handlers that might be failing, typically writing to a simple file or console output.

**Handler Failure Recovery**:

| Failure Type | Detection | Recovery Action | Backoff Strategy |
|--------------|-----------|----------------|------------------|
| Network timeout | Exception during send | Retry with exponential backoff | 1s, 2s, 4s, up to 60s |
| Disk full | Write operation failure | Disable temporarily, retry periodically | Check every 30 seconds |
| Permission denied | File access error | Log error, disable permanently | No retry |
| Memory exhaustion | Buffer allocation failure | Drop buffered messages, continue | Immediate retry with smaller buffer |

#### Asynchronous Handler Support

High-throughput applications require asynchronous handler dispatch to prevent slow output destinations from blocking application threads. The handler system supports both synchronous and asynchronous handlers through a unified interface with different execution strategies.

**Asynchronous Dispatch Architecture**:
1. Synchronous handlers execute immediately on the calling thread
2. Asynchronous handlers queue log records for background processing
3. Background worker threads process queued records with batching optimization
4. Queue overflow protection prevents memory exhaustion during traffic spikes
5. Graceful shutdown ensures all queued records get processed before exit

The asynchronous handler implementation uses bounded queues with overflow policies to handle traffic spikes. When queues fill up, the system can either block the calling thread, drop the oldest records, or drop the newest records based on configured policy.

| Queue Policy | Behavior | Use Case |
|--------------|----------|----------|
| Block | Wait for queue space | Critical logs that must not be lost |
| Drop Oldest | Remove old records for new ones | Real-time monitoring where recent data matters |
| Drop Newest | Reject new records when full | Batch processing where old data has value |
| Best Effort | Try async, fall back to sync | Development and testing scenarios |

#### Performance Optimization

Handler dispatch performance directly affects application throughput, especially for high-frequency logging. The system implements several optimizations to minimize latency and maximize throughput while maintaining correctness.

**Dispatch Performance Optimizations**:
1. **Handler Filtering**: Handlers can implement level filtering to avoid expensive formatting for filtered messages
2. **Lazy Formatting**: Message formatting occurs only when handlers need the formatted output
3. **Batching**: Asynchronous handlers batch multiple records to amortize network and I/O costs
4. **Connection Pooling**: Network handlers reuse connections to reduce setup overhead
5. **Memory Pooling**: LogRecord objects use object pools to reduce garbage collection pressure

The lazy formatting optimization proves particularly valuable when handlers have different formatting requirements. The structured LogRecord contains raw data, while handlers request formatted output only when needed for their specific destination.

> **Performance Insight**: The dispatch mechanism optimizes for the common case where all handlers succeed quickly. Error handling and recovery logic are designed to add minimal overhead in the success path while providing comprehensive recovery in failure scenarios.

![Handler Dispatch Sequence](./diagrams/handler-dispatch-sequence.svg)

#### Common Pitfalls

⚠️ **Pitfall: Handler Deadlock in Error Logging**
When handler error logging uses the same logger hierarchy, a failure in the error logger can cause recursive calls and deadlocks. For example, if the file handler fails and tries to log the error to a logger that also uses the file handler, the system can deadlock. **Fix**: Use a separate error logging path that writes only to console or a dedicated error file, never routing through the normal handler system.

⚠️ **Pitfall: Unbounded Handler Queues**
Asynchronous handlers with unbounded queues can consume unlimited memory during traffic spikes, leading to out-of-memory crashes. Without queue limits, a slow remote handler can accumulate millions of pending records. **Fix**: Use bounded queues with explicit overflow policies, monitor queue depth, and implement backpressure mechanisms that slow down logging when queues approach capacity.

⚠️ **Pitfall: Synchronous Network Handlers in Hot Paths**
Using synchronous network handlers in performance-critical code paths causes application latency to directly depend on network latency. A slow remote logging service can make the entire application unresponsive. **Fix**: Use asynchronous handlers for all network destinations, or implement timeout limits with fallback to local handlers when remote services are slow.

⚠️ **Pitfall: Handler State Corruption Under Concurrency**
Stateful handlers that don't properly synchronize internal state can corrupt output when accessed from multiple threads. File handlers that maintain internal buffers without proper locking can interleave output from different threads. **Fix**: Ensure all handler implementations are internally thread-safe, use atomic operations for simple state, and employ proper locking for complex state management.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Threading | `threading.RLock` for logger locks | `concurrent.futures.ThreadPoolExecutor` for async handlers |
| Logger Registry | `dict` with global lock | `weakref.WeakValueDictionary` for automatic cleanup |
| Handler Queues | `queue.Queue` for async handlers | `multiprocessing.Queue` for cross-process logging |
| Context Storage | `threading.local()` for per-thread context | `contextvars` for async-aware context |
| Serialization | `json.dumps()` for LogRecord conversion | `orjson` or `ujson` for high-performance serialization |

#### Recommended File Structure

```
logging-system/
  core/
    __init__.py              ← Public API exports
    logger.py                ← Logger class and hierarchy management
    record.py                ← LogRecord data structure
    levels.py                ← Level constants and utilities
    registry.py              ← Global logger registry and factory
  handlers/
    __init__.py              ← Handler base class and common handlers
    console.py               ← ConsoleHandler implementation
    file.py                  ← FileHandler with rotation
    remote.py                ← RemoteHandler for network logging
    async_handler.py         ← Asynchronous handler wrapper
  formatters/
    __init__.py              ← Formatter base class
    json_formatter.py        ← JSON formatting implementation
  context/
    __init__.py              ← Context propagation system
    storage.py               ← Thread-local and async context storage
  tests/
    test_logger.py           ← Logger hierarchy and level tests
    test_handlers.py         ← Handler dispatch and error tests
    test_concurrency.py      ← Thread safety and race condition tests
```

#### Infrastructure Starter Code

**Level Constants and Utilities** (`levels.py`):
```python
# Complete level management system
DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
FATAL = 50

LEVEL_NAMES = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARN: 'WARN',
    ERROR: 'ERROR',
    FATAL: 'FATAL'
}

NAME_TO_LEVEL = {name: level for level, name in LEVEL_NAMES.items()}

def should_log(message_level: int, configured_level: int) -> bool:
    """Determine if message should be logged based on level filtering."""
    return message_level >= configured_level

def level_name(level: int) -> str:
    """Convert numeric level to string name."""
    return LEVEL_NAMES.get(level, f'LEVEL-{level}')

def parse_level(level_input) -> int:
    """Parse string or integer level input to numeric level."""
    if isinstance(level_input, int):
        return level_input
    if isinstance(level_input, str):
        return NAME_TO_LEVEL.get(level_input.upper(), INFO)
    return INFO
```

**Thread-Safe Collections** (`utils.py`):
```python
import threading
from typing import Dict, Any, Optional, List
import weakref

class ThreadSafeDict:
    """Thread-safe dictionary with read-write locking."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False
    
    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

class LoggerRegistry:
    """Global registry for logger instances with thread safety."""
    
    def __init__(self):
        self._loggers: Dict[str, 'Logger'] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, name: str, factory_func) -> 'Logger':
        with self._lock:
            if name in self._loggers:
                return self._loggers[name]
            
            logger = factory_func(name)
            self._loggers[name] = logger
            return logger
    
    def get(self, name: str) -> Optional['Logger']:
        with self._lock:
            return self._loggers.get(name)
    
    def clear(self) -> None:
        """Clear all loggers - primarily for testing."""
        with self._lock:
            self._loggers.clear()
```

#### Core Logic Skeleton

**Logger Class Core** (`logger.py`):
```python
from typing import List, Dict, Any, Optional
import threading
from .levels import should_log, INFO
from .record import LogRecord

class Logger:
    """Core logger implementation with hierarchy and thread safety."""
    
    def __init__(self, name: str, parent: Optional['Logger'] = None):
        self.name = name
        self.parent = parent
        self.children: Dict[str, Logger] = {}
        self.handlers: List['Handler'] = []
        self.context: Dict[str, Any] = {}
        self.propagate = True
        
        self._level: Optional[int] = None
        self._effective_level: Optional[int] = None
        self._lock = threading.RLock()
    
    def log(self, level: int, message: str, **context) -> None:
        """Main logging entry point with filtering and dispatch."""
        # TODO 1: Check if message should be logged using should_log()
        # TODO 2: Create LogRecord with current timestamp and merged context
        # TODO 3: Collect all applicable handlers from self and ancestors
        # TODO 4: Dispatch record to each handler with error isolation
        # TODO 5: Handle propagation up the hierarchy if enabled
        # Hint: Use early return for filtered messages to optimize performance
        pass
    
    def get_effective_level(self) -> int:
        """Get the effective level, walking up hierarchy if needed."""
        # TODO 1: Check if we have a cached effective level
        # TODO 2: If not cached, walk up parent chain to find configured level
        # TODO 3: Cache the result for future calls
        # TODO 4: Return the found level or default to INFO
        # Hint: Use self._lock for thread safety when updating cache
        pass
    
    def set_level(self, level: int) -> None:
        """Set logger level and invalidate caches."""
        # TODO 1: Acquire write lock to prevent concurrent access
        # TODO 2: Set self._level to new value
        # TODO 3: Invalidate self._effective_level cache
        # TODO 4: Walk all descendants and invalidate their caches
        # TODO 5: Release lock
        # Hint: Cache invalidation prevents stale level inheritance
        pass
    
    def add_handler(self, handler: 'Handler') -> None:
        """Add handler to this logger's handler list."""
        # TODO 1: Acquire lock for thread-safe handler list modification
        # TODO 2: Check if handler is already in list to avoid duplicates
        # TODO 3: Append handler to self.handlers list
        # TODO 4: Release lock
        pass
    
    def collect_handlers(self) -> List['Handler']:
        """Collect all applicable handlers from hierarchy."""
        # TODO 1: Start with empty handler list
        # TODO 2: Add all handlers from self.handlers
        # TODO 3: If propagate is True, walk up parent chain
        # TODO 4: Add handlers from each ancestor until root or propagate=False
        # TODO 5: Return combined handler list
        # Hint: Check propagate flag at each level, not just current logger
        pass

    # Convenience methods for specific levels
    def debug(self, message: str, **context) -> None:
        self.log(DEBUG, message, **context)
    
    def info(self, message: str, **context) -> None:
        self.log(INFO, message, **context)
    
    def warn(self, message: str, **context) -> None:
        self.log(WARN, message, **context)
    
    def error(self, message: str, **context) -> None:
        self.log(ERROR, message, **context)
    
    def fatal(self, message: str, **context) -> None:
        self.log(FATAL, message, **context)
```

**Logger Factory Function**:
```python
_registry = LoggerRegistry()

def get_logger(name: str = '') -> Logger:
    """Get or create logger with proper hierarchy setup."""
    # TODO 1: Use registry to get existing logger or signal creation needed
    # TODO 2: If creating new logger, parse name to find parent path
    # TODO 3: Ensure parent loggers exist, creating them recursively
    # TODO 4: Create new logger with proper parent reference
    # TODO 5: Add new logger to parent's children dict
    # TODO 6: Initialize logger with inherited configuration
    # TODO 7: Register logger in global registry
    # TODO 8: Return logger instance
    # Hint: Split name on '.' to build hierarchy path
    pass
```

#### Milestone Checkpoint

After implementing the Logger Core, verify these behaviors:

**Level Filtering Test**:
```python
logger = get_logger('test')
logger.set_level(WARN)

logger.debug("Should not appear")  # Filtered out
logger.warn("Should appear")       # Visible
logger.error("Should appear")      # Visible
```

**Hierarchy Test**:
```python
parent = get_logger('parent')
child = get_logger('parent.child')

parent.set_level(ERROR)
assert child.get_effective_level() == ERROR  # Inherits from parent
```

**Thread Safety Test**:
```python
import concurrent.futures

def log_messages(logger, thread_id):
    for i in range(100):
        logger.info(f"Message {i} from thread {thread_id}")

logger = get_logger('concurrent')
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(log_messages, logger, i) for i in range(10)]
    # Should complete without deadlock or corrupted output
```

**Handler Dispatch Test**:
```python
logger = get_logger('dispatch')
logger.add_handler(ConsoleHandler())

logger.info("Test message")
# Should appear on console without errors
```

#### Language-Specific Hints

- Use `threading.RLock()` instead of `Lock()` to allow recursive locking within the same thread
- `time.time()` provides sufficient timestamp precision for most logging use cases  
- `sys.stderr.write()` is thread-safe and faster than `print()` for console handlers
- Use `**kwargs` for context parameters to provide clean API for arbitrary fields
- `weakref.WeakValueDictionary` can help prevent memory leaks in logger registry for long-running applications
- Consider using `__slots__` in LogRecord class to reduce memory overhead for high-volume logging

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Messages not appearing | Level filtering too restrictive | Check `get_effective_level()` vs message level | Lower logger level or use higher level messages |
| Duplicate log messages | Handler added multiple times | Inspect `logger.handlers` list | Check for duplicate handler addition |
| Deadlock on level change | Lock ordering violation | Thread dump showing lock contention | Acquire locks in consistent order (parent before child) |
| Context not inheriting | Parent-child relationship broken | Verify `logger.parent` references | Fix logger hierarchy creation in factory |
| Memory leak in long-running app | Logger registry holding references | Monitor logger count over time | Use WeakValueDictionary for registry storage |


## Structured Output Design

> **Milestone(s):** This section primarily addresses Milestone 2 (Structured Output) by defining JSON formatting, custom formatters, timestamp handling, and developer-friendly pretty printing. It builds on the Logger Core from Milestone 1 and prepares the foundation for context propagation in Milestone 3.

Think of structured logging output like a restaurant's order system. Traditional string-based logs are like servers shouting orders across a noisy kitchen — "Medium rare steak, extra sauce!" The information is there, but it's hard to parse, search, or aggregate. Structured JSON output is like a modern point-of-sale system that sends precise, machine-readable orders to each station: `{"item": "steak", "doneness": "medium_rare", "modifications": ["extra_sauce"], "table": 7, "timestamp": "2024-01-15T18:30:00Z"}`. Every piece of information has its designated place, making it trivial for kitchen display systems, inventory tracking, and analytics to process the data automatically.

The structured output system transforms the rich `LogRecord` objects from our logger core into various serialized formats suitable for different consumption scenarios. While the logger hierarchy and level filtering ensure the right messages reach the formatting stage, the output system determines how those messages appear to developers, monitoring systems, and log aggregation platforms.

### JSON Formatter

The JSON formatter serves as the primary structured output format, converting `LogRecord` objects into single-line JSON strings that maintain consistent field ordering and handle complex serialization scenarios gracefully.

> **Decision: Single-Line JSON Output**
> - **Context**: Log aggregation systems and streaming processors expect one JSON object per line to enable efficient parsing and processing of large log files
> - **Options Considered**: Multi-line pretty JSON, single-line compact JSON, custom delimited format
> - **Decision**: Single-line JSON with consistent field ordering
> - **Rationale**: Single-line format enables stream processing, consistent ordering aids debugging, and standard JSON ensures compatibility with existing tooling
> - **Consequences**: Enables efficient log processing but reduces human readability in raw form (addressed by separate pretty-print formatter)

| JSON Field | Type | Source | Description |
|------------|------|--------|-------------|
| `timestamp` | string | `LogRecord.timestamp` | ISO 8601 formatted timestamp with timezone |
| `level` | string | `LogRecord.level` mapped via `LEVEL_NAMES` | Human-readable level name (DEBUG, INFO, WARN, ERROR, FATAL) |
| `message` | string | `LogRecord.message` | Primary log message content |
| `logger` | string | `LogRecord.logger_name` | Hierarchical logger name (e.g., "api.auth.login") |
| `context` | object | `LogRecord.context` | Key-value pairs with request context and metadata |

The JSON formatter implements several sophisticated serialization strategies to handle edge cases that commonly break logging systems in production:

**Circular Reference Protection**: The `safe_serialize` function maintains a recursion depth counter and visited object set to detect circular references before they cause stack overflows. When a circular reference is detected, the formatter replaces the problematic object with a placeholder string `"<circular_reference>"`, allowing logging to continue rather than crashing the application.

**Non-Serializable Object Handling**: The `SafeJSONEncoder` extends Python's standard JSON encoder to handle common non-serializable types encountered in production logging. DateTime objects convert to ISO 8601 strings, Decimal objects serialize as strings to preserve precision, and custom objects fall back to their string representation via `str()`.

**Size-Based Truncation**: The `estimate_serialized_size` function provides fast size estimation without full serialization, enabling the formatter to truncate oversized context objects before they consume excessive memory or exceed log destination limits. Objects exceeding configurable size thresholds are replaced with summary metadata indicating the original type and truncated size.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `format` | `record: LogRecord` | `str` | Convert LogRecord to single-line JSON string |
| `safe_serialize` | `data: Any, max_depth: int = 10` | `str` | Serialize with circular reference protection |
| `estimate_serialized_size` | `data: Any` | `int` | Fast size estimation without full serialization |
| `to_dict` | `record: LogRecord` | `Dict[str, Any]` | Convert LogRecord to dictionary for JSON serialization |

**Field Ordering Strategy**: The formatter uses Python's `collections.OrderedDict` or dictionary insertion ordering (Python 3.7+) to ensure consistent field appearance across all log records. This deterministic ordering significantly improves developer experience when scanning logs manually and ensures that text-based diff tools can meaningfully compare log files.

Consider a concrete example where an authentication service logs a failed login attempt:

```text
Input LogRecord:
  timestamp: "2024-01-15T18:30:45.123456Z"
  level: 40 (ERROR)
  message: "Authentication failed"
  logger_name: "api.auth.login"
  context: {"user_id": "user_123", "ip_address": "192.168.1.100", "attempt_count": 3}

Output JSON:
{"timestamp":"2024-01-15T18:30:45.123456Z","level":"ERROR","message":"Authentication failed","logger":"api.auth.login","context":{"user_id":"user_123","ip_address":"192.168.1.100","attempt_count":3}}
```

The single-line output enables log aggregation systems to process each line as a discrete event while maintaining all the structured context necessary for querying and analysis.

### Formatter Plugin System

The formatter plugin system provides extensibility for custom output formats while maintaining a consistent interface that integrates seamlessly with the handler dispatch mechanism.

Think of the formatter plugin system like a universal printer driver architecture. Just as applications can send documents to any printer through a standard interface (regardless of whether it's an inkjet, laser, or plotter), the logging system can format records for any destination through a standard formatter interface. The application code doesn't need to know whether logs are going to JSON files, syslog servers, or custom monitoring dashboards — it just hands the `LogRecord` to the appropriate formatter.

> **Decision: Registry-Based Plugin Architecture**
> - **Context**: Different output destinations require different formatting (JSON for aggregation, colored text for development, structured formats for monitoring systems)
> - **Options Considered**: Inheritance-based formatters, function-based formatters, registry-based plugins
> - **Decision**: Registry-based plugin system with standard formatter interface
> - **Rationale**: Registry enables runtime formatter selection, standard interface ensures consistent behavior, and plugins can be distributed as separate packages
> - **Consequences**: Enables formatter extensibility and runtime configuration but adds complexity compared to fixed formatters

| Formatter Component | Type | Purpose | Registration Method |
|---------------------|------|---------|-------------------|
| `BaseFormatter` | Abstract Class | Defines standard interface for all formatters | N/A (base class) |
| `JSONFormatter` | Concrete Class | Single-line JSON output for log aggregation | `register_formatter("json", JSONFormatter())` |
| `PrettyFormatter` | Concrete Class | Human-readable colored output for development | `register_formatter("pretty", PrettyFormatter())` |
| `SyslogFormatter` | Concrete Class | RFC 3164 compatible syslog format | `register_formatter("syslog", SyslogFormatter())` |
| `FormatterRegistry` | Registry Class | Manages formatter registration and lookup | `get_formatter(name)`, `list_formatters()` |

The plugin registration process follows a simple two-step pattern that maintains type safety while enabling runtime flexibility:

1. **Formatter Implementation**: Custom formatters inherit from `BaseFormatter` and implement the required `format(record: LogRecord) -> str` method along with optional configuration methods.

2. **Registry Registration**: Formatters register themselves with the global `FormatterRegistry` using a unique string name, enabling selection via configuration files or runtime API calls.

| BaseFormatter Method | Parameters | Returns | Description |
|---------------------|------------|---------|-------------|
| `format` | `record: LogRecord` | `str` | Abstract method that subclasses must implement |
| `configure` | `options: Dict[str, Any]` | `None` | Optional configuration method for formatter-specific settings |
| `supports_color` | None | `bool` | Indicates whether formatter supports color output |
| `get_name` | None | `str` | Returns the formatter's registration name |

**Thread-Safe Registry Operations**: The `FormatterRegistry` uses a read-write lock to ensure thread-safe registration and lookup operations. Formatter registration typically occurs during application startup, while lookups happen during log processing, making the read-heavy access pattern well-suited to RWLock optimization.

**Configuration Integration**: Formatters can accept configuration parameters through the `configure` method, enabling customization without requiring code changes. For example, the JSON formatter accepts parameters for timestamp format, field ordering preferences, and size limits, while the pretty formatter accepts color scheme and indentation preferences.

Here's how the plugin system enables runtime formatter selection:

```text
Configuration-Driven Selection:
  handlers:
    - type: file
      path: "/var/log/app.json"
      formatter: "json"
    - type: console  
      formatter: "pretty"
      
Runtime Registration:
  1. Application startup registers built-in formatters (json, pretty, syslog)
  2. Custom formatter packages register additional formatters via plugin discovery
  3. Configuration loading selects formatters by name for each handler
  4. Handler dispatch uses assigned formatter for each destination
```

### Timestamp Formatting

Timestamp formatting provides the critical temporal context that enables log correlation, debugging, and audit trails across distributed systems. The timestamp system must balance precision, readability, and compatibility with various log processing systems.

Think of timestamp formatting like international time zone coordination for airline schedules. Airlines need to display departure times in local time for passenger convenience ("Flight 101 departs at 3:30 PM"), but they also need UTC timestamps for air traffic control coordination and schedule optimization. Similarly, our logging system needs human-readable timestamps for developers scanning logs locally, but also needs precise, sortable timestamps for log aggregation systems that merge streams from multiple servers across different time zones.

> **Decision: Multiple Timestamp Format Support**
> - **Context**: Different consumers need different timestamp formats (humans prefer readable formats, systems prefer sortable formats, legacy systems expect specific formats)
> - **Options Considered**: Single ISO 8601 format, Unix epoch only, configurable format per logger
> - **Decision**: Multiple timestamp formats with per-formatter configuration
> - **Rationale**: ISO 8601 provides precision and sortability, Unix epoch enables efficient processing, custom formats support legacy integration
> - **Consequences**: Enables format flexibility but requires careful timezone handling and format validation

| Timestamp Format | Example Output | Use Case | Pros | Cons |
|------------------|----------------|----------|------|------|
| ISO 8601 with Microseconds | `2024-01-15T18:30:45.123456Z` | Log aggregation, debugging | Sortable, precise, timezone-aware | Longer string length |
| ISO 8601 Basic | `2024-01-15T18:30:45Z` | General structured logging | Standard, readable, sortable | Lower precision |
| Unix Epoch | `1705344645.123456` | High-performance processing | Compact, fast parsing, efficient storage | Not human-readable |
| Custom strftime | `2024-01-15 18:30:45 UTC` | Legacy system integration | Flexible, familiar format | Non-standard, parsing complexity |

**Timezone Handling Strategy**: All timestamps are captured in UTC to avoid daylight saving time transitions and timezone inconsistencies that plague distributed logging. The `LogRecord.create` factory method uses `datetime.utcnow()` to ensure consistent temporal ordering regardless of the server's local timezone configuration.

**Precision Requirements**: The default timestamp format includes microsecond precision to enable ordering of log events that occur within the same second. This precision level proves essential when debugging race conditions or analyzing high-frequency operations where millisecond timing matters.

**Clock Synchronization Considerations**: While the logging system cannot control server clock synchronization, it provides mechanisms to detect and warn about significant clock skew. The timestamp formatter can optionally include a monotonic clock offset that enables relative timing analysis even when system clocks drift.

| Timestamp Method | Parameters | Returns | Description |
|------------------|------------|---------|-------------|
| `format_timestamp` | `timestamp: str, format_type: str` | `str` | Convert ISO timestamp to specified format |
| `parse_timestamp` | `timestamp_str: str` | `datetime` | Parse timestamp string to datetime object |
| `get_current_timestamp` | `precision: str = "microsecond"` | `str` | Generate current UTC timestamp with specified precision |
| `validate_timestamp` | `timestamp_str: str` | `bool` | Validate timestamp string format |

**Performance Optimization**: Timestamp formatting uses caching strategies to avoid repeated strftime operations on identical timestamps. The formatter maintains a small LRU cache of recently formatted timestamps, which significantly improves performance when processing bursts of log messages within the same second.

Consider how different timestamp formats serve different operational needs:

```text
Development Scenario - Pretty Formatter:
  Raw LogRecord: timestamp="2024-01-15T18:30:45.123456Z"
  Formatted Output: "[18:30:45.123] ERROR api.auth: Authentication failed"
  
Production JSON - JSON Formatter:  
  Raw LogRecord: timestamp="2024-01-15T18:30:45.123456Z"
  Formatted Output: {"timestamp":"2024-01-15T18:30:45.123456Z","level":"ERROR",...}
  
Legacy Syslog Integration - Syslog Formatter:
  Raw LogRecord: timestamp="2024-01-15T18:30:45.123456Z"  
  Formatted Output: "Jan 15 18:30:45 hostname app[1234]: Authentication failed"
```

### Developer-Friendly Pretty Print

The pretty print formatter transforms structured log data into human-readable, visually appealing output optimized for developer console environments and interactive debugging sessions.

Think of pretty printing like the difference between reading raw HTML source code and viewing a rendered web page. While the JSON formatter produces machine-readable output equivalent to HTML source (precise, parseable, but dense), the pretty formatter acts like a web browser, transforming that structured data into a visually organized, color-coded display that humans can quickly scan and comprehend. Just as web browsers use typography, spacing, and color to highlight important information, the pretty formatter uses console colors, indentation, and visual hierarchy to make log data accessible to developers.

> **Decision: Console-Optimized Visual Formatting**
> - **Context**: Developers need quick visual scanning of log output during debugging and development, but JSON format is difficult to read in terminal environments
> - **Options Considered**: Colored JSON with indentation, table-based layout, custom visual format
> - **Decision**: Multi-line colored output with hierarchical indentation and visual separators
> - **Rationale**: Color coding enables quick level identification, indentation shows context hierarchy, and compact layout maximizes information density
> - **Consequences**: Dramatically improves developer experience but produces larger output unsuitable for production log aggregation

**Color Scheme Strategy**: The pretty formatter uses a carefully designed color palette that remains readable across different terminal backgrounds and accommodates common forms of color vision differences. Critical information uses high-contrast colors, while contextual information uses muted tones that don't compete for attention.

| Log Level | Color | Terminal Code | Rationale |
|-----------|-------|---------------|-----------|
| DEBUG | Cyan | `\033[36m` | Cool color indicates low-priority information |
| INFO | Green | `\033[32m` | Green suggests normal, healthy operation |
| WARN | Yellow | `\033[33m` | Yellow universally signals caution |
| ERROR | Red | `\033[31m` | Red immediately draws attention to problems |
| FATAL | Bright Red + Bold | `\033[1;91m` | Maximum visual impact for critical issues |

**Layout Design Principles**: The pretty formatter uses consistent visual hierarchy to help developers quickly locate relevant information. Timestamps appear in muted colors to provide context without distraction, logger names use medium emphasis to show the source, and context fields are indented to clearly separate them from the primary message.

```text
Example Pretty Print Output Layout:

[18:30:45.123] ERROR api.auth.login                          ← timestamp + level + logger
Authentication failed for user                               ← primary message  
├── user_id: "user_123"                                     ← context fields with
├── ip_address: "192.168.1.100"                             ← tree-style indentation
├── attempt_count: 3                                        ← for visual hierarchy
└── correlation_id: "req_abc123def456"                      
                                                            ← blank line separator
```

**Context Field Rendering**: The formatter intelligently handles different context field types with appropriate visual treatment. String values appear quoted, numbers display without decoration, nested objects use deeper indentation levels, and large values trigger truncation with expansion hints.

| Context Field Type | Rendering Strategy | Example |
|-------------------|-------------------|---------|
| String | Quoted with escape handling | `user_name: "john.doe"` |
| Number | Direct display | `response_time_ms: 245` |
| Boolean | Lowercase display | `authenticated: true` |
| List | Bracketed with item count | `permissions: [3 items]` |
| Dict | Nested indentation | `metadata:` followed by indented key-value pairs |
| Large Object | Truncated with hint | `request_body: <dict with 15 fields>` |

**Terminal Capability Detection**: The formatter detects terminal capabilities to gracefully degrade when color support is unavailable or when output is redirected to files. This detection prevents ANSI escape codes from polluting log files while maintaining rich formatting in interactive terminals.

| PrettyFormatter Method | Parameters | Returns | Description |
|------------------------|------------|---------|-------------|
| `format` | `record: LogRecord` | `str` | Convert LogRecord to color-formatted multi-line string |
| `format_context` | `context: Dict[str, Any], indent: int` | `str` | Render context fields with tree-style indentation |
| `detect_terminal_caps` | None | `Dict[str, bool]` | Detect color support and terminal width |
| `colorize` | `text: str, color: str` | `str` | Apply color codes with terminal capability fallback |

**Performance Considerations**: While pretty formatting is more expensive than simple JSON serialization, it's designed specifically for development environments where the performance cost is acceptable in exchange for improved developer productivity. The formatter caches color capability detection and reuses formatting objects to minimize overhead.

**Width-Aware Formatting**: The formatter detects terminal width and adjusts layout accordingly. On narrow terminals, it abbreviates logger names and wraps long messages intelligently. On wide terminals, it can display additional context inline rather than using vertical indentation.

Consider how the same log record appears in different formatting contexts:

```text
JSON Formatter (Production):
{"timestamp":"2024-01-15T18:30:45.123456Z","level":"ERROR","message":"Authentication failed","logger":"api.auth.login","context":{"user_id":"user_123","ip_address":"192.168.1.100","attempt_count":3}}

Pretty Formatter (Development):
[18:30:45.123] ERROR api.auth.login
Authentication failed
├── user_id: "user_123"  
├── ip_address: "192.168.1.100"
└── attempt_count: 3
```

The pretty formatter transforms dense, machine-readable data into an immediately comprehensible visual representation that enables developers to quickly identify patterns, spot anomalies, and understand the flow of execution through their applications.

![Log Processing Data Flow](./diagrams/log-processing-flow.svg)

![Formatter Plugin Architecture](./diagrams/formatter-plugin-system.svg)

### Common Pitfalls

⚠️ **Pitfall: Non-Serializable Context Values**
Developers frequently add complex objects to logging context that cannot be serialized to JSON, such as database connections, file handles, or custom objects without `__dict__` methods. This causes the entire logging operation to fail with serialization exceptions, potentially masking the original issue being logged. The fix involves implementing the `SafeJSONEncoder` with fallback serialization that converts problematic objects to their string representation via `str()` or `repr()`, allowing logging to continue even with problematic context data.

⚠️ **Pitfall: Blocking I/O in Formatter**
Some custom formatters attempt to perform network requests or file I/O operations during the formatting phase, such as looking up additional metadata or validating data against external services. This introduces blocking operations directly in the logging hot path, causing application threads to stall while waiting for formatter completion. The solution involves moving all I/O operations to the handler level and keeping formatters as pure transformation functions that operate only on the provided `LogRecord` data.

⚠️ **Pitfall: Memory Exhaustion from Large Context Objects**
Applications sometimes log entire request payloads, response objects, or large data structures as context fields. When these objects contain megabytes of data, they can quickly exhaust available memory, especially under high logging volume. The `estimate_serialized_size` function provides protection by detecting oversized objects before serialization and replacing them with summary metadata, but this protection must be enabled and configured with appropriate thresholds.

⚠️ **Pitfall: Timezone Confusion in Timestamps**
Mixing local time and UTC timestamps across different servers creates impossible-to-debug timing issues where log events appear to occur in the wrong order or at impossible times. Always use UTC for log timestamps and convert to local time only in presentation layers. The `get_current_timestamp` function ensures UTC consistency, but developers must avoid using `datetime.now()` or other local time functions when creating custom timestamp formats.

⚠️ **Pitfall: Circular References in Context**
When logging objects that contain references to parent objects or self-references, the JSON serialization process can enter infinite loops, eventually causing stack overflow errors. This commonly occurs when logging request objects that contain references to the application context or when debugging recursive data structures. The `safe_serialize` function with depth limiting and visited-object tracking prevents this issue, but it must be used consistently across all formatters.

### Implementation Guidance

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| JSON Serialization | `json` standard library | `orjson` for high-performance serialization |
| Color Terminal Output | ANSI escape codes | `colorama` for cross-platform color support |
| Timestamp Parsing | `datetime.fromisoformat()` | `dateutil.parser` for flexible format support |
| Size Estimation | String length approximation | `sys.getsizeof()` with recursive object traversal |

**B. Recommended File Structure:**
```
structured_logging/
├── formatters/
│   ├── __init__.py              ← Formatter registry and base classes
│   ├── json_formatter.py        ← JSON formatter implementation  
│   ├── pretty_formatter.py      ← Developer console formatter
│   └── base_formatter.py        ← Abstract base formatter
├── utils/
│   ├── serialization.py         ← SafeJSONEncoder and safe_serialize
│   └── timestamps.py            ← Timestamp formatting utilities
└── tests/
    ├── test_json_formatter.py
    ├── test_pretty_formatter.py
    └── test_serialization.py
```

**C. Infrastructure Starter Code:**

**SafeJSONEncoder (Complete Implementation):**
```python
import json
import decimal
import datetime
from typing import Any, Set, Dict

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles non-serializable types gracefully."""
    
    def __init__(self, max_depth=10, max_size=10000):
        super().__init__()
        self.max_depth = max_depth
        self.max_size = max_size
        self._visited: Set[int] = set()
        self._current_depth = 0
    
    def encode(self, obj):
        self._visited.clear()
        self._current_depth = 0
        return super().encode(obj)
    
    def default(self, obj):
        # Handle common non-serializable types
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return f"<{type(obj).__name__} object>"
        else:
            return str(obj)

def safe_serialize(data: Any, max_depth: int = 10) -> str:
    """Serialize data to JSON with circular reference protection."""
    try:
        encoder = SafeJSONEncoder(max_depth=max_depth)
        return encoder.encode(data)
    except (TypeError, ValueError, RecursionError) as e:
        return json.dumps({"serialization_error": str(e), "type": str(type(data))})

def estimate_serialized_size(data: Any) -> int:
    """Estimate JSON size without full serialization."""
    if data is None:
        return 4  # "null"
    elif isinstance(data, bool):
        return 4 if data else 5  # "true" or "false"
    elif isinstance(data, int):
        return len(str(data))
    elif isinstance(data, float):
        return len(str(data))
    elif isinstance(data, str):
        return len(data) + 2  # Add quotes
    elif isinstance(data, (list, tuple)):
        return sum(estimate_serialized_size(item) for item in data) + len(data) + 1
    elif isinstance(data, dict):
        size = 2  # {}
        for key, value in data.items():
            size += len(str(key)) + 2  # quoted key
            size += estimate_serialized_size(value)
            size += 2  # colon and comma
        return size
    else:
        return len(str(data)) + 2  # quoted string representation
```

**Timestamp Utilities (Complete Implementation):**
```python
import datetime
from typing import Optional

def get_current_timestamp(precision: str = "microsecond") -> str:
    """Generate current UTC timestamp with specified precision."""
    now = datetime.datetime.utcnow()
    if precision == "second":
        return now.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif precision == "microsecond":
        return now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        raise ValueError(f"Unsupported precision: {precision}")

def format_timestamp(timestamp: str, format_type: str) -> str:
    """Convert ISO timestamp to specified format."""
    dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    if format_type == "iso":
        return timestamp
    elif format_type == "epoch":
        return str(dt.timestamp())
    elif format_type == "readable":
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        # Assume custom strftime format
        return dt.strftime(format_type)

class TimestampCache:
    """LRU cache for formatted timestamps to improve performance."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order = []
    
    def get_formatted(self, timestamp: str, format_type: str) -> str:
        cache_key = f"{timestamp}:{format_type}"
        
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]
        
        # Cache miss - format and store
        formatted = format_timestamp(timestamp, format_type)
        self._cache[cache_key] = formatted
        self._access_order.append(cache_key)
        
        # Evict oldest if over limit
        if len(self._cache) > self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        return formatted
```

**D. Core Logic Skeletons:**

**JSONFormatter (Core Implementation Skeleton):**
```python
from typing import Dict, Any
from .base_formatter import BaseFormatter
from ..models import LogRecord
from ..utils.serialization import safe_serialize, estimate_serialized_size

class JSONFormatter(BaseFormatter):
    """Formats log records as single-line JSON objects."""
    
    def __init__(self, max_context_size: int = 10000, timestamp_format: str = "iso"):
        self.max_context_size = max_context_size
        self.timestamp_format = timestamp_format
        
    def format(self, record: LogRecord) -> str:
        """Convert LogRecord to single-line JSON string."""
        # TODO 1: Convert LogRecord to dictionary using to_dict method
        # TODO 2: Check context size using estimate_serialized_size
        # TODO 3: If context too large, replace with size summary
        # TODO 4: Format timestamp according to configured format
        # TODO 5: Serialize dictionary to JSON using safe_serialize
        # TODO 6: Return single-line JSON string
        pass
    
    def to_dict(self, record: LogRecord) -> Dict[str, Any]:
        """Convert LogRecord to dictionary for JSON serialization."""  
        # TODO 1: Create base dictionary with timestamp, level, message, logger
        # TODO 2: Map numeric level to string name using LEVEL_NAMES
        # TODO 3: Add context fields if present and not empty
        # TODO 4: Ensure consistent field ordering (timestamp, level, message, logger, context)
        # Hint: Use collections.OrderedDict or rely on Python 3.7+ dict ordering
        pass
```

**PrettyFormatter (Core Implementation Skeleton):**
```python
import os
from typing import Dict, Any, List
from .base_formatter import BaseFormatter
from ..models import LogRecord

class PrettyFormatter(BaseFormatter):
    """Human-readable console formatter with colors and indentation."""
    
    # Color constants
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARN': '\033[33m',     # Yellow
        'ERROR': '\033[31m',    # Red
        'FATAL': '\033[1;91m',  # Bright Red + Bold
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = None):
        self.use_colors = use_colors if use_colors is not None else self._detect_color_support()
        
    def format(self, record: LogRecord) -> str:
        """Convert LogRecord to colored multi-line string."""
        # TODO 1: Format timestamp as short time (HH:MM:SS.mmm)
        # TODO 2: Apply color to log level based on severity
        # TODO 3: Create main line with timestamp, level, logger, and message
        # TODO 4: Format context fields with tree-style indentation
        # TODO 5: Combine main line and context lines with proper spacing
        # TODO 6: Add blank line separator for visual grouping
        # Hint: Use colorize method for color application with fallback
        pass
        
    def format_context(self, context: Dict[str, Any], indent_level: int = 1) -> List[str]:
        """Format context fields with tree-style indentation."""
        # TODO 1: If context is empty, return empty list
        # TODO 2: Create list to collect formatted lines  
        # TODO 3: Iterate through context items with enumerate for last-item detection
        # TODO 4: Use ├── for non-final items and └── for final item
        # TODO 5: Handle nested dictionaries with recursive calls
        # TODO 6: Truncate large values with summary hint
        # Hint: Use "├──" and "└──" Unicode characters for tree structure
        pass
    
    def colorize(self, text: str, color: str) -> str:
        """Apply color with terminal capability detection."""
        # TODO 1: Check if colors are enabled and supported
        # TODO 2: If not supported, return text without modification
        # TODO 3: If supported, wrap text with color code and reset
        # Hint: Always include RESET code after colored text
        pass
    
    def _detect_color_support(self) -> bool:
        """Detect if terminal supports color output.""" 
        # TODO 1: Check if stdout is a TTY (not redirected to file)
        # TODO 2: Check TERM environment variable for color capability
        # TODO 3: Return True only if both conditions are met
        # Hint: Use os.isatty() and os.environ.get()
        pass
```

**FormatterRegistry (Core Implementation Skeleton):**
```python
import threading
from typing import Dict, Optional, List
from .base_formatter import BaseFormatter

class FormatterRegistry:
    """Thread-safe registry for formatter plugins."""
    
    def __init__(self):
        self._formatters: Dict[str, BaseFormatter] = {}
        self._lock = threading.RLock()
        
    def register_formatter(self, name: str, formatter: BaseFormatter) -> None:
        """Register a formatter with given name."""
        # TODO 1: Acquire write lock
        # TODO 2: Validate formatter inherits from BaseFormatter
        # TODO 3: Store formatter in registry with name as key
        # TODO 4: Release lock
        # Hint: Use isinstance() to check formatter type
        pass
        
    def get_formatter(self, name: str) -> Optional[BaseFormatter]:
        """Retrieve formatter by name."""
        # TODO 1: Acquire read lock  
        # TODO 2: Look up formatter in registry
        # TODO 3: Return formatter or None if not found
        # TODO 4: Release lock
        pass
        
    def list_formatters(self) -> List[str]:
        """List all registered formatter names."""
        # TODO 1: Acquire read lock
        # TODO 2: Get list of registered names
        # TODO 3: Return sorted list of names
        # TODO 4: Release lock
        pass

# Global registry instance
_formatter_registry = FormatterRegistry()

def register_formatter(name: str, formatter: BaseFormatter) -> None:
    """Register formatter in global registry."""
    _formatter_registry.register_formatter(name, formatter)

def get_formatter(name: str) -> Optional[BaseFormatter]:
    """Get formatter from global registry."""
    return _formatter_registry.get_formatter(name)
```

**E. Language-Specific Hints:**
- Use `json.dumps(separators=(',', ':'))` for compact single-line JSON output
- `collections.OrderedDict` ensures consistent field ordering in Python versions before 3.7
- `sys.getsizeof()` provides object size estimation for truncation decisions
- `threading.RLock()` allows the same thread to acquire the lock multiple times
- `os.isatty(sys.stdout.fileno())` detects if output is going to a terminal vs file
- Unicode characters like `├──` and `└──` create clean tree-style formatting

**F. Milestone Checkpoint:**
After completing this milestone, test your structured output system:

**Test Commands:**
```bash
python -m pytest tests/test_formatters.py -v
python test_formatting_demo.py  # Custom demo script
```

**Expected JSON Output:**
```json
{"timestamp":"2024-01-15T18:30:45.123456Z","level":"ERROR","message":"Test message","logger":"test.formatter","context":{"key":"value"}}
```

**Expected Pretty Output:**
```
[18:30:45.123] ERROR test.formatter
Test message
└── key: "value"

```

**Verification Steps:**
1. JSON formatter produces valid single-line JSON parseable by `json.loads()`
2. Pretty formatter displays colors in terminal but plain text when redirected to file
3. Context fields appear consistently across different formatters
4. Large context objects are truncated appropriately
5. Non-serializable objects don't crash the formatter

**Common Issues to Check:**
- JSON output contains no newlines or pretty-printing
- Colors only appear in interactive terminals, not in log files
- Timestamp format matches configuration settings
- Context truncation prevents memory exhaustion
- Thread safety works under concurrent formatting load


## Context and Correlation Design

> **Milestone(s):** This section primarily addresses Milestone 3 (Context & Correlation) by defining request tracing, correlation ID systems, context propagation mechanisms, and async boundary preservation. Some elements support all milestones through the context enrichment capabilities of LogRecord objects.

Think of correlation IDs and context propagation like a **relay race baton**. In a relay race, runners must pass a physical baton from one runner to the next to maintain continuity and prove the race was completed legitimately. Similarly, in a distributed system handling requests, we need to pass a "context baton" containing correlation IDs and metadata from one function to the next, one service to the next, and even across async task boundaries. Without this baton-passing mechanism, we lose the ability to trace a request's journey through our system - it's like having runners complete their legs independently without proving they're part of the same race.

The fundamental challenge in context and correlation design lies in maintaining this continuity across three distinct boundaries: **function call boundaries** (nested calls within a thread), **thread boundaries** (when work moves between threads), and **async boundaries** (when coroutines suspend and resume). Each boundary requires a different propagation strategy, yet the developer experience should remain seamless - they should be able to log with full context regardless of these underlying complexity layers.

### Correlation ID System

A **correlation ID** serves as the unique fingerprint for a request's journey through your system. Think of it like a **shipping tracking number** - just as you can trace a package's movement from warehouse to warehouse using its tracking number, you can trace a request's flow through services, functions, and async operations using its correlation ID. This identifier links all log entries related to the same logical operation, enabling you to reconstruct the complete story of what happened during request processing.

The correlation ID system operates on four core principles: **uniqueness** (each request gets a globally unique identifier), **persistence** (the ID travels with the request context), **automatic injection** (developers don't manually add IDs to every log call), and **boundary crossing** (the ID survives function calls, thread switches, and async operations). These principles ensure that correlation works transparently without requiring constant developer intervention.

> **Decision: Correlation ID Generation Strategy**
> - **Context**: Need globally unique identifiers that are human-readable, sortable, and carry minimal overhead
> - **Options Considered**: UUID4 (random), UUID1 (timestamp-based), custom base62 format, request sequence numbers
> - **Decision**: UUID4 with optional prefix for service identification
> - **Rationale**: UUID4 provides strong uniqueness guarantees without coordination, has excellent library support across languages, and avoids potential timing attacks from timestamp-based IDs. Optional service prefixes (e.g., "api-server-550e8400") improve readability while maintaining uniqueness
> - **Consequences**: 36-character overhead per log entry, but enables reliable request tracing across service boundaries without central coordination

| Component | Responsibility | Key Behavior |
|-----------|---------------|--------------|
| `CorrelationIDGenerator` | Generate unique request identifiers | Creates UUID4 strings with optional service prefix, thread-safe generation |
| `RequestIDInjector` | Automatically add correlation IDs to new requests | Detects missing correlation context, generates new IDs, preserves existing IDs |
| `ContextualLogger` | Enrich log records with correlation data | Merges correlation ID with log message context, handles missing ID scenarios |
| `IDPropagator` | Carry correlation IDs across boundaries | Maintains ID in thread-local and async-local storage, handles inheritance |

The correlation ID lifecycle begins when a request enters your system. At the **entry point** (HTTP request handler, message queue consumer, scheduled job), the system either extracts an existing correlation ID from headers/metadata or generates a new one. This ID immediately becomes part of the request context and flows through all subsequent operations. When the request spawns child operations (database calls, service calls, async tasks), the correlation ID propagates to those contexts automatically.

**Correlation ID propagation** follows a specific priority order: First, check if the current context already contains a correlation ID and preserve it. Second, look for an incoming correlation ID from request headers, parent contexts, or message metadata. Third, generate a new correlation ID only if none exists. This priority ensures that correlation chains remain intact across service boundaries while preventing orphaned operations from lacking traceability.

| Propagation Scenario | Source | Action Taken |
|---------------------|---------|--------------|
| HTTP Request with `X-Correlation-ID` header | Request headers | Extract and use existing ID |
| HTTP Request without correlation header | Request context | Generate new UUID4 and set response header |
| Nested function call within request | Parent context | Inherit ID from thread-local or async-local storage |
| Background async task spawned from request | Parent task context | Copy ID to new task's context |
| Database query from request handler | Current context | Include ID in query metadata/comments |
| Inter-service call | Current context | Add ID to outgoing request headers |

⚠️ **Pitfall: Correlation ID Collision**
A common mistake is generating correlation IDs using timestamp-based schemes or sequential counters that can collide across service instances. This breaks request tracing because multiple unrelated requests share the same ID. Use UUID4 generation with proper entropy sources, and include service identifiers in the prefix if you need human-readable correlation IDs.

The correlation ID must be **automatically injected** into every `LogRecord` without requiring developers to manually include it in each logging call. This happens through the context enrichment process - when a `LogRecord` is created, the system queries the current correlation context and merges the ID into the record's context fields. The injection process should be transparent and fail gracefully if no correlation ID exists (logging the message without correlation rather than failing).

```table
| Field Name | Type | Description |
|------------|------|-------------|
| `correlation_id` | `str` | UUID4 identifier for request tracing |
| `request_id` | `str` | Alias for correlation_id for backward compatibility |
| `parent_id` | `str \| None` | ID of parent operation for nested request tracing |
| `operation_id` | `str \| None` | Unique identifier for sub-operations within a request |
| `session_id` | `str \| None` | User session identifier for user journey tracing |
```

### Context Propagation

**Context propagation** is the mechanism that carries key-value pairs through nested function calls without requiring explicit parameter passing. Think of it like **ambient lighting** in a room - just as ambient light illuminates everything in the space without needing to shine a spotlight on each object individually, context propagation makes contextual information (user ID, request metadata, feature flags) available to all code within a request's execution scope.

Context propagation solves the **parameter drilling problem** where contextual data must be passed as parameters through many layers of function calls, even when intermediate functions don't use the data. Without propagation, a user ID determined at the HTTP handler level would need explicit passing through business logic, data access, and utility functions just to reach a deep logging call. Context propagation makes this information ambient - available anywhere within the request scope without explicit passing.

> **Decision: Context Storage Mechanism**
> - **Context**: Need to store context data that's accessible from any function within a request without parameter passing
> - **Options Considered**: Thread-local storage only, async-local storage only, hybrid approach with both mechanisms
> - **Decision**: Hybrid approach using both thread-local and async-local storage with automatic synchronization
> - **Rationale**: Thread-local handles synchronous code paths efficiently, async-local preserves context across await boundaries, hybrid approach provides complete coverage without forcing async/await everywhere
> - **Consequences**: Slight memory overhead from dual storage, but enables seamless context access in both sync and async codepaths

The core of context propagation is the `LoggingContext` manager, which maintains a **dual storage strategy**. Thread-local storage handles synchronous execution paths where code runs on a single thread without interruption. Async-local storage (using asyncio context variables or equivalent) handles asynchronous execution where coroutines can suspend and resume on different threads. The context manager synchronizes between these storage mechanisms to provide a unified view.

| Context Operation | Thread-Local Behavior | Async-Local Behavior | Synchronization |
|------------------|----------------------|---------------------|-----------------|
| `get_current()` | Read from threading.local | Read from contextvars.ContextVar | Return union of both sources |
| `set_current(context)` | Write to threading.local | Write to contextvars.ContextVar | Update both storage locations |
| `add_fields(**fields)` | Merge with existing thread context | Merge with existing async context | Merge results and update both |
| `clear()` | Clear threading.local data | Clear contextvars context | Reset both storage mechanisms |

**Context inheritance** follows a layered approach where child contexts inherit all fields from their parent and can add additional fields without modifying the parent. This creates an **inheritance chain** similar to object-oriented inheritance - changes in child contexts are isolated from parents, but children automatically receive parent updates. The inheritance mechanism enables request-level context (user ID, correlation ID) to coexist with operation-level context (database transaction ID, cache keys) without conflicts.

```table
| Context Layer | Scope | Typical Fields | Inheritance Behavior |
|---------------|--------|----------------|---------------------|
| Request Level | Entire HTTP request | `user_id`, `correlation_id`, `request_method` | Inherited by all operations within request |
| Operation Level | Business logic operation | `operation_name`, `feature_flags`, `tenant_id` | Inherits request context, adds operation-specific fields |
| Function Level | Individual function call | `function_name`, `parameters`, `execution_id` | Inherits operation context, adds function-specific fields |
| Error Level | Error handling block | `error_type`, `error_message`, `recovery_action` | Inherits function context, adds error-specific fields |
```

The **context propagation algorithm** follows these steps for each nested function call:

1. **Context Capture**: When entering a new execution context (function call, async task creation), capture the current context state from both thread-local and async-local storage
2. **Context Inheritance**: Create a new context object that inherits all fields from the captured parent context
3. **Context Activation**: Set the new context as current in both storage mechanisms, making it available to nested calls
4. **Context Isolation**: Ensure changes to the new context don't affect the parent context that will be restored later
5. **Context Restoration**: When exiting the execution context, restore the previous context state to both storage mechanisms
6. **Context Cleanup**: Remove any temporary context data that shouldn't persist beyond the current execution scope

**Context field management** requires careful handling of data types and serialization. Context fields must be serializable to JSON for inclusion in log records, but the context system should accept arbitrary Python objects and handle serialization gracefully. Non-serializable objects (file handles, database connections, lambda functions) should be converted to string representations rather than causing serialization failures.

| Field Type | Storage Approach | Serialization Behavior | Example |
|------------|-----------------|----------------------|---------|
| Primitive types | Store directly | Serialize as-is | `"user_id": "12345"`, `"timeout": 30` |
| Collections | Store directly if serializable | Recursive serialization | `"tags": ["api", "production"]` |
| Custom objects | Store string representation | Convert to string before serialization | `"database": "<Connection:localhost:5432>"` |
| Circular references | Detect and replace with placeholder | Replace with reference placeholder | `"parent": "<CircularReference>"` |
| Large objects | Truncate or summarize | Limit size to prevent log bloat | `"request_body": "<Data:1024 bytes>"` |

⚠️ **Pitfall: Context Memory Leaks**
A critical mistake is storing large objects or circular references in context without cleanup, leading to memory leaks. Context should contain only essential metadata (strings, numbers, small collections). Large objects like request bodies, database result sets, or file contents should be summarized or referenced by ID rather than stored directly in context.

### Async Context Bridge

The **async context bridge** solves the most complex problem in context propagation: preserving logging context when execution crosses async/await boundaries. Think of async operations like **airline passengers changing planes** during a layover. Just as passengers must carry their luggage and boarding passes from one plane to the next to maintain their journey's continuity, async operations must carry their logging context from one coroutine to the next to maintain request tracing.

Async context preservation is challenging because **coroutines can suspend and resume** on different threads, traditional thread-local storage becomes unreliable, and **multiple coroutines** can run concurrently within the same request, each needing isolated context. The bridge mechanism ensures that when a coroutine suspends (awaits another operation), its logging context is preserved, and when it resumes, the same context is restored regardless of which thread it resumes on.

> **Decision: Async Context Storage Strategy**
> - **Context**: Async operations suspend and resume across thread boundaries, making thread-local storage insufficient
> - **Options Considered**: Manual context passing, asyncio.current_task() storage, contextvars.ContextVar, custom async-local implementation
> - **Decision**: contextvars.ContextVar with automatic synchronization to thread-local storage
> - **Rationale**: contextvars provides language-level async context support, automatically handles suspend/resume cycles, and integrates well with asyncio task management. Synchronization with thread-local ensures compatibility with synchronous code
> - **Consequences**: Requires Python 3.7+ for contextvars, slight overhead from dual storage, but provides robust async context preservation

The async context bridge operates through **context variables** that maintain their values across await boundaries. When an async function begins execution, it inherits the context variables from its caller. When it suspends on an await, the context variables are automatically preserved. When it resumes, the same context variables are restored, regardless of the underlying thread changes.

![Context Propagation Flow](./diagrams/context-propagation-flow.svg)

```table
| Context State | Event | Next State | Actions Taken |
|---------------|--------|------------|---------------|
| `Active` | Function entry | `Active` | Inherit parent context, create child context |
| `Active` | Await operation | `Suspended` | Store context in contextvars, suspend coroutine |
| `Suspended` | Operation completion | `Active` | Restore context from contextvars, resume coroutine |
| `Active` | Exception raised | `Error` | Preserve context for error logging, propagate exception |
| `Error` | Exception handled | `Active` | Restore pre-exception context, continue execution |
| `Active` | Function exit | `Cleaned` | Restore parent context, cleanup temporary fields |
```

The **context bridge implementation** requires careful coordination between async and sync storage mechanisms. The bridge maintains a **bidirectional synchronization** where changes to async context (via contextvars) are reflected in thread-local storage and vice versa. This ensures that code using either storage mechanism sees consistent context data.

**Async task spawning** is a critical point where context must be explicitly propagated. When creating new async tasks using `asyncio.create_task()`, `asyncio.gather()`, or similar mechanisms, the current context should be captured and passed to the new task. This prevents child tasks from losing their parent's logging context and enables proper correlation across parallel operations.

```table
| Async Operation | Context Propagation Strategy | Implementation Notes |
|----------------|------------------------------|----------------------|
| `await async_func()` | Automatic via contextvars | No manual intervention required |
| `asyncio.create_task()` | Copy current context to new task | Use contextvars.copy_context() for explicit copying |
| `asyncio.gather()` | Inherit context in each gathered operation | Each operation starts with caller's context |
| `asyncio.wait_for()` | Preserve context across timeout | Context maintained even if operation times out |
| Background tasks | Explicit context injection | Pass context as task parameter or use task-local storage |
| Task cancellation | Context cleanup on cancellation | Restore parent context when task is cancelled |
```

**Context isolation** between concurrent async operations is essential to prevent context pollution. When multiple async operations run concurrently (through `asyncio.gather()` or similar), each should maintain its own context copy. Changes to one operation's context shouldn't affect concurrent operations, even if they share the same parent context.

⚠️ **Pitfall: Context Not Surviving Async Boundaries**
A common mistake is assuming thread-local storage will work in async code. When an async function suspends and resumes, it might continue on a different thread, making thread-local data inaccessible. Always use contextvars.ContextVar for async operations and ensure proper synchronization with thread-local storage for mixed sync/async codebases.

The **context lifecycle management** in async environments follows a specific pattern:

1. **Context Inheritance**: New async operations inherit context from their creator through contextvars propagation
2. **Context Isolation**: Each async operation gets its own context copy to prevent interference between concurrent operations
3. **Context Synchronization**: Changes to async context are reflected in thread-local storage for compatibility with sync code
4. **Context Cleanup**: When async operations complete, temporary context fields are cleaned up to prevent memory accumulation
5. **Context Restoration**: Parent context is restored after child async operations complete
6. **Error Context**: Exception handling preserves context for error logging before propagating failures

![Async Context State Machine](./diagrams/async-context-states.svg)

### Request Context Middleware

**Request context middleware** serves as the **entry point orchestrator** for the entire context and correlation system. Think of it as the **hotel concierge** who greets guests at the entrance, gathers their information, provides them with key cards and room service menus, and ensures they have everything needed for their stay. Similarly, request middleware greets incoming HTTP requests, extracts or generates correlation IDs, gathers request metadata, and establishes the logging context that will be available throughout request processing.

The middleware operates at the **HTTP framework level**, intercepting requests before they reach business logic and responses before they're sent to clients. This positioning ensures that every request gets consistent context treatment regardless of which endpoint handles it, and that correlation data is properly extracted from incoming headers and injected into outgoing headers for distributed tracing.

> **Decision: Middleware Integration Strategy**
> - **Context**: Need to integrate context setup with various HTTP frameworks (Flask, Django, FastAPI) without framework-specific implementations
> - **Options Considered**: Framework-specific middleware for each platform, WSGI/ASGI middleware for universal compatibility, decorator-based approach
> - **Decision**: WSGI/ASGI middleware with framework-specific convenience wrappers
> - **Rationale**: WSGI/ASGI middleware works across all Python web frameworks, provides consistent behavior, and allows framework-specific optimizations through optional wrappers. This approach maximizes compatibility while enabling framework-specific features
> - **Consequences**: Requires understanding of WSGI/ASGI interfaces, but provides universal compatibility and consistent context behavior across different web frameworks

The **request context extraction** process follows a standardized approach for gathering contextual information from HTTP requests. The middleware examines request headers for existing correlation IDs, extracts user identification from authentication headers, captures request metadata (method, path, user agent), and determines request characteristics (content type, request size, client IP).

```table
| Context Source | Header/Field Name | Extraction Logic | Fallback Behavior |
|----------------|------------------|------------------|------------------|
| Correlation ID | `X-Correlation-ID`, `X-Request-ID` | Use first found header value | Generate new UUID4 |
| User Identity | `Authorization`, `X-User-ID` | Extract from JWT/token or direct header | Anonymous user context |
| Request Method | HTTP method | Direct from request object | Required field, no fallback |
| Request Path | URL path | Normalized path without query params | Required field, no fallback |
| Client IP | `X-Forwarded-For`, `X-Real-IP` | Use leftmost IP from forwarded headers | Fall back to direct connection IP |
| User Agent | `User-Agent` | Direct header value | 'Unknown' if header missing |
```

**Request metadata enrichment** automatically captures standard request information that's valuable for logging and debugging. This metadata becomes part of the base request context that all subsequent operations inherit. The enrichment process captures both standard HTTP metadata and application-specific context that can be determined at request time.

```table
| Metadata Field | Data Type | Description | Example Value |
|----------------|-----------|-------------|---------------|
| `request_id` | `str` | Correlation ID for this request | `"550e8400-e29b-41d4-a716-446655440000"` |
| `request_method` | `str` | HTTP method for the request | `"POST"` |
| `request_path` | `str` | URL path without query parameters | `"/api/users/123/orders"` |
| `request_size` | `int` | Content length in bytes | `1024` |
| `user_id` | `str \| None` | Authenticated user identifier | `"user_789"` |
| `client_ip` | `str` | Client IP address (respecting proxies) | `"192.168.1.100"` |
| `user_agent` | `str` | Client user agent string | `"Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."` |
| `content_type` | `str \| None` | Request content type | `"application/json"` |
| `request_start_time` | `str` | ISO timestamp when request processing began | `"2024-01-15T10:30:45.123456Z"` |
```

The **context activation process** sets up the complete logging environment for request processing. This involves creating the initial request context with extracted metadata, activating the context in both thread-local and async-local storage, configuring any request-specific logging behavior (debug mode for specific users, increased log levels for error investigation), and ensuring context propagation is ready for nested operations.

**Context activation algorithm**:

1. **Extract incoming context**: Parse correlation ID from headers, decode user information from authentication tokens, capture standard request metadata
2. **Generate missing context**: Create new correlation ID if none provided, set anonymous user context if unauthenticated, establish request timing information
3. **Create request context**: Combine extracted and generated context into a structured context object
4. **Activate storage mechanisms**: Set context in both thread-local storage (for sync code) and async-local storage (for async operations)
5. **Configure request-specific logging**: Apply any user-specific or path-specific logging configuration (debug mode, sampling rates)
6. **Establish cleanup hooks**: Register context cleanup functions to run after request completion

**Response header injection** ensures that correlation IDs and tracing information flow to downstream systems and clients. The middleware captures the final correlation ID from the request context and injects it into response headers. This enables client-side correlation and helps with distributed tracing when the client makes subsequent requests to other services.

```table
| Response Header | Content | Purpose |
|----------------|---------|---------|
| `X-Correlation-ID` | Current request correlation ID | Enable client-side request correlation |
| `X-Request-Duration` | Request processing time in milliseconds | Provide timing information to clients |
| `X-Response-Size` | Response content length | Complete request/response size tracking |
```

The **middleware error handling** ensures that context and correlation continue working even when request processing fails. When exceptions occur during request processing, the middleware captures error context (exception type, error message, stack trace ID) and includes it in the final log entries. The middleware also ensures that context cleanup happens regardless of whether the request succeeds or fails.

⚠️ **Pitfall: Context Leakage Between Requests**
A critical mistake is failing to clean up context between requests in applications that reuse threads (most web frameworks). Context from one request can leak into subsequent requests if cleanup isn't properly implemented. Always ensure context is cleared at the end of request processing and that new requests start with fresh context.

**Framework integration patterns** provide specific guidance for common Python web frameworks:

```table
| Framework | Integration Method | Key Considerations |
|-----------|-------------------|-------------------|
| Flask | Custom middleware or before_request/after_request hooks | Use Flask's application context for storage synchronization |
| Django | Custom middleware inheriting from Django middleware base | Integrate with Django's request/response cycle and user authentication |
| FastAPI | Dependency injection or custom middleware | Leverage FastAPI's async support and dependency injection for context setup |
| Tornado | RequestHandler subclass or custom middleware | Handle Tornado's async request processing and ensure context preservation |
| WSGI apps | WSGI middleware wrapping the application | Universal approach working with any WSGI-compatible framework |
| ASGI apps | ASGI middleware for async applications | Handle async request processing and ensure context survives async boundaries |
```

The middleware also handles **request context inheritance** for applications that spawn background tasks or make inter-service calls during request processing. When the application creates background tasks, the middleware-established context should be available to those tasks. When making outbound HTTP requests, the correlation ID should be automatically included in request headers.

### Implementation Guidance

The context and correlation system requires careful coordination between multiple storage mechanisms and careful handling of async boundaries. This implementation guidance provides concrete patterns for building robust context propagation that works across both synchronous and asynchronous code paths.

**A. Technology Recommendations Table:**
| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Correlation ID Generation | UUID4 with Python uuid module | Custom base62 IDs with service prefixes |
| Context Storage | threading.local + contextvars.ContextVar | Redis for distributed context sharing |
| Async Context | Python contextvars (3.7+) | Custom async-local implementation |
| HTTP Middleware | Simple WSGI middleware | Framework-specific middleware with optimizations |
| Context Serialization | JSON with custom encoder | MessagePack for efficiency |
| Context Cleanup | Manual cleanup in finally blocks | Automatic cleanup with context managers |

**B. Recommended File/Module Structure:**
```
project-root/
  logging_system/
    context/
      __init__.py              ← Public context API exports
      correlation.py           ← Correlation ID generation and management
      propagation.py           ← Thread-local and async context storage
      async_bridge.py          ← Async context preservation mechanisms
      middleware.py            ← HTTP request context middleware
      storage.py               ← Context storage backends (thread-local, async-local)
      serialization.py         ← Context serialization utilities
    core/
      logger.py                ← Logger class with context integration
      record.py                ← LogRecord with context fields
  tests/
    context/
      test_correlation.py      ← Correlation ID tests
      test_propagation.py      ← Context propagation tests
      test_async_bridge.py     ← Async context tests
      test_middleware.py       ← Middleware integration tests
```

**C. Infrastructure Starter Code:**

Complete correlation ID generation system:

```python
# logging_system/context/correlation.py
import uuid
import threading
from typing import Optional

class CorrelationIDGenerator:
    """Thread-safe correlation ID generator with optional service prefixes."""
    
    def __init__(self, service_name: Optional[str] = None):
        self.service_name = service_name
        self._lock = threading.Lock()
    
    def generate(self) -> str:
        """Generate a new correlation ID with optional service prefix."""
        correlation_id = str(uuid.uuid4())
        if self.service_name:
            return f"{self.service_name}-{correlation_id}"
        return correlation_id
    
    def is_valid(self, correlation_id: str) -> bool:
        """Validate correlation ID format."""
        if self.service_name:
            if not correlation_id.startswith(f"{self.service_name}-"):
                return False
            uuid_part = correlation_id[len(self.service_name) + 1:]
        else:
            uuid_part = correlation_id
        
        try:
            uuid.UUID(uuid_part)
            return True
        except ValueError:
            return False

# Global generator instance
_default_generator = CorrelationIDGenerator()

def generate_correlation_id() -> str:
    """Generate a new correlation ID using the default generator."""
    return _default_generator.generate()

def set_service_name(service_name: str) -> None:
    """Set service name for correlation ID generation."""
    global _default_generator
    _default_generator = CorrelationIDGenerator(service_name)
```

Complete context storage system:

```python
# logging_system/context/storage.py
import threading
import contextvars
from typing import Dict, Any, Optional
from copy import deepcopy

class ContextStorage:
    """Hybrid context storage using both thread-local and async-local mechanisms."""
    
    def __init__(self):
        self._thread_local = threading.local()
        self._async_var: contextvars.ContextVar = contextvars.ContextVar(
            'logging_context', 
            default={}
        )
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context from both storage mechanisms."""
        # Get thread-local context
        thread_context = getattr(self._thread_local, 'context', {})
        
        # Get async context
        async_context = self._async_var.get({})
        
        # Merge contexts (async takes precedence)
        merged_context = thread_context.copy()
        merged_context.update(async_context)
        
        return merged_context
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set context in both storage mechanisms."""
        # Set in thread-local storage
        self._thread_local.context = deepcopy(context)
        
        # Set in async storage
        self._async_var.set(deepcopy(context))
    
    def add_fields(self, **fields) -> None:
        """Add fields to current context."""
        current_context = self.get_context()
        current_context.update(fields)
        self.set_context(current_context)
    
    def clear_context(self) -> None:
        """Clear context from both storage mechanisms."""
        # Clear thread-local
        if hasattr(self._thread_local, 'context'):
            delattr(self._thread_local, 'context')
        
        # Clear async context
        self._async_var.set({})
    
    def copy_context(self) -> Dict[str, Any]:
        """Create a deep copy of current context for task spawning."""
        return deepcopy(self.get_context())

# Global storage instance
_context_storage = ContextStorage()
```

Complete request middleware foundation:

```python
# logging_system/context/middleware.py
import time
from typing import Dict, Any, Optional, Callable
from .correlation import generate_correlation_id
from .storage import _context_storage

class RequestContextMiddleware:
    """WSGI middleware for automatic request context setup."""
    
    def __init__(self, app: Callable, service_name: Optional[str] = None):
        self.app = app
        self.service_name = service_name
    
    def __call__(self, environ: Dict[str, Any], start_response: Callable):
        # Extract request context
        request_context = self._extract_request_context(environ)
        
        # Set up context storage
        _context_storage.set_context(request_context)
        
        request_start = time.time()
        
        def enhanced_start_response(status: str, headers: list, exc_info=None):
            # Add correlation headers to response
            headers.append(('X-Correlation-ID', request_context['correlation_id']))
            
            request_duration = int((time.time() - request_start) * 1000)
            headers.append(('X-Request-Duration', str(request_duration)))
            
            return start_response(status, headers, exc_info)
        
        try:
            # Process request with context
            return self.app(environ, enhanced_start_response)
        finally:
            # Clean up context
            _context_storage.clear_context()
    
    def _extract_request_context(self, environ: Dict[str, Any]) -> Dict[str, Any]:
        """Extract request context from WSGI environ."""
        # Extract correlation ID from headers
        correlation_id = (
            environ.get('HTTP_X_CORRELATION_ID') or
            environ.get('HTTP_X_REQUEST_ID') or
            generate_correlation_id()
        )
        
        # Extract basic request metadata
        context = {
            'correlation_id': correlation_id,
            'request_method': environ.get('REQUEST_METHOD', 'UNKNOWN'),
            'request_path': environ.get('PATH_INFO', '/'),
            'request_start_time': time.time(),
            'client_ip': self._get_client_ip(environ),
            'user_agent': environ.get('HTTP_USER_AGENT', 'Unknown'),
        }
        
        # Add service name if configured
        if self.service_name:
            context['service_name'] = self.service_name
        
        return context
    
    def _get_client_ip(self, environ: Dict[str, Any]) -> str:
        """Extract client IP respecting proxy headers."""
        # Check for forwarded headers first
        forwarded_for = environ.get('HTTP_X_FORWARDED_FOR')
        if forwarded_for:
            # Take the first IP from the chain
            return forwarded_for.split(',')[0].strip()
        
        real_ip = environ.get('HTTP_X_REAL_IP')
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return environ.get('REMOTE_ADDR', 'unknown')
```

**D. Core Logic Skeleton Code:**

Context propagation system for implementation:

```python
# logging_system/context/propagation.py
from typing import Dict, Any, Optional, ContextManager
from contextlib import contextmanager
from .storage import _context_storage

class LoggingContext:
    """Context manager for logging context propagation."""
    
    @staticmethod
    def get_current() -> Dict[str, Any]:
        """Get current logging context from storage."""
        # TODO 1: Retrieve context from hybrid storage mechanism
        # TODO 2: Handle case where no context exists (return empty dict)
        # TODO 3: Ensure returned context is a copy to prevent external modification
        pass
    
    @staticmethod
    def set_current(context: Dict[str, Any]) -> None:
        """Set current logging context in storage."""
        # TODO 1: Validate context is serializable (all values are JSON-compatible)
        # TODO 2: Set context in both thread-local and async-local storage
        # TODO 3: Handle storage errors gracefully (log warning, continue)
        pass
    
    @staticmethod
    def add_fields(**fields) -> None:
        """Add fields to current context without replacing existing context."""
        # TODO 1: Get current context from storage
        # TODO 2: Merge new fields with existing context (new fields take precedence)
        # TODO 3: Validate merged context is still serializable
        # TODO 4: Update storage with merged context
        # Hint: Use dict.update() but handle potential serialization issues
        pass
    
    @staticmethod
    def clear() -> None:
        """Clear current logging context from all storage."""
        # TODO 1: Clear context from thread-local storage
        # TODO 2: Clear context from async-local storage  
        # TODO 3: Handle storage errors (don't raise exceptions from cleanup)
        pass
    
    @staticmethod
    @contextmanager
    def inherit(additional_context: Optional[Dict[str, Any]] = None) -> ContextManager[Dict[str, Any]]:
        """Context manager that inherits parent context and restores it on exit."""
        # TODO 1: Capture current context as parent context
        # TODO 2: Create child context by copying parent context
        # TODO 3: Merge additional_context into child context if provided
        # TODO 4: Set child context as current context
        # TODO 5: Yield the child context to the with block
        # TODO 6: In finally block, restore parent context regardless of exceptions
        # Hint: Use try/finally to ensure parent context is always restored
        pass

def propagate_to_async_task(task_func: callable, *args, **kwargs):
    """Helper to propagate current context to a new async task."""
    # TODO 1: Capture current context before task creation
    # TODO 2: Create wrapper function that sets context before calling task_func
    # TODO 3: Return wrapped function that can be passed to asyncio.create_task()
    # Hint: The wrapper should set context, call original function, then clean up
    pass
```

Async context bridge for implementation:

```python
# logging_system/context/async_bridge.py
import asyncio
import contextvars
from typing import Dict, Any, Awaitable, TypeVar
from .propagation import LoggingContext

T = TypeVar('T')

async def preserve_context(coro: Awaitable[T]) -> T:
    """Ensure logging context is preserved across async operation."""
    # TODO 1: Capture current logging context before await
    # TODO 2: Execute the coroutine and await its result
    # TODO 3: Restore logging context after coroutine completes
    # TODO 4: Handle exceptions by preserving context during error propagation
    # Hint: Use try/finally to ensure context restoration
    pass

def create_task_with_context(coro: Awaitable[T], context: Optional[Dict[str, Any]] = None) -> asyncio.Task[T]:
    """Create async task with explicit context propagation."""
    # TODO 1: Use context parameter or capture current context if None
    # TODO 2: Create wrapper coroutine that sets context before running original
    # TODO 3: Create asyncio task from wrapper coroutine
    # TODO 4: Return the created task
    # Hint: Use contextvars.copy_context() for proper async context copying
    pass

async def gather_with_context(*coroutines: Awaitable) -> list:
    """Gather multiple coroutines while preserving context in each."""
    # TODO 1: Capture current context to propagate to all coroutines
    # TODO 2: Wrap each coroutine with context preservation
    # TODO 3: Use asyncio.gather() on wrapped coroutines
    # TODO 4: Return results from asyncio.gather()
    pass

class AsyncContextManager:
    """Context manager for async operations with logging context."""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        # TODO 1: Store provided context or capture current context
        # TODO 2: Initialize previous_context storage for restoration
        pass
    
    async def __aenter__(self) -> Dict[str, Any]:
        # TODO 1: Capture current context as previous_context
        # TODO 2: Set self.context as current context
        # TODO 3: Return the active context
        pass
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # TODO 1: Restore previous_context as current context
        # TODO 2: Handle any storage errors gracefully
        # TODO 3: Don't suppress exceptions unless they're context-related
        pass
```

**E. Language-Specific Hints:**

- Use `threading.local()` for thread-local storage in Python
- Use `contextvars.ContextVar` for async-local storage (requires Python 3.7+)
- Use `asyncio.current_task()` to get current task for task-local storage
- Use `copy.deepcopy()` to create isolated context copies
- Use `uuid.uuid4()` for correlation ID generation with strong uniqueness guarantees
- Use WSGI middleware pattern: `app(environ, start_response)` for universal framework compatibility
- Use `functools.wraps` when creating context wrapper functions to preserve function metadata
- Use `contextlib.contextmanager` decorator for simple context managers

**F. Milestone Checkpoint:**

After implementing Milestone 3, verify context and correlation functionality:

**Command to run**: `python -m pytest tests/context/ -v`

**Expected test output**:
```
test_correlation_id_generation ✓
test_context_propagation_sync ✓  
test_context_propagation_async ✓
test_async_context_bridge ✓
test_request_middleware_integration ✓
```

**Manual verification steps**:
1. **Start test web server**: `python examples/web_app.py`
2. **Make request with correlation ID**: `curl -H "X-Correlation-ID: test-123" localhost:8000/api/test`
3. **Verify logs contain correlation ID**: All log entries should include `"correlation_id": "test-123"`
4. **Make request without correlation ID**: `curl localhost:8000/api/test`  
5. **Verify generated correlation ID**: Logs should contain auto-generated UUID4 correlation ID
6. **Test nested function calls**: All log entries from nested functions should inherit same correlation ID
7. **Test async operations**: Async tasks spawned during request should preserve correlation context

**Signs something is wrong**:
- **Logs missing correlation IDs**: Context propagation not working, check storage mechanisms
- **Different correlation IDs within same request**: Context not propagating properly across function calls
- **Context lost in async operations**: Async context bridge not functioning, check contextvars usage
- **Context leaking between requests**: Context cleanup not working, check middleware finally blocks

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|----------------|-----|
| Correlation ID missing from logs | Context not set or not propagating | Add debug logging in context storage get/set methods | Ensure context middleware runs before request processing |
| Different correlation IDs in same request | New IDs being generated instead of inheriting | Check context inheritance in nested function calls | Use LoggingContext.inherit() context manager |
| Context lost in async operations | Thread-local storage used in async code | Add logging before/after async operations to track context | Use contextvars.ContextVar for async operations |
| Context leaking between requests | Context not cleared after request completion | Add request start/end logging with context state | Ensure context.clear() in middleware finally block |
| Memory usage growing over time | Context objects not being garbage collected | Monitor context storage size over time | Check for circular references in context data |
| Performance degradation | Context copying overhead | Profile context get/set operations | Optimize context serialization, limit context size |


## Error Handling and Edge Cases

> **Milestone(s):** This section provides critical error handling patterns for all three milestones, with handler failure recovery supporting Milestone 1 (Logger Core), serialization edge cases affecting Milestone 2 (Structured Output), and context cleanup preventing memory issues in Milestone 3 (Context & Correlation).

### The Safety Net Analogy

Think of error handling in a logging system like the safety nets in a circus. When an acrobat performs dangerous stunts high above the ground, there are multiple layers of protection: primary safety harnesses, backup cables, and finally the net below. Similarly, our logging system must have multiple layers of protection because **logging cannot be allowed to crash the main application**. When a file system is full, when network connections fail, when objects contain circular references, the logging system must gracefully degrade rather than bringing down the entire service.

The critical insight is that logging is a **non-critical path operation** from the application's perspective. If an e-commerce service cannot write logs, it should still process orders successfully. However, losing observability is serious, so we need sophisticated recovery mechanisms that maintain partial functionality while alerting operators to the underlying issues.

### Handler Failure Recovery

Handler failure recovery addresses the reality that output destinations are inherently unreliable. File systems can become full, network connections can timeout, and remote log aggregation services can become unavailable. The logging system must continue operating in these scenarios while providing mechanisms for recovery and alerting.

> **Decision: Multi-Handler Isolation**
> - **Context**: When one handler fails (e.g., network timeout), other handlers should continue working
> - **Options Considered**: Fail-fast on any handler error, retry all handlers together, isolate handler failures
> - **Decision**: Isolate handler failures so each handler's success/failure is independent
> - **Rationale**: A full disk shouldn't prevent console logging, and network issues shouldn't stop file logging
> - **Consequences**: Requires individual error handling per handler but maintains maximum availability

The handler dispatch mechanism implements the **Circuit Breaker Pattern** for each handler type. When a handler experiences repeated failures, it transitions to a failed state and stops attempting operations for a configured time period. This prevents cascading failures and reduces resource consumption during outages.

#### Handler Error States

Each handler maintains state information to track its health and implement appropriate recovery strategies:

| State | Description | Behavior | Recovery Trigger |
|-------|-------------|----------|------------------|
| `HEALTHY` | Handler operating normally | Process all log records immediately | N/A - default state |
| `DEGRADED` | Experiencing intermittent failures | Process records with retry logic | Successful operation after failure |
| `FAILED` | Consecutive failures exceeded threshold | Drop records or queue for later | Manual reset or timeout period |
| `RECOVERING` | Attempting to return to service | Test with low-priority records first | Sustained successful operations |

#### Failure Detection and Response

The logging system implements sophisticated failure detection that goes beyond simple exception catching. Different types of failures require different response strategies:

| Failure Type | Detection Method | Immediate Response | Recovery Strategy |
|--------------|------------------|-------------------|-------------------|
| File System Full | `OSError` with `errno.ENOSPC` | Switch to console handler, emit warning | Monitor disk space, resume when available |
| Permission Denied | `PermissionError` on file operations | Attempt alternative file location | Alert administrator, try alternative paths |
| Network Timeout | Socket timeout during remote handler | Buffer records locally | Exponential backoff retry, eventual discard |
| Serialization Error | JSON encoding exception | Log error record instead | Safe serialize with type conversion |

The **safe_call** function provides the foundation for handler isolation:

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `Callable` | The handler function to execute safely |
| `*args` | `Any` | Positional arguments for the handler function |
| `default` | `Any` | Default return value if function fails |
| `timeout` | `float` | Maximum execution time before considering failed |
| `**kwargs` | `Any` | Keyword arguments for the handler function |

When a handler fails, the system follows this recovery procedure:

1. **Immediate Isolation**: The failing handler is marked as degraded and subsequent calls use defensive timeouts
2. **Error Classification**: The exception type determines whether the failure is likely temporary or permanent
3. **Alternative Routing**: If available, log records are routed to backup handlers of the same type
4. **Graceful Degradation**: Critical system information is logged to console as a fallback
5. **Recovery Monitoring**: Background health checks test handler availability without affecting main log flow
6. **Automatic Restoration**: Successful operations gradually restore the handler to healthy status

> The key insight is that logging failures should never propagate as exceptions to the calling application code. A failed log operation returns normally but may emit diagnostic information through alternative channels.

#### Buffer Management for Failed Handlers

When handlers fail, the system implements intelligent buffering to preserve critical log information without consuming unbounded memory:

| Buffer Type | Capacity | Retention Policy | Flush Trigger |
|-------------|----------|------------------|---------------|
| `MEMORY_BUFFER` | 1000 records | FIFO replacement when full | Handler recovery |
| `DISK_BUFFER` | 100MB files | Rotate when size limit reached | Handler recovery or manual flush |
| `PRIORITY_BUFFER` | 100 ERROR/FATAL records | Never discard high-priority records | Handler recovery only |

⚠️ **Pitfall: Unbounded Buffer Growth**
A common mistake is buffering failed log records indefinitely, which leads to memory leaks during prolonged outages. Always implement size limits and explicit discard policies for buffered data.

### Serialization Edge Cases

Structured logging's reliance on JSON serialization introduces complex edge cases that can cause runtime failures. The system must handle non-serializable Python objects, circular references, deeply nested structures, and objects with dynamic or sensitive content gracefully.

> **Decision: Safe Serialization with Fallback**
> - **Context**: User code may pass any Python object as context, including non-serializable types
> - **Options Considered**: Reject non-serializable objects, convert everything to strings, selective type conversion
> - **Decision**: Implement custom JSON encoder with intelligent type conversion and circular reference detection
> - **Rationale**: Preserves maximum information while ensuring JSON output is always valid
> - **Consequences**: Requires custom encoder logic but provides robust handling of edge cases

#### Non-Serializable Object Handling

The `SafeJSONEncoder` provides intelligent conversion for Python objects that don't have native JSON representations:

| Object Type | Conversion Strategy | Example Input | Example Output |
|-------------|-------------------|---------------|----------------|
| `datetime` | ISO 8601 string conversion | `datetime(2023, 12, 25)` | `"2023-12-25T00:00:00Z"` |
| `Decimal` | String conversion preserving precision | `Decimal("123.456")` | `"123.456"` |
| `UUID` | String conversion | `UUID('123e4567-e89b-12d3-a456-426614174000')` | `"123e4567-e89b-12d3-a456-426614174000"` |
| `Exception` | Type and message extraction | `ValueError("Invalid input")` | `{"type": "ValueError", "message": "Invalid input"}` |
| `Custom Objects` | Repr string with type information | `MyClass(value=42)` | `{"type": "MyClass", "repr": "MyClass(value=42)"}` |

The serialization process follows a defensive strategy that prioritizes successful log output over perfect representation:

1. **Primary Serialization Attempt**: Use standard JSON encoder for basic types (str, int, float, bool, list, dict)
2. **Custom Type Conversion**: Apply type-specific converters for known problematic types like datetime and UUID
3. **Repr Fallback**: For unknown custom objects, capture the string representation and type information
4. **Error Isolation**: If individual fields fail serialization, replace with error markers rather than failing entire record
5. **Size Estimation**: Before full serialization, estimate the output size to prevent memory exhaustion

#### Circular Reference Protection

Circular references occur when objects contain references to themselves, either directly or through a chain of references. Without protection, JSON serialization would enter infinite loops and eventually crash with stack overflow errors.

The **safe_serialize** function implements circular reference detection using a tracking set:

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `Any` | The object to serialize, typically a dictionary |
| `max_depth` | `int` | Maximum nesting depth before truncation (default: 10) |
| `seen_objects` | `Set[int]` | Set of object IDs already being processed |
| `current_depth` | `int` | Current recursion depth for stack protection |

The circular reference detection algorithm works as follows:

1. **Object Identity Tracking**: For each object encountered, record its `id()` in the `seen_objects` set
2. **Reference Check**: Before processing any object, check if its ID is already in the tracking set
3. **Circular Reference Marker**: If a circular reference is detected, replace with a marker: `{"_circular_ref": "object_type_at_depth_N"}`
4. **Depth Limiting**: Even without circular references, stop recursion at maximum depth to prevent stack overflow
5. **Cleanup**: Remove object IDs from tracking set when exiting their scope to allow repeated references

#### Large Object Handling

Large objects in logging context can consume excessive memory and make log records unreadable. The system implements size-based truncation and sampling strategies:

| Size Threshold | Handling Strategy | Example |
|---------------|------------------|---------|
| < 1KB | Serialize completely | Small dictionaries, short strings |
| 1KB - 10KB | Serialize with truncation warning | Medium configuration objects |
| 10KB - 100KB | Serialize keys only, with size information | Large data structures |
| > 100KB | Replace with summary metadata | Database query results, file contents |

The **estimate_serialized_size** function provides efficient size estimation without full serialization:

1. **Type-Based Estimation**: Use heuristics for common types (strings: byte length, lists: sum of elements, etc.)
2. **Sampling**: For large containers, sample first N elements and extrapolate total size
3. **Early Termination**: Stop estimation when size threshold is clearly exceeded
4. **Memory Protection**: Never allocate memory proportional to estimated size during estimation

⚠️ **Pitfall: Sensitive Data in Logs**
Always implement field filtering for sensitive information like passwords, tokens, and personal data. Consider using a configurable blacklist of field names that should be redacted during serialization.

#### Encoding Error Recovery

When serialization fails despite all protective measures, the system implements progressive degradation:

1. **Field-Level Recovery**: If individual context fields fail, replace them with error descriptions and continue
2. **Message Preservation**: Always preserve the primary log message, even if all context fails serialization
3. **Diagnostic Information**: Include serialization error details in a special `_serialization_errors` field
4. **Raw Fallback**: As a last resort, convert the entire context to a string representation
5. **Error Logging**: Use a separate error channel to log serialization failures for debugging

### Context Cleanup and Memory Management

Context management in a structured logging system introduces subtle memory management challenges, particularly in long-running services with high request volumes. Without proper cleanup, context objects can accumulate indefinitely, leading to memory leaks and degraded performance.

> **Decision: Scoped Context with Automatic Cleanup**
> - **Context**: Context objects must be cleaned up when requests complete, but manual cleanup is error-prone
> - **Options Considered**: Manual cleanup in application code, reference counting, scoped context managers
> - **Decision**: Implement context managers with automatic cleanup and weak references for orphan detection
> - **Rationale**: Reduces developer burden while providing deterministic cleanup semantics
> - **Consequences**: Requires careful design of context inheritance and async context bridging

#### Context Lifecycle Management

The logging context follows a strict lifecycle that ensures proper resource management across different execution models:

| Lifecycle Phase | Thread-Local Context | Async Context | Cleanup Actions |
|-----------------|---------------------|---------------|-----------------|
| **Creation** | Store in `threading.local` | Set `contextvars.ContextVar` | Initialize parent reference |
| **Inheritance** | Copy parent context to child | `contextvars.copy_context()` | Establish parent-child relationship |
| **Modification** | Update thread-local storage | Create new context with changes | Preserve immutability of parent |
| **Propagation** | Manual passing between threads | Automatic with async tasks | Sync both storage mechanisms |
| **Cleanup** | Clear on request completion | Automatic scope exit | Remove circular references |

The dual storage strategy ensures context availability regardless of execution model:

| Storage Type | Use Case | Advantages | Limitations |
|--------------|----------|------------|-------------|
| `threading.local` | Synchronous request processing | Simple access, no parameter passing | Manual cleanup required |
| `contextvars.ContextVar` | Async/await operations | Automatic task isolation | Python 3.7+ only |
| **Hybrid Approach** | Mixed sync/async applications | Best of both worlds | Requires synchronization logic |

#### Memory Leak Prevention

Context objects can cause memory leaks through several mechanisms that require active prevention:

**Circular Reference Prevention:**
1. **Weak Parent References**: Child contexts hold weak references to parents to break reference cycles
2. **Context Isolation**: Changes in child contexts create new objects rather than modifying parents
3. **Explicit Cleanup**: Context managers ensure deterministic cleanup even when exceptions occur
4. **Orphan Detection**: Background monitoring identifies contexts that haven't been properly cleaned up

**Context Size Management:**
The system implements several strategies to prevent context objects from growing unboundedly:

| Strategy | Implementation | Trigger | Action |
|----------|----------------|---------|--------|
| **Field Count Limit** | Maximum 50 context fields | On context modification | Remove oldest fields (LRU) |
| **Value Size Limit** | Maximum 1KB per field value | On field assignment | Truncate with warning marker |
| **Total Size Limit** | Maximum 10KB per context | On context serialization | Summarize large fields |
| **Depth Limit** | Maximum 10 inheritance levels | On child context creation | Flatten inheritance chain |

⚠️ **Pitfall: Context Accumulation in Long-Running Tasks**
In long-running background tasks, context fields can accumulate over time if not properly scoped. Always use context managers or explicit cleanup in loops and long-running operations.

#### Async Context Bridge Implementation

The async context bridge ensures that logging context is preserved across async/await boundaries, which is critical for maintaining request tracing in async applications:

| Component | Responsibility | Implementation Strategy |
|-----------|----------------|------------------------|
| `AsyncContextManager` | Preserve context across await points | Copy context before async operations |
| `TaskContextPropagator` | Inject context into new tasks | Wrap task functions with context restoration |
| `CoroutineContextWrapper` | Maintain context in coroutines | Use `contextvars` for automatic propagation |

The context preservation algorithm for async operations:

1. **Pre-Async Capture**: Before any async operation, capture the current context from both storage mechanisms
2. **Context Packaging**: Create a context snapshot that includes all current fields and correlation IDs
3. **Async Task Wrapping**: Wrap the async function to restore context when the task begins execution
4. **Context Restoration**: When the async task starts, restore the captured context to both storage types
5. **Cleanup Registration**: Register cleanup callbacks to clear context when the task completes
6. **Exception Handling**: Ensure context cleanup occurs even when async operations raise exceptions

#### Context Memory Monitoring

The system provides monitoring and debugging capabilities for context memory usage:

| Metric | Collection Method | Purpose |
|--------|------------------|---------|
| `active_context_count` | Track context creation/destruction | Detect context leaks |
| `average_context_size` | Measure serialized context size | Identify oversized contexts |
| `context_inheritance_depth` | Track parent-child relationships | Prevent deep inheritance chains |
| `cleanup_failure_count` | Count failed cleanup operations | Monitor cleanup effectiveness |

**Context Debugging Tools:**

1. **Context Dump**: Function to serialize and display all active contexts for debugging
2. **Leak Detection**: Periodic scanning for contexts that haven't been accessed recently
3. **Size Analysis**: Breakdown of context memory usage by field type and size
4. **Inheritance Visualization**: Display parent-child relationships for complex context hierarchies

The context cleanup process implements several safety mechanisms:

1. **Timeout-Based Cleanup**: Contexts older than a configured age are automatically cleaned up
2. **Reference Counting**: Track how many active operations reference each context
3. **Graceful Degradation**: If cleanup fails, mark contexts as stale rather than leaving them indefinitely
4. **Background Monitoring**: Separate thread monitors context health and performs maintenance
5. **Emergency Cleanup**: When memory pressure is detected, aggressively clean up non-essential contexts

> The fundamental principle is that context management should be invisible to application developers while providing robust memory management and debugging capabilities for production operations.

![Handler Dispatch Sequence](./diagrams/handler-dispatch-sequence.svg)

### Common Pitfalls in Error Handling

⚠️ **Pitfall: Logging Errors That Break Application Flow**
The most critical mistake is allowing logging operations to throw exceptions that propagate to application code. Always wrap logging operations in try-catch blocks and provide default behaviors for all failure scenarios.

⚠️ **Pitfall: Synchronous I/O in High-Performance Paths**
Using synchronous file or network I/O for logging in request processing threads can significantly impact application performance. Consider async handlers or background thread dispatching for high-throughput scenarios.

⚠️ **Pitfall: Retrying Failed Operations Indefinitely**
Without proper backoff and circuit breaking, failed log handlers can consume CPU and network resources indefinitely. Implement exponential backoff and maximum retry limits.

⚠️ **Pitfall: Ignoring Handler Configuration Validation**
Invalid handler configurations (non-existent file paths, invalid network addresses) should be detected at startup rather than causing runtime failures. Implement configuration validation during logger initialization.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Recovery | Basic try/except with logging | Circuit breaker pattern with metrics |
| Serialization | Custom JSON encoder | Protocol Buffers with fallback |
| Context Storage | Threading.local only | Hybrid threading.local + contextvars |
| Buffer Management | In-memory lists | Persistent disk-backed queues |
| Monitoring | Simple counters | Prometheus metrics with alerting |

#### Infrastructure Starter Code

```python
import json
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set, Union
from collections import deque
from enum import Enum
import contextvars

class HandlerState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable objects gracefully."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_objects = set()
        self._current_depth = 0
        self.max_depth = kwargs.get('max_depth', MAX_DEPTH)
    
    def default(self, obj):
        # TODO: Implement type-specific conversion for datetime, UUID, Decimal, Exception
        # TODO: Add circular reference detection using object id tracking
        # TODO: Implement depth limiting to prevent stack overflow
        # TODO: Add fallback string conversion for unknown types
        pass

class ThreadSafeCounter:
    """Thread-safe counter for tracking handler failures."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def reset(self) -> None:
        with self._lock:
            self._value = 0
    
    @property
    def value(self) -> int:
        with self._lock:
            return self._value

class CircuitBreaker:
    """Circuit breaker for handler failure isolation."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self._failure_count = ThreadSafeCounter()
        self._last_failure_time = None
        self._state = HandlerState.HEALTHY
        self._lock = threading.RLock()
    
    def call(self, func, *args, **kwargs):
        # TODO: Check circuit breaker state before allowing call
        # TODO: Execute function and handle success/failure
        # TODO: Update failure count and state based on result
        # TODO: Implement timeout logic for failed state recovery
        pass

class ContextBuffer:
    """Memory-managed buffer for failed log records."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 10):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._buffer = deque()
        self._current_memory = 0
        self._lock = threading.RLock()
    
    def add_record(self, record: 'LogRecord') -> bool:
        # TODO: Estimate record memory size
        # TODO: Check if adding record would exceed limits
        # TODO: Remove old records if necessary (FIFO)
        # TODO: Add record to buffer and update memory tracking
        pass
    
    def flush_to_handler(self, handler: 'Handler') -> int:
        # TODO: Attempt to send buffered records to recovered handler
        # TODO: Remove successfully sent records from buffer
        # TODO: Return count of successfully flushed records
        pass

def safe_serialize(data: Any, max_depth: int = MAX_DEPTH) -> str:
    """Serialize data to JSON with circular reference and size protection."""
    # TODO: Create SafeJSONEncoder instance with depth limit
    # TODO: Implement size estimation before full serialization
    # TODO: Handle serialization exceptions gracefully
    # TODO: Return valid JSON string even on partial failures
    pass

def estimate_serialized_size(data: Any) -> int:
    """Estimate JSON serialization size without full serialization."""
    # TODO: Implement type-based size estimation
    # TODO: Sample large containers to avoid O(n) computation
    # TODO: Return conservative size estimate in bytes
    pass

def safe_call(func, *args, default=None, timeout=DEFAULT_TIMEOUT, **kwargs):
    """Safely execute function with timeout and exception handling."""
    # TODO: Implement timeout mechanism using threading.Timer or signal
    # TODO: Catch and log exceptions without propagating
    # TODO: Return default value on failure or timeout
    # TODO: Log diagnostic information about failures
    pass
```

#### Core Error Handling Skeleton

```python
class BaseHandler:
    """Base class for all log output handlers with error recovery."""
    
    def __init__(self, name: str):
        self.name = name
        self._circuit_breaker = CircuitBreaker()
        self._buffer = ContextBuffer()
        self._state = HandlerState.HEALTHY
        self._last_success_time = time.time()
    
    def handle(self, record: LogRecord) -> bool:
        # TODO 1: Check circuit breaker state - return False if failed
        # TODO 2: Attempt to send record using _write_record method
        # TODO 3: Handle success case - update circuit breaker, flush buffer
        # TODO 4: Handle failure case - add to buffer, update circuit breaker
        # TODO 5: Return True if handled (success or buffered), False if dropped
        pass
    
    def _write_record(self, record: LogRecord) -> None:
        # TODO: Subclasses implement actual output logic here
        # This method should raise exceptions for failures
        raise NotImplementedError("Subclasses must implement _write_record")
    
    def recover(self) -> bool:
        # TODO 1: Test handler availability with a dummy record
        # TODO 2: If successful, attempt to flush buffered records
        # TODO 3: Update handler state based on recovery success
        # TODO 4: Return True if handler is now operational
        pass

class FileHandler(BaseHandler):
    """File output handler with disk space and permission error handling."""
    
    def __init__(self, file_path: str):
        super().__init__(f"file:{file_path}")
        self.file_path = file_path
        self._file_handle = None
    
    def _write_record(self, record: LogRecord) -> None:
        # TODO 1: Check if file handle is open, open if necessary
        # TODO 2: Format record using configured formatter
        # TODO 3: Write formatted record to file
        # TODO 4: Flush to ensure data is written (consider sync vs async)
        # TODO 5: Handle specific errors: disk full, permission denied, etc.
        pass

class ContextStorage:
    """Dual storage for thread-local and async context."""
    
    def __init__(self):
        self._thread_local = threading.local()
        self._async_var = contextvars.ContextVar('logging_context', default={})
        self._active_contexts = weakref.WeakSet()
        self._lock = threading.RLock()
    
    def get_current(self) -> Dict[str, Any]:
        # TODO 1: Try to get context from async storage (contextvars)
        # TODO 2: Fall back to thread-local storage if async not available
        # TODO 3: Return empty dict if no context is set
        # TODO 4: Track context access for monitoring
        pass
    
    def set_current(self, context: Dict[str, Any]) -> None:
        # TODO 1: Validate context size and field count limits
        # TODO 2: Set context in both storage mechanisms
        # TODO 3: Register context for cleanup monitoring
        # TODO 4: Handle storage failures gracefully
        pass
    
    def cleanup_orphaned_contexts(self) -> int:
        # TODO 1: Identify contexts not accessed recently
        # TODO 2: Check if contexts have active references
        # TODO 3: Clean up orphaned contexts from storage
        # TODO 4: Return count of cleaned up contexts for monitoring
        pass

def preserve_context_async(coro):
    """Decorator to preserve logging context across async operations."""
    # TODO 1: Capture current context before async operation
    # TODO 2: Create wrapper that restores context when async function starts
    # TODO 3: Ensure context cleanup on async completion or exception
    # TODO 4: Handle nested async calls and context inheritance
    pass
```

#### Recommended File Structure

```
structured_logging/
  core/
    __init__.py
    logger.py                   ← Logger, LoggerRegistry
    record.py                   ← LogRecord, LogLevel constants
    handlers/
      __init__.py
      base.py                   ← BaseHandler, error recovery
      file_handler.py           ← FileHandler with disk error handling
      console_handler.py        ← ConsoleHandler with color/encoding errors
      remote_handler.py         ← NetworkHandler with timeout/retry logic
  formatters/
    __init__.py
    base.py                     ← BaseFormatter interface
    json_formatter.py           ← JSONFormatter with SafeJSONEncoder
    pretty_formatter.py         ← PrettyFormatter for development
  context/
    __init__.py
    storage.py                  ← ContextStorage, cleanup logic
    correlation.py              ← CorrelationIDGenerator
    middleware.py               ← RequestContextMiddleware
  utils/
    __init__.py
    serialization.py            ← safe_serialize, estimate_serialized_size
    safety.py                   ← safe_call, CircuitBreaker
    monitoring.py               ← Context monitoring and leak detection
  tests/
    test_error_handling.py      ← Handler failure scenarios
    test_serialization.py       ← Edge case serialization tests
    test_context_cleanup.py     ← Memory leak and cleanup tests
```

#### Milestone Checkpoints

**Checkpoint 1: Handler Failure Recovery**
```bash
# Test file handler with disk full simulation
python -m pytest tests/test_error_handling.py::test_file_handler_disk_full

# Expected: Handler switches to degraded state, logs continue to console
# Signs of issues: Exceptions propagate to application, logs are lost entirely
```

**Checkpoint 2: Serialization Edge Cases**
```bash
# Test circular reference handling
python -c "
import logging_system
logger = logging_system.get_logger('test')
obj = {'self': None}
obj['self'] = obj
logger.info('Test circular ref', context_obj=obj)
"

# Expected: JSON output with circular reference marker
# Signs of issues: RecursionError, stack overflow, or malformed JSON
```

**Checkpoint 3: Context Cleanup**
```bash
# Test context cleanup under load
python tests/stress_test_context.py --requests=10000 --async-tasks=100

# Expected: Stable memory usage, no growth over time
# Signs of issues: Memory usage grows linearly, eventual OutOfMemoryError
```

#### Language-Specific Hints

**Python Error Handling:**
- Use `errno` module constants instead of hardcoded error codes: `errno.ENOSPC` for disk full
- `threading.local()` provides thread-isolated storage for synchronous contexts
- `contextvars.ContextVar` automatically handles async context isolation in Python 3.7+
- Use `weakref.WeakSet()` to track context objects without preventing garbage collection
- `json.JSONEncoder.default()` method is the proper override point for custom serialization

**Memory Management:**
- `sys.getsizeof()` provides memory size estimation for Python objects
- `gc.collect()` can be called explicitly when cleaning up large numbers of contexts
- Use `__slots__` in frequently created classes to reduce memory overhead
- `deque` from collections provides efficient FIFO operations for buffering

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Application crashes with JSON errors | Circular references in context | Add debug logging in SafeJSONEncoder | Implement proper circular reference detection |
| Memory usage grows over time | Context objects not cleaned up | Monitor `_active_contexts` size | Add explicit cleanup in finally blocks |
| Logs missing during high load | Handler failures not recovered | Check handler state and buffer sizes | Implement proper circuit breaker recovery |
| Context not propagating to async tasks | Missing contextvars setup | Add trace logging in context bridge | Use `contextvars.copy_context()` correctly |
| File handler stops working | Disk full or permission errors | Check handler error logs and disk space | Implement fallback handlers and monitoring |


## Testing Strategy

> **Milestone(s):** This section provides comprehensive testing patterns for all three milestones, with unit testing supporting individual component validation in Milestone 1 (Logger Core), integration scenarios verifying cross-component behavior in Milestone 2 (Structured Output), and end-to-end testing ensuring complete system functionality in Milestone 3 (Context & Correlation).

Think of testing a structured logging system like conducting a symphony orchestra rehearsal. Just as a conductor must verify that individual musicians can play their parts correctly (unit testing), that sections can harmonize together (integration testing), and that the complete orchestra produces the intended musical experience (end-to-end testing), our logging system requires systematic validation at multiple levels. Each test layer reveals different types of issues: unit tests catch logic errors in individual components, integration tests expose interaction problems between subsystems, and milestone checkpoints verify that the complete system meets production requirements.

The testing strategy for a production-grade logging system presents unique challenges compared to typical application testing. Logging systems must handle concurrent access from multiple threads, preserve context across asynchronous boundaries, and maintain performance under heavy load while gracefully degrading when output destinations fail. Traditional testing approaches often miss these concerns because they focus on single-threaded, synchronous execution paths. Our comprehensive testing strategy addresses these real-world complexities by systematically validating thread safety, async context preservation, error recovery, and performance characteristics.

### Unit Testing Approach

Unit testing for structured logging components requires isolation of individual subsystems while simulating the complex interactions they will encounter in production. Unlike typical business logic testing, logging component tests must verify thread safety guarantees, memory management correctness, and graceful handling of malformed data. Each component operates within strict performance constraints since logging should never become the bottleneck in application execution.

The **formatter testing strategy** focuses on serialization correctness under edge cases that commonly break JSON output in production systems. Formatters must handle circular references, non-serializable objects, extremely large data structures, and malformed Unicode strings without throwing exceptions that could crash the application. Test cases should include deeply nested objects that exceed the maximum serialization depth, objects containing datetime instances and custom classes, dictionaries with non-string keys, and data structures containing binary data or control characters.

| Test Category | Input Type | Expected Behavior | Failure Mode to Verify |
|---------------|-----------|------------------|------------------------|
| Circular References | Object referencing itself | Truncated serialization with warning | Infinite recursion crash |
| Non-Serializable Objects | Custom class instances | String representation fallback | JSON serialization exception |
| Maximum Depth Exceeded | Nested dict 20 levels deep | Truncation at MAX_DEPTH=10 | Stack overflow from recursion |
| Large Data Structures | Dict with 10,000 key-value pairs | Size estimation and truncation | Memory exhaustion or timeout |
| Unicode Edge Cases | Strings with control characters | Escaped JSON output | Malformed JSON structure |
| Binary Data | Byte arrays in context fields | Base64 encoding or hex representation | Encoding errors |

The `JSONFormatter` testing requires verification that output always produces valid single-line JSON regardless of input complexity. Test the `safe_serialize` method with objects containing functions, lambda expressions, open file handles, and thread locks. Verify that the `estimate_serialized_size` function accurately predicts memory usage before full serialization to prevent memory exhaustion attacks. Test timestamp formatting with various precision levels and timezone configurations.

**Handler testing strategy** emphasizes failure isolation and recovery behavior since handlers represent external dependencies that will inevitably fail in production. Each handler type requires different failure simulation techniques. File handlers need tests for filesystem permission errors, disk space exhaustion, and network filesystem disconnections. Network handlers require simulation of connection timeouts, DNS resolution failures, and temporary service unavailability.

| Handler Type | Failure Scenarios | Recovery Verification | Buffer Behavior |
|--------------|------------------|---------------------|-----------------|
| FileHandler | Permission denied, disk full | Automatic retry with backoff | Buffer records until disk space available |
| NetworkHandler | Connection timeout, DNS failure | Circuit breaker activation | Queue records for batch retry |
| StdoutHandler | Broken pipe, process termination | Graceful degradation | Immediate failure, no buffering |
| RemoteHandler | Service unavailable, auth failure | Exponential backoff retry | Persistent queue with size limits |

The `CircuitBreaker` component requires time-based testing that verifies state transitions occur at the correct intervals. Mock the system clock to control timeout behavior and verify that the circuit breaker correctly transitions from HEALTHY to DEGRADED to FAILED states based on consecutive failure counts. Test the recovery mechanism by simulating successful operations after the timeout period expires.

**Context management testing** focuses on thread safety and memory leak prevention since context storage maintains state that could accumulate indefinitely. The `ContextStorage` component must handle concurrent access from multiple threads while preserving isolation between different execution contexts. Test scenarios should include rapid context creation and cleanup, nested context inheritance, and cleanup of orphaned contexts when threads terminate unexpectedly.

| Test Scenario | Thread Pattern | Context Behavior | Memory Verification |
|---------------|---------------|------------------|-------------------|
| Concurrent Context Creation | 50 threads creating contexts simultaneously | No data corruption or lost updates | All contexts properly isolated |
| Nested Context Inheritance | Parent context with 5 levels of children | Child inherits all parent fields | Changes in children don't affect parents |
| Orphaned Context Cleanup | Thread exits without calling clear() | Automatic cleanup after timeout | No memory leaks detected |
| Context Size Limits | Context with 1MB of data | Truncation to prevent memory issues | Graceful size limiting |

The `ThreadSafeCounter` and `ThreadSafeDict` utility classes require stress testing under high concurrency to verify that lock contention doesn't create deadlocks or performance bottlenecks. Use property-based testing to generate random sequences of operations and verify that the final state matches expected results.

**Logger hierarchy testing** verifies that configuration inheritance and level filtering work correctly across complex parent-child relationships. Create logger hierarchies with multiple levels and verify that level changes propagate to children, context inheritance works correctly, and handler collection follows the expected traversal order.

| Hierarchy Pattern | Level Configuration | Expected Behavior | Edge Case |
|------------------|-------------------|------------------|-----------|
| Root → App → Module | Root: INFO, App: DEBUG, Module: unset | Module effective level = DEBUG | Level inheritance through chain |
| Root → Service1/Service2 | Root: WARN, Services: unset | Both services inherit WARN | Sibling isolation |
| Deep Nesting (5 levels) | Alternating DEBUG/INFO levels | Correct effective level calculation | Performance of level resolution |
| Handler Propagation | Handlers at root and middle levels | Records reach all applicable handlers | No duplicate output |

⚠️ **Pitfall: Incomplete Thread Safety Testing**
Many developers test thread safety by running multi-threaded code and checking for obvious crashes, but this misses subtle race conditions that only appear under specific timing conditions. Race conditions in logging systems can cause corrupted log output, lost log records, or deadlocks that freeze the entire application. Instead of just running concurrent code, use tools like Python's `threading.Barrier` to synchronize thread execution and force specific interleavings. Test with thread counts that exceed CPU core counts to verify behavior under thread starvation.

### Integration Testing Scenarios

Integration testing for structured logging systems must verify that component interactions work correctly under realistic production conditions. Unlike unit tests that isolate individual components, integration tests exercise complete workflows that span multiple subsystems and expose emergent behaviors that only appear when components interact.

**Multi-threaded logging integration** represents one of the most critical test scenarios since logging systems must handle concurrent access from web server request handlers, background processing tasks, and periodic maintenance jobs simultaneously. The integration test creates a realistic multi-threaded environment that simulates actual application patterns rather than artificial concurrent access.

The test scenario establishes multiple thread pools representing different application subsystems: web request handlers generating high-frequency INFO and DEBUG messages with request context, background task processors generating periodic WARN and ERROR messages with job context, and system monitoring threads generating FATAL messages during simulated failures. Each thread type uses different loggers in the hierarchy and attaches different context fields to verify that the system correctly isolates and routes messages.

| Thread Type | Message Frequency | Log Levels | Context Fields | Handler Destinations |
|-------------|------------------|------------|----------------|-------------------|
| Web Requests | 100 messages/second | DEBUG, INFO | request_id, user_id, endpoint | stdout, file, metrics |
| Background Tasks | 10 messages/second | INFO, WARN | job_id, task_type, duration | file, remote collector |
| System Monitor | 1 message/second | ERROR, FATAL | component, health_status | file, alerting system |
| Maintenance Jobs | 0.1 messages/second | DEBUG, INFO | job_name, resource_usage | file only |

The integration test runs for a sustained period (60 seconds minimum) while monitoring for several failure modes. Output file corruption indicates insufficient locking around file writes. Missing log records suggest race conditions in handler dispatch. Incorrect context propagation shows thread-local storage issues. Memory leaks indicate improper cleanup of thread-specific context storage.

Verification requires parsing output files to confirm that all expected log records were written correctly, context fields are properly isolated between threads, timestamps show realistic ordering within reasonable bounds, and no partial JSON records appear in the output. Memory monitoring should show stable memory usage without continuous growth that would indicate context leaks.

**Async context preservation integration** tests the most complex aspect of modern logging systems: maintaining request context across asynchronous task boundaries. Python's asyncio, JavaScript's Promise chains, and similar async frameworks create execution contexts that don't follow traditional thread-local storage patterns. The logging system must preserve context when tasks are created, paused, resumed, and completed.

The integration test creates a realistic async web application scenario with nested async operations that simulate database queries, external API calls, and background processing. Each operation should inherit context from its parent and add its own contextual information without affecting sibling operations or the parent context.

```
Request Handler (correlation_id=123, user_id=456)
├── Database Query (operation=user_lookup, table=users)
├── External API Call (service=auth, endpoint=/validate)
└── Background Task (task=audit_log, priority=low)
    ├── Database Write (operation=insert, table=audit)
    └── Cache Update (operation=invalidate, key=user:456)
```

Each async operation writes log messages at different points in its execution lifecycle: before starting the operation, during processing, and after completion. The test verifies that context propagation works correctly across await boundaries, context isolation prevents interference between concurrent tasks, nested operations inherit parent context correctly, and context cleanup occurs when tasks complete or fail.

| Async Pattern | Context Behavior | Test Verification | Common Failure |
|---------------|------------------|------------------|----------------|
| Sequential await | Context preserved across each await | All messages contain full context chain | Context lost after first await |
| Concurrent tasks | Independent context for each task | No context bleeding between tasks | Shared mutable context state |
| Nested async calls | Child inherits parent + adds own fields | Proper context hierarchy in logs | Child modifications affect parent |
| Exception handling | Context preserved through exception unwinding | Error logs contain full request context | Context cleared on exception |

**End-to-end request tracing integration** demonstrates the complete correlation ID lifecycle from request ingress through all internal operations to final response. This test simulates a realistic microservice interaction where a user request triggers operations across multiple internal services, each adding their own contextual information while preserving the original correlation ID.

The test creates a mock web application with request middleware that extracts or generates correlation IDs, establishes request-scoped context, and ensures cleanup when requests complete. Multiple request handlers simulate different application endpoints with varying complexity: simple operations that complete synchronously, complex operations that spawn multiple async tasks, and error scenarios that require proper context preservation during exception handling.

| Request Type | Operations Triggered | Expected Context Fields | Correlation Tracking |
|--------------|-------------------|----------------------|-------------------|
| User Login | Auth service, user lookup, session creation | correlation_id, user_id, client_ip, user_agent | Single ID through all operations |
| Data Query | Query parser, database, result formatter | correlation_id, query_id, table, execution_time | Nested operation IDs |
| File Upload | Validation, storage, thumbnail generation | correlation_id, file_id, size, processing_status | Background task correlation |
| Error Scenario | Failed validation, error logging, cleanup | correlation_id, error_code, failed_operation | Error context preservation |

The integration test sends concurrent requests with different correlation ID patterns: requests with existing correlation IDs from upstream services, requests without correlation IDs that require generation, and requests with malformed correlation IDs that need sanitization. The test verifies that correlation IDs are properly preserved across all async operations, each request maintains independent context isolation, background tasks spawned by requests maintain correlation linkage, and error scenarios preserve context for debugging.

Verification involves parsing log output to construct correlation traces showing the complete lifecycle of each request. Each correlation ID should appear in log records from request start to completion, with proper nesting of operation-specific context and no correlation ID pollution between concurrent requests.

⚠️ **Pitfall: Unrealistic Test Conditions**
Many integration tests use artificial scenarios that don't reflect production behavior, such as perfect network conditions, unlimited memory, and immediate I/O operations. Real production environments have variable latency, intermittent failures, and resource constraints that can expose bugs not visible in idealized test conditions. Include realistic failure injection in integration tests: introduce random network delays for remote handlers, simulate disk pressure that slows file I/O, and test under memory pressure that triggers garbage collection pauses.

### Milestone Checkpoints

Milestone checkpoints provide concrete verification criteria for each stage of system implementation, ensuring that learners build foundational capabilities correctly before advancing to more complex features. Each checkpoint includes specific behavioral requirements, observable outputs, and diagnostic techniques for identifying common implementation errors.

**Milestone 1 Checkpoint: Logger Core**
The Logger Core milestone establishes the foundational logging infrastructure with proper level filtering, thread safety, and handler dispatch. The checkpoint verification ensures that these core capabilities work correctly under both normal operations and stress conditions.

The primary verification test creates multiple logger instances in a hierarchy and verifies that level filtering works correctly at each node. Create loggers for `root`, `app`, `app.database`, and `app.api` with different level configurations. Set the root logger to INFO, leave app unset (inheriting INFO), set app.database to DEBUG, and set app.api to WARN. Send messages at all log levels to each logger and verify that only appropriate messages appear in the output.

| Logger Name | Configured Level | Effective Level | DEBUG Visible | INFO Visible | WARN Visible |
|-------------|------------------|----------------|---------------|--------------|--------------|
| root | INFO | INFO | No | Yes | Yes |
| app | (unset) | INFO (inherited) | No | Yes | Yes |
| app.database | DEBUG | DEBUG | Yes | Yes | Yes |
| app.api | WARN | WARN | No | No | Yes |

Thread safety verification requires running the logging system under concurrent load while monitoring for output corruption. Create 10 threads that simultaneously write 1000 log messages each to the same logger instance. Each thread should use a unique thread identifier in its log messages. After completion, parse the output file and verify that all 10,000 messages are present, no partial messages appear (indicating corruption during writes), messages from different threads don't have interleaved content within single records, and thread identifiers are correctly preserved in each message.

Handler dispatch verification tests that log records reach multiple output destinations correctly and that handler failures don't prevent delivery to other handlers. Configure a logger with three handlers: stdout, file, and a mock network handler that can be programmatically failed. Write log messages while the mock handler is healthy, then trigger a failure in the mock handler and continue writing messages. Verify that messages continue to reach stdout and file handlers despite the network handler failure.

**Expected Behavior After Milestone 1:**
- Logger creation: `get_logger("app.module")` creates hierarchy automatically
- Level filtering: Only messages at or above configured level appear in output
- Thread safety: No output corruption under concurrent access from multiple threads
- Handler dispatch: Messages reach all configured handlers, with graceful failure handling
- Runtime reconfiguration: Level changes take effect immediately without restart

**Diagnostic Commands for Milestone 1:**
```bash
# Test basic functionality
python -c "
import logging_system
logger = logging_system.get_logger('test')
logger.set_level(logging_system.INFO)
logger.debug('This should not appear')
logger.info('This should appear')
"

# Test thread safety
python test_thread_safety.py --threads=10 --messages=1000

# Test handler dispatch
python test_handlers.py --with-failure-simulation
```

**Milestone 2 Checkpoint: Structured Output**
The Structured Output milestone adds JSON formatting, timestamp handling, and custom formatter support. The checkpoint verification focuses on output format correctness and formatter extensibility.

JSON format verification requires testing serialization behavior with complex data structures that commonly cause problems in production. Create log records with context containing nested dictionaries, arrays, datetime objects, and custom class instances. Verify that the JSON output is valid single-line JSON (no embedded newlines), contains all expected fields in consistent order, handles non-serializable objects gracefully with string representations, and includes properly formatted timestamps.

| Context Data Type | Serialization Behavior | Expected JSON Output |
|------------------|----------------------|-------------------|
| Nested Dict (3 levels) | Full serialization | Complete nested structure |
| Datetime Object | ISO 8601 conversion | "2024-01-15T10:30:45.123Z" |
| Custom Class Instance | String representation | "MyClass(id=123, name='test')" |
| Circular Reference | Truncation with warning | "[Circular Reference Detected]" |
| Large Array (1000 items) | Size-based truncation | First N items + "[...truncated]" |

Timestamp format verification tests that timestamps are generated correctly for different precision requirements and timezone configurations. Configure the timestamp formatter for ISO 8601 format and verify that all log records contain valid timestamps that are properly ordered chronologically. Test timezone handling by configuring UTC output and verifying that timestamps are consistent regardless of local system timezone.

Custom formatter verification tests the plugin system by registering a test formatter that produces CSV output instead of JSON. Create a custom formatter that outputs log records in comma-separated format with quoted fields. Register the formatter with the system and configure a logger to use it. Verify that log messages are formatted correctly in CSV format and that multiple formatters can coexist without interference.

**Expected Behavior After Milestone 2:**
- JSON output: Valid single-line JSON for every log record
- Timestamp formatting: Consistent ISO 8601 timestamps with millisecond precision
- Custom formatters: Ability to register and use custom output formats
- Pretty printing: Human-readable colored output for development console
- Serialization safety: Graceful handling of non-serializable objects

**Milestone 3 Checkpoint: Context & Correlation**
The Context & Correlation milestone adds request tracing, correlation ID generation, and context propagation across async boundaries. The checkpoint verification ensures that context is properly maintained throughout complex execution flows.

Correlation ID verification tests that unique identifiers are generated for each request and properly propagated through all operations. Create a mock request handler that generates a correlation ID and performs multiple nested operations: database query, external API call, and background task. Verify that all log records contain the same correlation ID, nested operations inherit the correlation ID correctly, concurrent requests have independent correlation IDs, and correlation IDs follow the expected format (service prefix + timestamp + random component).

| Operation Type | Correlation ID Presence | Context Inheritance | Isolation Verification |
|----------------|----------------------|-------------------|----------------------|
| Request Start | New correlation ID generated | Initial request context | Independent from other requests |
| Database Query | Same correlation ID | Request context + query metadata | No interference between queries |
| API Call | Same correlation ID | Request context + API metadata | Async operation isolation |
| Background Task | Same correlation ID | Request context + task metadata | Proper async context bridge |

Context propagation verification tests that context fields are properly inherited and isolated across different scopes. Create nested context scopes where each level adds additional fields while preserving parent context. Verify that child contexts inherit all parent fields, child context modifications don't affect parent contexts, context cleanup occurs when scopes exit, and memory usage remains stable during context creation/cleanup cycles.

Async context preservation verification tests the most complex scenario: maintaining context across asyncio task boundaries. Create an async request handler that spawns multiple concurrent tasks, each performing operations that add context fields. Verify that each async task maintains its own context isolation, context is properly inherited when tasks are created, await boundaries preserve context correctly, and context cleanup occurs when async tasks complete.

**Expected Behavior After Milestone 3:**
- Correlation ID generation: Unique IDs for each request with proper format
- Context propagation: Seamless context inheritance through nested function calls
- Async context preservation: Correct context maintenance across await boundaries
- Request middleware: Automatic context establishment for incoming requests
- Memory management: No context leaks during long-running operations

**Integration Verification Commands:**
```bash
# Test complete request flow
curl -H "X-Correlation-ID: test-123" http://localhost:8080/api/test
# Verify logs show correlation ID throughout request processing

# Test async context preservation
python test_async_context.py --concurrent-requests=50

# Test context isolation
python test_context_isolation.py --nested-levels=5

# Stress test complete system
python stress_test.py --duration=300 --rps=100
```

The milestone checkpoints provide a systematic progression that ensures each capability is solid before building additional complexity. This approach prevents the common problem of advanced features failing due to inadequate foundations, and provides clear diagnostic criteria when troubleshooting implementation issues.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Unit Testing Framework | `unittest` (Python standard library) | `pytest` with fixtures and parameterization |
| Thread Safety Testing | `threading.Thread` with manual synchronization | `concurrent.futures` with ThreadPoolExecutor |
| Async Testing | `asyncio.run()` with manual event loop | `pytest-asyncio` with async test fixtures |
| Mock Objects | `unittest.mock` for handler simulation | `responses` library for HTTP handler mocking |
| Performance Testing | Manual timing with `time.time()` | `cProfile` and `memory_profiler` for detailed analysis |
| Output Validation | String parsing with `json.loads()` | JSON Schema validation with `jsonschema` |
| Concurrency Testing | Manual thread creation | Property-based testing with `hypothesis` |

**B. Recommended File/Module Structure:**

```
structured-logging/
  src/logging_system/           ← main package
    __init__.py
    logger.py                   ← Logger, LoggerRegistry
    handlers.py                 ← Handler implementations
    formatters.py              ← JSON/Pretty formatters
    context.py                 ← Context management
  tests/                       ← test suite
    unit/                      ← component isolation tests
      test_logger.py           ← Logger hierarchy tests
      test_handlers.py         ← Handler failure/recovery tests
      test_formatters.py       ← Serialization edge cases
      test_context.py          ← Context isolation tests
    integration/               ← cross-component tests
      test_threading.py        ← Multi-threaded scenarios
      test_async.py           ← Async context preservation
      test_end_to_end.py      ← Complete request tracing
    checkpoints/              ← milestone verification
      test_milestone1.py       ← Logger core verification
      test_milestone2.py       ← Structured output verification  
      test_milestone3.py       ← Context correlation verification
    fixtures/                ← shared test utilities
      mock_handlers.py         ← Controllable handler mocks
      context_helpers.py       ← Context setup utilities
      data_generators.py       ← Test data creation
```

**C. Infrastructure Starter Code:**

```python
# tests/fixtures/mock_handlers.py
"""Mock handlers for controlled failure testing."""
import threading
import time
from typing import List, Optional
from logging_system.handlers import BaseHandler
from logging_system.data_model import LogRecord

class ControllableHandler(BaseHandler):
    """Handler with programmable failure behavior for testing."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.records: List[LogRecord] = []
        self.should_fail = False
        self.failure_delay = 0.0
        self.call_count = 0
        self._lock = threading.RLock()
    
    def _write_record(self, record: LogRecord) -> None:
        with self._lock:
            self.call_count += 1
            if self.should_fail:
                if self.failure_delay > 0:
                    time.sleep(self.failure_delay)
                raise IOError(f"Simulated failure in {self.name}")
            self.records.append(record)
    
    def simulate_failure(self, should_fail: bool, delay: float = 0.0):
        """Control handler failure behavior for testing."""
        with self._lock:
            self.should_fail = should_fail
            self.failure_delay = delay
    
    def get_records(self) -> List[LogRecord]:
        """Thread-safe access to captured records."""
        with self._lock:
            return self.records.copy()
    
    def clear_records(self):
        """Reset captured records for next test."""
        with self._lock:
            self.records.clear()
            self.call_count = 0

# tests/fixtures/data_generators.py
"""Test data generators for edge case testing."""
import datetime
import uuid
from typing import Any, Dict, List

class CircularReferenceObject:
    """Object that references itself for serialization testing."""
    def __init__(self, name: str):
        self.name = name
        self.self_ref = self
        self.nested = {'circular': self}

class NonSerializableObject:
    """Custom object that cannot be JSON serialized."""
    def __init__(self, value: Any):
        self.value = value
        self.function = lambda x: x * 2
    
    def __repr__(self):
        return f"NonSerializableObject({self.value})"

def generate_complex_context() -> Dict[str, Any]:
    """Generate context with various data types for testing."""
    return {
        'string_field': 'test_value',
        'numeric_field': 42,
        'datetime_field': datetime.datetime.now(),
        'nested_dict': {
            'level1': {'level2': {'level3': 'deep_value'}},
            'array': [1, 2, 3, 'mixed', True]
        },
        'circular_ref': CircularReferenceObject('test'),
        'non_serializable': NonSerializableObject(123),
        'large_array': list(range(1000)),
        'unicode_data': 'Special chars: 🚀 💡 ∑ ∆'
    }

def generate_correlation_id() -> str:
    """Generate test correlation ID with realistic format."""
    return f"test-{uuid.uuid4().hex[:8]}-{int(time.time())}"
```

**D. Core Logic Skeleton Code:**

```python
# tests/unit/test_logger.py
"""Unit tests for Logger hierarchy and level filtering."""
import unittest
import threading
import time
from logging_system import get_logger, LogLevel
from tests.fixtures.mock_handlers import ControllableHandler

class TestLoggerCore(unittest.TestCase):
    """Test Logger hierarchy, level filtering, and thread safety."""
    
    def test_level_filtering(self):
        """Verify that only messages at or above configured level are processed."""
        # TODO 1: Create logger with INFO level
        # TODO 2: Attach mock handler to capture output
        # TODO 3: Send DEBUG message - verify not processed
        # TODO 4: Send INFO message - verify processed
        # TODO 5: Send ERROR message - verify processed
        # TODO 6: Change level to DEBUG - verify DEBUG now processed
        pass
    
    def test_logger_hierarchy_inheritance(self):
        """Test that child loggers inherit configuration from parents."""
        # TODO 1: Create root logger with WARN level
        # TODO 2: Create child logger 'app.database' - verify inherits WARN
        # TODO 3: Set child to DEBUG - verify uses DEBUG, not inherited WARN
        # TODO 4: Clear child level - verify inherits WARN again
        # TODO 5: Test deep hierarchy: root.app.module.component
        pass
    
    def test_thread_safety_concurrent_logging(self):
        """Verify no output corruption under concurrent access."""
        logger = get_logger('thread_test')
        handler = ControllableHandler('test')
        logger.add_handler(handler)
        
        def worker_thread(thread_id: int, message_count: int):
            # TODO 1: Log message_count messages from this thread
            # TODO 2: Include thread_id in each message for verification
            # TODO 3: Use different log levels randomly
            # TODO 4: Add context fields unique to this thread
            pass
        
        # TODO 5: Start 10 worker threads with 100 messages each
        # TODO 6: Wait for all threads to complete
        # TODO 7: Verify all 1000 messages captured correctly
        # TODO 8: Verify no message corruption (partial records)
        # TODO 9: Verify thread isolation (no mixed content)
        pass

# tests/integration/test_async_context.py
"""Integration tests for async context preservation."""
import asyncio
import unittest
from logging_system import get_logger, LoggingContext
from tests.fixtures.mock_handlers import ControllableHandler

class TestAsyncContextPreservation(unittest.TestCase):
    """Test context preservation across async boundaries."""
    
    async def test_async_context_inheritance(self):
        """Verify context preserved across await boundaries."""
        logger = get_logger('async_test')
        handler = ControllableHandler('async')
        logger.add_handler(handler)
        
        async def database_operation(user_id: int):
            # TODO 1: Add database-specific context fields
            # TODO 2: Log start of database operation
            # TODO 3: Simulate async database call with asyncio.sleep()
            # TODO 4: Log completion with query results
            # TODO 5: Return operation results
            pass
        
        async def api_request_handler(request_id: str, user_id: int):
            # TODO 6: Set initial request context (correlation_id, user_id)
            # TODO 7: Log request start
            # TODO 8: Call database_operation - verify context preserved
            # TODO 9: Log request completion with timing
            pass
        
        # TODO 10: Run multiple concurrent requests
        # TODO 11: Verify each request has independent context
        # TODO 12: Verify nested operations inherit parent context
        # TODO 13: Verify no context bleeding between requests
        pass
    
    def test_concurrent_async_requests(self):
        """Test context isolation under concurrent async load."""
        # TODO 1: Create async request handlers with different correlation IDs
        # TODO 2: Run 50 concurrent requests with unique context
        # TODO 3: Each request spawns multiple async subtasks
        # TODO 4: Verify correlation IDs properly isolated
        # TODO 5: Verify no context pollution between requests
        pass
```

**E. Language-Specific Hints:**

- **Thread Safety**: Use `threading.RLock` for components that need recursive locking (loggers calling other loggers). Use `threading.Lock` for simple mutual exclusion in counters and caches.
- **Async Context**: Use `contextvars.ContextVar` for async-local storage that preserves across await boundaries. Combine with `threading.local` for thread-local fallback.
- **JSON Serialization**: Override `json.JSONEncoder.default()` method to handle non-serializable objects. Use `json.dumps(separators=(',', ':'))` for compact single-line output.
- **File I/O Testing**: Use `tempfile.NamedTemporaryFile()` for isolated file handler testing. Call `file.flush()` and `os.fsync()` to ensure data written before verification.
- **Memory Testing**: Use `tracemalloc.start()` to monitor memory usage during context lifecycle tests. Check for linear growth that indicates memory leaks.
- **Time Mocking**: Use `unittest.mock.patch('time.time')` to control timestamp generation for deterministic testing.

**F. Milestone Checkpoints:**

**Checkpoint 1 - Logger Core Verification:**
```bash
# Run core logger tests
python -m pytest tests/unit/test_logger.py -v

# Expected output shows:
# - test_level_filtering PASSED
# - test_logger_hierarchy_inheritance PASSED  
# - test_thread_safety_concurrent_logging PASSED
# - No output corruption warnings
# - All 1000 test messages captured correctly

# Manual verification:
python -c "
import logging_system
logger = logging_system.get_logger('manual_test')
logger.set_level(logging_system.INFO)
logger.debug('Should not see this')
logger.info('Should see this')
"
# Output should show only INFO message
```

**Checkpoint 2 - Structured Output Verification:**
```bash
# Test JSON formatting
python -m pytest tests/unit/test_formatters.py -v

# Verify JSON output format:
python test_json_output.py
# Should produce valid single-line JSON like:
# {"timestamp":"2024-01-15T10:30:45.123Z","level":20,"message":"test","logger_name":"test","context":{"key":"value"}}

# Test custom formatter registration:
python test_custom_formatter.py
# Should show CSV output: 2024-01-15T10:30:45.123Z,INFO,test,test,"key=value"
```

**Checkpoint 3 - Context & Correlation Verification:**
```bash
# Test complete context flow
python -m pytest tests/integration/test_end_to_end.py -v

# Test request tracing:
python simulate_request.py --correlation-id=test-123
# All log output should contain: "correlation_id":"test-123"

# Test async context preservation:
python test_async_flow.py --concurrent-requests=10
# Each request should maintain independent context throughout async operations
```

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Thread safety test fails randomly | Race condition in handler dispatch | Add debug logging around lock acquisition | Use RLock instead of Lock for recursive calls |
| Context not preserved across await | Missing contextvars setup | Check if ContextVar is properly copied | Use contextvars.copy_context() for task creation |
| JSON output contains newlines | Embedded newlines in log messages | Search for \n characters in test output | Strip newlines from message content before formatting |
| Memory usage grows during tests | Context cleanup not called | Monitor context storage size during test | Add explicit cleanup in test teardown |
| Async tests hang indefinitely | Deadlock in async context manager | Check for blocking operations in async code | Replace time.sleep() with asyncio.sleep() |
| Handler dispatch fails silently | Exception swallowed in handler | Add try/catch logging around handler calls | Check handler circuit breaker state |
| Correlation IDs not unique | Timestamp resolution too low | Print generated IDs to check for duplicates | Add random component to ID generation |
| Context bleeds between threads | Shared mutable context objects | Verify each thread gets independent context | Use copy.deepcopy() when inheriting context |


## Debugging Guide

> **Milestone(s):** This section provides essential debugging patterns for all three milestones, with race condition debugging supporting Milestone 1 (Logger Core), serialization issue diagnosis supporting Milestone 2 (Structured Output), and context propagation debugging supporting Milestone 3 (Context & Correlation).

Think of debugging a structured logging system like troubleshooting a complex telephone switchboard. When calls don't connect properly, the problem could be in the routing logic (level filtering), the switching mechanism (handler dispatch), the line quality (serialization), or the directory service (context propagation). Each type of failure has distinct symptoms, and skilled operators develop systematic approaches to isolate and resolve issues quickly.

The complexity of a production logging system creates multiple failure modes that can interact in subtle ways. A race condition in the logger hierarchy might only manifest under high concurrency, while context propagation issues may only appear in specific async execution patterns. Memory leaks can accumulate slowly until they cause cascading failures across the entire application. This debugging guide provides systematic approaches to identify, isolate, and resolve the most common issues learners encounter when implementing structured logging systems.

### Common Implementation Pitfalls

The most frequent implementation mistakes in structured logging systems fall into four categories: concurrency issues that corrupt shared state, context management problems that lose request correlation, blocking operations that degrade performance, and resource management failures that cause memory leaks. Each category requires different debugging approaches and prevention strategies.

#### Race Conditions in Logger Hierarchy

Race conditions in logger hierarchy management represent one of the most subtle and dangerous pitfalls in structured logging implementation. The logger hierarchy is a shared data structure accessed concurrently by multiple threads, and improper synchronization can lead to corrupted state, lost configuration changes, or inconsistent behavior that only manifests under high load.

⚠️ **Pitfall: Unsynchronized Logger Registry Access**

The most common race condition occurs when multiple threads simultaneously access the `LoggerRegistry._loggers` dictionary without proper locking. Consider a scenario where Thread A calls `get_logger("com.example.service")` while Thread B calls `get_logger("com.example.service.auth")`. Both threads may attempt to create parent loggers simultaneously, leading to duplicate entries, lost references, or partially initialized logger objects.

The symptom appears as intermittent failures where loggers seem to lose their configuration, handlers disappear randomly, or context fields vanish from log records. These issues are particularly insidious because they may not appear during development with single-threaded access patterns but emerge under production load.

The underlying problem is that dictionary operations in most languages are not atomic across multiple operations. Creating a logger hierarchy involves: checking if parent loggers exist, creating missing parents, establishing parent-child relationships, and updating the registry. Without proper synchronization, these operations can interleave in ways that corrupt the data structure.

| Race Condition Scenario | Thread A Action | Thread B Action | Corruption Result |
|-------------------------|-----------------|-----------------|-------------------|
| Simultaneous parent creation | Creates `com.example` logger | Creates `com.example` logger | Duplicate parent with different configurations |
| Hierarchy modification | Sets `com.example.service` level | Adds handler to `com.example.service` | Handler added to wrong logger instance |
| Context inheritance | Modifies parent context | Reads child effective context | Child sees inconsistent parent context |
| Level propagation | Changes parent level | Calculates child effective level | Child uses stale level cache |

The correct solution requires using a `RLock` (reader-writer lock) in the `LoggerRegistry` to protect all registry operations. Read operations like logger lookup can share the lock, while write operations like logger creation require exclusive access. Additionally, each individual `Logger` needs its own lock to protect configuration changes and context modifications.

⚠️ **Pitfall: Level Cache Invalidation Race**

Another subtle race condition occurs in the effective level calculation system. The `Logger._effective_level` cache improves performance by avoiding repeated hierarchy traversal, but invalidating this cache safely across multiple threads requires careful coordination.

When a parent logger's level changes, all descendant loggers must invalidate their cached effective levels. If this invalidation is not synchronized properly, a logger might use a stale cached level while its parent's configuration has changed, leading to incorrect level filtering behavior.

The symptom appears as messages that should be filtered continuing to appear, or messages that should be logged being incorrectly suppressed. The issue is timing-dependent and may only occur during configuration changes under load.

⚠️ **Pitfall: Handler List Modification During Iteration**

A particularly dangerous race condition occurs when one thread modifies a logger's handler list while another thread iterates over that list for message dispatch. This can cause handler dispatch to skip handlers, duplicate messages, or access deallocated handler objects leading to crashes.

The iteration happens in the `log()` method when collecting handlers from the logger hierarchy, while modification occurs when configuration changes add or remove handlers. Without proper synchronization, the iterator can become invalid mid-traversal.

#### Context Propagation Failures

Context propagation failures are among the most frustrating debugging challenges because they break the fundamental promise of structured logging: that related log entries can be correlated across service boundaries. When context propagation fails, correlation IDs disappear, request metadata is lost, and distributed tracing becomes impossible.

⚠️ **Pitfall: Thread-Local Context Not Propagating to New Threads**

The most common context propagation failure occurs when application code creates new threads without explicitly propagating the logging context. Thread-local storage isolates context between threads by design, so newly created threads start with empty context even if the parent thread has rich contextual information.

This manifests as correlation IDs and request metadata suddenly disappearing from log entries generated by background tasks, worker threads, or async operations. The symptoms are subtle because the logging system continues to function correctly—it simply loses the contextual information that makes logs useful for debugging distributed systems.

Consider a web request handler that spawns a background task to process uploaded data. The main request thread has correlation ID `req-12345` and user context `user_id: alice`, but the background thread starts with empty context. All log entries from the background processing lose this critical correlation information.

| Context Propagation Scenario | Parent Thread Context | Child Thread Context | Impact |
|------------------------------|----------------------|---------------------|---------|
| Background task creation | `{"correlation_id": "req-12345", "user_id": "alice"}` | `{}` (empty) | Background processing logs cannot be correlated to original request |
| Thread pool execution | `{"request_id": "order-456", "operation": "checkout"}` | Previous task's context | Worker thread sees stale context from previous task |
| Async task spawning | `{"trace_id": "abc123", "span_id": "def456"}` | `{}` (empty) | Distributed tracing chain broken |

The solution requires explicitly capturing context from the parent thread and setting it in the child thread before any logging occurs. This can be automated through context-aware thread creation utilities or manual context transfer at task boundaries.

⚠️ **Pitfall: Async Context Not Preserved Across Await Boundaries**

In async programming environments, context can be lost across `await` boundaries when the execution context switches between different async tasks or coroutines. This is particularly problematic in languages with cooperative multitasking where multiple async tasks share the same thread.

The symptom appears as correlation IDs and context fields randomly changing or disappearing in the middle of async function execution. A function might start with correlation ID `req-789` but after awaiting a database operation, find itself with a different correlation ID from another concurrent request.

This occurs because async runtimes may resume awaited tasks on different logical execution contexts, and the logging context storage mechanism fails to properly bridge these context switches. The issue is intermittent and load-dependent, making it particularly difficult to diagnose.

⚠️ **Pitfall: Context Memory Leaks in Long-Running Services**

Context storage systems that don't properly clean up unused contexts can accumulate memory over time, eventually causing out-of-memory errors in long-running services. This typically occurs when context storage uses strong references to context objects that should be garbage collected after request completion.

The symptom starts as gradually increasing memory usage that doesn't correspond to application load. Over days or weeks, memory consumption grows until the service becomes unstable. Memory profiling reveals large numbers of abandoned context objects still referenced by the logging system.

#### Blocking I/O in Log Handler Hot Path

Blocking I/O operations in the logging hot path represent a critical performance and reliability pitfall that can bring entire applications to a standstill. When log handlers perform synchronous file writes or network operations directly in the application's main execution thread, they introduce latency spikes and potential deadlocks that cascade through the entire system.

⚠️ **Pitfall: Synchronous File Writes Without Buffering**

The most common blocking I/O mistake is implementing file handlers that perform synchronous writes for every log record. When application code calls `logger.info("Processing user request")`, the thread blocks until the file write completes, including potential disk seeks, buffer flushes, and filesystem metadata updates.

Under normal conditions, this might add only microseconds of latency per log statement. However, during disk contention, filesystem cache pressure, or storage system degradation, each log write can block for milliseconds or seconds. In high-throughput applications generating thousands of log entries per second, this blocking behavior creates a performance cliff where logging overhead overwhelms application processing.

The symptom appears as application response times that correlate with logging volume rather than business logic complexity. Performance profiling reveals significant time spent in file system calls within logging code paths. Applications may appear to "freeze" during bursts of log activity.

| Blocking I/O Scenario | Normal Latency | Degraded Latency | Application Impact |
|----------------------|----------------|------------------|-------------------|
| Single log statement per request | 50μs | 10ms | 200x slowdown in request processing |
| Burst logging during error conditions | 1ms total | 500ms total | Request timeouts and cascading failures |
| High-frequency debug logging | 10ms/second overhead | 5s/second overhead | Complete application stall |

⚠️ **Pitfall: Network Handler Timeouts Without Circuit Breakers**

Network-based log handlers that send records to remote aggregation systems introduce even more severe blocking risks. Network operations can timeout, experience packet loss, or encounter service unavailability. Without proper timeout configuration and circuit breaker patterns, application threads can block indefinitely waiting for network log delivery.

This pitfall is particularly dangerous because it couples application availability to logging infrastructure availability. If the remote log aggregation service becomes slow or unavailable, the entire application performance degrades proportionally. In the worst case, all application threads become blocked waiting for log delivery, creating a complete service outage caused by logging infrastructure failure.

⚠️ **Pitfall: Lock Contention in Shared File Handles**

Even when file I/O is buffered and optimized, shared file handles can create lock contention bottlenecks. If multiple threads write to the same log file through a single `FileHandler` instance, the handler must serialize writes to maintain log record integrity. Under high concurrency, threads queue up waiting for file write access, creating a serialization bottleneck.

#### Memory Leaks in Context Management

Memory leaks in context management systems create insidious performance degradation that compounds over time. Unlike immediate failures that are quickly detected and resolved, memory leaks accumulate slowly and may not manifest symptoms until applications have been running for days or weeks under production load.

⚠️ **Pitfall: Context References in Thread-Local Storage**

The most common memory leak occurs when thread-local context storage maintains strong references to context objects beyond their useful lifetime. In long-running applications with thread pools, worker threads may be reused across many requests without properly clearing their thread-local context state.

Each request adds context fields like correlation IDs, user information, and request metadata to the thread-local storage. If this context is not explicitly cleared at request completion, the thread retains references to all context objects from every request it has ever processed. In a busy web server, a single worker thread might accumulate context from thousands of requests over its lifetime.

The symptom appears as steadily increasing memory usage in long-running services, with memory profiling revealing large numbers of abandoned context objects still reachable through thread-local storage. The leak rate correlates with request volume, and memory usage never decreases even during idle periods.

⚠️ **Pitfall: Circular References in Context Inheritance**

Context inheritance systems that allow arbitrary object nesting can inadvertently create circular references that prevent garbage collection. This typically occurs when context objects contain references to application objects that themselves hold logging context references.

For example, a request context might contain a reference to the current user object for logging purposes. If that user object maintains a reference back to the request context (perhaps through an embedded logger), a circular reference prevents both objects from being garbage collected even after the request completes.

⚠️ **Pitfall: Unbounded Context Field Accumulation**

Context systems that allow unlimited field accumulation without size limits can experience memory leaks when application code repeatedly adds context fields without bounds checking. This often occurs in long-running background tasks that add diagnostic information to context throughout their execution lifecycle.

A data processing job might add context fields for each processed record: `processing_record_1`, `processing_record_2`, etc. Over hours of execution, the context object grows to contain thousands of fields, consuming significant memory and degrading context access performance.

### Debugging Techniques

Effective debugging of structured logging systems requires systematic approaches that can isolate issues across the multiple layers of abstraction involved in log processing. Unlike simple debugging scenarios with clear cause-and-effect relationships, logging system bugs often involve subtle interactions between concurrent operations, context propagation mechanisms, and external I/O systems.

#### Adding Trace Logging for Internal Operations

Trace logging provides visibility into the internal operation of the logging system itself, creating a meta-logging capability that reveals how log records flow through the system architecture. This technique is particularly valuable for diagnosing issues in handler dispatch, level filtering, and context propagation where the symptoms are indirect effects of internal system state.

The key insight behind trace logging is creating a separate, simpler logging channel that operates independently of the main logging system being debugged. This trace system should be minimal and robust, avoiding the complex features that might themselves be the source of bugs in the main system.

**Internal Logger State Tracing**

Implementing trace logging for logger state changes reveals how the logger hierarchy evolves during application execution. This is particularly valuable for diagnosing race conditions in logger creation and configuration that only manifest under specific timing conditions.

The trace system should log every significant state change with enough detail to reconstruct the sequence of operations that led to the current state. Key events to trace include logger creation, parent-child relationship establishment, level changes, handler additions and removals, and context modifications.

| Trace Event Type | Information Logged | Debugging Value |
|------------------|-------------------|-----------------|
| Logger creation | Logger name, parent name, thread ID, timestamp | Reveals duplicate logger creation and hierarchy building order |
| Level changes | Logger name, old level, new level, effective level recalculation | Shows level inheritance and cache invalidation behavior |
| Handler operations | Handler add/remove, logger name, handler type, thread ID | Reveals handler configuration race conditions |
| Context modifications | Context field changes, inheritance relationships, thread boundaries | Shows context propagation and cleanup behavior |

The trace output should include thread identifiers and high-precision timestamps to reveal concurrency issues. For example, trace logs might show two threads simultaneously creating the same parent logger, or level changes that don't properly invalidate descendant logger caches.

**Handler Dispatch Flow Tracing**

Handler dispatch tracing reveals how log records flow through the handler selection and output generation process. This is essential for debugging issues where log records are lost, duplicated, or sent to incorrect destinations.

The trace system should log each step of the dispatch process: level filtering decisions, handler collection from the logger hierarchy, formatter invocation, and final output destination. When handlers fail, the trace logs show exactly which handlers were attempted and what errors occurred.

**Context Propagation Tracing**

Context propagation is one of the most difficult aspects of logging systems to debug because problems often manifest far from their root cause. A context field might be lost during async task creation but not discovered until much later when correlation fails during error investigation.

Effective context tracing logs every context operation: context creation, field additions, inheritance operations, thread boundary crossings, and cleanup operations. The trace includes context snapshots before and after each operation, making it possible to identify exactly where context corruption or loss occurs.

#### Inspecting Context State

Context state inspection provides real-time visibility into the current state of logging context across all active execution contexts. This technique is essential for debugging context propagation issues where the symptoms (missing correlation information) are distant from the cause (failed context inheritance).

**Context Storage Dump Utilities**

Building utilities to dump the current state of all context storage mechanisms reveals inconsistencies between thread-local and async-local context storage. These utilities should be callable from debugger sessions or triggered by special debug endpoints in web applications.

The dump output should show the complete context hierarchy for each active execution context, including inheritance relationships, field values, and storage mechanism details. Comparing context state across different storage mechanisms can reveal synchronization issues or failed context propagation.

| Context Storage View | Information Displayed | Debugging Application |
|---------------------|----------------------|----------------------|
| Thread-local contexts | Thread ID, context fields, inheritance chain | Reveals thread context isolation and cleanup issues |
| Async-local contexts | Task ID, context fields, parent task relationships | Shows async context propagation and coroutine switching |
| Global context registry | All active contexts, reference counts, cleanup status | Identifies memory leaks and orphaned contexts |
| Context inheritance tree | Parent-child relationships, field inheritance, override behavior | Reveals context inheritance logic errors |

**Context Field History Tracking**

Implementing context field history tracking maintains a record of how each context field was added, modified, or removed throughout request processing. This technique is particularly valuable for debugging cases where context fields have unexpected values or disappear mysteriously during processing.

The history tracker records not just field values but the stack trace or operation context where each change occurred. When debugging correlation failures, the history reveals exactly where correlation IDs were lost or overwritten with incorrect values.

**Context Boundary Verification**

Context boundary verification systematically checks that context propagation works correctly across all types of execution boundaries in the application. This includes thread creation, async task spawning, callback invocation, and external service calls.

The verification system can be implemented as assertions that automatically check context consistency at known boundary points, or as explicit verification calls inserted during debugging sessions. When context propagation fails, these checks identify the specific boundary where propagation broke down.

#### Testing Thread Safety

Thread safety testing for logging systems requires techniques that can reliably reproduce concurrency issues that may only manifest under specific timing conditions. Unlike functional testing where deterministic inputs produce predictable outputs, concurrency testing must deal with the inherent non-determinism of multi-threaded execution.

**Concurrent Logger Operation Testing**

Effective thread safety testing subjects the logger hierarchy to intensive concurrent operations that stress-test all synchronization mechanisms. This includes simultaneous logger creation, configuration changes, level modifications, and message logging across multiple threads.

The test framework should use barriers and coordination mechanisms to maximize the probability of race conditions occurring. Rather than simply launching multiple threads and hoping for conflicts, the tests should synchronize thread execution to create maximum contention at critical sections.

| Concurrency Test Scenario | Thread A Operations | Thread B Operations | Expected Behavior |
|---------------------------|--------------------|--------------------|-------------------|
| Simultaneous logger creation | Create logger `com.example.service` | Create logger `com.example.service.auth` | Both loggers created with correct parent-child relationship |
| Configuration race | Set logger level to DEBUG | Add handler to same logger | Both operations complete without corruption |
| Context modification | Add field `user_id: alice` to logger context | Read effective context from child logger | Child sees consistent context state |
| Handler dispatch | Log ERROR message | Remove handler from logger | Message logged to handlers present at start of operation |

**Memory Consistency Verification**

Memory consistency verification ensures that changes made by one thread are properly visible to other threads according to the memory model of the target language. This is particularly important for logging systems where configuration changes must be immediately visible to all threads performing logging operations.

The verification tests should check that level changes, handler modifications, and context updates made by one thread are immediately visible to other threads without requiring explicit synchronization in the application code. Memory consistency bugs often manifest as stale configuration being used intermittently under load.

**Stress Testing Under Realistic Load**

Realistic load testing subjects the logging system to conditions similar to production environments, including high message volume, frequent configuration changes, and mixed patterns of concurrent access. This type of testing often reveals performance bottlenecks and race conditions that only appear under sustained load.

The stress tests should monitor not just correctness but also performance characteristics under load. Thread contention, lock acquisition times, and message throughput should remain stable even under extreme concurrency levels.

### Symptom-Cause-Fix Troubleshooting

Systematic troubleshooting of structured logging systems requires mapping observable symptoms to their underlying causes and providing specific, actionable fixes. The challenge lies in the fact that logging system issues often have subtle symptoms that appear far from their root causes, and multiple independent issues can interact to create complex failure modes.

#### Thread Safety and Concurrency Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Logger configuration randomly reverts to defaults | Race condition in logger registry during concurrent logger creation | Add trace logging to logger creation and configuration methods. Check for duplicate logger instances with different configurations. | Implement proper locking in `LoggerRegistry` with `RLock` for reads and exclusive lock for writes. Ensure atomic logger creation and configuration. |
| Log messages occasionally missing from output | Handler list modified during iteration in concurrent threads | Add assertion to verify handler list stability during dispatch. Use concurrent logging load tests to reproduce. | Use copy-on-write pattern for handler lists or lock handler list during iteration and modification. |
| Inconsistent log levels applied to same logger | Effective level cache invalidation race condition | Monitor `_effective_level` cache values across threads. Add logging to level change and cache invalidation operations. | Implement atomic cache invalidation with proper memory barriers. Use versioning to detect stale cache entries. |
| Log records contain mixed context from different requests | Thread pool reusing threads without context cleanup | Inspect thread-local context storage at request boundaries. Check for context cleanup in thread pool management code. | Implement automatic context cleanup using request lifecycle hooks or context managers that clear state on exit. |
| Occasional crashes during high-volume logging | Memory corruption from unsynchronized data structure access | Use thread sanitizers or memory debugging tools. Add lock verification assertions in critical sections. | Add comprehensive locking to all shared data structures. Use lock ordering to prevent deadlocks. |

#### Context Propagation Problems

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Correlation IDs disappear in background tasks | Thread-local context not propagated to new threads | Trace context state before and after thread creation. Check context inheritance in task spawning code. | Implement context-aware thread creation utilities that capture parent context and set it in child threads. |
| Context fields randomly change during async operations | Async context not preserved across await boundaries | Add context state logging before and after each await operation. Monitor context during coroutine switches. | Use async-local storage with proper context copying across task boundaries. Implement async context managers. |
| Memory usage grows continuously in long-running services | Context objects not being garbage collected | Use memory profilers to identify context object accumulation. Check for circular references and strong reference chains. | Implement weak references in context storage. Add automatic context cleanup based on access time or reference counting. |
| Child logger context missing parent fields | Context inheritance not working correctly | Compare parent and child context objects. Trace context inheritance during logger hierarchy traversal. | Fix context inheritance logic to properly merge parent context with local context. Ensure inheritance occurs at access time, not creation time. |
| Context lost after exception handling | Exception unwinding clears context without restoration | Add context state logging in exception handlers. Check context cleanup in finally blocks and exception handling code. | Use context managers that automatically restore previous context state even when exceptions occur during processing. |

#### Serialization and Formatting Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Log output contains `[Object object]` or similar placeholder text | Non-serializable objects in context or log parameters | Add type checking to context field additions. Test serialization of all context objects before logging. | Implement `safe_serialize()` with fallback representations for non-serializable objects. Add type validation for context fields. |
| JSON output malformed or contains invalid characters | Circular references or special characters not being escaped properly | Validate JSON output with strict parsers. Check for circular object references in context data. | Use custom JSON encoder with circular reference detection and proper character escaping. Limit serialization depth. |
| Log timestamps inconsistent or incorrect format | Timestamp generation not thread-safe or using wrong timezone | Compare timestamps across threads and verify timezone handling. Check timestamp generation under concurrent load. | Use thread-safe timestamp generation with proper UTC handling. Implement timestamp caching for performance. |
| Pretty-printed logs missing colors or formatting | Terminal detection not working or color codes not supported | Test color output in different terminal environments. Check terminal capability detection logic. | Implement robust terminal capability detection with graceful fallback to plain text when colors not supported. |
| Large context objects cause out-of-memory errors | Context serialization creating oversized JSON objects | Monitor memory usage during serialization. Set limits on context object size and nesting depth. | Implement context size limits and truncation. Add `estimate_serialized_size()` checks before full serialization. |

#### Handler and Output Destination Problems

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Application hangs during log output | Blocking I/O in handlers without timeouts | Profile application during hang to identify blocked threads. Check file and network I/O patterns in handlers. | Implement non-blocking handlers with background queues. Add timeouts to all I/O operations. Use circuit breakers for unreliable destinations. |
| Log messages lost during handler failures | Handler exceptions not properly caught and recovered | Add error logging to handler dispatch. Monitor handler success/failure rates. Check exception handling in handler code. | Implement comprehensive exception handling in handler dispatch. Add handler health monitoring and automatic recovery. |
| Duplicate log messages in output | Handler dispatch calling same handler multiple times | Trace handler collection and dispatch logic. Check for duplicate handlers in logger hierarchy. | Fix handler collection to remove duplicates. Implement handler identity checking based on handler configuration. |
| Log files not being rotated or growing unbounded | File handler not implementing rotation or size limits | Monitor file sizes and rotation behavior. Check file handler configuration and rotation logic. | Implement proper file rotation with size and time-based triggers. Add monitoring for file handler disk usage. |
| Network handlers causing application instability | Network handlers blocking application threads on failures | Monitor network handler performance and failure rates. Check timeout and retry configuration. | Implement asynchronous network handlers with connection pooling. Add circuit breakers and exponential backoff for failed connections. |

#### Performance and Resource Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Application response time degrades with log volume | Logging operations blocking application threads | Profile application performance vs. logging volume. Measure time spent in logging operations. | Implement asynchronous logging with background processing queues. Optimize hot path operations and reduce lock contention. |
| Memory usage increases proportionally to log volume | Context or handler buffers accumulating without bounds | Monitor memory allocation patterns in logging code. Check for unbounded buffers or caches. | Implement buffer size limits and periodic cleanup. Add memory pressure detection and adaptive behavior. |
| CPU usage high during logging operations | Inefficient serialization or formatting algorithms | Profile CPU usage in logging hot paths. Identify expensive operations like JSON serialization or string formatting. | Optimize serialization with object pooling and caching. Use efficient formatting algorithms and reduce object allocation. |
| Disk I/O spikes causing system instability | Synchronous file writes without buffering or batching | Monitor disk I/O patterns and correlate with logging activity. Check file write patterns in handlers. | Implement write batching and buffering. Use asynchronous I/O or background writer threads. Add disk pressure detection. |
| Logger creation very slow under load | Logger registry lock contention or inefficient hierarchy traversal | Profile logger creation performance. Monitor lock acquisition times and registry operation costs. | Optimize logger registry with better data structures. Use read-write locks to reduce contention. Cache hierarchy traversal results. |

### Implementation Guidance

This implementation guidance provides concrete debugging infrastructure and techniques specifically tailored for Python-based structured logging systems. The focus is on building debugging tools that can be integrated into the logging system itself and used during development and troubleshooting.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Trace Logging | Python `logging` module with separate logger | Custom trace system with structured output |
| Thread Safety Testing | `threading.Thread` with manual barriers | `concurrent.futures` with systematic test framework |
| Memory Debugging | Built-in `tracemalloc` and `gc` modules | External tools like `objgraph` and `memory_profiler` |
| Concurrency Testing | Simple thread spawning with `threading` | Advanced concurrency testing with `pytest-xdist` |
| Context Inspection | Manual context dumps with `pprint` | Rich context visualization with custom formatters |

#### Recommended Module Structure

```
structured_logging/
├── core/
│   ├── logger.py              ← Logger, LoggerRegistry classes
│   ├── handlers.py            ← Handler implementations
│   ├── formatters.py          ← JSON and pretty formatters
│   └── context.py             ← Context management and propagation
├── debugging/
│   ├── __init__.py
│   ├── trace_logger.py        ← Internal tracing system
│   ├── context_inspector.py   ← Context state inspection utilities
│   ├── thread_safety_test.py  ← Concurrency testing framework
│   └── memory_tracker.py      ← Memory leak detection utilities
├── tests/
│   ├── test_race_conditions.py ← Systematic concurrency tests
│   ├── test_context_propagation.py ← Context boundary tests
│   └── test_memory_leaks.py    ← Long-running memory tests
└── examples/
    ├── debug_concurrent_app.py ← Example debugging setup
    └── trace_context_flow.py   ← Context propagation examples
```

#### Infrastructure Starter Code

**Complete Trace Logging System**

```python
import threading
import time
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import deque
import weakref

class TraceLogger:
    """
    Independent trace logging system for debugging the main logging infrastructure.
    Uses simple, robust mechanisms to avoid circular dependencies or complexity
    that might itself introduce bugs.
    """
    
    def __init__(self, output_file: Optional[str] = None, max_records: int = 10000):
        self._lock = threading.Lock()
        self._records = deque(maxlen=max_records)
        self._output_file = output_file
        self._start_time = time.perf_counter()
        
    def trace(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a trace event with high-precision timing and thread information."""
        with self._lock:
            record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'elapsed_ms': (time.perf_counter() - self._start_time) * 1000,
                'thread_id': threading.get_ident(),
                'thread_name': threading.current_thread().name,
                'event_type': event_type,
                'details': details.copy()
            }
            self._records.append(record)
            
            # Immediate output for debugging
            if self._output_file:
                self._write_to_file(record)
            else:
                self._write_to_stderr(record)
    
    def _write_to_file(self, record: Dict[str, Any]) -> None:
        """Write trace record to file with immediate flush."""
        try:
            with open(self._output_file, 'a', encoding='utf-8') as f:
                f.write(f"{record}\n")
                f.flush()
        except Exception as e:
            # Fallback to stderr if file write fails
            sys.stderr.write(f"TraceLogger file error: {e}\n")
            self._write_to_stderr(record)
    
    def _write_to_stderr(self, record: Dict[str, Any]) -> None:
        """Write trace record to stderr for immediate visibility."""
        sys.stderr.write(f"TRACE[{record['thread_name']}]: {record['event_type']} - {record['details']}\n")
        sys.stderr.flush()
    
    def get_records(self, event_type: Optional[str] = None, 
                   thread_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve trace records with optional filtering."""
        with self._lock:
            records = list(self._records)
            
        if event_type:
            records = [r for r in records if r['event_type'] == event_type]
        if thread_id:
            records = [r for r in records if r['thread_id'] == thread_id]
            
        return records
    
    def dump_summary(self) -> str:
        """Generate human-readable summary of trace activity."""
        with self._lock:
            records = list(self._records)
        
        if not records:
            return "No trace records available"
        
        summary_lines = [f"Trace Summary ({len(records)} records)"]
        summary_lines.append(f"Time range: {records[0]['elapsed_ms']:.2f}ms - {records[-1]['elapsed_ms']:.2f}ms")
        
        # Event type frequency
        event_counts = {}
        for record in records:
            event_counts[record['event_type']] = event_counts.get(record['event_type'], 0) + 1
        
        summary_lines.append("\nEvent Types:")
        for event_type, count in sorted(event_counts.items()):
            summary_lines.append(f"  {event_type}: {count}")
        
        # Thread activity
        thread_counts = {}
        for record in records:
            thread_id = record['thread_id']
            thread_counts[thread_id] = thread_counts.get(thread_id, 0) + 1
        
        summary_lines.append(f"\nThread Activity ({len(thread_counts)} threads):")
        for thread_id, count in sorted(thread_counts.items()):
            summary_lines.append(f"  Thread {thread_id}: {count} events")
        
        return "\n".join(summary_lines)

# Global trace logger instance for debugging
_debug_tracer = TraceLogger()

def trace_event(event_type: str, **details) -> None:
    """Convenience function for logging trace events."""
    _debug_tracer.trace(event_type, details)

def get_trace_summary() -> str:
    """Get debugging trace summary."""
    return _debug_tracer.dump_summary()
```

**Context State Inspector**

```python
import threading
import contextvars
import weakref
from typing import Dict, Any, List, Set, Optional
import json
from datetime import datetime

class ContextInspector:
    """
    Utility for inspecting and debugging context state across all storage mechanisms.
    Provides visibility into context propagation, inheritance, and cleanup behavior.
    """
    
    def __init__(self):
        self._inspection_lock = threading.Lock()
        self._context_history: List[Dict[str, Any]] = []
        self._active_contexts: weakref.WeakSet = weakref.WeakSet()
    
    def register_context(self, context_id: str, context_data: Dict[str, Any], 
                        storage_type: str) -> None:
        """Register a context for inspection tracking."""
        with self._inspection_lock:
            self._context_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'register',
                'context_id': context_id,
                'storage_type': storage_type,
                'thread_id': threading.get_ident(),
                'data_snapshot': context_data.copy(),
                'field_count': len(context_data)
            })
    
    def snapshot_all_contexts(self) -> Dict[str, Any]:
        """Take a complete snapshot of all context storage mechanisms."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'thread_local_contexts': self._snapshot_thread_local(),
            'async_contexts': self._snapshot_async_contexts(),
            'active_context_count': len(self._active_contexts),
            'history_length': len(self._context_history)
        }
        return snapshot
    
    def _snapshot_thread_local(self) -> Dict[str, Any]:
        """Snapshot all thread-local context storage."""
        # This would integrate with the actual ContextStorage implementation
        thread_contexts = {}
        
        # Iterate through all active threads and capture their context
        for thread in threading.enumerate():
            try:
                # This assumes the ContextStorage has a method to get context by thread
                # In real implementation, this would access the actual thread-local storage
                thread_context = self._get_thread_context(thread.ident)
                if thread_context:
                    thread_contexts[f"thread_{thread.ident}"] = {
                        'thread_name': thread.name,
                        'context': thread_context,
                        'field_count': len(thread_context)
                    }
            except Exception as e:
                thread_contexts[f"thread_{thread.ident}"] = {'error': str(e)}
        
        return thread_contexts
    
    def _snapshot_async_contexts(self) -> Dict[str, Any]:
        """Snapshot async context variables."""
        async_contexts = {}
        
        try:
            # Capture current async context if available
            current_context = contextvars.copy_context()
            context_vars = {}
            
            # This would iterate through actual context variables in use
            # In real implementation, this would access the ContextStorage async variables
            for var, value in current_context.items():
                if hasattr(var, 'name') and 'logging' in var.name:
                    context_vars[var.name] = value
            
            async_contexts['current_task'] = {
                'context_vars': context_vars,
                'var_count': len(context_vars)
            }
            
        except Exception as e:
            async_contexts['error'] = str(e)
        
        return async_contexts
    
    def _get_thread_context(self, thread_id: int) -> Optional[Dict[str, Any]]:
        """Get context for specific thread - placeholder for actual implementation."""
        # In real implementation, this would access the ContextStorage._thread_local
        return None
    
    def find_context_inconsistencies(self) -> List[Dict[str, Any]]:
        """Identify inconsistencies between different context storage mechanisms."""
        inconsistencies = []
        snapshot = self.snapshot_all_contexts()
        
        # Compare thread-local and async contexts for current thread
        current_thread_id = threading.get_ident()
        thread_context_key = f"thread_{current_thread_id}"
        
        thread_context = snapshot['thread_local_contexts'].get(thread_context_key, {}).get('context', {})
        async_context = snapshot['async_contexts'].get('current_task', {}).get('context_vars', {})
        
        # Check for missing fields
        for field in thread_context:
            if field not in async_context:
                inconsistencies.append({
                    'type': 'missing_in_async',
                    'field': field,
                    'thread_value': thread_context[field],
                    'thread_id': current_thread_id
                })
        
        for field in async_context:
            if field not in thread_context:
                inconsistencies.append({
                    'type': 'missing_in_thread_local',
                    'field': field,
                    'async_value': async_context[field],
                    'thread_id': current_thread_id
                })
        
        # Check for value mismatches
        for field in set(thread_context.keys()) & set(async_context.keys()):
            if thread_context[field] != async_context[field]:
                inconsistencies.append({
                    'type': 'value_mismatch',
                    'field': field,
                    'thread_value': thread_context[field],
                    'async_value': async_context[field],
                    'thread_id': current_thread_id
                })
        
        return inconsistencies
    
    def generate_context_report(self) -> str:
        """Generate comprehensive context debugging report."""
        snapshot = self.snapshot_all_contexts()
        inconsistencies = self.find_context_inconsistencies()
        
        report_lines = [
            "=== Context State Debugging Report ===",
            f"Generated at: {snapshot['timestamp']}",
            f"Active contexts: {snapshot['active_context_count']}",
            f"History entries: {snapshot['history_length']}",
            ""
        ]
        
        # Thread-local context summary
        report_lines.append("Thread-Local Contexts:")
        for thread_key, context_info in snapshot['thread_local_contexts'].items():
            if 'error' in context_info:
                report_lines.append(f"  {thread_key}: ERROR - {context_info['error']}")
            else:
                field_count = context_info.get('field_count', 0)
                thread_name = context_info.get('thread_name', 'unknown')
                report_lines.append(f"  {thread_key} ({thread_name}): {field_count} fields")
                
                # Show context fields if not too many
                if field_count <= 10:
                    for field, value in context_info.get('context', {}).items():
                        report_lines.append(f"    {field}: {value}")
        
        report_lines.append("")
        
        # Async context summary
        report_lines.append("Async Contexts:")
        async_info = snapshot['async_contexts']
        if 'error' in async_info:
            report_lines.append(f"  ERROR: {async_info['error']}")
        else:
            current_task = async_info.get('current_task', {})
            var_count = current_task.get('var_count', 0)
            report_lines.append(f"  Current task: {var_count} context variables")
            
            for var_name, value in current_task.get('context_vars', {}).items():
                report_lines.append(f"    {var_name}: {value}")
        
        report_lines.append("")
        
        # Inconsistencies
        if inconsistencies:
            report_lines.append(f"Context Inconsistencies ({len(inconsistencies)} found):")
            for inconsistency in inconsistencies:
                report_lines.append(f"  {inconsistency['type']}: {inconsistency['field']}")
                if 'thread_value' in inconsistency:
                    report_lines.append(f"    Thread-local: {inconsistency['thread_value']}")
                if 'async_value' in inconsistency:
                    report_lines.append(f"    Async: {inconsistency['async_value']}")
        else:
            report_lines.append("No context inconsistencies detected.")
        
        return "\n".join(report_lines)

# Global context inspector for debugging
_context_inspector = ContextInspector()

def snapshot_contexts() -> Dict[str, Any]:
    """Take snapshot of all context state for debugging."""
    return _context_inspector.snapshot_all_contexts()

def generate_context_debug_report() -> str:
    """Generate comprehensive context debugging report."""
    return _context_inspector.generate_context_report()
```

#### Core Logic Skeleton Code

**Thread Safety Testing Framework**

```python
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Dict, Any
import queue

class ConcurrencyTestFramework:
    """
    Framework for systematic testing of thread safety in logging components.
    Provides utilities for coordinated multi-threaded testing with barrier synchronization.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.test_results: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()
    
    def run_coordinated_test(self, test_func: Callable, num_threads: int = 5, 
                           iterations: int = 100, coordination_delay: float = 0.001) -> Dict[str, Any]:
        """
        Run coordinated multi-threaded test with barrier synchronization.
        
        Args:
            test_func: Function to test - should accept (thread_id, iteration, barrier)
            num_threads: Number of concurrent threads
            iterations: Number of iterations per thread
            coordination_delay: Delay to maximize contention probability
        """
        # TODO 1: Create barrier for thread coordination
        # TODO 2: Create results collection mechanism
        # TODO 3: Launch threads with coordinated start
        # TODO 4: Collect results and analyze for race conditions
        # TODO 5: Return summary with success/failure counts and timing information
        # Hint: Use threading.Barrier to synchronize thread starts
        pass
    
    def test_logger_hierarchy_creation(self, logger_factory: Callable) -> Dict[str, Any]:
        """
        Test concurrent logger creation for race conditions.
        
        Args:
            logger_factory: Function that creates loggers - get_logger(name)
        """
        # TODO 1: Define logger names that should create hierarchy conflicts
        # TODO 2: Create test function that creates loggers with barrier coordination
        # TODO 3: Run coordinated test with multiple threads creating same hierarchies
        # TODO 4: Verify no duplicate loggers or corrupted hierarchies were created
        # TODO 5: Check parent-child relationships are correct after concurrent creation
        # Hint: Test names like ["com.example", "com.example.service", "com.example.service.auth"]
        pass
    
    def test_concurrent_configuration_changes(self, logger_registry) -> Dict[str, Any]:
        """
        Test concurrent logger configuration changes.
        
        Args:
            logger_registry: Registry containing loggers to test
        """
        # TODO 1: Create test scenarios mixing level changes and handler additions
        # TODO 2: Coordinate threads to make changes at same time using barrier
        # TODO 3: Verify final configuration state is consistent
        # TODO 4: Check that effective level caches were properly invalidated
        # TODO 5: Ensure no configuration was lost or corrupted during changes
        pass
    
    def test_context_propagation_race_conditions(self, context_manager) -> Dict[str, Any]:
        """
        Test context propagation under concurrent access.
        
        Args:
            context_manager: ContextStorage instance to test
        """
        # TODO 1: Create test that sets different context in multiple threads
        # TODO 2: Add child context inheritance operations during concurrent updates
        # TODO 3: Verify context isolation between threads
        # TODO 4: Check that context inheritance works correctly under contention
        # TODO 5: Ensure no context corruption or cross-thread contamination
        pass

# Usage example for milestone checkpoints
def run_thread_safety_checkpoint(logger_system) -> bool:
    """
    Run comprehensive thread safety tests for milestone verification.
    Returns True if all tests pass, False if race conditions detected.
    """
    framework = ConcurrencyTestFramework()
    
    # Test logger creation under contention
    creation_results = framework.test_logger_hierarchy_creation(logger_system.get_logger)
    
    # Test configuration changes
    config_results = framework.test_concurrent_configuration_changes(logger_system.registry)
    
    # Test context propagation
    context_results = framework.test_context_propagation_race_conditions(logger_system.context_manager)
    
    # Analyze results
    all_passed = (
        creation_results.get('failures', 0) == 0 and
        config_results.get('failures', 0) == 0 and
        context_results.get('failures', 0) == 0
    )
    
    if not all_passed:
        print("Thread safety test failures detected:")
        print(f"Logger creation: {creation_results.get('failures', 0)} failures")
        print(f"Configuration: {config_results.get('failures', 0)} failures") 
        print(f"Context propagation: {context_results.get('failures', 0)} failures")
    
    return all_passed
```

#### Milestone Checkpoints

**Milestone 1: Logger Core Thread Safety Verification**

After implementing the logger core, run these verification steps:

```bash
# Run basic thread safety tests
python -m pytest tests/test_race_conditions.py::test_concurrent_logger_creation -v

# Expected output should show no race conditions:
# test_concurrent_logger_creation PASSED
# test_concurrent_level_changes PASSED  
# test_concurrent_handler_dispatch PASSED
```

Verify thread safety manually:
1. Start Python interpreter
2. Create logger registry and spawn 10 threads simultaneously creating loggers
3. Check that logger hierarchy is consistent and no duplicate loggers exist
4. Verify that all loggers have correct parent-child relationships

**Milestone 2: Structured Output Serialization Testing**

```bash
# Test JSON serialization under load
python -m pytest tests/test_serialization.py::test_concurrent_json_formatting -v

# Test with problematic objects
python examples/test_serialization_edge_cases.py
```

Expected behavior:
- All JSON output should be valid single-line JSON
- No serialization errors even with circular references
- Memory usage should remain stable during high-volume serialization

**Milestone 3: Context Propagation Verification**

```bash
# Test async context preservation
python -m pytest tests/test_context_propagation.py::test_async_context_preservation -v

# Test thread-local context isolation  
python examples/test_context_boundaries.py
```

Manual verification:
1. Create request with correlation ID in main thread
2. Spawn background task and verify correlation ID is present
3. Modify context in background task and verify main thread context unchanged
4. Check that context cleanup occurs after request completion

#### Language-Specific Python Debugging Tips

- Use `threading.get_ident()` to identify which thread is executing logging operations
- `threading.current_thread().name` provides human-readable thread identification
- `tracemalloc.start()` at application startup enables memory leak detection
- `gc.get_objects()` can find leaked context objects by type filtering  
- `weakref.WeakSet` prevents circular references in context tracking
- `contextvars.copy_context()` captures complete async context state
- `sys.stderr.write()` provides immediate debug output that bypasses the main logging system
- Use `threading.RLock()` instead of `Lock()` for recursive logger operations
- `time.perf_counter()` provides high-resolution timing for race condition detection
- `concurrent.futures.ThreadPoolExecutor` simplifies coordinated multi-threaded testing


## Future Extensions

> **Milestone(s):** This section outlines potential enhancements beyond the three core milestones that the current design accommodates. While Milestones 1-3 establish the foundation, these extensions represent production-scale optimizations and enterprise-level features that can be added incrementally.

Think of this logging system as a highway infrastructure. The three core milestones build the essential roads, traffic signals, and basic navigation systems that handle normal traffic flow. These future extensions are like adding express lanes for high-volume traffic (performance optimizations), connecting to airports and shipping ports (advanced output destinations), and installing smart traffic management systems that adapt to changing conditions (configuration management). Each extension leverages the solid foundation established by the core milestones while adding sophisticated capabilities for production environments.

The architectural decisions made in the core design deliberately accommodate these extensions without requiring fundamental restructuring. The handler dispatch mechanism supports pluggable output destinations, the context propagation system enables sampling decisions, and the thread-safe registry pattern allows dynamic reconfiguration. This forward-looking design ensures that organizations can start with the essential logging capabilities and scale up to enterprise requirements as their needs evolve.

### Performance Optimizations

Production logging systems must handle massive throughput while maintaining low latency for application threads. Think of performance optimizations as traffic management systems for a busy highway. During rush hour, you need express lanes for priority traffic (log sampling), speed limits to prevent accidents (rate limiting), and efficient on/off ramps that don't block the main flow (async handler dispatch). These optimizations ensure that logging enhances observability without degrading application performance.

The core design's thread-safe architecture and handler abstraction pattern provide the foundation for these performance enhancements. The `Handler` base class can be extended with buffering and batching capabilities, while the `LoggingContext` system enables sampling decisions based on request characteristics. The `CircuitBreaker` pattern already implemented for error handling extends naturally to performance protection by temporarily disabling expensive operations under load.

> **Decision: Log Sampling Strategy**
> - **Context**: High-throughput applications generate millions of log entries per second, overwhelming storage and network capacity while providing diminishing observability value
> - **Options Considered**: Random sampling, adaptive sampling, structured sampling
> - **Decision**: Implement adaptive sampling with context-aware rate limiting
> - **Rationale**: Adaptive sampling maintains full fidelity for errors and important events while reducing volume for routine operations. Context-aware decisions ensure complete request traces are preserved even when individual log entries are sampled
> - **Consequences**: Enables high-throughput logging with bounded resource usage while preserving debugging capability for critical scenarios

**Adaptive Sampling Architecture**

Adaptive sampling makes intelligent decisions about which log entries to preserve based on current system load, log entry importance, and correlation context. The sampling system operates at multiple levels: per-logger sampling rates, per-request sampling decisions, and global backpressure handling. This multi-layered approach ensures that critical information is never lost while routine operations can be sampled aggressively during high-load periods.

The `SamplingHandler` wraps existing handlers and makes sampling decisions before expensive operations like network transmission or disk writes. Sampling decisions are made deterministically based on correlation IDs, ensuring that when a request is selected for sampling, all related log entries are preserved together. This correlation-aware sampling maintains complete request traces while reducing overall volume.

| Component | Purpose | Key Responsibilities |
|-----------|---------|---------------------|
| `AdaptiveSampler` | Core sampling logic | Rate calculation, decision caching, context awareness |
| `SamplingHandler` | Handler wrapper | Pre-filtering, correlation tracking, metrics collection |
| `SamplingConfig` | Configuration model | Rate limits, importance rules, adaptation parameters |
| `LoadMonitor` | System monitoring | CPU/memory tracking, backpressure detection, rate adjustment |
| `SampleDecisionCache` | Decision storage | Correlation ID mapping, TTL management, memory bounds |

**Rate Limiting Implementation**

Rate limiting protects the logging system and downstream consumers from overload by enforcing maximum throughput limits. Think of rate limiting as traffic lights that meter the flow of vehicles onto a highway, preventing congestion that would slow down all traffic. The rate limiting system operates at multiple granularities: global limits prevent system-wide overload, per-logger limits prevent individual components from monopolizing resources, and per-level limits ensure that verbose debug logging doesn't crowd out important error messages.

The `TokenBucketRateLimiter` provides smooth rate limiting with burst capacity, allowing applications to handle traffic spikes while maintaining average rate limits. Token buckets are maintained per logger and per level, with hierarchical inheritance ensuring that child loggers respect parent limits. The rate limiter integrates with the sampling system to gracefully degrade service rather than dropping log entries randomly.

| Rate Limiter Type | Use Case | Configuration Parameters |
|-------------------|----------|-------------------------|
| `GlobalRateLimiter` | System-wide protection | tokens_per_second, burst_capacity, backpressure_threshold |
| `LoggerRateLimiter` | Per-component limits | logger_name, max_rate, inheritance_factor |
| `LevelRateLimiter` | Per-severity limits | log_level, rate_multiplier, priority_boost |
| `CorrelationRateLimiter` | Per-request limits | correlation_pattern, request_budget, spillover_handling |

**Asynchronous Handler Dispatch**

Asynchronous handler dispatch decouples log entry processing from application thread execution, preventing slow output destinations from blocking application performance. Think of async dispatch as a restaurant kitchen where orders are placed on a queue and prepared by dedicated cooks, allowing waiters to continue serving customers without waiting for each dish to be completed. This architectural pattern ensures that logging operations never block application threads, even when writing to slow destinations like remote collectors or network file systems.

The `AsyncHandler` maintains internal queues for each output destination and uses dedicated worker threads to process log entries in the background. Queue sizing and backpressure handling prevent memory exhaustion during traffic spikes, while priority queues ensure that high-severity log entries are processed first during congestion. The async system preserves ordering guarantees within each correlation context while allowing parallel processing across different requests.

> **Critical Design Insight**: Async dispatch introduces complexity around system shutdown and error propagation. The shutdown sequence must drain queues gracefully while the error handling system needs to surface async failures without blocking the main application flow.

| Async Component | Threading Model | Queue Management |
|-----------------|-----------------|------------------|
| `AsyncHandler` | Single worker per handler | Bounded blocking queue, overflow to disk |
| `BatchingHandler` | Periodic flush thread | Time and size-based batching, compression |
| `PriorityHandler` | Priority queue worker | Level-based ordering, starvation prevention |
| `ShutdownManager` | Graceful termination | Queue draining, timeout handling, force shutdown |

⚠️ **Pitfall: Queue Memory Explosion**
Under extreme load, async queues can consume unbounded memory if not properly configured. Implement overflow-to-disk mechanisms and queue size limits with backpressure signaling to prevent out-of-memory conditions. Monitor queue depths and implement alerting when queues approach capacity limits.

### Advanced Output Destinations

Production logging systems must integrate with enterprise infrastructure including search engines, metrics systems, and distributed tracing platforms. Think of advanced output destinations as specialized delivery services: while basic handlers are like local mail delivery, these advanced destinations are like express shipping to international locations with customs processing and tracking integration. Each destination has unique requirements for data format, authentication, batching, and error handling.

The `Handler` abstraction established in the core design provides a clean integration point for these advanced destinations. Each advanced handler implements the same `handle(record)` interface while internally managing the complexity of external system integration. This consistency ensures that advanced destinations can be mixed and matched with basic handlers, allowing applications to send the same log stream to multiple destinations simultaneously.

**Elasticsearch Integration**

Elasticsearch integration transforms the logging system into a powerful search and analytics platform by indexing log entries for real-time querying and dashboard visualization. Think of Elasticsearch as a library cataloging system that not only stores books but also creates detailed indexes by subject, author, and keyword, enabling researchers to find exactly what they need instantly. The Elasticsearch handler maps structured log entries to JSON documents with appropriate field mappings and index lifecycle management.

The `ElasticsearchHandler` batches log entries into bulk index operations to maximize throughput while managing index templates and field mappings automatically. Index rotation strategies partition log data by time period, enabling efficient storage management and query performance. The handler integrates with the circuit breaker pattern to handle Elasticsearch cluster outages gracefully while buffering entries for retry when connectivity is restored.

| Elasticsearch Component | Purpose | Configuration Options |
|-------------------------|---------|---------------------|
| `ElasticsearchHandler` | Bulk indexing | batch_size, flush_interval, index_template |
| `IndexManager` | Lifecycle management | rotation_policy, retention_days, shard_count |
| `MappingGenerator` | Schema creation | field_types, analyzers, dynamic_mapping |
| `BulkProcessor` | Batch optimization | max_batch_bytes, concurrent_requests, retry_policy |

> **Decision: Index Rotation Strategy**
> - **Context**: Log data accumulates rapidly and requires different retention policies based on age and importance
> - **Options Considered**: Single large index, daily rotation, size-based rotation
> - **Decision**: Implement daily index rotation with configurable retention policies
> - **Rationale**: Daily rotation provides optimal balance between query performance and operational simplicity, while enabling different retention policies for different log levels
> - **Consequences**: Enables efficient storage management and fast query performance while requiring index template management and automated cleanup processes

**Metrics Integration**

Metrics integration extracts quantitative signals from log entries to populate time-series databases and alerting systems. Think of metrics integration as transforming narrative log entries into numerical health indicators, similar to how a doctor extracts vital signs from patient observations. The metrics system counts log entries by level and logger, measures request durations from correlation context, and creates custom metrics from structured log fields.

The `MetricsHandler` analyzes log entries in real-time to emit counter, gauge, and histogram metrics to systems like Prometheus, StatsD, or CloudWatch. Metric extraction rules are configurable, allowing applications to define custom metrics based on log content without modifying application code. The metrics system provides immediate visibility into application health and performance trends that complement the detailed log analysis capabilities.

| Metrics Component | Metric Types | Export Destinations |
|-------------------|--------------|-------------------|
| `MetricsExtractor` | Counter, Gauge, Histogram | Field parsing, aggregation rules, labeling strategy |
| `PrometheusHandler` | Time-series metrics | /metrics endpoint, push gateway, service discovery |
| `StatsDHandler` | Real-time metrics | UDP transport, metric namespacing, sampling support |
| `CloudWatchHandler` | Cloud metrics | AWS API integration, custom namespaces, dimension mapping |

**Distributed Tracing Integration**

Distributed tracing integration connects log entries to trace spans and enables correlation across service boundaries in microservice architectures. Think of distributed tracing as creating a GPS tracking system for requests as they travel through multiple services, with log entries serving as detailed checkpoints along the journey. The tracing integration adds span context to log entries and creates trace annotations from log content.

The `TracingHandler` extracts trace context from correlation IDs and injects trace metadata into log entries for cross-system correlation. When integrated with systems like Jaeger or Zipkin, log entries become searchable annotations within trace visualizations, providing detailed context for performance analysis and debugging. The handler creates trace spans for long-running operations identified through log patterns and correlation timing.

| Tracing Component | Integration Method | Context Propagation |
|-------------------|-------------------|-------------------|
| `OpenTelemetryHandler` | OTEL protocol | W3C trace context, baggage propagation |
| `JaegerHandler` | Jaeger client | Uber trace header, span annotations |
| `ZipkinHandler` | Zipkin protocol | B3 propagation, span creation from logs |
| `TraceContextManager` | Context injection | Cross-service correlation, timing extraction |

### Configuration Management

Production logging systems require sophisticated configuration management to adapt to changing operational requirements without service restarts. Think of configuration management as the control tower for an airport, continuously adjusting flight paths, runway assignments, and traffic patterns based on weather, equipment status, and traffic volume. The configuration system must handle real-time updates, environment-specific settings, and complex inheritance hierarchies while maintaining system stability.

The core design's registry pattern and hierarchical structure provide the foundation for advanced configuration capabilities. The `LoggerRegistry` can be extended to support dynamic reconfiguration, while the inheritance system enables sophisticated configuration layering. Configuration changes propagate through the logger hierarchy automatically while preserving thread safety and avoiding configuration drift between components.

> **Decision: Configuration Hot-Reload Strategy**
> - **Context**: Production systems require configuration changes without service restart to handle incidents and adjust verbosity dynamically
> - **Options Considered**: File watching, API endpoints, configuration service polling
> - **Decision**: Implement multi-source configuration with API-driven hot reload and file watching fallback
> - **Rationale**: API endpoints enable programmatic configuration during incidents while file watching provides declarative configuration management for normal operations
> - **Consequences**: Enables rapid response to production issues while requiring additional API security and configuration validation logic

**Dynamic Reconfiguration System**

Dynamic reconfiguration allows logging behavior to change at runtime without service restarts, enabling rapid response to production incidents and performance optimization. The reconfiguration system validates configuration changes before applying them, ensuring that invalid configurations don't disrupt logging operations. Configuration changes are applied atomically across all affected loggers to prevent inconsistent states during transitions.

The `ConfigurationManager` coordinates configuration updates across multiple sources and applies changes through a phased rollout process. Validation rules prevent configurations that would cause performance problems or break system invariants. The system maintains configuration history and rollback capabilities to recover from problematic changes quickly.

| Configuration Component | Responsibility | Validation Rules |
|------------------------|----------------|------------------|
| `ConfigurationManager` | Change coordination | Atomicity, consistency, rollback support |
| `ValidationEngine` | Safety checking | Level hierarchy, handler validity, resource limits |
| `ChangeApplicator` | Update execution | Logger traversal, handler reconfiguration, cache invalidation |
| `ConfigurationHistory` | Audit trail | Change tracking, rollback support, compliance logging |

**Environment-Based Configuration**

Environment-based configuration adapts logging behavior automatically based on deployment context, ensuring that development, staging, and production environments have appropriate logging policies without manual configuration management. The environment system detects deployment context through environment variables, service discovery, or configuration files and applies environment-specific overlays to base configurations.

Different environments have fundamentally different logging needs: development requires verbose output with pretty formatting, staging needs production-like structured logging with extended retention, and production requires optimized performance with security-conscious field filtering. The environment system manages these differences through layered configuration that inherits base settings while overriding environment-specific concerns.

| Environment Type | Configuration Priorities | Typical Settings |
|------------------|-------------------------|------------------|
| `Development` | Readability, debugging | Pretty formatting, DEBUG level, console output |
| `Staging` | Production simulation | JSON formatting, INFO level, file + remote |
| `Production` | Performance, security | Optimized JSON, WARN level, filtered fields |
| `Testing` | Determinism, isolation | Structured output, captured handlers, no external systems |

⚠️ **Pitfall: Configuration Drift**
Environment-specific configurations can drift over time, leading to difficult-to-reproduce issues where bugs only appear in specific environments. Implement configuration validation tests that verify environment configurations produce expected behavior and maintain configuration documentation that explains environment-specific choices.

**Configuration File Formats**

Configuration file support enables declarative logging setup through YAML, JSON, or TOML files that define logger hierarchies, handler configurations, and formatter settings. Configuration files provide version-controlled logging policies that can be reviewed, tested, and deployed through standard infrastructure processes. The file format supports templating and environment variable substitution for flexible deployment scenarios.

The configuration parser validates file syntax and semantic correctness before applying changes, preventing invalid configurations from disrupting logging operations. Configuration files support includes and inheritance, enabling modular configuration that can be composed for different deployment scenarios. The system watches configuration files for changes and applies updates automatically when files are modified.

| Format | Strengths | Use Cases |
|--------|-----------|-----------|
| `YAML` | Human-readable, comments | Development configuration, documentation |
| `JSON` | Machine-readable, validation | API-driven configuration, service discovery |
| `TOML` | Type-safe, structured | Production configuration, complex hierarchies |
| `Environment Variables` | Container-friendly, simple | Docker deployment, cloud platforms |

> **Advanced Configuration Architecture**: The configuration system supports layered composition where base configurations provide common settings and environment-specific overlays add contextual modifications. This pattern enables maintainable configuration that avoids duplication while supporting diverse deployment requirements.

### Implementation Guidance

The future extensions build upon the solid foundation established in the core milestones while adding sophisticated capabilities for production environments. These extensions demonstrate the value of forward-looking architectural design that anticipates growth requirements without over-engineering the initial implementation.

**Technology Recommendations**

| Extension Category | Simple Option | Advanced Option |
|-------------------|---------------|-----------------|
| Performance Monitoring | Built-in metrics collection | Prometheus + Grafana integration |
| Elasticsearch Integration | HTTP client + JSON serialization | Official Elasticsearch Python client |
| Configuration Management | File watching + JSON parsing | etcd/Consul + configuration service |
| Async Processing | Threading module + queues | asyncio + aiohttp for async handlers |
| Load Testing | Manual threading tests | Locust or Artillery for realistic load |

**Recommended Module Structure**

These extensions integrate cleanly into the existing project structure while maintaining clear separation of concerns:

```
project-root/
├── core/
│   ├── logger.py              ← Core logger from Milestone 1
│   ├── formatters.py          ← JSON formatters from Milestone 2
│   └── context.py             ← Context system from Milestone 3
├── handlers/
│   ├── basic.py               ← File/Console handlers from core milestones
│   ├── elasticsearch.py      ← Elasticsearch integration
│   ├── metrics.py             ← Metrics extraction handlers
│   └── tracing.py             ← Distributed tracing integration
├── performance/
│   ├── sampling.py            ← Adaptive sampling logic
│   ├── rate_limiting.py       ← Token bucket rate limiters
│   └── async_dispatch.py      ← Asynchronous handler dispatch
├── config/
│   ├── manager.py             ← Dynamic configuration management
│   ├── validation.py          ← Configuration safety checking
│   └── environments.py       ← Environment-specific configuration
└── extensions/
    ├── load_testing.py        ← Performance testing utilities
    └── monitoring.py          ← Extension health monitoring
```

**Performance Optimization Starter Code**

```python
import threading
import time
import random
from typing import Dict, Any, Optional
from dataclasses import dataclass
from core.handlers import BaseHandler
from core.records import LogRecord

@dataclass
class SamplingConfig:
    """Configuration for adaptive sampling behavior."""
    base_sample_rate: float = 1.0
    max_sample_rate: float = 1.0
    min_sample_rate: float = 0.01
    adaptation_window: int = 60  # seconds
    importance_boost: Dict[str, float] = None
    
    def __post_init__(self):
        if self.importance_boost is None:
            self.importance_boost = {
                'ERROR': 10.0,
                'FATAL': 100.0,
                'WARN': 2.0
            }

class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter for smooth rate limiting."""
    
    def __init__(self, tokens_per_second: float, burst_capacity: int):
        self.tokens_per_second = tokens_per_second
        self.burst_capacity = burst_capacity
        self._tokens = float(burst_capacity)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire the specified number of tokens.
        Returns True if tokens were acquired, False if rate limited.
        """
        with self._lock:
            now = time.time()
            # TODO 1: Calculate tokens to add based on time elapsed
            # TODO 2: Add tokens to bucket without exceeding burst capacity
            # TODO 3: Check if enough tokens available for request
            # TODO 4: Deduct tokens if available and return success
            # TODO 5: Return False if insufficient tokens (rate limited)
            pass

class AdaptiveSampler:
    """Intelligent sampling that adapts to load and preserves important events."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self._current_rate = config.base_sample_rate
        self._decision_cache: Dict[str, bool] = {}
        self._load_monitor = LoadMonitor()
        self._lock = threading.RLock()
        
    def should_sample(self, record: LogRecord) -> bool:
        """
        Determine if this log record should be sampled (kept).
        Uses correlation ID for consistent sampling decisions.
        """
        # TODO 1: Check if correlation ID already has cached decision
        # TODO 2: Calculate importance multiplier based on log level
        # TODO 3: Get current system load from load monitor
        # TODO 4: Adjust sampling rate based on load and importance
        # TODO 5: Make deterministic decision based on correlation ID hash
        # TODO 6: Cache decision for correlation ID consistency
        # TODO 7: Return sampling decision
        pass
    
    def _calculate_adaptive_rate(self, base_importance: float, system_load: float) -> float:
        """Calculate sampling rate based on importance and system load."""
        # TODO: Implement adaptive rate calculation
        # Hint: Higher load should reduce sampling rate, higher importance should increase it
        pass

class LoadMonitor:
    """Monitor system load to inform sampling decisions."""
    
    def __init__(self, measurement_window: int = 30):
        self.measurement_window = measurement_window
        self._cpu_samples = []
        self._memory_samples = []
        self._lock = threading.Lock()
        self._monitoring = False
        
    def get_current_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        # TODO 1: Collect current CPU usage
        # TODO 2: Collect current memory usage  
        # TODO 3: Calculate average over measurement window
        # TODO 4: Return normalized load values (0.0 to 1.0)
        pass
```

**Elasticsearch Handler Infrastructure**

```python
import json
import threading
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from core.handlers import BaseHandler
from core.records import LogRecord

@dataclass
class ElasticsearchConfig:
    """Configuration for Elasticsearch integration."""
    hosts: List[str]
    index_prefix: str = "logs"
    batch_size: int = 100
    flush_interval: float = 5.0
    max_retries: int = 3
    timeout: float = 30.0
    
class BulkProcessor:
    """Efficient bulk processing for Elasticsearch indexing."""
    
    def __init__(self, config: ElasticsearchConfig, client):
        self.config = config
        self.client = client
        self._batch_buffer: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._last_flush = time.time()
        
    def add_document(self, record: LogRecord) -> None:
        """Add log record to batch buffer for bulk indexing."""
        document = {
            '_index': self._generate_index_name(record),
            '_source': record.to_dict()
        }
        
        with self._batch_lock:
            # TODO 1: Add document to batch buffer
            # TODO 2: Check if batch is full or flush interval exceeded
            # TODO 3: Trigger flush if either condition met
            # TODO 4: Handle buffer overflow protection
            pass
    
    def _generate_index_name(self, record: LogRecord) -> str:
        """Generate time-based index name for log record."""
        # TODO: Create index name like "logs-app-2023-12-01"
        # Hint: Use record timestamp and config.index_prefix
        pass
    
    def flush_batch(self) -> bool:
        """Flush current batch to Elasticsearch."""
        with self._batch_lock:
            if not self._batch_buffer:
                return True
                
            # TODO 1: Copy current batch and clear buffer
            # TODO 2: Execute bulk index operation
            # TODO 3: Handle partial failures and retries
            # TODO 4: Update last flush timestamp
            # TODO 5: Return success status
            pass

class ElasticsearchHandler(BaseHandler):
    """Handler that indexes log records to Elasticsearch for search and analytics."""
    
    def __init__(self, config: ElasticsearchConfig):
        super().__init__(name="elasticsearch")
        self.config = config
        self._client = None  # TODO: Initialize Elasticsearch client
        self._bulk_processor = BulkProcessor(config, self._client)
        
    def _write_record(self, record: LogRecord) -> None:
        """Write log record to Elasticsearch via bulk processor."""
        # TODO 1: Validate Elasticsearch client connection
        # TODO 2: Add record to bulk processor
        # TODO 3: Handle connection errors gracefully
        # TODO 4: Implement circuit breaker logic for outages
        pass
```

**Configuration Management Core**

```python
import threading
import json
import yaml
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from core.logger import LoggerRegistry

@dataclass
class ConfigurationChange:
    """Represents a configuration change with validation and rollback info."""
    change_id: str
    timestamp: float
    changes: Dict[str, Any]
    previous_values: Dict[str, Any]
    source: str  # "file", "api", "environment"
    
class ConfigurationValidator:
    """Validates configuration changes before applying them."""
    
    def __init__(self):
        self._validation_rules: List[Callable[[Dict[str, Any]], List[str]]] = []
        self._register_default_rules()
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of errors.
        Empty list means configuration is valid.
        """
        errors = []
        # TODO 1: Run all validation rules against configuration
        # TODO 2: Collect all validation errors from rules
        # TODO 3: Check for configuration consistency
        # TODO 4: Validate resource limits and constraints
        # TODO 5: Return comprehensive error list
        return errors
    
    def _register_default_rules(self):
        """Register built-in validation rules."""
        # TODO: Add validation rules for log levels, handler configs, etc.
        pass

class ConfigurationManager:
    """Manages dynamic configuration updates with validation and rollback."""
    
    def __init__(self, logger_registry: LoggerRegistry):
        self.logger_registry = logger_registry
        self._current_config: Dict[str, Any] = {}
        self._config_history: List[ConfigurationChange] = []
        self._validator = ConfigurationValidator()
        self._lock = threading.RLock()
        
    def apply_configuration(self, new_config: Dict[str, Any], source: str = "api") -> bool:
        """
        Apply new configuration with validation and rollback support.
        Returns True if configuration was applied successfully.
        """
        with self._lock:
            # TODO 1: Validate new configuration
            # TODO 2: Create configuration change record
            # TODO 3: Apply changes atomically
            # TODO 4: Update logger registry
            # TODO 5: Handle rollback on failure
            # TODO 6: Update configuration history
            pass
    
    def rollback_to_previous(self) -> bool:
        """Rollback to the previous configuration."""
        # TODO 1: Find most recent configuration change
        # TODO 2: Apply previous values
        # TODO 3: Validate rollback configuration
        # TODO 4: Update logger registry
        # TODO 5: Record rollback in history
        pass
```

**Milestone Checkpoints**

After implementing performance optimizations:
- Run `python -m performance.load_test` to generate high log volume
- Verify that sampling reduces output volume proportionally
- Check that rate limiting prevents resource exhaustion
- Confirm async dispatch doesn't block application threads

After implementing Elasticsearch integration:
- Start local Elasticsearch instance
- Send log entries through `ElasticsearchHandler`
- Query Elasticsearch to verify proper indexing and field mapping
- Test index rotation and retention policies

After implementing configuration management:
- Modify configuration files and verify hot reload
- Use API to change log levels without restart
- Test configuration validation with invalid settings
- Verify rollback functionality restores previous behavior

**Common Extension Pitfalls**

⚠️ **Pitfall: Elasticsearch Connection Pooling**
Elasticsearch handlers can exhaust connection pools under high load. Implement proper connection pooling with limits and timeouts. Use circuit breakers to prevent cascade failures when Elasticsearch is unavailable.

⚠️ **Pitfall: Configuration Race Conditions**
Concurrent configuration updates can create inconsistent states. Use proper locking and atomic updates. Validate that configuration changes don't conflict with ongoing operations.

⚠️ **Pitfall: Async Queue Memory Leaks**
Unbounded async queues consume memory indefinitely under sustained load. Implement queue size limits, overflow handling, and backpressure mechanisms. Monitor queue depths and implement alerting.

⚠️ **Pitfall: Sampling Bias**
Poor sampling strategies can lose critical debugging information. Ensure error logs are never sampled, maintain complete traces for sampled requests, and implement importance-based sampling that preserves context around failures.


## Glossary

> **Milestone(s):** This section provides essential terminology definitions that support all three milestones, with foundational terms like structured logging and log levels supporting Milestone 1 (Logger Core), JSON formatting and serialization terms supporting Milestone 2 (Structured Output), and correlation ID and context propagation terms supporting Milestone 3 (Context & Correlation).

This glossary provides precise definitions of all key terms used throughout the structured logging system design. Think of this glossary as the shared vocabulary that enables clear communication between team members - just as air traffic controllers use standardized terminology to prevent miscommunication during critical operations, we need consistent definitions to avoid ambiguity when discussing logging system components and behaviors.

Each term is defined with its specific meaning within the context of this logging system, including how it relates to other components and what behaviors it encompasses. The definitions progress from foundational concepts to more specialized terminology, building a comprehensive understanding of the system's vocabulary.

### Core Logging Concepts

**Structured Logging**: A logging approach that produces machine-readable output with consistent, queryable field structure rather than free-form text messages. Unlike traditional string-based logging where information is embedded in unstructured text, structured logging separates message content from contextual metadata, enabling automated log analysis, filtering, and correlation. Each log record contains well-defined fields such as timestamp, level, message, and arbitrary context data serialized in a standardized format like JSON.

**Log Level**: A numeric severity indicator that determines the importance and urgency of a log message. The system uses five standard levels with specific numeric values: `DEBUG` (10) for detailed diagnostic information useful during development, `INFO` (20) for general informational messages about normal system operation, `WARN` (30) for potentially problematic situations that don't prevent continued operation, `ERROR` (40) for error conditions that affect specific operations but don't halt the entire system, and `FATAL` (50) for critical errors that may cause system shutdown. Log levels enable filtering where messages below a configured threshold are suppressed to reduce noise.

**Log Level Filtering**: The mechanism for suppressing log messages below a configured severity threshold to control output volume and focus on relevant information. When a logger's minimum level is set to `WARN`, for example, only messages at `WARN`, `ERROR`, and `FATAL` levels are processed, while `DEBUG` and `INFO` messages are discarded before any formatting or output processing occurs. This filtering happens early in the logging pipeline to minimize performance impact from unwanted messages.

**Logger**: The primary interface through which application code generates log messages, maintaining configuration such as name, minimum level, output handlers, and contextual information. Loggers are organized in a hierarchical tree structure where child loggers inherit configuration from their parents unless explicitly overridden. Each logger processes incoming messages by checking level thresholds, enriching records with context, applying formatting, and dispatching to configured output destinations.

**Logger Hierarchy**: A tree structure of parent-child logger relationships that enables organized configuration management and context inheritance. Loggers are typically named using dot-separated paths like `com.example.service.database` where each segment represents a level in the hierarchy. Child loggers automatically inherit configuration settings like minimum level and output handlers from their parents, but can override these settings locally. This hierarchy prevents configuration duplication and provides logical organization matching application structure.

### Data Structures and Records

**LogRecord**: The fundamental data structure representing a single log event, containing all information needed to format and output the message. A LogRecord includes the timestamp when the event occurred, the numeric log level, the primary message text, the name of the originating logger, and a dictionary of contextual key-value pairs. LogRecords are immutable once created to ensure consistency across multiple output handlers and prevent accidental modification during processing.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | str | ISO 8601 formatted UTC timestamp when the log event occurred |
| `level` | int | Numeric log level indicating message severity (10-50 range) |
| `message` | str | Primary descriptive text explaining what happened |
| `logger_name` | str | Hierarchical name of the logger that created this record |
| `context` | Dict[str, Any] | Additional key-value pairs providing contextual information |

**Context**: A dictionary of key-value pairs that provides additional contextual information about the environment, request, or operation when a log event occurs. Context data automatically propagates through nested function calls and is inherited by child loggers, eliminating the need to manually pass contextual information through method parameters. Context fields commonly include correlation IDs, user identifiers, request URLs, database connection details, and business-specific metadata that aids in log analysis and debugging.

**Correlation ID**: A unique identifier that links related log entries across service boundaries, function calls, and asynchronous operations within the same logical request or transaction. Correlation IDs are typically generated at request entry points and automatically injected into every log record within that request scope. This enables distributed tracing where logs from multiple services, components, or threads can be correlated to understand the complete flow of a user request through the system.

### Output and Formatting

**Handler**: An abstract component responsible for delivering formatted log records to a specific output destination such as standard output, files, remote logging services, or databases. Each handler operates independently with its own error recovery, buffering, and retry logic to ensure that failures in one output destination don't affect others. Handlers can filter records based on level or content, apply destination-specific formatting, and implement circuit breaker patterns to handle temporary outages gracefully.

**Handler Dispatch**: The routing mechanism that delivers log records to one or more configured output handlers, ensuring that each handler receives a copy of the record for independent processing. The dispatch system handles concurrent access safely, manages handler failures through circuit breakers, and maintains buffer queues for temporary storage when handlers are unavailable. Failed handlers are automatically retried with exponential backoff while continuing to serve working destinations.

**Formatter**: A component that converts LogRecord objects into formatted strings suitable for output to specific destinations. Formatters implement different output styles such as single-line JSON for machine processing, colored console output for human readability during development, or custom formats required by specific log aggregation systems. The formatter system supports pluggable architecture where custom formatters can be registered and selected by name.

**JSON Formatter**: A concrete formatter implementation that serializes LogRecord objects as single-line JSON strings with consistent field ordering and proper handling of complex data types. The JSON formatter includes circular reference protection to prevent infinite loops when serializing objects that reference themselves, automatic timestamp formatting, and configurable field inclusion/exclusion. Output is optimized for log aggregation systems that expect one JSON object per line.

**Pretty Formatter**: A human-readable formatter designed for development and debugging that produces colored, indented output optimized for console viewing. Pretty formatters automatically detect terminal color capabilities, apply syntax highlighting to different log levels, format timestamps in human-friendly formats, and optionally pretty-print JSON context data with indentation. This formatter prioritizes readability over machine processing efficiency.

### Context Management and Propagation

**Context Propagation**: The mechanism for automatically carrying contextual information through nested function calls, asynchronous operations, and service boundaries without requiring explicit parameter passing. Context propagation uses thread-local storage for synchronous code and async-local storage for asynchronous operations, ensuring that context fields set at request entry points are available throughout the entire request processing pipeline.

**Ambient Context**: Contextual information that is automatically available anywhere within the current execution scope without being explicitly passed through method parameters. Ambient context includes correlation IDs, user session data, request metadata, and business context that was established earlier in the call chain. This pattern eliminates the need to modify function signatures throughout the codebase when adding new contextual information.

**Context Inheritance**: The behavior where child contexts automatically inherit all key-value pairs from their parent contexts and can add additional fields without affecting the parent. When a function creates a nested logging context, it receives all fields from the current context plus any new fields it adds locally. Changes made in child contexts are isolated and don't propagate back to parent contexts, ensuring proper scoping and preventing side effects.

**Async Context Bridge**: Infrastructure that preserves logging context when crossing asynchronous operation boundaries such as task creation, coroutine execution, and callback invocation. The bridge automatically captures the current context when async operations are initiated and restores that context when the async code executes, ensuring that correlation IDs and contextual information remain available throughout the entire request processing flow regardless of thread switches or event loop scheduling.

### Thread Safety and Concurrency

**Thread Safety**: The property that logging operations produce correct results when called concurrently from multiple threads without external synchronization. Thread-safe logging ensures that log records are not corrupted, context information is not mixed between threads, and internal data structures remain consistent under concurrent access. This is achieved through careful locking, immutable data structures, and atomic operations on shared state.

**Race Condition**: A concurrency bug where the correctness of logging operations depends on the relative timing of thread execution, potentially leading to corrupted output, mixed log records, or inconsistent context information. Common race conditions in logging systems include context corruption when multiple threads modify shared context simultaneously, interleaved output when multiple threads write to the same destination, and inconsistent state when logger configuration changes occur concurrently with logging operations.

**Immutability**: The design principle where LogRecord objects and context dictionaries cannot be modified after creation, preventing accidental changes that could corrupt logs or create race conditions. Immutable objects can be safely shared between multiple handlers and threads without synchronization overhead. When context changes are needed, new immutable objects are created rather than modifying existing ones.

### Error Handling and Recovery

**Circuit Breaker Pattern**: An error handling mechanism that prevents cascade failures by automatically disabling failed handlers and periodically attempting recovery. When a handler experiences consecutive failures exceeding a configured threshold, the circuit breaker opens and stops sending records to that handler while allowing other handlers to continue operating normally. The circuit breaker periodically attempts to close by testing the failed handler, automatically resuming normal operation when the handler recovers.

**Graceful Degradation**: The system's ability to maintain partial functionality when individual components fail, ensuring that logging continues to work even when some output destinations are unavailable. Failed file handlers fall back to console output, failed network handlers buffer records locally for later retry, and serialization failures are logged with simplified fallback representations rather than crashing the application.

**Handler Isolation**: The design principle that failures in one output handler do not affect the operation of other handlers or the logging system as a whole. Each handler maintains independent error state, recovery logic, and buffer management. When one handler fails, other handlers continue processing records normally, and the failed handler can recover independently without affecting system-wide logging operation.

**Safe Serialization**: JSON encoding with protection against edge cases that could cause failures or infinite loops, including circular references, non-serializable objects, and excessively large data structures. Safe serialization detects circular references and replaces them with reference markers, converts non-serializable objects to string representations, and truncates large objects to prevent memory exhaustion while preserving essential information.

### Performance and Optimization

**Buffer Management**: Intelligent queuing of log records when output handlers are temporarily unavailable, with configurable size limits and memory usage controls to prevent unbounded growth. Buffered records are automatically delivered when handlers recover, with oldest records discarded if buffer limits are exceeded. Buffer management includes both in-memory queues for short-term outages and persistent storage for longer-term handler failures.

**Level Filtering Optimization**: Early rejection of log messages below the configured threshold before expensive operations like context enrichment, formatting, or serialization occur. Level filtering uses numeric comparison for efficiency and caches effective levels after walking the logger hierarchy to avoid repeated parent traversal. This optimization ensures that disabled log levels have minimal performance impact.

**Lazy Evaluation**: Deferring expensive operations like complex context serialization, timestamp formatting, or message string formatting until they are actually needed by an active handler. If no handlers are configured or all handlers reject a message due to level filtering, expensive formatting operations are skipped entirely, improving performance for disabled log levels.

### Testing and Debugging

**Milestone Checkpoints**: Concrete verification criteria that validate correct implementation after completing each development stage. Checkpoints specify expected behavior, output formats, performance characteristics, and error handling that should be observed when testing the implemented functionality. Each checkpoint includes specific test commands, expected results, and troubleshooting guidance for common implementation issues.

**Integration Testing**: Comprehensive testing that verifies correct interaction between logging system components under realistic production conditions, including concurrent access, handler failures, context propagation across async boundaries, and end-to-end request tracing. Integration tests simulate real-world scenarios like network outages, disk full conditions, and high-concurrency load to ensure robust system behavior.

**Trace Logging**: An internal debugging system that records the behavior of the logging infrastructure itself, providing visibility into handler dispatch, context propagation, error recovery, and performance characteristics. Trace logging helps diagnose issues like why certain log messages are not appearing, how context flows through the system, and where performance bottlenecks occur during high-volume logging.

### Advanced Features

**Adaptive Sampling**: Intelligent sampling that automatically adjusts log retention rates based on system load, message importance, and available resources to maintain performance while preserving critical information. Adaptive sampling uses algorithms that increase sampling rates for error conditions and correlation contexts while reducing rates for routine operational messages during high-load periods.

**Rate Limiting**: Controlling logging throughput to prevent system overload during high-volume scenarios, using algorithms like token bucket rate limiting that allow burst capacity while maintaining sustainable average rates. Rate limiting can be applied per logger, per level, or per context to prevent specific components from overwhelming the logging system.

**Dynamic Reconfiguration**: The ability to change logging behavior at runtime without restarting the application, including adjusting log levels, adding or removing handlers, modifying formatter settings, and updating context propagation rules. Dynamic reconfiguration enables responsive troubleshooting and performance tuning in production environments without service interruption.

**Bulk Indexing**: Batching multiple log records into single operations for efficient delivery to external systems like Elasticsearch or database storage, reducing network overhead and improving throughput for high-volume logging scenarios. Bulk indexing includes automatic flushing based on time intervals or batch sizes, with retry logic for partial failures.

### Implementation Guidance

The terminology defined above forms the foundation for implementing a production-grade structured logging system. Understanding these concepts and their relationships is essential for making correct design decisions and avoiding common implementation pitfalls.

**Key Term Relationships:**

| Primary Term | Related Terms | Relationship Description |
|--------------|---------------|-------------------------|
| `LogRecord` | `Context`, `LogLevel`, `Formatter` | LogRecord contains Context data, has a LogLevel, processed by Formatters |
| `Logger` | `Logger Hierarchy`, `Handler Dispatch`, `Level Filtering` | Logger participates in hierarchy, dispatches to handlers, filters by level |
| `Context` | `Context Propagation`, `Correlation ID`, `Context Inheritance` | Context propagates through calls, contains correlation ID, supports inheritance |
| `Handler` | `Circuit Breaker`, `Buffer Management`, `Safe Serialization` | Handler uses circuit breaker, manages buffers, performs safe serialization |
| `Formatter` | `JSON Formatter`, `Pretty Formatter`, `Safe Serialization` | Formatters are concrete implementations using safe serialization techniques |

**Common Terminology Mistakes:**

⚠️ **Pitfall: Confusing "Context" with "Configuration"**
Context refers to runtime key-value data that describes the current execution environment (correlation IDs, user session, request metadata), while configuration refers to static settings that control logging behavior (minimum levels, handler settings, formatter choices). Context flows through application execution and changes frequently, while configuration is typically set at startup and changes infrequently.

⚠️ **Pitfall: Misunderstanding "Thread Safety" vs "Concurrent Performance"**
Thread safety means operations produce correct results under concurrent access, which is a correctness requirement. Concurrent performance refers to how well the system scales under concurrent load, which is an efficiency consideration. A system can be thread-safe but perform poorly under concurrency due to excessive locking, or can have good concurrent performance but be thread-unsafe due to race conditions.

⚠️ **Pitfall: Conflating "Structured Logging" with "JSON Format"**
Structured logging is the architectural approach of separating message content from contextual metadata in consistent, queryable field structure. JSON is one possible serialization format for structured logs, but structured logging can also use formats like XML, Protocol Buffers, or custom binary formats. The key is consistent field structure, not the specific serialization mechanism.

This glossary provides the vocabulary foundation needed to understand, implement, and maintain the structured logging system. Refer back to these definitions when encountering unfamiliar terms in other sections of the design document.

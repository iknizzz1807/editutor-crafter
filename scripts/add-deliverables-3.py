#!/usr/bin/env python3
"""Add deliverables to remaining projects missing them (part 3)."""

import yaml
from pathlib import Path

# Deliverables for remaining projects
DELIVERABLES = {
    "ai-agent-framework": {
        "Tool System": [
            "Tool interface definition",
            "Tool registry and lookup",
            "Tool execution with input/output",
            "Built-in tools (search, calculator, code execution)"
        ],
        "ReAct Loop (Reasoning + Acting)": [
            "Thought-Action-Observation cycle",
            "LLM prompt for reasoning",
            "Action parsing from LLM output",
            "Observation feeding back to LLM"
        ],
        "Planning & Task Decomposition": [
            "Task decomposition prompting",
            "Sub-task tracking",
            "Plan execution with re-planning",
            "Goal completion detection"
        ],
        "Memory & Context Management": [
            "Conversation memory storage",
            "Long-term memory (vector store)",
            "Context window management",
            "Memory retrieval for context"
        ],
        "Multi-Agent Collaboration": [
            "Agent-to-agent communication",
            "Role assignment",
            "Task delegation",
            "Result aggregation"
        ]
    },
    "api-gateway": {
        "Reverse Proxy & Routing": [
            "Request routing rules",
            "Path-based routing",
            "Header-based routing",
            "Load balancing to backends"
        ],
        "Request/Response Transformation": [
            "Header manipulation",
            "Request body transformation",
            "Response modification",
            "Protocol translation"
        ],
        "Authentication & Authorization Layer": [
            "JWT validation",
            "API key authentication",
            "OAuth token introspection",
            "Rate limiting per client"
        ],
        "Observability & Plugins": [
            "Request/response logging",
            "Metrics collection",
            "Distributed tracing headers",
            "Plugin system for extensibility"
        ]
    },
    "audit-logging": {
        "Audit Event Model": [
            "Event schema (who, what, when, where)",
            "Event serialization",
            "Actor and resource identification",
            "Event metadata"
        ],
        "Immutable Storage with Hash Chain": [
            "Append-only event storage",
            "Hash chain for tamper detection",
            "Event ordering guarantees",
            "Storage encryption"
        ],
        "Audit Query & Export": [
            "Time-range queries",
            "Actor/resource filtering",
            "Full-text search in events",
            "Export to compliance formats"
        ]
    },
    "build-raytracer": {
        "Output an Image": [
            "PPM image format output",
            "Pixel buffer management",
            "Color representation (RGB)",
            "Image file writing"
        ],
        "Ray Class and Background": [
            "Ray class (origin, direction)",
            "Ray point-at-parameter function",
            "Background gradient rendering",
            "Camera ray generation"
        ],
        "Sphere Intersection": [
            "Ray-sphere intersection math",
            "Hit record structure",
            "Intersection point and normal",
            "Front face determination"
        ],
        "Surface Normals and Multiple Objects": [
            "Normal vector calculation",
            "Normal visualization",
            "Hittable list for multiple objects",
            "Closest hit selection"
        ],
        "Antialiasing": [
            "Multiple samples per pixel",
            "Random ray offset within pixel",
            "Color averaging",
            "Image quality improvement"
        ],
        "Diffuse Materials": [
            "Random scatter direction",
            "Lambertian reflection",
            "Recursive ray tracing",
            "Gamma correction"
        ],
        "Metal and Reflections": [
            "Metal material class",
            "Reflection ray calculation",
            "Fuzziness parameter",
            "Material polymorphism"
        ],
        "Dielectrics (Glass)": [
            "Refraction (Snell's law)",
            "Fresnel equations (Schlick approximation)",
            "Total internal reflection",
            "Glass material implementation"
        ],
        "Positionable Camera": [
            "Camera positioning (lookfrom, lookat)",
            "Field of view control",
            "Up vector handling",
            "Aspect ratio configuration"
        ],
        "Depth of Field": [
            "Aperture and focus distance",
            "Lens ray offset",
            "Defocus blur effect",
            "Bokeh simulation"
        ]
    },
    "build-redis": {
        "TCP Server + RESP Protocol": [
            "TCP socket server",
            "RESP protocol parsing",
            "Command/response formatting",
            "Connection handling"
        ],
        "GET/SET/DEL Commands": [
            "In-memory key-value storage",
            "GET command implementation",
            "SET command implementation",
            "DEL command implementation"
        ],
        "Expiration (TTL)": [
            "EXPIRE command",
            "TTL storage per key",
            "Lazy expiration on access",
            "Active expiration background task"
        ],
        "Data Structures (List, Set, Hash)": [
            "List operations (LPUSH, RPUSH, LPOP, RPOP)",
            "Set operations (SADD, SREM, SMEMBERS)",
            "Hash operations (HSET, HGET, HDEL)",
            "Type checking per key"
        ],
        "Persistence (RDB Snapshots)": [
            "RDB file format",
            "BGSAVE background save",
            "RDB loading on startup",
            "Snapshot scheduling"
        ],
        "Persistence (AOF)": [
            "Append-only file writing",
            "Command logging",
            "AOF rewrite for compaction",
            "AOF loading on startup"
        ],
        "Pub/Sub": [
            "SUBSCRIBE command",
            "PUBLISH command",
            "Channel management",
            "Pattern subscriptions (PSUBSCRIBE)"
        ],
        "Cluster Mode (Sharding)": [
            "Hash slot assignment",
            "Cluster node communication",
            "MOVED/ASK redirections",
            "Cluster topology management"
        ]
    },
    "build-regex": {
        "Regex Parser": [
            "Regex tokenizer",
            "AST for regex patterns",
            "Operator precedence (|, concat, *, +, ?)",
            "Character class parsing"
        ],
        "Thompson's Construction (NFA)": [
            "NFA state and transition",
            "Epsilon transitions",
            "Concatenation NFA construction",
            "Alternation and Kleene star NFA"
        ],
        "NFA Simulation": [
            "Epsilon closure computation",
            "State set tracking",
            "Input character consumption",
            "Accept state detection"
        ],
        "DFA Conversion & Optimization": [
            "Subset construction algorithm",
            "DFA state minimization",
            "DFA table representation",
            "Fast matching with DFA"
        ]
    },
    "build-shell": {
        "Basic REPL and Command Execution": [
            "Read-eval-print loop",
            "Command line parsing",
            "fork() and execvp()",
            "Exit status handling"
        ],
        "Built-in Commands": [
            "cd command",
            "exit command",
            "export command",
            "echo command"
        ],
        "I/O Redirection": [
            "Output redirection (>)",
            "Input redirection (<)",
            "Append redirection (>>)",
            "File descriptor duplication"
        ],
        "Pipes": [
            "Pipe creation",
            "Pipeline execution",
            "File descriptor management",
            "Multi-stage pipelines"
        ],
        "Background Jobs": [
            "Background execution (&)",
            "Job table",
            "jobs command",
            "Process group management"
        ],
        "Job Control (fg, bg, Ctrl+Z)": [
            "SIGTSTP handling",
            "fg command",
            "bg command",
            "Signal forwarding"
        ]
    },
    "build-sqlite": {
        "SQL Tokenizer": [
            "Keyword recognition",
            "Identifier tokenization",
            "Literal parsing (strings, numbers)",
            "Operator tokenization"
        ],
        "SQL Parser (AST)": [
            "SELECT statement AST",
            "Expression AST",
            "Table and column references",
            "Parser error handling"
        ],
        "B-tree Page Format": [
            "Page header structure",
            "Cell format (key-value)",
            "Cell pointer array",
            "Overflow pages"
        ],
        "Table Storage": [
            "Table to B-tree mapping",
            "Row serialization",
            "Rowid management",
            "Schema storage"
        ],
        "SELECT Execution (Table Scan)": [
            "Table scan operator",
            "Row deserialization",
            "Column projection",
            "Result set building"
        ],
        "INSERT/UPDATE/DELETE": [
            "INSERT execution",
            "UPDATE execution",
            "DELETE execution",
            "B-tree modifications"
        ],
        "WHERE Clause and Indexes": [
            "WHERE evaluation",
            "Index scan operator",
            "Index creation",
            "Query optimization basics"
        ],
        "Query Planner": [
            "Plan enumeration",
            "Cost estimation",
            "Index selection",
            "Join ordering"
        ],
        "Transactions (BEGIN/COMMIT/ROLLBACK)": [
            "Transaction state",
            "Rollback journal",
            "Commit processing",
            "ACID guarantees"
        ],
        "WAL Mode": [
            "WAL file format",
            "Write to WAL",
            "Checkpoint mechanism",
            "Reader-writer concurrency"
        ]
    },
    "build-tcp-stack": {
        "Ethernet & ARP": [
            "Ethernet frame parsing",
            "ARP request/response",
            "ARP cache management",
            "MAC address resolution"
        ],
        "IP & ICMP": [
            "IP packet parsing/construction",
            "IP routing (simple)",
            "ICMP echo (ping)",
            "IP checksum"
        ],
        "TCP Connection Management": [
            "TCP state machine",
            "Three-way handshake",
            "Connection termination",
            "Timeout handling"
        ],
        "TCP Data Transfer & Flow Control": [
            "Sequence number management",
            "Sliding window",
            "Retransmission",
            "Congestion control basics"
        ]
    },
    "build-test-framework": {
        "Test Discovery & Execution": [
            "Test function detection",
            "Test file scanning",
            "Test execution runner",
            "Parallel execution option"
        ],
        "Assertions & Matchers": [
            "assertEqual, assertTrue",
            "assertRaises",
            "Custom matchers",
            "Assertion messages"
        ],
        "Fixtures & Setup/Teardown": [
            "Setup/teardown hooks",
            "Fixture injection",
            "Scope levels (function, class, module)",
            "Fixture dependencies"
        ],
        "Reporting & CLI": [
            "Pass/fail reporting",
            "Failure details",
            "CLI argument parsing",
            "JUnit XML output"
        ]
    },
    "build-text-editor": {
        "Raw Mode and Input": [
            "Terminal raw mode setup",
            "Keyboard input reading",
            "Special key handling",
            "Input buffering"
        ],
        "Screen Refresh": [
            "ANSI escape sequences",
            "Screen clearing",
            "Cursor positioning",
            "Efficient redraw"
        ],
        "File Viewing": [
            "File loading",
            "Scrolling (vertical, horizontal)",
            "Line number display",
            "Status bar"
        ],
        "Text Editing": [
            "Character insertion",
            "Character deletion",
            "Line operations",
            "Cursor movement"
        ],
        "Save and Undo": [
            "File saving",
            "Dirty flag tracking",
            "Undo stack",
            "Redo support"
        ],
        "Search": [
            "Incremental search",
            "Search highlighting",
            "Next/previous match",
            "Search and replace"
        ],
        "Syntax Highlighting": [
            "Token-based highlighting",
            "Language detection",
            "Color scheme",
            "Multi-line syntax"
        ]
    },
    "build-transformer": {
        "Self-Attention": [
            "Query, Key, Value projection",
            "Attention score computation",
            "Softmax and weighted sum",
            "Masked attention for decoder"
        ],
        "Transformer Block": [
            "Multi-head attention",
            "Feed-forward network",
            "Layer normalization",
            "Residual connections"
        ],
        "Training Pipeline": [
            "Data loading and batching",
            "Loss computation (cross-entropy)",
            "Optimizer (Adam)",
            "Learning rate scheduling"
        ],
        "Text Generation": [
            "Tokenization integration",
            "Autoregressive generation",
            "Sampling strategies",
            "Beam search (optional)"
        ]
    },
    "bytecode-vm": {
        "Instruction Set Design": [
            "Opcode enumeration",
            "Instruction format",
            "Operand encoding",
            "Instruction documentation"
        ],
        "Stack-Based Execution": [
            "Operand stack",
            "Push/pop operations",
            "Arithmetic instructions",
            "Comparison instructions"
        ],
        "Control Flow": [
            "Jump instructions",
            "Conditional jumps",
            "Loop support",
            "Label resolution"
        ],
        "Variables and Functions": [
            "Local variable slots",
            "Call instruction",
            "Return instruction",
            "Stack frames"
        ]
    },
    "cache-optimized-structures": {
        "Cache Fundamentals & Benchmarking": [
            "Cache hierarchy understanding",
            "Performance counter reading",
            "Benchmark harness",
            "Cache miss measurement"
        ],
        "Array of Structs vs Struct of Arrays": [
            "AoS implementation",
            "SoA implementation",
            "Access pattern comparison",
            "Performance benchmarking"
        ],
        "Cache-Friendly Hash Table": [
            "Open addressing with linear probing",
            "Cache line alignment",
            "Prefetching hints",
            "Comparison with chaining"
        ],
        "Cache-Oblivious B-Tree": [
            "Van Emde Boas layout",
            "Recursive layout construction",
            "Search implementation",
            "Performance analysis"
        ],
        "Blocked Matrix Operations": [
            "Blocked matrix multiplication",
            "Block size tuning",
            "Cache reuse optimization",
            "SIMD integration (optional)"
        ]
    },
    "calculator-parser": {
        "Basic Arithmetic": [
            "Number parsing",
            "Addition and subtraction",
            "Multiplication and division",
            "Parentheses grouping"
        ],
        "Unary and Power": [
            "Unary minus",
            "Exponentiation",
            "Right associativity for power",
            "Operator precedence"
        ],
        "Variables and Functions": [
            "Variable assignment",
            "Variable lookup",
            "Built-in functions (sin, cos, sqrt)",
            "Function call parsing"
        ]
    },
    "cd-deployment": {
        "Dual Environment Setup": [
            "Blue and green environment provisioning",
            "Environment health endpoints",
            "Environment isolation",
            "Configuration management"
        ],
        "Load Balancer Switching": [
            "Load balancer configuration",
            "Traffic switching mechanism",
            "Health check validation",
            "Instant rollback capability"
        ],
        "Deployment Automation": [
            "Deployment script/pipeline",
            "Artifact deployment to inactive",
            "Smoke test automation",
            "Switch trigger"
        ],
        "Rollback & Database Migrations": [
            "Rollback automation",
            "Database migration strategy",
            "Backward compatible migrations",
            "Migration rollback"
        ]
    },
    "cdc-system": {
        "Log Parsing & Change Events": [
            "Database log connection",
            "Log event parsing",
            "Change event construction",
            "Position/offset tracking"
        ],
        "Event Streaming & Delivery": [
            "Kafka/queue integration",
            "At-least-once delivery",
            "Event ordering per table",
            "Backpressure handling"
        ],
        "Schema Evolution & Compatibility": [
            "Schema registry integration",
            "Schema versioning",
            "Compatibility checking",
            "Schema migration events"
        ]
    },
    "cdn-implementation": {
        "Edge Cache Implementation": [
            "Cache storage layer",
            "Cache key generation",
            "TTL management",
            "Cache hit/miss handling"
        ],
        "Cache Invalidation": [
            "Purge by URL",
            "Purge by tag/surrogate key",
            "Soft purge (stale-while-revalidate)",
            "Invalidation propagation"
        ],
        "Origin Shield & Request Collapsing": [
            "Origin shield layer",
            "Request collapsing for concurrent requests",
            "Cache fill optimization",
            "Origin protection"
        ]
    },
    "chaos-engineering": {
        "Fault Injection Framework": [
            "Fault type definitions",
            "Target selection (pod, service, network)",
            "Fault scheduling",
            "Fault cleanup"
        ],
        "Experiment Orchestration": [
            "Experiment definition",
            "Steady state hypothesis",
            "Experiment execution",
            "Result analysis"
        ],
        "GameDay Automation": [
            "Scenario scripting",
            "Automated checks",
            "Incident simulation",
            "Recovery validation"
        ]
    },
    "chatbot-intent": {
        "Intent Classification": [
            "Training data preparation",
            "Intent classifier model",
            "Multi-intent handling",
            "Confidence thresholds"
        ],
        "Entity Extraction": [
            "Named entity recognition",
            "Slot filling",
            "Entity normalization",
            "Custom entity types"
        ],
        "Dialog Management": [
            "Dialog state tracking",
            "Context management",
            "Multi-turn conversation",
            "Slot confirmation"
        ],
        "Response Generation": [
            "Template-based responses",
            "Dynamic content insertion",
            "Fallback handling",
            "Response variation"
        ]
    },
    "ci-cd-pipeline": {
        "Pipeline Definition Parser": [
            "YAML pipeline parsing",
            "Stage and job definitions",
            "Dependency graph construction",
            "Variable substitution"
        ],
        "Job Executor": [
            "Docker container execution",
            "Script step runner",
            "Environment setup",
            "Output capture"
        ],
        "Artifact Management": [
            "Artifact upload after job",
            "Artifact download between jobs",
            "Artifact storage backend",
            "Artifact retention"
        ],
        "Deployment Strategies": [
            "Rolling deployment",
            "Blue-green deployment",
            "Canary deployment",
            "Rollback automation"
        ]
    },
    "circuit-breaker": {
        "Basic Circuit Breaker": [
            "State machine (closed, open, half-open)",
            "Failure threshold tracking",
            "Timeout for open state",
            "Success threshold for recovery"
        ],
        "Advanced Features": [
            "Bulkhead pattern",
            "Fallback functions",
            "Metrics collection",
            "Configuration per service"
        ],
        "Integration & Testing": [
            "HTTP client integration",
            "Decorator/wrapper pattern",
            "Integration tests",
            "Chaos testing"
        ]
    },
    "config-parser": {
        "INI Parser": [
            "Section parsing",
            "Key-value parsing",
            "Comment handling",
            "Multi-line values"
        ],
        "TOML Tokenizer": [
            "Token types for TOML",
            "String literal handling",
            "Date/time parsing",
            "Array and table tokens"
        ],
        "TOML Parser": [
            "Table parsing",
            "Array of tables",
            "Inline tables and arrays",
            "Value type inference"
        ],
        "YAML Subset Parser": [
            "Indentation-based structure",
            "Mapping parsing",
            "Sequence parsing",
            "Scalar types"
        ]
    },
    "container-basic": {
        "Process Namespace": [
            "PID namespace creation",
            "Process isolation verification",
            "Init process in namespace",
            "Namespace unsharing"
        ],
        "Mount Namespace": [
            "Mount namespace creation",
            "Root filesystem setup",
            "Bind mounts",
            "Proc/sys mounting"
        ],
        "Network Namespace": [
            "Network namespace creation",
            "Veth pair setup",
            "Bridge networking",
            "NAT configuration"
        ],
        "Cgroups (Resource Limits)": [
            "Cgroup creation",
            "Memory limit",
            "CPU limit",
            "Process limit"
        ]
    },
    "container-runtime": {
        "Process Isolation with Namespaces": [
            "PID namespace setup",
            "Mount namespace setup",
            "Network namespace setup",
            "UTS and IPC namespaces"
        ],
        "Resource Limits with Cgroups": [
            "Cgroup controller setup",
            "Memory limits",
            "CPU limits",
            "Device access control"
        ],
        "Overlay Filesystem": [
            "OverlayFS mount",
            "Layer management",
            "Copy-on-write",
            "Layer caching"
        ],
        "Container Networking": [
            "Veth pair creation",
            "Bridge setup",
            "IP assignment",
            "Port forwarding"
        ]
    },
    "disassembler": {
        "Binary File Loading": [
            "ELF/PE header parsing",
            "Section identification",
            "Entry point detection",
            "Symbol table loading"
        ],
        "Instruction Prefixes": [
            "REX prefix handling",
            "Operand size prefix",
            "Address size prefix",
            "Segment override"
        ],
        "Opcode Tables": [
            "Primary opcode lookup",
            "Extended opcode (0F prefix)",
            "VEX/EVEX prefixes",
            "Instruction metadata"
        ],
        "ModRM and SIB Decoding": [
            "ModRM byte parsing",
            "Register operand decoding",
            "Memory operand decoding",
            "SIB byte handling"
        ],
        "Output Formatting": [
            "Intel syntax output",
            "AT&T syntax option",
            "Address and bytes display",
            "Symbol resolution"
        ]
    },
    "distributed-cache": {
        "Consistent Hash Ring": [
            "Hash ring implementation",
            "Virtual nodes",
            "Key to node mapping",
            "Ring rebalancing"
        ],
        "Cache Node Implementation": [
            "LRU cache per node",
            "Get/set/delete operations",
            "TTL support",
            "Memory limit enforcement"
        ],
        "Cluster Communication": [
            "Node discovery",
            "Gossip protocol",
            "Request routing",
            "Node health checking"
        ],
        "Replication & Consistency": [
            "Replication factor configuration",
            "Read/write quorum",
            "Consistency level options",
            "Conflict resolution"
        ]
    },
    "distributed-tracing": {
        "Trace Context & Propagation": [
            "Trace ID generation",
            "Span ID generation",
            "Context propagation (W3C/B3)",
            "Baggage items"
        ],
        "Span Recording": [
            "Span start/finish",
            "Span attributes",
            "Span events/logs",
            "Error recording"
        ],
        "Collector & Storage": [
            "Span ingestion API",
            "Storage backend",
            "Sampling strategies",
            "Batch processing"
        ],
        "Query & Visualization": [
            "Trace search API",
            "Trace timeline view",
            "Service dependency map",
            "Latency analysis"
        ]
    },
    "ecommerce-basic": {
        "Product Catalog": [
            "Product model",
            "Category hierarchy",
            "Product listing API",
            "Product search"
        ],
        "Shopping Cart": [
            "Cart session management",
            "Add/remove items",
            "Quantity updates",
            "Cart persistence"
        ],
        "User Authentication": [
            "User registration",
            "Login/logout",
            "Password hashing",
            "Session management"
        ],
        "Checkout Process": [
            "Address collection",
            "Order creation",
            "Payment integration stub",
            "Order confirmation"
        ]
    },
    "ecs-arch": {
        "Entity Manager": [
            "Entity ID generation",
            "Entity creation/destruction",
            "Entity lookup",
            "Entity iteration"
        ],
        "Component Storage": [
            "Component type registry",
            "Sparse set storage",
            "Component add/remove",
            "Component lookup"
        ],
        "System Interface": [
            "System base class",
            "Component queries",
            "System execution order",
            "System dependencies"
        ],
        "Archetypes (Optional Advanced)": [
            "Archetype identification",
            "Archetype-based storage",
            "Archetype transitions",
            "Cache-friendly iteration"
        ]
    },
    "etl-pipeline": {
        "Pipeline DAG Definition": [
            "Task definition",
            "Dependency specification",
            "DAG validation",
            "DAG visualization"
        ],
        "Data Extraction & Loading": [
            "Source connectors (DB, API, file)",
            "Destination connectors",
            "Incremental extraction",
            "Bulk loading"
        ],
        "Data Transformations": [
            "Transform functions",
            "Data cleaning",
            "Schema mapping",
            "Aggregations"
        ],
        "Pipeline Orchestration & Monitoring": [
            "Scheduler integration",
            "Task execution tracking",
            "Failure handling",
            "Monitoring dashboard"
        ]
    },
    "event-sourcing": {
        "Event Store": [
            "Event append operation",
            "Event stream per aggregate",
            "Event versioning",
            "Event serialization"
        ],
        "Aggregate & Event Sourcing": [
            "Aggregate base class",
            "Event application",
            "Command handling",
            "Aggregate loading from events"
        ],
        "Projections": [
            "Projection handlers",
            "Read model updates",
            "Projection rebuilding",
            "Eventual consistency"
        ],
        "Snapshots": [
            "Snapshot creation",
            "Snapshot storage",
            "Loading with snapshots",
            "Snapshot scheduling"
        ]
    },
    "feature-flags": {
        "Flag Evaluation Engine": [
            "Flag definition storage",
            "Evaluation logic",
            "Context-based targeting",
            "Default values"
        ],
        "Real-time Flag Updates": [
            "Server-sent events or polling",
            "Flag change notification",
            "Client SDK updates",
            "Cache invalidation"
        ],
        "Flag Analytics & Experiments": [
            "Flag exposure logging",
            "A/B test assignment",
            "Metrics collection",
            "Experiment analysis"
        ]
    },
    "file-upload-service": {
        "Chunked Upload Protocol": [
            "Chunk upload endpoint",
            "Chunk ordering",
            "Resume interrupted uploads",
            "Chunk assembly"
        ],
        "Storage Abstraction": [
            "Storage interface",
            "Local filesystem backend",
            "S3-compatible backend",
            "Storage selection"
        ],
        "Virus Scanning & Validation": [
            "File type validation",
            "Size limits",
            "Virus scan integration",
            "Quarantine handling"
        ]
    },
    "gossip-protocol": {
        "Peer Management": [
            "Peer list maintenance",
            "Peer discovery",
            "Peer state tracking",
            "Peer removal"
        ],
        "Push Gossip": [
            "Random peer selection",
            "State delta calculation",
            "Push message sending",
            "Infection spread"
        ],
        "Pull Gossip & Anti-Entropy": [
            "Pull request/response",
            "State reconciliation",
            "Anti-entropy repair",
            "Consistency convergence"
        ],
        "Failure Detection": [
            "Heartbeat mechanism",
            "Failure suspicion",
            "SWIM protocol basics",
            "Failure dissemination"
        ]
    },
    "grpc-service": {
        "Proto Definition & Code Generation": [
            "Protocol buffer definition",
            "Code generation setup",
            "Message types",
            "Service definition"
        ],
        "Server Implementation": [
            "gRPC server setup",
            "Service implementation",
            "Request handling",
            "Error responses"
        ],
        "Interceptors & Middleware": [
            "Server interceptors",
            "Logging interceptor",
            "Authentication interceptor",
            "Metrics interceptor"
        ],
        "Client & Testing": [
            "gRPC client setup",
            "Client calls",
            "Integration tests",
            "Mock server for testing"
        ]
    },
    "hash-impl": {
        "Message Preprocessing": [
            "Padding to block size",
            "Length encoding",
            "Block parsing",
            "Endianness handling"
        ],
        "Message Schedule": [
            "Word expansion",
            "Schedule array construction",
            "Bit rotation functions",
            "XOR operations"
        ],
        "Compression Function": [
            "Working variables initialization",
            "Round function iteration",
            "Bitwise operations",
            "Intermediate hash update"
        ],
        "Final Hash Output": [
            "Final hash construction",
            "Hex string formatting",
            "Byte array output",
            "Test vector verification"
        ]
    },
    "hexdump": {
        "Basic Hex Output": [
            "File reading in chunks",
            "Byte to hex conversion",
            "Offset display",
            "Byte grouping"
        ],
        "ASCII Column": [
            "Printable character display",
            "Non-printable replacement",
            "Column alignment",
            "Line formatting"
        ],
        "Grouped Output": [
            "2-byte grouping option",
            "4-byte grouping option",
            "Endianness option",
            "Custom grouping"
        ],
        "CLI Options": [
            "Argument parsing",
            "Length limit option",
            "Skip offset option",
            "Output format selection"
        ]
    },
    "http-server-basic": {
        "TCP Server Basics": [
            "Socket creation and binding",
            "Connection acceptance",
            "Client handling",
            "Connection cleanup"
        ],
        "HTTP Request Parsing": [
            "Request line parsing",
            "Header parsing",
            "Body handling",
            "Error responses"
        ],
        "Static File Serving": [
            "File reading",
            "MIME type detection",
            "404 handling",
            "Directory listing"
        ],
        "Concurrent Connections": [
            "Thread per connection",
            "Thread pool option",
            "Non-blocking I/O option",
            "Connection limits"
        ]
    },
    "http2-server": {
        "Binary Framing": [
            "Frame header parsing",
            "Frame types (DATA, HEADERS, etc.)",
            "Frame serialization",
            "Frame validation"
        ],
        "HPACK Compression": [
            "Static table",
            "Dynamic table",
            "Header encoding",
            "Header decoding"
        ],
        "Stream Management": [
            "Stream state machine",
            "Stream prioritization",
            "Stream creation",
            "RST_STREAM handling"
        ],
        "Flow Control": [
            "Window updates",
            "Connection-level flow control",
            "Stream-level flow control",
            "Backpressure handling"
        ]
    },
    "https-client": {
        "TCP Socket & Record Layer": [
            "TCP connection establishment",
            "TLS record layer",
            "Record framing",
            "Record types"
        ],
        "ClientHello": [
            "ClientHello construction",
            "Cipher suite list",
            "Extensions",
            "ServerHello parsing"
        ],
        "Key Exchange": [
            "ECDHE key exchange",
            "Key derivation",
            "Certificate validation",
            "Finished message"
        ],
        "Encrypted Communication": [
            "Application data encryption",
            "Request sending",
            "Response receiving",
            "Connection close"
        ]
    },
    "infrastructure-as-code": {
        "Configuration Parser": [
            "HCL/YAML parsing",
            "Resource block extraction",
            "Variable interpolation",
            "Module support"
        ],
        "State Management": [
            "State file format",
            "State locking",
            "State diff computation",
            "Remote state backend"
        ],
        "Dependency Graph & Planning": [
            "Resource dependency extraction",
            "DAG construction",
            "Plan generation",
            "Plan preview"
        ],
        "Provider Abstraction": [
            "Provider interface",
            "CRUD operations",
            "Provider configuration",
            "Provider plugins"
        ]
    },
    "integration-testing": {
        "Test Database Setup": [
            "Database container setup",
            "Schema migrations",
            "Test data seeding",
            "Database cleanup"
        ],
        "API Integration Tests": [
            "HTTP client for tests",
            "Request/response assertions",
            "Authentication in tests",
            "Error case testing"
        ],
        "External Service Mocking": [
            "Mock server setup",
            "Response stubbing",
            "Request verification",
            "Network isolation"
        ]
    },
    "job-scheduler": {
        "Cron Expression Parser": [
            "Cron field parsing",
            "Next run time calculation",
            "Cron validation",
            "Human-readable output"
        ],
        "Job Queue with Priorities": [
            "Priority queue implementation",
            "Job submission",
            "Job dequeue by priority",
            "Delayed job support"
        ],
        "Worker Coordination": [
            "Worker registration",
            "Job assignment",
            "Worker heartbeat",
            "Failed worker handling"
        ]
    },
    "jwt-impl": {
        "JWT Structure": [
            "Header encoding",
            "Payload encoding",
            "Base64URL encoding",
            "Token assembly"
        ],
        "HMAC Signing": [
            "HMAC-SHA256 implementation",
            "Signature generation",
            "Signature verification",
            "Secret key handling"
        ],
        "Claims Validation": [
            "Expiration (exp) check",
            "Not before (nbf) check",
            "Issuer (iss) validation",
            "Custom claims"
        ]
    },
    "knn": {
        "Distance Calculation": [
            "Euclidean distance",
            "Manhattan distance",
            "Cosine similarity",
            "Distance matrix"
        ],
        "K-Nearest Neighbors Classification": [
            "K selection",
            "Neighbor finding",
            "Majority voting",
            "Weighted voting"
        ],
        "Improvements & Evaluation": [
            "Cross-validation",
            "K optimization",
            "KD-tree acceleration",
            "Accuracy metrics"
        ]
    },
    "leader-election": {
        "Node Communication": [
            "Node discovery",
            "Message passing",
            "Network partition handling",
            "Node failure detection"
        ],
        "Bully Algorithm": [
            "Election initiation",
            "Higher ID check",
            "Coordinator announcement",
            "Election timeout"
        ],
        "Ring Election": [
            "Ring topology",
            "Election message passing",
            "Coordinator selection",
            "Ring repair"
        ]
    },
    "linear-regression": {
        "Simple Linear Regression": [
            "Data loading",
            "Closed-form solution",
            "Prediction function",
            "R-squared calculation"
        ],
        "Gradient Descent": [
            "Cost function",
            "Gradient computation",
            "Parameter update",
            "Convergence check"
        ],
        "Multiple Linear Regression": [
            "Feature matrix construction",
            "Multi-variable gradient descent",
            "Feature normalization",
            "Regularization (optional)"
        ]
    },
    "lisp-interp": {
        "S-Expression Parser": [
            "Tokenizer for parens and atoms",
            "Recursive descent parser",
            "List construction",
            "Quote handling"
        ],
        "Basic Evaluation": [
            "Number evaluation",
            "Arithmetic primitives (+, -, *, /)",
            "Comparison primitives",
            "Boolean primitives"
        ],
        "Variables and Functions": [
            "Environment with bindings",
            "define form",
            "lambda form",
            "Function application"
        ],
        "List Operations & Recursion": [
            "car, cdr, cons primitives",
            "List construction",
            "Recursive function support",
            "Tail call optimization (optional)"
        ]
    },
    "llm-eval-framework": {
        "Dataset Management": [
            "Dataset loading formats",
            "Test case structure",
            "Ground truth storage",
            "Dataset versioning"
        ],
        "Evaluation Metrics": [
            "Exact match",
            "BLEU/ROUGE scores",
            "Semantic similarity",
            "Custom metric plugins"
        ],
        "Evaluation Runner": [
            "Model API integration",
            "Batch evaluation",
            "Progress tracking",
            "Result caching"
        ],
        "Reporting & Analysis": [
            "Score aggregation",
            "Error analysis",
            "Comparison reports",
            "Visualization"
        ]
    },
    "load-balancer-basic": {
        "HTTP Proxy Foundation": [
            "Request forwarding",
            "Response forwarding",
            "Header manipulation",
            "Connection handling"
        ],
        "Round Robin Distribution": [
            "Backend list management",
            "Round robin selection",
            "Request routing",
            "Backend cycling"
        ],
        "Health Checks": [
            "Health check endpoint",
            "Periodic checking",
            "Unhealthy backend removal",
            "Recovery detection"
        ],
        "Additional Algorithms": [
            "Least connections",
            "Weighted round robin",
            "IP hash",
            "Random selection"
        ]
    },
    "load-testing-framework": {
        "Virtual User Simulation": [
            "User scenario definition",
            "Request execution",
            "Think time simulation",
            "Session handling"
        ],
        "Distributed Workers": [
            "Worker node setup",
            "Work distribution",
            "Result collection",
            "Coordinator node"
        ],
        "Real-time Metrics & Reporting": [
            "Response time tracking",
            "Throughput calculation",
            "Error rate tracking",
            "Live dashboard"
        ]
    },
    "log-aggregator": {
        "Log Ingestion": [
            "HTTP log receiver",
            "Syslog receiver",
            "File tail agent",
            "Log parsing"
        ],
        "Log Index": [
            "Inverted index for logs",
            "Time-based partitioning",
            "Field extraction",
            "Index storage"
        ],
        "Log Query Engine": [
            "Query language",
            "Full-text search",
            "Field filtering",
            "Time range queries"
        ]
    },
    "logging-structured": {
        "Logger Core": [
            "Log level management",
            "Logger hierarchy",
            "Log record creation",
            "Handler dispatch"
        ],
        "Structured Output": [
            "JSON formatting",
            "Key-value pairs",
            "Timestamp formatting",
            "Custom formatters"
        ],
        "Context & Correlation": [
            "Context storage (thread-local)",
            "Correlation ID propagation",
            "Request context",
            "Async context"
        ]
    },
    "markdown-renderer": {
        "Block Elements": [
            "Heading parsing (#)",
            "Paragraph detection",
            "Code block (fenced)",
            "Blockquote"
        ],
        "Inline Elements": [
            "Bold and italic",
            "Inline code",
            "Links",
            "Images"
        ],
        "Lists": [
            "Unordered lists",
            "Ordered lists",
            "Nested lists",
            "List item content"
        ],
        "HTML Generation": [
            "HTML tag generation",
            "Escaping special characters",
            "Pretty printing",
            "Custom renderers"
        ]
    },
    "media-processing": {
        "Image Processing": [
            "Image loading",
            "Resize and crop",
            "Format conversion",
            "Thumbnail generation"
        ],
        "Video Transcoding": [
            "FFmpeg integration",
            "Codec selection",
            "Bitrate control",
            "Resolution scaling"
        ],
        "Processing Queue & Progress": [
            "Job queue",
            "Progress tracking",
            "Webhook notifications",
            "Error handling"
        ]
    },
    "memory-pool": {
        "Fixed-Size Pool": [
            "Pool initialization",
            "Block allocation",
            "Block deallocation",
            "Free list management"
        ],
        "Pool Growing": [
            "Automatic pool expansion",
            "Chunk allocation",
            "Memory tracking",
            "Pool statistics"
        ],
        "Thread Safety & Debugging": [
            "Lock-free allocation option",
            "Per-thread pools",
            "Leak detection",
            "Double-free detection"
        ]
    },
    "metrics-collector": {
        "Metrics Data Model": [
            "Metric types (counter, gauge, histogram)",
            "Label support",
            "Timestamp handling",
            "Metric metadata"
        ],
        "Scrape Engine": [
            "Target discovery",
            "HTTP scraping",
            "Scrape interval",
            "Scrape timeout"
        ],
        "Time Series Storage": [
            "Time series data structure",
            "Compression",
            "Retention policies",
            "Index for queries"
        ],
        "Query Engine": [
            "PromQL-style queries",
            "Range queries",
            "Aggregations",
            "Label matching"
        ]
    },
    "metrics-dashboard": {
        "Metrics Collection": [
            "Metric ingestion API",
            "Push vs pull model",
            "Metric types support",
            "Label handling"
        ],
        "Storage & Querying": [
            "Time-series storage",
            "Query language",
            "Aggregation functions",
            "Downsampling"
        ],
        "Visualization Dashboard": [
            "Dashboard configuration",
            "Chart rendering",
            "Real-time updates",
            "Dashboard sharing"
        ],
        "Alerting System": [
            "Alert rules",
            "Alert evaluation",
            "Notification channels",
            "Alert silencing"
        ]
    },
    "multiplayer-game-server": {
        "Game Loop & Tick System": [
            "Fixed timestep loop",
            "Game state updates",
            "Tick rate configuration",
            "State snapshots"
        ],
        "Client Prediction & Reconciliation": [
            "Client-side prediction",
            "Server authoritative state",
            "State reconciliation",
            "Smoothing corrections"
        ],
        "Lag Compensation": [
            "Server-side rewind",
            "Hit detection with lag",
            "Time synchronization",
            "Interpolation"
        ],
        "State Synchronization": [
            "Delta compression",
            "Priority-based updates",
            "Reliable vs unreliable",
            "Bandwidth optimization"
        ]
    },
    "neural-network-basic": {
        "Value Class with Autograd": [
            "Value wrapper class",
            "Operation tracking",
            "Gradient storage",
            "Parent references"
        ],
        "Backward Pass": [
            "Topological sort",
            "Backward propagation",
            "Chain rule application",
            "Gradient accumulation"
        ],
        "Neuron and Layer": [
            "Neuron class",
            "Layer class",
            "Forward pass",
            "Parameter collection"
        ],
        "Training Loop": [
            "Loss function",
            "Gradient zeroing",
            "Parameter update",
            "Training iteration"
        ]
    },
    "notification-service": {
        "Channel Abstraction & Routing": [
            "Channel interface (email, SMS, push)",
            "Channel implementations",
            "Routing rules",
            "Fallback channels"
        ],
        "Template System": [
            "Template storage",
            "Variable substitution",
            "Localization support",
            "Template versioning"
        ],
        "User Preferences & Unsubscribe": [
            "Preference storage",
            "Channel opt-out",
            "Unsubscribe links",
            "Preference management API"
        ],
        "Delivery Tracking & Analytics": [
            "Delivery status tracking",
            "Open/click tracking",
            "Bounce handling",
            "Analytics dashboard"
        ]
    },
    "oauth2-provider": {
        "Client Registration & Authorization Endpoint": [
            "Client registration API",
            "Client credentials storage",
            "Authorization endpoint",
            "User consent handling"
        ],
        "Token Endpoint & JWT Generation": [
            "Token endpoint",
            "Authorization code exchange",
            "JWT token generation",
            "Refresh token support"
        ],
        "Token Introspection & Revocation": [
            "Token introspection endpoint",
            "Token revocation endpoint",
            "Token validation",
            "Token lifecycle"
        ],
        "UserInfo Endpoint & Consent Management": [
            "UserInfo endpoint",
            "Scope to claims mapping",
            "Consent screen",
            "Consent persistence"
        ]
    },
    "packet-sniffer": {
        "Packet Capture Setup": [
            "Raw socket creation",
            "Promiscuous mode",
            "BPF filter option",
            "Packet buffer"
        ],
        "Ethernet Frame Parsing": [
            "Ethernet header parsing",
            "MAC address extraction",
            "EtherType handling",
            "Frame validation"
        ],
        "IP Header Parsing": [
            "IPv4 header parsing",
            "IPv6 header parsing",
            "Protocol field extraction",
            "Address extraction"
        ],
        "TCP/UDP Parsing": [
            "TCP header parsing",
            "UDP header parsing",
            "Port extraction",
            "Payload extraction"
        ],
        "Filtering and Output": [
            "Display filter",
            "Packet summary",
            "Hex dump option",
            "PCAP export"
        ]
    },
    "password-hashing": {
        "Basic Hashing with Salt": [
            "Salt generation",
            "Hash computation (SHA-256)",
            "Salt storage with hash",
            "Verification function"
        ],
        "Key Stretching": [
            "PBKDF2 implementation",
            "Iteration count",
            "Derived key length",
            "Timing attack prevention"
        ],
        "Modern Password Hashing": [
            "bcrypt integration",
            "Argon2 integration",
            "Cost factor tuning",
            "Migration support"
        ]
    },
    "payment-gateway": {
        "Payment Intent & Idempotency": [
            "Payment intent creation",
            "Idempotency key handling",
            "Intent state machine",
            "Intent expiration"
        ],
        "Payment Processing & 3DS": [
            "Payment method tokenization",
            "Charge creation",
            "3DS authentication flow",
            "Payment confirmation"
        ],
        "Refunds & Disputes": [
            "Refund processing",
            "Partial refunds",
            "Dispute handling",
            "Chargeback integration"
        ],
        "Webhook Reconciliation": [
            "Webhook endpoint",
            "Signature verification",
            "Event processing",
            "State reconciliation"
        ]
    },
    "platformer": {
        "Basic Movement and Gravity": [
            "Player entity",
            "Horizontal movement",
            "Gravity application",
            "Terminal velocity"
        ],
        "Jumping": [
            "Jump initiation",
            "Variable jump height",
            "Coyote time",
            "Jump buffering"
        ],
        "Tile-based Collision": [
            "Tilemap representation",
            "Collision detection",
            "Collision resolution",
            "One-way platforms"
        ],
        "Enemies and Hazards": [
            "Enemy entities",
            "Basic AI (patrol)",
            "Player-enemy collision",
            "Death and respawn"
        ]
    },
    "process-spawner": {
        "Basic Fork/Exec": [
            "fork() implementation",
            "exec() family usage",
            "Exit status collection",
            "waitpid() handling"
        ],
        "Pipe Communication": [
            "Pipe creation",
            "Parent-child communication",
            "Bi-directional pipes",
            "Pipe cleanup"
        ],
        "Process Pool": [
            "Worker process creation",
            "Task distribution",
            "Result collection",
            "Pool management"
        ]
    },
    "protocol-buffer": {
        "Varint Encoding": [
            "Varint encoding",
            "Varint decoding",
            "Signed varint (zigzag)",
            "64-bit support"
        ],
        "Wire Types": [
            "Wire type handling",
            "Length-delimited fields",
            "Fixed32/64 fields",
            "Field number extraction"
        ],
        "Schema Parser": [
            ".proto file lexer",
            "Message definition parsing",
            "Field definition parsing",
            "Import handling"
        ],
        "Message Serialization": [
            "Field encoding",
            "Message encoding",
            "Nested message handling",
            "Repeated field encoding"
        ]
    },
    "rag-system": {
        "Document Ingestion & Chunking": [
            "Document loading (PDF, text)",
            "Text chunking strategies",
            "Chunk overlap handling",
            "Metadata extraction"
        ],
        "Embedding Generation": [
            "Embedding model integration",
            "Batch embedding",
            "Embedding caching",
            "Model selection"
        ],
        "Vector Store & Retrieval": [
            "Vector database integration",
            "Similarity search",
            "Metadata filtering",
            "Re-ranking"
        ],
        "LLM Integration & Prompting": [
            "Context construction",
            "Prompt template",
            "LLM API call",
            "Response parsing"
        ],
        "Evaluation & Optimization": [
            "Retrieval metrics",
            "Answer quality metrics",
            "Chunk size optimization",
            "Prompt optimization"
        ]
    },
    "rate-limiter-distributed": {
        "Rate Limiting Algorithms": [
            "Token bucket implementation",
            "Sliding window counter",
            "Leaky bucket option",
            "Algorithm comparison"
        ],
        "Multi-tier Rate Limiting": [
            "Per-user limits",
            "Per-IP limits",
            "Global limits",
            "Quota management"
        ]
    },
    "rbac-system": {
        "Role & Permission Model": [
            "Role definition",
            "Permission definition",
            "Role-permission mapping",
            "User-role assignment"
        ],
        "ABAC Policy Engine": [
            "Attribute-based rules",
            "Policy definition language",
            "Policy evaluation",
            "Context attributes"
        ],
        "Resource-Based & Multi-tenancy": [
            "Resource ownership",
            "Tenant isolation",
            "Cross-tenant permissions",
            "Resource hierarchies"
        ],
        "Audit Logging & Policy Testing": [
            "Access decision logging",
            "Policy simulation",
            "Test suite for policies",
            "Compliance reporting"
        ]
    },
    "replicated-log": {
        "Log Storage": [
            "Append-only log structure",
            "Log entry format",
            "Log persistence",
            "Log compaction"
        ],
        "Replication Protocol": [
            "Leader election",
            "Log replication to followers",
            "Consistency guarantees",
            "Quorum writes"
        ],
        "Failure Detection": [
            "Heartbeat mechanism",
            "Failure timeout",
            "Leader takeover",
            "Split-brain prevention"
        ],
        "Client Interface": [
            "Append API",
            "Read API",
            "Subscribe to updates",
            "Consistency options"
        ]
    },
    "rest-api-design": {
        "CRUD Operations": [
            "Resource endpoints",
            "HTTP method mapping",
            "Response formatting",
            "Status codes"
        ],
        "Input Validation": [
            "Request validation",
            "Schema validation",
            "Error responses",
            "Sanitization"
        ],
        "Authentication & Authorization": [
            "JWT authentication",
            "API key option",
            "Role-based access",
            "Protected routes"
        ],
        "Rate Limiting & Throttling": [
            "Rate limit middleware",
            "Limit headers",
            "429 responses",
            "Per-client limits"
        ]
    },
    "rpc-basic": {
        "Message Protocol": [
            "Request message format",
            "Response message format",
            "Error format",
            "Serialization (JSON/msgpack)"
        ],
        "Server Implementation": [
            "TCP server",
            "Method registry",
            "Request dispatch",
            "Response sending"
        ],
        "Client Implementation": [
            "TCP client",
            "Method call API",
            "Response handling",
            "Timeout support"
        ]
    },
    "saga-orchestrator": {
        "Saga Definition": [
            "Step definition",
            "Compensation definition",
            "Step ordering",
            "Saga metadata"
        ],
        "Saga Orchestrator Engine": [
            "Saga execution",
            "Step invocation",
            "Failure handling",
            "Compensation execution"
        ],
        "Saga State Persistence": [
            "Saga instance state",
            "Step completion tracking",
            "Recovery on restart",
            "Idempotency"
        ]
    },
    "sandbox": {
        "Process Namespaces": [
            "PID namespace isolation",
            "Network namespace isolation",
            "Mount namespace setup",
            "User namespace mapping"
        ],
        "Filesystem Isolation": [
            "Chroot/pivot_root",
            "Read-only root",
            "Tmpfs for writable paths",
            "Bind mount restrictions"
        ],
        "Seccomp System Call Filtering": [
            "Seccomp-bpf filter",
            "Syscall whitelist",
            "Argument filtering",
            "Filter compilation"
        ],
        "Resource Limits with Cgroups": [
            "Memory limit",
            "CPU limit",
            "PID limit",
            "I/O bandwidth limit"
        ],
        "Capability Dropping": [
            "Capability bounding set",
            "Drop unnecessary caps",
            "No-new-privs flag",
            "Privilege verification"
        ]
    },
    "search-engine": {
        "Inverted Index": [
            "Term to document mapping",
            "Posting list structure",
            "Index construction",
            "Index persistence"
        ],
        "TF-IDF & BM25 Ranking": [
            "Term frequency calculation",
            "Document frequency",
            "TF-IDF scoring",
            "BM25 implementation"
        ],
        "Fuzzy Matching & Autocomplete": [
            "Edit distance",
            "Fuzzy search",
            "Prefix matching",
            "Autocomplete suggestions"
        ],
        "Query Parser & Filters": [
            "Query tokenization",
            "Boolean operators",
            "Field filtering",
            "Phrase queries"
        ]
    },
    "secret-management": {
        "Encrypted Secret Storage": [
            "Encryption at rest",
            "Master key handling",
            "Secret versioning",
            "Key rotation"
        ],
        "Access Policies & Authentication": [
            "Policy definition",
            "Token-based auth",
            "Policy evaluation",
            "Audit logging"
        ],
        "Dynamic Secrets": [
            "Dynamic secret generation",
            "Database credentials",
            "TTL management",
            "Secret revocation"
        ],
        "Unsealing & High Availability": [
            "Seal/unseal mechanism",
            "Key shares (Shamir)",
            "HA backend",
            "Standby nodes"
        ]
    },
    "semantic-search": {
        "Embedding Index": [
            "Document embedding",
            "Vector index construction",
            "Index persistence",
            "Incremental updates"
        ],
        "Query Processing": [
            "Query embedding",
            "Similarity search",
            "Result retrieval",
            "Query expansion"
        ],
        "Ranking & Relevance": [
            "Semantic similarity scoring",
            "Hybrid search (keyword + semantic)",
            "Re-ranking",
            "Relevance tuning"
        ],
        "Search API & UI": [
            "Search endpoint",
            "Result formatting",
            "Highlighting",
            "Search UI"
        ]
    },
    "service-mesh": {
        "Traffic Interception": [
            "Iptables rules",
            "Sidecar proxy",
            "Transparent proxying",
            "Protocol detection"
        ],
        "Service Discovery Integration": [
            "Endpoint discovery",
            "Service registry sync",
            "Health status tracking",
            "Load balancing"
        ],
        "mTLS and Certificate Management": [
            "Certificate generation",
            "Certificate rotation",
            "mTLS enforcement",
            "SPIFFE integration"
        ],
        "Load Balancing Algorithms": [
            "Round robin",
            "Least connections",
            "Weighted",
            "Locality-aware"
        ]
    },
    "session-management": {
        "Secure Session Creation & Storage": [
            "Session ID generation",
            "Session data storage",
            "Server-side sessions",
            "Storage backends"
        ],
        "Cookie Security & Transport": [
            "Secure cookie flags",
            "HttpOnly flag",
            "SameSite attribute",
            "Cookie encryption"
        ],
        "Multi-Device & Concurrent Sessions": [
            "Device tracking",
            "Session listing",
            "Session revocation",
            "Concurrent session limits"
        ]
    },
    "signal-handler": {
        "Basic Signal Handling": [
            "Signal handler registration",
            "Signal delivery",
            "Handler function",
            "Signal safety"
        ],
        "Signal Masking": [
            "Signal mask manipulation",
            "Blocking signals",
            "Pending signals",
            "Signal set operations"
        ],
        "Self-Pipe Trick": [
            "Pipe for signal notification",
            "Non-blocking write in handler",
            "Event loop integration",
            "Multiplexed I/O with signals"
        ]
    },
    "simple-gc": {
        "Object Model": [
            "Object header",
            "Reference tracking",
            "Object allocation",
            "Type information"
        ],
        "Root Discovery": [
            "Stack scanning",
            "Global variables",
            "Register contents",
            "Root set construction"
        ],
        "Mark Phase": [
            "Object graph traversal",
            "Mark bit setting",
            "Worklist processing",
            "Reachability analysis"
        ],
        "Sweep Phase": [
            "Heap traversal",
            "Unmarked object collection",
            "Free list update",
            "Memory reclamation"
        ]
    },
    "social-network": {
        "User Profiles & Follow System": [
            "User profile CRUD",
            "Follow/unfollow",
            "Follower/following lists",
            "Profile privacy"
        ],
        "Posts & Feed (Fan-out on Write)": [
            "Post creation",
            "Fan-out to followers",
            "Feed retrieval",
            "Feed pagination"
        ],
        "Likes, Comments & Interactions": [
            "Like functionality",
            "Comment system",
            "Interaction counts",
            "Notification triggers"
        ],
        "Notifications": [
            "Notification types",
            "Notification delivery",
            "Read/unread status",
            "Notification preferences"
        ],
        "Search & Discovery": [
            "User search",
            "Post search",
            "Hashtag support",
            "Trending topics"
        ],
        "Performance & Scaling": [
            "Caching strategy",
            "Database sharding",
            "CDN for media",
            "Rate limiting"
        ]
    },
    "software-3d": {
        "Line Drawing": [
            "Bresenham's line algorithm",
            "Line clipping",
            "Anti-aliased lines",
            "Framebuffer writing"
        ],
        "Triangle Rasterization": [
            "Edge function",
            "Scanline rasterization",
            "Barycentric coordinates",
            "Texture coordinate interpolation"
        ],
        "3D Transformations": [
            "Model matrix",
            "View matrix",
            "Projection matrix",
            "Vertex transformation"
        ],
        "Depth Buffer & Lighting": [
            "Z-buffer implementation",
            "Depth testing",
            "Flat shading",
            "Gouraud shading"
        ]
    },
    "sql-parser": {
        "SQL Tokenizer": [
            "Keyword tokens",
            "Identifier tokens",
            "Literal tokens",
            "Operator tokens"
        ],
        "SELECT Parser": [
            "SELECT clause",
            "FROM clause",
            "Column list",
            "Table references"
        ],
        "WHERE Clause": [
            "Condition parsing",
            "Comparison operators",
            "Logical operators",
            "Expression trees"
        ],
        "INSERT/UPDATE/DELETE": [
            "INSERT statement",
            "UPDATE statement",
            "DELETE statement",
            "Value lists"
        ]
    },
    "subscription-billing": {
        "Plans & Pricing": [
            "Plan definition",
            "Pricing tiers",
            "Feature entitlements",
            "Plan management API"
        ],
        "Subscription Lifecycle": [
            "Subscription creation",
            "Renewal processing",
            "Cancellation",
            "Reactivation"
        ],
        "Proration & Plan Changes": [
            "Proration calculation",
            "Upgrade/downgrade",
            "Credit application",
            "Mid-cycle changes"
        ],
        "Usage-Based Billing": [
            "Usage tracking",
            "Metering API",
            "Usage aggregation",
            "Overage charges"
        ]
    },
    "terminal-multiplexer": {
        "PTY Creation": [
            "Pseudoterminal allocation",
            "Master/slave setup",
            "PTY I/O handling",
            "Window size (TIOCSWINSZ)"
        ],
        "Terminal Emulation": [
            "ANSI escape parsing",
            "Cursor movement",
            "Screen buffer",
            "Scrollback"
        ],
        "Window Management": [
            "Pane splitting",
            "Pane resizing",
            "Window switching",
            "Layout algorithms"
        ],
        "Key Bindings and UI": [
            "Prefix key handling",
            "Custom key bindings",
            "Status bar",
            "Copy mode"
        ]
    },
    "tetris": {
        "Board & Tetrominoes": [
            "Game board grid",
            "Tetromino shapes",
            "Tetromino rotation data",
            "Color mapping"
        ],
        "Piece Falling & Controls": [
            "Piece spawning",
            "Gravity (auto-fall)",
            "Left/right movement",
            "Soft/hard drop"
        ],
        "Piece Rotation": [
            "Rotation states",
            "Wall kick tests",
            "SRS rotation system",
            "Collision checking"
        ],
        "Line Clearing & Scoring": [
            "Full line detection",
            "Line removal",
            "Scoring system",
            "Level progression"
        ]
    },
    "tokenizer": {
        "Basic Token Types": [
            "Token type enumeration",
            "Token structure",
            "Position tracking",
            "Token list output"
        ],
        "Scanning Logic": [
            "Character consumption",
            "Peek and advance",
            "Whitespace handling",
            "EOF handling"
        ],
        "Multi-character Tokens": [
            "Operators (==, !=, <=, >=)",
            "Identifiers",
            "Keywords",
            "Numbers"
        ],
        "Strings and Comments": [
            "String literal scanning",
            "Escape sequences",
            "Single-line comments",
            "Multi-line comments"
        ]
    },
    "topdown-shooter": {
        "Player Movement & Aiming": [
            "WASD movement",
            "Mouse aiming",
            "Player sprite rotation",
            "Movement speed"
        ],
        "Shooting & Projectiles": [
            "Projectile spawning",
            "Projectile movement",
            "Projectile collision",
            "Fire rate limiting"
        ],
        "Enemies & AI": [
            "Enemy spawning",
            "Chase behavior",
            "Enemy health",
            "Enemy-player collision"
        ],
        "Waves & Scoring": [
            "Wave system",
            "Score tracking",
            "Difficulty scaling",
            "Game over condition"
        ]
    },
    "transformer-scratch": {
        "Scaled Dot-Product Attention": [
            "Q, K, V computation",
            "Attention scores",
            "Softmax normalization",
            "Weighted sum"
        ],
        "Multi-Head Attention": [
            "Head splitting",
            "Parallel attention",
            "Head concatenation",
            "Output projection"
        ],
        "Position-wise Feed-Forward & Embeddings": [
            "Feed-forward network",
            "Token embeddings",
            "Positional encoding",
            "Embedding combination"
        ],
        "Encoder & Decoder Layers": [
            "Encoder layer",
            "Decoder layer",
            "Layer stacking",
            "Masked attention in decoder"
        ],
        "Full Transformer & Training": [
            "Encoder-decoder architecture",
            "Training loop",
            "Loss function",
            "Inference"
        ]
    },
    "unit-testing-basics": {
        "First Tests": [
            "Test function creation",
            "Assertion usage",
            "Test execution",
            "Pass/fail reporting"
        ],
        "Test Organization": [
            "Test file structure",
            "Test suites/classes",
            "Setup/teardown",
            "Test naming"
        ],
        "Mocking and Isolation": [
            "Mock objects",
            "Dependency injection",
            "Stub functions",
            "Spy objects"
        ]
    },
    "vector-clocks": {
        "Basic Vector Clock": [
            "Clock initialization",
            "Increment operation",
            "Merge operation",
            "Clock comparison"
        ],
        "Conflict Detection": [
            "Concurrent event detection",
            "Happens-before relationship",
            "Conflict marking",
            "Conflict resolution strategy"
        ],
        "Version Pruning": [
            "Clock garbage collection",
            "Minimum clock tracking",
            "Pruning algorithm",
            "Storage optimization"
        ],
        "Distributed Key-Value Store": [
            "Versioned values",
            "Read with version",
            "Write with version",
            "Conflict resolution on read"
        ]
    },
    "video-streaming": {
        "Video Upload": [
            "Upload endpoint",
            "Chunked upload support",
            "Video storage",
            "Metadata extraction"
        ],
        "Video Transcoding": [
            "FFmpeg integration",
            "Multiple quality levels",
            "Codec selection",
            "Transcoding queue"
        ],
        "Adaptive Streaming": [
            "HLS segment generation",
            "Manifest file creation",
            "Quality variant playlists",
            "Segment serving"
        ],
        "Video Player Integration": [
            "HLS.js integration",
            "Quality switching",
            "Playback controls",
            "Progress tracking"
        ]
    },
    "virtual-memory-sim": {
        "Page Table": [
            "Page table entry structure",
            "Page table construction",
            "Page table lookup",
            "Valid/invalid bits"
        ],
        "TLB": [
            "TLB structure",
            "TLB lookup",
            "TLB miss handling",
            "TLB flush"
        ],
        "Multi-level Page Tables": [
            "Two-level page table",
            "Page directory",
            "Page table hierarchy",
            "Memory savings"
        ],
        "Page Replacement": [
            "FIFO replacement",
            "LRU replacement",
            "Clock algorithm",
            "Page fault handling"
        ]
    },
    "wasm-emitter": {
        "WASM Binary Format": [
            "Module structure",
            "Section encoding",
            "LEB128 encoding",
            "Type section"
        ],
        "Expression Compilation": [
            "Literal values",
            "Binary operations",
            "Local variables",
            "Stack management"
        ],
        "Control Flow": [
            "Block and loop",
            "If-else",
            "Branch instructions",
            "Return"
        ],
        "Functions and Exports": [
            "Function type declaration",
            "Function body emission",
            "Export section",
            "Import section"
        ]
    },
    "webhook-delivery": {
        "Webhook Registration & Security": [
            "Endpoint registration",
            "Secret key generation",
            "Signature generation",
            "URL validation"
        ],
        "Delivery Queue & Retry Logic": [
            "Delivery queue",
            "HTTP delivery",
            "Retry with backoff",
            "Dead letter queue"
        ],
        "Circuit Breaker & Rate Limiting": [
            "Per-endpoint circuit breaker",
            "Rate limiting",
            "Backpressure handling",
            "Health tracking"
        ],
        "Event Log & Replay": [
            "Delivery log",
            "Event replay API",
            "Delivery status tracking",
            "Debugging tools"
        ]
    },
    "word2vec": {
        "Data Preprocessing": [
            "Text tokenization",
            "Vocabulary building",
            "Word to index mapping",
            "Subsampling frequent words"
        ],
        "Skip-gram Model": [
            "Embedding layer",
            "Context window",
            "Training pairs generation",
            "Forward pass"
        ],
        "Training with Negative Sampling": [
            "Negative sample selection",
            "Loss function",
            "Gradient computation",
            "Parameter updates"
        ],
        "Evaluation & Visualization": [
            "Word similarity",
            "Analogy tests",
            "t-SNE visualization",
            "Embedding export"
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    updated = 0
    missing = []

    for proj_id, proj in data.get('expert_projects', {}).items():
        if proj_id not in DELIVERABLES:
            # Check if already has deliverables
            has_deliv = any(m.get('deliverables') for m in proj.get('milestones', []))
            if not has_deliv:
                missing.append(proj_id)
            continue

        proj_deliverables = DELIVERABLES[proj_id]

        for milestone in proj.get('milestones', []):
            name = milestone.get('name')
            if name in proj_deliverables and not milestone.get('deliverables'):
                milestone['deliverables'] = proj_deliverables[name]
                updated += 1

    # Save
    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"Updated: {updated} milestones with deliverables")

    if missing:
        print(f"\nProjects still not in mapping ({len(missing)}):")
        for p in sorted(missing):
            print(f"  - {p}")
    else:
        print("\nAll projects now have deliverables!")


if __name__ == "__main__":
    main()

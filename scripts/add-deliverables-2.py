#!/usr/bin/env python3
"""Add deliverables to remaining projects missing them (part 2)."""

import yaml
from pathlib import Path

# Deliverables for remaining projects
DELIVERABLES = {
    "build-lsp": {
        "JSON-RPC & Initialization": [
            "JSON-RPC message parser and serializer",
            "Initialize request/response handling",
            "Server capabilities negotiation",
            "Shutdown and exit handling"
        ],
        "Document Synchronization": [
            "didOpen/didClose/didChange handlers",
            "Document version tracking",
            "Incremental text change application",
            "Document state management"
        ],
        "Language Features": [
            "Completion provider with items",
            "Hover provider with documentation",
            "Go-to-definition implementation",
            "Find references implementation"
        ],
        "Diagnostics & Code Actions": [
            "Diagnostic publishing on document change",
            "Error and warning reporting",
            "Code action suggestions",
            "Quick fix implementations"
        ]
    },
    "build-nn-framework": {
        "Tensor & Operations": [
            "N-dimensional tensor class",
            "Element-wise operations",
            "Matrix multiplication",
            "Broadcasting support"
        ],
        "Automatic Differentiation": [
            "Computation graph construction",
            "Backward pass traversal",
            "Gradient accumulation",
            "Gradient tape implementation"
        ],
        "Layers & Modules": [
            "Linear layer with parameters",
            "Activation functions",
            "Module base class",
            "Parameter registration"
        ],
        "Optimizers & Training": [
            "SGD optimizer",
            "Adam optimizer",
            "Training step abstraction",
            "Loss functions"
        ]
    },
    "build-observability-platform": {
        "Unified Data Model": [
            "Common schema for logs, metrics, traces",
            "Correlation identifiers (trace_id, span_id)",
            "Timestamp standardization",
            "Resource and attribute mapping"
        ],
        "Data Ingestion Pipeline": [
            "OTLP receiver implementation",
            "Batch processing for throughput",
            "Data validation and normalization",
            "Back-pressure handling"
        ],
        "Multi-Signal Storage": [
            "Time-series storage for metrics",
            "Full-text storage for logs",
            "Trace storage with parent-child",
            "Efficient correlation queries"
        ],
        "Unified Query Interface": [
            "Cross-signal query language",
            "Time-range filtering",
            "Aggregation support",
            "Query result correlation"
        ],
        "Alerting & Anomaly Detection": [
            "Multi-signal alert rules",
            "Anomaly detection algorithms",
            "Alert correlation",
            "Notification routing"
        ]
    },
    "build-os": {
        "Bootloader & Kernel Entry": [
            "Boot sector code (MBR or UEFI)",
            "Protected/long mode transition",
            "Kernel loading to memory",
            "Entry point setup"
        ],
        "Interrupts & Keyboard": [
            "IDT setup",
            "Interrupt handlers",
            "PIC/APIC configuration",
            "Keyboard driver"
        ],
        "Memory Management": [
            "Physical frame allocator",
            "Page table setup",
            "Virtual address mapping",
            "Heap allocator"
        ],
        "Process Management": [
            "Process control block",
            "Context switching",
            "Basic scheduler",
            "System call interface"
        ]
    },
    "build-react": {
        "Virtual DOM": [
            "Virtual DOM node structure",
            "createElement function",
            "Virtual tree construction",
            "Node type handling (element, text, component)"
        ],
        "Diffing Algorithm": [
            "Tree comparison algorithm",
            "Keyed element matching",
            "Patch generation",
            "Efficient update detection"
        ],
        "Reconciliation": [
            "DOM patch application",
            "Element creation and updates",
            "Property/attribute setting",
            "Child reconciliation"
        ],
        "Hooks": [
            "useState implementation",
            "useEffect implementation",
            "Hook state tracking",
            "Re-render triggering"
        ]
    },
    "build-shell": {
        "Input Processing": [
            "Line editing (readline-like)",
            "History storage and navigation",
            "Tab completion",
            "Prompt display"
        ],
        "Command Parsing": [
            "Tokenization of input",
            "Quote handling",
            "Variable expansion",
            "Glob pattern expansion"
        ],
        "Execution": [
            "fork/exec for external commands",
            "Pipeline construction",
            "Redirection handling",
            "Background jobs"
        ],
        "Job Control": [
            "Job table management",
            "Signal handling (SIGINT, SIGTSTP)",
            "Foreground/background control",
            "Job status reporting"
        ]
    },
    "build-spreadsheet": {
        "Cell Data Model": [
            "Grid data structure",
            "Cell value types (text, number, formula)",
            "Cell addressing (A1 notation)",
            "Cell formatting attributes"
        ],
        "Formula Parser": [
            "Expression tokenizer",
            "Operator precedence parsing",
            "Cell reference parsing",
            "Range parsing (A1:B10)"
        ],
        "Dependency Graph": [
            "Cell dependency tracking",
            "Topological sort for evaluation",
            "Circular reference detection",
            "Incremental recalculation"
        ],
        "UI & Interaction": [
            "Grid rendering",
            "Cell selection and editing",
            "Keyboard navigation",
            "Copy/paste support"
        ]
    },
    "build-tcp-stack": {
        "Ethernet & IP": [
            "Raw socket or TAP device",
            "Ethernet frame handling",
            "IP packet parsing/construction",
            "ARP implementation"
        ],
        "TCP State Machine": [
            "TCP state transitions",
            "Connection establishment (3-way handshake)",
            "Connection termination",
            "State timeout handling"
        ],
        "Data Transfer": [
            "Sequence number management",
            "Acknowledgment processing",
            "Receive buffer management",
            "Send buffer management"
        ],
        "Reliability": [
            "Retransmission timer",
            "Fast retransmit",
            "Congestion control (slow start, AIMD)",
            "Flow control with window"
        ]
    },
    "build-test-framework": {
        "Test Discovery": [
            "Test function/class detection",
            "Test file scanning",
            "Naming convention matching",
            "Test filtering by name/tag"
        ],
        "Assertions": [
            "assertEqual, assertTrue, assertFalse",
            "assertRaises for exceptions",
            "Assertion message formatting",
            "Custom assertion support"
        ],
        "Test Execution": [
            "Test runner loop",
            "Setup/teardown hooks",
            "Test isolation",
            "Parallel execution option"
        ],
        "Reporting": [
            "Pass/fail/skip counting",
            "Failure details with stack trace",
            "Output formatting (text, JUnit XML)",
            "Code coverage integration"
        ]
    },
    "build-tls": {
        "Handshake Protocol": [
            "ClientHello construction",
            "ServerHello parsing",
            "Certificate handling",
            "Key exchange (ECDHE)"
        ],
        "Key Derivation": [
            "HKDF implementation",
            "Handshake secrets",
            "Application secrets",
            "Key schedule"
        ],
        "Record Layer": [
            "Record framing",
            "AEAD encryption (AES-GCM, ChaCha20-Poly1305)",
            "Record decryption",
            "Record type handling"
        ],
        "Application Data": [
            "Encrypted data transfer",
            "Close notify handling",
            "Session resumption (optional)",
            "Alert protocol"
        ]
    },
    "build-transpiler": {
        "Source Parser": [
            "Lexer for source language",
            "Parser for source language",
            "AST construction",
            "Semantic analysis"
        ],
        "Target Code Generation": [
            "AST traversal for codegen",
            "Syntax transformation rules",
            "Target language formatting",
            "Source map generation"
        ],
        "Language Features": [
            "Variable transformation",
            "Function transformation",
            "Class/object transformation",
            "Module system handling"
        ],
        "Polyfills & Runtime": [
            "Runtime helper functions",
            "Polyfill injection",
            "Target environment detection",
            "Bundle output"
        ]
    },
    "bytecode-compiler": {
        "AST to Bytecode": [
            "AST traversal for emission",
            "Expression compilation",
            "Statement compilation",
            "Scope and variable handling"
        ],
        "Instruction Emission": [
            "Opcode selection",
            "Operand encoding",
            "Label resolution",
            "Constant pool management"
        ],
        "Control Flow": [
            "Conditional jumps",
            "Loop compilation",
            "Break/continue handling",
            "Function calls"
        ],
        "Bytecode Output": [
            "Bytecode file format",
            "Metadata section",
            "Debug info (line numbers)",
            "Disassembler for verification"
        ]
    },
    "consensus-algorithm": {
        "Paxos Basics": [
            "Proposer implementation",
            "Acceptor implementation",
            "Learner implementation",
            "Proposal number generation"
        ],
        "Phase 1 (Prepare)": [
            "Prepare request sending",
            "Promise response",
            "Highest accepted value tracking",
            "Quorum detection"
        ],
        "Phase 2 (Accept)": [
            "Accept request with value",
            "Accept response",
            "Value commitment",
            "Consensus detection"
        ],
        "Multi-Paxos": [
            "Leader election",
            "Stable leader optimization",
            "Log replication",
            "Catch-up mechanism"
        ]
    },
    "crdt-impl": {
        "G-Counter": [
            "Node-local counter storage",
            "Increment operation",
            "Merge function (max per node)",
            "Value computation (sum)"
        ],
        "PN-Counter": [
            "Positive and negative G-Counters",
            "Increment and decrement",
            "Merge both counters",
            "Value computation (P - N)"
        ],
        "LWW-Register": [
            "Value with timestamp storage",
            "Update with timestamp",
            "Merge (latest timestamp wins)",
            "Clock handling"
        ],
        "OR-Set": [
            "Add with unique tag",
            "Remove tracked adds",
            "Merge (union of adds, union of removes)",
            "Observed-remove semantics"
        ]
    },
    "database-index": {
        "B+ Tree Structure": [
            "Internal and leaf node types",
            "Key storage in internal nodes",
            "Key-value pairs in leaf nodes",
            "Leaf node linking"
        ],
        "Search Operations": [
            "Point query (exact match)",
            "Range scan (start to end)",
            "Prefix matching",
            "Cursor-based iteration"
        ],
        "Modifications": [
            "Insert with node splitting",
            "Delete with underflow handling",
            "Bulk loading optimization",
            "Key update"
        ],
        "Persistence": [
            "Page-based storage",
            "Buffer pool integration",
            "Write-ahead logging",
            "Crash recovery"
        ]
    },
    "diff-tool": {
        "File Reading": [
            "File loading into lines",
            "Encoding detection",
            "Newline handling",
            "Binary file detection"
        ],
        "LCS Algorithm": [
            "Longest common subsequence computation",
            "Dynamic programming table",
            "Backtracking for LCS",
            "Memory optimization"
        ],
        "Diff Generation": [
            "Edit script generation",
            "Chunk (hunk) formation",
            "Context lines inclusion",
            "Unified diff format"
        ],
        "Output Formatting": [
            "Unified diff output",
            "Side-by-side diff option",
            "Color-coded output",
            "Line number display"
        ]
    },
    "dns-client": {
        "Query Construction": [
            "DNS header construction",
            "Question section encoding",
            "Query type setting (A, AAAA, MX, etc.)",
            "Transaction ID generation"
        ],
        "Network Communication": [
            "UDP socket for DNS",
            "Query sending to resolver",
            "Response receiving",
            "Timeout and retry"
        ],
        "Response Parsing": [
            "DNS header parsing",
            "Answer section parsing",
            "Authority and additional sections",
            "Compression pointer handling"
        ],
        "Caching": [
            "TTL-based cache",
            "Cache lookup before query",
            "Negative caching",
            "Cache invalidation"
        ]
    },
    "encoding-impl": {
        "Base64": [
            "6-bit grouping of input",
            "Character mapping table",
            "Padding handling",
            "Decode inverse mapping"
        ],
        "Huffman Coding": [
            "Character frequency counting",
            "Huffman tree construction",
            "Code table generation",
            "Encode and decode"
        ],
        "Run-Length Encoding": [
            "Consecutive run detection",
            "Run length encoding format",
            "Decode run expansion",
            "Escape for literal values"
        ],
        "Variable-Length Integer": [
            "VarInt encoding (protobuf style)",
            "Continuation bit handling",
            "Decode to integer",
            "Signed integer handling (zigzag)"
        ]
    },
    "end-to-end-encryption": {
        "Key Exchange": [
            "X3DH key agreement protocol",
            "Identity key pairs",
            "Signed prekey bundles",
            "One-time prekeys"
        ],
        "Double Ratchet": [
            "Root chain ratchet",
            "Sending chain ratchet",
            "Receiving chain ratchet",
            "Message key derivation"
        ],
        "Message Encryption": [
            "Message key generation",
            "AEAD encryption (AES-GCM)",
            "Header encryption",
            "Message ordering"
        ],
        "Session Management": [
            "Session state storage",
            "Session initialization",
            "Out-of-order message handling",
            "Session reset"
        ]
    },
    "event-bus": {
        "Subscription Management": [
            "Event type to handler mapping",
            "Subscribe operation",
            "Unsubscribe operation",
            "Weak reference option"
        ],
        "Event Publishing": [
            "Event dispatch to handlers",
            "Synchronous delivery option",
            "Async delivery option",
            "Event queue for async"
        ],
        "Filtering": [
            "Event filtering by type",
            "Predicate-based filtering",
            "Topic/channel support",
            "Priority ordering"
        ],
        "Error Handling": [
            "Handler exception catching",
            "Dead letter handling",
            "Retry policies",
            "Error event publishing"
        ]
    },
    "event-loop": {
        "Poll/Epoll Integration": [
            "File descriptor registration",
            "Event polling (epoll/kqueue)",
            "Ready event handling",
            "Edge vs level triggering"
        ],
        "Callback Management": [
            "Callback registration for fd",
            "Read/write readiness callbacks",
            "Callback invocation on event",
            "Callback removal"
        ],
        "Timers": [
            "Timer registration with callback",
            "Timer wheel or heap",
            "Timer expiration check",
            "Periodic timer support"
        ],
        "Task Scheduling": [
            "Immediate task queue",
            "Next tick scheduling",
            "Idle callbacks",
            "Event loop iteration"
        ]
    },
    "expression-evaluator": {
        "Tokenizer": [
            "Number token extraction",
            "Operator token extraction",
            "Parenthesis handling",
            "Whitespace skipping"
        ],
        "Parser": [
            "Operator precedence handling",
            "Shunting-yard algorithm or recursive descent",
            "Parenthesis grouping",
            "Unary operators"
        ],
        "Evaluator": [
            "Evaluation of parsed expression",
            "Arithmetic operations",
            "Function call support (sin, cos, etc.)",
            "Variable substitution"
        ],
        "Error Handling": [
            "Syntax error detection",
            "Division by zero handling",
            "Undefined variable errors",
            "Error messages with position"
        ]
    },
    "ids-system": {
        "Packet Capture": [
            "Network interface binding",
            "Packet capture (libpcap/AF_PACKET)",
            "Protocol decoding",
            "Packet filtering"
        ],
        "Signature Detection": [
            "Signature rule loading",
            "Pattern matching engine",
            "Protocol-aware matching",
            "Rule optimization"
        ],
        "Anomaly Detection": [
            "Baseline traffic profiling",
            "Statistical anomaly detection",
            "Threshold-based alerts",
            "Machine learning (optional)"
        ],
        "Alerting": [
            "Alert generation with context",
            "Alert logging",
            "Alert aggregation",
            "Alert notification (email, webhook)"
        ]
    },
    "linker": {
        "Object File Parsing": [
            "ELF/Mach-O/PE parsing",
            "Section reading",
            "Symbol table extraction",
            "Relocation table reading"
        ],
        "Symbol Resolution": [
            "Global symbol table construction",
            "Undefined symbol resolution",
            "Multiple definition handling",
            "Weak symbol handling"
        ],
        "Relocation": [
            "Relocation application",
            "Address calculation",
            "Relocation type handling",
            "GOT/PLT for shared libs"
        ],
        "Output Generation": [
            "Section layout planning",
            "Section merging",
            "Executable header writing",
            "Final binary output"
        ]
    },
    "log-structured-fs": {
        "Log Segment Management": [
            "Segment file creation",
            "Sequential write append",
            "Segment rotation",
            "Segment sealing"
        ],
        "Inode Mapping": [
            "Inode to segment offset mapping",
            "Inode map persistence",
            "Inode allocation",
            "Directory entries"
        ],
        "Garbage Collection": [
            "Live block identification",
            "Segment cleaning",
            "Block copying to new segment",
            "Free segment reclamation"
        ],
        "Recovery": [
            "Checkpoint writing",
            "Log replay on mount",
            "Crash consistency",
            "Metadata recovery"
        ]
    },
    "memory-db": {
        "Data Structures": [
            "Hash table for key-value",
            "Skip list for sorted data",
            "Data type implementations",
            "Memory layout optimization"
        ],
        "Command Processing": [
            "Command parser",
            "Command dispatch",
            "Response formatting",
            "Pipelining support"
        ],
        "Persistence": [
            "Snapshot serialization",
            "Point-in-time dump",
            "Append-only log option",
            "Recovery on restart"
        ],
        "Networking": [
            "TCP server",
            "Client connection handling",
            "Protocol implementation",
            "Connection management"
        ]
    },
    "microservices-demo": {
        "Service Design": [
            "Service boundary definition",
            "API contract (REST/gRPC)",
            "Data ownership per service",
            "Inter-service communication"
        ],
        "Service Implementation": [
            "User service implementation",
            "Order service implementation",
            "Product service implementation",
            "Service configuration"
        ],
        "Service Discovery": [
            "Service registration",
            "Health check endpoint",
            "Discovery client",
            "Load balancing"
        ],
        "Observability": [
            "Distributed tracing integration",
            "Metrics exposure",
            "Centralized logging",
            "Service dashboard"
        ]
    },
    "nat-traversal": {
        "STUN Client": [
            "STUN binding request",
            "Server reflexive address extraction",
            "NAT type detection",
            "STUN response parsing"
        ],
        "TURN Client": [
            "TURN allocation request",
            "Relay address allocation",
            "Channel binding",
            "Data relay through TURN"
        ],
        "ICE Implementation": [
            "Candidate gathering",
            "Candidate exchange (signaling)",
            "Connectivity checks",
            "Candidate pair selection"
        ],
        "Hole Punching": [
            "Symmetric NAT handling",
            "Simultaneous open attempt",
            "Port prediction",
            "Fallback to relay"
        ]
    },
    "network-file-system": {
        "Protocol Design": [
            "RPC protocol definition",
            "Request/response messages",
            "File handle representation",
            "Error codes"
        ],
        "Server Implementation": [
            "File operation handlers",
            "Handle to path mapping",
            "Concurrent request handling",
            "Caching on server"
        ],
        "Client Implementation": [
            "VFS integration or FUSE",
            "RPC client",
            "Client-side caching",
            "Handle management"
        ],
        "Caching & Consistency": [
            "Attribute caching",
            "Data block caching",
            "Cache invalidation",
            "Consistency semantics"
        ]
    },
    "object-storage": {
        "Bucket Management": [
            "Bucket creation and deletion",
            "Bucket listing",
            "Bucket metadata",
            "Access control per bucket"
        ],
        "Object Operations": [
            "PUT object with data",
            "GET object retrieval",
            "DELETE object",
            "Object metadata handling"
        ],
        "Storage Backend": [
            "File-based storage",
            "Object to file mapping",
            "Content addressing option",
            "Deduplication"
        ],
        "S3 Compatibility": [
            "S3 REST API subset",
            "Authentication (signature v4)",
            "Multipart upload",
            "Range requests"
        ]
    },
    "parser-combinator": {
        "Basic Parsers": [
            "Character parser",
            "String parser",
            "Satisfy parser (predicate)",
            "End-of-input parser"
        ],
        "Combinators": [
            "Sequence combinator (and_then)",
            "Choice combinator (or_else)",
            "Many combinator (zero or more)",
            "Map combinator (transform)"
        ],
        "Error Handling": [
            "Parse error type",
            "Error position tracking",
            "Error recovery",
            "Error message improvement"
        ],
        "Grammar Construction": [
            "Recursive parser definition",
            "Left recursion handling",
            "Precedence with combinators",
            "Complete grammar example"
        ]
    },
    "peer-to-peer": {
        "Node Discovery": [
            "Bootstrap node connection",
            "Peer exchange protocol",
            "DHT for decentralized discovery",
            "NAT traversal integration"
        ],
        "Messaging Protocol": [
            "Message framing",
            "Message routing",
            "Request-response pattern",
            "Broadcast/multicast"
        ],
        "DHT Implementation": [
            "Kademlia routing table",
            "Node ID and distance",
            "Find node operation",
            "Store/retrieve values"
        ],
        "Data Synchronization": [
            "Merkle tree for sync",
            "Chunk-based data transfer",
            "Data availability",
            "Replication factor"
        ]
    },
    "penetration-testing-tool": {
        "Reconnaissance": [
            "DNS enumeration",
            "Subdomain discovery",
            "WHOIS lookup",
            "Technology fingerprinting"
        ],
        "Vulnerability Scanning": [
            "Port scanning",
            "Service detection",
            "CVE database lookup",
            "Vulnerability assessment"
        ],
        "Exploitation Framework": [
            "Exploit module loading",
            "Payload generation",
            "Exploit execution",
            "Post-exploitation helpers"
        ],
        "Reporting": [
            "Finding documentation",
            "Severity rating",
            "Remediation suggestions",
            "Report generation (PDF, HTML)"
        ]
    },
    "profiler": {
        "Sampling": [
            "Timer-based sampling (SIGPROF)",
            "Stack unwinding",
            "Symbol resolution",
            "Sample collection"
        ],
        "Instrumentation": [
            "Function entry/exit hooks",
            "Compile-time instrumentation",
            "Runtime instrumentation",
            "Call graph construction"
        ],
        "Data Collection": [
            "CPU time measurement",
            "Memory allocation tracking",
            "Lock contention tracking",
            "I/O wait tracking"
        ],
        "Visualization": [
            "Flame graph generation",
            "Call tree view",
            "Hot path identification",
            "Profile comparison"
        ]
    },
    "proxy-server": {
        "Connection Handling": [
            "Client connection accept",
            "Request parsing",
            "Upstream connection",
            "Response forwarding"
        ],
        "HTTP Proxy": [
            "HTTP CONNECT for HTTPS",
            "Request modification (headers)",
            "Response modification",
            "Caching proxy option"
        ],
        "SOCKS Proxy": [
            "SOCKS5 handshake",
            "Authentication support",
            "TCP relay",
            "UDP relay option"
        ],
        "Filtering": [
            "URL blacklist/whitelist",
            "Content filtering",
            "Request logging",
            "Access control"
        ]
    },
    "query-engine": {
        "Scan Operators": [
            "Full table scan",
            "Index scan",
            "Column projection",
            "Predicate filtering"
        ],
        "Join Operators": [
            "Nested loop join",
            "Hash join",
            "Sort-merge join",
            "Join condition evaluation"
        ],
        "Aggregation": [
            "Group by implementation",
            "Aggregate functions (sum, count, avg, min, max)",
            "Having clause filtering",
            "Distinct elimination"
        ],
        "Query Plan": [
            "Plan tree representation",
            "Plan execution iterator model",
            "Operator pipelining",
            "Memory management"
        ]
    },
    "secrets-manager": {
        "Secret Storage": [
            "Encrypted secret storage",
            "Master key management",
            "Secret versioning",
            "Secret metadata"
        ],
        "Access Control": [
            "RBAC for secrets",
            "Policy definition",
            "Access audit logging",
            "Token-based authentication"
        ],
        "Secret Operations": [
            "Create/update/delete secrets",
            "Secret retrieval API",
            "Secret rotation",
            "Dynamic secrets"
        ],
        "Integration": [
            "Kubernetes integration",
            "CI/CD integration",
            "SDK for applications",
            "Secret injection"
        ]
    },
    "semantic-analyzer": {
        "Symbol Table": [
            "Symbol table construction",
            "Scope management (enter/exit)",
            "Symbol lookup in scope chain",
            "Symbol redefinition detection"
        ],
        "Type System": [
            "Type representation",
            "Type checking expressions",
            "Type compatibility rules",
            "Type inference (optional)"
        ],
        "Name Resolution": [
            "Variable reference resolution",
            "Function call resolution",
            "Import/module resolution",
            "Forward reference handling"
        ],
        "Semantic Checks": [
            "Return type checking",
            "Break/continue validity",
            "Initialization checking",
            "Dead code warnings"
        ]
    },
    "service-discovery": {
        "Registration": [
            "Service registration API",
            "Health check configuration",
            "Metadata and tags",
            "TTL and heartbeat"
        ],
        "Discovery": [
            "Service lookup by name",
            "Instance listing",
            "Filtering by metadata",
            "DNS interface option"
        ],
        "Health Checking": [
            "Health check execution",
            "HTTP/TCP/gRPC health checks",
            "Unhealthy instance removal",
            "Health status API"
        ],
        "Load Balancing": [
            "Round-robin selection",
            "Weighted selection",
            "Zone-aware selection",
            "Client-side load balancing"
        ]
    },
    "session-manager": {
        "Session Creation": [
            "Session ID generation",
            "Session data storage",
            "Session cookie setting",
            "Secure session ID"
        ],
        "Session Storage": [
            "In-memory storage",
            "Redis/database storage",
            "Session serialization",
            "Storage adapter interface"
        ],
        "Session Operations": [
            "Get/set session data",
            "Session invalidation",
            "Session regeneration",
            "Sliding expiration"
        ],
        "Security": [
            "Session fixation prevention",
            "CSRF token in session",
            "Secure cookie flags",
            "Session binding (IP, user-agent)"
        ]
    },
    "sha-impl": {
        "Message Preprocessing": [
            "Message padding",
            "Length encoding",
            "Block division",
            "Big-endian handling"
        ],
        "Hash Functions": [
            "Initial hash values",
            "Round constants",
            "Compression function",
            "Message schedule"
        ],
        "SHA-256": [
            "32-bit word operations",
            "64 rounds processing",
            "Ch and Maj functions",
            "Sigma functions"
        ],
        "Testing": [
            "Test vectors verification",
            "Performance comparison",
            "Streaming hash support",
            "HMAC construction"
        ]
    },
    "shell-interpreter": {
        "Lexer": [
            "Word tokenization",
            "Operator recognition",
            "Quote state handling",
            "Here-doc handling"
        ],
        "Parser": [
            "Command parsing",
            "Pipeline parsing",
            "Conditional parsing (if/while)",
            "Function definition"
        ],
        "Expansion": [
            "Variable expansion",
            "Command substitution",
            "Arithmetic expansion",
            "Glob expansion"
        ],
        "Execution": [
            "Simple command execution",
            "Built-in dispatch",
            "Compound command execution",
            "Exit status handling"
        ]
    },
    "simple-assembler": {
        "Lexer": [
            "Mnemonic tokenization",
            "Operand parsing",
            "Label recognition",
            "Directive handling"
        ],
        "Parser": [
            "Instruction parsing",
            "Operand type detection",
            "Label definition tracking",
            "Directive processing"
        ],
        "Code Generation": [
            "Opcode lookup table",
            "Operand encoding",
            "Address calculation",
            "Label resolution"
        ],
        "Output": [
            "Binary output format",
            "Object file format (optional)",
            "Listing file generation",
            "Symbol table output"
        ]
    },
    "slab-allocator": {
        "Slab Structure": [
            "Slab layout (objects + metadata)",
            "Object slot management",
            "Free list within slab",
            "Slab to page mapping"
        ],
        "Cache Management": [
            "Cache per object size/type",
            "Cache creation and destruction",
            "Slab allocation for cache",
            "Cache statistics"
        ],
        "Allocation": [
            "Object allocation from cache",
            "Empty slab handling",
            "Partial slab preference",
            "New slab allocation"
        ],
        "Deallocation": [
            "Object return to cache",
            "Slab state transition",
            "Empty slab reclamation",
            "Magazine layer (optional)"
        ]
    },
    "snapshot-isolation": {
        "Version Storage": [
            "Multi-version record storage",
            "Version chain management",
            "Garbage collection of old versions",
            "Version visibility"
        ],
        "Snapshot Management": [
            "Snapshot timestamp assignment",
            "Active snapshot tracking",
            "Snapshot isolation level",
            "Snapshot read consistency"
        ],
        "Read Path": [
            "Version selection for snapshot",
            "Visibility check",
            "Consistent read without locks",
            "Snapshot too old handling"
        ],
        "Write Path": [
            "Write intent tracking",
            "Write-write conflict detection",
            "First-committer-wins",
            "Serializable snapshot isolation (optional)"
        ]
    },
    "spatial-index": {
        "R-tree Structure": [
            "Bounding box node",
            "Leaf entries (object MBR)",
            "Tree depth balancing",
            "Node capacity"
        ],
        "Insertion": [
            "Choose subtree algorithm",
            "Node splitting (linear, quadratic)",
            "MBR expansion",
            "Tree rebalancing"
        ],
        "Search": [
            "Range query (window)",
            "Point query",
            "Nearest neighbor",
            "Overlap test"
        ],
        "Deletion": [
            "Entry removal",
            "Underflow handling",
            "Reinsertion strategy",
            "Condense tree"
        ]
    },
    "sql-migrator": {
        "Migration Files": [
            "Migration file format",
            "Up and down scripts",
            "Migration ordering",
            "Migration generation command"
        ],
        "State Tracking": [
            "Applied migrations table",
            "Version tracking",
            "Pending migrations detection",
            "Migration history"
        ],
        "Execution": [
            "Migration runner",
            "Transaction wrapping",
            "Error handling and rollback",
            "Partial migration handling"
        ],
        "Commands": [
            "Migrate up command",
            "Migrate down (rollback)",
            "Migration status",
            "Reset database"
        ]
    },
    "state-machine": {
        "State Definition": [
            "State enumeration",
            "Initial state specification",
            "Final states",
            "State metadata"
        ],
        "Transitions": [
            "Transition rules (from, event, to)",
            "Guard conditions",
            "Transition actions",
            "Transition table"
        ],
        "Event Handling": [
            "Event dispatch",
            "Event queue (async)",
            "Unhandled event handling",
            "Internal transitions"
        ],
        "Execution": [
            "State entry/exit actions",
            "Hierarchical states (optional)",
            "State machine serialization",
            "Visualization (DOT graph)"
        ]
    },
    "template-engine": {
        "Template Parsing": [
            "Template syntax lexer",
            "Variable placeholder parsing",
            "Control structure parsing",
            "Template to AST"
        ],
        "Compilation": [
            "AST to render function",
            "Template caching",
            "Pre-compilation option",
            "Syntax validation"
        ],
        "Rendering": [
            "Context data binding",
            "Variable interpolation",
            "Conditional rendering",
            "Loop rendering"
        ],
        "Features": [
            "HTML escaping",
            "Custom filters",
            "Template inheritance",
            "Partial/include support"
        ]
    },
    "transaction-manager": {
        "Transaction Lifecycle": [
            "Begin transaction",
            "Commit transaction",
            "Rollback transaction",
            "Transaction state machine"
        ],
        "Concurrency Control": [
            "Lock manager",
            "Two-phase locking",
            "Deadlock detection",
            "Lock wait queue"
        ],
        "Recovery": [
            "Write-ahead logging",
            "Undo logging",
            "Redo logging",
            "ARIES recovery algorithm"
        ],
        "Isolation Levels": [
            "Read uncommitted",
            "Read committed",
            "Repeatable read",
            "Serializable"
        ]
    },
    "trie-impl": {
        "Node Structure": [
            "Children map/array",
            "End-of-word marker",
            "Value storage (optional)",
            "Compact representation"
        ],
        "Insertion": [
            "Character-by-character insertion",
            "Node creation as needed",
            "End marker setting",
            "Value association"
        ],
        "Search": [
            "Exact match search",
            "Prefix existence check",
            "Prefix listing (autocomplete)",
            "Longest prefix match"
        ],
        "Deletion": [
            "End marker removal",
            "Node cleanup (no children)",
            "Recursive deletion",
            "Prefix deletion"
        ]
    },
    "unit-test-framework": {
        "Test Definition": [
            "Test function decorator/macro",
            "Test class support",
            "Test naming",
            "Test metadata (tags, skip)"
        ],
        "Assertions": [
            "Equality assertions",
            "Boolean assertions",
            "Exception assertions",
            "Approximate equality"
        ],
        "Test Runner": [
            "Test discovery",
            "Test execution isolation",
            "Setup/teardown hooks",
            "Test filtering"
        ],
        "Output": [
            "Pass/fail summary",
            "Failure details",
            "Timing information",
            "Coverage integration"
        ]
    },
    "virtual-memory": {
        "Page Table": [
            "Page table entry format",
            "Multi-level page tables",
            "Page table walking",
            "TLB simulation"
        ],
        "Address Translation": [
            "Virtual to physical mapping",
            "Page offset extraction",
            "Page table lookup",
            "Permission checking"
        ],
        "Page Fault Handling": [
            "Page fault detection",
            "Demand paging",
            "Page allocation",
            "Copy-on-write"
        ],
        "Page Replacement": [
            "FIFO replacement",
            "LRU replacement",
            "Clock algorithm",
            "Working set tracking"
        ]
    },
    "wal-impl": {
        "Log Structure": [
            "Log record format",
            "LSN (Log Sequence Number)",
            "Log file segmentation",
            "Log header/footer"
        ],
        "Write Path": [
            "Log record appending",
            "Forced write (fsync)",
            "Group commit",
            "Write buffering"
        ],
        "Recovery": [
            "Log scanning",
            "Redo pass",
            "Undo pass",
            "Checkpoint integration"
        ],
        "Checkpointing": [
            "Checkpoint record",
            "Active transaction list",
            "Dirty page tracking",
            "Checkpoint coordination"
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


if __name__ == "__main__":
    main()

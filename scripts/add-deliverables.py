#!/usr/bin/env python3
"""Add deliverables to all projects missing them."""

import yaml
from pathlib import Path

# Comprehensive deliverables mapping by project_id -> milestone_name -> deliverables
DELIVERABLES = {
    "2pc-impl": {
        "Transaction Log": [
            "Write-ahead log with fsync durability",
            "Log record format (tx_id, state, participants)",
            "Recovery function to replay log on restart",
            "Log compaction/truncation after commit"
        ],
        "Prepare Phase": [
            "Coordinator sends PREPARE to all participants",
            "Participants acquire locks and respond VOTE_COMMIT/ABORT",
            "Timeout handling for unresponsive participants",
            "Prepare state persisted to log before response"
        ],
        "Commit Phase": [
            "Coordinator decides COMMIT if all voted yes",
            "COMMIT/ABORT message broadcast to participants",
            "Participants apply changes and release locks",
            "Acknowledgment collection from participants"
        ],
        "Failure Recovery": [
            "Coordinator crash recovery (re-read log, resume protocol)",
            "Participant crash recovery (check log, query coordinator)",
            "Blocking protocol handling for coordinator failure",
            "Presumed abort optimization"
        ]
    },
    "aes-impl": {
        "Galois Field Arithmetic": [
            "GF(2^8) multiplication using irreducible polynomial",
            "Lookup tables for S-box and inverse S-box",
            "XOR operations for field addition",
            "MixColumns multiplication implementation"
        ],
        "AES Core Operations": [
            "SubBytes transformation using S-box",
            "ShiftRows row rotation",
            "MixColumns column mixing",
            "AddRoundKey XOR with round key"
        ],
        "Key Expansion": [
            "Round constant (Rcon) generation",
            "Key schedule algorithm for 128/192/256-bit keys",
            "Correct number of round keys generated",
            "Word rotation and S-box application"
        ],
        "Encryption & Modes": [
            "Full AES encryption/decryption rounds",
            "ECB mode implementation",
            "CBC mode with IV handling",
            "CTR mode for streaming encryption"
        ]
    },
    "ast-builder": {
        "AST Node Definitions": [
            "Base Node class with source location tracking",
            "Expression node types (literal, binary, unary, identifier)",
            "Statement node types (if, while, return, block)",
            "Visitor pattern for tree traversal"
        ],
        "Recursive Descent Parser": [
            "Token stream consumption and lookahead",
            "Precedence climbing for operators",
            "Recursive functions for grammar rules",
            "Correct operator associativity handling"
        ],
        "Statement Parsing": [
            "Variable declaration parsing",
            "Function definition parsing with parameters",
            "Control flow statement parsing (if/while/for)",
            "Block statement with scope handling"
        ],
        "Error Recovery": [
            "Synchronization points for error recovery",
            "Multiple error collection (don't stop at first)",
            "Meaningful error messages with location",
            "Panic mode recovery to continue parsing"
        ]
    },
    "ast-interpreter": {
        "Expression Evaluation": [
            "Literal value evaluation (numbers, strings, bools)",
            "Binary operator evaluation with type checking",
            "Unary operator evaluation",
            "Short-circuit evaluation for && and ||"
        ],
        "Variables and Environment": [
            "Environment class with variable bindings",
            "Nested scopes with parent environment chain",
            "Variable declaration and assignment",
            "Undefined variable error handling"
        ],
        "Control Flow": [
            "If/else statement execution",
            "While loop execution with break/continue",
            "For loop desugaring to while",
            "Return statement with value propagation"
        ],
        "Functions": [
            "Function declaration storing in environment",
            "Function call with argument binding",
            "Closure capture of enclosing environment",
            "Recursion support"
        ]
    },
    "blog-platform": {
        "Project Setup & Database Schema": [
            "Database schema for users, posts, comments",
            "Migration files for schema versioning",
            "Database connection pooling setup",
            "Basic project structure with routes"
        ],
        "User Authentication": [
            "User registration with password hashing",
            "Login with JWT or session tokens",
            "Password reset functionality",
            "Protected route middleware"
        ],
        "Blog CRUD Operations": [
            "Create post with title, content, author",
            "Read posts with pagination",
            "Update post (author only)",
            "Delete post with soft delete option"
        ],
        "Frontend UI": [
            "Post listing page with pagination",
            "Single post view with comments",
            "Post editor with markdown support",
            "Responsive design for mobile"
        ]
    },
    "btree-impl": {
        "B-tree Node Structure": [
            "Node class with keys and children arrays",
            "Configurable order (minimum degree t)",
            "Leaf vs internal node distinction",
            "Disk page representation"
        ],
        "Search": [
            "Binary search within node keys",
            "Recursive descent to correct child",
            "Return value and found flag",
            "O(log n) time complexity"
        ],
        "Insert with Split": [
            "Find correct leaf for insertion",
            "Split full nodes during descent (proactive)",
            "Median key promotion to parent",
            "Handle root split creating new root"
        ],
        "Delete with Rebalancing": [
            "Delete from leaf node",
            "Delete from internal (predecessor/successor swap)",
            "Borrow from sibling when underflow",
            "Merge with sibling when borrow fails"
        ]
    },
    "build-allocator": {
        "Basic Allocator with sbrk": [
            "sbrk() system call for heap extension",
            "Block header with size and allocation status",
            "First-fit allocation strategy",
            "Simple free() implementation"
        ],
        "Free List Management": [
            "Explicit free list with next/prev pointers",
            "Coalescing adjacent free blocks",
            "Best-fit or next-fit strategy",
            "Boundary tags for efficient coalescing"
        ],
        "Segregated Free Lists": [
            "Size classes (e.g., 16, 32, 64, 128... bytes)",
            "Separate free list per size class",
            "Fast allocation from matching class",
            "Splitting larger blocks when needed"
        ],
        "Thread Safety & mmap": [
            "Per-thread arenas to reduce contention",
            "Lock-free fast path for small allocations",
            "mmap() for large allocations",
            "Thread-local caching"
        ]
    },
    "build-bittorrent": {
        "Torrent File Parsing": [
            "Bencode decoder (strings, integers, lists, dicts)",
            "Metainfo extraction (announce, info dict)",
            "Info hash calculation (SHA1 of info dict)",
            "Piece length and piece hashes extraction"
        ],
        "Tracker Communication": [
            "HTTP GET request to announce URL",
            "Compact peer list parsing",
            "Periodic announce with uploaded/downloaded",
            "Tracker response handling (interval, peers)"
        ],
        "Peer Protocol": [
            "TCP connection and handshake",
            "Message parsing (choke, unchoke, interested, have, bitfield, request, piece)",
            "Peer state machine (am_choking, am_interested, etc.)",
            "Request pipelining for throughput"
        ],
        "Piece Management & Seeding": [
            "Piece verification with SHA1 hash",
            "Rarest-first piece selection strategy",
            "Endgame mode for last pieces",
            "Upload to peers (seeding)"
        ]
    },
    "build-browser": {
        "HTML Parser": [
            "Tokenizer for HTML tags and text",
            "DOM tree construction from tokens",
            "Handle self-closing and void elements",
            "Basic error recovery for malformed HTML"
        ],
        "CSS Parser": [
            "CSS tokenizer (selectors, properties, values)",
            "Selector parsing (tag, class, id, combinators)",
            "Specificity calculation",
            "Style rule storage structure"
        ],
        "Layout": [
            "Box model calculation (margin, border, padding)",
            "Block layout with vertical stacking",
            "Inline layout with line breaking",
            "Layout tree from styled DOM"
        ],
        "Rendering": [
            "Paint list generation from layout tree",
            "Rectangle and text drawing commands",
            "Basic display list optimization",
            "Render to canvas or image buffer"
        ]
    },
    "build-bundler": {
        "Module Parsing": [
            "JavaScript/TypeScript parser integration",
            "Import/export statement extraction",
            "Dynamic import() detection",
            "CommonJS require() handling"
        ],
        "Module Resolution": [
            "Relative path resolution",
            "Node modules resolution algorithm",
            "Package.json main/module field handling",
            "Dependency graph construction"
        ],
        "Bundle Generation": [
            "Module wrapping in function scope",
            "Runtime module loader code",
            "Topological sort for module order",
            "Source map generation"
        ],
        "Tree Shaking": [
            "Dead code detection via static analysis",
            "Side effect annotation handling",
            "Unused export removal",
            "Bundle size optimization"
        ]
    },
    "build-ci-system": {
        "Pipeline Configuration Parser": [
            "YAML pipeline definition parsing",
            "Stage and job dependency graph",
            "Environment variable substitution",
            "Matrix build configuration"
        ],
        "Job Execution Engine": [
            "Docker container job execution",
            "Script step runner with logging",
            "Artifact collection and storage",
            "Parallel job execution"
        ],
        "Webhook & Queue System": [
            "GitHub/GitLab webhook handler",
            "Job queue with Redis or database",
            "Worker process pool",
            "Pipeline triggering on push/PR"
        ],
        "Web Dashboard": [
            "Build history and status display",
            "Real-time log streaming",
            "Pipeline visualization (DAG view)",
            "Build artifact download"
        ]
    },
    "build-debugger": {
        "Process Control": [
            "ptrace() attachment to process",
            "Process start/stop control",
            "Single-step execution (PTRACE_SINGLESTEP)",
            "Continue execution (PTRACE_CONT)"
        ],
        "Breakpoints": [
            "Software breakpoint via INT3 (0xCC)",
            "Original instruction preservation",
            "Breakpoint enable/disable",
            "Hit count tracking"
        ],
        "Symbol Tables": [
            "DWARF debug info parsing",
            "Function name to address mapping",
            "Source file and line number mapping",
            "Variable location information"
        ],
        "Variable Inspection": [
            "Read memory at variable address",
            "Type-aware value formatting",
            "Struct and array member access",
            "Register value reading"
        ]
    },
    "build-distributed-kv": {
        "Consistent Hashing": [
            "Hash ring implementation",
            "Virtual nodes for load balancing",
            "Node addition/removal with minimal rehashing",
            "Key to node mapping"
        ],
        "Replication": [
            "Configurable replication factor",
            "Quorum read/write (R + W > N)",
            "Replica synchronization",
            "Anti-entropy with Merkle trees"
        ],
        "Cluster Management": [
            "Node discovery and membership",
            "Gossip protocol for state propagation",
            "Failure detection with heartbeats",
            "Handoff for failed nodes"
        ],
        "Transactions": [
            "Single-key transactions with CAS",
            "Multi-key transactions (2PC or Percolator)",
            "MVCC for snapshot isolation",
            "Conflict resolution (last-write-wins or CRDTs)"
        ]
    },
    "build-dns": {
        "DNS Message Parsing": [
            "DNS header parsing (ID, flags, counts)",
            "Question section parsing",
            "Resource record parsing (A, AAAA, CNAME, MX, NS)",
            "DNS message serialization"
        ],
        "Authoritative Server": [
            "Zone file parsing",
            "Query matching against zone data",
            "SOA, NS record handling",
            "Authority and additional sections"
        ],
        "Recursive Resolver": [
            "Root server hints",
            "Iterative query following NS referrals",
            "CNAME following",
            "Response construction from resolution"
        ],
        "Caching & Performance": [
            "TTL-based cache with expiration",
            "Negative caching (NXDOMAIN)",
            "Cache lookup before recursive resolution",
            "Concurrent query handling"
        ]
    },
    "build-docker": {
        "Process Isolation (Namespaces)": [
            "PID namespace for process isolation",
            "Network namespace setup",
            "Mount namespace for filesystem",
            "UTS namespace for hostname"
        ],
        "Resource Limits (cgroups)": [
            "Memory limit cgroup setup",
            "CPU shares/quota limiting",
            "PIDs limit to prevent fork bombs",
            "Cgroup filesystem interaction"
        ],
        "Filesystem Isolation (chroot/pivot_root)": [
            "Root filesystem extraction",
            "pivot_root() for secure chroot",
            "Mount /proc, /sys inside container",
            "Unmount old root"
        ],
        "Layered Filesystem (OverlayFS)": [
            "OverlayFS mount with lower/upper/work dirs",
            "Layer stacking from image",
            "Copy-on-write behavior",
            "Layer caching and reuse"
        ],
        "Container Networking": [
            "Virtual ethernet pair (veth) creation",
            "Bridge network setup",
            "NAT/masquerade for outbound traffic",
            "Port forwarding to container"
        ],
        "Image Format and CLI": [
            "OCI image spec parsing",
            "Image pull from registry",
            "Container run command",
            "Container lifecycle management (start/stop/rm)"
        ]
    },
    "build-emulator": {
        "CPU Emulation": [
            "Instruction fetch-decode-execute loop",
            "Register file implementation",
            "ALU operations for target ISA",
            "Program counter and flags handling"
        ],
        "Memory System": [
            "Memory address space emulation",
            "Memory-mapped I/O regions",
            "ROM and RAM regions",
            "Bank switching if applicable"
        ],
        "Graphics": [
            "Display buffer (framebuffer)",
            "Sprite/tile rendering if applicable",
            "Scanline timing emulation",
            "Screen output to window"
        ],
        "Timing and Input": [
            "Cycle-accurate timing",
            "Input device mapping (keyboard/gamepad)",
            "Interrupt handling",
            "Audio timing synchronization"
        ]
    },
    "build-game-engine": {
        "Window & Rendering Foundation": [
            "Window creation with SDL/GLFW",
            "OpenGL/Vulkan context setup",
            "Render loop with delta time",
            "Basic sprite/mesh rendering"
        ],
        "Entity Component System": [
            "Entity ID management",
            "Component storage (dense arrays)",
            "System iteration over components",
            "Entity queries with component filters"
        ],
        "Physics & Collision": [
            "AABB collision detection",
            "Spatial partitioning (grid or quadtree)",
            "Rigid body physics integration",
            "Collision response and resolution"
        ],
        "Resource & Scene Management": [
            "Asset loading (textures, models, audio)",
            "Resource caching and reference counting",
            "Scene graph or level structure",
            "Scene serialization/deserialization"
        ]
    },
    "build-gc": {
        "Mark-Sweep Collector": [
            "Root set identification (stack, globals)",
            "Mark phase traversing object graph",
            "Sweep phase collecting unmarked objects",
            "Free list management for reclaimed memory"
        ],
        "Tri-color Marking": [
            "White/gray/black coloring scheme",
            "Worklist-based marking",
            "Write barrier for incremental marking",
            "Correct termination condition"
        ],
        "Generational Collection": [
            "Young/old generation separation",
            "Minor collection of young generation",
            "Object promotion to old generation",
            "Remembered set for cross-generation pointers"
        ],
        "Concurrent Collection": [
            "Concurrent marking with mutator",
            "Snapshot-at-the-beginning or incremental update",
            "Safe points for GC coordination",
            "Low-pause collection"
        ]
    },
    "build-git": {
        "Repository Initialization": [
            ".git directory structure creation",
            "HEAD file pointing to refs/heads/master",
            "objects and refs directories",
            "Initial config file"
        ],
        "Object Storage (Blobs)": [
            "SHA1 hash computation for content",
            "Zlib compression of objects",
            "Object file path from hash",
            "Blob object format (blob <size>\\0<content>)"
        ],
        "Tree Objects": [
            "Tree object format (mode name\\0hash entries)",
            "Directory to tree conversion",
            "Tree hash calculation",
            "Recursive tree building"
        ],
        "Commit Objects": [
            "Commit object format (tree, parent, author, message)",
            "Parent commit linking",
            "Timestamp and timezone handling",
            "Commit hash calculation"
        ],
        "References and Branches": [
            "Refs as files containing commit hash",
            "Branch creation as ref file",
            "HEAD as symbolic ref",
            "Detached HEAD handling"
        ],
        "Index (Staging Area)": [
            "Index file binary format parsing",
            "Add file to index",
            "Remove file from index",
            "Index to tree conversion"
        ],
        "Diff Algorithm": [
            "Myers diff algorithm implementation",
            "Line-based diff output",
            "Unified diff format",
            "Diff between trees"
        ],
        "Merge (Three-way)": [
            "Common ancestor finding",
            "Three-way merge algorithm",
            "Conflict detection and marking",
            "Merge commit creation"
        ]
    },
    "build-interpreter": {
        "Scanner (Lexer)": [
            "Character stream to token stream",
            "Token types (keywords, identifiers, literals, operators)",
            "Line and column tracking",
            "String and number literal scanning"
        ],
        "Representing Code (AST)": [
            "Expression AST node classes",
            "Statement AST node classes",
            "Visitor pattern for AST traversal",
            "Pretty printer for debugging"
        ],
        "Parsing Expressions": [
            "Recursive descent expression parser",
            "Operator precedence handling",
            "Grouping with parentheses",
            "Unary and binary expressions"
        ],
        "Evaluating Expressions": [
            "Tree-walking evaluator",
            "Runtime value representation",
            "Type checking at runtime",
            "Operator semantics implementation"
        ],
        "Statements and State": [
            "Print and expression statements",
            "Variable declaration and assignment",
            "Environment for variable storage",
            "Block statements with scoping"
        ],
        "Control Flow": [
            "If/else statement execution",
            "While loop execution",
            "For loop (desugared to while)",
            "Logical operators with short-circuit"
        ],
        "Functions": [
            "Function declaration",
            "Function call with arguments",
            "Return statement",
            "Local variables in function scope"
        ],
        "Closures": [
            "Closure captures enclosing environment",
            "Free variable resolution",
            "Closure as first-class value",
            "Nested function support"
        ],
        "Classes": [
            "Class declaration",
            "Instance creation",
            "Property get/set",
            "Method definition and this binding"
        ],
        "Inheritance": [
            "Superclass specification",
            "Method inheritance",
            "super keyword for parent methods",
            "Initializer chaining"
        ]
    },
    "build-jit": {
        "Bytecode Design": [
            "Bytecode instruction set design",
            "Bytecode encoding format",
            "Bytecode assembler/disassembler",
            "Bytecode verification"
        ],
        "Bytecode Interpreter": [
            "Dispatch loop (switch or computed goto)",
            "Operand stack management",
            "Local variable slots",
            "Call frame management"
        ],
        "JIT Compilation": [
            "Hot function detection",
            "IR generation from bytecode",
            "Machine code generation (x86-64)",
            "Code patching for OSR"
        ],
        "Optimizations": [
            "Inline caching for property access",
            "Type specialization",
            "Function inlining",
            "Dead code elimination"
        ]
    },
    "build-kafka": {
        "Log Storage Engine": [
            "Append-only log segments",
            "Segment rolling at size/time threshold",
            "Index files for offset lookup",
            "Memory-mapped file access"
        ],
        "Producer API": [
            "Message batching and compression",
            "Partitioner (key hash or round-robin)",
            "Acknowledgment modes (0, 1, all)",
            "Retry with idempotent producer"
        ],
        "Consumer Groups": [
            "Consumer group membership protocol",
            "Partition assignment strategies",
            "Offset commit and fetch",
            "Rebalancing on consumer join/leave"
        ],
        "Replication": [
            "Leader election per partition",
            "ISR (in-sync replicas) tracking",
            "Follower fetch from leader",
            "High watermark advancement"
        ]
    },
    "build-kubernetes": {
        "API Server": [
            "REST API for resource CRUD",
            "etcd integration for persistence",
            "Admission controllers",
            "Authentication and RBAC"
        ],
        "Scheduler": [
            "Node filtering (resource requests, affinity)",
            "Node scoring (least requested, balanced)",
            "Pod binding to selected node",
            "Priority and preemption"
        ],
        "Controller Manager": [
            "Control loop pattern (observe, diff, act)",
            "Deployment controller (ReplicaSet management)",
            "ReplicaSet controller (Pod count maintenance)",
            "Workqueue with rate limiting"
        ],
        "Kubelet": [
            "Pod lifecycle management",
            "Container runtime interface (CRI)",
            "Volume mounting",
            "Health checking (liveness, readiness)"
        ]
    },
    "build-linux-shell": {
        "Command Parsing": [
            "Input tokenization (words, operators)",
            "Quote handling (single, double, escape)",
            "Variable expansion ($VAR, ${VAR})",
            "Command substitution $(command)"
        ],
        "Process Execution": [
            "fork() and execvp() for command execution",
            "PATH searching for executables",
            "Exit status handling ($?)",
            "Background execution (&)"
        ],
        "Pipes and Redirection": [
            "Pipe creation with pipe()",
            "File redirection (>, <, >>)",
            "Stderr redirection (2>)",
            "Pipeline of multiple commands"
        ],
        "Job Control": [
            "Foreground/background job management",
            "SIGTSTP (Ctrl+Z) handling",
            "fg and bg builtins",
            "Job table maintenance"
        ],
        "Builtins": [
            "cd builtin with PWD update",
            "export for environment variables",
            "exit builtin",
            "source/. for script execution"
        ]
    },
    "build-llm": {
        "Tokenizer": [
            "BPE or SentencePiece tokenization",
            "Vocabulary loading from file",
            "Encode text to token IDs",
            "Decode token IDs to text"
        ],
        "Model Loading": [
            "Weight file format parsing (safetensors/GGUF)",
            "Model config loading (layers, heads, dims)",
            "Memory-mapped weight loading",
            "Quantized weight support"
        ],
        "Transformer Inference": [
            "Embedding lookup",
            "Multi-head attention computation",
            "RMSNorm/LayerNorm",
            "Feed-forward network"
        ],
        "KV Cache & Generation": [
            "Key-value cache for past tokens",
            "Autoregressive generation loop",
            "Sampling strategies (top-k, top-p, temperature)",
            "Stop token detection"
        ]
    },
    "build-load-balancer": {
        "TCP/UDP Proxy": [
            "Socket accept and forward",
            "Connection pooling to backends",
            "Timeout handling",
            "Graceful connection draining"
        ],
        "Load Balancing Algorithms": [
            "Round-robin distribution",
            "Least connections algorithm",
            "Weighted distribution",
            "Consistent hashing for sticky sessions"
        ],
        "Health Checks": [
            "TCP connect health check",
            "HTTP health check with expected response",
            "Health check interval and threshold",
            "Automatic backend removal/addition"
        ],
        "Configuration": [
            "Configuration file parsing",
            "Hot reload without restart",
            "Backend pool management",
            "Logging and metrics"
        ]
    },
    "build-lsm-tree": {
        "MemTable": [
            "Sorted in-memory structure (skip list or red-black tree)",
            "Write-ahead log for durability",
            "Size threshold for flush",
            "Concurrent read/write support"
        ],
        "SSTable": [
            "Sorted string table file format",
            "Block-based storage with index",
            "Bloom filter for existence check",
            "Compression per block"
        ],
        "Compaction": [
            "Level-based or size-tiered compaction",
            "Merge sorted runs",
            "Tombstone garbage collection",
            "Background compaction thread"
        ],
        "Read Path": [
            "MemTable lookup first",
            "Level-by-level SSTable search",
            "Bloom filter to skip SSTables",
            "Block cache for hot data"
        ]
    },
    "build-ml-framework": {
        "Tensor Operations": [
            "N-dimensional array (tensor) class",
            "Element-wise operations (add, mul, etc.)",
            "Matrix multiplication",
            "Broadcasting support"
        ],
        "Autograd": [
            "Computation graph construction",
            "Backward pass implementation",
            "Gradient accumulation",
            "Gradient tape or reverse-mode AD"
        ],
        "Neural Network Layers": [
            "Linear layer with weights and bias",
            "Activation functions (ReLU, sigmoid, tanh)",
            "Convolutional layer (Conv2D)",
            "Dropout and batch normalization"
        ],
        "Optimizer & Training": [
            "SGD optimizer with momentum",
            "Adam optimizer",
            "Loss functions (MSE, cross-entropy)",
            "Training loop with forward/backward/step"
        ]
    },
    "build-network-stack": {
        "Ethernet Frame Handling": [
            "Raw socket or TAP device setup",
            "Ethernet frame parsing (dst, src, type)",
            "Frame construction and sending",
            "ARP request/response handling"
        ],
        "IP Layer": [
            "IPv4 header parsing and construction",
            "IP fragmentation and reassembly",
            "Routing table lookup",
            "ICMP echo request/reply (ping)"
        ],
        "TCP Implementation": [
            "TCP state machine implementation",
            "Three-way handshake (SYN, SYN-ACK, ACK)",
            "Sequence number tracking",
            "Retransmission with timeout"
        ],
        "Socket API": [
            "Socket creation and binding",
            "Listen and accept for servers",
            "Connect for clients",
            "Send and receive with buffering"
        ]
    },
    "build-os-kernel": {
        "Bootloader": [
            "BIOS/UEFI boot protocol",
            "Kernel loading into memory",
            "Transition to protected/long mode",
            "Handoff to kernel entry point"
        ],
        "Memory Management": [
            "Physical memory allocator (bitmap or buddy)",
            "Virtual memory with page tables",
            "Kernel heap allocator",
            "User space memory mapping"
        ],
        "Process Management": [
            "Process control block (PCB)",
            "Context switching",
            "Process creation (fork equivalent)",
            "Process scheduling (round-robin)"
        ],
        "System Calls": [
            "System call interface (int 0x80 or syscall)",
            "System call handler dispatch",
            "Basic syscalls (read, write, exit)",
            "User/kernel mode transition"
        ]
    },
    "build-raft": {
        "Leader Election": [
            "Election timeout randomization",
            "RequestVote RPC implementation",
            "Vote granting logic",
            "Term number management"
        ],
        "Log Replication": [
            "AppendEntries RPC implementation",
            "Log consistency check",
            "Follower log repair",
            "Commit index advancement"
        ],
        "Persistence": [
            "Persist voted_for and current_term",
            "Persist log entries",
            "Crash recovery from persistent state",
            "Snapshot for log compaction"
        ],
        "Client Interaction": [
            "Leader redirect for client requests",
            "Linearizable read semantics",
            "Client request deduplication",
            "Cluster membership changes"
        ]
    },
    "build-raytracer": {
        "Ray Generation": [
            "Camera model with position and orientation",
            "Ray direction from pixel coordinates",
            "Field of view handling",
            "Anti-aliasing with multiple samples"
        ],
        "Intersection Tests": [
            "Ray-sphere intersection",
            "Ray-plane intersection",
            "Ray-triangle intersection",
            "BVH acceleration structure"
        ],
        "Shading": [
            "Diffuse (Lambertian) shading",
            "Specular reflection (Phong)",
            "Shadows via shadow rays",
            "Ambient occlusion"
        ],
        "Materials & Output": [
            "Material system (diffuse, metal, glass)",
            "Fresnel equations for glass",
            "Image output (PPM, PNG)",
            "Gamma correction"
        ]
    },
    "build-redis": {
        "Data Structures": [
            "String type with GET/SET",
            "List type with LPUSH/RPUSH/LPOP/RPOP",
            "Hash type with HGET/HSET",
            "Set and sorted set types"
        ],
        "Protocol": [
            "RESP protocol parsing",
            "Command parsing and dispatch",
            "Error response handling",
            "Bulk string handling"
        ],
        "Persistence": [
            "RDB snapshot (point-in-time dump)",
            "AOF (append-only file) logging",
            "Background save (BGSAVE)",
            "AOF rewrite for compaction"
        ],
        "Expiration": [
            "Key TTL with EXPIRE command",
            "Lazy expiration on access",
            "Active expiration sampling",
            "Memory eviction policies (LRU, LFU)"
        ]
    },
    "build-regex": {
        "Lexer": [
            "Character class parsing ([a-z])",
            "Escape sequence handling (\\d, \\w, \\s)",
            "Quantifier parsing (*, +, ?, {n,m})",
            "Alternation and grouping"
        ],
        "Parser (AST)": [
            "Regex AST node types",
            "Precedence handling (alternation < concatenation < quantifier)",
            "Capturing group numbering",
            "Non-capturing groups (?:...)"
        ],
        "NFA Construction": [
            "Thompson's construction algorithm",
            "Epsilon transitions for |, *, +, ?",
            "State and transition representation",
            "NFA fragment combination"
        ],
        "Matching": [
            "NFA simulation with state sets",
            "Backtracking for captures",
            "Match extraction with group positions",
            "Anchors (^, $) handling"
        ]
    },
    "build-rpc-framework": {
        "Protocol Design": [
            "Message framing (length-prefixed)",
            "Serialization format (protobuf, JSON, msgpack)",
            "Request/response correlation IDs",
            "Metadata/headers support"
        ],
        "Client Stub Generation": [
            "Interface definition parsing",
            "Client proxy code generation",
            "Async call support",
            "Timeout and retry configuration"
        ],
        "Server Skeleton": [
            "Service interface implementation",
            "Request dispatch to handlers",
            "Thread pool for concurrent requests",
            "Graceful shutdown"
        ],
        "Connection Management": [
            "Connection pooling",
            "Multiplexing requests over connection",
            "Reconnection with backoff",
            "Health checking"
        ]
    },
    "build-scheduler": {
        "Task Scheduling": [
            "Cron expression parsing",
            "Next run time calculation",
            "One-time scheduled tasks",
            "Recurring task management"
        ],
        "Execution Engine": [
            "Task execution in worker threads",
            "Task timeout handling",
            "Retry on failure",
            "Task output/error capture"
        ],
        "Persistence": [
            "Task definitions in database",
            "Execution history logging",
            "Missed job detection on restart",
            "Distributed locking for single execution"
        ],
        "API & Monitoring": [
            "REST API for task management",
            "Task status querying",
            "Execution logs viewing",
            "Alerting on failures"
        ]
    },
    "build-search-engine": {
        "Crawler": [
            "URL frontier queue",
            "Robots.txt parsing and respect",
            "HTML fetching and parsing",
            "Link extraction and normalization"
        ],
        "Inverted Index": [
            "Term to document posting list",
            "Term frequency storage",
            "Document frequency for IDF",
            "Index serialization to disk"
        ],
        "Query Processing": [
            "Query parsing (AND, OR, phrases)",
            "Posting list intersection",
            "TF-IDF scoring",
            "Top-k result retrieval"
        ],
        "Ranking": [
            "BM25 scoring algorithm",
            "PageRank or link analysis",
            "Result snippet generation",
            "Search result pagination"
        ]
    },
    "build-sqlite": {
        "SQL Parser": [
            "SQL tokenizer",
            "SELECT statement parsing",
            "INSERT/UPDATE/DELETE parsing",
            "CREATE TABLE parsing"
        ],
        "B-tree Storage": [
            "Page-based B-tree implementation",
            "Table B-tree (rowid keyed)",
            "Index B-tree",
            "Page cache (buffer pool)"
        ],
        "Query Execution": [
            "Table scan operator",
            "Index scan operator",
            "Join operator (nested loop)",
            "Sort and aggregation"
        ],
        "Transactions": [
            "Write-ahead logging (WAL)",
            "ACID transaction support",
            "Rollback journal alternative",
            "Checkpoint mechanism"
        ]
    },
    "build-text-editor": {
        "Terminal Handling": [
            "Raw mode terminal setup",
            "Cursor positioning with ANSI escapes",
            "Screen clearing and refresh",
            "Keyboard input handling"
        ],
        "Buffer Management": [
            "Text buffer data structure (gap buffer or rope)",
            "Line-based editing",
            "Cursor movement (char, word, line)",
            "Insert and delete operations"
        ],
        "File Operations": [
            "File loading into buffer",
            "Save buffer to file",
            "Dirty flag tracking",
            "Backup file creation"
        ],
        "Search & Syntax": [
            "Incremental search (Ctrl+F)",
            "Search and replace",
            "Basic syntax highlighting",
            "Line numbers display"
        ]
    },
    "build-vector-db": {
        "Vector Storage": [
            "Vector embedding storage format",
            "Metadata association with vectors",
            "Batch insert support",
            "Vector normalization options"
        ],
        "Index Structures": [
            "Brute-force exact search baseline",
            "IVF (Inverted File) index",
            "HNSW graph index",
            "Product quantization for compression"
        ],
        "Query Engine": [
            "k-NN query execution",
            "Filtered queries (metadata + similarity)",
            "Batch query support",
            "Approximate vs exact search modes"
        ],
        "API": [
            "REST API for CRUD operations",
            "Query API with filters",
            "Index management (create, rebuild)",
            "Collection management"
        ]
    },
    "build-virtualization": {
        "CPU Virtualization": [
            "KVM/HVF API usage",
            "vCPU creation and execution",
            "VM exit handling",
            "Register state management"
        ],
        "Memory Virtualization": [
            "Guest physical memory allocation",
            "Extended page tables (EPT/NPT)",
            "MMIO region handling",
            "Memory ballooning"
        ],
        "Device Emulation": [
            "Serial port (UART) emulation",
            "Virtio device model",
            "Interrupt injection",
            "PCI configuration space"
        ],
        "I/O Virtualization": [
            "Virtio-blk for disk",
            "Virtio-net for networking",
            "Virtqueue implementation",
            "DMA handling"
        ]
    },
    "build-wasm-runtime": {
        "Binary Parser": [
            "WASM binary format parsing",
            "Section parsing (type, function, memory, etc.)",
            "Function body decoding",
            "Validation of well-formedness"
        ],
        "Interpreter": [
            "Stack machine execution",
            "Instruction dispatch loop",
            "Control flow (block, loop, br, if)",
            "Call and return handling"
        ],
        "Memory & Tables": [
            "Linear memory implementation",
            "Memory grow operation",
            "Table elements for indirect calls",
            "Bounds checking"
        ],
        "Host Integration": [
            "Import function linking",
            "Export function exposure",
            "WASI system call implementation",
            "Memory sharing with host"
        ]
    },
    "build-web-framework": {
        "HTTP Routing": [
            "Route registration (path, method, handler)",
            "Path parameter extraction (/users/:id)",
            "Query string parsing",
            "Route matching and dispatch"
        ],
        "Middleware": [
            "Middleware chain execution",
            "Request/response modification",
            "Built-in middleware (logging, CORS, auth)",
            "Error handling middleware"
        ],
        "Request/Response": [
            "Request body parsing (JSON, form)",
            "Response builder (status, headers, body)",
            "Cookie handling",
            "File upload handling"
        ],
        "Template Engine": [
            "Template loading and caching",
            "Variable interpolation",
            "Control structures (if, for)",
            "Template inheritance/includes"
        ]
    },
    "build-web-server": {
        "HTTP Parsing": [
            "Request line parsing (method, path, version)",
            "Header parsing (key: value)",
            "Body handling (Content-Length)",
            "Chunked transfer encoding"
        ],
        "Response Generation": [
            "Status line construction",
            "Header formatting",
            "Body serialization",
            "Content-Type handling"
        ],
        "Concurrency": [
            "Thread pool for request handling",
            "Non-blocking I/O with epoll/kqueue",
            "Connection keep-alive",
            "Graceful shutdown"
        ],
        "Static Files": [
            "File serving from directory",
            "MIME type detection",
            "Range requests for partial content",
            "Directory listing"
        ]
    },
    "blockchain-basics": {
        "Block Structure": [
            "Block header (hash, prev_hash, timestamp, nonce)",
            "Transaction list in block",
            "Merkle root calculation",
            "Block serialization"
        ],
        "Hashing & Mining": [
            "SHA-256 hash function usage",
            "Proof of work algorithm",
            "Difficulty adjustment",
            "Nonce search loop"
        ],
        "Chain Management": [
            "Chain validation (hash links)",
            "Longest chain rule",
            "Block addition and propagation",
            "Fork resolution"
        ],
        "Transactions": [
            "Transaction structure (inputs, outputs)",
            "Digital signature verification",
            "UTXO model or account model",
            "Double-spend prevention"
        ]
    },
    "bytecode-vm": {
        "Instruction Set": [
            "Opcode design (stack-based or register)",
            "Instruction encoding format",
            "Operand types (immediate, register, address)",
            "Instruction documentation"
        ],
        "VM Execution": [
            "Fetch-decode-execute loop",
            "Program counter management",
            "Stack operations (push, pop)",
            "Arithmetic and logic operations"
        ],
        "Memory Model": [
            "Heap allocation instructions",
            "Global variable storage",
            "Local variable slots",
            "Garbage collection hooks"
        ],
        "Debugging": [
            "Instruction disassembler",
            "Execution tracing",
            "Breakpoint support",
            "Stack trace on error"
        ]
    },
    "chat-app": {
        "User Authentication": [
            "User registration with email/password",
            "Login with session or JWT",
            "Password hashing (bcrypt)",
            "Logout and session invalidation"
        ],
        "Real-time Messaging": [
            "WebSocket connection setup",
            "Message sending and receiving",
            "Message format (sender, content, timestamp)",
            "Typing indicators"
        ],
        "Chat Rooms": [
            "Room creation and joining",
            "Room member list management",
            "Message history loading",
            "Room-based message routing"
        ],
        "Message Persistence": [
            "Message storage in database",
            "Message retrieval with pagination",
            "Unread message tracking",
            "Message search"
        ]
    },
    "ci-cd-pipeline": {
        "Pipeline Definition": [
            "YAML pipeline configuration format",
            "Stage and job definitions",
            "Job dependencies",
            "Environment variables"
        ],
        "Build Execution": [
            "Git repository cloning",
            "Build command execution",
            "Build artifact collection",
            "Build log streaming"
        ],
        "Testing Integration": [
            "Test command execution",
            "Test result parsing",
            "Coverage report collection",
            "Test failure handling"
        ],
        "Deployment": [
            "Deployment target configuration",
            "SSH/cloud deployment scripts",
            "Rollback mechanism",
            "Deployment notifications"
        ]
    },
    "code-generator": {
        "IR Design": [
            "Intermediate representation format",
            "IR instructions (load, store, binop, call, etc.)",
            "Basic blocks and control flow graph",
            "SSA form (optional)"
        ],
        "Instruction Selection": [
            "IR to target instruction mapping",
            "Pattern matching for complex instructions",
            "Addressing mode selection",
            "Instruction legalization"
        ],
        "Register Allocation": [
            "Liveness analysis",
            "Interference graph construction",
            "Graph coloring algorithm",
            "Spill code generation"
        ],
        "Assembly Output": [
            "Assembly syntax generation",
            "Label and symbol management",
            "Stack frame layout",
            "Calling convention compliance"
        ]
    },
    "collaborative-editor": {
        "WebSocket Communication": [
            "WebSocket server setup",
            "Client connection management",
            "Message broadcasting to clients",
            "Connection state tracking"
        ],
        "Operational Transformation": [
            "Operation representation (insert, delete)",
            "Transform function implementation",
            "Server operation ordering",
            "Client operation buffer"
        ],
        "Document State": [
            "Document storage structure",
            "Operation application to document",
            "Version/revision tracking",
            "Document snapshot creation"
        ],
        "Cursor Awareness": [
            "Cursor position broadcasting",
            "Remote cursor display",
            "Cursor position transformation",
            "User presence indicators"
        ]
    },
    "compiler-optimization": {
        "Data Flow Analysis": [
            "Reaching definitions analysis",
            "Live variable analysis",
            "Available expressions analysis",
            "Iterative data flow framework"
        ],
        "Local Optimizations": [
            "Constant folding",
            "Algebraic simplification",
            "Dead code elimination",
            "Common subexpression elimination"
        ],
        "Loop Optimizations": [
            "Loop invariant code motion",
            "Strength reduction",
            "Loop unrolling",
            "Induction variable elimination"
        ],
        "SSA Optimizations": [
            "SSA construction (phi insertion)",
            "Sparse conditional constant propagation",
            "Global value numbering",
            "SSA destruction (phi elimination)"
        ]
    },
    "container-runtime": {
        "Container Lifecycle": [
            "Container create operation",
            "Container start/stop/delete",
            "Container state machine",
            "Container ID generation"
        ],
        "Namespace Setup": [
            "PID namespace isolation",
            "Network namespace setup",
            "Mount namespace for rootfs",
            "User namespace mapping"
        ],
        "Cgroup Management": [
            "Cgroup v1/v2 support",
            "Memory limit enforcement",
            "CPU quota/shares configuration",
            "PID limit setting"
        ],
        "Filesystem Setup": [
            "Rootfs extraction and setup",
            "pivot_root execution",
            "Bind mounts for volumes",
            "Overlay filesystem layering"
        ]
    },
    "cqrs-event-sourcing": {
        "Event Store": [
            "Event append operation",
            "Event stream per aggregate",
            "Event serialization format",
            "Optimistic concurrency control"
        ],
        "Command Handling": [
            "Command validation",
            "Aggregate loading from events",
            "Command to event translation",
            "Event persistence"
        ],
        "Event Projection": [
            "Event subscription mechanism",
            "Read model update handlers",
            "Projection rebuild capability",
            "Eventual consistency handling"
        ],
        "Query Side": [
            "Separate read database",
            "Optimized query models",
            "Query handlers",
            "Read model caching"
        ]
    },
    "database-engine": {
        "Storage Layer": [
            "Page-based storage format",
            "Buffer pool management",
            "Page read/write operations",
            "Free space management"
        ],
        "Query Parser": [
            "SQL tokenizer",
            "SQL grammar parser",
            "AST construction",
            "Semantic analysis"
        ],
        "Query Executor": [
            "Scan operators (table, index)",
            "Join operators",
            "Aggregation operators",
            "Query plan interpretation"
        ],
        "Transaction Manager": [
            "Transaction begin/commit/rollback",
            "Lock manager",
            "MVCC implementation",
            "Recovery logging"
        ]
    },
    "distributed-cache": {
        "Cache Operations": [
            "GET/SET/DELETE operations",
            "TTL-based expiration",
            "LRU eviction policy",
            "Atomic operations (CAS)"
        ],
        "Distribution": [
            "Consistent hashing for key distribution",
            "Node discovery and membership",
            "Rebalancing on node changes",
            "Virtual nodes for balance"
        ],
        "Replication": [
            "Primary-replica replication",
            "Read from replica option",
            "Failover to replica",
            "Consistency level configuration"
        ],
        "Client Library": [
            "Connection pooling",
            "Request routing to correct node",
            "Retry logic",
            "Serialization/deserialization"
        ]
    },
    "distributed-tracing": {
        "Trace Context": [
            "Trace ID and span ID generation",
            "Context propagation (headers)",
            "Parent-child span relationship",
            "Baggage for custom context"
        ],
        "Instrumentation": [
            "HTTP client/server instrumentation",
            "Database query instrumentation",
            "Manual span creation API",
            "Automatic instrumentation hooks"
        ],
        "Span Collection": [
            "Span reporting to collector",
            "Batching for efficiency",
            "Sampling strategies",
            "Error and exception recording"
        ],
        "Visualization": [
            "Trace storage backend",
            "Trace timeline visualization",
            "Service dependency graph",
            "Latency histogram"
        ]
    },
    "e-commerce-platform": {
        "Product Catalog": [
            "Product CRUD operations",
            "Category hierarchy",
            "Product variants (size, color)",
            "Product search and filtering"
        ],
        "Shopping Cart": [
            "Add/remove items",
            "Cart persistence (session or DB)",
            "Quantity updates",
            "Cart total calculation"
        ],
        "Checkout Flow": [
            "Address collection",
            "Shipping method selection",
            "Payment integration (Stripe)",
            "Order creation"
        ],
        "Order Management": [
            "Order status tracking",
            "Order history for users",
            "Inventory deduction",
            "Email notifications"
        ]
    },
    "ecs-engine": {
        "Entity Management": [
            "Entity ID generation",
            "Entity creation and destruction",
            "Deferred entity destruction",
            "Entity queries"
        ],
        "Component Storage": [
            "Archetype-based storage",
            "Component type registry",
            "Dense component arrays",
            "Sparse set for component lookup"
        ],
        "System Execution": [
            "System registration",
            "System query matching",
            "System execution ordering",
            "Parallel system execution"
        ],
        "World Management": [
            "World as entity container",
            "Resource storage (singletons)",
            "World serialization",
            "Multiple worlds support"
        ]
    },
    "file-sync": {
        "Change Detection": [
            "File modification time tracking",
            "Content hash comparison",
            "Directory scanning",
            "Ignore patterns (.gitignore style)"
        ],
        "Sync Protocol": [
            "File metadata exchange",
            "Delta sync (rsync-style)",
            "Full file transfer fallback",
            "Conflict detection"
        ],
        "Conflict Resolution": [
            "Last-write-wins strategy",
            "Manual conflict resolution",
            "Conflict file creation",
            "Merge for text files"
        ],
        "Network Layer": [
            "Peer discovery",
            "Secure connection (TLS)",
            "Chunk-based transfer",
            "Resume interrupted transfers"
        ]
    },
    "file-system": {
        "Disk Layout": [
            "Superblock structure",
            "Inode structure and allocation",
            "Data block allocation (bitmap)",
            "Directory entry format"
        ],
        "File Operations": [
            "Create file (allocate inode)",
            "Read file blocks",
            "Write file (allocate blocks)",
            "Delete file (free inode and blocks)"
        ],
        "Directory Operations": [
            "Directory as special file",
            "Add/remove directory entries",
            "Path resolution",
            "Rename operation"
        ],
        "FUSE Integration": [
            "FUSE library integration",
            "VFS operation handlers",
            "Mount/unmount",
            "Permission checking"
        ]
    },
    "fraud-detection": {
        "Feature Engineering": [
            "Transaction feature extraction",
            "Aggregated features (counts, sums over time)",
            "User behavior features",
            "Feature normalization"
        ],
        "Model Training": [
            "Imbalanced data handling (SMOTE, weighting)",
            "Model selection (Random Forest, XGBoost)",
            "Cross-validation",
            "Feature importance analysis"
        ],
        "Real-time Scoring": [
            "Model serving API",
            "Feature computation in real-time",
            "Score threshold configuration",
            "Low-latency inference"
        ],
        "Alerting": [
            "Alert generation on high scores",
            "Alert queue for review",
            "False positive feedback loop",
            "Alert aggregation"
        ]
    },
    "fuzzer": {
        "Input Generation": [
            "Random input generation",
            "Mutation strategies (bit flip, byte swap)",
            "Grammar-based generation",
            "Dictionary-based mutations"
        ],
        "Coverage Tracking": [
            "Compile-time instrumentation",
            "Edge coverage collection",
            "Coverage bitmap management",
            "New coverage detection"
        ],
        "Corpus Management": [
            "Interesting input saving",
            "Corpus minimization",
            "Seed input loading",
            "Input prioritization"
        ],
        "Crash Analysis": [
            "Crash detection",
            "Crash deduplication",
            "Stack trace collection",
            "Crash minimization"
        ]
    },
    "graphql-server": {
        "Schema & Type System": [
            "GraphQL schema with Object types",
            "Query type with field resolvers",
            "Mutation type for CRUD operations",
            "Input types for mutations"
        ],
        "Resolvers & Data Fetching": [
            "Root query resolvers",
            "Field resolvers for nested types",
            "Mutation resolvers with validation",
            "Context setup with database connection"
        ],
        "DataLoader & N+1 Prevention": [
            "DataLoader for each entity type",
            "Batch function implementation",
            "Per-request caching",
            "DataLoader in resolver context"
        ],
        "Subscriptions": [
            "WebSocket transport setup",
            "Subscription resolver with async iterator",
            "Pub/sub integration",
            "Subscription filtering"
        ]
    },
    "hash-table": {
        "Hash Function": [
            "Hash function implementation (FNV, MurmurHash)",
            "Hash to bucket index conversion",
            "Hash quality analysis",
            "Seed handling for randomization"
        ],
        "Collision Resolution": [
            "Separate chaining implementation",
            "Open addressing alternative",
            "Linear/quadratic probing",
            "Robin Hood hashing (optional)"
        ],
        "Dynamic Resizing": [
            "Load factor threshold",
            "Resize trigger on insert",
            "Rehashing all entries",
            "Shrinking on delete"
        ],
        "Operations": [
            "Insert operation",
            "Lookup operation",
            "Delete operation",
            "Iteration over entries"
        ]
    },
    "http-server": {
        "Socket Handling": [
            "TCP socket creation and binding",
            "Accept incoming connections",
            "Non-blocking I/O setup",
            "Connection timeout handling"
        ],
        "Request Parsing": [
            "HTTP request line parsing",
            "Header parsing into map",
            "Body reading (Content-Length)",
            "Query string parsing"
        ],
        "Response Building": [
            "Status line formatting",
            "Header serialization",
            "Body content handling",
            "Content-Type detection"
        ],
        "Routing": [
            "Path-based routing",
            "Method-based dispatch",
            "404 handling",
            "Static file serving"
        ]
    },
    "image-processor": {
        "Image Loading": [
            "Image format decoding (PNG, JPEG)",
            "Pixel buffer representation",
            "Color space handling (RGB, RGBA)",
            "Image metadata extraction"
        ],
        "Basic Transformations": [
            "Resize with interpolation",
            "Crop operation",
            "Rotate (90, 180, 270 degrees)",
            "Flip horizontal/vertical"
        ],
        "Filters": [
            "Brightness/contrast adjustment",
            "Grayscale conversion",
            "Blur filter (box, Gaussian)",
            "Sharpen filter"
        ],
        "Output": [
            "Image encoding (PNG, JPEG)",
            "Quality settings for JPEG",
            "Batch processing",
            "CLI interface"
        ]
    },
    "json-parser": {
        "Lexer": [
            "Token types (string, number, bool, null, delimiters)",
            "String parsing with escape sequences",
            "Number parsing (int, float, exponent)",
            "Whitespace skipping"
        ],
        "Parser": [
            "Recursive descent parser",
            "Object parsing (key-value pairs)",
            "Array parsing",
            "Nested structure handling"
        ],
        "Value Representation": [
            "JSON value type hierarchy",
            "Object as map/dict",
            "Array as list",
            "Primitive value storage"
        ],
        "Serialization": [
            "JSON value to string conversion",
            "Pretty print with indentation",
            "Escape string values properly",
            "Number formatting"
        ]
    },
    "key-value-store": {
        "Storage Engine": [
            "In-memory hash map storage",
            "File-based persistence option",
            "Log-structured storage (append-only)",
            "Compaction for log storage"
        ],
        "Operations": [
            "GET operation",
            "SET operation with optional TTL",
            "DELETE operation",
            "EXISTS check"
        ],
        "Network Protocol": [
            "Custom binary or text protocol",
            "Command parsing",
            "Response formatting",
            "Error responses"
        ],
        "Concurrency": [
            "Thread-safe operations",
            "Lock granularity (global vs per-key)",
            "Atomic compare-and-swap",
            "Multi-threaded server"
        ]
    },
    "kubernetes-operator": {
        "Custom Resource Definition": [
            "CRD YAML specification",
            "Schema validation",
            "CRD versioning",
            "Status subresource"
        ],
        "Controller Logic": [
            "Reconcile loop implementation",
            "Watch for CR changes",
            "Desired vs actual state diff",
            "Status update on CR"
        ],
        "Resource Management": [
            "Create child resources (Deployment, Service)",
            "Owner references for cleanup",
            "Resource update handling",
            "Resource deletion handling"
        ],
        "Testing & Deployment": [
            "Unit tests for reconciler",
            "Integration tests with envtest",
            "Operator deployment manifest",
            "RBAC permissions"
        ]
    },
    "lexer": {
        "Character Stream": [
            "Source code input handling",
            "Peek and advance operations",
            "Position tracking (line, column)",
            "End-of-file handling"
        ],
        "Token Recognition": [
            "Keyword vs identifier distinction",
            "Operator tokenization",
            "Literal recognition (strings, numbers)",
            "Comment skipping"
        ],
        "Token Output": [
            "Token type enumeration",
            "Token value/lexeme storage",
            "Token position information",
            "Token stream interface"
        ],
        "Error Handling": [
            "Invalid character handling",
            "Unterminated string detection",
            "Error recovery (skip to known token)",
            "Error message with location"
        ]
    },
    "llm-chatbot": {
        "API Integration": [
            "OpenAI/Anthropic API client",
            "Request formatting (messages array)",
            "Response parsing",
            "Error handling and retries"
        ],
        "Conversation Management": [
            "Message history storage",
            "Context window management",
            "System prompt configuration",
            "Conversation persistence"
        ],
        "Prompt Engineering": [
            "System prompt design",
            "Few-shot examples",
            "Output format instructions",
            "Temperature and parameter tuning"
        ],
        "User Interface": [
            "CLI chat interface",
            "Web chat interface",
            "Streaming response display",
            "Markdown rendering"
        ]
    },
    "load-balancer": {
        "Backend Pool": [
            "Backend server registration",
            "Backend metadata (weight, health)",
            "Dynamic backend addition/removal",
            "Backend configuration"
        ],
        "Load Balancing": [
            "Round-robin algorithm",
            "Least connections algorithm",
            "Weighted distribution",
            "Sticky sessions (IP hash)"
        ],
        "Health Checks": [
            "Periodic health check execution",
            "TCP or HTTP health check",
            "Unhealthy threshold",
            "Recovery detection"
        ],
        "Request Proxying": [
            "HTTP request forwarding",
            "Header modification (X-Forwarded-For)",
            "Connection pooling",
            "Timeout configuration"
        ]
    },
    "lock-free-queue": {
        "Atomic Operations": [
            "Compare-and-swap (CAS) primitives",
            "Memory ordering (acquire/release)",
            "Atomic load and store",
            "ABA problem understanding"
        ],
        "Queue Structure": [
            "Node-based linked queue",
            "Head and tail pointers",
            "Dummy/sentinel node",
            "Node memory layout"
        ],
        "Enqueue": [
            "Atomic tail update",
            "Link new node to queue",
            "Handle concurrent enqueues",
            "Memory ordering for visibility"
        ],
        "Dequeue": [
            "Atomic head update",
            "Empty queue detection",
            "Safe node reclamation",
            "Hazard pointers or epoch-based reclamation"
        ]
    },
    "log-aggregator": {
        "Log Collection": [
            "File tail with inotify/kqueue",
            "Syslog protocol receiver",
            "HTTP log ingestion endpoint",
            "Agent for remote collection"
        ],
        "Log Parsing": [
            "Regex-based log parsing",
            "JSON log parsing",
            "Timestamp extraction and normalization",
            "Field extraction"
        ],
        "Storage": [
            "Time-based log partitioning",
            "Compression for storage efficiency",
            "Retention policy enforcement",
            "Index for fast queries"
        ],
        "Query Interface": [
            "Full-text search",
            "Time range filtering",
            "Field-based filtering",
            "Log streaming (tail -f style)"
        ]
    },
    "merkle-tree": {
        "Tree Construction": [
            "Leaf node creation from data blocks",
            "Internal node hash computation",
            "Binary tree building (bottom-up)",
            "Odd number of nodes handling"
        ],
        "Root Hash": [
            "Root hash calculation",
            "Merkle root as tree fingerprint",
            "Root update on data change",
            "Efficient root recomputation"
        ],
        "Membership Proof": [
            "Proof generation (sibling hashes)",
            "Proof verification algorithm",
            "Proof size (O(log n))",
            "Proof serialization"
        ],
        "Applications": [
            "Data integrity verification",
            "Efficient sync (compare roots)",
            "Audit proof for inclusion",
            "Difference detection between trees"
        ]
    },
    "message-queue": {
        "Queue Operations": [
            "Enqueue message",
            "Dequeue message",
            "Peek without consuming",
            "Queue length query"
        ],
        "Persistence": [
            "Message serialization to disk",
            "Write-ahead log for durability",
            "Message acknowledgment",
            "Unacknowledged message redelivery"
        ],
        "Consumer Groups": [
            "Consumer registration",
            "Partition/queue assignment",
            "Consumer heartbeat",
            "Rebalancing on consumer change"
        ],
        "Networking": [
            "TCP server for clients",
            "Protocol for publish/subscribe",
            "Connection management",
            "Multi-tenant queue isolation"
        ]
    },
    "metrics-dashboard": {
        "Metric Collection": [
            "Metric ingestion API",
            "Metric types (counter, gauge, histogram)",
            "Label/tag support",
            "Prometheus scraping endpoint"
        ],
        "Storage": [
            "Time-series storage backend",
            "Downsampling for historical data",
            "Retention policies",
            "Efficient range queries"
        ],
        "Query Language": [
            "PromQL-style query language",
            "Aggregation functions",
            "Rate and increase calculations",
            "Label filtering"
        ],
        "Visualization": [
            "Dashboard configuration",
            "Graph/chart rendering",
            "Real-time updates",
            "Alert visualization"
        ]
    },
    "mini-redis": {
        "Data Types": [
            "String type implementation",
            "List type with push/pop",
            "Set type with add/remove/members",
            "Hash type with field operations"
        ],
        "Command Processing": [
            "Command parser",
            "Command dispatch to handlers",
            "RESP protocol encoding/decoding",
            "Pipeline support"
        ],
        "Expiration": [
            "TTL setting on keys",
            "Expiration checking on access",
            "Background expiration scanning",
            "PTTL/TTL commands"
        ],
        "Pub/Sub": [
            "Channel subscription",
            "Message publishing",
            "Pattern subscriptions",
            "Subscriber management"
        ]
    },
    "mini-shell": {
        "Command Execution": [
            "fork() and exec() for commands",
            "PATH searching",
            "Command not found error",
            "Exit status capture"
        ],
        "Pipes and Redirection": [
            "Pipe between commands",
            "stdin redirection (<)",
            "stdout redirection (>)",
            "Append redirection (>>)"
        ],
        "Signal Handling": [
            "SIGINT handling (Ctrl+C)",
            "SIGTSTP handling (Ctrl+Z)",
            "Child process signal forwarding",
            "Signal restoration after fork"
        ],
        "Builtins": [
            "cd command",
            "exit command",
            "export command",
            "pwd command"
        ]
    },
    "ml-model-serving": {
        "Model Loading": [
            "Model format loading (ONNX, TensorFlow, PyTorch)",
            "Model versioning",
            "Model warmup on load",
            "Memory management for models"
        ],
        "Inference API": [
            "REST API for predictions",
            "Input validation",
            "Batch prediction support",
            "Async inference option"
        ],
        "Performance": [
            "Request batching for throughput",
            "Model caching",
            "GPU inference support",
            "Latency monitoring"
        ],
        "A/B Testing": [
            "Traffic splitting configuration",
            "Model variant routing",
            "Metrics collection per variant",
            "Gradual rollout support"
        ]
    },
    "neural-network": {
        "Network Architecture": [
            "Layer abstraction (Dense, Conv, etc.)",
            "Sequential model building",
            "Parameter initialization",
            "Forward pass implementation"
        ],
        "Backpropagation": [
            "Gradient computation",
            "Chain rule application",
            "Gradient caching",
            "Backward pass implementation"
        ],
        "Optimizers": [
            "Gradient descent implementation",
            "Learning rate handling",
            "Momentum implementation",
            "Adam optimizer"
        ],
        "Training Loop": [
            "Batch iteration",
            "Loss computation",
            "Parameter updates",
            "Validation evaluation"
        ]
    },
    "oauth-provider": {
        "Client Registration": [
            "Client ID and secret generation",
            "Redirect URI registration",
            "Grant type configuration",
            "Client authentication"
        ],
        "Authorization Flow": [
            "Authorization endpoint",
            "User consent screen",
            "Authorization code generation",
            "Code to token exchange"
        ],
        "Token Management": [
            "Access token generation (JWT)",
            "Refresh token handling",
            "Token expiration",
            "Token revocation"
        ],
        "Resource Protection": [
            "Token validation endpoint",
            "Scope enforcement",
            "Token introspection",
            "Resource server integration"
        ]
    },
    "packet-analyzer": {
        "Packet Capture": [
            "Raw socket or libpcap integration",
            "Promiscuous mode setup",
            "BPF filter support",
            "Packet buffering"
        ],
        "Protocol Parsing": [
            "Ethernet frame parsing",
            "IP header parsing",
            "TCP/UDP header parsing",
            "Application layer detection"
        ],
        "Analysis": [
            "Connection tracking",
            "Traffic statistics",
            "Protocol distribution",
            "Anomaly detection"
        ],
        "Output": [
            "Packet summary display",
            "Hex dump view",
            "PCAP file export",
            "Real-time filtering"
        ]
    },
    "password-manager": {
        "Encryption": [
            "Master password derivation (PBKDF2/Argon2)",
            "Vault encryption (AES-256)",
            "Individual entry encryption",
            "Key storage security"
        ],
        "Vault Operations": [
            "Create new vault",
            "Unlock vault with master password",
            "Lock vault (clear keys)",
            "Vault backup/export"
        ],
        "Entry Management": [
            "Add password entry",
            "Edit entry",
            "Delete entry",
            "Search entries"
        ],
        "Password Generation": [
            "Random password generation",
            "Character set configuration",
            "Length configuration",
            "Passphrase generation"
        ]
    },
    "physics-engine": {
        "Rigid Body Dynamics": [
            "Position and velocity integration",
            "Force and torque application",
            "Mass and inertia tensor",
            "Angular velocity and rotation"
        ],
        "Collision Detection": [
            "Broad phase (spatial hash, BVH)",
            "Narrow phase (SAT, GJK)",
            "Contact point generation",
            "Collision layers/masks"
        ],
        "Collision Response": [
            "Impulse-based resolution",
            "Restitution (bounciness)",
            "Friction handling",
            "Position correction (penetration)"
        ],
        "Constraints": [
            "Distance constraint",
            "Hinge joint",
            "Fixed joint",
            "Constraint solver (sequential impulses)"
        ]
    },
    "plugin-system": {
        "Plugin Interface": [
            "Plugin interface/trait definition",
            "Plugin lifecycle hooks (init, shutdown)",
            "Plugin metadata (name, version)",
            "Dependency declaration"
        ],
        "Plugin Loading": [
            "Dynamic library loading",
            "Plugin discovery (scan directories)",
            "Version compatibility check",
            "Safe loading with error handling"
        ],
        "Plugin Communication": [
            "Event/message passing",
            "Service registration",
            "Shared context/state",
            "Inter-plugin dependencies"
        ],
        "Plugin Management": [
            "Enable/disable plugins",
            "Hot reload capability",
            "Plugin configuration",
            "Plugin isolation"
        ]
    },
    "protocol-buffers": {
        "Schema Parsing": [
            ".proto file lexer",
            "Message type parsing",
            "Field definition parsing",
            "Import handling"
        ],
        "Code Generation": [
            "Language-specific code output",
            "Message class generation",
            "Getter/setter generation",
            "Enum type generation"
        ],
        "Serialization": [
            "Wire format encoding",
            "Varint encoding",
            "Length-delimited fields",
            "Field number and type tags"
        ],
        "Deserialization": [
            "Wire format decoding",
            "Unknown field handling",
            "Type checking",
            "Default values"
        ]
    },
    "query-optimizer": {
        "Query Parsing": [
            "SQL query to AST",
            "Table and column resolution",
            "Type checking",
            "Query normalization"
        ],
        "Plan Enumeration": [
            "Join order enumeration",
            "Access path selection",
            "Plan space representation",
            "Physical operator choices"
        ],
        "Cost Estimation": [
            "Statistics collection (row counts, histograms)",
            "Selectivity estimation",
            "I/O and CPU cost model",
            "Join cost estimation"
        ],
        "Plan Selection": [
            "Dynamic programming for join ordering",
            "Heuristic pruning",
            "Best plan selection",
            "Plan caching"
        ]
    },
    "rate-limiter": {
        "Token Bucket": [
            "Bucket capacity configuration",
            "Token refill rate",
            "Token consumption on request",
            "Burst handling"
        ],
        "Sliding Window": [
            "Window size configuration",
            "Request count tracking",
            "Window sliding logic",
            "Timestamp-based counting"
        ],
        "Distributed Rate Limiting": [
            "Redis-based counter storage",
            "Atomic increment operations",
            "Sliding window in Redis",
            "Cluster-wide limits"
        ],
        "API Integration": [
            "Middleware implementation",
            "Rate limit headers",
            "429 response handling",
            "Client identification (IP, API key)"
        ]
    },
    "recommendation-engine": {
        "Data Collection": [
            "User interaction logging",
            "Item metadata storage",
            "Implicit feedback (views, clicks)",
            "Explicit feedback (ratings)"
        ],
        "Collaborative Filtering": [
            "User-item interaction matrix",
            "User similarity computation",
            "Item similarity computation",
            "Rating prediction"
        ],
        "Content-Based": [
            "Item feature extraction",
            "User profile from interactions",
            "Content similarity scoring",
            "Hybrid approach"
        ],
        "Serving": [
            "Recommendation API",
            "Real-time personalization",
            "Candidate retrieval",
            "Result ranking and filtering"
        ]
    },
    "rest-api": {
        "Resource Design": [
            "RESTful URL structure",
            "HTTP method semantics",
            "Resource representation (JSON)",
            "Pagination design"
        ],
        "CRUD Operations": [
            "Create resource (POST)",
            "Read resource (GET)",
            "Update resource (PUT/PATCH)",
            "Delete resource (DELETE)"
        ],
        "Validation": [
            "Request body validation",
            "Path parameter validation",
            "Query parameter validation",
            "Error response format"
        ],
        "Documentation": [
            "OpenAPI/Swagger spec",
            "API documentation generation",
            "Example requests/responses",
            "Authentication documentation"
        ]
    },
    "rsa-impl": {
        "Key Generation": [
            "Prime number generation",
            "Primality testing (Miller-Rabin)",
            "Modular arithmetic utilities",
            "Public/private key pair computation"
        ],
        "Encryption": [
            "Message to number conversion",
            "Modular exponentiation",
            "Padding scheme (PKCS#1 or OAEP)",
            "Ciphertext generation"
        ],
        "Decryption": [
            "Ciphertext to number conversion",
            "Private key exponentiation",
            "Padding removal",
            "Plaintext recovery"
        ],
        "Signatures": [
            "Hash of message",
            "Signature generation with private key",
            "Signature verification with public key",
            "Signature format"
        ]
    },
    "service-mesh": {
        "Sidecar Proxy": [
            "Traffic interception (iptables)",
            "HTTP/gRPC proxying",
            "Connection pooling",
            "Protocol detection"
        ],
        "Service Discovery": [
            "Service registry integration",
            "Endpoint health tracking",
            "DNS-based discovery",
            "Load balancing policies"
        ],
        "Traffic Management": [
            "Request routing rules",
            "Traffic splitting (canary)",
            "Retry and timeout policies",
            "Circuit breaker"
        ],
        "Observability": [
            "Automatic request tracing",
            "Metrics collection",
            "Access logging",
            "Distributed tracing headers"
        ]
    },
    "skiplist": {
        "Node Structure": [
            "Multi-level forward pointers",
            "Key-value storage in node",
            "Level height in node",
            "Sentinel head node"
        ],
        "Level Generation": [
            "Randomized level selection",
            "Geometric distribution",
            "Maximum level cap",
            "Level probability parameter"
        ],
        "Search": [
            "Top-down level traversal",
            "Forward pointer following",
            "O(log n) expected time",
            "Update array for inserts"
        ],
        "Insert/Delete": [
            "Find insertion point",
            "Update forward pointers",
            "Handle level increases",
            "Memory deallocation on delete"
        ]
    },
    "sse-server": {
        "Connection Management": [
            "HTTP keep-alive connection",
            "Content-Type: text/event-stream",
            "Connection state tracking",
            "Client disconnection detection"
        ],
        "Event Formatting": [
            "SSE message format (data:, event:, id:)",
            "Multi-line data handling",
            "Event type specification",
            "Comment lines for keep-alive"
        ],
        "Event Broadcasting": [
            "Channel subscription model",
            "Broadcast to all subscribers",
            "Targeted event delivery",
            "Event queue per connection"
        ],
        "Reconnection": [
            "Last-Event-Id header handling",
            "Event replay on reconnect",
            "Retry interval specification",
            "State recovery"
        ]
    },
    "static-analyzer": {
        "AST Traversal": [
            "Visitor pattern for AST",
            "Scope tracking during traversal",
            "Symbol table construction",
            "Type inference"
        ],
        "Rule Engine": [
            "Rule interface definition",
            "Rule registration",
            "Rule matching against AST nodes",
            "Configurable rule severity"
        ],
        "Bug Detection": [
            "Null pointer dereference",
            "Unused variable detection",
            "Unreachable code detection",
            "Resource leak detection"
        ],
        "Reporting": [
            "Issue location (file, line, column)",
            "Issue description and fix suggestion",
            "Output formats (text, JSON, SARIF)",
            "CI integration"
        ]
    },
    "tcp-impl": {
        "Connection Setup": [
            "SYN packet construction and sending",
            "SYN-ACK handling",
            "ACK completion of handshake",
            "Initial sequence number generation"
        ],
        "Data Transfer": [
            "Segment construction with sequence numbers",
            "Acknowledgment tracking",
            "Receive buffer management",
            "Send buffer management"
        ],
        "Flow Control": [
            "Sliding window implementation",
            "Window size advertisement",
            "Window updates",
            "Zero window handling"
        ],
        "Reliability": [
            "Retransmission timeout (RTO) calculation",
            "Fast retransmit on duplicate ACKs",
            "Out-of-order segment handling",
            "Connection teardown (FIN handshake)"
        ]
    },
    "thread-pool": {
        "Worker Management": [
            "Worker thread spawning",
            "Thread count configuration",
            "Worker loop implementation",
            "Graceful shutdown"
        ],
        "Task Queue": [
            "Thread-safe task queue",
            "Blocking wait for tasks",
            "Task submission API",
            "Queue size limits"
        ],
        "Scheduling": [
            "FIFO task scheduling",
            "Priority queue option",
            "Work stealing (optional)",
            "Task fairness"
        ],
        "Result Handling": [
            "Future/Promise for results",
            "Result retrieval blocking",
            "Exception propagation",
            "Timeout support"
        ]
    },
    "tls-impl": {
        "Handshake": [
            "ClientHello message",
            "ServerHello response",
            "Certificate exchange",
            "Key exchange (ECDHE)"
        ],
        "Key Derivation": [
            "Pre-master secret computation",
            "Master secret derivation",
            "Key expansion for encryption",
            "IV generation"
        ],
        "Record Protocol": [
            "Record layer framing",
            "Encryption with AES-GCM",
            "MAC computation",
            "Sequence number handling"
        ],
        "Certificate Validation": [
            "Certificate chain parsing",
            "Signature verification",
            "Expiration checking",
            "Common name/SAN validation"
        ]
    },
    "type-checker": {
        "Type Representation": [
            "Type AST (primitives, functions, generics)",
            "Type environment (symbol to type)",
            "Type equality and compatibility",
            "Subtyping rules"
        ],
        "Type Inference": [
            "Constraint generation",
            "Unification algorithm",
            "Type variable substitution",
            "Let-polymorphism (generalization)"
        ],
        "Type Checking": [
            "Expression type checking",
            "Function call type checking",
            "Assignment compatibility",
            "Return type checking"
        ],
        "Error Reporting": [
            "Type mismatch errors",
            "Undefined variable errors",
            "Error location tracking",
            "Helpful error messages"
        ]
    },
    "url-shortener": {
        "URL Storage": [
            "Database schema for URLs",
            "Original URL storage",
            "Short code storage",
            "Creation timestamp"
        ],
        "Short Code Generation": [
            "Base62 encoding",
            "Auto-increment ID approach",
            "Hash-based approach",
            "Collision handling"
        ],
        "Redirect Handling": [
            "Short code lookup",
            "HTTP 301/302 redirect",
            "Invalid code handling",
            "Expiration checking"
        ],
        "Analytics": [
            "Click counting",
            "Referrer tracking",
            "Geographic data (optional)",
            "Analytics dashboard"
        ]
    },
    "video-streaming": {
        "Video Processing": [
            "FFmpeg integration",
            "Transcoding to multiple qualities",
            "HLS segment generation",
            "Thumbnail extraction"
        ],
        "Manifest Generation": [
            "HLS master playlist",
            "Quality variant playlists",
            "Segment duration configuration",
            "Live vs VOD manifest"
        ],
        "Streaming Server": [
            "Segment file serving",
            "Playlist serving",
            "Byte-range requests",
            "CDN integration points"
        ],
        "Player Integration": [
            "HLS.js or Video.js setup",
            "Quality selection UI",
            "Buffering indication",
            "Error handling"
        ]
    },
    "web-crawler": {
        "URL Management": [
            "URL frontier queue",
            "Visited URL tracking",
            "URL normalization",
            "Domain-based politeness"
        ],
        "HTTP Fetching": [
            "HTTP client with timeouts",
            "Redirect following",
            "robots.txt fetching and parsing",
            "User-agent setting"
        ],
        "Content Parsing": [
            "HTML parsing",
            "Link extraction",
            "Content extraction",
            "Encoding detection"
        ],
        "Concurrency": [
            "Multiple concurrent fetchers",
            "Per-domain rate limiting",
            "Connection pooling",
            "Graceful shutdown"
        ]
    },
    "webhook-delivery": {
        "Event Ingestion": [
            "Event API endpoint",
            "Event validation",
            "Event queue storage",
            "Event deduplication"
        ],
        "Delivery Engine": [
            "HTTP POST to subscriber URL",
            "Signature generation (HMAC)",
            "Timeout handling",
            "Response validation"
        ],
        "Retry Logic": [
            "Exponential backoff",
            "Maximum retry attempts",
            "Retry queue",
            "Dead letter queue"
        ],
        "Monitoring": [
            "Delivery status tracking",
            "Success/failure metrics",
            "Latency tracking",
            "Alerting on failures"
        ]
    },
    "websocket-server": {
        "Handshake": [
            "HTTP upgrade request handling",
            "Sec-WebSocket-Key validation",
            "Sec-WebSocket-Accept computation",
            "Protocol negotiation"
        ],
        "Frame Handling": [
            "WebSocket frame parsing",
            "Mask/unmask payload",
            "Frame types (text, binary, ping, pong, close)",
            "Fragmented message handling"
        ],
        "Connection Management": [
            "Connection state tracking",
            "Ping/pong for keep-alive",
            "Close handshake",
            "Connection timeout"
        ],
        "Broadcasting": [
            "Connection registry",
            "Room/channel abstraction",
            "Broadcast to connections",
            "Targeted message delivery"
        ]
    },
    "workflow-engine": {
        "Workflow Definition": [
            "Workflow DSL or YAML format",
            "Step/task definition",
            "Conditional branching",
            "Parallel execution"
        ],
        "Execution Engine": [
            "Workflow instance management",
            "Step execution and state tracking",
            "Input/output passing between steps",
            "Error handling and compensation"
        ],
        "Persistence": [
            "Workflow state persistence",
            "Step execution history",
            "Resume from checkpoint",
            "Execution replay"
        ],
        "Monitoring": [
            "Workflow status dashboard",
            "Step execution timeline",
            "Error and retry visibility",
            "SLA tracking"
        ]
    },
    "alerting-system": {
        "Alert Rule Configuration": [
            "Rule definition (metric, threshold, duration)",
            "Alert severity levels",
            "Rule grouping and silencing",
            "Dynamic threshold support"
        ],
        "Alert Evaluation": [
            "Periodic rule evaluation",
            "Metric query execution",
            "Threshold comparison",
            "Duration-based triggering"
        ],
        "Notification Routing": [
            "Notification channel configuration",
            "Routing rules by severity/team",
            "Escalation policies",
            "On-call schedule integration"
        ],
        "Alert Lifecycle": [
            "Alert state machine (pending, firing, resolved)",
            "Alert deduplication",
            "Alert grouping",
            "Resolution detection"
        ]
    },
    "apm-system": {
        "Agent Instrumentation": [
            "Language-specific agent SDK",
            "Automatic framework instrumentation",
            "Custom span creation API",
            "Context propagation"
        ],
        "Trace Collection": [
            "Trace data ingestion API",
            "Trace sampling",
            "Trace storage backend",
            "Trace indexing"
        ],
        "Metrics Collection": [
            "Application metrics (latency, errors, throughput)",
            "Host metrics (CPU, memory)",
            "Custom metrics API",
            "Metric aggregation"
        ],
        "Visualization": [
            "Service map generation",
            "Trace waterfall view",
            "Performance dashboards",
            "Error tracking"
        ]
    },
    "audit-logging": {
        "Event Capture": [
            "Event interceptor/middleware",
            "Event schema definition",
            "Actor identification",
            "Resource identification"
        ],
        "Event Storage": [
            "Append-only storage",
            "Tamper-evident logging",
            "Event compression",
            "Partitioning by time/tenant"
        ],
        "Query Interface": [
            "Event search by actor/resource/time",
            "Full-text search in event data",
            "Aggregation queries",
            "Export to external systems"
        ],
        "Compliance": [
            "Retention policy enforcement",
            "Integrity verification",
            "Access control for audit logs",
            "Compliance reporting"
        ]
    },
    "cdc-system": {
        "Log Reading": [
            "Database log connection (binlog, WAL)",
            "Log position tracking",
            "Log event parsing",
            "Checkpoint management"
        ],
        "Event Conversion": [
            "Row change to event conversion",
            "Schema extraction",
            "Before/after image capture",
            "Event serialization"
        ],
        "Event Publishing": [
            "Kafka/message queue integration",
            "Topic routing by table",
            "At-least-once delivery",
            "Ordering guarantees"
        ],
        "Schema Management": [
            "Schema registry integration",
            "Schema evolution handling",
            "DDL event handling",
            "Compatibility checking"
        ]
    },
    "config-management": {
        "Configuration Storage": [
            "Hierarchical config storage",
            "Version control for configs",
            "Environment-specific overrides",
            "Secret management integration"
        ],
        "Client SDK": [
            "Configuration fetching",
            "Local caching",
            "Real-time updates (polling or push)",
            "Fallback to cached values"
        ],
        "Dynamic Updates": [
            "Change notification",
            "Hot reload support",
            "Gradual rollout",
            "Rollback capability"
        ],
        "Access Control": [
            "Authentication for config access",
            "RBAC for config modifications",
            "Audit logging for changes",
            "Approval workflows"
        ]
    },
    "feature-flags": {
        "Flag Definition": [
            "Flag types (boolean, string, number)",
            "Default values",
            "Flag metadata (description, owner)",
            "Flag grouping"
        ],
        "Targeting": [
            "User segment targeting",
            "Percentage rollout",
            "User attribute matching",
            "Environment targeting"
        ],
        "Evaluation": [
            "Flag evaluation SDK",
            "Local evaluation cache",
            "Server-side evaluation API",
            "Evaluation logging"
        ],
        "Management": [
            "Flag lifecycle (created, active, deprecated)",
            "Kill switch capability",
            "A/B test integration",
            "Flag usage tracking"
        ]
    },
    "gitops-deployment": {
        "Repository Sync": [
            "Git repository watching",
            "Manifest detection and parsing",
            "Desired state extraction",
            "Sync interval configuration"
        ],
        "Deployment Engine": [
            "Kubernetes API integration",
            "Resource creation/update/delete",
            "Health checking",
            "Sync status reporting"
        ],
        "Drift Detection": [
            "Actual vs desired state comparison",
            "Drift notification",
            "Auto-sync option",
            "Manual sync approval"
        ],
        "Rollback": [
            "Deployment history tracking",
            "Rollback to previous revision",
            "Rollback triggers (health check failure)",
            "Rollback notifications"
        ]
    },
    "llm-finetuning-pipeline": {
        "Dataset Preparation": [
            "Data loading and validation",
            "Prompt template formatting",
            "Train/validation split",
            "Data augmentation"
        ],
        "Training Configuration": [
            "LoRA/QLoRA parameter setup",
            "Hyperparameter configuration",
            "Gradient checkpointing",
            "Mixed precision training"
        ],
        "Training Loop": [
            "Training step implementation",
            "Loss computation",
            "Checkpoint saving",
            "Validation evaluation"
        ],
        "Model Export": [
            "Adapter weight saving",
            "Model merging (optional)",
            "Quantization for deployment",
            "Model evaluation metrics"
        ]
    },
    "mlops-platform": {
        "Experiment Tracking": [
            "Experiment metadata logging",
            "Metric and parameter logging",
            "Artifact storage",
            "Experiment comparison"
        ],
        "Model Registry": [
            "Model versioning",
            "Model metadata and lineage",
            "Model stage transitions",
            "Model approval workflow"
        ],
        "Training Pipelines": [
            "Pipeline definition (DAG)",
            "Pipeline execution engine",
            "Caching for pipeline steps",
            "Pipeline versioning"
        ],
        "Model Deployment": [
            "Deployment target configuration",
            "Canary deployment support",
            "Model monitoring",
            "Automated retraining triggers"
        ]
    },
    "multi-tenant-saas": {
        "Tenant Isolation": [
            "Tenant identification middleware",
            "Database schema isolation (shared vs separate)",
            "Row-level security policies",
            "Cross-tenant query prevention"
        ],
        "Tenant Management": [
            "Tenant onboarding API",
            "Tenant configuration storage",
            "Tenant-specific feature flags",
            "Tenant deletion/archival"
        ],
        "Authentication": [
            "Tenant-aware authentication",
            "SSO integration per tenant",
            "User-tenant association",
            "Role management per tenant"
        ],
        "Billing Integration": [
            "Usage metering per tenant",
            "Subscription plan management",
            "Billing API integration (Stripe)",
            "Invoice generation"
        ]
    },
    "background-job-processor": {
        "Job Queue": [
            "Job serialization and enqueue",
            "Queue backend (Redis, DB)",
            "Priority queues",
            "Scheduled job support"
        ],
        "Worker Execution": [
            "Job dequeue and execution",
            "Worker concurrency configuration",
            "Job timeout handling",
            "Process isolation"
        ],
        "Retry & Error Handling": [
            "Automatic retry on failure",
            "Exponential backoff",
            "Dead letter queue",
            "Error notification"
        ],
        "Monitoring": [
            "Job status tracking",
            "Queue depth metrics",
            "Worker health monitoring",
            "Job execution history"
        ]
    },
    "build-graphql-engine": {
        "Schema Parsing": [
            "GraphQL SDL parser",
            "Type definition extraction",
            "Directive handling",
            "Schema validation"
        ],
        "Query Execution": [
            "Query parsing and validation",
            "Execution plan generation",
            "Resolver execution",
            "Error handling"
        ],
        "Database Integration": [
            "Schema introspection from database",
            "SQL query generation from GraphQL",
            "Join optimization",
            "N+1 prevention"
        ],
        "Schema Stitching": [
            "Remote schema fetching",
            "Schema merging",
            "Type conflict resolution",
            "Cross-schema relationships"
        ]
    },
    "serverless-runtime": {
        "Function Packaging": [
            "Function code packaging",
            "Dependency bundling",
            "Container image building",
            "Function versioning"
        ],
        "Execution Environment": [
            "Function sandbox setup",
            "Resource limits (memory, CPU, time)",
            "Environment variable injection",
            "Temporary storage"
        ],
        "Invocation Handling": [
            "HTTP trigger handling",
            "Event trigger handling",
            "Request routing to function",
            "Response formatting"
        ],
        "Scaling": [
            "Scale to zero when idle",
            "Cold start handling",
            "Warm instance pooling",
            "Concurrent execution limits"
        ]
    },
    "data-quality-framework": {
        "Schema Validation": [
            "Schema definition language",
            "Type validation (null, type, range)",
            "Custom validation rules",
            "Schema versioning"
        ],
        "Data Profiling": [
            "Column statistics (min, max, avg, distinct)",
            "Null percentage tracking",
            "Value distribution histograms",
            "Data freshness monitoring"
        ],
        "Anomaly Detection": [
            "Statistical anomaly detection",
            "Trend deviation alerts",
            "Volume anomaly detection",
            "Schema drift detection"
        ],
        "Reporting": [
            "Data quality scores",
            "Quality trend dashboards",
            "Alert notifications",
            "Quality reports generation"
        ]
    },
    "workflow-orchestrator": {
        "DAG Definition": [
            "Task dependency specification",
            "DAG parsing and validation",
            "Task parameters and configuration",
            "Schedule configuration"
        ],
        "Scheduler": [
            "DAG run scheduling",
            "Task readiness detection",
            "Parallel task execution",
            "Backfill support"
        ],
        "Task Execution": [
            "Task execution in worker",
            "Task retry on failure",
            "Task timeout handling",
            "Task logging and output capture"
        ],
        "Monitoring": [
            "DAG run status tracking",
            "Task execution timeline",
            "Failure alerting",
            "SLA monitoring"
        ]
    },
    "vector-database": {
        "Vector Indexing": [
            "HNSW index implementation",
            "IVF index option",
            "Index build pipeline",
            "Index persistence"
        ],
        "Similarity Search": [
            "K-nearest neighbor query",
            "Distance metric support (L2, cosine, IP)",
            "Approximate vs exact mode",
            "Filter-based search"
        ],
        "Metadata Management": [
            "Vector-metadata association",
            "Metadata indexing",
            "Combined vector+metadata queries",
            "Metadata updates"
        ],
        "API": [
            "Insert/update/delete vectors",
            "Batch operations",
            "Search API with filters",
            "Index management API"
        ]
    },
    "stream-processing-engine": {
        "Event Ingestion": [
            "Kafka consumer integration",
            "Event deserialization",
            "Source partitioning",
            "Watermark generation"
        ],
        "Windowing": [
            "Tumbling window",
            "Sliding window",
            "Session window",
            "Window trigger and eviction"
        ],
        "Operators": [
            "Map/filter/flatMap operators",
            "Keyed aggregation",
            "Join operators (stream-stream, stream-table)",
            "Operator chaining"
        ],
        "State Management": [
            "Keyed state backend",
            "State checkpoint",
            "Exactly-once processing",
            "State recovery from checkpoint"
        ]
    },
    "order-matching-engine": {
        "Order Book": [
            "Price-level organization (sorted by price)",
            "Order queue per price level (FIFO)",
            "Bid and ask sides",
            "Order lookup by ID"
        ],
        "Order Types": [
            "Limit order handling",
            "Market order matching",
            "Stop order triggering",
            "Order cancellation"
        ],
        "Matching Logic": [
            "Price-time priority matching",
            "Partial fill handling",
            "Trade execution creation",
            "Match notification"
        ],
        "Performance": [
            "Low-latency data structures",
            "Lock-free order book option",
            "Throughput optimization",
            "Latency measurement"
        ]
    },
    "ledger-system": {
        "Account Structure": [
            "Chart of accounts hierarchy",
            "Account types (asset, liability, equity, revenue, expense)",
            "Account creation and management",
            "Account balance tracking"
        ],
        "Journal Entries": [
            "Double-entry transaction model",
            "Debit/credit validation",
            "Entry posting",
            "Entry reversal"
        ],
        "Reporting": [
            "Trial balance generation",
            "Balance sheet generation",
            "Income statement",
            "Account statement"
        ],
        "Audit Trail": [
            "Immutable entry logging",
            "Entry timestamp and user tracking",
            "Audit report generation",
            "Reconciliation support"
        ]
    },
    "build-vpn": {
        "TUN/TAP Interface": [
            "TUN device creation",
            "Packet reading from TUN",
            "Packet writing to TUN",
            "Interface configuration (IP, MTU)"
        ],
        "UDP Transport Layer": [
            "UDP socket for peer communication",
            "Packet encapsulation in UDP",
            "Peer address management",
            "NAT traversal basics"
        ],
        "Encryption Layer": [
            "Key derivation from shared secret",
            "Packet encryption (AES-GCM)",
            "Packet decryption and verification",
            "Nonce/IV management"
        ],
        "Key Exchange": [
            "Diffie-Hellman or ECDH key exchange",
            "Peer authentication",
            "Session key establishment",
            "Key rotation"
        ],
        "Routing Configuration": [
            "Route table manipulation",
            "Default gateway configuration",
            "Split tunneling option",
            "DNS configuration"
        ]
    },
    "cache-optimized-structures": {
        "Cache-Aware Arrays": [
            "Struct of Arrays (SoA) layout",
            "Array of Structs (AoS) comparison",
            "Prefetching strategies",
            "Cache line alignment"
        ],
        "B+ Tree for Disk/Cache": [
            "Node size matching cache line/page",
            "Sequential key scanning",
            "Bulk loading optimization",
            "Cache-friendly node layout"
        ],
        "Cache-Oblivious Algorithms": [
            "Van Emde Boas layout",
            "Cache-oblivious matrix multiplication",
            "Cache-oblivious sorting",
            "Theoretical analysis"
        ],
        "Benchmarking": [
            "Cache miss measurement",
            "Performance counters (perf)",
            "Comparison with naive implementations",
            "Memory access pattern analysis"
        ]
    },
    "sandbox": {
        "Namespace Isolation": [
            "PID namespace setup",
            "Network namespace isolation",
            "Mount namespace for filesystem",
            "User namespace for privilege drop"
        ],
        "Syscall Filtering": [
            "seccomp-bpf filter setup",
            "Whitelist/blacklist syscall policy",
            "Argument inspection",
            "Filter compilation"
        ],
        "Resource Limits": [
            "Cgroup CPU limits",
            "Cgroup memory limits",
            "File descriptor limits",
            "Process count limits"
        ],
        "Capability Control": [
            "Capability bounding set",
            "Drop unnecessary capabilities",
            "No-new-privileges flag",
            "Privilege escalation prevention"
        ]
    },
    "vulnerability-scanner": {
        "Host Discovery": [
            "ICMP ping sweep",
            "TCP/UDP host probing",
            "ARP scanning for local network",
            "Host database storage"
        ],
        "Port Scanning": [
            "TCP connect scan",
            "SYN scan (half-open)",
            "UDP scanning",
            "Service version detection"
        ],
        "Vulnerability Detection": [
            "Banner grabbing and analysis",
            "CVE database integration",
            "Version-based vulnerability matching",
            "Configuration weakness checks"
        ],
        "Reporting": [
            "Vulnerability report generation",
            "Severity classification (CVSS)",
            "Remediation recommendations",
            "Export formats (JSON, HTML, PDF)"
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
        print(f"\nProjects not in mapping ({len(missing)}):")
        for p in missing[:10]:
            print(f"  - {p}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()

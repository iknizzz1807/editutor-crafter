# EduTutor Crafter

> Project-based learning platform with AI review. Build real projects, get reviewed, level up.

## Vision

Part of the **editutor ecosystem**:

```
┌─────────────────────────────────────────────────────────┐
│                    LEARNING ECOSYSTEM                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────────────┐                                  │
│   │ editutor-crafter │ ← Roadmap + Projects             │
│   │     (this)       │   Step-by-step milestones        │
│   └────────┬─────────┘   AI code review                 │
│            │                                             │
│            ▼ build project                              │
│   ┌──────────────────┐                                  │
│   │   ai-editutor    │ ← Ask questions while coding     │
│   │    (plugin)      │   Learn in context               │
│   └────────┬─────────┘                                  │
│            │                                             │
│            ▼ knowledge saved                            │
│   ┌──────────────────┐                                  │
│   │ editutor-tracker │ ← Spaced repetition tests        │
│   │    (web app)     │   Reinforce learning             │
│   └────────┬─────────┘                                  │
│            │                                             │
│            └──────────► next project ───────────────────┘
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Core Concept

**Project-centric, not curriculum-centric.**

You pick projects that excite you. The system provides structure, milestones, and AI review to ensure you actually learn (not just copy-paste).

---

## Hierarchical Structure

```
Domain
└── Level (Beginner → Intermediate → Advanced → Expert)
    └── Projects (list)
        └── Milestones (sequential steps)
            └── Submissions (code + AI review)
```

---

## IT Domain Map (Complete)

### 10 Major Categories

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           IT LEARNING DOMAINS                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 🌐 APPLICATION DEVELOPMENT                                          │
│     ├── Web Development                                                 │
│     │   ├── Frontend (HTML/CSS, JS, React, Vue, Svelte)                │
│     │   ├── Backend (APIs, Auth, REST, GraphQL)                        │
│     │   └── Full-stack (integration, deployment)                       │
│     ├── Mobile Development                                              │
│     │   ├── iOS (Swift, UIKit, SwiftUI)                                │
│     │   ├── Android (Kotlin, Jetpack Compose)                          │
│     │   └── Cross-platform (React Native, Flutter)                     │
│     └── Desktop & CLI                                                   │
│         ├── Native (Qt, GTK, Win32, Cocoa)                             │
│         ├── Cross-platform (Electron, Tauri)                           │
│         └── CLI Tools                                                   │
│                                                                          │
│  2. ⚙️ SYSTEMS & LOW-LEVEL                                              │
│     ├── Systems Programming                                             │
│     │   ├── Memory management, pointers, allocation                    │
│     │   ├── Concurrency (threads, locks, atomics)                      │
│     │   ├── I/O, file systems                                          │
│     │   └── IPC, signals, sockets                                      │
│     ├── Networking                                                      │
│     │   ├── TCP/UDP, sockets programming                               │
│     │   ├── HTTP, WebSocket protocols                                  │
│     │   ├── Network protocols (DNS, TLS)                               │
│     │   └── P2P, distributed networking                                │
│     ├── Operating Systems                                               │
│     │   ├── Process management                                         │
│     │   ├── Memory (virtual memory, paging)                            │
│     │   ├── File systems                                               │
│     │   ├── Scheduling                                                  │
│     │   └── Kernel development                                         │
│     └── Embedded & IoT                                                  │
│         ├── Microcontrollers (Arduino, STM32, ESP32)                   │
│         ├── Real-time systems (RTOS)                                   │
│         ├── Hardware interfaces (GPIO, I2C, SPI)                       │
│         └── Firmware development                                       │
│                                                                          │
│  3. 🗄️ DATA & STORAGE                                                   │
│     ├── Databases                                                       │
│     │   ├── SQL (PostgreSQL, MySQL, query optimization)                │
│     │   ├── NoSQL (MongoDB, Redis, Cassandra)                          │
│     │   ├── Database internals (B-trees, WAL, MVCC)                    │
│     │   └── Build your own DB                                          │
│     └── Data Engineering                                                │
│         ├── ETL pipelines                                              │
│         ├── Stream processing (Kafka, Flink)                           │
│         ├── Data warehousing                                           │
│         └── Big data (Spark, Hadoop)                                   │
│                                                                          │
│  4. 🌍 DISTRIBUTED & CLOUD                                              │
│     ├── Distributed Systems                                             │
│     │   ├── CAP theorem, consistency models                            │
│     │   ├── Consensus (Raft, Paxos)                                    │
│     │   ├── Distributed storage                                        │
│     │   ├── Message queues (Kafka, RabbitMQ)                           │
│     │   └── Microservices architecture                                 │
│     └── Cloud & DevOps                                                  │
│         ├── Containers (Docker)                                        │
│         ├── Orchestration (Kubernetes)                                 │
│         ├── Cloud services (AWS, GCP, Azure)                           │
│         ├── Infrastructure as Code (Terraform)                         │
│         ├── CI/CD pipelines                                            │
│         ├── Monitoring & observability                                 │
│         └── Service mesh, load balancing                               │
│                                                                          │
│  5. 🤖 AI & MACHINE LEARNING                                            │
│     ├── Classical ML                                                    │
│     │   ├── Regression, classification                                 │
│     │   ├── Trees, forests, boosting                                   │
│     │   ├── Clustering, dimensionality reduction                       │
│     │   └── Feature engineering                                        │
│     ├── Deep Learning                                                   │
│     │   ├── Neural networks from scratch                               │
│     │   ├── CNNs (computer vision)                                     │
│     │   ├── RNNs, LSTMs (sequences)                                    │
│     │   ├── Transformers, attention                                    │
│     │   └── Training at scale                                          │
│     ├── NLP                                                             │
│     │   ├── Text processing, tokenization                              │
│     │   ├── Embeddings (Word2Vec, BERT)                                │
│     │   ├── Language models                                            │
│     │   └── RAG, fine-tuning                                           │
│     ├── Computer Vision                                                 │
│     │   ├── Image processing                                           │
│     │   ├── Object detection                                           │
│     │   ├── Segmentation                                               │
│     │   └── Video analysis                                             │
│     ├── Reinforcement Learning                                          │
│     │   ├── Q-learning, policy gradient                                │
│     │   ├── Game AI                                                     │
│     │   └── Robotics applications                                      │
│     └── MLOps                                                           │
│         ├── Model serving                                              │
│         ├── Experiment tracking                                        │
│         ├── Data versioning                                            │
│         └── Model monitoring                                           │
│                                                                          │
│  6. 🎮 GAME DEVELOPMENT                                                 │
│     ├── Game Programming                                                │
│     │   ├── Game loop, input handling                                  │
│     │   ├── 2D games (sprites, physics)                                │
│     │   ├── 3D games (transforms, cameras)                             │
│     │   └── Game AI (pathfinding, behavior trees)                      │
│     ├── Graphics Programming                                            │
│     │   ├── 2D rendering (SDL, raylib)                                 │
│     │   ├── 3D rendering (OpenGL, Vulkan)                              │
│     │   ├── Shaders (GLSL, HLSL)                                       │
│     │   └── Ray tracing                                                │
│     └── Game Engine Development                                         │
│         ├── Entity Component System (ECS)                              │
│         ├── Physics engine                                             │
│         ├── Audio system                                               │
│         ├── Asset pipeline                                             │
│         └── Scripting integration                                      │
│                                                                          │
│  7. 📝 LANGUAGES & COMPILERS                                            │
│     ├── Parsing & Lexing                                                │
│     │   ├── Lexers, tokenizers                                         │
│     │   ├── Recursive descent parsers                                  │
│     │   ├── Parser generators (ANTLR, yacc)                            │
│     │   └── AST design                                                  │
│     ├── Interpreters                                                    │
│     │   ├── Tree-walking interpreters                                  │
│     │   ├── Bytecode VMs                                               │
│     │   └── Stack vs register machines                                 │
│     ├── Compilers                                                       │
│     │   ├── IR design                                                   │
│     │   ├── Code generation                                            │
│     │   ├── Optimization passes                                        │
│     │   └── LLVM integration                                           │
│     ├── Type Systems                                                    │
│     │   ├── Static vs dynamic typing                                   │
│     │   ├── Type inference                                             │
│     │   ├── Generics, polymorphism                                     │
│     │   └── Dependent types                                            │
│     ├── Runtime Systems                                                 │
│     │   ├── Garbage collection                                         │
│     │   ├── Memory management                                          │
│     │   └── JIT compilation                                            │
│     └── Developer Tools                                                 │
│         ├── LSP servers                                                 │
│         ├── Debuggers                                                   │
│         ├── Linters, formatters                                        │
│         └── Build systems                                              │
│                                                                          │
│  8. 🔒 SECURITY                                                         │
│     ├── Cryptography                                                    │
│     │   ├── Symmetric encryption (AES)                                 │
│     │   ├── Asymmetric encryption (RSA, ECC)                           │
│     │   ├── Hashing, signatures                                        │
│     │   └── TLS/SSL implementation                                     │
│     ├── Web Security                                                    │
│     │   ├── OWASP Top 10                                               │
│     │   ├── Authentication & authorization                             │
│     │   ├── Input validation                                           │
│     │   └── Secure coding practices                                    │
│     └── Offensive Security                                              │
│         ├── Penetration testing                                        │
│         ├── Vulnerability research                                     │
│         ├── Reverse engineering                                        │
│         └── Binary exploitation                                        │
│                                                                          │
│  9. 🧮 CS FUNDAMENTALS                                                  │
│     ├── Data Structures                                                 │
│     │   ├── Arrays, linked lists, stacks, queues                       │
│     │   ├── Trees (BST, AVL, Red-Black, B-trees)                       │
│     │   ├── Graphs                                                      │
│     │   ├── Hash tables                                                 │
│     │   └── Advanced (skip lists, bloom filters, tries)                │
│     ├── Algorithms                                                      │
│     │   ├── Sorting, searching                                         │
│     │   ├── Graph algorithms (BFS, DFS, Dijkstra)                      │
│     │   ├── Dynamic programming                                        │
│     │   ├── Greedy algorithms                                          │
│     │   └── String algorithms                                          │
│     └── System Design                                                   │
│         ├── Scalability patterns                                       │
│         ├── Caching strategies                                         │
│         ├── Rate limiting                                              │
│         └── Design interviews prep                                     │
│                                                                          │
│  10. 🔧 SPECIALIZED (Future)                                            │
│      ├── Blockchain & Web3                                              │
│      ├── AR/VR Development                                              │
│      ├── Audio Programming                                              │
│      └── Scientific Computing                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Build Your Own X (Master Projects)

> Projects that take you from "knows how to use" → "deeply understands how it works"
>
> These are typically **Expert level** projects within each domain.

### Data & Storage

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Redis** | In-memory KV store | Data structures, persistence, pub/sub, RESP protocol |
| **Build Your Own SQLite** | Embedded database | B-tree, SQL parser, query planner, ACID transactions |
| **Build Your Own Kafka** | Message queue | Append-only log, partitions, consumer groups, replication |
| **Build Your Own Time-Series DB** | Metrics database | Compression, downsampling, retention policies |

### Systems & OS

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own OS** | Operating system kernel | Bootloader, scheduler, syscalls, drivers, memory management |
| **Build Your Own Shell** | Unix shell | Parsing, pipes, redirects, job control, builtins |
| **Build Your Own Docker** | Container runtime | Namespaces, cgroups, overlay filesystem, networking |
| **Build Your Own Memory Allocator** | malloc/free | Fragmentation, free lists, buddy system, slab allocation |
| **Build Your Own File System** | File system | Inodes, directories, journaling, block allocation |

### Networking

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own TCP/IP Stack** | Network stack | Ethernet frames, IP packets, TCP state machine, sockets |
| **Build Your Own HTTP Server** | Web server | HTTP/1.1, keep-alive, chunked transfer, HTTP/2, TLS |
| **Build Your Own Load Balancer** | Load balancer | Round-robin, least-connections, health checks, L4/L7 |
| **Build Your Own Proxy** | SOCKS/HTTP proxy | Tunneling, MITM, caching, connection pooling |
| **Build Your Own DNS Server** | DNS resolver | UDP protocol, recursion, caching, zone files |

### Distributed Systems

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Raft** | Consensus algorithm | Leader election, log replication, snapshots, membership |
| **Build Your Own Distributed KV** | Distributed database | Sharding, replication, consistency, partitioning |
| **Build Your Own MapReduce** | Distributed computing | Job scheduling, fault tolerance, shuffling |
| **Build Your Own Service Mesh** | Service mesh | Sidecar proxy, mTLS, circuit breaker, observability |

### Languages & Compilers

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Programming Language** | Full language | Lexer, parser, type checker, codegen, runtime |
| **Build Your Own Interpreter** | Lox/Lisp/Lua interpreter | Tree-walking or bytecode VM, closures, GC |
| **Build Your Own Compiler** | Compiler to Assembly/LLVM | IR design, optimization passes, target codegen |
| **Build Your Own Regex Engine** | Regex matcher | NFA/DFA construction, backtracking, optimization |
| **Build Your Own Garbage Collector** | GC | Mark-sweep, copying, generational, concurrent |
| **Build Your Own JIT Compiler** | JIT | Runtime codegen, tracing, inline caching |

### Developer Tools

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Git** | Version control | Objects, refs, pack files, merge algorithms, diff |
| **Build Your Own Text Editor** | Editor (vim-like) | Gap buffer/rope, syntax highlighting, modal editing |
| **Build Your Own Debugger** | Debugger | ptrace, breakpoints, stepping, symbol tables |
| **Build Your Own LSP Server** | Language server | LSP protocol, diagnostics, completion, go-to-definition |
| **Build Your Own Build System** | Build tool (make/bazel) | Dependency graph, incremental builds, caching |

### Game & Graphics

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Game Engine** | Game engine | ECS, renderer, physics, audio, scripting, editor |
| **Build Your Own Physics Engine** | Physics simulation | Collision detection, rigid body dynamics, constraints |
| **Build Your Own 3D Renderer** | Software renderer | Rasterization, shaders, shadows, PBR |
| **Build Your Own Ray Tracer** | Ray tracer | Path tracing, BVH acceleration, materials, GI |
| **Build Your Own Font Renderer** | Font rasterizer | TrueType parsing, bezier curves, hinting, subpixel |

### AI & ML

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Neural Network Framework** | PyTorch/TensorFlow clone | Autograd, tensor ops, layers, optimizers, GPU |
| **Build Your Own Transformer** | Transformer architecture | Self-attention, positional encoding, training loop |
| **Build Your Own Search Engine** | Search engine | Crawler, indexer, inverted index, ranking (TF-IDF, BM25) |
| **Build Your Own Recommendation System** | RecSys | Collaborative filtering, embeddings, ranking |

### Web & Browser

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own Web Browser** | Browser engine | HTML parser, CSS, layout engine, rendering, JS engine |
| **Build Your Own Web Framework** | Express/Django clone | Routing, middleware, templating, ORM |
| **Build Your Own React** | UI framework | Virtual DOM, reconciliation, hooks, fiber |
| **Build Your Own Bundler** | Webpack clone | Module resolution, tree shaking, code splitting |

### Security

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own TLS** | TLS implementation | Handshake, certificates, cipher suites, AEAD |
| **Build Your Own Password Manager** | Password manager | Encryption, key derivation (Argon2), secure storage |
| **Build Your Own Firewall** | Packet filter | iptables-like rules, stateful inspection, NAT |

### Embedded & Emulation

| Project | What You'll Build | Key Concepts |
|---------|-------------------|--------------|
| **Build Your Own RTOS** | Real-time OS | Scheduler, IPC, drivers, real-time guarantees |
| **Build Your Own Emulator** | NES/GameBoy/CHIP-8 | CPU emulation, memory mapping, graphics, input |

---

## Domain Examples with Build Your Own

### Systems Programming

```
Systems Programming
│
├── 🟢 Beginner
│   ├── Cat clone
│   ├── Wc clone
│   ├── Find clone
│   └── Simple HTTP client
│
├── 🟡 Intermediate
│   ├── Shell (basic - pipes, redirects)
│   ├── HTTP server (basic)
│   ├── Memory pool allocator
│   └── File watcher
│
├── 🟠 Advanced
│   ├── HTTP/2 server
│   ├── Container (namespaces only)
│   ├── Custom malloc
│   └── Thread pool
│
└── 🔴 Expert (Build Your Own)
    ├── ⭐ Build Your Own Docker
    │   ├── M1: Process isolation (namespaces)
    │   ├── M2: Resource limits (cgroups)
    │   ├── M3: Layered filesystem (overlay)
    │   ├── M4: Container networking
    │   ├── M5: Image format & registry
    │   └── M6: Docker CLI compatible
    │
    ├── ⭐ Build Your Own Shell (full)
    │   ├── M1: Lexer & parser
    │   ├── M2: Builtins (cd, export, etc.)
    │   ├── M3: Pipes & redirects
    │   ├── M4: Job control (fg, bg, jobs)
    │   ├── M5: Signal handling
    │   └── M6: Tab completion & history
    │
    ├── ⭐ Build Your Own Memory Allocator
    │   ├── M1: Simple bump allocator
    │   ├── M2: Free list allocator
    │   ├── M3: Buddy system
    │   ├── M4: Slab allocator
    │   └── M5: Thread-safe allocator
    │
    └── ⭐ Build Your Own OS
        ├── M1: Bootloader (BIOS/UEFI)
        ├── M2: Protected mode & GDT
        ├── M3: Interrupts & exceptions
        ├── M4: Physical memory manager
        ├── M5: Virtual memory & paging
        ├── M6: Kernel heap
        ├── M7: Process & scheduler
        ├── M8: System calls
        ├── M9: File system (FAT32 or ext2)
        └── M10: User space & shell
```

### Databases

```
Databases
│
├── 🟢 Beginner
│   ├── JSON file database
│   ├── CSV query engine
│   └── In-memory key-value store
│
├── 🟡 Intermediate
│   ├── B-tree implementation
│   ├── LSM tree
│   ├── SQL parser (SELECT, WHERE)
│   └── Simple indexing
│
├── 🟠 Advanced
│   ├── Query optimizer (basic)
│   ├── Transaction manager (ACID)
│   ├── WAL implementation
│   └── Replication (leader-follower)
│
└── 🔴 Expert (Build Your Own)
    ├── ⭐ Build Your Own Redis
    │   ├── M1: TCP server + RESP protocol
    │   ├── M2: GET/SET/DEL commands
    │   ├── M3: Expiration (TTL)
    │   ├── M4: Data structures (List, Set, Hash)
    │   ├── M5: Persistence (RDB snapshots)
    │   ├── M6: Persistence (AOF)
    │   ├── M7: Pub/Sub
    │   └── M8: Cluster mode (sharding)
    │
    ├── ⭐ Build Your Own SQLite
    │   ├── M1: SQL tokenizer
    │   ├── M2: SQL parser (AST)
    │   ├── M3: B-tree page format
    │   ├── M4: Table storage
    │   ├── M5: SELECT execution
    │   ├── M6: INSERT/UPDATE/DELETE
    │   ├── M7: Indexes
    │   ├── M8: Query planner
    │   ├── M9: Transactions (BEGIN/COMMIT)
    │   └── M10: WAL mode
    │
    └── ⭐ Build Your Own Distributed DB
        ├── M1: Single-node KV store
        ├── M2: Raft leader election
        ├── M3: Raft log replication
        ├── M4: Raft snapshots
        ├── M5: Client routing
        ├── M6: Sharding (hash/range)
        └── M7: Rebalancing
```

### Compilers & Languages

```
Compilers & Languages
│
├── 🟢 Beginner
│   ├── Calculator parser
│   ├── JSON parser
│   ├── Markdown parser
│   └── INI/TOML parser
│
├── 🟡 Intermediate
│   ├── Regex engine (basic NFA)
│   ├── Lisp interpreter
│   ├── Forth interpreter
│   └── Bytecode VM (stack-based)
│
├── 🟠 Advanced
│   ├── Type checker
│   ├── Closure implementation
│   ├── Compiler to C
│   ├── Register-based VM
│   └── Simple GC (mark-sweep)
│
└── 🔴 Expert (Build Your Own)
    ├── ⭐ Build Your Own Programming Language
    │   ├── M1: Lexer
    │   ├── M2: Parser (Pratt/recursive descent)
    │   ├── M3: AST & visitor pattern
    │   ├── M4: Type system
    │   ├── M5: IR generation
    │   ├── M6: Optimization passes
    │   ├── M7: Code generation (LLVM or native)
    │   ├── M8: Standard library
    │   └── M9: Package manager
    │
    ├── ⭐ Build Your Own Interpreter (Lox)
    │   ├── M1: Scanning (lexer)
    │   ├── M2: Representing code (AST)
    │   ├── M3: Parsing expressions
    │   ├── M4: Evaluating expressions
    │   ├── M5: Statements & state
    │   ├── M6: Control flow
    │   ├── M7: Functions
    │   ├── M8: Closures
    │   ├── M9: Classes
    │   └── M10: Inheritance
    │
    ├── ⭐ Build Your Own Garbage Collector
    │   ├── M1: Object representation
    │   ├── M2: Root scanning
    │   ├── M3: Mark phase
    │   ├── M4: Sweep phase
    │   ├── M5: Generational GC
    │   ├── M6: Concurrent marking
    │   └── M7: Compaction
    │
    └── ⭐ Build Your Own JIT
        ├── M1: Bytecode interpreter
        ├── M2: Basic block detection
        ├── M3: Native code emission
        ├── M4: Register allocation
        ├── M5: Inline caching
        ├── M6: Tracing JIT
        └── M7: Deoptimization
```

### Game Development

```
Game Development
│
├── 🟢 Beginner
│   ├── Pong
│   ├── Snake
│   ├── Breakout
│   └── Tetris
│
├── 🟡 Intermediate
│   ├── Platformer
│   ├── Top-down shooter
│   ├── Puzzle game (Sokoban)
│   └── Card game
│
├── 🟠 Advanced
│   ├── Software 3D renderer
│   ├── ECS architecture
│   ├── Multiplayer netcode
│   └── Procedural generation
│
└── 🔴 Expert (Build Your Own)
    ├── ⭐ Build Your Own Game Engine
    │   ├── M1: Window & input (SDL/GLFW)
    │   ├── M2: 2D sprite rendering
    │   ├── M3: Entity Component System
    │   ├── M4: Physics integration
    │   ├── M5: Audio system
    │   ├── M6: Asset pipeline
    │   ├── M7: Scripting (Lua)
    │   ├── M8: Scene serialization
    │   └── M9: Editor tools (ImGui)
    │
    ├── ⭐ Build Your Own Physics Engine
    │   ├── M1: Vector/matrix math
    │   ├── M2: Rigid body dynamics
    │   ├── M3: Collision detection (broad phase)
    │   ├── M4: Collision detection (narrow phase)
    │   ├── M5: Collision response
    │   ├── M6: Constraints & joints
    │   └── M7: Continuous collision detection
    │
    ├── ⭐ Build Your Own 3D Renderer
    │   ├── M1: Line drawing (Bresenham)
    │   ├── M2: Triangle rasterization
    │   ├── M3: Z-buffer
    │   ├── M4: Perspective projection
    │   ├── M5: Texture mapping
    │   ├── M6: Phong lighting
    │   ├── M7: Shadow mapping
    │   └── M8: Normal mapping
    │
    └── ⭐ Build Your Own Ray Tracer
        ├── M1: Ray-sphere intersection
        ├── M2: Multiple objects
        ├── M3: Diffuse materials
        ├── M4: Metal & glass
        ├── M5: Camera (DOF, motion blur)
        ├── M6: BVH acceleration
        ├── M7: Textures & UV mapping
        └── M8: Global illumination
```

### AI & Machine Learning

```
AI & Machine Learning
│
├── 🟢 Beginner
│   ├── Linear regression from scratch
│   ├── Logistic regression
│   ├── KNN classifier
│   └── Decision tree
│
├── 🟡 Intermediate
│   ├── Neural network (micrograd-style)
│   ├── CNN for MNIST
│   ├── Word embeddings (Word2Vec)
│   └── Random forest
│
├── 🟠 Advanced
│   ├── Transformer from scratch
│   ├── RL agent (Q-learning, DQN)
│   ├── GAN
│   └── Object detection (YOLO-style)
│
└── 🔴 Expert (Build Your Own)
    ├── ⭐ Build Your Own Neural Network Framework
    │   ├── M1: Tensor class
    │   ├── M2: Autograd (computational graph)
    │   ├── M3: Basic operations (add, mul, matmul)
    │   ├── M4: Activation functions
    │   ├── M5: Loss functions
    │   ├── M6: Optimizers (SGD, Adam)
    │   ├── M7: Layers (Linear, Conv2d)
    │   ├── M8: GPU support (CUDA)
    │   └── M9: Model serialization
    │
    ├── ⭐ Build Your Own Transformer
    │   ├── M1: Tokenizer (BPE)
    │   ├── M2: Embedding layer
    │   ├── M3: Positional encoding
    │   ├── M4: Self-attention
    │   ├── M5: Multi-head attention
    │   ├── M6: Feed-forward layers
    │   ├── M7: Encoder stack
    │   ├── M8: Decoder stack
    │   ├── M9: Training loop
    │   └── M10: Inference (generation)
    │
    └── ⭐ Build Your Own Search Engine
        ├── M1: Web crawler
        ├── M2: HTML parser & text extraction
        ├── M3: Inverted index
        ├── M4: TF-IDF scoring
        ├── M5: BM25 ranking
        ├── M6: Query parser
        ├── M7: Spell correction
        └── M8: PageRank (optional)
```

---

## Milestone Structure

Each milestone has:

```yaml
milestone:
  id: redis-01-ping-pong
  project: redis-clone
  name: "PING/PONG Protocol"

  description: |
    Implement basic Redis server that responds to PING command.

  # Clear, testable criteria
  acceptance_criteria:
    - Server listens on TCP port 6379
    - Responds to PING with +PONG\r\n (RESP protocol)
    - Handles multiple concurrent clients
    - Clean shutdown on SIGINT

  # Optional: automated tests
  tests:
    - command: "echo 'PING' | nc localhost 6379"
      expect: "+PONG"
    - command: "redis-benchmark -t ping -n 1000"
      expect: "exit_code: 0"

  # Hints (progressive reveal if stuck)
  hints:
    - "Look into Go's net.Listen for TCP"
    - "RESP protocol: https://redis.io/docs/reference/protocol-spec/"
    - "Use goroutines for concurrent clients"

  # Context for AI review
  review_focus:
    - Error handling approach
    - Concurrency model choice
    - Code organization
```

---

## Submit & Review Flow

```
┌─────────────────────────────────────────────────────────┐
│                    SUBMIT FLOW                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. User clicks "Submit Milestone"                      │
│         │                                                │
│         ▼                                                │
│  2. Extract project code                                │
│     ├── Tree structure                                  │
│     ├── Source files (smart selection, token budget)   │
│     └── Reuse ai-editutor context extraction logic     │
│         │                                                │
│         ▼                                                │
│  3. Run automated tests (if defined)                    │
│     ├── PASS → continue to AI review                   │
│     └── FAIL → instant feedback, no AI call needed     │
│         │                                                │
│         ▼                                                │
│  4. AI Review                                           │
│     ├── Check each acceptance criterion                │
│     ├── Code quality assessment                         │
│     ├── Architecture feedback                           │
│     └── Learning suggestions                            │
│         │                                                │
│         ▼                                                │
│  5. Verdict                                             │
│     ├── ACCEPT → unlock next milestone                 │
│     │           → generate concepts for tracker        │
│     └── REJECT → specific feedback                     │
│                 → must fix and resubmit                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## AI Review Prompt Template

```markdown
# Role
You are a senior engineer reviewing a milestone submission.
Be strict but educational. Reject if criteria not met.

# Context
Project: {{project.name}}
Milestone: {{milestone.name}}
Description: {{milestone.description}}

# Acceptance Criteria
{{#each milestone.acceptance_criteria}}
- [ ] {{this}}
{{/each}}

# Automated Test Results
{{test_results}}

# Submitted Code
## Project Structure
{{tree_structure}}

## Files
{{#each files}}
### {{this.path}}
```{{this.language}}
{{this.content}}
```
{{/each}}

# Your Task

## 1. Criteria Check
For each criterion, mark PASS or FAIL with brief explanation.

## 2. Verdict
- If ALL criteria pass → **ACCEPT**
- If ANY criterion fails → **REJECT**

## 3. Code Review (regardless of verdict)
- What's done well?
- What could be improved?
- Architecture observations
- Potential issues at scale

## 4. Learning Pointers
- Concepts to explore deeper
- Related topics for ai-editutor questions
- Resources if relevant

# Response Format
{
  "verdict": "ACCEPT" | "REJECT",
  "criteria_results": [
    {"criterion": "...", "status": "PASS|FAIL", "note": "..."}
  ],
  "feedback": {
    "strengths": ["..."],
    "improvements": ["..."],
    "concerns": ["..."]
  },
  "learning": {
    "concepts": ["...", "..."],
    "questions_to_explore": ["...", "..."]
  }
}
```

---

## Data Model

```
┌─────────────┐
│   Domain    │
├─────────────┤
│ id          │
│ name        │  "Game Development"
│ icon        │
│ description │
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│    Level    │
├─────────────┤
│ id          │
│ domain_id   │
│ name        │  "Beginner" | "Intermediate" | "Advanced" | "Expert"
│ order       │  1, 2, 3, 4
│ color       │  green, yellow, orange, red
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│   Project   │
├─────────────┤
│ id          │
│ level_id    │
│ name        │  "Pong Clone"
│ description │
│ tags[]      │  for "Build Your Own" projects
│ order       │
│ status      │  locked | available | in_progress | completed
│ repo_path   │  local path to project code
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│  Milestone  │
├─────────────┤
│ id          │
│ project_id  │
│ name        │  "Ball physics"
│ description │
│ criteria[]  │  acceptance criteria
│ hints[]     │  progressive hints
│ tests[]     │  automated test commands
│ order       │
│ status      │  locked | available | submitted | passed
└──────┬──────┘
       │ has many
       ▼
┌─────────────┐
│ Submission  │
├─────────────┤
│ id          │
│ milestone_id│
│ code        │  JSON snapshot (tree + files)
│ test_result │  automated test output
│ ai_review   │  JSON response from AI
│ verdict     │  ACCEPT | REJECT
│ created_at  │
└─────────────┘
```

---

## Unlock Logic

```
Level unlock:
├── Beginner: always unlocked
├── Intermediate: complete ≥2 Beginner projects in domain
├── Advanced: complete ≥2 Intermediate projects in domain
└── Expert: complete ≥2 Advanced projects in domain

Project unlock:
├── First project in level: auto unlocked when level unlocked
└── Others: complete ≥1 project in same level

Milestone unlock:
└── Sequential within project (must pass M1 → M2 → M3...)
```

Alternative: Flexible mode - everything unlocked, system only **recommends** order.

---

## UI Wireframes

### Domain Overview

```
┌─────────────────────────────────────────────────────────┐
│  🎮 Game Development                    [12/28 done]    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  🟢 Beginner ████████████░░ 3/4 projects                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │  Pong   │ │  Snake  │ │Breakout │ │ Tetris  │       │
│  │   ✓     │ │   ✓     │ │   ✓     │ │  🔒    │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                                                          │
│  🟡 Intermediate ████░░░░░░ 1/3 projects                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │Platform │ │ Shooter │ │ Puzzle  │                   │
│  │  ⏳ 3/5 │ │   ○     │ │   ○     │                   │
│  └─────────┘ └─────────┘ └─────────┘                   │
│                                                          │
│  🟠 Advanced 🔒 (complete 2 intermediate to unlock)     │
│                                                          │
│  🔴 Expert (Build Your Own) 🔒                          │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ │
│  │ ⭐ Game       │ │ ⭐ Physics    │ │ ⭐ 3D         │ │
│  │    Engine     │ │    Engine     │ │    Renderer   │ │
│  └───────────────┘ └───────────────┘ └───────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Project Detail

```
┌─────────────────────────────────────────────────────────┐
│  ← Back    ⭐ Build Your Own Redis          🔴 Expert   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Progress: ████████░░░░░░░░ 4/8 milestones              │
│                                                          │
│  ✓ M1: TCP server + RESP protocol                       │
│  ✓ M2: GET/SET/DEL commands                             │
│  ✓ M3: Expiration (TTL)                                 │
│  ✓ M4: Data structures (List, Set, Hash)               │
│  ⏳ M5: Persistence (RDB)         [Submit for Review]   │
│  🔒 M6: Persistence (AOF)                               │
│  🔒 M7: Pub/Sub                                         │
│  🔒 M8: Cluster mode                                    │
│                                                          │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  Current: M5 - RDB Persistence                          │
│                                                          │
│  Description:                                            │
│  Implement RDB snapshot persistence. Server should      │
│  save in-memory data to disk and restore on startup.   │
│                                                          │
│  Acceptance Criteria:                                    │
│  • SAVE command triggers snapshot                       │
│  • BGSAVE runs in background                           │
│  • Data restored on server restart                     │
│  • RDB file format compatible with Redis              │
│                                                          │
│  [View Hints]  [Open Project Folder]  [Submit Code]     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Submission Review

```
┌─────────────────────────────────────────────────────────┐
│  Review Result                              ✓ ACCEPTED  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Criteria Results:                                       │
│  ✓ SAVE command triggers snapshot                       │
│  ✓ BGSAVE runs in background                           │
│  ✓ Data restored on server restart                     │
│  ✓ RDB file format compatible                          │
│                                                          │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  Strengths:                                              │
│  • Clean fork() implementation for BGSAVE              │
│  • Proper signal handling for child process            │
│  • Good error recovery if snapshot fails               │
│                                                          │
│  Suggestions:                                            │
│  • Consider copy-on-write optimization                 │
│  • Add checksum validation for RDB file               │
│  • BGSAVE could report progress                        │
│                                                          │
│  Concepts to Explore:                                    │
│  • Copy-on-write (COW) memory                          │
│  • File system sync guarantees (fsync)                 │
│  • Crash recovery strategies                           │
│                                                          │
│  [Continue to M6: AOF Persistence]                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack (Proposed)

```
Frontend: React + Vite (consistent with editutor-tracker)
Backend: Go + Gin (consistent with editutor-tracker)
Database: SQLite (simple, local-first)
AI: Gemini API (or configurable)
Code extraction: Port ai-editutor logic (Lua → Go)
```

---

## Integration Points

### With ai-editutor

```
Platform                          ai-editutor
   │                                   │
   │  User builds project              │
   │                                   │
   │  ──────── questions ───────────>  │
   │                                   │
   │  <─────── knowledge.json ───────  │
   │                                   │
   │  Platform reads knowledge to      │
   │  understand what user struggled   │
   │  with during this milestone       │
   │                                   │
```

### With editutor-tracker

```
Platform                          Tracker
   │                                   │
   │  AI review generates              │
   │  "concepts to reinforce"          │
   │                                   │
   │  ──────── concepts ────────────>  │
   │                                   │
   │  Tracker creates tests            │
   │  for those concepts               │
   │                                   │
   │  <─────── test results ─────────  │
   │                                   │
   │  Platform sees mastery level      │
   │                                   │
```

---

## MVP Scope

### Phase 1: Core
- [ ] Domain/Level/Project/Milestone data model
- [ ] Basic UI: browse domains, projects, milestones
- [ ] Submit milestone: extract code (tree + files)
- [ ] AI review: call Gemini, parse response
- [ ] Pass/fail logic, unlock next milestone

### Phase 2: Content
- [ ] Populate 2-3 domains with real projects
- [ ] Write detailed milestones with criteria
- [ ] Add hints for common stuck points
- [ ] Include "Build Your Own" expert projects

### Phase 3: Integration
- [ ] Read ai-editutor knowledge.json
- [ ] Push concepts to tracker
- [ ] Unified progress dashboard

---

## Open Questions

1. **Local-first or cloud?** Store submissions locally or sync to cloud?
2. **Project templates?** Provide starter code or fully from scratch?
3. **Community?** Eventually allow sharing projects/milestones?
4. **Gamification level?** XP, levels, streaks like tracker?

---

## Resources

### Build Your Own X Sources
- [Build Your Own X (GitHub)](https://github.com/codecrafters-io/build-your-own-x)
- [Codecrafters](https://codecrafters.io)
- [Crafting Interpreters](https://craftinginterpreters.com)
- [Handmade Hero](https://handmadehero.org)
- [tinyrenderer](https://github.com/ssloy/tinyrenderer)
- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [nand2tetris](https://www.nand2tetris.org)
- [Writing an OS in Rust](https://os.phil-opp.com)
- [Ray Tracing in One Weekend](https://raytracing.github.io)

### Curriculum References
- [MIT 6.824 Distributed Systems](https://pdos.csail.mit.edu/6.824/)
- [MIT 6.828 Operating Systems](https://pdos.csail.mit.edu/6.828/)
- [Stanford CS143 Compilers](https://web.stanford.edu/class/cs143/)
- [CMU 15-445 Database Systems](https://15445.courses.cs.cmu.edu)
- [roadmap.sh](https://roadmap.sh)

### Project Idea Collections
- [Project-Based Learning](https://github.com/practical-tutorials/project-based-learning)
- [App Ideas Collection](https://github.com/florinpop17/app-ideas)
- [Mega Project List](https://github.com/karan/Projects)

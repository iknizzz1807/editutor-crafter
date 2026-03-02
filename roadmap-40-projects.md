# 40 Projects — Roadmap Kỹ Sư Không Thể Thay Thế

> Con đường từ fresher đến staff-level knowledge. Low-level, infrastructure, performance, AI/ML depth — kiến thức mà AI không thể thay thế.

---

## Tổng Quan

| Phase | Focus | Projects | Thời gian |
|-------|-------|----------|-----------|
| 1 | Nền Tảng | 4 | 2-3 tuần |
| 2 | OS & Systems | 6 | 6-8 tuần |
| 3 | Networking & Backend | 4 | 3-4 tuần |
| 4 | Language & Data Internals | 5 | 5-6 tuần |
| 5 | Distributed Systems | 3 | 4-5 tuần |
| 6 | Performance & Nanosecond | 8 | 8-12 tuần |
| 7 | AI/ML Infrastructure | 6 | 8-10 tuần |
| 8 | Capstone | 4 | 8-10 tuần |
| **Total** | | **40** | **12-16 tháng** |

---

## Phase 1 — Nền Tảng (4)

| # | Project | Core Lesson |
|---|---------|-------------|
| 1 | `tokenizer` | Máy tính đọc code thế nào — parsing mindset |
| 2 | `hash-impl` | Bit manipulation, crypto primitives |
| 3 | `http-server-basic` | TCP, sockets, request/response protocol |
| 4 | `build-event-loop` | epoll, async I/O — bản chất non-blocking |

**Skills acquired:** Parsing, bit operations, networking basics, async fundamentals

---

## Phase 2 — OS & Systems (6)

| # | Project | Core Lesson |
|---|---------|-------------|
| 5 | `build-strace` | Syscall tracing, kernel-userspace boundary |
| 6 | `build-kernel-module` | Code chạy trong kernel space |
| 7 | `container-basic` | Namespaces, cgroups — bản chất Docker |
| 8 | `filesystem` | Inodes, blocks, VFS — data sống ở đâu |
| 9 | `virtual-memory-sim` | Page tables, TLB, page faults |
| 10 | `build-shell` | Job control, piping, signals — OS integration |

**Skills acquired:** Kernel internals, process isolation, memory management, systems programming

---

## Phase 3 — Networking & Backend (4)

| # | Project | Core Lesson |
|---|---------|-------------|
| 11 | `message-queue` | Async patterns, pub/sub, persistence |
| 12 | `packet-sniffer` | Raw packets, Ethernet → IP → TCP layering |
| 13 | `build-tcp-stack` | TCP/IP từ zero — networking ở tầng sâu nhất |
| 14 | `profiler` | CPU sampling, flame graphs, perf analysis |

**Skills acquired:** Network protocols, observability, debugging production systems

---

## Phase 4 — Language & Data Internals (5)

| # | Project | Core Lesson |
|---|---------|-------------|
| 15 | `bytecode-vm` | Compile → bytecode → interpret |
| 16 | `linker` | Symbol resolution, relocation, ELF format |
| 17 | `wal-impl` | Write-Ahead Log, durability, crash recovery |
| 18 | `vector-database` | ANN search, embedding storage |
| 19 | `query-optimizer` | Cost-based planning, index selection |

**Skills acquired:** Compiler/VM internals, linking, database fundamentals

---

## Phase 5 — Distributed Systems (3)

| # | Project | Core Lesson |
|---|---------|-------------|
| 20 | `gossip-protocol` | Epidemic broadcast, cluster self-healing |
| 21 | `distributed-cache` | Consistent hashing, invalidation strategies |
| 22 | `build-raft` | Consensus — holy grail of distributed systems |

**Skills acquired:** Consensus, replication, distributed coordination

---

## Phase 6 — Performance & Nanosecond (8)

| # | Project | Core Lesson |
|---|---------|-------------|
| 23 | `memory-pool` | Custom allocator, fragmentation control |
| 24 | `simd-library` | CPU vectorization — instruction-level speed |
| 25 | `cache-optimized-structures` | Cache line alignment, prefetch, struct layout |
| 26 | `lock-free-structures` | CAS, atomics — concurrency không lock |
| 27 | `ecs-arch` | Data-oriented design — cache-friendly thinking |
| 28 | `io-uring-server` | Zero-syscall async I/O — thế hệ mới |
| 29 | `kernel-bypass-network-stack` | DPDK-style — network latency ~ns |
| 30 | `zero-copy-msg-bus` | Eliminate memcpy — IPC tốc độ tối đa |

**Skills acquired:** Memory optimization, CPU architecture, lock-free programming, nanosecond-level tuning

---

## Phase 7 — AI/ML Infrastructure (6)

| # | Project | Core Lesson |
|---|---------|-------------|
| 31 | `neural-network-basic` | Backprop từ zero — hiểu bản chất |
| 32 | `transformer-scratch` | Attention mechanism — hiểu LLM từ gốc |
| 33 | `build-gpu-compute` | CUDA programming — ngôn ngữ của NVIDIA |
| 34 | `tensor-quantization-engine` | INT8/FP16 — model optimization |
| 35 | `inference-engine` | Model serving, batching, KV-cache |
| 36 | `distributed-training-framework` | Multi-GPU, gradient sync — scale AI |

**Skills acquired:** Deep learning fundamentals, GPU programming, ML systems engineering

---

## Phase 8 — Capstone (4)

| # | Project | Core Lesson |
|---|---------|-------------|
| 37 | `build-redis` | Network server + data structures + persistence |
| 38 | `build-docker` | Container runtime từ zero |
| 39 | `build-allocator` | Memory allocator — low-level cuối cùng |
| 40 | `build-jit` | JIT compiler — generate native code |

**Skills acquired:** Systems integration, production-grade engineering

---

## Dependency Chains

```
# Parsing Track
tokenizer (1) → bytecode-vm (15) → build-jit (40)

# Networking Track  
http-server (3) → packet-sniffer (12) → build-tcp-stack (13) → kernel-bypass (29)

# Async Track
build-event-loop (4) → io-uring-server (28) → coroutine (implied in redis)

# Memory Track
virtual-memory-sim (9) → memory-pool (23) → build-allocator (39)

# Container Track
container-basic (7) → build-docker (38)

# Distributed Track
gossip (20) → distributed-cache (21) → build-raft (22)

# ML Track
neural-network (31) → transformer (32) → gpu-compute (33) → inference-engine (35) → distributed-training (36)

# Performance Track
hash-impl (2) → simd-library (24) → cache-optimized (25) → zero-copy-msg-bus (30)
```

---

## Milestone Checkpoints

| Done | Level | Can Apply |
|------|-------|-----------|
| Phase 1-2 (10) | Mid-level | Backend @ startup/mid-size |
| Phase 1-4 (19) | Senior | Senior backend, infra engineer |
| Phase 1-6 (30) | Senior @ Big Tech | Google L5, Meta E5, systems engineer |
| Phase 1-7 (36) | Senior + ML Infra | NVIDIA, ML infra @ big tech |
| Full 40 | Staff-level knowledge | Staff/Principal interviews, HFT |

---

## Lưu Ý

1. **Không rush** — 12-16 tháng là thực tế, 6 tháng là ảo tưởng
2. **Viết README** — mỗi project cần giải thích design decisions
3. **Benchmark** — đo performance, không nói suông
4. **Push GitHub** — portfolio quan trọng hơn certificate
5. **Làm sâu 1 project > làm hời 3 project**

---

*Last updated: Feb 2026*

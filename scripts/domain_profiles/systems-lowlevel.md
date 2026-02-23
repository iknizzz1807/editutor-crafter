# DOMAIN PROFILE: Systems & Low-Level Programming
# Applies to: systems, performance-engineering
# Projects: build-os, build-tcp, build-allocator, SIMD, io_uring, lock-free, DMA, etc.

## Fundamental Tension Type
Physical and hardware constraints. SOFTWARE wants infinite memory, instant I/O, unlimited parallelism. HARDWARE provides fixed pages, disk latency, cache lines, pipeline stalls. Every design decision negotiates with physics.

Examples:
- "CPU addresses 4KB pages, programs need 3B–3GB → allocator bridges the gap"
- "Disk reads cost 10ms, memory 100ns → 100,000× gap forces caching and prefetching"
- "CPU pipeline predicts branches — misprediction costs 15 cycles → branch-free code matters"

## Three-Level View
- **Level 1 — Application**: API/function call
- **Level 2 — OS/Kernel**: Syscalls, page tables, scheduler, interrupts
- **Level 3 — Hardware**: CPU pipeline, cache hierarchy (L1/L2/L3), TLB, DMA, memory bus, branch predictor

MANDATORY for every non-trivial operation.

## Soul Section: "Hardware Soul"
For every major operation:
- Which cache lines touched? Hot or cold?
- Branch predictable? Misprediction cost?
- TLB miss? Page fault? Context switch?
- SIMD opportunity? Vectorization?
- Memory access pattern? Sequential (prefetch-friendly) or random (cache-hostile)?

## Alternative Reality Comparisons
Linux kernel, FreeBSD, Redis, jemalloc/mimalloc/tcmalloc, io_uring, DPDK, SQLite internals, Go runtime, Rust std.

## TDD Emphasis
- Memory layout (byte offsets): MANDATORY for every struct
- Cache line analysis (64B): MANDATORY
- Lock ordering / concurrency: MANDATORY if multi-threaded
- Syscall list: specify exact syscalls and why
- Benchmarks: latency ns/μs, throughput ops/sec, memory bytes

## Cross-Domain Notes
Other domains may need to borrow from this profile when they involve:
- Process management (fork/exec/waitpid) → DevOps CI runners, container runtimes
- Network I/O (epoll/kqueue, TCP buffers) → web servers, game networking
- Memory-mapped I/O → database storage engines
- GPU pipeline → game rendering, ML training infrastructure



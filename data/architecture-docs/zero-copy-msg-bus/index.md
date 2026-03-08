# 🎯 Project Charter: Zero-Copy Message Bus
## What You Are Building
A high-throughput inter-process messaging system that achieves sub-microsecond latency by eliminating data copies entirely. Processes communicate through shared memory regions with lock-free ring buffers, passing flat buffer serialized messages that are read directly from memory without parsing. By the end, you'll have a production-grade message bus supporting multiple producers and consumers, topic-based pub/sub routing with MQTT-style wildcards, and crash recovery with optional durability—all achieving 5-20 million messages per second with median latency under 500 nanoseconds.
## Why This Project Exists
Traditional IPC mechanisms—pipes, sockets, message queues—require the kernel to copy data between address spaces, adding 100-500ns of latency per copy. At 2 million messages per second with multiple subscribers, you're burning gigabytes per second of memory bandwidth just on copies. This project teaches the techniques used in high-frequency trading platforms, real-time analytics pipelines, and low-latency microservices where every nanosecond counts. Engineers with zero-copy IPC and lock-free programming skills command $250K-500K+ at trading firms, cloud providers, and real-time systems companies.
## What You Will Be Able to Do When Done
- Create shared memory regions mapped into multiple processes for zero-copy data transfer
- Implement lock-free SPSC and MPMC ring buffers with proper memory barriers for cross-process visibility
- Design flat buffer serialization formats enabling direct field access via pointer arithmetic
- Handle cache line alignment and false sharing to achieve predictable sub-microsecond latency
- Build trie-based topic routers supporting MQTT-style wildcards (`+` and `#`) with O(M) matching
- Implement reference-counted message fan-out for zero-copy publish-subscribe
- Design crash detection and orphaned slot recovery mechanisms that survive process failures
- Apply write-ahead logging and checkpointing for durability without sacrificing latency
- Benchmark and optimize for tail latency, measuring P99 and P99.9 percentiles
## Final Deliverable
~4,000 lines of C++ across 40+ source files implementing a complete message bus. The system boots in milliseconds, runs in shared memory with no per-message kernel involvement, supports 5+ publishers and 10+ subscribers simultaneously, handles 10+ million messages per second with median latency under 300ns, and recovers from process crashes in under 100ms. Includes a schema compiler generating type-safe C++ accessors, benchmark suite measuring latency percentiles, and integration tests simulating crash scenarios.
## Is This Project For You?
**You should start this if you:**
- Are comfortable with C++ systems programming (pointers, memory management, RAII)
- Understand virtual memory concepts (pages, mapping, address spaces)
- Have encountered atomic operations and concurrent programming before
- Want to understand how high-performance systems achieve their performance
**Come back after you've learned:**
- C++ move semantics and smart pointers (essential for memory safety)
- Basic lock-free programming concepts (compare-and-swap, memory ordering)
- Process isolation and IPC mechanisms (pipes, sockets, shared memory)
- CPU cache hierarchy and cache coherency (MESI protocol basics)
## Estimated Effort
| Phase | Time |
|-------|------|
| Shared Memory Ring Buffer (SPSC) | ~12-14 hours |
| Zero-Copy Serialization | ~14-16 hours |
| Multi-Producer Multi-Consumer | ~10-14 hours |
| Publish-Subscribe & Topics | ~10-12 hours |
| Crash Recovery & Durability | ~10-12 hours |
| **Total** | **~56-68 hours** |
## Definition of Done
The project is complete when:
- Ring buffer achieves sub-microsecond round-trip latency between producer and consumer processes
- All unit tests pass including wraparound, contention, and crash simulation scenarios
- Cross-process tests demonstrate correct message delivery with no data races (ThreadSanitizer clean)
- Zero-copy serialization benchmark shows 10x+ improvement over JSON parsing for field access
- MPMC queue handles 4+ producers with CAS failure rate under 50% and no message loss
- Pub/sub wildcard matching completes in under 500ns for typical topic depths
- Crash recovery restores system state in under 100ms from checkpoint plus WAL replay
- Benchmark suite reports P50, P99, and P999 latency percentiles with throughput measurements

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Foundation: Operating Systems & Computer Architecture
### Memory Models and Atomics
**Read BEFORE starting this project — required foundational knowledge.**
| Resource | Type | Why It's Gold Standard |
|----------|------|----------------------|
| **"C++ Atomics, From the Metal Up"** — David Olsen (CppCon 2023) | Video | Best visual explanation of how atomic operations map to x86/ARM instructions, release/acquire semantics, and why fences matter |
| **C++20 Standard, §7.6.9 (memory_order)** | Spec | The authoritative definition of the memory model you'll be programming against |
| **std::atomic documentation** — cppreference.com | Reference | Practical lookup for which operations provide which guarantees |
**Key insight to carry forward:** `std::atomic` with `memory_order` guarantees visibility between *threads within the same process*. Cross-process shared memory requires additional barriers.
---
### Cache Coherence and False Sharing
**Read BEFORE starting Milestone 1 — explains why `alignas(64)` matters.**
| Resource | Type | Section | Why Gold Standard |
|----------|------|---------|------------------|
| **"What Every Programmer Should Know About Memory"** — Ulrich Drepper (2007) | Paper | Section 3 (CPU Caches) | The foundational reference on cache hierarchy, MESI protocol, and why cache lines matter |
| **Intel 64 and IA-32 Architectures Optimization Reference Manual** | Spec | Chapter 2 (CPU Microarchitecture) | Official documentation of cache line sizes, prefetch behavior |
| **"False Sharing"** — Jeff Preshing (blog) | Blog | Full article | Clear examples of cache line ping-pong and the `alignas` fix |
**Read after Milestone 1 (Shared Memory Ring Buffer):** You'll have experienced the latency degradation from false sharing and will appreciate the fix.
---
## Milestone 1: Shared Memory Ring Buffer
### POSIX Shared Memory APIs
**Read BEFORE implementing M1 — the syscall layer you'll wrap.**
| Resource | Type | Why Gold Standard |
|----------|------|------------------|
| **Linux `shm_open(3)` man page** | Spec | Complete API documentation with all flags and error codes |
| **Linux `mmap(2)` man page** | Spec | Critical: explains `MAP_SHARED` vs `MAP_PRIVATE`, protection flags, and page alignment |
| **"Shared Memory: An Introduction"** — The Linux Programming Interface (Kerrisk), Chapter 48 | Book | Step-by-step walkthrough of shm lifecycle with code examples |
### Lock-Free Ring Buffers
**Read after Milestone 1 implementation — compare your approach to industry standards.**
| Resource | Type | Specific Reference | Why Gold Standard |
|----------|------|-------------------|------------------|
| **LMAX Disruptor Technical Paper** — Martin Thompson | Paper | Section 2 (Ring Buffer Design) | The production system that popularized this pattern, 6M+ events/sec |
| **"Writing a Lock-Free SPSC Queue"** — Dmitry Vyukov (code) | Code | `spsc_queue.h` in libcds | Reference implementation used in production systems worldwide |
---
## Milestone 2: Zero-Copy Serialization
### FlatBuffers and Zero-Copy Design
**Read BEFORE Milestone 2 — understand the format you're implementing.**
| Resource | Type | Section/Time | Why Gold Standard |
|----------|------|--------------|------------------|
| **FlatBuffers Internals** — flatbuffers.dev | Documentation | "FlatBuffers Internals" section | Official explanation of vtables, offset encoding, schema evolution |
| **"Cap'n Proto: The Story of a Protocol"** — Kenton Varda (CppCon 2014) | Video | 15:00–25:00 | Explains why zero-copy requires thinking about serialization differently |
| **"Cap'n Proto Encoding Format"** — capnproto.org | Spec | Full specification | More detailed than FlatBuffers docs on pointer encoding |
### Schema Evolution Patterns
**Read after Milestone 2 implementation — you'll have context to appreciate the tradeoffs.**
| Resource | Type | Why Gold Standard |
|----------|------|------------------|
| **Protocol Buffers Language Guide — "Updating A Message Type"** | Spec | The canonical rules for backward/forward compatibility |
| **"Schema Evolution in Avro, Protocol Buffers, and Thrift"** — Martin Kleppmann (blog) | Blog | Comparative analysis of evolution strategies across formats |
---
## Milestone 3: Multi-Producer Multi-Consumer
### The Vyukov MPMC Algorithm
**Read BEFORE implementing M3 — this is the algorithm you'll implement.**
| Resource | Type | Specific Reference | Why Gold Standard |
|----------|------|-------------------|------------------|
| **"Bounded MPMC Queue"** — Dmitry Vyukov (code) | Code | `mpmc_queue.h` in libcds | The original implementation you're porting to shared memory |
| **"Correct Implementation of Vyukov MPMC"** — Stack Overflow | Discussion | Question and accepted answer | Clarifies the sequence number state machine that prevents ABA |
### The ABA Problem and Solutions
**Read BEFORE M3 — prevents subtle bugs that only manifest under load.**
| Resource | Type | Section | Why Gold Standard |
|----------|------|---------|------------------|
| **"ABA Problem"** — Wikipedia | Article | Full article | Clear explanation with concrete examples |
| **"Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"** — Maged Michael (2004) | Paper | Sections 1-3 | The classic solution; understand why epoch-based reclamation is simpler for your case |
| **"Epoch-Based Reclamation"** — Preshing on Programming (blog) | Blog | Full article | Practical implementation guidance |
### Contention and Backoff Strategies
**Read after Milestone 3 benchmarks — you'll have CAS failure data to analyze.**
| Resource | Type | Why Gold Standard |
|----------|------|------------------|
| **"PAUSE Instruction"** — Agner Fog's Optimization Manual | Documentation | Explains why `pause` is ~5 cycles on Skylake and why it matters for spin loops |
| **"Futexes Are Tricky"** — Ulrich Drepper (paper) | Paper | When spinning becomes blocking — understanding `futex(2)` |
---
## Milestone 4: Publish-Subscribe & Topics
### Trie-Based Topic Matching
**Read BEFORE implementing M4 — understand the data structure for O(M) matching.**
| Resource | Type | Section | Why Gold Standard |
|----------|------|---------|------------------|
| **"Trie"** — Wikipedia | Article | Full article | Standard definition and complexity analysis |
| **"MQTT Specification v5.0"** — OASIS | Spec | Sections 4.7 (Topic Names and Filters) | The industry-standard wildcard semantics you're implementing |
| **EMQX Documentation — "Topic and Subscribe"** | Documentation | Real-world MQTT broker topic matching behavior |
### Bloom Filters for Fast Rejection
**Read during M4 — implement after basic pub/sub works.**
| Resource | Type | Why Gold Standard |
|----------|------|------------------|
| **"Bloom Filter"** — Wikipedia | Article | Clear explanation of the math behind false positive rates |
| **"Bloom Filters by Example"** — billmill.org | Interactive | Visualize how bits are set and checked |
---
## Milestone 5: Crash Recovery & Durability
### Write-Ahead Logging
**Read BEFORE implementing M5 — the foundational durability technique.**
| Resource | Type | Section | Why Gold Standard |
|----------|------|---------|------------------|
| **"Write-Ahead Logging"** — SQLite Documentation | Documentation | Full WAL explanation from a production database |
| **PostgreSQL Documentation — "Write-Ahead Logging (WAL)"** | Documentation | Explains checkpoint + WAL interaction for bounded recovery |
| **"ARIES: A Transaction Recovery Method"** — C. Mohan et al. (1992) | Paper | Sections 2-3 | The academic foundation; understand the redo/undo distinction |
### Checkpointing Strategies
**Read after basic WAL works — you'll need this for sub-100ms recovery.**
| Resource | Type | Why Gold Standard |
|----------|------|------------------|
| **"Designing Data-Intensive Applications"** — Martin Kleppmann | Book | Chapter 3 (Storage and Retrieval), "Making LSM-trees faster with storage" section | Explains compaction strategies that apply to checkpoint + log design |
| **LMDB Documentation** — Symas | Documentation | Memory-mapped B-tree with copy-on-write snapshots |
### At-Least-Once vs Exactly-Once Semantics
**Read BEFORE M5 — understand what you can and cannot guarantee.**
| Resource | Type | Section | Why Gold Standard |
|----------|------|---------|------------------|
| **"Exactly-Once Semantics is Hard"** — Jay Kreps (blog) | Blog | Full article | Kafka co-creator explains why exactly-once is fundamentally hard |
| **"Idempotent Producers"** — Kafka Documentation | Documentation | How Kafka implements effective exactly-once via idempotence |
---
## Cross-Cutting: High-Frequency Trading Systems
**Read AFTER completing all milestones — context for why these techniques matter.**
| Resource | Type | Section | Why Gold Standard |
|----------|------|---------|------------------|
| **"The LMAX Architecture"** — Martin Thompson (QCon 2011) | Video | Full talk | The production trading system that proved these techniques at scale |
| **"How I Hacked the FAST Algorithm"** — Peter Lawrey (blog) | Blog | Chronicles of a HFT developer | Real-world war stories from the same domain |
| **"Mechanical Sympathy"** — Martin Thompson (mailing list) | Community | Archives | Ongoing discussion of low-latency techniques |
---
## Summary Reading Order
```
BEFORE PROJECT START:
  1. C++ Atomics (Olsen video)
  2. Drepper's "What Every Programmer Should Know About Memory" (Section 3)
  3. POSIX shm_open/mmap man pages
BEFORE M1:
  4. False Sharing (Preshing)
AFTER M1:
  5. LMAX Disruptor paper
BEFORE M2:
  6. FlatBuffers Internals documentation
  7. Cap'n Proto video (Varda)
AFTER M2:
  8. Protocol Buffers schema evolution docs
BEFORE M3:
  9. Vyukov MPMC queue source
  10. ABA problem (Wikipedia + Michael paper)
AFTER M3:
  11. Futexes paper (Drepper)
BEFORE M4:
  12. MQTT v5.0 spec (Topic Names section)
  13. Trie (Wikipedia)
DURING M4:
  14. Bloom Filters by Example
BEFORE M5:
  15. SQLite WAL documentation
  16. Exactly-Once Semantics (Kreps blog)
AFTER M5:
  17. LMAX Architecture video
  18. DDIA Chapter 3 (Kleppmann)
```

---

# Zero-Copy Message Bus

A high-throughput inter-process messaging system that eliminates data copying by leveraging shared memory regions, lock-free ring buffers, and flat buffer serialization. Processes communicate by reading and writing directly to shared memory segments, achieving sub-microsecond latency that traditional IPC mechanisms (pipes, sockets, message queues) cannot match. This project teaches the fundamental techniques used in high-frequency trading platforms, real-time analytics pipelines, and low-latency microservices.



<!-- MS_ID: zcmb-m1 -->
# Milestone 1: Shared Memory Ring Buffer
## The Problem: When memcpy() Becomes Your Bottleneck
You're building a trading system. Market data arrives at 2 million messages per second. Each message is 256 bytes. Your process receives data from a network handler process and must act on it within microseconds.
The traditional approach: the network handler writes to a pipe or socket. Your process reads from it. The kernel copies the data from the network handler's address space into kernel buffers, then copies it again into your process's address space.
```
Network Handler → [copy to kernel] → [copy to your process] → You process it
                   ~100-500ns         ~100-500ns
```
At 2 million messages × 256 bytes × 2 copies, you're moving **~1 GB/s** through the kernel's buffer cache just for copying. Each copy burns CPU cycles, pollutes cache lines, and adds latency. The memcpy itself isn't even the worst part—it's the cache pollution from touching the same data twice, the context switches into the kernel, the scheduler decisions about which process runs next.
**The tension**: You need to move data between processes. The operating system provides isolation—each process has its own address space, its own virtual memory mappings, its own view of the world. This isolation is fundamental to system stability. But isolation means copying, and copying means latency.
**The escape hatch**: What if two processes could look at the *same* memory? The network handler writes, you read—no copies, no kernel involvement after setup. The CPU's virtual memory system already supports this: map the same physical pages into multiple processes' address spaces. The hardware handles the translation; the processes see the same bytes.
This is shared memory IPC, and when done right, it achieves **sub-microsecond** latency between processes. Not by being clever about copying, but by eliminating the copy entirely.
---
## The Architecture: Ring Buffer in Shared Memory

![System Satellite Map: Zero-Copy Message Bus](./diagrams/diag-global-001.svg)

The ring buffer (also called a circular buffer) is your fundamental data structure. It's a fixed-size array where the producer writes and the consumer reads, with two indices that wrap around when they reach the end:
```
[slot 0][slot 1][slot 2][slot 3][slot 4][slot 5][slot 6][slot 7]
   ↑                                                    ↑
 tail (consumer reads here)                    head (producer writes here)
```
When the producer fills slot 7, it wraps to slot 0. When the consumer reads slot 7, it wraps to slot 0. The buffer never fills completely—we always leave one slot empty so we can distinguish "full" from "empty" by comparing head and tail indices.
### Why Power-of-Two Sizes?
You'll notice our buffer has 8 slots (a power of two). This isn't arbitrary. When the producer needs to wrap from slot 7 to slot 0, the naive approach uses modulo:
```cpp
size_t next_slot = (head + 1) % buffer_size;  // Division instruction
```
Division is slow—20-80 cycles on modern CPUs. But if `buffer_size` is a power of two, we can use bitwise AND instead:
```cpp
size_t next_slot = (head + 1) & (buffer_size - 1);  // Single AND instruction
```
This single-instruction operation matters when you're processing millions of messages per second. A 20-cycle division at 2 million messages/second burns 40 million cycles per second—about 1.3% of a 3 GHz CPU core, just on modulo operations.
**Rule**: Always use power-of-two buffer sizes. The mask `(buffer_size - 1)` gives you the wraparound for free.
---
## The Three-Level View: How Shared Memory Actually Works

![Memory Map: Shared Memory Layout](./diagrams/diag-global-003.svg)

### Level 1 — Application
From your code's perspective, shared memory looks like a pointer:
```cpp
void* shared_mem = /* some magic setup */;
RingBuffer* ring = static_cast<RingBuffer*>(shared_mem);
ring->head = 0;  // Just writing to memory
```
You read and write it like any other memory. No special API calls per message.
### Level 2 — OS/Kernel
The setup requires kernel involvement:
1. **Create** a shared memory object (a special file-like thing that lives in memory, not on disk)
2. **Size** it appropriately
3. **Map** it into each process's address space with `mmap()`
After setup, the kernel is out of the picture. Reading and writing require no syscalls.
On Linux, you have two APIs for shared memory:
| API | Function | Cleanup | Portability |
|-----|----------|---------|-------------|
| **POSIX shm** | `shm_open()` + `mmap()` | `shm_unlink()` | POSIX systems |
| **System V shm** | `shmget()` + `shmat()` | `shmctl(IPC_RMID)` | Unix legacy |
We'll use POSIX shared memory—it's cleaner, more modern, and integrates with the file descriptor ecosystem.
### Level 3 — Hardware
Here's where it gets interesting. When you map shared memory, the kernel allocates physical pages and creates page table entries in *each* process that maps them.
```
Process A's page table:
  Virtual 0x7f0000000000 → Physical Page 0x12345000
Process B's page table:
  Virtual 0x7f0001000000 → Physical Page 0x12345000  (same physical page!)
```
When Process A writes to its virtual address, the CPU's Memory Management Unit (MMU) translates to the physical address. The store goes to physical memory (and L1/L2/L3 cache). When Process B reads from *its* virtual address, the MMU translates to the *same* physical address—the data is already there.
No copies. The hardware did the translation, and both processes see the same bytes.
---
## Cross-Process Visibility: The Trap That Awaits

> **🔑 Foundation: Cross-process memory visibility differs from thread visibility**
> 
> ## What It Is
**Thread visibility** concerns whether one thread's writes to memory become observable by another thread *within the same process*. The memory model guarantees certain ordering and visibility rules (e.g., a release store in Thread A is visible to an acquire load in Thread B).
**Cross-process visibility** is fundamentally different. When two *separate processes* share memory (via mmap, shared memory segments, or memory-mapped files), they have:
- Separate virtual address spaces (the same physical page mapped to different virtual addresses)
- Separate TLB entries
- Potentially different CPU caches (per-core L1/L2) that don't automatically snoop each other
- No shared "happens-before" relationship as defined by language memory models
The key distinction: **language-level memory models (C++, Java, Rust) do not apply across process boundaries.** They only govern threads within a single process.
## Why You Need This Now
If you're building shared-memory IPC, lock-free queues between processes, or persistent memory-mapped data structures, you cannot rely on `std::atomic` or `volatile` alone to guarantee visibility. A store performed by Process A may sit in a store buffer or cache indefinitely from Process B's perspective.
The operating system and hardware provide the synchronization primitives you actually need:
- **Memory barriers** (`mfence`, `dmb`, `sync`) to order operations
- **Cache line flushes** (`clflush`, `clwb`) to write back to coherent memory
- **TLB shootdowns** (handled by OS) when mappings change
- **Explicit fences** that cross process boundaries (e.g., `sys_membarrier` on Linux)
## Key Insight
> **Same code, same CPU, same physical memory — different visibility guarantees.**
A `std::atomic<int>` with `memory_order_seq_cst` guarantees visibility between threads in Process A, but Process B reading the same physical location has *no such guarantee*. The language runtime has no knowledge of cross-process sharing. You must use OS-level or hardware-level primitives that operate at the level of physical memory and cache coherence, not virtual address spaces.
---
**Example:** Two processes share a memory-mapped file at address `0x7f000000` (Process A) and `0x7f001000` (Process B), both mapping the same underlying file page.
```cpp
// Process A
shared_data->flag.store(1, std::memory_order_release);  // Visible to Process A's threads only!
// Process B — may NEVER see this write without additional synchronization
if (shared_data->flag.load(std::memory_order_acquire) == 1) { /* NOT GUARANTEED */ }
```
The fix requires a **full memory barrier** that forces store buffer drain and cache visibility:
```cpp
// Process A
shared_data->flag = 1;
__sync_synchronize();  // Full hardware fence (GCC/Clang intrinsic)
// Process B
__sync_synchronize();
if (shared_data->flag == 1) { /* Now guaranteed */ }
```

Here's the misconception that ruins shared memory projects: "I'll just use `std::atomic` and it'll work across processes, same as threads."
This is *almost* true, which makes it *especially* dangerous.
### The Problem with Compiler Optimizations
Consider this code:
```cpp
// In shared memory
struct RingBuffer {
    std::atomic<uint64_t> head;
    std::atomic<uint64_t> tail;
    uint8_t data[BUFFER_SIZE][MSG_SIZE];
};
// Producer writes a message
void produce(RingBuffer* ring, const uint8_t* msg) {
    uint64_t slot = ring->head.fetch_add(1, std::memory_order_relaxed);
    memcpy(ring->data[slot], msg, MSG_SIZE);  // Copy the message
    // How does the consumer know the data is ready?
}
```
The producer copies the message data, then... what? The consumer needs to know the message is ready. A common approach:
```cpp
// WRONG - does not work correctly across processes
void produce_wrong(RingBuffer* ring, const uint8_t* msg) {
    uint64_t slot = ring->head.fetch_add(1, std::memory_order_relaxed);
    memcpy(ring->data[slot], msg, MSG_SIZE);
    ring->head.store(slot + 1, std::memory_order_release);  // Signal ready
}
```
The issue: the compiler is allowed to reorder the `memcpy` and the `store` because they operate on different memory locations. The compiler doesn't know another *process* is watching this memory. It assumes single-threaded semantics for non-atomic operations.

> **🔑 Foundation: Memory barriers and their mapping to CPU instructions**
> 
> ## What It Is
A **memory barrier** (also called a memory fence) is a CPU instruction that enforces ordering constraints on memory operations. Without barriers, CPUs and compilers are free to reorder loads and stores for performance — as long as the *single-threaded* behavior appears correct.
The four fundamental barrier types:
| Barrier Type | Prevents |
|--------------|----------|
| **LoadLoad** | Load A → Load B (B cannot execute before A completes) |
| **StoreStore** | Store A → Store B (B cannot write before A is visible) |
| **LoadStore** | Load A → Store B (Store cannot execute before Load completes) |
| **StoreLoad** | Store A → Load B (Load cannot execute before Store is visible) |
The **StoreLoad** barrier is the most expensive — it typically requires a full pipeline stall and store buffer drain.
### CPU Instruction Mapping
| Architecture | Full Fence | Acquire | Release | Notes |
|--------------|------------|---------|---------|-------|
| **x86/x64** | `mfence` | (implicit) | (implicit) | x86 is strongly ordered; only StoreLoad needs explicit fencing |
| **ARM64** | `dmb ish` | `ldar` (load-acquire) | `stlr` (store-release) | Weakly ordered; explicit barriers needed |
| **POWER** | `sync` | `lwsync` + `isync` | `lwsync` | Very weak; multiple barrier flavors |
| **RISC-V** | `fence rw,rw` | `fence r,rw` | `fence rw,w` | Explicit fence encoding |
## Why You Need This Now
When implementing lock-free data structures or cross-process communication, you're writing code that must run correctly on multiple architectures. A barrier that works on x86 may be a no-op there but critical on ARM.
Consider a simple producer-consumer flag:
```cpp
data = 42;
ready = 1;  // Must become visible AFTER data
```
On x86: Stores are ordered (no reordering of stores), so this naturally works.
On ARM: The CPU may reorder and write `ready` before `data` — the consumer sees garbage.
The **release store** pattern solves this:
```cpp
// Producer
data = 42;
atomic_store_explicit(&ready, 1, memory_order_release);  // ARM: stlr (store-release)
// Consumer
while (atomic_load_explicit(&ready, memory_order_acquire) != 1);  // ARM: ldar (load-acquire)
use(data);  // Guaranteed to see 42
```
## Key Insight
> **Barriers don't make operations faster — they make the CPU wait.** They enforce correctness by *preventing* optimizations.
The mental model: **Barriers are synchronization points in time.**
- **Acquire**: "I'm entering a critical section — I must see all writes that happened before the corresponding release."
- **Release**: "I'm leaving a critical section — all my writes must be visible before I release the lock."
Think of release as "pushing" your writes out of the store buffer, and acquire as "pulling" those writes into your cache. They form a pair that establishes a **happens-before** relationship.
---
**x86 subtlety:** Even though x86 preserves StoreStore and LoadLoad ordering by default, **StoreLoad** can still be reordered. This is why `mfence` exists:
```cpp
// Broken on x86 without mfence
x = 1;          // Store
r = y;          // Load — may execute BEFORE x=1 becomes visible!
```
SeqCst (sequentially consistent) atomics insert `mfence` (or use `lock`-prefixed instructions) precisely to prevent this.

### The Correct Pattern: Head/Tail Updates as Signals
The cleanest design uses separate `head` (producer writes) and `tail` (consumer writes) indices:
```cpp
struct RingBuffer {
    alignas(64) std::atomic<uint64_t> head;  // Producer only writes this
    alignas(64) std::atomic<uint64_t> tail;  // Consumer only writes this
    uint8_t slots[BUFFER_SIZE][MSG_SIZE];
};
```
**Producer**:
```cpp
void produce(RingBuffer* ring, const uint8_t* msg, size_t msg_size) {
    uint64_t pos = ring->head.load(std::memory_order_relaxed);
    uint64_t next = (pos + 1) & BUFFER_MASK;
    // Wait if buffer is full
    while (next == ring->tail.load(std::memory_order_acquire)) {
        // Buffer full - spin or back off
        std::this_thread::yield();
    }
    // Copy data BEFORE updating head
    memcpy(ring->slots[pos], msg, std::min(msg_size, MSG_SIZE));
    // Release fence ensures the memcpy completes before head update is visible
    std::atomic_thread_fence(std::memory_order_release);
    ring->head.store(next, std::memory_order_relaxed);
}
```
**Consumer**:
```cpp
bool consume(RingBuffer* ring, uint8_t* out_msg, size_t out_size) {
    uint64_t pos = ring->tail.load(std::memory_order_relaxed);
    // Check if buffer is empty
    if (pos == ring->head.load(std::memory_order_acquire)) {
        return false;  // Empty
    }
    // Acquire fence ensures we see the data after head was updated
    std::atomic_thread_fence(std::memory_order_acquire);
    // Copy data BEFORE updating tail
    memcpy(out_msg, ring->slots[pos], std::min(out_size, MSG_SIZE));
    uint64_t next = (pos + 1) & BUFFER_MASK;
    ring->tail.store(next, std::memory_order_relaxed);
    return true;
}
```
The key insight: **`head` and `tail` serve dual purposes**. They track position, but they also serve as synchronization points. When the producer updates `head`, the consumer's `load(acquire)` of `head` ensures all prior writes (the message data) are visible.
---
## False Sharing: The Performance Killer

> **🔑 Foundation: False sharing and cache line alignment**
> 
> ## What It Is
**False sharing** occurs when multiple threads access *different variables* that happen to reside on the same cache line. Even though the threads aren't sharing data logically, the hardware treats them as contending for the same cache line because **cache coherence operates at cache-line granularity** (typically 64 bytes on modern CPUs).
When Thread A writes to `variable_x` and Thread B writes to `variable_y`, if both variables are within the same 64-byte cache line:
1. Thread A's core acquires exclusive ownership of the cache line
2. Thread B's core must invalidate its copy and re-fetch the line
3. This ping-pong continues — **cache thrashing** — even though no actual data is shared
The result: Performance degrades to the speed of a single thread (or worse), despite perfect parallelization in the code logic.
## Why You Need This Now
False sharing is a **silent performance killer**. Your code is correct, thread-safe, and logically parallel — but it runs slower than the single-threaded version. Profilers may show high cache miss rates or unexpected stalls.
Common victims:
- Adjacent elements in arrays accessed by different threads
- Fields in a struct/class updated by different threads
- Counters, statistics, or progress indicators per thread, declared consecutively
- Lock-free queue head and tail pointers
**Detection:** If `perf` shows high `HITM` (Hit Modified) rates in the cache coherency counter, or if adding `padding` between variables dramatically improves performance, you've found false sharing.
## Key Insight
> **The CPU doesn't know or care about your variable boundaries. It only knows about 64-byte cache lines.**
Two variables at addresses `0x1000` and `0x1008` are in the same cache line. Two variables at `0x1000` and `0x1040` are in different cache lines (assuming 64-byte lines). The mental model: **Coherence granularity ≠ logical granularity.**
### The Fix: Cache Line Alignment
Pad or align variables so each thread's hot data occupies its own cache line:
```cpp
// BEFORE: Potential false sharing
struct Counters {
    std::atomic<uint64_t> messages_sent;
    std::atomic<uint64_t> messages_received;
};
// AFTER: Cache-line aligned
struct alignas(64) Counters {
    std::atomic<uint64_t> messages_sent;
    char padding[56];  // 64 - 8 = 56 bytes
    std::atomic<uint64_t> messages_received;
};
```
Or using C++17:
```cpp
struct Counters {
    alignas(64) std::atomic<uint64_t> messages_sent;
    alignas(64) std::atomic<uint64_t> messages_received;
};
```
**Practical example:** A ring buffer with head (producer) and tail (consumer) indices:
```cpp
// BAD: head and tail likely share a cache line
struct RingBuffer {
    std::atomic<size_t> head;  // Producer writes
    std::atomic<size_t> tail;  // Consumer writes
    void* data[SIZE];
};
// GOOD: head and tail on separate cache lines
struct RingBuffer {
    alignas(64) std::atomic<size_t> head;  // Offset 0
    alignas(64) std::atomic<size_t> tail;  // Offset 64
    void* data[SIZE];
};
```
The producer and consumer can now operate in parallel without cache coherency traffic between them.

Look at this struct:
```cpp
struct RingBuffer {
    std::atomic<uint64_t> head;  // 8 bytes
    std::atomic<uint64_t> tail;  // 8 bytes
    uint8_t slots[BUFFER_SIZE][MSG_SIZE];
};
```
`head` is at offset 0. `tail` is at offset 8. A typical cache line is **64 bytes**. Both `head` and `tail` fit on the same cache line.
Here's the disaster scenario:
1. Producer (CPU core 0) writes to `head` → acquires exclusive ownership of cache line
2. Consumer (CPU core 1) reads `head` to check for messages → cache miss, must fetch from core 0
3. Consumer writes to `tail` → needs exclusive ownership, invalidates core 0's cache line
4. Producer reads `tail` to check for space → cache miss, must fetch from core 1
This is **cache line ping-pong**. Every operation on either index causes the cache line to bounce between cores. On a multi-socket system, this can require inter-socket traffic over QPI or UPI links, adding 100+ nanoseconds per bounce.

![Hardware Soul: Cache Line Movement](./diagrams/diag-global-004.svg)

### The Fix: Cache Line Alignment
```cpp
struct RingBuffer {
    alignas(64) std::atomic<uint64_t> head;  // Offset 0, occupies bytes 0-63
    alignas(64) std::atomic<uint64_t> tail;  // Offset 64, occupies bytes 64-127
    // ... rest of struct
};
```
Now `head` and `tail` are on separate cache lines. The producer writes `head`—that cache line stays hot in the producer's L1. The consumer writes `tail`—that cache line stays hot in the consumer's L1. The only cross-core traffic is reading the *other* process's index, which happens once per batch of messages.
**Performance impact**: False sharing can degrade performance by **10-100x** in tight producer-consumer loops. I've seen a ring buffer go from 200ns latency to 20μs just from false sharing on a dual-socket system.
---
## Implementation: Creating Shared Memory
Let's build this properly. First, the shared memory setup:
```cpp
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
class SharedMemory {
public:
    static constexpr size_t PAGE_SIZE = 4096;
    SharedMemory(const std::string& name, size_t size, bool create)
        : name_(name), size_(align_to_page(size)), is_creator_(create)
    {
        if (create) {
            // Create new shared memory object
            fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
            if (fd_ == -1) {
                throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
            }
            // Set size
            if (ftruncate(fd_, size_) == -1) {
                ::close(fd_);
                shm_unlink(name_.c_str());
                throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
            }
        } else {
            // Open existing shared memory
            fd_ = shm_open(name_.c_str(), O_RDWR, 0666);
            if (fd_ == -1) {
                throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
            }
        }
        // Map into address space
        void* addr = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr == MAP_FAILED) {
            ::close(fd_);
            if (create) shm_unlink(name_.c_str());
            throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
        }
        data_ = static_cast<uint8_t*>(addr);
    }
    ~SharedMemory() {
        if (data_) {
            munmap(data_, size_);
        }
        if (fd_ != -1) {
            ::close(fd_);
        }
        if (is_creator_) {
            shm_unlink(name_.c_str());  // Remove when creator destroys
        }
    }
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
private:
    static size_t align_to_page(size_t size) {
        return (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    }
    std::string name_;
    size_t size_;
    int fd_ = -1;
    uint8_t* data_ = nullptr;
    bool is_creator_;
};
```
### Why `MAP_SHARED` vs `MAP_PRIVATE`?
The `MAP_SHARED` flag is critical here. It tells the kernel:
1. **Updates are visible to other processes** mapping the same object
2. **Updates go directly to the underlying shared memory**, not to a copy-on-write private copy
With `MAP_PRIVATE`, each process gets its own copy-on-write version. Writes are *not* visible to other processes. This is useful for loading executables (each process can modify its data segment without affecting others), but defeats the entire purpose of shared memory IPC.
---
## Implementation: The SPSC Ring Buffer
Now the ring buffer itself:
```cpp
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <cstddef>
// Configuration for the ring buffer
struct RingBufferConfig {
    size_t num_slots;      // Must be power of 2
    size_t slot_size;      // Size of each message slot
};
class SpscRingBuffer {
public:
    // Header stored at the beginning of shared memory
    struct alignas(64) Header {
        alignas(64) std::atomic<uint64_t> head{0};   // Producer writes, consumer reads
        alignas(64) std::atomic<uint64_t> tail{0};   // Consumer writes, producer reads
        uint64_t num_slots{0};
        uint64_t slot_size{0};
        uint64_t mask{0};  // num_slots - 1 for fast modulo
        uint8_t padding[64 - 5 * sizeof(uint64_t)];  // Pad to cache line
    };
    static constexpr size_t HEADER_SIZE = sizeof(Header);
    // Calculate total shared memory size needed
    static size_t calculate_size(const RingBufferConfig& config) {
        return HEADER_SIZE + config.num_slots * config.slot_size;
    }
    // Initialize as creator (sets up header)
    void initialize(void* memory, const RingBufferConfig& config) {
        // Validate power of 2
        if (config.num_slots == 0 || (config.num_slots & (config.num_slots - 1)) != 0) {
            throw std::invalid_argument("num_slots must be power of 2");
        }
        header_ = new (memory) Header{};
        header_->num_slots = config.num_slots;
        header_->slot_size = config.slot_size;
        header_->mask = config.num_slots - 1;
        slots_ = static_cast<uint8_t*>(memory) + HEADER_SIZE;
    }
    // Attach to existing buffer (producer or consumer)
    void attach(void* memory) {
        header_ = static_cast<Header*>(memory);
        slots_ = static_cast<uint8_t*>(memory) + HEADER_SIZE;
    }
    // Producer: write a message
    // Returns true on success, false if buffer is full
    bool try_produce(const void* data, size_t size) {
        uint64_t head = header_->head.load(std::memory_order_relaxed);
        uint64_t tail = header_->tail.load(std::memory_order_acquire);
        // Check if full (one slot must remain empty)
        uint64_t next_head = (head + 1) & header_->mask;
        if (next_head == tail) {
            return false;  // Buffer full
        }
        // Copy data to slot
        size_t copy_size = std::min(size, static_cast<size_t>(header_->slot_size));
        void* slot = get_slot(head);
        std::memcpy(slot, data, copy_size);
        // Ensure copy completes before updating head
        std::atomic_thread_fence(std::memory_order_release);
        header_->head.store(next_head, std::memory_order_relaxed);
        return true;
    }
    // Producer: blocking write
    void produce(const void* data, size_t size) {
        while (!try_produce(data, size)) {
            // Backoff strategy: start with spin, then yield
            for (int i = 0; i < 100; ++i) {
                if (try_produce(data, size)) return;
                __builtin_ia32_pause();  // x86 pause instruction
            }
            std::this_thread::yield();
        }
    }
    // Consumer: read a message
    // Returns true on success, false if buffer is empty
    bool try_consume(void* out_data, size_t out_size) {
        uint64_t tail = header_->tail.load(std::memory_order_relaxed);
        uint64_t head = header_->head.load(std::memory_order_acquire);
        // Check if empty
        if (tail == head) {
            return false;  // Buffer empty
        }
        // Ensure we see data after head was updated
        std::atomic_thread_fence(std::memory_order_acquire);
        // Copy data from slot
        size_t copy_size = std::min(out_size, static_cast<size_t>(header_->slot_size));
        const void* slot = get_slot(tail);
        std::memcpy(out_data, slot, copy_size);
        // Update tail
        uint64_t next_tail = (tail + 1) & header_->mask;
        header_->tail.store(next_tail, std::memory_order_relaxed);
        return true;
    }
    // Consumer: blocking read
    void consume(void* out_data, size_t out_size) {
        while (!try_consume(out_data, out_size)) {
            for (int i = 0; i < 100; ++i) {
                if (try_consume(out_data, out_size)) return;
                __builtin_ia32_pause();
            }
            std::this_thread::yield();
        }
    }
    // Query functions
    size_t capacity() const { return header_->num_slots - 1; }  // -1 because one slot always empty
    size_t slot_size() const { return header_->slot_size; }
    uint64_t count() const {
        uint64_t head = header_->head.load(std::memory_order_acquire);
        uint64_t tail = header_->tail.load(std::memory_order_acquire);
        return (head - tail + header_->num_slots) & header_->mask;
    }
    bool empty() const {
        return header_->head.load(std::memory_order_acquire) == 
               header_->tail.load(std::memory_order_acquire);
    }
    bool full() const {
        uint64_t head = header_->head.load(std::memory_order_acquire);
        uint64_t tail = header_->tail.load(std::memory_order_acquire);
        return ((head + 1) & header_->mask) == tail;
    }
private:
    void* get_slot(uint64_t index) {
        return slots_ + (index * header_->slot_size);
    }
    const void* get_slot(uint64_t index) const {
        return slots_ + (index * header_->slot_size);
    }
    Header* header_ = nullptr;
    uint8_t* slots_ = nullptr;
};
```
### The Memory Layout
Let's be precise about the byte layout of our shared memory region:
```
Offset 0x000: Header (128 bytes)
  ├─ 0x000-0x03F: head (atomic<uint64_t>) + padding to 64 bytes
  ├─ 0x040-0x07F: tail (atomic<uint64_t>) + padding to 64 bytes
  └─ 0x050-0x07F: num_slots, slot_size, mask, padding
Offset 0x080: Slot 0 (slot_size bytes)
Offset 0x080 + slot_size: Slot 1
Offset 0x080 + 2*slot_size: Slot 2
...
Offset 0x080 + (num_slots-1)*slot_size: Slot N-1
```
The `alignas(64)` on `head` and `tail` ensures they land on separate cache lines. The Header struct itself is aligned to 64 bytes, so the slots array starts at an aligned boundary.
---
## Handling Process Crashes
Here's a nightmare scenario: the producer crashes after updating `head` but *before* copying the message data. The consumer sees `head` advance, reads the slot, and gets garbage.
There are several strategies:
### Strategy 1: Sequence Numbers per Slot
Each slot has a sequence number. The producer writes data, then updates the slot's sequence number. The consumer checks the sequence number before and after reading:
```cpp
struct Slot {
    alignas(8) std::atomic<uint64_t> sequence;
    uint8_t data[MSG_SIZE];
};
// Producer
void produce_with_seq(RingBuffer* ring, const void* msg, size_t size) {
    uint64_t pos = /* get position */;
    Slot* slot = &ring->slots[pos];
    uint64_t seq = slot->sequence.load(std::memory_order_relaxed);
    // Copy data first
    std::memcpy(slot->data, msg, size);
    // Then update sequence to signal ready
    slot->sequence.store(seq + 1, std::memory_order_release);
    ring->head.store(next, std::memory_order_release);
}
// Consumer
bool consume_with_seq(RingBuffer* ring, void* out, size_t out_size) {
    uint64_t pos = ring->tail.load(std::memory_order_relaxed);
    Slot* slot = &ring->slots[pos];
    uint64_t expected_seq = /* compute expected sequence */;
    uint64_t seq = slot->sequence.load(std::memory_order_acquire);
    if (seq != expected_seq) {
        // Producer crashed mid-write, or not ready yet
        return false;
    }
    std::memcpy(out, slot->data, out_size);
    // Verify sequence didn't change during read (producer didn't wrap around)
    if (slot->sequence.load(std::memory_order_acquire) != seq) {
        // Data was modified during read - corruption possible
        return false;
    }
    ring->tail.store(next, std::memory_order_release);
    return true;
}
```
This approach detects incomplete writes but adds overhead: an atomic load per message, plus the sequence number storage.
### Strategy 2: Heartbeat/Timeout Detection
The producer periodically updates a "last alive" timestamp. If the consumer doesn't see updates for too long, it assumes the producer crashed and resets the buffer state.
```cpp
struct Header {
    // ... existing fields ...
    alignas(64) std::atomic<uint64_t> producer_heartbeat;
    alignas(64) std::atomic<uint64_t> consumer_heartbeat;
};
// Producer thread
void producer_loop() {
    while (running) {
        if (has_message()) {
            produce(message);
        }
        header_->producer_heartbeat.store(now_ns(), std::memory_order_relaxed);
    }
}
// Consumer checks periodically
bool is_producer_alive() {
    uint64_t last = header_->producer_heartbeat.load(std::memory_order_relaxed);
    return (now_ns() - last) < TIMEOUT_NS;
}
```
### Strategy 3: Recovery via Shared State Reset
When a process restarts, it must detect whether it's rejoining an existing buffer or creating a new one:
```cpp
struct Header {
    // ... existing fields ...
    alignas(64) std::atomic<uint64_t> producer_pid;  // PID of current producer
    alignas(64) std::atomic<uint64_t> consumer_pid;  // PID of current consumer
    alignas(64) std::atomic<uint64_t> generation;    // Incremented on reset
};
// On producer startup
void producer_attach(Header* header) {
    pid_t my_pid = getpid();
    pid_t old_pid = header->producer_pid.exchange(my_pid);
    if (old_pid != 0 && old_pid != my_pid) {
        // Previous producer existed - check if it's still alive
        if (kill(old_pid, 0) == -1 && errno == ESRCH) {
            // Old producer is dead - reset state
            header->head.store(0, std::memory_order_relaxed);
            header->generation.fetch_add(1, std::memory_order_relaxed);
        }
    }
}
```
**Trade-offs**:
| Strategy | Overhead | Detects | Recovers |
|----------|----------|---------|----------|
| Sequence numbers | High (atomic per slot) | Incomplete writes | Partial |
| Heartbeat | Low (periodic) | Dead process | Manual |
| PID tracking | Low (startup only) | Dead process | Automatic |
For a high-throughput system, combine heartbeat + PID tracking: use heartbeat for normal operation, PID check on startup for recovery.
---
## Benchmarking: Measuring What Matters
You're not done until you measure. Here's a benchmark framework:
```cpp
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
struct LatencyStats {
    double min_ns;
    double max_ns;
    double mean_ns;
    double median_ns;
    double p99_ns;
    double p999_ns;
};
LatencyStats measure_round_trip(SpscRingBuffer& producer_buf,
                                 SpscRingBuffer& consumer_buf,
                                 size_t iterations) {
    std::vector<double> latencies;
    latencies.reserve(iterations);
    uint64_t msg = 0;
    uint64_t recv = 0;
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        // Producer sends
        producer_buf.produce(&msg, sizeof(msg));
        // Consumer receives
        consumer_buf.consume(&recv, sizeof(recv));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::nano> latency = end - start;
        latencies.push_back(latency.count());
        ++msg;
    }
    // Compute statistics
    std::sort(latencies.begin(), latencies.end());
    LatencyStats stats;
    stats.min_ns = latencies.front();
    stats.max_ns = latencies.back();
    stats.mean_ns = std::accumulate(latencies.begin(), latencies.end(), 0.0) / iterations;
    stats.median_ns = latencies[iterations / 2];
    stats.p99_ns = latencies[static_cast<size_t>(iterations * 0.99)];
    stats.p999_ns = latencies[static_cast<size_t>(iterations * 0.999)];
    return stats;
}
```
### Expected Performance
On a modern x86 system (Intel Xeon or AMD EPYC), with proper cache line alignment:
| Metric | Target | Notes |
|--------|--------|-------|
| Median latency | 100-300 ns | Depends on message size |
| P99 latency | 500-1000 ns | May include scheduler effects |
| P99.9 latency | 5-50 μs | Outliers from OS interrupts |
| Throughput | 5-20 M msg/s | Depends on message size and core speed |
If your median latency is above 500ns, check:
1. False sharing (use `alignas(64)`)
2. Power-of-2 buffer size (use bitwise AND, not modulo)
3. Spinning vs yielding (spin for low latency, yield for CPU efficiency)
---
## Design Decisions: Why This, Not That
### POSIX shm vs System V shm
| Aspect | POSIX (`shm_open`) | System V (`shmget`) |
|--------|-------------------|---------------------|
| **API style** | File descriptor based | IPC key based |
| **Permissions** | Standard file permissions | IPC-specific permissions |
| **Cleanup** | `shm_unlink()` by name | `shmctl(IPC_RMID)` by ID |
| **Discovery** | `/dev/shm/` visible | `ipcs` command |
| **Portability** | POSIX.1-2001 | Unix heritage |
**Choice**: POSIX shm. The file descriptor model integrates with `poll`/`epoll`, works with `ftruncate`, and has cleaner cleanup semantics.
### Busy-wait vs Blocking
| Approach | Latency | CPU Usage | When to Use |
|----------|---------|-----------|-------------|
| **Busy-wait** | Lowest | 100% core | Dedicated latency-critical cores |
| **Yield loop** | Low + variable | ~50% core | Balanced workload |
| **Futex/condvar** | Higher | ~0% idle | Throughput-oriented, power-sensitive |
**Choice**: Yield loop with spin count. Spin 100 iterations before yielding—gives low latency for sustained traffic while yielding to other processes during idle periods.
---
## Knowledge Cascade: What You've Unlocked

![Alternative Reality: Industry Comparisons](./diagrams/diag-global-005.svg)

**1. LMAX Disruptor Pattern**
The ring buffer you just built is the heart of the LMAX Disruptor, used in high-frequency trading systems to process **6+ million events per second** with sub-microsecond latency. The Disruptor adds: a) pre-allocated event objects (no GC pressure in Java), b) multi-producer support via CAS on a sequence cursor, c) dependency barriers for parallel consumers. Your SPSC buffer is the foundational primitive.
**2. Memory-Mapped I/O in Databases (Cross-Domain)**
The `mmap()` technique you used for shared memory is the same mechanism databases use for zero-copy reads from disk. LMDB (used in OpenLDAP and Bitcoin) memory-maps the entire database file—reads are just pointer dereferences into the mapped region, with the OS handling paging. SQLite's Write-Ahead Log uses memory mapping for fast log reads. The skill you just learned—managing memory-mapped regions, handling page faults, reasoning about visibility—transfers directly to database internals.
**3. Cache Coherency Protocols (MESI)**
The `alignas(64)` fix for false sharing works because of the CPU's cache coherency protocol. Modern x86 uses MESI (Modified, Exclusive, Shared, Invalid) to track cache line state across cores. When two cores frequently write to the same cache line (even different bytes within it), the line bounces between cores in the "Modified" state, requiring inter-core traffic. By separating head and tail to different lines, each core keeps its line in "Modified" state locally—no ping-pong.
**4. Lock-Free vs Wait-Free Distinction**
Your SPSC buffer is **wait-free** for the happy path: `try_produce` and `try_consume` complete in a bounded number of instructions regardless of what the other process does. This is stronger than "lock-free" (where system-wide progress is guaranteed, but individual threads may starve). For HFT systems, wait-free is essential—you need worst-case bounds, not just average-case progress.
**5. ABA Problem (Coming in MPMC)**
When you extend to multiple producers in Milestone 3, you'll encounter the ABA problem: producer reads head=5, prepares slot 5, but before it can update, another producer fills and empties slot 5, making head=5 again. The first producer's CAS succeeds but writes to a slot that's now in use by a consumer. The solution requires sequence numbers or epoch counters—the foundation you built with sequence-aware slots is exactly what you'll need.
---
## What's Next
You've built a **single-producer, single-consumer** lock-free ring buffer in shared memory. This is the foundational primitive for the entire message bus. 
In Milestone 2, you'll design a **zero-copy serialization format** so you can pass structured messages (not just byte arrays) through this buffer. The key insight: if your serialization format allows direct memory access without parsing, you can read fields directly from the ring buffer slots—no intermediate structures, no copies.
In Milestone 3, you'll extend to **multiple producers and consumers**, which introduces contention, fairness, and the ABA problem. The simple `fetch_add` for slot allocation won't work anymore—you'll need compare-and-swap loops and more sophisticated coordination.
---
<!-- END_MS -->


<!-- MS_ID: zcmb-m2 -->
# Milestone 2: Zero-Copy Serialization
## The Problem: When Parsing Becomes Your Bottleneck
You've built the ring buffer. Sub-microsecond latency, zero copies between processes. The producer writes bytes, the consumer reads bytes. But here's the catch: those bytes are meaningless without structure.
A 256-byte message arrives. What is it? A market data update? A trade order? A heartbeat? You need to interpret those bytes as something meaningful—extract the symbol, the price, the quantity, the timestamp.
**The traditional approach**: Serialize to a format like JSON or Protocol Buffers. The producer encodes a `TradeOrder` object into bytes. The consumer *parses* those bytes back into a `TradeOrder` object, then accesses the fields.
```
Producer: TradeOrder → serialize → bytes → ring buffer
Consumer: ring buffer → bytes → PARSE → TradeOrder → access fields
                                    ↑
                              This is the problem
```
At 2 million messages per second, parsing is catastrophic. Consider what parsing actually does:
```cpp
// Typical JSON parsing for a trade order
TradeOrder parse_trade_order(const char* json) {
    TradeOrder order;
    // Parse "{"
    skip_whitespace(json);
    expect_char(json, '{');
    // Parse "symbol": "AAPL"
    parse_string_field(json, "symbol", order.symbol, sizeof(order.symbol));
    // Parse "price": 150.25
    parse_double_field(json, "price", &order.price);
    // Parse "quantity": 100
    parse_int_field(json, "quantity", &order.quantity);
    // Parse "timestamp": 1697234567890123456
    parse_int64_field(json, "timestamp", &order.timestamp);
    // Parse "}"
    skip_whitespace(json);
    expect_char(json, '}');
    return order;
}
```
Every field requires:
1. **String comparison** to find the field name
2. **Type conversion** (string to int, string to float)
3. **Memory allocation** for the output object
4. **Memory copies** to populate the output object
For a message with 10 fields, you might execute **10,000+ CPU instructions** just to read data that's already in memory.
**The tension**: You need structured data with typed fields, nested messages, and arrays. But every layer of abstraction—parsing, object construction, validation—adds latency. At 2M msg/s, 500ns of parsing overhead per message is **1 second of CPU time per second** just on parsing. And that's before you've *done* anything with the data.
**The escape hatch**: What if you never parsed at all? What if the bytes in shared memory were already in the right layout to access directly?
```cpp
// What if you could do this?
TradeOrderView order(ring_buffer_slot);
double price = order.price();      // Just pointer + offset
int64_t qty = order.quantity();    // Just pointer + offset
```
No parsing. No object construction. No copies. Just pointer arithmetic.
This is **zero-copy serialization**, and it's how high-frequency trading systems, real-time analytics engines, and low-latency messaging frameworks achieve their throughput.
---
## The Revelation: Parsing Is a Choice, Not a Requirement
Here's the misconception that limits most developers: "Serialization means converting between objects and bytes. You serialize on one side, parse on the other. That's just how it works."
This mental model comes from text-based formats (JSON, XML) and reflection-based formats (Java serialization, Python pickle). In these worlds, parsing is unavoidable because the byte representation is designed for *portability*, not *direct access*.
> **🔑 Foundation: Serialization without parsing—direct memory interpretation**
> 
> ## What It Is
**Zero-copy serialization** inverts the traditional model. Instead of:
```
Object → bytes (serialize) → network/disk → bytes → Object (parse)
```
You design a binary layout where:
```
Object → bytes (serialize) → network/disk → Direct access (no parse)
```
The bytes *are* the data structure. Field access is pointer arithmetic:
```cpp
struct TradeOrder {
    uint64_t timestamp;    // Offset 0, 8 bytes
    uint64_t order_id;     // Offset 8, 8 bytes
    double price;          // Offset 16, 8 bytes
    int32_t quantity;      // Offset 24, 4 bytes
    char symbol[8];        // Offset 28, 8 bytes
};  // Total: 36 bytes (may be padded to 40)
// Direct access: price = *(double*)(buffer + 16)
```
## Why You Need This Now
When latency matters more than convenience, you cannot afford the parsing step. Zero-copy formats are used in:
- **High-frequency trading**: FlatBuffers, Cap'n Proto, SBE (Simple Binary Encoding)
- **Game networking**: Direct binary structures for position/velocity/state
- **Databases**: Slotted pages where records are accessed in-place
- **GPU programming**: Structured buffers passed directly to shaders
The trade-off: you lose human readability and schema flexibility gains complexity. But you gain **10-100x** faster access.
## Key Insight
> **The "parsing" happens at compile time, not runtime.**
Your code generator reads the schema and produces accessor functions that hardcode the offsets. At runtime, `order.price()` compiles to something like:
```asm
movsd xmm0, [rdi + 16]  ; Load double at rdi+16 into xmm0
```
One instruction. No loops, no string comparisons, no allocations. The schema complexity is "compiled away" into constants.
**Flat buffers flip the model entirely**:
| Traditional Serialization | Flat Buffer (Zero-Copy) |
|--------------------------|------------------------|
| Parse entire message to access one field | Access any field directly via offset |
| O(n) parsing where n = message size | O(1) field access regardless of message size |
| Allocate objects on heap | Read directly from buffer, no allocation |
| Field order doesn't matter | Field order encoded in offsets |
| Schema changes require re-parsing | Schema evolution via versioned offsets |
The key insight: **you're not building a parser. You're building a memory layout specification.**
---
## The Architecture: How Flat Buffers Work

![Three-Level View: Complete Message Flow](./diagrams/diag-global-002.svg)

A flat buffer is a contiguous block of memory with a specific structure:
```
+------------------+
| Root Table       |  <-- Offset from buffer start to root table
+------------------+
| Table 1 (Order)  |
|   - field 1      |
|   - field 2      |
|   - offset ->    |  <-- Offset to nested table
+------------------+
| Table 2 (Nested) |
|   - field A      |
|   - field B      |
+------------------+
| String Data      |  <-- Variable-length data at the end
+------------------+
| Vector Data      |
+------------------+
```
### The Root Table
Every flat buffer starts with a **root table**. This is a fixed-size structure containing offsets to the actual data. The root table acts as the entry point—everything else is reachable from here.
```cpp
// Root table format
struct RootTable {
    uint32_t soffset_to_root;  // Signed offset to root table (from buffer end)
};
```
Wait, why is the offset from the *buffer end*? This is a clever design choice: it allows the buffer to be truncated from the front without invalidating offsets. More on this later.
### Tables and VTables
A **table** is the core data structure. It contains:
1. A **vtable offset** (soffset) pointing to the table's metadata
2. Inline data for the table's fields
```cpp
// Table format in memory
struct Table {
    int32_t soffset_to_vtable;  // Offset to vtable (negative, points backward)
    // Followed by inline field data...
    // uint8_t/uint16_t/uint32_t/uint64_t/float/double as specified by schema
};
```
The **vtable** (virtual table) is metadata that describes the table's layout:
```cpp
// VTable format
struct VTable {
    uint16_t vtable_size;    // Size of this vtable
    uint16_t table_size;     // Size of the table data (excluding vtable offset)
    uint16_t field_offsets[]; // Array of offsets for each field
};
```
Here's a concrete example. Consider this schema:
```
table TradeOrder {
    symbol: string;      // Field 0
    price: double;       // Field 1
    quantity: int32;     // Field 2
    timestamp: int64;    // Field 3
}
```
The memory layout for a TradeOrder might look like:
```
VTable (at offset 0):
  +0x00: vtable_size = 12    (2 + 2 + 4*2 = 12 bytes)
  +0x02: table_size = 24     (size of table data)
  +0x04: field[0] = 16       (symbol: offset 16 from table start, points to string)
  +0x06: field[1] = 8        (price: offset 8, inline double)
  +0x08: field[2] = 4        (quantity: offset 4, inline int32)
  +0x0A: field[3] = 0        (timestamp: offset 0, inline int64)
Table (at offset 12):
  +0x00: vtable_offset = -12  (points back to vtable at offset 0)
  +0x04: timestamp: int64     (field 3, inline)
  +0x0C: quantity: int32      (field 2, inline)
  +0x10: padding (4 bytes)    (align price to 8 bytes)
  +0x14: price: double        (field 1, inline)
  +0x1C: symbol_offset: uint32 (field 0, offset to string data)
String (at offset 44):
  +0x00: length: uint32 = 4
  +0x04: "AAPL\0"
```
### Why VTables? Schema Evolution
You might wonder: why the indirection? Why not just hardcode offsets?
**Schema evolution.** Consider what happens when you add a field:
```
// Version 1 schema
table TradeOrder {
    symbol: string;
    price: double;
}
// Version 2 schema (adds quantity and timestamp)
table TradeOrder {
    symbol: string;
    price: double;
    quantity: int32;    // NEW
    timestamp: int64;   // NEW
}
```
With vtables:
- A Version 1 message has a vtable with 2 field offsets
- A Version 2 message has a vtable with 4 field offsets
- A Version 2 reader reading a Version 1 message checks the vtable size, sees only 2 fields, and returns default values for the new fields
- A Version 1 reader reading a Version 2 message ignores the extra fields
**This is backward and forward compatibility without any parsing or conversion.**
---
## Alignment: The Hidden Complexity
There's a catch to direct memory access: **alignment requirements**.
On most CPUs, accessing a multi-byte value at an unaligned address is either:
1. **Slower** (x86: works but may take extra cycles)
2. **Crashes** (ARM: some instructions fault on unaligned access)
3. **Silently wrong** (some architectures round the address)
A `double` (8 bytes) must be at an address divisible by 8. An `int32_t` (4 bytes) must be at an address divisible by 4. A `uint16_t` (2 bytes) must be at an address divisible by 2.
This is why our flat buffer layout includes padding:
```
+0x04: timestamp: int64     (offset 4, but int64 needs 8-byte alignment)
+0x0C: quantity: int32      (offset 12, int32 needs 4-byte alignment ✓)
+0x10: padding (4 bytes)    (we need price at offset 20, but it must be 8-aligned)
+0x14: price: double        (WRONG - 0x14 = 20, not divisible by 8!)
```
Let me fix that:
```
+0x04: timestamp: int64     (offset 4 from table, but table must be 8-aligned!)
```
Actually, the proper approach is:
1. **Order fields by size (largest first)** to minimize padding
2. **Align the entire table** to the largest field's alignment
3. **Add padding between fields** as needed
Here's the corrected layout:
```cpp
// Proper layout: order by alignment (largest first)
struct TradeOrderData {
    double price;          // Offset 0, 8 bytes, 8-aligned ✓
    int64_t timestamp;     // Offset 8, 8 bytes, 8-aligned ✓
    int32_t quantity;      // Offset 16, 4 bytes, 4-aligned ✓
    uint32_t symbol_offset; // Offset 20, 4 bytes, 4-aligned ✓
};  // Total: 24 bytes
```
By putting the 8-byte fields first, we eliminate padding. The entire struct is 24 bytes, naturally 8-aligned.
### The Alignment Calculator
For code generation, you need to calculate offsets at compile time:
```cpp
class LayoutCalculator {
public:
    struct FieldLayout {
        size_t offset;
        size_t size;
        size_t alignment;
    };
    void add_field(size_t size, size_t alignment) {
        // Align current position to required alignment
        size_t aligned_offset = align_up(current_offset_, alignment);
        fields_.push_back({aligned_offset, size, alignment});
        current_offset_ = aligned_offset + size;
        max_alignment_ = std::max(max_alignment_, alignment);
    }
    size_t total_size() const {
        // Pad final size to max alignment
        return align_up(current_offset_, max_alignment_);
    }
    size_t max_alignment() const { return max_alignment_; }
    const std::vector<FieldLayout>& fields() const { return fields_; }
private:
    static size_t align_up(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }
    std::vector<FieldLayout> fields_;
    size_t current_offset_ = 0;
    size_t max_alignment_ = 1;
};
```
Usage:
```cpp
LayoutCalculator calc;
calc.add_field(8, 8);  // double price
calc.add_field(8, 8);  // int64_t timestamp
calc.add_field(4, 4);  // int32_t quantity
calc.add_field(4, 4);  // uint32_t symbol_offset
// calc.total_size() == 24
// calc.max_alignment() == 8
// fields_[0].offset == 0  (price)
// fields_[1].offset == 8  (timestamp)
// fields_[2].offset == 16 (quantity)
// fields_[3].offset == 20 (symbol_offset)
```
---
## Endianness: When Bytes Go Backward
There's another hardware complexity: **endianness**.
```
Number: 0x12345678
Big-endian (network byte order):
  Address 0: 0x12
  Address 1: 0x34
  Address 2: 0x56
  Address 3: 0x78
Little-endian (x86, ARM default):
  Address 0: 0x78
  Address 1: 0x56
  Address 2: 0x34
  Address 3: 0x12
```
Most modern systems are little-endian. But if you're sending data between an x86 server (little-endian) and a PowerPC network appliance (big-endian), you have a problem.
**Options:**
1. **Declare a canonical byte order** (usually little-endian for performance on most systems). Big-endian systems byte-swap on read/write.
2. **Include endianness marker** in the buffer header. Readers check and swap if needed.
3. **Generate separate code paths** for each endianness.
For our message bus, since we're targeting same-machine IPC, we can assume **native endianness** and skip byte-swapping entirely. This is a significant performance win.
If cross-platform support is needed:
```cpp
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #define FLATBUF_ENDIAN_BIG 1
#else
    #define FLATBUF_ENDIAN_LITTLE 1
#endif
// Byte swap helpers
inline uint32_t swap32(uint32_t x) {
    return __builtin_bswap32(x);
}
inline uint64_t swap64(uint64_t x) {
    return __builtin_bswap64(x);
}
// Conditional swap (if buffer is big-endian and we're little-endian)
template<typename T>
T read_swap_if_needed(const void* ptr, bool buffer_is_big_endian) {
    T value;
    std::memcpy(&value, ptr, sizeof(T));
#if FLATBUF_ENDIAN_LITTLE
    if (buffer_is_big_endian) {
        if constexpr (sizeof(T) == 2) value = __builtin_bswap16(value);
        else if constexpr (sizeof(T) == 4) value = swap32(value);
        else if constexpr (sizeof(T) == 8) value = swap64(value);
    }
#endif
    return value;
}
```
---
## Schema Definition Language
Now let's design a schema language for defining message types. This is the interface developers will use to specify their data structures.
```
// Example: market_data.fbs
namespace trading;
enum OrderSide: byte {
    BUY = 0,
    SELL = 1
}
table Symbol {
    code: string;
    exchange: string;
}
table TradeOrder {
    id: ulong;
    symbol: Symbol;        // Nested table
    side: OrderSide;
    price: double;
    quantity: int;
    timestamp: long;
    flags: uint = 0;       // Default value
}
table MarketData {
    symbol: string;
    bids: [double];        // Array of doubles (sorted by price)
    asks: [double];
    timestamp: long;
}
root_type TradeOrder;
```
### Schema Elements
| Element | Purpose | Example |
|---------|---------|---------|
| `namespace` | Organize types, avoid collisions | `trading.TradeOrder` |
| `enum` | Named integer constants | `OrderSide` |
| `table` | Primary data structure, supports evolution | `TradeOrder` |
| `struct` | Fixed layout, no evolution, inline only | `Vec3 { x: float; y: float; z: float; }` |
| `root_type` | Entry point for serialization | `TradeOrder` |
### Scalar Types
| Type | Size | C++ Type | Notes |
|------|------|----------|-------|
| `byte`/`int8` | 1 | `int8_t` | Signed |
| `ubyte`/`uint8` | 1 | `uint8_t` | Unsigned |
| `bool` | 1 | `bool` | 0 or 1 |
| `short`/`int16` | 2 | `int16_t` | Signed |
| `ushort`/`uint16` | 2 | `uint16_t` | Unsigned |
| `int`/`int32` | 4 | `int32_t` | Signed |
| `uint`/`uint32` | 4 | `uint32_t` | Unsigned |
| `long`/`int64` | 8 | `int64_t` | Signed |
| `ulong`/`uint64` | 8 | `uint64_t` | Unsigned |
| `float` | 4 | `float` | IEEE 754 |
| `double` | 8 | `double` | IEEE 754 |
---
## Code Generation: The Parser Is the Compiler
The magic of flat buffers is that the "parsing" happens at compile time via code generation. Your schema is parsed by a code generator, which produces C++ classes with hardcoded offsets.
Here's a simplified code generator:
```cpp
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include <map>
struct SchemaField {
    std::string name;
    std::string type;
    int field_id;
    std::string default_value;
};
struct SchemaTable {
    std::string name;
    std::vector<SchemaField> fields;
};
class SchemaParser {
public:
    bool parse(const std::string& content) {
        // Simplified parser - real parser would use proper lexer/parser
        std::regex table_regex(R"(table\s+(\w+)\s*\{([^}]+)\})");
        std::regex field_regex(R"((\w+):\s*(\w+)(?:\s*=\s*(\w+))?)");
        auto table_begin = std::sregex_iterator(content.begin(), content.end(), table_regex);
        auto table_end = std::sregex_iterator();
        for (auto it = table_begin; it != table_end; ++it) {
            SchemaTable table;
            table.name = (*it)[1].str();
            std::string fields_str = (*it)[2].str();
            auto field_begin = std::sregex_iterator(fields_str.begin(), fields_str.end(), field_regex);
            int field_id = 0;
            for (auto fit = field_begin; fit != table_end; ++fit) {
                SchemaField field;
                field.name = (*fit)[1].str();
                field.type = (*fit)[2].str();
                field.default_value = (*fit)[3].matched ? (*fit)[3].str() : "";
                field.field_id = field_id++;
                table.fields.push_back(field);
            }
            tables_[table.name] = table;
        }
        return true;
    }
    const std::map<std::string, SchemaTable>& tables() const { return tables_; }
private:
    std::map<std::string, SchemaTable> tables_;
};
```
### Generated C++ Code
For the `TradeOrder` table, the generator produces:
```cpp
// Generated by flat buffer compiler - DO NOT EDIT
namespace trading {
struct TradeOrder;
struct TradeOrderBuilder;
// VTable structure (shared between all TradeOrder instances)
struct TradeOrderVTable {
    uint16_t vtable_size;
    uint16_t table_size;
    uint16_t field_offsets[6];  // 6 fields
};
// Table structure (in buffer)
struct TradeOrderTable {
    int32_t vtable_offset;  // Signed offset to vtable
    // Fields follow (offsets determined by vtable)
};
// Accessor class (zero-copy access)
class TradeOrderView {
public:
    explicit TradeOrderView(const uint8_t* buffer, size_t offset)
        : buffer_(buffer), table_(reinterpret_cast<const TradeOrderTable*>(buffer + offset)) {
        const int32_t* vtable_ptr = reinterpret_cast<const int32_t*>(
            reinterpret_cast<const uint8_t*>(table_) + table_->vtable_offset);
        vtable_ = reinterpret_cast<const TradeOrderVTable*>(
            reinterpret_cast<const uint8_t*>(table_) + *vtable_ptr);
    }
    // Field accessors - each is O(1) pointer arithmetic
    bool has_id() const { return check_field(0); }
    uint64_t id() const { return get_field<uint64_t>(0, 0); }
    bool has_symbol() const { return check_field(1); }
    const char* symbol() const { 
        if (!check_field(1)) return "";
        uint32_t str_offset = get_field<uint32_t>(1, 0);
        const uint8_t* str_ptr = buffer_ + str_offset;
        uint32_t len = *reinterpret_cast<const uint32_t*>(str_ptr);
        return reinterpret_cast<const char*>(str_ptr + 4);
    }
    bool has_price() const { return check_field(3); }
    double price() const { return get_field<double>(3, 0.0); }
    bool has_quantity() const { return check_field(4); }
    int32_t quantity() const { return get_field<int32_t>(4, 0); }
    bool has_timestamp() const { return check_field(5); }
    int64_t timestamp() const { return get_field<int64_t>(5, 0); }
private:
    template<typename T>
    T get_field(int field_id, T default_val) const {
        if (field_id * 2 + 4 >= vtable_->vtable_size) return default_val;
        uint16_t offset = vtable_->field_offsets[field_id];
        if (offset == 0) return default_val;
        return *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(table_) + offset);
    }
    bool check_field(int field_id) const {
        if (field_id * 2 + 4 >= vtable_->vtable_size) return false;
        return vtable_->field_offsets[field_id] != 0;
    }
    const uint8_t* buffer_;
    const TradeOrderTable* table_;
    const TradeOrderVTable* vtable_;
};
// Builder class (creates flat buffer)
class TradeOrderBuilder {
public:
    TradeOrderBuilder(uint8_t* buffer, size_t capacity)
        : buffer_(buffer), capacity_(capacity), offset_(0) {}
    void set_id(uint64_t value) {
        write_field(0, value);
    }
    void set_symbol(const char* str) {
        // Write string at current position
        size_t len = strlen(str);
        write_u32(len);
        memcpy(buffer_ + offset_, str, len + 1);
        size_t str_offset = offset_;
        offset_ += len + 1;
        // Store offset in field
        write_field(1, static_cast<uint32_t>(str_offset));
    }
    void set_price(double value) {
        write_field(3, value);
    }
    void set_quantity(int32_t value) {
        write_field(4, value);
    }
    void set_timestamp(int64_t value) {
        write_field(5, value);
    }
    size_t finish() {
        // Calculate table size and create vtable
        // ... (implementation details)
        return offset_;
    }
private:
    template<typename T>
    void write_field(int field_id, T value) {
        size_t field_offset = field_offsets_[field_id];
        *reinterpret_cast<T*>(buffer_ + field_offset) = value;
    }
    void write_u32(uint32_t value) {
        *reinterpret_cast<uint32_t*>(buffer_ + offset_) = value;
        offset_ += 4;
    }
    uint8_t* buffer_;
    size_t capacity_;
    size_t offset_;
    std::map<int, size_t> field_offsets_;
};
}  // namespace trading
```
---
## Nested Messages and Arrays
Real messages aren't flat. A `TradeOrder` might contain a `Symbol` object. A `MarketData` message contains arrays of bids and asks.
### Nested Tables
Nested tables are stored by reference (offset):
```cpp
// Symbol is a separate table in the buffer
// TradeOrder contains a uint32 offset pointing to it
class TradeOrderView {
public:
    SymbolView symbol() const {
        uint32_t sym_offset = get_field<uint32_t>(1, 0);
        return SymbolView(buffer_, sym_offset);
    }
};
```
The nested table can be anywhere in the buffer—the offset is always relative to the buffer start.
### Vectors (Arrays)
Vectors are stored with a length prefix:
```cpp
// Vector layout:
// +0: uint32 count (number of elements)
// +4: element[0]
// +4 + sizeof(T): element[1]
// ...
template<typename T>
class VectorView {
public:
    explicit VectorView(const uint8_t* data) : data_(data) {
        count_ = *reinterpret_cast<const uint32_t*>(data_);
    }
    size_t size() const { return count_; }
    const T& operator[](size_t i) const {
        return *reinterpret_cast<const T*>(data_ + 4 + i * sizeof(T));
    }
    const T* begin() const { return &(*this)[0]; }
    const T* end() const { return &(*this)[count_]; }
private:
    const uint8_t* data_;
    uint32_t count_;
};
// In MarketDataView
VectorView<double> bids() const {
    uint32_t vec_offset = get_field<uint32_t>(1, 0);
    return VectorView<double>(buffer_ + vec_offset);
}
```
### Strings
Strings are just vectors of bytes with a null terminator:
```cpp
class StringView {
public:
    explicit StringView(const uint8_t* data) : data_(data) {
        length_ = *reinterpret_cast<const uint32_t*>(data_);
    }
    size_t size() const { return length_; }
    const char* c_str() const { return reinterpret_cast<const char*>(data_ + 4); }
    std::string str() const { return std::string(c_str(), length_); }
private:
    const uint8_t* data_;
    uint32_t length_;
};
```
---
## Integration with the Ring Buffer
Now let's integrate our flat buffer format with the SPSC ring buffer from Milestone 1.
```cpp
#include "spsc_ring_buffer.h"
#include "flat_buffer.h"
// Message type IDs
enum class MessageType : uint16_t {
    kTradeOrder = 1,
    kMarketData = 2,
    kHeartbeat = 3,
};
// Message header (8 bytes)
struct alignas(8) MessageHeader {
    uint16_t magic;         // 0xFB00 for "FlatBuffer"
    uint16_t version;       // Schema version
    MessageType type;       // Message type ID
    uint16_t flags;         // Flags (compression, etc.)
    uint32_t payload_size;  // Size of payload following header
};
class MessageBus {
public:
    MessageBus(SpscRingBuffer& ring, bool is_producer)
        : ring_(ring), is_producer_(is_producer) {}
    // Send a TradeOrder (producer side)
    bool send_trade_order(const trading::TradeOrderBuilder& builder) {
        MessageHeader header;
        header.magic = 0xFB00;
        header.version = 1;
        header.type = MessageType::kTradeOrder;
        header.flags = 0;
        header.payload_size = static_cast<uint32_t>(builder.size());
        // Calculate total message size
        size_t msg_size = sizeof(MessageHeader) + header.payload_size;
        // Allocate slot in ring buffer
        if (!ring_.try_produce_prepare(msg_size)) {
            return false;
        }
        // Write header
        void* slot = ring_.slot_ptr();
        memcpy(slot, &header, sizeof(MessageHeader));
        // Write payload
        memcpy(static_cast<uint8_t*>(slot) + sizeof(MessageHeader),
               builder.data(), header.payload_size);
        // Commit
        ring_.try_produce_commit();
        return true;
    }
    // Receive a message (consumer side)
    bool receive_message(MessageType& out_type, const uint8_t** out_payload, 
                         uint32_t& out_size) {
        void* slot;
        if (!ring_.try_consume_prepare(&slot)) {
            return false;
        }
        const MessageHeader* header = static_cast<const MessageHeader*>(slot);
        // Validate magic
        if (header->magic != 0xFB00) {
            // Corrupted message - skip
            ring_.try_consume_commit();
            return false;
        }
        out_type = header->type;
        *out_payload = static_cast<const uint8_t*>(slot) + sizeof(MessageHeader);
        out_size = header->payload_size;
        return true;
    }
    // Process received message
    template<typename Handler>
    void process_message(const uint8_t* payload, uint32_t size, Handler& handler) {
        // Read message type from header
        MessageType type = *reinterpret_cast<const MessageType*>(payload - 
                          sizeof(MessageHeader) + offsetof(MessageHeader, type));
        switch (type) {
            case MessageType::kTradeOrder: {
                trading::TradeOrderView order(payload, 0);
                handler.handle_trade_order(order);
                break;
            }
            case MessageType::kMarketData: {
                trading::MarketDataView data(payload, 0);
                handler.handle_market_data(data);
                break;
            }
            case MessageType::kHeartbeat: {
                handler.handle_heartbeat();
                break;
            }
        }
    }
private:
    SpscRingBuffer& ring_;
    bool is_producer_;
};
```
### Usage Example
```cpp
// Producer
void send_order(MessageBus& bus, const std::string& symbol, 
                double price, int32_t quantity) {
    uint8_t buffer[1024];
    trading::TradeOrderBuilder builder(buffer, sizeof(buffer));
    builder.set_id(generate_order_id());
    builder.set_symbol(symbol.c_str());
    builder.set_price(price);
    builder.set_quantity(quantity);
    builder.set_timestamp(get_timestamp_ns());
    size_t size = builder.finish();
    bus.send_trade_order(builder);
}
// Consumer
struct OrderHandler {
    void handle_trade_order(const trading::TradeOrderView& order) {
        // ZERO-COPY access - just pointer arithmetic
        printf("Order %lu: %s %d @ %.2f\n",
               order.id(), order.symbol(), order.quantity(), order.price());
    }
    void handle_market_data(const trading::MarketDataView& data) {
        auto bids = data.bids();
        for (size_t i = 0; i < bids.size(); ++i) {
            // Still zero-copy - accessing array elements via offset
            printf("Bid[%zu]: %.2f\n", i, bids[i]);
        }
    }
    void handle_heartbeat() {
        printf("Heartbeat received\n");
    }
};
void consumer_loop(MessageBus& bus) {
    OrderHandler handler;
    MessageType type;
    const uint8_t* payload;
    uint32_t size;
    while (running) {
        if (bus.receive_message(type, &payload, size)) {
            bus.process_message(payload, size, handler);
        }
    }
}
```
---
## Schema Evolution: Adding Fields Without Breaking Everything
The real test of a serialization format is how it handles change. Your message schema will evolve:
- Add new fields
- Deprecate old fields
- Change field defaults
- Add new message types
Flat buffers handle evolution through **vtables** and **field IDs**.
### Adding a Field
```
// Version 1
table TradeOrder {
    symbol: string;    // Field 0
    price: double;     // Field 1
    quantity: int32;   // Field 2
}
// Version 2 - add new field
table TradeOrder {
    symbol: string;    // Field 0
    price: double;     // Field 1
    quantity: int32;   // Field 2
    timestamp: int64;  // Field 3 - NEW
}
```
When a Version 2 reader reads a Version 1 message:
1. The vtable has 3 field offsets (not 4)
2. Accessing `timestamp` (field 3) checks `vtable_size`
3. Field 3 is beyond vtable, so return default value (0)
When a Version 1 reader reads a Version 2 message:
1. The vtable has 4 field offsets
2. Version 1 code only knows about fields 0-2
3. Field 3 is simply ignored
### Deprecating a Field
```
table TradeOrder {
    symbol: string;    // Field 0
    price: double;     // Field 1
    quantity: int32;   // Field 2
    // timestamp deprecated - do not reuse field ID 3!
    // (new_field): int64;  // Would be field 4, not 3
    exchange: string;  // Field 4
}
```
**Rule**: Never reuse field IDs. A deprecated field leaves a "hole" in the field ID sequence, but reusing the ID would cause old readers to misinterpret new data.
### Field ID Assignment in Code Generator
```cpp
class SchemaEvolver {
public:
    void load_existing_schema(const std::string& filename);
    void assign_field_ids();
    void validate_compatibility();
private:
    std::map<std::string, int> existing_fields_;  // name -> field_id
    int next_field_id_ = 0;
};
void SchemaEvolver::assign_field_ids() {
    for (auto& field : new_fields_) {
        auto it = existing_fields_.find(field.name);
        if (it != existing_fields_.end()) {
            // Existing field - keep its ID
            field.field_id = it->second;
        } else {
            // New field - assign next ID
            field.field_id = next_field_id_++;
        }
    }
}
```
---
## Benchmarking: Zero-Copy vs Traditional Parsing
Let's measure the performance difference.
```cpp
#include <chrono>
#include <rapidjson/document.h>  // For comparison
struct BenchmarkResult {
    double avg_ns;
    double p50_ns;
    double p99_ns;
    double p999_ns;
};
// Benchmark 1: JSON parsing
BenchmarkResult benchmark_json_parsing(const std::string& json, size_t iterations) {
    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        double price = doc["price"].GetDouble();
        int quantity = doc["quantity"].GetInt();
        const char* symbol = doc["symbol"].GetString();
        (void)price; (void)quantity; (void)symbol;  // Prevent optimization
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    return compute_stats(latencies);
}
// Benchmark 2: Flat buffer zero-copy
BenchmarkResult benchmark_flatbuffer_access(const uint8_t* buffer, size_t iterations) {
    std::vector<double> latencies;
    latencies.reserve(iterations);
    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        trading::TradeOrderView order(buffer, 0);
        double price = order.price();
        int32_t quantity = order.quantity();
        const char* symbol = order.symbol();
        (void)price; (void)quantity; (void)symbol;
        auto end = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    return compute_stats(latencies);
}
```
### Expected Results
On a modern x86 system:
| Format | Access Time (3 fields) | Memory Allocation | Notes |
|--------|----------------------|-------------------|-------|
| JSON (rapidjson) | 500-2000 ns | Yes (DOM tree) | String parsing, type conversion |
| Protocol Buffers | 200-800 ns | Yes (message object) | Varint decoding, field skipping |
| Flat Buffer (zero-copy) | 5-20 ns | No | Pointer arithmetic only |
**The 10x-100x difference** comes from eliminating:
1. String parsing (finding quotes, unescaping)
2. Type conversion (string to number)
3. Memory allocation (DOM construction)
4. Field name lookup (string comparison)
### Detailed Breakdown
```cpp
// What JSON parsing does for each field:
// 1. Find field name in JSON: O(n) string scan
// 2. Compare field name: multiple character comparisons
// 3. Skip whitespace
// 4. Parse value type
// 5. Convert string to typed value
// 6. Allocate storage in DOM
// What flat buffer does for each field:
// 1. Load offset from vtable: one memory access
// 2. Add offset to base: one add instruction
// 3. Load value: one memory access
// Total: 2-3 instructions
```
---
## The Three-Level View: How Zero-Copy Serialization Works
### Level 1 — Application
From your code's perspective, flat buffer access looks like method calls:
```cpp
TradeOrderView order(buffer, offset);
double price = order.price();  // Returns double directly
```
The generated accessor methods are inlined by the compiler, resulting in direct memory access.
### Level 2 — OS/Kernel
The OS is minimally involved:
- **Memory mapping**: The buffer is already mapped (from ring buffer setup)
- **No syscalls**: Accessing fields doesn't touch the kernel
- **Page faults**: Only on first access to a new page
This is the key difference from file-based serialization—no `read()` syscalls, no kernel buffer copies.
### Level 3 — Hardware
Here's what the CPU actually does:
```cpp
double price = order.price();
```
Compiles to (x86-64):
```asm
; Assume rdi contains buffer pointer, rsi contains table offset
mov eax, dword ptr [rdi + rsi - 4]  ; Load vtable offset (relative)
lea rcx, [rdi + rsi + eax]          ; Compute vtable address
movzx eax, word ptr [rcx + 10]      ; Load field offset from vtable (field 3 = offset 10)
test eax, eax                       ; Check if field present
je .return_default                  ; If 0, return default
movsd xmm0, qword ptr [rdi + rsi + eax - 4]  ; Load double at computed offset
ret
.return_default:
xorps xmm0, xmm0                    ; Return 0.0
ret
```
That's about 6-8 instructions for field access. Compare to JSON parsing which might execute **hundreds of instructions** per field.
### Cache Behavior
Flat buffers are cache-friendly:
- **Sequential access**: Reading fields sequentially hits the same cache line
- **No pointer chasing** (for inline fields): The data is right there
- **Vtable is shared**: All instances of the same table type share one vtable
```cpp
// Cache-friendly: sequential field access
for (const auto& order : orders) {
    sum += order.price();     // Likely in same cache line as other fields
    count += order.quantity();
}
// Cache-unfriendly: random string access
for (const auto& order : orders) {
    process(order.symbol());  // Each symbol is a separate cache miss
}
```
---
## Design Decisions: Why This, Not That
### Flat Buffers vs Protocol Buffers
| Aspect | Flat Buffers | Protocol Buffers |
|--------|--------------|------------------|
| **Access model** | Zero-copy, direct access | Parse into object |
| **Parsing** | None (offsets are constants) | Required (varint decoding) |
| **Memory allocation** | None (read in place) | Required (message object) |
| **Random field access** | O(1) via vtable | O(n) to skip unknown fields |
| **Schema evolution** | VTable versioning | Field numbers + wire types |
| **Binary size** | Larger (offsets, padding) | Smaller (varints, dense) |
| **Used by** | Games, HFT, real-time | RPC, config files |
**Choice**: Flat Buffers for our use case because:
- Ring buffer already has the bytes in memory
- Zero-copy access eliminates the parse step entirely
- Random access matters (you might only need some fields)
### Flat Buffers vs Cap'n Proto
| Aspect | Flat Buffers | Cap'n Proto |
|--------|--------------|-------------|
| **Pointer format** | 32-bit offsets | 64-bit far pointers |
| **Arena allocation** | No | Yes (builder arena) |
| **Streaming** | Requires full message | Can stream segments |
| **Inter-machine** | Requires canonicalization | Designed for network |
| **Used by** | Games, local IPC | Cloud, distributed |
**Choice**: Flat Buffers for same-machine IPC (simpler, smaller offsets). Cap'n Proto for distributed systems.
### Inlining vs Pointer Chasing
| Approach | Pros | Cons |
|----------|------|------|
| **Inline all data** | Cache-friendly, no indirection | Fixed size, wasted space |
| **Pointers to data** | Variable size, flexible | Cache misses, indirection |
| **Hybrid (flat buffer)** | Small data inline, large by reference | Complexity |
**Choice**: Hybrid. Small scalar fields (int, double) are inline. Variable-size data (strings, arrays) are referenced by offset.
---
## Knowledge Cascade: What You've Unlocked
**1. Cap'n Proto and Protobuf ZeroCopy (Industry Standards)**
The offset-based access pattern you've implemented is the same one used by Cap'n Proto ("infinity times faster than Protobuf") and Protobuf's ZeroCopy stream readers. The principle is universal: **the binary layout IS the data structure**. Cap'n Proto extends this to network transmission—the bytes on the wire are exactly the bytes in memory. You've essentially built a simplified version of these production-grade formats.
**2. GPU Memory Layout (Cross-Domain: Game Dev / ML)**
The alignment and padding rules you've mastered apply directly to GPU programming. When you declare a struct in GLSL/HLSL or define a uniform buffer, the same alignment constraints apply: `vec3` needs 16-byte alignment, `mat4` needs 16-byte alignment, floats need 4-byte alignment. Understanding flat buffers helps you understand why GPU structures are laid out the way they are—and why naive C-to-GPU struct copying can silently corrupt data.
**3. Database Storage Engines (Cross-Domain: Databases)**
The slotted page format used by PostgreSQL and InnoDB is essentially a flat buffer with variable-length records. Each page has:
- A header with metadata
- A slot array (like our vtable) pointing to record offsets
- Records stored from the end of the page growing backward
When you read a row from a database, the storage engine doesn't "parse" the row—it computes offsets and reads fields directly from the page buffer. The skills you've learned here transfer directly to understanding database internals.
**4. Schema Evolution Patterns**
The vtable approach to schema evolution is used across the industry:
- **Protocol Buffers**: Field numbers + wire types
- **Thrift**: Field IDs + type IDs
- **Avro**: Schema resolution at read time
- **Flat Buffers**: VTable with field offsets
The common principle: **never break existing readers**. New fields are ignored by old code. Removed fields leave holes in the ID space. Default values handle missing fields.
**5. SIMD-Friendly Data Layout**
Flat buffers naturally pack homogeneous arrays contiguously, which is perfect for SIMD vectorization:
```cpp
// Flat buffer vector: contiguous doubles
auto bids = market_data.bids();  // VectorView<double>
// Can be processed with AVX/SSE directly
__m256d prices = _mm256_load_pd(&bids[0]);  // Load 4 prices at once
```
This is why flat buffers are popular in game engines and HFT systems—the data layout enables vectorized processing without any restructuring.
---
## Implementation Checklist
Before moving to Milestone 3, ensure you have:
1. **Schema Definition Language**
   - Parse table definitions with field names, types, and default values
   - Support scalar types (int8-64, uint8-64, float, double, bool)
   - Support string and vector types
   - Support nested table references
2. **Code Generator**
   - Generate accessor classes (View) with inline field access
   - Generate builder classes for message construction
   - Generate vtables for each table type
   - Handle alignment and padding correctly
3. **Zero-Copy Access**
   - Field access compiles to pointer arithmetic (no parsing)
   - VTable lookup handles missing fields with defaults
   - Nested table access via offset indirection
   - Vector/string access via length-prefixed pointers
4. **Schema Evolution**
   - Field IDs assigned by schema position
   - Old readers ignore new fields
   - New readers use defaults for missing fields
   - Field IDs never reused
5. **Integration with Ring Buffer**
   - Message header with type ID and size
   - Serialize into ring buffer slot directly
   - Deserialize by creating View over slot memory
   - No copies between ring buffer and application
6. **Benchmarks**
   - Compare JSON parsing vs flat buffer access
   - Measure latency percentiles (p50, p99, p999)
   - Demonstrate 10x+ improvement
   - Profile cache behavior and memory access patterns
---
<!-- END_MS -->


<!-- MS_ID: zcmb-m3 -->
# Milestone 3: Multi-Producer Multi-Consumer
## The Problem: When CAS Becomes Your Enemy
You've built the SPSC ring buffer. It's beautiful—sub-microsecond latency, zero contention, elegant memory barriers. The producer writes `head`, the consumer writes `tail`, and they never step on each other's cache lines. It's almost peaceful.
Now reality hits. Your trading system doesn't have one network handler. It has eight—one per network interface card, each receiving 250K messages per second. Your analytics pipeline doesn't have one consumer. It has four—one for real-time alerts, one for persistence, one for aggregation, one for replay.
**The naive approach**: "I'll just use CAS instead of load/store. Lock-free is lock-free, right?"
```cpp
// The "obvious" MPMC extension
void produce_mpmc(RingBuffer* ring, const void* msg, size_t size) {
    uint64_t pos = ring->head.fetch_add(1, std::memory_order_relaxed);
    uint64_t slot = pos & ring->mask;
    // Wait for this slot to be consumed
    while (ring->sequence[slot].load(std::memory_order_acquire) != pos) {
        __builtin_ia32_pause();
    }
    memcpy(ring->slots[slot], msg, size);
    ring->sequence[slot].store(pos + 1, std::memory_order_release);
}
```
This looks reasonable. `fetch_add` atomically claims a slot. Each slot has a sequence number to coordinate producer-consumer handoff. What could go wrong?
**Everything.**
Here's what happens with 8 producers contending for the same ring buffer:
| Producers | CAS Success Rate | Avg Retries | Latency P99 |
|-----------|------------------|-------------|-------------|
| 1 (SPSC) | 100% | 0 | 200 ns |
| 2 | 85% | 0.2 | 450 ns |
| 4 | 60% | 1.1 | 1.2 μs |
| 8 | 25% | 4.8 | 8.5 μs |
With 8 producers, **75% of CAS operations fail**. Each failure means the CPU wasted cycles on a failed atomic operation, the cache line was invalidated by another core's successful CAS, and the producer must retry—re-reading the cache line that's now being hammered by 7 other producers.

![Hardware Soul: Cache Line Movement](./diagrams/diag-global-004.svg)

The cache coherency traffic alone can saturate the memory bus. On a dual-socket system, the inter-socket link (UPI or QPI) becomes the bottleneck—you're not limited by your algorithm, you're limited by physics.
**The tension**: You need multiple producers and consumers for throughput and redundancy. But sharing a single data structure between N writers creates exponential contention. The lock-free algorithms that work beautifully for 1 writer become pathological for 8.
**The deeper tension**: Lock-free doesn't mean wait-free. A lock-free algorithm guarantees system-wide progress—*some* thread makes progress—but individual threads can starve indefinitely. With 8 aggressive producers, a slightly slower producer might never successfully claim a slot.
---
## The Revelation: Lock-Free MPMC Is a Different Beast
Here's the misconception that ruins MPMC implementations: "Lock-free means no waiting. I just use atomic operations instead of locks, and everything works the same but faster."
This is catastrophically wrong. Let's trace through what actually happens.
### The ABA Problem: When CAS Lies to You
The Compare-And-Swap (CAS) operation is the foundation of lock-free programming. It atomically updates a value *only if* it still equals an expected value:
```cpp
bool cas(std::atomic<uint64_t>* addr, uint64_t expected, uint64_t desired) {
    // Pseudocode for what CAS does
    uint64_t current = addr->load();
    if (current == expected) {
        addr->store(desired);
        return true;
    }
    return false;
}
```
The assumption is: if `current == expected`, then *nothing changed* between your read and your write. This assumption is **false** in MPMC scenarios.

> **🔑 Foundation: The ABA problem occurs when a value changes A→B→A between read and CAS**
> 
> ## What It Is
The ABA problem is a subtle concurrency bug that afflicts compare-and-swap (CAS) operations. Here's the scenario:
1. Thread A reads a shared pointer value `X`
2. Thread A gets preempted
3. Thread B changes the pointer from `X` to `Y`, then frees `X`
4. Thread B (or another thread) allocates new memory that happens to be at address `X`
5. Thread A resumes, performs CAS comparing against `X` — it succeeds because the value "looks" unchanged
The CAS succeeds *incorrectly* because it only checks the value, not whether the state has actually remained stable. The intermediate changes (X→Y→X) are invisible to the CAS.
**Concrete example:** A lock-free stack using a "top" pointer. Thread A reads top=ptr_A, gets suspended. Thread B pops A, pushes C (which happens to be allocated at the same address as A). Thread A's CAS(top, ptr_A, ptr_A->next) succeeds — but `ptr_A->next` now points to garbage because the node was recycled.
## Why You Need It Right Now
If you're building or debugging lock-free data structures, the ABA problem will bite you. It's particularly insidious because:
- **It's rare and timing-dependent** — may not appear in testing
- **It corrupts silently** — no crash, just wrong data
- **Standard CAS doesn't help** — CAS only compares values, not history
Epoch-based solutions solve this by ensuring memory isn't recycled while any thread might still hold references to it.
## Key Insight
**CAS checks *equality of value*, not *stability of state*.** Think of it like checking if a hotel room number is still "302" — that doesn't tell you if the occupant changed while you weren't looking. The room number is the same, but the context is completely different.
Epoch-based reclamation adds the missing "version information" by guaranteeing that even if memory is logically freed, it won't be physically recycled until no thread could possibly still be referencing it.

Here's the concrete failure mode:
```
Time    Producer A              Producer B              Consumer              slot[5].seq
----    ---------              ---------              --------              -----------
T1      read head = 5                                                        5
T2      (about to CAS...)                                                     
T3                             CAS: 5 → 6, success                          6
T4                                                    read slot[5], seq=6   6
T5                                                    process message       
T6                                                    CAS: seq 6 → 7        7
T7      CAS: 5 → 6, success!                              (head now 6)      6
T8      write to slot[5]                                                      6
```
**What happened:**
1. Producer A read `head = 5`
2. Producer B claimed slot 5, wrote to it, consumer read it
3. The consumer advanced, slot 5 became available again
4. Producer A's CAS `5 → 6` succeeds—but slot 5 now contains a *different* message that hasn't been consumed yet!
5. Producer A overwrites a message that another producer just wrote
The CAS succeeded because the value was 5 again. But it was 5 for a *different reason*—the ring buffer wrapped around. Producer A just corrupted the queue.
This isn't theoretical. I've seen production systems lose messages intermittently for months before someone traced it to ABA.
### The Contention Explosion
Even without ABA, MPMC contention is brutal. Consider what happens when 8 producers all try to claim the next slot:
```
Cycle 0: All 8 producers read head = 100
Cycle 1: All 8 producers compute next_head = 101
Cycle 2: Producer 3's CAS succeeds, head = 101
         Producers 0,1,2,4,5,6,7's CAS fails (head is now 101, not 100)
Cycle 3: Failed producers re-read head = 101
Cycle 4: Producer 1's CAS succeeds, head = 102
         Producers 0,2,4,5,6,7's CAS fails
Cycle 5: Failed producers re-read head = 102
...
```
At any given moment, **7 out of 8 producers are failing**. Each failure:
1. Pollutes the cache with a failed write attempt
2. Requires re-reading the cache line (which another core just modified)
3. Triggers cache coherency traffic (invalidation + re-fetch)
The cache line containing `head` bounces between all 8 cores like a ping-pong ball at supersonic speed. This is **cache line thrashing**, and it kills performance.
### The Starvation Problem
Lock-free guarantees *someone* makes progress. It doesn't guarantee *you* make progress.
Consider this scenario:
- Producer A is on core 0 (dedicated, no other work)
- Producer B is on core 1 (shared with network interrupts)
- Both are spinning on the same `head` variable
Producer A, being faster, will win the CAS race 90% of the time. Producer B gets 10%. If the message rate increases, Producer B might get *nothing*—it keeps failing CAS, falling further behind, while Producer A hogs the queue.
This isn't just unfair. It's a correctness issue: if Producer B's messages are high-priority (e.g., order cancellations), they might be delayed indefinitely while Producer A's low-priority messages flow freely.
---
## The Architecture: Three Approaches to MPMC

![System Satellite Map: Zero-Copy Message Bus](./diagrams/diag-global-001.svg)

There are three fundamental strategies for MPMC queues, each with different trade-offs:
### Approach 1: Array-Based with Sequence Numbers (Dmitry Vyukov's Algorithm)
The classic MPMC queue uses per-slot sequence numbers to solve ABA and coordinate producers/consumers:
```cpp
struct MpmcRingBuffer {
    alignas(64) std::atomic<uint64_t> head;      // Next dequeue position
    alignas(64) std::atomic<uint64_t> tail;      // Next enqueue position
    std::atomic<uint64_t>* sequence;             // Per-slot sequence numbers
    uint8_t* slots;
    uint64_t mask;
    uint64_t capacity;
};
```
The algorithm:
1. **Producer**: `fetch_add` on `tail` to claim a slot, spin until slot's sequence equals claimed position, write data, update sequence to `position + 1`
2. **Consumer**: `fetch_add` on `head` to claim a slot, spin until slot's sequence equals `position + 1`, read data, update sequence to `position + capacity`
This is correct and reasonably efficient, but:
- **Still has CAS contention** on `head` and `tail`
- **Per-slot sequence numbers** add memory overhead
- **Spin waiting** burns CPU when the queue is full/empty
### Approach 2: Sharded Buffers (Partition-Based)
Instead of one queue, use N queues. Producers pick a shard based on thread ID, hash of message, or round-robin:
```cpp
struct ShardedMpmcQueue {
    SpscRingBuffer shards[NUM_SHARDS];
    std::atomic<uint32_t> next_shard{0};
};
```
Pros:
- **Minimal contention**: Each shard is SPSC (or MPSC)
- **Predictable latency**: No exponential backoff from CAS failures
- **Cache-friendly**: Each producer owns its cache lines
Cons:
- **Ordering not preserved** across shards
- **Load imbalance** possible (one hot shard)
- **Consumer complexity**: Must check all shards
### Approach 3: Hybrid with Lightweight Locks
For many workloads, a carefully optimized mutex outperforms lock-free algorithms under high contention:
```cpp
struct HybridMpmcQueue {
    alignas(64) std::atomic<uint32_t> lock{0};
    uint64_t head;
    uint64_t tail;
    uint8_t* slots;
    // ... rest of buffer
};
```
Modern mutexes (especially `std::mutex` on Linux with futex) are surprisingly efficient:
- **Uncontended lock**: ~25ns (one atomic CAS)
- **Contended lock**: Kernel puts thread to sleep, no CPU waste
- **Fairness**: Kernel scheduler ensures no starvation
The trade-off: latency spikes when threads are descheduled (can be 10-100μs).
---
## Implementation: The Vyukov MPMC Queue
Let's implement the production-grade MPMC queue. This is the algorithm used in many high-performance systems, including the Linux kernel's `kfifo` in certain modes.
```cpp
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
class MpmcRingBuffer {
public:
    // Constructor: allocate buffer with power-of-2 capacity
    explicit MpmcRingBuffer(size_t capacity, size_t slot_size)
        : capacity_(next_power_of_two(capacity)),
          mask_(capacity_ - 1),
          slot_size_(slot_size)
    {
        if (capacity < 2 || (capacity & (capacity - 1)) != 0) {
            throw std::invalid_argument("Capacity must be power of 2");
        }
        // Allocate slots
        slots_ = new uint8_t[capacity_ * slot_size_];
        // Allocate sequence numbers (one per slot)
        sequence_ = new std::atomic<uint64_t>[capacity_];
        // Initialize sequence numbers: slot[i] initially expects sequence i
        for (size_t i = 0; i < capacity_; ++i) {
            sequence_[i].store(i, std::memory_order_relaxed);
        }
        // Initialize head and tail
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }
    ~MpmcRingBuffer() {
        delete[] slots_;
        delete[] sequence_;
    }
    // Producer: try to enqueue without blocking
    // Returns true on success, false if queue is full
    bool try_enqueue(const void* data, size_t size) {
        uint64_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t seq = sequence_[pos & mask_].load(std::memory_order_acquire);
            int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);
            if (diff == 0) {
                // Slot is ready for us (sequence == position)
                // Try to claim it
                if (tail_.compare_exchange_weak(pos, pos + 1, 
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    // Successfully claimed!
                    break;
                }
                // CAS failed, another producer beat us
                // pos is updated by CAS failure, retry
            } else if (diff < 0) {
                // Slot is behind (queue is full)
                return false;
            } else {
                // Slot is ahead (another producer claimed but hasn't written)
                // Reload position and retry
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
        // We own the slot at (pos & mask_)
        size_t slot_idx = pos & mask_;
        void* slot = get_slot(slot_idx);
        size_t copy_size = std::min(size, slot_size_);
        std::memcpy(slot, data, copy_size);
        // Publish: update sequence to (pos + 1), signaling consumers
        sequence_[slot_idx].store(pos + 1, std::memory_order_release);
        return true;
    }
    // Producer: blocking enqueue with backoff
    void enqueue(const void* data, size_t size) {
        size_t spins = 0;
        while (!try_enqueue(data, size)) {
            exponential_backoff(spins++);
        }
    }
    // Consumer: try to dequeue without blocking
    // Returns true on success, false if queue is empty
    bool try_dequeue(void* out_data, size_t out_size) {
        uint64_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            uint64_t seq = sequence_[pos & mask_].load(std::memory_order_acquire);
            int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                // Slot has data (sequence == position + 1)
                // Try to claim it
                if (head_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
                // CAS failed, another consumer beat us
            } else if (diff < 0) {
                // Slot is empty
                return false;
            } else {
                // Slot is ahead (shouldn't happen in correct implementation)
                pos = head_.load(std::memory_order_relaxed);
            }
        }
        // We own the slot at (pos & mask_)
        size_t slot_idx = pos & mask_;
        const void* slot = get_slot(slot_idx);
        size_t copy_size = std::min(out_size, slot_size_);
        std::memcpy(out_data, slot, copy_size);
        // Release: update sequence to (pos + capacity), signaling producers
        sequence_[slot_idx].store(pos + capacity_, std::memory_order_release);
        return true;
    }
    // Consumer: blocking dequeue with backoff
    void dequeue(void* out_data, size_t out_size) {
        size_t spins = 0;
        while (!try_dequeue(out_data, out_size)) {
            exponential_backoff(spins++);
        }
    }
    size_t capacity() const { return capacity_; }
    size_t slot_size() const { return slot_size_; }
private:
    static size_t next_power_of_two(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }
    static void exponential_backoff(size_t spin_count) {
        if (spin_count < 32) {
            // Spin with pause instruction
            for (size_t i = 0; i < (1 << (spin_count / 4)); ++i) {
                __builtin_ia32_pause();
            }
        } else if (spin_count < 256) {
            // Yield to scheduler
            std::this_thread::yield();
        } else {
            // Sleep briefly
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    void* get_slot(size_t index) {
        return slots_ + (index * slot_size_);
    }
    const void* get_slot(size_t index) const {
        return slots_ + (index * slot_size_);
    }
    // Cache-line aligned head and tail to prevent false sharing
    alignas(64) std::atomic<uint64_t> head_;
    alignas(64) std::atomic<uint64_t> tail_;
    std::atomic<uint64_t>* sequence_;  // Per-slot sequence numbers
    uint8_t* slots_;
    const size_t capacity_;
    const size_t mask_;
    const size_t slot_size_;
};
```
### How the Sequence Numbers Work
The sequence number algorithm is subtle. Let's trace through an example with capacity=4:
```
Initial state:
  head = 0, tail = 0
  sequence = [0, 1, 2, 3]  (slot i expects sequence i)
Producer A calls enqueue():
  1. pos = tail.load() = 0
  2. seq = sequence[0].load() = 0
  3. diff = 0 - 0 = 0  (slot is ready!)
  4. CAS(tail, 0 → 1) succeeds
  5. Write to slot[0]
  6. sequence[0].store(1)  (now expects sequence 1 for consumer)
State after Producer A:
  head = 0, tail = 1
  sequence = [1, 1, 2, 3]
Producer B calls enqueue() simultaneously:
  1. pos = tail.load() = 1
  2. seq = sequence[1].load() = 1
  3. diff = 1 - 1 = 0  (slot is ready!)
  4. CAS(tail, 1 → 2) succeeds
  5. Write to slot[1]
  6. sequence[1].store(2)
State after Producer B:
  head = 0, tail = 2
  sequence = [1, 2, 2, 3]
Consumer calls dequeue():
  1. pos = head.load() = 0
  2. seq = sequence[0].load() = 1
  3. diff = 1 - (0+1) = 0  (data is ready!)
  4. CAS(head, 0 → 1) succeeds
  5. Read from slot[0]
  6. sequence[0].store(0 + 4) = 4  (wraps for next cycle)
State after Consumer:
  head = 1, tail = 2
  sequence = [4, 2, 2, 3]
```
The key insight: **sequence numbers encode both availability and ownership**. A slot with `sequence == position` is empty and ready for a producer. A slot with `sequence == position + 1` has data and is ready for a consumer. After the consumer reads, `sequence == position + capacity` signals the next cycle.
### Why This Avoids ABA
Remember the ABA problem? A value changes A→B→A, making CAS succeed incorrectly.
The sequence number approach makes ABA impossible because **sequence numbers never wrap to their original value within a single buffer cycle**. Each slot's sequence number increases monotonically: 0, 1, 2, 3, ... even after the ring buffer wraps around.
With a 64-bit sequence number and 1 billion messages per second, it would take **584 years** to overflow. The ABA problem is effectively impossible.
---
## Implementation: Sharded MPMC for High Contention
When CAS contention becomes pathological (8+ producers), sharding is often the better approach:
```cpp
#include <atomic>
#include <vector>
#include <thread>
class ShardedMpmcQueue {
public:
    // Configuration
    static constexpr size_t CACHE_LINE_SIZE = 64;
    struct alignas(CACHE_LINE_SIZE) Shard {
        // SPSC ring buffer (from Milestone 1)
        alignas(64) std::atomic<uint64_t> head{0};
        alignas(64) std::atomic<uint64_t> tail{0};
        uint8_t* slots{nullptr};
        size_t slot_size{0};
        size_t mask{0};
        // Padding to prevent false sharing
        char padding[CACHE_LINE_SIZE - 4 * sizeof(size_t)];
    };
    ShardedMpmcQueue(size_t num_shards, size_t shard_capacity, size_t slot_size)
        : num_shards_(num_shards),
          shard_capacity_(next_power_of_two(shard_capacity)),
          slot_size_(slot_size),
          producer_idx_(0),
          consumer_idx_(0),
          consumer_seq_(num_shards, 0)
    {
        shards_.reserve(num_shards);
        for (size_t i = 0; i < num_shards; ++i) {
            Shard shard;
            shard.slot_size = slot_size;
            shard.mask = shard_capacity_ - 1;
            shard.slots = new uint8_t[shard_capacity_ * slot_size];
            shards_.push_back(std::move(shard));
        }
    }
    ~ShardedMpmcQueue() {
        for (auto& shard : shards_) {
            delete[] shard.slots;
        }
    }
    // Producer: round-robin shard selection
    bool try_enqueue(const void* data, size_t size) {
        // Get next shard (atomic increment)
        size_t shard_idx = producer_idx_.fetch_add(1, std::memory_order_relaxed) % num_shards_;
        return enqueue_to_shard(shard_idx, data, size);
    }
    // Producer: hash-based shard selection (for ordering)
    bool try_enqueue_hashed(uint64_t key, const void* data, size_t size) {
        size_t shard_idx = key % num_shards_;
        return enqueue_to_shard(shard_idx, data, size);
    }
    // Consumer: check all shards for available messages
    bool try_dequeue(void* out_data, size_t out_size, size_t& out_shard_idx) {
        // Start from last consumed shard for fairness
        size_t start = consumer_idx_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < num_shards_; ++i) {
            size_t shard_idx = (start + i) % num_shards_;
            if (dequeue_from_shard(shard_idx, out_data, out_size)) {
                out_shard_idx = shard_idx;
                // Update start for next iteration
                consumer_idx_.store((shard_idx + 1) % num_shards_, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }
    // Batch consumer: drain all available messages from all shards
    template<typename Handler>
    size_t dequeue_batch(Handler& handler, size_t max_messages) {
        size_t count = 0;
        std::vector<uint8_t> buffer(slot_size_);
        while (count < max_messages) {
            size_t shard_idx;
            if (!try_dequeue(buffer.data(), slot_size_, shard_idx)) {
                break;
            }
            handler(buffer.data(), slot_size_, shard_idx);
            ++count;
        }
        return count;
    }
private:
    bool enqueue_to_shard(size_t shard_idx, const void* data, size_t size) {
        Shard& shard = shards_[shard_idx];
        uint64_t pos = shard.tail.load(std::memory_order_relaxed);
        uint64_t head = shard.head.load(std::memory_order_acquire);
        // Check if full
        uint64_t next = (pos + 1) & shard.mask;
        if (next == head) {
            return false;  // Shard is full
        }
        // Write data
        void* slot = shard.slots + (pos * slot_size_);
        std::memcpy(slot, data, std::min(size, slot_size_));
        // Publish
        std::atomic_thread_fence(std::memory_order_release);
        shard.tail.store(next, std::memory_order_relaxed);
        return true;
    }
    bool dequeue_from_shard(size_t shard_idx, void* out_data, size_t out_size) {
        Shard& shard = shards_[shard_idx];
        uint64_t pos = shard.head.load(std::memory_order_relaxed);
        uint64_t tail = shard.tail.load(std::memory_order_acquire);
        // Check if empty
        if (pos == tail) {
            return false;  // Shard is empty
        }
        // Read data
        const void* slot = shard.slots + (pos * slot_size_);
        std::memcpy(out_data, slot, std::min(out_size, slot_size_));
        // Publish
        std::atomic_thread_fence(std::memory_order_release);
        shard.head.store((pos + 1) & shard.mask, std::memory_order_relaxed);
        return true;
    }
    static size_t next_power_of_two(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }
    std::vector<Shard> shards_;
    const size_t num_shards_;
    const size_t shard_capacity_;
    const size_t slot_size_;
    alignas(64) std::atomic<uint32_t> producer_idx_;  // Round-robin counter
    alignas(64) std::atomic<uint32_t> consumer_idx_;  // Fairness tracker
    std::vector<uint64_t> consumer_seq_;  // Per-shard sequence tracking
};
```
### Sharding Trade-offs
| Aspect | Single MPMC | Sharded |
|--------|-------------|---------|
| **CAS contention** | O(N²) with N producers | O(N/Shards) |
| **Ordering** | FIFO across all messages | Per-shard FIFO only |
| **Latency variance** | High (exponential backoff) | Low (predictable) |
| **Memory overhead** | Low | Higher (N buffers) |
| **Load balancing** | Automatic | Requires tuning |
**When to use sharding:**
- 8+ producers with high message rate
- Ordering not required across all messages
- Can tolerate slightly higher memory usage
- Need predictable tail latency
**When to use single MPMC:**
- 2-4 producers
- Strict FIFO required
- Memory constrained
- Can tolerate latency variance
---
## Backpressure: When Consumers Can't Keep Up
A critical question: what happens when producers generate messages faster than consumers can process them?
Without backpressure, you have two bad options:
1. **Drop messages** silently (data loss)
2. **Buffer infinitely** (out of memory crash)
A proper MPMC queue needs **backpressure propagation**—a mechanism to signal producers to slow down.
### Approach 1: Blocking Producers
The simplest approach: block producers when the queue is full:
```cpp
void enqueue_with_backpressure(const void* data, size_t size) {
    size_t spins = 0;
    while (!try_enqueue(data, size)) {
        if (spins++ > 1000) {
            // Queue is persistently full - apply backpressure
            std::unique_lock<std::mutex> lock(backpressure_mutex_);
            not_full_cv_.wait(lock, [this] {
                return !this->is_full_approx();
            });
        } else {
            exponential_backoff(spins);
        }
    }
}
// Consumer signals when space is available
void dequeue_with_signal(void* out_data, size_t out_size) {
    bool was_full = is_full_approx();
    dequeue(out_data, out_size);
    if (was_full) {
        // Wake up blocked producers
        not_full_cv_.notify_all();
    }
}
```
### Approach 2: Rate Limiting
Instead of blocking, enforce a maximum production rate:
```cpp
class RateLimitedProducer {
public:
    RateLimitedProducer(size_t max_per_second)
        : interval_ns_(1'000'000'000 / max_per_second),
          last_send_(std::chrono::high_resolution_clock::now())
    {}
    bool try_send(MpmcRingBuffer& queue, const void* data, size_t size) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - last_send_).count();
        if (elapsed < interval_ns_) {
            // Rate limited - drop or queue
            return false;
        }
        if (queue.try_enqueue(data, size)) {
            last_send_ = now;
            return true;
        }
        return false;
    }
private:
    const size_t interval_ns_;
    std::chrono::high_resolution_clock::time_point last_send_;
};
```
### Approach 3: Credit-Based Flow Control
Each consumer grants "credits" to producers. A producer can only send if it has credits:
```cpp
struct CreditBasedFlowControl {
    std::atomic<int32_t> credits{INITIAL_CREDITS};
    // Consumer grants credits after processing
    void grant_credits(int32_t count) {
        credits.fetch_add(count, std::memory_order_release);
    }
    // Producer consumes a credit to send
    bool try_consume_credit() {
        int32_t current = credits.load(std::memory_order_acquire);
        while (current > 0) {
            if (credits.compare_exchange_weak(current, current - 1,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return true;
            }
        }
        return false;  // No credits available
    }
};
```
This is the approach used by Aeron (the high-performance messaging system used in trading):
```cpp
class AeronStyleFlowControl {
public:
    // Producer side
    bool send(const void* data, size_t size) {
        if (!flow_control_.try_consume_credit()) {
            return false;  // Backpressure applied
        }
        return queue_.try_enqueue(data, size);
    }
    // Consumer side
    void process_messages() {
        size_t processed = 0;
        uint8_t buffer[MAX_MSG_SIZE];
        while (queue_.try_dequeue(buffer, sizeof(buffer))) {
            handle_message(buffer);
            ++processed;
        }
        // Grant credits for processed messages
        if (processed > 0) {
            flow_control_.grant_credits(processed);
        }
    }
private:
    MpmcRingBuffer queue_;
    CreditBasedFlowControl flow_control_;
};
```
---
## Fairness: Preventing Starvation
Without explicit fairness mechanisms, aggressive producers/consumers can starve others:
### Producer Fairness
Problem: A fast producer on a dedicated core hogs the queue.
Solution: **Per-producer rate limiting** or **token bucket**:
```cpp
class FairMpmcQueue {
public:
    struct ProducerHandle {
        size_t producer_id;
        TokenBucket rate_limiter;
    };
    bool try_enqueue_fair(ProducerHandle& handle, const void* data, size_t size) {
        // Check if this producer has tokens
        if (!handle.rate_limiter.try_consume()) {
            return false;  // Rate limited for fairness
        }
        return queue_.try_enqueue(data, size);
    }
private:
    MpmcRingBuffer queue_;
};
```
### Consumer Fairness
Problem: A fast consumer processes all messages, leaving others idle.
Solution: **Work stealing with backoff**:
```cpp
class FairConsumer {
public:
    FairConsumer(MpmcRingBuffer& queue, size_t consumer_id)
        : queue_(queue), consumer_id_(consumer_id) {}
    void run() {
        size_t consecutive_empty = 0;
        while (running_) {
            uint8_t buffer[MAX_MSG_SIZE];
            if (queue_.try_dequeue(buffer, sizeof(buffer))) {
                process_message(buffer);
                consecutive_empty = 0;
            } else {
                ++consecutive_empty;
                if (consecutive_empty > 100) {
                    // Queue seems empty - back off to let other consumers check
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }
    }
private:
    void process_message(const void* data);
    MpmcRingBuffer& queue_;
    size_t consumer_id_;
};
```
---
## The Three-Level View: MPMC in Action
### Level 1 — Application
From your code's perspective, the MPMC queue looks like simple enqueue/dequeue:
```cpp
// Producer thread 1
void producer_thread_1() {
    while (running) {
        TradeOrder order = generate_order();
        queue.enqueue(&order, sizeof(order));
    }
}
// Producer thread 2
void producer_thread_2() {
    while (running) {
        MarketData data = receive_market_data();
        queue.enqueue(&data, sizeof(data));
    }
}
// Consumer thread
void consumer_thread() {
    uint8_t buffer[MAX_MSG_SIZE];
    while (running) {
        if (queue.try_dequeue(buffer, sizeof(buffer))) {
            process_message(buffer);
        }
    }
}
```
### Level 2 — OS/Kernel
The kernel's scheduler becomes a factor:
- **Context switches**: When a thread blocks on a full queue, the kernel switches to another thread
- **Scheduler latency**: A blocked thread may not wake immediately when space becomes available
- **NUMA effects**: On multi-socket systems, threads on different sockets have different memory access latencies
For ultra-low latency, you may need to:
- Pin threads to specific cores (`pthread_setaffinity_np`)
- Use real-time scheduling priorities (`SCHED_FIFO`)
- Disable power saving (CPU frequency scaling adds latency variance)
### Level 3 — Hardware

![Memory Map: Shared Memory Layout](./diagrams/diag-global-003.svg)

The hardware reality of MPMC:
**Cache Coherency Traffic (MESI Protocol)**:
1. Producer A on Core 0 writes `tail` → cache line in Modified state on Core 0
2. Producer B on Core 1 tries to write `tail` → cache miss, must invalidate Core 0's copy
3. Core 0 writes back to L3 (or memory), Core 1 fetches
4. This takes 40-100 cycles on same socket, 100-300 cycles across sockets
**Atomic Instruction Costs**:
| Operation | Cycles (uncontended) | Cycles (contended) |
|-----------|---------------------|-------------------|
| `load` | 1-4 | 1-4 |
| `store` | 1-4 | 1-4 |
| `CAS` | 10-20 | 50-200+ |
| `fetch_add` | 10-20 | 50-200+ |
The `LOCK` prefix (which CAS and `fetch_add` use) locks the cache line, preventing other cores from accessing it during the operation. Under contention, this creates a serialization point.
**Memory Bandwidth**:
Each failed CAS still performs a memory access. With 8 producers each failing 75% of the time at 1M msg/s:
- 8 × 1M × 4 (retries avg) × 64 bytes (cache line) = ~2 GB/s of wasted bandwidth
This is why sharding helps: it reduces the number of cores hammering each cache line.
---
## Benchmarking: Measuring Contention
You can't optimize what you don't measure. Here's how to benchmark MPMC contention:
```cpp
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
struct ContentionMetrics {
    size_t total_operations;
    size_t successful_operations;
    size_t cas_failures;
    size_t spins;
    double avg_latency_ns;
    double p99_latency_ns;
    double throughput_ops_per_sec;
};
class MpmcBenchmark {
public:
    static ContentionMetrics run_benchmark(
            MpmcRingBuffer& queue,
            size_t num_producers,
            size_t num_consumers,
            size_t messages_per_producer,
            size_t message_size) {
        std::atomic<size_t> total_produced{0};
        std::atomic<size_t> total_consumed{0};
        std::atomic<size_t> cas_failures{0};
        std::atomic<size_t> total_spins{0};
        std::atomic<bool> done{false};
        std::vector<double> producer_latencies;
        std::vector<double> consumer_latencies;
        std::mutex latency_mutex;
        // Producer threads
        std::vector<std::thread> producers;
        for (size_t p = 0; p < num_producers; ++p) {
            producers.emplace_back([&, p] {
                std::vector<uint8_t> msg(message_size, static_cast<uint8_t>(p));
                size_t local_failures = 0;
                size_t local_spins = 0;
                for (size_t i = 0; i < messages_per_producer; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();
                    size_t spins = 0;
                    while (!queue.try_enqueue(msg.data(), msg.size())) {
                        ++spins;
                        if (spins > 1000000) {
                            ++local_failures;
                            break;
                        }
                        __builtin_ia32_pause();
                    }
                    local_spins += spins;
                    auto end = std::chrono::high_resolution_clock::now();
                    double ns = std::chrono::duration<double, std::nano>(end - start).count();
                    {
                        std::lock_guard<std::mutex> lock(latency_mutex);
                        producer_latencies.push_back(ns);
                    }
                    total_produced.fetch_add(1, std::memory_order_relaxed);
                }
                cas_failures.fetch_add(local_failures, std::memory_order_relaxed);
                total_spins.fetch_add(local_spins, std::memory_order_relaxed);
            });
        }
        // Consumer threads
        std::vector<std::thread> consumers;
        for (size_t c = 0; c < num_consumers; ++c) {
            consumers.emplace_back([&] {
                std::vector<uint8_t> buffer(message_size);
                while (!done.load(std::memory_order_relaxed) ||
                       total_consumed.load(std::memory_order_relaxed) < 
                       num_producers * messages_per_producer) {
                    auto start = std::chrono::high_resolution_clock::now();
                    if (queue.try_dequeue(buffer.data(), buffer.size())) {
                        auto end = std::chrono::high_resolution_clock::now();
                        double ns = std::chrono::duration<double, std::nano>(end - start).count();
                        {
                            std::lock_guard<std::mutex> lock(latency_mutex);
                            consumer_latencies.push_back(ns);
                        }
                        total_consumed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }
        auto bench_start = std::chrono::high_resolution_clock::now();
        // Wait for completion
        for (auto& t : producers) t.join();
        done.store(true, std::memory_order_relaxed);
        for (auto& t : consumers) t.join();
        auto bench_end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(bench_end - bench_start).count();
        // Compute statistics
        ContentionMetrics metrics;
        metrics.total_operations = num_producers * messages_per_producer;
        metrics.successful_operations = total_consumed.load();
        metrics.cas_failures = cas_failures.load();
        metrics.spins = total_spins.load();
        if (!producer_latencies.empty()) {
            std::sort(producer_latencies.begin(), producer_latencies.end());
            metrics.avg_latency_ns = std::accumulate(
                producer_latencies.begin(), producer_latencies.end(), 0.0) / producer_latencies.size();
            metrics.p99_latency_ns = producer_latencies[
                static_cast<size_t>(producer_latencies.size() * 0.99)];
        }
        metrics.throughput_ops_per_sec = metrics.successful_operations / elapsed_sec;
        return metrics;
    }
};
```
### Expected Results: Single MPMC vs Sharded
On a modern 8-core x86 system:
**Single MPMC Queue (Vyukov algorithm)**:
| Producers | Consumers | Throughput | P99 Latency | CAS Fail Rate |
|-----------|-----------|------------|-------------|---------------|
| 1 | 1 | 12 M/s | 250 ns | 0% |
| 2 | 1 | 9 M/s | 600 ns | 15% |
| 4 | 1 | 6 M/s | 2.1 μs | 42% |
| 8 | 1 | 3.5 M/s | 12 μs | 68% |
**Sharded Queue (8 shards)**:
| Producers | Consumers | Throughput | P99 Latency | CAS Fail Rate |
|-----------|-----------|------------|-------------|---------------|
| 1 | 1 | 11 M/s | 300 ns | 0% |
| 2 | 1 | 10 M/s | 450 ns | 5% |
| 4 | 1 | 9 M/s | 800 ns | 12% |
| 8 | 1 | 8 M/s | 1.5 μs | 18% |
The sharded approach maintains **consistent performance** as producer count increases, while single MPMC degrades significantly.
---
## Design Decisions: Why This, Not That
### Lock-Free vs Lock-Based MPMC
| Aspect | Lock-Free | Lock-Based (Mutex) |
|--------|-----------|-------------------|
| **Uncontended latency** | 50-100 ns | 25-50 ns |
| **Contended latency** | 500 ns - 50 μs | 1-10 μs |
| **Worst-case latency** | Unbounded (starvation) | Bounded (scheduler) |
| **CPU usage when idle** | Spin (100% core) | Sleep (0% core) |
| **Fairness** | Not guaranteed | Guaranteed (kernel) |
| **Implementation complexity** | High | Low |
**Choice**: Use lock-free when:
- You have dedicated cores (no other work)
- You need sub-microsecond latency
- Message rate is consistently high
Use lock-based when:
- You share cores with other work
- Latency can tolerate 10μs variance
- Simplicity matters
### Sharding Granularity
| Shards | Pros | Cons |
|--------|------|------|
| 1 | Simple, ordered | High contention |
| N = num_producers | Minimal contention | Memory overhead |
| N = num_producers × 2 | Load balancing | More memory |
**Choice**: Start with `N = num_producers`. Add more shards if you see load imbalance.
### Backpressure Strategy
| Strategy | Latency | CPU Usage | Correctness |
|----------|---------|-----------|-------------|
| **Drop messages** | Best | Lowest | Data loss |
| **Block producers** | Medium | Medium | No loss |
| **Credit-based** | Best | Low | No loss |
| **Rate limit** | Good | Low | Possible loss |
**Choice**: Credit-based for trading systems (no loss, bounded latency). Rate limiting for telemetry (loss acceptable).
---
## Knowledge Cascade: What You've Unlocked
**1. ABA Problem and Epoch-Based Reclamation**
The sequence number solution you've implemented is one approach to ABA. The broader concept—**epoch-based reclamation**—is used in lock-free memory allocators, hazard pointers, and RCU (Read-Copy-Update) in the Linux kernel. The principle is universal: don't reuse a resource until you're sure no one has a stale reference to it.

> **🔑 Foundation: Hazard pointers and epoch-based reclamation are memory reclamation schemes for lock-free data structures that prevent use-after-free by tracking which memory locations threads are actively accessing**
> 
> ## What It Is
Both schemes solve the core problem in lock-free programming: **when is it safe to free memory?** In a lock-free world, Thread A might be traversing a node while Thread B removes and frees it — classic use-after-free.
### Hazard Pointers (Per-Pointer Tracking)
Each thread maintains a small set of "hazard pointers" — slots where it publishes which memory locations it's currently accessing. Before freeing memory, a thread must scan all hazard pointers from all threads to ensure nobody is still using it.
```
Thread A (reading):     hp[0] = &node->data  // "I'm using this node"
Thread B (freeing):     if (node not in any hp) free(node)
```
### Epoch-Based Reclamation (Global Generational Tracking)
Threads announce when they enter/exit critical sections by reading/writing a global epoch counter. Memory freed during epoch N is kept in a limbo list until *all threads* have moved past epoch N.
```
Thread A:               local_epoch = global_epoch  // "I'm active in epoch 3"
Thread B:               retire(node) // goes to epoch-3 limbo list
System:                 when all threads >= epoch 4, free epoch-3 limbo
```
## Why You Need It Right Now
You cannot build production-quality lock-free data structures without a reclamation scheme. The alternatives are:
- **Never free memory** — memory leak
- **Use locks** — defeats the purpose of lock-free
- **Hope for the best** — undefined behavior waiting to happen
Epoch-based reclamation is typically **2-10x faster** than hazard pointers for read-heavy workloads because it batches reclamation work. Hazard pointers have lower memory overhead and more predictable latency.
## Key Insight
**These schemes trade *deferred reclamation* for *safe concurrent access*.** Think of it like coat check at a club — you can't reuse a coat hanger until you're certain the customer has left the building. Hazard pointers track each coat individually; epoch-based schemes check "has everyone from the 9pm crowd left?"
The choice between them is usually:
- **Hazard pointers**: Predictable, fine-grained, better for heterogeneous workloads
- **Epoch-based**: Faster throughput, worse tail latency, better for uniform workloads

**2. Database Concurrency Control (Cross-Domain)**
The contention patterns you've seen—multiple writers competing for the same resource—are identical to database concurrency problems:
- **Optimistic Concurrency Control (OCC)**: Like CAS-based MPMC, assume no conflict, retry if wrong. Works well at low contention, degrades at high contention.
- **Pessimistic Locking (Two-Phase Locking)**: Like mutex-based MPMC, acquire locks before accessing. Predictable but can deadlock.
- **Multi-Version Concurrency Control (MVCC)**: Like sharded queues, give each transaction its own "view" of data to avoid conflicts.
The trade-offs are identical: optimistic approaches win at low contention, pessimistic at high contention. Understanding MPMC helps you understand why PostgreSQL uses MVCC, why MySQL InnoDB uses row-level locks, and why distributed databases often choose shard-based architectures.
**3. Contention Profiling**
The CAS failure rate metric you learned to measure is a powerful diagnostic tool. In production systems:
```bash
# Linux perf to measure cache coherency traffic
perf stat -e r10d1,r11d1,r12d1 -p <pid> sleep 10
# r10d1 = MEM_LOAD_RETIRED.L3_MISS (cache misses)
# r11d1 = MEM_STORE_RETIRED.L3_MISS (store misses)
# r12d1 = LOCK_CYCLES (atomic operations)
```
High `LOCK_CYCLES` with low throughput indicates contention. This is how you diagnose "slow" lock-free code that's actually thrashing the memory bus.
**4. Sharding Strategies**
The sharding approach you implemented is the same technique used by:
- **ConcurrentHashMap (Java)**: Segments (shards) with per-segment locks
- **Redis Cluster**: Hash slots distributed across nodes
- **Cassandra**: Partition keys determine node ownership
- **Kafka**: Partitions within topics
The principle: **partition by key to reduce contention**. When N entities compete for one resource, split into K resources and route by key. Contention drops from O(N²) to O((N/K)²).
**5. Backpressure Propagation**
The credit-based flow control you implemented is used throughout distributed systems:
- **TCP**: Receiver advertises window size, sender must not exceed it
- **HTTP/2**: Stream-level flow control with WINDOW_UPDATE frames
- **Kafka**: Consumer lag monitoring to detect backpressure
- **Akka Streams**: Reactive Streams specification with explicit backpressure
The principle: **never let a fast producer overwhelm a slow consumer**. The entire system runs at the speed of the slowest component—embrace it, don't fight it.
---
## Implementation Checklist
Before moving to Milestone 4, ensure you have:
1. **MPMC Ring Buffer (Vyukov Algorithm)**
   - Per-slot sequence numbers for ABA prevention
   - CAS-based slot claiming for producers
   - CAS-based slot claiming for consumers
   - Correct memory ordering (acquire/release semantics)
2. **Contention Management**
   - Exponential backoff in spin loops
   - Configurable spin/yield/sleep thresholds
   - Metrics for CAS failure rate
3. **Fairness Mechanisms**
   - Per-producer rate limiting (optional)
   - Round-robin consumer scheduling (optional)
   - Starvation detection (optional)
4. **Backpressure**
   - Blocking enqueue when queue is full
   - Credit-based flow control (optional)
   - Rate limiting (optional)
5. **Sharded Alternative**
   - Multiple SPSC shards
   - Round-robin or hash-based shard selection
   - Fair consumer across shards
6. **Benchmarks**
   - Throughput at various producer/consumer counts
   - Latency percentiles (p50, p99, p999)
   - CAS failure rate under contention
   - Comparison: single MPMC vs sharded
7. **Correctness Tests**
   - Multi-producer stress test (no lost messages)
   - ABA scenario test (sequence wraparound)
   - Consumer starvation test
   - Backpressure test (queue full scenario)
---
<!-- END_MS -->


<!-- MS_ID: zcmb-m4 -->
# Milestone 4: Publish-Subscribe & Topics
## The Problem: When Broadcasting Becomes Your Bottleneck
You've built the MPMC ring buffer. Multiple producers, multiple consumers, efficient contention handling. You can push millions of messages per second through shared memory with sub-microsecond latency. You're done, right?
Not even close.
Consider your trading system. You have:
- **Market data feeds** streaming price updates for 10,000+ symbols
- **Order management** publishing trade confirmations and order state changes
- **Risk monitoring** subscribing to all trades for position tracking
- **Persistence** subscribing to everything for audit logging
- **Alerting** subscribing to specific conditions (price thresholds, volume spikes)
- **Dashboard** subscribing to aggregated statistics
The naive approach: "Each subscriber creates its own consumer on the MPMC queue. Publishers broadcast to everyone."
```cpp
// The "obvious" broadcast approach
void broadcast_trade(const TradeOrder& order) {
    for (auto& subscriber : subscribers) {
        queue.enqueue(&order, sizeof(order));  // One copy per subscriber!
    }
}
```
At 2 million trades per second with 5 subscribers, you're **writing 10 million messages** to shared memory. Each message is 256 bytes. That's **2.5 GB/s** of memory bandwidth just for copies. Your ring buffer fills 5× faster. Your cache is thrashed 5× harder.
**But wait, it gets worse.**
The risk monitor needs *all* trades. The dashboard only needs *aggregated* stats. The alerting system only needs trades where `price > threshold`. You're sending every message to every subscriber, and most of them immediately discard it.
This is **message fan-out waste**: the combination of unnecessary copies (to separate queues) and unnecessary deliveries (to uninterested subscribers).
**The deeper problem**: What if a new subscriber joins mid-stream? They've missed the last 5 million trades. Do they start from zero? Do you replay history? How do you handle "late joiners" without blocking the live stream?
**And what about ordering?** If publisher A sends to topic "orders" and publisher B sends to topic "market-data", should a subscriber to both see them in timestamp order? In arrival order? What if they're on different shards?
**The tension**: You need flexible routing—publishers send to topics, subscribers receive from topics they care about. But every layer of indirection (topic matching, fan-out, filtering) adds latency. In a zero-copy world, you've worked hard to eliminate copies—now you're about to add them back through routing logic.
**The escape hatch**: What if the same message in shared memory could be read by multiple subscribers without copying? What if topic matching was O(1) instead of O(N)? What if you could filter *before* delivery, not after?
This is the **publish-subscribe pattern**, and implementing it efficiently on a zero-copy message bus requires solving three hard problems simultaneously:
1. **Topic routing** (how to match publishers to subscribers)
2. **Zero-copy fan-out** (how multiple readers share one message)
3. **Message lifecycle** (when can you reuse a slot if 3 of 5 subscribers have read it?)
---
## The Revelation: Topic Matching Is Not O(1)
Here's the misconception that ruins pub/sub implementations: "A topic is just a string. I'll use a hash map from topic to subscriber list. Lookup is O(1). Done."
```cpp
// The "obvious" topic routing
std::unordered_map<std::string, std::vector<Subscriber*>> topic_subscribers;
void publish(const std::string& topic, const void* msg, size_t size) {
    auto it = topic_subscribers.find(topic);
    if (it != topic_subscribers.end()) {
        for (auto* sub : it->second) {
            sub->deliver(msg, size);
        }
    }
}
```
This works for exact matches. But real pub/sub systems need **wildcards**:
- `stocks.*.price` — match any stock's price updates
- `orders.client123.#` — all order events for client 123
- `market.NYSE.>` — everything from NYSE
A hash map can't do wildcard matching. You'd have to:
1. Enumerate all possible matching topics (impossible with wildcards)
2. Or scan every subscription pattern (O(N) where N = number of subscriptions)
At 100,000 subscriptions and 2 million messages/second, an O(N) scan is **200 billion comparisons per second**. Your CPU will melt.

![System Satellite Map: Zero-Copy Message Bus](./diagrams/diag-global-001.svg)

### The Trie Solution: Prefix Trees for Topic Matching
The efficient approach uses a **trie** (prefix tree) to represent the topic hierarchy:
```
                    root
                   /    \
              stocks    orders
              /    \        \
            *      NASDAQ   client123
            |        |         |
          price    price      #
            |        |         |
         [sub A]  [sub B]   [sub C]
```
Topic matching becomes tree traversal:
- `stocks.AAPL.price` → match `stocks.*.price` (sub A)
- `stocks.NASDAQ.price` → match both `stocks.*.price` (sub A) and `stocks.NASDAQ.price` (sub B)
- `orders.client123.new` → match `orders.client123.#` (sub C)
**Time complexity**: O(M) where M = number of topic levels (typically 3-5), not O(N) subscriptions.

> **🔑 Foundation: A tree where each node represents a character or segment**
> 
> **What it IS**
A trie (pronounced "try") is a tree-like data structure where each node represents a character, and the path from the root to any node spells out a string. Think of it as an auto-complete tree: every branch point is a decision about the next character.
```
        (root)
       /   |   \
      c    d    t
      |    |    |
      a    o    o
      |    |    |
      t    g    p
     / \   |    |
   s   t   *    *
   |   |
   *   *
```
The paths `cat`, `cats`, `cat`, `dog`, and `top` are encoded here. The `*` markers indicate complete words.
**Key properties:**
- Lookup, insertion, and deletion are all **O(L)** where L is the length of the string
- This is independent of how many total strings are stored
- Strings with common prefixes share nodes automatically
**WHY you need it right now**
When building autocomplete, spell checkers, IP routing tables, or any system that needs to find strings by prefix, tries are the go-to structure. They let you answer "what words start with 'pre'" in O(L) time for the prefix, then enumerate matches. Compare this to scanning a list of 10,000 words—tries give you instant prefix-based filtering.
They also enable efficient longest-prefix matching (critical for IP routing) and dictionary implementations where memory can be saved by sharing common prefixes.
**ONE key insight**
A trie trades depth for breadth in a specific way: the tree's depth is bounded by the maximum string length, not the number of strings. This means whether you have 100 or 100 million strings, looking up "hello" always takes 5 node traversals (plus any branching overhead at each level).

### The MQTT Standard
Your topic hierarchy design isn't arbitrary—it follows **MQTT** conventions, the industry-standard pub/sub protocol used by AWS IoT, Azure IoT Hub, and millions of IoT devices:
| Wildcard | Meaning | Example |
|----------|---------|---------|
| `+` (single-level) | Matches exactly one level | `stocks/+/price` matches `stocks/AAPL/price` but not `stocks/AAPL/2024/price` |
| `#` (multi-level) | Matches zero or more levels | `stocks/#` matches `stocks/AAPL/price` and `stocks/NYSE/AAPL/price` |
| (no wildcard) | Exact match | `orders/new` matches only `orders/new` |
We'll use MQTT-style wildcards because:
1. **Proven at scale**: Billions of IoT messages per day
2. **Tool compatibility**: Works with existing MQTT brokers for bridging
3. **Clear semantics**: Single-level vs multi-level distinction is precise
---
## The Architecture: Topic Router + Shared Messages

![Three-Level View: Complete Message Flow](./diagrams/diag-global-002.svg)

Our pub/sub layer has three components:
### 1. Topic Registry (Trie-Based Router)
Maps topics to subscriber sets with O(M) lookup:
```cpp
class TopicRegistry {
    struct TrieNode {
        std::unordered_map<std::string, TrieNode*> children;
        std::vector<SubscriberID> exact_subscribers;   // Exact match at this node
        std::vector<SubscriberID> wildcard_subscribers; // + or # subscribers
    };
    TrieNode root_;
public:
    std::vector<SubscriberID> match(const std::string& topic);
    void subscribe(const std::string& pattern, SubscriberID id);
    void unsubscribe(SubscriberID id);
};
```
### 2. Shared Message Slots with Reference Counting
Multiple subscribers read the *same* memory. A reference count tracks how many readers remain:
```cpp
struct alignas(64) SharedMessageSlot {
    std::atomic<uint32_t> ref_count;  // Number of subscribers yet to read
    uint32_t message_size;
    uint64_t publish_timestamp;
    uint8_t data[];  // Flat buffer message
};
```
### 3. Per-Subscriber Cursors
Each subscriber tracks its position independently:
```cpp
struct SubscriberCursor {
    SubscriberID id;
    std::atomic<uint64_t> current_offset;  // Position in shared buffer
    std::vector<std::string> topic_patterns;
};
```
---
## Implementation: The Topic Trie
Let's build the trie-based topic router:
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <algorithm>
using SubscriberID = uint64_t;
class TopicRegistry {
public:
    // Wildcard types (MQTT-style)
    static constexpr char SINGLE_LEVEL_WILDCARD = '+';
    static constexpr char MULTI_LEVEL_WILDCARD = '#';
    static constexpr char TOPIC_SEPARATOR = '/';
private:
    struct TrieNode {
        // Children for exact segment matches
        std::unordered_map<std::string, std::unique_ptr<TrieNode>> exact_children;
        // Child for single-level wildcard (+)
        std::unique_ptr<TrieNode> wildcard_child;
        // Subscribers at this node (exact match)
        std::vector<SubscriberID> subscribers;
        // Subscribers for multi-level wildcard (#) - matches all descendants
        std::vector<SubscriberID> multi_level_subscribers;
        // Read-write lock for thread safety
        mutable std::shared_mutex mutex;
    };
    std::unique_ptr<TrieNode> root_;
    size_t total_subscriptions_ = 0;
public:
    TopicRegistry() : root_(std::make_unique<TrieNode>()) {}
    // Subscribe to a topic pattern (may contain + or #)
    void subscribe(const std::string& pattern, SubscriberID subscriber_id) {
        auto segments = split_topic(pattern);
        std::unique_lock<std::shared_mutex> lock(root_->mutex);
        TrieNode* current = root_.get();
        for (size_t i = 0; i < segments.size(); ++i) {
            const std::string& segment = segments[i];
            if (segment == std::string(1, MULTI_LEVEL_WILDCARD)) {
                // # must be the last segment
                std::unique_lock<std::shared_mutex> node_lock(current->mutex);
                current->multi_level_subscribers.push_back(subscriber_id);
                ++total_subscriptions_;
                return;
            }
            if (segment == std::string(1, SINGLE_LEVEL_WILDCARD)) {
                // + matches exactly one level
                std::unique_lock<std::shared_mutex> node_lock(current->mutex);
                if (!current->wildcard_child) {
                    current->wildcard_child = std::make_unique<TrieNode>();
                }
                current = current->wildcard_child.get();
            } else {
                // Exact segment match
                std::unique_lock<std::shared_mutex> node_lock(current->mutex);
                auto it = current->exact_children.find(segment);
                if (it == current->exact_children.end()) {
                    it = current->exact_children.insert({
                        segment, std::make_unique<TrieNode>()
                    }).first;
                }
                current = it->second.get();
            }
        }
        // Add subscriber at the final node
        std::unique_lock<std::shared_mutex> node_lock(current->mutex);
        current->subscribers.push_back(subscriber_id);
        ++total_subscriptions_;
    }
    // Unsubscribe from all topics
    void unsubscribe(SubscriberID subscriber_id) {
        // For simplicity, we track subscriptions in a flat list
        // Production systems would maintain reverse index
        std::unique_lock<std::shared_mutex> lock(root_->mutex);
        unsubscribe_recursive(root_.get(), subscriber_id);
    }
    // Match a topic and return all matching subscriber IDs
    std::vector<SubscriberID> match(const std::string& topic) const {
        auto segments = split_topic(topic);
        std::vector<SubscriberID> result;
        std::shared_lock<std::shared_mutex> lock(root_->mutex);
        match_recursive(root_.get(), segments, 0, result);
        // Remove duplicates (a subscriber might match multiple patterns)
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
    size_t total_subscriptions() const { return total_subscriptions_; }
private:
    // Split topic into segments
    static std::vector<std::string> split_topic(const std::string& topic) {
        std::vector<std::string> segments;
        size_t start = 0;
        size_t end = topic.find(TOPIC_SEPARATOR);
        while (end != std::string::npos) {
            segments.push_back(topic.substr(start, end - start));
            start = end + 1;
            end = topic.find(TOPIC_SEPARATOR, start);
        }
        segments.push_back(topic.substr(start));
        return segments;
    }
    // Recursive matching
    void match_recursive(const TrieNode* node,
                         const std::vector<std::string>& segments,
                         size_t depth,
                         std::vector<SubscriberID>& result) const {
        if (!node) return;
        std::shared_lock<std::shared_mutex> lock(node->mutex);
        // Add all multi-level wildcard subscribers (#) at this node
        result.insert(result.end(), 
                      node->multi_level_subscribers.begin(),
                      node->multi_level_subscribers.end());
        // If we've consumed all segments, add exact subscribers
        if (depth == segments.size()) {
            result.insert(result.end(),
                          node->subscribers.begin(),
                          node->subscribers.end());
            return;
        }
        const std::string& segment = segments[depth];
        // Try exact match
        auto it = node->exact_children.find(segment);
        if (it != node->exact_children.end()) {
            match_recursive(it->second.get(), segments, depth + 1, result);
        }
        // Try single-level wildcard (+)
        if (node->wildcard_child) {
            match_recursive(node->wildcard_child.get(), segments, depth + 1, result);
        }
    }
    // Recursive unsubscribe (simplified - production would track subscriptions per ID)
    void unsubscribe_recursive(TrieNode* node, SubscriberID subscriber_id) {
        if (!node) return;
        std::unique_lock<std::shared_mutex> lock(node->mutex);
        // Remove from all subscriber lists
        auto remove_id = [subscriber_id](std::vector<SubscriberID>& vec) {
            vec.erase(
                std::remove(vec.begin(), vec.end(), subscriber_id),
                vec.end()
            );
        };
        remove_id(node->subscribers);
        remove_id(node->multi_level_subscribers);
        if (total_subscriptions_ > 0) {
            --total_subscriptions_;
        }
        // Recurse into children
        for (auto& [key, child] : node->exact_children) {
            unsubscribe_recursive(child.get(), subscriber_id);
        }
        if (node->wildcard_child) {
            unsubscribe_recursive(node->wildcard_child.get(), subscriber_id);
        }
    }
};
```
### Trie Complexity Analysis
| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Subscribe | O(M) | O(M) per subscription |
| Unsubscribe | O(S × M) | — |
| Match | O(M + K) | — |
Where M = topic depth (segments), S = total subscriptions, K = matching subscribers.
For typical topics with 3-5 segments, this is effectively **constant time** regardless of subscription count.
---
## Zero-Copy Fan-Out: Reference Counting in Shared Memory
The key insight: **one message, multiple readers**. Instead of copying a message N times for N subscribers, write it once and track how many readers remain.
```cpp
#include <atomic>
#include <cstdint>
#include <cstring>
// Configuration
struct PubSubConfig {
    size_t num_slots;          // Must be power of 2
    size_t slot_size;          // Max message size
    size_t max_subscribers;    // Maximum concurrent subscribers per topic
};
class ZeroCopyPubSubBuffer {
public:
    // Message header stored in each slot
    struct alignas(64) MessageHeader {
        std::atomic<uint32_t> ref_count;     // Readers remaining
        uint32_t message_size;                // Actual message size
        uint64_t publish_timestamp_ns;        // For ordering and TTL
        uint32_t topic_hash;                  // Quick topic filter
        uint32_t sequence;                    // Message sequence number
        uint8_t padding[64 - 4*sizeof(uint32_t) - sizeof(uint64_t)];
    };
    static constexpr size_t HEADER_SIZE = sizeof(MessageHeader);
private:
    struct alignas(64) Slot {
        MessageHeader header;
        uint8_t data[0];  // Flexible array member
    };
    // Ring buffer state
    alignas(64) std::atomic<uint64_t> head_{0};  // Write position
    alignas(64) std::atomic<uint64_t> tail_{0};  // Reclaim position
    Slot* slots_;
    size_t slot_size_;
    size_t num_slots_;
    size_t mask_;
public:
    ZeroCopyPubSubBuffer(const PubSubConfig& config)
        : slot_size_(config.slot_size),
          num_slots_(next_power_of_two(config.num_slots)),
          mask_(num_slots_ - 1)
    {
        // Allocate slots with alignment
        size_t total_size = num_slots_ * (sizeof(MessageHeader) + slot_size_);
        slots_ = static_cast<Slot*>(aligned_alloc(64, total_size));
        std::memset(slots_, 0, total_size);
        // Initialize all ref counts to 0 (no readers)
        for (size_t i = 0; i < num_slots_; ++i) {
            slots_[i].header.ref_count.store(0, std::memory_order_relaxed);
        }
    }
    ~ZeroCopyPubSubBuffer() {
        free(slots_);
    }
    // Publisher: write a message with given subscriber count
    // Returns pointer to message data area, or nullptr if buffer full
    void* try_publish_begin(uint32_t num_subscribers, uint32_t topic_hash, 
                            uint32_t message_size) {
        uint64_t pos = head_.load(std::memory_order_relaxed);
        uint64_t slot_idx = pos & mask_;
        Slot* slot = &slots_[slot_idx];
        // Check if slot is free (ref_count == 0)
        uint32_t ref = slot->header.ref_count.load(std::memory_order_acquire);
        if (ref != 0) {
            return nullptr;  // Slot still has readers
        }
        // Reserve the slot with a CAS
        uint32_t expected = 0;
        if (!slot->header.ref_count.compare_exchange_strong(
                expected, num_subscribers,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return nullptr;  // Lost race
        }
        // Successfully reserved - fill header
        slot->header.message_size = std::min(message_size, 
                                             static_cast<uint32_t>(slot_size_));
        slot->header.publish_timestamp_ns = get_timestamp_ns();
        slot->header.topic_hash = topic_hash;
        slot->header.sequence = static_cast<uint32_t>(pos);
        return slot->data;
    }
    void publish_commit() {
        // Advance head
        head_.fetch_add(1, std::memory_order_release);
    }
    // Subscriber: get next message for reading
    // Returns pointer to data, sets out_size, or nullptr if no message
    const void* subscriber_read(uint64_t subscriber_offset, 
                                 uint32_t& out_size,
                                 uint32_t& out_topic_hash,
                                 uint64_t& out_next_offset) {
        uint64_t slot_idx = subscriber_offset & mask_;
        Slot* slot = &slots_[slot_idx];
        // Check if this slot has valid data
        uint32_t ref = slot->header.ref_count.load(std::memory_order_acquire);
        if (ref == 0) {
            return nullptr;  // No message (empty or already consumed)
        }
        // Read the message
        out_size = slot->header.message_size;
        out_topic_hash = slot->header.topic_hash;
        out_next_offset = subscriber_offset + 1;
        return slot->data;
    }
    // Subscriber: acknowledge message is done (decrement ref count)
    void subscriber_ack(uint64_t subscriber_offset) {
        uint64_t slot_idx = subscriber_offset & mask_;
        Slot* slot = &slots_[slot_idx];
        // Decrement reference count
        uint32_t prev = slot->header.ref_count.fetch_sub(1, std::memory_order_acq_rel);
        // If this was the last reader, try to advance tail for reclamation
        if (prev == 1) {
            try_reclaim();
        }
    }
    // Get current write position (for subscribers to track)
    uint64_t head_position() const {
        return head_.load(std::memory_order_acquire);
    }
    size_t slot_capacity() const { return slot_size_; }
    size_t num_slots() const { return num_slots_; }
private:
    static size_t next_power_of_two(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }
    static uint64_t get_timestamp_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch());
        return ns.count();
    }
    void try_reclaim() {
        // Advance tail past consecutive slots with ref_count == 0
        uint64_t current_tail = tail_.load(std::memory_order_relaxed);
        uint64_t current_head = head_.load(std::memory_order_acquire);
        while (current_tail < current_head) {
            uint64_t slot_idx = current_tail & mask_;
            uint32_t ref = slots_[slot_idx].header.ref_count.load(
                std::memory_order_acquire);
            if (ref != 0) {
                break;  // Slot still has readers
            }
            ++current_tail;
        }
        if (current_tail > tail_.load(std::memory_order_relaxed)) {
            tail_.store(current_tail, std::memory_order_release);
        }
    }
};
```
### The Reference Counting Lifecycle
```
Time    Publisher              Subscriber A          Subscriber B          Slot RefCount
----    ---------              ------------          ------------          -------------
T1      publish(msg, 2 subs)                                               2 (reserved)
T2      write data                                                         2
T3                             read msg                                    2
T4                             ack (ref--)                                 1
T5                                                   read msg              1
T6                                                   ack (ref--)           0 → reclaim
T7      slot now reusable                                                  0
```
**Key insight**: The slot isn't reused until *all* subscribers have acknowledged. Slow subscribers block slot reclamation—this is a fundamental tension.
---
## Per-Subscriber Filtering: The Pre-Delivery Optimization
Even with efficient topic routing, some subscribers receive messages they don't want. A subscriber to `stocks.AAPL.#` might only care about `price` messages, not `volume` or `dividend`.
**Option 1: Post-delivery filtering (wasteful)**
```cpp
// Subscriber receives everything, filters locally
void on_message(const Message& msg) {
    if (msg.type == "price") {
        process(msg);
    }
    // Discard everything else - wasted delivery!
}
```
**Option 2: Pre-delivery filtering (efficient)**
```cpp
// Broker checks filter before delivery
bool should_deliver(const Subscriber& sub, const MessageHeader& header) {
    // Quick hash check
    if ((sub.topic_hash_mask & header.topic_hash) == 0) {
        return false;
    }
    // Full pattern match
    return sub.filter.matches(header);
}
```
We implement filtering as a **predicate** evaluated before incrementing the reference count:
```cpp
#include <functional>
#include <regex>
class MessageFilter {
public:
    // Filter function: returns true if message should be delivered
    using FilterPredicate = std::function<bool(const uint8_t* data, uint32_t size)>;
    // Create a filter from a flat buffer field path and value
    static FilterPredicate create_field_filter(
            const std::string& field_path,
            const std::string& expected_value) {
        // Parse field path (e.g., "order.symbol" or "data.price")
        // Return predicate that extracts field and compares
        return [field_path, expected_value](const uint8_t* data, uint32_t size) {
            // Simplified - would use flat buffer accessors
            // In production: use generated accessor with field offset
            return true;  // Placeholder
        };
    }
    // Create a range filter (e.g., price > 100)
    static FilterPredicate create_range_filter(
            const std::string& field_path,
            double min_val,
            double max_val) {
        return [field_path, min_val, max_val](const uint8_t* data, uint32_t size) {
            // Extract numeric field and check range
            return true;  // Placeholder
        };
    }
    // Combine filters with AND/OR
    static FilterPredicate combine_and(
            std::vector<FilterPredicate> filters) {
        return [filters = std::move(filters)](const uint8_t* data, uint32_t size) {
            for (const auto& f : filters) {
                if (!f(data, size)) return false;
            }
            return true;
        };
    }
};
struct Subscriber {
    SubscriberID id;
    std::vector<std::string> topic_patterns;
    MessageFilter::FilterPredicate filter;
    uint64_t current_offset;      // Position in shared buffer
    uint32_t topic_hash_mask;     // Bloom filter for quick rejection
    bool active;
};
```
### Bloom Filter for Quick Rejection
For ultra-fast filtering, we use a **bloom filter** on topic hashes:

> **🔑 Foundation: A probabilistic data structure that tests set membership with no false negatives but possible false positives. Uses multiple hash functions to set bits in a bit array. Great for quick rejection**
> 
> **What it IS**
A Bloom filter is a space-efficient probabilistic data structure that answers the question "is this item possibly in the set?" It can tell you definitively that something is NOT present, but can only tell you that something MIGHT be present.
**How it works:**
1. Start with a bit array of m bits, all set to 0
2. Choose k different hash functions
3. To add an item: run it through all k hashes, each giving a position in the array. Set those k bits to 1.
4. To check an item: run it through all k hashes. If ANY of those bits are 0, the item was definitely never added. If ALL are 1, the item was probably added (but those bits might have been set by other items).
```
Adding "alice":
hash1("alice") → position 2  →  [0,0,1,0,0,0,0,0]
hash2("alice") → position 5  →  [0,0,1,0,0,1,0,0]
hash3("alice") → position 7  →  [0,0,1,0,0,1,0,1]
Checking "bob":
hash1("bob") → position 1  →  bit is 0  →  DEFINITELY NOT IN SET
Checking "carol":
hash1("carol") → position 2  →  bit is 1 ✓
hash2("carol") → position 5  →  bit is 1 ✓
hash3("carol") → position 7  →  bit is 1 ✓
→ POSSIBLY in set (could be false positive)
```
**WHY you need it right now**
Bloom filters shine as a first-line defense before expensive operations:
- **Databases**: Check if a key exists on disk before doing an I/O operation
- **Web crawlers**: Track which URLs you've already visited without storing every URL
- **Caching layers**: Avoid cache penetration by rejecting queries for data that definitely doesn't exist
- **Spell checkers**: Quickly reject words that definitely aren't in the dictionary
The pattern is always: Bloom filter first → if it says "no", you're done (instant). If it says "maybe", then do the expensive exact check.
**ONE key insight**
A Bloom filter's false positive rate depends on three knobs: the size of the bit array (m), the number of hash functions (k), and the number of items inserted (n). You can tune these at design time. The critical realization is that false positives increase as you add more items, but **false negatives are mathematically impossible**—if all k bits aren't set, that item was never added. This asymmetric guarantee is what makes Bloom filters useful: you can safely skip the expensive check when the filter says "no."

```cpp
class TopicBloomFilter {
    static constexpr size_t NUM_BITS = 256;
    static constexpr size_t NUM_HASHES = 3;
    std::bitset<NUM_BITS> bits_;
public:
    void add(const std::string& topic) {
        for (size_t i = 0; i < NUM_HASHES; ++i) {
            size_t hash = hash_topic(topic, i);
            bits_.set(hash % NUM_BITS);
        }
    }
    bool might_contain(const std::string& topic) const {
        for (size_t i = 0; i < NUM_HASHES; ++i) {
            size_t hash = hash_topic(topic, i);
            if (!bits_.test(hash % NUM_BITS)) {
                return false;  // Definitely not in set
            }
        }
        return true;  // Probably in set (or false positive)
    }
private:
    static size_t hash_topic(const std::string& topic, size_t seed) {
        // Simple hash - production would use xxHash or similar
        size_t h = seed;
        for (char c : topic) {
            h = h * 31 + c;
        }
        return h;
    }
};
```
With a bloom filter, 90%+ of non-matching topics are rejected in **nanoseconds** without touching the trie.
---
## Retained Messages: Handling Late Subscribers
A common requirement: when a subscriber connects, they want the "last known good" value for a topic. For example, a dashboard wants the current price of AAPL, not wait for the next price update.
**Retained messages** are stored per-topic and delivered to new subscribers immediately:
```cpp
class RetainedMessageStore {
    struct RetainedMessage {
        std::vector<uint8_t> data;
        uint64_t timestamp_ns;
        uint32_t topic_hash;
    };
    std::unordered_map<std::string, RetainedMessage> retained_;
    std::shared_mutex mutex_;
    size_t max_message_size_;
public:
    RetainedMessageStore(size_t max_size = 4096)
        : max_message_size_(max_size) {}
    // Store a retained message for a topic
    void store(const std::string& topic, const void* data, uint32_t size) {
        if (size > max_message_size_) return;
        std::unique_lock<std::shared_mutex> lock(mutex_);
        RetainedMessage& msg = retained_[topic];
        msg.data.assign(static_cast<const uint8_t*>(data),
                        static_cast<const uint8_t*>(data) + size);
        msg.timestamp_ns = get_timestamp_ns();
        msg.topic_hash = hash_topic(topic);
    }
    // Get retained message for a topic
    bool get(const std::string& topic, std::vector<uint8_t>& out_data,
             uint64_t& out_timestamp) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = retained_.find(topic);
        if (it == retained_.end()) {
            return false;
        }
        out_data = it->second.data;
        out_timestamp = it->second.timestamp_ns;
        return true;
    }
    // Clear retained message for a topic
    void clear(const std::string& topic) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        retained_.erase(topic);
    }
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return retained_.size();
    }
private:
    static uint64_t get_timestamp_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
    static uint32_t hash_topic(const std::string& topic) {
        uint32_t h = 0;
        for (char c : topic) {
            h = h * 31 + c;
        }
        return h;
    }
};
```
### Retained Message Integration
```cpp
class PubSubBroker {
    TopicRegistry registry_;
    ZeroCopyPubSubBuffer buffer_;
    RetainedMessageStore retained_;
    std::unordered_map<SubscriberID, Subscriber> subscribers_;
public:
    // Publish with retained flag
    void publish(const std::string& topic, const void* data, uint32_t size,
                 bool retain = false) {
        // Find matching subscribers
        auto subscriber_ids = registry_.match(topic);
        if (subscriber_ids.empty() && !retain) {
            return;  // No subscribers, no need to store
        }
        // Store retained message if requested
        if (retain) {
            retained_.store(topic, data, size);
        }
        // Write to shared buffer with ref count
        uint32_t topic_hash = hash_topic(topic);
        void* slot = buffer_.try_publish_begin(
            subscriber_ids.size(), topic_hash, size);
        if (!slot) {
            // Buffer full - apply backpressure
            handle_backpressure();
            return;
        }
        std::memcpy(slot, data, size);
        buffer_.publish_commit();
        // Notify subscribers (in production, use wait-free signaling)
        notify_subscribers(subscriber_ids);
    }
    // Subscribe with retained message delivery
    void subscribe(const std::string& pattern, SubscriberID id,
                   MessageFilter::FilterPredicate filter = nullptr) {
        registry_.subscribe(pattern, id);
        Subscriber sub;
        sub.id = id;
        sub.topic_patterns.push_back(pattern);
        sub.filter = filter;
        sub.current_offset = buffer_.head_position();
        sub.active = true;
        subscribers_[id] = std::move(sub);
        // Deliver retained messages matching this pattern
        deliver_retained_messages(pattern, id);
    }
private:
    void deliver_retained_messages(const std::string& pattern, SubscriberID id) {
        // For exact topics, check retained store
        if (pattern.find('+') == std::string::npos &&
            pattern.find('#') == std::string::npos) {
            std::vector<uint8_t> data;
            uint64_t timestamp;
            if (retained_.get(pattern, data, timestamp)) {
                // Deliver directly to subscriber
                deliver_to_subscriber(id, data.data(), data.size());
            }
        }
        // For wildcards, would need to scan retained store
        // (typically not done for performance reasons)
    }
    void deliver_to_subscriber(SubscriberID id, const void* data, uint32_t size);
    void notify_subscribers(const std::vector<SubscriberID>& ids);
    void handle_backpressure();
};
```
---
## Last-Will Messages: Disconnect Notification
When a subscriber disconnects unexpectedly (crash, network failure), other subscribers may need to know. MQTT calls this a **Last Will and Testament (LWT)** message.
```cpp
struct LastWillConfig {
    std::string topic;       // Topic to publish will to
    std::vector<uint8_t> payload;  // Will message content
    bool retain;             // Should will be retained?
    uint8_t qos;             // Quality of service level
};
class SubscriberManager {
    std::unordered_map<SubscriberID, LastWillConfig> will_configs_;
    std::unordered_map<SubscriberID, std::chrono::steady_clock::time_point> 
        last_heartbeat_;
    std::mutex mutex_;
    PubSubBroker& broker_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};
public:
    SubscriberManager(PubSubBroker& broker) : broker_(broker) {
        monitor_thread_ = std::thread(&SubscriberManager::monitor_loop, this);
    }
    ~SubscriberManager() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    // Register a subscriber with optional last-will
    void register_subscriber(SubscriberID id, 
                             const LastWillConfig& will = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!will.topic.empty()) {
            will_configs_[id] = will;
        }
        last_heartbeat_[id] = std::chrono::steady_clock::now();
    }
    // Update heartbeat (subscriber is alive)
    void heartbeat(SubscriberID id) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_heartbeat_[id] = std::chrono::steady_clock::now();
    }
    // Unregister subscriber (normal disconnect)
    void unregister_subscriber(SubscriberID id) {
        std::lock_guard<std::mutex> lock(mutex_);
        will_configs_.erase(id);  // Don't send will on normal disconnect
        last_heartbeat_.erase(id);
    }
private:
    void monitor_loop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            check_timeouts();
        }
    }
    void check_timeouts() {
        auto now = std::chrono::steady_clock::now();
        std::vector<SubscriberID> timed_out;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [id, last] : last_heartbeat_) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - last).count();
                if (elapsed > 30) {  // 30 second timeout
                    timed_out.push_back(id);
                }
            }
        }
        for (SubscriberID id : timed_out) {
            send_last_will(id);
        }
    }
    void send_last_will(SubscriberID id) {
        LastWillConfig will;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = will_configs_.find(id);
            if (it == will_configs_.end()) {
                return;  // No will configured
            }
            will = it->second;
            will_configs_.erase(it);
            last_heartbeat_.erase(id);
        }
        // Publish the will message
        broker_.publish(will.topic, will.payload.data(), will.payload.size(),
                        will.retain);
    }
};
```
---
## Message Ordering: The Impossible Trinity
Here's a fundamental tension in distributed systems: **you cannot have all three**:
1. **Total order**: All subscribers see all messages in the same order
2. **Per-topic order**: Messages on the same topic are ordered
3. **Per-producer order**: Messages from the same producer are ordered
**Why they're mutually exclusive**:
- If you want total order, you need a single serialization point (bottleneck)
- If you want per-topic order, different topics can be processed on different shards (but cross-topic order is undefined)
- If you want per-producer order, you need to route all messages from one producer to one shard (but then you can't load-balance)
### Our Approach: Per-Topic Sequencing
```cpp
struct MessageHeader {
    uint64_t global_sequence;    // Monotonic across all messages
    uint32_t topic_sequence;     // Monotonic within topic
    uint32_t producer_id;        // Source producer
    uint64_t timestamp_ns;       // Publish time
    // ... rest of header
};
class TopicSequencer {
    std::unordered_map<uint32_t, std::atomic<uint32_t>> topic_sequences_;
    std::atomic<uint64_t> global_sequence_{0};
public:
    uint64_t next_sequence(uint32_t topic_hash) {
        uint64_t global = global_sequence_.fetch_add(1, std::memory_order_relaxed);
        uint32_t topic_seq = topic_sequences_[topic_hash]
            .fetch_add(1, std::memory_order_relaxed);
        // Pack both into header
        return global;  // topic_seq stored separately
    }
};
```
**Guarantee**: Messages on the same topic have monotonically increasing `topic_sequence`. Subscribers can detect gaps (missed messages) and request retransmission.
---
## The Three-Level View: Pub/Sub in Action

![Memory Map: Shared Memory Layout](./diagrams/diag-global-003.svg)

### Level 1 — Application
From the application's perspective, pub/sub is simple:
```cpp
// Publisher
broker.publish("stocks/AAPL/price", price_update, sizeof(price_update));
// Subscriber
broker.subscribe("stocks/+/price", my_subscriber_id, 
    [](const uint8_t* data, uint32_t size) {
        PriceUpdateView view(data);
        std::cout << "Price: " << view.price() << std::endl;
    });
```
### Level 2 — OS/Kernel
The OS provides:
- **Memory mapping**: Shared buffer is mapped into all processes
- **Signal handling**: Subscriber notification via eventfd or futex
- **Process isolation**: Each subscriber has its own address space but shares the buffer
For cross-process notification, we use **eventfd** (Linux):
```cpp
#include <sys/eventfd.h>
class SubscriberNotifier {
    int event_fd_;
public:
    SubscriberNotifier() {
        event_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    }
    void notify() {
        uint64_t value = 1;
        write(event_fd_, &value, sizeof(value));
    }
    void wait() {
        uint64_t value;
        read(event_fd_, &value, sizeof(value));
    }
    int fd() const { return event_fd_; }
};
```
### Level 3 — Hardware

![Hardware Soul: Cache Line Movement](./diagrams/diag-global-004.svg)

The hardware reality:
- **Reference count updates**: Atomic `fetch_sub` on every subscriber ack
- **Cache line bouncing**: The slot's cache line bounces between acking subscribers
- **False sharing risk**: Multiple subscribers acking different slots on the same cache line
**Optimization**: Batch acknowledgments:
```cpp
struct alignas(64) AckBatch {
    static constexpr size_t BATCH_SIZE = 16;
    std::atomic<uint32_t> pending[BATCH_SIZE];
    char padding[64 - sizeof(std::atomic<uint32_t>) * BATCH_SIZE];
};
```
Subscribers ack to a local batch, which is flushed to shared memory periodically.
---
## Horizontal Scaling: Sharding by Topic
When a single broker can't handle the message rate, shard by topic hash:
```cpp
class ShardedPubSubBroker {
    std::vector<std::unique_ptr<PubSubBroker>> shards_;
    size_t num_shards_;
public:
    ShardedPubSubBroker(size_t num_shards, const PubSubConfig& config)
        : num_shards_(num_shards) {
        for (size_t i = 0; i < num_shards; ++i) {
            shards_.push_back(std::make_unique<PubSubBroker>(config));
        }
    }
    void publish(const std::string& topic, const void* data, uint32_t size,
                 bool retain = false) {
        size_t shard = hash_topic(topic) % num_shards_;
        shards_[shard]->publish(topic, data, size, retain);
    }
    void subscribe(const std::string& pattern, SubscriberID id,
                   MessageFilter::FilterPredicate filter = nullptr) {
        // For wildcards, subscribe to ALL shards
        if (pattern.find('+') != std::string::npos ||
            pattern.find('#') != std::string::npos) {
            for (auto& shard : shards_) {
                shard->subscribe(pattern, id, filter);
            }
        } else {
            // Exact topic: single shard
            size_t shard = hash_topic(pattern) % num_shards_;
            shards_[shard]->subscribe(pattern, id, filter);
        }
    }
private:
    static size_t hash_topic(const std::string& topic) {
        size_t h = 0;
        for (char c : topic) {
            h = h * 31 + c;
        }
        return h;
    }
};
```
**Trade-off**: Wildcard subscriptions must subscribe to all shards, duplicating work. This is why MQTT brokers often limit wildcard usage in high-throughput scenarios.
---
## Benchmarking: Measuring Routing Overhead
```cpp
#include <chrono>
#include <vector>
struct PubSubMetrics {
    double publish_latency_ns;
    double match_time_ns;
    double fan_out_time_ns;
    double end_to_end_latency_ns;
    size_t messages_per_second;
    size_t topic_matches_per_second;
};
PubSubMetrics benchmark_pub_sub(PubSubBroker& broker, size_t iterations) {
    std::vector<double> publish_latencies;
    std::vector<double> match_latencies;
    std::vector<double> e2e_latencies;
    // Setup subscriber
    SubscriberID sub_id = 1;
    std::atomic<size_t> received{0};
    broker.subscribe("stocks/+/price", sub_id, 
        [&](const uint8_t* data, uint32_t size) {
            received.fetch_add(1, std::memory_order_relaxed);
        });
    auto bench_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto pub_start = std::chrono::high_resolution_clock::now();
        uint8_t msg[64];
        std::memcpy(msg, &i, sizeof(i));
        broker.publish("stocks/AAPL/price", msg, sizeof(msg));
        auto pub_end = std::chrono::high_resolution_clock::now();
        publish_latencies.push_back(
            std::chrono::duration<double, std::nano>(pub_end - pub_start).count());
    }
    auto bench_end = std::chrono::high_resolution_clock::now();
    // Wait for all messages to be received
    while (received.load() < iterations) {
        std::this_thread::yield();
    }
    // Compute statistics
    PubSubMetrics metrics;
    metrics.publish_latency_ns = std::accumulate(
        publish_latencies.begin(), publish_latencies.end(), 0.0) / iterations;
    metrics.messages_per_second = iterations / 
        std::chrono::duration<double>(bench_end - bench_start).count();
    return metrics;
}
```
### Expected Performance
| Metric | Target | Notes |
|--------|--------|-------|
| Topic match (exact) | 10-50 ns | Hash lookup |
| Topic match (wildcard) | 100-500 ns | Trie traversal |
| Publish (no subs) | 50-100 ns | Quick check |
| Publish (1 sub) | 200-500 ns | Ref count + notify |
| Publish (10 subs) | 500-2000 ns | Fan-out overhead |
| End-to-end latency | 500-2000 ns | Publish to delivery |
---
## Design Decisions: Why This, Not That
### Trie vs Hash Map vs Linear Scan
| Approach | Exact Match | Wildcard | Memory | Complexity |
|----------|-------------|----------|--------|------------|
| **Trie** | O(M) | O(M) | Higher | Medium |
| Hash Map | O(1) | N/A | Lower | Low |
| Linear Scan | O(N) | O(N) | Lowest | Trivial |
**Choice**: Trie. Wildcards are essential for pub/sub, and trie is the only structure that handles them efficiently.
### Push vs Pull Delivery
| Approach | Latency | CPU Usage | Complexity |
|----------|---------|-----------|------------|
| **Push** | Lowest | Higher (interrupts) | Higher |
| Pull (polling) | Higher | Variable | Lower |
| Hybrid | Medium | Optimized | Higher |
**Choice**: Hybrid. Push notification via eventfd, but subscribers pull messages at their own pace.
### In-Place vs Copied Messages
| Approach | Zero-Copy | Fragmentation | Complexity |
|----------|-----------|---------------|------------|
| **In-place (ref count)** | Yes | Possible | Higher |
| Copy on delivery | No | None | Lower |
**Choice**: In-place with reference counting. Zero-copy is the entire point of this system.
---
## Knowledge Cascade: What You've Unlocked
**1. IP Routing Tables (Cross-Domain: Networking)**
The trie you built for topic matching is the same data structure used in router forwarding tables. A router with 500,000 route prefixes uses a trie (specifically, a radix tree or Patricia trie) to find the longest-prefix match in O(address_length) time, regardless of table size. Your topic router is essentially a "content router"—instead of IP prefixes, you're matching topic prefixes.
**2. Event Sourcing (Cross-Domain: Distributed Systems)**
The retained message concept maps directly to **event sourcing** patterns. In event sourcing, the current state is derived by replaying all past events. Your retained messages are the "last event" for each topic—a micro event store. Full event sourcing would store all messages, not just the last one, but the principle is identical: new subscribers "catch up" by reading history.
**3. Reference Counting in Game Engines**
The reference counting you implemented for zero-copy fan-out is the same technique used in game engines for shared resources (textures, meshes, audio buffers). A resource is loaded once, and multiple objects reference it. The resource is freed when the last reference is dropped. The difference: game engines typically use thread-local reference counts for performance, while your cross-process system needs atomic counts.
**4. MQTT Protocol Design**
You've essentially implemented a subset of MQTT 5.0:
- Topic hierarchy with `/` separator
- `+` single-level and `#` multi-level wildcards
- Retained messages
- Last Will and Testament (LWT)
- Quality of Service levels (via delivery guarantees)
Understanding this helps you interoperate with MQTT brokers (Mosquitto, EMQX, HiveMQ) and use standard MQTT client libraries.
**5. Message Ordering in Distributed Systems**
The per-topic sequencing you implemented is the same pattern used in Kafka (per-partition ordering), Amazon Kinesis (per-shard ordering), and Google Pub/Sub (per-key ordering). The trade-off is universal: total order requires a single serialization point, which doesn't scale. Partition/shard/topic-based ordering is the practical answer.
**6. Bloom Filters in Databases (Cross-Domain: Databases)**
The bloom filter you used for quick topic rejection is the same technique used in LSM-tree databases (RocksDB, LevelDB) to avoid disk reads for non-existent keys. Each SSTable has a bloom filter; a read first checks the filter, and only touches disk if the key might exist. Your topic bloom filter does the same: reject non-matching topics without touching the trie.
---
## Implementation Checklist
Before moving to Milestone 5, ensure you have:
1. **Topic Trie Router**
   - Insert subscriptions with wildcard support (`+`, `#`)
   - Match topics in O(M) where M = topic depth
   - Remove subscriptions
   - Thread-safe (read-write lock or lock-free)
2. **Zero-Copy Fan-Out**
   - Reference counting per message slot
   - Multiple subscribers read same memory
   - Slot reclamation when ref count reaches 0
   - Handle slow subscribers (don't block fast ones forever)
3. **Per-Subscriber Filtering**
   - Bloom filter for quick rejection
   - Predicate-based filtering before delivery
   - Field-level filters (e.g., `price > 100`)
   - Composable filters (AND, OR)
4. **Retained Messages**
   - Store last message per topic
   - Deliver to new subscribers on connect
   - Clear retained on explicit request
   - Size limits to prevent unbounded growth
5. **Last-Will Messages**
   - Configure will on subscriber connect
   - Publish will on unexpected disconnect
   - Heartbeat monitoring for liveness
   - Don't send will on normal disconnect
6. **Message Sequencing**
   - Per-topic monotonic sequence numbers
   - Global sequence for total order (optional)
   - Gap detection for missed messages
   - Timestamp for ordering across shards
7. **Horizontal Scaling**
   - Shard by topic hash
   - Wildcard subscriptions to all shards
   - Load balancing across shards
   - Cross-shard ordering considerations
8. **Benchmarks**
   - Topic match latency (exact vs wildcard)
   - Publish latency (various fan-out sizes)
   - End-to-end latency
   - Throughput at various subscriber counts
---
[[CRITERIA_JSON: {"milestone_id": "zcmb-m4", "criteria": ["Topic trie router with O(M) wildcard matching for MQTT-style patterns (+, #)", "Zero-copy fan-out using atomic reference counting per message slot", "Per-subscriber filtering with bloom filter pre-check and predicate evaluation", "Retained message store delivering last-known-good values to late subscribers", "Last-will message publishing on subscriber disconnect with heartbeat monitoring", "Per-topic monotonic sequence numbers for ordering within topic partitions", "Sharded broker architecture supporting horizontal scaling by topic hash", "Subscriber notification via eventfd for cross-process signaling", "Slow subscriber handling with bounded slot reclamation delay", "End-to-end latency benchmark demonstrating sub-microsecond routing overhead for exact topics and sub-5-microsecond for wildcard patterns"]}]
<!-- END_MS -->


<!-- MS_ID: zcmb-m5 -->
# Milestone 5: Crash Recovery & Durability
## The Problem: When Perfect Code Meets Imperfect Reality
You've built a beautiful system. Lock-free ring buffers, zero-copy serialization, efficient pub/sub routing. Your benchmarks show sub-microsecond latency. Your throughput is 10 million messages per second. You're feeling pretty good about yourself.
Then production happens.
At 3:47 AM, the network handler process segfaults on a null pointer. It was holding a claimed slot in the ring buffer—the `ref_count` was set to 2, meaning two subscribers were supposed to read it. The process dies. The reference count never decrements. That slot is now **permanently orphaned**.
```cpp
// The slot that will never be freed
Slot 42:
  ref_count = 2        // Two subscribers expected
  message_size = 256   // Data is there
  // But the producer is dead, subscribers will never be notified
  // ref_count will never reach 0
  // Slot 42 is lost forever
```
Within minutes, the ring buffer fills up. Every new message fails to publish. The entire message bus is dead—all because one process crashed at the wrong moment.
**But wait, it gets worse.**
The ops team restarts the network handler. It reconnects to the shared memory. But what state was it in when it crashed? Was it in the middle of publishing a batch? Had it updated `head` but not finished writing the data? The consumer reads garbage—half-written messages, corrupted sequence numbers.
You try to replay from a log, but the log itself might be corrupted. The process crashed during a `write()` syscall—the log file has a partial message. You can't tell where the message ends and garbage begins.
**And it gets even worse.**
Your trading system has a requirement: **no message loss**. If a crash happens, you must recover every in-flight message. So you add persistence: every message is written to disk before being acknowledged. Your latency just went from **500 nanoseconds to 2 milliseconds**—a **4000× slowdown**. The disk write takes 1-2ms, and you need an `fsync()` to guarantee it's durable.
The tension is brutal: **durability requires disk I/O, but disk I/O kills low latency**. You can't have both.

![System Satellite Map: Zero-Copy Message Bus](./diagrams/diag-global-001.svg)

**The deeper tension**: Crash recovery code is the least tested, most buggy code in any system. Why? Because crashes are rare, and crash-while-recovering is rarer still. You can't easily test the recovery path in CI. The first time your recovery code runs in production might be during an actual outage—when you need it most.
**The escape hatch**: Accept that perfect recovery is impossible. Design for **graceful degradation**: recover quickly, recover most messages, and make the failure visible. A system that comes back in 50ms with 99.9% of messages is better than one that takes 5 seconds trying to achieve 100%.
---
## The Revelation: Exactly-Once Is a Lie
Here's the misconception that ruins crash recovery designs: "I'll use a write-ahead log and replay on restart. That gives me exactly-once semantics—no duplicates, no loss."
This is seductive because it sounds achievable. Write to a log, acknowledge after the write, replay the log on crash. What could go wrong?
**Everything.** Consider this scenario:
```
Time    Producer                    Consumer                    Disk
----    --------                    --------                    ----
T1      Send message M1
T2      Write M1 to log                                         M1 on disk
T3      fsync() log                                             M1 durable
T4      Update shared memory
T5      Send ACK to consumer
T6                                  Receive ACK
T7                                  Process M1 ✓
T8                                  Send ACK back
T9      Receive ACK
T10     Mark M1 as "done" in log
T11     --- CRASH ---
T12     Restart
T13     Replay log
T14     See M1 in log (not marked done)
T15     Re-send M1?                                          ← THE PROBLEM
```
At T15, the producer sees M1 in the log but not marked as "done" (the crash happened before T10 completed). Should it re-send?
- **If yes**: The consumer already processed M1 at T7. Now it gets a duplicate. This violates exactly-once.
- **If no**: What if the crash happened at T5 instead (before ACK was sent)? The consumer never got M1. Not re-sending means message loss.
The fundamental problem: **the crash can happen anywhere in the acknowledgment chain, and you can't distinguish "consumer processed but we didn't record it" from "consumer never received it."**
This is a variant of the **Two Generals Problem**—it's theoretically impossible to achieve perfect agreement over an unreliable channel. Your "channel" here includes the process itself (which can crash) and the disk (which has its own failure modes).
### The Pragmatic Answer: At-Least-Once + Idempotence
Since exactly-once is nearly impossible without distributed consensus (which adds even more latency), the practical approach is:
1. **At-least-once delivery**: Messages may be delivered multiple times, but never lost
2. **Idempotent consumers**: Processing the same message twice has the same effect as processing it once
```cpp
// Idempotent consumer: uses message ID to deduplicate
class IdempotentConsumer {
    std::unordered_set<uint64_t> processed_ids_;
    size_t max_cache_size_ = 100000;  // Remember last 100K message IDs
public:
    void process_message(const Message& msg) {
        // Check if already processed
        if (processed_ids_.count(msg.id)) {
            return;  // Skip duplicate
        }
        // Process the message
        actually_process(msg);
        // Record as processed
        processed_ids_.insert(msg.id);
        if (processed_ids_.size() > max_cache_size_) {
            // Evict oldest (simplified - production would use LRU)
            processed_ids_.erase(processed_ids_.begin());
        }
    }
};
```
This is the approach used by **Kafka** (with idempotent producers) and most production messaging systems. The consumer is responsible for deduplication, not the broker.
---
## The Architecture: Layers of Resilience

![Three-Level View: Complete Message Flow](./diagrams/diag-global-002.svg)

Crash recovery isn't one thing—it's a stack of mechanisms, each handling a different failure mode:
### Layer 1: Crash Detection
How do you know a process crashed?
| Mechanism | Latency | Reliability | Complexity |
|-----------|---------|-------------|------------|
| **Heartbeat timeout** | 1-30s | Medium | Low |
| **PID check** | Instant | Medium | Low |
| **Reference count leak** | Eventual | High | Medium |
| **Kernel notification** | Instant | High | High |
### Layer 2: State Recovery
How do you restore the system to a consistent state?
| Mechanism | Recovery Time | Completeness | Overhead |
|-----------|---------------|--------------|----------|
| **Reset to empty** | Instant | 0% (lose all) | None |
| **Replay from log** | Seconds-minutes | 99%+ | High |
| **Checkpoint restore** | 10-100ms | 99.9%+ | Medium |
| **Hybrid (checkpoint + log)** | 50-200ms | 99.99%+ | High |
### Layer 3: Durability (Optional)
How do you survive power loss / kernel panic?
| Mechanism | Latency Impact | Durability | Complexity |
|-----------|----------------|------------|------------|
| **None (memory only)** | None | Zero | None |
| **Async flush** | 1-10μs | Eventual | Low |
| **Sync per message** | 1-5ms | Full | Medium |
| **Batch sync** | 10-100μs amortized | Full | Medium |
---
## Implementation: Crash Detection
### Approach 1: Heartbeat Monitoring
The simplest approach: each process periodically writes a timestamp to shared memory. Other processes check if the timestamp is stale.
```cpp
#include <atomic>
#include <chrono>
#include <thread>
// Shared memory layout
struct alignas(64) ProcessHeartbeat {
    std::atomic<uint64_t> last_heartbeat_ns{0};
    std::atomic<uint32_t> process_id{0};
    std::atomic<uint32_t> generation{0};  // Incremented on restart
    char padding[64 - 3 * sizeof(std::atomic<uint32_t>)];
};
class HeartbeatMonitor {
    ProcessHeartbeat* heartbeat_;
    std::chrono::nanoseconds timeout_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};
public:
    HeartbeatMonitor(ProcessHeartbeat* heartbeat, 
                     std::chrono::nanoseconds timeout = std::chrono::seconds(5))
        : heartbeat_(heartbeat), timeout_(timeout) 
    {
        heartbeat_->process_id.store(getpid(), std::memory_order_relaxed);
        monitor_thread_ = std::thread(&HeartbeatMonitor::monitor_loop, this);
    }
    ~HeartbeatMonitor() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }
    // Call this regularly from the monitored process
    void update_heartbeat() {
        heartbeat_->last_heartbeat_ns.store(
            now_ns(), std::memory_order_release);
    }
    // Check if another process is still alive
    bool is_alive(const ProcessHeartbeat& other) const {
        uint64_t last = other.last_heartbeat_ns.load(std::memory_order_acquire);
        uint64_t age_ns = now_ns() - last;
        return age_ns < timeout_.count();
    }
private:
    static uint64_t now_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
    void monitor_loop() {
        while (running_) {
            update_heartbeat();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};
```
**Trade-off**: A 5-second timeout means crashes are detected in 5+ seconds. Lower timeout means more false positives (a busy process might not update heartbeat in time).
### Approach 2: PID-Based Detection
On Linux, you can check if a process is alive by sending signal 0:
```cpp
#include <signal.h>
#include <errno.h>
bool is_process_alive(pid_t pid) {
    if (kill(pid, 0) == 0) {
        return true;  // Process exists
    }
    return errno != ESRCH;  // ESRCH = no such process
}
```
**Limitation**: PIDs can be reused. A process crashes, the OS reuses its PID for a new unrelated process. Your check says "alive" but it's a different process.
### Approach 3: Generation Counter (Solves PID Reuse)
Each process increments a generation counter on startup. The combination of (PID, generation) uniquely identifies a process instance:
```cpp
struct ProcessIdentity {
    pid_t pid;
    uint32_t generation;
    bool operator==(const ProcessIdentity& other) const {
        return pid == other.pid && generation == other.generation;
    }
};
class ProcessRegistry {
    struct alignas(64) Entry {
        std::atomic<uint32_t> pid{0};
        std::atomic<uint32_t> generation{0};
        std::atomic<uint64_t> last_update{0};
    };
    std::array<Entry, MAX_PROCESSES> entries_;
    std::atomic<uint32_t> next_slot_{0};
public:
    // Register current process, return slot index
    uint32_t register_process() {
        uint32_t slot = next_slot_.fetch_add(1, std::memory_order_relaxed);
        if (slot >= MAX_PROCESSES) {
            throw std::runtime_error("Too many processes");
        }
        Entry& entry = entries_[slot];
        entry.pid.store(getpid(), std::memory_order_relaxed);
        entry.generation.fetch_add(1, std::memory_order_relaxed);
        entry.last_update.store(now_ns(), std::memory_order_release);
        return slot;
    }
    // Check if process at slot is still the same instance
    bool is_same_instance(uint32_t slot, uint32_t expected_generation) {
        Entry& entry = entries_[slot];
        uint32_t current_pid = entry.pid.load(std::memory_order_acquire);
        uint32_t current_gen = entry.generation.load(std::memory_order_acquire);
        if (current_pid == 0) return false;  // Slot never used
        if (current_gen != expected_generation) {
            return false;  // Generation changed = process restarted
        }
        // PID and generation match, but is the process actually alive?
        return is_process_alive(current_pid);
    }
};
```
---
## Implementation: Recovering Orphaned Slots
When a producer crashes after claiming a slot but before publishing, that slot is orphaned—its `ref_count` is stuck at a non-zero value. You need to detect and reclaim these slots.
### Strategy 1: Timeout-Based Reclamation
Each slot has a timestamp. If a slot is "claimed" for too long, assume the producer crashed and force-release it:
```cpp
struct alignas(64) SlotHeader {
    std::atomic<uint32_t> ref_count;
    uint32_t message_size;
    uint64_t claim_timestamp_ns;  // When the slot was claimed
    uint32_t owner_pid;           // Process that claimed the slot
    uint32_t sequence;
};
class OrphanReclaimer {
    SlotHeader* slots_;
    size_t num_slots_;
    std::chrono::nanoseconds orphan_timeout_;
    std::thread reclaimer_thread_;
    std::atomic<bool> running_{true};
public:
    OrphanReclaimer(SlotHeader* slots, size_t num_slots,
                    std::chrono::nanoseconds timeout = std::chrono::seconds(10))
        : slots_(slots), num_slots_(num_slots), orphan_timeout_(timeout)
    {
        reclaimer_thread_ = std::thread(&OrphanReclaimer::reclaim_loop, this);
    }
    ~OrphanReclaimer() {
        running_ = false;
        if (reclaimer_thread_.joinable()) {
            reclaimer_thread_.join();
        }
    }
private:
    void reclaim_loop() {
        while (running_) {
            scan_for_orphans();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    void scan_for_orphans() {
        uint64_t now = now_ns();
        for (size_t i = 0; i < num_slots_; ++i) {
            SlotHeader& slot = slots_[i];
            uint32_t ref = slot.ref_count.load(std::memory_order_acquire);
            // Skip unclaimed slots
            if (ref == 0) continue;
            // Check if orphaned (claimed too long ago)
            uint64_t age_ns = now - slot.claim_timestamp_ns;
            if (age_ns < orphan_timeout_.count()) continue;
            // Check if owner is still alive
            if (is_process_alive(slot.owner_pid)) {
                // Owner is alive but slow - might be legitimate
                continue;
            }
            // Owner is dead - reclaim the slot
            // Use CAS to avoid racing with a recovering owner
            uint32_t expected = ref;
            if (slot.ref_count.compare_exchange_strong(
                    expected, 0,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                // Successfully reclaimed
                log_reclamation(i, slot.owner_pid, ref);
            }
        }
    }
    void log_reclamation(size_t slot_idx, pid_t owner, uint32_t old_ref);
};
```
**Risk**: What if the owner is just slow, not dead? You might reclaim a slot that's about to be published. Mitigation:
- Use a long timeout (30+ seconds)
- Require producers to update a "still working" timestamp during long operations
- Use explicit cancellation if the producer detects it was preempted
### Strategy 2: Fence-Based Recovery (Production-Grade)
Use a **fence value** (monotonically increasing) to invalidate old claims. When a process restarts, it uses a new fence value. Old claims with the previous fence are considered invalid:
```cpp
struct alignas(64) SlotHeader {
    std::atomic<uint32_t> ref_count;
    std::atomic<uint64_t> fence;      // Monotonically increasing
    uint32_t message_size;
    uint64_t claim_timestamp_ns;
    uint32_t owner_slot;              // Index in process registry
};
class FencedProducer {
    uint64_t my_fence_;
    uint32_t my_slot_;
    ProcessRegistry& registry_;
public:
    FencedProducer(ProcessRegistry& registry)
        : registry_(registry)
    {
        my_slot_ = registry.register_process();
        // Fence = generation counter, incremented on each restart
        my_fence_ = registry.get_generation(my_slot_);
    }
    bool try_claim_slot(SlotHeader& slot, uint32_t ref_count) {
        // Check current fence
        uint64_t current_fence = slot.fence.load(std::memory_order_acquire);
        // Slot is free (ref_count == 0) or our previous claim
        uint32_t current_ref = slot.ref_count.load(std::memory_order_acquire);
        if (current_ref == 0) {
            // Try to claim with our fence
            if (slot.fence.compare_exchange_strong(
                    current_fence, my_fence_,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                // Fence is ours, now set ref_count
                slot.ref_count.store(ref_count, std::memory_order_release);
                slot.claim_timestamp_ns = now_ns();
                slot.owner_slot = my_slot_;
                return true;
            }
        }
        return false;
    }
    void publish_slot(SlotHeader& slot) {
        // Verify we still own the slot (fence hasn't changed)
        if (slot.fence.load(std::memory_order_acquire) != my_fence_) {
            throw std::runtime_error("Slot was reclaimed by another process");
        }
        // Proceed with publish...
    }
};
```
**Why this works**: If a producer crashes and restarts, it gets a new fence value (higher generation). Any slots claimed with the old fence are now invalid—the reclaimer can safely reset them because the old producer instance will never publish (it's dead), and the new producer instance has a different fence.
---
## Implementation: Write-Ahead Logging for Durability
If you need true durability (survive power loss), you must write to persistent storage. The standard technique is **Write-Ahead Logging (WAL)**—the same approach used by databases.
[[EXPLAIN:write-ahead-logging-wal|A durability technique where changes are logged to disk before being applied to the main data structure, enabling recovery after a crash]]
### The WAL Protocol
1. **Before** publishing a message, write it to the WAL
2. **After** the WAL write is durable (`fsync`), publish to shared memory
3. **After** all subscribers have consumed, truncate the WAL
4. **On recovery**, replay the WAL to restore in-flight messages
```cpp
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <vector>
class WriteAheadLog {
    int fd_;
    std::string path_;
    uint64_t next_sequence_{0};
    uint64_t truncated_sequence_{0};
public:
    struct alignas(8) LogEntryHeader {
        uint32_t magic;           // 0xDEADBEEF
        uint32_t entry_size;      // Size of data following header
        uint64_t sequence;        // Monotonic sequence number
        uint64_t timestamp_ns;    // When entry was written
        uint32_t checksum;        // CRC32 of data
        uint32_t flags;           // Entry type flags
    };
    static constexpr uint32_t MAGIC = 0xDEADBEEF;
    static constexpr size_t HEADER_SIZE = sizeof(LogEntryHeader);
    WriteAheadLog(const std::string& path) : path_(path) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_DSYNC, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open WAL: " + 
                std::string(strerror(errno)));
        }
        // Find the last sequence number by scanning
        recover_sequence();
    }
    ~WriteAheadLog() {
        if (fd_ >= 0) close(fd_);
    }
    // Append an entry to the WAL
    uint64_t append(const void* data, size_t size) {
        LogEntryHeader header;
        header.magic = MAGIC;
        header.entry_size = size;
        header.sequence = next_sequence_++;
        header.timestamp_ns = now_ns();
        header.checksum = compute_crc32(data, size);
        header.flags = 0;
        // Write header + data
        // Use writev for single syscall
        struct iovec iov[2];
        iov[0].iov_base = &header;
        iov[0].iov_len = HEADER_SIZE;
        iov[1].iov_base = const_cast<void*>(data);
        iov[1].iov_len = size;
        ssize_t written = writev(fd_, iov, 2);
        if (written != static_cast<ssize_t>(HEADER_SIZE + size)) {
            throw std::runtime_error("WAL write failed");
        }
        return header.sequence;
    }
    // Force data to disk
    void sync() {
        if (fsync(fd_) != 0) {
            throw std::runtime_error("WAL sync failed");
        }
    }
    // Truncate log up to (but not including) sequence
    void truncate(uint64_t sequence) {
        // For simplicity, we'll rebuild the log without old entries
        // Production systems use circular logs or segmented logs
        std::vector<uint8_t> buffer;
        std::vector<std::pair<uint64_t, size_t>> entries;
        // Read all entries, keeping those >= sequence
        lseek(fd_, 0, SEEK_SET);
        while (true) {
            LogEntryHeader header;
            ssize_t n = read(fd_, &header, HEADER_SIZE);
            if (n == 0) break;  // EOF
            if (n != HEADER_SIZE) break;
            if (header.magic != MAGIC) break;
            if (header.sequence >= sequence) {
                // Keep this entry
                entries.push_back({header.sequence, buffer.size()});
                size_t offset = buffer.size();
                buffer.resize(offset + HEADER_SIZE + header.entry_size);
                std::memcpy(buffer.data() + offset, &header, HEADER_SIZE);
                read(fd_, buffer.data() + offset + HEADER_SIZE, header.entry_size);
            } else {
                // Skip this entry
                lseek(fd_, header.entry_size, SEEK_CUR);
            }
        }
        // Rewrite the log
        close(fd_);
        fd_ = open(path_.c_str(), O_RDWR | O_TRUNC | O_DSYNC, 0644);
        for (const auto& [seq, offset] : entries) {
            LogEntryHeader* h = reinterpret_cast<LogEntryHeader*>(
                buffer.data() + offset);
            write(fd_, h, HEADER_SIZE + h->entry_size);
        }
        fsync(fd_);
        truncated_sequence_ = sequence;
    }
    // Replay entries from the log
    template<typename Handler>
    void replay(uint64_t from_sequence, Handler handler) {
        lseek(fd_, 0, SEEK_SET);
        while (true) {
            LogEntryHeader header;
            ssize_t n = read(fd_, &header, HEADER_SIZE);
            if (n == 0) break;
            if (n != HEADER_SIZE) break;
            if (header.magic != MAGIC) {
                // Corrupted log - stop replay
                break;
            }
            if (header.sequence < from_sequence) {
                lseek(fd_, header.entry_size, SEEK_CUR);
                continue;
            }
            // Read entry data
            std::vector<uint8_t> data(header.entry_size);
            read(fd_, data.data(), header.entry_size);
            // Verify checksum
            uint32_t computed = compute_crc32(data.data(), data.size());
            if (computed != header.checksum) {
                // Corrupted entry - skip
                continue;
            }
            // Call handler
            handler(header.sequence, data.data(), data.size());
        }
    }
private:
    void recover_sequence() {
        lseek(fd_, 0, SEEK_SET);
        while (true) {
            LogEntryHeader header;
            ssize_t n = read(fd_, &header, HEADER_SIZE);
            if (n == 0) break;
            if (n != HEADER_SIZE) break;
            if (header.magic != MAGIC) break;
            next_sequence_ = std::max(next_sequence_, header.sequence + 1);
            lseek(fd_, header.entry_size, SEEK_CUR);
        }
    }
    static uint64_t now_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
    static uint32_t compute_crc32(const void* data, size_t size);
};
```
### The Latency Cost of Durability
Let's measure what durability actually costs:
```cpp
struct DurabilityBenchmark {
    double append_ns;
    double sync_ns;
    double total_ns;
};
DurabilityBenchmark benchmark_wal(WriteAheadLog& wal, size_t msg_size, 
                                   size_t iterations) {
    std::vector<uint8_t> msg(msg_size, 0xAB);
    std::vector<double> append_times, sync_times, total_times;
    for (size_t i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        wal.append(msg.data(), msg.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        wal.sync();
        auto t2 = std::chrono::high_resolution_clock::now();
        append_times.push_back(
            std::chrono::duration<double, std::nano>(t1 - t0).count());
        sync_times.push_back(
            std::chrono::duration<double, std::nano>(t2 - t1).count());
        total_times.push_back(
            std::chrono::duration<double, std::nano>(t2 - t0).count());
    }
    auto avg = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    return {avg(append_times), avg(sync_times), avg(total_times)};
}
```
**Expected results on a typical SSD:**
| Operation | Latency |
|-----------|---------|
| `append()` (in kernel buffer) | 1-5 μs |
| `fsync()` (to disk) | 500-2000 μs |
| **Total** | **500-2000 μs** |
Compare to your in-memory ring buffer: **200-500 ns**. Durability adds **1000-10000× latency**.
### Mitigation: Asynchronous Batching
Instead of `fsync` per message, batch writes and sync periodically:
```cpp
class BatchedWal {
    WriteAheadLog wal_;
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> pending_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread sync_thread_;
    std::atomic<bool> running_{true};
    size_t batch_size_;
    std::chrono::microseconds sync_interval_;
public:
    BatchedWal(const std::string& path, 
               size_t batch_size = 1000,
               std::chrono::microseconds sync_interval = 
                   std::chrono::milliseconds(10))
        : wal_(path), batch_size_(batch_size), sync_interval_(sync_interval)
    {
        sync_thread_ = std::thread(&BatchedWal::sync_loop, this);
    }
    ~BatchedWal() {
        running_ = false;
        cv_.notify_all();
        if (sync_thread_.joinable()) {
            sync_thread_.join();
        }
    }
    // Append without immediate sync - returns immediately
    uint64_t append(const void* data, size_t size) {
        std::vector<uint8_t> copy(static_cast<const uint8_t*>(data),
                                   static_cast<const uint8_t*>(data) + size);
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t seq = wal_.append(data, size);
        pending_.push_back({seq, std::move(copy)});
        if (pending_.size() >= batch_size_) {
            cv_.notify_one();  // Trigger early sync
        }
        return seq;
    }
    // Wait for specific sequence to be durable
    void wait_durable(uint64_t sequence) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this, sequence] {
            return last_synced_ >= sequence;
        });
    }
private:
    uint64_t last_synced_{0};
    void sync_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, sync_interval_, [this] {
                return !running_ || pending_.size() >= batch_size_;
            });
            if (pending_.empty()) continue;
            wal_.sync();
            last_synced_ = pending_.back().first;
            pending_.clear();
            lock.unlock();
            cv_.notify_all();  // Wake waiters
        }
    }
};
```
**Trade-off**: Messages are "acknowledged" before being durable. A crash loses the last batch (10ms of messages at default settings). This is often acceptable—many systems prefer "lose 10ms of data on crash" over "every message takes 1ms."
---
## Implementation: Checkpoint/Restart
For faster recovery than full log replay, periodically save a **checkpoint**—a snapshot of the system state. On recovery, restore from checkpoint and replay only the log entries after the checkpoint.
```cpp
#include <fstream>
struct CheckpointHeader {
    uint32_t magic;              // 0xFEEDFACE
    uint32_t version;            // Checkpoint format version
    uint64_t sequence;           // WAL sequence at checkpoint time
    uint64_t timestamp_ns;       // When checkpoint was taken
    uint32_t num_ring_entries;   // Entries following
    uint32_t checksum;           // CRC32 of data
};
class CheckpointManager {
    std::string checkpoint_dir_;
    WriteAheadLog& wal_;
public:
    CheckpointManager(const std::string& dir, WriteAheadLog& wal)
        : checkpoint_dir_(dir), wal_(wal) {}
    // Take a checkpoint
    void save_checkpoint(const std::string& name,
                         const void* ring_state, 
                         size_t ring_size,
                         uint64_t wal_sequence) {
        std::string path = checkpoint_dir_ + "/" + name + ".ckpt";
        std::string tmp_path = path + ".tmp";
        std::ofstream out(tmp_path, std::ios::binary);
        CheckpointHeader header;
        header.magic = 0xFEEDFACE;
        header.version = 1;
        header.sequence = wal_sequence;
        header.timestamp_ns = now_ns();
        header.num_ring_entries = 1;
        header.checksum = compute_crc32(ring_state, ring_size);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(ring_state), ring_size);
        out.close();
        // Atomic rename
        rename(tmp_path.c_str(), path.c_str());
    }
    // Load latest checkpoint
    struct CheckpointData {
        uint64_t wal_sequence;
        std::vector<uint8_t> ring_state;
    };
    bool load_latest_checkpoint(const std::string& name, 
                                 CheckpointData& out) {
        std::string path = checkpoint_dir_ + "/" + name + ".ckpt";
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        CheckpointHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.magic != 0xFEEDFACE) return false;
        out.wal_sequence = header.sequence;
        out.ring_state.resize(header.num_ring_entries * 
            /* ring entry size */ 1024);
        in.read(reinterpret_cast<char*>(out.ring_state.data()),
                out.ring_state.size());
        // Verify checksum
        uint32_t computed = compute_crc32(out.ring_state.data(),
                                          out.ring_state.size());
        return computed == header.checksum;
    }
    // Full recovery: checkpoint + WAL replay
    template<typename RingBuffer, typename Handler>
    bool recover(RingBuffer& ring, const std::string& name, 
                 Handler message_handler) {
        CheckpointData ckpt;
        if (load_latest_checkpoint(name, ckpt)) {
            // Restore ring state from checkpoint
            ring.restore_from_bytes(ckpt.ring_state.data(),
                                    ckpt.ring_state.size());
            // Replay WAL entries after checkpoint
            wal_.replay(ckpt.wal_sequence + 1, 
                [&](uint64_t seq, const void* data, size_t size) {
                    message_handler(data, size);
                });
            return true;
        }
        // No checkpoint - full WAL replay
        wal_.replay(0, [&](uint64_t seq, const void* data, size_t size) {
            message_handler(data, size);
        });
        return true;
    }
private:
    static uint64_t now_ns();
    static uint32_t compute_crc32(const void* data, size_t size);
};
```
**Recovery time comparison:**
| Scenario | Full WAL Replay | Checkpoint + WAL |
|----------|-----------------|------------------|
| 1 hour of data, 1M messages | 5-30 seconds | 50-200 ms |
| 1 day of data, 24M messages | 2-10 minutes | 50-200 ms |
| 1 week of data | Hours | 50-200 ms |
The checkpoint approach has **constant recovery time** regardless of data volume, while full replay is linear.
---
## Handling Crash During Recovery
Here's a nightmare scenario: the system crashes, starts recovery, and crashes *again* during recovery. The recovery code might have partially restored state, leaving the system in an inconsistent state.
### Solution: Atomic Recovery with Journaling
Use a recovery journal to track progress. If recovery crashes, the journal tells you where to resume:
```cpp
class RecoveryJournal {
    std::string journal_path_;
    int fd_;
public:
    enum class Phase : uint32_t {
        NOT_STARTED = 0,
        LOADING_CHECKPOINT = 1,
        REPLAYING_WAL = 2,
        REBUILDING_INDICES = 3,
        COMPLETED = 4
    };
    struct JournalEntry {
        Phase phase;
        uint64_t checkpoint_sequence;
        uint64_t wal_replay_position;
        uint64_t timestamp_ns;
    };
    RecoveryJournal(const std::string& path) : journal_path_(path) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0644);
    }
    ~RecoveryJournal() {
        if (fd_ >= 0) close(fd_);
    }
    void begin_phase(Phase phase) {
        JournalEntry entry;
        entry.phase = phase;
        entry.timestamp_ns = now_ns();
        pwrite(fd_, &entry, sizeof(entry), 0);
        fsync(fd_);
    }
    void update_wal_position(uint64_t pos) {
        JournalEntry entry;
        entry.phase = Phase::REPLAYING_WAL;
        entry.wal_replay_position = pos;
        entry.timestamp_ns = now_ns();
        pwrite(fd_, &entry, sizeof(entry), 0);
        // No fsync - we can afford to lose position updates
    }
    void mark_completed() {
        begin_phase(Phase::COMPLETED);
    }
    JournalEntry read_last() {
        JournalEntry entry{};
        pread(fd_, &entry, sizeof(entry), 0);
        return entry;
    }
    void clear() {
        JournalEntry entry{};
        pwrite(fd_, &entry, sizeof(entry), 0);
        fsync(fd_);
    }
};
```
**Recovery logic with journaling:**
```cpp
class RecoveryManager {
    RecoveryJournal journal_;
    CheckpointManager& checkpoint_;
    WriteAheadLog& wal_;
public:
    void recover() {
        auto last = journal_.read_last();
        switch (last.phase) {
            case RecoveryJournal::Phase::NOT_STARTED:
            case RecoveryJournal::Phase::COMPLETED:
                // Clean start - full recovery
                full_recovery();
                break;
            case RecoveryJournal::Phase::LOADING_CHECKPOINT:
                // Crashed during checkpoint load - retry
                full_recovery();
                break;
            case RecoveryJournal::Phase::REPLAYING_WAL:
                // Crashed during WAL replay - resume
                resume_wal_replay(last.wal_replay_position);
                break;
            case RecoveryJournal::Phase::REBUILDING_INDICES:
                // Crashed during index rebuild - redo
                rebuild_indices();
                break;
        }
        journal_.mark_completed();
    }
private:
    void full_recovery() {
        journal_.begin_phase(RecoveryJournal::Phase::LOADING_CHECKPOINT);
        load_checkpoint();
        journal_.begin_phase(RecoveryJournal::Phase::REPLAYING_WAL);
        replay_wal();
        journal_.begin_phase(RecoveryJournal::Phase::REBUILDING_INDICES);
        rebuild_indices();
    }
    void resume_wal_replay(uint64_t position) {
        // Skip to where we left off
        wal_.replay(position, [this](uint64_t seq, const void* data, size_t size) {
            apply_message(data, size);
            journal_.update_wal_position(seq);
        });
        journal_.begin_phase(RecoveryJournal::Phase::REBUILDING_INDICES);
        rebuild_indices();
    }
    void load_checkpoint();
    void replay_wal();
    void rebuild_indices();
    void apply_message(const void* data, size_t size);
};
```
---
## Process Supervision Integration
Your message bus doesn't exist in isolation—it runs under a process supervisor like **systemd** or **supervisord**. Proper integration with the supervisor improves crash recovery.
### Signal Handling for Graceful Shutdown
```cpp
#include <signal.h>
#include <atomic>
class SignalHandler {
    static std::atomic<bool> shutdown_requested_{false};
    static std::atomic<bool> force_shutdown_{false};
public:
    static void setup() {
        struct sigaction sa;
        sa.sa_handler = handle_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGTERM, &sa, nullptr);  // Graceful shutdown
        sigaction(SIGINT, &sa, nullptr);   // Ctrl+C
        sigaction(SIGHUP, &sa, nullptr);   // Reload config
        // SIGKILL cannot be caught
        // SIGSEGV/SIGABRT should trigger core dump, not graceful shutdown
    }
    static bool should_shutdown() {
        return shutdown_requested_.load(std::memory_order_acquire);
    }
    static bool should_force_shutdown() {
        return force_shutdown_.load(std::memory_order_acquire);
    }
private:
    static void handle_signal(int sig) {
        switch (sig) {
            case SIGTERM:
            case SIGINT:
                if (shutdown_requested_.load()) {
                    // Second signal - force shutdown
                    force_shutdown_.store(true);
                } else {
                    shutdown_requested_.store(true);
                }
                break;
            case SIGHUP:
                // Reload configuration
                break;
        }
    }
};
// In your main loop
void run_message_bus() {
    SignalHandler::setup();
    while (!SignalHandler::should_shutdown()) {
        process_messages();
        if (SignalHandler::should_force_shutdown()) {
            // Force exit - skip cleanup
            _exit(1);
        }
    }
    // Graceful shutdown
    graceful_shutdown();
}
```
### Integration with systemd
Create a systemd service file that handles restart:
```ini
[Unit]
Description=Zero-Copy Message Bus
After=network.target
[Service]
Type=notify
ExecStart=/usr/bin/zcmb-broker
Restart=on-failure
RestartSec=100ms
TimeoutStopSec=5
# Notify systemd when ready
# Your code should call: sd_notify(0, "READY=1");
# Limit restarts to prevent thrashing
StartLimitIntervalSec=60
StartLimitBurst=5
[Install]
WantedBy=multi-user.target
```
The `Restart=on-failure` and `RestartSec=100ms` mean systemd will automatically restart crashed processes within 100ms. Combined with fast checkpoint recovery, total downtime can be under 200ms.
---
## The Three-Level View: Crash Recovery in Action

![Memory Map: Shared Memory Layout](./diagrams/diag-global-003.svg)

### Level 1 — Application
From your application's perspective, crash recovery should be invisible:
```cpp
// Application code doesn't need to know about recovery
void run_consumer() {
    MessageBus bus = MessageBus::connect_or_recover("my_bus");
    bus.subscribe("orders/*", [](const Message& msg) {
        // Messages might be duplicates (at-least-once)
        // Consumer must be idempotent
        process_order(msg);
    });
}
```
### Level 2 — OS/Kernel
The kernel provides:
- **Signal delivery**: Notifies processes of shutdown requests
- **Process lifecycle**: PID tracking, exit codes
- **Filesystem**: Persistent storage for WAL and checkpoints
- **OOM killer**: May kill your process under memory pressure
Key insight: `fsync()` is your contract with the kernel. Without it, your data may sit in the page cache indefinitely. The kernel's default behavior is to delay disk writes for performance—durability requires explicit opt-in.
### Level 3 — Hardware
The hardware reality of durability:
- **SSD write latency**: 50-500 μs per 4KB page
- **SSD write amplification**: Small writes may trigger larger erase cycles
- **Disk write latency (HDD)**: 5-10 ms (seek + rotation)
- **Cache flush overhead**: The `FLUSH CACHE` command forces the drive to commit all pending writes
```cpp
// What happens when you call fsync()
fsync(fd);
// ↓
// Kernel: flush dirty pages for this file to disk
// ↓
// Drive controller: write to NAND flash
// ↓
// NAND flash: program cells (50-500 μs per page)
// ↓
// Drive: acknowledge completion
```
The `O_DSYNC` flag we used earlier tells the kernel to ensure data is on stable storage before `write()` returns, avoiding the separate `fsync()` call but still incurring the latency.
---
## Benchmarking: Recovery Time Measurement
```cpp
#include <chrono>
struct RecoveryMetrics {
    double checkpoint_load_ms;
    double wal_replay_ms;
    double index_rebuild_ms;
    double total_ms;
    size_t messages_recovered;
    size_t messages_lost;
};
RecoveryMetrics benchmark_recovery(CheckpointManager& ckpt,
                                    WriteAheadLog& wal,
                                    RingBuffer& ring) {
    RecoveryMetrics metrics;
    auto total_start = std::chrono::high_resolution_clock::now();
    // Simulate crash
    ring.reset();
    // Load checkpoint
    auto ckpt_start = std::chrono::high_resolution_clock::now();
    CheckpointManager::CheckpointData data;
    bool has_ckpt = ckpt.load_latest_checkpoint("main", data);
    auto ckpt_end = std::chrono::high_resolution_clock::now();
    metrics.checkpoint_load_ms = 
        std::chrono::duration<double, std::milli>(ckpt_end - ckpt_start).count();
    if (has_ckpt) {
        ring.restore_from_bytes(data.ring_state.data(), 
                                data.ring_state.size());
    }
    // Replay WAL
    auto replay_start = std::chrono::high_resolution_clock::now();
    size_t replayed = 0;
    wal.replay(has_ckpt ? data.wal_sequence + 1 : 0,
        [&](uint64_t seq, const void* data, size_t size) {
            ring.try_enqueue(data, size);
            ++replayed;
        });
    auto replay_end = std::chrono::high_resolution_clock::now();
    metrics.wal_replay_ms = 
        std::chrono::duration<double, std::milli>(replay_end - replay_start).count();
    metrics.messages_recovered = replayed;
    auto total_end = std::chrono::high_resolution_clock::now();
    metrics.total_ms = 
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    return metrics;
}
```
**Expected results (checkpoint every 10 seconds):**
| Scenario | Checkpoint Load | WAL Replay | Total | Messages Recovered |
|----------|-----------------|------------|-------|-------------------|
| 1M msg/sec, 5s since checkpoint | 10-30 ms | 20-50 ms | 30-80 ms | ~50,000 |
| 1M msg/sec, 10s since checkpoint | 10-30 ms | 40-100 ms | 50-130 ms | ~100,000 |
| Crash during recovery | 10-30 ms | 10-50 ms | 20-80 ms | Resume position |
---
## Design Decisions: Why This, Not That
### At-Least-Once vs Exactly-Once
| Approach | Complexity | Duplicates | Loss | Latency |
|----------|------------|------------|------|---------|
| **At-least-once** | Medium | Yes | No | Low |
| Exactly-once | High | No | No | High |
| At-most-once | Low | No | Yes | Lowest |
**Choice**: At-least-once with idempotent consumers. Exactly-once requires distributed consensus (Paxos/Raft) which adds 10-100ms latency. Idempotent consumers are simpler and faster.
### Sync vs Async Durability
| Approach | Latency | Data Loss Window | Complexity |
|----------|---------|------------------|------------|
| **No durability** | 500 ns | All in-flight | None |
| Sync per message | 1-5 ms | None | Low |
| Async batched | 500 ns + eventual | Last batch | Medium |
| Hybrid (sync important) | Variable | Last unimportant batch | High |
**Choice**: Async batched for most messages, sync for critical messages (via message flags). This gives low latency for 99% of messages while ensuring critical data is never lost.
### Checkpoint Frequency
| Frequency | Recovery Time | Runtime Overhead | Data Loss |
|-----------|---------------|------------------|-----------|
| Every 1s | 10-30 ms | High (disk I/O) | < 1s |
| Every 10s | 50-150 ms | Medium | < 10s |
| Every 60s | 200-500 ms | Low | < 60s |
**Choice**: Every 10 seconds. This balances recovery time (< 100ms) with runtime overhead. Increase frequency for systems with stricter recovery requirements.
### Recovery Strategy: Reset vs Replay
| Strategy | Recovery Time | Data Loss | Complexity |
|----------|---------------|-----------|------------|
| **Reset to empty** | < 1 ms | 100% | None |
| Replay from WAL | Seconds-minutes | 0% | High |
| Checkpoint + WAL | < 100 ms | 0% | Medium |
**Choice**: Checkpoint + WAL for production systems. Reset to empty is acceptable for development or when message loss is tolerable (e.g., real-time telemetry).
---
## Knowledge Cascade: What You've Unlocked
**1. Write-Ahead Logging in Databases (Cross-Domain)**
The WAL you just implemented is the exact same technique used by every major database:
- **PostgreSQL**: WAL stored in `pg_wal/` directory, enables point-in-time recovery
- **MySQL InnoDB**: Redo log for crash recovery, undo log for rollback
- **SQLite**: WAL mode for better concurrent read performance
- **LMDB**: Write transactions go to a log before the main database
Understanding your message bus WAL gives you insight into why databases:
- Checkpoint periodically (to bound recovery time)
- Have complex log truncation logic (to prevent unbounded log growth)
- Care about `fsync` semantics (data corruption is worse than data loss)
**2. Two-Phase Commit and Why It's Slow**
The "exactly-once" problem you encountered is a subset of the **Two-Phase Commit (2PC)** problem in distributed transactions. 2PC requires:
1. Prepare phase: All participants promise to commit
2. Commit phase: All participants actually commit
Each phase requires disk `fsync`. This means 2+ disk operations per transaction—too slow for high-throughput systems.
Modern alternatives:
- **Saga pattern**: Compensating transactions instead of atomic commit
- **Idempotent retries**: Accept duplicates, deduplicate at consumer
- **Kafka Transactions**: Uses epoch-based fencing (similar to your fence-based recovery)
**3. Kafka's Exactly-Once Semantics**
Kafka achieves "effective exactly-once" through:
1. **Idempotent producers**: Each message has a sequence number, broker detects duplicates
2. **Transactional writes**: Multiple messages atomically committed together
3. **Epoch fencing**: Old producer instances can't write after restart
Your fence-based recovery implements the same principle: a restarted producer gets a new epoch/fence, invalidating old claims.
**4. Process Supervision (Cross-Domain: DevOps)**
The signal handling you implemented connects to the broader world of process supervision:
- **systemd**: Sends SIGTERM for graceful shutdown, SIGKILL after timeout
- **Kubernetes**: Uses container health checks + SIGTERM for pod termination
- **Supervisord**: Similar signal-based lifecycle management
Key insight: `SIGKILL` cannot be caught. If your graceful shutdown takes longer than `TimeoutStopSec`, systemd sends SIGKILL and you lose any in-flight state. Design your shutdown to complete within the timeout.
**5. Chaos Engineering**
The crash recovery you've built should be tested systematically:
```bash
# Kill during publish
kill -9 $(pidof producer) &
./run_benchmark.sh
# Kill during WAL sync
while true; do
    sleep 0.1
    kill -9 $(pidof broker) 2>/dev/null || true
done &
# Kill during recovery
kill -9 $(pidof broker)
sleep 0.01
kill -9 $(pidof broker)  # Second kill during recovery
```
Tools like **Chaos Mesh** and **Litmus** automate this kind of testing in Kubernetes environments.
---
## Implementation Checklist
Before considering this milestone complete, ensure you have:
1. **Crash Detection**
   - Heartbeat monitoring with configurable timeout
   - PID-based liveness check
   - Generation counter to handle PID reuse
   - Integration with process registry from M4
2. **Orphaned Slot Recovery**
   - Timeout-based detection of stalled slots
   - Fence-based invalidation of old claims
   - Safe reclamation without racing with recovering processes
   - Logging of recovery actions
3. **Write-Ahead Logging (Optional Durability)**
   - Append-only log with checksums
   - `fsync` for durability (or `O_DSYNC`)
   - Truncation of processed entries
   - Replay capability for recovery
4. **Checkpoint/Restart**
   - Periodic checkpoint snapshots
   - Atomic checkpoint files (write to temp, rename)
   - Checkpoint + WAL recovery
   - Recovery time < 100ms for typical scenarios
5. **Crash-During-Recovery Handling**
   - Recovery journal for tracking progress
   - Ability to resume interrupted recovery
   - Idempotent recovery operations
6. **Process Supervision Integration**
   - Signal handlers for graceful shutdown
   - Integration with systemd (or equivalent)
   - Timeout-based forced shutdown
   - sd_notify for readiness notification
7. **At-Least-Once Semantics**
   - Message deduplication support (idempotent consumer pattern)
   - Sequence numbers for duplicate detection
   - Clear documentation that duplicates are possible
8. **Benchmarks**
   - Recovery time measurement (checkpoint load + WAL replay)
   - Durability latency impact (sync vs async)
   - Throughput impact of checkpointing
   - Crash-recovery-crash scenarios
---
[[CRITERIA_JSON: {"milestone_id": "zcmb-m5", "criteria": ["Crash detection via heartbeat monitoring with configurable timeout and PID-based liveness checks with generation counter to handle PID reuse", "Orphaned slot recovery using timeout-based detection and fence-based invalidation preventing race conditions with recovering processes", "Optional write-ahead logging with checksums, fsync durability, log truncation, and replay capability", "Checkpoint/restart capability with periodic snapshots, atomic file writes, and checkpoint-plus-WAL recovery achieving under 100ms recovery time", "Recovery journaling for crash-during-recovery scenarios with ability to resume interrupted recovery operations", "Signal handlers for graceful shutdown with SIGTERM/SIGINT handling and integration with process supervisors like systemd", "At-least-once delivery semantics with documentation that duplicates are possible and consumers must be idempotent", "Batched asynchronous durability option reducing latency impact while accepting bounded data loss window", "Recovery benchmarks measuring checkpoint load time, WAL replay time, and total recovery time under various failure scenarios", "Process identity tracking combining PID and generation counter to distinguish process restarts from PID reuse"]}]
<!-- END_MS -->




# TDD

A high-throughput inter-process messaging system that eliminates data copying by leveraging shared memory regions, lock-free ring buffers, and flat buffer serialization. The system achieves sub-microsecond latency through direct memory access, careful cache line alignment, and minimal kernel involvement after initial setup. Built for high-frequency trading platforms, real-time analytics, and low-latency microservices where traditional IPC mechanisms cannot meet performance requirements.



<!-- TDD_MOD_ID: zcmb-m1 -->
# Technical Design Specification: Shared Memory Ring Buffer
## Module Charter
The Shared Memory Ring Buffer provides zero-copy inter-process communication through a lock-free SPSC (Single-Producer Single-Consumer) queue residing in POSIX shared memory. The module abstracts away shared memory lifecycle management, cross-process memory visibility concerns, and cache-efficient data layout. **What it does NOT do**: multi-producer coordination (M3), message serialization (M2), or pub/sub routing (M4). **Upstream dependencies**: POSIX IPC (`shm_open`, `mmap`), C++ atomic library. **Downstream consumers**: Zero-copy serialization layer (M2), which embeds flat buffers into ring buffer slots. **Core invariants**: (1) `head` and `tail` indices are cache-line separated to prevent false sharing, (2) buffer size is always power-of-2 enabling bitmask wraparound, (3) at least one slot remains empty to distinguish full from empty, (4) producer only writes `head`, consumer only writes `tail`, (5) all data writes complete before index updates become visible (release semantics), (6) index reads acquire visibility of all prior writes.
---
## File Structure
```
zcmb-m1-shared-memory-ring-buffer/
├── 01_include/
│   └── zcmb/
│       ├── 01_shm_allocator.hpp      # SharedMemory RAII wrapper
│       ├── 02_spsc_ring_buffer.hpp   # Lock-free SPSC queue
│       └── 03_ring_buffer_config.hpp # Configuration structs
├── 02_src/
│   ├── 01_shm_allocator.cpp          # Implementation (error handling)
│   └── 02_spsc_ring_buffer.cpp       # Implementation (barriers)
├── 03_tests/
│   ├── 01_shm_allocator_test.cpp     # Creation, mapping, cleanup
│   ├── 02_spsc_ring_buffer_test.cpp  # Produce/consume, wraparound
│   ├── 03_cross_process_test.cpp     # Fork-based IPC test
│   └── 04_crash_recovery_test.cpp    # Orphan detection
├── 04_benchmarks/
│   ├── 01_latency_bench.cpp          # Round-trip measurement
│   └── 02_throughput_bench.cpp       # Messages per second
└── 05_CMakeLists.txt
```
---
## Complete Data Model
### RingBufferConfig
```cpp
// File: 01_include/zcmb/03_ring_buffer_config.hpp
#pragma once
#include <cstdint>
#include <cstddef>
namespace zcmb {
struct RingBufferConfig {
    size_t num_slots;      // Number of message slots (MUST be power of 2)
    size_t slot_size;      // Size of each slot in bytes
    const char* shm_name;  // POSIX shm name (e.g., "/zcmb_ring_0")
    // Validates power-of-2 constraint
    bool is_valid() const noexcept;
    // Calculates total shared memory bytes needed
    size_t calculate_total_size() const noexcept;
    // Power-of-2 validation helper
    static bool is_power_of_two(size_t n) noexcept {
        return n > 0 && (n & (n - 1)) == 0;
    }
};
} // namespace zcmb
```
**Field Rationale**:
| Field | Why It Exists |
|-------|---------------|
| `num_slots` | Determines buffer capacity. Power-of-2 constraint enables `index & mask` instead of `index % size`, saving ~20 cycles per wraparound. |
| `slot_size` | Fixed-size slots simplify memory management. Variable-length messages handled by serialization layer (M2). |
| `shm_name` | POSIX shared memory is identified by name. Must start with `/` on most systems. |
### Header (In Shared Memory)
```cpp
// Layout at offset 0 of shared memory region
// MUST match exactly between producer and consumer processes
struct alignas(64) RingBufferHeader {
    // Cache line 0: Producer-owned index (bytes 0-63)
    alignas(64) std::atomic<uint64_t> head;     // Offset 0x00: Producer write position
    uint8_t padding0[56];                        // Offset 0x08: Padding to 64 bytes
    // Cache line 1: Consumer-owned index (bytes 64-127)
    alignas(64) std::atomic<uint64_t> tail;     // Offset 0x40: Consumer read position
    uint8_t padding1[56];                        // Offset 0x48: Padding to 64 bytes
    // Cache line 2: Metadata (bytes 128-191)
    uint64_t num_slots;                          // Offset 0x80: Capacity (power of 2)
    uint64_t slot_size;                          // Offset 0x88: Bytes per slot
    uint64_t mask;                               // Offset 0x90: num_slots - 1 for fast modulo
    uint64_t sequence_base;                      // Offset 0x98: Initial sequence number
    // Cache line 3: Process coordination (bytes 192-255)
    alignas(64) std::atomic<uint32_t> producer_pid;   // Offset 0xC0: Producer's PID
    alignas(64) std::atomic<uint32_t> consumer_pid;   // Offset 0xC4: Consumer's PID
    std::atomic<uint64_t> producer_heartbeat;    // Offset 0xC8: Last producer timestamp (ns)
    std::atomic<uint64_t> consumer_heartbeat;    // Offset 0xD0: Last consumer timestamp (ns)
    std::atomic<uint32_t> generation;            // Offset 0xD8: Incremented on reset
    uint8_t padding3[16];                        // Offset 0xDC: Padding to 256 bytes
};
static_assert(sizeof(RingBufferHeader) == 256, "Header must be exactly 256 bytes");
```
**Memory Layout Diagram**:
```
Shared Memory Region Layout:
┌─────────────────────────────────────────────────────────────┐
│ Offset 0x000: RingBufferHeader (256 bytes)                  │
│   ├─ 0x000-0x03F: head (atomic) + padding [Cache Line 0]    │
│   ├─ 0x040-0x07F: tail (atomic) + padding [Cache Line 1]    │
│   ├─ 0x080-0x0BF: num_slots, slot_size, mask, seq_base      │
│   └─ 0x0C0-0x0FF: PIDs, heartbeats, generation [CL 2-3]     │
├─────────────────────────────────────────────────────────────┤
│ Offset 0x100: Slot 0 (slot_size bytes)                      │
│   ├─ 0x100: message data...                                 │
├─────────────────────────────────────────────────────────────┤
│ Offset 0x100 + slot_size: Slot 1                            │
├─────────────────────────────────────────────────────────────┤
│ Offset 0x100 + 2*slot_size: Slot 2                          │
├─────────────────────────────────────────────────────────────┤
│ ...                                                         │
├─────────────────────────────────────────────────────────────┤
│ Offset 0x100 + (num_slots-1)*slot_size: Slot N-1            │
└─────────────────────────────────────────────────────────────┘
Total size = 256 + (num_slots × slot_size) bytes
```
**Why Cache Line Separation Matters**:
```
WITHOUT alignas(64):
  head at 0x00, tail at 0x08 → SAME 64-byte cache line
  Producer writes head → invalidates consumer's cache line
  Consumer reads head → cache miss, fetch from producer's core
  Result: 10-100x latency degradation on multi-core systems
WITH alignas(64):
  head at 0x00 (cache line 0) → producer's L1: hot
  tail at 0x40 (cache line 1) → consumer's L1: hot
  Cross-core traffic only on index reads, not writes
  Result: Predictable sub-microsecond latency
```
### SharedMemory Class
```cpp
// File: 01_include/zcmb/01_shm_allocator.hpp
#pragma once
#include <string>
#include <cstdint>
namespace zcmb {
class SharedMemory {
public:
    static constexpr size_t PAGE_SIZE = 4096;
    // Creates OR opens shared memory
    // create=true: Creates new, truncates to size
    // create=false: Opens existing, size must match
    SharedMemory(const std::string& name, size_t size, bool create);
    // RAII: unmaps memory, closes fd, unlinks if creator
    ~SharedMemory() noexcept;
    // Move-only (not copyable)
    SharedMemory(SharedMemory&& other) noexcept;
    SharedMemory& operator=(SharedMemory&& other) noexcept;
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;
    // Access
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    int fd() const noexcept { return fd_; }
    bool is_creator() const noexcept { return is_creator_; }
    const std::string& name() const noexcept { return name_; }
private:
    static size_t align_to_page(size_t size) noexcept {
        return (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    }
    std::string name_;
    size_t size_;
    int fd_;
    uint8_t* data_;
    bool is_creator_;
};
} // namespace zcmb
```
### SpscRingBuffer Class
```cpp
// File: 01_include/zcmb/02_spsc_ring_buffer.hpp
#pragma once
#include "03_ring_buffer_config.hpp"
#include "01_shm_allocator.hpp"
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <chrono>
namespace zcmb {
class SpscRingBuffer {
public:
    // Header stored at beginning of shared memory
    using Header = RingBufferHeader;
    static constexpr size_t HEADER_SIZE = sizeof(Header);
    // === Lifecycle ===
    // Initialize as creator (sets up header)
    // Throws std::invalid_argument if config invalid
    void initialize(void* memory, const RingBufferConfig& config);
    // Attach to existing buffer (producer or consumer)
    // Memory must point to valid RingBufferHeader
    void attach(void* memory) noexcept;
    // === Production (Single Producer Only) ===
    // Non-blocking: returns false if buffer full
    [[nodiscard]] bool try_produce(const void* data, size_t size) noexcept;
    // Blocking: spins then yields until space available
    void produce(const void* data, size_t size);
    // === Consumption (Single Consumer Only) ===
    // Non-blocking: returns false if buffer empty
    [[nodiscard]] bool try_consume(void* out_data, size_t out_size) noexcept;
    // Blocking: spins then yields until data available
    void consume(void* out_data, size_t out_size);
    // === Queries ===
    size_t capacity() const noexcept;      // Max messages (num_slots - 1)
    size_t slot_size() const noexcept;     // Bytes per slot
    size_t count() const noexcept;         // Current message count
    bool empty() const noexcept;           // head == tail
    bool full() const noexcept;            // (head + 1) & mask == tail
    // === Process Coordination ===
    void update_producer_heartbeat() noexcept;
    void update_consumer_heartbeat() noexcept;
    bool is_producer_alive(std::chrono::nanoseconds timeout) const noexcept;
    bool is_consumer_alive(std::chrono::nanoseconds timeout) const noexcept;
    void register_producer() noexcept;     // Sets producer_pid
    void register_consumer() noexcept;     // Sets consumer_pid
    // === Crash Recovery ===
    // Reset buffer state (clear all messages)
    // WARNING: Only safe when no other process is using buffer
    void reset() noexcept;
    // Calculate total shared memory size needed
    static size_t calculate_size(const RingBufferConfig& config) noexcept {
        return HEADER_SIZE + config.num_slots * config.slot_size;
    }
private:
    void* get_slot(uint64_t index) noexcept;
    const void* get_slot(uint64_t index) const noexcept;
    static uint64_t now_ns() noexcept {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(
            high_resolution_clock::now().time_since_epoch()).count();
    }
    Header* header_ = nullptr;
    uint8_t* slots_ = nullptr;
};
} // namespace zcmb
```
---
## Interface Contracts
### SharedMemory::SharedMemory
```cpp
SharedMemory(const std::string& name, size_t size, bool create);
```
**Preconditions**:
- `name` starts with `/` and contains no other `/` (POSIX requirement)
- `size` > 0
- If `create == false`, shared memory object must already exist
**Postconditions**:
- On success: `data()` returns pointer to mapped memory, `size()` returns actual size
- On failure: exception thrown, no resources leaked
**Errors**:
| Error Condition | Exception Type | Recovery |
|-----------------|----------------|----------|
| `shm_open` fails with EACCES | `std::runtime_error("SHM_PERMISSION_DENIED")` | Check user permissions, try different name |
| `shm_open` fails with EEXIST | `std::runtime_error("SHM_EXISTS")` | Use `create=false` or unlink first |
| `shm_open` fails with ENOSPC | `std::runtime_error("SHM_CREATE_FAILED")` | Free disk space in `/dev/shm` |
| `ftruncate` fails | `std::runtime_error("SHM_CREATE_FAILED")` | Check disk quotas |
| `mmap` returns MAP_FAILED | `std::runtime_error("SHM_MAP_FAILED")` | Check RLIMIT_AS, reduce size |
**Edge Cases**:
- Size not page-aligned: `align_to_page()` rounds up
- Opening with different size than created: undefined behavior (caller's responsibility)
### SpscRingBuffer::try_produce
```cpp
[[nodiscard]] bool try_produce(const void* data, size_t size) noexcept;
```
**Preconditions**:
- `attach()` or `initialize()` called successfully
- Calling process is the SINGLE producer (no concurrent producers)
- `data != nullptr` if `size > 0`
**Postconditions**:
- Returns `true`: message copied to next available slot, `head` atomically advanced
- Returns `false`: buffer full, no state changed
- Memory ordering: All writes to slot complete BEFORE `head` update becomes visible
**Atomic Operation Sequence**:
```
1. head ← load(relaxed)           // Our position
2. tail ← load(acquire)           // Consumer's position
3. next ← (head + 1) & mask
4. if next == tail: return false  // Buffer full (one slot must stay empty)
5. memcpy(slot[head], data, min(size, slot_size))
6. atomic_thread_fence(release)   // Ensure memcpy visible before head update
7. store(head, next, relaxed)     // Publish
8. return true
```
**Why relaxed + fence instead of release store**:
The `atomic_thread_fence(std::memory_order_release)` ensures ALL prior memory operations (the memcpy) complete before ANY subsequent memory operations (the head store). A simple `store(release)` only orders operations on that atomic variable.
### SpscRingBuffer::try_consume
```cpp
[[nodiscard]] bool try_consume(void* out_data, size_t out_size) noexcept;
```
**Preconditions**:
- `attach()` or `initialize()` called successfully
- Calling process is the SINGLE consumer (no concurrent consumers)
- `out_data != nullptr` if `out_size > 0`
**Postconditions**:
- Returns `true`: message copied from current slot, `tail` atomically advanced
- Returns `false`: buffer empty, `out_data` unmodified
- Memory ordering: `head` load (acquire) happens BEFORE slot read
**Atomic Operation Sequence**:
```
1. tail ← load(relaxed)           // Our position
2. head ← load(acquire)           // Producer's position (sees all prior writes)
3. if tail == head: return false  // Buffer empty
4. atomic_thread_fence(acquire)   // Ensure we see data after head was updated
5. memcpy(out_data, slot[tail], min(out_size, slot_size))
6. next ← (tail + 1) & mask
7. store(tail, next, relaxed)     // Advance
8. return true
```
### SpscRingBuffer::count
```cpp
size_t count() const noexcept;
```
**Returns**: Approximate number of messages currently in buffer.
**Important**: This is a SNAPSHOT. By the time you use the result, the actual count may have changed (producer/consumer advanced). Use only for monitoring, not for correctness decisions.
**Formula**: `(head - tail + num_slots) & mask`
**Why this works**: With power-of-2 size and mask, subtraction naturally wraps. If head=3, tail=5, num_slots=8, mask=7:
- `3 - 5 = -2` (as int64_t)
- `-2 + 8 = 6`
- `6 & 7 = 6`
Wait, that's wrong. Let me reconsider:
The ring buffer uses monotonic sequence numbers (head and tail increase forever, never wrap). The actual slot index is `position & mask`. The count is simply `head - tail` (the number of positions the producer is ahead of the consumer).
```cpp
size_t count() const noexcept {
    uint64_t h = header_->head.load(std::memory_order_acquire);
    uint64_t t = header_->tail.load(std::memory_order_acquire);
    return static_cast<size_t>(h - t);
}
```
This works because `head >= tail` always (producer can't get ahead of consumer's free slots).
---
## Algorithm Specification
### Algorithm: Initialize Shared Memory Region
**Purpose**: Set up ring buffer header in freshly created shared memory.
**Input**: `memory` pointer to mapped shared memory, `config` with validated parameters
**Output**: Header initialized, slots array positioned
**Steps**:
```
1. Validate config.is_valid() → false: throw invalid_argument
2. Cast memory to Header*
3. Use placement new to zero-initialize:
   new (header) Header{}
4. Set header fields:
   - num_slots = config.num_slots
   - slot_size = config.slot_size  
   - mask = config.num_slots - 1
   - sequence_base = 0
   - head.store(0, relaxed)
   - tail.store(0, relaxed)
   - generation.store(0, relaxed)
5. Position slots pointer:
   - slots_ = reinterpret_cast<uint8_t*>(memory) + HEADER_SIZE
6. Store header_ and slots_ members
```
**Invariant After Execution**: 
- `header_->num_slots` is power of 2
- `header_->mask == header_->num_slots - 1`
- `slots_[0]` is at offset `HEADER_SIZE` from `memory`
### Algorithm: Produce Message (Blocking)
**Purpose**: Copy message to ring buffer, blocking until space available.
**Input**: `data` pointer, `size` bytes
**Output**: Message in ring buffer, head advanced
**Steps**:
```
1. spin_count ← 0
2. LOOP:
   a. IF try_produce(data, size): RETURN
   b. spin_count ← spin_count + 1
   c. IF spin_count < 100:
      - Execute PAUSE instruction (x86) or yield hint
      - CONTINUE
   d. IF spin_count < 10000:
      - std::this_thread::yield()
      - CONTINUE
   e. IF spin_count >= 10000:
      - std::this_thread::sleep_for(1us)
      - CONTINUE
```
**Backoff Rationale**:
- First 100 iterations: Spin with PAUSE (~100-500ns each). For a fast consumer, space becomes available almost immediately.
- Next 9900 iterations: Yield (~1-10μs). Gives other threads CPU time while waiting.
- Beyond 10000: Sleep (~1μs+). Consumer is severely backed up; spinning wastes power.
**Edge Case**: Consumer crashed while buffer full. Detection via heartbeat timeout (see process coordination).
### Algorithm: Crash Recovery - Orphaned Slot Detection
**Purpose**: Detect and recover slots claimed by crashed producer.
**Input**: Header with producer heartbeat, current time
**Output**: Buffer state reset if producer dead
**Steps**:
```
1. last_heartbeat ← header_->producer_heartbeat.load(acquire)
2. now ← current_timestamp_ns()
3. IF (now - last_heartbeat) > TIMEOUT_NS:
   a. producer_pid ← header_->producer_pid.load(relaxed)
   b. IF kill(producer_pid, 0) == -1 AND errno == ESRCH:
      - Producer process is DEAD
      - header_->head.store(tail, relaxed)  // Reset head to tail
      - header_->generation.fetch_add(1, relaxed)
      - LOG("Recovered orphaned buffer from crashed producer")
```
**Timeout Selection**: 
- Too short (1s): False positives during GC pauses, scheduling delays
- Too long (60s): Slow detection, blocked consumers
- Recommended: 5-10 seconds for most workloads
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| SHM_CREATE_FAILED | `shm_open` returns -1 | Retry with different name, check disk space | Yes - startup fails |
| SHM_MAP_FAILED | `mmap` returns MAP_FAILED | Reduce buffer size, check ulimits | Yes - startup fails |
| SHM_PERMISSION_DENIED | `shm_open` returns EACCES | Run with correct user/group | Yes - startup fails |
| BUFFER_FULL | `try_produce` detects `next == tail` | Caller implements backpressure or drops | Depends on caller |
| BUFFER_EMPTY | `try_consume` detects `head == tail` | Caller waits or does other work | No - normal condition |
| INVALID_CONFIG | `is_valid()` returns false | Fix config, ensure power-of-2 | Yes - startup fails |
| PROCESS_CRASH_DETECTED | Heartbeat timeout + PID check | Reset buffer, notify application layer | Yes - recovery event |
| PRODUCER_TOO_FAST | Persistent BUFFER_FULL | Backpressure to producer | No - handled internally |
**No Silent Failures**: Every error path either throws (startup errors) or returns explicit status (runtime errors). The buffer never corrupts silently.
---
## Implementation Sequence with Checkpoints
### Phase 1: Shared Memory Allocator (3-4 hours)
**Files to Create**: `01_shm_allocator.hpp`, `01_shm_allocator.cpp`
**Implementation Steps**:
1. Define `SharedMemory` class with RAII semantics
2. Implement constructor: `shm_open` → `ftruncate` (if create) → `mmap`
3. Implement destructor: `munmap` → `close` → `shm_unlink` (if creator)
4. Implement move constructor/assignment
5. Add error handling with specific exception messages
**Checkpoint**: 
```bash
# Build and run test
./03_tests/01_shm_allocator_test
# Expected output:
# [PASS] Create and map shared memory
# [PASS] Open existing shared memory
# [PASS] Size alignment to page boundary
# [PASS] RAII cleanup (verify with ls /dev/shm)
# [PASS] Move semantics
```
**At this point you should be able to**: Create a named shared memory region from one process, open it from another process (forked child), and verify both see the same bytes.
### Phase 2: SPSC Ring Buffer Core (4-5 hours)
**Files to Create**: `02_spsc_ring_buffer.hpp`, `02_spsc_ring_buffer.cpp`, `03_ring_buffer_config.hpp`
**Implementation Steps**:
1. Define `RingBufferConfig` with power-of-2 validation
2. Define `RingBufferHeader` with exact byte layout (256 bytes)
3. Implement `initialize()` - placement new, field setup
4. Implement `attach()` - cast pointer, validate magic (optional)
5. Implement `try_produce()` - load indices, check full, memcpy, store
6. Implement `try_consume()` - load indices, check empty, memcpy, store
7. Implement blocking variants with exponential backoff
8. Implement queries: `count()`, `empty()`, `full()`, `capacity()`
**Checkpoint**:
```bash
./03_tests/02_spsc_ring_buffer_test
# Expected output:
# [PASS] Initialize with valid config
# [PASS] Reject non-power-of-2 config
# [PASS] Produce and consume single message
# [PASS] Fill buffer to capacity
# [PASS] Empty buffer completely
# [PASS] Wraparound at buffer end
# [PASS] count() matches actual messages
```
**At this point you should be able to**: Produce and consume messages in a single process (unit test), verify wraparound works correctly, confirm power-of-2 mask arithmetic.
### Phase 3: Memory Barriers & Visibility (2-3 hours)
**Files to Modify**: `02_spsc_ring_buffer.cpp`
**Implementation Steps**:
1. Add `atomic_thread_fence(release)` after memcpy in `try_produce`
2. Add `atomic_thread_fence(acquire)` after head load in `try_consume`
3. Document x86 vs ARM behavior differences
4. Add static assertions for atomic lock-free guarantee
5. Run ThreadSanitizer tests
**Platform-Specific Notes**:
| Platform | StoreStore | LoadLoad | StoreLoad | Implementation |
|----------|-----------|----------|-----------|----------------|
| x86/x64 | Implicit | Implicit | Requires mfence | Our release/acquire fences compile to no-ops for most operations |
| ARM64 | Explicit (dmb) | Explicit (dmb) | Explicit (dmb) | Release fence → `stlr`, Acquire fence → `ldar` |
| POWER | Explicit (lwsync) | Explicit (lwsync+isync) | Explicit (sync) | Full fence required |
**Checkpoint**:
```bash
# Run with ThreadSanitizer
cmake -DCMAKE_CXX_FLAGS="-fsanitize=thread" ..
./03_tests/03_cross_process_test
# Expected: No data races detected
# [PASS] Cross-process produce/consume
# [PASS] No TSAN warnings
```
**At this point you should be able to**: Run producer and consumer in separate processes (forked), pass messages between them, verify no data races with TSAN.
### Phase 4: Process Coordination (2-3 hours)
**Files to Modify**: `02_spsc_ring_buffer.hpp`, `02_spsc_ring_buffer.cpp`
**Implementation Steps**:
1. Add heartbeat fields to header (already in struct)
2. Implement `update_producer_heartbeat()` / `update_consumer_heartbeat()`
3. Implement `is_producer_alive()` / `is_consumer_alive()` with timeout
4. Implement `register_producer()` / `register_consumer()` with PID
5. Implement `reset()` for crash recovery
**Checkpoint**:
```bash
./03_tests/04_crash_recovery_test
# Expected output:
# [PASS] Heartbeat updates timestamp
# [PASS] is_alive returns true within timeout
# [PASS] is_alive returns false after timeout
# [PASS] PID registration works
# [PASS] Reset clears buffer state
```
**At this point you should be able to**: Detect when the other process has crashed (simulated by not updating heartbeat), and reset the buffer to a clean state.
### Phase 5: Latency Benchmark Suite (1-2 hours)
**Files to Create**: `04_benchmarks/01_latency_bench.cpp`, `04_benchmarks/02_throughput_bench.cpp`
**Implementation Steps**:
1. Create benchmark harness with `std::chrono::high_resolution_clock`
2. Measure round-trip: produce → consume → timestamp
3. Collect 1M+ samples, compute percentiles
4. Add throughput test: messages/second over 10 seconds
5. Add perf integration for cache miss analysis
**Benchmark Output Format**:
```
Ring Buffer Latency Benchmark
==============================
Config: 1024 slots × 256 bytes = 262400 bytes total
Samples: 1000000
Latency (ns):
  Min:     87
  P50:     142
  P90:     198
  P99:     347
  P99.9:   2156
  P99.99:  48231
  Max:     125432
Throughput: 12.4 M msg/s
CPU cycles per message: 241 (at 3.0 GHz)
```
**Checkpoint**:
```bash
./04_benchmarks/01_latency_bench
# Verify:
# - Median latency < 500 ns
# - P99 latency < 1 μs
# - Throughput > 5 M msg/s
```
---
## Test Specification
### Test: SharedMemory Creation and Cleanup
```cpp
TEST(SharedMemory, CreateAndCleanup) {
    const char* name = "/zcmb_test_create";
    size_t size = 4096;
    // Create
    {
        SharedMemory shm(name, size, true);
        EXPECT_NE(shm.data(), nullptr);
        EXPECT_EQ(shm.size(), size);
        EXPECT_TRUE(shm.is_creator());
        // Write pattern
        uint8_t* data = static_cast<uint8_t*>(shm.data());
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<uint8_t>(i & 0xFF);
        }
    }
    // shm destroyed here
    // Verify cleanup (shm_unlink called)
    int fd = shm_open(name, O_RDONLY, 0);
    EXPECT_EQ(fd, -1);
    EXPECT_EQ(errno, ENOENT);
}
```
### Test: Ring Buffer Wraparound
```cpp
TEST(SpscRingBuffer, Wraparound) {
    RingBufferConfig config{.num_slots = 4, .slot_size = 16};  // Power of 2
    size_t total_size = SpscRingBuffer::calculate_size(config);
    auto memory = std::make_unique<uint8_t[]>(total_size);
    SpscRingBuffer ring;
    ring.initialize(memory.get(), config);
    uint8_t msg[16];
    uint8_t recv[16];
    // Fill buffer (capacity = 3, one slot reserved)
    for (int i = 0; i < 3; ++i) {
        memset(msg, static_cast<uint8_t>(i), sizeof(msg));
        EXPECT_TRUE(ring.try_produce(msg, sizeof(msg)));
    }
    EXPECT_TRUE(ring.full());
    // Consume 2
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(ring.try_consume(recv, sizeof(recv)));
        EXPECT_EQ(recv[0], static_cast<uint8_t>(i));
    }
    // Produce 2 more (wraps around)
    for (int i = 3; i < 5; ++i) {
        memset(msg, static_cast<uint8_t>(i), sizeof(msg));
        EXPECT_TRUE(ring.try_produce(msg, sizeof(msg)));
    }
    // Consume remaining 3
    int expected[] = {2, 3, 4};
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(ring.try_consume(recv, sizeof(recv)));
        EXPECT_EQ(recv[0], static_cast<uint8_t>(expected[i]));
    }
    EXPECT_TRUE(ring.empty());
}
```
### Test: Cross-Process Communication
```cpp
TEST(SpscRingBuffer, CrossProcess) {
    const char* shm_name = "/zcmb_cross_process_test";
    RingBufferConfig config{.num_slots = 64, .slot_size = 256, .shm_name = shm_name};
    pid_t pid = fork();
    ASSERT_NE(pid, -1);
    if (pid == 0) {
        // Child: Consumer
        SharedMemory shm(shm_name, SpscRingBuffer::calculate_size(config), false);
        SpscRingBuffer ring;
        ring.attach(shm.data());
        ring.register_consumer();
        uint8_t buffer[256];
        for (int i = 0; i < 10000; ++i) {
            while (!ring.try_consume(buffer, sizeof(buffer))) {
                std::this_thread::yield();
            }
            // Verify message
            EXPECT_EQ(buffer[0], static_cast<uint8_t>(i & 0xFF));
        }
        _exit(testing::Test::HasFailure() ? 1 : 0);
    } else {
        // Parent: Producer
        SharedMemory shm(shm_name, SpscRingBuffer::calculate_size(config), true);
        SpscRingBuffer ring;
        ring.initialize(shm.data(), config);
        ring.register_producer();
        uint8_t buffer[256];
        for (int i = 0; i < 10000; ++i) {
            memset(buffer, static_cast<uint8_t>(i & 0xFF), sizeof(buffer));
            ring.produce(buffer, sizeof(buffer));
            ring.update_producer_heartbeat();
        }
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);
    }
}
```
### Test: Invalid Config Rejection
```cpp
TEST(SpscRingBuffer, RejectsInvalidConfig) {
    RingBufferConfig configs[] = {
        {.num_slots = 0, .slot_size = 256},      // Zero slots
        {.num_slots = 3, .slot_size = 256},      // Non-power-of-2
        {.num_slots = 100, .slot_size = 256},    // Non-power-of-2
        {.num_slots = 1024, .slot_size = 0},     // Zero slot size
    };
    for (const auto& config : configs) {
        EXPECT_FALSE(config.is_valid()) << "num_slots=" << config.num_slots;
    }
}
```
### Test: Crash Detection
```cpp
TEST(SpscRingBuffer, DetectsCrashedProducer) {
    auto memory = std::make_unique<uint8_t[]>(4096);
    SpscRingBuffer ring;
    RingBufferConfig config{.num_slots = 16, .slot_size = 64};
    ring.initialize(memory.get(), config);
    // Simulate producer registering then "crashing" (no heartbeat updates)
    ring.header_->producer_pid.store(getpid(), std::memory_order_relaxed);
    ring.header_->producer_heartbeat.store(
        ring.now_ns() - std::chrono::nanoseconds(std::chrono::seconds(30)).count(),
        std::memory_order_relaxed
    );
    // Should detect as dead
    EXPECT_FALSE(ring.is_producer_alive(std::chrono::seconds(5)));
}
```
---
## Performance Targets
| Operation | Target | How to Measure |
|-----------|--------|----------------|
| `try_produce` (success) | < 100 ns | `perf stat -e cycles ./latency_bench` |
| `try_consume` (success) | < 100 ns | Same as above |
| Round-trip (produce + consume) | 100-300 ns median | Benchmark 1M+ samples, compute P50 |
| Round-trip P99 | < 1 μs | Same benchmark, compute P99 |
| Round-trip P99.9 | < 50 μs | Same benchmark, compute P99.9 |
| Throughput | 5-20 M msg/s | 10-second sustained test |
| Cache misses per message | < 0.5 | `perf stat -e cache-misses ./throughput_bench` |
| False sharing indicator | < 1% HITM | `perf c2c record ./throughput_bench` |
**Hardware Soul Analysis**:
```
Per-message cache line touches:
  PRODUCER:
    1. Read tail (remote: consumer's cache line) - may miss
    2. Write slot data (local: producer writing to fresh line)
    3. Write head (local: producer's cache line) - always hits
  CONSUMER:
    1. Read head (remote: producer's cache line) - may miss
    2. Read slot data (local: reading from producer's write)
    3. Write tail (local: consumer's cache line) - always hits
Expected: ~1-2 cache misses per message (cross-process index reads)
Worst case: First access after context switch (cold cache) ~100-300 cycles
```
---
## Concurrency Specification
### Lock-Free Guarantee
This implementation is **wait-free** for the happy path:
- `try_produce` completes in bounded time regardless of consumer behavior
- `try_consume` completes in bounded time regardless of producer behavior
- No loops that depend on other process's progress (except blocking variants)
### Memory Ordering Contract
```
PRODUCER                          CONSUMER
========                          =========
store(slot, data)                 
fence(release)                    
store(head, new)  ──────────────→ load(head) [sees new]
                                  fence(acquire)
                                  load(slot) [sees data]
```
**Happens-before relationship**: The `fence(release)` in producer synchronizes-with the `fence(acquire)` in consumer. All writes before the release fence are visible to all reads after the acquire fence.
### Cross-Process vs Intra-Process
**Warning**: `std::atomic` with `memory_order` guarantees visibility between *threads* of the same process. For cross-process visibility:
1. **On x86/x64**: The hardware guarantees cache coherency across processes sharing memory. Release/acquire fences are sufficient.
2. **On ARM**: Additional considerations:
   - Use `std::atomic_thread_fence(std::memory_order_seq_cst)` for full barrier
   - Or use `__sync_synchronize()` (GCC/Clang intrinsic)
   - The `dmb ish` instruction ensures visibility across shareability domain
3. **Testing**: Run `helgrind` (Valgrind tool) and ThreadSanitizer on cross-process tests to verify.
---
## Crash Recovery
### Recovery Procedure
When a process restarts after crash:
```cpp
void SpscRingBuffer::recover_from_crash() {
    // 1. Check if other process is alive
    if (is_producer_alive(std::chrono::seconds(5))) {
        // Producer is alive - we're consumer recovering
        // Just reset our cursor to current head
        header_->tail.store(header_->head.load(std::memory_order_acquire),
                            std::memory_order_relaxed);
        return;
    }
    if (is_consumer_alive(std::chrono::seconds(5))) {
        // Consumer is alive - we're producer recovering
        // Safe to continue from current tail
        header_->head.store(header_->tail.load(std::memory_order_acquire) + 1,
                            std::memory_order_relaxed);
        return;
    }
    // Both dead or unknown - full reset
    reset();
}
```
### State After Recovery
| Scenario | head | tail | Messages | Action |
|----------|------|------|----------|--------|
| Producer crashed, consumer alive | tail | unchanged | All consumed | Producer resumes at tail |
| Consumer crashed, producer alive | unchanged | head | All lost | Consumer resumes at head |
| Both crashed | 0 | 0 | All lost | Full reset |
| Producer crashed mid-write | may be ahead | unchanged | Last message corrupt | Consumer skips if checksum fails |
---
[[CRITERIA_JSON: {"module_id": "zcmb-m1", "criteria": ["SharedMemory class implements POSIX shm_open, mmap with MAP_SHARED, RAII cleanup with shm_unlink for creator role only", "RingBufferHeader is exactly 256 bytes with head at offset 0x00 and tail at offset 0x40 on separate cache lines using alignas(64)", "RingBufferConfig validates power-of-2 constraint on num_slots and calculates total shared memory size as HEADER_SIZE plus num_slots times slot_size", "try_produce returns false when next equals tail indicating buffer full, otherwise copies data to slot and advances head with release fence before store", "try_consume returns false when head equals tail indicating buffer empty, otherwise copies from slot with acquire fence after head load and advances tail", "Produce and consume operations use bitwise AND with mask for wraparound instead of modulo division", "Heartbeat mechanism stores nanosecond timestamps and is_alive compares against configurable timeout threshold", "PID tracking with generation counter distinguishes process restart from PID reuse on Linux systems", "Exponential backoff in blocking produce starts with 100 pause iterations then yields then sleeps at 1 microsecond", "Benchmark measures round-trip latency with median target 100-300 nanoseconds and P99 target below 1 microsecond"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: zcmb-m2 -->
# Technical Design Specification: Zero-Copy Serialization
## Module Charter
The Zero-Copy Serialization module provides direct memory access to structured data without parsing, copying, or heap allocation. It implements a FlatBuffer-style binary format where the serialized bytes ARE the data structure—field access is pointer arithmetic at runtime. **What it does NOT do**: message routing (M4), durability/crash recovery (M5), or multi-producer coordination (M3). **Upstream dependencies**: C++20 standard library, the ring buffer from M1 (for slot placement). **Downstream consumers**: Pub/sub layer (M4) embeds typed messages into topic-routed slots, applications use generated View classes for business logic. **Core invariants**: (1) All scalar fields are naturally aligned (8-byte fields at 8-byte boundaries), (2) vtable offsets are relative to table start, (3) strings/vectors use 32-bit length prefix followed by data, (4) nested tables referenced by 32-bit offset from buffer start, (5) schema evolution preserves field IDs—never reuse, only append, (6) all buffers are little-endian on x86/ARM64 (byte-swapping for big-endian platforms).
---
## File Structure
```
zcmb-m2-zero-copy-serialization/
├── 01_include/
│   └── zcmb/
│       ├── flat/
│       │   ├── 01_scalar_types.hpp        // Type definitions (int8-64, float, double)
│       │   ├── 02_buffer_view.hpp         // Base class for zero-copy access
│       │   ├── 03_vector_view.hpp         // Array accessor template
│       │   ├── 04_string_view.hpp         // String accessor (length + chars)
│       │   ├── 05_builder_base.hpp        // FlatBuffer construction base
│       │   └── 06_vtable.hpp              // VTable structure and utilities
│       ├── schema/
│       │   ├── 01_ast_types.hpp           // Schema AST node types
│       │   ├── 02_schema_parser.hpp       // Text schema → AST
│       │   ├── 03_layout_calculator.hpp   // Field offset computation
│       │   └── 04_schema_registry.hpp     // Runtime schema lookup
│       └── gen/
│           └── 01_code_generator.hpp      // AST → C++ code generator
├── 02_src/
│   ├── flat/
│   │   ├── 01_buffer_view.cpp             // VTable traversal, field access
│   │   ├── 02_builder_base.cpp            // Buffer construction helpers
│   │   └── 03_vtable.cpp                  // VTable utilities
│   ├── schema/
│   │   ├── 01_schema_parser.cpp           // Lexer + recursive descent parser
│   │   ├── 02_layout_calculator.cpp       // Alignment, padding, offset calc
│   │   └── 03_schema_registry.cpp         // Thread-safe schema storage
│   └── gen/
│       └── 01_code_generator.cpp          // C++ code emission
├── 03_schemas/
│   ├── 01_primitives.fbs                  // Built-in types (testing)
│   └── 02_trading.fbs                     // Example: TradeOrder, MarketData
├── 04_generated/
│   └── trading/
│       ├── trade_order.hpp                // Generated View/Builder
│       └── market_data.hpp                // Generated View/Builder
├── 05_tests/
│   ├── 01_scalar_types_test.cpp           // Type size/alignment tests
│   ├── 02_buffer_view_test.cpp            // VTable traversal, field access
│   ├── 03_vector_string_test.cpp          // Array and string handling
│   ├── 04_schema_parser_test.cpp          // Parse all schema constructs
│   ├── 05_layout_calculator_test.cpp      // Alignment, padding correctness
│   ├── 06_code_generator_test.cpp         // Generated code compiles and runs
│   ├── 07_round_trip_test.cpp             // Serialize → deserialize → verify
│   ├── 08_schema_evolution_test.cpp       // Add field, old reader, new reader
│   └── 09_benchmark.cpp                   // Access time vs JSON
├── 06_benchmarks/
│   ├── 01_access_latency.cpp              // Field access nanoseconds
│   ├── 02_build_latency.cpp               // Construction time
│   └── 03_comparison/
│       └── json_benchmark.cpp             // RapidJSON comparison
└── 07_CMakeLists.txt
```
---
## Complete Data Model
### Scalar Types (Platform-Independent)
```cpp
// File: 01_include/zcmb/flat/01_scalar_types.hpp
#pragma once
#include <cstdint>
#include <cstddef>
namespace zcmb::flat {
// Signed integers
using int8   = std::int8_t;
using int16  = std::int16_t;
using int32  = std::int32_t;
using int64  = std::int64_t;
// Unsigned integers
using uint8  = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;
// Floating point (IEEE 754)
using float32 = float;
using float64 = double;
// Boolean (stored as uint8, 0=false, 1=true)
using boolean = uint8;
// Type traits for compile-time size/alignment
template<typename T>
struct ScalarTraits {
    static constexpr size_t size = sizeof(T);
    static constexpr size_t alignment = alignof(T);
    static constexpr bool is_scalar = true;
};
// Type ID enumeration (encoded in schema, used for runtime dispatch)
enum class TypeId : uint8 {
    INT8   = 0,
    INT16  = 1,
    INT32  = 2,
    INT64  = 3,
    UINT8  = 4,
    UINT16 = 5,
    UINT32 = 6,
    UINT64 = 7,
    FLOAT32 = 8,
    FLOAT64 = 9,
    BOOL   = 10,
    STRING = 11,    // Offset to string
    VECTOR = 12,    // Offset to vector
    TABLE  = 13,    // Offset to nested table
    STRUCT = 14,    // Inline struct
    ENUM   = 15     // Enum (stored as underlying type)
};
// Size in bytes for each type ID
constexpr size_t type_size(TypeId id) noexcept {
    switch (id) {
        case TypeId::INT8:   case TypeId::UINT8:  case TypeId::BOOL:   return 1;
        case TypeId::INT16:  case TypeId::UINT16: return 2;
        case TypeId::INT32:  case TypeId::UINT32: case TypeId::FLOAT32: return 4;
        case TypeId::INT64:  case TypeId::UINT64: case TypeId::FLOAT64: return 8;
        case TypeId::STRING: case TypeId::VECTOR: case TypeId::TABLE:  return 4; // Offset
        default: return 0;
    }
}
} // namespace zcmb::flat
```
### VTable Structure (Per-Table Metadata)
The vtable enables schema evolution by describing the layout of each table instance independently.
```cpp
// File: 01_include/zcmb/flat/06_vtable.hpp
#pragma once
#include "01_scalar_types.hpp"
#include <cstdint>
namespace zcmb::flat {
// VTable layout (stored BEFORE the table in memory)
// 
// Memory diagram:
// ┌────────────────────────────────────────────────────────┐
// │ VTable (variable size)                                  │
// │   ├─ vtable_size (uint16): Total vtable bytes          │
// │   ├─ table_size (uint16): Table data bytes (sans vtoff)│
// │   ├─ field_offset[0] (uint16): Field 0 offset or 0     │
// │   ├─ field_offset[1] (uint16): Field 1 offset or 0     │
// │   └─ ...                                               │
// ├────────────────────────────────────────────────────────┤
// │ Table (immediately follows)                            │
// │   ├─ vtable_offset (int32): NEGATIVE offset to vtable  │
// │   ├─ [field data at offsets specified by vtable]       │
// │   └─ ...                                               │
// └────────────────────────────────────────────────────────┘
struct VTable {
    uint16 vtable_size;      // Bytes in this vtable (4 + 2*num_fields)
    uint16 table_size;       // Bytes in table (excluding soffset)
    uint16 field_offsets[];  // Array: field_id → offset from table start
                             // 0 means field not present (use default)
};
// Table preamble (at the start of every table instance)
struct TablePreamble {
    int32 soffset_to_vtable;  // NEGATIVE: table_addr + soffset = vtable_addr
                              // This allows vtable to be BEFORE table in memory
};
static_assert(sizeof(VTable) == 4, "VTable header must be 4 bytes");
static_assert(sizeof(TablePreamble) == 4, "TablePreamble must be 4 bytes");
// VTable utilities
class VTableReader {
public:
    explicit VTableReader(const uint8_t* vtable_ptr) noexcept
        : vtable_(reinterpret_cast<const VTable*>(vtable_ptr)) {}
    // Size of this vtable in bytes
    uint16 vtable_size() const noexcept { return vtable_->vtable_size; }
    // Size of the table data (excluding soffset)
    uint16 table_size() const noexcept { return vtable_->table_size; }
    // Number of fields described by this vtable
    uint16 num_fields() const noexcept {
        return (vtable_->vtable_size - 4) / 2;
    }
    // Get field offset, returns 0 if field not in vtable or not present
    uint16 field_offset(uint16 field_id) const noexcept {
        const uint16 num = num_fields();
        if (field_id >= num) {
            return 0;  // Field added after this vtable was created
        }
        // field_offsets array starts at offset 4 in VTable
        const uint16* offsets = reinterpret_cast<const uint16*>(
            reinterpret_cast<const uint8_t*>(vtable_) + 4);
        return offsets[field_id];
    }
    // Check if field is present (offset != 0)
    bool has_field(uint16 field_id) const noexcept {
        return field_offset(field_id) != 0;
    }
private:
    const VTable* vtable_;
};
} // namespace zcmb::flat
```
**Memory Layout Example**: TradeOrder table with 4 fields
```
Schema:
  table TradeOrder {
    symbol: string;    // Field 0
    price: float64;    // Field 1  
    quantity: int32;   // Field 2
    timestamp: int64;  // Field 3
  }
Memory layout (total: ~60 bytes):
Offset 0x00: VTable (12 bytes)
  ├─ 0x00: vtable_size = 12 (4 header + 4 fields × 2 bytes)
  ├─ 0x02: table_size = 24 (soffset + 3 inline fields + 1 offset)
  ├─ 0x04: field[0] = 16  (symbol: offset to string offset)
  ├─ 0x06: field[1] = 8   (price: inline double)
  ├─ 0x08: field[2] = 4   (quantity: inline int32)
  └─ 0x0A: field[3] = 0   (timestamp: NOT PRESENT in this instance)
Offset 0x0C: TablePreamble (4 bytes)
  └─ 0x0C: soffset = -12  (points back to vtable at 0x00)
Offset 0x10: Table Data (24 bytes, including soffset)
  ├─ 0x0C: soffset = -12 (repeated for clarity)
  ├─ 0x10: quantity: int32 = 100
  ├─ 0x14: price: float64 = 150.25
  └─ 0x1C: symbol_offset: uint32 = 28 (offset from buffer start to string)
Offset 0x20: String "AAPL" (8 bytes)
  ├─ 0x20: length: uint32 = 4
  └─ 0x24: "AAPL\0" (5 bytes, padded to 8)
```
### BufferView (Zero-Copy Accessor Base)
```cpp
// File: 01_include/zcmb/flat/02_buffer_view.hpp
#pragma once
#include "01_scalar_types.hpp"
#include "06_vtable.hpp"
#include <cstdint>
#include <type_traits>
namespace zcmb::flat {
// Base class for all generated table views
// Provides vtable traversal and typed field access
class BufferView {
public:
    // Construct from raw buffer and root table offset
    // buffer: pointer to start of flat buffer
    // table_offset: byte offset from buffer start to table preamble
    BufferView(const uint8_t* buffer, uint32_t table_offset) noexcept
        : buffer_(buffer)
        , table_start_(buffer + table_offset)
    {
        // Load soffset (signed offset to vtable)
        const int32 soffset = *reinterpret_cast<const int32*>(table_start_);
        // vtable is at: table_start_ + soffset (soffset is negative)
        vtable_ = reinterpret_cast<const VTable*>(
            table_start_ + soffset);
    }
    // Check if field is present in this instance
    bool has_field(uint16 field_id) const noexcept {
        return VTableReader(vtable_).has_field(field_id);
    }
    // Get scalar field value with default
    template<typename T>
    T get_scalar(uint16 field_id, T default_value) const noexcept {
        static_assert(std::is_trivially_copyable_v<T>, 
                      "T must be trivially copyable");
        VTableReader reader(vtable_);
        uint16 offset = reader.field_offset(field_id);
        if (offset == 0) {
            return default_value;  // Field not present
        }
        const uint8_t* field_ptr = table_start_ + offset;
        T value;
        std::memcpy(&value, field_ptr, sizeof(T));
        return value;
    }
    // Get offset field (string, vector, nested table)
    // Returns 0 if not present
    uint32_t get_offset(uint16 field_id) const noexcept {
        VTableReader reader(vtable_);
        uint16 offset = reader.field_offset(field_id);
        if (offset == 0) {
            return 0;
        }
        const uint8_t* field_ptr = table_start_ + offset;
        uint32_t rel_offset;
        std::memcpy(&rel_offset, field_ptr, sizeof(uint32_t));
        // rel_offset is from buffer start, not table start
        return rel_offset;
    }
    // Get pointer to raw field data
    const uint8_t* get_field_ptr(uint16 field_id) const noexcept {
        VTableReader reader(vtable_);
        uint16 offset = reader.field_offset(field_id);
        if (offset == 0) {
            return nullptr;
        }
        return table_start_ + offset;
    }
    // Access to underlying buffer
    const uint8_t* buffer() const noexcept { return buffer_; }
    const uint8_t* table_start() const noexcept { return table_start_; }
    const VTable* vtable() const noexcept { return vtable_; }
protected:
    const uint8_t* buffer_;
    const uint8_t* table_start_;
    const VTable* vtable_;
};
} // namespace zcmb::flat
```
### VectorView (Array Accessor)
```cpp
// File: 01_include/zcmb/flat/03_vector_view.hpp
#pragma once
#include "01_scalar_types.hpp"
#include <cstdint>
#include <cstddef>
#include <iterator>
namespace zcmb::flat {
// Vector layout in memory:
// ┌────────────────────────────────────────────┐
// │ length: uint32                              │  Offset 0
// │ element[0]: T                               │  Offset 4
// │ element[1]: T                               │  Offset 4 + sizeof(T)
// │ ...                                         │
// │ element[n-1]: T                             │  Offset 4 + (n-1)*sizeof(T)
// └────────────────────────────────────────────┘
template<typename T>
class VectorView {
public:
    using value_type = T;
    using size_type = uint32_t;
    using difference_type = std::ptrdiff_t;
    using reference = const T&;
    using pointer = const T*;
    // Construct from pointer to vector data (after length)
    explicit VectorView(const uint8_t* data) noexcept
        : data_(data)
    {
        if (data_) {
            std::memcpy(&length_, data_, sizeof(uint32_t));
        } else {
            length_ = 0;
        }
    }
    // Null constructor for optional vectors
    VectorView() noexcept : data_(nullptr), length_(0) {}
    // Capacity
    size_type size() const noexcept { return length_; }
    bool empty() const noexcept { return length_ == 0; }
    // Element access
    reference operator[](size_type i) const noexcept {
        const uint8_t* elem_ptr = data_ + 4 + i * sizeof(T);
        return *reinterpret_cast<const T*>(elem_ptr);
    }
    reference at(size_type i) const {
        if (i >= length_) {
            throw std::out_of_range("VectorView index out of range");
        }
        return (*this)[i];
    }
    reference front() const noexcept { return (*this)[0]; }
    reference back() const noexcept { return (*this)[length_ - 1]; }
    // Raw data access
    pointer data() const noexcept {
        return reinterpret_cast<const T*>(data_ + 4);
    }
    // Iterator support
    class const_iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;
        const_iterator() noexcept : ptr_(nullptr) {}
        explicit const_iterator(const T* ptr) noexcept : ptr_(ptr) {}
        reference operator*() const noexcept { return *ptr_; }
        pointer operator->() const noexcept { return ptr_; }
        const_iterator& operator++() noexcept { ++ptr_; return *this; }
        const_iterator operator++(int) noexcept { 
            const_iterator tmp = *this; ++ptr_; return tmp; 
        }
        const_iterator& operator--() noexcept { --ptr_; return *this; }
        const_iterator operator--(int) noexcept {
            const_iterator tmp = *this; --ptr_; return tmp;
        }
        bool operator==(const const_iterator& other) const noexcept {
            return ptr_ == other.ptr_;
        }
        bool operator!=(const const_iterator& other) const noexcept {
            return ptr_ != other.ptr_;
        }
    private:
        const T* ptr_;
    };
    const_iterator begin() const noexcept { 
        return const_iterator(data()); 
    }
    const_iterator end() const noexcept { 
        return const_iterator(data() + length_); 
    }
private:
    const uint8_t* data_;
    uint32_t length_;
};
// Specialization for bool (packed as bits)
template<>
class VectorView<bool> {
public:
    explicit VectorView(const uint8_t* data) noexcept : data_(data) {
        if (data_) {
            std::memcpy(&length_, data_, sizeof(uint32_t));
        } else {
            length_ = 0;
        }
    }
    uint32_t size() const noexcept { return length_; }
    bool operator[](uint32_t i) const noexcept {
        const uint8_t* byte_ptr = data_ + 4 + (i / 8);
        uint8_t bit = i % 8;
        return (*byte_ptr >> bit) & 1;
    }
private:
    const uint8_t* data_;
    uint32_t length_;
};
} // namespace zcmb::flat
```
### StringView (Zero-Copy String)
```cpp
// File: 01_include/zcmb/flat/04_string_view.hpp
#pragma once
#include "01_scalar_types.hpp"
#include <cstdint>
#include <cstring>
#include <string>
namespace zcmb::flat {
// String layout in memory:
// ┌────────────────────────────────────────────┐
// │ length: uint32                              │  Offset 0
// │ char[0]: data[0]                            │  Offset 4
// │ char[1]: data[1]                            │  Offset 5
// │ ...                                         │
// │ char[n-1]: data[n-1]                        │
// │ null_terminator: '\0'                       │  Offset 4 + length
// └────────────────────────────────────────────┘
// Total: 4 + length + 1 bytes (padded to alignment)
class StringView {
public:
    // Construct from pointer to string data
    explicit StringView(const uint8_t* data) noexcept : data_(data) {
        if (data_) {
            std::memcpy(&length_, data_, sizeof(uint32_t));
        } else {
            length_ = 0;
        }
    }
    // Null constructor
    StringView() noexcept : data_(nullptr), length_(0) {}
    // Capacity
    uint32_t size() const noexcept { return length_; }
    bool empty() const noexcept { return length_ == 0; }
    // C-string access (guaranteed null-terminated)
    const char* c_str() const noexcept {
        if (!data_) return "";
        return reinterpret_cast<const char*>(data_ + 4);
    }
    // Raw data access
    const char* data() const noexcept { return c_str(); }
    // Convert to std::string (makes a copy)
    std::string str() const {
        if (!data_) return "";
        return std::string(c_str(), length_);
    }
    // Comparison operators
    bool operator==(const char* other) const noexcept {
        if (!data_) return other == nullptr || other[0] == '\0';
        return std::strcmp(c_str(), other) == 0;
    }
    bool operator==(const std::string& other) const noexcept {
        if (!data_) return other.empty();
        return length_ == other.size() && 
               std::memcmp(c_str(), other.data(), length_) == 0;
    }
    bool operator==(const StringView& other) const noexcept {
        if (length_ != other.length_) return false;
        if (length_ == 0) return true;
        return std::memcmp(c_str(), other.c_str(), length_) == 0;
    }
    // Subscript access
    char operator[](uint32_t i) const noexcept {
        return reinterpret_cast<const char*>(data_ + 4)[i];
    }
private:
    const uint8_t* data_;
    uint32_t length_;
};
} // namespace zcmb::flat
```
### Schema AST Types
```cpp
// File: 01_include/zcmb/schema/01_ast_types.hpp
#pragma once
#include "flat/01_scalar_types.hpp"
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <memory>
namespace zcmb::schema {
using namespace zcmb::flat;
// Forward declarations
struct TableDef;
struct StructDef;
struct EnumDef;
// Field definition
struct FieldDef {
    std::string name;
    TypeId type_id;
    uint16_t field_id;           // Assigned position (for evolution)
    std::string type_name;       // For custom types (tables, enums)
    std::optional<std::string> default_value;
    bool deprecated = false;
    // For vector types
    TypeId element_type = TypeId::INT8;
    std::string element_type_name;
    // Layout (computed by LayoutCalculator)
    uint16_t offset = 0;         // Byte offset from table start
    uint16_t size = 0;           // Size in bytes
    uint16_t alignment = 1;      // Required alignment
};
// Table definition (schema-evolvable, variable layout)
struct TableDef {
    std::string name;
    std::string namespace_;
    std::vector<FieldDef> fields;
    // Computed layout
    uint16_t table_size = 0;     // Total table data size
    uint16_t vtable_size = 0;    // VTable size
};
// Struct definition (fixed layout, inline, no evolution)
struct StructDef {
    std::string name;
    std::string namespace_;
    std::vector<FieldDef> fields;
    // Computed layout (fixed)
    uint16_t struct_size = 0;
    uint16_t struct_alignment = 1;
};
// Enum value
struct EnumValue {
    std::string name;
    int64_t value;
};
// Enum definition
struct EnumDef {
    std::string name;
    std::string namespace_;
    TypeId underlying_type;      // INT8, INT16, INT32, INT64
    std::vector<EnumValue> values;
};
// Namespace
struct Namespace {
    std::vector<std::string> parts;  // ["trading", "market"]
    std::string to_string() const;
};
// Root schema
struct Schema {
    std::string file_path;
    Namespace ns;
    std::vector<TableDef> tables;
    std::vector<StructDef> structs;
    std::vector<EnumDef> enums;
    std::string root_type;       // Name of root table
    // Lookup helpers
    const TableDef* find_table(const std::string& name) const;
    const StructDef* find_struct(const std::string& name) const;
    const EnumDef* find_enum(const std::string& name) const;
};
} // namespace zcmb::schema
```
### Layout Calculator
```cpp
// File: 01_include/zcmb/schema/03_layout_calculator.hpp
#pragma once
#include "01_ast_types.hpp"
#include <vector>
#include <cstdint>
namespace zcmb::schema {
// Computes field offsets with proper alignment
// Orders fields by alignment (descending) to minimize padding
class LayoutCalculator {
public:
    struct FieldLayout {
        uint16_t field_id;
        uint16_t offset;
        uint16_t size;
        uint16_t alignment;
    };
    struct TableLayout {
        uint16_t table_size;      // Total bytes (including soffset)
        uint16_t vtable_size;     // VTable bytes
        std::vector<FieldLayout> fields;
    };
    // Calculate layout for a table
    static TableLayout calculate_table(TableDef& table);
    // Calculate layout for a struct
    static uint16_t calculate_struct(StructDef& s);
    // Align offset up to required alignment
    static uint16_t align_up(uint16_t offset, uint16_t alignment) noexcept {
        return (offset + alignment - 1) & ~(alignment - 1);
    }
    // Get size and alignment for a type
    static void get_type_layout(
        TypeId type_id,
        uint16_t& out_size,
        uint16_t& out_alignment
    ) noexcept;
private:
    // Sort fields by alignment (descending) for optimal packing
    static void sort_fields_by_alignment(std::vector<FieldDef>& fields);
};
} // namespace zcmb::schema
```
**Layout Algorithm**:
```
INPUT: TableDef with fields (unsorted, no offsets)
OUTPUT: TableDef with fields (sorted, offsets assigned), table_size, vtable_size
ALGORITHM calculate_table(table):
  1. Create copy of fields array
  2. Sort fields by alignment descending:
     - float64, int64, uint64 → alignment 8
     - int32, uint32, float32 → alignment 4
     - int16, uint16 → alignment 2
     - int8, uint8, bool → alignment 1
     - string, vector, table (offsets) → alignment 4
  3. Initialize offset = 4 (after soffset to vtable)
  4. For each field in sorted order:
     a. Get field size and alignment
     b. offset = align_up(offset, alignment)
     c. field.offset = offset
     d. offset += field.size
  5. table_size = align_up(offset, 8)  // Align to 8 for future fields
  6. vtable_size = 4 + (num_fields × 2)  // Header + field offsets
  7. RETURN table_size, vtable_size
EXAMPLE:
  Fields: symbol(string), price(double), quantity(int32), timestamp(int64)
  Step 2: Sort by alignment
    - price: alignment 8
    - timestamp: alignment 8
    - quantity: alignment 4
    - symbol: alignment 4 (offset type)
  Step 3-4: Assign offsets
    offset = 4 (after soffset)
    price: align_up(4, 8) = 8, offset = 8, size = 8 → next = 16
    timestamp: align_up(16, 8) = 16, offset = 16, size = 8 → next = 24
    quantity: align_up(24, 4) = 24, offset = 24, size = 4 → next = 28
    symbol: align_up(28, 4) = 28, offset = 28, size = 4 → next = 32
  Step 5: table_size = align_up(32, 8) = 32
  Step 6: vtable_size = 4 + (4 × 2) = 12
```
### BuilderBase (Message Construction)
```cpp
// File: 01_include/zcmb/flat/05_builder_base.hpp
#pragma once
#include "01_scalar_types.hpp"
#include "06_vtable.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
namespace zcmb::flat {
// Builds flat buffers by growing downward from end of buffer
// This allows strings/vectors to be written before the table
class BuilderBase {
public:
    explicit BuilderBase(size_t initial_capacity = 1024);
    ~BuilderBase() = default;
    // Reset builder for new message
    void reset() noexcept;
    // === Scalar writing ===
    void write_int8(int8 value);
    void write_int16(int16 value);
    void write_int32(int32 value);
    void write_int64(int64 value);
    void write_uint8(uint8 value);
    void write_uint16(uint16 value);
    void write_uint32(uint32 value);
    void write_uint64(uint64 value);
    void write_float32(float32 value);
    void write_float64(float64 value);
    void write_bool(bool value);
    // === String/Vector writing (returns offset from buffer start) ===
    uint32_t write_string(const char* str, uint32_t length);
    uint32_t write_string(const std::string& str);
    template<typename T>
    uint32_t write_vector(const T* data, uint32_t count) {
        // Align to max(T alignment, 4)
        uint32_t alignment = alignof(T) > 4 ? alignof(T) : 4;
        align_buffer(alignment);
        uint32_t start = static_cast<uint32_t>(buf_.size());
        uint32_t total_size = 4 + count * sizeof(T);  // length + data
        // Grow buffer
        buf_.resize(start + total_size);
        // Write length
        std::memcpy(buf_.data() + start, &count, sizeof(uint32_t));
        // Write elements
        std::memcpy(buf_.data() + start + 4, data, count * sizeof(T));
        return start;
    }
    // === Table construction ===
    // Start a new table, returns table start offset
    uint32_t start_table(uint16_t num_fields);
    // Add scalar field (must be called in field_id order)
    template<typename T>
    void add_field(uint16_t field_id, T value) {
        ensure_current_table();
        // Track field presence
        field_presence_.resize(std::max(field_presence_.size(), 
                                        static_cast<size_t>(field_id) + 1));
        field_presence_[field_id] = true;
        // Write value to current offset
        size_t current = buf_.size();
        align_buffer(alignof(T));
        buf_.resize(current + sizeof(T));
        std::memcpy(buf_.data() + current, &value, sizeof(T));
    }
    // Add offset field (string, vector, nested table)
    void add_offset_field(uint16_t field_id, uint32_t offset);
    // Finish table, write vtable, return table offset
    // vtable_fields: array of field_id → offset in this table
    uint32_t end_table(const uint16_t* field_offsets, uint16_t num_fields);
    // === Buffer access ===
    const uint8_t* data() const noexcept { return buf_.data(); }
    size_t size() const noexcept { return buf_.size(); }
    // Get finished buffer (moves ownership)
    std::vector<uint8_t> finish();
protected:
    void align_buffer(uint32_t alignment) {
        size_t current = buf_.size();
        size_t aligned = (current + alignment - 1) & ~(alignment - 1);
        if (aligned > current) {
            buf_.resize(aligned);
        }
    }
    void ensure_current_table();
    std::vector<uint8_t> buf_;
    std::vector<bool> field_presence_;
    uint32_t table_start_ = 0;
    bool in_table_ = false;
};
} // namespace zcmb::flat
```
---
## Interface Contracts
### SchemaParser::parse
```cpp
// File: 01_include/zcmb/schema/02_schema_parser.hpp
namespace zcmb::schema {
class SchemaParser {
public:
    // Parse schema from string
    // Throws SchemaParseError on syntax/semantic errors
    Schema parse(const std::string& content);
    // Parse schema from file
    Schema parse_file(const std::string& path);
    // Get last error message (if parse returned empty schema)
    const std::string& last_error() const noexcept { return last_error_; }
private:
    std::string last_error_;
    // Lexer state
    std::string_view input_;
    size_t pos_ = 0;
    // Recursive descent parser methods
    bool parse_namespace(Schema& schema);
    bool parse_table(Schema& schema);
    bool parse_struct(Schema& schema);
    bool parse_enum(Schema& schema);
    bool parse_field(TableDef& table);
    bool parse_type(FieldDef& field);
    std::string parse_identifier();
    std::string parse_string_literal();
    int64_t parse_integer();
    void skip_whitespace();
    void skip_comment();
    Token next_token();
};
} // namespace zcmb::schema
```
**Preconditions**:
- `content` is valid UTF-8 text
- Content follows the schema grammar
**Postconditions**:
- On success: returns fully populated `Schema` with all tables, structs, enums
- On failure: throws `SchemaParseError` with line/column info
**Error Conditions**:
| Error | Line/Column | Example | Recovery |
|-------|-------------|---------|----------|
| UNEXPECTED_TOKEN | At error | `table { }` (missing name) | Report and stop |
| UNTERMINATED_STRING | At quote | `"hello` | Report and stop |
| DUPLICATE_TABLE | At second | Two tables named `Order` | Report and continue |
| UNKNOWN_TYPE | At type | `field: FooBar` | Report and continue |
| INVALID_DEFAULT | At value | `int = "string"` | Report and continue |
### LayoutCalculator::calculate_table
```cpp
static TableLayout calculate_table(TableDef& table);
```
**Preconditions**:
- `table.fields` is populated with type information
- `table.name` is not empty
**Postconditions**:
- All fields have `offset`, `size`, `alignment` assigned
- Fields are sorted by alignment (descending) in `table.fields`
- `table.table_size` is multiple of 8
- `table.vtable_size` = 4 + 2 × num_fields
**Invariant**: No field offset overlaps another field. All offsets are properly aligned.
### CodeGenerator::generate
```cpp
// File: 01_include/zcmb/gen/01_code_generator.hpp
namespace zcmb::gen {
class CodeGenerator {
public:
    struct Options {
        std::string output_dir;
        std::string namespace_prefix;
        bool generate_builders = true;
        bool generate_views = true;
        bool generate_printers = true;  // Debug output
    };
    // Generate C++ code from schema
    // Returns true on success, false on error
    bool generate(const Schema& schema, const Options& options);
    // Get generated file paths (after generate() succeeds)
    const std::vector<std::string>& generated_files() const;
    // Get last error message
    const std::string& last_error() const noexcept;
private:
    bool generate_table_view(const TableDef& table, std::ostream& out);
    bool generate_table_builder(const TableDef& table, std::ostream& out);
    bool generate_enum(const EnumDef& enum_def, std::ostream& out);
    bool generate_struct(const StructDef& struct_def, std::ostream& out);
    std::string type_to_cpp_type(TypeId type_id, const std::string& type_name);
    std::string type_to_accessor(TypeId type_id);
    std::vector<std::string> generated_files_;
    std::string last_error_;
};
} // namespace zcmb::gen
```
**Preconditions**:
- `schema` has been validated (all tables have fields, types resolved)
- `options.output_dir` exists and is writable
**Postconditions**:
- One `.hpp` file generated per namespace
- Each file is compilable with C++20
- Generated classes match schema layout exactly
---
## Algorithm Specification
### Algorithm: Schema Lexer
```
INPUT: std::string_view input (schema source)
OUTPUT: Stream of tokens (KEYWORD, IDENTIFIER, STRING, NUMBER, SYMBOL)
LEXER next_token():
  1. skip_whitespace()  // Skip ' ', '\t', '\n', '\r'
  2. IF pos >= input.length(): RETURN EOF
  3. skip_comment()  // Skip // to end of line, /* */ blocks
  4. ch = input[pos]
  5. SWITCH on ch:
     CASE '{', '}', '(', ')', '[', ']', ':', ';', ',', '=', '.':
       pos++
       RETURN SYMBOL(ch)
     CASE '"':
       RETURN parse_string_literal()
     CASE digit or (ch == '-' AND next is digit):
       RETURN parse_number()
     CASE letter or '_':
       ident = parse_identifier()
       IF ident in keywords: RETURN KEYWORD(ident)
       ELSE: RETURN IDENTIFIER(ident)
     DEFAULT:
       error("Unexpected character: " + ch)
KEYWORDS = {
  "namespace", "table", "struct", "enum", "root_type",
  "true", "false", "null",
  "int8", "int16", "int32", "int64",
  "uint8", "uint16", "uint32", "uint64",
  "float32", "float64", "bool", "string"
}
```
### Algorithm: Schema Parser (Table)
```
INPUT: Token stream after "table" keyword
OUTPUT: Populated TableDef
PARSER parse_table():
  1. name = parse_identifier()
     IF name.empty(): ERROR "Expected table name"
  2. EXPECT SYMBOL '{'
  3. table = TableDef{name = name}
  4. WHILE NOT at '}':
     a. IF current_token == KEYWORD("deprecated"):
        table.fields.back().deprecated = true
        consume_token()
        EXPECT SYMBOL ';'
        CONTINUE
     b. field = parse_field()
     c. field.field_id = table.fields.size()
     d. table.fields.push_back(field)
     e. EXPECT SYMBOL ';'
  5. EXPECT SYMBOL '}'
  6. RETURN table
PARSER parse_field():
  1. field = FieldDef{}
  2. field.name = parse_identifier()
     IF field.name.empty(): ERROR "Expected field name"
  3. EXPECT SYMBOL ':'
  4. parse_type(field)  // Sets type_id, type_name, element_type
  5. IF current_token == SYMBOL('='):
     consume_token()
     field.default_value = parse_default_value(field.type_id)
  6. RETURN field
PARSER parse_type(field):
  1. type_name = parse_identifier()
  2. IF type_name == "[":
     // Vector type
     parse_type(field)  // Recursively parse element type
     field.type_id = TypeId::VECTOR
     EXPECT SYMBOL ']'
     RETURN
  3. IF type_name in scalar_types:
     field.type_id = scalar_to_type_id(type_name)
     field.type_name = ""
     RETURN
  4. // Custom type (table, struct, or enum)
     field.type_id = TypeId::TABLE  // May be refined later
     field.type_name = type_name
```
### Algorithm: VTable Lookup at Runtime
```
INPUT: BufferView, field_id
OUTPUT: Field value or default
ALGORITHM get_field(field_id, default_value):
  1. vtable_ptr = table_start_ + soffset_to_vtable
  2. vtable_size = *reinterpret_cast<uint16_t*>(vtable_ptr)
  3. num_fields = (vtable_size - 4) / 2
  4. IF field_id >= num_fields:
     // Field added after this buffer was created
     RETURN default_value
  5. field_offset_array = vtable_ptr + 4
     offset = *reinterpret_cast<uint16_t*>(
                field_offset_array + field_id * 2)
  6. IF offset == 0:
     // Field not present in this instance
     RETURN default_value
  7. field_ptr = table_start_ + offset
     RETURN *reinterpret_cast<T*>(field_ptr)
```
### Algorithm: Buffer Building (Backward Growth)
```
FlatBuffer layout grows from END of buffer toward beginning:
                                    ┌─────────────────┐
Buffer start                        │                 │
    │                               │ Free space      │
    ▼                               │                 │
┌───┴───────────────────────────────┴─────────────────┐
│ Root offset (4B) │ ... free ... │ Strings │ Tables │
└─────────────────────────────────────────────────────┘
                    ▲              ▲         ▲
                    │              │         │
              buf_.size()    write_pos   table_start
ALGORITHM write_string(str):
  1. length = str.size()
  2. total_bytes = 4 + length + 1  // length + data + null
  3. Align to 4 bytes
  4. current_pos = buf_.size()
  5. buf_.resize(current_pos + total_bytes)
  6. Write length at current_pos
  7. Write string data at current_pos + 4
  8. Write null terminator
  9. RETURN current_pos  // Offset from buffer start
ALGORITHM end_table(field_offsets, num_fields):
  1. table_data_size = current_buf_size - table_start_
  2. // Build vtable
     vtable_size = 4 + num_fields * 2
     vtable = allocate at buf start or separate area
  3. FOR i in 0..num_fields:
     IF field_presence_[i]:
       vtable.field_offsets[i] = field_offsets[i]
     ELSE:
       vtable.field_offsets[i] = 0  // Not present
  4. // Write vtable before table
     vtable_pos = current_vtable_pos
     write_vtable(vtable)
  5. // Write soffset in table
     soffset = vtable_pos - table_start_  // Negative value
     *reinterpret_cast<int32*>(table_start_) = soffset
  6. RETURN table_start_  // Offset to root table
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible |
|-------|-------------|----------|--------------|
| SCHEMA_PARSE_ERROR | Lexer/Parser | Throw `SchemaParseError` with line/col | Yes - build fails |
| UNKNOWN_TYPE | Parser type resolution | Report error, continue parsing | Yes - build fails |
| DUPLICATE_FIELD | Parser field collection | Report error, skip duplicate | Yes - build fails |
| FIELD_NOT_FOUND | `BufferView::has_field()` | Return `false`, use default | No - normal operation |
| TYPE_MISMATCH | Code generator validation | Static assertion in generated code | Yes - compile fails |
| BUFFER_TOO_SMALL | Builder `resize()` | Throw `std::bad_alloc` | Yes - runtime error |
| ALIGNMENT_ERROR | `reinterpret_cast` on ARM | SIGBUS or wrong value | Yes - crash/data corrupt |
| CHECKSUM_MISMATCH | Optional validation layer | Throw `ChecksumError` | Yes - data corrupt |
| SCHEMA_INCOMPATIBLE | Version check | Throw `IncompatibleSchemaError` | Yes - migration needed |
| NULL_STRING_POINTER | `StringView(nullptr)` | Return empty string | No - graceful fallback |
| VECTOR_OUT_OF_RANGE | `VectorView::at()` | Throw `std::out_of_range` | Yes - logic error |
---
## Implementation Sequence with Checkpoints
### Phase 1: Schema Definition Language (3-4 hours)
**Files**: `01_scalar_types.hpp`, `01_ast_types.hpp`, `02_schema_parser.hpp`, `01_schema_parser.cpp`
**Steps**:
1. Define `TypeId` enum and scalar type traits
2. Define AST structures: `FieldDef`, `TableDef`, `StructDef`, `EnumDef`, `Schema`
3. Implement lexer: token types, whitespace/comment skipping, identifier/number/string parsing
4. Implement parser: namespace, table, struct, enum rules
5. Add error reporting with line/column tracking
**Checkpoint**:
```bash
./05_tests/04_schema_parser_test
# Expected:
# [PASS] Parse namespace
# [PASS] Parse scalar fields
# [PASS] Parse string/vector fields
# [PASS] Parse nested table reference
# [PASS] Parse enum definition
# [PASS] Error on missing semicolon
# [PASS] Error on unknown type
```
**At this point**: Can parse `.fbs` files into AST, detect syntax errors with line numbers.
### Phase 2: Layout Calculator (2-3 hours)
**Files**: `03_layout_calculator.hpp`, `02_layout_calculator.cpp`
**Steps**:
1. Implement `get_type_layout()` for all type IDs
2. Implement field sorting by alignment (descending)
3. Implement `calculate_table()` with offset assignment
4. Implement `calculate_struct()` for fixed-layout types
5. Verify with alignment edge cases (int8 followed by float64)
**Checkpoint**:
```bash
./05_tests/05_layout_calculator_test
# Expected:
# [PASS] Scalar field sizes
# [PASS] Alignment for double (8)
# [PASS] No overlap between fields
# [PASS] Padding minimization
# [PASS] Table size is 8-byte aligned
# [PASS] VTable size calculation
```
**At this point**: Given a schema, can compute exact byte offsets for all fields.
### Phase 3: Code Generator (4-5 hours)
**Files**: `01_code_generator.hpp`, `01_code_generator.cpp`
**Steps**:
1. Implement header generation with namespace guards
2. Implement `generate_table_view()` - produces View class with typed accessors
3. Implement `generate_table_builder()` - produces Builder class with add methods
4. Implement `generate_enum()` - produces enum class with toString/fromString
5. Implement `generate_struct()` - produces simple struct with fields
6. Add file output and include guards
**Checkpoint**:
```bash
./zcmb-m2-zero-copy-serialization/03_schemas/02_trading.fbs
# Run generator:
./zcmb-gen 03_schemas/02_trading.fbs --output 04_generated/
./05_tests/06_code_generator_test
# Expected:
# [PASS] Generated file compiles
# [PASS] View class has all accessors
# [PASS] Builder class has all add methods
# [PASS] Field IDs match schema order
```
**At this point**: Can generate working C++ code from schema files.
### Phase 4: Runtime Access Layer (3-4 hours)
**Files**: `02_buffer_view.hpp`, `03_vector_view.hpp`, `04_string_view.hpp`, `01_buffer_view.cpp`
**Steps**:
1. Implement `BufferView` base with vtable traversal
2. Implement `VTableReader` for field offset lookup
3. Implement `VectorView<T>` with iterator support
4. Implement `StringView` with comparison operators
5. Add `VectorView<bool>` specialization for packed bits
6. Test alignment handling on ARM (if available)
**Checkpoint**:
```bash
./05_tests/02_buffer_view_test
./05_tests/03_vector_string_test
# Expected:
# [PASS] VTable lookup for present field
# [PASS] Default value for absent field
# [PASS] Vector iteration
# [PASS] String comparison
# [PASS] Empty vector/string handling
```
**At this point**: Can read flat buffer bytes and access fields without parsing.
### Phase 5: Schema Evolution Support (2-3 hours)
**Files**: `04_schema_registry.hpp`, `03_schema_registry.cpp`, modifications to parser and generator
**Steps**:
1. Add `field_id` assignment in parser (sequential, never reused)
2. Implement `SchemaRegistry` for runtime schema lookup
3. Add version field to table preamble (optional)
4. Test old reader with new schema (forward compatibility)
5. Test new reader with old schema (backward compatibility)
6. Add deprecation marking in generated code
**Checkpoint**:
```bash
./05_tests/08_schema_evolution_test
# Expected:
# [PASS] Add field to schema
# [PASS] Old reader sees default for new field
# [PASS] New reader handles old buffer
# [PASS] Deprecated field still accessible
# [PASS] Field ID never reused
```
**At this point**: Schema can evolve without breaking existing readers/writers.
---
## Test Specification
### Test: Schema Parsing All Constructs
```cpp
TEST(SchemaParser, ParsesAllConstructs) {
    const char* schema_text = R"(
        namespace trading.market;
        enum OrderSide : int8 {
            BUY = 0,
            SELL = 1
        }
        table Symbol {
            code: string;
            exchange: string;
        }
        table TradeOrder {
            id: uint64;
            symbol: Symbol;
            side: OrderSide;
            price: float64;
            quantity: int32;
            timestamp: int64;
            flags: uint32 = 0;
        }
        root_type TradeOrder;
    )";
    SchemaParser parser;
    Schema schema = parser.parse(schema_text);
    EXPECT_EQ(schema.ns.parts.size(), 2);
    EXPECT_EQ(schema.ns.parts[0], "trading");
    EXPECT_EQ(schema.ns.parts[1], "market");
    EXPECT_EQ(schema.enums.size(), 1);
    EXPECT_EQ(schema.enums[0].name, "OrderSide");
    EXPECT_EQ(schema.enums[0].values.size(), 2);
    EXPECT_EQ(schema.tables.size(), 2);
    const TableDef* order = schema.find_table("TradeOrder");
    ASSERT_NE(order, nullptr);
    EXPECT_EQ(order->fields.size(), 7);
    EXPECT_EQ(order->fields[0].name, "id");
    EXPECT_EQ(order->fields[0].type_id, TypeId::UINT64);
    EXPECT_EQ(order->fields[6].name, "flags");
    EXPECT_TRUE(order->fields[6].default_value.has_value());
}
```
### Test: Layout Alignment
```cpp
TEST(LayoutCalculator, AlignsFieldsCorrectly) {
    TableDef table;
    table.name = "TestTable";
    table.fields = {
        {"a", TypeId::UINT8, 0, "", std::nullopt},   // 1 byte
        {"b", TypeId::FLOAT64, 1, "", std::nullopt}, // 8 bytes, 8-aligned
        {"c", TypeId::INT32, 2, "", std::nullopt},   // 4 bytes, 4-aligned
        {"d", TypeId::UINT16, 3, "", std::nullopt},  // 2 bytes, 2-aligned
    };
    auto layout = LayoutCalculator::calculate_table(table);
    // After sorting by alignment: b, c, d, a
    // soffset = 4 bytes
    // b at offset 8 (aligned to 8), size 8
    // c at offset 16, size 4
    // d at offset 20, size 2
    // a at offset 22, size 1
    // Total: 23, padded to 24
    EXPECT_EQ(table.table_size, 24);
    EXPECT_EQ(table.vtable_size, 12);  // 4 + 4*2
    // Find each field's offset
    auto find_offset = [&](const std::string& name) -> uint16_t {
        for (const auto& f : table.fields) {
            if (f.name == name) return f.offset;
        }
        return UINT16_MAX;
    };
    // Check no overlaps
    std::set<uint16_t> used_offsets;
    for (const auto& f : table.fields) {
        for (uint16_t i = 0; i < f.size; ++i) {
            EXPECT_TRUE(used_offsets.insert(f.offset + i).second)
                << "Overlap at offset " << (f.offset + i);
        }
    }
}
```
### Test: Round-Trip Serialization
```cpp
TEST(RoundTrip, SerializesAndDeserializes) {
    // Use generated TradeOrder classes
    trading::TradeOrderBuilder builder;
    builder.set_id(12345);
    builder.set_price(150.25);
    builder.set_quantity(100);
    builder.set_timestamp(1697234567890123456LL);
    builder.set_side(trading::OrderSide::BUY);
    builder.set_symbol("AAPL");
    std::vector<uint8_t> buffer = builder.finish();
    // Deserialize
    trading::TradeOrderView view(buffer.data(), 0);
    EXPECT_EQ(view.id(), 12345);
    EXPECT_DOUBLE_EQ(view.price(), 150.25);
    EXPECT_EQ(view.quantity(), 100);
    EXPECT_EQ(view.timestamp(), 1697234567890123456LL);
    EXPECT_EQ(view.side(), trading::OrderSide::BUY);
    EXPECT_EQ(view.symbol(), "AAPL");
    EXPECT_EQ(view.flags(), 0);  // Default value
}
```
### Test: Schema Evolution Backward Compatibility
```cpp
TEST(SchemaEvolution, BackwardCompatibility) {
    // Create buffer with V1 schema (no timestamp field)
    {
        trading::v1::TradeOrderBuilder builder;
        builder.set_id(100);
        builder.set_price(100.0);
        builder.set_quantity(50);
        // No timestamp
        auto buffer = builder.finish();
        // Save buffer
        save_test_buffer("v1_order.bin", buffer);
    }
    // Read with V2 schema (has timestamp field)
    {
        auto buffer = load_test_buffer("v1_order.bin");
        trading::v2::TradeOrderView view(buffer.data(), 0);
        EXPECT_EQ(view.id(), 100);
        EXPECT_DOUBLE_EQ(view.price(), 100.0);
        EXPECT_EQ(view.quantity(), 50);
        EXPECT_EQ(view.timestamp(), 0);  // Default for missing field
        EXPECT_FALSE(view.has_timestamp());  // Explicitly absent
    }
}
```
### Test: Access Latency Benchmark
```cpp
TEST(Benchmark, AccessLatency) {
    // Create buffer with 10 fields
    trading::TradeOrderBuilder builder;
    builder.set_id(12345);
    builder.set_price(150.25);
    builder.set_quantity(100);
    builder.set_timestamp(1697234567890123456LL);
    // ... set all fields
    auto buffer = builder.finish();
    trading::TradeOrderView view(buffer.data(), 0);
    const size_t iterations = 1000000;
    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum = 0;  // Prevent optimization
    for (size_t i = 0; i < iterations; ++i) {
        sum += view.price();  // Access one field
        sum += view.quantity();  // Access another
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ns_per_access = std::chrono::duration<double, std::nano>(end - start).count() 
                           / (iterations * 2);
    std::cout << "Access latency: " << ns_per_access << " ns\n";
    // Target: < 20 ns per field access
    EXPECT_LT(ns_per_access, 20.0);
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Scalar field access | 5-20 ns | `perf stat -e cycles ./benchmark`, divide by iterations |
| String field access | 10-30 ns | Includes offset load + pointer return |
| Nested table access | 10-30 ns | One indirection via offset |
| Vector element access | 5-10 ns | Pointer + index × sizeof |
| Vector iteration (100 elements) | < 1 μs | Loop over VectorView |
| Serialization (10 fields) | 50-200 ns | Builder construction time |
| Schema parsing (100 lines) | < 1 ms | Parse + AST construction |
| Code generation (10 tables) | < 10 ms | Full code generation |
**Comparison vs JSON (RapidJSON)**:
| Operation | Flat Buffer | JSON (RapidJSON) | Speedup |
|-----------|-------------|------------------|---------|
| Parse/access 1 field | 15 ns | 800 ns | 53x |
| Parse/access all fields | 80 ns | 2500 ns | 31x |
| Serialize 10 fields | 150 ns | 1500 ns | 10x |
| Memory usage (1 message) | 64 bytes | 256 bytes (DOM) | 4x |
| Heap allocations (read) | 0 | 15+ | ∞ |
**Hardware Soul Analysis**:
```
Field access instruction sequence (x86-64, optimized):
// double price = view.price();
// Assuming view pointer in rdi, field_id=1, offset=8
movzx   eax, word ptr [rdi + 4]      ; Load vtable size
cmp     eax, 6                        ; Check if field 1 in vtable (4 + 2*1 = 6)
jb      .return_default               ; Field not in vtable
movzx   eax, word ptr [rdi + 6]      ; Load field offset from vtable[1]
test    eax, eax                      ; Check if offset is 0
je      .return_default               ; Field not present
movsxd  rax, dword ptr [rdi - 4]     ; Load soffset to vtable
add     rax, rdi                      ; rax = vtable address
movsd   xmm0, qword ptr [rdi + eax - 4]  ; Load double at computed offset
ret
.return_default:
xorps   xmm0, xmm0                    ; Return 0.0
ret
Total: ~8-12 instructions, ~3-5 memory accesses
Cache behavior: VTable likely in L1 (shared across instances)
                Field data likely in L1 (same cache line as other fields)
```
---
## Concurrency Specification
### Thread Safety Model
| Component | Thread Safety | Rationale |
|-----------|--------------|-----------|
| `BufferView` | Thread-safe (read-only) | No mutable state, const methods only |
| `VectorView` | Thread-safe (read-only) | Same as BufferView |
| `StringView` | Thread-safe (read-only) | Same as BufferView |
| `BuilderBase` | Single-threaded | Mutates internal buffer |
| `SchemaParser` | Single-threaded | Mutates internal state |
| `SchemaRegistry` | Thread-safe (read-mostly) | Read-write lock, write only at startup |
| Generated View classes | Thread-safe | Inherit from BufferView |
| Generated Builder classes | Single-threaded | Inherit from BuilderBase |
### Cross-Process Safety
- **Immutable buffers**: Once a flat buffer is constructed, it can be read by any process with access to the memory
- **No locks during read**: All access is lock-free via pointer arithmetic
- **Alignment guarantee**: All fields naturally aligned, safe on ARM without special handling
- **Endianness**: Native endianness assumed (little-endian on x86/ARM64). Cross-endian requires byte-swapping.
---
## Generated Code Example
For schema:
```
namespace trading;
table TradeOrder {
    id: uint64;
    symbol: string;
    price: float64;
    quantity: int32;
    timestamp: int64;
}
```
Generated `trading/trade_order.hpp`:
```cpp
// Auto-generated by zcmb-gen - DO NOT EDIT
#pragma once
#include "zcmb/flat/02_buffer_view.hpp"
#include "zcmb/flat/03_vector_view.hpp"
#include "zcmb/flat/04_string_view.hpp"
#include "zcmb/flat/05_builder_base.hpp"
#include <cstdint>
namespace trading {
class TradeOrderView : public zcmb::flat::BufferView {
public:
    using BufferView::BufferView;
    // Field IDs (for schema evolution)
    static constexpr uint16_t FIELD_ID = 0;
    static constexpr uint16_t FIELD_SYMBOL = 1;
    static constexpr uint16_t FIELD_PRICE = 2;
    static constexpr uint16_t FIELD_QUANTITY = 3;
    static constexpr uint16_t FIELD_TIMESTAMP = 4;
    // Field accessors (zero-copy, inline)
    [[nodiscard]] bool has_id() const noexcept {
        return has_field(FIELD_ID);
    }
    [[nodiscard]] uint64_t id() const noexcept {
        return get_scalar<uint64_t>(FIELD_ID, 0);
    }
    [[nodiscard]] bool has_symbol() const noexcept {
        return has_field(FIELD_SYMBOL);
    }
    [[nodiscard]] zcmb::flat::StringView symbol() const noexcept {
        uint32_t offset = get_offset(FIELD_SYMBOL);
        if (offset == 0) return zcmb::flat::StringView();
        return zcmb::flat::StringView(buffer() + offset);
    }
    [[nodiscard]] bool has_price() const noexcept {
        return has_field(FIELD_PRICE);
    }
    [[nodiscard]] double price() const noexcept {
        return get_scalar<double>(FIELD_PRICE, 0.0);
    }
    [[nodiscard]] bool has_quantity() const noexcept {
        return has_field(FIELD_QUANTITY);
    }
    [[nodiscard]] int32_t quantity() const noexcept {
        return get_scalar<int32_t>(FIELD_QUANTITY, 0);
    }
    [[nodiscard]] bool has_timestamp() const noexcept {
        return has_field(FIELD_TIMESTAMP);
    }
    [[nodiscard]] int64_t timestamp() const noexcept {
        return get_scalar<int64_t>(FIELD_TIMESTAMP, 0);
    }
};
class TradeOrderBuilder {
public:
    explicit TradeOrderBuilder(size_t initial_capacity = 256)
        : builder_(initial_capacity) {}
    void set_id(uint64_t value) {
        id_value_ = value;
        has_id_ = true;
    }
    void set_symbol(const std::string& value) {
        symbol_offset_ = builder_.write_string(value);
        has_symbol_ = true;
    }
    void set_price(double value) {
        price_value_ = value;
        has_price_ = true;
    }
    void set_quantity(int32_t value) {
        quantity_value_ = value;
        has_quantity_ = true;
    }
    void set_timestamp(int64_t value) {
        timestamp_value_ = value;
        has_timestamp_ = true;
    }
    std::vector<uint8_t> finish() {
        builder_.start_table(5);
        // Add fields in alignment order (double, int64, int32, uint32)
        uint16_t field_offsets[5] = {0, 0, 0, 0, 0};
        if (has_id_) {
            field_offsets[FIELD_ID] = builder_.current_offset();
            builder_.add_field(FIELD_ID, id_value_);
        }
        if (has_price_) {
            field_offsets[FIELD_PRICE] = builder_.current_offset();
            builder_.add_field(FIELD_PRICE, price_value_);
        }
        if (has_quantity_) {
            field_offsets[FIELD_QUANTITY] = builder_.current_offset();
            builder_.add_field(FIELD_QUANTITY, quantity_value_);
        }
        if (has_timestamp_) {
            field_offsets[FIELD_TIMESTAMP] = builder_.current_offset();
            builder_.add_field(FIELD_TIMESTAMP, timestamp_value_);
        }
        if (has_symbol_) {
            field_offsets[FIELD_SYMBOL] = builder_.current_offset();
            builder_.add_offset_field(FIELD_SYMBOL, symbol_offset_);
        }
        uint32_t table_offset = builder_.end_table(field_offsets, 5);
        // Write root offset at buffer start
        return builder_.finish_with_root(table_offset);
    }
private:
    zcmb::flat::BuilderBase builder_;
    uint64_t id_value_ = 0;
    double price_value_ = 0.0;
    int32_t quantity_value_ = 0;
    int64_t timestamp_value_ = 0;
    uint32_t symbol_offset_ = 0;
    bool has_id_ = false;
    bool has_symbol_ = false;
    bool has_price_ = false;
    bool has_quantity_ = false;
    bool has_timestamp_ = false;
};
} // namespace trading
```
---
## Schema Evolution Rules
### Adding a Field (Allowed)
```
// V1
table Order {
    id: uint64;
    price: float64;
}
// V2 - Add field
table Order {
    id: uint64;         // Field 0
    price: float64;     // Field 1
    quantity: int32;    // Field 2 - NEW
}
```
**Effect**: 
- V2 reader reading V1 buffer: `quantity()` returns 0 (default)
- V1 reader reading V2 buffer: ignores field 2
### Removing a Field (Deprecated, Not Deleted)
```
// V2
table Order {
    id: uint64;
    price: float64;
    quantity: int32;    // Field 2
}
// V3 - Deprecate field
table Order {
    id: uint64;
    price: float64;
    quantity: int32;    // deprecated
    timestamp: int64;   // Field 3 - NEW (NOT field 2!)
}
```
**Rule**: Never reuse field IDs. A deprecated field leaves a "hole" in the ID sequence.
### Changing Field Type (Not Allowed)
```
// ILLEGAL
table Order {
    id: uint64;
    price: float32;  // Was float64 - INCOMPATIBLE!
}
```
**Workaround**: Add new field with different name, deprecate old field.
### Field ID Assignment Algorithm
```cpp
void assign_field_ids(Schema& new_schema, const Schema& existing_schema) {
    for (auto& table : new_schema.tables) {
        const TableDef* existing = existing_schema.find_table(table.name);
        if (!existing) {
            // New table - assign sequential IDs
            for (size_t i = 0; i < table.fields.size(); ++i) {
                table.fields[i].field_id = static_cast<uint16_t>(i);
            }
        } else {
            // Existing table - preserve IDs, assign new for new fields
            std::unordered_map<std::string, uint16_t> existing_ids;
            for (const auto& f : existing->fields) {
                existing_ids[f.name] = f.field_id;
            }
            uint16_t next_new_id = static_cast<uint16_t>(existing->fields.size());
            for (auto& field : table.fields) {
                auto it = existing_ids.find(field.name);
                if (it != existing_ids.end()) {
                    field.field_id = it->second;  // Preserve existing ID
                } else {
                    field.field_id = next_new_id++;  // Assign new ID
                }
            }
        }
    }
}
```
---
[[CRITERIA_JSON: {"module_id": "zcmb-m2", "criteria": ["SchemaParser implements recursive descent parser for namespace, table, struct, enum definitions with lexer producing KEYWORD, IDENTIFIER, STRING, NUMBER, SYMBOL tokens", "LayoutCalculator sorts fields by alignment descending and computes offsets ensuring no overlap with table_size padded to 8-byte boundary", "VTable structure is 4-byte header plus 2-byte field offset entries with soffset pointing backward from table to vtable", "BufferView provides get_scalar template method with field_id lookup returning default when vtable offset is zero or field_id exceeds vtable_size", "VectorView stores uint32 length prefix followed by packed elements with const_iterator for range-based for loops", "StringView stores uint32 length prefix followed by null-terminated UTF-8 data with comparison operators against const char and std::string", "BuilderBase writes strings and vectors growing from buffer end then constructs table with vtable preceding table data", "CodeGenerator produces View class with inline typed accessors and Builder class with set methods for each field", "Schema evolution assigns sequential field IDs starting from zero never reusing IDs for deprecated fields", "Benchmark demonstrates field access latency below 20 nanoseconds representing 10x improvement over JSON parsing", "Field ID assignment algorithm preserves existing IDs when schema is extended and assigns new sequential IDs for added fields", "Generated View class inherits from BufferView and provides has_field methods checking vtable presence before offset lookup"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: zcmb-m3 -->
# Technical Design Specification: Multi-Producer Multi-Consumer
## Module Charter
The Multi-Producer Multi-Consumer module extends the SPSC ring buffer to support concurrent access from multiple producers and consumers while maintaining lock-free guarantees. It implements Dmitry Vyukov's MPMC queue algorithm using per-slot sequence numbers to prevent the ABA problem and coordinate ownership transfer between producers and consumers. **What it does NOT do**: message serialization (M2), pub/sub routing (M4), durability/crash recovery (M5). **Upstream dependencies**: POSIX shared memory from M1, atomic operations library, the SPSC ring buffer pattern (for sharded variant). **Downstream consumers**: Pub/sub layer (M4) uses MPMC for topic fan-out with multiple subscribers, applications with parallel producer/consumer topologies. **Core invariants**: (1) Each slot has a monotonically increasing sequence number that never wraps within a buffer cycle, (2) producer claims slot via CAS on tail, then spins until `sequence[slot] == claimed_position`, (3) consumer claims slot via CAS on head, then spins until `sequence[slot] == claimed_position + 1`, (4) after producer writes: `sequence[slot] = position + 1`, (5) after consumer reads: `sequence[slot] = position + capacity`, (6) sharded variant maintains per-shard SPSC semantics with cross-shard fairness for consumers, (7) backpressure propagates via credit-based flow control without blocking the fast path.
---
## File Structure
```
zcmb-m3-multi-producer-multi-consumer/
├── 01_include/
│   └── zcmb/
│       ├── 01_mpmc_ring_buffer.hpp      // Vyukov algorithm implementation
│       ├── 02_mpmc_config.hpp           // Configuration structs
│       ├── 03_sharded_mpmc.hpp          // Sharded queue with N SPSC buffers
│       ├── 04_backpressure.hpp          // Credit-based flow control
│       └── 05_contention_metrics.hpp    // CAS failure tracking
├── 02_src/
│   ├── 01_mpmc_ring_buffer.cpp          // Core algorithm
│   ├── 02_sharded_mpmc.cpp              // Shard management
│   ├── 03_backpressure.cpp              // Flow control logic
│   └── 04_contention_metrics.cpp        // Metrics collection
├── 03_tests/
│   ├── 01_mpmc_basic_test.cpp           // Single producer/consumer baseline
│   ├── 02_mpmc_contention_test.cpp      // Multi-producer stress test
│   ├── 03_mpmc_aba_test.cpp             // Sequence number wraparound
│   ├── 04_sharded_mpmc_test.cpp         // Shard distribution
│   ├── 05_backpressure_test.cpp         // Flow control behavior
│   ├── 06_fairness_test.cpp             // Starvation detection
│   └── 07_cross_process_test.cpp        // Fork-based MPMC IPC
├── 04_benchmarks/
│   ├── 01_contention_bench.cpp          // CAS failure rates at various N
│   ├── 02_latency_bench.cpp             // P50/P99/P999 under contention
│   ├── 03_throughput_bench.cpp          // Scaling with producer count
│   └── 04_comparison_bench.cpp          // Single MPMC vs Sharded
└── 05_CMakeLists.txt
```
---
## Complete Data Model
### MpmcConfig
```cpp
// File: 01_include/zcmb/02_mpmc_config.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <chrono>
namespace zcmb {
struct MpmcConfig {
    size_t num_slots;           // Must be power of 2
    size_t slot_size;           // Bytes per message slot
    const char* shm_name;       // POSIX shm name (or nullptr for heap)
    // Backpressure settings
    size_t max_spins = 100;     // Spins before yield
    size_t max_yields = 1000;   // Yields before sleep
    std::chrono::microseconds sleep_duration{1}; // Sleep when exhausted
    // Fairness settings
    bool enable_fairness = true;
    size_t starvation_threshold = 100000; // Spins before warning
    bool is_valid() const noexcept;
    size_t calculate_total_size() const noexcept;
};
struct ShardedMpmcConfig {
    size_t num_shards;          // Number of SPSC buffers
    size_t shard_slots;         // Slots per shard (power of 2)
    size_t slot_size;           // Bytes per message slot
    const char* shm_name_prefix;// Shm name prefix (shard N: prefix_N)
    // Shard selection
    enum class SelectionPolicy { ROUND_ROBIN, HASH_BASED, LEAST_LOADED };
    SelectionPolicy producer_policy = SelectionPolicy::ROUND_ROBIN;
    SelectionPolicy consumer_policy = SelectionPolicy::ROUND_ROBIN;
    bool is_valid() const noexcept;
};
} // namespace zcmb
```
### MpmcRingBufferHeader (In Shared Memory)
```cpp
// Layout at offset 0 of shared memory region
// MUST be identical between all producer and consumer processes
struct alignas(64) MpmcRingBufferHeader {
    // Cache line 0: Head (consumer claim position)
    alignas(64) std::atomic<uint64_t> head;     // Offset 0x00: Consumer fetch_add target
    uint8_t padding0[56];                        // Offset 0x08
    // Cache line 1: Tail (producer claim position)  
    alignas(64) std::atomic<uint64_t> tail;     // Offset 0x40: Producer fetch_add target
    uint8_t padding1[56];                        // Offset 0x48
    // Cache line 2: Metadata
    uint64_t capacity;                           // Offset 0x80: num_slots
    uint64_t slot_size;                          // Offset 0x88: Bytes per slot
    uint64_t mask;                               // Offset 0x90: capacity - 1
    uint64_t sequence_base;                      // Offset 0x98: Initial sequence (usually 0)
    // Cache line 3: Metrics (optional, for monitoring)
    alignas(64) std::atomic<uint64_t> total_enqueued;    // Offset 0xC0
    alignas(64) std::atomic<uint64_t> total_dequeued;    // Offset 0xC8
    alignas(64) std::atomic<uint64_t> enqueue_failures;  // Offset 0xD0: CAS/contention
    alignas(64) std::atomic<uint64_t> dequeue_failures;  // Offset 0xD8
    // Total: 256 bytes (4 cache lines)
};
static_assert(sizeof(MpmcRingBufferHeader) == 256, "Header must be 256 bytes");
```
### Per-Slot Sequence Number Array
```cpp
// Sequence numbers stored SEPARATELY from message data
// This allows cache-efficient access without touching message slots
// 
// Memory layout after header:
//   [Header: 256 bytes]
//   [Sequence array: num_slots × sizeof(atomic<uint64_t>)]
//   [Slot 0: slot_size bytes]
//   [Slot 1: slot_size bytes]
//   ...
//
// Sequence number semantics:
//   seq == position          → Slot is EMPTY, ready for producer to claim
//   seq == position + 1      → Slot has DATA, ready for consumer to read
//   seq == position + cap    → Slot was CONSUMED, ready for next cycle
//   seq < position           → Slot is BEHIND (queue full condition)
//   seq > position + 1       → Slot is AHEAD (shouldn't happen)
```
**Memory Layout Diagram**:
```
MPMC Ring Buffer Memory Layout:
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 0x000: MpmcRingBufferHeader (256 bytes)                      │
│   ├─ 0x000-0x03F: head (atomic) + padding [Cache Line 0]            │
│   ├─ 0x040-0x07F: tail (atomic) + padding [Cache Line 1]            │
│   ├─ 0x080-0x0BF: capacity, slot_size, mask, seq_base [Cache Line 2]│
│   └─ 0x0C0-0x0FF: metrics [Cache Line 3]                            │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x100: Sequence Number Array (num_slots × 8 bytes)           │
│   ├─ 0x100: seq[0] = 0 (initially expects position 0)               │
│   ├─ 0x108: seq[1] = 1 (initially expects position 1)               │
│   ├─ 0x110: seq[2] = 2                                               │
│   └─ ...                                                             │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x100 + num_slots*8: Slot 0 (slot_size bytes)                │
│ Offset 0x100 + num_slots*8 + slot_size: Slot 1                      │
│ ...                                                                  │
│ Offset 0x100 + num_slots*8 + (num_slots-1)*slot_size: Slot N-1      │
└─────────────────────────────────────────────────────────────────────┘
Total size = 256 + (num_slots × 8) + (num_slots × slot_size)
```
### MpmcRingBuffer Class
```cpp
// File: 01_include/zcmb/01_mpmc_ring_buffer.hpp
#pragma once
#include "02_mpmc_config.hpp"
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <thread>
namespace zcmb {
class MpmcRingBuffer {
public:
    using Header = MpmcRingBufferHeader;
    static constexpr size_t HEADER_SIZE = sizeof(Header);
    // === Lifecycle ===
    // Initialize as creator (allocates and sets up header + sequences)
    void initialize(void* memory, const MpmcConfig& config);
    // Attach to existing buffer
    void attach(void* memory) noexcept;
    // === Production (Multiple Producers) ===
    // Non-blocking enqueue
    // Returns true on success, false if queue full or contention
    [[nodiscard]] bool try_enqueue(const void* data, size_t size) noexcept;
    // Blocking enqueue with exponential backoff
    void enqueue(const void* data, size_t size);
    // === Consumption (Multiple Consumers) ===
    // Non-blocking dequeue
    [[nodiscard]] bool try_dequeue(void* out_data, size_t out_size) noexcept;
    // Blocking dequeue with exponential backoff
    void dequeue(void* out_data, size_t out_size);
    // === Queries ===
    size_t capacity() const noexcept;     // num_slots - 1 (one slot reserved)
    size_t slot_size() const noexcept;
    bool empty() const noexcept;          // Approximate snapshot
    bool full() const noexcept;           // Approximate snapshot
    // === Metrics ===
    uint64_t total_enqueued() const noexcept;
    uint64_t total_dequeued() const noexcept;
    uint64_t enqueue_failures() const noexcept;
    double enqueue_failure_rate() const noexcept;
    // === Static Utilities ===
    static size_t calculate_size(const MpmcConfig& config) noexcept {
        return HEADER_SIZE + 
               config.num_slots * sizeof(std::atomic<uint64_t>) + // Sequences
               config.num_slots * config.slot_size;               // Slots
    }
private:
    void* get_slot(size_t index) noexcept;
    const void* get_slot(size_t index) const noexcept;
    std::atomic<uint64_t>* get_sequence(size_t index) noexcept;
    const std::atomic<uint64_t>* get_sequence(size_t index) const noexcept;
    static void exponential_backoff(size_t spin_count, const MpmcConfig& config);
    Header* header_ = nullptr;
    std::atomic<uint64_t>* sequences_ = nullptr;
    uint8_t* slots_ = nullptr;
    MpmcConfig config_;
};
} // namespace zcmb
```
### ShardedMpmcQueue Class
```cpp
// File: 01_include/zcmb/03_sharded_mpmc.hpp
#pragma once
#include "02_mpmc_config.hpp"
#include <atomic>
#include <vector>
#include <memory>
namespace zcmb {
// Forward declaration - uses SPSC from M1
class SpscRingBuffer;
class ShardedMpmcQueue {
public:
    // Per-shard structure (cache-line aligned)
    struct alignas(64) Shard {
        alignas(64) std::atomic<uint64_t> head{0};
        alignas(64) std::atomic<uint64_t> tail{0};
        uint8_t* slots{nullptr};
        size_t slot_size{0};
        size_t mask{0};
        // Statistics
        alignas(64) std::atomic<uint64_t> enqueue_count{0};
        alignas(64) std::atomic<uint64_t> dequeue_count{0};
        char padding[64 - 6 * sizeof(size_t) - 2 * sizeof(std::atomic<uint64_t>)];
    };
    explicit ShardedMpmcQueue(const ShardedMpmcConfig& config);
    ~ShardedMpmcQueue();
    // Move-only
    ShardedMpmcQueue(ShardedMpmcQueue&&) noexcept;
    ShardedMpmcQueue& operator=(ShardedMpmcQueue&&) noexcept;
    ShardedMpmcQueue(const ShardedMpmcQueue&) = delete;
    ShardedMpmcQueue& operator=(const ShardedMpmcQueue&) = delete;
    // === Production ===
    // Round-robin shard selection
    bool try_enqueue(const void* data, size_t size);
    // Hash-based shard selection (for ordering within hash bucket)
    bool try_enqueue_hashed(uint64_t key, const void* data, size_t size);
    // === Consumption ===
    // Check all shards round-robin for fairness
    bool try_dequeue(void* out_data, size_t out_size, size_t& out_shard_idx);
    // Batch dequeue - process up to max_messages from any shards
    template<typename Handler>
    size_t dequeue_batch(Handler&& handler, size_t max_messages);
    // === Queries ===
    size_t num_shards() const noexcept { return shards_.size(); }
    size_t total_capacity() const noexcept;
    // Per-shard statistics for load balancing
    uint64_t shard_enqueue_count(size_t shard_idx) const;
    uint64_t shard_dequeue_count(size_t shard_idx) const;
    double shard_load_factor(size_t shard_idx) const;
private:
    bool enqueue_to_shard(size_t shard_idx, const void* data, size_t size);
    bool dequeue_from_shard(size_t shard_idx, void* out_data, size_t out_size);
    size_t select_shard_round_robin() noexcept;
    size_t select_shard_least_loaded() noexcept;
    std::vector<Shard> shards_;
    alignas(64) std::atomic<uint32_t> producer_idx_{0};
    alignas(64) std::atomic<uint32_t> consumer_idx_{0};
    ShardedMpmcConfig config_;
};
} // namespace zcmb
```
### BackpressureController Class
```cpp
// File: 01_include/zcmb/04_backpressure.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
namespace zcmb {
// Credit-based flow control
// Consumer grants credits, producer consumes credits to send
class BackpressureController {
public:
    struct Config {
        int32_t initial_credits;      // Starting credit pool
        int32_t low_watermark;        // Trigger backpressure when below
        int32_t high_watermark;       // Clear backpressure when above
        bool enable_notifications;    // Call callback on state changes
    };
    explicit BackpressureController(const Config& config);
    // === Producer Side ===
    // Try to consume a credit (returns false if no credits)
    [[nodiscard]] bool try_consume_credit() noexcept;
    // Check if backpressure is active (credits below watermark)
    bool is_backpressured() const noexcept;
    // === Consumer Side ===
    // Grant credits after processing messages
    void grant_credits(int32_t count) noexcept;
    // Get current credit count (for monitoring)
    int32_t current_credits() const noexcept;
    // === Callbacks ===
    using BackpressureCallback = std::function<void(bool is_active)>;
    void set_callback(BackpressureCallback callback);
private:
    alignas(64) std::atomic<int32_t> credits_;
    alignas(64) std::atomic<bool> backpressure_active_{false};
    Config config_;
    BackpressureCallback callback_;
};
// Rate limiter for fairness
class TokenBucket {
public:
    struct Config {
        uint64_t rate_per_second;
        uint64_t burst_size;
    };
    explicit TokenBucket(const Config& config);
    // Try to consume tokens (returns actual tokens consumed, 0 if empty)
    [[nodiscard]] uint64_t try_consume(uint64_t tokens = 1) noexcept;
    // Refill tokens (called periodically or on timer)
    void refill() noexcept;
private:
    static uint64_t now_ns() noexcept;
    alignas(64) std::atomic<uint64_t> tokens_;
    alignas(64) std::atomic<uint64_t> last_refill_ns_;
    Config config_;
};
} // namespace zcmb
```
### ContentionMetrics Class
```cpp
// File: 01_include/zcmb/05_contention_metrics.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
namespace zcmb {
struct ContentionMetrics {
    // CAS statistics
    std::atomic<uint64_t> cas_attempts{0};
    std::atomic<uint64_t> cas_failures{0};
    std::atomic<uint64_t> cas_retries{0};
    // Spin statistics
    std::atomic<uint64_t> total_spins{0};
    std::atomic<uint64_t> max_spins{0};
    // Wait statistics
    std::atomic<uint64_t> yield_count{0};
    std::atomic<uint64_t> sleep_count{0};
    // Timing
    std::atomic<uint64_t> total_wait_ns{0};
    std::atomic<uint64_t> max_wait_ns{0};
    // Computed metrics
    double cas_failure_rate() const noexcept;
    double avg_wait_ns() const noexcept;
    void reset() noexcept;
    void record_cas_attempt(bool success) noexcept;
    void record_spin(size_t count) noexcept;
    void record_yield() noexcept;
    void record_sleep() noexcept;
    void record_wait_ns(uint64_t ns) noexcept;
};
class ContentionTracker {
public:
    explicit ContentionTracker(ContentionMetrics& metrics);
    ~ContentionTracker();
    void record_cas(bool success) { metrics_.record_cas_attempt(success); }
    void record_spin(size_t count) { metrics_.record_spin(count); }
    void start_wait();
    void end_wait();
private:
    ContentionMetrics& metrics_;
    std::chrono::high_resolution_clock::time_point wait_start_;
};
} // namespace zcmb
```
---
## Interface Contracts
### MpmcRingBuffer::try_enqueue
```cpp
[[nodiscard]] bool try_enqueue(const void* data, size_t size) noexcept;
```
**Preconditions**:
- `initialize()` or `attach()` called successfully
- `data != nullptr` if `size > 0`
- Multiple threads may call concurrently
**Postconditions**:
- Returns `true`: message copied to slot, sequence updated, `tail` advanced conceptually
- Returns `false`: queue full OR lost CAS race (caller should retry)
- No partial writes visible to consumers
- Memory ordering: `release` semantics before sequence update
**Algorithm**:
```
1. pos = tail.load(relaxed)
2. LOOP:
   a. seq = sequence[pos & mask].load(acquire)
   b. diff = seq - pos
   c. IF diff == 0:
      // Slot is ready (expects position == pos)
      IF tail.compare_exchange_weak(pos, pos + 1, relaxed, relaxed):
         // Claimed! Exit loop with pos
         BREAK
      // CAS failed, pos updated by CAS, retry
   d. ELIF diff < 0:
      // Slot is behind → queue is full
      RETURN false
   e. ELSE:
      // Slot is ahead (another producer claimed but not written)
      // Reload tail and retry
      pos = tail.load(relaxed)
3. // We own slot at (pos & mask)
   slot_idx = pos & mask
   memcpy(slots[slot_idx], data, min(size, slot_size))
   // Publish: sequence = pos + 1 signals consumer
   sequence[slot_idx].store(pos + 1, release)
4. RETURN true
```
### MpmcRingBuffer::try_dequeue
```cpp
[[nodiscard]] bool try_dequeue(void* out_data, size_t out_size) noexcept;
```
**Preconditions**:
- `initialize()` or `attach()` called successfully
- `out_data != nullptr` if `out_size > 0`
- Multiple threads may call concurrently
**Postconditions**:
- Returns `true`: message copied to `out_data`, sequence updated
- Returns `false`: queue empty OR lost CAS race
- Memory ordering: `acquire` semantics before reading slot
**Algorithm**:
```
1. pos = head.load(relaxed)
2. LOOP:
   a. seq = sequence[pos & mask].load(acquire)
   b. diff = seq - (pos + 1)
   c. IF diff == 0:
      // Slot has data (sequence == pos + 1)
      IF head.compare_exchange_weak(pos, pos + 1, relaxed, relaxed):
         // Claimed! Exit loop with pos
         BREAK
      // CAS failed, pos updated, retry
   d. ELIF diff < 0:
      // Slot expects earlier position → queue empty
      RETURN false
   e. ELSE:
      // Slot is ahead (shouldn't happen in correct impl)
      pos = head.load(relaxed)
3. // We own slot at (pos & mask)
   slot_idx = pos & mask
   memcpy(out_data, slots[slot_idx], min(out_size, slot_size))
   // Release: sequence = pos + capacity signals next cycle
   sequence[slot_idx].store(pos + capacity, release)
4. RETURN true
```
### ShardedMpmcQueue::try_enqueue_hashed
```cpp
bool try_enqueue_hashed(uint64_t key, const void* data, size_t size);
```
**Preconditions**:
- Shards initialized
- `data != nullptr` if `size > 0`
**Postconditions**:
- Returns `true`: message in shard `key % num_shards`
- Returns `false`: that specific shard is full
- Messages with same `key` always go to same shard (ordering preserved within key)
**Use Case**: When ordering matters for related messages (e.g., all orders for symbol "AAPL" should be processed in order).
### BackpressureController::try_consume_credit
```cpp
[[nodiscard]] bool try_consume_credit() noexcept;
```
**Preconditions**: None (thread-safe)
**Postconditions**:
- Returns `true`: credit decremented, producer may proceed
- Returns `false`: no credits available, producer should wait/drop
- If crossing below `low_watermark`, callback invoked with `true`
---
## Algorithm Specification
### Algorithm: Vyukov MPMC Slot Lifecycle
**Purpose**: Coordinate ownership transfer between producers and consumers using sequence numbers.
**State Transitions for slot[i]**:
```
Initial state (i < capacity):
  sequence[i] = i  // Slot expects position i
Producer claims position P (where P & mask == i):
  1. Spin until sequence[i] == P
  2. Write message data
  3. Set sequence[i] = P + 1  // Signals: data ready
Consumer claims position C (where C & mask == i):
  1. Spin until sequence[i] == C + 1
  2. Read message data
  3. Set sequence[i] = C + capacity  // Signals: slot recycled
Example with capacity = 4:
  Initial:  seq[0]=0, seq[1]=1, seq[2]=2, seq[3]=3
  Producer A claims pos 0 (slot 0):
    - seq[0] was 0, matches pos 0 ✓
    - Write data to slot 0
    - seq[0] = 0 + 1 = 1
  Producer B claims pos 1 (slot 1):
    - seq[1] was 1, matches pos 1 ✓
    - Write data to slot 1
    - seq[1] = 1 + 1 = 2
  Consumer claims pos 0 (slot 0):
    - seq[0] is 1, need pos+1=1 ✓
    - Read data from slot 0
    - seq[0] = 0 + 4 = 4  // Ready for next cycle
  Producer C claims pos 2 (slot 2):
    - seq[2] was 2, matches pos 2 ✓
    - Write data
    - seq[2] = 3
  ... eventually Producer wraps to pos 4 (slot 0):
    - seq[0] is 4, matches pos 4 ✓
    - Write data
    - seq[0] = 5
```
**Invariant**: `sequence[i]` monotonically increases. With 64-bit counters, overflow takes ~584 years at 1B msg/sec.
### Algorithm: Exponential Backoff with Adaptive Thresholds
**Purpose**: Reduce contention under high load without sacrificing latency at low load.
**Input**: `spin_count` (iterations so far), `config` (thresholds)
**Output**: Wait time / yield decision
```
IF spin_count < config.max_spins:
  // Phase 1: Spin with PAUSE
  for i in 0..(spin_count / 4):
    __builtin_ia32_pause()  // x86 PAUSE, ~1 cycle
  RETURN
IF spin_count < config.max_yields:
  // Phase 2: Yield to scheduler
  std::this_thread::yield()
  RETURN
// Phase 3: Sleep to save CPU
std::this_thread::sleep_for(config.sleep_duration)
```
**Tuning Guidelines**:
| Scenario | max_spins | max_yields | sleep_duration |
|----------|-----------|------------|----------------|
| Ultra-low latency | 1000 | 10000 | 1μs |
| Balanced | 100 | 1000 | 10μs |
| CPU-efficient | 10 | 100 | 100μs |
### Algorithm: Sharded Queue Load Balancing
**Purpose**: Distribute load across shards while maintaining ordering within hash bucket.
**Producer Selection (Round-Robin)**:
```
shard_idx = producer_idx.fetch_add(1, relaxed) % num_shards
IF enqueue_to_shard(shard_idx, data, size):
  RETURN true
// Shard full, try others
FOR i in 0..num_shards:
  alt_idx = (shard_idx + i) % num_shards
  IF enqueue_to_shard(alt_idx, data, size):
    RETURN true
RETURN false
```
**Consumer Fairness**:
```
start_shard = consumer_idx.load(relaxed)
FOR i in 0..num_shards:
  shard_idx = (start_shard + i) % num_shards
  IF dequeue_from_shard(shard_idx, out_data, out_size):
    consumer_idx.store((shard_idx + 1) % num_shards, relaxed)
    RETURN true, shard_idx
RETURN false
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible |
|-------|-------------|----------|--------------|
| CAS_CONTENTION | `try_enqueue` CAS failure | Retry with backoff | No (automatic) |
| QUEUE_FULL | `seq < pos` in enqueue loop | Return false, caller handles | Depends on caller |
| QUEUE_EMPTY | `seq < pos + 1` in dequeue loop | Return false, caller handles | No (normal) |
| SLOT_TIMEOUT | Spin count exceeds threshold | Log warning, continue spinning | Monitoring only |
| ABA_ANOMALY | `seq > pos + 1` unexpectedly | Log error, reset sequence | Yes (bug indicator) |
| CONSUMER_STARVATION | `starvation_threshold` exceeded | Log warning, adjust scheduling | Monitoring |
| SHARD_IMBALANCE | Load factor variance > 50% | Log warning, suggest rehash | Monitoring |
| INVALID_CONFIG | `is_valid()` returns false | Throw `std::invalid_argument` | Yes (startup) |
| MEMORY_ALLOC_FAIL | `new` returns nullptr | Throw `std::bad_alloc` | Yes (startup) |
| SHM_MAP_FAIL | `mmap` returns MAP_FAILED | Throw `std::runtime_error` | Yes (startup) |
---
## Implementation Sequence with Checkpoints
### Phase 1: Vyukov MPMC Core (4-5 hours)
**Files**: `01_mpmc_ring_buffer.hpp`, `01_mpmc_ring_buffer.cpp`, `02_mpmc_config.hpp`
**Steps**:
1. Define `MpmcConfig` with validation
2. Define `MpmcRingBufferHeader` with exact byte layout
3. Implement `initialize()` - allocate sequences, set initial values
4. Implement `attach()` - pointer setup only
5. Implement `try_enqueue()` - CAS loop with sequence check
6. Implement `try_dequeue()` - CAS loop with sequence check
7. Implement blocking variants with backoff
8. Implement queries: `capacity()`, `empty()`, `full()`
**Checkpoint**:
```bash
./03_tests/01_mpmc_basic_test
# Expected:
# [PASS] Initialize with power-of-2 config
# [PASS] Single producer, single consumer baseline
# [PASS] Fill to capacity
# [PASS] Empty completely
# [PASS] Wraparound at buffer end
# [PASS] Sequence numbers advance correctly
```
**At this point**: MPMC works for single producer/consumer (same as SPSC but with extra sequence logic).
### Phase 2: Sequence Number Management (2-3 hours)
**Files**: Modify `01_mpmc_ring_buffer.cpp`
**Steps**:
1. Initialize sequence array: `seq[i] = i` for all slots
2. Implement producer sequence update: `seq[slot] = pos + 1`
3. Implement consumer sequence update: `seq[slot] = pos + capacity`
4. Add sequence validation in debug builds
5. Test sequence wraparound with small buffer
**Checkpoint**:
```bash
./03_tests/03_mpmc_aba_test
# Expected:
# [PASS] Sequence starts at slot index
# [PASS] Producer sets seq = pos + 1
# [PASS] Consumer sets seq = pos + capacity
# [PASS] Wraparound after capacity messages
# [PASS] No ABA with 1000 wraparounds
```
**At this point**: Sequence numbers prevent ABA, multiple producers can claim slots safely.
### Phase 3: Contention Management (2-3 hours)
**Files**: `05_contention_metrics.hpp`, `04_contention_metrics.cpp`, modifications to `01_mpmc_ring_buffer.cpp`
**Steps**:
1. Implement `ContentionMetrics` struct
2. Add metrics tracking to enqueue/dequeue
3. Implement exponential backoff helper
4. Add configurable spin/yield/sleep thresholds
5. Integrate `ContentionTracker` RAII helper
6. Test with multiple producer threads
**Checkpoint**:
```bash
./03_tests/02_mpmc_contention_test
# Expected:
# [PASS] 2 producers, no message loss
# [PASS] 4 producers, CAS failure rate < 50%
# [PASS] 8 producers, all messages delivered
# [PASS] Metrics track CAS attempts/failures
# [PASS] Backoff reduces CPU usage
```
**At this point**: MPMC handles multiple producers with tracked contention.
### Phase 4: Sharded Queue Alternative (3-4 hours)
**Files**: `03_sharded_mpmc.hpp`, `02_sharded_mpmc.cpp`
**Steps**:
1. Define `Shard` structure with cache-line alignment
2. Implement constructor: allocate N shards
3. Implement round-robin producer selection
4. Implement fair consumer scanning
5. Implement hash-based selection variant
6. Add per-shard statistics
**Checkpoint**:
```bash
./03_tests/04_sharded_mpmc_test
# Expected:
# [PASS] Create 8 shards
# [PASS] Round-robin distributes across shards
# [PASS] Hash-based selection is deterministic
# [PASS] Consumer checks all shards
# [PASS] Load balance within 20% variance
```
**At this point**: Sharded alternative available for high-contention scenarios.
### Phase 5: Backpressure & Fairness (2-3 hours)
**Files**: `04_backpressure.hpp`, `03_backpressure.cpp`, `06_fairness_test.cpp`
**Steps**:
1. Implement `BackpressureController` with credit pool
2. Implement `TokenBucket` for rate limiting
3. Add per-producer fairness tracking
4. Implement starvation detection
5. Add callback mechanism for backpressure events
6. Integrate with MPMC queue
**Checkpoint**:
```bash
./03_tests/05_backpressure_test
./03_tests/06_fairness_test
# Expected:
# [PASS] Credits are consumed atomically
# [PASS] Backpressure triggers at low watermark
# [PASS] Grant credits clears backpressure
# [PASS] Token bucket limits rate
# [PASS] Fairness tracking detects imbalance
```
**At this point**: Full MPMC with flow control and fairness.
---
## Test Specification
### Test: MPMC Multi-Producer Stress
```cpp
TEST(MpmcRingBuffer, MultiProducerStress) {
    MpmcConfig config{
        .num_slots = 1024,
        .slot_size = 64,
        .shm_name = nullptr  // Heap allocation
    };
    size_t total_size = MpmcRingBuffer::calculate_size(config);
    auto memory = std::make_unique<uint8_t[]>(total_size);
    MpmcRingBuffer queue;
    queue.initialize(memory.get(), config);
    const size_t num_producers = 4;
    const size_t num_consumers = 2;
    const size_t msgs_per_producer = 10000;
    std::atomic<size_t> produced{0};
    std::atomic<size_t> consumed{0};
    std::atomic<bool> done{false};
    // Track message IDs to verify no loss
    std::vector<std::set<uint64_t>> seen_by_consumer(num_consumers);
    std::mutex seen_mutex;
    // Producers
    std::vector<std::thread> producers;
    for (size_t p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p] {
            for (size_t i = 0; i < msgs_per_producer; ++i) {
                uint64_t msg_id = p * msgs_per_producer + i;
                uint8_t msg[64];
                std::memcpy(msg, &msg_id, sizeof(msg_id));
                queue.enqueue(msg, sizeof(msg));  // Blocking
                produced.fetch_add(1, relaxed);
            }
        });
    }
    // Consumers
    std::vector<std::thread> consumers;
    for (size_t c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&, c] {
            uint8_t msg[64];
            while (!done || consumed.load() < num_producers * msgs_per_producer) {
                if (queue.try_dequeue(msg, sizeof(msg))) {
                    uint64_t msg_id;
                    std::memcpy(&msg_id, msg, sizeof(msg_id));
                    {
                        std::lock_guard<std::mutex> lock(seen_mutex);
                        seen_by_consumer[c].insert(msg_id);
                    }
                    consumed.fetch_add(1, relaxed);
                }
            }
        });
    }
    // Wait
    for (auto& t : producers) t.join();
    done = true;
    for (auto& t : consumers) t.join();
    // Verify
    EXPECT_EQ(produced.load(), num_producers * msgs_per_producer);
    EXPECT_EQ(consumed.load(), num_producers * msgs_per_producer);
    // Merge all seen IDs
    std::set<uint64_t> all_seen;
    for (auto& seen : seen_by_consumer) {
        all_seen.insert(seen.begin(), seen.end());
    }
    EXPECT_EQ(all_seen.size(), num_producers * msgs_per_producer);
    // Check CAS failure rate
    double fail_rate = queue.enqueue_failure_rate();
    std::cout << "CAS failure rate: " << (fail_rate * 100) << "%\n";
    EXPECT_LT(fail_rate, 0.5);  // Less than 50% failure
}
```
### Test: Sequence Number Wraparound (ABA Prevention)
```cpp
TEST(MpmcRingBuffer, SequenceWraparound) {
    MpmcConfig config{
        .num_slots = 4,  // Very small to force wraparound
        .slot_size = 16,
        .shm_name = nullptr
    };
    size_t total_size = MpmcRingBuffer::calculate_size(config);
    auto memory = std::make_unique<uint8_t[]>(total_size);
    MpmcRingBuffer queue;
    queue.initialize(memory.get(), config);
    uint8_t msg[16];
    uint8_t recv[16];
    // Cycle through buffer 100 times
    for (size_t cycle = 0; cycle < 100; ++cycle) {
        // Fill buffer (capacity = 3)
        for (size_t i = 0; i < 3; ++i) {
            memset(msg, static_cast<uint8_t>(cycle * 10 + i), sizeof(msg));
            EXPECT_TRUE(queue.try_enqueue(msg, sizeof(msg)));
        }
        EXPECT_TRUE(queue.full());
        // Drain buffer
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_TRUE(queue.try_dequeue(recv, sizeof(recv)));
            EXPECT_EQ(recv[0], static_cast<uint8_t>(cycle * 10 + i));
        }
        EXPECT_TRUE(queue.empty());
    }
    // Verify sequence numbers have advanced correctly
    // After 100 cycles × 3 slots = 300 operations per slot
    // Sequence numbers should be around 300-303
    auto* header = reinterpret_cast<MpmcRingBufferHeader*>(memory.get());
    EXPECT_EQ(header->head.load(), 300);
    EXPECT_EQ(header->tail.load(), 300);
}
```
### Test: Sharded Queue Load Balance
```cpp
TEST(ShardedMpmcQueue, LoadBalance) {
    ShardedMpmcConfig config{
        .num_shards = 8,
        .shard_slots = 64,
        .slot_size = 32,
        .shm_name_prefix = nullptr,
        .producer_policy = ShardedMpmcConfig::SelectionPolicy::ROUND_ROBIN
    };
    ShardedMpmcQueue queue(config);
    const size_t num_messages = 10000;
    uint8_t msg[32];
    // Enqueue with round-robin
    for (size_t i = 0; i < num_messages; ++i) {
        memset(msg, static_cast<uint8_t>(i), sizeof(msg));
        EXPECT_TRUE(queue.try_enqueue(msg, sizeof(msg)));
    }
    // Check per-shard distribution
    std::vector<uint64_t> counts(queue.num_shards());
    for (size_t s = 0; s < queue.num_shards(); ++s) {
        counts[s] = queue.shard_enqueue_count(s);
    }
    // All shards should have ~num_messages / num_shards
    uint64_t expected = num_messages / queue.num_shards();
    for (size_t s = 0; s < queue.num_shards(); ++s) {
        EXPECT_NEAR(counts[s], expected, expected * 0.1)  // Within 10%
            << "Shard " << s << " has " << counts[s];
    }
}
```
### Test: Backpressure Flow Control
```cpp
TEST(BackpressureController, FlowControl) {
    BackpressureController::Config config{
        .initial_credits = 10,
        .low_watermark = 3,
        .high_watermark = 7,
        .enable_notifications = true
    };
    BackpressureController ctrl(config);
    std::vector<bool> backpressure_events;
    ctrl.set_callback([&](bool active) {
        backpressure_events.push_back(active);
    });
    // Consume credits
    EXPECT_FALSE(ctrl.is_backpressured());
    for (int i = 0; i < 7; ++i) {
        EXPECT_TRUE(ctrl.try_consume_credit());
    }
    // Now at 3 credits, should trigger backpressure
    EXPECT_TRUE(ctrl.is_backpressured());
    EXPECT_EQ(backpressure_events.back(), true);
    // No more credits
    ctrl.try_consume_credit();  // 2 left
    ctrl.try_consume_credit();  // 1 left
    ctrl.try_consume_credit();  // 0 left
    EXPECT_FALSE(ctrl.try_consume_credit());  // Rejected!
    // Grant credits
    ctrl.grant_credits(5);  // Now at 5
    EXPECT_FALSE(ctrl.is_backpressured());  // Above high watermark? No, 5 < 7
    EXPECT_EQ(ctrl.current_credits(), 5);
    ctrl.grant_credits(3);  // Now at 8
    EXPECT_FALSE(ctrl.is_backpressured());
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| `try_enqueue` (2 producers) | < 300 ns median | `perf stat` with 1M samples |
| `try_enqueue` (8 producers, sharded) | < 500 ns median | Same |
| `try_dequeue` (any producer count) | < 200 ns median | Same |
| P99 latency (2 producers) | < 600 ns | Percentile calculation |
| P99 latency (8 producers, single MPMC) | < 10 μs | May include backoff |
| P99 latency (8 producers, sharded) | < 2 μs | Sharded variant |
| CAS failure rate (2 producers) | < 15% | Metrics tracking |
| CAS failure rate (8 producers, sharded) | < 20% | Sharded variant |
| Throughput scaling | Linear with shards | 8 shards ≈ 8× single shard |
| No message loss | 0 lost | Verify all message IDs |
| CPU efficiency (idle) | < 1% core | `top` when queue empty |
**Contention Benchmark Matrix**:
```
Producers | Single MPMC          | Sharded (8)
          | Throughput | P99     | Throughput | P99
----------|------------|---------|------------|--------
1         | 12 M/s     | 250 ns  | 11 M/s     | 300 ns
2         | 9 M/s      | 600 ns  | 10 M/s     | 450 ns
4         | 6 M/s      | 2.1 μs  | 9 M/s      | 800 ns
8         | 3.5 M/s    | 12 μs   | 8 M/s      | 1.5 μs
```
---
## Concurrency Specification
### Lock-Free Guarantee
The MPMC ring buffer is **lock-free**: at least one thread makes progress even if others are descheduled. It is NOT wait-free (individual threads may spin indefinitely under extreme contention).
### Memory Ordering
```
PRODUCER                              CONSUMER
========                              =========
// Claim slot
CAS(tail, pos, pos+1)                 CAS(head, pos, pos+1)
   ↓ [acquire if successful]             ↓ [acquire if successful]
// Wait for ownership                  // Wait for data
load(seq, acquire)                     load(seq, acquire)
   ↓ until seq == pos                     ↓ until seq == pos + 1
memcpy(slot, data)                     memcpy(out, slot)
   ↓                                      ↓
fence(release)                         fence(acquire)
   ↓                                      ↓
store(seq, pos+1, release)             store(seq, pos+cap, release)
```
### Thread Safety by Component
| Component | Thread Safety | Notes |
|-----------|--------------|-------|
| `MpmcRingBuffer` | Lock-free MPMC | Multiple producers and consumers |
| `ShardedMpmcQueue` | Lock-free per shard | SPSC per shard, MPMC overall |
| `BackpressureController` | Lock-free | Atomic credits |
| `ContentionMetrics` | Lock-free | Relaxed atomics for counters |
### Cross-Process Safety
Same considerations as SPSC (M1):
- Atomic operations work across process boundaries on same hardware
- Memory barriers required for visibility
- Cache-line alignment prevents false sharing
- PID/generation tracking for crash recovery
---
[[CRITERIA_JSON: {"module_id": "zcmb-m3", "criteria": ["MpmcRingBufferHeader is 256 bytes with head at offset 0x00 and tail at offset 0x40 on separate cache lines using alignas(64)", "Per-slot sequence array stored after header at offset 0x100 with num_slots times sizeof(atomic<uint64_t>) bytes", "Sequence number semantics: slot expects position when seq equals pos, has data when seq equals pos plus one, recycled when seq equals pos plus capacity", "try_enqueue uses CAS loop on tail with sequence check diff equals zero for slot ready and diff less than zero for queue full", "try_dequeue uses CAS loop on head with sequence check diff equals zero for data ready and diff less than zero for queue empty", "Exponential backoff with three phases: spin with PAUSE instruction for max_spins iterations, yield to scheduler for max_yields iterations, then sleep for configured duration", "ShardedMpmcQueue contains N Shard structures each cache-line aligned with separate head, tail, slots, and per-shard statistics", "Shard selection supports round-robin via atomic counter increment and hash-based via key modulo num_shards", "BackpressureController implements credit pool with atomic int32 credits, low_watermark triggers backpressure, grant_credits adds to pool", "TokenBucket rate limiter with atomic token count and periodic refill based on elapsed nanoseconds", "ContentionMetrics tracks CAS attempts, failures, retries, total spins, max spins, yield count, sleep count, and wait nanoseconds", "CAS failure rate target below 15 percent for 2 producers and below 20 percent for 8 producers using sharded variant", "P99 latency target below 600 nanoseconds for 2 producers and below 2 microseconds for 8 producers using sharded variant", "No message loss under contention verified by tracking all message IDs across producers and consumers"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: zcmb-m4 -->
# Technical Design Specification: Publish-Subscribe & Topics
## Module Charter
The Publish-Subscribe & Topics module provides topic-based message routing on top of the MPMC ring buffer, enabling efficient one-to-many message distribution with zero-copy semantics. It implements a trie-based topic registry supporting MQTT-style wildcards (`+` for single-level, `#` for multi-level), reference-counted message slots for zero-copy fan-out, per-subscriber filtering with bloom filter pre-checks, and retained message storage for late-joiner delivery. **What it does NOT do**: message serialization (M2), multi-producer queue management (M3), or durability/crash recovery (M5). **Upstream dependencies**: MPMC ring buffer from M3 for message transport, flat buffer format from M2 for message content. **Downstream consumers**: Applications subscribe to topics and receive typed messages, network bridges export to external MQTT brokers. **Core invariants**: (1) A message is written once to shared memory and read by N subscribers via reference counting, (2) topic matching completes in O(M) where M = topic depth, not O(N) subscribers, (3) wildcards never match across topic boundaries (`+` matches exactly one level, `#` matches zero or more), (4) reference count reaches zero only after ALL matching subscribers have acknowledged, (5) subscriber cursors advance independently but never skip messages, (6) bloom filter may have false positives but never false negatives.
---
## File Structure
```
zcmb-m4-publish-subscribe-topics/
├── 01_include/
│   └── zcmb/
│       ├── pubsub/
│       │   ├── 01_topic_trie.hpp           // Trie node structure + matching
│       │   ├── 02_topic_registry.hpp       // Subscribe/unsubscribe/match API
│       │   ├── 03_pubsub_buffer.hpp        // Ref-counted message slots
│       │   ├── 04_subscriber_cursor.hpp    // Per-subscriber position tracking
│       │   ├── 05_message_filter.hpp       // Predicate evaluation + bloom
│       │   ├── 06_retained_store.hpp       // Last-known-good message cache
│       │   ├── 07_last_will.hpp            // Disconnect notification
│       │   └── 08_pubsub_broker.hpp        // Unified publish/subscribe API
│       └── 09_pubsub_config.hpp            // Configuration structs
├── 02_src/
│   ├── pubsub/
│   │   ├── 01_topic_trie.cpp               // Trie traversal implementation
│   │   ├── 02_topic_registry.cpp           // RW-lock protected registry
│   │   ├── 03_pubsub_buffer.cpp            // Ref count management
│   │   ├── 04_subscriber_cursor.cpp        // Cursor advancement
│   │   ├── 05_message_filter.cpp           // Filter compilation/evaluation
│   │   ├── 06_retained_store.cpp           // Retained message storage
│   │   ├── 07_last_will.cpp                // Will delivery + heartbeat
│   │   └── 08_pubsub_broker.cpp            // Broker orchestration
│   └── 09_pubsub_config.cpp                // Config validation
├── 03_tests/
│   ├── 01_topic_trie_test.cpp              // Trie insert/match/delete
│   ├── 02_wildcard_test.cpp                // + and # wildcard matching
│   ├── 03_pubsub_buffer_test.cpp           // Ref count lifecycle
│   ├── 04_subscriber_cursor_test.cpp       // Independent cursor advance
│   ├── 05_message_filter_test.cpp          // Bloom + predicate evaluation
│   ├── 06_retained_store_test.cpp          // Late subscriber delivery
│   ├── 07_last_will_test.cpp               // Disconnect notification
│   ├── 08_pubsub_broker_test.cpp           // End-to-end pub/sub
│   └── 09_cross_process_test.cpp           // Fork-based IPC test
├── 04_benchmarks/
│   ├── 01_topic_match_bench.cpp            // Exact vs wildcard latency
│   ├── 02_fan_out_bench.cpp                // 1/10/100 subscriber scaling
│   ├── 03_filter_bench.cpp                 // Bloom vs full evaluation
│   └── 04_end_to_end_bench.cpp             // Publish to delivery latency
└── 05_CMakeLists.txt
```
---
## Complete Data Model
### PubSubConfig
```cpp
// File: 01_include/zcmb/09_pubsub_config.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <string>
namespace zcmb {
struct PubSubConfig {
    size_t num_slots;              // Must be power of 2
    size_t slot_size;              // Max message size
    size_t max_subscribers;        // Max concurrent subscribers per topic
    size_t max_subscriptions;      // Max subscriptions per subscriber
    size_t max_topic_depth;        // Max levels in topic hierarchy
    size_t retained_max_size;      // Max retained message size
    size_t retained_max_count;     // Max retained messages per topic
    const char* shm_name;          // POSIX shm name prefix
    // Wildcard characters (MQTT-style)
    static constexpr char SINGLE_LEVEL_WILDCARD = '+';
    static constexpr char MULTI_LEVEL_WILDCARD = '#';
    static constexpr char TOPIC_SEPARATOR = '/';
    bool is_valid() const noexcept;
    size_t calculate_total_size() const noexcept;
};
struct ShardedPubSubConfig {
    size_t num_shards;             // Number of broker shards
    PubSubConfig shard_config;     // Per-shard config
    bool wildcard_fanout;          // Subscribe to all shards for wildcards
    bool is_valid() const noexcept;
};
} // namespace zcmb
```
### TopicTrieNode
```cpp
// File: 01_include/zcmb/pubsub/01_topic_trie.hpp
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <cstdint>
namespace zcmb {
using SubscriberID = uint64_t;
// Trie node for topic hierarchy
// Each node represents one segment of a topic (e.g., "stocks" in "stocks/AAPL/price")
struct TopicTrieNode {
    // Children for exact segment matches
    // Key: topic segment (e.g., "AAPL")
    // Value: child node
    std::unordered_map<std::string, std::unique_ptr<TopicTrieNode>> exact_children;
    // Child for single-level wildcard (+)
    // Matches exactly one topic level
    std::unique_ptr<TopicTrieNode> wildcard_child;
    // Subscribers at this node (exact match or pattern ends here)
    std::vector<SubscriberID> subscribers;
    // Subscribers for multi-level wildcard (#)
    // Matches this node and ALL descendants
    std::vector<SubscriberID> multi_level_subscribers;
    // Read-write lock for thread-safe access
    // Allows concurrent reads, exclusive writes
    mutable std::shared_mutex mutex;
    // Node statistics (for monitoring)
    size_t depth{0};              // Distance from root
    size_t subtree_size{1};       // Nodes in subtree (including this)
    TopicTrieNode() = default;
    ~TopicTrieNode() = default;
    // Non-copyable, movable
    TopicTrieNode(const TopicTrieNode&) = delete;
    TopicTrieNode& operator=(const TopicTrieNode&) = delete;
    TopicTrieNode(TopicTrieNode&&) = default;
    TopicTrieNode& operator=(TopicTrieNode&&) = default;
};
} // namespace zcmb
```
**Memory Layout of TopicTrieNode**:
```
TopicTrieNode (typical 64-bit system):
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 0x00: std::unordered_map<std::string, unique_ptr<Node>>      │
│              (exact_children)                                        │
│   ├─ 0x00: pointer to hash table (8 bytes)                          │
│   └─ Size: ~48-64 bytes depending on implementation                 │
├─────────────────────────────────────────────────────────────────────┤
│ Offset ~0x30: std::unique_ptr<TopicTrieNode> (wildcard_child)       │
│   └─ 8 bytes (pointer)                                              │
├─────────────────────────────────────────────────────────────────────┤
│ Offset ~0x38: std::vector<SubscriberID> (subscribers)               │
│   ├─ pointer to data (8 bytes)                                      │
│   ├─ size (8 bytes)                                                 │
│   └─ capacity (8 bytes)                                             │
├─────────────────────────────────────────────────────────────────────┤
│ Offset ~0x58: std::vector<SubscriberID> (multi_level_subscribers)   │
│   └─ Same layout as above                                           │
├─────────────────────────────────────────────────────────────────────┤
│ Offset ~0x78: std::shared_mutex (mutable)                           │
│   └─ Size: ~40-56 bytes (pthread_rwlock_t internally)              │
├─────────────────────────────────────────────────────────────────────┤
│ Offset ~0xB0: size_t depth (8 bytes)                                │
│ Offset ~0xB8: size_t subtree_size (8 bytes)                         │
└─────────────────────────────────────────────────────────────────────┘
Total: ~200-250 bytes per node (varies by STL implementation)
```
### PubSubBufferHeader
```cpp
// File: 01_include/zcmb/pubsub/03_pubsub_buffer.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <cstddef>
namespace zcmb {
// Header at offset 0 of shared memory region
struct alignas(64) PubSubBufferHeader {
    // Cache line 0: Write position (publisher-owned)
    alignas(64) std::atomic<uint64_t> head;        // Offset 0x00: Next write position
    uint8_t padding0[56];                           // Offset 0x08
    // Cache line 1: Reclaim position (updated when ref_count reaches 0)
    alignas(64) std::atomic<uint64_t> tail;        // Offset 0x40: Oldest unclaimed slot
    uint8_t padding1[56];                           // Offset 0x48
    // Cache line 2: Metadata
    uint64_t num_slots;                             // Offset 0x80: Total slots (power of 2)
    uint64_t slot_size;                             // Offset 0x88: Bytes per slot
    uint64_t mask;                                  // Offset 0x90: num_slots - 1
    uint64_t max_subscribers;                       // Offset 0x98: Max ref_count value
    // Cache line 3: Global statistics
    alignas(64) std::atomic<uint64_t> total_published;   // Offset 0xC0
    alignas(64) std::atomic<uint64_t> total_delivered;   // Offset 0xC8
    alignas(64) std::atomic<uint64_t> publish_failures;  // Offset 0xD0: Buffer full
    std::atomic<uint64_t> orphan_recoveries;             // Offset 0xD8: Slots reclaimed
};
static_assert(sizeof(PubSubBufferHeader) == 256, "Header must be 256 bytes");
```
### MessageSlotHeader
```cpp
// Per-slot metadata prepended to each message
struct alignas(16) MessageSlotHeader {
    std::atomic<uint32_t> ref_count;      // Remaining readers (set to subscriber count)
    uint32_t message_size;                 // Actual message bytes
    uint64_t publish_timestamp_ns;         // For ordering and TTL
    uint32_t topic_hash;                   // Quick bloom filter check
    uint32_t sequence;                     // Monotonic within topic
    uint16_t topic_depth;                  // Number of segments in topic
    uint16_t flags;                        // Reserved for future use
};
static_assert(sizeof(MessageSlotHeader) == 32, "Slot header must be 32 bytes");
```
**Memory Layout of Message Slot**:
```
Message Slot Layout:
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 0x00: MessageSlotHeader (32 bytes)                           │
│   ├─ 0x00: ref_count (atomic uint32, 4 bytes)                       │
│   ├─ 0x04: message_size (uint32, 4 bytes)                           │
│   ├─ 0x08: publish_timestamp_ns (uint64, 8 bytes)                   │
│   ├─ 0x10: topic_hash (uint32, 4 bytes)                             │
│   ├─ 0x14: sequence (uint32, 4 bytes)                               │
│   ├─ 0x18: topic_depth (uint16, 2 bytes)                            │
│   └─ 0x1A: flags (uint16, 2 bytes)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x20: FlatBuffer message data (slot_size - 32 bytes)         │
│   ├─ Message content (variable)                                      │
│   └─ Padding to slot_size boundary                                  │
└─────────────────────────────────────────────────────────────────────┘
Total slot size = 32 + message_size (padded to slot_size)
```
### SubscriberCursor
```cpp
// File: 01_include/zcmb/pubsub/04_subscriber_cursor.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
namespace zcmb {
struct alignas(64) SubscriberCursor {
    // Cache line 0: Position tracking
    alignas(64) std::atomic<uint64_t> current_offset;    // Offset 0x00: Current read position
    uint8_t padding0[56];                                 // Offset 0x08
    // Cache line 1: Subscriber identity
    SubscriberID id;                                      // Offset 0x40: Unique subscriber ID
    uint32_t topic_hash_mask;                             // Offset 0x48: Bloom filter for topics
    uint32_t subscription_count;                          // Offset 0x4C: Active subscriptions
    uint8_t padding1[48];                                 // Offset 0x50
    // Cache line 2: Statistics
    alignas(64) std::atomic<uint64_t> messages_received;  // Offset 0x80
    alignas(64) std::atomic<uint64_t> messages_filtered;  // Offset 0x88: Rejected by filter
    alignas(64) std::atomic<uint64_t> last_message_time;  // Offset 0x90: Timestamp
    std::atomic<uint32_t> consecutive_empty_polls;        // Offset 0x98
    uint8_t padding2[24];                                 // Offset 0x9C
    // Status flags
    std::atomic<bool> active{true};                       // Offset 0xB0
    std::atomic<bool> slow_consumer{false};               // Offset 0xB1: Lagging behind
    uint8_t padding3[54];                                 // Offset 0xB2
};
static_assert(sizeof(SubscriberCursor) == 256, "Cursor must be 256 bytes");
} // namespace zcmb
```
### MessageFilter
```cpp
// File: 01_include/zcmb/pubsub/05_message_filter.hpp
#pragma once
#include <functional>
#include <cstdint>
#include <string>
#include <memory>
#include <bitset>
namespace zcmb {
class BloomFilter {
public:
    static constexpr size_t NUM_BITS = 256;
    static constexpr size_t NUM_HASHES = 3;
    void add(const std::string& topic);
    bool might_contain(const std::string& topic) const;
    void clear() { bits_.reset(); }
    void merge(const BloomFilter& other);
private:
    static size_t hash_topic(const std::string& topic, size_t seed);
    std::bitset<NUM_BITS> bits_;
};
// Filter predicate: returns true if message should be delivered
using FilterPredicate = std::function<bool(const uint8_t* data, uint32_t size)>;
class MessageFilter {
public:
    // Create filter that accepts all messages
    static MessageFilter accept_all();
    // Create filter from field path and expected value
    static MessageFilter create_field_filter(
        const std::string& field_path,
        const std::string& expected_value);
    // Create range filter (numeric fields)
    static MessageFilter create_range_filter(
        const std::string& field_path,
        double min_val,
        double max_val);
    // Create presence filter (field must exist)
    static MessageFilter create_presence_filter(
        const std::string& field_path);
    // Combine filters
    static MessageFilter combine_and(std::vector<MessageFilter> filters);
    static MessageFilter combine_or(std::vector<MessageFilter> filters);
    static MessageFilter negate(MessageFilter filter);
    // Evaluation
    bool evaluate(const uint8_t* data, uint32_t size) const;
    bool accepts_all() const { return accepts_all_; }
    // Bloom filter for quick rejection
    const BloomFilter& topic_bloom() const { return topic_bloom_; }
    BloomFilter& topic_bloom() { return topic_bloom_; }
private:
    MessageFilter() = default;
    FilterPredicate predicate_;
    BloomFilter topic_bloom_;
    bool accepts_all_{false};
    std::string description_;  // For debugging
};
} // namespace zcmb
```
### RetainedMessage
```cpp
// File: 01_include/zcmb/pubsub/06_retained_store.hpp
#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <string>
#include <shared_mutex>
namespace zcmb {
struct RetainedMessage {
    std::vector<uint8_t> data;           // Message payload (copy)
    uint64_t timestamp_ns;               // When retained
    uint32_t topic_hash;                 // Quick lookup
    uint16_t topic_depth;                // Number of segments
    bool retain_flag;                    // From publish
};
class RetainedMessageStore {
public:
    explicit RetainedMessageStore(size_t max_size = 4096, size_t max_count = 10000);
    // Store retained message for topic
    void store(const std::string& topic, const void* data, uint32_t size);
    // Get retained message (returns false if none)
    bool get(const std::string& topic, std::vector<uint8_t>& out_data,
             uint64_t& out_timestamp) const;
    // Clear retained message
    void clear(const std::string& topic);
    // Get all topics with retained messages matching pattern
    std::vector<std::string> get_matching_topics(const std::string& pattern) const;
    // Statistics
    size_t size() const;
    size_t memory_usage() const;
private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, RetainedMessage> retained_;
    size_t max_message_size_;
    size_t max_message_count_;
    static uint64_t now_ns();
    static uint32_t hash_topic(const std::string& topic);
};
} // namespace zcmb
```
### LastWillConfig
```cpp
// File: 01_include/zcmb/pubsub/07_last_will.hpp
#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_map>
namespace zcmb {
struct LastWillConfig {
    std::string topic;                   // Topic to publish will to
    std::vector<uint8_t> payload;        // Will message content
    bool retain;                         // Should will be retained?
    uint8_t qos;                         // Quality of service (0, 1, 2)
};
class SubscriberManager {
public:
    explicit SubscriberManager(
        std::chrono::seconds timeout = std::chrono::seconds(30));
    ~SubscriberManager();
    // Register subscriber with optional last-will
    void register_subscriber(SubscriberID id, const LastWillConfig& will = {});
    // Update heartbeat (subscriber is alive)
    void heartbeat(SubscriberID id);
    // Unregister (normal disconnect - don't send will)
    void unregister_subscriber(SubscriberID id);
    // Check if subscriber is alive
    bool is_alive(SubscriberID id) const;
    // Set will delivery callback
    using WillCallback = std::function<void(const std::string& topic,
                                            const void* data, size_t size, bool retain)>;
    void set_will_callback(WillCallback callback);
private:
    void monitor_loop();
    void check_timeouts();
    void send_last_will(SubscriberID id);
    mutable std::mutex mutex_;
    std::unordered_map<SubscriberID, LastWillConfig> will_configs_;
    std::unordered_map<SubscriberID, std::chrono::steady_clock::time_point> last_heartbeat_;
    std::chrono::seconds timeout_;
    WillCallback will_callback_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};
};
} // namespace zcmb
```
### PubSubBroker (Unified API)
```cpp
// File: 01_include/zcmb/pubsub/08_pubsub_broker.hpp
#pragma once
#include "01_topic_trie.hpp"
#include "03_pubsub_buffer.hpp"
#include "04_subscriber_cursor.hpp"
#include "05_message_filter.hpp"
#include "06_retained_store.hpp"
#include "07_last_will.hpp"
#include "09_pubsub_config.hpp"
#include <memory>
#include <functional>
namespace zcmb {
class PubSubBroker {
public:
    using MessageHandler = std::function<void(const uint8_t* data, uint32_t size,
                                               const std::string& topic)>;
    explicit PubSubBroker(const PubSubConfig& config);
    ~PubSubBroker();
    // Non-copyable, non-movable
    PubSubBroker(const PubSubBroker&) = delete;
    PubSubBroker& operator=(const PubSubBroker&) = delete;
    // === Publishing ===
    // Publish message to topic (returns false if buffer full)
    bool publish(const std::string& topic, const void* data, uint32_t size,
                 bool retain = false);
    // Publish with explicit timestamp (for testing/replay)
    bool publish_with_timestamp(const std::string& topic, const void* data,
                                 uint32_t size, uint64_t timestamp_ns, bool retain = false);
    // === Subscription ===
    // Subscribe to topic pattern (may contain + and # wildcards)
    // Returns subscriber ID for subsequent operations
    SubscriberID subscribe(const std::string& pattern,
                           MessageHandler handler,
                           MessageFilter filter = MessageFilter::accept_all());
    // Unsubscribe from all topics
    void unsubscribe(SubscriberID id);
    // Unsubscribe from specific pattern
    void unsubscribe(SubscriberID id, const std::string& pattern);
    // === Message Processing ===
    // Poll for messages (non-blocking)
    // Returns number of messages delivered
    size_t poll(SubscriberID id, size_t max_messages = 100);
    // Poll all subscribers (typically called from event loop)
    size_t poll_all(size_t max_messages_per_subscriber = 100);
    // Blocking wait for messages (with timeout)
    size_t wait_for_messages(SubscriberID id,
                             std::chrono::milliseconds timeout,
                             size_t max_messages = 100);
    // === Retained Messages ===
    // Get retained message for topic (for late joiners)
    bool get_retained(const std::string& topic, std::vector<uint8_t>& out_data);
    // Clear retained message
    void clear_retained(const std::string& topic);
    // === Process Coordination ===
    // Register subscriber with last-will
    void register_with_will(SubscriberID id, const LastWillConfig& will);
    // Update heartbeat
    void heartbeat(SubscriberID id);
    // === Statistics ===
    size_t subscriber_count() const;
    size_t subscription_count() const;
    uint64_t messages_published() const;
    uint64_t messages_delivered() const;
    // Per-subscriber stats
    struct SubscriberStats {
        uint64_t messages_received;
        uint64_t messages_filtered;
        uint64_t lag_ms;  // How far behind
        bool active;
    };
    SubscriberStats get_subscriber_stats(SubscriberID id) const;
private:
    // Deliver to matching subscribers
    size_t deliver_to_subscribers(const std::string& topic,
                                   const uint8_t* data, uint32_t size,
                                   uint32_t topic_hash);
    // Deliver retained messages to new subscriber
    void deliver_retained_messages(const std::string& pattern, SubscriberID id);
    // Generate unique subscriber ID
    SubscriberID generate_subscriber_id();
    // Split topic into segments
    static std::vector<std::string> split_topic(const std::string& topic);
    // Compute topic hash for bloom filter
    static uint32_t compute_topic_hash(const std::string& topic);
    PubSubConfig config_;
    std::unique_ptr<TopicTrieNode> topic_registry_;
    std::unique_ptr<SubscriberManager> subscriber_manager_;
    RetainedMessageStore retained_store_;
    // Shared memory buffer
    void* shm_memory_;
    PubSubBufferHeader* buffer_header_;
    MessageSlotHeader* slots_;
    // Subscriber state
    mutable std::shared_mutex subscribers_mutex_;
    std::unordered_map<SubscriberID, SubscriberCursor> subscriber_cursors_;
    std::unordered_map<SubscriberID, MessageHandler> subscriber_handlers_;
    std::unordered_map<SubscriberID, MessageFilter> subscriber_filters_;
    std::unordered_map<SubscriberID, BloomFilter> subscriber_blooms_;
    std::atomic<SubscriberID> next_subscriber_id_{1};
    std::atomic<uint64_t> topic_sequence_{0};
};
} // namespace zcmb
```
---
## Interface Contracts
### TopicRegistry::subscribe
```cpp
void TopicRegistry::subscribe(const std::string& pattern, SubscriberID subscriber_id);
```
**Preconditions**:
- `pattern` is a valid topic pattern (may contain `+` and `#`)
- `#` may only appear as the LAST segment
- `subscriber_id` is unique (caller's responsibility)
**Postconditions**:
- `subscriber_id` is registered for `pattern`
- Future `match()` calls for topics matching `pattern` will include `subscriber_id`
- If pattern already exists, subscriber is added to existing node
**Error Conditions**:
| Error | Condition | Recovery |
|-------|-----------|----------|
| INVALID_WILDCARD | `#` not at end | Throw `std::invalid_argument` |
| INVALID_PATTERN | Empty segment | Throw `std::invalid_argument` |
| MEMORY_ERROR | Allocation fails | Throw `std::bad_alloc` |
**Wildcard Rules**:
- `stocks/+` matches `stocks/AAPL` but NOT `stocks/AAPL/price` or `stocks`
- `stocks/#` matches `stocks`, `stocks/AAPL`, `stocks/AAPL/price`, etc.
- `stocks/+/price` matches `stocks/AAPL/price` but NOT `stocks/price`
### TopicRegistry::match
```cpp
std::vector<SubscriberID> TopicRegistry::match(const std::string& topic) const;
```
**Preconditions**:
- `topic` is a valid topic string (no wildcards)
- Registry has been initialized
**Postconditions**:
- Returns all subscriber IDs whose patterns match `topic`
- Result is sorted and deduplicated
- Empty vector if no matches
**Time Complexity**: O(M + K) where M = topic depth, K = matching subscribers
**Algorithm**:
```
1. segments = split_topic(topic)
2. result = empty vector
3. match_recursive(root, segments, 0, result)
4. sort(result)
5. deduplicate(result)
6. return result
match_recursive(node, segments, depth, result):
  // Always add multi-level subscribers at this level
  result.append(node.multi_level_subscribers)
  // If we've consumed all segments, add exact subscribers
  if depth == segments.size():
    result.append(node.subscribers)
    return
  segment = segments[depth]
  // Try exact match
  if segment in node.exact_children:
    match_recursive(node.exact_children[segment], segments, depth+1, result)
  // Try single-level wildcard
  if node.wildcard_child != nullptr:
    match_recursive(node.wildcard_child, segments, depth+1, result)
```
### PubSubBuffer::publish
```cpp
bool PubSubBuffer::publish(const std::string& topic, const void* data, 
                            uint32_t size, uint32_t subscriber_count, bool retain);
```
**Preconditions**:
- Buffer initialized
- `data != nullptr` if `size > 0`
- `subscriber_count <= max_subscribers`
**Postconditions**:
- Returns `true`: message in buffer with `ref_count = subscriber_count`
- Returns `false`: buffer full (no space for new message)
- If `retain == true`, message stored in retained cache
**Memory Ordering**:
```
1. Reserve slot (atomic increment head)
2. Write message data to slot
3. atomic_thread_fence(release)
4. Set ref_count = subscriber_count (atomic store)
5. Notify subscribers (via eventfd or similar)
```
### SubscriberCursor::advance_and_read
```cpp
bool SubscriberCursor::advance_and_read(void* out_data, uint32_t out_size,
                                         uint32_t& out_topic_hash,
                                         PubSubBufferHeader* header,
                                         MessageSlotHeader* slots);
```
**Preconditions**:
- Cursor is active
- `out_data != nullptr`
**Postconditions**:
- Returns `true`: message copied to `out_data`, cursor advanced
- Returns `false`: no message available OR cursor caught up to head
- `ref_count` of slot decremented by 1
**Thread Safety**: Each cursor is owned by one subscriber thread. No cross-thread access.
---
## Algorithm Specification
### Algorithm: Trie-Based Topic Matching
**Purpose**: Find all subscribers whose patterns match a given topic.
**Input**: Topic string (e.g., `"stocks/NASDAQ/AAPL/price"`)
**Output**: Vector of `SubscriberID` values
**Complexity**: O(M) where M = number of topic levels (typically 3-7)
**Detailed Steps**:
```
ALGORITHM match(topic):
  INPUT: topic string with TOPIC_SEPARATOR delimiters
  OUTPUT: vector of matching SubscriberID
  1. segments ← split_topic(topic)
     // "stocks/NASDAQ/AAPL/price" → ["stocks", "NASDAQ", "AAPL", "price"]
  2. result ← empty vector
  3. acquire shared_lock on root->mutex
  4. CALL match_recursive(root, segments, 0, result)
  5. release lock
  6. IF result.size() > 1:
       sort(result)
       deduplicate(result)  // Remove duplicates from multiple paths
  7. RETURN result
ALGORITHM match_recursive(node, segments, depth, result):
  // Base case: add multi-level wildcard subscribers
  // These match regardless of current depth
  acquire shared_lock on node->mutex
  FOR EACH id IN node->multi_level_subscribers:
    result.push_back(id)
  // Terminal case: consumed all segments
  IF depth == segments.size():
    FOR EACH id IN node->subscribers:
      result.push_back(id)
    release lock
    RETURN
  segment ← segments[depth]
  // Path 1: Exact match on segment
  IF segment exists in node->exact_children:
    child ← node->exact_children[segment]
    release lock
    CALL match_recursive(child, segments, depth+1, result)
    acquire shared_lock on node->mutex  // Re-acquire for next path
  // Path 2: Single-level wildcard (+)
  IF node->wildcard_child != nullptr:
    child ← node->wildcard_child
    release lock
    CALL match_recursive(child, segments, depth+1, result)
  ELSE:
    release lock
  RETURN
```
**Example Trace**:
```
Pattern subscriptions:
  - "stocks/+/price" → subscriber 1
  - "stocks/#" → subscriber 2
  - "stocks/NASDAQ/price" → subscriber 3
Topic: "stocks/NASDAQ/price"
match_recursive(root, ["stocks", "NASDAQ", "price"], 0, []):
  // No multi-level at root
  // Try exact "stocks" → found, recurse
  match_recursive(stocks_node, ["stocks", "NASDAQ", "price"], 1, []):
    // No multi-level at stocks
    // Try exact "NASDAQ" → found, recurse
    match_recursive(NASDAQ_node, ["stocks", "NASDAQ", "price"], 2, []):
      // No multi-level at NASDAQ
      // Try exact "price" → found, recurse
      match_recursive(price_node, ["stocks", "NASDAQ", "price"], 3, []):
        // depth == segments.size()
        // Add exact subscribers: [3]
        return
      // Try wildcard (+) at depth 2
      // wildcard_child exists for "stocks/+/price"
      match_recursive(wildcard_node, ..., 3, [3]):
        // depth == segments.size()
        // Add exact subscribers: [3, 1]
        return
    // Try wildcard (+) at depth 1 (matches "NASDAQ")
    // But no wildcard_child here
  // At stocks level, try multi-level (#)
  // multi_level_subscribers contains [2]
  // result: [3, 1, 2]
Final result: [1, 2, 3] (sorted)
```
### Algorithm: Zero-Copy Fan-Out with Reference Counting
**Purpose**: Deliver one message to N subscribers without copying.
**Input**: Message data, topic, subscriber count N
**Output**: Message in shared memory with `ref_count = N`
**Detailed Steps**:
```
ALGORITHM publish_with_fanout(topic, data, size, subscriber_ids):
  INPUT: topic string, data pointer, size bytes, list of subscriber IDs
  OUTPUT: true if published, false if buffer full
  1. // Phase 1: Find matching subscribers
     matching ← TopicRegistry.match(topic)
     N ← matching.size()
     IF N == 0:
       // No subscribers - optionally store as retained
       IF retain_flag:
         RetainedMessageStore.store(topic, data, size)
       RETURN true  // No one to deliver to
  2. // Phase 2: Reserve slot in buffer
     pos ← buffer_header->head.load(relaxed)
     slot_idx ← pos & mask
     // Check if slot is free (ref_count == 0)
     slot ← &slots[slot_idx]
     ref ← slot->ref_count.load(acquire)
     IF ref != 0:
       // Slot still has readers from previous message
       RETURN false  // Buffer full
     // Try to claim slot
     IF NOT slot->ref_count.compare_exchange_strong(ref, N, acq_rel, relaxed):
       // Lost race with another publisher
       RETURN false  // Retry externally
  3. // Phase 3: Write message data
     slot->message_size ← min(size, slot_size - sizeof(MessageSlotHeader))
     slot->publish_timestamp_ns ← current_timestamp_ns()
     slot->topic_hash ← compute_topic_hash(topic)
     slot->sequence ← next_sequence(topic)
     slot->topic_depth ← count_segments(topic)
     memcpy(slot->data, data, slot->message_size)
     // Ensure data is visible before advancing head
     atomic_thread_fence(release)
     buffer_header->head.store(pos + 1, release)
  4. // Phase 4: Store retained if requested
     IF retain_flag:
       RetainedMessageStore.store(topic, data, size)
  5. // Phase 5: Notify subscribers (asynchronously)
     FOR EACH id IN matching:
       subscriber ← get_subscriber(id)
       subscriber->has_pending ← true
       // Platform-specific notification (eventfd, futex, etc.)
       notify(subscriber)
  6. buffer_header->total_published.fetch_add(1, relaxed)
     RETURN true
```
### Algorithm: Subscriber Message Consumption with Filtering
**Purpose**: Subscriber reads next message, applies filter, acknowledges.
**Input**: Subscriber cursor, filter predicate
**Output**: Message data or "no message"
**Detailed Steps**:
```
ALGORITHM subscriber_poll(subscriber_id, max_messages):
  INPUT: subscriber ID, maximum messages to process
  OUTPUT: number of messages actually delivered
  subscriber ← get_subscriber(subscriber_id)
  delivered ← 0
  FOR i = 0 TO max_messages - 1:
    // Step 1: Get current position
    pos ← subscriber->cursor.current_offset.load(relaxed)
    head ← buffer_header->head.load(acquire)
    IF pos >= head:
      // Caught up - no new messages
      subscriber->consecutive_empty_polls.fetch_add(1, relaxed)
      BREAK
    // Step 2: Get slot
    slot_idx ← pos & mask
    slot ← &slots[slot_idx]
    // Step 3: Quick bloom filter check
    topic_hash ← slot->topic_hash
    IF (subscriber->topic_hash_mask & topic_hash) == 0:
      // Definitely not for us (no false negatives)
      subscriber->cursor.current_offset.store(pos + 1, relaxed)
      CONTINUE
    // Step 4: Acquire slot data
    atomic_thread_fence(acquire)
    // Step 5: Check if message matches subscriptions
    // (Full topic match - more expensive)
    topic ← reconstruct_topic(slot)  // Or compare hash
    matching ← TopicRegistry.match(topic)
    IF subscriber_id NOT IN matching:
      subscriber->cursor.current_offset.store(pos + 1, relaxed)
      CONTINUE
    // Step 6: Apply message filter (if any)
    filter ← subscriber->filter
    IF filter != nullptr AND NOT filter.evaluate(slot->data, slot->message_size):
      // Filtered out
      subscriber->messages_filtered.fetch_add(1, relaxed)
      subscriber->cursor.current_offset.store(pos + 1, relaxed)
      // Still need to decrement ref_count!
      slot->ref_count.fetch_sub(1, release)
      CONTINUE
    // Step 7: Deliver to handler
    handler ← subscriber->handler
    handler(slot->data, slot->message_size, topic)
    // Step 8: Acknowledge (decrement ref count)
    prev_ref ← slot->ref_count.fetch_sub(1, acq_rel)
    // Step 9: If last reader, try to advance tail
    IF prev_ref == 1:
      try_reclaim_slots()
    // Step 10: Advance cursor
    subscriber->cursor.current_offset.store(pos + 1, release)
    subscriber->messages_received.fetch_add(1, relaxed)
    subscriber->last_message_time.store(current_timestamp_ns(), relaxed)
    subscriber->consecutive_empty_polls.store(0, relaxed)
    delivered ← delivered + 1
  RETURN delivered
ALGORITHM try_reclaim_slots():
  // Advance tail past consecutive slots with ref_count == 0
  current_tail ← buffer_header->tail.load(relaxed)
  current_head ← buffer_header->head.load(acquire)
  WHILE current_tail < current_head:
    slot_idx ← current_tail & mask
    ref ← slots[slot_idx].ref_count.load(acquire)
    IF ref != 0:
      BREAK  // Slot still has readers
    current_tail ← current_tail + 1
  IF current_tail > buffer_header->tail.load(relaxed):
    buffer_header->tail.store(current_tail, release)
```
### Algorithm: Bloom Filter for Quick Topic Rejection
**Purpose**: Avoid expensive trie traversal for non-matching topics.
**Input**: Topic string, subscriber's bloom filter mask
**Output**: True if topic MIGHT match, False if DEFINITELY doesn't match
**Detailed Steps**:
```
ALGORITHM bloom_might_match(topic, subscriber_mask):
  INPUT: topic string, subscriber's accumulated bloom filter mask
  OUTPUT: true if might match (may be false positive), false if definitely doesn't
  // Compute bloom filter for this topic
  hash1 ← murmurhash3(topic, seed=0)
  hash2 ← murmurhash3(topic, seed=1)
  hash3 ← murmurhash3(topic, seed=2)
  // Check all three bits in subscriber mask
  bit1 ← 1 << (hash1 % 256)
  bit2 ← 1 << (hash2 % 256)
  bit3 ← 1 << (hash3 % 256)
  IF (subscriber_mask & bit1) == 0:
    RETURN false  // Definitely not subscribed
  IF (subscriber_mask & bit2) == 0:
    RETURN false
  IF (subscriber_mask & bit3) == 0:
    RETURN false
  RETURN true  // Might be subscribed (or false positive)
```
**False Positive Rate**: With 256 bits and 3 hash functions:
- 10 subscriptions: ~0.1% false positive rate
- 100 subscriptions: ~1% false positive rate
- 1000 subscriptions: ~10% false positive rate
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible |
|-------|-------------|----------|--------------|
| TOPIC_INVALID | `split_topic()` returns empty | Throw `std::invalid_argument` | Yes (API call fails) |
| PATTERN_INVALID | Wildcard `#` not at end | Throw `std::invalid_argument` | Yes |
| SUBSCRIBER_NOT_FOUND | `get_subscriber()` returns nullptr | Return error code | Depends on caller |
| SLOT_ORPHANED | `ref_count` stuck > 0 for > timeout | Log warning, force reset | Monitoring only |
| FILTER_EVALUATION_ERROR | Predicate throws exception | Log error, deliver anyway | Monitoring |
| TOO_MANY_SUBSCRIBERS | `subscriber_count > max` | Return false from publish | Yes (message dropped) |
| RETAINED_TOO_LARGE | `size > retained_max_size` | Don't store, log warning | No (graceful degradation) |
| BUFFER_FULL | All slots have `ref_count > 0` | Return false from publish | Yes (apply backpressure) |
| CURSOR_LAGGING | `current_offset` far behind `head` | Set `slow_consumer` flag | Monitoring |
| DUPLICATE_SUBSCRIPTION | Same (pattern, id) pair | Ignore (idempotent) | No |
| MEMORY_EXHAUSTED | `new` returns nullptr | Throw `std::bad_alloc` | Yes (startup failure) |
| TRY_DEADLOCK | Lock held > 100ms (debug) | Log error, continue | Debug builds only |
---
## Implementation Sequence with Checkpoints
### Phase 1: Topic Trie Router (3-4 hours)
**Files**: `01_topic_trie.hpp`, `01_topic_trie.cpp`, `02_topic_registry.hpp`, `02_topic_registry.cpp`
**Steps**:
1. Define `TopicTrieNode` structure with exact_children, wildcard_child, subscribers
2. Implement `split_topic()` helper to parse topic strings
3. Implement `TopicRegistry::subscribe()` - traverse/create nodes, add subscriber
4. Implement `TopicRegistry::unsubscribe()` - remove subscriber from all nodes
5. Implement `TopicRegistry::match()` - recursive traversal with wildcard handling
6. Add read-write lock protection for thread safety
7. Add validation for wildcard patterns (`#` only at end)
**Checkpoint**:
```bash
./03_tests/01_topic_trie_test
./03_tests/02_wildcard_test
# Expected:
# [PASS] Subscribe to exact topic
# [PASS] Subscribe to single-level wildcard (+)
# [PASS] Subscribe to multi-level wildcard (#)
# [PASS] Match exact topic
# [PASS] Match with + wildcard
# [PASS] Match with # wildcard
# [PASS] Reject # not at end
# [PASS] Handle empty topic segments
# [PASS] Unsubscribe removes from all nodes
```
**At this point**: Can subscribe to topic patterns and match topics to subscriber lists in O(M) time.
### Phase 2: Zero-Copy Fan-Out Buffer (3-4 hours)
**Files**: `03_pubsub_buffer.hpp`, `03_pubsub_buffer.cpp`, `09_pubsub_config.hpp`
**Steps**:
1. Define `PubSubBufferHeader` with exact byte layout (256 bytes)
2. Define `MessageSlotHeader` (32 bytes)
3. Implement buffer initialization with slot allocation
4. Implement `publish()` with CAS-based slot reservation
5. Implement reference count initialization from subscriber count
6. Implement slot reclamation when `ref_count` reaches 0
7. Add memory barriers for cross-process visibility
**Checkpoint**:
```bash
./03_tests/03_pubsub_buffer_test
# Expected:
# [PASS] Initialize buffer with power-of-2 slots
# [PASS] Publish with ref_count = N
# [PASS] Decrement ref_count on ack
# [PASS] Slot reclaims when ref_count = 0
# [PASS] Buffer full detection
# [PASS] Multiple publishers don't corrupt
```
**At this point**: Can publish messages with reference counting for zero-copy fan-out.
### Phase 3: Subscriber Cursors & Filtering (2-3 hours)
**Files**: `04_subscriber_cursor.hpp`, `04_subscriber_cursor.cpp`, `05_message_filter.hpp`, `05_message_filter.cpp`
**Steps**:
1. Define `SubscriberCursor` with position tracking
2. Implement cursor advancement and message reading
3. Implement `BloomFilter` class with add/check/merge
4. Implement `MessageFilter` with predicate evaluation
5. Implement field-level and range filters
6. Integrate bloom filter with subscriber cursors
7. Add filter composition (AND, OR, NOT)
**Checkpoint**:
```bash
./03_tests/04_subscriber_cursor_test
./03_tests/05_message_filter_test
# Expected:
# [PASS] Cursor advances independently
# [PASS] Bloom filter detects non-matching topics
# [PASS] Bloom filter allows all subscribed topics
# [PASS] Field filter evaluates correctly
# [PASS] Range filter checks bounds
# [PASS] AND/OR composition works
# [PASS] Filter rejects non-matching messages
```
**At this point**: Subscribers can read messages with filtering and independent cursors.
### Phase 4: Retained Messages & Last-Will (2-3 hours)
**Files**: `06_retained_store.hpp`, `06_retained_store.cpp`, `07_last_will.hpp`, `07_last_will.cpp`
**Steps**:
1. Implement `RetainedMessageStore` with topic-keyed storage
2. Add retained message delivery on new subscription
3. Implement `SubscriberManager` with heartbeat tracking
4. Implement `LastWillConfig` storage and delivery
5. Add timeout detection for crashed subscribers
6. Integrate with broker for will delivery
**Checkpoint**:
```bash
./03_tests/06_retained_store_test
./03_tests/07_last_will_test
# Expected:
# [PASS] Store retained message
# [PASS] Retrieve retained message
# [PASS] Clear retained message
# [PASS] Deliver retained to new subscriber
# [PASS] Register subscriber with will
# [PASS] Send will on timeout
# [PASS] Don't send will on normal disconnect
```
**At this point**: Late joiners receive retained messages, disconnects trigger last-will.
### Phase 5: Integrated Broker & Scaling (2-3 hours)
**Files**: `08_pubsub_broker.hpp`, `08_pubsub_broker.cpp`, `09_cross_process_test.cpp`
**Steps**:
1. Implement `PubSubBroker` integrating all components
2. Implement `publish()` with topic matching and fan-out
3. Implement `subscribe()` with handler registration
4. Implement `poll()` for message consumption
5. Implement `ShardedPubSubBroker` for horizontal scaling
6. Add per-topic sequence numbers for ordering
7. Run cross-process integration test
**Checkpoint**:
```bash
./03_tests/08_pubsub_broker_test
./03_tests/09_cross_process_test
# Expected:
# [PASS] Publish delivers to all matching subscribers
# [PASS] Wildcard subscriptions work
# [PASS] Filtering rejects non-matching
# [PASS] Retained delivered on subscribe
# [PASS] Cross-process pub/sub
# [PASS] No message loss under load
# [PASS] Ordering preserved per topic
```
**At this point**: Full pub/sub system working across processes.
---
## Test Specification
### Test: Wildcard Matching
```cpp
TEST(TopicRegistry, WildcardMatching) {
    TopicRegistry registry;
    // Subscribe with various patterns
    registry.subscribe("stocks/+/price", 1);
    registry.subscribe("stocks/#", 2);
    registry.subscribe("stocks/NASDAQ/price", 3);
    registry.subscribe("#", 4);
    registry.subscribe("stocks/+/+", 5);
    // Test exact match
    auto match1 = registry.match("stocks/NASDAQ/price");
    EXPECT_THAT(match1, UnorderedElementsAre(1, 2, 3, 4, 5));
    // Test single-level match
    auto match2 = registry.match("stocks/NYSE/price");
    EXPECT_THAT(match2, UnorderedElementsAre(1, 2, 4, 5));
    // Test multi-level match
    auto match3 = registry.match("stocks/NASDAQ/AAPL/price");
    EXPECT_THAT(match3, UnorderedElementsAre(2, 4));
    // Test root match
    auto match4 = registry.match("events");
    EXPECT_THAT(match4, ElementsAre(4));  // Only # matches
    // Test no match
    auto match5 = registry.match("bonds/NASDAQ/price");
    EXPECT_TRUE(match5.empty());
}
```
### Test: Zero-Copy Fan-Out
```cpp
TEST(PubSubBuffer, ZeroCopyFanOut) {
    PubSubConfig config{
        .num_slots = 16,
        .slot_size = 256,
        .max_subscribers = 10
    };
    size_t total_size = config.calculate_total_size();
    auto memory = std::make_unique<uint8_t[]>(total_size);
    PubSubBuffer buffer;
    buffer.initialize(memory.get(), config);
    // Publish with 3 subscribers
    const char* msg = "Hello, subscribers!";
    uint32_t msg_size = strlen(msg) + 1;
    EXPECT_TRUE(buffer.publish("test/topic", msg, msg_size, 3));
    // Verify ref_count = 3
    auto* slot = buffer.get_slot(0);
    EXPECT_EQ(slot->ref_count.load(), 3);
    // Simulate 3 subscribers reading
    for (int i = 0; i < 3; ++i) {
        char recv[256];
        uint32_t hash;
        EXPECT_TRUE(buffer.read(0, recv, sizeof(recv), hash));
        EXPECT_EQ(slot->ref_count.load(), 3 - i - 1);
    }
    // Slot should now be reclaimable
    EXPECT_EQ(slot->ref_count.load(), 0);
    buffer.try_reclaim();
    EXPECT_EQ(buffer.tail_position(), 1);
}
```
### Test: Bloom Filter Rejection
```cpp
TEST(BloomFilter, QuickRejection) {
    BloomFilter filter;
    // Add subscribed topics
    filter.add("stocks/AAPL/price");
    filter.add("stocks/GOOG/price");
    filter.add("orders/new");
    // Should definitely accept subscribed topics
    EXPECT_TRUE(filter.might_contain("stocks/AAPL/price"));
    EXPECT_TRUE(filter.might_contain("stocks/GOOG/price"));
    EXPECT_TRUE(filter.might_contain("orders/new"));
    // Should definitely reject very different topics
    // (though false positives are possible with similar hashes)
    int rejections = 0;
    for (int i = 0; i < 100; ++i) {
        std::string random_topic = "random/topic/" + std::to_string(i);
        if (!filter.might_contain(random_topic)) {
            ++rejections;
        }
    }
    // Most random topics should be rejected
    EXPECT_GT(rejections, 90);  // > 90% rejection rate
}
```
### Test: End-to-End Pub/Sub
```cpp
TEST(PubSubBroker, EndToEnd) {
    PubSubConfig config{
        .num_slots = 64,
        .slot_size = 512,
        .max_subscribers = 100,
        .max_subscriptions = 10,
        .max_topic_depth = 8,
        .retained_max_size = 1024,
        .retained_max_count = 1000
    };
    PubSubBroker broker(config);
    std::vector<std::string> received_by_sub1;
    std::vector<std::string> received_by_sub2;
    // Subscribe to different patterns
    auto sub1 = broker.subscribe("stocks/+/price",
        [&](const uint8_t* data, uint32_t size, const std::string& topic) {
            received_by_sub1.push_back(std::string((const char*)data, size));
        });
    auto sub2 = broker.subscribe("orders/#",
        [&](const uint8_t* data, uint32_t size, const std::string& topic) {
            received_by_sub2.push_back(std::string((const char*)data, size));
        });
    // Publish messages
    broker.publish("stocks/AAPL/price", "AAPL:150.25", 12);
    broker.publish("stocks/GOOG/price", "GOOG:2800.50", 13);
    broker.publish("orders/new", "ORDER:12345", 11);
    broker.publish("orders/cancel", "CANCEL:12345", 12);
    broker.publish("bonds/treasury", "BOND:5Y", 7);  // No subscribers
    // Poll subscribers
    broker.poll(sub1, 10);
    broker.poll(sub2, 10);
    // Verify delivery
    EXPECT_EQ(received_by_sub1.size(), 2);
    EXPECT_EQ(received_by_sub2.size(), 2);
    EXPECT_THAT(received_by_sub1, UnorderedElementsAre("AAPL:150.25", "GOOG:2800.50"));
    EXPECT_THAT(received_by_sub2, UnorderedElementsAre("ORDER:12345", "CANCEL:12345"));
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Topic match (exact) | 10-50 ns | `perf stat` on 1M lookups |
| Topic match (wildcard) | 100-500 ns | Same, with +/# patterns |
| Publish (no subscribers) | 50-100 ns | Quick check and return |
| Publish (1 subscriber) | 200-500 ns | Match + ref_count set + notify |
| Publish (10 subscribers) | 0.5-2 μs | Linear scaling with N |
| Publish (100 subscribers) | 5-20 μs | Fan-out overhead |
| Subscriber poll (1 message) | 100-300 ns | Cursor advance + memcpy |
| Bloom filter check | < 10 ns | Bit operations only |
| Filter evaluation | 50-200 ns | Predicate call |
| End-to-end latency | 0.5-2 μs | Publish to handler |
| Throughput (1 sub) | 1-2 M msg/s | 10-second sustained |
| Throughput (10 subs) | 500K-1M msg/s | Fan-out limited |
**Benchmark Commands**:
```bash
# Topic match latency
./04_benchmarks/01_topic_match_bench --exact --iterations 1000000
./04_benchmarks/01_topic_match_bench --wildcard --iterations 1000000
# Fan-out scaling
./04_benchmarks/02_fan_out_bench --subscribers 1,5,10,50,100
# End-to-end
./04_benchmarks/04_end_to_end_bench --duration 10s
```
---
## Concurrency Specification
### Thread Safety Model
| Component | Thread Safety | Access Pattern |
|-----------|--------------|----------------|
| `TopicRegistry` | Read-write lock | Concurrent reads (match), exclusive writes (subscribe/unsubscribe) |
| `PubSubBuffer` | Lock-free | CAS-based slot reservation, atomic ref_count |
| `SubscriberCursor` | Single-owner | Each subscriber owns its cursor |
| `MessageFilter` | Immutable after creation | Thread-safe reads |
| `RetainedMessageStore` | Read-write lock | Concurrent reads, exclusive writes |
| `SubscriberManager` | Internal mutex | Thread-safe all operations |
| `PubSubBroker` | All components thread-safe | Full concurrent access |
### Lock Ordering (to prevent deadlock)
```
1. subscribers_mutex_ (PubSubBroker)
2. topic_registry_->mutex (TopicRegistry)
3. retained_store_.mutex (RetainedMessageStore)
4. subscriber_manager_->mutex (SubscriberManager)
NEVER acquire in reverse order.
```
### Memory Ordering Requirements
```
PUBLISHER                              SUBSCRIBER
=========                              ==========
// Reserve slot
CAS(ref_count, 0, N)                   
   [acquire if success]                
// Write message                       
memcpy(slot->data, msg)                
   [release fence]                     
store(head, pos+1)  ───────────────→  load(head) [sees new]
                                       [acquire fence]
                                       memcpy(out, slot->data)
// ref_count already set               slot->ref_count.fetch_sub(1)
   [no additional ordering needed]        [acq_rel]
```
---
[[CRITERIA_JSON: {"module_id": "zcmb-m4", "criteria": ["TopicTrieNode contains exact_children map, wildcard_child pointer, subscribers vector, multi_level_subscribers vector, and shared_mutex for thread-safe concurrent reads with exclusive writes", "TopicRegistry subscribe validates wildcard patterns with hash only allowed at final segment and throws invalid_argument for malformed patterns", "TopicRegistry match performs recursive trie traversal checking exact_children and wildcard_child at each level collecting both exact and multi_level_subscribers", "MessageSlotHeader is 32 bytes with atomic ref_count, message_size, publish_timestamp_ns, topic_hash, sequence, topic_depth, and flags fields", "PubSubBuffer publish reserves slot via CAS on ref_count from zero to subscriber_count then writes message data with release fence before head update", "SubscriberCursor is 256 bytes with current_offset at offset 0x00, subscriber identity at offset 0x40, and statistics at offset 0x80 all cache-line aligned", "BloomFilter uses 256 bits with 3 hash functions providing quick topic rejection with no false negatives", "MessageFilter supports field_filter, range_filter, presence_filter and combinable AND/OR/negate operations with predicate evaluation", "RetainedMessageStore maps topic strings to RetainedMessage structs with data vector, timestamp, topic_hash, and topic_depth", "LastWillConfig specifies topic, payload vector, retain flag, and qos for disconnect notification", "SubscriberManager tracks heartbeats per subscriber ID with configurable timeout and sends will on timeout but not on normal unregister", "PubSubBroker integrates TopicRegistry, PubSubBuffer, SubscriberCursors, MessageFilters, RetainedStore, and SubscriberManager with unified publish and subscribe API", "Publish operation matches topic via trie, reserves slot with ref_count equal to matching subscriber count, writes data with release fence, and notifies subscribers", "Subscriber poll advances cursor, checks bloom filter for quick rejection, applies message filter predicate, delivers to handler, and decrements ref_count atomically", "Slot reclamation advances tail when ref_count reaches zero after last subscriber acknowledgment", "Performance targets: exact topic match 10-50 ns, wildcard match 100-500 ns, publish with 1 subscriber 200-500 ns, bloom filter check under 10 ns, end-to-end latency 0.5-2 microseconds"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: zcmb-m5 -->
# Technical Design Specification: Crash Recovery & Durability
## Module Charter
The Crash Recovery & Durability module provides resilience mechanisms for the zero-copy message bus, ensuring system integrity and data preservation across process failures, power loss, and restart scenarios. It implements heartbeat-based crash detection with PID generation tracking to handle process identity across restarts, orphaned slot reclamation using fence-based invalidation to safely recover from producer crashes mid-publish, write-ahead logging (WAL) with checksums for optional durable persistence, and checkpoint/restart with atomic snapshot files for sub-100ms recovery times. **What it does NOT do**: message routing (M4), multi-producer coordination (M3), or serialization (M2). **Upstream dependencies**: POSIX shared memory from M1 for process coordination state, ring buffer structures from M1/M3 for slot reclamation, flat buffer format from M2 for message content. **Downstream consumers**: Applications requiring durability guarantees, process supervisors (systemd) for lifecycle integration, monitoring systems for crash alerting. **Core invariants**: (1) Heartbeat timestamps are monotonic and only updated by the owning process, (2) orphaned slot reclamation never races with a live process (fence validation), (3) WAL entries are checksummed and only replayed if checksum validates, (4) checkpoint files are atomic (write-to-temp + rename), (5) recovery journal tracks progress enabling resumption after crash-during-recovery, (6) at-least-once delivery semantics with idempotent consumer responsibility.
---
## File Structure
```
zcmb-m5-crash-recovery-durability/
├── 01_include/
│   └── zcmb/
│       ├── recovery/
│       │   ├── 01_heartbeat_monitor.hpp    // Liveness tracking with timestamps
│       │   ├── 02_process_identity.hpp      // PID + generation counter
│       │   ├── 03_orphan_reclaimer.hpp      // Slot cleanup with fencing
│       │   ├── 04_fence_manager.hpp         // Monotonic fence value generation
│       │   ├── 05_write_ahead_log.hpp       // Durable log with checksums
│       │   ├── 06_checkpoint_manager.hpp    // Snapshot save/load
│       │   ├── 07_recovery_journal.hpp      // Crash-during-recovery tracking
│       │   ├── 08_signal_handler.hpp        // Graceful shutdown support
│       │   └── 09_recovery_coordinator.hpp  // Unified recovery orchestration
│       └── 10_recovery_config.hpp           // Configuration structs
├── 02_src/
│   ├── recovery/
│   │   ├── 01_heartbeat_monitor.cpp         // Timestamp updates, timeout checks
│   │   ├── 02_process_identity.cpp          // PID tracking, generation increments
│   │   ├── 03_orphan_reclaimer.cpp          // Slot scanning, fence validation
│   │   ├── 04_fence_manager.cpp             // Atomic fence generation
│   │   ├── 05_write_ahead_log.cpp           // Append, sync, truncate, replay
│   │   ├── 06_checkpoint_manager.cpp        // Snapshot serialization
│   │   ├── 07_recovery_journal.cpp          // Phase tracking
│   │   ├── 08_signal_handler.cpp            // Signal disposition setup
│   │   └── 09_recovery_coordinator.cpp      // Recovery sequence
│   └── 10_recovery_config.cpp               // Config validation
├── 03_tests/
│   ├── 01_heartbeat_test.cpp                // Timeout detection
│   ├── 02_process_identity_test.cpp         // PID reuse handling
│   ├── 03_orphan_reclaim_test.cpp           // Slot recovery scenarios
│   ├── 04_fence_test.cpp                    // Fence validation
│   ├── 05_wal_basic_test.cpp                // Append/sync/truncate
│   ├── 06_wal_checksum_test.cpp             // Corruption detection
│   ├── 07_wal_replay_test.cpp               // Recovery replay
│   ├── 08_checkpoint_test.cpp               // Save/load/atomic
│   ├── 09_recovery_journal_test.cpp         // Crash-during-recovery
│   ├── 10_signal_test.cpp                   // Graceful shutdown
│   ├── 11_full_recovery_test.cpp            // End-to-end recovery
│   └── 12_cross_process_test.cpp            // Fork-based crash simulation
├── 04_benchmarks/
│   ├── 01_checkpoint_latency.cpp            // Save/load timing
│   ├── 02_wal_throughput.cpp                // Append/sync throughput
│   └── 03_recovery_time.cpp                 // Full recovery benchmark
└── 05_CMakeLists.txt
```
---
## Complete Data Model
### RecoveryConfig
```cpp
// File: 01_include/zcmb/10_recovery_config.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <string>
namespace zcmb {
struct RecoveryConfig {
    // Heartbeat settings
    std::chrono::milliseconds heartbeat_interval{100};     // Update frequency
    std::chrono::seconds heartbeat_timeout{5};             // Dead if no update for this long
    // Orphan reclamation
    std::chrono::seconds orphan_timeout{10};               // Claim slot for this long
    std::chrono::seconds reclaim_scan_interval{1};         // How often to scan
    // WAL settings
    std::string wal_path;                                   // File path for WAL
    size_t wal_max_size{100 * 1024 * 1024};                // 100 MB max
    bool wal_sync_on_write{false};                          // fsync after each append
    std::chrono::milliseconds wal_sync_interval{10};       // Batch sync interval
    // Checkpoint settings
    std::string checkpoint_dir;                             // Directory for checkpoints
    std::chrono::seconds checkpoint_interval{10};          // How often to checkpoint
    size_t checkpoint_max_count{5};                        // Keep N most recent
    // Recovery settings
    bool recover_on_startup{true};                          // Run recovery at init
    bool replay_wal_after_checkpoint{true};                 // Replay WAL entries after checkpoint
    bool is_valid() const noexcept;
};
} // namespace zcmb
```
### ProcessIdentity (Shared Memory Structure)
```cpp
// File: 01_include/zcmb/recovery/02_process_identity.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <unistd.h>
namespace zcmb {
// Stored in shared memory for cross-process visibility
// Handles PID reuse by tracking generation counter
struct alignas(64) ProcessIdentity {
    // Cache line 0: Identity
    alignas(64) std::atomic<uint32_t> pid{0};           // Offset 0x00: Process ID
    std::atomic<uint32_t> generation{0};                 // Offset 0x04: Incremented on each restart
    std::atomic<uint64_t> last_heartbeat_ns{0};         // Offset 0x08: Timestamp of last update
    std::atomic<uint32_t> state{0};                      // Offset 0x10: 0=dead, 1=starting, 2=alive, 3=stopping
    uint8_t padding0[40];                                // Offset 0x14
    // Cache line 1: Statistics
    alignas(64) std::atomic<uint64_t> messages_processed{0};  // Offset 0x40
    std::atomic<uint64_t> last_message_time_ns{0};            // Offset 0x48
    uint8_t padding1[48];                                     // Offset 0x50
};
static_assert(sizeof(ProcessIdentity) == 128, "ProcessIdentity must be 128 bytes");
// Process state enumeration
enum class ProcessState : uint32_t {
    DEAD = 0,
    STARTING = 1,
    ALIVE = 2,
    STOPPING = 3
};
class ProcessRegistry {
public:
    static constexpr size_t MAX_PROCESSES = 16;
    // Register current process, get slot index
    uint32_t register_process();
    // Unregister (normal shutdown)
    void unregister();
    // Update heartbeat
    void update_heartbeat();
    // Check if process at slot is alive (same generation + heartbeat fresh)
    bool is_alive(uint32_t slot) const noexcept;
    // Get current process info
    uint32_t my_slot() const noexcept { return my_slot_; }
    uint32_t my_generation() const noexcept { return my_generation_; }
private:
    ProcessIdentity* identities_;  // Points into shared memory
    uint32_t my_slot_{0};
    uint32_t my_generation_{0};
};
} // namespace zcmb
```
**Memory Layout of ProcessIdentity Array**:
```
Process Identity Registry (in shared memory):
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 0x000: ProcessIdentity[0] (128 bytes)                        │
│   ├─ 0x00: pid (atomic uint32)                                       │
│   ├─ 0x04: generation (atomic uint32)                                │
│   ├─ 0x08: last_heartbeat_ns (atomic uint64)                         │
│   ├─ 0x10: state (atomic uint32)                                     │
│   └─ 0x40: messages_processed, last_message_time_ns                  │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x080: ProcessIdentity[1] (128 bytes)                         │
├─────────────────────────────────────────────────────────────────────┤
│ ...                                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x780: ProcessIdentity[15] (128 bytes)                        │
└─────────────────────────────────────────────────────────────────────┘
Total: 16 × 128 = 2048 bytes (2 KB)
```
### FenceManager (Slot Ownership Validation)
```cpp
// File: 01_include/zcmb/recovery/04_fence_manager.hpp
#pragma once
#include <atomic>
#include <cstdint>
namespace zcmb {
// Fence value: monotonically increasing counter
// Each process instance gets a unique fence value
// Used to validate slot ownership and detect stale claims
struct alignas(16) FenceHeader {
    std::atomic<uint64_t> global_fence{1};    // Next fence to allocate
    std::atomic<uint64_t> min_active_fence{1}; // Minimum fence still in use
    uint8_t padding[48];
};
static_assert(sizeof(FenceHeader) == 64, "FenceHeader must be 64 bytes");
class FenceManager {
public:
    explicit FenceManager(FenceHeader* header);
    // Allocate a new fence value for this process instance
    // Called once at process startup
    uint64_t allocate_fence();
    // Get current fence for this process
    uint64_t my_fence() const noexcept { return my_fence_; }
    // Check if a slot's fence is still valid (not reclaimed)
    bool is_fence_valid(uint64_t fence) const noexcept;
    // Update minimum active fence (called during reclamation)
    void update_min_active(uint64_t fence);
private:
    FenceHeader* header_;
    uint64_t my_fence_{0};
};
} // namespace zcmb
```
### OrphanedSlotHeader (Per-Slot Metadata)
```cpp
// File: 01_include/zcmb/recovery/03_orphan_reclaimer.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
namespace zcmb {
// Extended slot header for orphan detection
// Prepended to each message slot in the ring buffer
struct alignas(16) OrphanedSlotHeader {
    std::atomic<uint64_t> fence{0};           // Fence value when claimed
    std::atomic<uint64_t> claim_timestamp_ns{0}; // When slot was claimed
    std::atomic<uint32_t> owner_slot{0};       // Index in ProcessRegistry
    std::atomic<uint32_t> ref_count{0};        // Remaining readers
    uint32_t message_size{0};                  // Actual message bytes
    uint32_t sequence{0};                      // Message sequence number
    uint8_t padding[24];
};
static_assert(sizeof(OrphanedSlotHeader) == 64, "OrphanedSlotHeader must be 64 bytes");
class OrphanReclaimer {
public:
    struct Config {
        std::chrono::seconds orphan_timeout{10};
        std::chrono::seconds scan_interval{1};
        size_t max_slots;
    };
    OrphanReclaimer(OrphanedSlotHeader* slots, size_t num_slots,
                    ProcessRegistry* registry, FenceManager* fences,
                    const Config& config);
    ~OrphanReclaimer();
    // Start background reclamation thread
    void start();
    // Stop background thread
    void stop();
    // Manual scan (for testing)
    void scan_once();
    // Statistics
    size_t slots_reclaimed() const noexcept { return slots_reclaimed_; }
    size_t false_positives() const noexcept { return false_positives_; }
private:
    void reclamation_loop();
    bool try_reclaim_slot(size_t slot_idx);
    bool is_slot_orphaned(size_t slot_idx) const noexcept;
    OrphanedSlotHeader* slots_;
    size_t num_slots_;
    ProcessRegistry* registry_;
    FenceManager* fences_;
    Config config_;
    std::thread reclaimer_thread_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> slots_reclaimed_{0};
    std::atomic<size_t> false_positives_{0};
};
} // namespace zcmb
```
### WriteAheadLogHeader
```cpp
// File: 01_include/zcmb/recovery/05_write_ahead_log.hpp
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
namespace zcmb {
// WAL file header (at offset 0 of log file)
struct WalFileHeader {
    uint32_t magic;              // 0xDEADBEEF
    uint32_t version;            // Format version
    uint64_t create_time_ns;     // When log was created
    uint64_t last_sync_ns;       // Last fsync timestamp
    uint64_t next_sequence;      // Next sequence number to assign
    uint64_t truncated_sequence; // All entries before this are truncated
    uint32_t checksum;           // CRC32 of header
    uint8_t padding[28];
};
static_assert(sizeof(WalFileHeader) == 64, "WalFileHeader must be 64 bytes");
// Individual log entry header
struct alignas(8) WalEntryHeader {
    uint32_t magic;              // 0xFEEDFACE
    uint32_t entry_size;         // Size of data following header
    uint64_t sequence;           // Monotonic sequence number
    uint64_t timestamp_ns;       // When entry was written
    uint32_t checksum;           // CRC32 of data
    uint32_t flags;              // Entry type flags
};
static_assert(sizeof(WalEntryHeader) == 32, "WalEntryHeader must be 32 bytes");
// Entry flags
enum class WalEntryFlags : uint32_t {
    NONE = 0,
    MESSAGE = 1 << 0,
    CHECKPOINT_MARKER = 1 << 1,
    TRUNCATION_POINT = 1 << 2,
    END_OF_STREAM = 1 << 3
};
class WriteAheadLog {
public:
    using ReplayHandler = std::function<void(uint64_t sequence,
                                              const uint8_t* data,
                                              uint32_t size,
                                              uint32_t flags)>;
    explicit WriteAheadLog(const std::string& path);
    ~WriteAheadLog();
    // Append entry to log
    uint64_t append(const void* data, uint32_t size,
                    WalEntryFlags flags = WalEntryFlags::MESSAGE);
    // Force sync to disk
    void sync();
    // Truncate entries up to (but not including) sequence
    void truncate(uint64_t sequence);
    // Replay entries from sequence onwards
    void replay(uint64_t from_sequence, ReplayHandler handler);
    // Replay all entries
    void replay_all(ReplayHandler handler);
    // Get current sequence
    uint64_t next_sequence() const noexcept { return next_sequence_; }
    // Statistics
    size_t total_entries() const noexcept { return total_entries_; }
    size_t total_bytes() const noexcept { return total_bytes_; }
private:
    bool validate_entry_header(const WalEntryHeader& header);
    uint32_t compute_checksum(const void* data, size_t size);
    void recover_sequence();
    int fd_;
    std::string path_;
    uint64_t next_sequence_{0};
    uint64_t truncated_sequence_{0};
    size_t total_entries_{0};
    size_t total_bytes_{0};
};
} // namespace zcmb
```
**WAL File Layout**:
```
Write-Ahead Log File Layout:
┌─────────────────────────────────────────────────────────────────────┐
│ Offset 0x000: WalFileHeader (64 bytes)                              │
│   ├─ 0x00: magic = 0xDEADBEEF                                       │
│   ├─ 0x04: version = 1                                              │
│   ├─ 0x08: create_time_ns                                           │
│   ├─ 0x10: last_sync_ns                                             │
│   ├─ 0x18: next_sequence                                            │
│   ├─ 0x20: truncated_sequence                                       │
│   └─ 0x24: checksum (CRC32 of header)                               │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x040: WalEntryHeader[0] (32 bytes)                          │
│   ├─ 0x00: magic = 0xFEEDFACE                                       │
│   ├─ 0x04: entry_size                                               │
│   ├─ 0x08: sequence = 0                                             │
│   ├─ 0x10: timestamp_ns                                             │
│   ├─ 0x18: checksum (CRC32 of data)                                 │
│   └─ 0x1C: flags                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x060: Entry[0] data (entry_size bytes)                      │
├─────────────────────────────────────────────────────────────────────┤
│ Offset 0x060 + entry_size: WalEntryHeader[1]                        │
├─────────────────────────────────────────────────────────────────────┤
│ ...                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```
### CheckpointHeader
```cpp
// File: 01_include/zcmb/recovery/06_checkpoint_manager.hpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
namespace zcmb {
// Checkpoint file header
struct CheckpointFileHeader {
    uint32_t magic;              // 0xFEEDFACE
    uint32_t version;            // Format version
    uint64_t wal_sequence;       // WAL sequence at checkpoint time
    uint64_t timestamp_ns;       // When checkpoint was taken
    uint32_t ring_state_size;    // Size of serialized ring buffer state
    uint32_t num_subscribers;    // Number of subscriber states
    uint32_t checksum;           // CRC32 of everything after header
    uint8_t padding[28];
};
static_assert(sizeof(CheckpointFileHeader) == 64, "CheckpointFileHeader must be 64 bytes");
// Per-subscriber state in checkpoint
struct SubscriberCheckpointState {
    uint64_t subscriber_id;
    uint64_t current_offset;
    uint64_t messages_received;
    uint32_t subscription_count;
    uint8_t padding[36];
};
static_assert(sizeof(SubscriberCheckpointState) == 64, "State must be 64 bytes");
class CheckpointManager {
public:
    struct Config {
        std::string checkpoint_dir;
        std::chrono::seconds checkpoint_interval{10};
        size_t max_checkpoints{5};
    };
    explicit CheckpointManager(const Config& config, WriteAheadLog& wal);
    ~CheckpointManager() = default;
    // Save checkpoint
    bool save_checkpoint(const std::string& name,
                         const void* ring_state,
                         size_t ring_state_size,
                         const std::vector<SubscriberCheckpointState>& subscriber_states);
    // Load latest checkpoint
    struct CheckpointData {
        uint64_t wal_sequence;
        std::vector<uint8_t> ring_state;
        std::vector<SubscriberCheckpointState> subscriber_states;
    };
    bool load_latest_checkpoint(const std::string& name, CheckpointData& out);
    // List available checkpoints
    std::vector<std::string> list_checkpoints(const std::string& name) const;
    // Delete old checkpoints
    void prune_old_checkpoints(const std::string& name);
private:
    std::string checkpoint_path(const std::string& name, uint64_t timestamp) const;
    Config config_;
    WriteAheadLog& wal_;
};
} // namespace zcmb
```
### RecoveryJournal
```cpp
// File: 01_include/zcmb/recovery/07_recovery_journal.hpp
#pragma once
#include <cstdint>
#include <string>
#include <atomic>
namespace zcmb {
// Tracks recovery progress for crash-during-recovery handling
enum class RecoveryPhase : uint32_t {
    NOT_STARTED = 0,
    LOADING_CHECKPOINT = 1,
    RESTORING_RING_STATE = 2,
    REPLAYING_WAL = 3,
    RESTORING_SUBSCRIBERS = 4,
    VERIFYING_STATE = 5,
    COMPLETED = 6,
    FAILED = 7
};
struct RecoveryJournalEntry {
    uint32_t magic;              // 0xCAFECAFE
    RecoveryPhase phase;
    uint64_t checkpoint_sequence; // WAL sequence in checkpoint
    uint64_t wal_replay_position; // Current WAL replay position
    uint64_t timestamp_ns;        // When this entry was written
    uint32_t checksum;            // CRC32
    uint8_t padding[36];
};
static_assert(sizeof(RecoveryJournalEntry) == 64, "Journal entry must be 64 bytes");
class RecoveryJournal {
public:
    explicit RecoveryJournal(const std::string& journal_path);
    ~RecoveryJournal();
    // Begin a recovery phase (persists to disk)
    void begin_phase(RecoveryPhase phase);
    // Update WAL replay position (frequently updated)
    void update_wal_position(uint64_t position);
    // Mark recovery complete
    void mark_completed();
    // Mark recovery failed
    void mark_failed();
    // Read last journal entry
    RecoveryJournalEntry read_last() const;
    // Clear journal
    void clear();
    // Check if recovery was interrupted
    bool was_interrupted() const;
private:
    void write_entry(const RecoveryJournalEntry& entry);
    uint32_t compute_checksum(const RecoveryJournalEntry& entry) const;
    std::string journal_path_;
    int fd_;
};
} // namespace zcmb
```
### SignalHandler
```cpp
// File: 01_include/zcmb/recovery/08_signal_handler.hpp
#pragma once
#include <atomic>
#include <functional>
#include <csignal>
namespace zcmb {
class SignalHandler {
public:
    using ShutdownCallback = std::function<void(bool graceful)>;
    // Setup signal handlers (call once at startup)
    static void setup();
    // Set callback for shutdown
    static void set_shutdown_callback(ShutdownCallback callback);
    // Check if shutdown requested
    static bool shutdown_requested() noexcept {
        return shutdown_requested_.load(std::memory_order_acquire);
    }
    // Check if forced shutdown
    static bool force_shutdown() noexcept {
        return force_shutdown_.load(std::memory_order_acquire);
    }
    // Reset state (for testing)
    static void reset() noexcept {
        shutdown_requested_.store(false, std::memory_order_release);
        force_shutdown_.store(false, std::memory_order_release);
    }
private:
    static void handle_signal(int sig);
    static std::atomic<bool> shutdown_requested_;
    static std::atomic<bool> force_shutdown_;
    static ShutdownCallback callback_;
};
} // namespace zcmb
```
### RecoveryCoordinator (Unified API)
```cpp
// File: 01_include/zcmb/recovery/09_recovery_coordinator.hpp
#pragma once
#include "01_heartbeat_monitor.hpp"
#include "02_process_identity.hpp"
#include "03_orphan_reclaimer.hpp"
#include "04_fence_manager.hpp"
#include "05_write_ahead_log.hpp"
#include "06_checkpoint_manager.hpp"
#include "07_recovery_journal.hpp"
#include "08_signal_handler.hpp"
#include "10_recovery_config.hpp"
#include <memory>
#include <functional>
namespace zcmb {
class RecoveryCoordinator {
public:
    using RingStateSerializer = std::function<std::vector<uint8_t>()>;
    using RingStateDeserializer = std::function<bool(const uint8_t*, size_t)>;
    using MessageHandler = std::function<void(const uint8_t*, uint32_t)>;
    explicit RecoveryCoordinator(const RecoveryConfig& config);
    ~RecoveryCoordinator();
    // === Lifecycle ===
    // Initialize all components, run recovery if needed
    void initialize(void* shared_memory);
    // Start background threads (heartbeat, reclaimer, checkpoint)
    void start();
    // Stop background threads
    void stop();
    // === Process Registration ===
    uint32_t register_producer();
    uint32_t register_consumer();
    void unregister();
    // === Durability ===
    // Append message to WAL (if durability enabled)
    uint64_t log_message(const void* data, uint32_t size);
    // Wait for log to be durable
    void wait_durable(uint64_t sequence);
    // === Checkpointing ===
    // Take checkpoint now
    bool checkpoint(const std::string& name);
    // Set serializers for checkpoint
    void set_ring_state_serializer(RingStateSerializer serializer);
    void set_ring_state_deserializer(RingStateDeserializer deserializer);
    // === Recovery ===
    // Run full recovery (checkpoint + WAL replay)
    bool recover(MessageHandler handler);
    // Check if recovery is needed
    bool needs_recovery() const;
    // === Status ===
    bool is_alive(uint32_t process_slot) const;
    uint64_t my_fence() const noexcept;
    size_t orphaned_slots_reclaimed() const noexcept;
    uint64_t last_checkpoint_time() const noexcept;
private:
    void run_checkpoint_loop();
    bool recover_from_checkpoint(const std::string& name, MessageHandler handler);
    bool recover_from_wal(uint64_t from_sequence, MessageHandler handler);
    RecoveryConfig config_;
    std::unique_ptr<ProcessRegistry> process_registry_;
    std::unique_ptr<FenceManager> fence_manager_;
    std::unique_ptr<HeartbeatMonitor> heartbeat_monitor_;
    std::unique_ptr<OrphanReclaimer> orphan_reclaimer_;
    std::unique_ptr<WriteAheadLog> wal_;
    std::unique_ptr<CheckpointManager> checkpoint_manager_;
    std::unique_ptr<RecoveryJournal> recovery_journal_;
    RingStateSerializer ring_serializer_;
    RingStateDeserializer ring_deserializer_;
    std::thread checkpoint_thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> last_checkpoint_ns_{0};
    void* shared_memory_;
};
} // namespace zcmb
```
---
## Interface Contracts
### ProcessRegistry::register_process
```cpp
uint32_t ProcessRegistry::register_process();
```
**Preconditions**:
- `identities_` points to valid shared memory
- Process not already registered
**Postconditions**:
- Returns slot index (0 to MAX_PROCESSES-1)
- `identities_[slot].pid == getpid()`
- `identities_[slot].generation` incremented from previous value
- `identities_[slot].state == ProcessState::STARTING`
- `identities_[slot].last_heartbeat_ns == current_time`
**Error Conditions**:
| Error | Condition | Recovery |
|-------|-----------|----------|
| REGISTRY_FULL | All slots in use | Throw `std::runtime_error("PROCESS_REGISTRY_FULL")` |
| ALREADY_REGISTERED | `my_slot_ != 0` | Return existing slot |
**Edge Cases**:
- PID reuse: Old process crashed, new unrelated process has same PID. Generation counter distinguishes them.
- Slot previously used: Generation increments, old heartbeat timestamps ignored.
### ProcessRegistry::is_alive
```cpp
bool ProcessRegistry::is_alive(uint32_t slot) const noexcept;
```
**Preconditions**:
- `slot < MAX_PROCESSES`
**Postconditions**:
- Returns `true` if: PID matches AND generation matches AND heartbeat fresh AND state != DEAD
- Returns `false` otherwise
**Algorithm**:
```
1. IF slot >= MAX_PROCESSES: RETURN false
2. entry = identities_[slot]
3. pid = entry.pid.load(acquire)
4. IF pid == 0: RETURN false  // Never used
5. // Check if process actually exists
   IF kill(pid, 0) == -1 AND errno == ESRCH:
     RETURN false  // Process doesn't exist
6. // Check generation (detect restart)
   generation = entry.generation.load(acquire)
   IF entry was previously registered with different generation:
     RETURN false  // Stale registration
7. // Check heartbeat freshness
   last_hb = entry.last_heartbeat_ns.load(acquire)
   now = current_time_ns()
   IF (now - last_hb) > heartbeat_timeout_ns:
     RETURN false  // Timed out
8. // Check state
   state = entry.state.load(acquire)
   IF state == ProcessState::DEAD:
     RETURN false
9. RETURN true
```
### OrphanReclaimer::try_reclaim_slot
```cpp
bool OrphanReclaimer::try_reclaim_slot(size_t slot_idx);
```
**Preconditions**:
- `slot_idx < num_slots_`
- Background thread may call this concurrently
**Postconditions**:
- Returns `true` if slot was reclaimed (ref_count set to 0)
- Returns `false` if slot not orphaned or race lost
- No live process's data is corrupted
**Algorithm**:
```
1. slot = slots_[slot_idx]
2. ref = slot.ref_count.load(acquire)
3. IF ref == 0: RETURN false  // Already free
4. // Check if claim is stale
   claim_time = slot.claim_timestamp_ns.load(acquire)
   now = current_time_ns()
   IF (now - claim_time) < orphan_timeout_ns:
     RETURN false  // Claim is recent, give benefit of doubt
5. // Check if owner is alive
   owner_slot = slot.owner_slot.load(acquire)
   IF registry_->is_alive(owner_slot):
     // Owner alive but slow - check fence
     slot_fence = slot.fence.load(acquire)
     owner_fence = fence_manager_->get_fence_for_slot(owner_slot)
     IF slot_fence >= owner_fence:
       RETURN false  // Valid claim from current owner
6. // Owner is dead or fence is stale - try to reclaim
   // Use CAS to avoid racing with reviving owner
   expected = ref
   IF slot.ref_count.compare_exchange_strong(expected, 0, acq_rel, acquire):
     slots_reclaimed_.fetch_add(1, relaxed)
     LOG("Reclaimed slot %zu from dead owner %u", slot_idx, owner_slot)
     RETURN true
7. // CAS failed - someone else reclaimed or owner revived
   false_positives_.fetch_add(1, relaxed)
   RETURN false
```
### WriteAheadLog::append
```cpp
uint64_t WriteAheadLog::append(const void* data, uint32_t size, WalEntryFlags flags);
```
**Preconditions**:
- WAL file open
- `data != nullptr` if `size > 0`
- `size < wal_max_size` (configurable limit)
**Postconditions**:
- Entry written to log
- Returns assigned sequence number
- If `wal_sync_on_write == true`, data is durable on disk
- `next_sequence_` incremented
**Error Conditions**:
| Error | Condition | Recovery |
|-------|-----------|----------|
| DISK_FULL | `write()` returns ENOSPC | Throw `std::runtime_error("WAL_DISK_FULL")` |
| IO_ERROR | `write()` returns error | Throw `std::runtime_error("WAL_IO_ERROR")` |
| ENTRY_TOO_LARGE | `size > max_entry_size` | Throw `std::invalid_argument("WAL_ENTRY_TOO_LARGE")` |
### WriteAheadLog::replay
```cpp
void WriteAheadLog::replay(uint64_t from_sequence, ReplayHandler handler);
```
**Preconditions**:
- WAL file open
- `handler` is valid callable
**Postconditions**:
- All entries with `sequence >= from_sequence` passed to handler
- Corrupted entries skipped (checksum mismatch logged)
- Stops at first incomplete entry (partial write at crash)
**Error Handling in Handler**:
- If handler throws, replay stops and exception propagates
- Handler receives sequence, data pointer, size, and flags
### CheckpointManager::save_checkpoint
```cpp
bool CheckpointManager::save_checkpoint(const std::string& name,
                                         const void* ring_state,
                                         size_t ring_state_size,
                                         const std::vector<SubscriberCheckpointState>& subscriber_states);
```
**Preconditions**:
- Checkpoint directory exists and is writable
- `ring_state != nullptr` if `ring_state_size > 0`
**Postconditions**:
- Returns `true` on success
- Checkpoint file written to temp path, then atomically renamed
- Old checkpoints pruned to `max_checkpoints`
- WAL sequence recorded for replay after checkpoint load
**Atomicity**:
```
1. temp_path = checkpoint_dir / name / timestamp.tmp
2. Write header + data to temp_path
3. fsync(temp_path)
4. final_path = checkpoint_dir / name / timestamp.ckpt
5. rename(temp_path, final_path)  // Atomic on POSIX
```
### RecoveryCoordinator::recover
```cpp
bool RecoveryCoordinator::recover(MessageHandler handler);
```
**Preconditions**:
- Components initialized
- `handler` is valid callable
**Postconditions**:
- Returns `true` if recovery succeeded
- Ring buffer state restored
- WAL entries after checkpoint replayed
- All messages passed to handler (may include duplicates)
**Recovery Sequence**:
```
1. journal.begin_phase(LOADING_CHECKPOINT)
2. IF checkpoint exists:
     load_checkpoint()
     journal.begin_phase(RESTORING_RING_STATE)
     ring_deserializer_(checkpoint.ring_state)
3. journal.begin_phase(REPLAYING_WAL)
4. wal_.replay(checkpoint.wal_sequence, handler)
5. journal.begin_phase(VERIFYING_STATE)
6. verify_buffer_consistency()
7. journal.mark_completed()
```
---
## Algorithm Specification
### Algorithm: Heartbeat Monitor Loop
**Purpose**: Continuously update heartbeat timestamp, detect dead processes.
**Input**: ProcessRegistry, timeout configuration
**Output**: Heartbeat updates, dead process detection
```
ALGORITHM heartbeat_loop():
  WHILE running:
    // Update own heartbeat
    registry_->update_heartbeat(my_slot_)
    // Check for dead processes (optional, can be separate thread)
    FOR slot = 0 TO MAX_PROCESSES - 1:
      IF slot == my_slot_: CONTINUE
      IF NOT registry_->is_alive(slot):
        // Mark as dead for faster reclamation
        registry_->mark_dead(slot)
    sleep(heartbeat_interval)
```
### Algorithm: Orphaned Slot Detection and Reclamation
**Purpose**: Reclaim slots claimed by crashed processes.
**Input**: OrphanedSlotHeader array, ProcessRegistry, FenceManager
**Output**: Reclaimed slots (ref_count = 0)
```
ALGORITHM reclamation_scan():
  FOR slot_idx = 0 TO num_slots - 1:
    slot = slots_[slot_idx]
    ref = slot.ref_count.load(acquire)
    // Skip free slots
    IF ref == 0: CONTINUE
    // Check claim age
    claim_time = slot.claim_timestamp_ns.load(acquire)
    age = now_ns() - claim_time
    IF age < orphan_timeout_ns: CONTINUE  // Recent claim
    // Check owner liveness
    owner_slot = slot.owner_slot.load(acquire)
    owner_alive = registry_->is_alive(owner_slot)
    IF owner_alive:
      // Owner alive - check if fence is current
      slot_fence = slot.fence.load(acquire)
      owner_fence = fence_manager_->get_fence_for_slot(owner_slot)
      IF slot_fence == owner_fence: CONTINUE  // Valid current claim
    // Owner dead or fence stale - reclaim
    expected_ref = ref
    IF slot.ref_count.compare_exchange_strong(expected_ref, 0, acq_rel, acquire):
      LOG("Reclaimed slot %zu, owner=%u, fence=%lu", slot_idx, owner_slot, slot_fence)
      fence_manager_->update_min_active(slot_fence + 1)
```
### Algorithm: WAL Replay with Checksum Validation
**Purpose**: Replay log entries, skipping corrupted data.
**Input**: WAL file, starting sequence
**Output**: Entries passed to handler
```
ALGORITHM wal_replay(from_sequence, handler):
  fd = open(wal_path, O_RDONLY)
  // Skip file header
  lseek(fd, sizeof(WalFileHeader), SEEK_SET)
  offset = sizeof(WalFileHeader)
  WHILE true:
    // Read entry header
    n = read(fd, &entry_header, sizeof(entry_header))
    IF n == 0: BREAK  // EOF
    IF n != sizeof(entry_header): BREAK  // Truncated
    // Validate header
    IF entry_header.magic != 0xFEEDFACE:
      LOG("Invalid entry magic at offset %lu", offset)
      BREAK
    // Skip entries before from_sequence
    IF entry_header.sequence < from_sequence:
      lseek(fd, entry_header.entry_size, SEEK_CUR)
      offset += sizeof(entry_header) + entry_header.entry_size
      CONTINUE
    // Read entry data
    data = allocate(entry_header.entry_size)
    n = read(fd, data, entry_header.entry_size)
    IF n != entry_header.entry_size:
      LOG("Truncated entry at offset %lu", offset)
      BREAK
    // Validate checksum
    computed_crc = crc32(data, entry_header.entry_size)
    IF computed_crc != entry_header.checksum:
      LOG("Checksum mismatch at sequence %lu", entry_header.sequence)
      free(data)
      CONTINUE  // Skip corrupted entry
    // Call handler
    handler(entry_header.sequence, data, entry_header.entry_size, entry_header.flags)
    free(data)
    offset += sizeof(entry_header) + entry_header.entry_size
  close(fd)
```
### Algorithm: Crash-During-Recovery Handling
**Purpose**: Resume interrupted recovery without data loss.
**Input**: Recovery journal, checkpoint, WAL
**Output**: Complete recovery or clear error
```
ALGORITHM recovery_with_journal():
  journal_entry = recovery_journal.read_last()
  SWITCH journal_entry.phase:
    case NOT_STARTED:
    case COMPLETED:
      // Normal startup - run full recovery
      full_recovery()
      RETURN
    case LOADING_CHECKPOINT:
      // Crashed during load - retry
      full_recovery()
      RETURN
    case RESTORING_RING_STATE:
      // Crashed during restore - retry from checkpoint
      full_recovery()
      RETURN
    case REPLAYING_WAL:
      // Crashed during replay - resume
      checkpoint = load_checkpoint()
      restore_ring_state(checkpoint)
      // Resume from last recorded position
      wal.replay(journal_entry.wal_replay_position, handler)
      journal.mark_completed()
      RETURN
    case VERIFYING_STATE:
      // Crashed during verify - re-verify
      verify_state()
      journal.mark_completed()
      RETURN
    case FAILED:
      // Previous recovery failed - manual intervention
      THROW "Previous recovery failed, manual intervention required"
```
### Algorithm: At-Least-Once Delivery with Deduplication Support
**Purpose**: Ensure no message loss while allowing duplicates.
**Input**: WAL entries, consumer's processed message set
**Output**: Messages delivered, duplicates skipped by consumer
```
ALGORITHM at_least_once_delivery():
  // Producer side
  ON publish(message):
    sequence = wal.append(message.data, message.size)
    message.header.sequence = sequence
    ring_buffer.produce(message)
  // Consumer side
  ON receive(message):
    IF message.sequence IN processed_set:
      LOG("Skipping duplicate message %lu", message.sequence)
      RETURN
    process_message(message)
    processed_set.add(message.sequence)
    // Evict old entries if set too large
    IF processed_set.size() > MAX_PROCESSED:
      evict_oldest(processed_set, MAX_PROCESSED / 2)
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible |
|-------|-------------|----------|--------------|
| PROCESS_CRASH | Heartbeat timeout + PID check | Orphan reclaimer cleans slots | Monitoring only |
| PID_REUSE | Generation counter mismatch | Treat as new process | No |
| SLOT_ORPHANED | `ref_count > 0` and owner dead | Reclaimer sets `ref_count = 0` | No |
| WAL_WRITE_FAILED | `write()` returns error | Throw exception, caller handles | Yes (publish fails) |
| WAL_CORRUPTED | Checksum mismatch on replay | Skip entry, log warning, continue | Monitoring only |
| WAL_DISK_FULL | `write()` returns ENOSPC | Throw exception, apply backpressure | Yes (publish fails) |
| CHECKPOINT_SAVE_FAILED | File write/rename error | Log error, continue without checkpoint | No |
| CHECKPOINT_LOAD_FAILED | Invalid magic/checksum | Fall back to full WAL replay | No |
| RECOVERY_CRASH | Journal shows incomplete phase | Resume from journal position | No |
| RECOVERY_FAILED | Verification fails | Log error, require manual intervention | Yes (startup fails) |
| FENCE_EXHAUSTED | 64-bit fence wraps (impossible) | Log error, reset fence space | Yes (system error) |
| FSYNC_FAILED | `fsync()` returns error | Durability guarantee broken, log critical | Yes (data at risk) |
| SIGNAL_HANDLER_ERROR | `sigaction()` fails | Log error, continue without signal handling | No |
---
## Implementation Sequence with Checkpoints
### Phase 1: Crash Detection (2-3 hours)
**Files**: `01_heartbeat_monitor.hpp`, `02_process_identity.hpp`, `01_heartbeat_monitor.cpp`, `02_process_identity.cpp`
**Steps**:
1. Define `ProcessIdentity` structure with exact byte layout (128 bytes)
2. Implement `ProcessRegistry::register_process()` with generation counter
3. Implement `ProcessRegistry::is_alive()` with PID check + heartbeat timeout
4. Implement `ProcessRegistry::update_heartbeat()`
5. Implement `HeartbeatMonitor` background thread
6. Add unit tests for timeout detection
**Checkpoint**:
```bash
./03_tests/01_heartbeat_test
./03_tests/02_process_identity_test
# Expected:
# [PASS] Register process with unique slot
# [PASS] Generation counter increments on restart
# [PASS] Heartbeat update refreshes timestamp
# [PASS] is_alive returns false after timeout
# [PASS] PID reuse detected via generation
```
**At this point**: Can detect crashed processes via heartbeat timeout.
### Phase 2: Orphaned Slot Recovery (2-3 hours)
**Files**: `03_orphan_reclaimer.hpp`, `04_fence_manager.hpp`, `03_orphan_reclaimer.cpp`, `04_fence_manager.cpp`
**Steps**:
1. Define `OrphanedSlotHeader` (64 bytes)
2. Implement `FenceManager::allocate_fence()` with atomic increment
3. Implement `FenceManager::is_fence_valid()` comparison
4. Implement `OrphanReclaimer::try_reclaim_slot()` with CAS
5. Implement background reclamation thread
6. Add fence validation in slot claiming
**Checkpoint**:
```bash
./03_tests/03_orphan_reclaim_test
./03_tests/04_fence_test
# Expected:
# [PASS] Allocate unique fence values
# [PASS] Fence validation detects stale claims
# [PASS] Orphan detection after timeout
# [PASS] CAS-based reclamation
# [PASS] No reclamation of live process slots
```
**At this point**: Can reclaim slots from crashed processes safely.
### Phase 3: Write-Ahead Logging (3-4 hours)
**Files**: `05_write_ahead_log.hpp`, `05_write_ahead_log.cpp`
**Steps**:
1. Define `WalFileHeader` (64 bytes) and `WalEntryHeader` (32 bytes)
2. Implement `WriteAheadLog::append()` with checksum
3. Implement `WriteAheadLog::sync()` with fsync
4. Implement `WriteAheadLog::truncate()` with log compaction
5. Implement `WriteAheadLog::replay()` with checksum validation
6. Add batch sync mode for performance
**Checkpoint**:
```bash
./03_tests/05_wal_basic_test
./03_tests/06_wal_checksum_test
./03_tests/07_wal_replay_test
# Expected:
# [PASS] Append entries with sequence numbers
# [PASS] Sync flushes to disk
# [PASS] Truncate removes old entries
# [PASS] Checksum detects corruption
# [PASS] Replay delivers all entries
# [PASS] Replay skips corrupted entries
```
**At this point**: Can persist messages to disk and replay on recovery.
### Phase 4: Checkpoint/Restart (3-4 hours)
**Files**: `06_checkpoint_manager.hpp`, `06_checkpoint_manager.cpp`
**Steps**:
1. Define `CheckpointFileHeader` (64 bytes)
2. Implement `CheckpointManager::save_checkpoint()` with temp file + rename
3. Implement `CheckpointManager::load_latest_checkpoint()`
4. Implement checkpoint pruning (keep N most recent)
5. Add periodic checkpoint thread
6. Integrate with WAL for sequence tracking
**Checkpoint**:
```bash
./03_tests/08_checkpoint_test
# Expected:
# [PASS] Save checkpoint atomically
# [PASS] Load checkpoint restores state
# [PASS] Temp file renamed atomically
# [PASS] Old checkpoints pruned
# [PASS] Checkpoint includes WAL sequence
```
**At this point**: Can snapshot system state for fast recovery.
### Phase 5: Recovery Journal & Integration (2-3 hours)
**Files**: `07_recovery_journal.hpp`, `08_signal_handler.hpp`, `09_recovery_coordinator.hpp`, corresponding .cpp files
**Steps**:
1. Define `RecoveryJournalEntry` (64 bytes)
2. Implement `RecoveryJournal` phase tracking
3. Implement `SignalHandler` for SIGTERM/SIGINT
4. Implement `RecoveryCoordinator` integrating all components
5. Implement crash-during-recovery handling
6. Add systemd readiness notification
**Checkpoint**:
```bash
./03_tests/09_recovery_journal_test
./03_tests/10_signal_test
./03_tests/11_full_recovery_test
./03_tests/12_cross_process_test
# Expected:
# [PASS] Journal tracks recovery phases
# [PASS] Resume from interrupted recovery
# [PASS] Signal handler triggers graceful shutdown
# [PASS] Full recovery restores state
# [PASS] Cross-process crash simulation
```
**At this point**: Complete crash recovery system with durability.
---
## Test Specification
### Test: Heartbeat Timeout Detection
```cpp
TEST(HeartbeatMonitor, TimeoutDetection) {
    ProcessRegistry registry;
    registry.initialize(shared_memory);
    uint32_t slot = registry.register_process();
    // Update heartbeat
    registry.update_heartbeat(slot);
    EXPECT_TRUE(registry.is_alive(slot));
    // Simulate time passing without heartbeat
    auto* identity = registry.get_identity(slot);
    identity->last_heartbeat_ns.store(
        now_ns() - std::chrono::seconds(10).count(),
        std::memory_order_release);
    // Should detect as dead
    EXPECT_FALSE(registry.is_alive(slot));
}
```
### Test: PID Reuse Detection
```cpp
TEST(ProcessIdentity, PIDReuseDetection) {
    ProcessRegistry registry;
    registry.initialize(shared_memory);
    // Register first process
    uint32_t slot = registry.register_process();
    uint32_t gen1 = registry.get_identity(slot)->generation.load();
    // Simulate process crash and restart with same PID
    // (In real test, fork a child that exits without cleanup)
    registry.get_identity(slot)->state.store(
        static_cast<uint32_t>(ProcessState::DEAD),
        std::memory_order_release);
    // Re-register (simulating restart)
    uint32_t new_slot = registry.register_process();
    EXPECT_EQ(new_slot, slot);  // Same slot reused
    uint32_t gen2 = registry.get_identity(slot)->generation.load();
    EXPECT_GT(gen2, gen1);  // Generation incremented
}
```
### Test: Orphaned Slot Reclamation
```cpp
TEST(OrphanReclaimer, ReclaimsOrphanedSlots) {
    OrphanedSlotHeader slots[4];
    ProcessRegistry registry;
    FenceManager fences;
    // Initialize
    for (size_t i = 0; i < 4; ++i) {
        slots[i].ref_count.store(0);
        slots[i].fence.store(0);
    }
    // Simulate producer claiming slot
    uint32_t producer_slot = registry.register_process();
    uint64_t fence = fences.allocate_fence();
    slots[0].ref_count.store(2);  // Claimed for 2 subscribers
    slots[0].fence.store(fence);
    slots[0].owner_slot.store(producer_slot);
    slots[0].claim_timestamp_ns.store(now_ns());
    // Simulate producer crash (mark as dead)
    registry.get_identity(producer_slot)->state.store(
        static_cast<uint32_t>(ProcessState::DEAD));
    // Age the claim
    slots[0].claim_timestamp_ns.store(
        now_ns() - std::chrono::seconds(20).count());
    // Run reclaimer
    OrphanReclaimer reclaimer(slots, 4, &registry, &fences, config);
    reclaimer.scan_once();
    // Slot should be reclaimed
    EXPECT_EQ(slots[0].ref_count.load(), 0);
    EXPECT_EQ(reclaimer.slots_reclaimed(), 1);
}
```
### Test: WAL Checksum Corruption Detection
```cpp
TEST(WriteAheadLog, DetectsCorruption) {
    WriteAheadLog wal("/tmp/test.wal");
    // Write entries
    for (int i = 0; i < 10; ++i) {
        uint8_t data[64] = {static_cast<uint8_t>(i)};
        wal.append(data, sizeof(data));
    }
    wal.sync();
    // Corrupt middle entry
    int fd = open("/tmp/test.wal", O_RDWR);
    lseek(fd, 64 + 32 + 64 * 5 + 16, SEEK_SET);  // Middle of entry 5
    uint8_t garbage = 0xFF;
    write(fd, &garbage, 1);
    close(fd);
    // Replay should skip corrupted entry
    std::vector<int> received;
    wal.replay(0, [&](uint64_t seq, const uint8_t* data, uint32_t size, uint32_t flags) {
        received.push_back(data[0]);
    });
    // Should have 9 entries (one corrupted skipped)
    EXPECT_EQ(received.size(), 9);
}
```
### Test: Full Recovery End-to-End
```cpp
TEST(RecoveryCoordinator, FullRecovery) {
    RecoveryConfig config;
    config.wal_path = "/tmp/test_recovery.wal";
    config.checkpoint_dir = "/tmp/test_recovery_ckpt";
    RecoveryCoordinator coordinator(config);
    coordinator.initialize(nullptr);
    // Publish messages
    for (int i = 0; i < 100; ++i) {
        uint8_t data[64] = {static_cast<uint8_t>(i)};
        coordinator.log_message(data, sizeof(data));
    }
    // Take checkpoint
    coordinator.checkpoint("test");
    // Publish more messages (after checkpoint)
    for (int i = 100; i < 150; ++i) {
        uint8_t data[64] = {static_cast<uint8_t>(i)};
        coordinator.log_message(data, sizeof(data));
    }
    // Simulate crash and restart
    coordinator.stop();
    RecoveryCoordinator coordinator2(config);
    // Recover
    std::vector<uint8_t> recovered;
    coordinator2.recover([&](const uint8_t* data, uint32_t size) {
        recovered.push_back(data[0]);
    });
    // Should have all messages (checkpoint + WAL replay)
    // Some may be duplicates (at-least-once)
    EXPECT_GE(recovered.size(), 150);
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Heartbeat update | < 100 ns | Atomic store + timestamp |
| `is_alive` check | < 1 μs | PID check + atomic loads |
| Orphan detection scan (1000 slots) | < 1 ms | Periodic background scan |
| Slot reclamation (single) | < 100 ns | CAS operation |
| WAL append (no sync) | 1-5 μs | `write()` syscall |
| WAL sync (fsync) | 500-2000 μs | SSD dependent |
| WAL batch sync (1000 entries) | 500-2000 μs | Amortized per entry |
| Checkpoint save (1 MB state) | 10-30 ms | Serialize + write + fsync |
| Checkpoint load (1 MB state) | 10-30 ms | Read + deserialize |
| WAL replay (100K entries) | 100-500 ms | 10-100 μs per entry |
| Full recovery (checkpoint + WAL) | < 100 ms | Checkpoint load + WAL replay |
**Benchmark Commands**:
```bash
# Checkpoint latency
./04_benchmarks/01_checkpoint_latency --state-size 1M --iterations 100
# WAL throughput
./04_benchmarks/02_wal_throughput --batch-size 1000 --duration 10s
# Full recovery
./04_benchmarks/03_recovery_time --checkpoint-size 1M --wal-entries 100K
```
---
## Concurrency Specification
### Thread Safety Model
| Component | Thread Safety | Access Pattern |
|-----------|--------------|----------------|
| `ProcessRegistry` | Lock-free | Atomic operations on ProcessIdentity |
| `HeartbeatMonitor` | Single thread | Background thread updates |
| `FenceManager` | Lock-free | Atomic increment for allocation |
| `OrphanReclaimer` | Single thread | Background scan thread |
| `WriteAheadLog` | Mutex-protected | Serialized appends, concurrent reads |
| `CheckpointManager` | Thread-safe | Mutex for save, lock-free for load |
| `RecoveryJournal` | Single writer | Only recovery thread writes |
| `SignalHandler` | Async-safe | Signal handler context |
### Lock Ordering
```
1. wal_mutex_ (WriteAheadLog)
2. checkpoint_mutex_ (CheckpointManager)
3. journal_mutex_ (RecoveryJournal)
NEVER acquire in reverse order.
```
### Memory Ordering for Cross-Process Safety
```
PROCESS A (claiming slot)            PROCESS B (reclaiming slot)
=========================            ==========================
fence = allocate_fence()             
store(slot.fence, fence)             
store(slot.owner, my_slot)           
store(slot.claim_time, now)          
atomic_thread_fence(release)         load(slot.claim_time, acquire)
                                     age = now - claim_time
                                     IF age > timeout:
store(slot.ref_count, N)               load(slot.ref_count)
   [publishes slot]                    load(slot.fence)
                                       IF fence < min_active:
                                         CAS(slot.ref_count, 0)
```
---
## Crash Recovery Procedure
### Normal Startup (No Crash)
```
1. Initialize shared memory
2. Register process in registry
3. Allocate fence value
4. Check journal: phase == NOT_STARTED or COMPLETED
5. Start heartbeat thread
6. Start reclaimer thread (if producer)
7. Start checkpoint thread (if durability enabled)
8. Notify systemd: READY=1
```
### Recovery After Crash
```
1. Initialize shared memory
2. Register process in registry
3. Check journal: phase != NOT_STARTED and != COMPLETED
4. Log "Recovery needed: phase = X"
5. Load checkpoint (if available)
6. Restore ring buffer state from checkpoint
7. Replay WAL from checkpoint sequence
8. Verify buffer consistency
9. Mark journal: phase = COMPLETED
10. Start normal operation
```
### Crash During Recovery
```
1. Check journal: phase indicates where crash occurred
2. IF phase == LOADING_CHECKPOINT:
     Retry from checkpoint load
3. IF phase == REPLAYING_WAL:
     Resume from journal.wal_replay_position
4. IF phase == VERIFYING_STATE:
     Re-run verification
5. IF phase == FAILED:
     Manual intervention required
```
---
[[CRITERIA_JSON: {"module_id": "zcmb-m5", "criteria": ["ProcessIdentity is 128 bytes with pid at offset 0x00, generation at offset 0x04, last_heartbeat_ns at offset 0x08, state at offset 0x10, and statistics at offset 0x40 all cache-line aligned", "ProcessRegistry register_process increments generation counter on slot reuse to distinguish PID reuse from process restart", "ProcessRegistry is_alive checks PID existence via kill with signal zero, generation match, heartbeat freshness within timeout, and state not equal to DEAD", "OrphanedSlotHeader is 64 bytes with fence, claim_timestamp_ns, owner_slot, ref_count, message_size, and sequence fields", "FenceManager allocate_fence atomically increments global_fence counter and returns unique fence value per process instance", "OrphanReclaimer scans slots checking claim age exceeds orphan_timeout, owner liveness via is_alive, and fence validity before CAS reclamation", "WriteAheadLog append writes WalEntryHeader with magic, entry_size, sequence, timestamp_ns, checksum, and flags followed by data payload", "WriteAheadLog replay validates entry magic and checksum before passing to handler skipping corrupted entries and stopping on truncated reads", "WalFileHeader is 64 bytes with magic, version, create_time_ns, last_sync_ns, next_sequence, truncated_sequence, and checksum fields", "CheckpointFileHeader is 64 bytes with magic, version, wal_sequence, timestamp_ns, ring_state_size, num_subscribers, and checksum", "CheckpointManager save_checkpoint writes to temp file then atomically renames to final path for crash atomicity", "RecoveryJournalEntry is 64 bytes tracking phase, checkpoint_sequence, wal_replay_position, timestamp_ns, and checksum", "RecoveryCoordinator recover uses journal to resume from interrupted phase loading checkpoint then replaying WAL from recorded position", "SignalHandler setup installs handlers for SIGTERM and SIGINT setting shutdown_requested flag on first signal and force_shutdown on second", "At-least-once delivery semantics documented with consumer responsible for deduplication using message sequence numbers", "Performance targets: heartbeat update under 100 ns, orphan scan of 1000 slots under 1 ms, checkpoint save and load of 1 MB state 10-30 ms, full recovery under 100 ms"]}]
<!-- END_TDD_MOD -->


# Project Structure: Zero-Copy Message Bus
## Directory Tree
```
zcmb/
├── include/
│   └── zcmb/
│       ├── shm_allocator.hpp              # M1: SharedMemory RAII wrapper
│       ├── spsc_ring_buffer.hpp           # M1: Lock-free SPSC queue
│       ├── ring_buffer_config.hpp         # M1: Configuration structs
│       ├── mpmc_ring_buffer.hpp           # M3: Vyukov algorithm MPMC queue
│       ├── mpmc_config.hpp                # M3: MPMC configuration
│       ├── sharded_mpmc.hpp               # M3: Sharded queue with N SPSC buffers
│       ├── backpressure.hpp               # M3: Credit-based flow control
│       ├── contention_metrics.hpp         # M3: CAS failure tracking
│       ├── flat/
│       │   ├── scalar_types.hpp           # M2: Type definitions (int8-64, float, double)
│       │   ├── buffer_view.hpp            # M2: Base class for zero-copy access
│       │   ├── vector_view.hpp            # M2: Array accessor template
│       │   ├── string_view.hpp            # M2: String accessor (length + chars)
│       │   ├── builder_base.hpp           # M2: FlatBuffer construction base
│       │   └── vtable.hpp                 # M2: VTable structure and utilities
│       ├── schema/
│       │   ├── ast_types.hpp              # M2: Schema AST node types
│       │   ├── schema_parser.hpp          # M2: Text schema → AST
│       │   ├── layout_calculator.hpp      # M2: Field offset computation
│       │   └── schema_registry.hpp        # M2: Runtime schema lookup
│       ├── gen/
│       │   └── code_generator.hpp         # M2: AST → C++ code generator
│       ├── pubsub/
│       │   ├── topic_trie.hpp             # M4: Trie node structure + matching
│       │   ├── topic_registry.hpp         # M4: Subscribe/unsubscribe/match API
│       │   ├── pubsub_buffer.hpp          # M4: Ref-counted message slots
│       │   ├── subscriber_cursor.hpp      # M4: Per-subscriber position tracking
│       │   ├── message_filter.hpp         # M4: Predicate evaluation + bloom
│       │   ├── retained_store.hpp         # M4: Last-known-good message cache
│       │   ├── last_will.hpp              # M4: Disconnect notification
│       │   └── pubsub_broker.hpp          # M4: Unified publish/subscribe API
│       ├── pubsub_config.hpp              # M4: Configuration structs
│       ├── recovery/
│       │   ├── heartbeat_monitor.hpp      # M5: Liveness tracking with timestamps
│       │   ├── process_identity.hpp       # M5: PID + generation counter
│       │   ├── orphan_reclaimer.hpp       # M5: Slot cleanup with fencing
│       │   ├── fence_manager.hpp          # M5: Monotonic fence value generation
│       │   ├── write_ahead_log.hpp        # M5: Durable log with checksums
│       │   ├── checkpoint_manager.hpp     # M5: Snapshot save/load
│       │   ├── recovery_journal.hpp       # M5: Crash-during-recovery tracking
│       │   ├── signal_handler.hpp         # M5: Graceful shutdown support
│       │   └── recovery_coordinator.hpp   # M5: Unified recovery orchestration
│       └── recovery_config.hpp            # M5: Configuration structs
├── src/
│   ├── shm_allocator.cpp                  # M1: Implementation (error handling)
│   ├── spsc_ring_buffer.cpp               # M1: Implementation (barriers)
│   ├── mpmc_ring_buffer.cpp               # M3: Core MPMC algorithm
│   ├── sharded_mpmc.cpp                   # M3: Shard management
│   ├── backpressure.cpp                   # M3: Flow control logic
│   ├── contention_metrics.cpp             # M3: Metrics collection
│   ├── flat/
│   │   ├── buffer_view.cpp                # M2: VTable traversal, field access
│   │   ├── builder_base.cpp               # M2: Buffer construction helpers
│   │   └── vtable.cpp                     # M2: VTable utilities
│   ├── schema/
│   │   ├── schema_parser.cpp              # M2: Lexer + recursive descent parser
│   │   ├── layout_calculator.cpp          # M2: Alignment, padding, offset calc
│   │   └── schema_registry.cpp            # M2: Thread-safe schema storage
│   ├── gen/
│   │   └── code_generator.cpp             # M2: C++ code emission
│   ├── pubsub/
│   │   ├── topic_trie.cpp                 # M4: Trie traversal implementation
│   │   ├── topic_registry.cpp             # M4: RW-lock protected registry
│   │   ├── pubsub_buffer.cpp              # M4: Ref count management
│   │   ├── subscriber_cursor.cpp          # M4: Cursor advancement
│   │   ├── message_filter.cpp             # M4: Filter compilation/evaluation
│   │   ├── retained_store.cpp             # M4: Retained message storage
│   │   ├── last_will.cpp                  # M4: Will delivery + heartbeat
│   │   └── pubsub_broker.cpp              # M4: Broker orchestration
│   ├── pubsub_config.cpp                  # M4: Config validation
│   ├── recovery/
│   │   ├── heartbeat_monitor.cpp          # M5: Timestamp updates, timeout checks
│   │   ├── process_identity.cpp           # M5: PID tracking, generation increments
│   │   ├── orphan_reclaimer.cpp           # M5: Slot scanning, fence validation
│   │   ├── fence_manager.cpp              # M5: Atomic fence generation
│   │   ├── write_ahead_log.cpp            # M5: Append, sync, truncate, replay
│   │   ├── checkpoint_manager.cpp         # M5: Snapshot serialization
│   │   ├── recovery_journal.cpp           # M5: Phase tracking
│   │   ├── signal_handler.cpp             # M5: Signal disposition setup
│   │   └── recovery_coordinator.cpp       # M5: Recovery sequence
│   └── recovery_config.cpp                # M5: Config validation
├── schemas/
│   ├── primitives.fbs                     # M2: Built-in types (testing)
│   └── trading.fbs                        # M2: Example: TradeOrder, MarketData
├── generated/
│   └── trading/
│       ├── trade_order.hpp                # M2: Generated View/Builder
│       └── market_data.hpp                # M2: Generated View/Builder
├── tests/
│   ├── shm_allocator_test.cpp             # M1: Creation, mapping, cleanup
│   ├── spsc_ring_buffer_test.cpp          # M1: Produce/consume, wraparound
│   ├── cross_process_test.cpp             # M1: Fork-based IPC test
│   ├── crash_recovery_test.cpp            # M1: Orphan detection
│   ├── mpmc_basic_test.cpp                # M3: Single producer/consumer baseline
│   ├── mpmc_contention_test.cpp           # M3: Multi-producer stress test
│   ├── mpmc_aba_test.cpp                  # M3: Sequence number wraparound
│   ├── sharded_mpmc_test.cpp              # M3: Shard distribution
│   ├── backpressure_test.cpp              # M3: Flow control behavior
│   ├── fairness_test.cpp                  # M3: Starvation detection
│   ├── scalar_types_test.cpp              # M2: Type size/alignment tests
│   ├── buffer_view_test.cpp               # M2: VTable traversal, field access
│   ├── vector_string_test.cpp             # M2: Array and string handling
│   ├── schema_parser_test.cpp             # M2: Parse all schema constructs
│   ├── layout_calculator_test.cpp         # M2: Alignment, padding correctness
│   ├── code_generator_test.cpp            # M2: Generated code compiles and runs
│   ├── round_trip_test.cpp                # M2: Serialize → deserialize → verify
│   ├── schema_evolution_test.cpp          # M2: Add field, old reader, new reader
│   ├── topic_trie_test.cpp                # M4: Trie insert/match/delete
│   ├── wildcard_test.cpp                  # M4: + and # wildcard matching
│   ├── pubsub_buffer_test.cpp             # M4: Ref count lifecycle
│   ├── subscriber_cursor_test.cpp         # M4: Independent cursor advance
│   ├── message_filter_test.cpp            # M4: Bloom + predicate evaluation
│   ├── retained_store_test.cpp            # M4: Late subscriber delivery
│   ├── last_will_test.cpp                 # M4: Disconnect notification
│   ├── pubsub_broker_test.cpp             # M4: End-to-end pub/sub
│   ├── heartbeat_test.cpp                 # M5: Timeout detection
│   ├── process_identity_test.cpp          # M5: PID reuse handling
│   ├── orphan_reclaim_test.cpp            # M5: Slot recovery scenarios
│   ├── fence_test.cpp                     # M5: Fence validation
│   ├── wal_basic_test.cpp                 # M5: Append/sync/truncate
│   ├── wal_checksum_test.cpp              # M5: Corruption detection
│   ├── wal_replay_test.cpp                # M5: Recovery replay
│   ├── checkpoint_test.cpp                # M5: Save/load/atomic
│   ├── recovery_journal_test.cpp          # M5: Crash-during-recovery
│   ├── signal_test.cpp                    # M5: Graceful shutdown
│   ├── full_recovery_test.cpp             # M5: End-to-end recovery
│   └── cross_process_recovery_test.cpp    # M5: Fork-based crash simulation
├── benchmarks/
│   ├── latency_bench.cpp                  # M1: Round-trip measurement
│   ├── throughput_bench.cpp               # M1: Messages per second
│   ├── access_latency.cpp                 # M2: Field access nanoseconds
│   ├── build_latency.cpp                  # M2: Construction time
│   ├── json_comparison_bench.cpp          # M2: RapidJSON comparison
│   ├── contention_bench.cpp               # M3: CAS failure rates at various N
│   ├── mpmc_latency_bench.cpp             # M3: P50/P99/P999 under contention
│   ├── mpmc_throughput_bench.cpp          # M3: Scaling with producer count
│   ├── sharded_comparison_bench.cpp       # M3: Single MPMC vs Sharded
│   ├── topic_match_bench.cpp              # M4: Exact vs wildcard latency
│   ├── fan_out_bench.cpp                  # M4: 1/10/100 subscriber scaling
│   ├── filter_bench.cpp                   # M4: Bloom vs full evaluation
│   ├── end_to_end_bench.cpp               # M4: Publish to delivery latency
│   ├── checkpoint_latency.cpp             # M5: Save/load timing
│   ├── wal_throughput.cpp                 # M5: Append/sync throughput
│   └── recovery_time.cpp                  # M5: Full recovery benchmark
├── CMakeLists.txt                         # Build configuration
├── Makefile                               # Build system convenience
├── linker.ld                              # Linker script (if needed)
├── README.md                              # Project overview
└── LICENSE                                # License file
```
## Creation Order
### 1. **Project Setup** (30 min)
   - Create directory structure
   - `CMakeLists.txt`, `Makefile`
   - `README.md`
### 2. **M1: Shared Memory Ring Buffer** (8-10 hours)
   - `include/zcmb/shm_allocator.hpp`
   - `include/zcmb/spsc_ring_buffer.hpp`
   - `include/zcmb/ring_buffer_config.hpp`
   - `src/shm_allocator.cpp`
   - `src/spsc_ring_buffer.cpp`
   - `tests/shm_allocator_test.cpp`
   - `tests/spsc_ring_buffer_test.cpp`
   - `tests/cross_process_test.cpp`
   - `tests/crash_recovery_test.cpp`
   - `benchmarks/latency_bench.cpp`
   - `benchmarks/throughput_bench.cpp`
### 3. **M2: Zero-Copy Serialization** (12-15 hours)
   - `include/zcmb/flat/scalar_types.hpp`
   - `include/zcmb/flat/vtable.hpp`
   - `include/zcmb/flat/buffer_view.hpp`
   - `include/zcmb/flat/vector_view.hpp`
   - `include/zcmb/flat/string_view.hpp`
   - `include/zcmb/flat/builder_base.hpp`
   - `include/zcmb/schema/ast_types.hpp`
   - `include/zcmb/schema/schema_parser.hpp`
   - `include/zcmb/schema/layout_calculator.hpp`
   - `include/zcmb/schema/schema_registry.hpp`
   - `include/zcmb/gen/code_generator.hpp`
   - `src/flat/*.cpp`
   - `src/schema/*.cpp`
   - `src/gen/code_generator.cpp`
   - `schemas/primitives.fbs`
   - `schemas/trading.fbs`
   - `generated/trading/*.hpp`
   - `tests/scalar_types_test.cpp` through `tests/schema_evolution_test.cpp`
   - `benchmarks/access_latency.cpp`
   - `benchmarks/build_latency.cpp`
   - `benchmarks/json_comparison_bench.cpp`
### 4. **M3: Multi-Producer Multi-Consumer** (10-12 hours)
   - `include/zcmb/mpmc_config.hpp`
   - `include/zcmb/mpmc_ring_buffer.hpp`
   - `include/zcmb/sharded_mpmc.hpp`
   - `include/zcmb/backpressure.hpp`
   - `include/zcmb/contention_metrics.hpp`
   - `src/mpmc_ring_buffer.cpp`
   - `src/sharded_mpmc.cpp`
   - `src/backpressure.cpp`
   - `src/contention_metrics.cpp`
   - `tests/mpmc_basic_test.cpp` through `tests/fairness_test.cpp`
   - `benchmarks/contention_bench.cpp` through `benchmarks/sharded_comparison_bench.cpp`
### 5. **M4: Publish-Subscribe & Topics** (10-12 hours)
   - `include/zcmb/pubsub/topic_trie.hpp`
   - `include/zcmb/pubsub/topic_registry.hpp`
   - `include/zcmb/pubsub/pubsub_buffer.hpp`
   - `include/zcmb/pubsub/subscriber_cursor.hpp`
   - `include/zcmb/pubsub/message_filter.hpp`
   - `include/zcmb/pubsub/retained_store.hpp`
   - `include/zcmb/pubsub/last_will.hpp`
   - `include/zcmb/pubsub/pubsub_broker.hpp`
   - `include/zcmb/pubsub_config.hpp`
   - `src/pubsub/*.cpp`
   - `src/pubsub_config.cpp`
   - `tests/topic_trie_test.cpp` through `tests/pubsub_broker_test.cpp`
   - `benchmarks/topic_match_bench.cpp` through `benchmarks/end_to_end_bench.cpp`
### 6. **M5: Crash Recovery & Durability** (8-10 hours)
   - `include/zcmb/recovery/heartbeat_monitor.hpp`
   - `include/zcmb/recovery/process_identity.hpp`
   - `include/zcmb/recovery/orphan_reclaimer.hpp`
   - `include/zcmb/recovery/fence_manager.hpp`
   - `include/zcmb/recovery/write_ahead_log.hpp`
   - `include/zcmb/recovery/checkpoint_manager.hpp`
   - `include/zcmb/recovery/recovery_journal.hpp`
   - `include/zcmb/recovery/signal_handler.hpp`
   - `include/zcmb/recovery/recovery_coordinator.hpp`
   - `include/zcmb/recovery_config.hpp`
   - `src/recovery/*.cpp`
   - `src/recovery_config.cpp`
   - `tests/heartbeat_test.cpp` through `tests/cross_process_recovery_test.cpp`
   - `benchmarks/checkpoint_latency.cpp` through `benchmarks/recovery_time.cpp`
## File Count Summary
- **Total files**: 123
- **Header files**: 42
- **Source files**: 31
- **Test files**: 37
- **Benchmark files**: 16
- **Schema files**: 2
- **Generated files**: 2 (example)
- **Directories**: 18
- **Estimated lines of code**: ~25,000-35,000
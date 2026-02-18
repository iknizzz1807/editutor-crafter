# AUDIT & FIX: io-uring-server

## CRITIQUE
- **Missing Provided Buffers (IORING_REGISTER_PBUF_RING)**: The audit correctly identifies this critical omission. In high-concurrency async I/O, pre-allocating buffers per operation doesn't scale. Provided buffer rings allow the kernel to select buffers from a pool, essential for servers handling 10K+ connections where pre-allocation wastes memory.
- **Short Reads/Writes Not Addressed**: Async I/O operations can return partial completions. A read of 4096 bytes might complete with only 1024 bytes. The network server milestone has no AC for handling partial I/O, which will cause data corruption or protocol violations.
- **SQPOLL Mode Missing**: IORING_SETUP_SQPOLL is one of io_uring's most significant features — a kernel thread polls the SQ, eliminating the io_uring_enter syscall entirely. This is absent from all milestones.
- **Memory Barrier Detail Insufficient**: The pitfall about memory barriers is mentioned but the AC doesn't require demonstrating correct barrier placement. On weakly-ordered architectures (or even x86 with compiler reordering), incorrect barriers corrupt ring state.
- **Zero-Copy Nuances**: IORING_OP_SEND_ZC requires kernel 6.0+ and the notification CQE (with IORING_CQE_F_NOTIF flag) must be harvested before the buffer can be reused. The pitfall mentions this but the AC doesn't verify it.
- **Error Handling Gaps**: No AC for handling EBUSY, EAGAIN, or CQE error codes. io_uring CQEs can return negative res values that must be handled per-operation.
- **Milestone 2 Scope is Narrow**: A file I/O server is an odd intermediate step. It would be more educational to go directly to network I/O since that's where io_uring shines most. The file I/O milestone should focus on demonstrating fixed buffers and registered file descriptors.
- **No Mention of Registered File Descriptors**: IORING_REGISTER_FILES reduces per-operation fd lookup overhead. This is a significant optimization missed entirely.

## FIXED YAML
```yaml
id: io-uring-server
name: "io_uring High-Performance Server"
description: >-
  Build a high-performance TCP server using Linux io_uring, progressing from basic
  SQ/CQ operations through advanced features including zero-copy I/O, provided
  buffer rings, SQ polling, and linked operations. Benchmark against epoll.
difficulty: expert
estimated_hours: "40-55"
essence: >-
  Asynchronous I/O through shared ring buffers between kernel and userspace,
  eliminating syscall overhead by batching submission/completion queue operations,
  using provided buffer rings for scalable memory management, and enabling
  zero-copy data transfer for high-throughput workloads.
why_important: >-
  Building this teaches you modern Linux I/O architecture that outperforms
  traditional epoll/select mechanisms, a critical skill for high-performance
  systems like databases, web servers, and real-time applications where I/O
  latency directly impacts throughput.
learning_outcomes:
  - Implement submission and completion queue management with memory-mapped ring buffers
  - Design buffer management strategies including fixed buffers and provided buffer rings
  - Build batched syscall interfaces that minimize kernel transitions
  - Handle short reads/writes and partial I/O completions correctly
  - Debug asynchronous I/O race conditions and completion ordering issues
  - Optimize network servers using linked operations, SQ polling, and registered file descriptors
  - Benchmark I/O subsystems and analyze performance differences between io_uring and epoll
  - Implement proper resource cleanup and error handling for kernel ring buffers
skills:
  - Asynchronous I/O Programming
  - Ring Buffer Management
  - Zero-copy Data Transfer
  - Linux Kernel Interfaces
  - High-performance Networking
  - Systems Benchmarking
  - Memory-mapped I/O
  - Buffer Pool Management
tags:
  - io-uring
  - async-io
  - high-performance
  - linux
  - networking
  - zero-copy
  - expert
architecture_doc: architecture-docs/io-uring-server/index.md
languages:
  recommended:
    - C
    - Rust
    - Zig
  also_possible: []
resources:
  - name: Lord of the io_uring""
    url: https://unixism.net/loti/
    type: tutorial
  - name: io_uring Kernel Documentation""
    url: https://kernel.dk/io_uring.pdf
    type: reference
  - name: liburing GitHub Repository""
    url: https://github.com/axboe/liburing
    type: reference
  - name: io_uring by Example (zero-copy, provided buffers)""
    url: https://github.com/shuveb/io_uring-by-example
    type: tutorial
prerequisites:
  - type: project
    name: build-event-loop
  - type: skill
    name: Linux system calls (read, write, mmap, syscall ABI)
  - type: skill
    name: C or Rust memory management
milestones:
  - id: io-uring-server-m1
    name: "Basic SQ/CQ Operations & Buffer Strategies"
    description: >-
      Set up io_uring instance, perform basic submission/completion queue operations
      with correct memory barriers, and implement both fixed buffer registration
      and provided buffer rings (IORING_REGISTER_PBUF_RING).
    estimated_hours: "10-14"
    concepts:
      - io_uring architecture (SQ, CQ, SQE, CQE)
      - Memory-mapped shared ring buffers
      - Memory barriers for ring head/tail synchronization
      - Fixed buffer registration (IORING_REGISTER_BUFFERS)
      - Provided buffer rings (IORING_REGISTER_PBUF_RING)
      - Batched syscalls via io_uring_enter
    skills:
      - Ring buffer management
      - System call batching
      - Memory barrier synchronization
      - Buffer pool design
    acceptance_criteria:
      - io_uring instance is initialized with io_uring_setup syscall and SQ/CQ rings are mmap'd into userspace
      - SQ and CQ sizes are configured correctly — CQ is at least 2x SQ size (kernel default) to prevent CQ overflow
      - Memory barriers (io_uring_smp_store_release for tail, io_uring_smp_load_acquire for head) are correctly placed between ring updates
      - Batched submission demonstrated by submitting 10+ SQEs with a single io_uring_enter call and verifying all CQEs are harvested
      - Fixed buffers are registered with IORING_REGISTER_BUFFERS and used with IORING_OP_READ_FIXED/WRITE_FIXED; benchmark shows reduced per-op overhead vs non-fixed
      - Provided buffer ring is set up with IORING_REGISTER_PBUF_RING; operations select buffers from the ring and the buffer group ID is correctly specified in SQEs
      - CQE error handling checks res field for negative errno values and handles EAGAIN, ECANCELED, and EBUSY appropriately
      - All registered buffers and rings are properly unregistered during cleanup
    pitfalls:
      - SQ and CQ have separate sizes; CQ overflow (IORING_CQ_OVERFLOW) silently drops completions if CQ is full and you don't check the overflow counter
      - Missing memory barriers between head/tail updates causes data races even on x86 due to compiler reordering
      - io_uring_enter with IORING_ENTER_GETEVENTS blocks; without it, you must poll CQ manually
      - Provided buffer rings require kernel 5.19+; check kernel version or feature probe with IORING_REGISTER_PBUF_RING
      - Fixed buffer registration fails if buffers are not page-aligned on some kernel versions
    deliverables:
      - io_uring setup module with mmap'd shared memory rings and configurable queue depths
      - SQE preparation helpers for read/write with both regular and fixed buffer variants
      - CQE harvesting loop with proper error code checking and overflow detection
      - Provided buffer ring setup and replenishment logic for scalable buffer management
      - Batched submission test demonstrating multiple ops per io_uring_enter call

  - id: io-uring-server-m2
    name: "File I/O Server with Registered Resources"
    description: >-
      Build a file server using io_uring for all I/O operations, utilizing
      registered file descriptors and fixed buffers, with correct handling of
      short reads.
    estimated_hours: "8-12"
    concepts:
      - Asynchronous file I/O with io_uring
      - Registered file descriptors (IORING_REGISTER_FILES)
      - Short reads and partial I/O completions
      - Direct I/O vs buffered I/O tradeoffs
    skills:
      - Asynchronous file I/O
      - Registered file descriptor management
      - Short read/write handling
      - Direct I/O alignment requirements
    acceptance_criteria:
      - File server serves read requests using IORING_OP_READ with correct file descriptor and offset parameters
      - File descriptors are registered with IORING_REGISTER_FILES and SQEs use IOSQE_FIXED_FILE flag with the registered index
      - Short reads (CQE res < requested bytes) are handled by resubmitting a read for the remaining bytes at the updated offset
      - Multiple concurrent file reads (at least 64 in-flight) are handled with correct buffer-to-request association
      - Direct I/O (O_DIRECT) variant is implemented with properly aligned buffers (512-byte or 4K-byte alignment depending on filesystem)
      - Benchmark demonstrates at least 2x throughput improvement over synchronous pread for 64+ concurrent 4KB random reads on SSD
      - Buffer lifetime is guaranteed to extend until CQE is harvested; no buffer is reused while an async operation references it
    pitfalls:
      - IORING_OP_READ requires explicit offset; passing -1 uses the file's current position which is racy with concurrent ops
      - Registered file descriptors use indices (not fd numbers); confusing them causes EBADF
      - O_DIRECT requires buffer alignment to filesystem block size; misalignment causes EINVAL
      - Buffer reuse before CQE harvest causes data corruption that is extremely difficult to debug
      - Short reads are common on pipes, sockets, and even files near EOF; treating them as errors breaks the server
    deliverables:
      - File server handling concurrent async reads using IORING_OP_READ
      - Registered file descriptor table with add/remove/update support
      - Short read handler resubmitting partial completions
      - Buffer pool with ownership tracking preventing premature reuse
      - Benchmark suite comparing io_uring vs synchronous and vs O_DIRECT file I/O

  - id: io-uring-server-m3
    name: "TCP Network Server"
    description: >-
      Build a TCP echo server using io_uring for accept, read, write, and close
      operations. Handle connection lifecycle entirely through io_uring with
      multishot accept and provided buffers. Handle short writes and connection
      cleanup.
    estimated_hours: "10-14"
    concepts:
      - Async TCP accept/read/write via io_uring
      - Multishot accept (IORING_ACCEPT_MULTISHOT)
      - Provided buffers for receive operations
      - Connection state machine
      - Short writes on non-blocking sockets
    skills:
      - Asynchronous TCP server implementation
      - Multishot operation handling
      - Connection state tracking
      - Socket lifecycle management
      - Provided buffer replenishment under load
    acceptance_criteria:
      - IORING_OP_ACCEPT is used for async connection acceptance; no blocking accept() calls
      - Multishot accept (IORING_ACCEPT_MULTISHOT) is used where kernel supports it, reusing a single SQE for multiple accepts
      - Receive operations use provided buffer rings (IORING_OP_RECV with buffer group ID) to avoid pre-allocating per-connection buffers
      - Short writes (CQE res < requested send size) are handled by resubmitting the remaining bytes
      - Connection lifecycle (accept → read → process → write → close) is managed entirely through io_uring without epoll
      - Connection cleanup cancels all in-flight SQEs for a disconnected socket using IORING_OP_ASYNC_CANCEL before closing the fd
      - Server handles at least 10,000 concurrent connections with stable latency (p99 < 10ms for echo of 64-byte messages)
      - Provided buffer ring is replenished when buffer count drops below 25% of initial capacity
    pitfalls:
      - Multishot accept CQEs have IORING_CQE_F_MORE flag set; its absence means the multishot was terminated and must be resubmitted
      - Provided buffers for recv require checking CQE flags (IORING_CQE_F_BUFFER) and extracting buffer ID from cqe->flags >> IORING_CQE_BUFFER_SHIFT
      - Connection cleanup without ASYNC_CANCEL leaks SQEs and causes use-after-free when CQEs arrive for closed fds
      - Forgetting to replenish provided buffer rings under load causes ENOBUFS errors for new recv operations
      - TCP_NODELAY should be set on accepted sockets to avoid Nagle's algorithm adding latency to small writes
    deliverables:
      - TCP echo server with IORING_OP_ACCEPT for async connection handling
      - Multishot accept implementation with automatic resubmission on termination
      - Provided buffer ring integration for receive operations
      - Connection state tracker managing per-connection in-flight operations
      - Short write handler for partial send completions
      - Scalability test demonstrating 10K+ concurrent connections

  - id: io-uring-server-m4
    name: "Zero-copy, SQ Polling, Linked Ops & Benchmarks"
    description: >-
      Implement advanced io_uring features including zero-copy sends, SQ polling
      mode, linked SQE chains, and IO_DRAIN ordering. Produce comprehensive
      benchmarks comparing io_uring vs epoll under varied workloads.
    estimated_hours: "10-14"
    concepts:
      - Zero-copy send (IORING_OP_SEND_ZC)
      - SQ polling mode (IORING_SETUP_SQPOLL)
      - Linked SQEs (IOSQE_IO_LINK)
      - IO_DRAIN for ordering guarantees
      - CQE notification flag (IORING_CQE_F_NOTIF) for zero-copy buffer lifecycle
    skills:
      - Zero-copy buffer lifecycle management
      - SQ polling configuration and idle timeout tuning
      - Operation chaining with failure semantics
      - Performance profiling and benchmarking
      - Comparing async I/O models quantitatively
    acceptance_criteria:
      - IORING_OP_SEND_ZC is used for network sends; buffer is not freed until the notification CQE (IORING_CQE_F_NOTIF flag) is received
      - Each zero-copy send produces two CQEs — the completion CQE and the notification CQE; both are correctly harvested and distinguished
      - SQ polling mode (IORING_SETUP_SQPOLL) is enabled with configurable idle timeout; benchmark shows reduced syscall count vs non-SQPOLL mode
      - Linked SQEs (IOSQE_IO_LINK) create operation chains where failure of any link cancels subsequent operations in the chain
      - IOSQE_IO_DRAIN is used where ordering is required; benchmark demonstrates the throughput cost of IO_DRAIN vs unordered submission
      - Comprehensive benchmark suite measures latency (p50, p95, p99) and throughput for io_uring vs epoll under at least 3 workloads — file I/O heavy, network I/O heavy, and mixed
      - Benchmark results include syscall counts (measured via strace or perf) showing io_uring's batching advantage
      - All advanced features gracefully fall back or report clear errors on unsupported kernel versions
    pitfalls:
      - Zero-copy send requires kernel 6.0+; IORING_OP_SEND_ZC returns EINVAL on older kernels
      - Zero-copy notification CQE can arrive out of order relative to other CQEs; buffer tracking must handle this
      - SQPOLL thread consumes CPU even when idle until sq_thread_idle timeout; set it appropriately (default 1000ms is often too high)
      - SQPOLL requires CAP_SYS_NICE or io_uring_register with IORING_REGISTER_IOWQ_AFF on unprivileged processes
      - Linked SQEs fail as a chain — if operation N fails, operations N+1..end are cancelled with -ECANCELED
      - IO_DRAIN forces serialization of all prior SQEs; using it frequently defeats the purpose of async I/O
    deliverables:
      - Zero-copy send implementation with dual-CQE handling (completion + notification)
      - SQ polling mode configuration with idle timeout tuning and privilege handling
      - Linked SQE chains for compound operations with correct error propagation
      - IO_DRAIN usage for operations requiring ordering guarantees
      - Benchmark suite with latency percentiles, throughput, and syscall count comparison across io_uring vs epoll
```
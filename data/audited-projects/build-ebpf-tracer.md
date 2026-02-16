# AUDIT & FIX: build-ebpf-tracer

## CRITIQUE
- **Audit Finding 1 (Verifier Education):** Valid and critical. The eBPF verifier is the most unique and frustrating aspect of eBPF programming. Understanding its constraints (bounded loops, no unbounded memory access, stack size limits, pointer arithmetic rules) is essential. M1 pitfalls mention it briefly but no AC tests the learner's understanding of verifier constraints.
- **Audit Finding 2 (BTF/CO-RE):** Valid and important. Without BTF (BPF Type Format) and CO-RE (Compile Once, Run Everywhere), eBPF programs break across kernel versions because struct offsets change. Modern eBPF development with libbpf requires BTF for portability. M1 should address this.
- **Prerequisite Issue:** Prerequisites list 'build-strace' and 'build-kernel-module' as project prerequisites. These are reasonable but the 'build-kernel-module' prerequisite is unusual for an eBPF project since eBPF specifically avoids writing kernel modules. The prerequisite should be 'Linux kernel basics' or 'syscall fundamentals' rather than kernel module development.
- **M2 Histogram Detail:** The AC says 'Generate log2 histogram buckets for latency distribution in kernel space' which implies doing the histogram bucketing in the eBPF program itself. This is correct and efficient but the deliverable should clarify that this uses BPF_MAP_TYPE_ARRAY with log2-indexed buckets.
- **M3 TCP Tracepoint Specificity:** Using sock:inet_sock_set_state is correct and specific. Good.
- **M4 Scope:** M4 combines multiple eBPF programs, per-CPU maps, terminal UI, AND runtime probe management. This is overscoped for 7-12 hours.
- **Missing Security Context:** eBPF programs run in kernel space - there's no discussion of security implications (CAP_BPF, unprivileged eBPF restrictions, potential for kernel crashes via verifier bugs).
- **Missing Map Types Coverage:** BPF_MAP_TYPE_PERF_EVENT_ARRAY vs BPF_MAP_TYPE_RINGBUF distinction matters - ring buffer is newer and more efficient. M1 uses ring buffer but doesn't explain why it's preferred over perf event arrays.
- **Estimated Hours:** Ranges are used (7-10, 7-12) which is inconsistent. Total range is 28-44, project says 30-45.

## FIXED YAML
```yaml
id: build-ebpf-tracer
name: eBPF Tracing Tool
description: >-
  Dynamic kernel instrumentation using eBPF programs on kprobes and
  tracepoints with efficient kernel-to-userspace data transfer via BPF maps.
difficulty: advanced
estimated_hours: 40
essence: >-
  Dynamic kernel instrumentation via eBPF bytecode that safely executes in
  kernel space after passing the kernel verifier, extracting runtime
  telemetry through kprobes, tracepoints, and BPF maps with CO-RE
  (Compile Once Run Everywhere) for cross-kernel portability.
why_important: >-
  Building this teaches low-level Linux kernel internals and observability
  techniques used in production tools like Cilium, Falco, and bpftrace.
  These are essential skills for systems programming, performance engineering,
  and security monitoring.
learning_outcomes:
  - Write eBPF programs in C that pass the kernel verifier's safety constraints
  - Understand and work within verifier limitations (bounded loops, stack size, pointer safety)
  - Use BTF and CO-RE for eBPF programs portable across kernel versions
  - Attach kprobes and tracepoints to kernel functions for dynamic instrumentation
  - Transfer data efficiently between kernel and userspace using BPF ring buffers and maps
  - Measure syscall latency distributions using entry/exit probe correlation
  - Trace TCP connection lifecycle events using kernel tracepoints
  - Build multi-source observability dashboards combining multiple eBPF data sources
skills:
  - eBPF C programming
  - BPF verifier constraints
  - BTF and CO-RE portability
  - Kprobes and tracepoints
  - BPF maps (hash, array, ring buffer, per-CPU)
  - libbpf framework
  - Performance analysis
  - Kernel-userspace communication
tags:
  - ebpf
  - tracing
  - observability
  - linux-kernel
  - performance
  - advanced
architecture_doc: architecture-docs/build-ebpf-tracer/index.md
languages:
  recommended:
    - C
    - Rust
  also_possible:
    - Go
resources:
  - name: BPF Performance Tools (Brendan Gregg)
    url: https://www.brendangregg.com/bpf-performance-tools-book.html
    type: book
  - name: eBPF Documentation
    url: https://ebpf.io/what-is-ebpf/
    type: reference
  - name: libbpf Bootstrap Examples
    url: https://github.com/libbpf/libbpf-bootstrap
    type: tool
  - name: BPF CO-RE Reference Guide
    url: https://nakryiko.com/posts/bpf-core-reference-guide/
    type: tutorial
prerequisites:
  - type: skill
    name: Linux syscall fundamentals and /proc filesystem
  - type: skill
    name: C programming with pointer arithmetic
  - type: skill
    name: Basic understanding of kernel vs userspace separation
milestones:
  - id: build-ebpf-tracer-m1
    name: eBPF Fundamentals & Kprobe Tracer
    description: >-
      Write an eBPF program, understand verifier constraints, attach to a
      kprobe, and collect data in userspace via ring buffer using CO-RE
      for portability.
    acceptance_criteria:
      - eBPF program is written in C and compiled to BPF bytecode using clang with -target bpf and BTF debug info (-g)
      - Program attaches to a kprobe (e.g., do_sys_openat2) using libbpf and the generated skeleton
      - CO-RE (BPF_CORE_READ) is used for accessing kernel struct fields, making the program portable across kernel versions with different struct layouts
      - Data is passed from kernel to userspace via BPF_MAP_TYPE_RINGBUF (not perf event array) for efficient batched event delivery
      - Userspace consumer polls the ring buffer and displays traced events showing PID, comm (process name), timestamp, and function arguments
      - At least one verifier rejection is intentionally triggered (e.g., unbounded loop, out-of-bounds access) and the verifier error log is captured and explained
      - bpf_probe_read_kernel is used for reading kernel memory safely; direct pointer dereference is avoided
      - The program runs with CAP_BPF (or root) and the required capability is documented
    pitfalls:
      - The eBPF verifier rejects unbounded loops, out-of-bounds memory access, and uninitialized variables - every code path must be provably safe
      - bpf_probe_read_kernel (not bpf_probe_read_user) must be used for kernel memory - using the wrong variant causes silent data corruption
      - Ring buffer requires proper polling with ring_buffer__poll in userspace; busy-waiting wastes CPU, too-slow polling drops events
      - Without BTF and CO-RE, the eBPF program hardcodes struct offsets that break on different kernel versions - always use BPF_CORE_READ
      - Kernel headers must be available (either installed or generated via bpftool btf dump) for BTF-enabled compilation
      - eBPF stack size is limited to 512 bytes - large local variables must use BPF per-CPU arrays instead
    concepts:
      - eBPF program lifecycle (compile, load, verify, attach, run)
      - BPF verifier safety constraints and bounded execution
      - CO-RE and BTF for cross-kernel portability
      - Ring buffer vs perf event array for kernel-userspace data transfer
      - Kprobe attachment to kernel function entry/exit
      - CAP_BPF capability and security model
    skills:
      - eBPF C programming
      - BPF verifier understanding
      - libbpf and skeleton generation
      - Ring buffer management
      - CO-RE programming
    deliverables:
      - eBPF C program compiled to BPF bytecode with BTF debug info
      - Kprobe attachment to a kernel function using libbpf skeleton
      - Ring buffer map for efficient kernel-to-userspace event delivery
      - Userspace consumer displaying PID, comm, timestamp, and function info
      - Verifier rejection demonstration with captured error log and explanation
      - CO-RE usage for portable kernel struct field access
    estimated_hours: 10

  - id: build-ebpf-tracer-m2
    name: Syscall Latency Histogram
    description: >-
      Measure and display per-syscall latency distributions using paired
      entry/exit probes and in-kernel histogram computation.
    acceptance_criteria:
      - Entry probe records start timestamp (bpf_ktime_get_ns) in a BPF hash map keyed by thread ID (current->pid)
      - Exit probe retrieves start timestamp, computes duration, and accumulates into a log2 histogram bucket in a BPF array map
      - Histogram buckets are indexed by log2 of the latency in microseconds (bucket 0 = 0-1us, bucket 1 = 1-2us, bucket 2 = 2-4us, etc.)
      - Userspace reads the histogram map periodically and displays a formatted histogram similar to bpftrace output
      - Multiple syscalls are traced simultaneously (e.g., read, write, openat) with separate histogram per syscall number
      - Hash map has a bounded size with graceful handling when full (drop new entries, not crash)
    pitfalls:
      - Entry/exit probe correlation requires keying by thread ID (pid in kernel = thread ID), not process group ID (tgid) - a thread could be interrupted between entry and exit
      - bpf_ktime_get_ns returns monotonic nanoseconds, not wall clock time - this is correct for latency measurement but not for timestamps
      - BPF hash map has a fixed maximum size; when full, bpf_map_update_elem returns -E2BIG - handle this by dropping the entry, not crashing
      - Histogram bucket computation must handle zero latency (log2(0) is undefined) - use bucket 0 for sub-microsecond latencies
      - Nested syscalls (syscall from within a syscall handler) can overwrite the start timestamp for the same thread - this is rare but possible
    concepts:
      - Entry/exit probe pairing for duration measurement
      - BPF hash map for per-thread state correlation
      - In-kernel log2 histogram computation for efficient aggregation
      - BPF array map for fixed-size histogram storage
      - Monotonic vs wall clock time in kernel tracing
    skills:
      - Paired probe correlation
      - BPF map usage (hash, array)
      - Histogram data structures
      - Latency measurement
    deliverables:
      - Entry kprobe recording start timestamp in hash map keyed by thread ID
      - Exit kprobe computing duration and updating log2 histogram bucket
      - Per-syscall histogram stored in BPF array maps
      - Formatted histogram display in userspace with microsecond buckets
      - Graceful map-full handling dropping new entries without crashing
    estimated_hours: 10

  - id: build-ebpf-tracer-m3
    name: TCP Connection Tracer
    description: >-
      Trace TCP connection lifecycle events using kernel tracepoints with
      configurable filtering.
    acceptance_criteria:
      - Attaches to the sock:inet_sock_set_state tracepoint for TCP state change tracking
      - Captures source IP, destination IP, source port, destination port, old state, and new state for each transition
      - Handles both IPv4 (4-byte) and IPv6 (16-byte) addresses correctly using the address family field
      - Tracks connection duration by recording the timestamp of the ESTABLISHED transition and computing delta at CLOSE/TIME_WAIT
      - Configurable filtering via BPF maps allows filtering by PID, destination port, or address range at runtime without reloading the program
      - Outputs connection events to userspace showing formatted IP addresses, ports, state transitions, and duration
    pitfalls:
      - IPv4 and IPv6 have different address sizes (4 vs 16 bytes) - must check the address family field before reading the address to avoid reading garbage
      - TCP state machine has many transitions (SYN_SENT, SYN_RECV, ESTABLISHED, FIN_WAIT1, FIN_WAIT2, TIME_WAIT, CLOSE, etc.) - don't try to track all of them initially, focus on ESTABLISHED and CLOSE
      - Tracepoint arguments are accessed through the tracepoint context struct, not via register reading as with kprobes - the struct layout comes from the tracepoint format definition
      - Connection tracking across state transitions requires correlating events by socket pointer or (saddr, daddr, sport, dport) tuple - socket pointer is more reliable
      - Short-lived connections may transition through multiple states between polling intervals - ring buffer ensures no events are lost
    concepts:
      - Kernel tracepoints vs kprobes (stable ABI vs function-specific)
      - TCP state machine transitions and connection lifecycle
      - BPF map-based runtime configuration for dynamic filtering
      - IPv4/IPv6 dual-stack address handling
      - Socket-based event correlation across state transitions
    skills:
      - Tracepoint attachment
      - Network protocol tracing
      - IPv4/IPv6 handling
      - Event correlation
      - Runtime filtering
    deliverables:
      - Tracepoint attachment for sock:inet_sock_set_state
      - IP and port extraction handling both IPv4 and IPv6
      - Connection duration tracking from ESTABLISHED to CLOSE
      - BPF map-based filtering by PID, port, and address
      - Formatted output with IP addresses, ports, states, and duration
    estimated_hours: 10

  - id: build-ebpf-tracer-m4
    name: Multi-Source Dashboard
    description: >-
      Build a terminal dashboard combining multiple eBPF data sources
      with per-CPU aggregation and runtime probe management.
    acceptance_criteria:
      - Runs multiple eBPF programs simultaneously (syscall latency, TCP connections, and one additional tracer such as scheduler or file I/O)
      - Uses BPF_MAP_TYPE_PERCPU_ARRAY and BPF_MAP_TYPE_PERCPU_HASH for high-performance data aggregation with minimal inter-CPU contention
      - Per-CPU map values are correctly summed across all CPUs in userspace before display
      - Terminal UI displays live-updating metrics including syscall histogram, active connections, and top-N processes by syscall count
      - Runtime configuration allows enabling/disabling individual probes and changing filters without restarting the tool
      - Performance overhead of running all probes simultaneously is measured and documented (should be <2% CPU overhead on idle system)
    pitfalls:
      - Per-CPU maps require reading and summing values from each CPU in userspace - forgetting this gives per-CPU values, not totals
      - Too many active probes on hot kernel paths (e.g., every syscall entry) can add measurable overhead to system performance - measure and document
      - Terminal UI refresh rate must balance between responsiveness and CPU overhead; 1-2 Hz is typically sufficient
      - Attaching and detaching probes at runtime requires careful lifecycle management to avoid resource leaks (file descriptors, maps)
      - Ring buffer consumers for multiple programs must be polled efficiently (epoll or equivalent), not busy-waited individually
    concepts:
      - Multi-program eBPF coordination and lifecycle management
      - Per-CPU data structures for contention-free aggregation
      - CPU-side summation in userspace for per-CPU map values
      - Runtime probe management (attach, detach, reconfigure)
      - Performance overhead measurement and budgeting
    skills:
      - Multi-source data aggregation
      - Terminal UI programming
      - Per-CPU map usage
      - Runtime probe management
      - Performance overhead analysis
    deliverables:
      - Multi-program eBPF manager handling concurrent probe lifecycle
      - Per-CPU map aggregation with correct cross-CPU summation
      - Terminal dashboard with live histograms, counters, and top-N displays
      - Runtime probe enable/disable and filter reconfiguration
      - Performance overhead measurement documenting CPU cost of active probes
    estimated_hours: 10

```
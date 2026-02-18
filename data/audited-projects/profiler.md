# AUDIT & FIX: profiler

## CRITIQUE
- **Kernel stack capture (M1 AC 2)**: Capturing kernel stack frames requires perf_event_open(2) or eBPF on Linux, NOT signal-based sampling. SIGPROF-based profiling only captures user-space stacks. The AC conflates two fundamentally different mechanisms. For an intermediate project, user-space only is appropriate.
- **Sampling rate range (M1 AC 3)**: 10KHz sampling is extremely aggressive and will cause significant overhead via signals. On modern Linux, perf_event_open can handle this, but SIGPROF at 10KHz is problematic. The AC should cap at a reasonable range or clarify the mechanism.
- **M2 DWARF parsing**: Full DWARF parsing is an enormous undertaking. The AC should clarify: use libdw/libdwarf/addr2line as a dependency, or implement from scratch? Implementing from scratch would be a separate advanced project.
- **M2 AC 4 JIT support**: JIT symbol resolution requires reading /tmp/perf-<pid>.map files (the jitdump convention). This is niche and shouldn't be a hard AC for an intermediate project.
- **M4 leak detection vs sampling**: The project mixes two fundamentally different profiling approaches: sampling (CPU profiler, signal-based) and instrumentation (memory profiler, function interposition). These are architecturally distinct. The AC should acknowledge this.
- **M4 recursive malloc**: Intercepting malloc while the interceptor itself needs to allocate memory is a classic deadlock/infinite-recursion bug. The AC doesn't require addressing this, but it's the #1 implementation challenge.
- **Estimated hours**: 45 hours with 4 milestones at 11 hours each. This is reasonable but the milestones are unequally complex. Symbol resolution with DWARF could easily take 20+ hours alone.
- **Language mismatch**: Prerequisites require 'C programming' but recommended languages include Python and Go, which have very different profiling APIs. Signal-based stack sampling is a C/Rust concept; Python has sys.setprofile/settrace.

## FIXED YAML
```yaml
id: profiler
name: CPU/Memory Profiler
description: Sampling profiler with flame graphs and heap allocation tracking
difficulty: intermediate
estimated_hours: "40-55"
essence: >
  Signal-driven periodic stack sampling with symbol resolution to capture
  and aggregate call hierarchies, combined with heap allocation interception
  and hierarchical flame graph visualization to identify CPU and memory
  performance bottlenecks.
why_important: >
  Building this teaches fundamental systems programming concepts like signal
  handling, stack unwinding, symbol resolution, and function interposition,
  while developing practical performance analysis skills critical for
  optimizing production systems.
learning_outcomes:
  - Implement timer-based sampling using ITIMER_PROF/SIGPROF for periodic stack capture
  - Collect user-space stack traces from signal handlers safely
  - Resolve instruction pointer addresses to function names using symbol tables
  - Aggregate stack samples and generate flame graph visualizations
  - Intercept heap allocation functions to track memory usage by call site
  - Detect potential memory leaks by tracking unfreed allocations
skills:
  - Stack sampling
  - Timer signals
  - Symbol resolution
  - Flame graphs
  - Memory tracking via interposition
  - Profile analysis
tags:
  - cpu
  - flame-graphs
  - instrumentation
  - intermediate
  - memory
  - sampling
  - tool
architecture_doc: architecture-docs/profiler/index.md
languages:
  recommended:
    - C
    - Rust
  also_possible:
    - Go
    - Python
resources:
  - name: Flame Graphs
    url: "https://www.brendangregg.com/flamegraphs.html"
    type: documentation
  - name: Linux perf Tutorial
    url: "https://perfwiki.github.io/main/tutorial/"
    type: tutorial
  - name: Go pprof Package
    url: "https://pkg.go.dev/runtime/pprof"
    type: documentation
  - name: Memory Profiling Introduction
    url: "https://easyperf.net/blog/2024/02/12/Memory-Profiling-Part1"
    type: article
prerequisites:
  - type: skill
    name: C or Rust programming
  - type: skill
    name: Unix process model (fork, signals, file descriptors)
  - type: skill
    name: Basic understanding of ELF and symbol tables
milestones:
  - id: profiler-m1
    name: Stack Sampling
    description: Periodically sample user-space call stacks using timer signals.
    estimated_hours: "10-12"
    concepts:
      - Signal-based interruption for periodic sampling
      - Stack frame walking via frame pointers
      - Async-signal-safe operations
      - Sampling bias and frequency tradeoffs
    skills:
      - Signal handling (SIGPROF)
      - Timer configuration (setitimer/timer_create)
      - Stack walking via frame pointers or libunwind
      - Async-signal-safe programming
    acceptance_criteria:
      - Configure ITIMER_PROF or timer_create to deliver SIGPROF at a configurable interval
      - Support sampling rates from 10Hz to 1000Hz; warn the user if rate exceeds 1000Hz due to overhead
      - In the signal handler, capture the user-space call stack as an array of instruction pointer addresses
      - Signal handler must be async-signal-safe: no malloc, no printf, no mutex locks; write to a pre-allocated ring buffer only
      - Store captured stack samples (array of IPs + sample count) in a pre-allocated lock-free data structure
      - Support targeting a specific process by PID (attach via ptrace or self-profiling via LD_PRELOAD)
      - Support targeting specific threads within a process when self-profiling
      - Measure and report profiler overhead as a percentage of target process wall time; must be < 5% at 100Hz
      - Note: This milestone captures USER-SPACE stacks only. Kernel stack capture requires perf_event_open and is out of scope.
    pitfalls:
      - "Signal handlers that call malloc, printf, or lock mutexes cause deadlocks or undefined behavior"
      - Binaries compiled without frame pointers (-fomit-frame-pointer, default in GCC -O2+) break frame-pointer-based stack walking; use libunwind as fallback
      - Sampling too frequently (> 1KHz) causes measurable overhead that distorts results
      - Race conditions accessing the sample buffer from both signal handler and reader thread; use lock-free ring buffer
      - Tight loops or blocking syscalls may bias sample distribution
    deliverables:
      - Timer-based SIGPROF delivery at configurable frequency
      - Async-signal-safe signal handler capturing instruction pointer chain
      - Pre-allocated ring buffer for lock-free sample storage
      - Frame-pointer-based stack walker with libunwind fallback
      - Overhead measurement reporting

  - id: profiler-m2
    name: Symbol Resolution
    description: Convert raw instruction pointer addresses to human-readable function names.
    estimated_hours: "10-12"
    concepts:
      - ELF symbol table parsing
      - Address-to-symbol mapping
      - ASLR and /proc/pid/maps
      - Name demangling
    skills:
      - Symbol table loading from ELF binaries
      - Address space layout parsing
      - C++ name demangling
      - Shared library symbol resolution
    acceptance_criteria:
      - Parse /proc/<pid>/maps to determine loaded libraries and their base addresses (for ASLR)
      - Load .symtab and .dynsym from the main executable and each loaded shared library
      - Resolve raw instruction pointer addresses to function names by finding the symbol whose address range contains the IP
      - For addresses not found in symbol tables, display as hex address with library name and offset (e.g., libc.so.6+0x12345)
      - Demangle C++ symbol names using __cxa_demangle or equivalent
      - "Optionally use addr2line, libdw, or libdwarf for source file and line number resolution (bonus, not required)"
      - Cache symbol lookups using a sorted array + binary search for O(log n) resolution per address
      - Handle stripped binaries gracefully (display hex offset when no symbols available)
    pitfalls:
      - ASLR randomizes library base addresses on each run; must parse /proc/pid/maps at capture time
      - Symbol tables may not cover all functions (stripped binaries, JIT code); handle gracefully
      - Inlined functions appear under the parent function's address range; DWARF is needed for inline info (bonus)
      - Slow symbol resolution can bottleneck the profiler if done synchronously; batch-resolve after capture
      - dlopen'd libraries may not appear in initial maps scan; refresh maps periodically
    deliverables:
      - /proc/<pid>/maps parser for library base addresses
      - ELF symbol table loader for executables and shared libraries
      - Address-to-function resolver with binary search
      - C++ name demangler
      - Symbol cache for repeated lookups
      - Graceful handling of unresolvable addresses

  - id: profiler-m3
    name: Flame Graph Generation
    description: Aggregate samples and visualize as interactive flame graphs.
    estimated_hours: "8-10"
    concepts:
      - Stack folding and aggregation
      - Hierarchical call tree construction
      - SVG generation
      - Flame graph semantics (x-axis is NOT time)
    skills:
      - Data aggregation algorithms
      - SVG generation
      - Interactive visualization
    acceptance_criteria:
      - Aggregate captured samples by unique call stack signature (sequence of resolved function names)
      - Output folded stack format compatible with Brendan Gregg's flamegraph.pl tool
      - Generate standalone SVG flame graph from aggregated data (no external tool dependency for basic output)
      - "X-axis width represents sample count (proportion of CPU time), NOT chronological time; document this clearly"
      - Support zoom (click to zoom into a subtree) and search (highlight matching function names) in generated SVG
      - Color-code frames by category: user code (warm colors), library code (cool colors), unknown (grey)
      - Support inverted flame graph (icicle chart) showing callee-focused view via command-line flag
      - Display top-N hottest functions as a text summary alongside the flame graph
    pitfalls:
      - Not normalizing stack depths causes unbalanced visual representation
      - Flame graph x-axis is alphabetically sorted within each level, NOT temporal; document this
      - Truncating long function names without tooltips makes graphs unreadable
      - Very deep stacks (> 100 frames) can make flame graphs too tall; consider truncation or collapsing
    deliverables:
      - Stack aggregation by unique call chain with sample counts
      - Folded stack format output (compatible with flamegraph.pl)
      - SVG flame graph generator with color coding
      - Zoom and search interactivity in SVG
      - Inverted icicle chart option
      - Text summary of top-N hottest functions

  - id: profiler-m4
    name: Memory Profiling
    description: Track heap allocations via function interposition to identify memory-heavy code paths.
    estimated_hours: "10-12"
    concepts:
      - Function interposition via LD_PRELOAD
      - Allocation metadata tracking
      - Leak detection by diffing alloc/free
      - Backtrace capture at allocation sites
    skills:
      - malloc/free interception
      - Allocation metadata management
      - Backtrace capture (backtrace() or libunwind)
      - Leak analysis
    acceptance_criteria:
      - Intercept malloc, calloc, realloc, and free via LD_PRELOAD shared library (or compile-time wrapping)
      - Solve the recursive-malloc problem: use a static fallback allocator (e.g., small static buffer with bump allocation) for allocations made by the interceptor itself before the real malloc is resolved
      - Record allocation size and call stack (backtrace) for each malloc/calloc/realloc call
      - Record free calls and match them to their corresponding allocation by pointer address
      - Track realloc correctly (free old + alloc new) to maintain accurate allocation map
      - At program exit (or on signal), report top-N allocation call sites ranked by total bytes allocated
      - Report potential leaks (allocations never freed before exit) with their originating call stacks
      - Generate an allocation flame graph (flame graph where width represents bytes allocated, not CPU samples)
      - Measure and report memory profiler overhead; metadata tracking should use < 10% of the profiled program's peak memory
    pitfalls:
      - Recursive malloc: your interceptor calls a function that calls malloc, which re-enters your interceptor—infinite recursion. Use a thread-local guard flag and static fallback allocator.
      - realloc with NULL pointer is equivalent to malloc; realloc to size 0 may be equivalent to free (implementation-defined)
      - Thread-local storage allocations and static constructor allocations happen before LD_PRELOAD initializes; some allocations will be missed
      - Excessive metadata per allocation (full backtrace) can consume significant memory; limit backtrace depth (e.g., 16 frames)
      - Long-lived allocations (e.g., global data structures) are NOT leaks but will appear as unfreed at exit; distinguish by lifetime
    deliverables:
      - LD_PRELOAD shared library intercepting malloc/calloc/realloc/free
      - Recursive-malloc guard with static fallback allocator
      - Per-allocation metadata recording (size, backtrace, timestamp)
      - Allocation/free matching by pointer address
      - Top-N allocation site report by total bytes
      - Leak report (unfreed allocations at exit)
      - Allocation flame graph
```
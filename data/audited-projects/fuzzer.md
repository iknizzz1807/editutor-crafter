# AUDIT & FIX: fuzzer

## CRITIQUE
- **Missing Deterministic vs Non-Deterministic Phase Distinction**: AFL's efficiency comes from running deterministic mutations (walking bit flips, arithmetic, interesting values) first for systematic coverage, then switching to non-deterministic 'havoc' mode for random combinations. This two-phase strategy is a key architectural decision not mentioned in the milestones.
- **Coverage Instrumentation Underspecified**: M2 says 'instrument target for code coverage using llvm-cov or similar tooling' but llvm-cov is a coverage reporting tool, NOT an instrumentation tool. AFL uses compiler passes (afl-gcc/afl-clang-fast) with custom instrumentation, or __sanitizer_cov* callbacks. This is technically inaccurate.
- **Shared Memory Bitmap Not Explained**: The shared memory bitmap (typically 64KB) uses a hash of (previous_block, current_block) to track edges. The hash collision rate and bitmap size tradeoff are critical design decisions not addressed.
- **Fork Server Missing**: AFL's major performance optimization is the fork server: the target starts once, then forks for each test case, avoiding repeated execve() overhead. This is absent and is the single biggest performance differentiator (10-100x speedup).
- **Sanitizer Integration Absent**: Modern fuzzers use AddressSanitizer (ASAN), MemorySanitizer (MSAN), and UndefinedBehaviorSanitizer (UBSAN) to detect bugs beyond crashes. This is not mentioned.
- **Energy Scheduling Underspecified**: M5 mentions 'coverage-weighted selection' but doesn't address power schedules (how many mutations to apply to each seed). AFL uses exponential power schedules; AFLFast uses explore/exploit schedules.
- **Corpus Minimization Algorithm Not Specified**: M4 says 'minimize corpus' but doesn't mention the algorithm (greedy set cover: keep minimum inputs covering all edges).
- **Overall Structure Is Sound**: The milestone progression (execute → instrument → mutate → manage → orchestrate) is logical and follows the AFL architecture correctly.

## FIXED YAML
```yaml
id: fuzzer
name: Coverage-Guided Fuzzing Framework
description: >-
  Coverage-guided fuzzer implementing compile-time instrumentation, deterministic
  and non-deterministic mutation strategies, fork server optimization, corpus
  management, and crash triage.
difficulty: advanced
estimated_hours: "55-70"
essence: >-
  Compile-time code instrumentation inserting edge-tracking callbacks into
  target programs, fork-server-based execution for high throughput, two-phase
  mutation (deterministic walking mutations then non-deterministic havoc),
  shared-memory coverage bitmap feedback for input selection, and corpus
  management with minimization and crash deduplication.
why_important: >-
  Building a fuzzer teaches you low-level program analysis, instrumentation
  techniques, and security vulnerability discovery—skills critical for
  security engineering, compiler tooling, and building robust systems that
  handle untrusted input.
learning_outcomes:
  - Implement process execution harness with fork, exec, signal handling, and timeout enforcement
  - Design fork server optimization avoiding repeated execve() overhead for 10-100x throughput improvement
  - Instrument target programs at compile time to track edge coverage via shared memory bitmap
  - Implement deterministic mutation phase (walking bit flips, arithmetic, interesting values)
  - Implement non-deterministic havoc mutation phase with random combined mutations
  - Build coverage-guided corpus management with greedy set-cover minimization
  - Design crash triage with stack-hash-based deduplication and input minimization
  - Orchestrate the fuzzing loop with energy-based scheduling and adaptive mutation selection
skills:
  - Compile-Time Instrumentation
  - Fork Server Optimization
  - Shared Memory Coverage Bitmap
  - Deterministic Mutation Strategies
  - Non-Deterministic Havoc Mutation
  - Corpus Management & Minimization
  - Crash Detection & Triage
  - Parallel Fuzzing
tags:
  - advanced
  - coverage
  - crash-detection
  - mutation
  - security
  - testing
  - tool
  - instrumentation
architecture_doc: architecture-docs/fuzzer/index.md
languages:
  recommended:
    - C
    - Rust
  also_possible:
    - Go
resources:
  - name: The Fuzzing Book
    url: https://www.fuzzingbook.org/
    type: book
  - name: AFL++ Documentation
    url: https://aflplus.plus/
    type: documentation
  - name: AFL Technical Whitepaper
    url: https://lcamtuf.coredump.cx/afl/technical_details.txt
    type: reference
  - name: libFuzzer LLVM Docs
    url: https://llvm.org/docs/LibFuzzer.html
    type: documentation
  - name: Google Fuzzing Tutorials
    url: https://github.com/google/fuzzing/blob/master/tutorial/libFuzzerTutorial.md
    type: tutorial
prerequisites:
  - type: skill
    name: C programming and compilation
  - type: skill
    name: Process management (fork, exec, signals, wait)
  - type: skill
    name: Shared memory (shmget/shmat or mmap)
  - type: skill
    name: Basic understanding of x86-64 or target architecture
milestones:
  - id: fuzzer-m1
    name: Target Execution & Fork Server
    description: >-
      Build the execution harness for running target programs with test
      inputs, including a fork server for high-throughput execution.
    acceptance_criteria:
      - Execute target program via fork() + execve() with test input delivered via stdin, file argument, or command-line argument
      - Capture exit code, signal number (SIGSEGV=11, SIGABRT=6, SIGBUS=7), and execution time for each run
      - Kill target process after configurable timeout (default 1 second) via SIGKILL to prevent hangs
      - Set resource limits on target: memory (RLIMIT_AS), CPU time (RLIMIT_CPU), file size (RLIMIT_FSIZE)
      - Redirect target stderr to capture crash diagnostics (assertion messages, sanitizer output)
      - Implement fork server: target process starts once, then forks for each test case; parent communicates via control pipe
      - Fork server uses a pipe pair for synchronization: fuzzer signals 'go', target forks and runs, child exit status returned
      - Measure and report executions per second; fork server should achieve >1000 exec/sec on simple targets
      - Isolate target process from fuzzer state (separate PID, file descriptors closed except stdin/control pipe)
    pitfalls:
      - Without fork server, execve() overhead dominates; fork server provides 10-100x speedup
      - Must set SIGCHLD handler or waitpid() correctly to avoid zombie processes
      - RLIMIT_AS must be set high enough for ASAN-instrumented binaries (ASAN uses shadow memory)
      - Target may block on stdin if input delivery mode is wrong; use file-based input for targets that seek()
      - Fork server protocol must handle target crashes gracefully; respawn fork server if child's parent dies
      - Stderr capture buffer can overflow for verbose targets; limit capture size
    concepts:
      - Fork server architecture for high-throughput execution
      - Process forking, execution, and signal handling
      - Resource limit enforcement via setrlimit
      - Inter-process communication via pipes
    skills:
      - Process execution and lifecycle management
      - Fork server protocol implementation
      - Signal handling and timeout enforcement
      - Resource limit configuration
    deliverables:
      - Basic execution harness with fork/exec, signal capture, and timeout
      - Fork server implementation with pipe-based synchronization protocol
      - Configurable input delivery (stdin, file, argv)
      - Resource limit enforcement (memory, CPU time, file size)
      - Execution speed measurement (exec/sec reporting)
      - Crash detection classifying outcomes (clean exit, crash, timeout, OOM)
    estimated_hours: "10-14"

  - id: fuzzer-m2
    name: Coverage Instrumentation & Tracking
    description: >-
      Instrument target programs at compile time to track edge coverage
      via a shared memory bitmap, and detect new coverage from test inputs.
    acceptance_criteria:
      - Implement compile-time instrumentation pass that inserts edge-tracking code at each basic block boundary
      - Instrumentation computes edge hash as (previous_block_id XOR current_block_id) and increments shared_memory[hash % MAP_SIZE]
      - Shared memory bitmap (default 64KB) is allocated via shmget/shmat or mmap and shared between fuzzer and target
      - Previous_block_id is stored in a thread-local variable and updated to (current_block_id >> 1) after each edge
      - After each execution, compare target's coverage bitmap against global coverage bitmap to detect new edges
      - New edge detected when any byte in target bitmap has a value in a new hit-count bucket (1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+)
      - Clear target coverage bitmap before each execution (or use fork to get clean copy)
      - Provide wrapper compiler script (e.g., fuzzer-cc) that invokes real compiler with instrumentation added
      - Verify instrumentation by running two different inputs and confirming different coverage bitmaps
    pitfalls:
      - Bitmap size 64KB is a tradeoff: larger = fewer hash collisions but more cache misses; 64KB is the AFL sweet spot
      - Hash collisions (different edges mapping to same bitmap slot) create blind spots; use prime-based block IDs to reduce collisions
      - Do NOT instrument library code (libc, etc.) unless specifically needed; it adds noise and slows execution
      - Coverage bitmap must be reset between executions; fork server handles this naturally (each fork gets clean bitmap)
      - Hit-count bucketing (not raw counts) is important: distinguishes 'hit once' from 'hit many times' without overflow sensitivity
      - Using llvm-cov or gcov for coverage is too slow for fuzzing (file I/O overhead); shared memory is required
    concepts:
      - Edge coverage vs basic block coverage (edge is more precise)
      - Shared memory bitmap for coverage feedback
      - Hit-count bucketing for coverage sensitivity
      - Compile-time vs runtime instrumentation tradeoffs
    skills:
      - Compiler instrumentation (LLVM pass or GCC plugin or manual source instrumentation)
      - Shared memory IPC
      - Coverage bitmap analysis and comparison
      - Compile wrapper scripting
    deliverables:
      - Compile-time instrumentation inserting edge-tracking callbacks at basic block boundaries
      - Shared memory bitmap allocation and management (64KB default)
      - Edge hash computation using XOR of previous and current block IDs
      - Coverage comparison detecting new edges via hit-count bucketing
      - Wrapper compiler script for instrumenting target programs
      - Coverage visualization tool showing exercised edges
    estimated_hours: "10-14"

  - id: fuzzer-m3
    name: Mutation Engine (Deterministic & Havoc)
    description: >-
      Implement two-phase mutation: deterministic walking mutations for
      systematic exploration, then non-deterministic havoc mode for
      random combined mutations.
    acceptance_criteria:
      - "Deterministic phase (applied in order for each seed input):"
      - Walking bit flips: flip 1, 2, and 4 bits at each bit position sequentially
      - Walking byte flips: flip 1, 2, and 4 bytes at each byte position
      - Arithmetic mutations: add/subtract values 1-35 to each 1/2/4-byte integer field (both endiannesses)
      - Interesting value substitution: replace 1/2/4-byte fields with known interesting values (0, 1, -1, MAX_INT, MIN_INT, etc.)
      - Dictionary token insertion and replacement: insert/replace known tokens (from user-provided dictionary or auto-discovered) at each position
      - "Non-deterministic havoc phase (applied after deterministic phase completes for a seed):"
      - Havoc applies random combinations of mutations: bit flip, byte set, arithmetic, block delete, block insert, block overwrite, splice with another corpus input
      - Number of havoc mutations per input is configurable (default: energy-based, more for high-value seeds)
      - Track which mutation strategies discovered new coverage to inform adaptive selection
      - Support user-provided dictionary files for format-aware token insertion
    pitfalls:
      - Deterministic phase is slow but thorough; it should complete fully for each seed before moving to havoc
      - Arithmetic mutations must try BOTH little-endian and big-endian interpretations
      - Interesting values are architecture-specific: include both 32-bit and 64-bit boundary values
      - Over-mutating in havoc mode produces inputs too different from parent, losing structural validity
      - Dictionary tokens for structured formats (XML, JSON, SQL) dramatically improve format-aware fuzzing
      - Splice mutations (combining two corpus inputs) require compatible parent selection
    concepts:
      - Two-phase mutation strategy (deterministic then non-deterministic)
      - Walking mutations for systematic bit/byte exploration
      - Interesting values for boundary condition testing
      - Havoc mode for rapid random exploration
      - Dictionary-based format-aware mutation
    skills:
      - Mutation strategy implementation
      - Binary data manipulation at bit and byte level
      - Endianness-aware integer manipulation
      - Adaptive strategy selection
    deliverables:
      - Deterministic mutation suite: walking bit flips, byte flips, arithmetic, interesting values
      - Dictionary mutation: token insertion and replacement from user-provided wordlist
      - Havoc mutation combining random operations (flip, set, delete, insert, overwrite, splice)
      - Mutation strategy tracker recording which strategies found new coverage
      - Energy-based mutation count: high-value seeds get more havoc iterations
    estimated_hours: "10-14"

  - id: fuzzer-m4
    name: Corpus Management & Crash Triage
    description: >-
      Manage the input corpus with coverage-based selection, greedy set-cover
      minimization, and crash deduplication via stack-hash triage.
    acceptance_criteria:
      - Add inputs to corpus when they discover new edge coverage (new bitmap bytes or new hit-count buckets)
      - Store corpus inputs as individual files in a persistent directory for resumable fuzzing
      - Support loading initial seed corpus from user-provided directory
      - Implement corpus minimization using greedy set-cover: find minimum subset of inputs covering all observed edges
      - Implement input minimization (test case reduction): shrink crash-triggering input to smallest reproducing version
      - Input minimization uses binary search + sequential byte removal while verifying crash still reproduces
      - Deduplicate crashes by computing stack hash from crash signal + top N return addresses (or unique edge path to crash)
      - Export crash-reproducing inputs to dedicated crash directory with metadata (signal, stack hash, timestamp)
      - Assign energy scores to corpus inputs: prioritize inputs covering rare edges or recently discovered paths
      - Track corpus statistics: total inputs, total edges covered, edges per input, coverage growth over time
    pitfalls:
      - Corpus grows unbounded without periodic minimization; schedule minimization runs during idle periods
      - Greedy set-cover is NP-hard optimally but greedy algorithm achieves good approximation
      - Input minimization can accidentally remove the crash trigger; verify crash reproduces after each reduction step
      - Stack-hash deduplication may group different bugs (hash collision) or split same bug (ASLR varies addresses)
      - Don't discard inputs covering rare edges even if they're 'low energy'; they may be the only path to deep bugs
      - Parallel fuzzer instances need corpus synchronization; periodic scan of shared corpus directory
    concepts:
      - Greedy set-cover for corpus minimization
      - Delta debugging for input minimization
      - Stack-hash crash deduplication
      - Energy-based seed scheduling
    skills:
      - Corpus storage and management
      - Set-cover approximation algorithms
      - Test case minimization techniques
      - Crash classification and deduplication
    deliverables:
      - Coverage-based corpus addition storing new-coverage inputs to disk
      - Seed corpus loader for user-provided initial inputs
      - Corpus minimization via greedy set-cover algorithm
      - Input minimization reducing crash inputs to smallest reproducing version
      - Crash deduplication via stack-hash classification
      - Crash output directory with reproduction metadata
      - Energy scoring for corpus inputs prioritizing rare coverage
    estimated_hours: "10-14"

  - id: fuzzer-m5
    name: Fuzzing Loop & Orchestration
    description: >-
      Implement the main fuzzing loop orchestrating seed selection, mutation
      phase progression, execution, coverage analysis, and reporting.
    acceptance_criteria:
      - Main loop: select seed from corpus → apply mutations → execute target → analyze coverage → update corpus → repeat
      - Seed selection uses energy-based scheduling: seeds with more unexplored potential get more execution time
      - For each seed: complete deterministic mutation phase first, then apply havoc phase with energy-proportional iterations
      - Track and display real-time statistics: exec/sec, total executions, unique crashes, unique edges, corpus size, last new edge timestamp
      - Support graceful pause and resume: save fuzzer state (current seed, phase, position) to enable resumable campaigns
      - Implement parallel fuzzing with multiple worker processes sharing corpus via filesystem directory
      - Workers periodically scan shared corpus directory to import new inputs discovered by other workers
      - Handle worker crashes gracefully without losing corpus or state
      - Campaign completion detection: alert when no new coverage found for configurable duration (stall detection)
      - Total coverage should increase monotonically over time; verify no coverage regression
    pitfalls:
      - Don't starve low-energy seeds; ensure all corpus entries eventually get mutated (minimum energy floor)
      - Parallel workers must handle filesystem race conditions when reading/writing corpus files
      - State persistence is critical: a crash of the fuzzer itself should not lose hours of work
      - exec/sec varies wildly based on target complexity; don't compare across different targets
      - Stall detection (no new coverage) doesn't mean all bugs found; may need different seeds or dictionaries
      - Deterministic phase must track progress per-seed to resume correctly after fuzzer restart
    concepts:
      - Energy-based seed scheduling (power schedules)
      - Deterministic-then-havoc phase progression per seed
      - Parallel fuzzing with corpus synchronization
      - Fuzzing campaign state management
    skills:
      - Event loop and orchestration design
      - Real-time statistics collection and display
      - Parallel process coordination
      - State serialization for resume capability
    deliverables:
      - Main fuzzing loop with seed selection, mutation, execution, and corpus update
      - Energy-based seed scheduler with deterministic-then-havoc phase progression
      - Real-time statistics display (exec/sec, coverage, crashes, corpus size)
      - State persistence for resumable fuzzing campaigns
      - Parallel fuzzing with multiple workers and corpus synchronization
      - Stall detection alerting when coverage growth plateaus
    estimated_hours: "10-14"
```
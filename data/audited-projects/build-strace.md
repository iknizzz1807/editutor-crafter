# AUDIT & FIX: build-strace

## CRITIQUE
- **Architecture specificity is critical**: The entire project is x86_64-specific (register names like rax, rdi, etc. are x86_64 ABI). This must be explicitly stated as a requirement, not left ambiguous. ARM and other architectures have completely different syscall conventions.
- **PTRACE_PEEKDATA string reading is non-trivial**: The audit correctly identifies that reading strings requires an iterative word-by-word loop. A word is typically 8 bytes on x86_64, and you must scan each word for a null terminator byte. This is a significant implementation challenge that deserves its own AC.
- **Entry/exit state tracking is the #1 pitfall**: ptrace stops the tracee twice per syscall (once at entry, once at exit). The tracer must track whether it's at entry or exit to know whether to read arguments or the return value. This is mentioned as a pitfall in M1 but should be an explicit AC.
- **Signal injection vs signal delivery confusion**: When a traced process receives a signal, ptrace reports it as a stop. The tracer must decide whether to inject the signal back or suppress it. This is a critical correctness issue not mentioned in any AC.
- **M4 timing measurement**: The AC says 'using clock_gettime' but doesn't specify CLOCK_MONOTONIC. Using CLOCK_REALTIME is subject to NTP adjustments and can produce negative durations.
- **Missing AC for attach-to-running-process**: Real strace can attach to already-running processes with PTRACE_ATTACH. This is missing from all milestones.
- **Error return detection is platform-specific**: On x86_64, syscall error is indicated by return value in range [-4096, -1]. This is a crucial detail for correctly printing errno values.
- **PTRACE_PEEKDATA returns the word in the ERRNO path**: On error, PTRACE_PEEKDATA returns -1 and sets errno. But -1 might also be valid data. Must check errno explicitly (set to 0 before call, check after).

## FIXED YAML
```yaml
id: build-strace
name: System Call Tracer (strace clone)
description: ptrace-based syscall interception and decoding for x86_64 Linux
difficulty: intermediate
estimated_hours: "22-35"
essence: >
  Process introspection through kernel-mediated interception of system calls on x86_64 Linux,
  requiring register manipulation, word-by-word argument decoding from tracee memory, and
  process state management across fork/exec boundaries.
why_important: >
  Understanding ptrace-based tracing reveals how debuggers, profilers, and security tools
  interact with the kernel to observe program behavior — foundational knowledge for systems
  programming and tooling development.
learning_outcomes:
  - Implement ptrace-based process attachment and syscall interception using PTRACE_SYSCALL
  - Track syscall entry/exit state to correctly read arguments vs return values
  - Decode syscall arguments by reading registers and dereferencing pointers from tracee memory word-by-word
  - Handle multi-process tracing across fork and exec with PTRACE_O_TRACEFORK
  - Build syscall filtering and timing statistics
  - Handle signal delivery to traced processes correctly
skills:
  - ptrace System Call
  - x86_64 Syscall ABI
  - Process Control
  - Register Manipulation
  - Remote Memory Inspection
  - Signal Handling in Traced Processes
  - Multi-process Debugging
tags:
  - ptrace
  - debugging
  - system-calls
  - linux
  - x86_64
  - process-control
architecture_doc: architecture-docs/build-strace/index.md
languages:
  recommended:
    - C
    - Rust
  also_possible:
    - Go
resources:
  - name: ptrace(2) man page
    url: https://man7.org/linux/man-pages/man2/ptrace.2.html
    type: reference
  - name: How strace works
    url: https://blog.packagecloud.io/how-does-strace-work/
    type: article
  - name: x86_64 syscall table
    url: https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/
    type: reference
prerequisites:
  - type: project
    name: process-spawner
  - type: project
    name: signal-handler
milestones:
  - id: build-strace-m1
    name: Basic ptrace Syscall Intercept
    description: >
      Use ptrace to trace a child process, stopping at each syscall entry and exit.
      Target architecture: x86_64 Linux.
    acceptance_criteria:
      - Fork a child process; child calls PTRACE_TRACEME then exec's the target program
      - Parent uses PTRACE_SYSCALL + waitpid loop to stop the child at each syscall boundary
      - Explicitly track entry/exit state with a toggle flag: on entry, read syscall number from orig_rax register; on exit, read return value from rax register
      - Print syscall number and return value for each intercepted syscall (e.g., "syscall(0) = 5")
      - Detect error returns: on x86_64, return values in range [-4096, -1] indicate error; display as -1 ERRNO (e.g., -1 ENOENT")"
      - Handle signals delivered to the tracee: when waitpid reports a signal-delivery stop (not a syscall stop), re-inject the signal using PTRACE_SYSCALL with the signal number, do not suppress it
      - Handle tracee exit cleanly: detect WIFEXITED/WIFSIGNALED from waitpid and exit the tracer
    pitfalls:
      - ptrace stops twice per syscall (entry and exit); failing to toggle state causes argument reads on exit and return-value reads on entry, producing garbage output
      - Register layout is x86_64-specific: orig_rax for syscall number, rax for return value; on i386 it's orig_eax/eax, on ARM it's completely different
      - PTRACE_TRACEME must be called in the child BEFORE exec(); after exec the tracee stops automatically on the first signal (SIGTRAP)
      - Signal stops and syscall stops are both reported via waitpid; must distinguish them using PTRACE_O_TRACESYSGOOD (sets bit 7 of signal in status) or by checking the signal number
      - Suppressing signals intended for the tracee (by passing 0 to PTRACE_SYSCALL instead of the signal number) breaks the traced program
    concepts:
      - ptrace system call and tracing lifecycle
      - x86_64 syscall ABI (register conventions)
      - Syscall entry/exit state tracking
      - Signal delivery in traced processes
    skills:
      - ptrace API usage
      - Process control with waitpid
      - x86_64 register inspection
      - Signal handling in tracer context
    deliverables:
      - Child process creation with PTRACE_TRACEME + exec
      - PTRACE_SYSCALL + waitpid interception loop
      - Entry/exit toggle with correct register reads
      - Error return detection with errno mapping
      - Signal re-injection for non-syscall stops
    estimated_hours: "5-8"

  - id: build-strace-m2
    name: Argument Decoding
    description: >
      Map syscall numbers to names and decode arguments including strings
      read word-by-word from tracee memory.
    acceptance_criteria:
      - Map x86_64 syscall numbers to human-readable names using a syscall table (at least the 50 most common syscalls)
      - Decode integer arguments from x86_64 argument registers: rdi, rsi, rdx, r10, r8, r9 (in that order)
      - Read string arguments (e.g., file paths for open/stat) from tracee memory using PTRACE_PEEKDATA in a word-by-word loop until a null terminator byte is found within a read word
      - Handle PTRACE_PEEKDATA correctly: set errno to 0 before call, check errno after (since -1 is both a valid data value and an error indicator)
      - Truncate displayed strings at a configurable maximum length (default 32 bytes) with "..." suffix
      - Format output matching strace style: syscall_name(arg1, arg2, ...) = return_value (e.g., 'open(/etc/passwd", O_RDONLY) = 3')"
      - Decode flag arguments for common syscalls: open() flags (O_RDONLY, O_WRONLY, O_CREAT), mmap() prot and flags
    pitfalls:
      - PTRACE_PEEKDATA reads one word (8 bytes on x86_64) at a time; string reading requires looping and scanning each word for a null byte, handling the case where the null is in the middle of a word
      - PTRACE_PEEKDATA returns -1 on error but -1 (0xFFFFFFFFFFFFFFFF) is also valid data; must pre-clear errno and check it to distinguish error from data
      - Some syscalls have different argument semantics depending on flags (e.g., clone's arguments vary by flag bits); decode common cases, skip exotic ones
      - String pointers may be NULL (e.g., execve with NULL envp); must check for NULL before attempting PEEKDATA
      - Buffer overread: a string in the tracee might not be null-terminated within readable memory; set a maximum read length to prevent infinite loops
    concepts:
      - x86_64 syscall ABI argument registers
      - Remote process memory reading (PTRACE_PEEKDATA)
      - Word-by-word string extraction with null terminator scan
      - Flag decoding with bitmask analysis
    skills:
      - Remote memory access across address spaces
      - Syscall ABI knowledge and argument interpretation
      - String marshalling from word-aligned reads
      - Bitmask flag decoding
    deliverables:
      - Syscall number-to-name mapping table for x86_64
      - Register-based argument extraction per syscall signature
      - Word-by-word string reader with null terminator detection and length limit
      - Flag decoder for common syscalls (open, mmap, etc.)
      - strace-style formatted output
    estimated_hours: "6-8"

  - id: build-strace-m3
    name: Multi-Process and Fork Following
    description: >
      Handle traced processes that fork or exec, tracing all descendants.
    acceptance_criteria:
      - Set PTRACE_O_TRACEFORK, PTRACE_O_TRACEVFORK, and PTRACE_O_TRACECLONE options on the tracee so child processes are automatically traced
      - Use waitpid(-1, ...) to catch events from ANY traced process (not just the original)
      - Detect fork/vfork/clone events via PTRACE_EVENT_FORK/VFORK/CLONE and retrieve new child PID with PTRACE_GETEVENTMSG
      - Tag all output lines with the PID of the process that made the syscall (e.g., "[pid 1234] open(...) = 3")
      - Handle PTRACE_EVENT_EXEC correctly: the tracee's memory map changes entirely after exec; reset any cached state for that PID
      - Handle multiple simultaneously traced processes without losing events or mixing up entry/exit state between PIDs
      - Maintain per-PID state tracking (entry/exit toggle, current syscall number) in a hash map
    pitfalls:
      - Must set PTRACE_O_TRACEFORK BEFORE the tracee calls fork, otherwise the child runs untraced and may exit before we can attach
      - waitpid(-1) returns events from any traced child; must dispatch to the correct per-PID state based on the returned PID
      - PTRACE_EVENT_EXEC stops look like SIGTRAP stops; must use status>>16 to extract the event type
      - After exec, the tracee has a new memory map; any cached memory pointers or string addresses from before exec are invalid
      - A traced child that forks creates a grandchild that must also be traced; this requires the PTRACE_O_TRACE* options to be inherited
    concepts:
      - Multi-process ptrace tracing
      - PTRACE_EVENT dispatching
      - Per-PID state management
      - Fork/exec lifecycle in traced processes
    skills:
      - Multi-process coordination
      - Event-driven process management
      - Per-process state tracking with hash maps
      - ptrace option configuration
    deliverables:
      - PTRACE_O_TRACE* option setup for automatic child tracing
      - waitpid(-1) loop handling events from all traced processes
      - Per-PID state map tracking entry/exit toggle and current syscall
      - PID-tagged output for multi-process traces
      - Correct exec handling resetting per-PID cached state
    estimated_hours: "5-8"

  - id: build-strace-m4
    name: Filtering and Statistics
    description: >
      Add syscall filtering by name and generate timing/frequency statistics.
    acceptance_criteria:
      - Filter output to show only specified syscalls by name (e.g., -e trace=open,read,write); unmatched syscalls are still traced (for dependency tracking) but not printed
      - Collect per-syscall wall-clock timing using clock_gettime(CLOCK_MONOTONIC): record time at entry, compute duration at exit
      - Generate summary table (like strace -c) showing: syscall name, call count, error count, cumulative time, percentage of total time; sorted by cumulative time descending
      - Support -o filename flag to redirect trace output to a file instead of stderr
      - Support -p PID flag to attach to an already-running process using PTRACE_ATTACH (instead of fork+exec)
      - Detach cleanly from attached process on tracer exit or SIGINT using PTRACE_DETACH
    pitfalls:
      - Timing must use CLOCK_MONOTONIC not CLOCK_REALTIME; CLOCK_REALTIME is subject to NTP adjustments that can produce negative durations
      - Error detection requires checking if the x86_64 return value is in [-4096, -1]; simply checking < 0 misclassifies legitimate negative return values
      - PTRACE_ATTACH sends SIGSTOP to the target; must waitpid for the SIGSTOP before beginning PTRACE_SYSCALL tracing
      - File output with -o must handle concurrent writes from multiple traced processes; use per-line write() calls or a mutex
      - Signal filter: when attached to a running process, existing signal handlers must not be disrupted; always re-inject signals
    concepts:
      - Syscall filtering and pattern matching
      - High-resolution timing with CLOCK_MONOTONIC
      - Statistical aggregation and reporting
      - Process attachment vs fork-based tracing
    skills:
      - Performance profiling and timing
      - Statistical data collection
      - PTRACE_ATTACH lifecycle management
      - Output management and file redirection
    deliverables:
      - Syscall name filter with -e trace=name,name syntax
      - Per-syscall timing with CLOCK_MONOTONIC
      - Summary statistics table (count, errors, time, percentage)
      - File output option with -o flag
      - PTRACE_ATTACH support for tracing running processes with -p PID
      - Clean detach on exit/signal
    estimated_hours: "6-10"
```
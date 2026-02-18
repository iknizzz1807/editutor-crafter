# AUDIT & FIX: build-jit

## CRITIQUE
- Critical security/correctness issue: Milestone 1 AC says 'Allocate executable memory using mmap with PROT_READ|PROT_WRITE|PROT_EXEC'. On modern hardened systems (macOS, OpenBSD, many Linux configs with SELinux), simultaneous W+X is forbidden (W^X policy). Must map as RW, write code, then mprotect to RX. This is a fundamental correctness issue, not just a pitfall.
- Milestone 1 pitfall mentions W^X but the AC contradicts it by requiring simultaneous RWX. The AC and pitfall are in direct conflict.
- Missing deoptimization/guard milestone: JITs that do speculative optimization (M4 mentions constant folding, hot path) MUST have a deoptimization story. If a guard fails, the JIT must fall back to the interpreter. This is not mentioned anywhere.
- Milestone 2 pitfall about 'cqo before idiv' is correct for signed division but should also mention that unsigned division uses 'xor rdx, rdx' before 'div', not 'cqo'.
- Milestone 3 AC 'Support function calls between JIT-compiled and interpreted code' is vague — no mention of how the transition works (trampoline mechanics, state saving/restoring).
- Milestone 4 mentions 'constant folding' and 'dead code elimination' as optimizations during JIT but these are typically done at bytecode/IR level, not during machine code emission. The milestone conflates IR-level optimization with code generation.
- No milestone covers handling of exceptions or error paths in JIT-compiled code.
- Estimated hours (40-60 total) is extremely aggressive for a correct JIT compiler with calling conventions, hot path detection, and optimization.
- Prerequisites list 'bytecode-vm' and 'bytecode-compiler' projects but the milestones seem to assume building from scratch rather than extending existing work. This should be clarified.
- No mention of instruction cache coherence (icache flush) after writing code, which is required on ARM and sometimes needed on x86 with certain mmap patterns.

## FIXED YAML
```yaml
id: build-jit
name: JIT Compiler
description: >-
  Build a JIT compiler backend that translates bytecode to native x86-64
  machine code at runtime, with register allocation, calling conventions,
  hot path detection, and deoptimization guards.
difficulty: expert
estimated_hours: "60-90"
essence: >-
  Runtime translation of bytecode to native x86-64 machine code with W^X
  compliant executable memory management, register allocation mapping
  stack-based bytecode to registers, System V AMD64 ABI calling convention
  compliance, profiling-guided tiered compilation, and deoptimization guards
  for safe fallback to interpretation.
why_important: >-
  Building a JIT compiler provides deep understanding of the boundary between
  high-level languages and hardware. It teaches how production VMs (V8, JVM
  HotSpot, PyPy, LuaJIT) achieve performance through runtime code generation,
  and exposes the critical interplay between speculation, guards, and
  deoptimization.
learning_outcomes:
  - Implement x86-64 instruction encoding with REX prefixes and ModR/M bytes
  - Manage executable memory with W^X compliance (mmap RW, mprotect to RX)
  - Design register allocation for translating stack-based bytecode to register-based native code
  - Build function prologues/epilogues following System V AMD64 ABI
  - Implement profiling counters and hot path detection for tiered compilation
  - Build deoptimization guards that safely fall back to interpreter on speculation failure
  - Debug machine code using disassemblers (objdump, capstone) and runtime analysis
  - Flush instruction cache where required for correctness
skills:
  - x86-64 Assembly and Encoding
  - Executable Memory Management
  - Calling Conventions (System V AMD64 ABI)
  - Runtime Code Generation
  - Register Allocation
  - Binary Encoding (REX, ModR/M, SIB)
  - Performance Profiling
  - Low-level Debugging
tags:
  - jit-compilation
  - x86-64
  - code-generation
  - runtime
  - compilers
  - expert
  - build-from-scratch
architecture_doc: architecture-docs/build-jit/index.md
languages:
  recommended:
    - C
    - Rust
    - Zig
  also_possible: []
resources:
  - name: Adventures in JIT Compilation""
    url: https://eli.thegreenplace.net/2017/adventures-in-jit-compilation-part-1-an-interpreter/
    type: article
  - name: x86-64 Instruction Reference""
    url: https://www.felixcloutier.com/x86/
    type: reference
  - name: Intel 64 and IA-32 Software Developer Manuals""
    url: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
    type: reference
prerequisites:
  - type: project
    name: bytecode-vm
  - type: project
    name: bytecode-compiler
  - type: skill
    name: x86-64 assembly basics
milestones:
  - id: build-jit-m1
    name: "x86-64 Code Emitter with W^X Memory"
    description: >-
      Build a machine code emitter that generates x86-64 instructions into
      executable memory with proper W^X (Write XOR Execute) compliance.
    acceptance_criteria:
      - >-
        Allocate memory using mmap with PROT_READ|PROT_WRITE (no PROT_EXEC)
        for code writing.
      - >-
        After code emission is complete, change page permissions to
        PROT_READ|PROT_EXEC using mprotect, removing write permission
        (W^X compliance).
      - >-
        On platforms requiring it (macOS ARM64 via Rosetta, native ARM),
        flush the instruction cache after permission change using
        __builtin___clear_cache or equivalent.
      - >-
        Emit basic x86-64 instructions (mov reg/imm, add, sub, imul, ret)
        as correctly encoded byte sequences with REX prefix for 64-bit
        operands.
      - >-
        Support all 16 general-purpose registers (rax-r15) with correct
        REX.B/REX.R encoding for registers r8-r15.
      - >-
        ModR/M byte encoding correctly handles register-to-register,
        register-to-memory, and immediate operand forms.
      - >-
        Generated code is callable as a C function pointer and returns
        correct results for at least 5 test cases (e.g., return 42,
        add two arguments, negate a value).
      - >-
        Emitted bytes match expected encoding verified by disassembling
        output with a disassembler (objdump -d or Capstone).
    pitfalls:
      - >-
        Mapping memory with PROT_READ|PROT_WRITE|PROT_EXEC simultaneously
        is forbidden on hardened systems (macOS, OpenBSD, SELinux). Always
        use the two-phase approach: write then mprotect to executable.
      - >-
        REX prefix (0x48) is required for 64-bit operand size; forgetting
        it silently operates on 32-bit values with zero-extension, producing
        subtly wrong results for large values.
      - >-
        Registers r8-r15 require REX.B (for ModR/M r/m field) or REX.R
        (for ModR/M reg field) — mixing these up produces wrong register
        operands.
      - >-
        Casting the mmap'd buffer to a function pointer requires ensuring
        the buffer address is page-aligned and the calling convention
        matches the emitted prologue/epilogue.
    concepts:
      - x86-64 instruction encoding (REX, ModR/M, SIB, displacement)
      - W^X memory protection policy
      - Executable memory management
      - Machine code generation and verification
    skills:
      - x86-64 Assembly Programming
      - Memory Protection Management (mmap/mprotect)
      - Binary Code Emission
      - Low-level Debugging with Disassemblers
    deliverables:
      - >-
        Executable memory manager: allocate RW, emit code, mprotect to RX,
        with proper cleanup (munmap).
      - >-
        x86-64 instruction encoder for mov, add, sub, imul, cmp, ret with
        register, immediate, and memory operands.
      - >-
        REX prefix generation for 64-bit operations and extended registers
        (r8-r15).
      - >-
        ModR/M and SIB byte encoder for all addressing modes.
      - >-
        Test harness that emits code, changes permissions, calls via
        function pointer, and verifies results.
      - >-
        Disassembly verification utility comparing emitted bytes against
        expected instruction encoding.
    estimated_hours: "12-18"

  - id: build-jit-m2
    name: "Expression JIT: Bytecode to Native Arithmetic"
    description: >-
      JIT-compile arithmetic and comparison bytecode operations to native
      x86-64 code with register allocation and jump patching.
    acceptance_criteria:
      - >-
        Translate bytecode arithmetic operations (ADD, SUB, MUL, DIV) to
        corresponding x86-64 instructions (add, sub, imul, idiv/div).
      - >-
        Signed division correctly emits 'cqo; idiv' (sign-extend rax
        into rdx:rax); unsigned division emits 'xor rdx, rdx; div'.
      - >-
        Simple register allocator maps the top N stack slots to machine
        registers and spills to the stack frame when expression depth
        exceeds available registers.
      - >-
        Comparison bytecodes (EQ, LT, GT, etc.) emit cmp followed by
        conditional set (setcc) or conditional jump (jcc) instructions.
      - >-
        Forward jumps are emitted with placeholder offsets and back-patched
        once the target address is known. Back-patching produces correct
        relative offsets.
      - >-
        JIT-compiled expressions produce bit-identical results to the
        interpreter for a test suite of at least 20 expressions including
        edge cases (division by zero, integer overflow, nested operations).
    pitfalls:
      - >-
        idiv requires the dividend in rdx:rax. Forgetting to emit cqo
        (signed) or zero rdx (unsigned) before the division instruction
        causes incorrect results or SIGFPE.
      - >-
        Forward jump offsets in x86-64 are relative to the END of the
        jump instruction, not the beginning. Off-by-N errors in patching
        are extremely common.
      - >-
        Register spilling: when the operand stack depth exceeds available
        registers, values must be spilled to the stack frame. Failing to
        track spill slots causes silent data corruption.
      - >-
        imul has different encoding forms (one-operand, two-operand,
        three-operand) with different semantics. Use the two-operand
        form for simplicity.
    concepts:
      - Bytecode-to-native translation
      - Register allocation (linear scan or simple)
      - Forward reference patching
      - x86-64 division semantics
    skills:
      - Runtime Code Generation
      - Register Management and Spilling
      - Control Flow Patching
      - Expression Compilation
    deliverables:
      - >-
        Bytecode-to-x86-64 translator for arithmetic (add, sub, mul, div)
        and comparison (cmp, setcc, jcc) operations.
      - >-
        Simple register allocator mapping VM stack slots to machine
        registers with stack spilling.
      - >-
        Forward jump back-patching mechanism for conditional branches.
      - >-
        Division handling for both signed (cqo+idiv) and unsigned
        (xor rdx+div) cases.
      - >-
        Correctness test suite comparing JIT output to interpreter output
        for 20+ expressions.
    estimated_hours: "10-15"

  - id: build-jit-m3
    name: "Function JIT with ABI Compliance"
    description: >-
      JIT-compile full functions with System V AMD64 ABI calling convention
      compliance and JIT-to-interpreter trampolines.
    acceptance_criteria:
      - >-
        Generated function prologue pushes rbp, sets up stack frame,
        saves all callee-saved registers used (rbx, r12-r15, rbp),
        and aligns rsp to 16 bytes.
      - >-
        Generated function epilogue restores callee-saved registers,
        tears down the stack frame, and returns via ret.
      - >-
        Arguments are read from the correct registers (rdi, rsi, rdx,
        rcx, r8, r9 for the first 6 integer args) and from the stack
        for additional arguments, per System V AMD64 ABI.
      - >-
        Return values are placed in rax (integer) per ABI convention.
      - >-
        Stack is 16-byte aligned immediately before every call instruction,
        verified by a test that calls a C library function (e.g., printf)
        from JIT-compiled code without crashing.
      - >-
        Trampoline mechanism allows JIT-compiled code to call interpreter
        functions and vice versa, correctly saving/restoring VM state
        across the transition.
      - >-
        Local variables are allocated in the stack frame with correct
        offsets, supporting at least 32 local variables.
      - >-
        Recursive JIT-compiled functions work correctly for at least
        depth 100 (e.g., factorial, fibonacci).
    pitfalls:
      - >-
        Stack alignment: the stack must be 16-byte aligned BEFORE the call
        instruction (which pushes 8 bytes for the return address). This
        means rsp must be 16-byte aligned after the prologue minus 8.
      - >-
        Callee-saved registers (rbx, rbp, r12-r15) must be preserved.
        If JIT code uses rbx without saving/restoring it, calling C
        functions will corrupt the caller's state.
      - >-
        Trampoline complexity: switching between JIT and interpreter
        requires saving the JIT's register state and restoring the
        interpreter's VM state (IP, stack pointer, frame pointer). This
        is easy to get wrong.
      - >-
        Variadic C functions (printf) require rax to contain the number
        of vector register arguments used (0 for integer-only calls).
    concepts:
      - System V AMD64 ABI calling convention
      - Stack frame construction and layout
      - Callee-saved vs caller-saved registers
      - JIT/interpreter trampolines
    skills:
      - x86-64 ABI Implementation
      - Stack Frame Construction
      - Cross-mode Execution Bridging
      - Function Call Compilation
    deliverables:
      - >-
        Function prologue/epilogue generator following System V AMD64 ABI.
      - >-
        Stack frame allocator for local variables with correct alignment.
      - >-
        Argument marshaling from ABI registers/stack to local variable
        slots.
      - >-
        Trampoline for JIT→interpreter and interpreter→JIT transitions
        with VM state save/restore.
      - >-
        Test suite including recursive functions, multi-argument functions,
        and calls to C library functions.
    estimated_hours: "12-18"

  - id: build-jit-m4
    name: "Hot Path Detection and Tiered Compilation"
    description: >-
      Add profiling-guided tiered compilation: interpret cold code, JIT-compile
      hot functions when execution count exceeds a threshold.
    acceptance_criteria:
      - >-
        Per-function invocation counter is incremented on each interpreted
        call with minimal overhead (< 5% interpreter slowdown).
      - >-
        When a function's counter exceeds a configurable threshold (default:
        1000), it is scheduled for JIT compilation.
      - >-
        JIT compilation replaces the function's dispatch entry so subsequent
        calls execute native code directly without interpreter overhead.
      - >-
        Basic optimization passes during JIT compilation: constant folding
        (evaluate compile-time constants), dead store elimination (remove
        unused assignments), and strength reduction (replace multiply by
        power-of-2 with shift).
      - >-
        Benchmark demonstrates at least 5x speedup on a compute-intensive
        loop (e.g., sum 1 to 1M) when JIT-compiled vs. interpreted.
      - >-
        Cold functions (below threshold) continue to execute in the
        interpreter with negligible profiling overhead.
    pitfalls:
      - >-
        Counter overhead: incrementing a counter on every function call
        and loop back-edge adds overhead to ALL code, including cold code.
        Use a simple decrement-and-branch-on-zero pattern for minimal
        overhead.
      - >-
        On-stack replacement (OSR) for hot loops already executing in the
        interpreter is extremely complex. For a first implementation,
        JIT the whole function on the NEXT call instead.
      - >-
        If bytecode can be dynamically modified (eval, hot-reload), JIT-
        compiled code must be invalidated. Track dependencies and invalidate
        on modification.
      - >-
        Optimization correctness: constant folding must respect evaluation
        order and side effects. Folding 'a / 0' to a constant is wrong.
    concepts:
      - Tiered compilation (interpreter → baseline JIT)
      - Profiling counters and hot code detection
      - Basic compiler optimizations
      - Code cache management
    skills:
      - Performance Profiling
      - Adaptive Optimization
      - Code Cache Management
      - Optimization Pass Implementation
    deliverables:
      - >-
        Per-function invocation counter with low-overhead increment.
      - >-
        Threshold-based JIT trigger that compiles and installs native code.
      - >-
        Constant folding, dead store elimination, and strength reduction
        optimization passes.
      - >-
        Code cache managing JIT-compiled function entries with invalidation
        support.
      - >-
        Benchmark suite comparing interpreted vs. JIT-compiled execution.
    estimated_hours: "10-15"

  - id: build-jit-m5
    name: "Deoptimization Guards and Safe Fallback"
    description: >-
      Implement speculative guards in JIT-compiled code that detect when
      assumptions are violated and safely fall back to the interpreter.
    acceptance_criteria:
      - >-
        Type guards emit a check before type-specialized operations (e.g.,
        'if value is not integer, deoptimize') that branches to a
        deoptimization stub on failure.
      - >-
        Deoptimization stub saves the current machine state (registers,
        stack), reconstructs the interpreter's VM state (instruction
        pointer, operand stack, local variables), and transfers control
        to the interpreter.
      - >-
        After deoptimization, the interpreter continues execution from
        the exact bytecode instruction corresponding to the failed guard,
        with correct VM state.
      - >-
        Guard failure counter tracks how often each guard fails. Functions
        with frequent guard failures are blacklisted from JIT compilation
        to avoid repeated compile-deopt cycles.
      - >-
        A test case demonstrates: JIT compiles a function assuming integer
        arguments, then calls it with a non-integer, triggers deoptimization,
        and the interpreter produces the correct result.
      - >-
        No memory leaks from deoptimized code: invalidated native code
        buffers are freed (or reused) after all active frames have left
        the deoptimized code.
    pitfalls:
      - >-
        VM state reconstruction is the hardest part: every register and
        stack slot in the JIT frame must be mapped back to the corresponding
        interpreter local variable or operand stack entry. This requires
        maintaining a side table of register-to-bytecode-slot mappings.
      - >-
        Deoptimizing in the middle of an expression (e.g., after evaluating
        LHS but before RHS) requires the interpreter to resume at the
        correct sub-expression point, which may require partial expression
        values on the operand stack.
      - >-
        Guard placement: too many guards eliminate the speedup from JIT
        compilation. Profile which guards actually fail and specialize
        only on stable types.
      - >-
        Dangling function pointers: if another frame holds a return address
        into deoptimized code, freeing that code causes a crash. Use
        reference counting or defer freeing until the code is provably
        unreachable.
    concepts:
      - Speculative optimization and guards
      - Deoptimization and interpreter fallback
      - VM state reconstruction
      - Guard failure profiling and blacklisting
    skills:
      - State Mapping (JIT ↔ Interpreter)
      - Guard Insertion and Placement
      - Safe Code Invalidation
      - Debugging Deoptimization Bugs
    deliverables:
      - >-
        Guard emission in JIT code for type checks and assumption
        verification.
      - >-
        Deoptimization stub that reconstructs interpreter VM state from
        JIT machine state.
      - >-
        Register-to-bytecode-slot mapping side table maintained during
        JIT compilation.
      - >-
        Guard failure counter and function blacklisting mechanism.
      - >-
        Safe code buffer invalidation with deferred freeing.
      - >-
        End-to-end test demonstrating speculate → guard fail → deoptimize
        → correct interpreter result.
    estimated_hours: "12-18"
```
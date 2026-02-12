# JIT Compiler: Design Document


## Overview

A Just-In-Time compiler that extends a bytecode virtual machine with native x86-64 code generation, transforming frequently-executed bytecode into optimized machine code at runtime. The key architectural challenge is bridging the gap between high-level bytecode operations and low-level machine instructions while managing executable memory and maintaining correctness.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones — understanding the core problem drives the entire JIT compiler design

The fundamental challenge of runtime code execution lies in balancing two competing forces: the speed of development and deployment versus the speed of execution. Traditional bytecode virtual machines prioritize portability and ease of implementation, but they pay a steep performance penalty through interpretation overhead. Just-In-Time compilation represents a sophisticated solution that transforms this trade-off from a binary choice into a dynamic optimization strategy.

### Mental Model: From Translation to Native Fluency

Think of the difference between bytecode interpretation and JIT compilation like the progression from using a human translator to achieving native language fluency. When you travel to a foreign country and rely on a translator, every conversation follows a predictable but slow pattern: you speak in your native language, the translator processes your words, converts them to the local language, delivers the message, waits for the response, translates it back, and finally conveys the answer to you. This process works correctly and allows complex communication, but it's inherently slow due to the multiple translation steps.

**Bytecode interpretation** operates exactly like this translation scenario. Every time your program needs to execute an instruction, the virtual machine acts as the translator: it reads a bytecode instruction (your native language), decodes what that instruction means (translation process), performs the corresponding machine operations (delivers the message), and then moves to the next instruction. Each bytecode instruction requires multiple CPU cycles to decode and dispatch, creating a constant translation overhead that accumulates across millions of instruction executions.

**Just-In-Time compilation** is analogous to learning the local language through immersion. Initially, you still rely on the translator (interpreter) for basic communication. However, as you notice certain conversations happening repeatedly—ordering coffee, asking for directions, introducing yourself—you begin to memorize these common phrases in the native language. Eventually, for these frequent interactions, you can speak directly without the translator, dramatically increasing both the speed and naturalness of communication.

The JIT compiler identifies frequently executed code paths (the "common conversations") and translates entire sequences of bytecode instructions into native machine code that the CPU can execute directly. Once a function has been compiled to native code, subsequent executions bypass the interpreter entirely, eliminating translation overhead for hot code paths while maintaining the flexibility of interpretation for rarely executed code.

This mental model reveals why JIT compilation is particularly effective: most programs follow the 80/20 rule, where 20% of the code accounts for 80% of the execution time. By focusing native compilation efforts on this critical 20%, JIT compilers achieve most of the performance benefits of ahead-of-time compilation while preserving the development agility of interpreted languages.

### Interpretation Performance Bottlenecks

The performance gap between bytecode interpretation and native execution stems from several fundamental overhead sources that compound with each instruction execution. Understanding these bottlenecks is crucial for appreciating why JIT compilation provides such dramatic speedups.

**Instruction Decode Overhead** represents the most pervasive performance penalty in bytecode interpretation. Each bytecode instruction must be fetched from the instruction stream, decoded to determine its operation type and operands, and dispatched to the appropriate handler function. This process typically involves multiple conditional branches or switch statements that modern CPUs struggle to predict accurately, leading to pipeline stalls and instruction cache misses.

Consider a simple arithmetic expression like `a + b * c` compiled to bytecode. The interpreter might execute this sequence:

1. `LOAD_LOCAL 0` (load variable `a`)
2. `LOAD_LOCAL 1` (load variable `b`) 
3. `LOAD_LOCAL 2` (load variable `c`)
4. `MUL` (multiply `b * c`)
5. `ADD` (add `a + result`)

Each instruction requires the interpreter to fetch the opcode, decode it through a switch statement, extract operands, perform the operation, and update the virtual machine state. This five-instruction sequence might require 50-100 CPU cycles in the interpreter, while the equivalent native x86-64 code could execute in 5-10 cycles.

**Virtual Stack Manipulation** creates additional overhead through constant memory traffic and bounds checking. Bytecode VMs typically use a stack-based architecture where operands are pushed and popped from a virtual execution stack. Every arithmetic operation involves multiple stack manipulations: popping operands, performing the computation, and pushing the result. Each stack operation requires memory access, stack pointer updates, and often bounds checking to prevent stack overflow or underflow.

The interpreter must maintain explicit stack pointer tracking and perform safety checks that native code can eliminate through compiler analysis. For instance, the multiplication operation `MUL` in the sequence above requires popping two values from the stack, multiplying them, and pushing the result—three memory operations that native code could eliminate by keeping values in CPU registers.

**Indirect Function Calls and Branch Prediction Failures** severely impact modern CPU performance due to their interaction with processor optimization features. Bytecode interpreters typically implement instruction dispatch through function pointer tables or large switch statements. Each bytecode instruction execution involves an indirect branch whose target address depends on the runtime instruction stream, making it nearly impossible for the CPU's branch predictor to anticipate correctly.

Modern processors rely heavily on speculative execution and branch prediction to maintain high instruction throughput. When branch predictions fail—as they frequently do in interpreter dispatch loops—the CPU must discard speculatively executed instructions and restart from the correct path, creating substantial performance penalties that can stall execution for dozens of cycles.

**Type Checking and Dynamic Dispatch** overhead affects dynamically typed languages where the interpreter must verify operand types and select appropriate operations at runtime. Even in statically typed bytecode, the interpreter often includes runtime type assertions and overflow checking that native code could optimize away or handle more efficiently.

For example, an integer addition in the interpreter might involve checking that both operands are integers, verifying that the addition won't overflow, performing the arithmetic, and boxing the result into a tagged value representation. Native code generated by a JIT compiler could reduce this to a single `add` instruction with optional overflow checking.

**Memory Access Patterns and Cache Efficiency** problems arise from the interpreter's data structures and execution patterns. Bytecode instructions are typically stored in a linear array that's accessed sequentially, while the interpreter's dispatch logic creates irregular memory access patterns that can lead to cache misses. The virtual machine state (program counter, stack pointer, local variables) often resides in different memory regions, causing additional cache line misses.

> The cumulative effect of these bottlenecks explains why interpreted bytecode often runs 10-100 times slower than equivalent native code. JIT compilation addresses each bottleneck systematically: eliminating decode overhead through direct native execution, reducing stack manipulation through register allocation, enabling branch prediction through straight-line code sequences, specializing operations based on observed types, and improving cache efficiency through compact native code generation.

### Compilation Approach Comparison

The choice between ahead-of-time (AOT) compilation and just-in-time (JIT) compilation involves fundamental trade-offs that affect development workflow, deployment complexity, startup performance, and steady-state execution speed. Understanding these trade-offs helps explain why JIT compilation represents an optimal middle ground for many use cases.

> **Decision: Compilation Strategy Selection**
> - **Context**: Programs need to balance development agility, deployment flexibility, startup time, and execution performance across diverse runtime environments
> - **Options Considered**: Pure interpretation, ahead-of-time compilation, just-in-time compilation, hybrid tiered compilation
> - **Decision**: JIT compilation with interpreter fallback
> - **Rationale**: JIT compilation provides near-native performance for hot code while maintaining development flexibility and deployment simplicity of interpreted languages
> - **Consequences**: Enables rapid development cycles with production-ready performance, at the cost of implementation complexity and warm-up time

| Compilation Approach | Development Speed | Deployment Complexity | Startup Time | Steady-State Performance | Platform Flexibility |
|---------------------|-------------------|----------------------|---------------|--------------------------|---------------------|
| Pure Interpretation | Excellent - instant feedback | Minimal - copy bytecode | Fast - no compilation | Poor - 10-100x slower | Perfect - portable bytecode |
| Ahead-of-Time (AOT) | Poor - long build cycles | High - per-platform builds | Fast - ready to run | Excellent - fully optimized | Limited - separate binaries |
| Just-in-Time (JIT) | Good - fast iteration | Low - single bytecode | Slow - compilation overhead | Excellent - optimized hot paths | Good - adaptive compilation |
| Tiered JIT | Good - fast iteration | Low - single bytecode | Medium - progressive optimization | Excellent - best of both worlds | Good - adaptive compilation |

**Ahead-of-Time Compilation Characteristics** center around performing all optimization work before program deployment. AOT compilers analyze the entire program, apply sophisticated optimizations like inlining and loop unrolling, and generate highly optimized machine code for the target platform. This approach delivers maximum steady-state performance since no compilation overhead exists at runtime.

However, AOT compilation imposes significant constraints on development workflow and deployment flexibility. Any code change requires a complete recompilation cycle that can take minutes or hours for large applications. Cross-platform deployment requires maintaining separate compiled binaries for each target architecture, increasing build complexity and distribution overhead. Perhaps most critically, AOT compilers must make optimization decisions based on static analysis without runtime behavior information, potentially missing opportunities that only become apparent during actual program execution.

**Pure Interpretation Benefits** include unmatched development agility and deployment simplicity. Code changes take effect immediately without compilation delays, enabling rapid prototyping and interactive development. A single bytecode file runs unchanged across all platforms, eliminating architecture-specific build processes. The interpreter can provide rich debugging and introspection capabilities since it maintains complete program metadata at runtime.

The primary drawback of pure interpretation is execution speed. As detailed in the performance bottlenecks section, interpretation overhead typically reduces execution speed by 1-2 orders of magnitude compared to native code. This performance penalty makes pure interpretation unsuitable for computationally intensive applications, despite its development advantages.

**Just-in-Time Compilation Advantages** emerge from its ability to combine the development benefits of interpretation with the performance benefits of native compilation. JIT compilers observe actual program behavior at runtime, enabling optimizations that static analysis cannot discover. For example, a JIT compiler can inline function calls that are monomorphic at runtime even when static analysis suggests they might be polymorphic.

JIT compilation supports adaptive optimization strategies where the compiler can specialize code based on observed data types, branch patterns, and call frequencies. If program behavior changes over time, the JIT compiler can recompile code with different optimizations, providing a level of adaptability that AOT compilation cannot match.

**Tiered Compilation Strategy** represents the most sophisticated approach, combining multiple compilation levels to optimize the trade-off between compilation time and code quality. A typical tiered JIT system might include:

1. **Interpreter Level**: Executes all code initially while collecting profiling data
2. **Quick Compiler**: Generates unoptimized native code for warm functions to eliminate interpretation overhead
3. **Optimizing Compiler**: Applies expensive optimizations to hot functions based on extensive profiling data

This approach minimizes startup overhead by avoiding expensive compilation for cold code while ensuring that hot code receives the full benefit of aggressive optimization.

**Runtime Information Advantages** represent one of JIT compilation's most significant benefits over AOT compilation. Static compilers must make conservative assumptions about program behavior, such as assuming that any virtual function call could invoke any implementation or that any array access could be out of bounds. JIT compilers can observe actual program behavior and make aggressive optimizations based on runtime patterns.

For instance, if a JIT compiler observes that a virtual function call site always invokes the same implementation, it can inline that function and eliminate the virtual dispatch overhead. If an array access pattern never goes out of bounds, the compiler can eliminate bounds checking. These optimizations can provide speedups that AOT compilation cannot achieve.

**Memory Overhead and Complexity Trade-offs** must be carefully considered in JIT compiler design. JIT compilation requires maintaining both bytecode and compiled native code in memory, increasing memory usage compared to pure AOT or interpretation approaches. The compiler itself consumes additional memory and CPU cycles during compilation, creating overhead that must be amortized over many executions of the compiled code.

The implementation complexity of JIT compilers significantly exceeds that of interpreters or AOT compilers. JIT compilers must handle dynamic memory management for executable code, coordinate between interpreted and compiled code execution, manage compilation threads to avoid blocking program execution, and implement deoptimization when optimistic assumptions prove incorrect.

> The key insight driving JIT compilation adoption is that most applications exhibit highly skewed execution patterns where a small fraction of code accounts for most execution time. By focusing compilation effort on this hot code while interpreting the remainder, JIT compilers achieve near-optimal performance with acceptable complexity and memory overhead.

### Implementation Guidance

The transition from understanding JIT compilation concepts to implementing a working compiler requires careful technology choices and project organization. This guidance provides concrete recommendations for building a JIT compiler that balances learning objectives with practical implementation constraints.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Machine Code Generation | Hand-coded x86-64 instruction encoding | LLVM IR generation with LLVM backend |
| Executable Memory Management | `mmap()` with `PROT_EXEC` flags | Platform abstraction layer with VirtualAlloc/mmap |
| Profiling Infrastructure | Simple execution counters | Statistical sampling profiler |
| Bytecode Analysis | Linear scan with basic block detection | Control flow graph with dominance analysis |
| Register Allocation | Linear scan with spilling | Graph coloring or live range splitting |
| Debugging Support | Hexdump of generated machine code | DWARF debug information generation |

For learning purposes, the simple options provide sufficient functionality while keeping implementation complexity manageable. The hand-coded instruction encoding approach, while more verbose than using LLVM, offers deeper insight into machine code generation fundamentals.

**B. Recommended File Structure**

```c
jit-compiler/
├── src/
│   ├── vm/
│   │   ├── vm.h                    // VM core interfaces
│   │   ├── vm.c                    // Basic VM implementation  
│   │   ├── bytecode.h              // Bytecode instruction definitions
│   │   └── interpreter.c           // Bytecode interpreter
│   ├── jit/
│   │   ├── jit.h                   // JIT compiler public interface
│   │   ├── compiler.c              // High-level compilation orchestration
│   │   ├── emitter.h               // Machine code emitter interface
│   │   ├── emitter.c               // x86-64 instruction encoding
│   │   ├── codegen.c               // Bytecode to native translation
│   │   ├── regalloc.c              // Simple register allocation
│   │   └── profiler.c              // Execution frequency tracking
│   ├── memory/
│   │   ├── executable.h            // Executable memory management
│   │   └── executable.c            // mmap-based code buffer allocation
│   └── utils/
│       ├── disasm.c                // x86-64 disassembler for debugging
│       └── test_helpers.c          // Common test utilities
├── tests/
│   ├── vm_tests.c                  // Interpreter correctness tests
│   ├── jit_tests.c                 // JIT compilation tests
│   ├── benchmarks.c                // Performance comparison tests
│   └── integration_tests.c         // End-to-end functionality tests
├── examples/
│   ├── arithmetic.bc               // Simple arithmetic bytecode
│   ├── fibonacci.bc                // Recursive function example
│   └── mandelbrot.bc               // Compute-intensive benchmark
└── Makefile
```

This organization separates concerns clearly: VM core functionality remains independent of JIT implementation, enabling incremental development and testing. The memory management abstraction simplifies platform-specific executable memory handling.

**C. Infrastructure Starter Code**

**Executable Memory Allocator** (`src/memory/executable.c`):

```c
#include "executable.h"
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

typedef struct CodeBuffer {
    void* memory;
    size_t size;
    size_t used;
    int writable;
} CodeBuffer;

CodeBuffer* code_buffer_create(size_t size) {
    CodeBuffer* buffer = malloc(sizeof(CodeBuffer));
    if (!buffer) return NULL;
    
    // Allocate with read/write initially for code generation
    buffer->memory = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buffer->memory == MAP_FAILED) {
        free(buffer);
        return NULL;
    }
    
    buffer->size = size;
    buffer->used = 0;
    buffer->writable = 1;
    return buffer;
}

int code_buffer_make_executable(CodeBuffer* buffer) {
    if (!buffer->writable) return 0; // Already executable
    
    // Change protection to read/execute (W^X security)
    if (mprotect(buffer->memory, buffer->size, PROT_READ | PROT_EXEC) != 0) {
        return -1;
    }
    buffer->writable = 0;
    return 0;
}

void* code_buffer_allocate(CodeBuffer* buffer, size_t bytes) {
    if (!buffer->writable || buffer->used + bytes > buffer->size) {
        return NULL;
    }
    
    void* ptr = (char*)buffer->memory + buffer->used;
    buffer->used += bytes;
    return ptr;
}

void code_buffer_destroy(CodeBuffer* buffer) {
    if (buffer) {
        munmap(buffer->memory, buffer->size);
        free(buffer);
    }
}
```

**Profiling Infrastructure** (`src/jit/profiler.c`):

```c
#include "profiler.h"
#include <stdlib.h>
#include <string.h>

typedef struct ProfileEntry {
    uint32_t function_id;
    uint32_t invocation_count;
    uint32_t compilation_threshold;
    int is_compiled;
} ProfileEntry;

typedef struct Profiler {
    ProfileEntry* entries;
    size_t capacity;
    uint32_t default_threshold;
} Profiler;

Profiler* profiler_create(size_t max_functions, uint32_t threshold) {
    Profiler* prof = malloc(sizeof(Profiler));
    if (!prof) return NULL;
    
    prof->entries = calloc(max_functions, sizeof(ProfileEntry));
    if (!prof->entries) {
        free(prof);
        return NULL;
    }
    
    prof->capacity = max_functions;
    prof->default_threshold = threshold;
    return prof;
}

int profiler_record_invocation(Profiler* prof, uint32_t function_id) {
    if (function_id >= prof->capacity) return -1;
    
    ProfileEntry* entry = &prof->entries[function_id];
    entry->function_id = function_id;
    entry->invocation_count++;
    
    if (entry->compilation_threshold == 0) {
        entry->compilation_threshold = prof->default_threshold;
    }
    
    return entry->invocation_count >= entry->compilation_threshold && 
           !entry->is_compiled;
}

void profiler_mark_compiled(Profiler* prof, uint32_t function_id) {
    if (function_id < prof->capacity) {
        prof->entries[function_id].is_compiled = 1;
    }
}
```

**D. Core Logic Skeleton Code**

**JIT Compiler Interface** (`src/jit/compiler.c`):

```c
// compile_function transforms a bytecode function into native x86-64 machine code.
// Returns a function pointer that can be called directly, or NULL on compilation failure.
typedef int (*CompiledFunction)(int* args, int arg_count);

CompiledFunction compile_function(BytecodeFunction* func, CodeBuffer* code_buffer) {
    // TODO 1: Analyze bytecode to identify basic blocks and control flow
    // TODO 2: Perform simple register allocation mapping stack slots to x86-64 registers  
    // TODO 3: Generate function prologue following System V AMD64 calling convention
    // TODO 4: Translate each bytecode instruction to equivalent x86-64 instruction sequence
    // TODO 5: Handle forward references for jumps with two-pass code generation
    // TODO 6: Generate function epilogue and return sequence
    // TODO 7: Patch jump targets and relocate any PC-relative addresses
    // TODO 8: Return function pointer cast from code buffer memory address
    
    // Hint: Start with rdi/rsi/rdx for first 3 integer arguments per AMD64 ABI
    // Hint: Use rbp-relative addressing for local variables in stack frame
    return NULL;
}
```

**x86-64 Instruction Emitter** (`src/jit/emitter.c`):

```c
// emit_mov_reg_imm generates "mov reg, immediate" instruction into code buffer.
// Handles REX prefix for 64-bit operands and ModR/M byte encoding for register.
void emit_mov_reg_imm(CodeBuffer* buffer, X86Register reg, int64_t immediate) {
    // TODO 1: Emit REX prefix (0x48) for 64-bit operation
    // TODO 2: Emit MOV opcode (0xB8 + register encoding for immediate form)
    // TODO 3: Emit 64-bit immediate value in little-endian byte order
    // TODO 4: Update code buffer position pointer
    
    // Hint: RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7
    // Hint: "mov rax, 0x1234567890ABCDEF" = [0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x90, 0x78, 0x56, 0x34, 0x12]
}

// emit_add_reg_reg generates "add dst, src" instruction for register-to-register addition.
// Uses ModR/M byte to encode both source and destination registers.
void emit_add_reg_reg(CodeBuffer* buffer, X86Register dst, X86Register src) {
    // TODO 1: Emit REX prefix (0x48) for 64-bit operation  
    // TODO 2: Emit ADD opcode (0x01) for register-to-register form
    // TODO 3: Emit ModR/M byte: 0xC0 | (src << 3) | dst
    // TODO 4: Update code buffer position pointer
    
    // Hint: ModR/M byte format: [mod:2][reg:3][r/m:3] where mod=11b for register mode
}
```

**E. Language-Specific Hints**

- Use `mmap()` with `PROT_READ|PROT_WRITE|PROT_EXEC` for executable memory on Linux/macOS
- On some systems, use separate `mmap()` calls with `mprotect()` to enforce W^X policy
- Cast code buffer addresses to function pointers: `CompiledFunction func = (CompiledFunction)code_address`
- Use `__builtin_expect()` to hint branch probability in hot interpreter loops
- Enable `-fno-strict-aliasing` compiler flag when casting between pointer types
- Use `objdump -D -b binary -m i386:x86-64` to disassemble generated machine code for debugging
- Implement stack alignment checks with `assert((rsp & 0xF) == 0)` before function calls

**F. Milestone Checkpoints**

**Milestone 1 Verification**:
```bash
# Compile and run basic machine code emitter tests
gcc -g -Wall src/jit/*.c src/memory/*.c tests/emitter_tests.c -o test_emitter
./test_emitter

# Expected output:
# ✓ mov_reg_imm: Generated correct instruction bytes
# ✓ add_reg_reg: Generated correct ModR/M encoding  
# ✓ executable_memory: Function call returned expected value (42)
# ✓ basic_arithmetic: 3 + 5 = 8 computed in generated code
```

**Milestone 2 Verification**:
```bash
# Test bytecode to native translation
./test_jit examples/arithmetic.bc

# Expected behavior:
# - Interpreter result: 42
# - JIT compiled result: 42  
# - Generated code size: 87 bytes
# - Performance improvement: 15.3x speedup
```

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| Segmentation fault when calling generated code | Incorrect instruction encoding or stack corruption | Use `objdump` to disassemble, check stack alignment | Verify REX prefix and ModR/M byte encoding |
| Generated code returns wrong result | Register allocation error or incorrect operation translation | Add debug prints showing register assignments | Check bytecode to x86-64 operation mapping |
| JIT compilation never triggers | Profiling threshold too high or counter not incrementing | Print invocation counts and threshold values | Lower threshold or fix counter increment logic |
| Memory allocation failures | Code buffer too small or memory leak | Monitor memory usage and buffer allocation patterns | Increase buffer size or implement code garbage collection |
| Performance regression vs interpreter | Compilation overhead exceeds execution savings | Benchmark compilation time vs execution time savings | Increase compilation threshold or optimize compilation speed |


## Goals and Non-Goals

> **Milestone(s):** All milestones — defines the scope and boundaries that guide the entire JIT compiler implementation

The success of any complex system implementation depends critically on establishing clear boundaries between what will be built and what will be explicitly excluded. A JIT compiler represents a particularly challenging scope management problem because the field encompasses decades of research and optimization techniques, from simple expression compilation to sophisticated adaptive optimization systems used in production virtual machines like the JVM's HotSpot or JavaScript's V8 engine.

Our JIT compiler implementation deliberately balances educational value with implementation complexity, focusing on core concepts that demonstrate the fundamental principles of runtime code generation while avoiding advanced features that would obscure these principles or require months of additional development. This section establishes explicit boundaries to prevent scope creep and ensure the implementation remains achievable within the expert-level difficulty target.

Think of this scoping exercise like planning a mountain climbing expedition. An experienced climber doesn't attempt Everest on their first major climb — they choose a challenging peak that will teach essential skills (rope work, weather assessment, high-altitude acclimatization) without requiring specialized equipment or years of preparation. Our JIT compiler follows the same philosophy: we're climbing a significant technical peak that teaches all the fundamental concepts, but we're not attempting the equivalent of Everest's oxygen-deprived death zone.

### Primary Goals

The primary goals represent the core functionality that our JIT compiler must deliver to provide a complete learning experience in runtime code generation. These goals are structured around the four major conceptual areas that define modern JIT compilation: machine code generation, bytecode translation, runtime profiling, and execution integration.

#### Machine Code Generation and Memory Management

Our first fundamental goal centers on **low-level machine code generation** — the ability to translate abstract operations into concrete x86-64 instructions and manage the complex memory requirements of executable code. This goal encompasses several interconnected capabilities that form the foundation of all JIT compilation.

The implementation must demonstrate **executable memory allocation** using the POSIX `mmap` system call with appropriate protection flags. This involves understanding the critical security and performance implications of memory pages that contain executable machine code. The system must properly handle the W^X (write-xor-execute) security policy implemented by modern operating systems, which prevents memory pages from being simultaneously writable and executable to mitigate code injection attacks.

Our x86-64 instruction encoding capability must cover the essential instruction categories required for basic computation: data movement (`mov` instructions with register and immediate operands), arithmetic operations (`add`, `sub`, `mul`, `idiv`), comparison operations (`cmp`), and control flow (`jmp`, conditional jump variants like `je`, `jne`, `jl`, `jg`), and function entry/exit (`ret`). The implementation must correctly handle x86-64's complex instruction encoding requirements, including REX prefixes for 64-bit operations, ModR/M byte encoding for operand specification, and immediate value encoding in various formats.

**Register and operand encoding** represents a particularly critical aspect of this goal. The implementation must support the complete x86-64 register set (RAX through R15) and correctly generate the binary encoding for register-to-register operations, register-immediate operations, and basic memory addressing modes. This includes understanding when REX prefixes are required and how to encode them correctly.

The **function pointer integration** aspect of this goal requires that generated machine code follows standard calling conventions closely enough that it can be invoked through C function pointers. This means the generated code must preserve required registers, maintain stack alignment, and handle argument passing and return values according to the System V AMD64 ABI.

| Capability | Input | Output | Validation Criteria |
|------------|-------|---------|-------------------|
| Executable Memory Allocation | Size requirement | `CodeBuffer*` with executable pages | Memory is writable during generation, executable after protection change |
| Instruction Encoding | Mnemonic + operands | Raw bytes in buffer | Bytes match expected x86-64 encoding |
| Register Operand Handling | Register enum + values | ModR/M byte encoding | Correct register field encoding |
| Function Invocation | Generated code buffer | Function pointer cast | Callable from C with correct results |

#### Bytecode to Native Translation

The second primary goal focuses on **systematic translation** from high-level bytecode operations to equivalent native instruction sequences. This goal represents the conceptual heart of JIT compilation — bridging the semantic gap between virtual machine abstractions and physical processor capabilities.

Our translation capability must handle the complete set of arithmetic and comparison operations typically found in bytecode instruction sets. This includes not only direct instruction mapping (where a single bytecode ADD instruction becomes a single x86-64 `add` instruction) but also complex mappings where single bytecode operations expand into multiple native instructions. Division operations exemplify this complexity — a single bytecode DIV instruction must expand into dividend setup in the RDX:RAX register pair, the `cqo` sign extension instruction, the `idiv` instruction, and proper handling of the quotient result.

**Stack-to-register mapping** represents a fundamental translation challenge. Bytecode virtual machines typically operate on an abstract stack where operands are pushed and popped in last-in-first-out order. Physical processors operate on a fixed set of registers with specific addressing capabilities. Our translation layer must implement a strategy for mapping stack operations to register operations, handling cases where expression depth exceeds available registers, and maintaining semantic equivalence between bytecode execution and native execution.

The **control flow translation** aspect of this goal requires handling bytecode's typically abstract jump instructions (which reference bytecode addresses) and translating them to native conditional jumps (which reference physical memory addresses). This involves forward reference resolution — generating native jump instructions before knowing their final target addresses, then patching these instructions once target addresses are determined.

**Correctness verification** represents a critical aspect of this goal. The translated native code must produce mathematically identical results to bytecode interpretation across all possible input values and execution paths. This includes proper handling of integer overflow, division by zero, and other edge cases that might have different behavior between bytecode and native execution.

| Translation Type | Bytecode Pattern | Native Pattern | Complexity Factor |
|------------------|------------------|----------------|-------------------|
| Simple Arithmetic | ADD, SUB | Single instruction | 1:1 mapping |
| Complex Arithmetic | DIV, MUL | Multi-instruction sequence | 1:N mapping |
| Stack Operations | PUSH val, ADD | Register allocation + instruction | Abstract to concrete |
| Control Flow | JUMP_IF_ZERO addr | Compare + conditional jump | Forward reference resolution |

#### Runtime Profiling and Hot Path Detection

The third primary goal establishes **intelligent compilation triggering** through runtime execution monitoring. This goal transforms our implementation from a simple translator into a true JIT compiler that makes dynamic optimization decisions based on observed program behavior.

Our profiling system must implement **lightweight execution counting** that tracks function invocation frequency and loop iteration counts without introducing significant overhead to interpreted execution. The profiling instrumentation should add no more than a few nanoseconds per function call or loop iteration — overhead that becomes negligible once hot code paths are identified and compiled to native code.

The **threshold-based compilation triggering** mechanism must implement configurable policies for deciding when bytecode should be compiled to native code. This involves not only simple invocation count thresholds but also more sophisticated policies that consider compilation cost versus expected benefit. Functions that are called frequently but execute quickly might not benefit from JIT compilation due to the overhead of the compilation process itself.

**Tiered execution coordination** requires seamless transitions between interpreted and native execution modes. When a function transitions from interpreted to compiled status, the system must handle ongoing executions gracefully — completing current interpreted invocations while routing new invocations to the compiled version.

The profiling system must also implement **compilation state management**, tracking which functions are currently being compiled (to avoid duplicate compilation efforts), which functions have been successfully compiled, and which functions have failed compilation (to avoid repeated futile compilation attempts).

| Profiling Component | Data Tracked | Trigger Condition | Action Taken |
|---------------------|-------------|------------------|--------------|
| Function Invocation Counter | Call count per function | Count exceeds threshold | Queue for compilation |
| Loop Iteration Counter | Iteration count per loop | Hot loop detected | Include in function compilation |
| Compilation State Tracker | Per-function compilation status | State transitions | Route execution appropriately |
| Performance Monitor | Compilation time vs execution time | Cost-benefit analysis | Adjust thresholds dynamically |

#### System Integration and Performance Validation

The fourth primary goal ensures that our JIT compiler integrates seamlessly with the existing bytecode virtual machine and delivers measurable performance improvements. This goal validates that our implementation achieves its fundamental purpose — accelerating program execution through runtime optimization.

**Transparent VM integration** requires that the JIT compiler extends the existing bytecode VM without modifying its core interpreter logic or bytecode format. The integration should be achievable through well-defined extension points, allowing the JIT compiler to be enabled or disabled without affecting the correctness of interpreted execution.

The **calling convention compatibility** aspect of this goal ensures that JIT-compiled functions can call interpreted functions and vice versa without complex marshaling or data conversion overhead. This requires careful attention to argument passing, return value handling, exception propagation, and stack frame management across the interpreted-native boundary.

**Performance benchmarking** represents the ultimate validation of our implementation. The system must demonstrate significant speedup (target: 3-10x improvement) on compute-intensive bytecode programs while maintaining identical functional behavior to interpretation. The benchmarking must cover various program patterns — arithmetic-heavy computations, loop-intensive algorithms, and function-call-heavy code — to demonstrate the broad applicability of JIT compilation benefits.

**Memory and resource management** ensures that the JIT compiler itself doesn't introduce excessive memory overhead or resource leaks. The implementation must properly clean up compiled code when functions are redefined, manage executable memory allocation efficiently, and provide mechanisms for controlling the total amount of memory dedicated to compiled code.

> **Decision: Focus on Single-Architecture x86-64 Implementation**
> - **Context**: JIT compilers can target multiple processor architectures, but each architecture requires significant specialized knowledge of instruction encoding, calling conventions, and performance characteristics.
> - **Options Considered**: 
>   1. Multi-architecture abstraction layer supporting x86-64, ARM64, and RISC-V
>   2. Single x86-64 implementation with clean architecture for future extension
>   3. Architecture-agnostic intermediate representation with pluggable backends
> - **Decision**: Single x86-64 implementation with clean separation between bytecode translation and machine code generation
> - **Rationale**: x86-64 remains the dominant development and deployment architecture, has excellent documentation and tooling support, and allows deep focus on JIT compilation concepts without diluting effort across multiple instruction sets. The clean separation ensures future extensibility.
> - **Consequences**: Implementation can dive deep into x86-64 optimization opportunities and calling convention details. Future multi-architecture support will require refactoring but not redesigning core algorithms.

### Explicit Non-Goals

The explicit non-goals section defines advanced features and optimizations that, while valuable in production JIT compilers, would significantly increase implementation complexity without proportional educational benefit. These exclusions allow our implementation to focus on fundamental concepts while acknowledging the rich ecosystem of advanced techniques used in modern JIT systems.

#### Advanced Optimization Techniques

Our implementation explicitly excludes sophisticated optimization passes that require complex program analysis or extensive infrastructure support. While these optimizations provide significant performance benefits in production systems, they would overshadow the core learning objectives of our JIT compiler.

**Inlining and interprocedural optimization** represent powerful techniques used in production JIT compilers to eliminate function call overhead and enable cross-function optimization opportunities. However, implementing effective inlining requires sophisticated heuristics for deciding when inlining improves performance, complex code size management to prevent excessive code bloat, and intricate handling of recursive function calls and indirect calls through function pointers. These complexities would require several additional milestones to implement properly.

**Advanced register allocation algorithms** like graph coloring, linear scan with spilling, or static single assignment (SSA) form optimizations provide significant performance improvements but require extensive compiler theory background. Our simple register allocation strategy (mapping stack slots directly to registers with basic spill handling) demonstrates the core concepts without requiring implementation of complex interference graph algorithms or live range analysis.

**Loop optimization techniques** including loop unrolling, loop-invariant code motion, and vectorization can provide dramatic performance improvements for numerical code. However, these optimizations require sophisticated loop analysis, dependence analysis, and target-specific knowledge of SIMD instruction sets. The complexity of implementing effective loop optimization would double or triple our implementation effort.

**Profile-guided optimization (PGO)** beyond basic hot path detection involves collecting detailed execution profiles (branch frequencies, value profiles, type profiles) and using this information to guide complex optimization decisions. While conceptually interesting, PGO requires extensive profiling infrastructure and optimization heuristics that would obscure the fundamental JIT compilation concepts.

| Excluded Optimization | Reason for Exclusion | Complexity Factors | Alternative Learning Path |
|----------------------|---------------------|-------------------|------------------------|
| Function Inlining | Requires complex heuristics and code size management | Call graph analysis, recursive handling | Study LLVM inlining passes |
| Advanced Register Allocation | Needs extensive compiler theory background | Interference graphs, live range analysis | Implement graph coloring separately |
| Loop Vectorization | Requires SIMD instruction knowledge | Dependence analysis, target-specific codegen | Focus on scalar optimization first |
| Profile-Guided Optimization | Extensive profiling infrastructure needed | Multi-dimensional profiling, optimization heuristics | Study production PGO systems |

#### Multi-Threading and Concurrency

Our implementation excludes concurrent compilation and multi-threaded execution support, focusing instead on single-threaded execution with clean extension points for future concurrent features.

**Concurrent compilation** in production JIT systems allows compilation to occur on background threads while the main program continues executing in interpreted mode. This approach improves responsiveness by eliminating compilation pauses but introduces significant complexity in compilation queue management, memory synchronization between compiler and execution threads, and handling of concurrent compilation requests for the same function.

**Thread-safe code generation** requires careful synchronization of executable memory allocation, compilation state management, and code cache updates. The complexity of implementing correct concurrent code generation without race conditions or memory corruption would require extensive experience with lock-free programming and memory ordering semantics.

**On-stack replacement (OSR)** allows JIT-compiled code to replace interpreted code even while functions are actively executing (for example, in the middle of a long-running loop). OSR provides significant performance benefits for long-running computations but requires complex stack frame transformation, register state migration, and careful handling of execution state consistency.

The single-threaded focus allows our implementation to concentrate on the core algorithms of bytecode translation and machine code generation without the additional complexity of concurrent programming. Extension points in the design ensure that concurrent features could be added later without requiring fundamental architectural changes.

#### Garbage Collection and Memory Management Integration

Modern JIT compilers in garbage-collected languages must integrate closely with the runtime's memory management system, but this integration involves complex interactions that would significantly expand our implementation scope.

**Garbage collection integration** requires JIT-generated code to participate in garbage collection by providing stack maps (identifying which stack slots and registers contain object references at each potential collection point), write barriers (tracking object reference modifications for generational collectors), and safe point insertion (ensuring that garbage collection can occur at regular intervals).

**Object layout optimization** involves JIT compilation making assumptions about object field layouts, method dispatch tables, and type hierarchy information. When the garbage collector moves objects or the runtime system modifies type information, the JIT-compiled code must be invalidated and potentially recompiled. This invalidation mechanism requires complex dependency tracking and code patching capabilities.

**Allocation fast paths** allow JIT-compiled code to perform object allocation inline rather than calling into the runtime system, but these fast paths must handle allocation failure, garbage collection triggering, and synchronization with concurrent collectors.

Our implementation focuses on primitive data types (integers, floating-point numbers) that don't require garbage collection, allowing the JIT compiler to demonstrate core concepts without the additional complexity of runtime system integration.

#### Production-Quality Features

Several features essential for production JIT compilers are explicitly excluded from our educational implementation due to their complexity and specialized requirements.

**Debugging and profiling integration** in production JIT systems includes generating DWARF debug information for JIT-compiled code, supporting debugger breakpoints in generated code, and integrating with performance profiling tools. These features require detailed knowledge of debugging formats and tool integration protocols that would significantly expand the implementation scope.

**Security hardening** includes control flow integrity (CFI) protection, return-oriented programming (ROP) mitigation, and code signing for generated code. While important for production systems, these security features involve platform-specific security mechanisms and cryptographic operations that are orthogonal to the core JIT compilation learning objectives.

**Adaptive optimization** systems that continuously monitor program behavior and recompile functions with different optimization levels based on changing execution patterns represent the cutting edge of JIT compiler technology. However, implementing effective adaptive optimization requires extensive performance modeling, optimization heuristics, and runtime overhead management.

**Cross-platform compatibility** beyond x86-64 Linux would require abstracting memory management, executable code generation, and calling convention handling across multiple operating systems and processor architectures. This abstraction layer would add significant complexity without enhancing understanding of the core JIT compilation algorithms.

> **Decision: Exclude Garbage Collection Integration**  
> - **Context**: Many languages requiring JIT compilation use garbage collection for automatic memory management, requiring tight integration between generated code and the GC system.
> - **Options Considered**:
>   1. Implement simple mark-and-sweep GC with stack scanning support
>   2. Focus on primitive types only, avoiding GC integration entirely  
>   3. Provide hooks for future GC integration without implementing GC
> - **Decision**: Focus on primitive types only, avoiding GC integration complexity
> - **Rationale**: GC integration requires extensive additional infrastructure (stack maps, write barriers, safe points) that would overshadow JIT compilation concepts. Primitive types demonstrate all core JIT principles while keeping complexity manageable.
> - **Consequences**: Implementation can focus deeply on code generation and optimization without GC complexity. Real-world application would require significant additional work for GC integration.

The boundary between goals and non-goals reflects a deliberate choice to prioritize depth over breadth. Our implementation will provide a thorough understanding of JIT compilation fundamentals — machine code generation, bytecode translation, runtime profiling, and execution integration — while acknowledging the rich ecosystem of advanced techniques that production systems employ. This approach ensures that developers who complete the implementation will have a solid foundation for understanding and extending more sophisticated JIT compilation systems.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Rationale |
|-----------|---------------|----------------|-----------|
| Memory Management | `mmap`/`munmap` system calls directly | Custom executable memory allocator with pooling | Direct system calls teach fundamental concepts |
| Instruction Encoding | Hand-coded byte arrays with helper functions | Instruction encoding library or assembler | Manual encoding builds deep understanding |
| Profiling Storage | Simple arrays with linear search | Hash tables or B-trees for counter lookup | Arrays sufficient for educational scope |
| Error Handling | Return codes with explicit checking | Exception handling or error propagation macros | Explicit checking makes error paths visible |
| Testing Framework | Custom assertion macros | Full unit testing framework (CUnit, etc.) | Custom macros keep dependencies minimal |

#### Recommended File Structure

The implementation should organize code to clearly separate the different aspects of JIT compilation, making the system understandable and maintainable:

```
jit-compiler/
├── src/
│   ├── vm/
│   │   ├── vm.h                    # VM integration points
│   │   ├── vm.c                    # Extend existing bytecode VM
│   │   └── bytecode.h              # Bytecode instruction definitions
│   ├── jit/
│   │   ├── jit.h                   # Main JIT compiler interface
│   │   ├── jit.c                   # Compilation orchestration
│   │   ├── profiler.h              # Hot path detection interface
│   │   ├── profiler.c              # Execution counting and thresholds
│   │   ├── translator.h            # Bytecode-to-native translation
│   │   ├── translator.c            # Operation mapping logic
│   │   ├── emitter.h               # Machine code generation interface  
│   │   ├── emitter.c               # x86-64 instruction encoding
│   │   ├── memory.h                # Executable memory management
│   │   ├── memory.c                # mmap/mprotect wrapper functions
│   │   └── registers.h             # x86-64 register definitions
│   └── runtime/
│       ├── calling.h               # Calling convention support
│       ├── calling.c               # Prologue/epilogue generation
│       └── trampolines.c           # JIT-interpreter transitions
├── tests/
│   ├── test_emitter.c              # Machine code generation tests
│   ├── test_translator.c           # Translation correctness tests
│   ├── test_profiler.c             # Hot path detection tests
│   └── integration_tests.c         # End-to-end JIT compiler tests
└── benchmarks/
    ├── arithmetic_bench.c          # Compute-intensive test programs
    ├── loop_bench.c                # Loop-heavy benchmarks
    └── function_bench.c            # Function call benchmarks
```

#### Core Data Structure Definitions

These complete data structure definitions provide the foundation for JIT compiler implementation:

```c
#include <stdint.h>
#include <stddef.h>
#include <sys/mman.h>

// x86-64 register enumeration for instruction encoding
typedef enum {
    REG_RAX = 0, REG_RCX = 1, REG_RDX = 2, REG_RBX = 3,
    REG_RSP = 4, REG_RBP = 5, REG_RSI = 6, REG_RDI = 7,
    REG_R8 = 8,  REG_R9 = 9,  REG_R10 = 10, REG_R11 = 11,
    REG_R12 = 12, REG_R13 = 13, REG_R14 = 14, REG_R15 = 15
} X86Register;

// Executable memory buffer for generated machine code
typedef struct {
    void* memory;           // mmap'd memory region
    size_t size;            // Total allocated size
    size_t used;            // Bytes currently used
    int writable;           // 1 if writable, 0 if execute-only
} CodeBuffer;

// Function execution profiling data
typedef struct {
    uint32_t function_id;           // Unique identifier for function
    uint32_t invocation_count;      // Number of times called
    uint32_t compilation_threshold; // Calls needed to trigger compilation
    int is_compiled;                // 1 if native code exists
} ProfileEntry;

// Runtime profiler for hot path detection
typedef struct {
    ProfileEntry* entries;          // Array of per-function profile data
    size_t capacity;               // Maximum number of functions
    uint32_t default_threshold;    // Default compilation threshold
} Profiler;

// Function pointer type for calling JIT-compiled code
typedef int64_t (*CompiledFunction)(int64_t arg1, int64_t arg2, int64_t arg3);

// Bytecode function representation (interface to existing VM)
typedef struct {
    uint32_t function_id;          // Unique function identifier
    uint8_t* bytecode;             // Bytecode instruction array
    size_t bytecode_length;        // Number of bytecode bytes
    int argument_count;            // Function parameter count
    int local_variable_count;      // Local variable slots needed
} BytecodeFunction;

// Constants for x86-64 instruction encoding
#define REX_PREFIX          0x48    // REX.W prefix for 64-bit operations
#define MOV_REG_IMM_OPCODE  0xB8    // MOV register, immediate (base opcode)
#define ADD_REG_REG_OPCODE  0x01    // ADD register, register
#define SUB_REG_REG_OPCODE  0x29    // SUB register, register
#define CMP_REG_REG_OPCODE  0x39    // CMP register, register
#define RET_OPCODE          0xC3    // RET (return from function)

// Memory protection flags
#ifndef PROT_READ
#define PROT_READ  0x1
#define PROT_WRITE 0x2  
#define PROT_EXEC  0x4
#endif
```

#### Machine Code Emitter Infrastructure

Complete implementation of the machine code emitter that handles executable memory management and basic instruction encoding:

```c
// memory.c - Executable memory management
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

CodeBuffer* code_buffer_create(size_t size) {
    CodeBuffer* buffer = malloc(sizeof(CodeBuffer));
    if (!buffer) return NULL;
    
    // Allocate memory with read/write permissions initially
    buffer->memory = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buffer->memory == MAP_FAILED) {
        free(buffer);
        return NULL;
    }
    
    buffer->size = size;
    buffer->used = 0;
    buffer->writable = 1;
    return buffer;
}

int code_buffer_make_executable(CodeBuffer* buffer) {
    if (!buffer || !buffer->writable) return -1;
    
    // Change protection to read/execute (W^X security policy)
    if (mprotect(buffer->memory, buffer->size, PROT_READ | PROT_EXEC) != 0) {
        return -1;
    }
    
    buffer->writable = 0;
    return 0;
}

void code_buffer_destroy(CodeBuffer* buffer) {
    if (buffer) {
        if (buffer->memory != MAP_FAILED) {
            munmap(buffer->memory, buffer->size);
        }
        free(buffer);
    }
}

// emitter.c - Basic instruction encoding functions  
void emit_mov_reg_imm(CodeBuffer* buffer, X86Register reg, int64_t immediate) {
    if (!buffer || !buffer->writable) return;
    
    uint8_t* code = (uint8_t*)buffer->memory + buffer->used;
    
    // REX.W prefix for 64-bit operation
    *code++ = REX_PREFIX;
    
    // MOV opcode + register encoding
    *code++ = MOV_REG_IMM_OPCODE + (reg & 0x7);
    
    // 64-bit immediate value (little-endian)
    *(int64_t*)code = immediate;
    code += 8;
    
    buffer->used += 10; // REX + opcode + 8-byte immediate
}

void emit_add_reg_reg(CodeBuffer* buffer, X86Register dest, X86Register src) {
    if (!buffer || !buffer->writable) return;
    
    uint8_t* code = (uint8_t*)buffer->memory + buffer->used;
    
    // REX.W prefix for 64-bit operation
    *code++ = REX_PREFIX;
    
    // ADD opcode
    *code++ = ADD_REG_REG_OPCODE;
    
    // ModR/M byte: 11 (register mode) + src<<3 + dest
    *code++ = 0xC0 | ((src & 0x7) << 3) | (dest & 0x7);
    
    buffer->used += 3;
}

void emit_ret(CodeBuffer* buffer) {
    if (!buffer || !buffer->writable) return;
    
    uint8_t* code = (uint8_t*)buffer->memory + buffer->used;
    *code = RET_OPCODE;
    buffer->used += 1;
}
```

#### Core Logic Skeleton Functions

These skeleton functions define the interfaces that learners should implement, with detailed TODO comments mapping to the algorithm steps:

```c
// profiler.c - Hot path detection skeleton
Profiler* profiler_create(size_t max_functions, uint32_t default_threshold) {
    // TODO 1: Allocate Profiler structure
    // TODO 2: Allocate ProfileEntry array for max_functions entries
    // TODO 3: Initialize all entries with function_id = 0, counts = 0
    // TODO 4: Set default_threshold and capacity fields
    // TODO 5: Return initialized profiler or NULL on allocation failure
}

int profiler_record_invocation(Profiler* profiler, uint32_t function_id) {
    // TODO 1: Find ProfileEntry for function_id (linear search is fine)
    // TODO 2: If not found, allocate new entry if space available
    // TODO 3: Increment invocation_count for the function
    // TODO 4: Check if invocation_count >= compilation_threshold
    // TODO 5: Return 1 if compilation should be triggered, 0 otherwise
    // Hint: Use profiler->default_threshold for new functions
}

// translator.c - Bytecode-to-native translation skeleton
CompiledFunction compile_function(BytecodeFunction* func, CodeBuffer* buffer) {
    // TODO 1: Generate function prologue (save rbp, set up stack frame)
    // TODO 2: Iterate through bytecode instructions
    // TODO 3: For each instruction, call appropriate emit_* function
    // TODO 4: Handle LOAD_CONST -> emit_mov_reg_imm
    // TODO 5: Handle ADD -> emit_add_reg_reg  
    // TODO 6: Handle return values in RAX register
    // TODO 7: Generate function epilogue (restore rbp, ret)
    // TODO 8: Make buffer executable and return function pointer
    // Hint: Cast buffer->memory to CompiledFunction after making executable
}

// Integration skeleton - main JIT orchestration
int jit_execute_function(uint32_t function_id, BytecodeFunction* func, 
                        int64_t* args, int64_t* result) {
    // TODO 1: Check if function already compiled (lookup in code cache)
    // TODO 2: If compiled, call native function directly and return result
    // TODO 3: Record invocation in profiler
    // TODO 4: Check if profiler indicates compilation should be triggered
    // TODO 5: If trigger activated, compile function and cache result
    // TODO 6: Execute appropriate version (interpreted or compiled)
    // TODO 7: Handle compilation failures gracefully (fall back to interpreter)
}
```

#### Milestone Checkpoints

After completing each milestone, verify progress with these concrete checkpoints:

**Milestone 1 Checkpoint - Basic x86-64 Emitter:**
```bash
# Compile and run emitter tests
gcc -o test_emitter test_emitter.c src/jit/memory.c src/jit/emitter.c
./test_emitter

# Expected output:
# Testing executable memory allocation... PASS
# Testing MOV instruction encoding... PASS  
# Testing ADD instruction encoding... PASS
# Testing generated code execution... PASS
# Simple calculation result: 42
```

**Milestone 2 Checkpoint - Expression JIT:**
```bash
# Test arithmetic expression compilation
gcc -o test_translator test_translator.c src/jit/*.c src/vm/vm.c
./test_translator

# Expected behavior:
# Compiles bytecode sequence: LOAD_CONST 10, LOAD_CONST 32, ADD, RETURN
# JIT result matches interpreter result: 42
# Performance improvement visible on repeated execution
```

#### Language-Specific Implementation Hints

- **Memory Management**: Use `mmap` with `MAP_ANONYMOUS | MAP_PRIVATE` for executable memory allocation. Always check for `MAP_FAILED` return value.
- **Instruction Encoding**: x86-64 instructions are variable length. Keep track of buffer->used carefully and ensure you don't exceed buffer->size.
- **Function Pointers**: Cast generated code to function pointer only after calling `code_buffer_make_executable`. Calling writable memory as a function causes segmentation fault.
- **Register Encoding**: x86-64 registers R8-R15 require REX prefix. Use `(reg & 0x7)` for the low 3 bits in ModR/M encoding.
- **Stack Alignment**: System V AMD64 ABI requires 16-byte stack alignment before `call` instruction. Use `pushq %rbp; movq %rsp, %rbp` in function prologue.
- **Debugging Generated Code**: Use `objdump -D -b binary -m i386:x86-64 <code_file>` to disassemble generated machine code for debugging.

These implementation guidelines provide the infrastructure and interfaces needed to build a working JIT compiler while leaving the core learning challenges for the developer to solve.


## High-Level Architecture

> **Milestone(s):** All milestones — the architectural foundation supports the complete JIT compiler implementation from basic instruction emission through hot path optimization

The architecture of a JIT compiler presents a unique challenge: it must seamlessly bridge two distinct execution paradigms while maintaining both correctness and performance. Think of this as designing a hybrid vehicle that can switch between electric and combustion engines mid-journey — the transition must be invisible to the passenger, yet each engine operates on completely different principles. Our JIT compiler similarly orchestrates multiple specialized components that work together to transform slow bytecode interpretation into fast native execution.

The architectural complexity stems from managing multiple concurrent concerns: tracking execution patterns to identify optimization opportunities, translating high-level operations into low-level machine instructions, managing memory with specific security constraints, and coordinating between interpreted and compiled code execution modes. Each component has distinct responsibilities yet must integrate smoothly with both the existing bytecode VM and other JIT subsystems.

![JIT Compiler System Architecture](./diagrams/system-architecture.svg)

### Component Overview and Responsibilities

The JIT compiler architecture consists of four primary components that extend the existing bytecode VM: the **Runtime Profiler** for execution tracking, the **Bytecode-to-Native Translator** for code transformation, the **Machine Code Emitter** for low-level instruction generation, and the **Code Cache Manager** for executable memory management. Each component addresses a specific aspect of the compilation challenge while maintaining clean interfaces for integration.

**Runtime Profiler** serves as the intelligence gathering component of the system. It monitors bytecode execution patterns by instrumenting function calls and loop iterations with lightweight counters. The profiler maintains a `Profiler` structure containing an array of `ProfileEntry` records, each tracking a specific function's invocation frequency against configurable compilation thresholds. When a function's execution count exceeds its threshold, the profiler signals the compilation pipeline to transform that function from bytecode to native code.

| Component | Primary Responsibility | Key Data Structures | Interface Points |
|-----------|----------------------|-------------------|-----------------|
| Runtime Profiler | Execution frequency tracking and hot path detection | `Profiler`, `ProfileEntry` | `profiler_record_invocation()`, threshold checking |
| Bytecode-to-Native Translator | High-level operation mapping and register allocation | Translation tables, register mappings | `compile_function()`, operation translators |
| Machine Code Emitter | x86-64 instruction encoding and memory management | `CodeBuffer`, instruction encoders | `emit_mov_reg_imm()`, `emit_add_reg_reg()` |
| Code Cache Manager | Executable memory allocation and function dispatch | Compiled function registry, memory pools | `code_buffer_create()`, `code_buffer_make_executable()` |

**Bytecode-to-Native Translator** acts as the system's linguistic interpreter, converting abstract bytecode operations into concrete machine instruction sequences. This component understands both the semantics of bytecode operations (stack manipulations, arithmetic, control flow) and the capabilities of the x86-64 architecture (registers, addressing modes, instruction formats). It implements a simple register allocation strategy that maps commonly-used stack slots to machine registers, reducing memory traffic for performance-critical operations.

**Machine Code Emitter** functions as the system's assembly line, taking high-level instruction requests and producing precise binary machine code. It handles the intricate details of x86-64 instruction encoding, including REX prefix generation for 64-bit operations, ModR/M byte construction for register/memory operand specification, and immediate value encoding. The emitter manages `CodeBuffer` structures that represent regions of executable memory, handling the complex security requirements of modern operating systems that enforce W^X (write-exclusive-or-execute) policies.

**Code Cache Manager** serves as the system's memory orchestrator, allocating executable memory regions and maintaining the mapping between bytecode functions and their compiled native equivalents. It interfaces with the operating system's memory management through `mmap` system calls, requesting pages with appropriate protection flags (`PROT_READ|PROT_WRITE|PROT_EXEC` or transitioning between writable and executable states). The cache manager also handles function pointer generation and dispatch, enabling seamless transitions between interpreted and compiled execution modes.

> The key architectural insight is that each component operates at a different abstraction level — the profiler works with execution patterns, the translator works with operation semantics, the emitter works with instruction encoding, and the cache manager works with memory management — yet they must coordinate precisely to maintain system correctness.

**Component Interaction Patterns** follow a clear dependency hierarchy. The Runtime Profiler operates independently, monitoring execution without knowledge of compilation details. When a compilation trigger fires, the Profiler notifies the Translator, which requests services from both the Emitter (for instruction generation) and the Cache Manager (for memory allocation). The Emitter and Cache Manager work closely together, as instruction generation requires writable memory that must later become executable.

**Error Propagation Strategy** flows upward through the component hierarchy. Low-level failures (memory allocation errors, instruction encoding errors) propagate to higher-level components, which can make informed decisions about fallback strategies. For example, if the Machine Code Emitter encounters an encoding error for a complex instruction, the Translator can attempt a simpler instruction sequence. If compilation fails entirely, execution falls back to bytecode interpretation with no visible impact on program correctness.

### VM Integration Strategy

The JIT compiler extends the existing bytecode VM through a **non-invasive augmentation** approach that preserves the VM's original interpretation capabilities while adding compilation as an optional performance enhancement. Think of this as adding a turbocharger to an existing engine — the engine continues to operate normally, but when conditions are right, the turbocharger engages to provide additional power.

> **Decision: Transparent JIT Integration**
> - **Context**: The JIT compiler must integrate with an existing bytecode VM without breaking existing functionality or requiring significant VM modifications
> - **Options Considered**: 
>   1. VM Core Modification (embed JIT directly into interpreter loop)
>   2. Transparent Augmentation (JIT as separate layer that hooks into VM)
>   3. Separate JIT VM (completely new VM with JIT built-in)
> - **Decision**: Transparent Augmentation approach
> - **Rationale**: Preserves existing VM stability, allows gradual JIT feature development, enables easy fallback to interpretation, and minimizes integration complexity
> - **Consequences**: Enables incremental JIT development but requires careful attention to VM state synchronization and calling convention compatibility

| Integration Approach | VM Modification Required | Fallback Capability | Development Risk | Chosen? |
|---------------------|-------------------------|-------------------|-----------------|---------|
| VM Core Modification | Extensive changes to interpreter loop | Limited - hard to disable JIT | High - can break existing VM | No |
| Transparent Augmentation | Minimal - only function dispatch hooks | Complete - seamless fallback | Low - JIT operates independently | **Yes** |
| Separate JIT VM | Complete rewrite required | None - no original VM | Very High - rebuild everything | No |

**Function Dispatch Augmentation** represents the primary integration point between the VM and JIT compiler. The existing VM's function call mechanism gains a lightweight check that consults the Code Cache Manager to determine whether a compiled version of the target function exists. If a compiled version is available, execution transfers to the native code through a `CompiledFunction` function pointer. If no compiled version exists, execution proceeds through the original bytecode interpretation path.

The dispatch augmentation requires minimal changes to the existing VM structure:

| VM Component | Modification Required | Purpose | Implementation Complexity |
|--------------|---------------------|---------|--------------------------|
| Function Call Handler | Add compiled function lookup | Check for JIT-compiled version before interpretation | Low - single conditional check |
| Stack Frame Setup | Add native/interpreted mode flag | Track execution mode for proper cleanup | Low - single flag addition |
| Return Handler | Add mode-aware return processing | Handle returns from both native and interpreted functions | Medium - dual return paths |
| Exception Handler | Add native code exception support | Propagate exceptions from JIT-compiled code | Medium - exception translation |

**VM State Synchronization** ensures that both interpreted and compiled code operate on consistent program state. The bytecode VM maintains its execution state in specific data structures (stack, local variables, program counter), while native code operates directly on machine registers and stack frames. The JIT integration layer provides **state marshaling** functions that convert between these representations when transitioning between execution modes.

**Memory Management Integration** leverages the existing VM's memory allocation infrastructure while adding JIT-specific executable memory management. The Code Cache Manager coordinates with the VM's garbage collector (if present) to ensure that compiled functions remain valid while their corresponding bytecode functions are reachable. When bytecode functions become unreachable, their compiled counterparts can be safely deallocated, freeing executable memory for new compilations.

**Profiling Integration** instruments the existing VM's function call mechanism with lightweight execution counters. The profiler adds a single `profiler_record_invocation()` call to the VM's function dispatch logic, incrementing the invocation count for each called function. This instrumentation adds minimal overhead to interpreted execution while providing the data necessary for hot path detection.

> The critical design principle is that JIT integration should be **additive rather than transformative** — the VM gains new capabilities without losing existing functionality, and JIT failures never compromise program correctness.

**Compilation Trigger Integration** operates asynchronously with respect to the main execution thread. When the profiler identifies a hot function, it can trigger compilation in several ways:

1. **Immediate Synchronous Compilation**: Pause execution, compile the function, then resume with native code
2. **Deferred Asynchronous Compilation**: Continue interpretation while compilation occurs in background, use compiled version on subsequent calls
3. **Threshold-Based Compilation**: Compile after multiple invocations exceed threshold, balancing compilation cost against execution benefit

| Trigger Strategy | Compilation Latency | Implementation Complexity | Impact on Current Execution | Best For |
|------------------|-------------------|--------------------------|---------------------------|----------|
| Immediate Synchronous | Zero (compiled immediately) | Low - no threading | Pause during compilation | Short functions, low compilation cost |
| Deferred Asynchronous | One call (available next invocation) | High - requires threading | No pause | Long-running applications |
| Threshold-Based | Multiple calls until threshold | Medium - counter management | Minimal pause | Balanced performance/complexity |

Our implementation uses **Threshold-Based Compilation** for its optimal balance of performance benefit and implementation simplicity, with configurable thresholds per function based on estimated compilation cost.

### Recommended File Structure

The JIT compiler implementation follows a modular organization that separates concerns while maintaining clear dependency relationships. Each component resides in its own source file with well-defined interfaces, enabling independent development and testing of JIT subsystems.

```
jit-compiler/
├── src/
│   ├── vm/
│   │   ├── vm_core.c              ← Original VM implementation (minimal modifications)
│   │   ├── vm_core.h              ← VM public interface
│   │   └── bytecode.h             ← Bytecode operation definitions
│   │
│   ├── jit/
│   │   ├── jit_compiler.c         ← Main JIT orchestration and public API
│   │   ├── jit_compiler.h         ← JIT compiler public interface
│   │   │
│   │   ├── profiler.c             ← Runtime profiling and hot path detection
│   │   ├── profiler.h             ← Profiler data structures and interface
│   │   │
│   │   ├── translator.c           ← Bytecode to native translation logic
│   │   ├── translator.h           ← Translation interface and register allocation
│   │   │
│   │   ├── emitter.c              ← x86-64 machine code emission
│   │   ├── emitter.h              ← Instruction encoding and CodeBuffer management
│   │   │
│   │   └── cache.c                ← Compiled function cache and executable memory
│   │   └── cache.h                ← Code cache interface and memory management
│   │
│   ├── platform/
│   │   ├── memory_linux.c         ← Linux-specific executable memory allocation
│   │   ├── memory_macos.c         ← macOS-specific executable memory allocation
│   │   ├── memory_windows.c       ← Windows-specific executable memory allocation
│   │   └── platform.h             ← Platform abstraction interface
│   │
│   └── main.c                     ← Example driver program and benchmarks
│
├── tests/
│   ├── unit/
│   │   ├── test_profiler.c        ← Profiler unit tests
│   │   ├── test_translator.c      ← Translation correctness tests
│   │   ├── test_emitter.c         ← Machine code generation tests
│   │   └── test_cache.c           ← Memory management tests
│   │
│   ├── integration/
│   │   ├── test_jit_integration.c ← End-to-end JIT compilation tests
│   │   └── test_vm_integration.c  ← VM and JIT interaction tests
│   │
│   └── benchmarks/
│       ├── benchmark_suite.c      ← Performance comparison benchmarks
│       └── test_programs/         ← Bytecode test programs for benchmarking
│
├── docs/
│   └── design.md                  ← This design document
│
└── Makefile                       ← Build configuration and targets
```

**Source File Organization Rationale** separates the JIT compiler into logical components that can be developed and tested independently. The `jit/` directory contains all JIT-specific code, while the `vm/` directory contains the original bytecode VM with minimal modifications. This organization supports incremental development where each milestone can focus on a specific component without affecting others.

**Header File Dependency Structure** follows a clear hierarchy to avoid circular dependencies:

| Header File | Dependencies | Exports | Used By |
|-------------|-------------|---------|---------|
| `bytecode.h` | None | Bytecode operation definitions | All JIT components |
| `platform.h` | System headers | Memory allocation interface | `emitter.h`, `cache.h` |
| `profiler.h` | `bytecode.h` | `Profiler`, `ProfileEntry` types | `jit_compiler.h` |
| `emitter.h` | `platform.h` | `CodeBuffer`, instruction emission | `translator.h`, `cache.h` |
| `cache.h` | `emitter.h` | Code cache and function dispatch | `jit_compiler.h` |
| `translator.h` | `emitter.h`, `bytecode.h` | Bytecode translation interface | `jit_compiler.h` |
| `jit_compiler.h` | All JIT headers | Main JIT API | `vm_core.h`, user code |

**Platform Abstraction Layer** isolates operating system-specific functionality into separate files, enabling portability across Linux, macOS, and Windows. Each platform implements the same interface defined in `platform.h`, but uses appropriate system calls (`mmap` on Unix systems, `VirtualAlloc` on Windows) for executable memory management.

**Test Organization Structure** supports comprehensive validation at multiple levels. Unit tests verify individual component functionality in isolation, integration tests validate component interactions, and benchmarks measure performance improvements. The `test_programs/` directory contains representative bytecode programs that exercise various JIT compilation scenarios, from simple arithmetic to complex control flow patterns.

> The modular file structure enables **independent milestone development** — Milestone 1 focuses on `emitter.c`, Milestone 2 adds `translator.c`, Milestone 3 integrates with `cache.c`, and Milestone 4 completes with `profiler.c`. Each milestone can be tested and validated before proceeding to the next.

**Build Configuration Strategy** uses a single Makefile with targets for each development phase:

| Make Target | Purpose | Files Compiled | Output |
|-------------|---------|---------------|---------|
| `make milestone1` | Basic x86-64 emitter | `emitter.c`, `platform.c`, test harness | Instruction emission demo |
| `make milestone2` | Expression JIT | Add `translator.c`, `bytecode.h` | Arithmetic expression compiler |
| `make milestone3` | Function JIT | Add `cache.c`, `vm_core.c` integration | Full function compilation |
| `make milestone4` | Hot path optimization | Add `profiler.c`, complete integration | Production JIT compiler |
| `make test` | All unit tests | All source files, test files | Test suite execution |
| `make benchmark` | Performance evaluation | All source files, benchmark programs | Performance measurements |

This build structure supports incremental development where each milestone produces a working, testable system that validates the implemented functionality before adding complexity.

### Implementation Guidance

**Technology Stack Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Memory Management | `mmap()` with manual protection changes | Memory pool allocator with multiple regions |
| Instruction Encoding | Direct byte array manipulation | Structured instruction builder with validation |
| Platform Abstraction | Preprocessor conditionals (#ifdef) | Runtime platform detection with function pointers |
| Error Handling | Return codes with errno | Structured exception handling with cleanup |
| Testing Framework | Simple assert() macros | Full unit test framework (CUnit, Unity) |
| Build System | Single Makefile with basic rules | CMake with cross-platform configuration |

**Recommended Implementation Sequence:**

The milestone-driven development approach ensures each phase builds upon stable foundations:

1. **Milestone 1 Foundation**: Implement `emitter.c` with basic instruction encoding and executable memory allocation
2. **Milestone 2 Translation**: Add `translator.c` with bytecode-to-native mapping for arithmetic operations
3. **Milestone 3 Integration**: Implement `cache.c` and integrate with existing VM for function-level compilation
4. **Milestone 4 Optimization**: Add `profiler.c` for hot path detection and threshold-based compilation

**Platform-Specific Memory Management:**

```c
// platform/memory_linux.c - Linux/macOS implementation
#include <sys/mman.h>
#include <unistd.h>

void* allocate_executable_memory(size_t size) {
    // Request readable, writable, executable memory
    void* memory = mmap(NULL, size, 
                       PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (memory == MAP_FAILED) {
        return NULL;
    }
    
    return memory;
}

int make_memory_executable(void* memory, size_t size) {
    // Some systems require W^X policy - change from writable to executable
    return mprotect(memory, size, PROT_READ | PROT_EXEC);
}

void deallocate_executable_memory(void* memory, size_t size) {
    munmap(memory, size);
}
```

```c
// platform/memory_windows.c - Windows implementation  
#include <windows.h>

void* allocate_executable_memory(size_t size) {
    // Use VirtualAlloc for executable memory on Windows
    return VirtualAlloc(NULL, size,
                       MEM_COMMIT | MEM_RESERVE,
                       PAGE_EXECUTE_READWRITE);
}

int make_memory_executable(void* memory, size_t size) {
    DWORD old_protection;
    return VirtualProtect(memory, size, PAGE_EXECUTE_READ, &old_protection);
}

void deallocate_executable_memory(void* memory, size_t size) {
    VirtualFree(memory, 0, MEM_RELEASE);
}
```

**Core JIT Compiler Interface:**

```c
// jit/jit_compiler.h - Main JIT compiler API
#ifndef JIT_COMPILER_H
#define JIT_COMPILER_H

#include "vm/bytecode.h"
#include "profiler.h"
#include "cache.h"

typedef struct {
    Profiler* profiler;
    CodeCache* cache;
    int compilation_enabled;
    uint32_t default_threshold;
} JITCompiler;

// Initialize JIT compiler with default configuration
JITCompiler* jit_compiler_create(uint32_t default_threshold);

// Clean up JIT compiler and free all resources
void jit_compiler_destroy(JITCompiler* jit);

// Record function invocation and trigger compilation if threshold exceeded
int jit_compiler_record_call(JITCompiler* jit, uint32_t function_id, 
                             BytecodeFunction* function);

// Look up compiled version of function (returns NULL if not compiled)
CompiledFunction jit_compiler_get_compiled(JITCompiler* jit, uint32_t function_id);

// Force compilation of specific function (for testing/debugging)
int jit_compiler_compile_function(JITCompiler* jit, uint32_t function_id,
                                  BytecodeFunction* function);

#endif
```

**JIT Compiler Orchestration Implementation:**

```c
// jit/jit_compiler.c - Main coordination logic
#include "jit_compiler.h"
#include "translator.h"
#include <stdlib.h>

JITCompiler* jit_compiler_create(uint32_t default_threshold) {
    // TODO 1: Allocate JITCompiler structure
    // TODO 2: Initialize profiler with default threshold
    // TODO 3: Initialize code cache for compiled functions
    // TODO 4: Set compilation enabled flag
    // TODO 5: Return initialized JIT compiler or NULL on failure
    // Hint: Each component has its own create function
}

int jit_compiler_record_call(JITCompiler* jit, uint32_t function_id,
                            BytecodeFunction* function) {
    // TODO 1: Call profiler_record_invocation() for this function
    // TODO 2: Check if return value indicates threshold exceeded
    // TODO 3: If threshold exceeded, trigger compilation
    // TODO 4: Call compile_function() to generate native code
    // TODO 5: Store compiled function in cache for future calls
    // Hint: Use jit_compiler_compile_function() for actual compilation
}

CompiledFunction jit_compiler_get_compiled(JITCompiler* jit, uint32_t function_id) {
    // TODO 1: Look up function in code cache
    // TODO 2: Return compiled function pointer or NULL if not found
    // Hint: This is a simple cache lookup operation
}
```

**VM Integration Points:**

```c
// vm/vm_core.c - Modified function dispatch (minimal changes to existing VM)
#include "jit/jit_compiler.h"

// Global JIT compiler instance (initialized at VM startup)
extern JITCompiler* global_jit_compiler;

// Modified function call handler (existing VM function with JIT augmentation)
int vm_call_function(VM* vm, uint32_t function_id) {
    BytecodeFunction* function = vm_get_function(vm, function_id);
    
    // JIT integration: record call and check for compiled version
    jit_compiler_record_call(global_jit_compiler, function_id, function);
    CompiledFunction compiled = jit_compiler_get_compiled(global_jit_compiler, function_id);
    
    if (compiled != NULL) {
        // TODO 1: Set up arguments for native calling convention
        // TODO 2: Call compiled function with proper argument passing
        // TODO 3: Handle return value and update VM state
        // TODO 4: Return from VM function call
        // Hint: This requires System V AMD64 calling convention
    }
    
    // Fallback: execute via original bytecode interpretation
    return vm_interpret_function(vm, function);
}
```

**Milestone Verification Checkpoints:**

After implementing each milestone, verify functionality with these checkpoints:

**Milestone 1 Checkpoint:**
```bash
# Compile and run basic emitter test
make milestone1
./test_emitter

# Expected output:
# Generated machine code: 48 B8 2A 00 00 00 00 00 00 00 C3
# Function call result: 42
# Test passed: basic instruction emission working
```

**Milestone 2 Checkpoint:**
```bash
# Compile and run expression translation test
make milestone2  
./test_translator

# Expected output:
# Compiling expression: LOAD_CONST 10, LOAD_CONST 20, ADD, RETURN
# Generated code size: 23 bytes
# Interpreter result: 30
# JIT result: 30
# Test passed: expression compilation matches interpretation
```

**Milestone 3 Checkpoint:**
```bash
# Run full VM integration test
make milestone3
./test_vm_integration

# Expected output:
# Function factorial(5) - Interpreter: 120, JIT: 120
# Function fibonacci(10) - Interpreter: 55, JIT: 55
# Test passed: function compilation produces correct results
```

**Milestone 4 Checkpoint:**
```bash
# Run performance benchmark
make milestone4
./benchmark_suite

# Expected output:
# Running arithmetic_heavy benchmark...
# Interpreter time: 1.234s
# JIT time: 0.456s  
# Speedup: 2.7x
# Hot path compilation triggered after 100 calls
# Test passed: JIT provides significant speedup
```

**Common Implementation Pitfalls:**

⚠️ **Pitfall: Executable Memory Permissions**
Many platforms now enforce W^X security policies where memory pages cannot be both writable and executable simultaneously. Attempting to write machine code to executable memory will cause segmentation faults. Always allocate memory as writable, generate code, then change protection to executable before calling the generated function.

⚠️ **Pitfall: x86-64 Instruction Encoding Errors**
x86-64 instruction encoding is complex, with multiple prefixes, opcodes, and addressing modes. The REX prefix (0x48) is required for 64-bit register operations, and the ModR/M byte determines register encoding. Incorrect encoding produces invalid instructions that cause SIGILL crashes. Validate generated instructions with a disassembler during development.

⚠️ **Pitfall: Stack Alignment Requirements**
The System V AMD64 ABI requires the stack pointer to be 16-byte aligned before function calls. Misaligned stacks cause crashes in optimized code or when calling system functions. Always ensure proper stack alignment in generated function prologues and when calling between JIT and interpreted code.

⚠️ **Pitfall: Memory Leaks in Executable Memory**
Executable memory allocated with `mmap()` or `VirtualAlloc()` must be explicitly freed with corresponding deallocation functions. Standard `free()` cannot deallocate executable memory. Track all executable memory allocations and ensure proper cleanup to avoid resource leaks.


## Data Model

> **Milestone(s):** All milestones — the data model provides the foundational structures that support the complete JIT compiler from basic machine code emission through advanced hot path optimization

The data model serves as the **structural backbone** of our JIT compiler, defining how we represent, transform, and manage code at every stage of execution. Think of it as the **architectural blueprint** that defines the rooms, corridors, and infrastructure of a building — every component must understand and interact with these fundamental structures to create a cohesive system. The data model bridges three distinct worlds: the high-level bytecode representation from our VM, the low-level machine code we generate, and the runtime profiling information that drives optimization decisions.

### Mental Model: Digital DNA of Code Execution

Imagine the data model as the **digital DNA** of our JIT compiler — just as biological DNA contains the instructions for building and maintaining a living organism, our data structures contain all the information needed to transform, execute, and optimize code at runtime. Each structure serves a specific role:

- **BytecodeFunction** acts like the original genetic template, containing the pure instruction sequence and metadata
- **CompiledFunction** represents the evolved, optimized form after environmental adaptation (JIT compilation)  
- **CodeBuffer** serves as the cellular environment where transformation occurs, providing the space and tools for machine code synthesis
- **ProfileEntry** functions like cellular memory, recording environmental conditions (execution frequency) that trigger evolutionary adaptations (compilation)

This biological analogy helps us understand that our data model isn't just static storage — it's a living system where structures evolve and adapt based on runtime conditions.

### Core Data Structures

The foundation of our JIT compiler rests on four primary data structures that represent different aspects of code compilation and execution. Each structure has been carefully designed to support efficient transformation between bytecode interpretation and native execution while maintaining the profiling information necessary for optimization decisions.

#### BytecodeFunction Structure

The `BytecodeFunction` represents the source material for JIT compilation, containing all the information necessary to understand and compile a function from its bytecode representation. This structure serves as the immutable template that describes what the function does and how it should be compiled.

| Field | Type | Description |
|-------|------|-------------|
| function_id | uint32_t | Unique identifier for this function across the VM |
| bytecode | uint8_t* | Pointer to the raw bytecode instruction sequence |
| bytecode_length | size_t | Total number of bytes in the bytecode array |
| argument_count | int | Number of parameters this function expects |
| local_variable_count | int | Number of local variables allocated on the stack frame |

The `BytecodeFunction` structure embodies several important design decisions. The `function_id` serves as a stable identifier that remains constant across compilation and execution, enabling the profiler to track the same logical function through different execution modes. The raw bytecode pointer approach allows direct memory access to instruction sequences without additional indirection, critical for performance in the compilation hot path.

> **Decision: Separate Bytecode Length Field**
> - **Context**: Bytecode arrays need bounds checking during compilation to prevent buffer overruns
> - **Options Considered**: 
>   - Null-terminated bytecode (like C strings)
>   - Length embedded in first bytes of bytecode
>   - Separate length field
> - **Decision**: Separate length field
> - **Rationale**: Explicit length checking prevents compilation errors and supports bytecode sequences containing zero bytes as valid instruction data
> - **Consequences**: Slightly larger structure but eliminates entire class of bounds-checking bugs during compilation

The argument and local variable counts provide essential information for stack frame calculation during native code generation, allowing the compiler to allocate the correct amount of stack space and generate proper function prologues.

#### CompiledFunction Type

The `CompiledFunction` represents the final result of JIT compilation — a function pointer to executable native machine code that can be called directly from C. This type bridges the gap between our managed bytecode world and the unmanaged native execution environment.

| Component | Type | Description |
|-----------|------|-------------|
| Function Signature | int64_t (*)(int64_t*, int64_t*, size_t) | Pointer to compiled native code |
| Parameters | int64_t* args | Pointer to argument array |
| Parameters | int64_t* locals | Pointer to local variable array |  
| Parameters | size_t frame_size | Size of the stack frame in bytes |
| Return Value | int64_t | Function result value |

The `CompiledFunction` follows the System V AMD64 calling convention, accepting arguments and locals as pointer arrays to maintain compatibility with the VM's data representation. This design allows seamless transition between interpreted and compiled execution without complex marshaling operations.

> **Decision: Function Pointer with Array Parameters**
> - **Context**: Need efficient way to pass VM state to compiled native code
> - **Options Considered**:
>   - Individual register parameters (rdi, rsi, rdx, etc.)
>   - Single structure containing all VM state
>   - Array pointers for arguments and locals
> - **Decision**: Array pointers for arguments and locals
> - **Rationale**: Provides direct memory access to VM data without copying, supports variable argument counts, and simplifies native code generation
> - **Consequences**: Requires pointer indirection but eliminates data copying and supports functions with any number of parameters

#### CodeBuffer Structure

The `CodeBuffer` manages executable memory allocation and machine code generation, serving as both the workspace for code emission and the final container for executable instructions. This structure handles the complex requirements of executable memory management while providing a simple interface for instruction emission.

| Field | Type | Description |
|-------|------|-------------|
| memory | void* | Pointer to allocated executable memory region |
| size | size_t | Total size of allocated memory buffer |
| used | size_t | Number of bytes currently used for generated code |
| writable | int | Boolean flag indicating current memory protection state |

The `CodeBuffer` design addresses the security requirements of modern operating systems that implement **W^X (Write XOR Execute) policies**. The `writable` flag tracks whether the memory is currently in write mode (for code generation) or execute mode (for function calls), enabling the buffer to toggle between these states safely.

> The critical insight here is that executable memory cannot be writable and executable simultaneously on modern systems. The CodeBuffer abstracts this complexity by managing protection state transitions automatically.

The buffer maintains both total size and used space to support incremental code generation while preventing buffer overruns. This design enables multiple functions to share the same code buffer while maintaining proper bounds checking.

#### ProfileEntry Structure  

The `ProfileEntry` tracks execution frequency and compilation state for each function, providing the data necessary for hot path detection and compilation triggering. This structure bridges the gap between execution monitoring and compilation decisions.

| Field | Type | Description |
|-------|------|-------------|
| function_id | uint32_t | Unique identifier matching BytecodeFunction.function_id |
| invocation_count | uint32_t | Number of times this function has been called |
| compilation_threshold | uint32_t | Invocation count that triggers JIT compilation |
| is_compiled | int | Boolean flag indicating whether function is already compiled |

The profile entry design supports **threshold-based compilation** by maintaining both current invocation counts and the threshold that triggers compilation. The `is_compiled` flag prevents redundant compilation attempts and enables efficient lookup of already-compiled functions.

| Current State | Invocation Count | Action Taken |
|---------------|------------------|--------------|
| Cold | < threshold | Increment counter, continue interpreting |
| Hot | >= threshold | Trigger JIT compilation |
| Compiled | Any | Call compiled code directly |

#### Profiler Structure

The `Profiler` manages the collection of profile entries across all functions in the system, providing centralized execution frequency tracking and compilation triggering logic.

| Field | Type | Description |
|-------|------|-------------|
| entries | ProfileEntry* | Array of profile entries for all tracked functions |
| capacity | size_t | Maximum number of functions that can be tracked |
| default_threshold | uint32_t | Default invocation count threshold for new functions |

The profiler maintains a simple array-based structure optimized for frequent access during function invocation. The fixed capacity design avoids dynamic memory allocation during profiling operations, ensuring consistent performance overhead.

### Bytecode to Native Code Mapping

The transformation from bytecode instructions to native x86-64 machine code requires a systematic mapping that preserves semantic correctness while enabling efficient execution. This mapping forms the core translation logic of our JIT compiler.

#### Mental Model: Code Transformation Pipeline

Think of bytecode-to-native mapping as a **language translation service** with specialized dictionaries. Just as translating from English to French requires understanding both grammar structures and cultural context, translating bytecode to machine code requires understanding both instruction semantics and hardware capabilities. Our mapping tables serve as the translation dictionaries, while the register allocator manages the grammatical structure of the target language.

#### Instruction Mapping Table

Each bytecode operation maps to one or more x86-64 instruction sequences, with the complexity varying based on the operation's semantic requirements.

| Bytecode Operation | x86-64 Sequence | Registers Used | Special Handling |
|-------------------|-----------------|----------------|------------------|
| LOAD_CONST | mov rax, imm64 | RAX | 64-bit immediate requires REX prefix |
| ADD | add rax, rbx | RAX, RBX | Result stored in first operand |
| SUB | sub rax, rbx | RAX, RBX | Order matters: RAX = RAX - RBX |
| MUL | imul rax, rbx | RAX, RBX | Signed multiplication |
| DIV | cqo; idiv rbx | RAX, RDX, RBX | Dividend in RDX:RAX, quotient in RAX |
| CMP | cmp rax, rbx | RAX, RBX | Sets flags for conditional jumps |
| JMP_IF_EQUAL | je label | None | Requires forward patching for target |
| CALL | call address | All | Must preserve callee-saved registers |
| RETURN | mov rax, result; ret | RAX | Return value in RAX per ABI |

The mapping reveals several complexity patterns. Simple operations like addition translate directly to single instructions, while complex operations like division require multi-instruction sequences with specific register constraints.

⚠️ **Pitfall: Division Register Requirements**
Division operations require the dividend to be split across RDX (high 64 bits) and RAX (low 64 bits) before calling `idiv`. Forgetting to emit the `cqo` instruction to sign-extend RAX into RDX will cause incorrect results for negative numbers and potential crashes from invalid division states.

#### Register Allocation Strategy

The VM's stack-based execution model must be mapped to the x86-64 register-based architecture. Our simple register allocator uses a **stack simulation** approach where the top stack elements are maintained in registers when possible.

| Stack Depth | Preferred Register | Backup Strategy |
|-------------|-------------------|-----------------|
| Top (TOS) | RAX | Always in register |
| TOS-1 | RBX | Spill to memory if needed |
| TOS-2 | RCX | Spill to memory if needed |
| TOS-3 | RDX | Spill to memory if needed |
| Deeper | Memory | Stack frame allocation |

This allocation strategy provides good performance for common expression evaluation while maintaining simplicity. The register allocator tracks the current mapping and generates load/store instructions when spilling is required.

> **Decision: Simple Stack Simulation**
> - **Context**: VM uses stack-based execution while x86-64 uses registers
> - **Options Considered**:
>   - Full graph-coloring register allocation
>   - Linear scan allocation  
>   - Simple stack simulation with fixed register mapping
> - **Decision**: Simple stack simulation
> - **Rationale**: Provides 80% of the performance benefit with 20% of the implementation complexity, suitable for educational JIT compiler
> - **Consequences**: May generate suboptimal code for complex expressions but enables straightforward implementation and debugging

#### Jump Target Resolution

Bytecode jump instructions reference target addresses that may not be known during initial code generation, requiring a **forward patching** mechanism to resolve addresses after code generation completes.

| Jump Type | Resolution Strategy | Data Structure |
|-----------|-------------------|----------------|
| Backward Jump | Direct address calculation | Known target offset |
| Forward Jump | Patch list with placeholders | Array of patch locations |
| Conditional Forward | Patch list + condition code | Condition flags + patch list |

The forward patching process maintains a list of incomplete jump instructions during code generation, then resolves all addresses in a second pass after code generation completes.

```
Forward Jump Resolution Process:
1. Encounter forward jump instruction
2. Emit jump opcode with placeholder target (0x00000000)  
3. Record jump location in patch list
4. Continue code generation
5. Encounter jump target label
6. Record target address
7. After code generation: iterate patch list
8. Calculate relative offsets for each jump
9. Update placeholder addresses with calculated offsets
```

### Profiling and Optimization Data

The profiling subsystem collects runtime execution data to drive compilation decisions and optimization strategies. This data provides the intelligence that transforms our JIT compiler from a simple translator into an adaptive optimization system.

#### Mental Model: Performance Thermometer

Think of the profiling system as a **performance thermometer** that measures the "temperature" of different code paths. Just as a medical thermometer detects fever by measuring body temperature, our profiler detects "hot" code paths by measuring execution frequency. When a function's temperature (invocation count) rises above the fever threshold, we prescribe medicine (JIT compilation) to improve performance.

#### JITCompiler Structure

The `JITCompiler` serves as the central coordination point between profiling, compilation, and execution, maintaining the state necessary for tiered compilation decisions.

| Field | Type | Description |
|-------|------|-------------|
| profiler | Profiler* | Pointer to the execution frequency profiler |
| cache | CodeCache* | Pointer to compiled function storage |
| compilation_enabled | int | Boolean flag to enable/disable JIT compilation |
| default_threshold | uint32_t | Default invocation threshold for new functions |

The JIT compiler structure provides centralized control over compilation policy while maintaining separation between profiling data collection and compilation triggering logic.

#### Compilation Decision Matrix

The decision to compile a function depends on multiple factors beyond simple invocation count, enabling more sophisticated compilation policies as the system matures.

| Function Characteristics | Compilation Decision | Rationale |
|--------------------------|---------------------|-----------|
| High invocation count + Small bytecode size | Compile immediately | High ROI, low compilation cost |
| High invocation count + Large bytecode size | Compile with delay | High ROI but higher compilation cost |
| Low invocation count + Any size | Continue interpreting | Compilation cost exceeds benefit |
| Error-prone functions | Never compile | Fallback to interpreter for debugging |

This decision matrix enables the JIT compiler to make intelligent compilation choices based on the characteristics of individual functions rather than applying uniform policies.

#### Optimization Tracking Data

As the JIT compiler matures, additional profiling data enables more sophisticated optimizations based on observed runtime behavior patterns.

| Optimization Type | Data Collected | Compilation Impact |
|------------------|-----------------|-------------------|
| Constant Folding | Frequently used constant values | Replace variable loads with immediate values |
| Dead Code Elimination | Unreachable code paths | Skip code generation for unused branches |
| Type Specialization | Common argument types | Generate specialized versions for frequent types |
| Inlining Candidates | Call site frequency | Inline frequently called small functions |

The optimization tracking data transforms compilation from simple translation into intelligent specialization based on actual runtime usage patterns.

> The power of JIT compilation lies not just in translating bytecode to native code, but in using runtime information to generate code that is specialized for the actual execution patterns of the program.

#### State Transition Management

Functions progress through multiple states as they move from cold interpretation through hot path detection to native compilation. The state management system tracks these transitions and ensures proper coordination between different execution modes.

| Current State | Trigger Event | Next State | Actions Taken |
|---------------|---------------|------------|---------------|
| Interpreting | First invocation | Profiling | Initialize profile entry |
| Profiling | Invocation count < threshold | Profiling | Increment counter, continue interpreting |
| Profiling | Invocation count >= threshold | Compiling | Trigger background compilation |
| Compiling | Compilation success | Compiled | Update function table, enable native calls |
| Compiling | Compilation failure | Interpreting | Log error, continue interpreting |
| Compiled | Any invocation | Compiled | Call compiled code directly |

This state machine ensures clean transitions between execution modes while handling compilation failures gracefully by falling back to interpretation.

### Implementation Guidance

The data model implementation requires careful attention to memory management, structure alignment, and cross-platform compatibility. The following guidance provides practical implementation strategies for each core structure.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Memory Allocation | malloc/free with mmap for executable memory | Custom memory pools with region-based allocation |
| Profiling Storage | Fixed-size arrays with linear search | Hash tables with dynamic resizing |
| Jump Patching | Simple array of patch records | Red-black tree for efficient target lookup |
| Error Handling | Return codes with errno | Exception-like error propagation |

#### Recommended File Structure

```
jit-compiler/
  src/
    data_model/
      bytecode_function.h       ← BytecodeFunction definition
      bytecode_function.c       ← BytecodeFunction operations  
      code_buffer.h            ← CodeBuffer and executable memory
      code_buffer.c            ← Memory allocation implementation
      profiler.h               ← Profiling structures  
      profiler.c               ← Profile tracking implementation
      jit_compiler.h           ← Main JITCompiler structure
      jit_compiler.c           ← Compilation coordination logic
    machine_code/
      x86_registers.h          ← X86Register enum and utilities
      instruction_emitter.h    ← Machine code generation functions
    tests/
      test_data_model.c        ← Unit tests for all structures
```

#### Core Data Structure Implementation

**BytecodeFunction Creation and Management:**

```c
// bytecode_function.h
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint32_t function_id;
    uint8_t* bytecode;
    size_t bytecode_length;
    int argument_count;
    int local_variable_count;
} BytecodeFunction;

// Create a new bytecode function from raw instruction data
BytecodeFunction* bytecode_function_create(uint32_t function_id, 
                                         const uint8_t* bytecode, 
                                         size_t length,
                                         int arg_count, 
                                         int local_count) {
    // TODO 1: Allocate BytecodeFunction structure
    // TODO 2: Allocate separate memory for bytecode copy (don't share pointers)
    // TODO 3: Copy bytecode data to prevent external modification
    // TODO 4: Initialize all fields including validation of counts >= 0
    // TODO 5: Return pointer to initialized structure
    // Hint: Use memcpy for bytecode copying, validate length > 0
}

// Validate bytecode instruction sequence for compilation safety
int bytecode_function_validate(const BytecodeFunction* func) {
    // TODO 1: Check for null pointer and basic field validation
    // TODO 2: Scan bytecode for valid instruction opcodes
    // TODO 3: Verify jump targets are within bytecode bounds
    // TODO 4: Check that function doesn't end mid-instruction
    // TODO 5: Return 1 for valid, 0 for invalid
    // Hint: Create lookup table of valid opcodes for quick validation
}
```

**CodeBuffer with W^X Policy Support:**

```c
// code_buffer.h
#include <sys/mman.h>
#include <stddef.h>

typedef struct {
    void* memory;
    size_t size;
    size_t used;
    int writable;
} CodeBuffer;

// Allocate executable memory buffer with initial write permissions
CodeBuffer* code_buffer_create(size_t size) {
    // TODO 1: Allocate CodeBuffer structure
    // TODO 2: Use mmap with PROT_READ|PROT_WRITE|PROT_EXEC for memory
    // TODO 3: Handle mmap failure (returns MAP_FAILED)
    // TODO 4: Initialize size, used=0, writable=1
    // TODO 5: Return initialized buffer or NULL on failure
    // Hint: Round size up to page boundary for mmap efficiency
}

// Switch buffer from writable to executable (W^X policy)
int code_buffer_make_executable(CodeBuffer* buffer) {
    // TODO 1: Validate buffer is currently writable
    // TODO 2: Use mprotect to change permissions to PROT_READ|PROT_EXEC
    // TODO 3: Handle mprotect failure and return error code
    // TODO 4: Update writable flag to 0
    // TODO 5: Return 0 for success, -1 for failure
    // Hint: Some platforms require this transition for security
}

// Emit single byte to code buffer with bounds checking
int code_buffer_emit_byte(CodeBuffer* buffer, uint8_t byte) {
    // TODO 1: Verify buffer is in writable mode
    // TODO 2: Check that used < size (bounds checking)
    // TODO 3: Store byte at memory + used offset
    // TODO 4: Increment used counter
    // TODO 5: Return bytes written (1) or error code (-1)
    // Hint: Cast memory to uint8_t* for byte-level access
}
```

**Profiler with Threshold-Based Compilation:**

```c
// profiler.h
typedef struct {
    uint32_t function_id;
    uint32_t invocation_count;
    uint32_t compilation_threshold;
    int is_compiled;
} ProfileEntry;

typedef struct {
    ProfileEntry* entries;
    size_t capacity;
    uint32_t default_threshold;
} Profiler;

// Create profiler with specified capacity and default threshold
Profiler* profiler_create(size_t max_functions, uint32_t threshold) {
    // TODO 1: Allocate Profiler structure
    // TODO 2: Allocate array of ProfileEntry structures
    // TODO 3: Initialize all entries with function_id=0 (invalid)
    // TODO 4: Set capacity and default_threshold fields
    // TODO 5: Return initialized profiler
    // Hint: Use calloc to zero-initialize all profile entries
}

// Record function invocation and check compilation threshold
int profiler_record_invocation(Profiler* profiler, uint32_t function_id) {
    // TODO 1: Find existing profile entry or create new one
    // TODO 2: Increment invocation_count atomically (if multithreaded)
    // TODO 3: Check if invocation_count >= compilation_threshold
    // TODO 4: Return 1 if compilation should be triggered, 0 otherwise
    // TODO 5: Handle profiler capacity limits gracefully
    // Hint: Linear search is acceptable for educational implementation
}
```

**Machine Code Generation Infrastructure:**

```c
// x86_registers.h
typedef enum {
    REG_RAX = 0,
    REG_RCX = 1, 
    REG_RDX = 2,
    REG_RBX = 3,
    REG_RSP = 4,
    REG_RBP = 5,
    REG_RSI = 6,
    REG_RDI = 7,
} X86Register;

// Constants for x86-64 instruction encoding
#define REX_PREFIX 0x48
#define MOV_REG_IMM_OPCODE 0xB8
#define ADD_REG_REG_OPCODE 0x01
#define RET_OPCODE 0xC3

// Generate move immediate instruction: mov reg, imm64
void emit_mov_reg_imm(CodeBuffer* buffer, X86Register reg, int64_t immediate) {
    // TODO 1: Emit REX prefix (0x48) for 64-bit operation
    // TODO 2: Emit MOV_REG_IMM_OPCODE + register number
    // TODO 3: Emit 8-byte immediate value in little-endian format
    // TODO 4: Update buffer used counter
    // Hint: Use bit manipulation to extract bytes from immediate value
}

// Generate register addition: add dest, src  
void emit_add_reg_reg(CodeBuffer* buffer, X86Register dest, X86Register src) {
    // TODO 1: Emit REX prefix for 64-bit operation
    // TODO 2: Emit ADD_REG_REG_OPCODE
    // TODO 3: Emit ModR/M byte encoding dest and src registers
    // TODO 4: Update buffer used counter  
    // Hint: ModR/M byte = 0xC0 | (dest << 3) | src for register-to-register
}
```

#### Language-Specific Hints

**Memory Management:**
- Use `mmap()` with `MAP_ANONYMOUS | MAP_PRIVATE` flags for executable memory allocation
- Always check for `MAP_FAILED` return value from mmap calls
- Use `mprotect()` to transition between writable and executable memory states
- Call `munmap()` in cleanup functions to prevent memory leaks

**Atomic Operations:**
- Use `__sync_fetch_and_add()` for thread-safe invocation counter increments
- Consider using `volatile` keyword for fields accessed from multiple threads
- Profile entry lookups may need reader-writer locks in multithreaded environments

**Platform Compatibility:**
- Some platforms (like iOS) prohibit writable+executable memory entirely
- Use `#ifdef` guards for platform-specific memory allocation strategies  
- Test on multiple platforms to ensure portable behavior

#### Milestone Checkpoints

**After implementing core data structures:**

Run the following validation test:
```bash
gcc -o test_data_model test_data_model.c data_model/*.c
./test_data_model
```

Expected output:
```
BytecodeFunction creation: PASS
CodeBuffer allocation: PASS  
Profiler threshold detection: PASS
x86-64 instruction emission: PASS
All data model tests passed
```

**After implementing machine code emission:**

Create a simple test that generates native code:
```c
// Generate function: int add_two_numbers(int a, int b) { return a + b; }
CodeBuffer* buffer = code_buffer_create(4096);
emit_mov_reg_imm(buffer, REG_RAX, 5);    // mov rax, 5
emit_mov_reg_imm(buffer, REG_RBX, 3);    // mov rbx, 3  
emit_add_reg_reg(buffer, REG_RAX, REG_RBX); // add rax, rbx
emit_ret(buffer);                        // ret
code_buffer_make_executable(buffer);

int64_t (*compiled_func)() = (int64_t (*)()) buffer->memory;
int64_t result = compiled_func(); // Should return 8
```

If this crashes or returns wrong results, check:
- REX prefix emission (must be 0x48 for 64-bit operations)
- Little-endian byte order for immediate values  
- Proper ModR/M byte calculation for register operands


## Machine Code Emitter

> **Milestone(s):** Milestone 1 (Basic x86-64 Emitter), Milestone 2 (Expression JIT), Milestone 3 (Function JIT and Calling Convention), Milestone 4 (Hot Path Detection and Optimization) — the machine code emitter serves as the foundational layer that converts high-level bytecode operations into executable native instructions throughout all compilation phases

The **Machine Code Emitter** represents the lowest level of our JIT compiler architecture, functioning as the critical bridge between abstract bytecode operations and concrete x86-64 machine instructions. This component must handle the intricate details of instruction encoding, register management, and executable memory allocation while providing a clean abstraction for higher-level compilation phases. The emitter's design fundamentally shapes the performance characteristics and correctness guarantees of the entire JIT system, making it essential to establish robust foundations that can support increasingly sophisticated optimization passes.

![Executable Memory Layout](./diagrams/memory-layout.svg)

The machine code emitter operates at the intersection of several complex domains: x86-64 instruction set architecture, operating system memory management, and runtime code generation. Each domain introduces unique constraints and opportunities that must be carefully balanced. The x86-64 instruction set provides powerful capabilities but demands precise encoding of register operands, immediate values, and addressing modes. Operating system memory management enforces security policies like **W^X (Write XOR Execute)** that require careful coordination between code generation and execution phases. Runtime code generation introduces dynamic requirements where instruction sequences must be assembled incrementally as bytecode analysis proceeds.

### Mental Model: Digital Assembly Line

Understanding machine code generation requires thinking beyond traditional compilation models toward a **digital assembly line** metaphor. In a physical manufacturing assembly line, raw materials enter at one end, pass through specialized workstations that perform specific transformations, and emerge as finished products ready for use. Similarly, our machine code emitter receives abstract operation requests (move this value, add these registers, call this function), processes them through specialized encoding stages, and produces binary instruction sequences ready for CPU execution.

Each workstation in our digital assembly line has a specific responsibility and set of tools. The **Memory Allocation Station** secures executable memory regions and manages their lifecycle, ensuring proper permissions and alignment. The **Instruction Encoding Station** transforms operation mnemonics and operands into precise binary representations, handling the complex rules of x86-64 instruction formats. The **Register Management Station** tracks register usage and resolves operand references, ensuring that abstract register names map to concrete hardware registers. The **Quality Control Station** validates instruction sequences for correctness and applies final patches like jump target resolution.

The assembly line operates under strict **just-in-time manufacturing** principles where each instruction is produced exactly when needed, in the correct sequence, with no waste or rework. Unlike traditional ahead-of-time compilation where all code can be analyzed before generation, our JIT assembly line must make encoding decisions with limited forward visibility. This constraint demands careful design of each workstation to minimize dependencies and enable efficient single-pass generation while maintaining correctness guarantees.

> **Key Insight**: The assembly line metaphor emphasizes that machine code generation is not a monolithic transformation but a coordinated sequence of specialized operations, each with distinct inputs, outputs, and quality requirements. This perspective helps identify clean interfaces between emitter components and guides error handling strategies.

### Executable Memory Management

The foundation of any JIT compiler rests on its ability to allocate, manage, and execute dynamically generated machine code. **Executable memory management** involves acquiring memory pages with execute permissions, coordinating between write and execute phases to comply with security policies, and maintaining efficient allocation strategies that minimize fragmentation and overhead.

Modern operating systems implement **W^X (Write XOR Execute) security policies** that prevent memory pages from being simultaneously writable and executable. This policy mitigates code injection attacks by ensuring that attackers cannot both write malicious code and execute it from the same memory region. For JIT compilers, W^X introduces a coordination challenge: code must be written to memory during generation but then executed during runtime, requiring explicit permission transitions.

The `CodeBuffer` structure serves as our primary abstraction for managing executable memory regions, encapsulating both the raw memory allocation and the metadata necessary for proper lifecycle management:

| Field Name | Type | Description |
|------------|------|-------------|
| `memory` | `void*` | Pointer to the allocated memory region, initially writable for code generation |
| `size` | `size_t` | Total allocated size of the memory region in bytes |
| `used` | `size_t` | Number of bytes currently occupied by generated instructions |
| `writable` | `int` | Boolean flag indicating current permission state (1=writable, 0=executable) |

The `allocate_executable_memory` function provides platform-specific memory allocation with appropriate size and alignment requirements. On Unix-like systems, this function utilizes `mmap` with specific flags to request memory pages that can eventually hold executable code:

> **Decision: mmap-based Allocation Strategy**
> - **Context**: JIT compilers require memory regions that can transition between writable (during code generation) and executable (during code execution) states
> - **Options Considered**: 
>   1. Standard malloc with mprotect for permission changes
>   2. mmap with MAP_ANONYMOUS for direct OS page allocation
>   3. Pre-allocated executable pools with internal management
> - **Decision**: mmap with MAP_ANONYMOUS and explicit mprotect transitions
> - **Rationale**: mmap provides direct control over memory permissions and alignment, enables efficient large allocations without heap fragmentation, and offers portable abstractions across Unix systems
> - **Consequences**: Requires platform-specific implementation but provides optimal performance and security compliance, with straightforward debugging of memory permission issues

| Allocation Strategy | Pros | Cons |
|-------------------|------|------|
| malloc + mprotect | Simple integration with existing allocators | Potential heap fragmentation, limited permission control |
| mmap + mprotect | Direct OS control, optimal alignment, no heap impact | Platform-specific, requires explicit lifecycle management |
| Pre-allocated pools | Predictable performance, reduced syscall overhead | Complex internal management, potential memory waste |

The memory lifecycle follows a strict state machine that ensures security compliance while enabling efficient code generation:

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| Unallocated | `code_buffer_create` called | Writable | Allocate memory with PROT_READ\|PROT_WRITE, initialize metadata |
| Writable | Emit instruction | Writable | Append instruction bytes to buffer, increment `used` counter |
| Writable | `code_buffer_make_executable` called | Executable | Call mprotect with PROT_READ\|PROT_EXEC, set `writable` flag to 0 |
| Executable | Function call | Executable | CPU executes instructions, no permission changes needed |
| Executable | Buffer destruction | Unallocated | Unmap memory region, free metadata structures |

⚠️ **Pitfall: Permission Transition Timing**
Many JIT implementations attempt to emit additional instructions after transitioning memory to executable mode, resulting in segmentation faults. The `writable` flag in `CodeBuffer` prevents this error by tracking permission state. Always verify that `buffer->writable == 1` before calling any emit functions, and ensure that `code_buffer_make_executable` is called exactly once after all instructions have been generated.

The `make_memory_executable` function encapsulates the critical permission transition that enables generated code execution. This operation is irreversible within our current design, emphasizing the importance of completing all code generation before making the transition:

```c
// Platform-specific implementation requirements:
// - Unix: mprotect(buffer->memory, buffer->size, PROT_READ | PROT_EXEC)  
// - Windows: VirtualProtect with PAGE_EXECUTE_READ
// - Verification: attempt to write single byte should fail with access violation
```

> **Security Consideration**: The transition from writable to executable represents a critical security boundary. Production implementations should consider additional hardening measures such as code signing verification, control flow integrity checks, or integration with system-level exploit mitigation frameworks.

Memory alignment requirements add another layer of complexity to executable memory management. x86-64 instructions have varying alignment preferences, and function entry points typically require specific alignment for optimal performance. Our allocation strategy must account for these requirements while maintaining efficient memory utilization:

| Alignment Target | Requirement | Rationale |
|------------------|-------------|-----------|
| Instruction boundaries | 1-byte aligned | x86-64 supports unaligned instructions but prefers alignment |
| Function entry points | 16-byte aligned | Optimal cache line utilization and call/return performance |
| Memory pages | 4KB aligned | Required by OS memory management for mprotect operations |
| Large allocations | 2MB aligned | Enables huge page optimizations on supported systems |

### x86-64 Instruction Encoding

The heart of machine code generation lies in **x86-64 instruction encoding**, the process of converting abstract operation requests into precise binary representations that the CPU can execute directly. This encoding process must handle the complex variable-length instruction format of x86-64, manage register operand encoding, and ensure correct generation of immediate values and addressing modes.

![Register Allocation Strategy](./diagrams/register-allocation.svg)

x86-64 instructions follow a sophisticated variable-length encoding scheme that balances backward compatibility with modern 64-bit extensions. Understanding this encoding requires familiarity with several key components that combine to form complete instructions: prefixes, opcodes, ModR/M bytes, SIB bytes, and displacement/immediate fields.

The **REX prefix** represents the most crucial addition for 64-bit operations, enabling access to extended registers and 64-bit operand sizes. The REX prefix uses the byte value `0x48` for most common 64-bit operations, though specific bit patterns within the REX byte control various encoding aspects:

| REX Bit Pattern | Field Name | Purpose |
|-----------------|------------|---------|
| `0100WRXB` | REX structure | Base pattern for 64-bit operations |
| `W` (bit 3) | Operand width | 1 = 64-bit operands, 0 = 32-bit operands |
| `R` (bit 2) | Register extension | Extends ModR/M reg field for registers R8-R15 |
| `X` (bit 1) | Index extension | Extends SIB index field for registers R8-R15 |
| `B` (bit 0) | Base extension | Extends ModR/M r/m or SIB base field for registers R8-R15 |

The **ModR/M byte** provides the primary mechanism for specifying instruction operands, encoding both register-to-register and register-to-memory operations within a single byte format. Understanding ModR/M encoding is essential for generating correct operand references:

| ModR/M Field | Bits | Purpose |
|--------------|------|---------|
| `mod` | 7-6 | Addressing mode: 11=register-direct, 10/01/00=memory with displacement |
| `reg` | 5-3 | Register operand (source or destination depending on instruction) |
| `r/m` | 2-0 | Register/memory operand (combined with mod field for interpretation) |

Our instruction emitter provides specialized functions for common instruction patterns, abstracting the complex encoding details while maintaining precise control over generated code. The `emit_mov_reg_imm` function demonstrates typical encoding requirements for move immediate operations:

The move immediate instruction follows the pattern: REX prefix (if needed) + base opcode + register encoding + immediate value. For 64-bit operations, this becomes: `0x48` (REX.W) + `0xB8` (base MOV opcode) + register encoding + 8-byte immediate value.

| Instruction Component | Encoding Details |
|----------------------|------------------|
| REX prefix | `0x48` for 64-bit immediate moves to registers RAX-R15 |
| Base opcode | `0xB8` for MOV register, immediate (register encoded in opcode) |
| Register encoding | Add register number to base opcode (RAX=0, RCX=1, RDX=2, etc.) |
| Immediate value | 8 bytes in little-endian format for 64-bit values |

Register-to-register arithmetic operations require more complex encoding using ModR/M bytes to specify both source and destination operands. The `emit_add_reg_reg` function illustrates this pattern:

| Encoding Step | Details |
|---------------|---------|
| REX prefix | `0x48` to enable 64-bit operation on both operands |
| Opcode | `0x01` for ADD register to register/memory |
| ModR/M byte | `0xC0` base + (src_reg << 3) + dst_reg for register-direct addressing |

The `X86Register` enumeration provides a clean abstraction over x86-64 register names while maintaining direct correspondence to hardware encoding values:

| Register Name | Enum Value | Hardware Encoding | Usage Notes |
|---------------|------------|------------------|-------------|
| `RAX` | 0 | 000 | Primary accumulator, function return values |
| `RCX` | 1 | 001 | Counter register, fourth function argument |
| `RDX` | 2 | 010 | Data register, third function argument |
| `RBX` | 3 | 011 | Base register, callee-saved |
| `RSP` | 4 | 100 | Stack pointer, requires special handling |
| `RBP` | 5 | 101 | Base pointer, frame pointer for functions |
| `RSI` | 6 | 110 | Source index, second function argument |
| `RDI` | 7 | 111 | Destination index, first function argument |

Extended registers R8-R15 require REX prefix extensions but follow similar encoding patterns with appropriate bit field modifications. The instruction emitter handles these extensions automatically based on register operand values.

> **Decision: Direct Byte Emission Strategy**
> - **Context**: JIT compilers can generate machine code through various approaches including intermediate representations, external assemblers, or direct byte emission
> - **Options Considered**:
>   1. External assembler integration (gas, nasm) with temporary files
>   2. Intermediate representation with separate encoding phase  
>   3. Direct byte emission with instruction-specific encoding functions
> - **Decision**: Direct byte emission with specialized encoding functions for each instruction pattern
> - **Rationale**: Eliminates external dependencies, provides precise control over generated code, enables efficient single-pass generation, and simplifies debugging through direct correspondence between high-level operations and binary output
> - **Consequences**: Requires detailed understanding of x86-64 encoding but delivers optimal performance and complete control over instruction selection and optimization

Immediate value encoding varies significantly based on instruction type and operand size requirements. Our emitter supports several immediate encoding patterns:

| Immediate Type | Size | Encoding | Usage |
|----------------|------|----------|-------|
| 8-bit signed | 1 byte | Sign-extended to operand size | Small constants, displacement values |
| 32-bit signed | 4 bytes | Sign-extended to 64-bit in 64-bit mode | Most integer constants |
| 64-bit absolute | 8 bytes | Full 64-bit value | Large constants, absolute addresses |

> **Encoding Insight**: x86-64 cannot encode 64-bit immediate values in most arithmetic instructions. Only MOV register, immediate supports full 64-bit immediates. Other instructions are limited to 32-bit signed immediates that are sign-extended to 64 bits. This limitation influences instruction selection during JIT compilation.

### Common Pitfalls

Machine code generation introduces numerous opportunities for subtle errors that can manifest as crashes, incorrect results, or security vulnerabilities. Understanding these pitfalls and their prevention strategies is crucial for building robust JIT compilers.

⚠️ **Pitfall: Incorrect REX Prefix Usage**
The most frequent error in x86-64 code generation involves missing or incorrect REX prefixes when working with 64-bit operands or extended registers. Without proper REX prefixes, instructions operate on 32-bit values (zeroing upper 32 bits) or fail to encode extended registers correctly. This manifests as mysterious data corruption where only the lower 32 bits of 64-bit values are processed correctly.

**Detection**: Values larger than 32 bits get truncated unexpectedly, or attempts to use registers R8-R15 result in different registers being used instead.

**Prevention**: Always emit `REX_PREFIX` (0x48) before instructions that operate on 64-bit values or use extended registers. Verify that register encoding functions properly handle the REX extension bits for registers above R7.

**Example Fix**: For a 64-bit ADD operation between RAX and RCX, the correct encoding is `0x48 0x01 0xC8`, not `0x01 0xC8`. The missing REX prefix would result in a 32-bit operation that zeros the upper 32 bits of RAX.

⚠️ **Pitfall: ModR/M Byte Calculation Errors**
ModR/M byte encoding requires precise bit manipulation to specify operand addressing modes and register selections. Common errors include swapping source and destination register positions, using incorrect mod field values for register-direct addressing, or failing to account for special cases like RSP register handling.

**Detection**: Instructions appear to use wrong registers, memory access violations when expecting register operations, or reversed operand order in non-commutative operations.

**Prevention**: Use consistent helper functions for ModR/M calculation that clearly separate mod, reg, and r/m field computations. Verify calculations against x86-64 reference documentation for each instruction pattern.

**Example Fix**: For ADD RAX, RCX (add RCX to RAX), the correct ModR/M byte is `0xC1` (mod=11 for register-direct, reg=001 for RCX source, r/m=000 for RAX destination), not `0xC8` which would encode ADD RCX, RAX.

⚠️ **Pitfall: Memory Permission Timing**
W^X security policies require careful coordination between code generation and execution phases. Attempting to write instructions after transitioning memory to executable mode causes segmentation faults, while attempting to execute code before completing the transition results in permission errors.

**Detection**: Segmentation faults when emitting instructions, or execution failures with permission errors when calling generated code.

**Prevention**: Maintain clear separation between generation and execution phases. Use the `writable` flag in `CodeBuffer` to track permission state and assert appropriate conditions in emit functions.

**Example Fix**: Always complete all instruction emission before calling `code_buffer_make_executable`, and never attempt to emit additional instructions after the transition. If dynamic patching is needed, allocate separate writable memory regions for modifications.

⚠️ **Pitfall: Instruction Length Miscalculation**
Variable-length x86-64 instructions make buffer size estimation challenging. Underestimating buffer requirements leads to buffer overruns, while overestimating wastes memory and impacts performance. Additionally, jump target calculations depend on accurate instruction length prediction.

**Detection**: Buffer overruns during code generation, incorrect jump targets leading to execution of invalid instructions, or excessive memory usage.

**Prevention**: Implement conservative buffer sizing with bounds checking on all emit operations. For jump targets, use two-pass generation or conservative displacement estimation with padding.

**Example Fix**: When emitting a conditional jump, initially emit a 32-bit displacement (6 bytes total) even if an 8-bit displacement might suffice. This ensures consistent instruction lengths for jump target calculations.

⚠️ **Pitfall: Stack Alignment Violations**
The System V AMD64 ABI requires 16-byte stack alignment before function calls. Generated code that violates this requirement may experience crashes in system libraries or math functions that assume proper alignment.

**Detection**: Crashes in library functions with error messages about stack alignment, or segmentation faults when calling generated functions from C code.

**Prevention**: Track stack pointer modifications during code generation and ensure alignment requirements are met before call instructions. Emit stack adjustment instructions as needed.

**Example Fix**: Before emitting a call instruction, check that (RSP % 16 == 8) to account for the return address push. If not, emit `SUB RSP, 8` to achieve proper alignment, and remember to restore with `ADD RSP, 8` after the call.

⚠️ **Pitfall: Register Clobbering Without Preservation**
Generated code must respect calling conventions regarding callee-saved registers. Modifying registers like RBX, R12-R15, or RBP without proper preservation causes unpredictable behavior when returning to calling code.

**Detection**: Intermittent corruption of variables in calling functions, crashes when returning from generated code, or debugger showing unexpected register values.

**Prevention**: Implement systematic register preservation in function prologues and epilogues. Use caller-saved registers (RAX, RCX, RDX, RSI, RDI, R8-R11) for temporary values when possible.

**Example Fix**: If generated code uses RBX for temporary storage, emit `PUSH RBX` in the function prologue and `POP RBX` in the epilogue. Alternatively, redesign register allocation to prefer caller-saved registers for temporary values.

### Implementation Guidance

The machine code emitter serves as the foundation for all higher-level JIT compilation phases, requiring robust implementation of executable memory management and instruction encoding capabilities. This implementation guidance provides complete starter code for memory management infrastructure and skeletal frameworks for core instruction emission functionality.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Memory Management | `mmap` with `mprotect` for Unix systems | Cross-platform abstraction with Windows `VirtualAlloc` support |
| Instruction Encoding | Direct byte array manipulation | Structured instruction builder with validation |
| Debug Support | Printf debugging with hex dumps | Integration with disassembly libraries (capstone, objdump) |
| Testing Framework | Manual test functions with assertions | Property-based testing with random instruction generation |

#### Recommended File Structure

```
jit-compiler/
  src/
    emitter/
      code_buffer.c          ← executable memory management
      code_buffer.h          ← memory allocation interfaces
      x86_encoder.c          ← instruction encoding implementation
      x86_encoder.h          ← instruction emission interfaces
      registers.h            ← register enumeration and utilities
    tests/
      test_emitter.c         ← comprehensive emitter tests
    examples/
      basic_emission.c       ← simple code generation examples
  Makefile                   ← build configuration with debug flags
```

#### Infrastructure Starter Code (Complete)

**File: `src/emitter/code_buffer.h`**
```c
#ifndef CODE_BUFFER_H
#define CODE_BUFFER_H

#include <stdint.h>
#include <stddef.h>

// Memory protection constants for cross-platform compatibility
#define PROT_READ  0x1
#define PROT_WRITE 0x2  
#define PROT_EXEC  0x4

// CodeBuffer manages executable memory regions for JIT compilation
typedef struct {
    void* memory;        // Pointer to allocated executable memory
    size_t size;         // Total size of allocated region
    size_t used;         // Bytes currently used for instructions
    int writable;        // 1 if memory is writable, 0 if executable
} CodeBuffer;

// Function pointer type for calling JIT-compiled code
typedef int64_t (*CompiledFunction)(int64_t arg1, int64_t arg2);

// Memory management interface
CodeBuffer* code_buffer_create(size_t size);
int code_buffer_make_executable(CodeBuffer* buffer);
void code_buffer_destroy(CodeBuffer* buffer);

// Platform-specific memory allocation functions
void* allocate_executable_memory(size_t size);
int make_memory_executable(void* memory, size_t size);

#endif // CODE_BUFFER_H
```

**File: `src/emitter/code_buffer.c`**
```c
#include "code_buffer.h"
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

CodeBuffer* code_buffer_create(size_t size) {
    CodeBuffer* buffer = malloc(sizeof(CodeBuffer));
    if (!buffer) return NULL;
    
    buffer->memory = allocate_executable_memory(size);
    if (!buffer->memory) {
        free(buffer);
        return NULL;
    }
    
    buffer->size = size;
    buffer->used = 0;
    buffer->writable = 1;  // Initially writable for code generation
    
    return buffer;
}

void* allocate_executable_memory(size_t size) {
    // Round up to page boundary for mmap
    size_t page_size = getpagesize();
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);
    
    void* memory = mmap(NULL, aligned_size, 
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, 
                       -1, 0);
    
    if (memory == MAP_FAILED) {
        return NULL;
    }
    
    return memory;
}

int code_buffer_make_executable(CodeBuffer* buffer) {
    assert(buffer != NULL);
    assert(buffer->writable == 1);  // Must be writable before transition
    
    if (make_memory_executable(buffer->memory, buffer->size) != 0) {
        return -1;
    }
    
    buffer->writable = 0;  // Mark as executable
    return 0;
}

int make_memory_executable(void* memory, size_t size) {
    return mprotect(memory, size, PROT_READ | PROT_EXEC);
}

void code_buffer_destroy(CodeBuffer* buffer) {
    if (buffer) {
        if (buffer->memory) {
            munmap(buffer->memory, buffer->size);
        }
        free(buffer);
    }
}
```

**File: `src/emitter/registers.h`**
```c
#ifndef REGISTERS_H
#define REGISTERS_H

// x86-64 register enumeration matching hardware encoding values
typedef enum {
    RAX = 0, RCX = 1, RDX = 2, RBX = 3,
    RSP = 4, RBP = 5, RSI = 6, RDI = 7,
    R8 = 8,  R9 = 9,  R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15
} X86Register;

// Helper functions for register encoding
static inline int register_needs_rex(X86Register reg) {
    return reg >= R8;  // Extended registers require REX prefix
}

static inline uint8_t register_encoding(X86Register reg) {
    return (uint8_t)(reg & 0x7);  // Lower 3 bits for ModR/M encoding
}

static inline const char* register_name(X86Register reg) {
    static const char* names[] = {
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
    };
    return (reg <= R15) ? names[reg] : "invalid";
}

#endif // REGISTERS_H
```

#### Core Logic Skeleton Code

**File: `src/emitter/x86_encoder.h`**
```c
#ifndef X86_ENCODER_H
#define X86_ENCODER_H

#include "code_buffer.h"
#include "registers.h"
#include <stdint.h>

// x86-64 instruction encoding constants
#define REX_PREFIX 0x48                // REX.W prefix for 64-bit operations
#define MOV_REG_IMM_OPCODE 0xB8        // Base opcode for mov reg, imm
#define ADD_REG_REG_OPCODE 0x01        // Opcode for add reg, reg
#define SUB_REG_REG_OPCODE 0x29        // Opcode for sub reg, reg  
#define CMP_REG_REG_OPCODE 0x39        // Opcode for cmp reg, reg
#define RET_OPCODE 0xC3                // Return instruction

// Instruction emission interface
void emit_mov_reg_imm(CodeBuffer* buffer, X86Register reg, int64_t immediate);
void emit_add_reg_reg(CodeBuffer* buffer, X86Register dst, X86Register src);
void emit_sub_reg_reg(CodeBuffer* buffer, X86Register dst, X86Register src);
void emit_cmp_reg_reg(CodeBuffer* buffer, X86Register left, X86Register right);
void emit_ret(CodeBuffer* buffer);

// Helper functions for instruction encoding
void emit_byte(CodeBuffer* buffer, uint8_t byte);
void emit_rex_prefix(CodeBuffer* buffer, X86Register reg1, X86Register reg2);
uint8_t encode_modrm(uint8_t mod, uint8_t reg, uint8_t rm);

#endif // X86_ENCODER_H
```

**File: `src/emitter/x86_encoder.c`**
```c
#include "x86_encoder.h"
#include <assert.h>
#include <string.h>

void emit_byte(CodeBuffer* buffer, uint8_t byte) {
    assert(buffer != NULL);
    assert(buffer->writable == 1);  // Must be writable
    assert(buffer->used < buffer->size);  // Must have space
    
    ((uint8_t*)buffer->memory)[buffer->used] = byte;
    buffer->used++;
}

void emit_mov_reg_imm(CodeBuffer* buffer, X86Register reg, int64_t immediate) {
    // TODO 1: Check if buffer is writable and has sufficient space (10 bytes max)
    // TODO 2: Emit REX prefix (0x48) for 64-bit operation
    // TODO 3: Emit base opcode (0xB8) + register encoding for MOV reg, imm64
    // TODO 4: Emit 8-byte immediate value in little-endian format
    // Hint: Use emit_byte() for each byte, handle endianness manually
}

void emit_add_reg_reg(CodeBuffer* buffer, X86Register dst, X86Register src) {
    // TODO 1: Verify buffer is writable and has space (3 bytes)
    // TODO 2: Emit REX prefix for 64-bit operation on both registers
    // TODO 3: Emit ADD opcode (0x01) for register-to-register addition  
    // TODO 4: Calculate and emit ModR/M byte: mod=11, reg=src, r/m=dst
    // TODO 5: Handle extended register encoding in REX prefix if needed
    // Hint: ModR/M = 0xC0 + (src << 3) + dst for register-direct addressing
}

void emit_sub_reg_reg(CodeBuffer* buffer, X86Register dst, X86Register src) {
    // TODO 1: Similar structure to emit_add_reg_reg but with SUB opcode (0x29)
    // TODO 2: Verify register operand order matches x86-64 semantics (dst = dst - src)
    // TODO 3: Handle REX prefix requirements for extended registers
}

void emit_cmp_reg_reg(CodeBuffer* buffer, X86Register left, X86Register right) {
    // TODO 1: Emit REX prefix and CMP opcode (0x39) 
    // TODO 2: Create ModR/M byte with right as reg field, left as r/m field
    // TODO 3: Remember CMP sets flags but doesn't store result
    // Hint: CMP left, right computes (left - right) and sets flags accordingly
}

void emit_ret(CodeBuffer* buffer) {
    // TODO 1: Verify buffer space (1 byte)
    // TODO 2: Emit RET opcode (0xC3) - no operands needed
    // Simple but critical for proper function exit
}

uint8_t encode_modrm(uint8_t mod, uint8_t reg, uint8_t rm) {
    // TODO 1: Validate that mod, reg, rm are all <= 3, <= 7, <= 7 respectively
    // TODO 2: Combine fields: (mod << 6) | (reg << 3) | rm
    // TODO 3: Return the combined byte value
    // This is a fundamental building block for most x86-64 instructions
}

void emit_rex_prefix(CodeBuffer* buffer, X86Register reg1, X86Register reg2) {
    // TODO 1: Check if either register requires REX prefix (>= R8)
    // TODO 2: Start with base REX value (0x48 for REX.W = 64-bit operands)
    // TODO 3: Set REX.R bit if reg1 >= R8, REX.B bit if reg2 >= R8  
    // TODO 4: Emit the constructed REX byte
    // TODO 5: Handle case where no REX prefix is needed (don't emit anything)
}
```

#### Language-Specific Hints

**Memory Management in C:**
- Use `mmap` with `MAP_PRIVATE | MAP_ANONYMOUS` for executable memory allocation
- Always check return values: `mmap` returns `MAP_FAILED` on error, not NULL
- Use `getpagesize()` to determine system page size for proper alignment
- Call `munmap` in cleanup code to avoid memory leaks

**Byte Manipulation:**
- Use `memcpy` for copying multi-byte values in correct endianness
- Cast immediate values to appropriate sizes: `(uint32_t)imm` for 32-bit values
- Use bit shifting for ModR/M construction: `(mod << 6) | (reg << 3) | rm`
- Verify buffer bounds before every emit operation to prevent overruns

**Testing and Debugging:**
- Print generated code as hex bytes for manual verification: `printf("%02x ", byte)`
- Use `objdump -D -b binary -m i386:x86-64 /tmp/code.bin` to disassemble generated code
- Create simple test functions that perform known operations with expected results
- Test edge cases: register R15 (highest), immediate value 0, negative immediates

#### Milestone Checkpoint

**After implementing the machine code emitter:**

1. **Compilation Test**: `gcc -o test_emitter src/emitter/*.c tests/test_emitter.c -DDEBUG`
2. **Basic Functionality**: The test program should successfully:
   - Allocate executable memory (4KB buffer)
   - Generate simple instruction sequence (MOV RAX, 42; RET)
   - Transition memory to executable mode
   - Call generated code and receive correct return value (42)

3. **Expected Output**:
   ```
   Allocated 4096 bytes of executable memory
   Generated 10 bytes of code: 48 B8 2A 00 00 00 00 00 00 00 C3
   Made memory executable
   Calling generated function...
   Result: 42 (expected: 42) ✓
   ```

4. **Signs of Success**:
   - No segmentation faults during code generation or execution
   - Generated code bytes match expected x86-64 encoding
   - Function calls return correct values for simple arithmetic

5. **Common Issues and Diagnostics**:
   - **Segfault during emit**: Check that buffer is still writable (`buffer->writable == 1`)
   - **Segfault during execution**: Verify `code_buffer_make_executable` was called successfully
   - **Wrong results**: Compare generated bytes against x86-64 reference or use objdump for disassembly
   - **Permission denied**: Check that system allows executable memory (some containers/VMs restrict this)

**Next Steps**: With a working machine code emitter, proceed to implement bytecode-to-native translation for arithmetic expressions, building on this foundational layer to create increasingly sophisticated JIT compilation capabilities.


## Bytecode to Native Translator

> **Milestone(s):** Milestone 2 (Expression JIT), Milestone 3 (Function JIT and Calling Convention) — this component performs the core translation from high-level bytecode operations to low-level x86-64 machine instructions

The **bytecode to native translator** represents the intellectual heart of our JIT compiler, transforming abstract virtual machine operations into concrete processor instructions. This component bridges two fundamentally different computational models: the stack-based, platform-independent world of bytecode interpretation and the register-based, hardware-specific domain of native machine execution. Understanding this translation process requires grasping not just the mechanical conversion of instructions, but the deeper challenge of maintaining semantic equivalence while exploiting the performance characteristics of modern processors.

### Mental Model: Code Transformation Pipeline

Think of the bytecode to native translator as a sophisticated language translation service, but instead of converting between human languages, it converts between computational languages. Just as a skilled human translator doesn't merely substitute words but adapts idioms, cultural context, and communication patterns to preserve meaning across languages, our bytecode translator must adapt computational patterns to preserve program behavior across execution models.

Consider how a human translator approaches a complex document. They first understand the overall structure and intent, then break it down into logical sections, and finally translate each section while maintaining coherence with the whole. Similarly, our translator examines the bytecode function structure, identifies logical blocks of operations, and converts each block to native instructions while ensuring the translated code maintains the same computational behavior as the original bytecode.

The translation process involves three fundamental transformations. **Semantic preservation** ensures that the translated code produces identical results to the original bytecode under all possible inputs. **Execution model adaptation** converts from the VM's stack-based operation model to the processor's register-based model. **Performance optimization** leverages native instruction capabilities to execute the same operations more efficiently than interpretation.

This mental model helps us understand why bytecode translation is more complex than simple instruction substitution. Each bytecode operation may require multiple native instructions to implement correctly, and the translator must coordinate these instruction sequences to maintain program correctness while maximizing execution efficiency.

### Operation Mapping Strategies

The foundation of bytecode to native translation lies in establishing systematic mappings between VM operations and their native instruction equivalents. Each bytecode operation represents a well-defined computational primitive, and our translator must generate native instruction sequences that implement identical behavior while respecting the target processor's execution model and calling conventions.

![Data Model and Relationships](./diagrams/data-model.svg)

The translation process begins with **bytecode analysis**, where we examine the instruction stream to understand data flow patterns, control dependencies, and register usage requirements. This analysis phase identifies optimization opportunities and determines the most efficient native instruction sequences for each bytecode pattern.

Consider the fundamental arithmetic operations that form the backbone of most computations. The VM's `OP_ADD` bytecode instruction pops two values from the evaluation stack, computes their sum, and pushes the result back onto the stack. The native translation must implement this same behavior using processor registers and memory operations, but it can eliminate the actual stack manipulation when the values are already in registers from previous operations.

| Bytecode Operation | Stack Effect | Native Instruction Sequence | Register Usage |
|-------------------|--------------|---------------------------|----------------|
| `OP_LOAD_CONST value` | push value | `mov rax, immediate` | rax = value |
| `OP_ADD` | pop a, pop b, push(a+b) | `add rdi, rsi` | rdi += rsi |
| `OP_SUB` | pop a, pop b, push(a-b) | `sub rdi, rsi` | rdi -= rsi |
| `OP_MUL` | pop a, pop b, push(a*b) | `imul rdi, rsi` | rdi *= rsi |
| `OP_DIV` | pop a, pop b, push(a/b) | `mov rax, rdi; cqo; idiv rsi` | rax = rdi/rsi |
| `OP_CMP` | pop a, pop b, push(cmp) | `cmp rdi, rsi` | sets flags |
| `OP_JMP_IF_FALSE offset` | pop condition, maybe jump | `test rax, rax; jz label` | conditional branch |

The complexity of operation mapping becomes apparent with operations like division, which requires careful setup of processor registers according to x86-64 conventions. The `OP_DIV` operation must place the dividend in the `rax` register, extend it to a 128-bit value using the `cqo` instruction, then perform signed division with `idiv`, retrieving the quotient from `rax` and handling potential divide-by-zero exceptions.

> **Decision: Direct Instruction Mapping vs. Runtime Helper Calls**
> - **Context**: Complex bytecode operations like division, string manipulation, or object allocation can be implemented either as direct native instruction sequences or as calls to runtime helper functions
> - **Options Considered**: Inline native instructions for all operations; runtime helper calls for complex operations; hybrid approach based on operation complexity
> - **Decision**: Use direct native instructions for simple arithmetic and comparisons, runtime helpers for complex operations like object allocation and string manipulation
> - **Rationale**: Direct instructions provide maximum performance for common operations while runtime helpers reduce code complexity and compilation time for rarely-used operations
> - **Consequences**: Enables aggressive optimization of hot arithmetic code while maintaining implementation simplicity for edge cases

**Comparison mapping** operations require special attention because they bridge computational operations with control flow. The VM's comparison bytecodes produce boolean values on the evaluation stack, but x86-64 comparisons set processor flags that influence subsequent conditional jump instructions. Our translator must coordinate comparison operations with their associated conditional branches to generate efficient native code sequences.

The translation of **control flow operations** presents unique challenges because bytecode addresses don't correspond directly to native code addresses. When the VM executes `OP_JMP_IF_FALSE 42`, it performs a conditional jump to bytecode offset 42. The native translation must compute the corresponding native code address and generate appropriate conditional jump instructions, handling the fact that the target native address might not be known during initial code generation.

**Memory access operations** require careful attention to data layout and alignment requirements. The VM's `OP_LOAD_LOCAL 3` operation loads the value of local variable 3 onto the evaluation stack. The native translation must compute the appropriate stack frame offset, generate memory load instructions with correct addressing modes, and ensure the loaded value is available in the expected register for subsequent operations.

### Simple Register Allocation

Register allocation represents one of the most critical aspects of generating efficient native code, transforming the VM's infinite evaluation stack abstraction into the processor's finite register resources. Our simple register allocation strategy focuses on common execution patterns while providing straightforward fallback mechanisms for complex cases that exceed available register resources.

![Register Allocation Strategy](./diagrams/register-allocation.svg)

The **stack-to-register mapping** process begins by analyzing the bytecode instruction sequence to understand stack depth requirements and value lifetimes. Most arithmetic expressions exhibit predictable stack behavior where values are pushed, consumed by operations, and results pushed back. Our allocator exploits this pattern by maintaining a **virtual stack** that maps stack positions to physical registers whenever possible.

The register allocation algorithm maintains a **register assignment table** that tracks which physical registers currently hold which virtual stack positions. When translating a bytecode operation that pushes a value, the allocator assigns the next available register. When translating an operation that pops values, the allocator identifies which registers contain the required operands and generates instructions that operate directly on those registers.

| Virtual Stack Position | Physical Register | Status | Lifetime |
|----------------------|------------------|---------|----------|
| Stack[0] (bottom) | rdi | occupied | long-term |
| Stack[1] | rsi | occupied | medium-term |
| Stack[2] | rdx | occupied | short-term |
| Stack[3] | rcx | available | - |
| Stack[4] | r8 | available | - |
| Stack[5] | r9 | available | - |

The **register selection priority** follows x86-64 calling convention guidelines to minimize register save/restore overhead when calling other functions. We prioritize caller-saved registers (`rax`, `rcx`, `rdx`, `rdi`, `rsi`, `r8`, `r9`, `r10`, `r11`) for short-term intermediate values and reserve callee-saved registers (`rbx`, `r12`, `r13`, `r14`, `r15`) for values with longer lifetimes that span function calls.

Consider the translation of a simple arithmetic expression: `a + b * c`. The bytecode sequence loads three values onto the stack, performs multiplication, then addition. Our register allocator assigns `a` to `rdi`, `b` to `rsi`, and `c` to `rdx`. The multiplication `b * c` becomes `imul rsi, rdx`, storing the result in `rsi`. The final addition `a + (b * c)` becomes `add rdi, rsi`, storing the final result in `rdi`. This register allocation eliminates all stack manipulation overhead present in the interpreted version.

**Spill handling** becomes necessary when the expression complexity exceeds available register resources. When all registers are occupied and a new value must be stored, the allocator selects a register for **spilling** based on value lifetime analysis. Values with the longest remaining lifetimes are spilled to memory locations in the stack frame, freeing their registers for immediate use. Spilled values are reloaded into registers when they're needed for subsequent operations.

The spill selection algorithm prioritizes spilling values that won't be needed soon and avoids spilling values that will be required by the next few instructions. This **lazy spill strategy** minimizes memory traffic by keeping frequently-used values in registers while ensuring that register pressure never exceeds available resources.

> **Decision: Expression-Level vs. Function-Level Register Allocation**
> - **Context**: Register allocation can operate at different granularities, from individual expression trees to entire function bodies
> - **Options Considered**: Expression-level allocation with simple spilling; function-level allocation with liveness analysis; hybrid approach with local and global phases
> - **Decision**: Expression-level allocation with predictive spilling for the initial implementation
> - **Rationale**: Expression-level allocation provides good performance for arithmetic-heavy code while remaining simple to implement and debug, suitable for learning JIT compilation principles
> - **Consequences**: Enables efficient translation of common bytecode patterns while avoiding the complexity of global register allocation algorithms

**Register state synchronization** ensures that register assignments remain consistent across bytecode instruction boundaries. After translating each bytecode instruction, the allocator updates its register assignment table to reflect the new stack configuration. This synchronization process handles cases where operations modify the evaluation stack depth or reorder stack elements.

The **calling convention integration** aspect of register allocation becomes crucial when generating function calls or managing function entry/exit sequences. Our allocator must ensure that function arguments are placed in the correct registers (`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`) according to the System V AMD64 ABI, and that caller-saved registers are preserved across function calls when their values are needed afterward.

### Control Flow Translation

Control flow translation represents the most complex aspect of bytecode to native conversion, requiring coordination between instruction generation, address resolution, and execution state management. The fundamental challenge lies in converting the VM's program counter based jumps to the processor's instruction pointer based branches while maintaining identical execution behavior under all possible conditions.

The **forward reference problem** emerges because bytecode jump targets are specified as offsets from the current instruction, but native code addresses aren't known until instruction emission completes. When translating a conditional jump instruction that targets bytecode 20 instructions ahead, the native code address of that target won't be determined until we've translated all intervening instructions and computed their native code sizes.

Our **two-pass translation strategy** addresses forward references systematically. The first pass performs bytecode analysis and generates native instruction sequences, but leaves jump target addresses unresolved. During this pass, we maintain a **jump patch table** that records the locations of incomplete jump instructions and their intended bytecode targets. The second pass resolves all forward references by computing final native addresses and patching the incomplete jump instructions with correct target addresses.

| Bytecode Offset | Native Code Offset | Patch Locations | Instruction Type |
|----------------|-------------------|-----------------|------------------|
| 0 | 0x0000 | - | entry point |
| 3 | 0x0008 | - | load constant |
| 6 | 0x0010 | [0x0018] | conditional jump to offset 15 |
| 9 | 0x0020 | - | arithmetic operation |
| 12 | 0x0028 | [0x0030] | unconditional jump to offset 20 |
| 15 | 0x0038 | - | jump target from offset 6 |
| 18 | 0x0040 | - | arithmetic operation |
| 20 | 0x0048 | - | jump target from offset 12 |

The **conditional branch translation** process converts VM conditional jumps into native conditional jump instructions while preserving exact branching behavior. The VM's `OP_JMP_IF_FALSE offset` instruction pops a value from the evaluation stack and jumps to the target offset if the value is false (zero). The native translation must test the register containing this value and generate a conditional jump instruction with the appropriate condition code.

```
Bytecode sequence:
  OP_LOAD_CONST 42
  OP_LOAD_CONST 0
  OP_CMP
  OP_JMP_IF_FALSE 10
  ... (true branch instructions)
  OP_JMP 15
  ... (false branch instructions at offset 10)
  ... (continuation at offset 15)
```

The native translation generates a sequence that tests the comparison result and branches accordingly. The comparison operation sets processor flags based on the operand values, and the subsequent conditional jump instruction examines these flags to determine whether to branch. This translation preserves the exact same branching decisions as the original bytecode while executing more efficiently.

**Loop translation** requires special attention because loops represent the most performance-critical control flow patterns in typical programs. The VM's loop constructs use backward jumps to implement iteration, but the native translation can leverage processor features like branch prediction and loop optimization to achieve superior performance.

Consider a simple counting loop in bytecode:
```
loop_start (offset 0):
  OP_LOAD_LOCAL 0    ; load counter variable
  OP_LOAD_CONST 100  ; load loop limit
  OP_CMP             ; compare counter with limit
  OP_JMP_IF_FALSE loop_end  ; exit if counter >= limit
  ... (loop body instructions)
  OP_LOAD_LOCAL 0    ; load counter
  OP_LOAD_CONST 1    ; load increment
  OP_ADD             ; increment counter
  OP_STORE_LOCAL 0   ; store updated counter
  OP_JMP loop_start  ; jump back to loop condition
loop_end:
```

The native translation optimizes this pattern by keeping the loop counter in a register throughout the loop execution, eliminating repeated memory accesses to the local variable. The counter increment and comparison can be combined into efficient instruction sequences that minimize branch overhead and maximize pipeline efficiency.

> **Decision: Jump Table vs. Switch Statement Translation**
> - **Context**: VM switch statements can be translated as jump tables for dense case values or as cascaded comparisons for sparse case values
> - **Options Considered**: Always use jump tables; always use comparison chains; hybrid approach based on case density analysis
> - **Decision**: Use comparison chains for switches with fewer than 8 cases, jump tables for dense switches with 8 or more cases
> - **Rationale**: Comparison chains provide better performance for small switches due to lower setup overhead, while jump tables excel for large dense switches by providing O(1) case selection
> - **Consequences**: Enables optimal performance for both small and large switch statements while adding minimal complexity to the translation logic

**Exception handling integration** ensures that native code maintains proper exception semantics when runtime errors occur. Division by zero, null pointer dereferences, and other runtime exceptions must be detected and handled consistently with the interpreted execution model. The native code includes explicit checks for these conditions and generates appropriate exception-raising sequences when errors are detected.

The **state synchronization** requirement means that when control flow transfers between native and interpreted code, the execution state must be properly marshaled between the two execution models. Register values must be converted back to stack-based representation when transferring to interpreted code, and stack-based state must be loaded into registers when transferring to native code.

### Common Pitfalls

⚠️ **Pitfall: Incorrect x86-64 Division Setup**
Division operations in x86-64 require specific register configuration that differs from other arithmetic operations. The dividend must be placed in `rax`, extended to 128 bits using `cqo` (which sign-extends `rax` into `rdx:rax`), and then divided using `idiv` with the divisor operand. Forgetting the `cqo` instruction or using the wrong registers causes incorrect results or processor exceptions. Always generate the complete sequence: `mov rax, dividend; cqo; idiv divisor_register`.

⚠️ **Pitfall: Forward Jump Target Calculation Errors**
Computing jump target addresses before all instructions are emitted leads to incorrect branch destinations because instruction sizes aren't known until emission completes. This commonly manifests as jumps that land in the middle of instructions or jump to wrong locations entirely. Use two-pass compilation: first pass emits instructions with placeholder jump targets, second pass patches all jump instructions with correct target addresses computed from the final instruction layout.

⚠️ **Pitfall: Register Allocation State Corruption**
Failing to maintain consistent register allocation state across instruction boundaries causes registers to contain unexpected values, leading to incorrect computations. This typically occurs when spill/reload logic doesn't properly track which registers contain which values. Maintain a register assignment table that's updated after each instruction translation, and verify that register contents match expected values before generating instructions that depend on specific register assignments.

⚠️ **Pitfall: Stack Pointer Alignment Violations**
x86-64 requires 16-byte stack alignment before function calls, but bytecode operations may leave the stack in misaligned states. Calling functions with misaligned stacks causes crashes or undefined behavior. Before generating call instructions, ensure the stack pointer is properly aligned by adjusting it to the nearest 16-byte boundary and compensating for any local variables or temporary storage.

⚠️ **Pitfall: Condition Code Flag Corruption**
x86-64 comparison operations set processor flags that are consumed by subsequent conditional jump instructions, but intervening instructions may modify these flags before the jump is executed. This causes conditional branches to test incorrect conditions, leading to wrong execution paths. Ensure that no flag-modifying instructions are emitted between comparison operations and their associated conditional jumps, or save/restore flags when intervening instructions are necessary.

### Implementation Guidance

The bytecode to native translator bridges high-level VM operations with low-level machine instructions, requiring careful coordination between instruction analysis, register management, and code generation. This implementation provides the foundation for converting abstract bytecode operations into efficient native execution while maintaining semantic correctness.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Instruction Analysis | Linear scan with basic pattern matching | Control flow graph with dataflow analysis |
| Register Allocation | Expression-level with simple spilling | Function-level with liveness analysis |
| Jump Resolution | Two-pass with patch table | Single-pass with lazy binding |
| Optimization | Peephole optimization on instruction sequences | Block-level optimization with SSA form |

**Recommended File Structure:**
```
src/
  jit/
    translator.c              ← main translation logic
    translator.h              ← public interfaces and types
    register_allocator.c      ← register assignment and spilling
    register_allocator.h      ← register allocation interfaces
    instruction_mapping.c     ← bytecode to native instruction mapping
    instruction_mapping.h     ← operation mapping definitions
    control_flow.c           ← jump and branch translation
    control_flow.h           ← control flow translation interfaces
    translator_test.c        ← comprehensive translation tests
  bytecode/
    opcodes.h               ← bytecode operation definitions (from prerequisite)
    vm.h                    ← VM state structures (from prerequisite)
  machine/
    emitter.h              ← machine code emitter interface (from previous milestone)
```

**Core Translation Infrastructure:**

```c
#include <stdint.h>
#include <stdlib.h>
#include "bytecode/opcodes.h"
#include "bytecode/vm.h"
#include "machine/emitter.h"

// Translation context maintains state during bytecode to native conversion
typedef struct TranslationContext {
    BytecodeFunction* function;          // source bytecode function
    CodeBuffer* code_buffer;            // target native code buffer
    RegisterAllocator* allocator;       // register assignment tracker
    JumpPatchTable* patch_table;        // forward reference resolution
    uint32_t current_bytecode_offset;   // current bytecode instruction position
    uint32_t current_native_offset;     // current native code generation position
    int compilation_failed;             // error flag for compilation failures
} TranslationContext;

// Register allocator tracks virtual stack to physical register mapping
typedef struct RegisterAllocator {
    X86Register register_assignments[MAX_EXPRESSION_DEPTH];  // virtual stack to register mapping
    int register_status[X86_REGISTER_COUNT];                // register availability status
    int current_stack_depth;                                // current virtual stack depth
    int max_stack_depth;                                    // maximum observed stack depth
    uint32_t spill_locations[MAX_EXPRESSION_DEPTH];         // stack frame offsets for spilled values
    int spill_count;                                        // number of currently spilled values
} RegisterAllocator;

// Jump patch table resolves forward references after instruction emission
typedef struct JumpPatchEntry {
    uint32_t bytecode_target_offset;    // target bytecode instruction offset
    uint32_t native_patch_location;     // native code location to patch
    JumpType jump_type;                 // conditional, unconditional, or call
} JumpPatchEntry;

typedef struct JumpPatchTable {
    JumpPatchEntry* entries;            // array of pending patches
    size_t capacity;                    // maximum number of patch entries
    size_t count;                       // current number of pending patches
    uint32_t* bytecode_to_native_map;   // mapping from bytecode to native offsets
    size_t map_size;                    // size of the offset mapping table
} JumpPatchTable;

// Translation result encapsulates the outcome of bytecode conversion
typedef struct TranslationResult {
    CompiledFunction compiled_function; // callable native function pointer
    size_t native_code_size;           // size of generated native code
    int register_count_used;           // number of registers utilized
    int spill_count;                   // number of values spilled to memory
    TranslationError error;            // detailed error information if compilation failed
} TranslationResult;

// Initialize translation context for bytecode function conversion
TranslationContext* translation_context_create(BytecodeFunction* function, CodeBuffer* code_buffer) {
    TranslationContext* ctx = malloc(sizeof(TranslationContext));
    ctx->function = function;
    ctx->code_buffer = code_buffer;
    ctx->allocator = register_allocator_create(MAX_EXPRESSION_DEPTH);
    ctx->patch_table = jump_patch_table_create(MAX_JUMP_COUNT);
    ctx->current_bytecode_offset = 0;
    ctx->current_native_offset = 0;
    ctx->compilation_failed = 0;
    return ctx;
}

// Register allocator manages virtual stack to physical register assignment
RegisterAllocator* register_allocator_create(int max_depth) {
    RegisterAllocator* allocator = malloc(sizeof(RegisterAllocator));
    
    // Initialize register assignments to unassigned state
    for (int i = 0; i < MAX_EXPRESSION_DEPTH; i++) {
        allocator->register_assignments[i] = X86_REGISTER_NONE;
        if (i < MAX_EXPRESSION_DEPTH) {
            allocator->spill_locations[i] = 0;
        }
    }
    
    // Mark all registers as available initially
    for (int i = 0; i < X86_REGISTER_COUNT; i++) {
        allocator->register_status[i] = REGISTER_AVAILABLE;
    }
    
    allocator->current_stack_depth = 0;
    allocator->max_stack_depth = max_depth;
    allocator->spill_count = 0;
    return allocator;
}

// Jump patch table manages forward reference resolution
JumpPatchTable* jump_patch_table_create(size_t capacity) {
    JumpPatchTable* table = malloc(sizeof(JumpPatchTable));
    table->entries = malloc(sizeof(JumpPatchEntry) * capacity);
    table->capacity = capacity;
    table->count = 0;
    
    // Initialize bytecode to native offset mapping
    table->bytecode_to_native_map = malloc(sizeof(uint32_t) * MAX_BYTECODE_SIZE);
    table->map_size = MAX_BYTECODE_SIZE;
    
    // Initialize all mappings to invalid offset
    for (size_t i = 0; i < table->map_size; i++) {
        table->bytecode_to_native_map[i] = INVALID_NATIVE_OFFSET;
    }
    
    return table;
}
```

**Core Translation Logic Skeleton:**

```c
// Main translation entry point - converts bytecode function to native code
TranslationResult translate_function(BytecodeFunction* function, CodeBuffer* code_buffer) {
    // TODO 1: Create translation context with function and code buffer
    // TODO 2: Generate function prologue following System V AMD64 ABI
    // TODO 3: Perform first pass - translate instructions and record jump patches
    // TODO 4: Perform second pass - resolve all forward jump references
    // TODO 5: Generate function epilogue and return instruction
    // TODO 6: Make code buffer executable and return callable function pointer
    // TODO 7: Handle any compilation errors by cleaning up and returning error result
}

// First pass translation - converts bytecode instructions to native sequences
int translate_instructions_first_pass(TranslationContext* ctx) {
    // TODO 1: Iterate through all bytecode instructions in function
    // TODO 2: Update bytecode-to-native offset mapping for each instruction
    // TODO 3: Dispatch each bytecode instruction to appropriate translation handler
    // TODO 4: Update register allocator state after each instruction
    // TODO 5: Record jump patch entries for forward references
    // TODO 6: Handle translation errors by setting error flag and returning
    // Hint: Use switch statement on bytecode opcode to dispatch to handlers
}

// Translate arithmetic bytecode operations to native instruction sequences
int translate_arithmetic_operation(TranslationContext* ctx, Opcode opcode) {
    // TODO 1: Determine source registers for operands based on stack depth
    // TODO 2: Generate appropriate x86-64 arithmetic instruction (add, sub, mul, etc.)
    // TODO 3: Handle special cases like division requiring rdx:rax setup
    // TODO 4: Update register allocator to reflect stack depth change
    // TODO 5: Handle register spilling if no registers available for result
    // Hint: Division requires "mov rax, operand1; cqo; idiv operand2"
}

// Translate conditional jump instructions with forward reference handling
int translate_conditional_jump(TranslationContext* ctx, Opcode opcode, uint32_t target_offset) {
    // TODO 1: Generate comparison instruction testing top stack value
    // TODO 2: Emit conditional jump instruction with placeholder target
    // TODO 3: Record jump patch entry for target offset resolution
    // TODO 4: Update register allocator to reflect popped condition value
    // TODO 5: Handle different jump conditions (zero, non-zero, etc.)
    // Hint: Use "test reg, reg; jz label" for jump-if-false operations
}
```

**Register Allocation Implementation:**

```c
// Allocate physical register for new virtual stack position
X86Register allocate_register_for_push(RegisterAllocator* allocator) {
    // TODO 1: Check if stack depth exceeds maximum expression depth
    // TODO 2: Find first available register from allocation priority list
    // TODO 3: If no register available, select victim register for spilling
    // TODO 4: Update register assignment table and status arrays
    // TODO 5: Increment current stack depth counter
    // Hint: Priority order should be rdi, rsi, rdx, rcx, r8, r9 for caller-saved
}

// Release physical register when value is popped from virtual stack
void deallocate_register_for_pop(RegisterAllocator* allocator) {
    // TODO 1: Verify stack depth is greater than zero
    // TODO 2: Get register assignment for top stack position
    // TODO 3: Mark register as available in status array
    // TODO 4: Clear register assignment for stack position
    // TODO 5: Decrement current stack depth counter
    // Hint: Always pop from top of stack (highest stack position)
}

// Spill register contents to memory when no registers available
uint32_t spill_register_to_memory(RegisterAllocator* allocator, X86Register reg, CodeBuffer* code_buffer) {
    // TODO 1: Allocate stack frame offset for spilled value
    // TODO 2: Generate mov instruction to store register to stack location
    // TODO 3: Update spill location tracking for affected stack position
    // TODO 4: Mark register as available for new allocation
    // TODO 5: Return stack frame offset for later reload
    // Hint: Use negative offsets from rbp for local variable storage
}

// Reload spilled value from memory back into register
void reload_spilled_value(RegisterAllocator* allocator, uint32_t spill_offset, X86Register target_reg, CodeBuffer* code_buffer) {
    // TODO 1: Verify spill offset is valid and contains spilled value
    // TODO 2: Generate mov instruction to load from stack location to register
    // TODO 3: Update register assignment table for loaded value
    // TODO 4: Mark target register as occupied
    // TODO 5: Clear spill location entry since value is now in register
    // Hint: Stack frame addressing uses [rbp - offset] format
}
```

**Jump Resolution Implementation:**

```c
// Second pass - resolve all forward jump references to native addresses
int resolve_jump_patches(TranslationContext* ctx) {
    // TODO 1: Iterate through all recorded jump patch entries
    // TODO 2: Look up target native offset from bytecode-to-native mapping
    // TODO 3: Calculate relative jump distance from patch location to target
    // TODO 4: Patch jump instruction at recorded location with computed offset
    // TODO 5: Verify all jump targets are valid and within code buffer bounds
    // Hint: x86-64 relative jumps use signed 32-bit offsets from end of jump instruction
}

// Add forward jump reference to patch table for later resolution
int record_jump_patch(JumpPatchTable* table, uint32_t target_bytecode_offset, uint32_t patch_location, JumpType type) {
    // TODO 1: Verify patch table has capacity for new entry
    // TODO 2: Create new patch entry with target offset and patch location
    // TODO 3: Set jump type (conditional, unconditional, call) for correct patching
    // TODO 4: Add entry to patch table and increment count
    // TODO 5: Return success/failure status for error handling
    // Hint: Conditional jumps use different opcodes than unconditional jumps
}

// Update bytecode-to-native offset mapping during instruction emission
void update_offset_mapping(JumpPatchTable* table, uint32_t bytecode_offset, uint32_t native_offset) {
    // TODO 1: Verify bytecode offset is within valid range
    // TODO 2: Store native offset in mapping table at bytecode offset index
    // TODO 3: Handle case where bytecode offset already has mapping (error condition)
    // TODO 4: Ensure mapping table has sufficient capacity for offset
    // Hint: Each bytecode instruction should map to exactly one native code location
}
```

**Milestone Checkpoint:**

After implementing the bytecode to native translator, verify the following functionality:

1. **Arithmetic Translation Test**: Compile and execute simple arithmetic expressions, comparing results with interpreter output. Expected: identical results for all arithmetic operations including edge cases like division by zero.

2. **Register Allocation Verification**: Monitor register usage during expression compilation, ensuring efficient register utilization without unnecessary spills. Expected: expressions with depth ≤ 6 should require no spills.

3. **Control Flow Translation Test**: Compile and execute functions with conditional branches and loops, verifying correct execution paths under all conditions. Expected: identical branching behavior to interpreted execution.

4. **Jump Resolution Verification**: Inspect generated native code to ensure all jump instructions have correct target addresses and no unresolved forward references remain. Expected: all jumps land at instruction boundaries with correct offsets.

Common issues at this checkpoint:
- **Incorrect arithmetic results**: Usually caused by wrong register operand ordering or missing instruction prefixes
- **Segmentation faults**: Often from unaligned stack access or incorrect jump target calculations
- **Infinite loops**: Typically from incorrect jump offset calculations or condition code handling


## Function Compilation and ABI

> **Milestone(s):** Milestone 3 (Function JIT and Calling Convention) — this section covers the complete function compilation process, including calling convention compliance, stack frame management, and seamless transitions between interpreted and native code execution modes

### Mental Model: Diplomatic Protocol

Understanding calling conventions as standardized communication protocols between functions provides an intuitive framework for comprehending this complex aspect of JIT compilation. Consider how diplomats from different countries must follow precise protocols when conducting international meetings: they must agree on the order of speaking, the language to use, how documents are exchanged, and who sits where. Similarly, when JIT-compiled functions interact with interpreted code or other native functions, they must follow the **System V AMD64 calling convention** — a standardized protocol that governs how arguments are passed, where return values are placed, which registers must be preserved, and how the stack is managed.

Just as a diplomatic breach of protocol can derail international negotiations, violating calling convention rules results in crashes, corruption, or silent data loss. The calling convention serves as the "universal translator" that enables seamless communication between code compiled at different times, by different compilers, or even written in different languages. Our JIT compiler must generate **prologue and epilogue code** that acts like diplomatic ceremonial procedures — formal opening and closing rituals that ensure proper protocol compliance.

The **trampoline functions** we implement serve as interpreters at international meetings, translating between the "native code protocol" and the "bytecode interpreter protocol." When a JIT-compiled function needs to call back into the interpreter, the trampoline marshals arguments from native registers into the interpreter's expected format, just as a diplomatic interpreter translates not only language but also cultural context and procedural expectations.

### System V AMD64 Calling Convention

The System V AMD64 Application Binary Interface defines the standardized rules that govern function calls on x86-64 Linux and macOS systems. This calling convention ensures that functions compiled by different compilers, at different times, or even written in different programming languages can successfully call each other. Our JIT compiler must generate code that strictly adheres to these rules to enable seamless interoperability with the existing bytecode interpreter and any native library functions.

The calling convention specifies precise rules for **argument passing** through a combination of registers and stack allocation. The first six integer arguments are passed in registers `rdi`, `rsi`, `rdx`, `rcx`, `r8`, and `r9` respectively. Additional arguments beyond the sixth are pushed onto the stack in reverse order. Floating-point arguments use the SSE registers `xmm0` through `xmm7`, but our bytecode VM focuses on integer operations, simplifying our implementation requirements.

**Return value handling** follows equally strict rules. Integer return values up to 64 bits are returned in the `rax` register. Larger return values or complex structures follow more intricate rules involving multiple registers or memory addresses passed as implicit parameters. Our bytecode functions typically return simple integer values, allowing us to focus on the `rax` return mechanism.

**Register preservation** represents one of the most critical aspects of calling convention compliance. Registers are classified into two categories: **caller-saved** and **callee-saved**. Caller-saved registers (`rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`, `r9`, `r10`, `r11`) may be modified by the called function without preservation — the calling function must save these values before the call if needed. Callee-saved registers (`rbx`, `rbp`, `r12`, `r13`, `r14`, `r15`) must be preserved by the called function — if the function modifies these registers, it must restore their original values before returning.

> **Decision: Register Allocation Strategy**
> - **Context**: JIT-compiled functions need register allocation that balances performance with ABI compliance requirements
> - **Options Considered**: 
>   1. Use only caller-saved registers (simple but limits optimization)
>   2. Use callee-saved registers with full preservation (complex but optimal)
>   3. Hybrid approach using caller-saved primarily with selective callee-saved usage
> - **Decision**: Primarily use caller-saved registers with limited callee-saved register usage
> - **Rationale**: Minimizes prologue/epilogue complexity while providing sufficient registers for most expressions, reducing implementation complexity during initial development
> - **Consequences**: May result in more register spilling for complex expressions but ensures robust ABI compliance

| Register Category | Registers | Preservation Rule | Usage in JIT |
|------------------|-----------|-------------------|--------------|
| Argument Passing | rdi, rsi, rdx, rcx, r8, r9 | Caller-saved | Function entry argument extraction |
| Return Value | rax | Caller-saved | Function exit return value placement |
| General Purpose Caller-Saved | r10, r11 | Caller-saved | Primary allocation for expression evaluation |
| General Purpose Callee-Saved | rbx, r12, r13, r14, r15 | Must preserve | Secondary allocation for longer-lived values |
| Stack Pointer | rsp | Special handling | Stack frame management and alignment |
| Base Pointer | rbp | Callee-saved (optional) | Optional frame pointer for debugging |

**Stack alignment** requires particular attention due to ABI mandates. The stack pointer must be 16-byte aligned immediately before any `call` instruction. Since the `call` instruction pushes an 8-byte return address, the called function receives a stack pointer that is 8 bytes offset from 16-byte alignment. The function's prologue must account for this offset when allocating stack space, ensuring that any subsequent function calls maintain proper alignment.

> The critical insight for stack alignment is that misalignment doesn't always cause immediate crashes — it may only manifest when calling system libraries or optimized code that uses SSE instructions, making it a particularly insidious debugging challenge.

### Stack Frame Management

Proper stack frame management forms the foundation of reliable function compilation, encompassing the allocation of local variable space, preservation of callee-saved registers, and maintenance of stack pointer alignment throughout function execution. The **function prologue** establishes the stack frame at function entry, while the **epilogue** tears down the frame and restores the execution environment for the caller.

The **prologue generation** process follows a precise sequence of operations that must be executed atomically from the perspective of debugging and exception handling. Our JIT compiler generates prologue code that first preserves any callee-saved registers that the function will modify, then allocates stack space for local variables and temporary storage. The prologue must calculate the exact stack space requirements based on the bytecode function's local variable count and the maximum expression depth that might require register spilling.

**Local variable allocation** requires mapping the bytecode function's local variable slots to specific stack frame offsets. Each local variable receives a dedicated stack slot at a negative offset from the frame base pointer. The allocation must account for variable size (our implementation assumes 64-bit values), proper alignment requirements, and efficient addressing modes for x86-64 memory operands. The mapping between local variable indices and stack offsets remains constant throughout the function's execution, enabling consistent access patterns.

> **Decision: Stack Frame Layout Strategy**
> - **Context**: Need to organize stack frame for local variables, register spills, and ABI compliance
> - **Options Considered**:
>   1. Fixed frame layout with rbp as frame pointer
>   2. Dynamic rsp-relative addressing without frame pointer
>   3. Hybrid approach using rbp selectively
> - **Decision**: Use rbp as frame pointer with fixed layout
> - **Rationale**: Simplifies address calculation for local variables and register spills, improves debugging experience, and provides stable reference point for stack operations
> - **Consequences**: Requires rbp preservation overhead but significantly simplifies memory address generation

| Stack Frame Component | Offset from rbp | Size | Purpose |
|----------------------|-----------------|------|---------|
| Previous rbp | 0 | 8 bytes | Caller's frame pointer (pushed by prologue) |
| Local Variable N-1 | -8 | 8 bytes | Last local variable (highest index) |
| Local Variable N-2 | -16 | 8 bytes | Second-to-last local variable |
| ... | ... | ... | Additional local variables |
| Local Variable 0 | -(N*8) | 8 bytes | First local variable (lowest index) |
| Register Spill Area | -(N*8 + spill_size) | Variable | Temporary storage for register spills |
| Stack Alignment Padding | Variable | 0-8 bytes | Maintain 16-byte alignment |

**Register spill handling** becomes necessary when the expression evaluation depth exceeds the available register count or when function calls require preservation of live values across call boundaries. The JIT compiler must generate code to save register contents to designated stack frame locations, perform the operation requiring the register, then reload the saved values. Spill locations are allocated dynamically as needed, with the stack frame size adjusted during compilation to accommodate the maximum spill requirements discovered during code generation.

The **epilogue generation** process reverses the prologue operations in the correct order to ensure proper stack unwinding. The epilogue must restore all callee-saved registers from their stack locations, deallocate the stack frame space, and execute the return instruction. The order of operations is critical: registers must be restored before the stack pointer adjustment, and the frame pointer must be restored last to maintain proper stack unwinding for debugging tools.

```
Stack Frame Allocation Algorithm:
1. Calculate total local variable space: local_count * 8 bytes
2. Estimate maximum register spill space based on expression complexity
3. Add alignment padding to ensure 16-byte boundary before function calls
4. Generate prologue: push rbp, mov rbp rsp, sub rsp frame_size
5. Save callee-saved registers that will be modified
6. Generate epilogue: restore callee-saved registers, mov rsp rbp, pop rbp, ret
```

### JIT-Interpreter Transitions

The seamless transition between JIT-compiled native code and bytecode interpretation represents one of the most complex aspects of the entire system architecture. These transitions must preserve execution state, maintain calling convention compliance, and handle the fundamental differences between native register-based execution and stack-based bytecode interpretation. The **state marshaling** process converts between these two execution models without losing data or violating ABI requirements.

**JIT-to-interpreter transitions** occur when JIT-compiled code needs to call a function that hasn't been compiled yet or when calling built-in VM operations that remain implemented in the interpreter. The transition mechanism must extract arguments from native registers, package them into the interpreter's expected stack format, invoke the interpreter function, and convert the result back to native calling convention format. This process requires careful attention to register preservation, stack alignment, and error propagation.

The **trampoline implementation** serves as the bridge between execution modes, providing standardized entry and exit points that handle the mechanical aspects of state conversion. A JIT-to-interpreter trampoline receives arguments in native registers, saves the caller's state, sets up the interpreter's execution context, invokes the bytecode function through the interpreter's dispatch mechanism, captures the result, restores the caller's native state, and returns the result in the appropriate register.

**Interpreter-to-JIT transitions** occur when interpreted code calls a function that has been JIT-compiled. The interpreter must detect that a compiled version exists, extract arguments from its stack-based representation, place them in the appropriate registers according to the calling convention, invoke the compiled function, and integrate the returned result back into its stack-based execution model.

> **Decision: Transition Mechanism Design**
> - **Context**: Need efficient and reliable mechanism for switching between interpreted and native execution modes
> - **Options Considered**:
>   1. Inline state marshaling at each call site
>   2. Centralized trampoline functions for all transitions  
>   3. Dynamic trampoline generation per function signature
> - **Decision**: Centralized trampoline functions with standardized interfaces
> - **Rationale**: Reduces code duplication, centralizes transition logic for easier debugging, and provides consistent error handling across all transitions
> - **Consequences**: Slight performance overhead for transitions but significantly improved maintainability and reliability

| Transition Type | Source State | Target State | Marshaling Requirements |
|----------------|--------------|--------------|------------------------|
| JIT → Interpreter | Registers (rdi, rsi, etc.) | VM stack slots | Extract args, push to VM stack, call interpreter |
| Interpreter → JIT | VM stack slots | Registers (rdi, rsi, etc.) | Pop from VM stack, place in registers, call native |
| JIT → JIT | Registers | Registers | Direct native function call (no marshaling) |
| Interpreter → Interpreter | VM stack | VM stack | Normal interpreter dispatch (no marshaling) |

**Error handling across transitions** requires special consideration because exceptions or runtime errors may occur in either execution mode and must be properly propagated back to the caller regardless of the execution model mismatch. The trampoline functions must establish error handling contexts that can capture interpreter exceptions and convert them to appropriate native code error indications, or vice versa.

**Performance optimization** of transitions focuses on minimizing the overhead of state marshaling, particularly for frequently called functions. The system can cache argument layouts, pre-allocate marshaling buffers, and optimize the most common transition patterns. However, correctness always takes precedence over performance — a fast but incorrect transition mechanism would undermine the entire JIT compilation benefit.

The **transition state machine** tracks the current execution mode and manages the context switching process. Each function maintains metadata indicating whether it's currently being interpreted, has been JIT-compiled, or is in the process of compilation. This metadata guides the transition mechanism's decisions about which trampoline to use and how to handle the state conversion.

⚠️ **Pitfall: Stack Alignment Violations**
Many developers forget that the System V AMD64 ABI requires 16-byte stack alignment before function calls. When generating JIT code that calls back into the interpreter, failure to maintain proper alignment can cause crashes in system library functions or optimized code that uses SSE instructions. Always verify stack alignment with `(rsp & 0xF) == 0` before call instructions.

⚠️ **Pitfall: Register Clobbering in Transitions**
The trampoline functions must carefully preserve all caller-saved registers that might contain live values during JIT-to-interpreter transitions. A common mistake is assuming that only argument registers need preservation, forgetting that the calling JIT code might have live values in other caller-saved registers like r10 and r11.

⚠️ **Pitfall: Incomplete Error Propagation**
When interpreter functions encounter runtime errors (division by zero, stack overflow, etc.), these errors must be properly converted to native code error indications. Simply returning error codes isn't sufficient — the system must maintain exception context and stack unwinding information across the transition boundary.

![Function Execution State Machine](./diagrams/execution-state-machine.svg)

![Executable Memory Layout](./diagrams/memory-layout.svg)

![JIT-Interpreter Call Sequence](./diagrams/calling-sequence.svg)

### Implementation Guidance

This section provides the concrete implementation building blocks for function compilation with proper ABI compliance and seamless JIT-interpreter transitions.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Stack Frame Management | Fixed rbp-based frames with manual offset calculation | Dynamic frame layout with compiler-assisted optimization |
| Calling Convention | System V AMD64 with basic register usage | Full register optimization with advanced spill strategies |
| State Marshaling | Simple argument copying with type assumptions | Generic marshaling with runtime type information |
| Error Handling | Basic return code propagation | Full exception context preservation |

**File Structure:**

```
jit-compiler/
  src/
    function_compiler.c       ← main function compilation logic
    function_compiler.h       ← public interfaces and types
    abi_support.c            ← System V AMD64 calling convention helpers
    abi_support.h            ← ABI constants and function signatures
    trampoline.c             ← JIT-interpreter transition trampolines
    trampoline.h             ← trampoline function declarations
    stack_frame.c            ← stack frame management utilities
    stack_frame.h            ← frame layout structures and functions
  tests/
    test_function_compiler.c  ← comprehensive function compilation tests
    test_abi_compliance.c     ← calling convention correctness verification
    test_transitions.c        ← JIT-interpreter transition tests
```

**Infrastructure Starter Code:**

Complete ABI support implementation with register management and stack alignment:

```c
// abi_support.h
#ifndef ABI_SUPPORT_H
#define ABI_SUPPORT_H

#include <stdint.h>
#include <stddef.h>

// System V AMD64 register assignments
typedef enum {
    X86_REGISTER_RAX = 0,
    X86_REGISTER_RCX = 1, 
    X86_REGISTER_RDX = 2,
    X86_REGISTER_RBX = 3,
    X86_REGISTER_RSP = 4,
    X86_REGISTER_RBP = 5,
    X86_REGISTER_RSI = 6,
    X86_REGISTER_RDI = 7,
    X86_REGISTER_R8 = 8,
    X86_REGISTER_R9 = 9,
    X86_REGISTER_R10 = 10,
    X86_REGISTER_R11 = 11,
    X86_REGISTER_R12 = 12,
    X86_REGISTER_R13 = 13,
    X86_REGISTER_R14 = 14,
    X86_REGISTER_R15 = 15,
    X86_REGISTER_NONE = 255
} X86Register;

// ABI constants
#define STACK_ALIGNMENT 16
#define ARGUMENT_REGISTER_COUNT 6
#define CALLEE_SAVED_REGISTER_COUNT 5

extern const X86Register ARGUMENT_REGISTERS[ARGUMENT_REGISTER_COUNT];
extern const X86Register CALLEE_SAVED_REGISTERS[CALLEE_SAVED_REGISTER_COUNT];

typedef struct {
    int argument_count;
    X86Register register_args[ARGUMENT_REGISTER_COUNT];
    int stack_args_size;
    int requires_alignment_padding;
} ABICallInfo;

// Calculate calling convention layout for function call
ABICallInfo* abi_analyze_call(int argument_count);
void abi_call_info_destroy(ABICallInfo* info);

// Stack frame size calculation with alignment
size_t abi_calculate_frame_size(int local_count, int max_spill_count);
int abi_is_callee_saved_register(X86Register reg);
int abi_is_caller_saved_register(X86Register reg);

#endif
```

```c
// abi_support.c
#include "abi_support.h"
#include <stdlib.h>

const X86Register ARGUMENT_REGISTERS[ARGUMENT_REGISTER_COUNT] = {
    X86_REGISTER_RDI, X86_REGISTER_RSI, X86_REGISTER_RDX,
    X86_REGISTER_RCX, X86_REGISTER_R8, X86_REGISTER_R9
};

const X86Register CALLEE_SAVED_REGISTERS[CALLEE_SAVED_REGISTER_COUNT] = {
    X86_REGISTER_RBX, X86_REGISTER_R12, X86_REGISTER_R13,
    X86_REGISTER_R14, X86_REGISTER_R15
};

ABICallInfo* abi_analyze_call(int argument_count) {
    ABICallInfo* info = malloc(sizeof(ABICallInfo));
    info->argument_count = argument_count;
    
    // First 6 arguments in registers
    int reg_arg_count = (argument_count < ARGUMENT_REGISTER_COUNT) ? 
                        argument_count : ARGUMENT_REGISTER_COUNT;
    for (int i = 0; i < reg_arg_count; i++) {
        info->register_args[i] = ARGUMENT_REGISTERS[i];
    }
    
    // Remaining arguments on stack
    int stack_arg_count = (argument_count > ARGUMENT_REGISTER_COUNT) ?
                         argument_count - ARGUMENT_REGISTER_COUNT : 0;
    info->stack_args_size = stack_arg_count * 8;
    
    // Check if stack padding needed for 16-byte alignment
    info->requires_alignment_padding = (info->stack_args_size % STACK_ALIGNMENT) != 0;
    
    return info;
}

void abi_call_info_destroy(ABICallInfo* info) {
    free(info);
}

size_t abi_calculate_frame_size(int local_count, int max_spill_count) {
    size_t base_size = local_count * 8;  // 8 bytes per local variable
    base_size += max_spill_count * 8;     // 8 bytes per spill slot
    
    // Round up to 16-byte alignment
    return (base_size + STACK_ALIGNMENT - 1) & ~(STACK_ALIGNMENT - 1);
}

int abi_is_callee_saved_register(X86Register reg) {
    for (int i = 0; i < CALLEE_SAVED_REGISTER_COUNT; i++) {
        if (CALLEE_SAVED_REGISTERS[i] == reg) return 1;
    }
    return reg == X86_REGISTER_RBP;  // rbp is also callee-saved
}

int abi_is_caller_saved_register(X86Register reg) {
    return !abi_is_callee_saved_register(reg) && reg != X86_REGISTER_RSP;
}
```

Complete stack frame management with prologue/epilogue generation:

```c
// stack_frame.h
#ifndef STACK_FRAME_H
#define STACK_FRAME_H

#include <stdint.h>
#include <stddef.h>
#include "abi_support.h"
#include "code_buffer.h"

typedef struct {
    int local_variable_count;
    int max_spill_slots;
    size_t total_frame_size;
    X86Register used_callee_saved[CALLEE_SAVED_REGISTER_COUNT];
    int callee_saved_count;
    int uses_frame_pointer;
} StackFrameLayout;

// Stack frame layout calculation
StackFrameLayout* stack_frame_create_layout(int local_count, int max_spill_count, 
                                           const X86Register* used_registers, int register_count);
void stack_frame_destroy_layout(StackFrameLayout* layout);

// Offset calculations
int32_t stack_frame_local_variable_offset(const StackFrameLayout* layout, int local_index);
int32_t stack_frame_spill_slot_offset(const StackFrameLayout* layout, int spill_index);

// Code generation
int emit_function_prologue(CodeBuffer* buffer, const StackFrameLayout* layout);
int emit_function_epilogue(CodeBuffer* buffer, const StackFrameLayout* layout);
int emit_save_callee_saved_registers(CodeBuffer* buffer, const StackFrameLayout* layout);
int emit_restore_callee_saved_registers(CodeBuffer* buffer, const StackFrameLayout* layout);

#endif
```

**Core Logic Skeleton Code:**

Main function compilation with proper ABI compliance:

```c
// Function compilation with complete ABI support
typedef struct {
    uint32_t function_id;
    uint8_t* bytecode;
    size_t bytecode_length;
    int argument_count;
    int local_variable_count;
} BytecodeFunction;

typedef int64_t (*CompiledFunction)(int64_t arg1, int64_t arg2, int64_t arg3, 
                                  int64_t arg4, int64_t arg5, int64_t arg6);

// Compile bytecode function to native code with full ABI compliance
CompiledFunction compile_function(BytecodeFunction* function, CodeBuffer* code_buffer) {
    // TODO 1: Analyze function to determine register usage and stack requirements
    // - Scan bytecode to estimate maximum expression depth
    // - Determine which callee-saved registers will be needed
    // - Calculate total stack frame size including locals and spills
    
    // TODO 2: Create stack frame layout based on analysis results
    // - Use stack_frame_create_layout() with calculated requirements
    // - Account for argument count and local variable count
    // - Ensure proper alignment for function calls
    
    // TODO 3: Generate function prologue following System V AMD64 ABI
    // - Use emit_function_prologue() to establish stack frame
    // - Save callee-saved registers that will be modified
    // - Set up frame pointer if using rbp-based addressing
    
    // TODO 4: Generate argument extraction code
    // - Move arguments from calling convention registers to local storage
    // - Handle case where argument count exceeds register capacity
    // - Store arguments in predictable stack frame locations
    
    // TODO 5: Translate bytecode instructions to native code
    // - Use translate_function() from bytecode translator
    // - Apply register allocation strategy appropriate for stack frame
    // - Handle function calls with proper stack alignment
    
    // TODO 6: Generate function epilogue with proper cleanup
    // - Use emit_function_epilogue() to tear down stack frame
    // - Restore all callee-saved registers in reverse order
    // - Ensure return value is placed in rax register
    
    // TODO 7: Finalize and return function pointer
    // - Make memory executable using code_buffer_make_executable()
    // - Cast buffer memory to CompiledFunction pointer
    // - Return function pointer ready for native calls
    
    return NULL;
}
```

Trampoline implementation for seamless transitions:

```c
// JIT to interpreter trampoline - handles state marshaling
typedef struct {
    int64_t* stack_slots;
    size_t stack_capacity;
    size_t stack_top;
    // Additional interpreter state...
} InterpreterState;

int64_t jit_to_interpreter_trampoline(uint32_t function_id, int argument_count,
                                     int64_t arg1, int64_t arg2, int64_t arg3,
                                     int64_t arg4, int64_t arg5, int64_t arg6) {
    // TODO 1: Allocate or acquire interpreter state context
    // - Get thread-local interpreter instance
    // - Ensure stack has sufficient capacity for arguments
    // - Save current native register state if needed
    
    // TODO 2: Marshal arguments from registers to interpreter stack
    // - Push arguments onto interpreter stack in correct order
    // - Handle argument count validation and bounds checking
    // - Convert native calling convention to stack-based format
    
    // TODO 3: Invoke interpreter function dispatch mechanism
    // - Look up bytecode function by function_id
    // - Set interpreter program counter to function start
    // - Execute function through normal interpreter dispatch loop
    
    // TODO 4: Extract return value from interpreter state
    // - Pop return value from interpreter stack
    // - Perform any necessary type conversion or validation
    // - Handle case where function doesn't return a value
    
    // TODO 5: Clean up interpreter state and restore native context
    // - Clear temporary stack allocations
    // - Restore any saved native register state
    // - Handle any exceptions or errors that occurred during interpretation
    
    return 0;  // Return extracted value
}

// Interpreter to JIT trampoline - handles reverse marshaling  
CompiledFunction interpreter_to_jit_trampoline(uint32_t function_id) {
    // TODO 1: Look up JIT-compiled function in code cache
    // - Check if function has been compiled
    // - Verify compiled code is still valid and executable
    // - Handle case where compilation failed or was invalidated
    
    // TODO 2: Return wrapper function that handles argument marshaling
    // - Create wrapper that extracts arguments from interpreter stack
    // - Place arguments in appropriate calling convention registers
    // - Call actual JIT-compiled function with proper ABI compliance
    
    // TODO 3: Handle return value marshaling in wrapper
    // - Extract return value from rax register
    // - Push result back onto interpreter stack
    // - Handle any exceptions thrown by native code
    
    return NULL;  // Return wrapper function pointer
}
```

**Language-Specific Implementation Hints:**

- Use `mprotect()` to change memory protection after code generation completes, never keep memory both writable and executable simultaneously
- The `__attribute__((sysv_abi))` attribute in GCC can help verify calling convention compliance during development
- Use `gdb` with `disas` command to inspect generated machine code and verify instruction sequences
- Linux `/proc/self/maps` shows memory protection flags for debugging executable memory allocation
- The `objdump -D` command can disassemble memory regions to verify instruction encoding correctness

**Milestone Checkpoint:**

After implementing function compilation with ABI support:

1. **Compile and test basic function:**
   ```bash
   gcc -g -o test_function_compiler test_function_compiler.c function_compiler.c abi_support.c stack_frame.c
   ./test_function_compiler
   ```

2. **Expected output:** Successfully compile simple arithmetic functions, verify calling convention compliance, and demonstrate JIT-interpreter transitions

3. **Manual verification:** Use `gdb` to single-step through generated code, verify stack frame layout matches expectations, and confirm register preservation

4. **Performance check:** Compare execution time of JIT-compiled functions versus interpretation for compute-intensive operations

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Segmentation fault on function call | Stack misalignment or bad function pointer | Use `gdb` to check stack pointer alignment before call | Ensure 16-byte stack alignment and proper code buffer setup |
| Wrong return values | Register preservation issue or incorrect epilogue | Check if callee-saved registers are properly restored | Add register save/restore in prologue/epilogue |
| Crashes in system functions | Calling convention violation | Verify argument registers and stack layout match ABI | Review argument marshaling and register usage |
| Memory corruption | Stack frame size calculation error | Check local variable and spill slot offset calculations | Recalculate frame size with proper alignment |


## Hot Path Detection and Profiling

> **Milestone(s):** Milestone 4 (Hot Path Detection and Optimization) — this section covers the profiling system that identifies frequently executed code paths and triggers JIT compilation when execution frequency exceeds configured thresholds

The challenge of runtime performance optimization lies in selectively applying expensive compilation techniques only where they provide the greatest benefit. Just as an experienced doctor uses a thermometer to identify which patients have a fever requiring treatment, a JIT compiler uses profiling data to identify which code paths are "hot" enough to warrant the cost of native compilation.

### Mental Model: Performance Thermometer

Understanding hot path detection requires thinking of code execution frequency as temperature measurement. In the same way a thermometer measures heat to identify patients who need medical attention, execution counters measure code "temperature" to identify functions that need compilation attention.

**Cold Code** represents the majority of program functions that execute infrequently. Like patients with normal body temperature, these functions run adequately under interpretation and don't justify the overhead of compilation. The interpreter handles them efficiently enough for their usage patterns.

**Warming Code** consists of functions that execute moderately frequently but haven't yet crossed the compilation threshold. These are like patients with slightly elevated temperatures that require monitoring but don't yet need intensive treatment. The profiler tracks their execution counts and watches for trends.

**Hot Code** represents the critical few functions that execute very frequently and consume the majority of program runtime. Like patients with high fevers requiring immediate intervention, these functions desperately need the performance benefits of native compilation. The JIT compiler prioritizes these functions for immediate translation.

**Compilation Threshold** acts as the diagnostic boundary that separates normal from abnormal execution frequency. Just as a doctor uses 100.4°F as the fever threshold, the JIT compiler uses a configurable invocation count threshold to trigger compilation decisions. Functions exceeding this threshold receive native compilation treatment.

The profiler maintains lightweight execution counters that impose minimal overhead during interpretation, similar to how a digital thermometer provides quick, accurate readings without disrupting the patient. These counters accumulate data across all function invocations, building a comprehensive picture of program behavior over time.

![Function Execution State Machine](./diagrams/execution-state-machine.svg)

The state machine governing function execution demonstrates how functions progress from cold interpretation through warming phases to hot native execution, with threshold-based transitions driving the compilation pipeline.

### Execution Counter Instrumentation

The foundation of hot path detection lies in lightweight execution frequency measurement that captures function invocation patterns without significantly impacting interpretation performance. The profiling system must balance measurement accuracy with runtime overhead, ensuring that the cost of profiling never exceeds the benefits of optimization.

**Profiling Data Structure Design** requires careful attention to memory layout and access patterns. The `Profiler` maintains an array of `ProfileEntry` structures, each tracking the execution statistics for a specific function. The profiler uses the function's unique identifier as an index into this array, enabling constant-time counter updates during function invocation.

| Field Name | Type | Description |
|------------|------|-------------|
| entries | ProfileEntry* | Array of profiling data indexed by function ID |
| capacity | size_t | Maximum number of functions that can be tracked simultaneously |
| default_threshold | uint32_t | Default compilation threshold for newly registered functions |

Each `ProfileEntry` maintains the complete execution history and compilation state for a single function, encapsulating both statistical data and compilation metadata in a compact structure optimized for frequent access.

| Field Name | Type | Description |
|------------|------|-------------|
| function_id | uint32_t | Unique identifier for the function being profiled |
| invocation_count | uint32_t | Total number of times this function has been called |
| compilation_threshold | uint32_t | Invocation count that triggers JIT compilation for this function |
| is_compiled | int | Boolean flag indicating whether this function has been JIT compiled |

> **Design Insight**: The profiler uses function ID as a direct array index rather than hash table lookup to minimize profiling overhead. This requires the bytecode VM to assign sequential function IDs, but eliminates hash computation and collision handling during the critical path of function invocation.

**Counter Update Strategy** determines how and when execution counters increment during program execution. The profiler integrates with the bytecode VM's function call mechanism, incrementing counters immediately before function execution begins. This placement ensures that every function invocation contributes to the frequency measurement, regardless of whether the function executes through interpretation or native compilation.

The `profiler_record_invocation` function implements the core profiling logic with minimal computational overhead. The function performs three essential operations: locates the appropriate profile entry, increments the invocation counter, and evaluates whether the compilation threshold has been exceeded.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| profiler_record_invocation | Profiler* profiler, uint32_t function_id | int | Increments invocation counter and returns 1 if compilation threshold exceeded |
| profiler_create | uint32_t default_threshold | Profiler* | Allocates and initializes profiler with specified default threshold |
| profiler_reset_function | Profiler* profiler, uint32_t function_id | void | Resets invocation count to zero for the specified function |
| profiler_set_threshold | Profiler* profiler, uint32_t function_id, uint32_t threshold | int | Sets custom compilation threshold for specific function |

**Overhead Minimization Techniques** ensure that profiling instrumentation doesn't degrade interpretation performance significantly. The profiler employs several optimization strategies to keep overhead below 5% of total execution time.

**Direct Array Access** eliminates hash table overhead by using function IDs as direct indices into the profile entry array. This approach requires O(1) time for counter updates but demands careful coordination with the bytecode VM's function ID assignment strategy.

**Minimal Computation per Invocation** restricts the profiling code path to simple arithmetic operations and conditional comparisons. The profiler avoids expensive operations like memory allocation, string manipulation, or complex mathematical calculations during the hot path of function invocation.

**Cache-Friendly Data Layout** organizes profile entries in a compact array structure that maximizes CPU cache utilization. Each `ProfileEntry` fits within a single cache line, and the sequential access pattern during program execution maintains good spatial locality.

> **Architecture Decision: Synchronous vs. Asynchronous Profiling**
> - **Context**: Profiling can occur synchronously during function calls or asynchronously via sampling
> - **Options Considered**: 
>   - Synchronous counting: increment counters during each function call
>   - Sampling-based profiling: periodically sample execution state
>   - Hybrid approach: synchronous counting with asynchronous analysis
> - **Decision**: Synchronous counting with immediate threshold evaluation
> - **Rationale**: Synchronous profiling provides 100% accuracy for function invocation frequency, enabling precise threshold-based compilation triggers. Sampling introduces statistical uncertainty that could miss short-lived hot functions or delay compilation of critical paths.
> - **Consequences**: Adds small overhead to every function call but guarantees accurate hot path detection

| Profiling Approach | Accuracy | Overhead | Complexity | Implementation Difficulty |
|-------------------|----------|----------|------------|---------------------------|
| Synchronous Counting | 100% accurate | Low (1-5%) | Simple | Easy |
| Statistical Sampling | 90-95% accurate | Very Low (<1%) | Complex | Hard |
| Hybrid Approach | 98% accurate | Medium (2-8%) | Very Complex | Very Hard |

**Integration with VM Function Calls** requires careful coordination between the profiler and the bytecode VM's execution engine. The profiler hooks into the VM's function dispatch mechanism, recording invocations before transferring control to either the interpreter or JIT-compiled native code.

The profiling integration point occurs at the highest level of function dispatch, ensuring that all execution paths through both interpreted and native code contribute to the frequency measurements. This placement prevents double-counting when JIT-compiled functions call other JIT-compiled functions, while maintaining accurate statistics across execution mode transitions.

### Compilation Trigger Policies

The decision of when to compile a function from bytecode to native code represents a critical trade-off between compilation cost and execution benefit. Compilation trigger policies define the rules and thresholds that govern this decision-making process, balancing the expensive upfront cost of JIT compilation against the ongoing performance benefits of native execution.

**Threshold-Based Compilation** forms the foundation of most JIT compilation strategies. When a function's invocation count exceeds its configured compilation threshold, the system initiates JIT compilation for that function. The threshold value determines the aggressiveness of the compilation strategy — lower thresholds compile more functions but incur higher compilation overhead, while higher thresholds reduce compilation cost but may miss optimization opportunities.

> **Design Insight**: The compilation threshold should be calibrated based on the average compilation time and the expected performance improvement from native code. If compilation takes 1000x longer than interpretation but native code runs 10x faster, the break-even point occurs at approximately 100 invocations.

The `JITCompiler` structure maintains the compilation trigger logic and integrates profiling data with compilation decisions. The compiler evaluates compilation triggers during function invocation, initiating compilation when thresholds are exceeded while ensuring that compilation overhead doesn't impact program execution.

| Field Name | Type | Description |
|------------|------|-------------|
| profiler | Profiler* | Profiling subsystem that tracks function execution frequency |
| cache | CodeCache* | Storage for compiled native code functions |
| compilation_enabled | int | Global flag enabling or disabling JIT compilation |
| default_threshold | uint32_t | Default compilation threshold for new functions |

**Dynamic Threshold Adjustment** allows the JIT compiler to adapt compilation triggers based on observed program behavior and compilation success rates. Functions that compile successfully and demonstrate significant performance improvements may have their thresholds lowered for similar functions, while functions that fail compilation or show minimal improvement may trigger threshold increases.

**Compilation Trigger Evaluation** occurs during the function call sequence, immediately after profiling counter updates. The `jit_compiler_record_call` method integrates profiling data collection with compilation trigger evaluation, ensuring that threshold decisions reflect the most current execution frequency data.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| jit_compiler_record_call | JITCompiler* compiler, uint32_t function_id, BytecodeFunction* function | int | Records function invocation and triggers compilation if threshold exceeded |
| jit_compiler_set_threshold | JITCompiler* compiler, uint32_t function_id, uint32_t threshold | int | Sets custom compilation threshold for specific function |
| jit_compiler_compile_function | JITCompiler* compiler, BytecodeFunction* function | int | Initiates compilation process for the specified function |
| jit_compiler_get_compiled_function | JITCompiler* compiler, uint32_t function_id | CompiledFunction | Retrieves native code function pointer if available |

The compilation trigger algorithm follows a systematic evaluation process that considers multiple factors beyond simple invocation counts. The algorithm evaluates compilation eligibility, resource availability, and compilation feasibility before initiating the expensive compilation process.

1. **Invocation Count Evaluation**: The algorithm first compares the function's current invocation count against its configured compilation threshold. Functions that haven't reached their threshold remain in interpreted execution mode.

2. **Compilation Status Check**: The algorithm verifies that the function hasn't already been compiled to avoid redundant compilation efforts. Previously compiled functions skip the compilation trigger logic entirely.

3. **Resource Availability Assessment**: The algorithm checks whether sufficient memory and computational resources are available for compilation. Resource-constrained environments may defer compilation even for hot functions.

4. **Function Complexity Analysis**: The algorithm evaluates whether the function's bytecode complexity justifies compilation overhead. Very simple functions may not benefit from native compilation despite high invocation counts.

5. **Compilation Queue Management**: The algorithm manages the compilation queue to prevent compilation backlog from impacting runtime performance. Functions exceeding thresholds enter a compilation queue for background processing.

> **Architecture Decision: Immediate vs. Deferred Compilation**
> - **Context**: Functions can be compiled immediately when thresholds are exceeded or queued for later compilation
> - **Options Considered**: 
>   - Immediate compilation: compile function synchronously when threshold exceeded
>   - Background compilation: queue function for asynchronous compilation
>   - Next-call compilation: trigger compilation but compile before next invocation
> - **Decision**: Next-call compilation with immediate fallback to interpretation
> - **Rationale**: Next-call compilation avoids blocking current function execution while ensuring compiled code is available for subsequent invocations. Background compilation introduces complexity for managing compilation state and code installation.
> - **Consequences**: First invocation after threshold may still use interpretation, but compilation overhead doesn't impact current execution path

| Compilation Strategy | Latency Impact | Implementation Complexity | Resource Usage | Compilation Timing |
|---------------------|----------------|---------------------------|----------------|-------------------|
| Immediate | High | Low | Bursty | Synchronous |
| Background | None | High | Smooth | Asynchronous |
| Next-call | Low | Medium | Moderate | Deferred |

**Compilation Failure Handling** addresses scenarios where JIT compilation cannot successfully translate a function to native code. Compilation failures may result from unsupported bytecode operations, resource exhaustion, or internal compiler errors. The trigger policy must gracefully handle these failures without disrupting program execution.

When compilation fails, the trigger policy typically implements a backoff strategy that increases the compilation threshold for the failed function, reducing the likelihood of repeated compilation attempts. This approach prevents compilation failures from creating performance bottlenecks while allowing future compilation attempts if execution patterns change significantly.

**Adaptive Threshold Management** enables the JIT compiler to learn from compilation successes and failures, adjusting thresholds based on observed program behavior. Functions that demonstrate significant performance improvements after compilation may influence threshold decisions for similar functions, while functions that show minimal improvement may trigger more conservative compilation policies.

![JIT Compilation Pipeline](./diagrams/compilation-pipeline.svg)

The compilation pipeline demonstrates the complete flow from hot path detection through native code generation and installation, showing how compilation triggers initiate the complex process of bytecode-to-native translation.

### Tiered Execution Strategy

The orchestration of multiple execution modes — interpretation, profiling, and native compilation — requires a sophisticated strategy for coordinating these different approaches while maintaining program correctness and optimal performance. Tiered execution strategy defines how the JIT compiler manages the transition between execution modes and coordinates the various subsystems involved in runtime optimization.

**Execution Mode Transitions** govern how functions move between different execution strategies based on their observed behavior and compilation status. The tiered execution strategy implements a state machine that tracks each function's current execution mode and manages transitions between modes based on profiling data and compilation results.

The function execution state machine defines five distinct states that capture the complete lifecycle of function execution optimization:

| Current State | Trigger Event | Next State | Actions Taken |
|--------------|---------------|------------|---------------|
| Cold Interpretation | First invocation | Warming Interpretation | Initialize profiling entry, begin counter tracking |
| Warming Interpretation | Invocation count < threshold | Warming Interpretation | Increment counter, continue interpretation |
| Warming Interpretation | Invocation count ≥ threshold | Compilation Triggered | Mark for compilation, continue interpretation |
| Compilation Triggered | Next invocation | Compiling | Begin JIT compilation process |
| Compiling | Compilation success | Native Execution | Install compiled code, switch to native calls |
| Compiling | Compilation failure | Warming Interpretation | Reset threshold with backoff, continue interpretation |
| Native Execution | All subsequent invocations | Native Execution | Execute compiled code directly |

**Mixed-Mode Execution Coordination** enables programs to seamlessly combine interpreted and native code execution within a single program run. The tiered execution strategy ensures that functions in different execution modes can call each other correctly, maintaining proper calling conventions and state consistency across mode transitions.

The coordination mechanism handles several critical scenarios that arise during mixed-mode execution:

**Interpreted-to-Native Calls** occur when an interpreted function calls a function that has been JIT-compiled to native code. The execution system must marshal the interpreter's virtual stack state into the native calling convention, ensuring that arguments are passed correctly and return values are handled properly.

**Native-to-Interpreted Calls** happen when JIT-compiled native code calls a function that remains in interpreted mode. The execution system must convert from native calling conventions back to the interpreter's virtual machine state, preserving register contents and stack alignment requirements.

**Recursive Mixed Calls** involve complex call chains that alternate between native and interpreted execution multiple times. The execution system must maintain a consistent call stack representation that supports proper unwinding and exception handling across execution mode boundaries.

> **Design Insight**: Mixed-mode execution requires trampolines — small bridge functions that handle the conversion between execution modes. These trampolines manage the state marshaling necessary to maintain correctness while adding minimal overhead to cross-mode function calls.

**Performance Monitoring and Optimization** tracks the effectiveness of compilation decisions and adjusts execution strategies based on observed performance improvements. The tiered execution strategy includes mechanisms for measuring the performance impact of JIT compilation and using this data to refine future compilation decisions.

The monitoring system tracks several key performance metrics:

| Metric | Purpose | Collection Method | Usage |
|--------|---------|------------------|-------|
| Compilation Time | Cost of JIT compilation | Timestamp before/after compilation | Threshold adjustment |
| Execution Speed Improvement | Benefit of native code | Compare interpreted vs native timing | ROI calculation |
| Memory Usage | Native code memory consumption | Track code buffer allocation | Resource management |
| Compilation Success Rate | Reliability of JIT compilation | Count successful vs failed compilations | Trigger policy tuning |

**Code Installation and Management** coordinates the process of installing newly compiled native code into the runtime system while maintaining execution correctness. The installation process must atomically replace interpreted execution with native execution, ensuring that concurrent function invocations don't observe inconsistent state.

The code installation algorithm follows a carefully orchestrated sequence that minimizes disruption to ongoing execution:

1. **Compilation Completion Verification**: The installation process first verifies that compilation completed successfully and produced valid native code that passes basic correctness checks.

2. **Code Buffer Finalization**: The newly generated native code undergoes memory protection changes, transitioning from writable during compilation to executable for runtime use.

3. **Function Pointer Installation**: The runtime system atomically updates the function dispatch table to redirect future invocations to the native code instead of the interpreter.

4. **Metadata Updates**: The profiling and execution tracking systems update their records to reflect the function's transition to native execution mode.

5. **Resource Cleanup**: The installation process releases any temporary resources used during compilation, including intermediate representation data structures and compilation context objects.

> **Architecture Decision: Atomic vs. Gradual Code Installation**
> - **Context**: Native code can be installed atomically or gradually with version management
> - **Options Considered**:
>   - Atomic installation: single operation switches all future calls to native code
>   - Versioned installation: maintain both versions and gradually migrate callers
>   - Copy-and-patch: modify existing code in place
> - **Decision**: Atomic installation with function pointer updates
> - **Rationale**: Atomic installation provides simplicity and correctness guarantees while avoiding the complexity of version management. Function pointer updates are atomic on x86-64, ensuring that concurrent threads observe consistent state.
> - **Consequences**: Simple implementation but requires careful coordination to prevent races between compilation and installation

| Installation Strategy | Correctness Guarantees | Implementation Complexity | Memory Usage | Concurrency Safety |
|-----------------------|------------------------|---------------------------|--------------|-------------------|
| Atomic Function Pointer | Strong | Low | Low | High |
| Version Management | Strong | High | High | High |
| Copy-and-Patch | Moderate | Very High | Low | Low |

**Deoptimization and Fallback Strategies** handle scenarios where native code execution encounters conditions that require falling back to interpretation. While less common than forward optimization, deoptimization ensures that the JIT compiler can gracefully handle unexpected runtime conditions without compromising program correctness.

Deoptimization scenarios include runtime type errors, unsupported operations that weren't anticipated during compilation, and resource exhaustion that prevents continued native execution. The tiered execution strategy includes mechanisms for detecting these conditions and safely transferring execution back to the interpreter.

The fallback process involves several critical steps:

1. **Deoptimization Trigger Detection**: The native code or runtime system detects a condition that requires falling back to interpretation, such as an unsupported operation or type mismatch.

2. **Execution State Reconstruction**: The deoptimization system reconstructs the equivalent interpreter state from the native execution context, including virtual stack contents and local variable values.

3. **Control Transfer**: The system transfers execution control back to the interpreter at the appropriate bytecode instruction, ensuring that program semantics remain consistent.

4. **Native Code Invalidation**: The deoptimization system may invalidate the native code to prevent future executions that could trigger the same deoptimization condition.

**Compilation Priority Management** determines the order in which multiple functions receive JIT compilation when system resources are limited. The priority system considers factors beyond simple invocation frequency, including function complexity, call graph position, and compilation success probability.

The priority algorithm evaluates several factors when ordering compilation candidates:

**Execution Frequency Weight** considers not just raw invocation counts but also recent execution patterns. Functions with consistently high invocation rates receive higher priority than functions with sporadic usage spikes.

**Compilation Cost Estimation** evaluates the expected compilation time based on function size and complexity. Functions with favorable cost-to-benefit ratios receive higher priority for compilation resources.

**Call Graph Analysis** considers the position of functions within the program's call graph. Functions that are called by many other functions or that serve as entry points to hot code paths receive elevated priority.

### Implementation Guidance

This implementation guidance provides concrete code structures and implementation strategies for building the hot path detection and profiling system. The profiling infrastructure must integrate seamlessly with the existing bytecode VM while imposing minimal runtime overhead.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Counter Storage | Fixed-size array indexed by function ID | Hash table with dynamic resizing |
| Threshold Management | Single global threshold | Per-function adaptive thresholds |
| Compilation Trigger | Synchronous check during function call | Asynchronous background compilation |
| Performance Monitoring | Simple timing measurements | Statistical profiling with confidence intervals |

**B. Recommended File/Module Structure:**

```
jit-compiler/
  src/profiler/
    profiler.h                 ← profiling data structures and API
    profiler.c                 ← execution counter implementation
    profiler_test.c           ← unit tests for profiling logic
  src/compiler/
    jit_compiler.h            ← main JIT compiler interface
    jit_compiler.c            ← compilation trigger and tiered execution
    hot_path_detector.h       ← hot path analysis algorithms
    hot_path_detector.c       ← threshold management and trigger policies
  src/vm/
    vm_integration.h          ← VM hooks for profiling integration
    vm_integration.c          ← function call instrumentation
  examples/
    profiling_demo.c          ← demonstration of hot path detection
  benchmarks/
    profiling_overhead.c      ← profiling overhead measurement
```

**C. Infrastructure Starter Code (COMPLETE, ready to use):**

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// Profiling data structures - complete implementation
typedef struct {
    uint32_t function_id;
    uint32_t invocation_count;
    uint32_t compilation_threshold;
    int is_compiled;
} ProfileEntry;

typedef struct {
    ProfileEntry* entries;
    size_t capacity;
    uint32_t default_threshold;
} Profiler;

// Create profiler with specified default compilation threshold
Profiler* profiler_create(uint32_t default_threshold) {
    Profiler* profiler = malloc(sizeof(Profiler));
    if (!profiler) return NULL;
    
    profiler->capacity = 1024; // Support up to 1024 functions initially
    profiler->entries = calloc(profiler->capacity, sizeof(ProfileEntry));
    profiler->default_threshold = default_threshold;
    
    if (!profiler->entries) {
        free(profiler);
        return NULL;
    }
    
    // Initialize all entries to invalid state
    for (size_t i = 0; i < profiler->capacity; i++) {
        profiler->entries[i].function_id = UINT32_MAX;
        profiler->entries[i].invocation_count = 0;
        profiler->entries[i].compilation_threshold = default_threshold;
        profiler->entries[i].is_compiled = 0;
    }
    
    return profiler;
}

// High-resolution timer for performance measurement
uint64_t get_time_nanoseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Performance measurement context
typedef struct {
    uint64_t compilation_start_time;
    uint64_t total_compilation_time;
    uint32_t successful_compilations;
    uint32_t failed_compilations;
    double average_speedup;
} PerformanceMonitor;

PerformanceMonitor* performance_monitor_create(void) {
    PerformanceMonitor* monitor = calloc(1, sizeof(PerformanceMonitor));
    return monitor;
}

void performance_monitor_start_compilation(PerformanceMonitor* monitor) {
    monitor->compilation_start_time = get_time_nanoseconds();
}

void performance_monitor_end_compilation(PerformanceMonitor* monitor, int success) {
    uint64_t end_time = get_time_nanoseconds();
    uint64_t compilation_time = end_time - monitor->compilation_start_time;
    
    monitor->total_compilation_time += compilation_time;
    if (success) {
        monitor->successful_compilations++;
    } else {
        monitor->failed_compilations++;
    }
}
```

**D. Core Logic Skeleton Code (signature + TODOs only):**

```c
/**
 * Record function invocation and check compilation threshold.
 * Returns 1 if compilation should be triggered, 0 otherwise.
 * This is the critical path function called on every function invocation.
 */
int profiler_record_invocation(Profiler* profiler, uint32_t function_id) {
    // TODO 1: Validate profiler and function_id parameters
    // TODO 2: Check if function_id exceeds profiler capacity - handle gracefully
    // TODO 3: Initialize profile entry if this is first invocation (function_id == UINT32_MAX)
    // TODO 4: Increment invocation_count atomically (consider thread safety)
    // TODO 5: Compare invocation_count against compilation_threshold
    // TODO 6: Return 1 if threshold exceeded and function not already compiled
    // TODO 7: Return 0 for all other cases (below threshold or already compiled)
    // Hint: This function must be extremely fast - avoid any expensive operations
}

/**
 * Main JIT compiler function invocation handler.
 * Integrates profiling with compilation triggering.
 */
int jit_compiler_record_call(JITCompiler* compiler, uint32_t function_id, 
                           BytecodeFunction* function) {
    // TODO 1: Call profiler_record_invocation to update counters and check threshold
    // TODO 2: If threshold not exceeded, return 0 (continue with interpretation)
    // TODO 3: If function already compiled, return compiled function pointer
    // TODO 4: Check compilation_enabled flag - respect global compilation disable
    // TODO 5: Verify function is suitable for compilation (not too simple/complex)
    // TODO 6: Trigger compilation process (may be immediate or deferred)
    // TODO 7: Update function's compilation status to prevent duplicate compilation
    // TODO 8: Return appropriate code indicating compilation triggered
    // Hint: Consider resource availability before triggering expensive compilation
}

/**
 * Adaptive threshold management based on compilation success/failure.
 * Adjusts compilation thresholds based on observed performance improvements.
 */
void adjust_compilation_threshold(JITCompiler* compiler, uint32_t function_id, 
                                int compilation_successful, double performance_improvement) {
    // TODO 1: Retrieve current ProfileEntry for the function
    // TODO 2: If compilation failed, increase threshold (backoff strategy)
    // TODO 3: If compilation succeeded but improvement < 2x, increase threshold slightly  
    // TODO 4: If compilation succeeded with good improvement, keep or lower threshold
    // TODO 5: Apply minimum and maximum threshold bounds (e.g., 10-10000)
    // TODO 6: Consider global compilation success rate for system-wide adjustments
    // TODO 7: Update ProfileEntry with new threshold value
    // Hint: Use exponential backoff for failures, linear adjustment for successes
}

/**
 * Tiered execution coordinator - manages transitions between execution modes.
 * Handles the complex state machine for function execution optimization.
 */
int coordinate_execution_mode(JITCompiler* compiler, uint32_t function_id, 
                            BytecodeFunction* function, void** execution_target) {
    // TODO 1: Check current execution state for the function
    // TODO 2: If in Cold/Warming Interpretation state, call profiler and continue interpreting
    // TODO 3: If compilation threshold reached, transition to Compilation Triggered state
    // TODO 4: If in Compiling state, continue interpretation but check compilation completion
    // TODO 5: If compilation completed successfully, transition to Native Execution
    // TODO 6: If compilation failed, transition back to Warming with adjusted threshold
    // TODO 7: Set *execution_target to appropriate function pointer (interpreter or native)
    // TODO 8: Return execution mode code for caller's execution dispatch
    // Hint: This function implements the state machine transitions from the design
}

/**
 * Code installation manager - atomically installs compiled native code.
 * Ensures thread-safe transition from interpreted to native execution.
 */
int install_compiled_function(JITCompiler* compiler, uint32_t function_id, 
                            CompiledFunction compiled_function, size_t code_size) {
    // TODO 1: Validate compiled_function pointer is not NULL
    // TODO 2: Make code buffer executable using make_memory_executable
    // TODO 3: Perform basic sanity checks on generated code (entry point, alignment)
    // TODO 4: Update function dispatch table atomically (critical section)
    // TODO 5: Update ProfileEntry to mark function as compiled (is_compiled = 1)
    // TODO 6: Update performance monitoring statistics
    // TODO 7: Release any compilation-time resources (TranslationContext, etc.)
    // TODO 8: Return success/failure status
    // Hint: Use atomic operations or locks to ensure thread safety during installation
}
```

**E. Language-Specific Hints:**

- **Function ID Management**: Use a simple counter starting from 0 for function IDs. The VM should assign IDs during bytecode loading to ensure sequential, dense numbering for array indexing.

- **Atomic Counter Updates**: Use `__sync_fetch_and_add(&entry->invocation_count, 1)` for thread-safe counter increments. This GCC builtin provides atomic increment without explicit locking.

- **Memory Protection**: Use `mprotect(code_buffer, size, PROT_READ | PROT_EXEC)` to make compiled code executable. Call this after code generation completes but before installing function pointers.

- **Threshold Calibration**: Start with default thresholds around 100-1000 invocations. Measure compilation time vs execution improvement to calibrate optimal thresholds for your workload.

- **Compiler Integration**: Hook profiling into the VM's main function dispatch mechanism. Add the profiling call immediately before branching to interpretation or native execution.

**F. Milestone Checkpoint:**

After implementing hot path detection:

- **Test Command**: `gcc -o profiler_test profiler.c profiler_test.c && ./profiler_test`
- **Expected Output**: All profiling tests pass, showing correct counter incrementation and threshold detection
- **Manual Verification**: Run a simple loop-heavy bytecode program and observe functions transitioning from interpretation to native execution after reaching thresholds
- **Performance Check**: Profiling overhead should be < 5% of interpretation time. Use the profiling_overhead benchmark to measure this.
- **Signs of Problems**: If compilation never triggers, check threshold values and counter incrementation. If performance degrades significantly, check for expensive operations in the profiling hot path.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Functions never get compiled despite high invocation counts | Profiling counters not incrementing correctly | Add debug prints in `profiler_record_invocation` | Verify VM integration hooks profiling before every function call |
| Compilation triggers but performance doesn't improve | Generated native code is incorrect or slow | Compare native vs interpreted output, measure execution time | Check bytecode translation correctness, verify register allocation |
| High profiling overhead impacts performance | Expensive operations in profiling hot path | Profile the profiler - measure time spent in `profiler_record_invocation` | Remove expensive operations, use simpler data structures |
| Compilation never completes or crashes | Compilation process has bugs or resource issues | Enable compilation debugging, check memory usage during compilation | Review compilation pipeline, add error handling and resource cleanup |
| Mixed interpreted/native execution produces wrong results | State marshaling between execution modes is incorrect | Test simple functions that call between modes, verify argument passing | Review trampoline implementation, check calling convention compliance |


## Runtime Optimization

> **Milestone(s):** Milestone 4 (Hot Path Detection and Optimization) — this section covers the optimization passes that transform raw bytecode-to-native translations into efficient machine code, demonstrating how runtime information enables optimizations impossible at compile time

The true power of Just-In-Time compilation emerges not merely from translating bytecode to native instructions, but from the sophisticated optimizations enabled by runtime information. Unlike ahead-of-time compilers that must make conservative assumptions about program behavior, JIT compilers observe actual execution patterns and can aggressively optimize based on concrete runtime data. This section explores how our JIT compiler transforms basic bytecode-to-native translations into highly optimized machine code through constant folding, dead code elimination, and runtime specialization techniques.

### Mental Model: Code Refinement Workshop

Think of JIT optimization as a master craftsman's workshop where raw materials undergo progressive refinement. When a blacksmith forges a blade, they begin with rough iron ore, gradually heating, hammering, and folding the metal to remove impurities and align the grain structure. Each refinement pass makes the metal stronger and more precise. Similarly, our JIT optimizer takes the raw "metal" of direct bytecode translation and applies successive refinement passes—constant folding removes computational "impurities," dead code elimination strips away unused material, and runtime specialization aligns the code structure with observed execution patterns.

Just as the blacksmith uses heat and pressure unavailable to the original miner, the JIT optimizer wields runtime information unavailable to static compilers. The blacksmith knows the blade's intended use—cutting, thrusting, or ceremonial display—and shapes the metal accordingly. Our JIT optimizer observes how functions actually execute—which branches are taken, which values remain constant, which code paths are never traversed—and sculpts the machine code to match these observed patterns. The result is not merely translated bytecode, but refined, specialized native code optimized for the specific execution context.

### Constant Folding and Propagation

**Constant folding** represents the most fundamental optimization in our JIT compiler's arsenal, evaluating constant expressions at compile time rather than repeatedly computing the same values during execution. Unlike static compilers that must conservatively assume variables might change, our JIT optimizer observes actual runtime values and can aggressively fold expressions when operands remain constant across multiple invocations.

The optimization process begins during the bytecode analysis phase, where our compiler tracks the flow of constant values through the virtual machine's stack operations. When the translator encounters arithmetic bytecodes with constant operands, it performs the computation immediately and emits a single `MOV` instruction with the computed result, rather than generating separate instructions for loading operands and performing the operation.

Consider a bytecode sequence that computes `(5 + 3) * 2 - 1`. The naive translation would generate:

| Bytecode Operation | Generated Instructions | Description |
|-------------------|----------------------|-------------|
| `LOAD_CONST 5` | `mov rax, 5` | Load first constant |
| `LOAD_CONST 3` | `mov rbx, 3` | Load second constant |
| `ADD` | `add rax, rbx` | Perform addition |
| `LOAD_CONST 2` | `mov rcx, 2` | Load multiplication factor |
| `MUL` | `imul rax, rcx` | Perform multiplication |
| `LOAD_CONST 1` | `mov rdx, 1` | Load subtraction operand |
| `SUB` | `sub rax, rdx` | Perform subtraction |

Our constant folding optimization recognizes that all operands are compile-time constants, evaluates the entire expression `(5 + 3) * 2 - 1 = 15`, and emits a single instruction: `mov rax, 15`. This transformation eliminates six instructions and three arithmetic operations, reducing both code size and execution time.

**Constant propagation** extends this concept by tracking constant values across multiple bytecode operations and through function boundaries. When a variable is assigned a constant value and never modified within the observed execution paths, the optimizer substitutes the constant value at all use sites. This creates opportunities for additional constant folding in downstream operations.

> **Decision: Interprocedural vs. Intraprocedural Constant Propagation**
> - **Context**: Constant propagation can operate within single functions (intraprocedural) or across function calls (interprocedural), with different complexity and benefit trade-offs
> - **Options Considered**: 
>   1. Intraprocedural only - simpler implementation, limited optimization scope
>   2. Full interprocedural - maximum optimization potential, significant implementation complexity
>   3. Limited interprocedural - track constants through direct function calls only
> - **Decision**: Intraprocedural constant propagation with limited interprocedural analysis for leaf functions
> - **Rationale**: Provides substantial optimization benefits for mathematical computations while keeping implementation complexity manageable. Most performance-critical loops contain arithmetic operations within single functions where intraprocedural analysis suffices
> - **Consequences**: Enables aggressive optimization of computational kernels while maintaining reasonable compilation speed and implementation complexity

The constant propagation algorithm maintains a **constant value table** during bytecode analysis, tracking which virtual stack positions and local variables hold known constant values. When the translator encounters operations that modify these tracked values, it updates the constant table accordingly. Operations that could potentially modify values in unknown ways (such as function calls or array assignments) conservatively invalidate the constant assumptions for affected variables.

| Optimization Component | Data Structure | Fields | Purpose |
|----------------------|----------------|--------|---------|
| `ConstantTable` | Hash table | `entries ConstantEntry*`, `capacity size_t`, `count size_t` | Maps variables to constant values |
| `ConstantEntry` | Value record | `variable_id uint32_t`, `constant_value int64_t`, `is_valid int`, `last_assignment_offset uint32_t` | Tracks constant value for specific variable |
| `PropagationContext` | Analysis state | `table ConstantTable*`, `invalidation_points uint32_t*`, `point_count size_t` | Manages constant propagation during compilation |

The optimization pass processes bytecode instructions sequentially, maintaining constant value assumptions and applying folding transformations when opportunities arise:

1. **Initialize constant table** with parameter values marked as unknown (since they vary across function calls)
2. **Process each bytecode instruction** in execution order, updating constant table with assignment operations
3. **Check for constant operands** before emitting arithmetic operations, applying folding when all operands are known constants
4. **Invalidate constant assumptions** when encountering operations that could modify variables in unknown ways
5. **Propagate constants forward** by substituting known values at variable use sites
6. **Generate optimized instruction sequence** using folded constants and propagated values

⚠️ **Pitfall: Incorrect Constant Invalidation**
A common error involves failing to invalidate constant assumptions when variables might be modified through indirect means. For example, if a function takes a pointer parameter and modifies the pointed-to value, all variables that might alias that memory location must be marked as non-constant. The optimizer must conservatively invalidate constant assumptions for any operation that could potentially modify memory or variables in ways not directly visible in the bytecode sequence.

**Runtime-observed constants** represent a more aggressive optimization opportunity unique to JIT compilation. By monitoring actual execution, the optimizer can identify variables that remain constant across multiple function invocations, even if they are not compile-time constants in the traditional sense. For example, a configuration parameter loaded from a file might remain unchanged throughout the program's execution, enabling the optimizer to treat it as a constant for optimization purposes.

### Dead Code Elimination

**Dead code elimination** removes unreachable or unused code sequences from the generated machine code, reducing both memory footprint and execution time. In the context of JIT compilation, dead code arises from several sources: unreachable bytecode following unconditional jumps, operations whose results are never used, and conditional branches that are never taken based on observed runtime behavior.

The elimination process operates through multiple analysis passes that identify different categories of dead code:

| Dead Code Category | Detection Method | Elimination Strategy | Runtime Benefit |
|-------------------|------------------|---------------------|-----------------|
| Unreachable instructions | Control flow analysis | Skip instruction translation | Reduces code size |
| Unused computation results | Liveness analysis | Eliminate result generation | Reduces execution time |
| Never-taken branches | Profile-guided analysis | Remove branch instruction | Improves branch prediction |
| Dead stores | Data flow analysis | Skip memory write operations | Reduces memory traffic |

**Control flow analysis** constructs a **control flow graph** representing all possible execution paths through the bytecode function. Instructions that are not reachable through any valid execution path are marked for elimination. This commonly occurs after unconditional jumps, return statements, or in conditional branches that handle error conditions never observed during profiling.

The control flow analysis algorithm processes bytecode instructions to build a complete graph of execution paths:

1. **Identify basic blocks** by finding instruction sequences with single entry and exit points
2. **Construct control flow edges** connecting basic blocks based on jump instructions and conditional branches
3. **Perform reachability analysis** starting from the function entry point, marking all reachable basic blocks
4. **Mark unreachable blocks** for elimination, ensuring their instructions are not translated to native code
5. **Update jump targets** to account for eliminated blocks, ensuring remaining control flow remains correct
6. **Compact instruction sequence** to remove gaps left by eliminated blocks

**Liveness analysis** determines which computed values are actually used by subsequent operations, enabling elimination of computations whose results are never consumed. This analysis tracks the **live ranges** of virtual stack positions and local variables, identifying operations that produce values never read by downstream instructions.

| Analysis Phase | Data Structure | Purpose | Algorithm |
|---------------|----------------|---------|-----------|
| `LivenessAnalysis` | `live_in BitSet*`, `live_out BitSet*`, `basic_blocks BasicBlock*` | Track variable liveness | Backward dataflow analysis |
| `UseDefChains` | `definitions UseDefEntry*`, `uses UseDefEntry*` | Map variable definitions to uses | Forward scanning with def-use linking |
| `DeadCodeMarker` | `eliminated_instructions BitSet*`, `elimination_count size_t` | Track eliminated operations | Mark-and-sweep elimination |

The liveness analysis employs a **backward dataflow analysis** algorithm that propagates liveness information from variable uses back to their definitions:

1. **Initialize liveness sets** for each basic block, marking function parameters and return values as live
2. **Process basic blocks in reverse order**, propagating liveness information backward through the control flow graph
3. **Compute live-in and live-out sets** for each block based on variable uses and definitions within the block
4. **Iterate until convergence** as liveness information propagates throughout the function
5. **Mark dead operations** whose results are not live at any subsequent program point
6. **Eliminate unused computations** by skipping instruction translation for marked operations

**Profile-guided dead code elimination** leverages runtime execution statistics to identify code paths that are never executed in practice. Unlike static analysis that must conservatively assume all syntactically reachable code might execute, profile-guided optimization can eliminate branches and exception handlers that are never taken during observed execution.

> **Decision: Conservative vs. Aggressive Dead Code Elimination**
> - **Context**: Dead code elimination can be conservative (removing only provably dead code) or aggressive (removing code that appears unused based on profiling data)
> - **Options Considered**:
>   1. Conservative elimination - only remove statically proven dead code
>   2. Aggressive elimination - remove code unused during profiling period
>   3. Adaptive elimination - start conservative, become more aggressive with longer profiling
> - **Decision**: Adaptive elimination with configurable aggressiveness levels
> - **Rationale**: Balances optimization benefits with correctness concerns. Conservative elimination handles safety-critical paths, while aggressive elimination optimizes hot paths with sufficient profiling data
> - **Consequences**: Requires sophisticated profiling infrastructure and deoptimization support, but enables significant performance improvements for specialized execution patterns

The profile-guided elimination process maintains **execution frequency counters** for each basic block and conditional branch, identifying code regions that are never executed during the profiling period:

1. **Instrument basic blocks** with lightweight execution counters during initial bytecode interpretation
2. **Collect execution statistics** over multiple function invocations to establish reliable frequency data
3. **Identify cold code regions** with execution frequencies below configured thresholds
4. **Analyze code dependencies** to ensure eliminated regions do not affect frequently executed paths
5. **Generate specialized code** that omits cold regions while maintaining correctness for hot paths
6. **Install deoptimization guards** to handle cases where eliminated code would need to execute

### Runtime Specialization

**Runtime specialization** represents the most sophisticated optimization category in our JIT compiler, enabling code generation tailored to specific execution contexts observed during runtime profiling. Unlike static optimizations that must handle all possible inputs correctly, runtime specialization generates code optimized for the common case while providing fallback mechanisms for uncommon scenarios.

The specialization process analyzes runtime behavior patterns to identify optimization opportunities invisible to static analysis:

| Specialization Type | Observation Method | Optimization Strategy | Fallback Mechanism |
|--------------------|-------------------|----------------------|-------------------|
| Type specialization | Value type profiling | Generate type-specific code paths | Runtime type check with deoptimization |
| Branch specialization | Branch frequency profiling | Optimize for frequently taken direction | Maintain both branch paths |
| Loop specialization | Iteration count profiling | Unroll loops with known iteration bounds | General loop for variable bounds |
| Call site specialization | Target function profiling | Inline frequently called functions | Standard call mechanism |

**Type specialization** optimizes operations based on the observed types of operands during runtime execution. In a dynamically typed bytecode VM, arithmetic operations must handle multiple data types (integers, floating-point numbers, strings), requiring runtime type checking and dispatch. When profiling reveals that specific operations consistently receive the same types, the optimizer can generate specialized code paths that assume those types, eliminating runtime type checks for the common case.

The type specialization algorithm maintains **type profile data** for each operation, tracking the frequency distribution of operand types across multiple executions:

1. **Instrument type-sensitive operations** with lightweight profiling code that records operand types
2. **Accumulate type frequency statistics** over the profiling period to identify dominant type patterns
3. **Generate specialized code paths** for type combinations that exceed frequency thresholds
4. **Insert runtime type guards** that verify type assumptions and redirect to specialized code when valid
5. **Provide generic fallback paths** for type combinations not covered by specialized versions
6. **Update specialization decisions** based on ongoing runtime feedback to adapt to changing type patterns

| Specialization Component | Data Structure | Fields | Purpose |
|------------------------|----------------|--------|---------|
| `TypeProfile` | Frequency tracker | `operation_id uint32_t`, `type_frequencies uint32_t[]`, `total_samples uint32_t` | Tracks type distribution for operations |
| `SpecializationDecision` | Optimization choice | `specialized_types int[]`, `guard_instructions uint32_t[]`, `fallback_offset uint32_t` | Encodes specialization strategy |
| `GuardInstructions` | Runtime checks | `type_check_code uint8_t*`, `deopt_target uint32_t`, `guard_count int` | Implements runtime type verification |

**Branch specialization** optimizes conditional jumps based on the observed frequency of branch directions during runtime execution. When profiling reveals that a conditional branch is taken in the same direction 90% of the time, the optimizer can restructure the generated code to optimize for the common case while ensuring the uncommon case remains correct.

The branch specialization optimization employs several techniques to improve performance of predictable branches:

1. **Branch reordering** places the frequently taken path immediately after the conditional instruction, improving instruction cache locality
2. **Condition inversion** modifies the branch condition to test for the uncommon case, falling through to the common case
3. **Code duplication** creates specialized versions of code following frequently taken branches, enabling additional optimizations
4. **Predication** converts simple conditional operations to predicated instructions on architectures that support them

**Loop specialization** optimizes iterative constructs based on observed iteration patterns and loop-carried dependencies. When profiling reveals that loops typically execute a specific number of iterations or access data in predictable patterns, the optimizer can apply transformations like loop unrolling, vectorization, or strength reduction.

> **Decision: Loop Unrolling Strategy**
> - **Context**: Loop unrolling can improve performance by reducing branch overhead and enabling instruction-level parallelism, but increases code size and compile time
> - **Options Considered**:
>   1. Fixed unrolling factor - simple implementation, may not match actual iteration counts
>   2. Profile-guided unrolling - unroll based on observed iteration patterns
>   3. Dynamic unrolling - generate multiple unrolled versions for different iteration counts
> - **Decision**: Profile-guided unrolling with dynamic factor selection
> - **Rationale**: Maximizes performance benefits by matching unrolling factor to actual execution patterns while avoiding excessive code bloat from over-unrolling
> - **Consequences**: Requires sophisticated profiling and code generation capabilities, but provides optimal performance for loop-intensive code

The loop specialization process analyzes loop execution characteristics to determine appropriate optimization strategies:

1. **Profile loop iteration counts** to identify loops with predictable iteration patterns
2. **Analyze loop-carried dependencies** to determine which transformations preserve correctness
3. **Estimate performance impact** of different optimization strategies based on loop characteristics
4. **Generate specialized loop versions** optimized for common iteration patterns
5. **Insert runtime checks** to select appropriate loop version based on actual iteration count
6. **Apply secondary optimizations** enabled by loop transformations, such as strength reduction or vectorization

The runtime specialization infrastructure requires sophisticated **deoptimization support** to handle cases where specialized assumptions become invalid. When runtime conditions change—such as type patterns shifting or branch frequencies diverging from profiled behavior—the system must safely transition back to generic code that handles all cases correctly.

![JIT Compilation Pipeline](./diagrams/compilation-pipeline.svg)

**Call site specialization** optimizes function calls based on observed call targets and argument patterns. In dynamically dispatched systems, function calls may involve virtual dispatch or indirect calls through function pointers. When profiling reveals that specific call sites consistently invoke the same target functions, the optimizer can generate direct calls or even inline the called functions entirely.

The call site specialization algorithm tracks call target frequency and argument characteristics:

1. **Monitor call site behavior** to identify frequently called functions and common argument patterns
2. **Analyze callee functions** to determine inlining profitability based on size and optimization potential
3. **Generate specialized call sequences** using direct calls or inlined code for common targets
4. **Insert call target guards** that verify the called function matches specialized assumptions
5. **Maintain polymorphic call stubs** for handling uncommon call targets not covered by specialization
6. **Update specialization decisions** based on ongoing call pattern observations

⚠️ **Pitfall: Over-Specialization**
Aggressive runtime specialization can lead to code bloat and reduced performance if specialization decisions are based on insufficient profiling data or unstable runtime patterns. Each specialized code path increases memory consumption and compilation time while potentially reducing instruction cache effectiveness. The optimization system must balance specialization benefits against these costs, using statistical confidence measures to ensure specialization decisions are based on stable, representative runtime behavior.

The integration of all optimization passes requires careful coordination to maximize their combined effectiveness. Constant folding creates opportunities for dead code elimination by simplifying control flow, while dead code elimination enables more aggressive constant propagation by reducing the number of variables that must be tracked. Runtime specialization amplifies the benefits of both optimizations by creating contexts where more aggressive assumptions can be made safely.

### Implementation Guidance

The runtime optimization system requires sophisticated analysis and transformation capabilities that build upon the basic bytecode-to-native translation infrastructure. This implementation guidance provides the necessary components for building a complete optimization pipeline.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| Constant Analysis | Hash table with linear probing | SSA-based value numbering |
| Dead Code Detection | Basic block reachability | Full dataflow analysis framework |
| Profile Storage | Simple counters in memory | Persistent profiling database |
| Deoptimization | Function-level recompilation | On-stack replacement (OSR) |
| Optimization Pipeline | Sequential pass execution | Iterative worklist algorithm |

**B. Recommended File/Module Structure:**

```
jit-compiler/
  src/
    optimization/
      optimizer.c               ← main optimization coordinator
      optimizer.h               ← optimization pipeline interface
      constant_folding.c        ← constant analysis and folding
      constant_folding.h        ← constant propagation interface
      dead_code.c              ← dead code elimination passes
      dead_code.h              ← liveness analysis interface
      runtime_specialization.c ← profile-guided optimizations
      runtime_specialization.h ← specialization interface
      profile_data.c           ← execution profiling infrastructure
      profile_data.h           ← profiling data structures
      deoptimization.c         ← fallback and guard handling
      deoptimization.h         ← deoptimization interface
    analysis/
      control_flow.c           ← control flow graph construction
      control_flow.h           ← CFG analysis interface
      dataflow.c               ← dataflow analysis framework
      dataflow.h               ← dataflow algorithms
      liveness.c               ← variable liveness analysis
      liveness.h               ← liveness computation interface
    test/
      optimization_test.c      ← optimization correctness tests
      benchmark.c              ← performance measurement suite
```

**C. Infrastructure Starter Code:**

```c
// profile_data.c - Complete profiling infrastructure
#include "profile_data.h"
#include <stdlib.h>
#include <string.h>

ProfileData* profile_data_create(uint32_t function_count) {
    ProfileData* profile = malloc(sizeof(ProfileData));
    profile->function_profiles = calloc(function_count, sizeof(FunctionProfile));
    profile->function_count = function_count;
    profile->total_samples = 0;
    profile->sampling_enabled = 1;
    return profile;
}

void profile_data_record_type(ProfileData* profile, uint32_t operation_id, 
                             int operand_type) {
    if (!profile->sampling_enabled) return;
    
    TypeProfile* type_profile = &profile->type_profiles[operation_id];
    if (operand_type >= 0 && operand_type < MAX_TYPE_COUNT) {
        type_profile->type_frequencies[operand_type]++;
        type_profile->total_samples++;
        profile->total_samples++;
    }
}

void profile_data_record_branch(ProfileData* profile, uint32_t branch_id, 
                               int taken) {
    BranchProfile* branch_profile = &profile->branch_profiles[branch_id];
    if (taken) {
        branch_profile->taken_count++;
    } else {
        branch_profile->not_taken_count++;
    }
    branch_profile->total_count++;
}

int profile_data_should_specialize(ProfileData* profile, uint32_t operation_id, 
                                  double confidence_threshold) {
    TypeProfile* type_profile = &profile->type_profiles[operation_id];
    if (type_profile->total_samples < MIN_SAMPLES_FOR_SPECIALIZATION) {
        return 0;
    }
    
    uint32_t max_frequency = 0;
    for (int i = 0; i < MAX_TYPE_COUNT; i++) {
        if (type_profile->type_frequencies[i] > max_frequency) {
            max_frequency = type_profile->type_frequencies[i];
        }
    }
    
    double confidence = (double)max_frequency / type_profile->total_samples;
    return confidence >= confidence_threshold;
}

// control_flow.c - Complete control flow analysis
#include "control_flow.h"
#include <stdlib.h>

ControlFlowGraph* cfg_create(BytecodeFunction* function) {
    ControlFlowGraph* cfg = malloc(sizeof(ControlFlowGraph));
    cfg->basic_blocks = malloc(sizeof(BasicBlock) * MAX_BASIC_BLOCKS);
    cfg->block_count = 0;
    cfg->function = function;
    
    // Build basic blocks by scanning for leaders (jump targets and jump sources)
    uint8_t* leaders = calloc(function->bytecode_length, 1);
    leaders[0] = 1; // Function entry is always a leader
    
    for (uint32_t i = 0; i < function->bytecode_length; ) {
        Opcode op = function->bytecode[i];
        switch (op) {
            case OP_JUMP:
            case OP_JUMP_IF_FALSE: {
                uint32_t target = *(uint32_t*)&function->bytecode[i + 1];
                leaders[target] = 1; // Jump target is a leader
                if (i + 5 < function->bytecode_length) {
                    leaders[i + 5] = 1; // Instruction after jump is a leader
                }
                i += 5;
                break;
            }
            default:
                i += bytecode_instruction_length(op);
                break;
        }
    }
    
    // Create basic blocks from leaders
    for (uint32_t i = 0; i < function->bytecode_length; i++) {
        if (leaders[i] && cfg->block_count < MAX_BASIC_BLOCKS) {
            BasicBlock* block = &cfg->basic_blocks[cfg->block_count++];
            block->start_offset = i;
            block->is_reachable = (i == 0); // Only entry block initially reachable
            block->predecessor_count = 0;
            block->successor_count = 0;
        }
    }
    
    free(leaders);
    return cfg;
}

void cfg_compute_reachability(ControlFlowGraph* cfg) {
    int changed = 1;
    while (changed) {
        changed = 0;
        for (uint32_t i = 0; i < cfg->block_count; i++) {
            BasicBlock* block = &cfg->basic_blocks[i];
            if (!block->is_reachable) continue;
            
            for (int j = 0; j < block->successor_count; j++) {
                BasicBlock* successor = block->successors[j];
                if (!successor->is_reachable) {
                    successor->is_reachable = 1;
                    changed = 1;
                }
            }
        }
    }
}
```

**D. Core Logic Skeleton Code:**

```c
// optimizer.h - Main optimization interface
typedef struct OptimizationContext {
    BytecodeFunction* function;
    ProfileData* profile;
    TranslationContext* translation;
    ConstantTable* constants;
    ControlFlowGraph* cfg;
    LivenessAnalysis* liveness;
    int optimization_level;
    double specialization_threshold;
} OptimizationContext;

// Creates optimization context for function compilation
OptimizationContext* optimization_context_create(BytecodeFunction* function, 
                                                 ProfileData* profile);

// Main optimization pipeline entry point
int optimize_function(OptimizationContext* ctx) {
    // TODO 1: Run constant folding pass to evaluate compile-time expressions
    // TODO 2: Perform dead code elimination based on reachability analysis
    // TODO 3: Apply runtime specialization based on profile data
    // TODO 4: Coordinate optimization passes for maximum effectiveness
    // TODO 5: Validate optimization correctness and update translation context
    // Hint: Order matters - constant folding enables dead code elimination
}

// constant_folding.c - Constant propagation and folding
typedef struct ConstantFoldingPass {
    ConstantTable* constant_table;
    BytecodeFunction* function;
    int* folded_operations;
    uint32_t fold_count;
} ConstantFoldingPass;

int constant_folding_run(OptimizationContext* ctx) {
    // TODO 1: Initialize constant table with function parameters marked unknown
    // TODO 2: Process bytecode instructions in execution order
    // TODO 3: Track constant assignments and propagate values forward
    // TODO 4: Identify arithmetic operations with all constant operands
    // TODO 5: Replace constant expressions with computed results
    // TODO 6: Update translation context with folded instruction mapping
    // TODO 7: Invalidate constants affected by memory operations or calls
    // Hint: Use dataflow analysis to track constants across basic blocks
}

int evaluate_constant_expression(Opcode operation, int64_t left_operand, 
                                int64_t right_operand, int64_t* result) {
    // TODO 1: Handle arithmetic operations (ADD, SUB, MUL, DIV)
    // TODO 2: Handle comparison operations (EQ, NE, LT, LE, GT, GE)  
    // TODO 3: Handle bitwise operations (AND, OR, XOR, SHL, SHR)
    // TODO 4: Check for overflow conditions and division by zero
    // TODO 5: Return success/failure indication for valid operations
    // Hint: Use checked arithmetic to detect overflow conditions
}

// dead_code.c - Dead code elimination
typedef struct DeadCodePass {
    ControlFlowGraph* cfg;
    LivenessAnalysis* liveness;
    BitSet* eliminated_instructions;
    uint32_t elimination_count;
} DeadCodePass;

int dead_code_elimination_run(OptimizationContext* ctx) {
    // TODO 1: Build control flow graph for reachability analysis
    // TODO 2: Compute liveness information for all variables
    // TODO 3: Mark unreachable basic blocks for elimination
    // TODO 4: Mark operations with unused results for elimination
    // TODO 5: Mark never-taken branches based on profile data
    // TODO 6: Update translation context to skip eliminated operations
    // TODO 7: Compact remaining instructions to remove gaps
    // Hint: Use backward dataflow analysis for liveness computation
}

int compute_variable_liveness(OptimizationContext* ctx, LivenessAnalysis* analysis) {
    // TODO 1: Initialize live-in and live-out sets for each basic block
    // TODO 2: Process basic blocks in reverse topological order
    // TODO 3: Compute use and definition sets for each block
    // TODO 4: Propagate liveness information backward through CFG
    // TODO 5: Iterate until liveness sets reach fixed point
    // TODO 6: Mark operations whose results are never live
    // Hint: Function parameters and return values are always live
}

// runtime_specialization.c - Profile-guided optimization
typedef struct RuntimeSpecialization {
    ProfileData* profile;
    SpecializationDecision* decisions;
    uint32_t decision_count;
    double confidence_threshold;
} RuntimeSpecialization;

int runtime_specialization_run(OptimizationContext* ctx) {
    // TODO 1: Analyze profile data for specialization opportunities
    // TODO 2: Identify operations with dominant type patterns
    // TODO 3: Find branches with predictable taken/not-taken patterns
    // TODO 4: Generate specialized code paths for common cases
    // TODO 5: Insert runtime guards to verify specialization assumptions
    // TODO 6: Create fallback paths for uncommon cases
    // TODO 7: Update translation context with specialization decisions
    // Hint: Use statistical confidence measures to validate specialization
}

int generate_type_specialized_operation(OptimizationContext* ctx, 
                                       uint32_t operation_id, int specialized_type) {
    // TODO 1: Generate type-specific instruction sequence
    // TODO 2: Emit runtime type check guard instructions  
    // TODO 3: Create fallback jump to generic operation handler
    // TODO 4: Record guard locations for deoptimization support
    // TODO 5: Update register allocation for specialized operands
    // Hint: Type guards should be lightweight and rarely taken
}
```

**E. Language-Specific Hints:**

- **Memory Management**: Use `malloc`/`free` for optimization data structures, but consider memory pools for frequently allocated/deallocated objects during compilation
- **Bit Operations**: Use bitwise operations for efficient set operations in liveness analysis (`|` for union, `&` for intersection)
- **Profile Data**: Store profiling counters in cache-aligned structures to avoid false sharing in multi-threaded environments
- **Floating Point**: Use `fenv.h` functions to handle floating-point exceptions during constant folding of floating-point operations
- **Platform Specifics**: Conditional compilation (`#ifdef`) for architecture-specific optimizations like vectorization or predicated instructions

**F. Milestone Checkpoint:**

After implementing the optimization pipeline:

1. **Correctness Verification**: Run `./test_optimizer --verify-correctness` to ensure optimized code produces identical results to unoptimized translation
2. **Performance Measurement**: Execute `./benchmark_optimizer --compare-speedup` to measure optimization effectiveness on computational workloads  
3. **Profile Analysis**: Use `./profile_analyzer --dump-statistics` to verify profiling data collection and specialization trigger logic
4. **Expected Behavior**: 
   - Constant folding should eliminate 60-80% of compile-time computable expressions
   - Dead code elimination should remove 10-30% of unreachable code in typical programs
   - Runtime specialization should trigger for operations with >90% type consistency
   - Overall speedup should be 2-5x for mathematical computations compared to unoptimized code

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|----------------|-----|
| Optimized code produces wrong results | Incorrect constant folding | Compare optimized vs unoptimized instruction sequences | Add overflow checking and validation to constant evaluation |
| Performance regression after optimization | Over-aggressive specialization | Profile compilation time and code size growth | Increase confidence thresholds and add specialization cost model |
| Segmentation fault in optimized code | Dead code eliminated incorrectly | Check liveness analysis for false negatives | Fix dataflow analysis to handle indirect variable access |
| Optimization pass never triggers | Insufficient profiling data | Examine profile data collection and threshold settings | Lower compilation thresholds or extend profiling period |


## Interactions and Data Flow

> **Milestone(s):** All milestones — understanding data flow and component interactions is essential throughout the entire JIT compiler implementation, from basic machine code emission through advanced optimization

The complete execution of a JIT compiler represents a sophisticated dance between multiple subsystems, each with distinct responsibilities yet requiring seamless coordination. Understanding this choreography is crucial for both implementing the system correctly and debugging it effectively when things go wrong. The flow of data and control through the system involves multiple state transitions, format conversions, and execution mode switches that must work together to deliver the promise of JIT compilation: transparent acceleration of hot code paths.

### Mental Model: Orchestra Performance

Think of JIT compilation like a symphony orchestra where multiple musicians (components) must coordinate perfectly to create a harmonious performance. The conductor (profiler) watches the performance and decides when to bring in new instruments (JIT compilation) to enhance frequently played themes (hot paths). The sheet music (bytecode) gets translated into individual instrument parts (native code) by the arranger (translator), while the stage manager (memory manager) ensures everyone has the right space and equipment. Each musician knows their part, but the magic happens in their coordination.

Just as an orchestra seamlessly transitions between movements without the audience noticing the complexity behind the scenes, our JIT compiler must transparently switch between interpreted and native execution while maintaining perfect program semantics. The audience (user program) should only notice improved performance, never the intricate coordination happening backstage.

### Complete Execution Flow

The journey from bytecode interpretation to native execution follows a carefully orchestrated sequence of state transitions and component interactions. This flow represents the heart of tiered execution strategy, where programs begin in an interpreted state and gradually transition to optimized native code execution based on observed runtime behavior.

![JIT Compilation Pipeline](./diagrams/compilation-pipeline.svg)

The complete execution flow encompasses several distinct phases, each with specific responsibilities and data transformations. Understanding this flow is essential for implementing correct state transitions and debugging execution issues.

#### Initial Interpretation Phase

Program execution begins in the traditional interpreted mode, where the existing bytecode VM processes instructions through its dispatcher loop. During this phase, the JIT compiler remains dormant, but critical infrastructure is already in place. The profiler maintains execution counters for each function, incrementing them with minimal overhead on each function invocation. This lightweight instrumentation represents the first touch point between the existing VM and the JIT compilation system.

The profiler initialization occurs during VM startup, creating a `Profiler` structure that maps function identifiers to `ProfileEntry` records. Each entry contains the current invocation count, the compilation threshold, and a flag indicating whether the function has already been compiled. The default threshold typically ranges from 1000 to 10000 invocations, balancing compilation overhead against potential performance gains.

**Function Invocation Flow in Interpretation Mode:**

1. The VM dispatcher encounters a function call bytecode instruction
2. The dispatcher extracts the target function ID from the bytecode operand
3. Before transferring control, it calls `profiler_record_invocation()` with the function ID
4. The profiler increments the invocation counter for the target function
5. If the counter is below the compilation threshold, execution proceeds normally
6. The VM establishes a new stack frame and begins interpreting the function's bytecode
7. Upon function return, control transfers back to the calling context

This instrumentation adds only a few CPU cycles per function call, making the performance overhead negligible compared to the cost of bytecode interpretation itself. The key insight is that profiling must be lightweight enough not to harm cold code performance while providing accurate hot path detection.

#### Hot Path Detection Trigger

The transition from interpretation to compilation occurs when the profiler detects that a function has exceeded its compilation threshold. This moment represents a critical decision point where the system must coordinate between multiple components to initiate the compilation process while maintaining correct program execution.

**Hot Path Detection Algorithm:**

1. During function invocation, `profiler_record_invocation()` increments the counter
2. The function compares the new counter value against the stored threshold
3. If the threshold is exceeded and the function isn't already compiled, compilation triggers
4. The profiler marks the function as "compilation in progress" to prevent duplicate compilation
5. The system allocates a `TranslationContext` to manage the compilation process
6. The function continues executing in interpreted mode during compilation
7. Upon successful compilation, the profiler updates the function's execution pointer

The critical design decision here involves handling compilation timing. The system could compile synchronously (blocking the current execution) or asynchronously (compiling in the background). Our design chooses synchronous compilation for simplicity, though this creates a brief pause during the compilation process.

> **Decision: Synchronous vs Asynchronous Compilation**
> - **Context**: When a function reaches its compilation threshold, we must decide whether to compile immediately or defer compilation to a background thread
> - **Options Considered**: Synchronous compilation (compile immediately), asynchronous compilation (background thread), lazy compilation (compile on next invocation)
> - **Decision**: Synchronous compilation with immediate installation
> - **Rationale**: Synchronous compilation simplifies state management, eliminates race conditions between compilation and execution, and ensures compiled code is immediately available. While this creates a brief compilation pause, the complexity reduction outweighs the temporary performance impact
> - **Consequences**: Enables simpler error handling and debugging but introduces compilation latency on the triggering invocation

| Compilation Approach | Pros | Cons | Implementation Complexity |
|---------------------|------|------|---------------------------|
| Synchronous | Simple state management, immediate availability, no race conditions | Compilation latency on trigger invocation | Low |
| Asynchronous | No compilation latency, continuous execution | Complex state management, race conditions, delayed availability | High |
| Lazy | Deferred latency, simple trigger logic | Compilation overhead on multiple invocations | Medium |

#### Compilation Phase Coordination

When compilation triggers, the system must coordinate between multiple components while maintaining correct program state. This coordination represents one of the most complex aspects of JIT compiler implementation, requiring careful state management and error handling.

The compilation phase begins with the creation of a `TranslationContext` that serves as the central coordination point for the entire compilation process. This context encapsulates the bytecode function being compiled, the target code buffer, register allocation state, jump patch information, and error tracking. The context provides a single point of truth for compilation state, simplifying error handling and resource cleanup.

**Compilation Coordination Sequence:**

1. The `JITCompiler` receives a compilation request with the target `BytecodeFunction`
2. It allocates a new `CodeBuffer` with appropriate size estimates based on bytecode length
3. A `TranslationContext` is created linking the function and code buffer
4. The system initializes a `RegisterAllocator` for managing virtual stack to register mapping
5. A `JumpPatchTable` is allocated to track forward jump references during compilation
6. The main translation process begins with `translate_function()`
7. Upon successful compilation, the generated code is made executable
8. The compiled function is installed in the VM's function dispatch table
9. All temporary resources are cleaned up, leaving only the executable native code

The error handling during this sequence is crucial. If compilation fails at any stage, the system must clean up all allocated resources and fall back to continued interpretation. The VM's correctness cannot be compromised by compilation failures, making robust error recovery essential.

#### Native Code Execution Phase

Once compilation succeeds, subsequent invocations of the function execute the generated native code directly, bypassing the interpreter entirely. This transition represents the payoff for the compilation investment, where execution speed can improve by 10-100x over interpretation depending on the nature of the compiled function.

The transition to native execution involves careful state marshaling between the interpreter's virtual machine state and the native code's execution environment. The interpreter maintains program state in abstract data structures (bytecode stack, local variable arrays, program counter), while native code expects state in physical machine registers and memory locations following the System V AMD64 calling convention.

**Native Execution Transition:**

1. A function call bytecode instruction identifies a compiled function
2. The VM extracts arguments from the virtual stack into physical registers
3. The system calls the compiled function through a properly typed function pointer
4. Native code executes directly on the processor without interpretation overhead
5. Return values are marshaled back into the virtual machine's expected format
6. Control returns to the interpreter for continued execution

This seamless transition between execution modes is invisible to the running program, maintaining perfect semantic compatibility while delivering significant performance improvements. The key insight is that the JIT compiler augments rather than replaces the interpreter, creating a tiered execution system that combines the startup speed of interpretation with the runtime performance of native code.

![Function Execution State Machine](./diagrams/execution-state-machine.svg)

### JIT Compilation Pipeline

The internal compilation pipeline transforms bytecode functions into optimized native machine code through a series of well-defined stages. Each stage has specific inputs, outputs, and transformation responsibilities, creating a clear separation of concerns that simplifies both implementation and debugging.

Understanding the compilation pipeline is essential for implementing correct bytecode-to-native translation and for diagnosing compilation failures. The pipeline represents a classic compiler architecture adapted for runtime compilation constraints, where compilation speed matters as much as code quality.

#### Pipeline Stage Overview

The compilation pipeline consists of five main stages, each building upon the results of previous stages. This staged approach allows for clear error handling boundaries and enables future optimizations to be inserted at appropriate points in the pipeline.

| Pipeline Stage | Input | Output | Primary Responsibility |
|----------------|-------|--------|----------------------|
| Function Analysis | `BytecodeFunction` | Control flow graph, basic blocks | Bytecode parsing and structure analysis |
| First Pass Translation | Bytecode instructions | Native instructions + patch table | Instruction-by-instruction translation |
| Register Allocation | Virtual stack operations | Physical register assignments | Stack-to-register mapping |
| Jump Resolution | Forward references | Complete native code | Address resolution and patching |
| Code Finalization | Raw native code | Executable function | Memory protection and function pointer creation |

Each stage maintains clear interfaces and error reporting, allowing the pipeline to abort compilation at any point and fall back to interpretation. This robust error handling is crucial because compilation operates on user-provided bytecode that may contain unexpected patterns or exceed resource limits.

#### Function Analysis Stage

The pipeline begins with analyzing the bytecode function to understand its structure, control flow, and resource requirements. This analysis phase builds the foundation for all subsequent compilation stages by creating abstract representations of the function's behavior.

**Analysis Tasks:**

1. **Bytecode Validation**: Verify that the bytecode is well-formed and contains valid instruction opcodes
2. **Control Flow Discovery**: Identify basic blocks, jump targets, and loop structures
3. **Stack Depth Analysis**: Compute maximum virtual stack depth for register allocation planning
4. **Resource Estimation**: Estimate native code size for memory allocation
5. **Optimization Opportunity Identification**: Detect patterns suitable for runtime optimization

The analysis stage creates a `ControlFlowGraph` representing the function's basic blocks and their relationships. Each `BasicBlock` contains a contiguous sequence of bytecode instructions with a single entry point and single exit point. This representation simplifies subsequent compilation stages by providing clear boundaries for translation and optimization.

The stack depth analysis is particularly crucial for our register allocation strategy. Since we map virtual stack positions to physical registers, understanding the maximum stack depth determines how many registers we need and whether spill handling is required. Functions with stack depth exceeding available registers require more complex compilation strategies.

#### First Pass Translation Stage

The first pass translation converts bytecode instructions to native x86-64 instructions while building a table of forward jump references that cannot be resolved immediately. This stage focuses on correctness rather than optimization, ensuring that each bytecode operation has a semantically equivalent native instruction sequence.

**Translation Process:**

1. **Instruction Iteration**: Process bytecode instructions in sequential order
2. **Operation Mapping**: Convert each bytecode opcode to equivalent native instruction(s)
3. **Register Assignment**: Allocate physical registers for virtual stack operations
4. **Jump Recording**: Record forward jump references in the patch table
5. **Address Mapping**: Maintain correspondence between bytecode and native addresses

The translation stage uses the `RegisterAllocator` to maintain a mapping between virtual stack positions and physical x86-64 registers. As bytecode instructions push and pop values, the allocator assigns and releases registers accordingly. When register pressure exceeds available registers, the allocator generates spill code to store values in the function's stack frame.

**Arithmetic Operation Translation Example:**

Consider the bytecode sequence for computing `a + b * c`:
1. `LOAD_LOCAL 0` (load variable `a`)
2. `LOAD_LOCAL 1` (load variable `b`) 
3. `LOAD_LOCAL 2` (load variable `c`)
4. `MUL` (multiply `b * c`)
5. `ADD` (add `a + (b * c)`)

The translator processes this sequence as follows:
1. `LOAD_LOCAL 0`: Allocate register RAX, emit `mov rax, [rbp-8]` (assuming `a` is first local variable)
2. `LOAD_LOCAL 1`: Allocate register RBX, emit `mov rbx, [rbp-16]`
3. `LOAD_LOCAL 2`: Allocate register RCX, emit `mov rcx, [rbp-24]`
4. `MUL`: Generate `imul rbx, rcx`, deallocate RCX, result in RBX
5. `ADD`: Generate `add rax, rbx`, deallocate RBX, result in RAX

This translation maintains the semantic meaning of the bytecode while converting abstract stack operations to concrete register manipulations. The register allocator ensures that each virtual stack position has a corresponding physical location.

#### Register Allocation Pass

The register allocation pass assigns physical x86-64 registers to virtual stack positions, generating spill code when necessary and optimizing register usage patterns. This stage transforms the abstract stack-based computation model into efficient register-based native code.

Our register allocation strategy uses a simple but effective approach suitable for expression-level allocation. The allocator maintains a mapping between virtual stack depths and physical registers, assigning registers in a predetermined order and generating spill code when registers are exhausted.

**Register Allocation Algorithm:**

1. **Initialize Register Pool**: Create available register set (RAX, RBX, RCX, RDX, R8-R15)
2. **Process Stack Operations**: For each push operation, allocate next available register
3. **Handle Register Pressure**: When registers exhausted, spill least recently used values
4. **Generate Spill Code**: Emit store instructions to save spilled values to stack frame
5. **Generate Reload Code**: Emit load instructions to restore spilled values when needed
6. **Track Register Liveness**: Deallocate registers when values are consumed

The allocator uses a simple stack-based allocation strategy where virtual stack positions map directly to physical registers in allocation order. This approach works well for expression compilation while being simple to implement and debug.

| Virtual Stack Position | Physical Register | Spill Location |
|-----------------------|------------------|----------------|
| 0 (stack bottom) | RAX | [rbp-32] |
| 1 | RBX | [rbp-40] |
| 2 | RCX | [rbp-48] |
| 3 | RDX | [rbp-56] |
| ... | ... | ... |
| 14 | R15 | [rbp-144] |
| 15 (spilled) | (memory only) | [rbp-152] |

When the virtual stack depth exceeds available registers, the allocator generates spill code that stores register contents to predetermined stack frame locations. The spill locations are allocated in the function prologue, ensuring adequate stack frame space for worst-case register pressure.

#### Jump Resolution Pass

The jump resolution pass completes the compilation process by resolving all forward jump references recorded during the first pass translation. This stage converts relative offsets and symbolic jump targets into concrete native code addresses, producing fully executable machine code.

Forward jumps present a classic chicken-and-egg problem in code generation: we need to know the target address to generate the jump instruction, but we don't know the target address until we've generated all the intervening instructions. The jump patch table solves this by recording incomplete jumps during the first pass and filling in the correct addresses during a second pass.

**Jump Resolution Process:**

1. **Iterate Patch Table**: Process each recorded forward jump reference
2. **Calculate Target Address**: Determine native code address corresponding to bytecode target
3. **Compute Relative Offset**: Calculate displacement from jump instruction to target
4. **Update Jump Instruction**: Patch the incomplete jump instruction with correct offset
5. **Validate Jump Range**: Ensure the displacement fits in the instruction's immediate field

The `JumpPatchTable` maintains a mapping between bytecode instruction offsets and their corresponding native code addresses. When resolving a forward jump, the system looks up the target bytecode offset in this mapping to find the native address, then calculates the relative displacement needed for the x86-64 jump instruction.

**Jump Patch Table Structure:**

| Field | Type | Description |
|-------|------|-------------|
| `entries` | `JumpPatchEntry*` | Array of pending jump patches |
| `capacity` | `size_t` | Maximum number of patch entries |
| `count` | `size_t` | Current number of pending patches |
| `bytecode_to_native_map` | `uint32_t*` | Mapping from bytecode to native addresses |
| `map_size` | `size_t` | Size of the address mapping array |

Each `JumpPatchEntry` records the native code location of an incomplete jump instruction, the target bytecode offset, and the jump type (conditional or unconditional). During resolution, the system updates the native instruction at the recorded location with the correct displacement value.

#### Code Finalization Stage

The final pipeline stage transforms the raw native code into an executable function by setting appropriate memory permissions and creating a typed function pointer that can be called directly from the VM. This stage handles the security-sensitive transition from writable code generation to executable code execution.

Modern operating systems implement W^X (Write XOR Execute) policies that prevent memory pages from being both writable and executable simultaneously. This security measure prevents many code injection attacks but requires JIT compilers to carefully manage memory permissions during code generation.

**Finalization Process:**

1. **Complete Code Generation**: Ensure all instructions and patches are finalized
2. **Add Function Epilogue**: Generate proper return instruction sequence
3. **Change Memory Protection**: Switch code buffer from writable to executable
4. **Create Function Pointer**: Cast executable memory to appropriate function pointer type
5. **Validate Generated Code**: Perform basic sanity checks on the native code
6. **Install in Function Cache**: Store the compiled function for future invocations

The memory protection change represents a critical security boundary. Once the code buffer becomes executable, it cannot be modified without changing permissions again. This immutability ensures that the generated code cannot be tampered with during execution, preventing various security vulnerabilities.

The function pointer creation requires careful attention to calling convention compatibility. The generated native code must follow the System V AMD64 ABI exactly, ensuring that arguments are received in the correct registers and return values are provided in the expected format.

### Data Flow Patterns

Understanding how data flows through the JIT compiler system is crucial for implementing correct state management and debugging execution issues. The system involves multiple data format transformations and state representations, each optimized for different stages of the compilation and execution process.

![System Architecture](./diagrams/system-architecture.svg)

The data flow patterns reveal how information moves between the major system components, transforming from high-level bytecode representations to low-level native code execution. These transformations must preserve program semantics while enabling significant performance improvements.

#### Bytecode to Native Code Transformation

The core data flow involves transforming bytecode instructions and their operands into semantically equivalent native machine instructions. This transformation operates at multiple levels: individual instruction translation, virtual stack to register mapping, and abstract addresses to concrete memory locations.

**Data Transformation Layers:**

| Abstraction Level | Representation | Example |
|------------------|----------------|---------|
| High-level bytecode | Opcode + operands | `ADD` (stack[top-1] + stack[top]) |
| Virtual machine state | Stack positions + locals | Virtual stack depth 2, local vars in array |
| Register allocation | Physical registers | Values in RAX and RBX registers |
| Native instructions | x86-64 assembly | `add rax, rbx` |
| Machine code | Raw bytes | `0x48 0x01 0xD8` |

Each transformation layer serves a specific purpose in bridging the semantic gap between bytecode abstractions and native execution realities. The bytecode level provides a clean, platform-independent representation of program operations. The virtual machine state layer manages program state in abstract containers optimized for interpretation. The register allocation layer maps abstract state to physical hardware resources. The native instruction layer represents operations in processor-specific forms. Finally, the machine code layer provides the binary representation that processors actually execute.

The key insight is that these transformations must preserve program semantics exactly while optimizing for native execution performance. Any deviation from bytecode semantics represents a bug that could cause incorrect program behavior.

#### Execution State Marshaling

The transition between interpreted and native execution requires careful marshaling of program state between different representational formats. The interpreter maintains state in virtual machine abstractions (bytecode stack, local variable arrays, program counter), while native code expects state in physical machine locations (registers, memory addresses, instruction pointer).

**State Marshaling Components:**

1. **Virtual Stack to Register Conversion**: Map bytecode stack positions to x86-64 registers
2. **Local Variable Address Translation**: Convert variable indices to stack frame offsets
3. **Return Address Management**: Handle call/return semantics across execution modes
4. **Exception State Preservation**: Maintain exception handling capability during transitions
5. **Profiling Data Continuity**: Preserve execution counters and optimization state

The marshaling process must handle mismatches between abstract and concrete representations. For example, the bytecode virtual stack can grow to arbitrary depth, while physical registers are limited to 16 general-purpose registers on x86-64. The register allocator addresses this mismatch through spill handling, but the marshaling code must understand when values reside in registers versus memory.

**Execution State Mapping:**

| VM State Component | Native Equivalent | Marshaling Method |
|-------------------|-------------------|-------------------|
| Virtual stack position 0 | RAX register | Direct register assignment |
| Virtual stack position 1 | RBX register | Direct register assignment |
| Virtual stack position 15+ | Stack frame offset | Memory load/store operations |
| Local variable N | [RBP - (N+1)*8] | Calculated stack offset |
| Function arguments | RDI, RSI, RDX, RCX, R8, R9 | ABI-compliant register usage |
| Return value | RAX register | Standard calling convention |

The marshaling process operates bidirectionally, converting interpreter state to native format when entering compiled code and converting native state back to interpreter format when returning. This bidirectional conversion ensures seamless integration between execution modes.

#### Memory Management Flow

The memory management subsystem handles the complex lifecycle of executable code buffers, ensuring that generated machine code is properly allocated, protected, and eventually released. This flow involves multiple security-sensitive operations that must be handled correctly to prevent memory leaks and security vulnerabilities.

**Memory Management Stages:**

1. **Initial Allocation**: Allocate writable memory pages for code generation
2. **Code Generation**: Fill allocated pages with generated machine instructions
3. **Permission Transition**: Change page protection from writable to executable
4. **Function Installation**: Register executable code in the VM's dispatch table
5. **Execution Phase**: Execute generated code directly from protected pages
6. **Cache Management**: Track compiled functions for memory usage optimization
7. **Cleanup**: Eventually release executable pages when functions become obsolete

The memory protection transitions represent critical security boundaries that must be handled correctly. Modern operating systems provide `mmap()` system calls with specific protection flags (`PROT_READ`, `PROT_WRITE`, `PROT_EXEC`) that control page access permissions. JIT compilers must coordinate these permissions carefully to maintain both functionality and security.

**Memory Protection State Machine:**

| Current State | Operation | New State | Security Implications |
|---------------|-----------|-----------|----------------------|
| Unallocated | `mmap(PROT_READ\|PROT_WRITE)` | Writable | Safe for code generation |
| Writable | Code emission | Writable | Generated instructions accumulate |
| Writable | `mprotect(PROT_READ\|PROT_EXEC)` | Executable | Code becomes immutable and runnable |
| Executable | Function execution | Executable | Native code runs directly |
| Executable | `munmap()` | Unallocated | Memory returned to system |

The transition from writable to executable represents the security-critical moment where generated code becomes immutable and trusted. Once this transition occurs, the code cannot be modified without explicitly changing permissions again, preventing various code injection attacks that might target JIT compilers.

#### Profile Data Evolution

The profiling subsystem tracks execution frequency and behavioral patterns that guide compilation decisions and optimizations. This data flow operates continuously throughout program execution, accumulating statistical information that drives the tiered compilation strategy.

**Profiling Data Lifecycle:**

1. **Counter Initialization**: Create execution counters for each function during VM startup
2. **Invocation Tracking**: Increment counters on each function call with minimal overhead
3. **Threshold Evaluation**: Compare counters against compilation thresholds during profiling
4. **Compilation Triggering**: Use profile data to make compilation decisions
5. **Optimization Guidance**: Feed profile information to optimization passes
6. **Adaptive Adjustment**: Modify thresholds based on observed performance characteristics

The profiling system must balance accuracy against overhead, collecting enough information to make good compilation decisions without significantly slowing down program execution. This balance is achieved through lightweight instrumentation that adds only a few CPU cycles per function invocation.

**Profile Data Structures Evolution:**

| Execution Phase | Data Collected | Storage Format | Usage Pattern |
|-----------------|----------------|----------------|---------------|
| Initial execution | Basic invocation counts | `ProfileEntry` counters | Simple threshold comparison |
| Hot path detection | Detailed frequency data | Extended `ProfileData` | Compilation trigger analysis |
| Compilation guidance | Type and branch profiles | `TypeProfile`, `BranchProfile` | Optimization decision making |
| Performance monitoring | Speedup measurements | `PerformanceMonitor` | Threshold adjustment and tuning |

The evolution from simple counters to detailed profiling data reflects the system's growing understanding of program behavior. Initial execution phases collect minimal data to reduce overhead, while later phases gather detailed information to guide sophisticated optimizations.

### Common Pitfalls

⚠️ **Pitfall: Incomplete State Marshaling**

A frequent mistake involves incomplete marshaling of program state when transitioning between interpreted and native execution modes. Developers often correctly handle obvious state like function arguments and return values but miss subtle state like exception handling context or profiling counters. This incomplete marshaling can cause crashes when exceptions occur during native code execution or lead to incorrect profiling data that affects compilation decisions.

The problem manifests as intermittent crashes or incorrect program behavior that appears only when specific functions become compiled. Debugging is challenging because the issue depends on execution history (which functions have been compiled) rather than just current program state.

**Detection**: Monitor for crashes that occur only after functions have been compiled, or for cases where program behavior changes after compilation triggers. Use debugging tools to compare interpreter state before and after native function calls.

**Solution**: Create comprehensive state marshaling checklists that cover all VM state components. Implement state validation routines that verify interpreter state consistency before and after native code execution. Design marshaling code to be symmetric - every piece of state converted for native execution must be properly restored when returning to interpretation.

⚠️ **Pitfall: Race Conditions in Compilation Triggers**

Multi-threaded programs can trigger race conditions where multiple threads simultaneously decide to compile the same function. Without proper synchronization, this can lead to duplicate compilation work, memory leaks from multiple code buffers, or crashes when multiple threads modify the same profiling data structures.

The issue typically manifests as occasional crashes during compilation or as multiple compiled versions of the same function consuming excessive memory. The problem is intermittent and may only occur under high load when multiple threads execute the same hot functions simultaneously.

**Detection**: Look for crashes during compilation phase or excessive memory usage from compiled code. Monitor for duplicate entries in function caches or inconsistent profiling counter states.

**Solution**: Implement atomic operations for profiling counter updates and compilation state changes. Use compare-and-swap operations to ensure only one thread can transition a function from "eligible for compilation" to "compilation in progress". Design the profiling system to handle concurrent access gracefully, possibly using thread-local counters that merge periodically.

⚠️ **Pitfall: Memory Protection Ordering Violations**

Incorrect ordering of memory protection operations can create security vulnerabilities or cause crashes. Common mistakes include attempting to execute code before changing permissions to executable, or forgetting to change permissions from writable before execution, leaving executable code pages writable.

These violations can manifest as segmentation faults when attempting to execute code with incorrect permissions, or as security vulnerabilities that allow runtime code modification. The timing of these errors depends on operating system behavior and may vary between development and production environments.

**Detection**: Monitor for segmentation faults during native code execution. Use memory debugging tools to verify that executable pages are not writable during execution. Implement assertions that check page permissions before executing generated code.

**Solution**: Create a strict memory protection state machine that enforces correct transitions. Implement wrapper functions that combine code generation completion with appropriate permission changes. Never attempt to execute code until permissions have been verified as executable. Consider using platform-specific APIs that provide atomic allocation-and-protection operations.

⚠️ **Pitfall: Jump Patch Resolution Errors**

Forward jump resolution is particularly error-prone because it involves address arithmetic and instruction encoding details. Common mistakes include calculating incorrect relative offsets, patching the wrong instruction bytes, or exceeding the range limits of jump instructions.

These errors typically manifest as incorrect program control flow, infinite loops, or crashes when jump instructions transfer control to invalid addresses. The symptoms may be immediate (crash on first jump) or delayed (incorrect logic that causes later failures).

**Detection**: Implement verification routines that check jump target validity before patching. Use disassemblers to inspect generated code and verify that jump instructions have correct targets. Test with functions that contain various jump patterns (forward, backward, conditional, unconditional).

**Solution**: Create comprehensive test suites that exercise all jump patterns. Implement bounds checking for relative offset calculations. Use symbolic debugging information to track the relationship between bytecode addresses and native addresses. Consider using trampolines for long-distance jumps that exceed x86-64 relative addressing limits.

### Implementation Guidance

The implementation of JIT compiler interactions and data flow requires careful coordination between multiple subsystems while maintaining performance and correctness. This guidance provides concrete implementation strategies for managing the complex state transitions and data transformations involved in tiered compilation.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Memory Management | `mmap()` with manual protection changes | Custom memory allocator with pool management |
| State Marshaling | Direct register assignment with fixed mapping | Dynamic register allocation with live range analysis |
| Profiling | Simple counters with fixed thresholds | Adaptive thresholds with statistical analysis |
| Jump Patching | Two-pass compilation with fixed-size patch table | Single-pass with dynamic relocation tables |
| Debugging Support | Printf-style logging with execution tracing | GDB integration with symbol table generation |

#### File Structure Organization

```c
project-root/
  src/
    jit/
      jit_compiler.c           ← main JIT compiler coordination
      jit_compiler.h           ← public JIT compiler interface
      execution_flow.c         ← execution flow management
      execution_flow.h         ← flow control interfaces
      state_marshaling.c       ← interpreter/native state conversion
      state_marshaling.h       ← marshaling interfaces
      profiler.c              ← execution profiling and hot path detection
      profiler.h              ← profiling interfaces
      memory_manager.c        ← executable memory management
      memory_manager.h        ← memory management interfaces
      compilation_pipeline.c   ← compilation coordination
      compilation_pipeline.h   ← pipeline interfaces
    vm/
      interpreter.c           ← existing bytecode interpreter
      interpreter.h           ← interpreter interfaces (to be extended)
  tests/
    integration/
      execution_flow_test.c   ← end-to-end execution flow tests
      state_marshaling_test.c ← state conversion correctness tests
      profiling_test.c        ← profiling behavior verification
```

#### Core Execution Flow Implementation

```c
// execution_flow.h
#include <stdint.h>
#include <stddef.h>

// Function execution mode tracking
typedef enum {
    EXECUTION_MODE_INTERPRETED,
    EXECUTION_MODE_COMPILING,
    EXECUTION_MODE_NATIVE,
    EXECUTION_MODE_ERROR
} ExecutionMode;

// Complete execution flow coordinator
typedef struct {
    Profiler* profiler;
    JITCompiler* compiler;
    PerformanceMonitor* monitor;
    ExecutionMode default_mode;
    int profiling_enabled;
} ExecutionFlowManager;

// Main execution flow entry point - handles mode selection and transitions
int execute_function_with_flow_control(ExecutionFlowManager* manager, 
                                     uint32_t function_id, 
                                     BytecodeFunction* function,
                                     int64_t* arguments, 
                                     int argument_count,
                                     int64_t* return_value);

// Hot path detection and compilation trigger logic
int check_compilation_eligibility(ExecutionFlowManager* manager, 
                                uint32_t function_id, 
                                BytecodeFunction* function);

// Coordinate transition between execution modes
int transition_execution_mode(ExecutionFlowManager* manager,
                            uint32_t function_id,
                            ExecutionMode from_mode,
                            ExecutionMode to_mode);
```

```c
// execution_flow.c - Core coordination implementation skeleton
#include "execution_flow.h"

ExecutionFlowManager* execution_flow_manager_create(uint32_t default_threshold) {
    ExecutionFlowManager* manager = malloc(sizeof(ExecutionFlowManager));
    if (!manager) return NULL;
    
    // TODO 1: Initialize profiler with default compilation threshold
    // TODO 2: Create JIT compiler instance with appropriate configuration
    // TODO 3: Initialize performance monitoring for compilation tracking
    // TODO 4: Set default execution mode to interpreted
    // TODO 5: Enable profiling by default for hot path detection
    
    return manager;
}

int execute_function_with_flow_control(ExecutionFlowManager* manager, 
                                     uint32_t function_id, 
                                     BytecodeFunction* function,
                                     int64_t* arguments, 
                                     int argument_count,
                                     int64_t* return_value) {
    // TODO 1: Check current execution mode for the function
    // TODO 2: If interpreted mode, record invocation and check compilation threshold
    // TODO 3: If threshold exceeded, trigger compilation process
    // TODO 4: If compilation succeeds, update execution mode to native
    // TODO 5: Execute function using appropriate mode (interpreted or native)
    // TODO 6: Marshal return value and state back to interpreter format
    // TODO 7: Update performance monitoring data
    // Hint: Use profiler_record_invocation() to handle threshold checking
    // Hint: Compilation failures should fall back to interpretation gracefully
}
```

#### State Marshaling Infrastructure

```c
// state_marshaling.h
#include "vm_types.h"  // InterpreterState and related types

// State conversion between interpreter and native execution
typedef struct {
    InterpreterState* interpreter_state;
    X86Register* register_assignments;
    int64_t* spill_locations;
    size_t spill_count;
    int marshaling_failed;
} StateMarshalingContext;

// Convert interpreter state to native calling convention
int marshal_interpreter_to_native(StateMarshalingContext* context,
                                 int64_t* arguments,
                                 int argument_count,
                                 ABICallInfo* call_info);

// Convert native return state back to interpreter format
int marshal_native_to_interpreter(StateMarshalingContext* context,
                                 int64_t return_value,
                                 InterpreterState* target_state);

// Handle state transitions for function calls between modes
int transition_call_state(StateMarshalingContext* context,
                         ExecutionMode from_mode,
                         ExecutionMode to_mode);
```

#### Compilation Pipeline Coordination

```c
// compilation_pipeline.h
#include "bytecode_translator.h"
#include "register_allocator.h"

// Pipeline stage coordination
typedef struct {
    BytecodeFunction* source_function;
    TranslationContext* translation;
    OptimizationContext* optimization;
    CodeBuffer* target_buffer;
    TranslationResult* result;
    int pipeline_stage;
    int compilation_failed;
} CompilationPipelineContext;

// Main pipeline execution with error handling
TranslationResult* execute_compilation_pipeline(BytecodeFunction* function,
                                              uint32_t optimization_level);

// Individual pipeline stages with clear interfaces
int pipeline_stage_analysis(CompilationPipelineContext* context);
int pipeline_stage_translation(CompilationPipelineContext* context);
int pipeline_stage_optimization(CompilationPipelineContext* context);
int pipeline_stage_finalization(CompilationPipelineContext* context);
```

```c
// compilation_pipeline.c - Pipeline coordination skeleton
TranslationResult* execute_compilation_pipeline(BytecodeFunction* function,
                                              uint32_t optimization_level) {
    CompilationPipelineContext* context = create_pipeline_context(function, optimization_level);
    if (!context) return NULL;
    
    // TODO 1: Execute function analysis stage - parse bytecode, build CFG
    // TODO 2: Execute first-pass translation - convert instructions to native
    // TODO 3: Execute register allocation pass - assign physical registers
    // TODO 4: Execute jump resolution pass - patch forward references
    // TODO 5: Execute optimization passes if optimization level > 0
    // TODO 6: Execute code finalization - set memory permissions, create function pointer
    // TODO 7: Clean up intermediate data structures, return final result
    // Hint: Each stage should check for errors and abort pipeline if failures occur
    // Hint: Maintain compilation statistics for performance monitoring
    
    return extract_translation_result(context);
}
```

#### Memory Management Integration

```c
// memory_manager.h
#include <sys/mman.h>

// Executable memory pool management
typedef struct {
    void* base_address;
    size_t total_size;
    size_t used_size;
    CodeBuffer** allocated_buffers;
    size_t buffer_count;
    int protection_state;  // W^X compliance tracking
} ExecutableMemoryPool;

// High-level memory management interface
ExecutableMemoryPool* executable_memory_pool_create(size_t initial_size);
CodeBuffer* allocate_compilation_buffer(ExecutableMemoryPool* pool, size_t estimated_size);
int finalize_compilation_buffer(ExecutableMemoryPool* pool, CodeBuffer* buffer);
int release_compilation_buffer(ExecutableMemoryPool* pool, CodeBuffer* buffer);
```

#### Profiling Integration

```c
// profiler.c - Core profiling implementation skeleton
int profiler_record_invocation(Profiler* profiler, uint32_t function_id) {
    // TODO 1: Locate or create ProfileEntry for the function ID
    // TODO 2: Atomically increment invocation counter (for thread safety)
    // TODO 3: Compare current count against compilation threshold
    // TODO 4: Return compilation recommendation (0 = continue interpreting, 1 = compile)
    // TODO 5: Update profiling statistics for performance monitoring
    // Hint: Use __sync_fetch_and_add for atomic counter updates
    // Hint: Avoid expensive operations in this hot path
}

void adjust_compilation_threshold(JITCompiler* compiler, uint32_t function_id, 
                                int compilation_success, double observed_speedup) {
    // TODO 1: Locate ProfileEntry for the function
    // TODO 2: Calculate new threshold based on compilation outcome
    // TODO 3: If compilation failed, increase threshold to reduce retry frequency
    // TODO 4: If compilation succeeded with good speedup, decrease threshold
    // TODO 5: If compilation succeeded with poor speedup, increase threshold
    // TODO 6: Clamp threshold values to reasonable ranges (100-50000)
    // Hint: Use exponential adjustment - multiply/divide by factors of 2
}
```

#### Testing and Validation Framework

```c
// tests/integration/execution_flow_test.c
#include "execution_flow.h"
#include "test_framework.h"

// Comprehensive execution flow testing
void test_interpreter_to_native_transition() {
    // TODO 1: Create simple bytecode function with arithmetic operations
    // TODO 2: Execute function in interpretation mode below threshold
    // TODO 3: Execute function repeatedly to trigger compilation
    // TODO 4: Verify that compiled version produces identical results
    // TODO 5: Test state marshaling by checking intermediate values
    // TODO 6: Verify performance improvement after compilation
}

void test_mixed_mode_execution() {
    // TODO 1: Create functions that call each other (some compiled, some interpreted)
    // TODO 2: Verify that state marshaling works correctly across call boundaries
    // TODO 3: Test exception handling across execution mode transitions
    // TODO 4: Verify that profiling continues correctly in mixed-mode execution
}
```

#### Debugging and Diagnostics

```c
// debugging support for execution flow
#define DEBUG_EXECUTION_FLOW 1

#if DEBUG_EXECUTION_FLOW
#define TRACE_EXECUTION(fmt, ...) \
    fprintf(stderr, "[EXEC] " fmt "\n", ##__VA_ARGS__)
#define TRACE_COMPILATION(fmt, ...) \
    fprintf(stderr, "[COMP] " fmt "\n", ##__VA_ARGS__)
#define TRACE_MARSHALING(fmt, ...) \
    fprintf(stderr, "[MARSH] " fmt "\n", ##__VA_ARGS__)
#else
#define TRACE_EXECUTION(fmt, ...)
#define TRACE_COMPILATION(fmt, ...)
#define TRACE_MARSHALING(fmt, ...)
#endif

// Execution flow validation helpers
int validate_interpreter_state(InterpreterState* state);
int validate_native_function_pointer(CompiledFunction function);
int validate_state_marshaling_consistency(StateMarshalingContext* context);
```

#### Platform-Specific Considerations

```c
// Platform abstraction for executable memory
#ifdef __linux__
    #define EXECUTABLE_MEMORY_FLAGS (MAP_PRIVATE | MAP_ANONYMOUS)
    #define PAGE_SIZE_BYTES 4096
#elif __APPLE__
    #define EXECUTABLE_MEMORY_FLAGS (MAP_PRIVATE | MAP_ANON)
    #define PAGE_SIZE_BYTES 4096
    // macOS may require additional entitlements for JIT compilation
#endif

// Thread safety for profiling counters
#ifdef __GNUC__
    #define ATOMIC_INCREMENT(ptr) __sync_fetch_and_add(ptr, 1)
    #define ATOMIC_COMPARE_SWAP(ptr, old, new) __sync_bool_compare_and_swap(ptr, old, new)
#else
    // Fallback to mutex-based synchronization for other compilers
    extern pthread_mutex_t profiling_mutex;
    #define ATOMIC_INCREMENT(ptr) do { pthread_mutex_lock(&profiling_mutex); (*ptr)++; pthread_mutex_unlock(&profiling_mutex); } while(0)
#endif

```


## Error Handling and Edge Cases

> **Milestone(s):** All milestones — robust error handling is essential throughout the JIT compiler implementation, from basic machine code emission through advanced optimization passes

Error handling in a JIT compiler presents unique challenges that traditional compilers don't face. Unlike ahead-of-time compilation where errors simply prevent executable generation, JIT compilation errors occur during program execution and must be handled gracefully without crashing the running application. The system must maintain correctness guarantees while providing seamless fallback to interpretation when compilation fails.

### Mental Model: Safety Net Architecture

Think of JIT compiler error handling like a circus safety net system. Trapeze artists (bytecode functions) perform increasingly complex maneuvers (compilation optimizations) at greater heights (native code performance), but every performance level has safety nets underneath. When an artist falls (compilation fails), they don't hit the ground (crash the program) — they bounce safely into the net and can climb back up to try again or perform at a safer height (interpretation). The key insight is that the safety nets are always there, tested, and reliable, even when the high-flying performance attempts fail.

This mental model emphasizes three critical principles: **graceful degradation** (falling back to interpretation maintains correctness), **transparent recovery** (users don't see the failure), and **progressive retry** (future attempts can succeed even after failures). The safety net infrastructure must be as robust as the performance system itself.

![Error Handling and Recovery Flow](./diagrams/error-recovery.svg)

### Compilation Error Recovery

Compilation error recovery addresses failures during the bytecode-to-native translation process. These errors range from instruction encoding failures and register allocation exhaustion to optimization pass crashes and memory allocation problems. The recovery strategy must ensure that compilation failures never prevent program execution — interpretation provides the guaranteed fallback path.

#### Error Classification and Response Strategies

The JIT compiler encounters several classes of compilation errors, each requiring different recovery strategies:

| Error Class | Examples | Recovery Strategy | Performance Impact |
|-------------|----------|-------------------|-------------------|
| Resource Exhaustion | Register allocation failure, executable memory full | Immediate fallback to interpretation | None — interpretation continues |
| Translation Failures | Unsupported bytecode sequence, invalid instruction combination | Mark function as non-compilable | Permanent interpretation for this function |
| Optimization Crashes | CFG construction failure, liveness analysis error | Retry compilation without optimizations | Reduced optimization level |
| Memory Protection Errors | mmap failure, mprotect failure | System-wide compilation disable | All functions use interpretation |

> **Design Insight**: The key architectural principle is **fail-safe compilation** — any compilation error must leave the system in a state where interpretation can proceed normally. This requires careful separation between compilation state and interpreter state, ensuring that compilation failures cannot corrupt the execution environment.

The `TranslationResult` structure captures both successful compilation results and detailed error information:

| Field | Type | Description |
|-------|------|-------------|
| compiled_function | CompiledFunction | Function pointer to compiled code (NULL on error) |
| native_code_size | size_t | Size of generated machine code in bytes |
| register_count_used | int | Number of registers allocated during compilation |
| spill_count | int | Number of values spilled to memory |
| error | TranslationError | Detailed error information for failed compilations |

#### Translation Error Recovery Process

When compilation fails, the recovery process follows a structured sequence that preserves system stability while gathering diagnostic information:

1. **Error Detection and Context Capture**: The compilation pipeline immediately stops execution and captures the current translation context, including the failing bytecode instruction, register allocation state, and code generation progress. This information enables both immediate recovery decisions and future debugging.

2. **State Cleanup and Memory Management**: All partially allocated resources are released, including executable memory buffers, register allocator state, and jump patch tables. The cleanup process ensures no memory leaks or resource exhaustion from failed compilation attempts.

3. **Error Classification and Logging**: The error type determines both immediate recovery strategy and future compilation decisions. Temporary errors (like memory exhaustion) allow retry attempts, while permanent errors (like unsupported instruction sequences) mark the function as non-compilable.

4. **Fallback Decision and Implementation**: Based on error classification, the system either immediately falls back to interpretation, schedules a retry with reduced optimization, or disables compilation entirely. The fallback decision considers both error severity and system resource availability.

5. **Profile Update and Future Compilation**: The profiling system records the compilation failure to prevent immediate retry attempts and adjusts compilation thresholds based on failure patterns. Functions with repeated compilation failures may have compilation permanently disabled.

> **Decision: Immediate Fallback vs. Retry Strategy**
> - **Context**: When compilation fails, the system can either immediately fall back to interpretation or attempt retry with different parameters
> - **Options Considered**: Always retry, never retry, adaptive retry based on error type
> - **Decision**: Adaptive retry based on error classification
> - **Rationale**: Resource exhaustion errors often resolve after memory cleanup, while translation errors indicate fundamental incompatibilities that won't improve with retry
> - **Consequences**: Enables recovery from temporary failures while avoiding infinite retry loops for permanent failures

#### Error Recovery Implementation Patterns

The compilation error recovery follows established patterns that separate error detection, classification, and response:

```c
typedef enum {
    TRANSLATION_SUCCESS = 0,
    TRANSLATION_ERROR_REGISTER_EXHAUSTION,
    TRANSLATION_ERROR_MEMORY_ALLOCATION,
    TRANSLATION_ERROR_INVALID_BYTECODE,
    TRANSLATION_ERROR_UNSUPPORTED_OPERATION,
    TRANSLATION_ERROR_OPTIMIZATION_FAILURE,
    TRANSLATION_ERROR_ABI_VIOLATION
} TranslationError;
```

Each error type maps to specific recovery actions that balance system stability with performance recovery opportunities.

### Runtime Exception Handling

Runtime exception handling addresses errors that occur during execution of JIT-compiled native code. Unlike compilation errors which affect the translation process, runtime exceptions occur in generated machine code and must be handled within the context of native code execution. The challenge lies in maintaining proper exception semantics while coordinating between native code and interpreter exception handling mechanisms.

#### Exception Sources in JIT-Compiled Code

JIT-compiled code can generate exceptions from several sources, each requiring different handling strategies:

| Exception Source | Common Causes | Detection Method | Recovery Strategy |
|------------------|---------------|------------------|-------------------|
| Arithmetic Errors | Division by zero, integer overflow | CPU trap/signal | Convert to VM exception and unwind |
| Memory Access Violations | Null pointer dereference, bounds violation | Segmentation fault | Crash recovery with interpreter fallback |
| Stack Overflow | Deep recursion in compiled code | Stack guard page fault | Stack unwinding and interpreter transition |
| System Call Failures | File operations, network errors | Return code checking | Propagate to VM exception system |

The critical challenge is that native code exceptions must be converted back into VM-level exceptions that maintain proper semantics with interpreted code. This requires **exception translation bridges** that map native exceptions to bytecode exception types.

#### Exception Context Marshaling

When exceptions occur in JIT-compiled code, the system must marshal the native execution context back into interpreter-compatible exception state:

| Context Element | Native Representation | Interpreter Representation | Marshaling Process |
|-----------------|----------------------|----------------------------|-------------------|
| Stack Trace | Native call stack | Bytecode instruction pointers | Map native addresses to bytecode offsets |
| Local Variables | Register/memory values | VM stack slots | Convert native state to interpreter state |
| Exception Object | Native exception structure | VM object reference | Create VM exception object from native data |
| Program Counter | Native instruction address | Bytecode instruction offset | Reverse-map using address translation table |

The marshaling process requires maintaining **bidirectional address mapping** between native code addresses and bytecode instruction offsets throughout compilation. This mapping enables accurate exception reporting and proper stack trace generation.

> **Design Insight**: Exception handling in JIT-compiled code requires treating the native code as an extension of the interpreter, not a separate execution environment. All exceptions must be translatable back to bytecode-level semantics to maintain proper program behavior and debugging information.

#### Signal Handler Integration

Native code exceptions typically manifest as operating system signals (SIGSEGV, SIGFPE, etc.) that must be caught and converted into VM exceptions:

1. **Signal Handler Registration**: The JIT system registers signal handlers for all exceptions that can occur in compiled code. These handlers are installed globally but must differentiate between exceptions in JIT code versus exceptions in other system components.

2. **Exception Source Identification**: When a signal occurs, the handler examines the faulting instruction address to determine if the exception originated from JIT-compiled code. The executable memory allocator maintains address range information to enable this identification.

3. **Context Extraction and Conversion**: The signal handler extracts the native execution context (registers, stack pointer, instruction pointer) and converts it to interpreter state using the address mapping tables maintained during compilation.

4. **VM Exception Creation and Throw**: The converted context is used to create a proper VM exception object that maintains correct stack trace information and exception semantics. The exception is then thrown through the interpreter's normal exception handling mechanism.

5. **Execution Mode Transition**: After exception creation, execution transitions back to interpreted mode, ensuring that exception handling and stack unwinding proceed through well-tested interpreter code paths.

> **Decision: Signal-Based vs. Return Code Exception Handling**
> - **Context**: Native code exceptions can be handled through signal handlers or by checking return codes after every potentially-failing operation
> - **Options Considered**: Signal handlers, explicit return code checking, mixed approach
> - **Decision**: Signal handlers for unexpected exceptions, return codes for expected failures
> - **Rationale**: Signal handlers provide transparent exception handling without code generation overhead, while return codes handle expected failures like file operations gracefully
> - **Consequences**: Enables seamless exception handling with minimal code generation overhead but requires careful signal handler implementation

### Memory Management Edge Cases

Memory management edge cases in JIT compilation involve failures in executable memory allocation, memory protection changes, and cleanup of compiled code. These failures can occur at any point during compilation or execution and require careful handling to prevent memory corruption, resource leaks, and system instability.

#### Executable Memory Allocation Failures

Executable memory allocation can fail for several reasons, each requiring different recovery strategies:

| Failure Type | Cause | Detection | Recovery Strategy |
|--------------|-------|-----------|-------------------|
| Virtual Memory Exhaustion | Process address space limit reached | mmap returns -1 with ENOMEM | Garbage collect unused compiled code |
| Permission Denial | Security policy prevents executable memory | mmap returns -1 with EPERM | Disable JIT compilation system-wide |
| Fragmentation | Available memory not contiguous | mmap succeeds but size insufficient | Try smaller allocation or compaction |
| W^X Policy Violation | Platform prevents simultaneous write+execute | mprotect fails during finalization | Use two-phase compilation with memory copying |

The `ExecutableMemoryPool` manages these failure scenarios through a combination of resource tracking, garbage collection, and graceful degradation:

| Field | Type | Description |
|-------|------|-------------|
| base_address | void* | Start of allocated executable memory region |
| total_size | size_t | Total size of memory pool in bytes |
| used_size | size_t | Currently allocated memory within pool |
| allocated_buffers | CodeBuffer** | Array of active compilation buffers |
| buffer_count | size_t | Number of active buffers in pool |
| protection_state | int | Current memory protection flags (read/write/execute) |

#### Memory Pool Management and Recovery

When executable memory allocation fails, the memory pool implements a multi-stage recovery process:

1. **Immediate Allocation Retry**: For transient failures, the allocator immediately retries with exponential backoff, allowing temporary resource contention to resolve. This handles cases where another process briefly exhausts system resources.

2. **Garbage Collection Trigger**: If immediate retry fails, the system triggers garbage collection of unused compiled code. Functions that haven't been executed recently are candidates for deallocation, freeing their executable memory for new compilation attempts.

3. **Allocation Size Reduction**: When memory pressure persists, the allocator attempts smaller buffer allocations, potentially requiring multiple buffers per function but enabling compilation to proceed with reduced memory footprint.

4. **Pool Compaction**: As a last resort before disabling compilation, the memory pool performs compaction, moving active compiled code to eliminate fragmentation and create larger contiguous regions for new allocations.

5. **Compilation Disable**: If all recovery strategies fail, the system permanently disables JIT compilation for the current process, ensuring that interpretation can continue normally without memory allocation pressure.

> **Design Insight**: Memory management in JIT systems must be defensive and adaptive. Unlike traditional memory allocation where failure can be reported to the user, JIT memory allocation failures must be handled transparently while maintaining system performance and stability.

#### W^X Policy Compliance

Modern operating systems increasingly enforce W^X (Write XOR Execute) policies that prevent memory pages from being simultaneously writable and executable. This security measure complicates JIT compilation, which needs to write machine code and then execute it:

| Phase | Memory Protection | Operations Allowed | Required Transitions |
|-------|-------------------|-------------------|---------------------|
| Code Generation | PROT_READ \| PROT_WRITE | Instruction emission, patching | None — writable for modification |
| Address Resolution | PROT_READ \| PROT_WRITE | Jump patching, relocation | None — still needs write access |
| Code Finalization | PROT_READ \| PROT_EXEC | Function execution only | mprotect to remove write permission |
| Runtime Execution | PROT_READ \| PROT_EXEC | Normal function calls | None — executable for performance |

The two-phase compilation approach handles W^X compliance:

1. **Compilation Phase**: All code generation occurs in writable but non-executable memory. The machine code emitter writes instructions, performs jump patching, and completes all address resolution while maintaining write permissions.

2. **Finalization Phase**: After compilation completes, the system calls `mprotect` to change memory protection from writable to executable. This transition is atomic and irreversible — once memory becomes executable, it cannot be modified again.

3. **Error Recovery**: If the protection change fails (due to system policy or resource exhaustion), the system must deallocate the compiled code and fall back to interpretation. The partially-compiled code cannot be executed safely.

#### Memory Cleanup and Resource Management

Proper memory cleanup requires handling both normal deallocation and error recovery scenarios:

```c
typedef struct {
    void* base_address;
    size_t total_size;
    size_t used_size;
    CodeBuffer** allocated_buffers;
    size_t buffer_count;
    int protection_state;
} ExecutableMemoryPool;
```

The cleanup process must handle several edge cases:

- **Partial Compilation Cleanup**: When compilation fails partway through, any allocated executable memory must be deallocated even though no valid compiled code exists. This prevents memory leaks from failed compilation attempts.

- **Exception-Safe Deallocation**: If exceptions occur during memory cleanup (unlikely but possible in signal handlers), the cleanup process must not leave the memory pool in an inconsistent state or leak resources.

- **Cross-Platform Memory Management**: Different operating systems have different requirements for executable memory deallocation. The cleanup code must handle platform-specific requirements while maintaining consistent behavior across systems.

⚠️ **Pitfall: Forgetting Memory Protection Cleanup**

A common error is forgetting to properly deallocate executable memory when compilation fails after memory allocation but before code generation. This leads to memory leaks that gradually exhaust the executable memory pool. The fix is to always use RAII-style resource management where memory allocation automatically registers a cleanup handler that runs regardless of compilation success or failure.

⚠️ **Pitfall: Race Conditions in Memory Pool Management**

Another common mistake is failing to properly synchronize access to the executable memory pool when multiple threads attempt compilation simultaneously. This can lead to double-allocation or corruption of the buffer tracking structures. The solution is to protect all memory pool operations with appropriate locking and use atomic operations for reference counting.

### Common Pitfalls

Several error handling pitfalls frequently occur in JIT compiler implementations:

⚠️ **Pitfall: Assuming Compilation Always Succeeds**

Many JIT implementations fail to properly handle compilation failures, assuming that if bytecode is valid, native compilation will succeed. This assumption breaks when resources are exhausted, when encountering unsupported instruction sequences, or when platform-specific limitations are reached. The fix is to always check compilation results and have fallback paths tested as thoroughly as the success paths.

⚠️ **Pitfall: Leaking Compilation State on Errors**

When compilation fails, it's easy to forget to clean up intermediate state like register allocator structures, jump patch tables, and code buffers. This leads to memory leaks and resource exhaustion over time. The solution is to use structured cleanup with either explicit cleanup functions or RAII-style resource management that ensures cleanup occurs regardless of error conditions.

⚠️ **Pitfall: Inconsistent Exception Semantics Between Modes**

A subtle but critical error is allowing exceptions in JIT-compiled code to have different semantics than the same exceptions in interpreted code. For example, if division by zero throws different exception types or has different stack traces, program behavior becomes unpredictable. The fix is to ensure that all exceptions are converted through a common translation layer that maintains identical semantics.

⚠️ **Pitfall: Signal Handler Reentrancy Issues**

Signal handlers for runtime exceptions can create reentrancy problems if they call functions that aren't signal-safe. This is particularly problematic when the signal handler needs to allocate memory or perform I/O for logging. The solution is to keep signal handlers minimal, using only signal-safe functions, and defer complex error handling to safe execution contexts.

⚠️ **Pitfall: Platform-Specific Memory Management Assumptions**

Different platforms have different requirements for executable memory management. Code that works on Linux might fail on macOS due to different W^X policies, or code that works on x86-64 might fail on ARM due to different cache coherency requirements. The fix is to abstract platform-specific operations through a common interface and test on all target platforms.

### Implementation Guidance

Error handling in JIT compilation requires careful coordination between multiple system components. The following implementation guidance provides concrete approaches for building robust error recovery systems.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Logging | fprintf to stderr with error codes | Structured logging with severity levels and context |
| Signal Handling | Basic signal handlers with global state | Advanced signal handling with signal masks and contexts |
| Memory Management | Simple malloc/free with manual cleanup | Memory pools with automatic resource management |
| Exception Translation | Direct mapping between native and VM exceptions | Full exception context marshaling with stack traces |

#### File Structure

```
jit-compiler/
  src/error_handling/
    compilation_errors.c      ← compilation error recovery
    runtime_exceptions.c      ← runtime exception handling  
    memory_management.c       ← memory allocation and cleanup
    signal_handlers.c         ← native exception signal handling
    error_recovery.h          ← common error handling interface
  src/memory/
    executable_pool.c         ← executable memory pool management
    memory_protection.c       ← W^X policy handling
  tests/error_handling/
    test_compilation_errors.c ← compilation failure test suite
    test_runtime_exceptions.c ← runtime exception test suite
    test_memory_errors.c      ← memory management error tests
```

#### Infrastructure Starter Code

**Error Recovery Infrastructure** (error_recovery.h):
```c
#include <stdint.h>
#include <stddef.h>
#include <signal.h>

typedef enum {
    TRANSLATION_SUCCESS = 0,
    TRANSLATION_ERROR_REGISTER_EXHAUSTION,
    TRANSLATION_ERROR_MEMORY_ALLOCATION,
    TRANSLATION_ERROR_INVALID_BYTECODE,
    TRANSLATION_ERROR_UNSUPPORTED_OPERATION,
    TRANSLATION_ERROR_OPTIMIZATION_FAILURE,
    TRANSLATION_ERROR_ABI_VIOLATION
} TranslationError;

typedef struct {
    CompiledFunction compiled_function;
    size_t native_code_size;
    int register_count_used;
    int spill_count;
    TranslationError error;
    char error_message[256];
    uint32_t failing_bytecode_offset;
    int retry_recommended;
} TranslationResult;

typedef struct {
    void* base_address;
    size_t total_size;
    size_t used_size;
    CodeBuffer** allocated_buffers;
    size_t buffer_count;
    int protection_state;
    uint32_t allocation_failures;
    uint32_t successful_allocations;
} ExecutableMemoryPool;

// Error recovery functions
TranslationResult* create_error_result(TranslationError error, const char* message);
int should_retry_compilation(TranslationError error, int attempt_count);
void cleanup_failed_compilation(TranslationContext* context);
void log_compilation_error(TranslationError error, const char* details);

// Memory management functions
ExecutableMemoryPool* executable_memory_pool_create(size_t initial_size);
CodeBuffer* allocate_compilation_buffer(ExecutableMemoryPool* pool, size_t size);
int finalize_compilation_buffer(ExecutableMemoryPool* pool, CodeBuffer* buffer);
void release_compilation_buffer(ExecutableMemoryPool* pool, CodeBuffer* buffer);
int handle_memory_allocation_failure(ExecutableMemoryPool* pool, size_t requested_size);

// Signal handling functions
void install_jit_signal_handlers(void);
void remove_jit_signal_handlers(void);
int is_address_in_jit_code(void* address);
void convert_native_exception_to_vm(int signal, siginfo_t* info, void* context);
```

**Memory Pool Implementation** (executable_pool.c):
```c
#include "error_recovery.h"
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

ExecutableMemoryPool* executable_memory_pool_create(size_t initial_size) {
    ExecutableMemoryPool* pool = malloc(sizeof(ExecutableMemoryPool));
    if (!pool) return NULL;
    
    // Align size to page boundary
    size_t page_size = getpagesize();
    initial_size = (initial_size + page_size - 1) & ~(page_size - 1);
    
    // Allocate executable memory region
    void* memory = mmap(NULL, initial_size, 
                       PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (memory == MAP_FAILED) {
        free(pool);
        return NULL;
    }
    
    pool->base_address = memory;
    pool->total_size = initial_size;
    pool->used_size = 0;
    pool->allocated_buffers = calloc(64, sizeof(CodeBuffer*));
    pool->buffer_count = 0;
    pool->protection_state = PROT_READ | PROT_WRITE;
    pool->allocation_failures = 0;
    pool->successful_allocations = 0;
    
    return pool;
}

int handle_memory_allocation_failure(ExecutableMemoryPool* pool, size_t requested_size) {
    pool->allocation_failures++;
    
    // Strategy 1: Garbage collect unused buffers
    int freed_buffers = 0;
    for (size_t i = 0; i < pool->buffer_count; i++) {
        CodeBuffer* buffer = pool->allocated_buffers[i];
        if (buffer && buffer->used == 0) {
            release_compilation_buffer(pool, buffer);
            freed_buffers++;
        }
    }
    
    if (freed_buffers > 0) {
        return 1; // Retry allocation
    }
    
    // Strategy 2: Try smaller allocation
    if (requested_size > 4096) {
        return 1; // Caller should retry with smaller size
    }
    
    // Strategy 3: Expand pool if possible
    size_t expansion_size = pool->total_size;
    void* new_memory = mmap(NULL, expansion_size,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (new_memory != MAP_FAILED) {
        // Successfully expanded - caller should retry
        munmap(new_memory, expansion_size); // Clean up test allocation
        return 1;
    }
    
    // All strategies failed
    return 0;
}
```

#### Core Logic Skeleton

**Compilation Error Recovery** (compilation_errors.c):
```c
TranslationResult* execute_compilation_with_recovery(BytecodeFunction* function, 
                                                   uint32_t optimization_level) {
    // TODO 1: Create translation context and allocate compilation buffer
    // TODO 2: Attempt compilation with current optimization level
    // TODO 3: If compilation fails, classify error type and determine retry strategy
    // TODO 4: For retryable errors, attempt compilation with reduced optimization
    // TODO 5: If all compilation attempts fail, create error result with fallback info
    // TODO 6: Clean up all allocated resources regardless of success/failure
    // TODO 7: Update profiling data to prevent immediate retry of failed compilations
    // TODO 8: Log error details for debugging while preserving performance
    
    // Hint: Use should_retry_compilation() to determine retry strategy
    // Hint: Always call cleanup_failed_compilation() in error paths
    // Hint: Set retry_recommended flag based on error type for future attempts
}

int should_retry_compilation(TranslationError error, int attempt_count) {
    // TODO 1: Return immediately false if attempt_count exceeds maximum retries
    // TODO 2: For TRANSLATION_ERROR_REGISTER_EXHAUSTION, retry with reduced optimization
    // TODO 3: For TRANSLATION_ERROR_MEMORY_ALLOCATION, retry after brief delay
    // TODO 4: For TRANSLATION_ERROR_OPTIMIZATION_FAILURE, retry without optimizations
    // TODO 5: For permanent errors (invalid bytecode, unsupported operation), never retry
    // TODO 6: Consider system resource availability in retry decision
    
    // Hint: Use exponential backoff for memory allocation retries
    // Hint: Track retry patterns to avoid infinite retry loops
}

void cleanup_failed_compilation(TranslationContext* context) {
    // TODO 1: Release allocated executable memory buffer
    // TODO 2: Free register allocator state and spill locations
    // TODO 3: Clean up jump patch table and address mappings
    // TODO 4: Reset translation context to safe initial state
    // TODO 5: Log resource cleanup for debugging memory leak issues
    
    // Hint: Set context->compilation_failed = 1 to prevent further use
    // Hint: Use defensive programming - check pointers before freeing
}
```

**Runtime Exception Handling** (runtime_exceptions.c):
```c
void jit_signal_handler(int signal, siginfo_t* info, void* context) {
    // TODO 1: Check if faulting address is within JIT-compiled code regions
    // TODO 2: Extract native execution context (registers, stack, instruction pointer)
    // TODO 3: Map native instruction address back to bytecode offset
    // TODO 4: Convert native exception to appropriate VM exception type
    // TODO 5: Marshal native execution state to interpreter state format
    // TODO 6: Create VM exception object with proper stack trace information
    // TODO 7: Transition execution back to interpreter mode for exception handling
    // TODO 8: Ensure signal handler is reentrant-safe and uses only signal-safe functions
    
    // Hint: Use sigsetjmp/siglongjmp for safe signal handler return
    // Hint: Keep signal handler minimal - defer complex work to safe context
    // Hint: Use atomic operations for any shared state modification
}

void convert_native_exception_to_vm(int signal, siginfo_t* info, void* context) {
    // TODO 1: Map signal type (SIGSEGV, SIGFPE, etc.) to VM exception type
    // TODO 2: Extract fault address and instruction pointer from signal context
    // TODO 3: Use address mapping table to find corresponding bytecode instruction
    // TODO 4: Build VM-compatible stack trace from native call stack
    // TODO 5: Create exception object with bytecode-level debugging information
    // TODO 6: Set up interpreter state for exception propagation
    
    // Hint: Different signals require different exception type mappings
    // Hint: Preserve as much debugging information as possible for stack traces
}
```

#### Language-Specific Hints

**C-Specific Implementation Details:**
- Use `sigaction()` instead of `signal()` for reliable signal handling across platforms
- Use `mmap()` with MAP_ANONYMOUS for executable memory allocation
- Use `mprotect()` to change memory protection for W^X compliance  
- Use `getpagesize()` to ensure memory allocations are page-aligned
- Use `errno` to distinguish different types of system call failures
- Use `__builtin_return_address(0)` to capture call stack information
- Use `setjmp()/longjmp()` for non-local error recovery in signal handlers

**Platform-Specific Considerations:**
- Linux: Use `/proc/self/maps` to verify executable memory allocation
- macOS: Handle additional W^X restrictions and code signing requirements  
- Windows: Use `VirtualAlloc()` with PAGE_EXECUTE_READWRITE for executable memory
- ARM: Use cache coherency instructions after code generation
- Consider using `MAP_JIT` flag on platforms that support it for JIT-specific optimizations

#### Milestone Checkpoints

**After Compilation Error Recovery Implementation:**
```bash
# Test compilation error handling
gcc -o test_errors test_compilation_errors.c compilation_errors.c -ldl
./test_errors

# Expected output:
# Testing register exhaustion recovery... PASS
# Testing memory allocation failure... PASS  
# Testing invalid bytecode handling... PASS
# Testing optimization failure recovery... PASS
# All compilation error tests passed: 4/4
```

**After Runtime Exception Implementation:**
```bash  
# Test runtime exception handling
gcc -o test_exceptions test_runtime_exceptions.c runtime_exceptions.c -ldl
./test_exceptions

# Expected output:
# Testing division by zero exception... PASS
# Testing null pointer dereference... PASS
# Testing stack overflow handling... PASS  
# Testing signal handler installation... PASS
# All runtime exception tests passed: 4/4
```

**After Memory Management Implementation:**
```bash
# Test memory management edge cases
gcc -o test_memory test_memory_errors.c memory_management.c -ldl  
./test_memory

# Expected output:
# Testing executable memory allocation... PASS
# Testing W^X policy compliance... PASS
# Testing memory pool exhaustion... PASS
# Testing cleanup on allocation failure... PASS
# All memory management tests passed: 4/4
```

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| JIT compilation hangs indefinitely | Infinite retry loop on compilation failure | Check compilation attempt counter and error logs | Implement maximum retry limit and proper error classification |
| Segmentation fault during exception handling | Signal handler using non-signal-safe functions | Use gdb to examine signal handler call stack | Replace unsafe functions with signal-safe alternatives |
| Memory leak during repeated compilation failures | Missing cleanup in error paths | Use valgrind to track memory allocation/deallocation | Add cleanup_failed_compilation() calls to all error paths |
| Inconsistent exception behavior between modes | Different exception translation for same errors | Compare exception objects from interpreted vs compiled code | Ensure all exceptions go through common translation layer |
| Compilation fails with EPERM on memory allocation | W^X policy preventing executable memory | Check mmap/mprotect return codes and errno values | Implement two-phase compilation with memory protection changes |


## Testing Strategy

> **Milestone(s):** All milestones — comprehensive testing ensures correctness and performance validation throughout the entire JIT compiler implementation, from basic instruction emission through advanced optimization

The reliability of a JIT compiler fundamentally depends on its ability to produce native code that behaves identically to the original bytecode interpretation while delivering measurable performance improvements. Testing a JIT compiler presents unique challenges that go far beyond traditional software testing: we must verify correctness across two entirely different execution models (interpreted vs. native), validate performance improvements without introducing regressions, and ensure robustness in the face of dynamic code generation and executable memory management.

### Mental Model: Quality Assurance Laboratory

Think of JIT compiler testing as running a sophisticated quality assurance laboratory for a manufacturing process that transforms raw materials (bytecode) into finished products (native machine code). Just as a pharmaceutical lab must verify that a generic drug produces identical therapeutic effects to the original while meeting purity and potency standards, our testing strategy must verify that JIT-compiled code produces identical computational results to interpretation while meeting performance and reliability standards.

The laboratory operates at multiple levels: **chemical analysis** (correctness verification ensures identical behavior at the instruction level), **clinical trials** (performance benchmarking validates therapeutic effectiveness), and **production quality control** (milestone checkpoints ensure each manufacturing stage meets specifications). Each level requires different testing methodologies, measurement tools, and success criteria, but all must work together to provide confidence in the final product.

This multi-layered approach recognizes that JIT compilation involves multiple transformations—bytecode analysis, native code generation, optimization passes, and runtime integration—each of which can introduce subtle bugs that only manifest under specific conditions or workload patterns.

### Correctness Verification: Ensuring JIT Output Matches Interpreter Behavior Exactly

Correctness verification forms the foundation of JIT compiler testing because any deviation from interpreter behavior represents a critical bug that could corrupt program execution. The challenge lies in verifying equivalent behavior across fundamentally different execution models: the interpreter operates on a virtual stack with explicit bytecode dispatch, while the JIT compiler generates native code that directly manipulates processor registers and memory.

#### Reference Implementation Strategy

The correctness verification strategy centers on treating the existing bytecode interpreter as the **reference implementation** against which all JIT-compiled code must be validated. This approach provides several critical advantages: the interpreter behavior is already validated and trusted, comparison can be performed at multiple granularity levels (instruction-by-instruction, function-by-function, or program-by-program), and the same test cases can validate both execution modes without requiring separate test suites.

The verification process operates through systematic comparison at multiple execution boundaries. At the finest granularity, individual bytecode operations are executed in both interpreter and JIT modes, with stack state comparison after each operation. At the function level, complete functions are executed with identical inputs in both modes, comparing final return values and any side effects. At the program level, complete programs run in both modes with comparison of final program state and output.

> **Decision: Dual-Mode Execution Framework**
> - **Context**: Need systematic way to execute identical code paths in both interpreter and JIT modes for comparison
> - **Options Considered**: Separate test harnesses for each mode, shared test framework with mode selection, hybrid approach with automatic fallback
> - **Decision**: Shared test framework with runtime mode selection and automatic comparison
> - **Rationale**: Eliminates test case duplication, ensures identical inputs to both execution modes, enables automated regression detection, supports gradual JIT rollout with interpreter fallback
> - **Consequences**: Requires additional infrastructure for mode coordination, enables comprehensive correctness validation, simplifies test maintenance

| Verification Level | Scope | Comparison Points | Detection Capability |
|-------------------|-------|-------------------|---------------------|
| Operation-Level | Single bytecode instruction | Stack state, register assignments | Instruction translation bugs |
| Expression-Level | Arithmetic/logical expressions | Intermediate and final results | Register allocation errors |
| Function-Level | Complete function execution | Arguments, return values, local variables | Calling convention violations |
| Program-Level | Full program execution | Final state, output, side effects | Integration and optimization bugs |

#### Systematic Test Case Generation

Comprehensive correctness verification requires systematic generation of test cases that exercise all possible code paths and edge conditions in both the bytecode translator and the native code generator. The test generation strategy employs multiple approaches to achieve comprehensive coverage while maintaining manageable test suite size and execution time.

**Boundary value testing** focuses on edge cases that frequently expose off-by-one errors, overflow conditions, and incorrect assumption handling. These test cases target integer arithmetic overflow and underflow conditions, division by zero and remainder operations, maximum and minimum representable values for each data type, stack depth limits and register allocation boundaries, and jump distances that span different instruction encoding formats.

**Combinatorial testing** systematically explores interactions between different bytecode operations and execution contexts. This approach generates test cases covering all combinations of arithmetic operations with different operand types, nested expression structures that stress register allocation, control flow patterns including loops, conditionals, and function calls, and mixed operation sequences that could expose incorrect optimization assumptions.

**Property-based testing** automatically generates large numbers of test cases based on high-level properties that must hold regardless of specific input values. Key properties include **arithmetic commutativity** (a + b equals b + a in both modes), **associativity** ((a + b) + c equals a + (b + c)), **idempotency** (operations that should not change program state), and **invariant preservation** (data structure invariants maintained across function calls).

#### State Comparison Mechanisms

Accurate state comparison requires careful design to handle the fundamental differences between interpreter and JIT execution models while detecting genuine behavioral differences. The interpreter maintains explicit stack data structures and program counters, while JIT-compiled code uses processor registers and native call stacks that are not directly comparable.

The state comparison framework employs **execution boundary normalization**, where comparisons occur at well-defined points where both execution modes can be reduced to equivalent representations. Function entry and exit points provide natural comparison boundaries where arguments and return values can be directly compared. Expression evaluation boundaries allow comparison of intermediate results without requiring identical internal representation.

| Comparison Point | Interpreter State | JIT State | Normalization Strategy |
|------------------|-------------------|-----------|----------------------|
| Function Entry | Argument stack, local variable slots | Register assignments, stack frame | Extract values to common format |
| Function Exit | Return value, modified stack | Return register, caller frame | Compare final values only |
| Exception Points | Exception object, stack trace | Signal context, native trace | Map to common exception representation |
| Memory Operations | Virtual memory state | Native memory state | Compare final memory contents |

**State marshaling** converts execution state from both modes into a common representation suitable for comparison. The marshaling process extracts relevant data (variable values, memory contents, execution results) while filtering out implementation-specific details (register assignments, stack layouts, instruction pointers). This normalization enables precise comparison while accommodating legitimate differences in internal representation.

#### Regression Detection and Isolation

Correctness verification must not only detect when JIT output differs from interpreter behavior, but also provide sufficient diagnostic information to isolate the root cause and guide debugging efforts. The verification framework implements systematic regression detection with detailed error reporting and failure reproduction capabilities.

**Differential execution tracing** records detailed execution traces from both interpreter and JIT modes, enabling post-hoc analysis of where behavior first diverges. The tracing system captures operation sequences, intermediate values, control flow decisions, and memory access patterns at sufficient detail to reconstruct the complete execution path. When behavioral differences are detected, the trace comparison identifies the specific operation where divergence occurred.

The regression isolation process employs **binary search reduction**, automatically reducing complex failing test cases to minimal examples that still exhibit the behavioral difference. This reduction process systematically removes operations, simplifies expressions, and reduces input ranges until the smallest possible test case that demonstrates the bug is identified.

### Performance Benchmarking: Measuring and Validating JIT Compilation Benefits

Performance benchmarking validates that JIT compilation delivers its primary promise: significant execution speed improvements over bytecode interpretation. However, JIT performance measurement presents unique challenges because compilation overhead must be amortized across multiple function invocations, different workload patterns may benefit differently from various optimization strategies, and performance improvements must be sustainable across diverse program structures.

#### Comprehensive Benchmark Suite Design

A robust benchmarking strategy requires carefully designed benchmark programs that represent realistic workload patterns while isolating specific performance characteristics. The benchmark suite must cover different computational patterns, various code structures, and diverse optimization opportunities to provide comprehensive performance validation.

**Micro-benchmarks** isolate specific operations or code patterns to measure fine-grained performance characteristics. These benchmarks focus on arithmetic-intensive loops that benefit from register allocation, function call overhead and calling convention efficiency, memory access patterns and cache locality, and control flow patterns including conditional branches and nested loops. Micro-benchmarks provide precise measurement of specific optimizations but may not reflect real-world performance.

**Macro-benchmarks** evaluate performance on complete programs or substantial computational tasks that reflect realistic usage patterns. These benchmarks include mathematical computations (numerical integration, linear algebra), algorithmic problems (sorting, searching, graph algorithms), and data processing tasks (parsing, transformation, analysis). Macro-benchmarks provide realistic performance validation but make it harder to isolate specific performance factors.

| Benchmark Category | Examples | Performance Focus | Measurement Strategy |
|--------------------|----------|-------------------|---------------------|
| Arithmetic | Mathematical expressions, numerical loops | Register allocation, instruction selection | Operations per second |
| Control Flow | Nested conditionals, loop structures | Branch prediction, jump optimization | Execution time reduction |
| Function Calls | Recursive algorithms, call-heavy code | Calling convention efficiency | Call overhead reduction |
| Mixed Workloads | Complete algorithms, realistic programs | Overall system performance | Total execution speedup |

#### Baseline Measurement and Comparison

Accurate performance measurement requires establishing reliable baselines and controlling for measurement variability that could mask genuine performance differences. JIT compiler performance evaluation must account for compilation overhead, warmup effects, and system-level performance variations that could affect measurement accuracy.

**Baseline establishment** measures interpreter performance under controlled conditions to provide accurate comparison targets. The measurement process includes multiple execution runs to account for system variance, isolation from other system activity that could affect timing, consistent memory and CPU state at measurement start, and measurement of both execution time and resource utilization (memory, CPU cycles).

The performance comparison methodology accounts for **JIT compilation overhead** by measuring both the cost of compilation itself and the break-even point where compilation costs are recovered through faster execution. This analysis determines how many function invocations are required to amortize compilation overhead and validates that frequently executed code achieves net performance improvements.

> **Decision: Amortized Performance Measurement**
> - **Context**: JIT compilation incurs upfront costs that must be amortized across multiple executions to show net benefit
> - **Options Considered**: Single execution measurement, fixed iteration count, adaptive measurement until convergence
> - **Decision**: Adaptive measurement with statistical convergence detection
> - **Rationale**: Single executions may not amortize compilation costs, fixed iterations may under or over-measure, adaptive measurement finds true steady-state performance
> - **Consequences**: More complex measurement infrastructure, more accurate real-world performance prediction, enables optimization of compilation thresholds

#### Performance Regression Detection

Systematic performance regression detection ensures that code changes, optimization improvements, or system updates do not inadvertently reduce JIT compiler effectiveness. The regression detection system must distinguish between genuine performance changes and measurement noise while providing actionable feedback for performance investigations.

**Statistical performance analysis** applies rigorous statistical methods to detect genuine performance changes amidst measurement variability. The analysis process collects multiple performance samples from each benchmark, applies statistical tests to determine if performance differences are significant, calculates confidence intervals for performance improvements, and tracks performance trends over time to detect gradual degradation.

Performance regression detection employs **threshold-based alerting** where performance degradation beyond acceptable limits triggers automatic investigation. The alerting system defines acceptable performance variance ranges for each benchmark, escalates significant performance regressions for investigation, tracks performance improvement trends to validate optimization effectiveness, and maintains historical performance data for long-term trend analysis.

| Performance Metric | Measurement Method | Regression Threshold | Investigation Trigger |
|--------------------|-------------------|---------------------|----------------------|
| Execution Speedup | Time ratio (interpreter/JIT) | < 1.5x improvement | Immediate investigation |
| Compilation Overhead | Time to compile function | > 10x execution time | Threshold tuning needed |
| Memory Usage | Peak memory consumption | > 2x interpreter usage | Memory optimization review |
| Throughput | Operations per second | < 90% of baseline | Performance regression analysis |

#### Optimization Validation

Performance benchmarking must validate that specific optimizations deliver their intended performance benefits while not introducing performance regressions in other areas. Each optimization technique implemented in the JIT compiler requires targeted performance validation to ensure its effectiveness and guide future optimization priorities.

**Optimization-specific benchmarks** measure the performance impact of individual optimization passes by comparing performance with and without specific optimizations enabled. Constant folding validation measures speedup on expression-heavy code with many compile-time constants. Dead code elimination validation uses benchmarks with unreachable code paths to verify elimination effectiveness. Register allocation validation compares performance with different register allocation strategies.

The optimization validation process employs **controlled experimentation** where identical code is compiled with different optimization settings to isolate performance effects. This approach enables precise measurement of optimization benefits, identification of optimization opportunities with highest impact, detection of optimization interactions that may reduce effectiveness, and validation that optimization complexity is justified by performance gains.

### Milestone Checkpoints: Verification Steps After Each Implementation Milestone

Systematic milestone checkpoints provide structured validation points throughout JIT compiler implementation, ensuring that each development phase produces working functionality before proceeding to more complex features. The checkpoint strategy prevents accumulation of bugs across milestones while providing clear success criteria and debugging guidance for each implementation phase.

#### Milestone 1: Basic x86-64 Emitter Checkpoint

The basic x86-64 emitter checkpoint validates that the fundamental machine code generation infrastructure operates correctly before building higher-level translation capabilities. This milestone focuses on executable memory management, instruction encoding accuracy, and basic function call mechanics.

**Executable memory validation** verifies that the memory allocation and protection mechanisms work correctly across different operating systems and hardware configurations. The validation process includes allocation of memory pages with appropriate protection flags (`PROT_READ|PROT_WRITE` initially, `PROT_EXEC` after code generation), verification that generated code can be executed as function pointers, testing of W^X policy compliance on systems that enforce write-xor-execute, and validation of memory cleanup and deallocation.

**Instruction encoding verification** ensures that the x86-64 instruction encoder generates correct binary machine code that produces expected computational results. The verification tests cover basic arithmetic instructions (mov, add, sub) with register and immediate operands, proper REX prefix generation for 64-bit operations, ModR/M byte encoding for different addressing modes, and function return instruction encoding.

| Test Category | Test Cases | Expected Results | Failure Indicators |
|---------------|------------|------------------|-------------------|
| Memory Allocation | mmap with PROT_EXEC | Successfully allocated executable page | Segmentation fault on execution |
| Instruction Encoding | mov rax, 42; ret | Return value 42 when called | Incorrect return value or crash |
| Register Operations | add rax, rbx; ret | Correct sum in return value | Arithmetic errors or wrong registers |
| Immediate Values | mov rdi, 100; ret | Correct immediate value return | Wrong immediate encoding |

#### Milestone 2: Expression JIT Checkpoint

The expression JIT checkpoint validates bytecode-to-native translation for arithmetic expressions, register allocation for stack operations, and control flow translation accuracy. This milestone ensures that the translation pipeline produces correct results for computational code patterns.

**Bytecode translation accuracy** verifies that each bytecode operation translates to functionally equivalent native code sequences. The validation process includes systematic testing of all implemented bytecode operations (ADD, SUB, MUL, DIV, CMP), verification that complex expressions produce identical results to interpreter execution, testing of edge cases including overflow conditions and division by zero, and validation of operation precedence and associativity preservation.

**Register allocation correctness** ensures that the mapping between virtual stack operations and physical processor registers maintains program semantics while optimizing performance. The validation tests cover simple expressions that fit entirely in registers, complex expressions requiring register spilling to memory, nested expressions with varying stack depths, and register reuse patterns that minimize memory traffic.

The expression JIT checkpoint employs **differential testing** where identical expressions execute in both interpreter and JIT modes with comprehensive result comparison. This testing approach automatically detects translation bugs, register allocation errors, and optimization mistakes while providing detailed diagnostic information for debugging.

| Validation Focus | Test Methodology | Success Criteria | Debug Information |
|-----------------|------------------|-------------------|-------------------|
| Operation Translation | Side-by-side execution comparison | Identical results across all test cases | Operation-by-operation trace comparison |
| Register Allocation | Expression complexity stress testing | No register spill errors or corruption | Register state at each operation boundary |
| Control Flow | Branch and jump instruction testing | Correct conditional execution paths | Control flow graph verification |
| Edge Cases | Boundary value and error condition testing | Proper exception handling in both modes | Exception state comparison |

#### Milestone 3: Function JIT and Calling Convention Checkpoint

The function JIT checkpoint validates complete function compilation including calling convention compliance, stack frame management, and seamless integration between JIT-compiled and interpreted code. This milestone ensures that the JIT compiler can handle realistic function call patterns and maintain ABI compliance.

**Calling convention compliance** verifies that generated code follows the System V AMD64 ABI specification for argument passing, return value handling, and register preservation. The validation process includes functions with various argument counts (0-6 register arguments plus stack arguments), different return value types (integers, pointers, void), proper callee-saved register preservation across function calls, and correct stack alignment maintenance (16-byte alignment requirement).

**Stack frame management** validates that function prologue and epilogue code correctly establishes and tears down stack frames while preserving calling function state. The validation tests cover local variable allocation and access patterns, nested function calls with proper frame chaining, exception unwinding through JIT-compiled stack frames, and memory cleanup on function return.

| ABI Compliance Area | Validation Method | Compliance Check | Debugging Tools |
|-------------------|------------------|------------------|-----------------|
| Argument Passing | Multi-argument function calls | Correct values in rdi, rsi, rdx, rcx, r8, r9 | Register state inspection |
| Return Values | Functions returning different types | Proper rax usage and value preservation | Return value tracing |
| Stack Alignment | Call instruction execution | 16-byte alignment before calls | Stack pointer verification |
| Register Preservation | Callee-saved register usage | Proper save/restore of rbx, r12-r15, rbp | Before/after register comparison |

#### Milestone 4: Hot Path Detection and Optimization Checkpoint

The hot path detection checkpoint validates the complete tiered execution system including profiling accuracy, compilation triggering, optimization effectiveness, and performance improvement measurement. This milestone ensures that the JIT compiler delivers its primary value proposition: automatic performance optimization for frequently executed code.

**Profiling system validation** verifies that execution frequency tracking operates correctly without significant performance overhead on interpreted code. The validation process includes accuracy of invocation counting across different execution patterns, proper threshold-based compilation triggering, minimal overhead measurement for counter instrumentation, and correct handling of profiling data across multiple program runs.

**Optimization effectiveness measurement** validates that implemented optimizations deliver measurable performance improvements on appropriate workload patterns. The validation includes performance comparison between optimized and unoptimized JIT code, verification that optimizations preserve program semantics, measurement of optimization overhead versus performance benefit, and validation that optimized code maintains correctness under stress testing.

The final milestone checkpoint employs comprehensive **end-to-end validation** that exercises the complete JIT compilation system under realistic workload conditions. This validation demonstrates that hot path detection correctly identifies performance-critical code sections, JIT compilation produces faster code than interpretation, optimization passes improve performance without introducing bugs, and the system gracefully handles compilation failures and edge cases.

| System Integration Area | Validation Approach | Performance Criteria | Robustness Testing |
|------------------------|--------------------|--------------------|-------------------|
| Hot Path Detection | Execution pattern analysis | Correct identification of frequently executed functions | False positive/negative rate measurement |
| Compilation Triggering | Threshold sensitivity testing | Appropriate compilation timing | Compilation overhead analysis |
| Optimization Pipeline | Before/after performance comparison | Measurable speedup on target workloads | Optimization correctness verification |
| System Robustness | Stress testing and error injection | Graceful degradation on failures | Error recovery validation |

### Implementation Guidance

The testing strategy implementation requires a systematic approach that builds testing capabilities alongside JIT compiler development, ensuring that each milestone can be thoroughly validated before proceeding to more complex functionality.

#### Testing Infrastructure Setup

The testing infrastructure must support dual-mode execution (interpreter and JIT) with comprehensive result comparison and detailed diagnostic reporting. The infrastructure provides the foundation for all correctness verification and performance measurement activities.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Test Framework | Simple assert macros with custom harness | Unity test framework with fixtures |
| Performance Measurement | gettimeofday() timing loops | RDTSC cycle counting with statistical analysis |
| Memory Debugging | Basic mmap/munmap validation | Valgrind integration for memory leak detection |
| Code Coverage | Manual coverage tracking | gcov/lcov for automated coverage reporting |

**Recommended File Structure:**
```
project-root/
  tests/
    correctness/
      test_basic_emission.c      ← Milestone 1 tests
      test_expression_jit.c      ← Milestone 2 tests  
      test_function_jit.c        ← Milestone 3 tests
      test_optimization.c        ← Milestone 4 tests
    performance/
      benchmarks/
        micro_arithmetic.c       ← Arithmetic operation benchmarks
        macro_algorithms.c       ← Complete algorithm benchmarks
      benchmark_runner.c         ← Performance measurement harness
    framework/
      test_harness.c            ← Dual-mode execution framework
      result_comparison.c       ← State comparison utilities
      performance_measurement.c ← Timing and statistics utilities
    integration/
      end_to_end_tests.c       ← Complete system validation
```

#### Correctness Testing Framework

```c
// test_harness.c - Dual-mode execution framework for correctness verification

#include "jit_compiler.h"
#include "interpreter.h"

typedef enum {
    EXECUTION_MODE_INTERPRETER_ONLY,
    EXECUTION_MODE_JIT_ONLY,
    EXECUTION_MODE_COMPARISON
} TestExecutionMode;

typedef struct {
    char* test_name;
    BytecodeFunction* test_function;
    int64_t* input_args;
    int arg_count;
    int64_t expected_result;
    TestExecutionMode mode;
    int repeat_count;
} TestCase;

// Execute test case in both modes and compare results
int run_correctness_test(TestCase* test) {
    // TODO: Initialize interpreter state with test inputs
    // TODO: Execute function in interpreter mode, record result and execution trace
    // TODO: Initialize JIT compiler, compile function to native code
    // TODO: Execute compiled function with identical inputs
    // TODO: Compare results - return 0 for match, 1 for mismatch
    // TODO: On mismatch, generate detailed diagnostic report showing divergence point
}

// Generate systematic test cases for operation coverage
TestCase* generate_arithmetic_tests(int* test_count) {
    // TODO: Create test cases covering all arithmetic operations
    // TODO: Include boundary values (INT64_MAX, INT64_MIN, 0, -1)
    // TODO: Generate combination tests (a+b*c, (a-b)/(c+d), etc.)
    // TODO: Create expression depth tests to stress register allocation
    // Hint: Start with 100+ test cases, expand based on coverage analysis
}
```

#### Performance Benchmarking Infrastructure

```c
// benchmark_runner.c - Performance measurement and analysis

#include <time.h>
#include <sys/time.h>
#include "performance_measurement.h"

typedef struct {
    double* samples;
    int sample_count;
    double mean;
    double std_dev;
    double confidence_interval_95;
} PerformanceStats;

// Measure JIT vs interpreter performance with statistical analysis
PerformanceStats* benchmark_function_performance(BytecodeFunction* func, 
                                               int64_t* inputs, int input_count,
                                               int warmup_runs, int measurement_runs) {
    // TODO: Run function multiple times in interpreter mode, collect timing samples
    // TODO: Compile function to native code, run warmup executions
    // TODO: Run measurement executions in JIT mode, collect timing samples  
    // TODO: Calculate statistical measures (mean, std dev, confidence intervals)
    // TODO: Return performance comparison with speedup ratio
    // Hint: Use RDTSC for high-precision timing, account for CPU frequency scaling
}

// Comprehensive benchmark suite execution
void run_benchmark_suite() {
    // TODO: Load benchmark functions (arithmetic, algorithms, mixed workloads)
    // TODO: Execute each benchmark with statistical measurement
    // TODO: Generate performance report with speedup analysis
    // TODO: Check for performance regressions against historical baselines
    // TODO: Alert on significant performance changes (>10% regression)
}
```

#### Milestone Checkpoint Implementation

```c
// test_basic_emission.c - Milestone 1 validation

// Test executable memory allocation and basic instruction emission
int test_executable_memory_allocation() {
    // TODO: Allocate executable memory using code_buffer_create()
    // TODO: Generate simple function (mov rax, 42; ret) using emit_* functions
    // TODO: Make memory executable with code_buffer_make_executable()
    // TODO: Cast to function pointer and call: CompiledFunction func = (CompiledFunction)buffer->memory
    // TODO: Verify return value is 42, cleanup memory
    // Expected: Successful execution with correct return value
}

// Test x86-64 instruction encoding accuracy
int test_instruction_encoding() {
    CodeBuffer* buffer = code_buffer_create(1024);
    
    // TODO: Emit mov rax, 100
    // TODO: Emit mov rbx, 200  
    // TODO: Emit add rax, rbx
    // TODO: Emit ret
    // TODO: Execute and verify return value is 300
    // TODO: Test with different registers and immediate values
    // Expected: All arithmetic operations produce correct results
}
```

#### Debug Integration and Error Analysis

```c
// debugging_utilities.c - JIT compiler debugging support

// Disassemble generated machine code for inspection
void disassemble_generated_code(CodeBuffer* buffer) {
    // TODO: Integrate with objdump or capstone disassembler
    // TODO: Display human-readable assembly alongside byte values
    // TODO: Annotate with bytecode source mapping when available
    // TODO: Highlight potential encoding errors (incorrect prefixes, wrong operands)
}

// Compare execution traces between interpreter and JIT modes
void analyze_execution_divergence(ExecutionTrace* interpreter_trace, 
                                ExecutionTrace* jit_trace) {
    // TODO: Find first point where traces diverge (operation, stack state, result)
    // TODO: Display side-by-side comparison of execution state
    // TODO: Identify likely root cause (register allocation, instruction encoding, ABI violation)
    // TODO: Generate suggestions for debugging (breakpoint locations, state inspection)
}
```

#### Milestone Checkpoint Validation

**Milestone 1 Checkpoint:**
```bash
# Compile and run basic emission tests
gcc -o test_emission test_basic_emission.c jit_compiler.c -DDEBUG_MODE
./test_emission

# Expected output:
# [PASS] Executable memory allocation
# [PASS] Basic instruction encoding (mov, add, ret)  
# [PASS] Register operand handling
# [PASS] Immediate value encoding
# All basic emission tests passed (4/4)
```

**Milestone 2 Checkpoint:**
```bash
# Run expression JIT correctness tests
gcc -o test_expressions test_expression_jit.c jit_compiler.c interpreter.c
./test_expressions --mode=comparison --verbose

# Expected output:
# [PASS] Arithmetic operations: 47/47 tests
# [PASS] Complex expressions: 23/23 tests  
# [PASS] Register allocation: 15/15 tests
# [PASS] Control flow: 12/12 tests
# Expression JIT correctness validated (97/97 tests)
```

**Performance Validation Commands:**
```bash
# Run performance benchmarks
./benchmark_runner --suite=arithmetic --iterations=1000
# Expected: 2-5x speedup on arithmetic-heavy code

./benchmark_runner --suite=algorithms --statistical-analysis
# Expected: Detailed performance report with confidence intervals
```

**Debugging Workflow Example:**

| Symptom | Likely Cause | Diagnostic Command | Resolution |
|---------|--------------|-------------------|------------|
| Segmentation fault in JIT code | Memory protection error | `gdb ./test --args function_id` | Check mmap flags and W^X compliance |
| Wrong arithmetic results | Instruction encoding error | `./debug_disasm --function=test_add` | Verify REX prefixes and ModR/M bytes |
| Performance slower than interpreter | Missing optimizations | `./benchmark --profile --detailed` | Check compilation overhead and register allocation |
| JIT compilation failure | Unsupported bytecode pattern | `./test --trace-compilation --function=failing_func` | Implement missing bytecode translation |


## Debugging Guide

> **Milestone(s):** All milestones — robust debugging capabilities are essential throughout the JIT compiler implementation, from basic machine code emission through advanced optimization passes

Debugging a JIT compiler presents unique challenges that combine the complexity of systems programming with the intricacies of runtime code generation. Unlike traditional application debugging where source code directly corresponds to execution, JIT debugging involves multiple layers of abstraction: the original bytecode, the intermediate translation state, the generated machine code, and the runtime execution environment. The debugging process requires both high-level reasoning about algorithmic correctness and low-level investigation of binary instruction sequences.

### Mental Model: Detective Investigation with Multiple Crime Scenes

Think of JIT debugging as a detective investigating a complex case with multiple crime scenes. The "crime" is incorrect program behavior, but the evidence is scattered across different locations: the bytecode represents the planned crime, the translation process shows how the plan was executed, the generated machine code reveals what actually happened, and the runtime state shows the aftermath. Just as a detective must piece together evidence from multiple sources to understand what went wrong, a JIT debugger must correlate information across these different representations to identify the root cause of failures.

The detective analogy extends to the investigation process itself. Sometimes the evidence points to an obvious culprit (a clear segmentation fault in generated code), but more often the case requires methodical examination of each piece of evidence, cross-referencing witness statements (execution traces), and following leads that initially seem unrelated (register states that don't match expectations). The key insight is that JIT bugs often manifest in one layer while originating in another, requiring systematic investigation across all abstraction levels.

### Common Bug Patterns

Understanding frequently encountered JIT compiler issues helps developers quickly identify and resolve problems. These patterns emerge from the inherent complexity of runtime code generation and the interaction between multiple system components.

#### Stack Corruption Patterns

Stack corruption represents one of the most common and dangerous categories of JIT bugs. These issues arise from improper stack frame management, incorrect calling convention implementation, or register allocation errors that cause stack pointer misalignment.

⚠️ **Pitfall: Stack Misalignment Crashes**
The System V AMD64 ABI requires the stack pointer to be 16-byte aligned before any call instruction. When JIT-generated code violates this requirement, called functions may crash with cryptic errors in completely unrelated code sections. The bug manifests as sporadic crashes that seem to occur randomly in library functions, making the root cause extremely difficult to trace. The fix involves ensuring that function prologues always align the stack pointer and that call sites maintain proper alignment accounting.

| Bug Pattern | Symptom | Diagnostic Signature | Root Cause |
|-------------|---------|---------------------|-------------|
| Stack pointer misalignment | Crashes in library functions like `printf` | `(rsp + 8) % 16 != 0` before call instructions | Missing stack alignment in function prologue |
| Frame pointer corruption | Local variables contain garbage values | `rbp` points outside valid stack region | Incorrect prologue/epilogue register preservation |
| Return address overwrite | Function returns to invalid address | Stack canary corruption or buffer overflow | Register spilling overwrites stack data |
| Callee-saved register damage | Intermittent variable corruption | Register values differ before/after function calls | Missing register preservation in prologue/epilogue |
| Stack overflow in generated code | Segmentation fault with deep call stack | `rsp` approaches stack limit during execution | Excessive local variable allocation or infinite recursion |

#### Register Allocation Failures

Register allocation bugs occur when the JIT compiler incorrectly manages the mapping between virtual stack positions and physical processor registers. These errors can cause subtle data corruption or obvious crashes depending on the specific failure mode.

⚠️ **Pitfall: Register Reuse Without Spilling**
When expression depth exceeds available registers, the allocator must spill values to memory before reusing registers. Failing to implement proper spilling causes the compiler to overwrite live values, leading to incorrect computation results. The bug is particularly insidious because it may only manifest with complex expressions, making it difficult to reproduce systematically. The fix requires implementing comprehensive liveness tracking and automatic spill generation when register pressure exceeds capacity.

| Bug Pattern | Symptom | Diagnostic Signature | Root Cause |
|-------------|---------|---------------------|-------------|
| Live value overwrite | Incorrect arithmetic results | Register contains wrong value during computation | Missing spill before register reuse |
| Spill location conflicts | Memory corruption in local variables | Multiple values spilled to same stack slot | Incorrect spill slot allocation |
| Register leak across operations | Gradual register exhaustion | Available register count decreases over time | Missing register deallocation after use |
| ABI register violation | Argument values corrupted in function calls | Scratch registers overwritten during call setup | Improper caller-saved register handling |
| Temporary register conflicts | Intermediate values lost during complex expressions | Expression evaluation produces wrong results | Insufficient temporary register management |

#### Instruction Encoding Errors

Machine code generation errors occur when the JIT compiler produces invalid or incorrect x86-64 instruction bytes. These bugs range from obvious invalid opcodes to subtle encoding mistakes that produce functionally different instructions.

⚠️ **Pitfall: Missing REX Prefix for 64-bit Operations**
x86-64 instructions require a REX prefix (typically `0x48`) to operate on 64-bit registers and memory operands. Omitting this prefix causes instructions to operate on 32-bit values instead, leading to incorrect results when working with large integers or pointer values. The bug manifests as truncated results or crashes when dereferencing pointers with high-order bits set. The fix involves systematically adding REX prefixes to all instructions that operate on 64-bit data.

| Bug Pattern | Symptom | Diagnostic Signature | Root Cause |
|-------------|---------|---------------------|-------------|
| Invalid opcode generation | Illegal instruction exception | SIGILL signal with specific instruction address | Incorrect instruction encoding tables |
| Wrong operand size | Truncated arithmetic results | Operations produce 32-bit instead of 64-bit results | Missing REX prefix in instruction encoding |
| Incorrect ModR/M byte | Wrong register or memory operand | Instruction affects different register than intended | Incorrect register encoding in ModR/M byte |
| Branch offset errors | Jump to wrong location | Control flow transfers to unexpected addresses | Incorrect relative offset calculation |
| Memory addressing errors | Segmentation fault during memory access | Invalid effective address calculation | Wrong SIB byte or displacement encoding |

#### Memory Management Issues

Executable memory management presents unique challenges because memory pages must transition between writable (during code generation) and executable (during code execution) states. Memory management bugs can cause crashes, security violations, or subtle corruption issues.

⚠️ **Pitfall: W^X Policy Violations**
Modern operating systems enforce Write XOR Execute (W^X) policies where memory pages cannot be both writable and executable simultaneously. Attempting to modify executable memory or execute writable memory triggers security violations. The bug manifests as permission denied errors or segmentation faults when trying to modify generated code or execute data pages. The fix involves properly managing memory protection states throughout the compilation process, ensuring memory is writable during generation and executable during execution.

| Bug Pattern | Symptom | Diagnostic Signature | Root Cause |
|-------------|---------|---------------------|-------------|
| Permission denied during code modification | `mprotect()` fails with `EPERM` | Cannot change memory protection flags | W^X policy prevents writable+executable memory |
| Segmentation fault during code execution | Crash when jumping to generated code | Memory protection flags don't include `PROT_EXEC` | Memory not marked executable before use |
| Memory leak in code buffers | Gradual memory consumption increase | Allocated executable memory never freed | Missing cleanup of compilation buffers |
| Code buffer overflow | Corruption of adjacent memory regions | Generated code exceeds allocated buffer size | Insufficient buffer size estimation |
| Memory fragmentation | Allocation failures despite available memory | `mmap()` fails even with sufficient virtual address space | Poor memory pool management strategy |

#### Calling Convention Violations

Calling convention bugs occur when JIT-generated code fails to properly implement the System V AMD64 ABI. These violations can cause crashes, incorrect results, or intermittent failures that are extremely difficult to debug.

⚠️ **Pitfall: Incorrect Argument Register Usage**
The System V AMD64 ABI specifies that integer arguments are passed in registers `rdi`, `rsi`, `rdx`, `rcx`, `r8`, and `r9` in order. Using the wrong registers or incorrect order causes functions to receive garbage values as arguments. The bug manifests as functions operating on incorrect data, leading to wrong results or crashes. The fix involves implementing systematic argument register allocation that follows the ABI specification exactly.

| Bug Pattern | Symptom | Diagnostic Signature | Root Cause |
|-------------|---------|---------------------|-------------|
| Wrong argument values | Functions receive garbage parameters | Arguments don't match expected values in called function | Incorrect argument register allocation |
| Return value corruption | Function results are wrong or corrupted | `rax` doesn't contain expected return value | Improper return value handling |
| Register preservation failure | Caller's register values corrupted after function call | Callee-saved registers modified without restoration | Missing register save/restore in prologue/epilogue |
| Variadic function call errors | Crashes when calling `printf` or similar functions | `al` register not set to floating-point argument count | Missing variadic function call convention handling |
| Red zone violations | Stack corruption in leaf functions | Local data overwritten by signal handlers or interrupts | Incorrect use of stack red zone in generated code |

### Debugging Tools and Techniques

Effective JIT debugging requires a combination of specialized tools and systematic techniques that address the unique challenges of runtime code generation. The debugging process must handle multiple representation levels simultaneously while providing visibility into the dynamic compilation process.

#### Machine Code Inspection Tools

Understanding generated machine code requires tools that can disassemble binary instructions and correlate them with high-level operations. These tools bridge the gap between human-readable assembly and the binary instruction sequences produced by the JIT compiler.

| Tool Category | Simple Option | Advanced Option | Use Case |
|---------------|---------------|-----------------|----------|
| Disassembler | `objdump -D` on code dumps | Capstone Engine integration | Converting binary instructions to assembly mnemonics |
| Debugger | GDB with JIT debugging support | Intel Pin or DynamoRIO | Interactive debugging of generated code |
| Hex dump | `xxd` or `hexdump` utilities | Custom binary analysis tools | Examining raw instruction bytes |
| Memory analyzer | Valgrind with JIT support | AddressSanitizer with JIT mode | Detecting memory errors in generated code |
| Performance profiler | `perf` with JIT symbol support | Intel VTune with dynamic code analysis | Identifying performance bottlenecks in JIT code |

The most fundamental debugging technique involves generating human-readable disassembly from the binary machine code produced by the JIT compiler. This process requires implementing a disassembly function that uses libraries like Capstone Engine to convert instruction bytes back into assembly mnemonics.

> **Design Insight**: Implementing built-in disassembly support during development provides immediate visibility into code generation correctness. Rather than relying on external tools, integrating disassembly directly into the JIT compiler enables automatic verification of instruction encoding and immediate debugging feedback during compilation failures.

#### Execution State Comparison

Since JIT-compiled code must produce identical results to bytecode interpretation, systematic comparison of execution states provides the most reliable method for detecting correctness issues. This technique involves running identical code through both execution modes and identifying the first point where results diverge.

The comparison process requires capturing detailed execution traces from both the interpreter and JIT-compiled code, then analyzing these traces to identify divergence points. The traces must include register states, memory contents, and intermediate computation results at comparable execution points.

| Comparison Level | Information Captured | Debugging Value | Implementation Complexity |
|-----------------|---------------------|-----------------|-------------------------|
| Final results only | Function return values and side effects | Basic correctness verification | Simple - single comparison point |
| Operation-level traces | Results after each bytecode operation | Identifies first incorrect operation | Moderate - requires instrumentation |
| Instruction-level traces | Register and memory state after each instruction | Pinpoints exact failure location | Complex - high overhead and data volume |
| Register allocation traces | Virtual-to-physical register mapping throughout execution | Debugs register allocation correctness | Moderate - compiler-internal data |
| Memory access traces | All memory reads and writes with addresses and values | Identifies memory corruption and access violations | Complex - requires memory instrumentation |

#### Signal Handler Integration

JIT-compiled code can trigger various system signals (segmentation faults, illegal instructions, arithmetic exceptions) that must be properly handled and converted into meaningful debugging information. Signal handler integration provides structured exception handling for generated code.

The signal handling system must distinguish between signals originating from JIT-compiled code versus other program components, extract meaningful context from the signal delivery information, and provide debugging information that correlates native code addresses with source bytecode locations.

> **Critical Implementation Detail**: Signal handlers for JIT-compiled code must be extremely careful about async-signal-safety. They cannot call malloc, printf, or most library functions safely. The signal handler should only capture essential debugging information and defer detailed analysis to a safe execution context.

#### Memory Access Debugging

Memory-related bugs in JIT-compiled code can be particularly difficult to diagnose because they may corrupt data structures used by other program components. Specialized memory debugging techniques help isolate and identify these issues.

Memory access debugging involves instrumenting generated code to validate all memory operations, implementing guard pages around code buffers to detect overruns, and maintaining detailed allocation tracking for executable memory regions.

| Memory Debug Technique | Detection Capability | Performance Impact | Implementation Effort |
|------------------------|---------------------|-------------------|----------------------|
| Guard pages around code buffers | Buffer overrun detection | Low - only affects overruns | Simple - use `mprotect()` for guard regions |
| Memory access instrumentation | Invalid pointer dereference detection | High - overhead on every access | Complex - requires code generation modification |
| Address sanitizer integration | Use-after-free and buffer overflow detection | Moderate - runtime checking overhead | Moderate - requires compiler integration |
| Executable memory tracking | Memory leak and double-free detection | Low - only affects allocation/deallocation | Simple - maintain allocation metadata |
| Stack canary insertion | Stack corruption detection | Low - single check per function | Moderate - requires prologue/epilogue modification |

#### Performance Debugging and Profiling

Performance issues in JIT compilers can be subtle because they involve the interplay between compilation overhead and execution speedup. Performance debugging requires measuring both the cost of compilation and the benefit of optimized execution.

The performance analysis process involves establishing baseline performance using pure interpretation, measuring compilation overhead for each function, tracking execution frequency to validate hot path detection, and analyzing the amortized performance benefit over multiple function invocations.

Performance debugging must account for the fact that JIT compilation introduces upfront costs that are only amortized over multiple executions. Short-running programs may perform worse with JIT compilation enabled, while long-running programs should show significant improvement.

### Symptom-Cause-Fix Reference

This reference provides a systematic approach to diagnosing and resolving common JIT compiler issues. Each entry follows the pattern: observable symptom, likely underlying cause, diagnostic steps, and specific fix recommendations.

#### Runtime Crashes and Exceptions

Runtime crashes represent the most obvious category of JIT bugs, but the symptoms often provide limited information about root causes. Systematic diagnosis requires correlating crash locations with generated code regions and analyzing the execution context at failure time.

| Symptom | Likely Cause | Diagnostic Steps | Fix Strategy |
|---------|--------------|------------------|--------------|
| Segmentation fault when jumping to generated code | Memory not marked executable | Check memory protection flags with `cat /proc/PID/maps` | Call `mprotect()` with `PROT_EXEC` before execution |
| Illegal instruction exception at known JIT address | Invalid instruction encoding | Disassemble code at crash address, verify instruction bytes | Fix instruction encoding tables and REX prefix usage |
| Stack overflow in recursive JIT code | Missing base case or incorrect stack frame size | Use `ulimit -s` to check stack limit, examine frame size | Implement proper recursion termination and optimize frame layout |
| Random crashes in library functions | Stack misalignment before function calls | Check `(rsp + 8) % 16` before call instructions | Add stack alignment code in function prologue |
| Crashes when accessing local variables | Frame pointer corruption or incorrect offset calculation | Compare `rbp` with expected stack frame layout | Fix prologue generation and local variable offset calculation |

#### Incorrect Computation Results

Computational errors in JIT-compiled code are often more challenging to diagnose than crashes because the program continues executing with wrong data. These bugs require systematic comparison between interpreter and JIT execution results.

| Symptom | Likely Cause | Diagnostic Steps | Fix Strategy |
|---------|--------------|------------------|--------------|
| Arithmetic operations return wrong results | Missing REX prefix causing 32-bit instead of 64-bit operations | Compare JIT results with interpreter for same inputs | Add REX prefix (`0x48`) to all 64-bit arithmetic instructions |
| Function arguments have incorrect values | Wrong argument register allocation | Print register values in called function, compare with ABI specification | Fix argument passing to use `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9` in order |
| Local variables contain garbage data | Register spilling overwrites stack data | Check spill slot allocation for conflicts | Implement proper spill slot management with unique addresses |
| Conditional branches take wrong path | Incorrect condition code translation | Trace branch decisions in both interpreter and JIT modes | Fix condition flag translation from bytecode comparisons to x86-64 flags |
| Loop results differ from interpreter | Register allocation errors in loop bodies | Compare register states at loop entry and exit | Implement proper register liveness analysis for loop-carried values |

#### Memory Management Failures

Memory management issues in JIT compilers can cause resource leaks, security violations, or corruption of other program components. These problems require careful tracking of executable memory allocation and deallocation.

| Symptom | Likely Cause | Diagnostic Steps | Fix Strategy |
|---------|--------------|------------------|--------------|
| `mmap()` fails with insufficient memory | Executable memory leaks from failed compilations | Check `/proc/PID/maps` for unreleased executable regions | Implement cleanup of failed compilation buffers |
| Permission denied when modifying generated code | Attempting to modify executable memory | Verify memory protection state during modification attempts | Use separate writable and executable memory regions |
| Gradual memory consumption increase | Code buffers not released after function replacement | Monitor executable memory usage over time | Implement reference counting and cleanup for replaced functions |
| Memory corruption in interpreter data structures | Generated code writes outside allocated buffers | Use memory debugging tools to detect buffer overruns | Add bounds checking to code buffer allocation and emission |
| Address space fragmentation | Poor memory pool management | Analyze memory allocation patterns and fragmentation | Implement memory pooling with proper size classes |

#### Calling Convention and ABI Issues

Calling convention bugs are particularly insidious because they may only manifest when calling certain types of functions or passing specific argument patterns. These issues require deep understanding of the System V AMD64 ABI specification.

| Symptom | Likely Cause | Diagnostic Steps | Fix Strategy |
|---------|--------------|------------------|--------------|
| Variadic functions like `printf` crash or produce garbage output | Missing floating-point argument count in `al` register | Trace register state before variadic function calls | Set `al` to count of floating-point arguments before variadic calls |
| Function returns wrong values | Return value not properly placed in `rax` register | Check `rax` contents immediately after function execution | Ensure function epilogue places return value in `rax` |
| Caller's register values corrupted after JIT function calls | Callee-saved registers not preserved | Compare register values before and after function calls | Implement proper save/restore of `rbx`, `r12`-`r15`, `rbp` in prologue/epilogue |
| Stack corruption after function returns | Red zone violations or improper stack cleanup | Examine stack contents before and after function calls | Avoid using red zone in non-leaf functions, ensure proper stack cleanup |
| JIT-to-interpreter transitions fail | State marshaling errors between execution modes | Trace state conversion at transition boundaries | Implement proper conversion between interpreter stack and native registers |

#### Performance and Optimization Issues

Performance problems in JIT compilers can be counterintuitive because compilation overhead may outweigh execution benefits for certain usage patterns. Performance debugging requires systematic measurement and analysis of both compilation costs and execution improvements.

| Symptom | Likely Cause | Diagnostic Steps | Fix Strategy |
|---------|--------------|------------------|--------------|
| JIT-enabled programs run slower than interpreter-only | Compilation threshold too low, compiling cold code | Profile compilation frequency and execution counts | Increase compilation thresholds, improve hot path detection |
| Hot functions never get compiled | Compilation threshold too high or profiling disabled | Verify profiling counters and threshold logic | Lower compilation thresholds, ensure profiling instrumentation works |
| Compilation takes excessive time | Inefficient optimization passes or register allocation | Profile compilation time breakdown by component | Optimize slow compilation phases, implement compilation timeouts |
| Generated code performs poorly | Missing optimizations or suboptimal instruction selection | Compare generated code with hand-optimized assembly | Implement additional optimization passes, improve instruction selection |
| Memory usage grows excessively with JIT enabled | Compiled code cache not implementing eviction policy | Monitor code cache size and hit rates | Implement LRU or similar cache eviction policy |

#### Integration and System-Level Issues

System-level integration problems occur when the JIT compiler interacts incorrectly with other program components, the operating system, or external libraries. These issues often manifest as intermittent failures that are difficult to reproduce consistently.

| Symptom | Likely Cause | Diagnostic Steps | Fix Strategy |
|---------|--------------|------------------|--------------|
| Program hangs during JIT compilation | Deadlock in compilation pipeline or resource contention | Use `gdb` to examine thread states and lock ownership | Implement proper locking order, add compilation timeouts |
| JIT compilation fails silently | Error handling swallows compilation failures | Add verbose error reporting and logging | Implement comprehensive error reporting with fallback to interpretation |
| Inconsistent behavior between program runs | Race conditions in compilation or code installation | Run under thread sanitizer, examine concurrent access patterns | Add proper synchronization around compilation and code installation |
| Platform-specific crashes | Endianness issues or architecture-specific instruction encoding | Test on different architectures, examine instruction byte sequences | Implement platform-specific instruction encoding validation |
| Integration test failures | Mismatch between JIT and interpreter behavior for edge cases | Create minimal test cases that isolate the behavioral differences | Fix edge case handling in bytecode translation |

> **Debugging Philosophy**: The most effective JIT debugging approach combines systematic methodology with deep technical understanding. Rather than randomly trying fixes, successful debugging requires understanding the layered architecture: bytecode semantics, translation correctness, machine code validity, and system integration. Each layer provides evidence that must be synthesized into a coherent understanding of the failure mode.

### Implementation Guidance

This guidance provides concrete tools and techniques for implementing effective debugging capabilities in your JIT compiler. The focus is on building debugging infrastructure that provides maximum diagnostic value with minimal performance impact during normal execution.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Disassembly | Manual instruction decoding with lookup tables | Capstone Engine library integration |
| Memory debugging | Manual guard pages with `mprotect()` | AddressSanitizer integration |
| Performance profiling | Simple timestamp-based measurements | Integration with `perf` or Intel VTune |
| Signal handling | Basic `sigaction()` with minimal handlers | Complete exception context capture and analysis |
| Execution tracing | Printf-based logging with conditional compilation | Structured trace buffer with binary format |

#### Recommended File Structure

```
jit-compiler/
  src/
    debug/
      debug.c                    ← main debugging interface
      debug.h                    ← debugging function declarations
      disassembler.c            ← machine code disassembly
      memory_debug.c            ← executable memory debugging
      signal_handler.c          ← JIT signal handling
      trace.c                   ← execution trace recording
      performance.c             ← performance measurement utilities
    test/
      debug_test.c              ← debugging functionality tests
      crash_reproduction.c      ← systematic crash reproduction
    tools/
      code_inspector.c          ← standalone code analysis tool
      trace_analyzer.c          ← trace comparison and analysis
  include/
    debug_config.h              ← debugging configuration macros
```

#### Infrastructure Starter Code

**Complete Signal Handler Implementation:**

```c
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include "debug.h"

// Global JIT debugging context - async-signal-safe access only
static struct {
    void* jit_code_start;
    void* jit_code_end;
    int signal_handling_enabled;
    volatile sig_atomic_t crash_detected;
} debug_context = {0};

// Initialize JIT signal handling - call during JIT compiler startup
void install_jit_signal_handlers() {
    struct sigaction sa;
    sa.sa_sigaction = jit_signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    
    sigaction(SIGSEGV, &sa, NULL);  // Segmentation fault
    sigaction(SIGILL, &sa, NULL);   // Illegal instruction
    sigaction(SIGFPE, &sa, NULL);   // Floating point exception
    sigaction(SIGBUS, &sa, NULL);   // Bus error
    
    debug_context.signal_handling_enabled = 1;
}

// Check if address belongs to JIT-generated code
int is_address_in_jit_code(void* addr) {
    return (addr >= debug_context.jit_code_start && 
            addr < debug_context.jit_code_end);
}

// Update JIT code region bounds when allocating new executable memory
void update_jit_code_region(void* start, void* end) {
    debug_context.jit_code_start = start;
    debug_context.jit_code_end = end;
}

// Signal handler for JIT-related crashes - async-signal-safe only
void jit_signal_handler(int signal, siginfo_t* info, void* context) {
    ucontext_t* uc = (ucontext_t*)context;
    void* fault_addr = info->si_addr;
    void* instruction_addr = (void*)uc->uc_mcontext.gregs[REG_RIP];
    
    // Only handle signals from JIT-generated code
    if (!is_address_in_jit_code(instruction_addr)) {
        // Re-raise signal with default handler
        signal(signal, SIG_DFL);
        raise(signal);
        return;
    }
    
    debug_context.crash_detected = 1;
    
    // Write crash information to stderr (async-signal-safe)
    const char* signal_name = (signal == SIGSEGV) ? "SIGSEGV" :
                             (signal == SIGILL) ? "SIGILL" :
                             (signal == SIGFPE) ? "SIGFPE" :
                             (signal == SIGBUS) ? "SIGBUS" : "UNKNOWN";
    
    write(STDERR_FILENO, "\nJIT CRASH: ", 12);
    write(STDERR_FILENO, signal_name, strlen(signal_name));
    write(STDERR_FILENO, " at instruction ", 16);
    
    // Convert addresses to hex strings (async-signal-safe)
    char addr_buf[32];
    snprintf(addr_buf, sizeof(addr_buf), "%p", instruction_addr);
    write(STDERR_FILENO, addr_buf, strlen(addr_buf));
    
    // Terminate process - cannot safely continue after JIT crash
    _exit(128 + signal);
}
```

**Complete Disassembly Support:**

```c
#include "debug.h"

// Simple x86-64 instruction disassembler for common JIT instructions
typedef struct {
    uint8_t opcode;
    const char* mnemonic;
    int operand_count;
    const char* format;
} InstructionInfo;

static const InstructionInfo instruction_table[] = {
    {0x48, "REX.W", 0, ""},
    {0xB8, "mov", 2, "rax, 0x%lx"},
    {0x01, "add", 2, "%s, %s"},
    {0x29, "sub", 2, "%s, %s"},
    {0x39, "cmp", 2, "%s, %s"},
    {0xC3, "ret", 0, ""},
    {0xE8, "call", 1, "0x%x"},
    {0x74, "je", 1, "0x%x"},
    {0x75, "jne", 1, "0x%x"},
    // Add more instructions as needed
};

static const char* register_names[] = {
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
};

// Disassemble generated code buffer for debugging
void disassemble_generated_code(CodeBuffer* buffer) {
    if (!buffer || !buffer->memory) {
        printf("No code buffer to disassemble\n");
        return;
    }
    
    printf("Disassembly of generated code (%zu bytes):\n", buffer->used);
    printf("Address          Bytes                Instruction\n");
    printf("===============================================\n");
    
    uint8_t* code = (uint8_t*)buffer->memory;
    size_t offset = 0;
    
    while (offset < buffer->used) {
        printf("%p: ", (void*)((uintptr_t)buffer->memory + offset));
        
        // Print instruction bytes (up to 8 bytes)
        size_t byte_count = (buffer->used - offset < 8) ? buffer->used - offset : 8;
        for (size_t i = 0; i < byte_count; i++) {
            printf("%02x ", code[offset + i]);
        }
        for (size_t i = byte_count; i < 8; i++) {
            printf("   ");  // Padding for alignment
        }
        
        // Decode instruction
        uint8_t opcode = code[offset];
        const InstructionInfo* info = NULL;
        
        // Look up instruction in table
        for (size_t i = 0; i < sizeof(instruction_table) / sizeof(instruction_table[0]); i++) {
            if (instruction_table[i].opcode == opcode) {
                info = &instruction_table[i];
                break;
            }
        }
        
        if (info) {
            printf("%-8s", info->mnemonic);
            
            // Decode operands based on instruction format
            if (info->operand_count == 0) {
                printf("\n");
                offset += 1;
            } else if (opcode == 0xB8 && offset + 8 < buffer->used) {
                // mov rax, immediate (64-bit)
                int64_t immediate = *(int64_t*)(code + offset + 1);
                printf("rax, 0x%lx\n", immediate);
                offset += 9;  // 1 byte opcode + 8 byte immediate
            } else {
                printf("(operand decoding not implemented)\n");
                offset += 1;
            }
        } else {
            printf("(unknown opcode: 0x%02x)\n", opcode);
            offset += 1;
        }
    }
    
    printf("===============================================\n\n");
}
```

#### Core Logic Skeleton Code

**Main Debugging Interface:**

```c
// Initialize debugging subsystem with configuration options
int debug_initialize(int enable_tracing, int enable_disassembly, int enable_memory_debug) {
    // TODO 1: Initialize signal handlers for JIT crash detection
    // TODO 2: Set up execution tracing if requested
    // TODO 3: Enable memory debugging features if requested
    // TODO 4: Initialize performance monitoring
    // TODO 5: Create trace buffers and analysis structures
    // Hint: Call install_jit_signal_handlers() first for crash handling
}

// Record execution trace entry for comparison debugging
void debug_record_execution(uint32_t bytecode_offset, int64_t stack_values[], int stack_depth) {
    // TODO 1: Check if tracing is enabled and buffer has space
    // TODO 2: Capture current execution state (registers, stack, PC)
    // TODO 3: Store trace entry with timestamp and execution context
    // TODO 4: Handle trace buffer overflow by cycling old entries
    // Hint: Use lightweight copying - avoid expensive operations in hot path
}

// Compare execution traces between interpreter and JIT modes
void debug_analyze_execution_divergence(ExecutionTrace* interpreter_trace, 
                                      ExecutionTrace* jit_trace) {
    // TODO 1: Find first point where execution traces diverge
    // TODO 2: Analyze stack states, register values, and memory contents at divergence
    // TODO 3: Print detailed comparison showing expected vs actual values
    // TODO 4: Suggest potential causes based on divergence patterns
    // TODO 5: Generate debugging report with actionable information
    // Hint: Start comparison from beginning and stop at first difference
}
```

**Memory Debugging Support:**

```c
// Validate executable memory allocation and protection
int debug_validate_executable_memory(void* memory, size_t size) {
    // TODO 1: Check memory alignment (should be page-aligned)
    // TODO 2: Verify memory protection flags using /proc/self/maps or mprotect
    // TODO 3: Test memory accessibility (read/execute but not write)
    // TODO 4: Validate memory is within expected address ranges
    // TODO 5: Check for memory leaks by tracking allocation/deallocation
    // Hint: Use mprotect with PROT_READ|PROT_EXEC to test protection
}

// Install guard pages around code buffers to detect overruns
int debug_install_guard_pages(CodeBuffer* buffer) {
    // TODO 1: Allocate guard pages before and after code buffer
    // TODO 2: Set guard pages to no access (PROT_NONE) to trigger faults
    // TODO 3: Register guard page addresses for signal handler recognition
    // TODO 4: Update buffer metadata to include guard page information
    // Hint: Use mmap with MAP_ANONYMOUS and mprotect with PROT_NONE
}

// Check for code buffer corruption or overruns
int debug_verify_code_buffer_integrity(CodeBuffer* buffer) {
    // TODO 1: Verify buffer metadata is consistent (size, used, capacity)
    // TODO 2: Check guard pages haven't been modified
    // TODO 3: Validate code buffer contents haven't been corrupted
    // TODO 4: Ensure buffer bounds haven't been exceeded
    // TODO 5: Report any integrity violations with detailed information
    // Hint: Store checksums of guard pages and verify them periodically
}
```

#### Language-Specific Hints

**C-Specific Debugging Techniques:**
- Use `mmap()` with `MAP_ANONYMOUS | MAP_PRIVATE` for executable memory allocation
- Signal handlers must be async-signal-safe - avoid `printf`, `malloc`, or mutex operations
- Use `write()` instead of `printf()` in signal handlers for crash reporting
- Implement guard pages with `mprotect(addr, size, PROT_NONE)` to detect buffer overruns
- Use `__builtin_return_address(0)` to capture call stack information
- Link with `-rdynamic` to get symbol names in backtraces

**Memory Management Debugging:**
- Check memory protection with `cat /proc/PID/maps` to verify executable regions
- Use `valgrind --tool=memcheck` with JIT support enabled for memory error detection
- Implement allocation tracking with file/line information using `__FILE__` and `__LINE__`
- Test W^X compliance by attempting to write to executable memory (should fail)

**Performance Debugging:**
- Use `clock_gettime(CLOCK_MONOTONIC, &ts)` for high-resolution timing measurements
- Implement statistical analysis to detect genuine performance changes vs noise
- Account for compilation overhead when measuring JIT performance benefits
- Profile both interpretation and compilation time separately

#### Milestone Checkpoints

**After Milestone 1 (Basic x86-64 Emitter):**
- Verify generated machine code with `objdump -D -b binary -m i386:x86-64 code.bin`
- Test executable memory allocation with `cat /proc/self/maps | grep rwx`
- Confirm basic instructions execute correctly: `mov`, `add`, `ret`
- Check instruction encoding matches x86-64 reference manual

**After Milestone 2 (Expression JIT):**
- Compare arithmetic results between interpreter and JIT for identical inputs
- Verify register allocation doesn't corrupt values during complex expressions
- Test edge cases: integer overflow, division by zero, maximum expression depth
- Confirm conditional jumps take correct branches based on comparison results

**After Milestone 3 (Function JIT and Calling Convention):**
- Test function calls between JIT and interpreter modes in both directions
- Verify stack alignment with `(rsp + 8) % 16 == 0` before call instructions
- Confirm argument passing follows System V AMD64 ABI specification
- Test callee-saved register preservation across function calls

**After Milestone 4 (Hot Path Detection and Optimization):**
- Verify hot functions get compiled when execution count exceeds threshold
- Measure performance improvement on compute-intensive test programs
- Test optimization correctness by comparing optimized vs unoptimized results
- Confirm compilation overhead is amortized over multiple function invocations

#### Debugging Tips by Category

| Problem Category | Symptom | Investigation Method | Quick Fix |
|-----------------|---------|---------------------|-----------|
| **Instruction Encoding** | Illegal instruction crash | Disassemble at crash address | Verify REX prefix and ModR/M byte |
| **Memory Management** | Permission denied | Check `/proc/PID/maps` | Call `mprotect()` with correct flags |
| **Register Allocation** | Wrong computation results | Compare register states | Fix register spilling logic |
| **Calling Convention** | Function argument errors | Print argument registers | Follow System V AMD64 ABI exactly |
| **Stack Management** | Stack corruption | Check alignment and frame size | Implement proper prologue/epilogue |
| **Control Flow** | Wrong branch destinations | Trace jump target calculations | Fix relative offset computation |

**Emergency Debugging Checklist:**
1. Is the crash in JIT-generated code? Check instruction pointer against code buffer range
2. Are instruction bytes valid? Disassemble around crash location
3. Is stack properly aligned? Check `(rsp + 8) % 16 == 0` before calls  
4. Are registers preserved correctly? Compare before/after function calls
5. Is memory protection correct? Verify executable flag is set
6. Do results match interpreter? Run identical code in both modes


## Future Extensions

> **Milestone(s):** Beyond Milestone 4 — these extensions represent advanced capabilities that build upon the complete JIT compiler foundation

The journey from a basic JIT compiler to a production-grade runtime system involves numerous architectural extensions and optimizations. Think of the current implementation as a solid foundation — like a well-built house with good bones — that can support increasingly sophisticated additions. Each extension category represents a different dimension of growth: broader hardware support, deeper optimization capabilities, and more advanced runtime features.

The extensions discussed in this section are not merely academic exercises but represent the evolutionary path that real-world JIT compilers follow. Modern systems like the Java HotSpot VM, V8 JavaScript engine, and .NET CoreCLR have all traversed similar upgrade paths, starting with single-architecture implementations and gradually expanding their capabilities. Understanding these potential extensions provides insight into the broader landscape of runtime optimization technology and helps architects plan for future requirements.

The key insight is that each extension builds incrementally upon existing infrastructure. The machine code emitter's abstraction layer enables multi-architecture support. The profiling system's execution counters provide the data foundation for advanced optimizations. The compilation pipeline's modular design accommodates new optimization passes. This architectural foresight — designing core components to support future growth — distinguishes professional-grade systems from academic prototypes.

### Multi-Architecture Support: Extending beyond x86-64 to ARM, RISC-V, and other targets

The transition from a single-architecture JIT compiler to a multi-architecture system represents one of the most significant architectural challenges in runtime system development. Think of this transformation like converting a specialized craftsperson's workshop — optimized for creating one specific type of product — into a flexible manufacturing facility capable of producing the same quality output across multiple product lines while maintaining efficiency and quality standards.

The fundamental challenge lies in abstracting away architecture-specific details while preserving the performance characteristics that make JIT compilation worthwhile. Different processor architectures have vastly different instruction sets, register configurations, calling conventions, and performance characteristics. ARM processors typically have more registers but different instruction encoding formats. RISC-V provides a cleaner instruction set but lacks some of the complex addressing modes available in x86-64. Each architecture requires specialized knowledge to generate optimal code.

#### Architecture Abstraction Layer Design

The foundation for multi-architecture support begins with designing an abstraction layer that captures the essential concepts of machine code generation while hiding architecture-specific implementation details. This abstraction must be powerful enough to express the optimization opportunities available on each target platform while remaining simple enough that bytecode translation logic can remain largely architecture-independent.

The core abstraction revolves around the concept of **virtual instructions** — an intermediate representation that sits between bytecode operations and actual machine instructions. These virtual instructions capture the semantic intent of operations (load value, perform arithmetic, store result) without committing to specific instruction encodings or register assignments. The architecture-specific backends then translate these virtual instructions into optimal native code sequences for their target platform.

> **Decision: Virtual Instruction Intermediate Representation**
> - **Context**: Multiple architectures have different instruction sets, register counts, and optimization opportunities, requiring a way to share translation logic while enabling architecture-specific code generation
> - **Options Considered**: 
>   - Direct architecture-specific translation for each target
>   - Shared high-level IR with architecture-specific backends
>   - LLVM integration for multi-architecture support
> - **Decision**: Custom virtual instruction IR with pluggable architecture backends
> - **Rationale**: Provides the right abstraction level for JIT compilation while maintaining control over code generation quality and compilation speed
> - **Consequences**: Enables code reuse for bytecode analysis and optimization while allowing architecture-specific tuning for performance-critical code generation

The virtual instruction design must capture several key architectural concepts in a platform-neutral way:

| Virtual Instruction Component | Description | Architecture Mapping |
|-------------------------------|-------------|---------------------|
| Operation Type | Semantic operation (add, load, store, branch) | Maps to native instruction families |
| Operand Descriptors | Virtual registers, memory locations, immediates | Translated to physical registers/addressing modes |
| Data Types | Integer sizes, floating point formats | Mapped to native data type support |
| Constraint Annotations | Register preferences, alignment requirements | Guide backend optimization decisions |
| Dependency Information | Input/output relationships between instructions | Enable instruction scheduling and optimization |

#### Architecture Backend Interface

Each architecture backend implements a standardized interface that handles the translation from virtual instructions to native machine code. This interface must provide enough flexibility for architecture-specific optimizations while maintaining consistency in the overall compilation pipeline.

The backend interface operates through several key abstraction points:

| Backend Interface Method | Parameters | Responsibilities | Architecture-Specific Behavior |
|-------------------------|------------|------------------|-------------------------------|
| `initialize_backend` | Target configuration | Set up architecture constants | Configure register sets, calling conventions |
| `analyze_virtual_function` | Virtual instruction sequence | Analyze optimization opportunities | Architecture-specific pattern recognition |
| `allocate_registers` | Virtual registers, constraints | Map virtual to physical registers | Handle architecture register count differences |
| `emit_instruction_sequence` | Virtual instructions, register assignments | Generate native machine code | Architecture-specific instruction encoding |
| `optimize_code_sequence` | Generated code buffer | Apply post-generation optimizations | Peephole optimizations, instruction scheduling |

The register allocation interface requires particular attention because different architectures provide vastly different register resources. x86-64 provides 16 general-purpose registers with specific roles (stack pointer, frame pointer). ARM architectures typically provide 31 general-purpose registers with more uniform capabilities. RISC-V provides 32 registers with clean separation between integer and floating-point register files.

#### ARM Architecture Backend Implementation

ARM processors present both opportunities and challenges for JIT compilation. The larger register set (typically 31 general-purpose registers versus x86-64's 16) reduces register pressure and enables more values to remain in registers during expression evaluation. However, ARM's load-store architecture requires different code generation patterns, and the instruction encoding differs significantly from x86-64's variable-length instruction format.

The ARM backend implementation focuses on several key areas:

**Instruction Encoding Differences**: ARM uses fixed-width 32-bit instructions (in ARM64) with different bit field layouts for opcodes, registers, and immediate values. The backend must implement ARM-specific instruction encoding routines while maintaining the same virtual instruction interface used by other architectures.

**Calling Convention Adaptation**: ARM's Procedure Call Standard (AAPCS) differs from x86-64's System V ABI in register usage patterns. ARM passes arguments in registers r0-r7, uses r29 as frame pointer, and r30 as link register. The ARM backend must implement these conventions while presenting the same high-level function call interface.

**Load-Store Architecture Implications**: Unlike x86-64, which supports memory operands in arithmetic instructions, ARM requires explicit load/store instructions to move data between memory and registers. The ARM backend must decompose virtual instructions that assume memory operands into appropriate load-store sequences.

#### RISC-V Architecture Backend Implementation

RISC-V represents the cleanest target architecture from a JIT compilation perspective. Its regular instruction set, consistent encoding patterns, and well-defined extension system make it an attractive target for code generation experiments. However, RISC-V's relative newness means less optimization knowledge and fewer established patterns compared to mature architectures.

The RISC-V backend implementation emphasizes simplicity and correctness over aggressive optimization:

**Regular Instruction Format**: RISC-V's consistent instruction encoding with well-defined bit fields simplifies instruction generation compared to x86-64's complex variable-length encoding. The backend can use template-based instruction generation for most operations.

**Extension Handling**: RISC-V's modular extension system (RV32I base with optional M, A, F, D extensions) requires the backend to detect available extensions and generate appropriate instruction sequences. Missing extensions must be handled through software emulation or runtime checks.

**Optimization Opportunities**: RISC-V's large register file and simple instruction set create opportunities for aggressive register allocation and instruction scheduling that may not be available on other architectures.

#### Cross-Architecture Code Generation Pipeline

The multi-architecture compilation pipeline extends the existing single-architecture design with additional abstraction layers and backend selection logic. The pipeline maintains the same high-level structure while adding architecture-specific processing stages.

The extended pipeline operates through these stages:

1. **Bytecode Analysis**: Architecture-independent analysis of bytecode structure, control flow, and optimization opportunities. This stage remains unchanged from the single-architecture implementation.

2. **Virtual Instruction Generation**: Translation from bytecode operations to virtual instructions. This stage captures the semantic intent of operations without committing to specific machine instructions.

3. **Architecture Selection**: Runtime determination of target architecture based on current execution environment. This enables the same JIT compiler binary to generate optimal code for different processor types.

4. **Backend-Specific Optimization**: Architecture-specific analysis and optimization of virtual instruction sequences. Each backend can apply optimizations that leverage architecture-specific features.

5. **Native Code Generation**: Translation from optimized virtual instructions to actual machine code using the selected architecture backend.

6. **Post-Generation Optimization**: Architecture-specific peephole optimizations and instruction scheduling applied to the generated machine code.

The critical insight is that most compilation complexity remains architecture-independent. Control flow analysis, constant propagation, dead code elimination, and most other optimizations operate on the virtual instruction representation. Only the final code generation stages require architecture-specific knowledge.

#### Runtime Architecture Detection and Selection

Supporting multiple architectures in a single JIT compiler binary requires runtime detection of the target processor and dynamic selection of the appropriate backend. This capability enables deployment flexibility while maintaining optimal performance on each supported platform.

The architecture detection system operates through several mechanisms:

| Detection Method | Information Gathered | Reliability | Performance Impact |
|------------------|---------------------|-------------|-------------------|
| Compile-time constants | Base architecture (x86-64, ARM64, RISC-V) | Perfect | None |
| Runtime CPU detection | Specific processor model, feature flags | High | Minimal (cached) |
| Performance profiling | Actual instruction performance characteristics | Variable | Moderate |
| User configuration | Explicit architecture selection | Perfect | None |

The backend selection logic must handle cases where multiple backends might be available for the same base architecture. For example, different ARM processors may benefit from different optimization strategies, or RISC-V implementations with different extension sets may require different code generation approaches.

#### Common Pitfalls in Multi-Architecture Design

⚠️ **Pitfall: Architecture-Specific Logic Leaking into Shared Code**
One of the most common mistakes in multi-architecture JIT design is allowing architecture-specific assumptions to creep into supposedly generic code. For example, assuming that all architectures have the same number of argument registers, or that immediate values have the same size limits across architectures. This creates subtle bugs that only manifest on specific target platforms. The solution is rigorous abstraction discipline — all architecture-specific constants and behaviors must be encapsulated within backend implementations, with the shared code operating only on virtual instruction abstractions.

⚠️ **Pitfall: Inadequate Testing Across Architectures**
Multi-architecture support dramatically increases the testing burden because bugs may only manifest on specific target platforms. A common mistake is testing primarily on the developer's native architecture (typically x86-64) with minimal validation on other targets. This leads to regressions and correctness issues on less-tested architectures. The solution is systematic cross-architecture testing, ideally automated through continuous integration systems that build and test on multiple target platforms.

⚠️ **Pitfall: Performance Regression on Secondary Architectures**
While maintaining correctness across architectures is essential, it's equally important to ensure that performance remains competitive on all supported targets. A common mistake is optimizing primarily for one "primary" architecture while treating others as secondary concerns. This can result in significantly degraded performance on some platforms, undermining the value of multi-architecture support. The solution is architecture-specific performance benchmarking and optimization tuning for each supported target.

### Advanced Optimization Techniques: Inlining, loop optimizations, and profile-guided optimization

The transition from basic JIT compilation to advanced optimization represents a shift from correctness-focused translation to performance-oriented code transformation. Think of this evolution like upgrading from a competent translator who accurately conveys meaning to a skilled literary translator who captures not just the content but the style, rhythm, and emotional impact of the original work. Advanced optimizations don't just make code run correctly — they make it run beautifully.

Advanced optimizations operate on a fundamentally different principle than basic compilation. Instead of translating each bytecode operation in isolation, advanced optimizations consider the broader context of program execution patterns, data flow relationships, and runtime behavior. These optimizations often involve speculative transformations that improve performance for common cases while maintaining correctness for all cases.

The key insight driving advanced optimization is that **runtime information reveals optimization opportunities invisible at static analysis time**. A static compiler must generate code that handles all possible execution scenarios. A JIT compiler can observe actual program behavior, identify hot execution patterns, and generate specialized code optimized for the specific workloads it observes. This runtime specialization enables optimizations that would be impossible or unsafe in traditional ahead-of-time compilation.

#### Function Inlining and Call Site Optimization

Function inlining represents one of the most impactful optimization techniques available to JIT compilers. By replacing function call sites with copies of the called function's body, inlining eliminates call overhead and creates larger optimization scopes that enable secondary optimizations. However, effective inlining requires sophisticated heuristics to balance code size growth against performance improvements.

The inlining decision process operates through several analysis phases:

**Call Site Analysis**: The optimizer examines each function call site to determine inlining feasibility and profitability. This analysis considers factors like caller and callee function sizes, call frequency, argument characteristics, and optimization opportunities that inlining would enable.

| Inlining Decision Factor | Favorable Conditions | Unfavorable Conditions | Weight in Decision |
|-------------------------|---------------------|------------------------|-------------------|
| Callee Function Size | Small functions (< 50 bytecode instructions) | Large functions (> 200 instructions) | High |
| Call Site Frequency | Hot paths (called frequently) | Cold code (rarely executed) | Very High |
| Argument Patterns | Constant arguments, simple types | Variable arguments, complex types | Medium |
| Secondary Optimization Opportunities | Enables constant folding, dead code elimination | No additional optimization potential | High |
| Code Cache Pressure | Plenty of available cache space | Limited cache space | Medium |

**Profile-Guided Inlining**: The most effective inlining decisions leverage runtime profiling data to identify the most profitable inlining opportunities. The profiler collects detailed information about call site behavior, including call frequencies, argument value distributions, and return value patterns.

The profile-guided inlining system operates through several data collection mechanisms:

1. **Call Site Profiling**: Track invocation counts for each call site to identify hot calls that benefit most from inlining.

2. **Argument Profiling**: Monitor argument values passed to functions to identify opportunities for constant propagation after inlining.

3. **Return Value Profiling**: Track return value patterns to identify functions that often return constants or predictable values.

4. **Caller-Callee Relationship Analysis**: Analyze the relationship between callers and callees to identify chains of small functions that should be inlined together.

**Recursive and Polymorphic Call Handling**: Advanced inlining systems must handle complex call patterns including recursive calls and polymorphic dispatch. Recursive inlining requires careful depth limiting to prevent infinite expansion. Polymorphic calls benefit from speculative inlining with guard conditions that handle cases where speculation fails.

The implementation approach for polymorphic inlining uses **guarded inlining** with fallback paths:

1. **Profile Analysis**: Identify the most common target function for each polymorphic call site based on runtime observations.

2. **Speculative Inlining**: Generate optimized code that assumes the most common target, with guard checks that verify the assumption at runtime.

3. **Fallback Path Generation**: Create fallback code that handles cases where the speculation fails, typically by calling the original polymorphic dispatch mechanism.

4. **Guard Optimization**: Optimize the guard checks themselves to minimize the overhead of speculation verification.

#### Loop Optimization and Vectorization

Loop optimization represents another critical area for advanced JIT compilation because loops often contain the most performance-sensitive code in applications. Effective loop optimization requires recognizing loop structures in bytecode, analyzing data dependencies, and applying transformations that improve execution efficiency.

**Loop Recognition and Classification**: The first step in loop optimization involves identifying loop structures within the bytecode control flow graph and classifying them according to their optimization potential.

| Loop Type | Characteristics | Optimization Opportunities | Implementation Complexity |
|-----------|-----------------|---------------------------|--------------------------|
| Simple Counting Loop | Fixed iteration count, simple increment | Unrolling, vectorization, strength reduction | Low |
| Data-Driven Loop | Iteration depends on data values | Predication, speculative execution | Medium |
| Nested Loop Structure | Multiple loop levels | Loop interchange, tiling, fusion | High |
| Irregular Control Flow | Complex branching within loop | Limited optimization, region-based approach | Very High |

**Loop Unrolling**: Loop unrolling reduces loop overhead by replicating loop body instructions multiple times per iteration. This optimization reduces branch instruction frequency and creates larger basic blocks that enable better instruction scheduling and register allocation.

The loop unrolling implementation must balance several competing factors:

1. **Unroll Factor Selection**: Determine how many loop iterations to combine into each unrolled iteration. Higher unroll factors reduce branch overhead but increase code size and register pressure.

2. **Remainder Loop Handling**: Generate separate code to handle cases where the total iteration count is not evenly divisible by the unroll factor.

3. **Register Pressure Management**: Ensure that unrolling doesn't exhaust available registers and force expensive register spilling.

4. **Cache Impact Analysis**: Consider the impact of increased code size on instruction cache performance.

**Strength Reduction**: Strength reduction replaces expensive operations with cheaper equivalent operations. This optimization is particularly effective in loops where expensive operations are repeated with predictable patterns.

Common strength reduction patterns include:

- Replacing multiplication by loop index with incremental addition
- Converting array indexing calculations to pointer arithmetic
- Simplifying address calculation patterns
- Optimizing loop-invariant expressions

**Vectorization and SIMD Utilization**: Modern processors provide SIMD (Single Instruction, Multiple Data) instructions that can perform the same operation on multiple data elements simultaneously. JIT compilers can leverage runtime information to identify vectorization opportunities and generate SIMD-optimized code.

The vectorization process operates through several analysis phases:

1. **Data Dependency Analysis**: Verify that loop iterations can be executed in parallel without data hazards.

2. **Memory Access Pattern Analysis**: Identify regular memory access patterns that can benefit from vector load/store instructions.

3. **Operation Compatibility Check**: Ensure that loop operations can be expressed using available SIMD instructions.

4. **Cost-Benefit Analysis**: Verify that vectorization provides sufficient performance improvement to justify the complexity.

#### Profile-Guided Optimization Framework

Profile-guided optimization (PGO) represents the pinnacle of JIT compilation sophistication. By collecting detailed runtime behavior data and using this information to guide optimization decisions, PGO can achieve performance improvements that are impossible through static analysis alone.

**Multi-Tiered Profiling Strategy**: Effective PGO requires a sophisticated profiling system that collects different types of behavioral data without significantly impacting program performance.

| Profiling Type | Data Collected | Collection Overhead | Optimization Impact |
|-----------------|-----------------|-------------------|-------------------|
| Basic Block Profiling | Execution frequency of code regions | Very Low (< 1%) | High for control flow optimization |
| Branch Profiling | Taken/not-taken frequencies | Low (< 2%) | Medium for predication and layout |
| Value Profiling | Distribution of variable values | Medium (5-10%) | High for specialization |
| Memory Access Profiling | Cache behavior, access patterns | High (10-20%) | Medium for data layout optimization |
| Call Graph Profiling | Function call relationships | Low (< 3%) | High for inlining decisions |

**Adaptive Optimization Triggers**: The PGO system must determine when sufficient profiling data has been collected to make reliable optimization decisions. This requires sophisticated statistical analysis to distinguish genuine behavior patterns from noise.

The adaptive trigger system operates through several mechanisms:

1. **Sample Size Analysis**: Ensure that sufficient execution samples have been collected to make statistically significant optimization decisions.

2. **Behavior Stability Detection**: Monitor whether program behavior remains consistent over time or changes significantly during execution.

3. **Cost-Benefit Threshold Management**: Only trigger expensive optimizations when the expected performance improvement justifies the compilation cost.

4. **Deoptimization Trigger Conditions**: Detect when program behavior changes sufficiently to invalidate existing optimizations.

**Speculative Optimization with Guard Conditions**: PGO enables speculative optimizations that assume common case behavior while providing fallback mechanisms for uncommon cases.

The speculative optimization framework includes several key components:

- **Assumption Tracking**: Record all assumptions made during speculative optimization to enable invalidation when assumptions become false.
- **Guard Condition Generation**: Create efficient runtime checks that verify speculation assumptions.
- **Deoptimization Infrastructure**: Provide mechanisms to fall back to unoptimized code when speculation fails.
- **Re-optimization Triggers**: Detect when changed behavior patterns justify re-compiling with different assumptions.

#### Common Pitfalls in Advanced Optimization

⚠️ **Pitfall: Over-Optimization Leading to Code Bloat**
Advanced optimizations like inlining and loop unrolling can dramatically increase code size, potentially causing instruction cache pressure that negates performance improvements. A common mistake is applying aggressive optimizations without considering their impact on cache behavior. The solution is holistic performance modeling that considers both CPU execution efficiency and memory hierarchy effects.

⚠️ **Pitfall: Profile Instability and Over-Fitting**
Optimizing based on profiling data can lead to code that performs very well on observed workloads but poorly on different inputs. This "over-fitting" problem is particularly acute when profiling samples are too small or unrepresentative. The solution is robust statistical analysis of profiling data, with confidence intervals and stability checks before applying aggressive optimizations.

⚠️ **Pitfall: Deoptimization Overhead Negating Benefits**
Speculative optimizations require deoptimization mechanisms to handle cases where assumptions fail. If speculation fails frequently, the overhead of deoptimization can eliminate the benefits of optimization. The solution is careful profiling-based prediction of speculation success rates and conservative application of speculative optimizations when success rates are marginal.

### Advanced Runtime Features: On-stack replacement, deoptimization, and dynamic recompilation

Advanced runtime features represent the sophisticated machinery that enables seamless transitions between different optimization levels and execution modes while the program is running. Think of these features like the advanced safety and control systems in a high-performance race car — they operate invisibly during normal operation but provide crucial capabilities for handling exceptional conditions and optimizing performance in real-time.

These features distinguish production-grade JIT compilers from academic implementations. While basic JIT compilation focuses on translating bytecode to native code, advanced runtime features enable the system to continuously adapt its behavior based on evolving program characteristics. This adaptive capability is what allows modern JIT compilers to achieve performance that often exceeds statically compiled code by specializing for actual runtime behavior rather than conservative worst-case assumptions.

The key architectural insight is that advanced runtime features require **seamless state management across compilation boundaries**. The system must be able to capture the complete execution state at arbitrary points, transform that state between different code representations, and resume execution in a different optimization context without any visible impact on program semantics.

#### On-Stack Replacement (OSR) Implementation

On-Stack Replacement represents one of the most technically challenging features in JIT compilation. OSR enables the runtime to replace currently executing code with a more optimized version without waiting for function boundaries or control flow transitions. This capability is essential for optimizing long-running loops and recursive functions that spend significant time in unoptimized code before triggering compilation.

**Execution State Capture**: The foundation of OSR lies in the ability to capture complete execution state at arbitrary points within a function. This state includes register contents, stack frame layout, local variables, and the current position within the instruction stream.

The state capture mechanism operates through several coordinated components:

| State Component | Capture Method | Complexity | Critical Requirements |
|-----------------|---------------|------------|----------------------|
| Register Values | Read from physical processor registers | Low | Must handle register allocation differences |
| Stack Frame | Copy stack memory contents | Medium | Must account for frame layout differences |
| Local Variables | Extract from registers and stack slots | High | Requires mapping between optimization levels |
| Control Flow Position | Map instruction pointer to bytecode offset | High | Must handle inlined functions and optimizations |
| Heap References | Identify and preserve object pointers | Very High | Critical for garbage collection correctness |

**State Transformation and Mapping**: Once execution state is captured, it must be transformed to match the expectations of the optimized code. This transformation is complicated by the fact that optimized and unoptimized code may have completely different register allocation, stack frame layouts, and even control flow structures due to inlining and other optimizations.

The state transformation process operates through several mapping phases:

1. **Variable Location Mapping**: Determine where each logical variable is stored in both the source (unoptimized) and target (optimized) code representations.

2. **Control Flow Position Translation**: Map the current execution position in unoptimized code to the corresponding position in optimized code, accounting for inlining and code motion optimizations.

3. **Stack Frame Reconstruction**: Build the new stack frame expected by optimized code, potentially combining or splitting frames due to inlining decisions.

4. **Register State Initialization**: Populate processor registers with values expected by the optimized code at the target execution position.

**OSR Point Selection and Instrumentation**: Effective OSR requires careful selection of points where replacement can occur safely and efficiently. These OSR points must be instrumented in the unoptimized code to enable state capture and replacement when triggered.

> **Decision: Loop Header OSR Point Strategy**
> - **Context**: Long-running loops need optimization before completion, but OSR points add overhead to unoptimized code execution
> - **Options Considered**: 
>   - OSR at every bytecode instruction
>   - OSR only at loop headers and function entries  
>   - OSR at compiler-selected safe points
> - **Decision**: OSR primarily at loop headers with selective additional safe points
> - **Rationale**: Loop headers provide the highest performance impact with minimal instrumentation overhead, while additional safe points handle edge cases
> - **Consequences**: Enables optimization of long-running computations while keeping instrumentation overhead below 5% of interpretation performance

The OSR point instrumentation includes several key elements:

- **Execution Counter Integration**: OSR points typically piggyback on existing profiling counters to minimize overhead.
- **State Capture Triggers**: Mechanisms to initiate state capture and replacement when optimization thresholds are reached.
- **Fallback Handling**: Error recovery mechanisms for cases where OSR fails due to state capture or transformation problems.

**Deoptimization Infrastructure**: OSR requires bidirectional capability — not only optimizing from unoptimized to optimized code, but also deoptimizing from optimized back to unoptimized code when speculative optimizations are invalidated.

The deoptimization infrastructure includes several critical components:

| Deoptimization Component | Purpose | Implementation Complexity | Performance Impact |
|-------------------------|---------|---------------------------|-------------------|
| Assumption Tracking | Record speculative optimization assumptions | Medium | Low (metadata only) |
| Invalidation Triggers | Detect when assumptions become false | High | Variable (depends on check frequency) |
| State Reconstruction | Rebuild unoptimized execution state | Very High | High during deoptimization events |
| Code Patching | Remove or redirect optimized code | Medium | Low (infrequent operation) |

#### Dynamic Recompilation and Code Versioning

Dynamic recompilation extends the basic JIT compilation model by maintaining multiple versions of the same function optimized for different execution contexts or behavioral patterns. This capability enables the runtime to adapt to changing program characteristics over time without losing previous optimization investments.

**Multi-Version Code Management**: The dynamic recompilation system maintains a versioning hierarchy where each function can have multiple compiled representations optimized for different scenarios.

The code versioning system operates through several organizational principles:

1. **Specialization Hierarchy**: Functions are specialized for increasingly specific execution contexts, from generic implementations to highly specialized versions for particular input patterns.

2. **Version Selection Logic**: Runtime dispatch mechanisms select the most appropriate code version based on current execution context and argument characteristics.

3. **Version Lifecycle Management**: Automated systems manage the creation, optimization, and eventual retirement of code versions based on utility and resource consumption.

4. **Memory Management**: Efficient storage and retrieval of multiple code versions without excessive memory consumption.

**Adaptive Compilation Triggers**: Dynamic recompilation requires sophisticated triggering mechanisms that detect when program behavior has changed sufficiently to justify creating new optimized versions.

| Recompilation Trigger | Detected Condition | Response Strategy | Implementation Notes |
|-----------------------|-------------------|-------------------|---------------------|
| Behavior Pattern Change | Significant shift in execution patterns | Generate new specialized version | Requires statistical change detection |
| Performance Regression | Current version performing poorly | Recompile with different optimizations | Must distinguish real regression from noise |
| New Optimization Opportunities | Runtime discovers new patterns | Create more aggressive optimized version | Balances risk/reward of speculation |
| Resource Pressure | Memory or cache pressure from multiple versions | Consolidate or retire underused versions | Requires usage tracking and cleanup logic |

**Profile Evolution Tracking**: Dynamic recompilation systems must track how program behavior evolves over time to make intelligent decisions about when and how to recompile functions.

The profile evolution system includes several tracking mechanisms:

- **Behavioral Drift Detection**: Statistical analysis to identify when current behavior differs significantly from the patterns used for existing optimizations.
- **Performance Trend Analysis**: Monitoring to detect performance degradation that might indicate suboptimal code versions.
- **Resource Utilization Tracking**: Analysis of memory and cache impact from maintaining multiple code versions.
- **Optimization Opportunity Discovery**: Continuous analysis to identify new optimization opportunities not captured in existing versions.

#### Speculative Optimization and Guard Mechanisms

Advanced JIT compilers leverage speculative optimization to achieve performance that exceeds what's possible through conservative compilation strategies. Speculative optimizations make assumptions about program behavior based on observed patterns, generate optimized code for the assumed common cases, and provide fallback mechanisms for cases where assumptions prove incorrect.

**Guard Condition Implementation**: Effective speculative optimization requires efficient guard mechanisms that verify assumptions with minimal performance overhead.

The guard implementation strategy operates through several optimization techniques:

1. **Guard Consolidation**: Combine multiple related checks into single guard conditions to reduce branching overhead.

2. **Guard Scheduling**: Position guards to minimize impact on critical execution paths while ensuring assumption verification.

3. **Guard Elimination**: Remove redundant guards through data flow analysis and assumption propagation.

4. **Guard Specialization**: Generate different guard implementations based on the specific assumptions being verified.

**Assumption Management Framework**: Speculative optimization requires comprehensive tracking of all assumptions made during compilation to enable invalidation when assumptions become false.

| Assumption Type | Examples | Verification Method | Invalidation Cost |
|-----------------|----------|-------------------|------------------|
| Type Assumptions | Variable always contains integer values | Runtime type checks | Low - local deoptimization |
| Value Assumptions | Variable usually contains specific constant | Value comparison guards | Low - local fallback |
| Control Flow Assumptions | Branch usually taken in specific direction | Branch frequency monitoring | Medium - affects scheduling |
| Call Target Assumptions | Polymorphic call usually targets specific function | Target address verification | High - affects inlining |
| Memory Layout Assumptions | Object fields remain at fixed offsets | Object header checks | Very High - affects all object access |

**Deoptimization and Reoptimization Cycles**: When speculative assumptions are invalidated, the system must not only fall back to safe code but also learn from the invalidation to make better optimization decisions in the future.

The deoptimization learning process includes several feedback mechanisms:

- **Assumption Failure Analysis**: Determine why speculative assumptions failed and whether the failure represents a temporary anomaly or fundamental behavior change.
- **Reoptimization Strategy Selection**: Choose appropriate optimization strategies for recompilation based on updated behavioral understanding.
- **Speculation Aggressiveness Adjustment**: Modify the threshold for speculative optimization based on success/failure rates.
- **Alternative Optimization Exploration**: Try different optimization strategies when previous speculative approaches prove unreliable.

#### Advanced Runtime Feature Integration

The integration of OSR, dynamic recompilation, and speculative optimization creates a sophisticated runtime system that can continuously adapt to program behavior while maintaining correctness and performance. This integration requires careful coordination between all system components.

**Unified State Management**: Advanced runtime features share common infrastructure for execution state capture, transformation, and restoration. This shared infrastructure reduces implementation complexity and ensures consistency across different optimization scenarios.

**Performance Monitoring and Feedback**: Advanced runtime features require comprehensive performance monitoring to validate that optimizations are providing expected benefits and to guide future optimization decisions.

The monitoring system tracks several key metrics:

- **Optimization Success Rates**: Percentage of speculative optimizations that provide performance improvements without frequent deoptimization.
- **State Transition Overhead**: Cost of OSR and deoptimization operations compared to execution time savings from optimizations.
- **Memory Utilization**: Resource consumption from maintaining multiple code versions and optimization metadata.
- **Cache Impact**: Effect of code size growth from optimization on instruction cache performance.

#### Common Pitfalls in Advanced Runtime Features

⚠️ **Pitfall: State Capture Overhead Exceeding Optimization Benefits**
OSR and deoptimization require continuous tracking of execution state, which can add significant overhead to program execution. A common mistake is implementing overly comprehensive state tracking that costs more than the optimizations save. The solution is selective state tracking that focuses on optimization-critical information while minimizing instrumentation overhead.

⚠️ **Pitfall: Speculation Assumption Instability**
Speculative optimizations can become counterproductive when based on unstable behavioral patterns. Code that frequently deoptimizes and reoptimizes can perform worse than conservatively compiled code. The solution is robust statistical analysis of behavioral patterns with confidence thresholds that prevent optimization based on insufficient or unstable data.

⚠️ **Pitfall: Memory Explosion from Multiple Code Versions**
Dynamic recompilation can lead to unbounded memory consumption if not carefully managed. Each function potentially has multiple optimized versions, and without proper lifecycle management, memory usage can grow without bound. The solution is aggressive version lifecycle management with automatic cleanup of underutilized code versions based on usage patterns and resource pressure.

### Implementation Guidance

The implementation of future extensions for the JIT compiler requires careful architectural planning and incremental development strategies. These extensions represent significant engineering investments that should be approached systematically to ensure successful integration with the existing system.

#### Technology Recommendations

| Extension Category | Simple Implementation | Advanced Implementation |
|-------------------|----------------------|------------------------|
| Multi-Architecture | Abstract instruction interface with direct encoding | LLVM integration for multiple targets |
| Advanced Optimization | Basic inlining with size limits | Full profile-guided optimization framework |
| Runtime Features | Simple OSR at function boundaries | Complete state capture with speculative optimization |

#### Recommended Development Phase Structure

The extensions should be implemented in carefully planned phases that build upon each other:

```
extensions/
  multi-arch/
    arch-interface.h          ← Architecture abstraction layer
    x86-64-backend.c         ← Reference x86-64 implementation  
    arm64-backend.c          ← ARM architecture backend
    riscv-backend.c          ← RISC-V architecture backend
  optimization/
    inlining.c               ← Function inlining implementation
    loop-opt.c               ← Loop optimization passes
    profile-guided.c         ← PGO framework
  runtime-features/
    osr.c                    ← On-stack replacement
    deoptimization.c         ← Speculative optimization support
    versioning.c             ← Dynamic recompilation
  testing/
    cross-arch-tests.c       ← Multi-architecture validation
    optimization-benchmarks.c ← Performance measurement
    runtime-feature-tests.c  ← Advanced feature validation
```

#### Multi-Architecture Infrastructure Starter Code

```c
// Architecture abstraction interface
typedef struct ArchBackend {
    const char* architecture_name;
    int register_count;
    int stack_alignment;
    
    // Backend initialization
    int (*initialize)(struct ArchBackend* backend, 
                     const char* target_config);
    
    // Instruction emission interface
    void (*emit_mov_reg_imm)(CodeBuffer* buffer, int reg, int64_t value);
    void (*emit_add_reg_reg)(CodeBuffer* buffer, int dest, int src);
    void (*emit_function_prologue)(CodeBuffer* buffer, 
                                  StackFrameLayout* frame);
    void (*emit_function_epilogue)(CodeBuffer* buffer, 
                                   StackFrameLayout* frame);
    
    // Architecture-specific optimization
    int (*optimize_instruction_sequence)(CodeBuffer* buffer,
                                        uint32_t start_offset,
                                        uint32_t length);
} ArchBackend;

// Virtual instruction representation
typedef struct VirtualInstruction {
    enum VInstrType {
        VINSTR_LOAD,
        VINSTR_STORE, 
        VINSTR_ARITHMETIC,
        VINSTR_BRANCH,
        VINSTR_CALL
    } type;
    
    int virtual_registers[3];  // src1, src2, dest
    int64_t immediate_value;
    uint32_t flags;           // Architecture hints
} VirtualInstruction;

// Cross-architecture compilation context
typedef struct CrossArchContext {
    ArchBackend* current_backend;
    VirtualInstruction* virtual_code;
    uint32_t instruction_count;
    CodeBuffer* target_buffer;
    RegisterAllocator* virtual_allocator;
} CrossArchContext;
```

#### Advanced Optimization Framework Skeleton

```c
// Profile-guided optimization context
typedef struct OptimizationContext {
    BytecodeFunction* function;
    ProfileData* profile;
    ControlFlowGraph* cfg;
    
    // Optimization passes
    int (*run_inlining_pass)(struct OptimizationContext* ctx);
    int (*run_loop_optimization)(struct OptimizationContext* ctx);
    int (*run_speculative_optimization)(struct OptimizationContext* ctx);
} OptimizationContext;

// Function inlining implementation skeleton
int run_inlining_pass(OptimizationContext* ctx) {
    // TODO 1: Identify all call sites in the function
    // TODO 2: For each call site, analyze inlining profitability
    // TODO 3: Check callee function size and complexity
    // TODO 4: Verify inlining doesn't exceed code size budgets
    // TODO 5: Perform actual inlining transformation
    // TODO 6: Update control flow graph and profiling data
    // Hint: Start with simple inlining of leaf functions only
}

// Profile-guided specialization
int run_speculative_optimization(OptimizationContext* ctx) {
    // TODO 1: Analyze profile data for stable behavioral patterns
    // TODO 2: Identify opportunities for type specialization
    // TODO 3: Generate guard conditions for speculative assumptions
    // TODO 4: Create specialized code paths for common cases
    // TODO 5: Implement fallback paths for assumption failures
    // TODO 6: Track assumption validity for future deoptimization
    // Hint: Focus on operations with high type stability (>90% same type)
}
```

#### OSR and Advanced Runtime Feature Implementation

```c
// On-stack replacement infrastructure
typedef struct OSRPoint {
    uint32_t bytecode_offset;
    uint32_t native_offset;
    StateMapping* register_map;
    StateMapping* stack_map;
    int can_osr_here;
} OSRPoint;

// Execution state capture for OSR
typedef struct ExecutionState {
    int64_t register_values[16];  // x86-64 general registers
    int64_t* stack_frame;
    size_t stack_frame_size;
    uint32_t current_bytecode_offset;
    void* heap_references;        // GC root set
} ExecutionState;

// OSR implementation skeleton
int perform_osr_transition(uint32_t function_id, 
                          OSRPoint* osr_point,
                          CompiledFunction optimized_version) {
    // TODO 1: Capture current execution state
    // TODO 2: Validate OSR point compatibility
    // TODO 3: Transform state for optimized code expectations
    // TODO 4: Set up stack frame for optimized function
    // TODO 5: Transfer control to optimized code at correct position
    // TODO 6: Handle OSR failure by continuing in unoptimized code
    // Hint: Use setjmp/longjmp for control transfer
}

// Deoptimization support
int trigger_deoptimization(void* failing_guard_address,
                          ExecutionState* current_state) {
    // TODO 1: Identify which speculative assumptions failed
    // TODO 2: Locate corresponding unoptimized code position
    // TODO 3: Reconstruct execution state for unoptimized code
    // TODO 4: Invalidate affected optimized code versions
    // TODO 5: Transfer control back to interpreter or unoptimized code
    // TODO 6: Record deoptimization for future optimization decisions
    // Hint: Use signal handlers to catch assumption failures
}
```

#### Language-Specific Implementation Hints

For C implementation of advanced features:

- **Multi-Architecture**: Use function pointers for backend dispatch, compile-time architecture detection with `#ifdef` blocks
- **OSR State Capture**: Use `setjmp`/`longjmp` for control flow transfer, inline assembly for register capture
- **Memory Management**: Implement custom allocators for code versioning, use mmap for executable memory pools
- **Profiling Integration**: Use signal-based sampling profilers, low-overhead counter instrumentation
- **Cross-Platform**: Use autotools or CMake for multi-architecture build systems

#### Milestone Checkpoints for Extensions

**Multi-Architecture Extension Checkpoint**:
- Compile and run identical bytecode on x86-64 and ARM64 (if available)
- Verify generated code correctness with cross-architecture test suite
- Measure performance parity between architectures (within 10%)
- Test architecture detection and backend selection logic

**Advanced Optimization Checkpoint**:
- Implement basic function inlining with correctness verification
- Demonstrate measurable performance improvement on loop-heavy benchmarks
- Validate profile-guided optimization decisions through before/after comparison
- Verify optimization doesn't break correctness on edge cases

**Advanced Runtime Features Checkpoint**:
- Successfully perform OSR on long-running loop without visible glitches
- Demonstrate deoptimization and recovery from failed speculative assumptions
- Measure overhead of advanced features (should be < 15% in worst case)
- Validate state consistency across optimization level transitions


## Glossary

> **Milestone(s):** All milestones — comprehensive technical terminology provides essential reference material throughout the entire JIT compiler implementation

The complexity of JIT compilation introduces a rich vocabulary of technical terms spanning multiple domains: computer architecture, compiler theory, virtual machines, operating systems, and performance optimization. This glossary provides precise definitions for all terminology used throughout this design document, serving as both a learning resource for newcomers and a reference for experienced developers working on the JIT compiler implementation.

### Mental Model: Technical Dictionary as Navigation Chart

Think of this glossary as a navigation chart for exploring the JIT compilation landscape. Just as sailors use charts with detailed explanations of nautical terms, landmarks, and navigation symbols to safely traverse unfamiliar waters, developers need precise definitions of technical terminology to navigate the complex interactions between bytecode interpretation, machine code generation, memory management, and runtime optimization. Each term represents a landmark in the conceptual territory, with definitions providing the coordinates needed to understand its relationship to other concepts in the system.

The terminology spans multiple levels of abstraction, from low-level hardware concepts like instruction encoding and memory protection, through mid-level system concepts like calling conventions and profiling, to high-level algorithmic concepts like hot path detection and optimization passes. Understanding these terms and their relationships is essential for making informed design decisions and implementing robust solutions.

### Architecture and Hardware Terms

The foundation of JIT compilation rests on understanding the target processor architecture and the interface between software and hardware. These terms define the essential concepts for generating executable machine code.

| Term | Definition | Context |
|------|------------|---------|
| **x86-64** | The 64-bit extension of the x86 instruction set architecture, supporting 64-bit addresses and extended register sets | Primary target architecture for JIT code generation |
| **Instruction Encoding** | The process of converting assembly mnemonics and operands into binary machine code format | Core responsibility of the machine code emitter |
| **REX Prefix** | A byte prefix (0x48) required for 64-bit operations in x86-64, extending instruction encoding | Essential for generating correct 64-bit instructions |
| **ModR/M Byte** | An instruction encoding byte that specifies register/memory operand layout in x86-64 | Determines how operands are interpreted by the processor |
| **Machine Code Generation** | The process of converting high-level operations into binary CPU instructions | Primary function of the JIT compilation pipeline |
| **Register** | A high-speed storage location within the processor, directly accessible by instructions | Target for bytecode stack value allocation |
| **Immediate Value** | A constant value encoded directly within an instruction, rather than loaded from memory | Used for constant operands in generated code |
| **Opcode** | The portion of an instruction that specifies the operation to be performed | Determines the fundamental instruction behavior |
| **Operand** | The data values that an instruction operates upon, specified as registers, memory, or immediates | Targets of instruction operations |
| **Assembly Language** | Human-readable representation of machine instructions using mnemonics | Intermediate representation for understanding generated code |

### Memory Management and Execution

JIT compilation requires sophisticated memory management to allocate, protect, and execute dynamically generated code. These terms define the memory management concepts essential for secure and correct execution.

| Term | Definition | Context |
|------|------------|---------|
| **Executable Memory** | Memory pages with execute permission, allowing stored instructions to be run as code | Required for JIT-generated machine code |
| **Memory Protection** | Operating system mechanism controlling read, write, and execute access to memory pages | Security foundation for executable code management |
| **W^X Policy** | Security model where memory pages are either writable or executable, never both simultaneously | Modern security requirement affecting code generation workflow |
| **mmap** | System call for mapping files or anonymous memory into process address space with specified permissions | Primary mechanism for allocating executable memory |
| **Memory Page** | Fixed-size memory unit (typically 4KB) that serves as the granularity for memory protection | Basic unit for executable memory allocation |
| **Code Buffer** | A memory region allocated for storing generated machine code instructions | Container for JIT-compiled functions |
| **Memory Pool** | A pre-allocated region of memory divided into smaller buffers for efficient allocation | Optimization for frequent code buffer allocation |
| **Guard Pages** | Memory pages with no access permissions, used to detect buffer overruns | Debugging aid for detecting memory corruption |
| **Page Fault** | Hardware exception triggered when accessing memory without appropriate permissions | Mechanism for detecting security violations |
| **Virtual Memory** | Operating system abstraction providing each process with its own address space | Foundation for memory protection and management |

### Bytecode and Virtual Machine Concepts

The JIT compiler extends an existing bytecode virtual machine, requiring deep understanding of bytecode execution semantics and virtual machine architecture.

| Term | Definition | Context |
|------|------------|---------|
| **Bytecode** | Platform-independent instruction format used by virtual machines, higher-level than machine code | Source format for JIT compilation |
| **Bytecode Interpretation** | Execution of bytecode instructions through a virtual machine dispatcher loop | Baseline execution mode that JIT compilation optimizes |
| **Virtual Machine** | Software implementation of a computer system that executes bytecode instructions | Host environment for the JIT compiler |
| **Virtual Stack** | Abstraction mapping bytecode stack operations to register assignments | Bridge between stack-based bytecode and register-based machine code |
| **Opcode** | Operation code identifying the specific bytecode instruction to execute | Unit of bytecode-to-native translation |
| **Stack-Based Architecture** | Virtual machine design where operations consume and produce values on a stack | Typical bytecode execution model |
| **Dispatcher Loop** | Main interpretation loop that fetches, decodes, and executes bytecode instructions | Core of the bytecode interpreter |
| **Bytecode Function** | A complete function represented as a sequence of bytecode instructions | Unit of compilation for function-level JIT |
| **Local Variables** | Function-scoped variables accessible by bytecode instructions within a function | Require stack frame allocation in compiled code |
| **Control Flow** | The order of instruction execution, including jumps, branches, and function calls | Requires careful handling in bytecode-to-native translation |

### JIT Compilation Fundamentals

These terms define the core concepts of just-in-time compilation, distinguishing it from ahead-of-time compilation and interpretation.

| Term | Definition | Context |
|------|------------|---------|
| **Just-In-Time Compilation** | Dynamic compilation of bytecode to native machine code during program execution | Core technology implemented by this project |
| **Hot Path Detection** | Identification of frequently executed code paths that benefit from JIT compilation | Profiling-guided optimization strategy |
| **Compilation Threshold** | The execution count at which a function or loop becomes eligible for JIT compilation | Policy parameter controlling compilation timing |
| **Tiered Compilation** | Execution strategy using multiple optimization levels, from interpretation to optimized native code | Architecture for balancing compilation cost and execution speed |
| **Mixed-Mode Execution** | Seamless combination of interpreted and native code execution within the same program | Essential capability for practical JIT systems |
| **Compilation Pipeline** | Series of stages that transform bytecode into native machine code | Structured approach to bytecode-to-native translation |
| **Translation Context** | Data structure maintaining state during bytecode-to-native compilation | Coordination mechanism for compilation stages |
| **Compilation Trigger** | Event or condition that initiates JIT compilation of a function or code region | Policy mechanism for managing compilation overhead |
| **Code Cache** | Storage system for compiled native code functions, enabling reuse across invocations | Performance optimization avoiding redundant compilation |
| **Transparent Augmentation** | Extending VM capabilities without modifying core functionality or visible behavior | Design principle for JIT integration |

### Register Allocation and Code Generation

Efficient translation from stack-based bytecode to register-based machine code requires sophisticated register allocation and instruction selection strategies.

| Term | Definition | Context |
|------|------------|---------|
| **Register Allocation** | Assignment of virtual stack positions to physical processor registers | Core optimization for efficient native code generation |
| **Register Pressure** | Situation where the number of active values exceeds available processor registers | Triggers register spilling to memory |
| **Register Spilling** | Storing register contents to memory when registers are exhausted | Fallback mechanism for complex expressions |
| **Spill Slot** | Stack frame location used to temporarily store spilled register values | Memory allocation for register pressure handling |
| **Virtual Register** | Conceptual register representing a value during compilation, later mapped to physical registers | Abstraction for register allocation algorithms |
| **Live Range** | The span of instructions during which a value is active and potentially needed | Key concept for register allocation optimization |
| **Expression Depth** | The maximum number of simultaneously active values in an expression tree | Determines minimum register requirements |
| **Register Assignment** | The mapping between virtual stack positions and specific processor registers | Result of the register allocation process |
| **Reload** | Loading a previously spilled value from memory back into a register | Recovery operation for spilled values |
| **Caller-Saved Registers** | Registers that the calling function must preserve across function calls | ABI requirement affecting register allocation |

### Calling Conventions and ABI

Proper function compilation requires adherence to standardized calling conventions that govern parameter passing, return values, and register preservation.

| Term | Definition | Context |
|------|------------|---------|
| **Application Binary Interface (ABI)** | Standardized rules governing function calls, parameter passing, and register usage | Foundation for interoperability between compiled code |
| **System V AMD64 ABI** | The calling convention standard for x86-64 Linux and macOS systems | Primary ABI for JIT-compiled functions |
| **Calling Convention** | Standardized rules for function argument passing and register preservation | Protocol enabling function calls between different code |
| **Function Prologue** | Code sequence at function entry that establishes the stack frame | Standard setup for function execution |
| **Function Epilogue** | Code sequence at function exit that tears down stack frame and returns | Standard cleanup for function completion |
| **Stack Frame** | Memory region allocated for a function's local variables and execution state | Per-function memory organization |
| **Frame Pointer** | Register pointing to a fixed location within the current stack frame | Reference point for local variable access |
| **Stack Alignment** | Requirement for stack pointer to maintain 16-byte alignment before function calls | ABI requirement for proper function operation |
| **Parameter Passing** | Mechanism for transferring argument values from caller to callee | ABI-defined protocol for function communication |
| **Return Value** | Result value transferred from callee back to caller upon function completion | ABI-defined protocol for function results |

### Control Flow and Jump Handling

Bytecode control flow instructions require careful translation to maintain correct program semantics while handling forward references and address resolution.

| Term | Definition | Context |
|------|------------|---------|
| **Forward Reference** | A jump instruction that targets a location not yet compiled | Common situation requiring address patching |
| **Jump Patching** | Process of updating incomplete jump instructions with correct target addresses | Two-pass compilation technique |
| **Forward Patching** | Resolving jump target addresses after code generation completes | Solution for forward reference problem |
| **Conditional Jump** | Branch instruction that transfers control based on a condition | Requires translation of bytecode comparisons |
| **Unconditional Jump** | Branch instruction that always transfers control to the target | Direct translation from bytecode jumps |
| **Jump Table** | Data structure recording forward references for later address resolution | Implementation mechanism for jump patching |
| **Basic Block** | Sequence of instructions with single entry and exit points | Unit of control flow analysis |
| **Control Flow Graph** | Representation of all possible execution paths through a function | Analysis structure for optimization |
| **Branch Prediction** | Processor mechanism for speculatively executing likely branch paths | Hardware optimization affecting performance |
| **Branch Target** | The destination address of a jump or branch instruction | Must be correctly calculated during compilation |

### Profiling and Performance Monitoring

Effective JIT compilation requires runtime profiling to identify optimization opportunities and measure compilation effectiveness.

| Term | Definition | Context |
|------|------------|---------|
| **Execution Counter** | Runtime tracking of how many times a function or loop has been executed | Foundation for hot path detection |
| **Hot Path** | Frequently executed code region that benefits from JIT compilation optimization | Target for compilation resources |
| **Cold Path** | Infrequently executed code that should remain interpreted to avoid compilation overhead | Regions to avoid compiling |
| **Execution Frequency** | Rate at which specific code regions are executed during program runtime | Metric for compilation decision making |
| **Profile Data** | Runtime statistics about program execution behavior and patterns | Input for optimization decisions |
| **Profiling Overhead** | Performance cost of collecting execution statistics during program runtime | Must be minimized to avoid slowing interpretation |
| **Instrumentation** | Addition of measurement code to track execution behavior | Mechanism for collecting profile data |
| **Performance Counter** | Hardware or software mechanism for measuring execution statistics | Tool for detailed performance analysis |
| **Sampling Profiler** | Profiling approach that periodically samples execution state rather than tracking every event | Low-overhead alternative to exhaustive profiling |
| **Threshold Policy** | Rules governing when execution frequency justifies compilation cost | Policy mechanism for resource management |

### Runtime Optimization Techniques

JIT compilation enables optimizations based on runtime behavior that are impossible with ahead-of-time compilation.

| Term | Definition | Context |
|------|------------|---------|
| **Constant Folding** | Compile-time evaluation of expressions with constant operands | Basic optimization reducing runtime computation |
| **Constant Propagation** | Tracking of constant values across operations to enable additional optimizations | Analysis enabling constant folding |
| **Dead Code Elimination** | Removal of instructions that produce values never used by subsequent code | Optimization reducing code size and execution time |
| **Runtime Specialization** | Generating code optimized for specific operand types or common execution patterns | Profile-guided optimization technique |
| **Type Specialization** | Optimizing operations for specific operand types observed during profiling | Runtime optimization based on observed behavior |
| **Branch Specialization** | Optimizing conditional jumps based on observed branch frequency patterns | Profile-guided control flow optimization |
| **Speculative Optimization** | Optimizations based on assumptions about common case behavior | Aggressive optimization requiring guard conditions |
| **Guard Condition** | Runtime check verifying that speculative optimization assumptions remain valid | Safety mechanism for speculative optimizations |
| **Deoptimization** | Fallback from optimized to generic code when assumptions become invalid | Recovery mechanism for failed speculation |
| **Profile-Guided Optimization** | Optimizations based on runtime execution behavior data collected during profiling | Data-driven optimization strategy |

### Error Handling and Recovery

Robust JIT compilation requires comprehensive error handling for compilation failures, runtime exceptions, and resource management issues.

| Term | Definition | Context |
|------|------------|---------|
| **Compilation Error** | Failure during the process of translating bytecode to native machine code | Recoverable condition requiring fallback to interpretation |
| **Runtime Exception** | Error condition occurring during execution of JIT-compiled native code | Requires conversion to VM exception format |
| **Graceful Degradation** | System behavior that falls back to interpretation when JIT compilation fails | Robustness requirement for production systems |
| **Exception Marshaling** | Converting between native exception context and virtual machine exception format | Bridge between execution environments |
| **Signal Handler** | Operating system mechanism for handling hardware exceptions and signals | Interface for catching native code exceptions |
| **Exception Context** | CPU register and stack state captured when an exception occurs | Information needed for exception handling and debugging |
| **Fail-Safe Compilation** | Ensuring compilation failures don't corrupt interpreter state or program execution | Safety requirement for JIT integration |
| **Resource Cleanup** | Proper deallocation of memory and system resources when compilation fails | Memory management responsibility |
| **Error Recovery** | Process of returning to a consistent state after compilation or runtime errors | Fault tolerance mechanism |
| **Transparent Recovery** | Error handling that is invisible to the executing program | Seamless fallback maintaining program semantics |

### Testing and Verification Concepts

Ensuring JIT compiler correctness requires comprehensive testing strategies that verify both functional correctness and performance characteristics.

| Term | Definition | Context |
|------|------------|---------|
| **Correctness Verification** | Ensuring JIT-compiled code produces identical results to bytecode interpretation | Primary requirement for JIT compiler validation |
| **Differential Testing** | Systematic comparison of results between interpreter and JIT execution modes | Core testing methodology for correctness |
| **Reference Implementation** | Treating bytecode interpreter as the authoritative standard for correct behavior | Baseline for JIT compiler validation |
| **Dual-Mode Execution** | Running identical code in both interpreter and JIT modes for result comparison | Testing technique for correctness verification |
| **Performance Benchmarking** | Measuring and validating the performance benefits of JIT compilation | Validation of optimization effectiveness |
| **Regression Testing** | Detecting correctness or performance degradation across code changes | Quality assurance for ongoing development |
| **Statistical Analysis** | Applying statistical methods to detect genuine performance changes in noisy measurements | Rigorous approach to performance validation |
| **Amortized Performance** | Accounting for compilation overhead across multiple function executions | Realistic performance measurement approach |
| **Test Coverage** | Measure of how thoroughly test cases exercise the JIT compiler implementation | Quality metric for test suite completeness |
| **Milestone Checkpoint** | Verification step confirming correct implementation after each development stage | Structured approach to incremental validation |

### Debugging and Diagnostics

Complex JIT compilation systems require sophisticated debugging capabilities to diagnose issues in generated code and compilation processes.

| Term | Definition | Context |
|------|------------|---------|
| **Machine Code Disassembly** | Converting binary instructions back to human-readable assembly language | Essential debugging tool for generated code inspection |
| **Execution Trace** | Record of program execution including instruction sequence and state changes | Debugging aid for comparing interpreter and JIT behavior |
| **Execution Divergence** | Point where interpreter and JIT execution produce different results | Diagnostic indicator of JIT compilation bugs |
| **Memory Corruption** | Unintended modification of memory contents, often causing unpredictable behavior | Common category of JIT compiler bugs |
| **Segmentation Fault** | Hardware exception caused by accessing memory without proper permissions | Common symptom of incorrect address generation |
| **Illegal Instruction** | Hardware exception caused by invalid or malformed instruction encoding | Symptom of incorrect machine code generation |
| **Stack Overflow** | Condition where function call depth exceeds available stack space | Potential issue with ABI compliance or recursion |
| **Register Corruption** | Unintended modification of processor register contents | ABI violation affecting program correctness |
| **Debug Symbol** | Metadata associating generated machine code with source bytecode locations | Aid for debugging and profiling JIT-compiled code |
| **Crash Dump** | Snapshot of program state at the time of failure | Post-mortem debugging information |

### Advanced JIT Concepts

These terms represent sophisticated JIT compilation techniques that extend beyond basic bytecode-to-native translation.

| Term | Definition | Context |
|------|------------|---------|
| **On-Stack Replacement** | Technique to replace executing code with optimized version without function boundaries | Advanced optimization requiring complex state management |
| **Dynamic Recompilation** | Creating multiple optimized versions of the same function for different contexts | Adaptive optimization strategy |
| **Code Versioning** | Maintaining multiple compiled representations of the same function | Infrastructure for dynamic recompilation |
| **Inlining** | Optimization replacing function calls with copies of function body | Aggressive optimization reducing call overhead |
| **Loop Optimization** | Specialized optimizations for repetitive code structures | Performance optimization for computational kernels |
| **Escape Analysis** | Determining whether object references can escape local scope | Optimization enabling stack allocation |
| **Alias Analysis** | Determining whether different memory references might refer to the same location | Foundation for memory optimization |
| **Interprocedural Analysis** | Analysis across function boundaries to enable global optimizations | Advanced compiler analysis technique |
| **Runtime Feedback** | Using execution behavior to guide subsequent optimization decisions | Adaptive optimization strategy |
| **Adaptive Compilation** | Dynamically adjusting compilation strategies based on runtime characteristics | Self-tuning JIT compilation approach |

### System Integration Terms

JIT compilation requires integration with operating system services and existing virtual machine infrastructure.

| Term | Definition | Context |
|------|------------|---------|
| **System Call** | Interface for requesting operating system services from user-space programs | Required for memory management and signal handling |
| **Memory Mapping** | Operating system service for allocating and managing virtual memory regions | Foundation for executable memory allocation |
| **Signal Handling** | Operating system mechanism for delivering asynchronous notifications to processes | Interface for handling hardware exceptions |
| **Process Address Space** | Virtual memory layout of a running program | Context for executable memory allocation |
| **Dynamic Linking** | Runtime resolution of function references between separately compiled modules | Integration challenge for JIT-compiled code |
| **Thread Safety** | Property of code that functions correctly in multithreaded environments | Requirement for JIT compiler implementation |
| **Atomic Operations** | Instructions that complete as a single, uninterruptible unit | Primitives for thread-safe data structure updates |
| **Memory Barriers** | Instructions ensuring memory operation ordering across processor cores | Concurrency primitive for shared data structures |
| **Cache Coherence** | Hardware protocol ensuring consistent view of memory across processor cores | Underlying mechanism affecting performance |
| **Virtual Memory System** | Operating system subsystem managing process address spaces | Platform service underlying executable memory management |

### Performance and Optimization Metrics

Measuring JIT compilation effectiveness requires understanding various performance metrics and their implications.

| Term | Definition | Context |
|------|------------|---------|
| **Throughput** | Rate of useful work completion, measured in operations per unit time | Primary performance metric for computational code |
| **Latency** | Time delay between request and response, critical for interactive applications | Performance metric for individual operations |
| **Compilation Overhead** | Performance cost of translating bytecode to native code | Must be amortized across multiple executions |
| **Warm-up Time** | Period required for JIT optimizations to reach steady-state performance | Characteristic behavior of JIT-compiled systems |
| **Code Quality** | Efficiency of generated machine code compared to optimal hand-written assembly | Measure of JIT compiler sophistication |
| **Memory Overhead** | Additional memory required for JIT compilation infrastructure and generated code | Resource cost of JIT compilation |
| **Compilation Speed** | Rate at which bytecode is translated to native machine code | Affects compilation overhead and warm-up time |
| **Execution Speed** | Rate of native code execution compared to interpretation | Primary benefit justifying JIT compilation cost |
| **Optimization Effectiveness** | Degree to which optimizations improve performance over baseline code generation | Measure of optimization pass value |
| **Resource Utilization** | Efficiency of processor, memory, and system resource usage | Overall system efficiency metric |

### Future Extension Terminology

These terms represent advanced capabilities that extend beyond the basic JIT compiler implementation.

| Term | Definition | Context |
|------|------------|---------|
| **Multi-Architecture Support** | Capability to generate native code for different processor architectures | Portability extension for cross-platform deployment |
| **Virtual Instruction** | Architecture-neutral intermediate representation between bytecode and native code | Abstraction enabling multi-architecture support |
| **Architecture Backend** | Component responsible for generating code for a specific processor architecture | Modular design enabling multiple targets |
| **Cross-Compilation** | Generating code for a different architecture than the one running the compiler | Deployment flexibility for diverse environments |
| **Garbage Collection Integration** | Coordination between JIT-compiled code and automatic memory management | Advanced runtime system integration |
| **Object Layout Optimization** | Optimizing memory layout of data structures based on access patterns | Advanced optimization requiring runtime analysis |
| **Polymorphic Inline Cache** | Optimization technique for dynamic method dispatch in object-oriented languages | Advanced optimization for dynamic languages |
| **Trace Compilation** | JIT strategy that compiles frequently executed paths through multiple functions | Alternative compilation approach to function-based compilation |
| **Superblock Formation** | Creating large compilation units by combining frequently executed basic blocks | Advanced compilation strategy for complex control flow |
| **Profile-Directed Feedback** | Using detailed execution profiles to guide aggressive optimization decisions | Advanced optimization strategy requiring sophisticated profiling |

### x86-64 Instruction Set Terminology

Understanding x86-64 instructions is essential for implementing the machine code emitter and verifying generated code correctness.

| Instruction | Encoding | Purpose | Usage Context |
|-------------|----------|---------|---------------|
| **MOV** | 0xB8 + reg (reg, imm) | Move immediate value to register | Loading constants and transferring values |
| **ADD** | 0x01 (reg, reg) | Add source register to destination register | Arithmetic operations |
| **SUB** | 0x29 (reg, reg) | Subtract source register from destination register | Arithmetic operations |
| **MUL** | 0xF7 /4 (reg) | Unsigned multiply with rax | Multiplication operations |
| **IMUL** | 0xF7 /5 (reg) | Signed multiply with rax | Signed multiplication |
| **DIV** | 0xF7 /6 (reg) | Unsigned divide rdx:rax by operand | Division operations |
| **IDIV** | 0xF7 /7 (reg) | Signed divide rdx:rax by operand | Signed division |
| **CMP** | 0x39 (reg, reg) | Compare registers and set flags | Conditional operations |
| **JMP** | 0xEB (rel8), 0xE9 (rel32) | Unconditional jump to target | Control flow transfer |
| **JZ/JE** | 0x74 (rel8), 0x0F 0x84 (rel32) | Jump if zero/equal flag set | Conditional control flow |
| **JNZ/JNE** | 0x75 (rel8), 0x0F 0x85 (rel32) | Jump if zero/equal flag clear | Conditional control flow |
| **CALL** | 0xE8 (rel32), 0xFF /2 (reg) | Call function at target address | Function invocation |
| **RET** | 0xC3 | Return from function | Function completion |
| **PUSH** | 0x50 + reg | Push register onto stack | Stack manipulation |
| **POP** | 0x58 + reg | Pop stack top into register | Stack manipulation |
| **CQO** | 0x48 0x99 | Sign-extend rax to rdx:rax | Division preparation |

### Register and Addressing Terminology

x86-64 register organization and addressing modes form the foundation for register allocation and instruction generation.

| Register | Encoding | ABI Role | Allocation Strategy |
|----------|----------|----------|-------------------|
| **RAX** | 0 | Return value, accumulator | Temporary, return results |
| **RBX** | 3 | Callee-saved general purpose | Long-lived values |
| **RCX** | 1 | 4th argument register | Parameter passing, temporary |
| **RDX** | 2 | 3rd argument, multiply/divide high | Parameter passing, arithmetic |
| **RSI** | 6 | 2nd argument register | Parameter passing |
| **RDI** | 7 | 1st argument register | Parameter passing |
| **RBP** | 5 | Frame pointer (optional) | Stack frame reference |
| **RSP** | 4 | Stack pointer | Stack management |
| **R8** | 8 | 5th argument register | Parameter passing, temporary |
| **R9** | 9 | 6th argument register | Parameter passing, temporary |
| **R10** | 10 | Caller-saved temporary | Temporary values |
| **R11** | 11 | Caller-saved temporary | Temporary values |
| **R12** | 12 | Callee-saved general purpose | Long-lived values |
| **R13** | 13 | Callee-saved general purpose | Long-lived values |
| **R14** | 14 | Callee-saved general purpose | Long-lived values |
| **R15** | 15 | Callee-saved general purpose | Long-lived values |

### Implementation Guidance

This comprehensive terminology reference serves as the foundation for consistent communication throughout the JIT compiler implementation. Each term represents a precisely defined concept with specific implications for system design and implementation.

#### Terminology Usage Guidelines

When implementing the JIT compiler, consistent use of this terminology ensures clear communication within the development team and with external stakeholders. The following guidelines help maintain consistency:

**Architecture-Specific Terms**: Always specify the target architecture when discussing instruction sets, calling conventions, or register allocation. Use "x86-64" rather than generic terms like "assembly" or "machine code" when the discussion is specific to the Intel/AMD 64-bit architecture.

**Precision in Error Descriptions**: When documenting or reporting errors, use precise terminology to distinguish between different failure modes. "Compilation error" refers to failures during bytecode-to-native translation, while "runtime exception" refers to errors during execution of JIT-compiled code.

**ABI and Calling Convention References**: Always specify "System V AMD64 ABI" when discussing calling conventions, as different platforms use different standards. This precision prevents confusion and ensures correct implementation.

**Performance and Optimization Context**: Distinguish between different types of optimizations by using specific terms like "constant folding," "dead code elimination," and "runtime specialization" rather than generic terms like "optimization" or "improvement."

#### Cross-Reference Integration

This glossary integrates with all other sections of the design document, providing precise definitions for terms used throughout the system architecture, implementation guidance, and testing strategies. When encountering unfamiliar terminology in other sections, refer to this glossary for authoritative definitions and contextual information.

The terminology establishes the conceptual framework for understanding JIT compilation as a systematic engineering discipline, with each term representing a specific aspect of the complex interactions between virtual machines, processor architecture, memory management, and runtime optimization. Mastery of this vocabulary enables clear thinking about design trade-offs, implementation strategies, and debugging approaches throughout the JIT compiler development process.

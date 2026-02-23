# DOMAIN PROFILE: Compilers & Language Implementation
# Applies to: compilers
# Projects: tokenizer, AST, interpreter, bytecode VM, JIT, GC, regex, type checker, linker, etc.

## Fundamental Tension Type
Formal language theory. EXPRESSIVENESS (what programmer wants to write) vs ANALYZABILITY (what compiler can reason about/optimize). Every language decision sits on this spectrum.

Secondary: compile time vs runtime perf, type system expressiveness vs decidability (Rice's theorem), grammar ambiguity resolution.

## Three-Level View
- **Level 1 — Source Language**: Code the programmer writes, semantics, syntax sugar
- **Level 2 — Intermediate Representation**: AST, IR, bytecode, SSA — how compiler sees the program
- **Level 3 — Target/Runtime**: Machine code, VM execution, registers, calling conventions, GC pauses

## Soul Section: "Formal Soul"
- Grammar class? (Regular → CF → CS)
- Theoretical limit? (Halting problem, Rice's, undecidability)
- Time complexity of this analysis/transformation?
- Sound (rejects all bad) or complete (accepts all good)?
- What does this optimization ASSUME? When does assumption break?

## Alternative Reality Comparisons
GCC/LLVM, V8/SpiderMonkey, CPython/PyPy, Go compiler, Rust compiler, Lua/LuaJIT, Zig, Cranelift.

## TDD Emphasis
- AST node definitions: MANDATORY — enum/struct for every node
- Token definitions: MANDATORY — every variant with pattern
- Grammar: BNF/EBNF for parsed language
- Memory layout: YES for AST nodes, symbol tables, bytecode instructions, GC object headers. NO for helpers.
- Visitor/walker contracts: interface specs
- Tests: input source → expected AST/bytecode/output per feature
- Benchmarks: parse lines/sec, compile time, runtime vs reference
- Skip: cache line alignment, lock ordering (unless concurrent GC)

## Cross-Domain Notes
Borrow from systems-lowlevel when: JIT compilation (executable memory, mprotect), GC (page management, mmap).
Borrow from ai-ml when: DSL for ML (custom operators, autodiff).




## Artist Examples for This Domain
- **data_walk**: String "x = 5" -> Lexer -> Tokens [IDENT, EQ, INT] -> Parser -> AST Node [Assign].
- **structure_layout**: Stack frame layout during a function call (Params, Return Addr, Locals).
- **state_evolution**: Symbol Table state before and after entering a new scope.
- **bytecode_walk**: Instructions being fetched from the instruction stream into the VM register.

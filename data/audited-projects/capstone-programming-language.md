# AUDIT & FIX: capstone-programming-language

## CRITIQUE
- The essence mentions 'Hindley-Milner type inference unification' combined with 'generics with constraint checking'. Hindley-Milner infers principal types for parametric polymorphism, but constraint checking (bounded polymorphism / type classes) is a substantial extension not covered by vanilla HM. The project should clarify whether it's HM with let-polymorphism or a constraint-based system, as mixing them is a research-level problem that's grossly under-scoped.
- Milestone 2 lists 'Generic functions and types with constraint checking' as an AC but provides no guidance on the constraint language (interfaces? type classes? trait bounds?). This is a design decision that affects the entire type system and should be specified.
- Milestone 2 does not mention Symbol Table construction, which is foundational for the claimed scope resolution and use-before-define detection. The audit correctly identified this.
- Milestone 3 mentions 'SSA form' in concepts and 'register allocation' in ACs, but the bytecode VM in M4 is stack-based. SSA and register allocation are irrelevant for a stack-based bytecode compiler — they're relevant for a register-based IR or native code generation. This is a conceptual mismatch.
- Milestone 4 deliverables mention 'Write barrier implementation tracking cross-generational references for incremental GC' but the AC only requires mark-and-sweep. Write barriers and generational/incremental GC are far beyond the scope of this milestone and contradict the stated acceptance criteria.
- Milestone 5 claims 'Self-hosting compiler written in the language itself' as a deliverable, but the AC only says 'run a non-trivial program (e.g., a simple calculator)'. Self-hosting is orders of magnitude harder and would likely double the project hours. These are contradictory.
- Estimated hours (80-120) is aggressive given the scope includes type inference, generics, optimization passes, GC, stdlib, and REPL. A more realistic estimate is 120-200.
- 'Closures compile to captured variable environments' in M3 is listed as an AC but closure conversion is a complex topic that needs its own detailed treatment.
- No mention of error recovery strategy in the type checker — how to continue checking after a type error.

## FIXED YAML
```yaml
id: capstone-programming-language
name: "Capstone: Complete Programming Language"
description: >-
  Build a complete statically-typed programming language from scratch —
  lexer, parser, symbol table, type checker with inference, bytecode
  compiler, stack-based virtual machine, and mark-and-sweep garbage
  collector. Integrates all compiler/language sub-projects into one
  cohesive executable pipeline.
difficulty: expert
estimated_hours: "120-200"
essence: >-
  Recursive descent parsing with lookahead, symbol table construction for
  lexical scoping, Hindley-Milner-inspired type inference with let-
  polymorphism, stack-based bytecode compilation with constant folding,
  bytecode interpretation with computed-goto dispatch, and tricolor
  mark-sweep garbage collection — orchestrating lexical scanning, context-
  free grammar recognition, static type constraint solving, and automatic
  memory reclamation into a single executable pipeline.
why_important: >-
  Building a complete language from scratch gives you the deepest possible
  understanding of how programming languages work — from parsing to
  execution. This knowledge makes you better at debugging, optimizing,
  and designing APIs in any language, and is foundational for compiler
  engineering and runtime development roles.
learning_outcomes:
  - Design and implement a complete language pipeline from source text to execution
  - Build a symbol table with lexical scoping, shadowing, and forward reference resolution
  - Implement type inference with let-polymorphism and parametric generics
  - Design a stack-based bytecode instruction set balancing simplicity and expressiveness
  - Compile closures using upvalue capture and closure conversion
  - Build a mark-and-sweep garbage collector integrated with the VM
  - Implement a standard library with I/O, collections, and string operations
  - Write comprehensive test suites covering parsing, type errors, and runtime behavior
skills:
  - Lexical Analysis
  - Parsing and Error Recovery
  - Symbol Tables and Scope Resolution
  - Type Systems and Inference
  - Bytecode Compilation
  - Virtual Machine Implementation
  - Garbage Collection
  - Language Design
tags:
  - compilers
  - languages
  - capstone
  - systems-programming
  - expert
architecture_doc: architecture-docs/capstone-programming-language/index.md
languages:
  recommended:
    - Rust
    - C
    - Go
  also_possible:
    - Java
resources:
  - name: Crafting Interpreters""
    url: https://craftinginterpreters.com/
    type: book
  - name: Writing An Interpreter In Go""
    url: https://interpreterbook.com/
    type: book
  - name: Engineering a Compiler (3rd Edition)""
    url: https://www.elsevier.com/books/engineering-a-compiler/cooper/978-0-12-815412-0
    type: book
  - name: Types and Programming Languages (Pierce)""
    url: https://www.cis.upenn.edu/~bcpierce/tapl/
    type: book
prerequisites:
  - type: project
    id: tokenizer
    name: Tokenizer/Lexer
  - type: project
    id: ast-builder
    name: AST Builder
  - type: project
    id: type-checker
    name: Type Checker
  - type: project
    id: bytecode-compiler
    name: Bytecode Compiler
  - type: project
    id: bytecode-vm
    name: Bytecode VM
  - type: project
    id: simple-gc
    name: Simple GC
milestones:
  - id: capstone-programming-language-m1
    name: "Language Design, Lexer, and Parser"
    description: >-
      Design the language syntax and semantics. Implement a lexer and
      recursive descent parser that produces an AST with source location
      tracking and panic-mode error recovery.
    acceptance_criteria:
      - >-
        Language specification document defines: primitive types (int, float,
        bool, string), control flow (if/else, while, for), functions with
        parameters and return types, struct/record types, and module/import
        syntax.
      - >-
        Formal grammar is documented in EBNF notation covering all language
        constructs, with operator precedence table.
      - >-
        Lexer produces tokens with source location metadata (file, line,
        column, span) for every token type in the language.
      - >-
        Lexer correctly handles string literals with escape sequences
        (\n, \t, \\\", \\\\), multiline strings, and integer/float
        literal formats.
      - >-
        Parser produces a typed AST with source spans attached to every
        node for error reporting.
      - >-
        Panic-mode error recovery: parser synchronizes on statement
        boundaries after syntax errors, reporting at least 3 distinct
        errors in a file with multiple syntax mistakes.
      - >-
        Parser correctly handles operator precedence (unary > multiplicative
        > additive > comparison > logical) and associativity without
        ambiguity.
      - >-
        Test suite covers: all token types, operator precedence, nested
        expressions, error recovery, malformed inputs.
    pitfalls:
      - >-
        Ambiguous grammar requiring backtracking: use precedence climbing
        or Pratt parsing for expressions to avoid LL(1) conflicts.
      - >-
        Poor error messages: include expected token type, found token,
        and source location in every parse error.
      - >-
        Operator precedence conflicts between unary minus and binary minus
        — handle in the expression parser with correct binding power.
      - >-
        Not designing the grammar for the type system upfront: adding
        generics syntax later often requires grammar changes that break
        existing parsing.
    concepts:
      - Language design and grammar specification
      - Lexical analysis and tokenization
      - Recursive descent parsing
      - Panic-mode error recovery
      - Source location tracking
    skills:
      - Language design
      - Lexer implementation
      - Recursive descent parser
      - Error recovery strategies
    deliverables:
      - Language specification document with EBNF grammar
      - Lexer with source location metadata on all tokens
      - Recursive descent parser with AST output
      - Panic-mode error recovery reporting multiple errors per file
      - Test suite for lexer and parser
    estimated_hours: "18-25"

  - id: capstone-programming-language-m2
    name: "Symbol Table, Type Checker, and Semantic Analysis"
    description: >-
      Build a symbol table for lexical scoping, implement type checking
      with Hindley-Milner-inspired inference and let-polymorphism, and
      perform semantic analysis.
    acceptance_criteria:
      - >-
        Symbol table maps identifiers to their declarations with scope
        nesting: inner scopes shadow outer scopes, and lookup traverses
        the scope chain.
      - >-
        Use-before-define errors are detected and reported with the
        location of the usage and a suggestion to declare the variable.
      - >-
        Type inference for local variables works correctly (let x = 42
        infers int; let f = fn(a) { a + 1 } infers fn(int) -> int).
      - >-
        Let-polymorphism: a polymorphic function like 'let id = fn(x) { x }'
        can be used at multiple types (id(42) and id("hello")) without
        explicit type annotations.
      - >-
        Parametric generic types (e.g., List<T>, Option<T>) are supported
        with correct instantiation and unification.
      - >-
        Type error messages include expected type, actual type, and source
        location of both the expectation and the offending expression.
      - >-
        Mutually recursive functions are supported with explicit type
        annotations (inference for mutual recursion is optional).
      - >-
        Type checker continues after errors (error recovery) and reports
        all type errors in a file, not just the first one.
      - >-
        Semantic analysis detects: unreachable code after return, unused
        variables (warning), and missing return in non-void functions.
    pitfalls:
      - >-
        Infinite types from recursive inference (e.g., let f = fn(x) { f })
        must be detected with an occurs check during unification. Without
        it, the unifier loops infinitely.
      - >-
        Confusing let-polymorphism with rank-2 polymorphism. HM only
        generalizes at let-bindings, not at function arguments. This
        restriction must be clearly documented.
      - >-
        Symbol table lifetime: in languages with closures, captured
        variables must remain resolvable even after the enclosing scope
        exits. The symbol table must support upvalue resolution.
      - >-
        Forgetting to instantiate polymorphic types at each use site
        leads to false unification between unrelated type variables.
    concepts:
      - Symbol tables and lexical scoping
      - Hindley-Milner type inference
      - Unification algorithm
      - Let-polymorphism and generalization
      - Occurs check for infinite type detection
    skills:
      - Symbol table construction
      - Type inference and unification
      - Semantic analysis
      - Error recovery in type checking
    deliverables:
      - Symbol table with scope chain, shadowing, and upvalue resolution
      - Type inference engine with unification and let-polymorphism
      - Occurs check preventing infinite types
      - Semantic analyzer (unreachable code, unused variables, missing returns)
      - Type error reporting with expected vs. actual and source locations
      - Test suite for scoping, inference, generics, and error cases
    estimated_hours: "25-35"

  - id: capstone-programming-language-m3
    name: "Bytecode Compiler with Optimization"
    description: >-
      Compile the typed AST to a stack-based bytecode instruction set
      with closure conversion, constant folding, and dead code elimination.
    acceptance_criteria:
      - >-
        Bytecode instruction set is documented: each opcode has a name,
        operand format, stack effect (number of values consumed/produced),
        and semantics description.
      - >-
        All typed AST nodes compile to correct bytecode sequences, verified
        by round-trip testing (compile, disassemble, check expected output).
      - >-
        Constant folding evaluates compile-time constant expressions
        (e.g., 2 + 3 compiles to CONST 5, not CONST 2; CONST 3; ADD).
      - >-
        Dead code elimination removes bytecode after unconditional returns
        or jumps that cannot be reached.
      - >-
        Closures are compiled with upvalue capture: free variables in
        the closure body are resolved to upvalue indices at compile time,
        and a CLOSURE instruction creates the closure object at runtime.
      - >-
        Bytecode disassembler outputs human-readable instruction listing
        with operand values and source line mappings.
      - >-
        Compiled bytecode includes a constant pool for string and numeric
        literals, and a line number table for runtime error reporting.
    pitfalls:
      - >-
        Incorrect optimization changing program semantics: constant folding
        must not fold expressions with side effects (function calls,
        variable assignments) or division by zero.
      - >-
        Closure upvalue resolution: variables captured by closures must
        be 'closed over' (moved to heap) when the enclosing scope exits.
        Forgetting this causes dangling stack references.
      - >-
        Stack effect miscounting: if any bytecode instruction's stack
        effect is wrong, the VM stack will underflow or overflow at
        runtime. Validate stack depth statically during compilation.
      - >-
        Line number table granularity: mapping too few bytecodes to
        source lines makes runtime errors unhelpful.
    concepts:
      - Stack-based bytecode design
      - Constant folding and dead code elimination
      - Closure conversion and upvalue capture
      - Constant pool and line number tables
    skills:
      - Bytecode instruction set design
      - Optimization passes
      - Closure conversion
      - Bytecode serialization
    deliverables:
      - Documented bytecode instruction set with stack effects
      - AST-to-bytecode compiler for all language constructs
      - Constant folding and dead code elimination passes
      - Closure compilation with upvalue capture
      - Bytecode disassembler with source line mapping
      - Constant pool and line number table
    estimated_hours: "20-28"

  - id: capstone-programming-language-m4
    name: "Virtual Machine and Garbage Collector"
    description: >-
      Implement a stack-based VM executing the bytecode, with a stop-the-
      world mark-and-sweep garbage collector for automatic memory management.
    acceptance_criteria:
      - >-
        VM executes all bytecode instructions with correct semantics,
        verified against a test suite of at least 50 programs covering
        arithmetic, control flow, functions, closures, and structs.
      - >-
        Instruction dispatch uses computed goto (GCC/Clang) or a dense
        switch for performance, not a chain of if-else.
      - >-
        Mark-and-sweep GC correctly identifies all roots (VM stack, global
        variables, open upvalues, compiler temporaries) and frees all
        unreachable objects.
      - >-
        GC triggers automatically when allocated bytes exceed a
        configurable threshold (e.g., 1MB), with the threshold growing
        after each collection (adaptive scheduling).
      - >-
        Runtime errors (division by zero, type mismatch, stack overflow)
        produce stack traces with source file, line number, and function
        name.
      - >-
        Stack overflow detection: VM detects when call depth exceeds a
        configurable limit (e.g., 1024 frames) and reports a runtime
        error instead of crashing.
      - >-
        GC stress test mode: trigger GC on every allocation to catch
        missing root registrations and dangling pointer bugs.
    pitfalls:
      - >-
        Dangling references from incorrect root scanning: if the compiler
        creates a temporary GC object (e.g., concatenated string) that
        is not on the VM stack, a GC triggered before the temporary is
        stored will free it. Push all temporaries to the stack or a
        root set before allocating again.
      - >-
        GC during compilation: if the compiler allocates GC objects (string
        constants, function objects), a GC triggered during compilation
        must be able to trace the compiler's in-progress state.
      - >-
        Stack overflow from deeply recursive programs: use an explicit
        frame limit check, not reliance on the C stack (which produces
        segfaults instead of error messages).
      - >-
        Upvalue closing: when a local variable goes out of scope but is
        captured by a closure, it must be moved from the stack to the
        heap. Failing to close upvalues causes closures to read garbage.
    concepts:
      - Stack-based bytecode interpretation
      - Instruction dispatch strategies
      - Stop-the-world mark-and-sweep GC
      - Root scanning and object tracing
      - Adaptive GC scheduling
    skills:
      - VM implementation
      - Garbage collection
      - Memory management
      - Runtime error handling
    deliverables:
      - Stack-based bytecode interpreter with computed-goto or switch dispatch
      - Mark-and-sweep GC with root scanning of stack, globals, and upvalues
      - Adaptive GC scheduling with configurable thresholds
      - GC stress test mode for debugging
      - Stack traces with source locations on runtime errors
      - Stack overflow detection and reporting
    estimated_hours: "22-30"

  - id: capstone-programming-language-m5
    name: "Standard Library, REPL, and Integration Testing"
    description: >-
      Build a standard library with I/O, collections, and strings. Implement
      an interactive REPL. Write integration tests demonstrating the
      language can run non-trivial programs.
    acceptance_criteria:
      - >-
        Standard library provides: file I/O (open, read, write, close),
        print/println, string operations (length, substring, concatenation,
        split), math functions (abs, min, max, sqrt), and a dynamic
        array/list type.
      - >-
        Standard library functions are callable from user code through
        a native function interface (the VM can call host-language
        functions).
      - >-
        REPL supports: line editing (readline or equivalent), command
        history, and multiline input (detects incomplete expressions and
        prompts for continuation).
      - >-
        REPL displays the type and value of each evaluated expression.
      - >-
        Integration test suite includes at least 5 non-trivial programs:
        fibonacci, a simple calculator (parsing and evaluating expressions),
        a linked list implementation, a sorting algorithm, and a string
        processing utility.
      - >-
        All integration test programs produce correct output, verified
        by automated test harness comparing actual output to expected
        output.
      - >-
        Error handling: standard library functions return error values
        (not crash) on invalid input (e.g., file not found, invalid
        argument).
    pitfalls:
      - >-
        Standard library performance: native functions must manage GC
        roots correctly. If a native function allocates GC objects, it
        must register them as roots to prevent collection.
      - >-
        REPL state management: variables defined in one REPL line must
        be available in subsequent lines, requiring persistent global
        scope across evaluations.
      - >-
        Self-hosting is NOT required for this milestone. It is a stretch
        goal that would roughly double the project scope. Don't conflate
        'run a non-trivial program' with 'self-hosting compiler'.
      - >-
        Native function interface must handle type checking at the boundary
        between user code and native code to prevent type-unsafe calls.
    concepts:
      - Standard library design
      - Native function interface (FFI)
      - REPL loop with persistent state
      - Integration testing
    skills:
      - Standard library design
      - REPL implementation
      - Native function interface
      - Integration testing
    deliverables:
      - Standard library modules (I/O, strings, math, collections)
      - Native function interface for host-language function calls
      - Interactive REPL with readline, history, and multiline support
      - Integration test suite with 5+ non-trivial programs
      - Automated test harness for output comparison
    estimated_hours: "15-20"
```
# AUDIT & FIX: wasm-emitter

## CRITIQUE
- **Technical Inaccuracy (Confirmed - Loop/Branch Direction):** Milestone 3 describes 'Loop with br_if' as a backward branch. In WASM, jumping to a 'loop' label goes to the START of the loop body, but jumping to a 'block' label goes to the END of the block. This distinction is critical for structured control flow and differs from typical assembly.
- **Logical Gap (Confirmed - Function Section):** Milestone 4 mentions function calls by index but omits the 'Function Section' which maps these indices to the 'Type Section' signatures. Without this section, the WASM module is invalid—the function index alone doesn't specify the signature.
- **Hour Ranges:** Using ranges like '5-8' is imprecise. Converting to single estimates.
- **Estimated Hours:** 25-40 range is reasonable; estimate ~32 hours.

## FIXED YAML
```yaml
id: wasm-emitter
name: WebAssembly Emitter
description: >-
  AST to WebAssembly binary module compilation with LEB128 encoding,
  structured control flow, function sections, and proper export declarations.
difficulty: advanced
estimated_hours: 32
essence: >-
  Stack-based bytecode generation targeting WebAssembly's binary module format,
  requiring manual encoding of type sections, function signatures, and
  instruction opcodes while maintaining structured control flow constraints.
why_important: >-
  Building a WASM compiler teaches low-level code generation and binary format
  encoding—fundamental skills for systems programming, language implementation,
  and understanding how high-level code becomes executable machine instructions.
learning_outcomes:
  - Emit valid WASM binary module structure with correct section ordering
  - Encode LEB128 integers for variable-length encoding
  - Compile expressions to WASM stack instructions
  - Generate structured control flow with correct block/loop semantics
  - Emit function sections mapping indices to type signatures
  - Export functions for host access
skills:
  - Binary Format Encoding
  - LEB128 Encoding
  - Stack-Based Code Generation
  - Structured Control Flow
  - WASM Module Structure
tags:
  - webassembly
  - wasm
  - compiler
  - binary-format
  - advanced
architecture_doc: architecture-docs/wasm-emitter/index.md
languages:
  recommended:
    - Rust
    - C
    - Go
  also_possible:
    - Python
    - TypeScript
resources:
  - name: WebAssembly Specification
    url: https://webassembly.github.io/spec/core/
    type: documentation
  - name: WASM Binary Toolkit
    url: https://github.com/WebAssembly/wabt
    type: tool
  - name: WASM Opcode Matrix
    url: https://pengowray.github.io/wasm-ops/
    type: reference
prerequisites:
  - type: skill
    name: Binary file format concepts
  - type: skill
    name: Stack machine understanding
  - type: skill
    name: AST traversal
milestones:
  - id: wasm-emitter-m1
    name: WASM Binary Format
    description: >-
      Understand and emit valid WASM module structure with sections.
    acceptance_criteria:
      - Magic number 0x00 0x61 0x73 0x6D and version 1 header are emitted correctly
      - Section encoding writes section ID byte followed by LEB128-encoded content length
      - LEB128 integers encode both signed and unsigned values in variable-length format
      - Type section lists all function signatures used by the module
      - Sections are emitted in the correct order (type before function, function before code, etc.)
    pitfalls:
      - LEB128 edge cases: values near 128, 16384, etc. require multiple bytes
      - Section ordering: WASM requires sections in specific numeric order
      - Size calculation: section size must be computed BEFORE writing the section content
    concepts:
      - Binary formats have fixed headers and variable-length sections
      - LEB128 encoding compresses common small integers
      - Sections organize module content by type
    skills:
      - Binary file format design
      - Low-level byte manipulation
      - WebAssembly module structure
      - Variable-length integer encoding
    deliverables:
      - Module structure emitter writing the top-level WASM binary module wrapper
      - Section encoding logic writing section IDs, sizes, and payloads in binary format
      - LEB128 encoding functions converting integers to variable-length binary representation
      - Type section emitter writing function signature definitions with parameter and return types
    estimated_hours: 8

  - id: wasm-emitter-m2
    name: Expression Compilation
    description: >-
      Compile arithmetic expressions to WASM stack instructions.
    acceptance_criteria:
      - i32 arithmetic operations produce correct results via WASM stack instructions
      - Stack-based code generation pushes operands and applies operators in correct order
      - Local variables are read and written using local.get and local.set instructions
      - Constants emit correct i32.const or f64.const instructions with LEB128-encoded values
      - Operand stack depth is tracked to ensure correct final depth
    pitfalls:
      - Operand order for non-commutative ops: WASM is stack-based, left operand pushed first
      - Signed vs unsigned: WASM has distinct operations for signed vs unsigned (e.g., i32.div_s vs i32.div_u)
      - Stack imbalance: each code path must leave the stack at the same depth
    concepts:
      - Stack machines push operands, then apply operators
      - WASM has typed instructions (i32, i64, f32, f64)
      - Local variables are indexed within a function
    skills:
      - Stack-based VM programming
      - Expression tree compilation
      - Type-aware code generation
      - Register-less computation
    deliverables:
      - Literal value emitter producing WASM const instructions for integer and float values
      - Binary operation compiler generating WASM i32.add, i32.sub, i32.mul, i32.div_s instructions
      - Local variable compiler generating local.get and local.set instructions by index
      - Stack management logic ensuring operand stack depth is correct after each instruction
    estimated_hours: 8

  - id: wasm-emitter-m3
    name: Control Flow
    description: >-
      Compile if/else and loops with correct WASM block/loop semantics.
    acceptance_criteria:
      - block creates a label that br branches to the END (exits the block)
      - loop creates a label that br branches to the START (repeats the loop)
      - If/else/end implements two-branch conditional execution correctly
      - br_if conditionally branches based on stack top (1 = branch, 0 = fall through)
      - Break to labels targets the right enclosing block by depth index (0 = innermost)
    pitfalls:
      - Label depth calculation: depth 0 targets innermost enclosing structure
      - Block type annotations: blocks returning values need result type (e.g., (result i32))
      - Unreachable code after br: code after unconditional br is unreachable but must still validate
      - loop vs block semantics: br to loop jumps to START, br to block jumps to END—this is counterintuitive
    concepts:
      - Structured control flow: all control flow is block-based
      - Labels are indexed by depth from innermost
      - block is for forward jumps (exit), loop is for backward jumps (repeat)
    skills:
      - Structured control flow compilation
      - Label scope management
      - Branch target calculation
      - Loop and block construction
    deliverables:
      - Block structure emitter generating WASM block with matching end marker
      - Loop structure emitter generating WASM loop with matching end marker (br goes to start)
      - If-else emitter generating WASM if, else, and end instructions for conditional branches
      - Branch instruction emitter generating br and br_if for jumping to enclosing labels
      - Return instruction emitter generating WASM return to exit the current function
    estimated_hours: 10

  - id: wasm-emitter-m4
    name: Functions and Exports
    description: >-
      Compile function definitions with function sections and exports.
    acceptance_criteria:
      - Type section lists all function signatures (parameter types, return types)
      - Function section maps each function index to its corresponding type index
      - Code section format includes local count declarations followed by expression bytecode
      - Export section maps string names to function indices for host-callable entry points
      - Function calls emit the correct call instruction with the target function index
      - Import section can declare functions provided by the host runtime
    pitfalls:
      - Parameter vs local indices: parameters come first, then locals; index from 0
      - Body size encoding: function body size must be computed before writing
      - Export name encoding: export names are UTF-8 strings with length prefix
      - Function section vs Code section: function section has type indices, code section has bodies
    concepts:
      - Function section declares which type signature each function uses
      - Code section contains the actual bytecode for each function
      - Exports make functions visible to the host environment
      - Imports allow using host-provided functions
    skills:
      - Function signature design
      - Module export systems
      - ABI design and implementation
      - Function body encoding
    deliverables:
      - Type section emitter writing function signature definitions
      - Function section emitter mapping function indices to type indices
      - Code section emitter writing local declarations and instruction sequences
      - Export section emitter making selected functions visible by name to the host
      - Import section emitter declaring functions provided by the host runtime
      - Call instruction emitter generating function calls by index
    estimated_hours: 6
```

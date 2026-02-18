# AUDIT & FIX: ast-builder

## CRITIQUE
- **Logical Gap (Confirmed - Lookahead):** Milestone 2 (Recursive Descent) omits the concept of 'Lookahead' or 'Peeking', which is the technical requirement for a recursive descent parser to decide which production rule to apply. Without lookahead, the parser cannot distinguish between ambiguous grammar constructs like `foo * bar` (multiplication vs pointer declaration in C).
- **Technical Inaccuracy (Confirmed - Scope Resolution):** Milestone 3 mentions 'lexical scope boundaries' for block statements, but an AST Builder only represents structure; scope resolution is a semantic analysis task usually performed AFTER the AST is built, not during parsing. The parser should not attempt to build symbol tables or resolve scopes.
- **Hour Ranges:** Using ranges like '2-3' is imprecise. Converting to single estimates.
- **Estimated Hours:** 12-20 range is reasonable; estimate ~16 hours.

## FIXED YAML
```yaml
id: ast-builder
name: AST Builder
description: >-
  Token stream to abstract syntax tree transformation using recursive descent
  parsing with lookahead, operator precedence handling, and proper error
  recovery.
difficulty: intermediate
estimated_hours: 16
essence: >-
  Transforming flat token sequences into hierarchical tree structures that
  encode operator precedence, associativity, and syntactic relationships by
  recursively applying context-free grammar production rules and resolving
  ambiguities through parsing algorithms like recursive descent or precedence
  climbing.
why_important: >-
  Parsers are the foundation of every compiler, interpreter, linter, and code
  analysis tool. Understanding grammar-to-code translation and tree-based
  program representation is essential for building developer tools, language
  processors, and working with compiler frontends.
learning_outcomes:
  - Design AST node types for expressions, statements, and declarations
  - Implement recursive descent parsing with single-token lookahead
  - Handle operator precedence and associativity correctly
  - Build error recovery mechanisms for fault-tolerant parsing
  - Implement the visitor pattern for AST traversal
skills:
  - Recursive Descent Parsing
  - Operator Precedence
  - AST Design
  - Error Recovery
  - Visitor Pattern
tags:
  - parser
  - ast
  - compiler-frontend
  - intermediate
  - recursive-descent
architecture_doc: architecture-docs/ast-builder/index.md
languages:
  recommended:
    - Rust
    - Python
    - TypeScript
  also_possible:
    - Go
    - Java
resources:
  - name: Crafting Interpreters - Parsing
    url: https://craftinginterpreters.com/parsing-expressions.html
    type: book
  - name: Pratt Parsing
    url: https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/
    type: article
  - name: Recursive Descent
    url: https://www.engr.mun.ca/~theo/Misc/exp_parsing.htm
    type: tutorial
prerequisites:
  - type: skill
    name: Tree data structures
  - type: skill
    name: Recursion
  - type: skill
    name: Basic grammar/BNF notation
milestones:
  - id: ast-builder-m1
    name: AST Node Definitions
    description: >-
      Define AST node types for your language with source location tracking.
    acceptance_criteria:
      - Expression nodes represent literals, binary operators, unary operators, and identifiers as distinct types
      - Statement nodes represent if-else, while, return, and block constructs with correct child references
      - Visitor pattern or tagged union dispatch enables processing each node type without type-casting
      - Every AST node includes source location data (file, line, column) for error reporting and debugging
    pitfalls:
      - Missing node types: plan ahead for all language constructs before coding
      - No location info: source locations are critical for error messages—add them from day one
      - Mutable vs immutable nodes: immutable nodes are easier to reason about but may complicate optimization passes
    concepts:
      - AST nodes are the internal representation of program structure
      - Expression vs statement: expressions produce values, statements perform actions
      - Visitor pattern enables double dispatch for type-safe traversal
    skills:
      - Data structure design
      - Tree traversal implementation
      - Type system basics
      - Pattern matching
    deliverables:
      - Base Node class with source location tracking including file name, line number, and column offset
      - Expression node types covering literal values, binary operations, unary operations, and identifier references
      - Statement node types covering if-else branches, while loops, return statements, and block groupings
      - Visitor pattern implementation enabling type-safe tree traversal without modifying node classes
    estimated_hours: 3

  - id: ast-builder-m2
    name: Recursive Descent Parser
    description: >-
      Implement recursive descent parsing with lookahead for expressions.
    acceptance_criteria:
      - Each grammar production rule is implemented as a separate recursive parsing function
      - Single-token lookahead (peek) enables predictive parsing decisions without consuming tokens
      - Operator precedence is respected so multiplication binds tighter than addition in the resulting AST
      - Function call expressions are parsed with the callee followed by a parenthesized argument list
      - Token consumption advances the stream correctly and reports errors on unexpected token types
    pitfalls:
      - Wrong precedence order: verify * binds tighter than +, comparisons bind looser than arithmetic
      - Left recursion in grammar causes infinite loop—rewrite grammar to eliminate left recursion
      - Infinite loops from missing token consumption—every parsing function must advance the token stream
    concepts:
      - Recursive descent: one function per grammar rule, calling other functions recursively
      - Lookahead (peek) lets the parser see the next token without consuming it
      - Operator precedence determines which operations are grouped together
    skills:
      - Recursive function design
      - Operator precedence implementation
      - Grammar translation to code
      - Parser combinator patterns
    deliverables:
      - Token stream consumer with single-token lookahead (peek) and consume (advance) operations
      - Precedence climbing algorithm that correctly handles operator precedence levels and grouping
      - Recursive parsing functions with one function per grammar production rule
      - Operator associativity handling that correctly nests left-associative and right-associative operators
    estimated_hours: 6

  - id: ast-builder-m3
    name: Statement Parsing
    description: >-
      Parse statements and declarations without attempting scope resolution.
    acceptance_criteria:
      - Parser correctly handles let/var/const declarations with optional initializer expressions
      - If/else statements are parsed with condition, then-branch, and optional else-branch as child nodes
      - While loops are parsed with a condition expression and a body statement or block
      - Block statements group zero or more statements within braces—scope resolution happens later
      - Dangling else ambiguity is resolved (typically by associating else with nearest if)
    pitfalls:
      - Dangling else ambiguity: use else matches nearest if" rule and document it"
      - Missing semicolons: decide if semicolons are required or if newlines can terminate statements
      - Block scope boundaries: the parser builds the AST structure; semantic analysis builds scope info
    concepts:
      - Statement parsing builds the control flow structure of the AST
      - Declarations add names to the program but don't resolve them
      - Block statements create hierarchical structure for later semantic analysis
    skills:
      - Control flow parsing
      - Declaration handling
      - Statement vs expression handling
      - Ambiguity resolution
    deliverables:
      - Variable declaration parsing supporting let, var, and const with optional initializer expressions
      - Function definition parsing including parameter list, return type annotation, and body block
      - Control flow statement parsing for if/else, while, and for constructs with correct body nesting
      - Block statement parsing that groups multiple statements within braces
    estimated_hours: 5

  - id: ast-builder-m4
    name: Error Recovery
    description: >-
      Implement error handling and recovery for fault-tolerant parsing.
    acceptance_criteria:
      - Parser reports multiple syntax errors in a single pass instead of stopping at the first error
      - After an error, the parser synchronizes to the next statement boundary and continues parsing
      - Error messages include the file name, line number, column, and a description of what was expected
      - Source locations in error messages accurately point to the token where the error was detected
      - Synchronization points are statement boundaries (semicolon, closing brace, keywords like 'if', 'while')
    pitfalls:
      - Stopping at first error: users want to see all errors at once, not fix one at a time
      - Bad sync points: synchronizing on every token causes cascading errors—sync on statement boundaries
      - Cascading errors: one error can trigger many false positives if sync is poor—track and suppress
    concepts:
      - Error recovery allows parsing to continue after detecting an error
      - Panic mode: discard tokens until a known synchronization point
      - Synchronization points are typically statement or declaration boundaries
    skills:
      - Error message design
      - Recovery strategy implementation
      - Fault tolerance
      - Diagnostic reporting
    deliverables:
      - Synchronization points that advance the token stream to a known recovery position after an error
      - Multiple error collection that continues parsing after the first error to report additional issues
      - Meaningful error messages that include the expected token, actual token, and source location
      - Panic mode recovery that discards tokens until a statement boundary is found to resume parsing
    estimated_hours: 2
```

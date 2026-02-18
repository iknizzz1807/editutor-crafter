# AUDIT & FIX: type-checker

## CRITIQUE
- **Logical Gap (Confirmed - Occurs Check):** Milestone 3 mentions Unification but omits the 'Occurs Check'. Without the Occurs Check, the unification algorithm can enter infinite recursion when trying to unify a type variable with a type containing that same variable (e.g., t1 = list<t1>). This produces infinite types and crashes the type checker.
- **Hour Ranges:** Using ranges like '3-4' is imprecise. Converting to single estimates.
- **Estimated Hours:** 20-35 range is reasonable; estimate ~28 hours.

## FIXED YAML
```yaml
id: type-checker
name: Type Checker
description: >-
  Hindley-Milner style type inference with constraint generation, unification
  with occurs check, and let-polymorphism for a functional programming
  language.
difficulty: advanced
estimated_hours: 28
essence: >-
  Constraint generation and unification over type equations to automatically
  infer the most general types while detecting conflicts through substitution
  propagation and the occurs check algorithm.
why_important: >-
  Type checkers are fundamental to modern language implementation, teaching
  you how compilers reason about program correctness and enabling you to build
  safer programming languages with advanced features like generics and type
  inference.
learning_outcomes:
  - Design type representations for primitives, functions, and type variables
  - Implement constraint generation from expressions
  - Build the unification algorithm with occurs check
  - Implement let-polymorphism with generalization and instantiation
  - Produce clear type error messages with source locations
skills:
  - Type Inference
  - Unification Algorithm
  - Constraint Solving
  - Polymorphism
  - Error Reporting
tags:
  - type-system
  - type-inference
  - hindley-milner
  - advanced
  - compiler
architecture_doc: architecture-docs/type-checker/index.md
languages:
  recommended:
    - Haskell
    - OCaml
    - Rust
  also_possible:
    - Python
    - TypeScript
resources:
  - name: Types and Programming Languages (TAPL)
    url: https://www.cis.upenn.edu/~bcpierce/tapl/
    type: book
  - name: Algorithm W
    url: "https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system"
    type: article
  - name: Write You a Haskell
    url: http://dev.stephendiehl.com/fun/
    type: tutorial
prerequisites:
  - type: skill
    name: Functional programming concepts
  - type: skill
    name: AST representation and traversal
  - type: skill
    name: Basic type theory (what types are)
milestones:
  - id: type-checker-m1
    name: Type Representation
    description: >-
      Define type representations, type variables, and type environment.
    acceptance_criteria:
      - Primitive types int, bool, and string are represented as distinct type nodes
      - Function types represent parameter types and return type as a single arrow type (T1 -> T2)
      - Type variables for generics are represented as named or numbered placeholder type nodes
      - Type environment maps variable names to their declared or inferred types
      - Type equality checks handle alpha-equivalence for type variables
    pitfalls:
      - Forgetting unit/void type: some expressions have no meaningful return type
      - Mutable type variables: type variables should be immutable; use fresh names instead of mutation
      - Scope handling: type environments must handle nested scopes correctly
    concepts:
      - Type representations model the types in the language
      - Type environments map names to types, similar to runtime environments
      - Type variables are placeholders for unknown types
    skills:
      - Algebraic data types for type representation
      - Symbol table implementation
      - Environment management and scoping rules
    deliverables:
      - Type AST supporting primitives, functions, and type variables
      - Type environment mapping symbols to types with scope chain support
      - Type equality and compatibility checks for structural comparison
      - Fresh type variable generation for inference
    estimated_hours: 4

  - id: type-checker-m2
    name: Basic Type Checking
    description: >-
      Check types for expressions and statements with explicit annotations.
    acceptance_criteria:
      - Literals have known types inferred from their syntactic form (numbers -> int, strings -> string)
      - Binary operators check that both operand types are compatible with the operator
      - Function calls verify that argument types match the declared parameter types
      - Assignments check that the right-hand side type is compatible with the variable type
      - Type errors are reported with source location and a clear description of the mismatch
    pitfalls:
      - Operator type rules: each operator has specific type requirements (e.g., + works on int and string differently)
      - Function arity: argument count must match parameter count
      - Return type tracking: functions must return their declared type in all branches
    concepts:
      - Type checking verifies that expressions have consistent types
      - Type rules define the valid type combinations for each construct
      - Error reporting guides the programmer to fix type mismatches
    skills:
      - Type rule implementation for expressions
      - Type checking visitor pattern
      - Semantic error reporting and recovery
    deliverables:
      - Type annotation parser extracting explicit type declarations from source AST
      - Expression type inference deducing types from literals, operators, and context
      - Type compatibility checker verifying assignment and argument type correctness
      - Error reporting module producing clear messages with source location for type mismatches
    estimated_hours: 7

  - id: type-checker-m3
    name: Type Inference with Unification
    description: >-
      Infer types using constraint generation and unification with occurs check.
    acceptance_criteria:
      - Infer variable types from initializer expressions without explicit annotations
      - Infer function return types from the body expression or return statements
      - Unification algorithm finds most general unifier for a set of type constraints
      - Occurs check prevents infinite types (e.g., t1 = list<t1> fails with occurs check error)
      - Type constraint conflicts produce clear error messages indicating the conflicting types
    pitfalls:
      - Infinite types: without occurs check, unification can loop forever or produce nonsense types
      - Substitution application: substitutions must be applied correctly and composed in the right order
      - Occurs check: must be performed BEFORE applying a substitution that would create a cycle
    concepts:
      - Type inference discovers types without explicit annotations
      - Constraints are type equalities that must be satisfied
      - Unification finds substitutions that satisfy all constraints
      - Occurs check prevents recursive type definitions through type variables
    skills:
      - Constraint generation and solving
      - Unification algorithm implementation
      - Type substitution and composition
      - Occurs check implementation
    deliverables:
      - Constraint generation pass collecting type equality constraints from expressions
      - Unification algorithm solving type constraints by finding consistent type substitutions
      - Occurs check detecting and rejecting infinite type constructions
      - Type variable substitution applying solved constraints to replace type variables with concrete types
    estimated_hours: 10

  - id: type-checker-m4
    name: Let-Polymorphism
    description: >-
      Add support for polymorphic types with generalization and instantiation.
    acceptance_criteria:
      - Let polymorphism allows a single definition to be used at multiple types
      - Generalize types at let bindings by quantifying over free type variables
      - Instantiate polymorphic types at use sites with fresh type variables
      - Type schemes with forall quantification represent polymorphic types correctly
      - The identity function (fun x -> x) gets polymorphic type forall a. a -> a
    pitfalls:
      - Value restriction: in languages with mutation, not all values can be polymorphic
      - Generalization timing: generalize only at let bindings, not at lambda boundaries
      - Monomorphization vs polymorphism: decide if polymorphism is resolved at compile time or runtime
    concepts:
      - Polymorphism allows code to work with multiple types
      - Generalization quantifies over type variables at binding sites
      - Instantiation replaces quantified variables with fresh ones at use sites
      - Type schemes capture polymorphic types with explicit quantification
    skills:
      - Generic type parameter handling
      - Type scheme instantiation and generalization
      - Let-polymorphism implementation
      - Free variable computation
    deliverables:
      - Generic type parameters declared on functions and data type definitions
      - Type variable instantiation replacing generic parameters with fresh type variables at use sites
      - Constraint solving resolving type variable bindings during polymorphic function application
      - Type generalization at let bindings quantifying over unconstrained type variables
      - Type scheme representation with explicit forall quantification
    estimated_hours: 7
```

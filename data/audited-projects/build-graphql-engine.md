# AUDIT & FIX: build-graphql-engine

## CRITIQUE
- **N+1 strategy contradiction**: The essence and M5 both claim the engine will use SQL JOINs for nested fields AND DataLoader-style batching with IN clauses. These are fundamentally different strategies: JOINs resolve relationships in a single query at the SQL level, while DataLoader batches resolver calls at the application level. Real engines choose one primary strategy (Hasura uses JOINs; Apollo uses DataLoader). Implementing both in one project without clarifying when each is used creates confusion. The project should commit to one primary strategy (JOINs for the SQL compiler path) and mention DataLoader as an alternative/fallback.
- **Missing validation milestone**: There is no milestone for query validation between parsing (M1) and execution (M3). The GraphQL spec requires validation of: field existence on types, argument type correctness, fragment type conditions, variable usage, query depth limits, and fragment cycle detection. Without validation, invalid queries hit the executor and produce confusing errors or security vulnerabilities (e.g., deeply nested queries for DoS).
- **M1 and M2 ordering concern**: M2 (Schema & Type System) logically should come before or alongside M1 (Parser), because validation (which should follow parsing) requires the schema. However, the parser itself doesn't need the schema, so this ordering is acceptable if validation is added as a separate milestone between M2 and M3.
- **Estimated hours are unrealistic**: 120 hours total with 24 hours per milestone × 5 milestones = 120. But M5 (SQL compilation with JOINs, filtering, pagination, aggregation, and DataLoader) is easily 40+ hours for an expert. The hours are suspiciously uniform.
- **'game-dev' tag is nonsensical**: A GraphQL engine has nothing to do with game development. This is a tagging error.
- **M3 execution doesn't mention fragment resolution**: The execution engine must inline fragment spreads into selection sets before resolution. This is a critical step between parsing and execution that's missing.
- **M4 database reflection lacks schema change handling**: What happens when the database schema changes after the GraphQL schema is generated? No AC for schema refresh or change detection.
- **No AC for mutations**: M3 mentions mutations in passing (M1 parsing) but there's no AC for actually executing mutations (INSERT, UPDATE, DELETE SQL generation) in M4 or M5.
- **SQL injection risk**: M5 mentions parameterization in concepts but has no AC requiring parameterized queries. This is a critical security requirement that must be an explicit AC.
- **No mention of authorization**: Real GraphQL engines need row-level or field-level authorization. While this could be out of scope, it should be acknowledged.

## FIXED YAML
```yaml
id: build-graphql-engine
name: Build Your Own GraphQL Engine
description: GraphQL query parsing, schema generation from database, and query-to-SQL compilation
difficulty: expert
estimated_hours: "130-170"
essence: >
  Parsing GraphQL queries into abstract syntax trees, validating them against a type system
  schema, and compiling them into optimized SQL queries through database metadata reflection
  and JOIN-based query planning to eliminate N+1 problems.
why_important: >
  Building a GraphQL engine requires mastering query language design, compiler construction,
  and database query optimization — skills critical for backend infrastructure, API design, and
  understanding how modern data layers like Hasura, PostGraphile, and Apollo Server work.
learning_outcomes:
  - Implement a complete GraphQL parser producing spec-compliant ASTs
  - Build a type system supporting all GraphQL type kinds with validation
  - Validate queries against schema for field existence, type correctness, and fragment validity
  - Execute queries with field resolution, null propagation, and error collection
  - Auto-generate GraphQL schemas from database metadata via reflection
  - Compile GraphQL queries to optimized SQL with JOINs instead of N+1 resolver calls
skills:
  - GraphQL specification
  - Query parsing and AST construction
  - Type system design and validation
  - Schema introspection
  - Database reflection and metadata querying
  - SQL query planning and JOIN optimization
  - Real-time subscriptions (stretch goal)
tags:
  - api
  - build-from-scratch
  - expert
  - framework
  - introspection
  - resolvers
  - schema
  - sql-compilation
architecture_doc: architecture-docs/build-graphql-engine/index.md
languages:
  recommended:
    - Python
    - Go
    - Rust
  also_possible:
    - TypeScript
resources:
  - name: GraphQL Official Specification
    url: https://spec.graphql.org/
    type: documentation
  - name: How to GraphQL Tutorial
    url: https://www.howtographql.com/
    type: tutorial
  - name: Building a GraphQL to SQL Compiler
    url: https://hasura.io/blog/building-a-graphql-to-sql-compiler-on-postgres-ms-sql-and-mysql
    type: article
  - name: GraphQL Introspection Guide
    url: https://graphql.org/learn/introspection/
    type: documentation
prerequisites:
  - type: skill
    name: GraphQL fundamentals (queries, mutations, schemas)
  - type: skill
    name: SQL and relational database concepts
  - type: skill
    name: Compiler basics (lexing, parsing, ASTs)
milestones:
  - id: build-graphql-engine-m1
    name: GraphQL Parser
    description: Parse GraphQL query documents into a spec-compliant AST.
    acceptance_criteria:
      - Parser converts valid GraphQL query strings into a complete AST with operation definitions, selection sets, fields, arguments, and directives
      - Query, mutation, and subscription operations are recognized as distinct AST operation types
      - Named fragments (fragment ... on Type) and inline fragments (... on Type) are parsed with their type conditions and selection sets correctly attached
      - Variable definitions are parsed with names, types (including NonNull and List wrappers), and optional default values
      - Directive parsing handles @-prefixed directives with argument lists on fields, fragments, and operations
      - String literals handle block strings (triple-quoted), escape sequences, and Unicode escapes per the GraphQL spec
      - Parse errors include line number, column number, and a description of expected vs. actual token
      - Parser handles all spec examples from the GraphQL specification without error
    pitfalls:
      - Not handling block string (triple-quote) indentation stripping per the spec
      - Failing to validate balanced braces and parentheses before attempting full parse leads to confusing error messages
      - Confusing field arguments with directive arguments during AST construction
      - Not preserving source location metadata on AST nodes makes later validation error messages useless
      - Fragment spread vs inline fragment disambiguation requires lookahead (spread has no type condition keyword)
    concepts:
      - Tokenization of GraphQL syntax
      - Recursive descent parsing for nested selection sets
      - AST node design and source location tracking
      - GraphQL operation, fragment, and directive grammar rules
    skills:
      - Lexer and tokenizer implementation
      - Recursive descent parser construction
      - AST design with location metadata
      - GraphQL specification compliance
    deliverables:
      - GraphQL lexer producing tokens (names, strings, numbers, punctuation) with position info
      - Recursive descent parser building AST from token stream
      - AST node types for operations, selection sets, fields, arguments, directives, fragments, variables
      - Error reporting with line, column, and expected-vs-actual description
      - Test suite covering all GraphQL specification grammar examples
    estimated_hours: "22-28"

  - id: build-graphql-engine-m2
    name: Schema & Type System
    description: Build the type system representation and schema validation.
    acceptance_criteria:
      - All GraphQL type kinds are supported: Scalar, Object, Interface, Union, Enum, InputObject, List, NonNull
      - Built-in scalar types (Int, Float, String, Boolean, ID) are predefined with proper serialize/parseValue/parseLiteral
      - Custom scalar types can be registered with user-defined serialize, parseValue, and parseLiteral functions
      - Object types declare fields with return types, argument definitions, and interface implementations
      - Schema validation detects: undefined type references, duplicate type names, missing required fields on interface implementations, and invalid input object recursion (circular non-nullable input fields)
      - Interface types validate that all implementing object types provide every field declared by the interface with compatible types
      - Introspection queries (__schema, __type, __typename) are supported per the GraphQL specification, returning the full schema structure
      - SDL (Schema Definition Language) parser constructs the schema from type definition strings as an alternative to programmatic construction
    pitfalls:
      - Circular type references between objects (User has posts, Post has author) are valid and must use lazy resolution; circular INPUT types with non-nullable fields are invalid
      - Interface implementation validation must check argument compatibility, not just field name matching
      - Forgetting NonNull and List wrapper types in introspection output breaks client tooling like GraphiQL
      - Allowing invalid combinations like NonNull(NonNull(T)) must be rejected
    concepts:
      - GraphQL type system and type kinds
      - Schema definition and validation
      - Introspection meta-fields
      - Input vs output type distinction
    skills:
      - Type system design and constraint validation
      - Schema definition language parsing
      - Introspection implementation
      - Recursive type validation
    deliverables:
      - Type definitions for all 8 type kinds
      - Built-in and custom scalar type support
      - Schema validation with comprehensive error reporting
      - Introspection query implementation (__schema, __type, __typename)
      - SDL parser for schema-from-string construction
    estimated_hours: "22-28"

  - id: build-graphql-engine-m3
    name: Query Validation
    description: >
      Validate parsed query ASTs against the schema before execution, implementing
      the validation rules from the GraphQL specification.
    acceptance_criteria:
      - Field validation: every field in a selection set exists on the corresponding object/interface type; unknown fields produce a validation error with field name and parent type
      - Argument validation: required arguments are present, argument types match schema definitions, and unknown arguments are rejected
      - Fragment validation: fragment type conditions reference existing types; fragment spreads reference defined fragments; fragment cycles (A spreads B, B spreads A) are detected and rejected
      - Variable validation: all used variables are defined in the operation header; all defined variables are used; variable types are compatible with argument types where they're used
      - Query depth limiting: configurable maximum query depth (default 10) rejects deeply nested queries to prevent DoS
      - Query complexity analysis: configurable maximum field count or cost estimate rejects overly expensive queries
      - Validation errors include the field/fragment/variable name, the source location, and a human-readable description
      - Multiple validation errors are collected and returned together, not just the first one
    pitfalls:
      - Fragment cycle detection requires building a directed graph of fragment spreads and running cycle detection; naive recursive expansion hits infinite loops
      - Variable type compatibility must account for NonNull and List wrappers (e.g., String variable is compatible with String argument but not String! argument)
      - Query depth limiting must handle fragments: a fragment spread can add arbitrary depth
      - Skipping validation entirely is a security risk: allows malicious queries to DoS the executor
    concepts:
      - GraphQL validation rules (spec section 5)
      - Fragment cycle detection via graph analysis
      - Type compatibility checking with wrappers
      - Query cost analysis and depth limiting
    skills:
      - Graph-based cycle detection
      - Type compatibility algorithms
      - Multi-pass AST visitor pattern
      - Security-oriented input validation
    deliverables:
      - Field existence and type validation visitor
      - Argument presence, type, and unknown-argument validation
      - Fragment validation with cycle detection
      - Variable definition and usage validation
      - Configurable query depth and complexity limits
      - Validation error collection with source locations
    estimated_hours: "18-24"

  - id: build-graphql-engine-m4
    name: Query Execution
    description: Execute validated queries against the schema with resolver dispatch.
    acceptance_criteria:
      - Queries execute against the schema and return a data object matching the requested selection set shape
      - Each field is resolved by calling its resolver function with (parent, args, context, info); default resolver reads parent[fieldName]
      - Fragment spreads and inline fragments are inlined into the selection set before field resolution, applying type conditions correctly
      - Non-null fields that resolve to null propagate the null to the nearest nullable parent field per the GraphQL specification's null-bubbling rules
      - Field-level errors are collected in an errors array alongside partial data; the entire query does not abort on a single field error
      - Mutations execute root fields serially (not in parallel) per the GraphQL specification
      - Execution context object carries authentication, request-scoped state, and DataLoader instances to resolvers
      - Sibling fields on the same selection set level are resolved concurrently when resolvers are async
    pitfalls:
      - Null propagation must bubble up through nested non-null types: if a deeply nested non-null field is null, the null may propagate several levels up
      - Forgetting to resolve fragments by checking type conditions leads to returning fields from the wrong type in union/interface queries
      - Mutation serial execution is required by spec; executing mutation root fields in parallel violates specification guarantees
      - Not providing a default field resolver (reading property from parent) forces users to write trivial resolvers for every field
      - Errors in one branch of the response must not corrupt other branches
    concepts:
      - Field resolver dispatch and default resolvers
      - Fragment inlining and type condition checking
      - Null propagation and bubbling rules
      - Serial vs parallel execution semantics
      - Error collection with partial results
    skills:
      - Resolver execution engine implementation
      - Null propagation algorithms
      - Async parallel and serial execution
      - Error boundary management
    deliverables:
      - Field resolver dispatch with (parent, args, context, info) signature
      - Fragment inlining with type condition evaluation
      - Null propagation engine per GraphQL spec
      - Error collection returning partial data + errors array
      - Serial mutation execution and parallel query execution
      - Execution context with DataLoader support
    estimated_hours: "22-28"

  - id: build-graphql-engine-m5
    name: Database Schema Reflection
    description: Auto-generate a GraphQL schema from database table metadata.
    acceptance_criteria:
      - Database tables and columns are introspected via information_schema (PostgreSQL) or equivalent, reading names, types, nullability, and primary keys
      - Each table generates a GraphQL object type with one field per column; column types map to GraphQL scalars (integer->Int, varchar->String, boolean->Boolean, timestamp->String, numeric->Float)
      - Nullable columns produce nullable GraphQL fields; NOT NULL columns produce NonNull fields
      - Foreign key relationships produce nested object fields: a posts.author_id FK to users.id generates an author: User field on Post
      - One-to-many reverse relationships are detected: users has a posts: [Post] field based on the FK from posts
      - Root Query type includes: single-record lookup by primary key (e.g., user(id: ID!): User) and collection queries (e.g., users: [User]) for each table
      - Root Mutation type includes basic CRUD: insert, update (by PK), and delete (by PK) for each table
      - Schema regeneration can be triggered to pick up database schema changes
    pitfalls:
      - Database-specific type mappings vary widely: PostgreSQL arrays, JSON columns, enums, and composite types need special handling or must be explicitly excluded
      - Self-referential foreign keys (e.g., employee.manager_id -> employee.id) must produce valid recursive GraphQL types without stack overflow during schema generation
      - Many-to-many relationships via join tables need special detection: a table with exactly two FKs and no other meaningful columns is likely a join table
      - Not respecting column nullability leads to incorrect NonNull annotations and runtime null-propagation errors
      - Generated type names may conflict with GraphQL reserved names or contain characters invalid in GraphQL identifiers
    concepts:
      - Database metadata introspection (information_schema)
      - SQL-to-GraphQL type mapping
      - Foreign key relationship detection
      - Automatic CRUD generation
    skills:
      - Database introspection queries
      - Automated schema generation
      - Relationship detection and modeling
      - Type mapping strategies
    deliverables:
      - Table-to-type mapper generating GraphQL object types from database tables
      - Column-to-field mapper with SQL-to-GraphQL scalar type conversion
      - FK relationship detector generating nested object and collection fields
      - Root query generator with by-PK lookup and collection queries
      - Root mutation generator with insert, update, delete operations
      - Schema refresh mechanism for database changes
    estimated_hours: "22-28"

  - id: build-graphql-engine-m6
    name: Query-to-SQL Compilation
    description: >
      Compile GraphQL queries into efficient SQL queries using JOINs for nested selections,
      eliminating N+1 queries at the SQL compilation level rather than the resolver level.
    acceptance_criteria:
      - GraphQL field selections compile to SQL SELECT clauses containing only the requested columns, not SELECT *
      - Nested object fields (via foreign keys) compile to SQL JOINs rather than separate queries; a query requesting user { posts { title } } generates a single SQL statement with a JOIN
      - One-to-many relationships use lateral joins or subqueries (depending on database) to avoid cartesian product duplication of parent rows
      - Filter arguments (where: {name: {eq: "Alice"}}) compile to parameterized SQL WHERE clauses; all user input is parameterized (never string-interpolated) to prevent SQL injection
      - Pagination arguments (first, offset, or cursor-based after/before) compile to SQL LIMIT/OFFSET or keyset pagination clauses
      - Ordering arguments (orderBy: {field: createdAt, direction: DESC}) compile to SQL ORDER BY clauses
      - Aggregate fields (count, sum, avg, min, max) compile to SQL aggregate functions with appropriate GROUP BY
      - Generated SQL is logged in debug mode for inspection and optimization
      - Benchmark: a 3-level nested query generates at most 1 SQL statement (not N+1)
    pitfalls:
      - JOINs on one-to-many relationships produce cartesian products that duplicate parent rows; must use lateral joins, subqueries, or JSON aggregation to avoid this
      - Not parameterizing filter values creates SQL injection vulnerabilities; this is a critical security requirement
      - Deeply nested queries (5+ levels) can generate extremely complex JOINs; must set a maximum JOIN depth and fall back to separate queries beyond it
      - Database-specific SQL dialect differences (PostgreSQL LATERAL JOIN vs MySQL correlated subquery) must be abstracted behind a dialect layer
      - Forgetting LIMIT on collection queries allows unbounded result sets that exhaust memory
      - Aggregate queries with filters need correct WHERE clause placement (before vs after GROUP BY)
    concepts:
      - Query planning and SQL generation
      - JOIN compilation from nested selections
      - Lateral joins for one-to-many without duplication
      - Parameterized query construction for security
      - Keyset vs offset pagination in SQL
    skills:
      - SQL query generation and optimization
      - JOIN strategy selection (inner, left, lateral)
      - Query parameterization for injection prevention
      - Database dialect abstraction
    deliverables:
      - Selection-to-SELECT column mapper
      - Nested field-to-JOIN compiler (with lateral join for one-to-many)
      - Filter-to-parameterized-WHERE translator
      - Pagination and ordering SQL clause generator
      - Aggregate query compiler with GROUP BY support
      - SQL dialect abstraction layer (at least PostgreSQL)
      - Query logging for debugging generated SQL
    estimated_hours: "24-34"
```
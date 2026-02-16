# AUDIT & FIX: capstone-database-engine

## CRITIQUE
- **Logical Gap (Confirmed - Lock Manager):** MVCC provides read isolation via snapshots, but write-write conflicts require explicit locking. The original has no mention of a lock manager. Two concurrent transactions updating the same row need either pessimistic locking (lock manager) or optimistic validation (first-committer-wins). Additionally, DDL operations (ALTER TABLE, DROP TABLE) require schema-level locks to prevent concurrent modifications.
- **Technical Inaccuracy (Confirmed - Statistics Collection):** The cost-based optimizer in M3 'estimates cardinality' but has no prerequisite for ANALYZE or any statistics collection mechanism. Without statistics, cardinality estimation is pure guesswork. Statistics collection should be part of M3 or a separate milestone.
- **M2 B+ tree:** 'B+ tree index supporting insert, delete, point lookup, and range scan'—needs to distinguish between table B-tree (clustered, data in all nodes as in SQLite) and index B+tree (non-clustered, data only in leaves). The description says 'B+ tree' but tables typically use B-tree (clustered).
- **M3 Hash Join:** 'Memory-aware spilling to disk' is mentioned as a deliverable for hash join, but this is an extremely complex feature (external hash join). For a capstone that already has 5 milestones and 100+ hours, this might be over-scoped.
- **M4 MVCC + WAL:** Combines two massive topics (MVCC and WAL) into one milestone. Each is a full project. The ARIES recovery protocol alone (analysis, redo, undo phases) is substantial. This milestone is over-scoped at 22 hours.
- **M5 Wire Protocol:** Implementing the PostgreSQL wire protocol is a significant undertaking. The AC mentions 'Connection pooling support' which is a server-side feature requiring thread/goroutine management, not just protocol implementation.
- **Missing: AGGREGATE functions and GROUP BY.** The parser AC mentions GROUP BY but no execution milestone implements aggregation or grouping.
- **Overall Scope:** 100-140 hours is realistic but the milestone distribution is uneven. M1 and M5 are relatively light; M3 and M4 are overloaded.

## FIXED YAML
```yaml
id: capstone-database-engine
name: "Capstone: Complete Database Engine"
description: >-
  Build a complete relational database engine from scratch: SQL parser, bytecode
  compiler, cost-based query optimizer with statistics, B-tree/B+tree storage
  with buffer pool, MVCC transactions, write-ahead logging, lock management,
  and PostgreSQL wire protocol compatibility.
difficulty: expert
estimated_hours: 130
essence: >-
  SQL parsing and compilation into a logical plan, optimized via cost-based
  query planning with collected statistics, executed by a volcano-style iterator
  engine with hash join and sort-merge join, operating over a page-based B-tree
  storage layer with buffer pool management, MVCC snapshot isolation with a lock
  manager for write-write conflicts, ARIES-style write-ahead logging for crash
  recovery, and PostgreSQL wire protocol for client connectivity.
why_important: >-
  Databases are the most critical infrastructure in any application. Building one
  from scratch gives you unmatched understanding of indexing, transactions, query
  optimization, and storage—knowledge that directly improves your ability to
  design schemas, write efficient queries, debug production issues, and architect
  data-intensive systems.
learning_outcomes:
  - Implement a complete SQL pipeline from lexing through parsing to logical plan generation
  - Build a page-based storage engine with buffer pool, B-tree tables, and B+tree indexes
  - Design a volcano-style query executor with hash join and sort-merge join
  - Implement a cost-based query optimizer with statistics collection (ANALYZE)
  - Build MVCC transactions with snapshot isolation and a lock manager for write conflicts
  - Implement ARIES-style write-ahead logging for crash recovery
  - Design a PostgreSQL-compatible wire protocol for client connectivity
skills:
  - SQL Parsing
  - Query Optimization
  - B-tree/B+tree Indexing
  - Buffer Pool Management
  - MVCC
  - Lock Management
  - Write-Ahead Logging
  - Crash Recovery
  - Wire Protocol
tags:
  - databases
  - storage-engines
  - capstone
  - systems-programming
  - expert
architecture_doc: architecture-docs/capstone-database-engine/index.md
languages:
  recommended:
    - Rust
    - C
    - Go
  also_possible:
    - Java
    - C++
resources:
  - name: Architecture of a Database System
    url: https://dsf.berkeley.edu/papers/fntdb07-architecture.pdf
    type: paper
  - name: CMU 15-445 Database Systems Course
    url: https://15445.courses.cs.cmu.edu/
    type: course
  - name: SQLite Architecture
    url: https://www.sqlite.org/arch.html
    type: documentation
  - name: PostgreSQL Wire Protocol Documentation
    url: https://www.postgresql.org/docs/current/protocol.html
    type: documentation
  - name: ARIES Recovery Algorithm Paper
    url: https://cs.stanford.edu/people/chr101/aries.pdf
    type: paper
prerequisites:
  - type: project
    id: build-sqlite
    name: Build Your Own SQLite (or equivalent B-tree + SQL experience)
  - type: skill
    name: B-tree/B+tree data structures
  - type: skill
    name: SQL fundamentals
  - type: skill
    name: File I/O and binary formats
  - type: skill
    name: Concurrency and locking
milestones:
  - id: capstone-database-engine-m1
    name: SQL Frontend & System Catalog
    description: >-
      Implement SQL lexer, parser (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE,
      CREATE INDEX), logical plan generator, and system catalog for schema storage.
    acceptance_criteria:
      - Lexer tokenizes SQL keywords, identifiers, string/numeric literals, and operators case-insensitively
      - Parser handles SELECT with column list, FROM, WHERE, JOIN (INNER, LEFT), GROUP BY, HAVING, ORDER BY, LIMIT
      - Parser handles INSERT with VALUES clause and optional column list
      - Parser handles UPDATE with SET clause and WHERE, DELETE with WHERE
      - Parser handles CREATE TABLE with column definitions (types: INTEGER, TEXT, REAL, BOOLEAN), NOT NULL, PRIMARY KEY, UNIQUE constraints
      - Parser handles CREATE INDEX and DROP TABLE
      - Logical plan generator transforms parsed AST into a tree of logical operators (Scan, Filter, Project, Join, Sort, Limit, Aggregate)
      - System catalog persists table schemas, column types, constraints, and index definitions; survives database restart
      - Ambiguous column references in JOINs are detected and reported as errors
    pitfalls:
      - LEFT JOIN requires preserving unmatched rows from the left table with NULLs for right columns—different from INNER JOIN logic
      - NULL handling in WHERE: 'column = NULL' is always false; must use 'IS NULL'
      - System catalog concurrency: DDL operations must not corrupt the catalog under concurrent access
      - Logical plan must normalize equivalent representations (e.g., 'WHERE a=1 AND b=2' and 'WHERE b=2 AND a=1' produce the same plan)
    concepts:
      - Lexical analysis and recursive-descent parsing for SQL grammar
      - Logical query plan represents relational algebra operations as a tree
      - System catalog is the metadata repository for all database objects
      - Semantic validation checks type compatibility and name resolution
    skills:
      - SQL parsing
      - Logical plan generation
      - Catalog design
      - Type validation
    deliverables:
      - SQL lexer and recursive-descent parser for DDL and DML
      - Logical plan generator producing operator tree from AST
      - System catalog persisting table schemas, column types, constraints, and index metadata
      - Semantic validator checking column existence, type compatibility, and ambiguous references
    estimated_hours: 18

  - id: capstone-database-engine-m2
    name: Storage Engine & Buffer Pool
    description: >-
      Build a page-based storage engine with buffer pool manager, B-tree
      for table storage (clustered), and B+tree for secondary indexes.
    acceptance_criteria:
      - Fixed-size pages (4KB) use slotted page format for variable-length records with cell pointer array
      - Buffer pool manages configurable number of page frames with LRU eviction and dirty page tracking
      - Pin/Unpin API prevents eviction of actively-used pages; pin leak detection logs warnings
      - Table B-tree stores rows clustered by primary key (rowid) with data in all nodes
      - Index B+tree stores (indexed column value → primary key) with data only in leaf nodes and linked leaf pages for range scans
      - B-tree/B+tree supports insert, delete, point lookup, and range scan with correct node splitting and merging
      - Sequential table scan iterates all rows via leaf page traversal; index scan uses B+tree for filtered lookups
      - Storage engine handles at least 100K row inserts into a single table (verified by test)
    pitfalls:
      - Page splits during B-tree insert can cascade up the tree; handle multi-level splits correctly
      - Buffer pool deadlocks occur when operations pin too many pages simultaneously—limit concurrent pin count
      - Overflow pages for records larger than a page require linked page chains—handle or reject oversized records
      - Forgetting to flush dirty pages on shutdown loses committed data
      - Linked leaf pages in B+tree must be maintained during splits for correct range scans
    concepts:
      - Slotted page format uses cell pointers for variable-length records within fixed-size pages
      - Buffer pool caches pages in memory with LRU eviction for frequently-accessed pages
      - Table B-tree is clustered (data in all nodes, keyed by primary key)
      - Index B+tree is non-clustered (data only in leaves, linked for range scans)
      - Node splitting maintains tree balance on insert overflow
    skills:
      - Page format design
      - Buffer pool implementation
      - B-tree/B+tree algorithms
      - Index scan and table scan execution
    deliverables:
      - Slotted page format with cell pointers and free space management
      - LRU buffer pool manager with dirty page tracking and pin/unpin API
      - Table B-tree with insert, delete, point lookup, and sequential scan
      - Index B+tree with insert, delete, point lookup, range scan, and linked leaf pages
      - 100K row insertion test verifying correctness and performance
    estimated_hours: 24

  - id: capstone-database-engine-m3
    name: Query Execution & Optimization
    description: >-
      Implement a volcano-style query executor with hash join and sort-merge
      join, aggregate functions, and a cost-based optimizer with statistics.
    acceptance_criteria:
      - Volcano iterator model: each operator implements open()/next()/close() with demand-driven pull semantics
      - Sequential scan operator iterates all rows from a table via B-tree cursor
      - Filter operator evaluates WHERE predicates and passes matching rows
      - Project operator selects and computes output columns
      - Hash join builds a hash table on the smaller input and probes with the larger input; handles memory limits by partitioning to disk if needed
      - Sort-merge join sorts both inputs on join key and merges matching rows
      - Aggregate operator implements COUNT, SUM, AVG, MIN, MAX with GROUP BY support using hash-based grouping
      - ANALYZE command collects per-table row count and per-column distinct value count, persisted in catalog
      - Cost-based optimizer estimates I/O cost for table scan vs index scan; selects index scan when estimated selectivity < 20%
      - For multi-table JOINs, optimizer selects join order to minimize estimated intermediate result size using dynamic programming for ≤6 tables
      - EXPLAIN displays the chosen physical plan with operator types, index names, and estimated row counts
    pitfalls:
      - Cardinality estimation errors cascade through multi-table joins—a 10x error in one table becomes 100x after a join
      - Hash join memory overflow without spilling causes OOM on large tables—implement grace hash join or restrict to in-memory hash join with documented limits
      - Stale statistics (before ANALYZE is run) cause the optimizer to choose poor plans—use conservative defaults when no statistics exist
      - AVG must return a floating-point result even for integer columns; COUNT(*) counts NULLs but COUNT(col) does not
      - Optimizer search space explodes exponentially with table count; limit to dynamic programming and avoid exhaustive search
    concepts:
      - Volcano model uses demand-driven (pull) execution where each operator requests rows from its child
      - Hash join is optimal for equality joins on large unsorted inputs
      - Sort-merge join is optimal when inputs are already sorted on the join key
      - Cardinality estimation uses column statistics (distinct count, row count) to predict operator output size
      - Cost model estimates I/O pages for each candidate plan
    skills:
      - Iterator-based execution
      - Join algorithms
      - Aggregate computation
      - Statistics collection
      - Cost-based optimization
    deliverables:
      - Volcano iterator operators: SeqScan, IndexScan, Filter, Project, HashJoin, SortMergeJoin, Aggregate, Sort, Limit
      - Hash join with in-memory hash table and documented memory limit
      - Sort-merge join with external sort for large inputs
      - Aggregate operator with COUNT, SUM, AVG, MIN, MAX and GROUP BY
      - ANALYZE command collecting and persisting table/column statistics
      - Cost-based optimizer choosing scan type and join order
      - EXPLAIN command displaying physical plan
    estimated_hours: 26

  - id: capstone-database-engine-m4
    name: MVCC Transactions & Lock Manager
    description: >-
      Implement MVCC with snapshot isolation for read consistency, a lock
      manager for write-write conflict detection, and transaction lifecycle
      management.
    acceptance_criteria:
      - BEGIN starts a transaction; COMMIT makes changes permanent; ROLLBACK undoes all changes
      - Each transaction receives a unique, monotonically increasing transaction ID (txn_id)
      - MVCC maintains multiple row versions; each version is tagged with the creating transaction ID
      - Snapshot isolation: a transaction sees only rows committed before its start timestamp; uncommitted and later-committed rows are invisible
      - Lock manager provides row-level exclusive locks for writes; two transactions attempting to UPDATE the same row result in one waiting or aborting (deadlock detection via wait-for graph or timeout)
      - Write skew anomaly is documented as a known limitation of snapshot isolation (vs serializable)
      - DDL operations (CREATE TABLE, DROP TABLE, CREATE INDEX) acquire schema-level locks preventing concurrent DDL and DML on the same table
      - Garbage collection reclaims old row versions that are no longer visible to any active transaction
    pitfalls:
      - Write skew (two transactions reading overlapping data and writing non-overlapping rows) is possible under snapshot isolation—document this explicitly
      - Lock ordering inconsistency causes deadlocks; implement deadlock detection (wait-for graph cycle detection) or lock timeout
      - Long-running transactions prevent garbage collection of old row versions, causing version bloat
      - DDL without schema locks can corrupt the catalog if two DDL statements run concurrently
      - Transaction ID wraparound must be handled for very long-running databases (not required for capstone but document the limitation)
    concepts:
      - MVCC stores multiple versions of each row, tagged with transaction IDs
      - Snapshot isolation provides each transaction a consistent view as of its start time
      - Row-level locking prevents write-write conflicts on the same row
      - Deadlock detection identifies cycles in the wait-for graph
      - Version garbage collection reclaims row versions invisible to all transactions
    skills:
      - MVCC implementation
      - Lock manager design
      - Deadlock detection
      - Transaction lifecycle management
      - Version garbage collection
    deliverables:
      - Transaction manager with BEGIN, COMMIT, ROLLBACK and transaction ID generation
      - MVCC row versioning with transaction ID-based visibility checking
      - Snapshot isolation providing consistent reads without blocking writes
      - Row-level lock manager with exclusive write locks
      - Deadlock detection via wait-for graph or configurable lock timeout
      - Schema-level locks for DDL operations
      - Version garbage collection for old row versions
    estimated_hours: 22

  - id: capstone-database-engine-m5
    name: Write-Ahead Logging & Crash Recovery
    description: >-
      Implement ARIES-style write-ahead logging for crash recovery with
      redo/undo phases and periodic checkpointing.
    acceptance_criteria:
      - Every data page modification is logged to the WAL BEFORE the dirty page is written to the database file (write-ahead property)
      - WAL records include: LSN (log sequence number), transaction ID, page ID, before-image (for undo), and after-image (for redo)
      - COMMIT record in WAL is fsynced before acknowledging commit to the client (durability guarantee)
      - Crash recovery implements ARIES three-phase protocol: analysis (identify active transactions and dirty pages), redo (replay all logged changes), undo (rollback uncommitted transactions)
      - Periodic checkpointing writes the set of dirty pages and active transactions to the WAL, bounding recovery time
      - After injecting a crash (kill -9) mid-transaction, recovery correctly restores the database to the last consistent state (committed transactions present, uncommitted transactions rolled back)
      - WAL is truncated after checkpoint to prevent unbounded growth
    pitfalls:
      - Writing dirty pages to the database before their WAL records are fsynced violates the write-ahead property and causes unrecoverable corruption
      - ARIES redo must be idempotent—applying the same redo record twice must produce the same result (use LSN comparison to skip already-applied records)
      - Long periods without checkpointing increase recovery time because redo must replay from the last checkpoint
      - WAL growing without truncation exhausts disk space; checkpoint frequency must be tuned
      - Undo phase must handle the case where the undo operation itself is logged (compensation log records) to ensure recovery is idempotent
    concepts:
      - Write-ahead property: log before data ensures crash recovery is always possible
      - ARIES analysis phase identifies dirty pages and in-flight transactions at crash time
      - ARIES redo phase replays ALL logged changes to bring pages to their crash-time state
      - ARIES undo phase rolls back uncommitted transactions using before-images
      - Compensation Log Records (CLR) make undo operations idempotent during repeated recovery
    skills:
      - WAL implementation
      - ARIES recovery protocol
      - Checkpoint design
      - Crash testing methodology
    deliverables:
      - WAL with LSN, transaction ID, page ID, before-image, and after-image per record
      - WAL fsync on COMMIT for durability
      - ARIES three-phase crash recovery (analysis, redo, undo)
      - Periodic checkpointing recording dirty pages and active transactions
      - WAL truncation after checkpoint
      - Crash recovery integration test verifying correct state after simulated crash
    estimated_hours: 22

  - id: capstone-database-engine-m6
    name: Wire Protocol & Client Interface
    description: >-
      Implement a PostgreSQL-compatible wire protocol enabling standard
      clients (psql, JDBC drivers) to connect and query your database.
    acceptance_criteria:
      - Server accepts TCP connections and implements PostgreSQL wire protocol v3 (startup message, simple query, termination)
      - psql can connect, execute SQL queries, and display results correctly
      - Simple Query protocol: client sends a query string, server parses, executes, and returns RowDescription + DataRow + CommandComplete messages
      - Column type OIDs in RowDescription map internal types to PostgreSQL-compatible type identifiers (INT4, TEXT, FLOAT8, BOOL)
      - Error responses use PostgreSQL error message format with SQLSTATE codes and human-readable messages
      - Multiple concurrent client connections are supported with independent transaction states
      - Connection lifecycle: startup handshake → authentication (trust or password) → query loop → termination
    pitfalls:
      - PostgreSQL wire protocol has subtle message framing (4-byte length prefix)—off-by-one errors cause connection drops
      - Type OID mapping must be exact for PostgreSQL client libraries to correctly deserialize column values
      - Connection state must be isolated: one client's transaction must not affect another client's session
      - Not cleaning up connection state (locks, open transactions) on unexpected disconnect causes resource leaks
      - SSL negotiation request must be handled (even if only to reject it) to prevent protocol confusion with clients that try SSL first
    concepts:
      - PostgreSQL wire protocol v3 uses message-based framing with type byte and length prefix
      - Simple Query flow: Query → RowDescription → DataRow* → CommandComplete → ReadyForQuery
      - Type OIDs identify column types for client-side deserialization
      - Connection lifecycle manages authentication, session state, and cleanup
    skills:
      - Wire protocol implementation
      - PostgreSQL compatibility
      - Connection management
      - Type mapping
    deliverables:
      - TCP server accepting PostgreSQL wire protocol v3 connections
      - Startup message handling with authentication (trust mode minimum)
      - Simple Query protocol implementation (parse → execute → format → respond)
      - RowDescription and DataRow message formatting with correct type OIDs
      - Error response messages with SQLSTATE codes
      - Concurrent client connection support with independent session state
      - Connection cleanup on disconnect releasing locks and rolling back open transactions
    estimated_hours: 18
```
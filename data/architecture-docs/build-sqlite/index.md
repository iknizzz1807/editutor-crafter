# Build Your Own SQLite: Design Document


## Overview

This document outlines the design of an embedded, file-based SQL database engine. The key architectural challenge is efficiently organizing, querying, and persisting structured data on disk while supporting a subset of SQL and maintaining data integrity, all within a single-file architecture that balances simplicity with performance.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## 1. Context and Problem Statement

> **Milestone(s):** Foundational (prerequisite for all milestones)

This project addresses one of the most fundamental challenges in software engineering: **how to persist, organize, and retrieve structured data efficiently and reliably**. Every application that outlives a single execution—from a simple to-do list app to a complex financial system—needs to solve this problem. At its core, building a database is about bridging two worlds: the high-level, logical world of application data (tables, rows, columns) and the low-level, physical reality of storing bytes on disk.

### Mental Model: The Digital Filing Cabinet

Imagine you're organizing a physical library. You could simply throw all books into a giant pile—finding a specific book would require sifting through the entire pile every time. A slightly better approach would be to arrange them alphabetically by title on one very long shelf—still inefficient as your collection grows. What you really want is a **card catalog system**: a structured index that lets you find any book by title, author, or subject in a few steps, then go directly to its specific shelf location.

A relational database is this digital equivalent—a sophisticated filing system with multiple interconnected catalogs (indexes) and strict organizational rules (schema). The database file is the entire library building: shelves (pages), books (rows), and card catalogs (indexes) all contained within. The SQL language is the standardized request form you fill out ("find all books about databases published after 2010"), and the query engine is the librarian who knows how to navigate the system efficiently.

This mental model helps clarify key database concepts:
- **Tables** are like bookshelves organized by category (fiction, non-fiction)
- **Rows** are individual books on those shelves
- **Indexes** are card catalogs that provide alternative lookup paths
- **Transactions** are like checking out multiple books at once—either all succeed (commit) or none do (rollback)
- **The B-tree** is the hierarchical organization system that makes finding any book take roughly the same small number of steps, whether you have 100 or 1,000,000 books

### The Core Challenge: Structured Data on Disk

Storing structured data on disk presents four fundamental tensions that every database design must balance:

1. **Structure vs. Flexibility**: How do we impose enough structure for efficient querying while allowing for schema evolution and variable-length data?

2. **Persistence vs. Performance**: How do we guarantee data survives crashes (durability) without sacrificing write speed? Every `fsync()` that ensures data hits disk is orders of magnitude slower than writing to memory.

3. **Order vs. Mutability**: How do we maintain sorted order for efficient range queries while supporting frequent inserts, updates, and deletes?

4. **Complexity vs. Simplicity**: How do we provide powerful query capabilities (JOINs, aggregations, transactions) while keeping the implementation understandable and maintainable?

These tensions manifest in concrete design decisions:

| Tension | Database Manifestation | Simple Approach | Sophisticated Approach |
|---------|----------------------|-----------------|------------------------|
| Structure vs. Flexibility | Column types and schema | Fixed-width records | Variable-length encoding with type tags |
| Persistence vs. Performance | Transaction durability | Sync after every write | Write-ahead logging with batch fsync |
| Order vs. Mutability | Index maintenance | Rebuild entire index on change | B-tree with local rebalancing |
| Complexity vs. Simplicity | Query execution | Interpret AST directly | Optimized bytecode virtual machine |

The **single-file constraint** of SQLite introduces additional challenges: all data, indexes, schema metadata, and free space tracking must coexist in one contiguous byte stream. This eliminates the "throw another file at it" solution and forces careful spatial planning—like designing a self-contained apartment where every square inch serves multiple purposes.

> **Key Insight**: The hardest part of database design isn't implementing any single feature, but making all features work together harmoniously within tight resource constraints. A well-designed database is like a Swiss Army knife—each tool (B-tree, parser, transaction manager) must fit precisely with the others.

### Existing Approaches and Their Trade-offs

Before designing our own solution, it's valuable to examine how existing systems approach the problem. Each represents a different point in the design space:

#### 1. Full-Featured RDBMS (PostgreSQL, MySQL)
These systems prioritize features and performance over simplicity:

**Strengths**:
- Sophisticated query optimization with cost-based planners
- Advanced concurrency control (MVCC)
- Extensive type system and SQL compliance
- Client-server architecture enables multi-process access

**Trade-offs**:
- Complex installation and management
- Multiple files/directories (data, indexes, logs, configuration)
- Larger memory footprint
- Steeper learning curve for internals

**Architecture Decision Record: Embedded vs. Client-Server**

> **Decision: Build an Embedded Database (Like SQLite) Rather Than Client-Server**
> 
> **Context**: We need to choose a fundamental architecture that determines how applications interact with our database. The system must be simple to deploy and suitable for educational implementation.
> 
> **Options Considered**:
> 1. **Client-server architecture**: Database runs as a separate process; applications connect via network/socket
> 2. **Embedded library**: Database runs in the same process as the application; accessed via function calls
> 3. **Hybrid**: Embedded with optional server mode
> 
> **Decision**: Embedded library architecture (like SQLite)
> 
> **Rationale**:
> - **Educational focus**: Fewer moving parts (no network protocol, connection pooling, authentication)
> - **Single-file constraint**: Naturally aligns with embedded model where database is a regular file
> - **Zero configuration**: No separate installation or daemon management
> - **Implementation simplicity**: Direct function calls instead of message serialization/deserialization
> 
> **Consequences**:
> - **Enables**: Simple deployment, no separate process management, faster intra-process communication
> - **Requires**: Careful handling of concurrent access from multiple threads in same process
> - **Limits**: No remote access without application-layer proxying, single process can become bottleneck

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Client-server | Natural concurrency, remote access, process isolation | Network overhead, complex protocol, separate installation | No |
| Embedded library | Zero configuration, fast intra-process calls, single file | Limited concurrency, crashes take down database | **Yes** |
| Hybrid | Flexibility for different use cases | Maximum complexity, dual code paths | No |

#### 2. Simple File Formats (CSV, JSON, Custom Binary)
Many applications start with simple file-based storage:

**Strengths**:
- Trivial to implement and debug
- Human-readable (CSV, JSON)
- No external dependencies
- Easy backup/restore (just copy the file)

**Trade-offs**:
- No transaction guarantees
- Poor performance for large datasets
- No indexing (full scans always)
- Ad-hoc querying difficult
- Concurrency issues (file locking or corruption)

#### 3. Specialized Embedded Databases (SQLite, LevelDB, LMDB)
These systems optimize for specific niches:

**SQLite's Design Philosophy** (which we'll emulate):
- **Serverless**: No separate server process
- **Zero-configuration**: No setup or administration
- **Transactional**: ACID-compliant with rollback journal
- **Single-file**: Entire database in one cross-platform file
- **Public domain**: Unrestricted use

**Trade-offs SQLite Accepts**:
- Weaker concurrency (writer locks whole database in default mode)
- Limited to single-machine access
- Simpler query optimizer than enterprise RDBMS

#### 4. In-Memory Structures (Maps, Lists, Trees)
Some applications keep everything in memory:

**Strengths**:
- Exceptional performance (nanosecond access)
- Rich data structures available
- No serialization/deserialization overhead

**Trade-offs**:
- Volatile (data lost on crash/restart)
- Limited by RAM size
- Manual persistence implementation needed
- Poor memory locality for large datasets

> **Design Principle**: Our system should occupy the "sweet spot" between these extremes—more capable than simple file formats but simpler than full RDBMS, specifically designed for learning database internals.

#### The Evolution of Data Persistence Approaches

Understanding why databases evolved as they did helps inform our design:

| Era | Dominant Approach | Why It Worked Then | Why It's Limited Now |
|-----|------------------|-------------------|---------------------|
| 1960s-70s | Hierarchical/Network DBs | Matched mainframe batch processing | Rigid schema, navigational queries complex |
| 1980s | Relational (Codd) | Declarative queries, mathematical foundation | Overhead for simple cases, "one size fits all" |
| 1990s | Client-server RDBMS | Centralized management, SQL standard | Scaling limits, administration complexity |
| 2000s | NoSQL/key-value | Web-scale, flexible schema, simple APIs | Lost ACID guarantees, ad-hoc querying difficult |
| 2010s+ | NewSQL/Embedded | Balance of SQL familiarity with modern needs | Still evolving, implementation complexity |

Our project follows the embedded lineage that proved surprisingly durable: **simplicity, reliability, and good-enough performance for most use cases**.

### The Core Problem Statement

Given these trade-offs, our specific problem is:

> **Design and implement an embedded SQL database engine that stores all data in a single file, supports a practical subset of SQL (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, CREATE INDEX), provides ACID transactions, and maintains acceptable performance through B-tree indexing—all while being understandable enough to serve as an educational tool for learning database internals.**

This problem decomposes into several sub-problems we'll address in subsequent sections:

1. **Language Interpretation**: How to parse SQL text into executable operations
2. **Storage Organization**: How to layout data on disk for efficient retrieval and mutation
3. **Query Execution**: How to transform high-level queries into low-level storage operations
4. **Transaction Management**: How to provide atomicity and durability guarantees
5. **Concurrency & Recovery**: How to handle multiple readers/writers and crash recovery

Each sub-problem interacts with the others, creating a classic systems engineering challenge: changes to the B-tree format affect the transaction journal, which affects recovery, which affects concurrency. The rest of this document outlines how to navigate these interdependencies.

### Common Pitfalls in Understanding the Problem Space

Before diving into solutions, let's address common misunderstandings:

⚠️ **Pitfall: Assuming databases just "write data to files"**
- **What's wrong**: Thinking persistence is as simple as `fwrite(row_data)` without considering corruption, partial writes, or concurrent access
- **Why it matters**: Without proper journaling, a crash during write leaves the database corrupted
- **How to avoid**: Design atomic write operations using write-ahead logging or shadow paging from the beginning

⚠️ **Pitfall: Optimizing prematurely for large datasets**
- **What's wrong**: Starting with complex paging strategies, compression, or sophisticated caching before basic correctness
- **Why it matters**: Complexity obscures bugs; educational value comes from understanding fundamentals first
- **How to avoid**: Implement straightforward B-trees with fixed-size pages first, optimize only after basic operations work

⚠️ **Pitfall: Underestimating the parser/compiler component**
- **What's wrong**: Treating SQL parsing as "just string splitting" rather than a full compiler frontend
- **Why it matters**: Complex WHERE clauses, nested expressions, and type checking require proper parsing
- **How to avoid**: Use recursive descent parsing with proper AST design from Milestone 2

⚠️ **Pitfall: Ignoring the single-file constraint implications**
- **What's wrong**: Designing components as if they had separate files, then trying to merge them
- **Why it matters**: Spatial allocation, free space management, and file growth strategies differ fundamentally
- **How to avoid**: Design all components knowing they share one contiguous byte array; use page-based allocation

### Implementation Guidance

While this foundational section doesn't correspond to a specific coding milestone, setting up your project structure correctly from the start prevents headaches later. Here's the recommended approach:

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Build System | Makefile with gcc | CMake with cross-platform support |
| Testing Framework | Simple test runner + asserts | Unity/CTest with coverage reporting |
| Debugging | printf debugging + hexdump | GDB/LLDB with custom pretty-printers |
| File I/O | Standard C stdio.h | Memory-mapped I/O for performance |
| Data Structures | Custom implementations | Reuse vetted libraries (uthash, etc.) |

**B. Recommended File/Module Structure:**

```
build-your-own-sqlite/
├── Makefile                    # Build configuration
├── src/                        # All source files
│   ├── main.c                  # REPL entry point
│   ├── sql/                    # SQL parsing components
│   │   ├── tokenizer.c         # Milestone 1: Tokenizer
│   │   ├── tokenizer.h
│   │   ├── parser.c            # Milestone 2: Parser
│   │   └── parser.h
│   ├── storage/                # B-tree storage engine
│   │   ├── pager.c             # Milestone 3: Page management
│   │   ├── pager.h
│   │   ├── btree.c             # Milestone 4: B-tree operations
│   │   └── btree.h
│   ├── execution/              # Query execution engine
│   │   ├── executor.c          # Milestones 5-8: Execution
│   │   └── executor.h
│   ├── transaction/            # Transaction management
│   │   ├── journal.c           # Milestone 9: Rollback journal
│   │   ├── journal.h
│   │   ├── wal.c               # Milestone 10: Write-ahead log
│   │   └── wal.h
│   └── utils/                  # Utilities
│       ├── common.h            # Common defines, constants
│       ├── errors.c            # Error handling
│       └── errors.h
├── test/                       # Test files
│   ├── test_runner.c           # Test harness
│   ├── test_tokenizer.c        # Milestone 1 tests
│   ├── test_parser.c           # Milestone 2 tests
│   └── ...                     # Other component tests
└── samples/                    # Example databases and queries
    ├── create_sample.sql
    └── queries.sql
```

**C. Infrastructure Starter Code:**

Here's a complete, reusable foundation for the database file interface:

```c
// src/storage/pager.c - Complete starter code for page management
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "pager.h"
#include "../utils/errors.h"

// Page size fixed at 4KB (SQLite default)
#define PAGE_SIZE 4096

struct Pager {
    FILE* file;
    uint32_t file_length;
    uint32_t num_pages;
    void* pages[TABLE_MAX_PAGES];  // Cache of in-memory pages
};

Pager* pager_open(const char* filename) {
    FILE* file = fopen(filename, "rb+");
    if (!file) {
        // File doesn't exist - create it
        file = fopen(filename, "wb+");
        if (!file) {
            fprintf(stderr, "Unable to open file %s: %s\n", 
                    filename, strerror(errno));
            return NULL;
        }
    }
    
    Pager* pager = malloc(sizeof(Pager));
    if (!pager) {
        fclose(file);
        return NULL;
    }
    
    pager->file = file;
    
    // Get file size
    fseek(file, 0, SEEK_END);
    pager->file_length = ftell(file);
    pager->num_pages = pager->file_length / PAGE_SIZE;
    
    // If file_length not multiple of PAGE_SIZE, we have a partial page
    if (pager->file_length % PAGE_SIZE != 0) {
        fprintf(stderr, "Database file %s is not a multiple of page size\n", 
                filename);
        exit(EXIT_FAILURE);
    }
    
    // Initialize page cache to NULL
    for (uint32_t i = 0; i < TABLE_MAX_PAGES; i++) {
        pager->pages[i] = NULL;
    }
    
    return pager;
}

void* pager_get_page(Pager* pager, uint32_t page_num) {
    if (page_num >= TABLE_MAX_PAGES) {
        fprintf(stderr, "Page number %d out of bounds\n", page_num);
        return NULL;
    }
    
    // Cache miss - load from file
    if (pager->pages[page_num] == NULL) {
        void* page = malloc(PAGE_SIZE);
        if (!page) {
            fprintf(stderr, "Memory allocation failed for page %d\n", page_num);
            return NULL;
        }
        
        // Calculate offset in file
        uint32_t file_offset = page_num * PAGE_SIZE;
        
        // Read page if it exists in file
        if (file_offset <= pager->file_length) {
            fseek(pager->file, file_offset, SEEK_SET);
            size_t bytes_read = fread(page, 1, PAGE_SIZE, pager->file);
            if (bytes_read < PAGE_SIZE && ferror(pager->file)) {
                fprintf(stderr, "Error reading page %d: %s\n", 
                        page_num, strerror(errno));
                free(page);
                return NULL;
            }
        }
        // If page doesn't exist in file yet, it will be all zeros
        
        pager->pages[page_num] = page;
        
        // Update page count if we're extending the file
        if (page_num >= pager->num_pages) {
            pager->num_pages = page_num + 1;
        }
    }
    
    return pager->pages[page_num];
}

void pager_flush_page(Pager* pager, uint32_t page_num) {
    if (pager->pages[page_num] == NULL) {
        fprintf(stderr, "Tried to flush null page %d\n", page_num);
        return;
    }
    
    uint32_t file_offset = page_num * PAGE_SIZE;
    fseek(pager->file, file_offset, SEEK_SET);
    size_t bytes_written = fwrite(pager->pages[page_num], 1, PAGE_SIZE, pager->file);
    if (bytes_written < PAGE_SIZE) {
        fprintf(stderr, "Error writing page %d: %s\n", 
                page_num, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

void pager_close(Pager* pager) {
    if (!pager) return;
    
    // Flush all cached pages
    for (uint32_t i = 0; i < pager->num_pages; i++) {
        if (pager->pages[i]) {
            pager_flush_page(pager, i);
            free(pager->pages[i]);
        }
    }
    
    // Close file
    if (pager->file) {
        fclose(pager->file);
    }
    
    free(pager);
}
```

**D. Core Logic Skeleton Code:**

While the main database logic comes later, here's a skeleton for the REPL (Read-Eval-Print Loop) that will drive all components:

```c
// src/main.c - Database REPL skeleton
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sql/parser.h"
#include "storage/pager.h"
#include "execution/executor.h"

#define INPUT_BUFFER_SIZE 1024

typedef struct {
    Pager* pager;
    // TODO: Add other database state (transaction state, schema cache, etc.)
} Database;

void print_prompt() {
    printf("db > ");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <database_file>\n", argv[0]);
        return 1;
    }
    
    // TODO 1: Initialize database connection
    // - Open database file using pager_open()
    // - Load schema from system tables
    // - Initialize transaction state
    
    // TODO 2: Create REPL loop
    // - Print prompt
    // - Read input (support multi-line statements)
    // - Parse SQL statement
    // - Execute statement
    // - Print results or errors
    // - Continue until ".exit" command
    
    // TODO 3: Handle meta-commands (start with ".")
    // - .exit - close database and exit
    // - .tables - list all tables
    // - .schema - show table schemas
    // - .help - show help
    
    // TODO 4: Clean shutdown
    // - Rollback any active transaction
    // - Close database file
    // - Free all allocated memory
    
    printf("Database REPL not yet implemented.\n");
    return 0;
}
```

**E. Language-Specific Hints for C:**

- **Error Handling**: Use consistent error codes rather than mixing error styles
- **Memory Management**: Every `malloc()` should have a corresponding `free()`; consider using arena allocators for temporary query results
- **File Operations**: Always check return values of `fread()`, `fwrite()`, `fseek()`; use `fflush()` and `fsync()` for durability
- **Structure Packing**: Use `#pragma pack(1)` or `__attribute__((packed))` for on-disk structures to avoid padding
- **Portability**: Use fixed-width types (`uint32_t`, `int64_t`) from `stdint.h` for cross-platform consistency

**F. Milestone Checkpoint (Foundation):**

Before starting Milestone 1, verify your environment is set up correctly:

1. **Compilation Test**: Run `make` (or `gcc -o db src/main.c src/storage/pager.c -I./src`) - should compile without errors
2. **Basic File I/O Test**: Create a simple test program that creates a 4KB file, writes a pattern, reads it back
3. **REPL Framework**: Create a simple loop that reads lines and prints them back until ".exit"
4. **Expected Behavior**: 
   ```
   $ ./db test.db
   db > hello;
   Unrecognized command: hello;
   db > .exit
   Goodbye!
   ```

**G. Debugging Tips for Initial Setup:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|------|
| "Segmentation fault" on startup | Uninitialized pointers in Pager struct | Use `valgrind` or add debug prints after each malloc | Initialize all struct fields after allocation |
| Database file grows unexpectedly | Pages not being reused | Add logging to track page allocations | Implement free page list in page 1 |
| Multi-line SQL statements fail | Input buffer too small | Check buffer size vs. actual SQL length | Use dynamic allocation or detect continuation |
| File corruption after crash | Missing fsync calls | Add fsync to pager_flush_page | Call `fsync(fileno(pager->file))` after writes |

---


## 2. Goals and Non-Goals
> **Milestone(s):** Foundational (establishes scope for all milestones)

This educational implementation aims to build a **minimal viable SQLite** — a functional embedded database that demonstrates core database engine concepts while remaining tractable for a single developer or small team. Think of this project as building a **teaching skeleton** rather than a production database: it should have all the essential bones (storage, querying, transactions) but without the complex musculature of a full commercial database (optimizations, extensive SQL support, high concurrency).

The primary constraint is **educational value per line of code**. Every feature must directly illuminate a fundamental database concept. Features that would require disproportionate implementation complexity for their learning value are explicitly excluded, even if they're standard in production databases. This disciplined scoping ensures the project remains achievable while providing deep insight into how databases work under the hood.

### Goals (Must-Have Features)

These features represent the essential capabilities that demonstrate core database engine principles. Each maps directly to one or more milestones in the development roadmap.

| Feature Category | Specific Capability | Description | Corresponding Milestone(s) | Learning Objective |
|-----------------|---------------------|-------------|----------------------------|-------------------|
| **SQL Language Support** | **Basic DML**: `SELECT`, `INSERT`, `UPDATE`, `DELETE` | Support fundamental data manipulation with filtering via `WHERE` clauses | Milestone 1, 2, 5, 6, 7 | SQL parsing, AST construction, query execution |
| | **Basic DDL**: `CREATE TABLE`, `CREATE INDEX` | Define table schemas and build secondary indexes for faster lookups | Milestone 2, 4, 7 | Schema management, B-tree construction |
| | **Expressions**: arithmetic, comparisons, boolean logic | Evaluate expressions in `WHERE` clauses and column projections | Milestone 2, 7 | Expression trees, predicate evaluation |
| **Storage Engine** | **B-tree organization** with rowid primary key | Store all table rows in a clustered B-tree index keyed by 64-bit rowid | Milestone 3, 4 | B-tree structure, page layout, variable-length records |
| | **Secondary B-tree indexes** | Create non-clustered indexes on specific columns for faster lookups | Milestone 7 | Secondary index structure, index maintenance |
| | **Fixed-size pages (4096 bytes)** | Organize storage in standard-sized pages matching OS/hardware blocks | Milestone 3 | Page-based I/O, buffer management |
| **Query Execution** | **Table scan** with column projection | Read all rows sequentially with optional column filtering | Milestone 5 | B-tree traversal, cursor pattern, result set building |
| | **Index scan** for equality and range queries | Use B-tree search to locate rows by indexed column values | Milestone 7, 8 | Index lookup, query planning basics |
| | **Basic cost-based query planning** | Choose between table scan and index scan based on estimated I/O cost | Milestone 8 | Cost estimation, plan enumeration |
| **Transactions & Durability** | **ACID transactions** with `BEGIN`/`COMMIT`/`ROLLBACK` | Group operations atomically with rollback capability | Milestone 9 | Atomicity, consistency, isolation basics |
| | **Rollback journal** for crash recovery | Record original page content before modification for undo | Milestone 9 | Crash recovery, atomic commit |
| | **Write-Ahead Logging (WAL) mode** | Alternative journaling for better concurrency (reader/writer) | Milestone 10 | WAL protocol, MVCC basics, checkpointing |
| | **`fsync()` for durability** | Ensure writes reach persistent storage at commit boundaries | Milestone 9, 10 | Durability guarantees, I/O ordering |
| **Constraints** | **`NOT NULL` constraint** | Reject attempts to insert NULL into constrained columns | Milestone 6 | Constraint validation during write |
| | **`UNIQUE` constraint** (via index) | Enforce uniqueness using underlying B-tree index structure | Milestone 6, 7 | Constraint enforcement, index maintenance |
| **System Architecture** | **Single-file database** | All data, indexes, and metadata in one portable file | All milestones | File format design, self-contained storage |
| | **Embedded C library** API | Clean C interface following SQLite's style (open, exec, close) | Integration across milestones | Library design, clean abstraction boundaries |
| | **REPL (Read-Eval-Print Loop)** | Interactive shell for executing SQL commands | Throughout development | User interface, debugging tool |

> **Key Insight:** These goals were selected because each one demonstrates a *fundamental database concept* that cannot be omitted without losing understanding of how databases work. For example, B-trees are non-negotiable because they're the standard structure for disk-based indexing; transactions are essential because they demonstrate how databases maintain consistency despite failures.

**Architecture Decision Record: Feature Selection Philosophy**

> **Decision: Educational Value Over Production Completeness**
> - **Context**: We're building a teaching database, not a production system. Learners need to understand core concepts without being overwhelmed by edge cases, optimizations, or rarely-used features.
> - **Options Considered**:
>   1. **Maximalist**: Implement extensive SQL-92 compliance with all common extensions
>   2. **Minimalist**: Implement only the absolute minimum needed to store and retrieve data
>   3. **Concept-focused**: Implement features that best illustrate database engine internals
> - **Decision**: Choose option 3 (concept-focused), prioritizing features that teach important lessons about database architecture.
> - **Rationale**: 
>   - Learners will understand B-trees better by implementing them than by using a black-box storage layer
>   - Transactions teach atomicity and durability more effectively than any theoretical explanation
>   - Query planning illustrates the cost-based optimization mindset even in simplified form
>   - Each implemented feature should answer "how do databases do X?" for fundamental X
> - **Consequences**:
>   - The database will be impractical for most real applications (missing too many features)
>   - Learners will gain deep insight into the *mechanisms* rather than surface-level API knowledge
>   - Implementation remains tractable within a few thousand lines of code

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Maximalist (full SQL) | Practical for real use, tests edge cases | Implementation complexity explodes, obscures core concepts | ✗ |
| Minimalist (bare bones) | Very simple to implement | Misses too many educational opportunities (transactions, indexes) | ✗ |
| Concept-focused | Teaches why databases work, not just how to use them | Not production-ready, limited SQL dialect | ✓ |

### Non-Goals (Explicitly Out of Scope)

These features are deliberately excluded to maintain focus on core educational objectives. Many are standard in production databases but would add disproportionate complexity without corresponding educational value.

| Feature Category | Specific Exclusion | Reason for Exclusion | Alternative Approach for Learning |
|-----------------|-------------------|----------------------|-----------------------------------|
| **SQL Language** | **`JOIN` operations** (any type) | Implementation complexity is high relative to educational value; joins involve multiple algorithms (nested loop, hash, merge) and query planning complexity | Focus on single-table operations; learners can extend later as an exercise |
| | **Subqueries, Views, CTEs** | These are syntactic sugar that build on fundamental execution primitives; implementing them doesn't teach new storage/execution concepts | Demonstrate same capabilities through multiple queries in application code |
| | **Triggers, Stored Procedures** | Move logic into database engine rather than illustrating engine internals; complex to implement correctly | Not needed to understand storage, indexing, or transaction mechanisms |
| | **Full SQL-92 compliance** | Would require extensive parser complexity and edge case handling | Implement a clean subset that demonstrates parsing concepts without getting bogged down |
| **Type System** | **Extensive type system** (DATE, BLOB, etc.) | Type handling adds complexity without illuminating core database concepts | Support only INTEGER, TEXT, and REAL (SQLite's storage classes) to demonstrate type representation |
| | **Type coercion rules** | Complex specification with many edge cases | Use simple, predictable rules: store what's given, retrieve as requested |
| **Concurrency** | **Multiple concurrent writers** | Requires sophisticated locking protocols (MVCC, WAL with readers) that are complex to implement correctly | Support single writer or WAL mode with one writer + multiple readers (Milestone 10) |
| | **Fine-grained locking** (row-level) | Adds significant complexity to transaction manager | Use simpler approaches: database-level lock or WAL for reader/writer concurrency |
| **Optimization** | **Query optimization beyond index selection** | Full query optimization is a vast topic (join ordering, subquery flattening, etc.) | Implement only basic cost-based index selection to illustrate the optimization mindset |
| | **Statistics collection/updating** | Maintaining histograms and cardinality estimates adds significant complexity | Use simple heuristics or hard-coded estimates for planning decisions |
| **Storage** | **Multiple database files or ATTACH** | Complicates transaction atomicity and storage management | Single-file design keeps architecture simple and portable |
| | **Compression, encryption** | These are features built atop core storage engine; don't illuminate database fundamentals | Focus on the raw storage format without transformations |
| **Administration** | **VACUUM, ANALYZE commands** | Maintenance operations important for production but not for understanding core algorithms | Skip or implement as trivial no-ops |
| | **Backup/restore API** | Important practical feature but doesn't teach new database concepts | Learners can copy the single file for backup |
| **Compatibility** | **SQLite API compatibility** | Full compatibility would constrain design choices and add testing burden | Implement a clean subset inspired by SQLite but not bound by exact compatibility |
| | **File format compatibility** | Reading/writing actual SQLite files requires implementing full format complexity | Use a simpler, custom file format that illustrates the same concepts |

> **Design Principle:** Every line of code should answer a "how does a database do X?" question. If implementing a feature would require significant code that doesn't directly illuminate a core database concept, it's a candidate for exclusion.

**Common Pitfalls in Scope Definition**

⚠️ **Pitfall: Creeping Featurism**
- **Description**: Adding "just one more" SQL feature because it seems useful or interesting
- **Why It's Wrong**: Each additional feature increases code complexity geometrically (interactions between features), pushing the project beyond educational scope into a never-ending implementation effort
- **How to Avoid**: Strictly adhere to the feature list above. When tempted to add something, ask: "Does this teach a fundamentally new database concept that isn't already covered?"

⚠️ **Pitfall: Premature Optimization**
- **Description**: Spending time on performance optimizations before core functionality works
- **Why It's Wrong**: Educational value comes from understanding the algorithms, not from micro-optimizations. A simple, clear implementation is more valuable than a fast but opaque one
- **How to Avoid**: Implement the straightforward approach first (e.g., linear scan). Only optimize (e.g., add indexes) when it demonstrates a specific educational concept

⚠️ **Pitfall: Production-Grade Error Handling**
- **Description**: Attempting to handle every possible error case with detailed messages and recovery
- **Why It's Wrong**: Error handling code can dominate the codebase and obscure the core algorithms. Some error cases are extremely rare in educational contexts
- **How to Avoid**: Handle obvious, common errors (file not found, malformed SQL) with clear messages. For complex error scenarios, use assertions or simple error returns

**Concrete Example: The JOIN Exclusion**
Consider a query like `SELECT * FROM users, orders WHERE users.id = orders.user_id`. Implementing this requires:
1. Multiple table access methods
2. Join algorithms (nested loop at minimum)
3. More complex query planning
4. Result set combining columns from multiple tables

While joins are fundamental to SQL, their implementation complexity is disproportionate to their educational value for understanding storage, indexing, and single-table query execution. A learner who understands how to scan a single table and evaluate WHERE clauses has 90% of what's needed to understand joins conceptually.

### Implementation Guidance

**Technology Recommendations**

| Component | Simple Option (Recommended) | Advanced Option (Alternative) |
|-----------|-----------------------------|-------------------------------|
| SQL Parser | Hand-written recursive descent in C | Parser generator (Lemon, yacc/bison) with grammar file |
| B-tree Storage | In-memory pages as `uint8_t[4096]` arrays | Memory-mapped file I/O for automatic paging |
| Query Execution | Iterator model with explicit cursor objects | Vectorized execution model for batch processing |
| Transaction Journaling | Simple rollback journal (copy-before-write) | Full WAL implementation with checkpointing |
| File I/O | Standard C `stdio.h` (fopen, fread, fwrite) | POSIX file API (open, read, write) for finer control |

**Recommended File/Module Structure**
```
build-your-own-sqlite/
├── src/
│   ├── main.c                    # REPL entry point, main loop
│   ├── database.h/c              # Database lifecycle (open/close)
│   ├── pager.h/c                 # Page cache management (Pager type)
│   ├── btree.h/c                 # B-tree operations (insert, search, split)
│   ├── table.h/c                 # Table-level operations (create, insert, select)
│   ├── cursor.h/c                # Cursor for iteration
│   ├── parser.h/c                # SQL parsing (tokenizer + parser)
│   ├── ast.h/c                   # Abstract Syntax Tree definitions
│   ├── execution.h/c             # Query execution engine
│   ├── planner.h/c               # Query planner (cost estimation)
│   ├── transaction.h/c           # Transaction management
│   ├── wal.h/c                   # Write-Ahead Log implementation
│   └── utils.h/c                 # Utilities (varint encoding, etc.)
├── include/                      # Public API headers
│   └── sqlite_simple.h           # Library interface
├── tests/                        # Test suite
│   ├── test_parser.c
│   ├── test_btree.c
│   └── test_integration.c
└── Makefile                      # Build system
```

**Infrastructure Starter Code**
```c
/* src/pager.h - Complete Pager implementation */
#ifndef PAGER_H
#define PAGER_H

#include <stdint.h>
#include <stdio.h>

#define PAGE_SIZE 4096
#define TABLE_MAX_PAGES 100

typedef struct {
    FILE* file;
    uint32_t file_length;
    uint32_t num_pages;
    void* pages[TABLE_MAX_PAGES];
} Pager;

Pager* pager_open(const char* filename);
void* pager_get_page(Pager* pager, uint32_t page_num);
void pager_flush_page(Pager* pager, uint32_t page_num);
void pager_close(Pager* pager);

#endif
```

```c
/* src/pager.c - Complete Pager implementation */
#include "pager.h"
#include <stdlib.h>
#include <string.h>

Pager* pager_open(const char* filename) {
    FILE* file = fopen(filename, "rb+");
    if (!file) {
        file = fopen(filename, "wb+");
        if (!file) return NULL;
    }
    
    Pager* pager = malloc(sizeof(Pager));
    pager->file = file;
    
    fseek(file, 0, SEEK_END);
    pager->file_length = ftell(file);
    pager->num_pages = pager->file_length / PAGE_SIZE;
    
    if (pager->file_length % PAGE_SIZE != 0) {
        fprintf(stderr, "Database file is not a whole number of pages. Corrupt file.\n");
        exit(EXIT_FAILURE);
    }
    
    for (uint32_t i = 0; i < TABLE_MAX_PAGES; i++) {
        pager->pages[i] = NULL;
    }
    
    return pager;
}

void* pager_get_page(Pager* pager, uint32_t page_num) {
    if (page_num >= TABLE_MAX_PAGES) {
        fprintf(stderr, "Page number out of bounds: %d >= %d\n", page_num, TABLE_MAX_PAGES);
        exit(EXIT_FAILURE);
    }
    
    if (pager->pages[page_num] == NULL) {
        void* page = malloc(PAGE_SIZE);
        uint32_t num_pages = pager->file_length / PAGE_SIZE;
        
        if (page_num < num_pages) {
            fseek(pager->file, page_num * PAGE_SIZE, SEEK_SET);
            size_t bytes_read = fread(page, PAGE_SIZE, 1, pager->file);
            if (bytes_read != 1) {
                fprintf(stderr, "Error reading page %d\n", page_num);
                exit(EXIT_FAILURE);
            }
        }
        
        pager->pages[page_num] = page;
        
        if (page_num >= pager->num_pages) {
            pager->num_pages = page_num + 1;
        }
    }
    
    return pager->pages[page_num];
}

void pager_flush_page(Pager* pager, uint32_t page_num) {
    if (pager->pages[page_num] == NULL) {
        return;
    }
    
    fseek(pager->file, page_num * PAGE_SIZE, SEEK_SET);
    size_t bytes_written = fwrite(pager->pages[page_num], PAGE_SIZE, 1, pager->file);
    if (bytes_written != 1) {
        fprintf(stderr, "Error writing page %d\n", page_num);
        exit(EXIT_FAILURE);
    }
    
    fflush(pager->file);
}

void pager_close(Pager* pager) {
    for (uint32_t i = 0; i < pager->num_pages; i++) {
        if (pager->pages[i] != NULL) {
            pager_flush_page(pager, i);
            free(pager->pages[i]);
            pager->pages[i] = NULL;
        }
    }
    
    fclose(pager->file);
    free(pager);
}
```

**Core Logic Skeleton Code**
```c
/* src/database.h - Database type and operations */
#ifndef DATABASE_H
#define DATABASE_H

#include "pager.h"

typedef struct {
    Pager* pager;
    // TODO: Add transaction state, schema cache, etc.
} Database;

Database* database_open(const char* filename);
void database_close(Database* db);
int database_execute(Database* db, const char* sql);

#endif
```

```c
/* src/database.c - Database operations skeleton */
#include "database.h"
#include "parser.h"
#include "execution.h"
#include <stdlib.h>

Database* database_open(const char* filename) {
    Database* db = malloc(sizeof(Database));
    if (!db) return NULL;
    
    db->pager = pager_open(filename);
    if (!db->pager) {
        free(db);
        return NULL;
    }
    
    // TODO 1: Read schema from first page of database
    // TODO 2: Initialize transaction state to IDLE
    // TODO 3: Load system catalog (list of tables)
    
    return db;
}

void database_close(Database* db) {
    // TODO 1: Check for active transaction, rollback if needed
    // TODO 2: Flush all dirty pages
    // TODO 3: Free schema cache
    
    pager_close(db->pager);
    free(db);
}

int database_execute(Database* db, const char* sql) {
    // TODO 1: Parse SQL into AST using parser_parse(sql)
    // TODO 2: Validate AST against schema (check table exists, columns exist)
    // TODO 3: Plan query using planner_create_plan(ast)
    // TODO 4: Execute plan using execution_execute_plan(db, plan)
    // TODO 5: Handle errors at each step, return appropriate error codes
    // TODO 6: For SELECT queries, print results to stdout
    // TODO 7: For modification queries, return rows affected count
    
    return 0; // Success
}
```

**Language-Specific Hints (C)**
- Use `fdatasync()` or `fsync()` after critical writes for durability guarantees
- Implement variable-length integer encoding using SQLite's varint algorithm for compact storage
- Use `memcpy()` for serializing structures to byte buffers to avoid alignment issues
- Consider using a union for page types to access both header and cell areas conveniently
- For the REPL, use `readline()` or simple `fgets()` with line editing support

**Milestone Checkpoint**
After implementing the basic pager and database open/close:
1. **Test command**: `make && ./sqlite test.db`
2. **Expected output**: `sqlite> ` prompt appears
3. **Verify behavior**: 
   - Creates `test.db` file if it doesn't exist
   - File size is 0 bytes initially
   - Can exit with `.exit` command
4. **Debugging signs**:
   - If program crashes on start: Check `pager_open` file handling
   - If no prompt appears: Check main REPL loop in `main.c`
   - If file isn't created: Check file permissions and `fopen` modes

**Debugging Tips**
| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Database file grows but queries return no data | Pages not being initialized with proper B-tree headers | Use hex dump (`xxd test.db`) to check page contents | Ensure `btree_init_empty_page()` sets page type and cell count |
| INSERT works but SELECT returns wrong rows | Row serialization/deserialization mismatch | Print raw byte values during serialization and deserialization | Ensure same byte order and field sizes in both directions |
| Program crashes after many INSERTs | Memory leak in page cache | Use valgrind or count malloc/free calls | Ensure all allocated pages are freed in `pager_close()` |
| WAL mode causes infinite loop | Checkpoint not removing old frames | Add debug prints to WAL frame header reading | Ensure checkpoint updates WAL header's checkpoint field |


## 3. High-Level Architecture

> **Milestone(s):** All (provides the architectural foundation for every milestone)

This section presents the top-level organization of our SQLite implementation. Think of the system as a **layered pipeline** where SQL commands flow downward from human-readable text to on-disk binary data, and query results flow upward from raw bytes to formatted output. Each layer has a specific responsibility and communicates with adjacent layers through well-defined interfaces, allowing us to build and test components independently.

### Component Overview and Responsibilities

At its core, our database engine follows a **three-layer architecture** that mirrors how humans interact with physical filing systems:

1. **Front Desk (Parser Layer)**: Receives written requests (SQL statements) and translates them into precise internal work orders (AST).
2. **Office Manager (Execution Layer)**: Receives work orders, makes decisions about the most efficient way to accomplish them, and directs workers to access or modify files.
3. **Filing Room (Storage Layer)**: Where the actual data lives on shelves (B-tree pages), with clerks (cursors) that can quickly locate, retrieve, or update records.

![System Component Diagram](./diagrams/sys-component.svg)

These three primary components work together through clean interfaces:

| Component | Primary Responsibility | Key Data Structures Owned | Critical Operations |
|-----------|-----------------------|---------------------------|---------------------|
| **SQL Parser** | Translates SQL text into executable internal representations | `Token` stream, `AST` nodes (SELECT, INSERT, CREATE) | Tokenization, Syntax validation, AST construction |
| **Query Engine** | Plans and executes queries, enforces constraints, manages transactions | `ExecutionPlan`, `Cursor`, `TransactionState` | Query planning, Operator execution, Constraint checking |
| **B-tree Storage** | Manages on-disk data organization, provides efficient key-value access | `Pager`, `Page` buffers, `BTreeNode` structures | Page management, B-tree operations (insert/split/delete) |
| **Transaction Manager** | Ensures ACID properties through journaling/WAL | `Journal` file, `WAL` frames, `Lock` structures | Journaling, Rollback, Checkpointing, Recovery |

> **Key Insight**: This separation of concerns allows us to evolve each component independently. For example, we could replace the B-tree storage with a different indexing structure without changing how the parser or query engine works, as long as we maintain the same interface for key-value operations.

**Component Interactions** follow a clear flow:
1. **SQL Text** → `Parser` → **AST**
2. **AST** → `Query Planner` → **Execution Plan**
3. **Execution Plan** → `Operators` → `Storage Cursors` → **Page Access**
4. **Page Modifications** → `Transaction Manager` → **Journal/WAL Records**
5. **Commit** → `Transaction Manager` → **Durability Guarantees**

**Architecture Decision: Component Decomposition Strategy**

> **Decision: Three-Layer Architecture with Clean Interfaces**
> - **Context**: We need to balance educational clarity with practical functionality. A monolithic design would be simpler to start but harder to understand, test, and extend. We must choose a decomposition that aligns with the learning milestones while providing clear separation of concerns.
> - **Options Considered**:
>   1. **Monolithic Single Module**: All code in one module/file
>   2. **Two-Layer (Frontend/Backend)**: Parser as frontend, everything else as backend
>   3. **Three-Layer (Parser/Engine/Storage)**: Clear separation between parsing, execution, and storage
>   4. **Microkernel with Plugins**: Core engine with pluggable components for parsing, storage, etc.
> - **Decision**: Three-layer architecture (Parser/Engine/Storage) with Transaction Manager as a cross-cutting concern
> - **Rationale**: 
>   - Aligns perfectly with the milestone progression (Parser → Storage → Execution → Transactions)
>   - Each layer has distinct responsibilities that match natural cognitive boundaries
>   - Enables independent testing: we can test the parser with mock SQL, the storage engine with synthetic data, and the query engine with pre-built ASTs
>   - Transaction management affects both storage (page writes) and execution (isolation), so it logically sits orthogonal to the main layers
> - **Consequences**:
>   - **Positive**: Clear learning progression, easier debugging (faults are localized), better code organization
>   - **Negative**: Slight overhead from interface abstractions, requires careful design of cross-layer APIs
>   - **Mitigation**: Define simple, focused interfaces that don't over-generalize

**Comparison of Architecture Options**:

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Monolithic** | Simplest to implement initially, no interface overhead | Becomes unmanageable quickly, hard to test components in isolation, violates single responsibility principle | Fails at educational clarity beyond very small codebases |
| **Two-Layer** | Clear separation between SQL and execution | Storage and execution logic mixed together, transaction handling ambiguous | Doesn't provide enough separation for independent milestone development |
| **Three-Layer** | Clear responsibilities, matches milestone structure, enables independent testing | Requires careful interface design, slightly more upfront design work | **CHOSEN**: Best balance of clarity, testability, and alignment with learning goals |
| **Microkernel** | Maximum flexibility, could support multiple SQL dialects or storage engines | Significant complexity overhead, over-engineering for educational project | Too complex for learning objectives; flexibility not needed |

The **Transaction Manager** deserves special attention as a cross-cutting component. Unlike the three main layers that process data in a linear pipeline, the transaction manager interacts with multiple layers:

- **With Storage**: Intercepts page writes to journal/WAL before modifying the main database file
- **With Query Engine**: Provides isolation between concurrent transactions
- **With Parser**: Processes `BEGIN`, `COMMIT`, `ROLLBACK` statements as transaction control commands

This architecture is visualized in the system component diagram, where arrows show data flow and dependency directions. Notice that:
- Dependencies flow downward: Parser depends on nothing, Query Engine depends on Parser's AST, Storage depends on nothing (it's a foundation)
- Transaction Manager has bidirectional relationships with both Query Engine and Storage
- The Database File and optional WAL File are external resources managed by Storage and Transaction Manager respectively

### Recommended File/Module Structure

Organizing code files effectively is crucial for managing complexity as the project grows. We recommend a **module-per-component** structure with clear public interfaces and private implementations. Think of each module as a **specialized workshop** within a larger factory:

- **Parser Workshop**: Raw materials (SQL text) enter, precise blueprints (AST) exit
- **Engine Workshop**: Blueprints enter, work instructions for the storage team exit  
- **Storage Workshop**: Work instructions enter, physical products (data pages) are retrieved or modified
- **Transaction Office**: Oversees all workshops, maintains logs of all activities

**Module Dependencies and Interfaces**:

| Module | Public Interface (Header File) | Private Implementation | Dependencies |
|--------|-------------------------------|------------------------|--------------|
| **parser** | `parser.h` (AST types, `parse_statement()`) | `tokenizer.c`, `ast_builder.c` | None (pure text to AST) |
| **storage** | `pager.h`, `btree.h` | `page.c`, `cell.c` | System I/O (`<stdio.h>`) |
| **query** | `executor.h`, `planner.h` | `operators.c`, `cursor.c` | parser (AST types), storage (pager/btree) |
| **transaction** | `transaction.h`, `wal.h` | `journal.c`, `recovery.c` | storage (pager), system I/O |
| **main** | `database.h` (public API) | `database.c` (glue code) | All modules |

**Key Interface Contracts**:

| Interface | Provider Module | Consumer Module | Contract Description |
|-----------|----------------|-----------------|----------------------|
| `AST* parse_statement(const char* sql)` | Parser | Query Engine | Returns fully parsed AST or NULL on error; caller responsible for freeing |
| `Cursor* btree_open_cursor(BTree* tree, Key* key)` | Storage | Query Engine | Returns positioned cursor; caller must close cursor when done |
| `int pager_write_page(Pager* pager, uint32_t page_num)` | Storage | Transaction | Writes page to storage; returns 0 on success, error code otherwise |
| `void journal_begin_transaction(Journal* journal)` | Transaction | Query Engine | Starts recording page modifications for potential rollback |

**Module Communication Patterns**:

1. **Synchronous Direct Calls** for most operations (simplest, adequate for single-threaded embedded use)
2. **Callback Registration** for transaction hooks (storage notifies transaction manager before page modifications)
3. **Data Structures as Messages** (ASTs flow from parser to engine, execution plans flow from engine to storage)

**Architecture Decision: Module Boundary Strategy**

> **Decision: Compilation Unit per Component with Header File Interfaces**
> - **Context**: We need to decide how to physically organize code files. The options range from a single massive file to many small files. We must balance compile times, encapsulation, and navigability.
> - **Options Considered**:
>   1. **Single File**: All code in one `.c` file
>   2. **Header-Only Library**: Implementation in header files with `static` functions
>   3. **Component-Based**: Each major component in separate `.c/.h` file pairs
>   4. **Class-Style**: Each data structure gets its own `.c/.h` pair (even if tiny)
> - **Decision**: Component-based organization with one `.c/.h` pair per architectural component
> - **Rationale**:
>   - **Compilation Speed**: Independent compilation of components speeds up development
>   - **Encapsulation**: Private functions remain truly private (not exposed in headers)
>   - **Cognitive Load**: Developers can focus on one component at a time
>   - **Testing**: Components can be tested in isolation by mocking their dependencies
>   - **Educational Value**: Mirrors real-world C project structure
> - **Consequences**:
>   - **Positive**: Clean separation, faster incremental builds, better encapsulation
>   - **Negative**: More files to navigate, need to manage header dependencies carefully
>   - **Mitigation**: Use a clear directory structure and `#include` guards religiously

**Module Dependency Rules** (enforced through build configuration):

1. **No Circular Dependencies**: Dependencies must form a directed acyclic graph
2. **Higher Layers Depend on Lower Layers**: Parser → Engine → Storage (transaction manager is special)
3. **Public Interfaces Only**: Modules communicate only through published header files
4. **Private Implementation Freedom**: Implementation files can change without affecting dependents

**Build System Implications**: This structure works well with simple Makefiles:
- Each `.c` file compiles to its own `.o` object file
- Header files declare interfaces without exposing implementation details
- Final linking combines all object files into the executable
- Changes to one component's implementation only require recompiling that component

### Implementation Guidance

**A. Technology Recommendations Table**:

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Build System** | Plain Makefile | CMake with cross-platform support |
| **Testing** | Simple test runner with asserts | Unity/CTest with test fixtures |
| **Memory Management** | Manual `malloc`/`free` with careful tracking | Arena/region allocator for query execution |
| **Error Handling** | Return error codes with descriptive messages | Error code system with error recovery contexts |
| **Serialization** | Manual byte-by-byte packing/unpacking | Generated serialization code (could use macros) |

**B. Recommended File/Module Structure**:

```
sqlite_educational/
├── Makefile                    # Build configuration
├── src/                        # All source code
│   ├── main.c                  # REPL entry point, command dispatch
│   ├── include/                # Public header files
│   │   ├── database.h          # Main public API (database_open/close/execute)
│   │   ├── parser.h            # Parser public interface
│   │   ├── storage.h           # Storage public interface
│   │   ├── query.h             # Query engine public interface
│   │   └── transaction.h       # Transaction public interface
│   ├── parser/                 # Milestone 1-2: SQL Parser
│   │   ├── parser.c            # Main parser implementation
│   │   ├── tokenizer.c         # Tokenizer (lexer) implementation
│   │   ├── ast.c               # AST node definitions and utilities
│   │   └── parser_private.h    # Private parser headers
│   ├── storage/                # Milestone 3-4: B-tree Storage
│   │   ├── pager.c             # Page cache management
│   │   ├── btree.c             # B-tree operations
│   │   ├── page.c              # Page layout and serialization
│   │   ├── cell.c              # Cell (key-value) handling
│   │   └── storage_private.h   # Private storage headers
│   ├── query/                  # Milestone 5-8: Query Execution
│   │   ├── executor.c          # Query execution engine
│   │   ├── planner.c           # Query planner/optimizer
│   │   ├── operators.c         # Scan, filter, project operators
│   │   ├── cursor.c            # B-tree cursor implementation
│   │   └── query_private.h     # Private query headers
│   └── transaction/            # Milestone 9-10: Transactions
│       ├── transaction.c       # Transaction state management
│       ├── journal.c           # Rollback journal implementation
│       ├── wal.c               # Write-ahead logging (advanced)
│       ├── recovery.c          # Crash recovery
│       └── transaction_private.h
├── tests/                      # Test suites
│   ├── test_parser.c
│   ├── test_storage.c
│   ├── test_query.c
│   └── test_runner.c
└── tools/                      # Development utilities
    ├── hexdump.c               # View database file contents
    └── sql_shell.c             # Alternative REPL implementation
```

**C. Infrastructure Starter Code**:

Here's complete, working code for the foundational `Pager` component that manages page caching and file I/O. This is infrastructure that learners can use without modification:

```c
// src/storage/pager.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include "include/storage.h"

#define PAGE_SIZE 4096
#define TABLE_MAX_PAGES 100

typedef struct Pager {
    int file_descriptor;
    uint32_t file_length;
    uint32_t num_pages;
    void* pages[TABLE_MAX_PAGES];
} Pager;

Pager* pager_open(const char* filename) {
    int fd = open(filename, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        fprintf(stderr, "Unable to open file %s: %s\n", filename, strerror(errno));
        return NULL;
    }
    
    off_t file_length = lseek(fd, 0, SEEK_END);
    
    Pager* pager = malloc(sizeof(Pager));
    pager->file_descriptor = fd;
    pager->file_length = file_length;
    pager->num_pages = (file_length + PAGE_SIZE - 1) / PAGE_SIZE;
    
    for (uint32_t i = 0; i < TABLE_MAX_PAGES; i++) {
        pager->pages[i] = NULL;
    }
    
    return pager;
}

void* pager_get_page(Pager* pager, uint32_t page_num) {
    if (page_num >= TABLE_MAX_PAGES) {
        fprintf(stderr, "Page number out of bounds: %d >= %d\n", page_num, TABLE_MAX_PAGES);
        return NULL;
    }
    
    if (pager->pages[page_num] == NULL) {
        void* page = malloc(PAGE_SIZE);
        uint32_t num_pages = pager->file_length / PAGE_SIZE;
        
        if (pager->file_length % PAGE_SIZE) {
            num_pages += 1;
        }
        
        if (page_num < num_pages) {
            lseek(pager->file_descriptor, page_num * PAGE_SIZE, SEEK_SET);
            ssize_t bytes_read = read(pager->file_descriptor, page, PAGE_SIZE);
            if (bytes_read == -1) {
                fprintf(stderr, "Error reading page %d: %s\n", page_num, strerror(errno));
                free(page);
                return NULL;
            }
        }
        
        pager->pages[page_num] = page;
        
        if (page_num >= pager->num_pages) {
            pager->num_pages = page_num + 1;
        }
    }
    
    return pager->pages[page_num];
}

void pager_flush_page(Pager* pager, uint32_t page_num) {
    if (pager->pages[page_num] == NULL) {
        fprintf(stderr, "Tried to flush null page: %d\n", page_num);
        return;
    }
    
    off_t offset = lseek(pager->file_descriptor, page_num * PAGE_SIZE, SEEK_SET);
    if (offset == -1) {
        fprintf(stderr, "Error seeking to page %d: %s\n", page_num, strerror(errno));
        return;
    }
    
    ssize_t bytes_written = write(pager->file_descriptor, pager->pages[page_num], PAGE_SIZE);
    if (bytes_written == -1) {
        fprintf(stderr, "Error writing page %d: %s\n", page_num, strerror(errno));
    }
}

void pager_close(Pager* pager) {
    for (uint32_t i = 0; i < pager->num_pages; i++) {
        if (pager->pages[i] != NULL) {
            pager_flush_page(pager, i);
            free(pager->pages[i]);
            pager->pages[i] = NULL;
        }
    }
    
    close(pager->file_descriptor);
    free(pager);
}
```

**D. Core Logic Skeleton Code**:

Here's the skeleton for the main `database_execute` function that ties all components together. Learners will fill in the TODOs:

```c
// src/main.c (simplified version)
#include "include/database.h"
#include "include/parser.h"
#include "include/query.h"
#include "include/transaction.h"

int database_execute(Database* db, const char* sql) {
    // TODO 1: Parse the SQL statement into an AST
    // Call parse_statement() from parser module
    // Handle parse errors: return appropriate error code
    
    // TODO 2: Check if this is a transaction control statement (BEGIN/COMMIT/ROLLBACK)
    // If yes, handle via transaction module and return
    
    // TODO 3: Check if this is a DDL statement (CREATE TABLE/INDEX)
    // If yes, update schema catalog and storage structures
    
    // TODO 4: For DML statements (SELECT/INSERT/UPDATE/DELETE):
    //   a. Generate execution plan using planner module
    //   b. Begin transaction if not already in one (autocommit mode)
    //   c. Execute the plan using executor module
    //   d. If autocommit, commit the transaction
    
    // TODO 5: Handle errors at each step:
    //   - Parse errors: return syntax error
    //   - Planning errors: return semantic error  
    //   - Execution errors: return constraint violation or I/O error
    //   - Transaction errors: return rollback needed
    
    // TODO 6: For SELECT statements, format and print results
    // For other statements, print affected row count
    
    return 0; // Success
}
```

**E. Language-Specific Hints (C)**:

1. **Memory Management**: Use a consistent pattern: allocate with `malloc`, check for NULL, free with `free`. Consider using `valgrind` to detect leaks.
2. **Error Handling**: Return 0 for success, negative error codes for failures. Use `errno` for system call errors.
3. **File I/O**: Always check return values of `open`, `read`, `write`, `lseek`. Use `fsync()` for durability guarantees.
4. **Structure Packing**: Use `#pragma pack(1)` or compiler attributes for on-disk structures to avoid padding.
5. **Debugging**: Compile with `-g -Og` for debugging, use `gdb` with breakpoints on critical functions.

**F. Milestone Checkpoint**:

After implementing the basic architecture (even with stub components), you should be able to:

1. **Build the project**: `make` should compile without errors
2. **Run the REPL**: `./sqlite db.test` should start an interactive shell
3. **Execute simple commands**: Typing `.exit` should cleanly exit
4. **Check component linking**: All .o files should link successfully into the executable

**Expected early output**:
```
$ ./sqlite test.db
SQLite Educational Version 0.1
Enter ".help" for usage hints.
sqlite> .exit
$
```

**Debugging if linking fails**:
- **Symptom**: "undefined reference to `parse_statement`"
- **Cause**: Forgot to implement the function or link the object file
- **Fix**: Check `parser.c` has the implementation, and `Makefile` includes `parser.o` in the link command

---


## 4. Data Model

> **Milestone(s):** 1, 2, 3, 4, 5, 6, 7, 9, 10 (Foundational for all components)

This section defines the core in‑memory and on‑disk data structures that form the backbone of our SQLite implementation. Think of the data model as the **DNA of the database** – it encodes how information is represented at every stage, from a raw SQL string typed by a user to the binary bits written on disk. A well‑designed data model ensures that components can communicate efficiently, that data can be persisted reliably, and that the system remains comprehensible as it grows.

The data model is split into two complementary realms:

*   **In‑Memory Structures**: These are transient, high‑level representations used during query processing. They are optimized for fast traversal and manipulation by the C runtime. Examples include tokens from the lexer, nodes of the abstract syntax tree (AST), and in‑memory row buffers.
*   **On‑Disk (Binary) Structures**: These are persistent, byte‑oriented layouts stored in the database file (and auxiliary files like the WAL). They are optimized for compact storage, efficient disk I/O, and crash consistency. Examples include B‑tree page headers, variable‑length record cells, and journal frames.

The bridge between these two realms is **serialization and deserialization** logic, which translates between the in‑memory `Row` structure and the on‑disk `LeafCell` format, or between a `Page` buffer and its constituent `PageHeader` and cell pointers.

### In-Memory Structures

Think of in‑memory structures as the **workspace of a librarian**. When a patron requests a book, the librarian doesn't bring the entire bookshelf to the desk. Instead, they fetch the relevant book, open it to the needed page, and perhaps jot down notes on a notepad. The bookshelf is the on‑disk storage; the book on the desk, the open page, and the notepad are the in‑memory structures – ephemeral, fast, and tailored for the task at hand.

These structures are designed to be **opaque to the storage layer**. The B‑tree component manipulates pages as raw byte buffers; it doesn't need to know about AST nodes. Conversely, the parser produces an AST but has no knowledge of how rows are packed into B‑tree cells. This separation of concerns is key to maintainability.

#### Token (Lexical Analysis)
A `Token` is the smallest meaningful unit of a SQL statement, produced by the lexer (tokenizer). It's akin to a word or punctuation mark in a sentence.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `type` | `TokenType` (enum) | The category of the token (e.g., `TOKEN_KEYWORD_SELECT`, `TOKEN_IDENTIFIER`, `TOKEN_NUMBER_LITERAL`, `TOKEN_STRING_LITERAL`, `TOKEN_OPERATOR_EQ`, `TOKEN_SEMICOLON`). |
| `value` | `char*` (dynamically allocated) | The exact text of the token as it appeared in the source SQL string. For a keyword like `SELECT`, this would be `"SELECT"` (or `"select"` if case‑insensitive). For a string literal `'hello'`, this would be `"hello"` (with escape sequences already processed). |
| `position` | `uint32_t` | The byte offset in the original SQL input string where this token begins. Used for generating meaningful error messages (e.g., "Syntax error near position 15"). |

#### ASTNode (Abstract Syntax Tree)
An `ASTNode` represents a syntactic element of a parsed SQL statement. The tree structure captures the nested relationships defined by the SQL grammar. For example, a `SELECT` statement node has children for the column list, the `FROM` clause, and the `WHERE` clause.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `type` | `ASTType` (enum) | The type of AST node (e.g., `AST_SELECT_STMT`, `AST_INSERT_STMT`, `AST_CREATE_TABLE_STMT`, `AST_EXPRESSION`, `AST_COLUMN_DEF`). |
| `children` | `ASTNode**` (array) | A dynamically allocated array of pointers to child nodes. For a `SELECT` node, children might be: `[column_list_node, from_table_node, where_expr_node]`. For a binary expression like `age > 21`, children are `[left_operand_node, right_operand_node]`. |
| `value` | `union` | A union storing node‑specific data. The active member depends on `type`. Common variants: <br> • `str_val` (`char*`): For identifier names (table `"users"`), string literals, aliases.<br> • `int_val` (`int64_t`): For integer literals.<br> • `float_val` (`double`): For floating‑point literals.<br> • `bool_val` (`bool`): For `TRUE`/`FALSE` keywords.<br> • `op_val` (`Operator`): For comparison/arithmetic operators (`+`, `=`, `AND`).<br> • `data_type` (`DataType`): For column type specifiers (`INTEGER`, `TEXT`). |

> **Design Insight:** Using a union for `value` keeps the AST node memory‑efficient. Each node carries only the data relevant to its role, avoiding a sprawling struct with many unused fields.

#### Row (In‑Memory Record)
A `Row` is the deserialized representation of a table row, used by the query execution engine for filtering, projection, and returning results. It is the "working copy" of a record.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `rowid` | `int64_t` | The primary key value for this row. Serves as the key in the table's B‑tree. Automatically assigned if not provided during `INSERT`. |
| `columns` | `ColumnValue*` (array) | A dynamically allocated array of column values, in the order defined by the table schema. |
| `column_count` | `uint32_t` | The number of columns in this row (i.e., the length of the `columns` array). |

A `ColumnValue` is a tagged union representing a single cell's data:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `type` | `DataType` (enum) | The data type of this column (`DT_INTEGER`, `DT_FLOAT`, `DT_TEXT`, `DT_BLOB`, `DT_NULL`). |
| `value` | `union` | The actual value, interpreted according to `type`: <br> • `int_val` (`int64_t`): For `DT_INTEGER`.<br> • `float_val` (`double`): For `DT_FLOAT`.<br> • `text_val` (`char*`): For `DT_TEXT` (dynamically allocated, null‑terminated string).<br> • `blob_val` (`void*`, `size_t length`): For `DT_BLOB` (pointer to binary data and its length).<br> • (empty): For `DT_NULL`. |

#### Cursor (B‑tree Iterator)
A `Cursor` is an iterator‑like object that provides a position within a B‑tree, enabling traversal and record‑at‑a‑time processing. Think of it as a **bookmark** that remembers your current page and row while scanning a table.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `btree` | `BTree*` | Pointer to the B‑tree object this cursor is associated with. Provides access to the root page and comparison functions. |
| `current_page` | `uint32_t` | The page number of the B‑tree node (leaf or internal) where the cursor is currently positioned. |
| `cell_index` | `uint32_t` | The index within the current page's cell pointer array that points to the cell the cursor is on. For a leaf page, this corresponds to a specific row. For an internal page, this is a separator key. |
| `end_of_table` | `bool` | A flag indicating whether the cursor has advanced beyond the last entry in the table. When `true`, `cursor_next()` should return `false`. |

> **Mental Model:** A `Cursor` is like a player's token on a board game. The `btree` is the game board, `current_page` is which board segment they're on, and `cell_index` is the exact square. Moving the token (`cursor_next`) follows the rules of the game (B‑tree traversal).

#### TransactionState
The `TransactionState` tracks the lifecycle of an active transaction, ensuring ACID properties can be enforced.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `state` | `TransactionStatus` (enum) | Current state: `TRANSACTION_IDLE`, `TRANSACTION_ACTIVE`, `TRANSACTION_COMMITTING`, `TRANSACTION_ROLLBACK`. |
| `journal` | `Journal*` | Pointer to the rollback journal or WAL journal object that records pre‑update page images for atomic rollback. |
| `dirty_pages` | `uint32_t*` (array) | List of page numbers that have been modified in the current transaction and are now dirty in the page cache. Used to efficiently flush or journal pages on commit. |
| `dirty_count` | `uint32_t` | Number of entries in the `dirty_pages` array. |

### On-Disk (Binary) Structures

Think of on‑disk structures as the **blueprint for a shipping container**. The container has a standard size (4096‑byte page), a fixed header area describing its contents, and an internal layout optimized for packing variable‑sized boxes (cells) efficiently. Every byte's position is precisely defined so that any part of the database can be located and interpreted without external context.

These structures are designed with **platform portability** in mind. We use fixed‑size integer types (e.g., `uint32_t`) and define byte order (endianness) to ensure the same database file can be read on different architectures.

#### Page (The Fundamental Unit of I/O)
A `Page` is a fixed‑size (4096‑byte) block of data that is read from or written to the database file as a single unit. The B‑tree storage engine sees pages as opaque buffers; the `Pager` component manages caching and I/O for pages.

In memory, a page is typically represented as a byte array (`uint8_t[PAGE_SIZE]`). Its on‑disk layout is structured as follows:

![Data Model Class Diagram](./diagrams/data-model-diagram.svg)

#### PageHeader (Every Page Starts With This)
The first few bytes of every page are a header that describes the page's type and internal layout. This allows the B‑tree traversal code to quickly determine how to interpret the rest of the page.

| Field Offset (Bytes) | Field Name | Type (in‑memory) | Description |
| :--- | :--- | :--- | :--- |
| 0 | `page_type` | `uint8_t` | Type of page: `PAGE_TYPE_INTERNAL` (0x05), `PAGE_TYPE_LEAF` (0x0D), `PAGE_TYPE_FREELIST` (0x0A). SQLite uses specific values for b‑tree pages; we adopt similar conventions. |
| 1 | `freeblock_start` | `uint16_t` | Offset within the page (from page start) to the first free block on the page's freeblock list. Used for managing space fragmentation. 0 indicates no freeblock. |
| 3 | `cell_count` | `uint16_t` | Number of cells (key‑value pairs) stored on this page. |
| 5 | `cell_content_start` | `uint16_t` | Offset within the page where the cell content area begins (i.e., the first byte past the last cell's data). Grows upward from the end of the page. |
| 7 | `fragmented_free_bytes` | `uint8_t` | Number of fragmented free bytes within the cell content area (due to deletions of variable‑length records). |
| 8 | `rightmost_child` | `uint32_t` | **For internal pages only**: The page number of the right‑most child pointer (for keys greater than all separator keys). For leaf pages, this field is unused (reserved). |

> **Note:** All multi‑byte integer fields in the on‑disk header are stored in **big‑endian (network) byte order** to ensure portability. The in‑memory `PageHeader` struct uses the native byte order; conversion happens during serialization/deserialization.

#### Cell Pointer Array
Immediately following the page header (at byte offset 12 for leaf pages, 16 for internal pages to account for the `rightmost_child`), there is an array of 2‑byte cell pointer offsets.

*   Each cell pointer is a `uint16_t` (big‑endian) giving the offset from the start of the page to where that cell's data begins.
*   The array has `cell_count` entries, sorted in **ascending order by the cell's key**. This enables binary search within the page.
*   Cell pointers grow **downward** from the end of the header toward the cell content area.

#### Cell Content Area
The cell content area starts at the `cell_content_start` offset (typically near the end of the page) and grows **upward** towards the cell pointer array. Each cell is stored as a contiguous byte sequence at the offset specified by its cell pointer.

There are two fundamental cell types, distinguished by the page type:

##### LeafCell (Stores Row Data on Leaf Pages)
A LeafCell stores a key (`rowid`) and the serialized row data (values for all columns).

| Field | Type (on‑disk) | Description |
| :--- | :--- | :--- |
| `payload_size` | `varint` | Variable‑length integer encoding of the total size of the row data (excluding the `rowid` and this length field). |
| `rowid` | `varint` | The primary key for this row, stored as a variable‑length integer (SQLite's "varint" encoding). |
| `row_data` | `uint8_t[payload_size]` | The serialized row, consisting of a series of column values packed according to the table's schema. |

##### InternalCell (Stores Separator Keys on Internal Pages)
An InternalCell stores a separator key and a child page pointer, used to navigate the B‑tree.

| Field | Type (on‑disk) | Description |
| :--- | :--- | :--- |
| `child_page` | `uint32_t` (big‑endian) | The page number of the left child subtree that contains keys **less than** `key`. |
| `key` | `varint` | The separator key (a `rowid`). All keys in the subtree rooted at `child_page` are less than this `key`. |

> **ADR: Variable‑Length Integer Encoding for Keys and Payloads**
> *   **Context:** We need to store integer values (like `rowid` and payload sizes) that can range from very small (a few bytes) to very large (8‑byte). Using a fixed 8‑byte field for every cell would waste significant space, especially for small keys and short rows.
> *   **Options Considered:**
>     1.  **Fixed‑size integers (e.g., `int64_t`):** Simple to parse, but wastes space for small values.
>     2.  **SQLite‑style "varint":** Encode the integer in 1 to 9 bytes, where the high bit of each byte indicates continuation. Efficient for small values, slightly more complex to encode/decode.
>     3.  **Length‑prefixed string representation:** Convert integer to decimal string, then store length + string. Simple but inefficient for both space and comparison.
> *   **Decision:** Use SQLite‑style **variable‑length integer (varint)** encoding for `rowid` and `payload_size` fields within cells.
> *   **Rationale:** Space efficiency is critical for database performance, as it directly affects the number of pages needed and thus I/O operations. The varint encoding is a proven, compact binary format that maintains lexicographic ordering for positive integers (important for B‑tree key comparisons). The encoding/decoding overhead is minimal compared to the I/O savings.
> *   **Consequences:** We must implement helper functions `varint_encode()` and `varint_decode()`. The B‑tree comparison logic must decode keys to compare them, or we must ensure the encoded form is comparable byte‑for‑byte (which SQLite's encoding does for positive integers).

#### Journal/WAL Frame (For Crash Recovery)
When using a rollback journal or Write‑Ahead Log, modifications are recorded as frames. Each frame corresponds to one database page before or after a change.

**Rollback Journal Frame (for UNDO):**
| Field | Type (on‑disk) | Description |
| :--- | :--- | :--- |
| `page_number` | `uint32_t` (big‑endian) | The database page number being modified. |
| `original_page_image` | `uint8_t[PAGE_SIZE]` | The complete content of the page **before** the modification. On rollback, this image is copied back to the main database file. |

**WAL Frame (for REDO):**
| Field | Type (on‑disk) | Description |
| :--- | :--- | Description |
| `page_number` | `uint32_t` (big‑endian) | The database page number being modified. |
| `db_size_after_commit` | `uint32_t` (big‑endian) | The size of the database (in pages) after the transaction containing this frame commits. Used to prevent reading past the end of the file. |
| `salt` | `uint32_t[2]` (big‑endian) | Random values changed on each checkpoint; used to detect mismatches between WAL and main DB. |
| `checksum` | `uint32_t[2]` (big‑endian) | Checksum of the frame header and page data for integrity detection. |
| `page_image` | `uint8_t[PAGE_SIZE]` | The complete content of the page **after** the modification. On recovery, this image is copied to the main database file. |

### Common Pitfalls

⚠️ **Pitfall: Ignoring Byte Order (Endianness) in On‑Disk Structures**
*   **Description:** Writing multi‑byte integers (like `uint32_t page_number`) directly from memory to disk without conversion. The bytes are written in the CPU's native byte order (little‑endian on x86).
*   **Why it's wrong:** If the database file is moved to a machine with a different byte order (e.g., a big‑endian ARM server), all integer fields will be read incorrectly, corrupting the B‑tree structure.
*   **Fix:** Define a canonical byte order (big‑endian is standard for network‑portable formats). Use helper functions `store_big_endian_u32()` and `load_big_endian_u32()` when reading/writing header fields and cell pointers.

⚠️ **Pitfall: Forgetting to Handle NULL in ColumnValue Union**
*   **Description:** The `ColumnValue` union has no explicit field for a `NULL` value, only a `type` tag `DT_NULL`.
*   **Why it's wrong:** If code attempts to access, say, `int_val` when `type` is `DT_NULL`, it will read uninitialized memory, causing undefined behavior.
*   **Fix:** Always check `type == DT_NULL` before accessing any union member. Provide a safe accessor function like `column_value_as_int64(const ColumnValue* cv, int64_t fallback)` that returns the fallback for `DT_NULL`.

⚠️ **Pitfall: Not Bounding Token/Identifier Lengths**
*   **Description:** The lexer reads identifiers and string literals into dynamically allocated `char*` without imposing a maximum length.
*   **Why it's wrong:** A malicious or malformed input could contain an extremely long identifier (e.g., 1GB), causing the lexer to exhaust memory.
*   **Fix:** Define a sensible maximum (e.g., `MAX_IDENTIFIER_LEN = 255`) and truncate or error after that limit. This also simplifies storage, as column names can be stored in fixed‑length fields in the system catalog.

⚠️ **Pitfall: Storing Pointers in On‑Disk Structures**
*   **Description:** Accidentally writing a memory address (pointer) into a page buffer that gets persisted to disk.
*   **Why it's wrong:** Pointers are only valid for the current process's address space. When the database is reopened, those addresses are meaningless and will cause crashes or corruption.
*   **Fix:** Never store `char*` or `void*` directly. For variable‑length data (like `TEXT`), store the length followed by the bytes. In‑memory, you can have a `Row` with pointers to offsets within a page buffer, but those pointers must be recomputed after each page load.

### Implementation Guidance

#### A. Technology Recommendations Table
| Component | Simple Option (for learning) | Advanced Option (for robustness) |
| :--- | :--- | :--- |
| **Data Serialization** | Manual byte‑by‑byte writing/reading with helper functions for endianness. | Use a mini‑library like `sqlite3.c`'s `putVarint()`/`getVarint()` and `put4byte()`/`get4byte()`. |
| **Memory Management** | Manual `malloc()`/`free()` with clear ownership conventions. | Use arena/region allocators for AST and query execution to batch frees. |
| **String Handling** | Null‑terminated `char*` with `strdup()`/`free()`. | Length‑prefixed strings (`struct { size_t len; char data[]; }`) to support embedded nulls in `BLOB`. |

#### B. Recommended File/Module Structure
```
build-your-own-sqlite/
├── src/
│   ├── main.c                      # REPL entry point
│   ├── include/
│   │   ├── data_model.h            # Central header for all structures below
│   │   ├── token.h                 # Token and TokenType definitions
│   │   ├── ast.h                   # ASTNode and ASTType definitions
│   │   ├── row.h                   # Row and ColumnValue definitions
│   │   ├── cursor.h                # Cursor definition
│   │   ├── transaction.h           # TransactionState definition
│   │   ├── page.h                  # Page, PageHeader, LeafCell, InternalCell definitions
│   │   └── journal.h               # Journal frame definitions
│   ├── parser/                     # Milestone 1, 2
│   │   ├── tokenizer.c
│   │   └── parser.c
│   ├── storage/                    # Milestone 3, 4
│   │   ├── pager.c                 # Page cache
│   │   ├── btree.c                 # B-tree operations
│   │   └── serialization.c         # Row <-> Cell conversion, varint helpers
│   ├── query/                      # Milestone 5, 6, 7, 8
│   │   ├── executor.c
│   │   └── planner.c
│   └── transaction/                # Milestone 9, 10
│       ├── journal.c
│       └── wal.c
└── test/
    ├── test_tokenizer.c
    ├── test_parser.c
    └── ...
```

#### C. Infrastructure Starter Code
**`src/include/data_model.h` (foundational types)**
```c
#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <stdint.h>
#include <stdbool.h>

#define PAGE_SIZE 4096
#define TABLE_MAX_PAGES 100
#define MAX_IDENTIFIER_LEN 255

// ---------- Token (Milestone 1) ----------
typedef enum {
    TOKEN_KEYWORD_SELECT,
    TOKEN_KEYWORD_FROM,
    TOKEN_KEYWORD_WHERE,
    TOKEN_KEYWORD_INSERT,
    // ... other keywords
    TOKEN_IDENTIFIER,
    TOKEN_STRING_LITERAL,
    TOKEN_NUMBER_LITERAL,
    TOKEN_OPERATOR_EQ,
    TOKEN_OPERATOR_NE,
    TOKEN_OPERATOR_LT,
    TOKEN_OPERATOR_GT,
    TOKEN_OPERATOR_AND,
    TOKEN_OPERATOR_OR,
    TOKEN_COMMA,
    TOKEN_SEMICOLON,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_EOF,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char* value;        // Dynamically allocated, must be freed
    uint32_t position;
} Token;

// ---------- AST (Milestone 2) ----------
typedef enum {
    AST_SELECT_STMT,
    AST_INSERT_STMT,
    AST_CREATE_TABLE_STMT,
    AST_EXPRESSION,
    AST_COLUMN_DEF,
    // ... other node types
} ASTType;

typedef enum {
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_GT,
    OP_AND,
    OP_OR,
    OP_PLUS,
    OP_MINUS
} Operator;

typedef enum {
    DT_INTEGER,
    DT_FLOAT,
    DT_TEXT,
    DT_BLOB,
    DT_NULL
} DataType;

typedef struct ASTNode {
    ASTType type;
    struct ASTNode** children;  // Array of pointers, NULL-terminated or with separate count
    union {
        char* str_val;
        int64_t int_val;
        double float_val;
        bool bool_val;
        Operator op_val;
        DataType data_type;
    } value;
} ASTNode;

// ---------- Row (Milestone 4, 5) ----------
typedef struct {
    DataType type;
    union {
        int64_t int_val;
        double float_val;
        char* text_val;     // Dynamically allocated, null-terminated
        struct {
            void* data;
            size_t length;
        } blob_val;
    } value;
} ColumnValue;

typedef struct {
    int64_t rowid;
    ColumnValue* columns;
    uint32_t column_count;
} Row;

// ---------- Cursor (Milestone 5) ----------
typedef struct BTree BTree;  // Forward declaration

typedef struct {
    BTree* btree;
    uint32_t current_page;
    uint32_t cell_index;
    bool end_of_table;
} Cursor;

// ---------- TransactionState (Milestone 9) ----------
typedef enum {
    TRANSACTION_IDLE,
    TRANSACTION_ACTIVE,
    TRANSACTION_COMMITTING,
    TRANSACTION_ROLLBACK
} TransactionStatus;

typedef struct Journal Journal;  // Forward declaration

typedef struct {
    TransactionStatus state;
    Journal* journal;
    uint32_t* dirty_pages;
    uint32_t dirty_count;
} TransactionState;

// ---------- Page (Milestone 3) ----------
typedef enum {
    PAGE_TYPE_INTERNAL = 0x05,
    PAGE_TYPE_LEAF = 0x0D,
    PAGE_TYPE_FREELIST = 0x0A
} PageType;

typedef struct {
    uint8_t page_type;           // 0
    uint16_t freeblock_start;    // 1
    uint16_t cell_count;         // 3
    uint16_t cell_content_start; // 5
    uint8_t fragmented_free_bytes; // 7
    uint32_t rightmost_child;    // 8 (internal pages only)
} PageHeader;

// In-memory representation of a page
typedef struct {
    PageHeader header;
    uint8_t data[PAGE_SIZE - sizeof(PageHeader)]; // Flexible array member alternative
} Page;

#endif // DATA_MODEL_H
```

**`src/storage/serialization.c` (varint and endian helpers)**
```c
#include "data_model.h"
#include <arpa/inet.h>  // For htonl/ntohl (or implement manually for portability)

// Encode a 64-bit unsigned integer into varint encoding.
// Returns number of bytes written.
int varint_encode(uint64_t value, uint8_t* out) {
    int i = 0;
    while (value > 0x7F) {
        out[i++] = (uint8_t)((value & 0x7F) | 0x80);
        value >>= 7;
    }
    out[i++] = (uint8_t)(value & 0x7F);
    return i;
}

// Decode a varint from memory. Returns number of bytes read.
int varint_decode(const uint8_t* in, uint64_t* out) {
    uint64_t result = 0;
    int shift = 0;
    int i = 0;
    uint8_t byte;
    do {
        byte = in[i++];
        result |= (uint64_t)(byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    *out = result;
    return i;
}

// Store a 32-bit integer in big-endian order.
void store_big_endian_u32(uint8_t* buf, uint32_t value) {
    buf[0] = (value >> 24) & 0xFF;
    buf[1] = (value >> 16) & 0xFF;
    buf[2] = (value >> 8) & 0xFF;
    buf[3] = value & 0xFF;
}

// Load a 32-bit integer from big-endian order.
uint32_t load_big_endian_u32(const uint8_t* buf) {
    return ((uint32_t)buf[0] << 24) |
           ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8) |
           (uint32_t)buf[3];
}
```

#### D. Core Logic Skeleton Code
**`src/parser/tokenizer.c` (next_token function)**
```c
Token* next_token(const char* input, uint32_t* pos) {
    Token* token = malloc(sizeof(Token));
    if (!token) return NULL;
    token->value = NULL;

    // TODO 1: Skip whitespace and comments, updating *pos.
    // TODO 2: If at end of input, set type = TOKEN_EOF and return.
    // TODO 3: Check for single‑character punctuation (',', ';', '(', ')'), set appropriate type.
    // TODO 4: Check for multi‑character operators ('=', '!=', '<', '>', 'AND', 'OR').
    //         Note: SQL keywords are case‑insensitive; convert to uppercase for comparison.
    // TODO 5: If it starts with a letter or underscore, read an identifier/keyword.
    //         Determine if it's a reserved keyword (set TOKEN_KEYWORD_*) or a regular identifier.
    // TODO 6: If it starts with a digit, read a numeric literal (integer or float).
    // TODO 7: If it starts with a single quote, read a string literal until closing quote.
    //         Handle escaped quotes (two single quotes '').
    // TODO 8: For identifiers and literals, dynamically allocate token->value and copy the text.
    // TODO 9: Set token->position to the start offset of this token.
    // TODO 10: If none of the above, set type = TOKEN_ERROR and optionally store the offending character.

    return token;
}
```

**`src/storage/btree.c` (deserialize a leaf cell into a Row)**
```c
// Given a pointer to the start of a LeafCell within a page buffer,
// deserialize it into a Row structure according to the given table schema.
// Returns true on success, false on corruption.
bool deserialize_leaf_cell(const uint8_t* cell_start, const TableSchema* schema, Row* row) {
    const uint8_t* p = cell_start;
    uint64_t payload_size;
    uint64_t rowid;

    // TODO 1: Decode varint payload_size from p. Advance p by bytes read.
    // TODO 2: Decode varint rowid from p. Advance p.
    // TODO 3: Set row->rowid = (int64_t)rowid.
    // TODO 4: Allocate row->columns = malloc(schema->column_count * sizeof(ColumnValue)).
    // TODO 5: For each column in schema order:
    //   a) Determine column type from schema.
    //   b) Based on type, read appropriate bytes from p:
    //        - INTEGER: varint -> column.value.int_val
    //        - FLOAT: 8‑byte IEEE‑754 -> column.value.float_val
    //        - TEXT: varint length N, then N bytes (allocate + null‑terminate) -> column.value.text_val
    //        - BLOB: varint length N, then N bytes (allocate) -> column.value.blob_val.data/.length
    //        - NULL: skip, set column.type = DT_NULL
    //   c) Advance p accordingly.
    // TODO 6: Verify that total bytes read from cell_start equals payload_size + bytes for rowid.
    // TODO 7: Set row->column_count = schema->column_count.
    // TODO 8: Return true. On any error, free allocated memory and return false.
}
```

#### E. Language-Specific Hints (C)
*   **Memory Ownership:** Document which component owns each pointer. For example, `Token->value` is allocated by the tokenizer and must be freed by the parser after constructing the AST. The AST nodes own their children and `str_val` strings.
*   **Zero‑Length Arrays:** For the in‑memory `Page` struct, consider using a flexible array member `uint8_t data[];` and allocating `sizeof(PageHeader) + PAGE_SIZE`. This avoids wasting space for the `data` array pointer.
*   **Alignment:** The `PageHeader` struct may have padding between fields. Use `pragma pack(1)` or compiler‑specific attributes to ensure it matches the on‑disk layout exactly, or serialize field‑by‑field.
*   **Error Propagation:** Use a consistent error‑handling pattern (e.g., returning `bool` with an out‑parameter for error details, or returning a nullable pointer with `NULL` indicating error).

#### F. Milestone Checkpoint (Data Model Validation)
After implementing the foundational data structures, run a simple test to ensure they can round‑trip:

```bash
# Compile and run a small test that:
# 1. Creates a Row with integer and text columns.
# 2. Serializes it into a mock LeafCell buffer.
# 3. Deserializes the buffer back into a Row.
# 4. Compares original and deserialized values.
gcc -o test_serialization src/storage/serialization.c test/test_serialization.c
./test_serialization
```
**Expected Output:** `All serialization tests passed.`

**Signs of Trouble:**
*   **Segmentation fault:** Likely a NULL pointer dereference. Check that all allocations succeed and pointers are initialized.
*   **Incorrect values:** Byte‑order mismatch or varint encoding/decoding bug. Use a hex dump to inspect the buffer.
*   **Memory leak:** Use `valgrind ./test_serialization` to detect unfreed allocations.


## 5.1 Component: SQL Parser (Milestone 1, 2)

> **Milestone(s):** 1 (SQL Tokenizer), 2 (SQL Parser)

The SQL Parser component serves as the gateway between human-readable SQL statements and the internal operations of our database engine. It transforms unstructured text into a structured, machine-processable representation that accurately captures the intent of the SQL command while rejecting syntactically invalid input.

### Mental Model: The Language Translator

Imagine you're translating instructions between two specialists: a **business analyst** who speaks human languages (SQL) and a **machine operator** who only understands precise internal commands. The business analyst writes requests like "Show me all customers from New York who spent over $100." Your job is to:

1. **Break it into meaningful chunks** (tokenization): Recognize "Show me" as a request for data, "customers" as a table name, "from New York" as a location filter, etc.
2. **Understand the sentence structure** (parsing): Recognize that this is a data retrieval request (`SELECT`), identify what data to retrieve (`*` for all columns), which table to get it from (`customers`), and what conditions to apply (`location = 'New York' AND amount > 100`).
3. **Create a precise work order** (AST generation): Build a structured document that specifies exactly: operation type = SELECT, target table = customers, return columns = all, filters = [location equals "New York", amount greater than 100].

This component doesn't execute anything — it only translates. Like a human translator who might flag ambiguous phrases ("Do you mean current customers or all historical ones?"), the parser validates syntax and structure but doesn't check if the table exists or if columns are properly typed (that's the execution engine's job).

### Interface and Public API

The parser exposes a minimal API: one primary function that takes SQL text and returns an Abstract Syntax Tree (AST), plus supporting structures for tokens and AST nodes. All parsing errors are reported through structured error codes rather than printing to stdout/stderr.

**Primary Function:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `parse_statement` | `const char* sql` (SQL string) | `AST*` (pointer to AST root) | Parses a complete SQL statement (SELECT, INSERT, CREATE TABLE, etc.) into an abstract syntax tree. Returns `NULL` on parse error. Sets global error state accessible via `get_parser_error()`. |

**Supporting Data Structures:**

| Type | Fields | Description |
|------|--------|-------------|
| `Token` | `type TokenType`<br>`value char*`<br>`position uint32_t` | Represents a lexical token from SQL input. `type` categorizes the token (keyword, identifier, literal, etc.). `value` holds the raw text (for identifiers/literals) or `NULL` for fixed tokens. `position` tracks character offset in source for error reporting. |
| `ASTNode` | `type ASTType`<br>`children ASTNode**`<br>`value union` | The fundamental building block of the AST. `type` indicates what kind of node (SELECT_STMT, COLUMN_REF, BINARY_OP, etc.). `children` is a dynamic array of child nodes (NULL-terminated). `value` union holds node-specific data (string, number, operator, etc.). |
| `AST` | `type ASTType`<br>`children ASTNode**` | Root container for a parsed statement. `type` indicates statement type (SELECT, INSERT, CREATE). `children` contains statement-specific top-level nodes (e.g., for SELECT: column list, table list, where clause). |

**Token Types (partial enumeration):**

| Constant | Value | Description |
|----------|-------|-------------|
| `TOKEN_SELECT` | 1 | `SELECT` keyword |
| `TOKEN_FROM` | 2 | `FROM` keyword |
| `TOKEN_WHERE` | 3 | `WHERE` keyword |
| `TOKEN_IDENTIFIER` | 20 | Table/column names (case-preserving) |
| `TOKEN_STRING` | 21 | String literal with quotes removed |
| `TOKEN_NUMBER` | 22 | Numeric literal |
| `TOKEN_EQUAL` | 30 | `=` operator |
| `TOKEN_LESS` | 31 | `<` operator |
| `TOKEN_GREATER` | 32 | `>` operator |
| `TOKEN_AND` | 40 | `AND` logical operator |
| `TOKEN_OR` | 41 | `OR` logical operator |
| `TOKEN_LPAREN` | 50 | `(` punctuation |
| `TOKEN_RPAREN` | 51 | `)` punctuation |
| `TOKEN_COMMA` | 52 | `,` punctuation |
| `TOKEN_SEMICOLON` | 53 | `;` statement terminator |
| `TOKEN_EOF` | 100 | End of input |

**AST Node Types (partial enumeration):**

| Constant | Description | Value Union Field |
|----------|-------------|-------------------|
| `AST_SELECT` | SELECT statement root | None |
| `AST_COLUMN_LIST` | List of column references | None |
| `AST_COLUMN_REF` | Reference to column | `str_val` (column name) |
| `AST_TABLE_REF` | Reference to table | `str_val` (table name) |
| `AST_WHERE_CLAUSE` | WHERE clause container | None |
| `AST_BINARY_OP` | Binary operator expression | `op_val` (operator type) |
| `AST_LITERAL` | Constant value | `int_val`, `float_val`, or `str_val` |
| `AST_CREATE_TABLE` | CREATE TABLE statement | `str_val` (table name) |
| `AST_COLUMN_DEF` | Column definition | `str_val` (column name), `data_type` (type) |

**Error Handling Functions:**

| Function | Returns | Description |
|----------|---------|-------------|
| `get_parser_error()` | `const char*` | Returns the most recent parser error message (or NULL if no error). Reset by successful `parse_statement` call. |
| `get_parser_error_position()` | `uint32_t` | Returns character offset in input where error occurred. |

The component follows a "parse-one-statement" model, suitable for our interactive REPL or single-statement execution. Multiple statements separated by semicolons require repeated calls with advancing input pointers.

### Internal Behavior: Lexing and Parsing

The parser operates in two sequential phases: **lexical analysis** (breaking text into tokens) followed by **syntax analysis** (arranging tokens into hierarchical structure). This separation simplifies implementation and enables better error reporting.

#### Phase 1: Lexical Analysis (Tokenizer)

The lexer scans the SQL string character by character, grouping characters into tokens based on syntactic categories. It implements a **finite state machine** with states for identifiers, numbers, strings, and operators.

**Tokenizer Algorithm:**
1. **Initialize**: Set current position `pos = 0`, create empty token list.
2. **Skip whitespace**: Advance past spaces, tabs, newlines (these are never tokens).
3. **Check for EOF**: If at end of string, emit `TOKEN_EOF` and return.
4. **Classify by first character**:
   - **Letter or underscore**: Enter identifier state. Continue reading while next character is letter, digit, or underscore. Check if matches SQL keyword (case-insensitively). If keyword, emit corresponding token; otherwise emit `TOKEN_IDENTIFIER` with the raw text.
   - **Digit**: Enter number state. Read integer part, then if decimal point followed by digits, read fractional part. Convert to internal numeric representation, emit `TOKEN_NUMBER` with numeric value.
   - **Single quote (`'`)**: Enter string literal state. Read characters until closing single quote. Handle escaped quotes (`''` → single quote in string). Store string without surrounding quotes, emit `TOKEN_STRING`.
   - **Operator characters** (`=`, `<`, `>`, `!`, etc.): Check for multi-character operators (`!=`, `<=`, `>=`). Emit corresponding operator token.
   - **Punctuation** (`(`, `)`, `,`, `;`): Emit corresponding single-character token.
   - **Anything else**: Report lexical error "unexpected character", abort.
5. **Repeat** from step 2 until EOF.

**Key Design Points:**
- Keywords are recognized **case-insensitively** (`SELECT`, `Select`, `select` all produce `TOKEN_SELECT`).
- Identifiers preserve original case but comparisons are case-insensitive per SQL standard.
- String literals support only single quotes; escaped quotes are handled during lexing.
- Numeric literals support integers (`123`) and decimals (`123.45`), but not scientific notation initially.
- The lexer doesn't validate SQL semantics (e.g., doesn't check if table exists).

#### Phase 2: Syntax Analysis (Parser)

The parser consumes the token stream using **recursive descent parsing**, a top-down method where each grammar rule corresponds to a parsing function. It constructs the AST bottom-up as functions return.

**Parsing Algorithm for SELECT Statement:**
```
parse_statement() → AST*:
  1. Peek at first token
  2. If token is TOKEN_SELECT: return parse_select_statement()
  3. If token is TOKEN_INSERT: return parse_insert_statement()
  4. If token is TOKEN_CREATE: return parse_create_statement()
  5. Otherwise: report error "unexpected token, expected SELECT, INSERT, or CREATE"

parse_select_statement() → AST*:
  1. Consume TOKEN_SELECT
  2. Parse column list using parse_column_list()
  3. Consume TOKEN_FROM
  4. Parse table reference using parse_table_reference()
  5. If next token is TOKEN_WHERE:
      a. Consume TOKEN_WHERE
      b. Parse expression using parse_expression() as WHERE clause
  6. Build AST_SELECT node with column list, table, and optional WHERE clause
  7. Return node

parse_expression() → ASTNode*:
  Implements operator precedence using Pratt parsing or precedence climbing:
  1. Parse primary (literal, identifier, parenthesized expression)
  2. While next token is operator with higher or equal precedence than current:
      a. Get operator precedence from precedence table
      b. Parse right-hand side expression
      c. Create AST_BINARY_OP node
  3. Return root of expression tree
```

**Expression Precedence Table (highest to lowest):**

| Operator(s) | Precedence | Associativity |
|-------------|------------|---------------|
| `( )` (parentheses) | 7 | N/A |
| `-` (unary) | 6 | Right |
| `*`, `/` | 5 | Left |
| `+`, `-` (binary) | 4 | Left |
| `=`, `<`, `>`, `<=`, `>=`, `!=` | 3 | Left |
| `AND` | 2 | Left |
| `OR` | 1 | Left |

**Parsing Example Walkthrough:**
For the SQL: `SELECT name, age FROM users WHERE age > 18 AND active = 1`

1. **Lexer produces tokens**: `TOKEN_SELECT`, `TOKEN_IDENTIFIER("name")`, `TOKEN_COMMA`, `TOKEN_IDENTIFIER("age")`, `TOKEN_FROM`, `TOKEN_IDENTIFIER("users")`, `TOKEN_WHERE`, `TOKEN_IDENTIFIER("age")`, `TOKEN_GREATER`, `TOKEN_NUMBER(18)`, `TOKEN_AND`, `TOKEN_IDENTIFIER("active")`, `TOKEN_EQUAL`, `TOKEN_NUMBER(1)`, `TOKEN_EOF`

2. **Parser builds AST**:
   - Root: `AST_SELECT`
     - Column list: `AST_COLUMN_LIST` with two `AST_COLUMN_REF` nodes ("name", "age")
     - Table: `AST_TABLE_REF` ("users")
     - Where clause: `AST_BINARY_OP` (`AND`)
       - Left: `AST_BINARY_OP` (`>`)
         - Left: `AST_COLUMN_REF` ("age")
         - Right: `AST_LITERAL` (18)
       - Right: `AST_BINARY_OP` (`=`)
         - Left: `AST_COLUMN_REF` ("active")
         - Right: `AST_LITERAL` (1)

The resulting AST provides a complete, unambiguous representation that the query execution engine can process without re-parsing the original SQL.

### ADR: Recursive Descent vs. Parser Generator

> **Decision: Manual Recursive Descent Parser**
>
> **Context**: We need a parser for a well-defined subset of SQL that's maintainable, debuggable, and has no external dependencies. The parser is a core learning component where understanding the parsing process is educational. We expect to handle 10-15 statement types and simple expressions.
>
> **Options Considered**:
> 1. **Manual recursive descent parser** (hand-written functions for each grammar rule)
> 2. **Parser generator** (ANTLR, yacc/bison, lemon) with grammar file
> 3. **Parser combinator library** (only applicable in functional languages)
>
> **Decision**: Implement a hand-written recursive descent parser in C.
>
> **Rationale**:
> - **Educational value**: Writing the parser manually teaches how parsing works, precedence handling, and AST construction — core compiler concepts.
> - **Debuggability**: When a parsing bug occurs, we can step through C functions rather than debugging generated code or grammar conflicts.
> - **No build dependencies**: Avoids complex build toolchains for parser generation, keeping the project self-contained.
> - **Good fit for SQL subset**: SQL has relatively simple, non-ambiguous grammar for our target statements; recursive descent handles this cleanly.
> - **Performance**: Hand-written parsers can be optimized for our specific needs without generator overhead.
>
> **Consequences**:
> - **Maintenance burden**: Adding new SQL features requires manually updating parser functions.
> - **Error reporting**: We must implement our own error recovery and messaging.
> - **Grammar changes**: Modifying operator precedence requires code changes rather than grammar file edits.
> - **Limited scalability**: If we expand to full SQL, the parser could become unwieldy (but our scope is limited).

**Comparison Table:**

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Recursive Descent** | Full control, debuggable, no dependencies, educational | Manual maintenance, error recovery harder | **Chosen** - Best for learning and our limited SQL subset |
| **Parser Generator** | Grammar as declarative spec, automatic error recovery, handles complex grammars | Build dependency, generated code harder to debug, less educational | Too much magic for learning; debugging generated C code is difficult |
| **Parser Combinators** | Elegant, compositional, great error messages | Not available in C, requires functional language | Language mismatch (project uses C) |

### Common Pitfalls

⚠️ **Pitfall 1: Case-Sensitive Keyword Matching**
- **Description**: Checking keywords with exact case match (`strcmp("SELECT", token_text)`) instead of case-insensitive comparison.
- **Why it's wrong**: SQL keywords are case-insensitive per standard. `SeLeCt` should parse as a SELECT statement.
- **Fix**: Use case-insensitive comparison (`strcasecmp` on POSIX, `_stricmp` on Windows) or convert to uppercase before comparing.

⚠️ **Pitfall 2: Not Handling Escaped Quotes in Strings**
- **Description**: Treating `'it''s'` as a string ending at the second single quote, producing `it` instead of `it's`.
- **Why it's wrong**: SQL uses doubled quotes for escaping: `'it''s'` → `it's`. Also affects empty strings: `''''` → `'`.
- **Fix**: In string lexing state, when encountering a single quote, peek at next character. If it's another single quote, consume both and add one quote to string value.

⚠️ **Pitfall 3: Incorrect Operator Precedence**
- **Description**: Parsing `a = 1 AND b = 2 OR c = 3` as `(a = 1 AND b = 2) OR c = 3` instead of `a = 1 AND (b = 2 OR c = 3)` because AND and OR have same precedence.
- **Why it's wrong**: SQL gives AND higher precedence than OR. The expression should parse as `(a = 1) AND (b = 2 OR c = 3)`.
- **Fix**: Implement proper precedence levels (see table above) using precedence climbing or explicit precedence values in parser functions.

⚠️ **Pitfall 4: Left Recursion in Grammar**
- **Description**: Writing parsing function that calls itself without consuming tokens first, causing infinite recursion:
  ```c
  ASTNode* parse_expression() {
      ASTNode* left = parse_expression();  // Infinite recursion!
      // ... parse operator ...
  }
  ```
- **Why it's wrong**: Direct left recursion causes stack overflow. Grammar like `expression → expression AND expression` needs transformation.
- **Fix**: Rewrite using iteration or precedence climbing:
  ```c
  ASTNode* parse_expression() {
      ASTNode* left = parse_primary();
      while (is_operator(next_token)) {
          Token op = consume_token();
          ASTNode* right = parse_primary();
          left = create_binary_op(left, op, right);
      }
      return left;
  }
  ```

⚠️ **Pitfall 5: Not Tracking Token Positions for Error Reporting**
- **Description**: Reporting "syntax error" without indicating where in the SQL string the error occurred.
- **Why it's wrong**: Users can't fix errors without knowing location, especially in longer SQL statements.
- **Fix**: Store character offset in each `Token` structure. When parsing fails, report error with line/column or character position.

⚠️ **Pitfall 6: Memory Leaks on Parse Errors**
- **Description**: Allocating AST nodes during parsing but not freeing them when syntax error occurs mid-parse.
- **Why it's wrong**: Memory leaks accumulate with invalid SQL inputs in REPL mode.
- **Fix**: Use arena allocator for AST nodes that can be freed all at once, or implement rollback cleanup in each parsing function.

### Implementation Guidance

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Lexer | Hand-written finite state machine | Table-driven lexer with state transition table |
| Parser | Recursive descent with precedence climbing | Pratt parser with operator precedence tables |
| Memory Management | Arena allocator for AST nodes | Reference counting with pool allocator |
| Error Reporting | Simple error string with position | Detailed error codes with suggestion hints |

**B. Recommended File/Module Structure**

```
build-your-own-sqlite/
├── src/
│   ├── parser/
│   │   ├── lexer.c           # Tokenization implementation
│   │   ├── lexer.h           # Token types, Token struct, lexer functions
│   │   ├── parser.c          # Recursive descent parsing
│   │   ├── parser.h          # AST types, parse_statement() declaration
│   │   ├── ast.c             # AST construction/manipulation functions
│   │   └── ast.h             # AST node structures, visitor pattern if used
│   ├── database.c            # Calls parse_statement()
│   └── main.c                # REPL that calls database_execute()
├── include/
│   └── parser.h              # Public parser API (forward from src/parser/parser.h)
└── tests/
    └── test_parser.c         # Parser unit tests
```

**C. Infrastructure Starter Code**

Complete lexer implementation with error handling:

```c
/* lexer.h */
#ifndef LEXER_H
#define LEXER_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    TOKEN_SELECT = 1,
    TOKEN_FROM,
    TOKEN_WHERE,
    TOKEN_INSERT,
    TOKEN_INTO,
    TOKEN_VALUES,
    TOKEN_CREATE,
    TOKEN_TABLE,
    TOKEN_IDENTIFIER,
    TOKEN_STRING,
    TOKEN_NUMBER,
    TOKEN_EQUAL,
    TOKEN_LESS,
    TOKEN_GREATER,
    TOKEN_LESS_EQUAL,
    TOKEN_GREATER_EQUAL,
    TOKEN_NOT_EQUAL,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_COMMA,
    TOKEN_SEMICOLON,
    TOKEN_EOF,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char* value;      /* For identifiers, strings, numbers (dynamically allocated) */
    uint32_t position;/* Character offset in source where token starts */
} Token;

typedef struct {
    const char* source;
    uint32_t position;
    uint32_t length;
    char error_msg[256];
} Lexer;

/* Public API */
Lexer* lexer_create(const char* source);
void lexer_destroy(Lexer* lexer);
Token* lexer_next_token(Lexer* lexer);
const char* lexer_get_error(Lexer* lexer);

#endif /* LEXER_H */
```

```c
/* lexer.c */
#include "lexer.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define KEYWORD_COUNT 10

typedef struct {
    const char* keyword;
    TokenType token_type;
} Keyword;

static Keyword keywords[KEYWORD_COUNT] = {
    {"SELECT", TOKEN_SELECT},
    {"FROM", TOKEN_FROM},
    {"WHERE", TOKEN_WHERE},
    {"INSERT", TOKEN_INSERT},
    {"INTO", TOKEN_INTO},
    {"VALUES", TOKEN_VALUES},
    {"CREATE", TOKEN_CREATE},
    {"TABLE", TOKEN_TABLE},
    {"AND", TOKEN_AND},
    {"OR", TOKEN_OR}
};

Lexer* lexer_create(const char* source) {
    Lexer* lexer = malloc(sizeof(Lexer));
    if (!lexer) return NULL;
    
    lexer->source = source;
    lexer->position = 0;
    lexer->length = strlen(source);
    lexer->error_msg[0] = '\0';
    
    return lexer;
}

void lexer_destroy(Lexer* lexer) {
    free(lexer);
}

static bool is_keyword(const char* text, TokenType* token_type) {
    char upper[256];
    size_t i = 0;
    while (text[i] && i < 255) {
        upper[i] = toupper(text[i]);
        i++;
    }
    upper[i] = '\0';
    
    for (int j = 0; j < KEYWORD_COUNT; j++) {
        if (strcmp(upper, keywords[j].keyword) == 0) {
            *token_type = keywords[j].token_type;
            return true;
        }
    }
    return false;
}

Token* lexer_next_token(Lexer* lexer) {
    /* Skip whitespace */
    while (lexer->position < lexer->length && 
           isspace(lexer->source[lexer->position])) {
        lexer->position++;
    }
    
    /* Check for EOF */
    if (lexer->position >= lexer->length) {
        Token* token = malloc(sizeof(Token));
        token->type = TOKEN_EOF;
        token->value = NULL;
        token->position = lexer->position;
        return token;
    }
    
    char current = lexer->source[lexer->position];
    Token* token = malloc(sizeof(Token));
    token->position = lexer->position;
    token->value = NULL;
    
    /* Identifier or keyword */
    if (isalpha(current) || current == '_') {
        uint32_t start = lexer->position;
        while (lexer->position < lexer->length &&
               (isalnum(lexer->source[lexer->position]) || 
                lexer->source[lexer->position] == '_')) {
            lexer->position++;
        }
        
        uint32_t length = lexer->position - start;
        char* text = malloc(length + 1);
        strncpy(text, &lexer->source[start], length);
        text[length] = '\0';
        
        TokenType keyword_type;
        if (is_keyword(text, &keyword_type)) {
            token->type = keyword_type;
            free(text);  /* Don't need value for keywords */
        } else {
            token->type = TOKEN_IDENTIFIER;
            token->value = text;
        }
        return token;
    }
    
    /* Number literal */
    if (isdigit(current)) {
        uint32_t start = lexer->position;
        while (lexer->position < lexer->length &&
               isdigit(lexer->source[lexer->position])) {
            lexer->position++;
        }
        
        /* Optional decimal part */
        if (lexer->position < lexer->length &&
            lexer->source[lexer->position] == '.') {
            lexer->position++;
            while (lexer->position < lexer->length &&
                   isdigit(lexer->source[lexer->position])) {
                lexer->position++;
            }
        }
        
        uint32_t length = lexer->position - start;
        char* text = malloc(length + 1);
        strncpy(text, &lexer->source[start], length);
        text[length] = '\0';
        
        token->type = TOKEN_NUMBER;
        token->value = text;
        return token;
    }
    
    /* String literal */
    if (current == '\'') {
        lexer->position++;  /* Skip opening quote */
        uint32_t start = lexer->position;
        
        while (lexer->position < lexer->length) {
            if (lexer->source[lexer->position] == '\'') {
                /* Check for escaped quote (two consecutive quotes) */
                if (lexer->position + 1 < lexer->length &&
                    lexer->source[lexer->position + 1] == '\'') {
                    lexer->position += 2;  /* Skip both quotes */
                    continue;
                }
                break;
            }
            lexer->position++;
        }
        
        if (lexer->position >= lexer->length) {
            snprintf(lexer->error_msg, sizeof(lexer->error_msg),
                    "Unterminated string literal at position %u", start);
            token->type = TOKEN_ERROR;
            return token;
        }
        
        uint32_t length = lexer->position - start;
        char* text = malloc(length + 1);
        
        /* Copy string, handling escaped quotes */
        uint32_t dest_idx = 0;
        for (uint32_t src_idx = start; src_idx < start + length; src_idx++) {
            if (lexer->source[src_idx] == '\'' && 
                src_idx + 1 < start + length &&
                lexer->source[src_idx + 1] == '\'') {
                text[dest_idx++] = '\'';
                src_idx++;  /* Skip the second quote */
            } else {
                text[dest_idx++] = lexer->source[src_idx];
            }
        }
        text[dest_idx] = '\0';
        
        lexer->position++;  /* Skip closing quote */
        token->type = TOKEN_STRING;
        token->value = text;
        return token;
    }
    
    /* Operators and punctuation */
    switch (current) {
        case '=':
            token->type = TOKEN_EQUAL;
            lexer->position++;
            break;
        case '<':
            if (lexer->position + 1 < lexer->length &&
                lexer->source[lexer->position + 1] == '=') {
                token->type = TOKEN_LESS_EQUAL;
                lexer->position += 2;
            } else {
                token->type = TOKEN_LESS;
                lexer->position++;
            }
            break;
        case '>':
            if (lexer->position + 1 < lexer->length &&
                lexer->source[lexer->position + 1] == '=') {
                token->type = TOKEN_GREATER_EQUAL;
                lexer->position += 2;
            } else {
                token->type = TOKEN_GREATER;
                lexer->position++;
            }
            break;
        case '!':
            if (lexer->position + 1 < lexer->length &&
                lexer->source[lexer->position + 1] == '=') {
                token->type = TOKEN_NOT_EQUAL;
                lexer->position += 2;
            } else {
                snprintf(lexer->error_msg, sizeof(lexer->error_msg),
                        "Unexpected character '!' at position %u", lexer->position);
                token->type = TOKEN_ERROR;
            }
            break;
        case '(':
            token->type = TOKEN_LPAREN;
            lexer->position++;
            break;
        case ')':
            token->type = TOKEN_RPAREN;
            lexer->position++;
            break;
        case ',':
            token->type = TOKEN_COMMA;
            lexer->position++;
            break;
        case ';':
            token->type = TOKEN_SEMICOLON;
            lexer->position++;
            break;
        default:
            snprintf(lexer->error_msg, sizeof(lexer->error_msg),
                    "Unexpected character '%c' at position %u", current, lexer->position);
            token->type = TOKEN_ERROR;
            break;
    }
    
    return token;
}

const char* lexer_get_error(Lexer* lexer) {
    return lexer->error_msg[0] ? lexer->error_msg : NULL;
}
```

**D. Core Logic Skeleton Code**

AST node structure and creation functions:

```c
/* ast.h */
#ifndef AST_H
#define AST_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    AST_SELECT,
    AST_INSERT,
    AST_CREATE_TABLE,
    AST_COLUMN_LIST,
    AST_COLUMN_REF,
    AST_TABLE_REF,
    AST_WHERE_CLAUSE,
    AST_BINARY_OP,
    AST_LITERAL,
    AST_COLUMN_DEF,
    AST_TYPE_INT,
    AST_TYPE_TEXT
} ASTType;

typedef enum {
    OP_EQUAL,
    OP_LESS,
    OP_GREATER,
    OP_LESS_EQUAL,
    OP_GREATER_EQUAL,
    OP_NOT_EQUAL,
    OP_AND,
    OP_OR,
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE
} Operator;

typedef enum {
    DATA_TYPE_INTEGER,
    DATA_TYPE_TEXT,
    DATA_TYPE_FLOAT
} DataType;

typedef struct ASTNode {
    ASTType type;
    struct ASTNode** children;  /* NULL-terminated array */
    union {
        char* str_val;          /* For identifiers, string literals */
        int64_t int_val;        /* For integer literals */
        double float_val;       /* For float literals */
        bool bool_val;          /* For boolean literals */
        Operator op_val;        /* For binary operators */
        DataType data_type;     /* For column type specifications */
    } value;
} ASTNode;

typedef struct {
    ASTType type;
    ASTNode** children;
} AST;

/* AST creation/destruction */
ASTNode* ast_node_create(ASTType type);
void ast_node_add_child(ASTNode* parent, ASTNode* child);
void ast_node_destroy(ASTNode* node);

AST* ast_create(ASTType type);
void ast_destroy(AST* ast);

/* Convenience constructors */
ASTNode* ast_create_column_ref(const char* column_name);
ASTNode* ast_create_table_ref(const char* table_name);
ASTNode* ast_create_literal_int(int64_t value);
ASTNode* ast_create_literal_string(const char* value);
ASTNode* ast_create_binary_op(Operator op, ASTNode* left, ASTNode* right);

#endif /* AST_H */
```

Parser skeleton with TODOs:

```c
/* parser.c */
#include "parser.h"
#include "lexer.h"
#include "ast.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    Lexer* lexer;
    Token* current_token;
    char error_msg[256];
} Parser;

static Parser* parser = NULL;
static char global_error_msg[256] = {0};

static void parser_set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(global_error_msg, sizeof(global_error_msg), fmt, args);
    va_end(args);
}

const char* get_parser_error() {
    return global_error_msg[0] ? global_error_msg : NULL;
}

uint32_t get_parser_error_position() {
    return parser && parser->current_token ? parser->current_token->position : 0;
}

static Token* parser_peek() {
    return parser->current_token;
}

static Token* parser_consume() {
    Token* current = parser->current_token;
    if (current->type != TOKEN_EOF) {
        parser->current_token = lexer_next_token(parser->lexer);
    }
    return current;
}

static bool parser_expect(TokenType expected) {
    if (parser->current_token->type == expected) {
        parser_consume();
        return true;
    }
    return false;
}

AST* parse_statement(const char* sql) {
    /* TODO 1: Initialize parser with lexer for the SQL string */
    /* TODO 2: Get first token */
    /* TODO 3: Check token type to determine statement type */
    /* TODO 4: Call appropriate parsing function based on statement type */
    /* TODO 5: Handle parse errors and cleanup */
    /* TODO 6: Return AST or NULL on error */
    return NULL;
}

static AST* parse_select_statement() {
    /* TODO 1: Consume SELECT token (already peeked by caller) */
    /* TODO 2: Parse column list (could be * or comma-separated identifiers) */
    /* TODO 3: Expect FROM token */
    /* TODO 4: Parse table reference */
    /* TODO 5: If next token is WHERE, parse WHERE clause */
    /* TODO 6: Build and return SELECT AST node */
    return NULL;
}

static ASTNode* parse_expression() {
    /* TODO 1: Parse primary expression (literal, identifier, or parenthesized expr) */
    /* TODO 2: While next token is a binary operator */
    /* TODO 3:   Get operator precedence */
    /* TODO 4:   Parse right-hand side expression */
    /* TODO 5:   Create binary operator node */
    /* TODO 6: Return expression tree */
    return NULL;
}

static ASTNode* parse_primary_expression() {
    /* TODO 1: Check current token type */
    /* TODO 2: If token is identifier, create column reference node */
    /* TODO 3: If token is number, create integer literal node */
    /* TODO 4: If token is string, create string literal node */
    /* TODO 5: If token is '(' parse parenthesized expression */
    /* TODO 6: Otherwise report error */
    return NULL;
}

static AST* parse_insert_statement() {
    /* TODO 1: Consume INSERT token */
    /* TODO 2: Expect INTO token */
    /* TODO 3: Parse table name */
    /* TODO 4: Parse optional column list in parentheses */
    /* TODO 5: Expect VALUES token */
    /* TODO 6: Parse value list in parentheses */
    /* TODO 7: Build and return INSERT AST node */
    return NULL;
}

static AST* parse_create_table_statement() {
    /* TODO 1: Consume CREATE token */
    /* TODO 2: Expect TABLE token */
    /* TODO 3: Parse table name */
    /* TODO 4: Expect '(' token */
    /* TODO 5: Parse column definitions (comma separated) */
    /* TODO 6: Expect ')' token */
    /* TODO 7: Build and return CREATE TABLE AST node */
    return NULL;
}
```

**E. Language-Specific Hints (C)**

1. **Memory Management**: Use arena allocation for AST nodes. Allocate all nodes from a single block; on parse error, free the entire arena at once.

2. **String Handling**: For token values, use `strdup` to copy strings. Remember to `free` them when destroying tokens/AST.

3. **Error Reporting**: Store error messages in thread-local or global variables since C doesn't have exceptions.

4. **Unicode Identifiers**: SQLite supports Unicode in identifiers. For simplicity, you can limit to ASCII initially, but use `iswalnum` from `<wctype.h>` for full Unicode support.

5. **Performance**: For keywords, use a perfect hash function (like `gperf`) if you have many keywords, but for our subset, linear search is fine.

**F. Milestone Checkpoint**

After implementing Milestone 1 (Tokenizer):

1. **Run Test Command**: `./tests/test_lexer`
2. **Expected Output**: All tests pass, showing tokens correctly identified for various SQL snippets.
3. **Manual Verification**: In REPL, type `.test_tokenizer SELECT * FROM users` and see token list printed.
4. **Signs of Problems**: 
   - String `'it''s'` tokenizes as `it` instead of `it's` → Escaped quote handling bug.
   - `SELECT` and `select` produce different tokens → Case-insensitive comparison missing.
   - Memory usage grows with each invalid SQL → Token memory not freed on errors.

After implementing Milestone 2 (Parser):

1. **Run Test Command**: `./tests/test_parser`
2. **Expected Output**: Tests pass for valid SQL, informative errors for invalid SQL.
3. **Manual Verification**: In REPL, type `.test_ast SELECT name FROM users WHERE age > 18` and see AST structure printed.
4. **Signs of Problems**:
   - `a = 1 AND b = 2 OR c = 3` parses incorrectly → Operator precedence wrong.
   - Multiple statements cause crash → Parser doesn't stop at semicolon.
   - Parentheses in expressions not handled → `(a + b) * c` fails.

**G. Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Parser reports "unexpected token" at start of valid statement | Lexer not recognizing keywords case-insensitively | Print token type/value; test with `SeLeCt` vs `SELECT` | Use case-insensitive string comparison |
| String `'O''Brien'` becomes `O` | Lexer stops at first single quote after O | Add debug print in string lexing; check for escaped quote handling | Implement escaped quote detection (two quotes → one) |
| Expression `1 + 2 * 3` evaluates as `9` instead of `7` | Operator precedence: treating all operators equal | Print AST structure; check if `*` is higher in tree than `+` | Implement precedence climbing with proper precedence table |
| Parser crashes on large SQL | Memory leak or stack overflow from left recursion | Use valgrind; check recursion depth on expression parsing | Rewrite left-recursive grammar to iterative form |
| WHERE clause with AND/OR produces wrong AST grouping | Associativity handled incorrectly | Print AST; check if `a AND b OR c` groups as `(a AND b) OR c` | Ensure AND has higher precedence than OR in parser |

---


## 5.2 Component: B-tree Storage Engine (Milestone 3, 4)

> **Milestone(s):** 3 (B-tree Page Format), 4 (Table Storage)

The B-tree Storage Engine is the **heart of the database** — the component responsible for organizing, storing, and retrieving structured data on disk with optimal efficiency. While the parser handles language understanding and the query engine processes logical operations, the storage engine translates those operations into physical data manipulation. Its design directly determines the database's performance characteristics, storage efficiency, and reliability.

### Mental Model: The Phone Book Index

Think of the B-tree Storage Engine as a **highly organized library indexing system**. Imagine a massive phone book (your data) that must support two fundamental operations: quickly finding any person's phone number by name (point lookup) and efficiently listing all names in alphabetical order (range scan).

A naïve approach — storing entries in insertion order — would require scanning the entire phone book for every lookup. A slightly better approach — keeping entries sorted — allows binary search but makes insertions expensive (shifting all subsequent entries). The B-tree solves this with a **multi-level index**:

1. **Leaf Pages (The Actual Phone Book)**: At the bottom level, data entries are stored in sorted order within fixed-size pages (like chapters in a binder). Each page contains 50-100 entries, fully sorted within that page.
2. **Internal Pages (Chapter Index)**: Above the leaves, a directory page contains the first entry of each leaf page and pointers to those pages. This allows you to quickly determine which "chapter" contains your target name.
3. **Root Page (Master Index)**: At the very top, a single page directs you to the appropriate section of the directory. For large databases, multiple levels of internal pages create a shallow, wide tree.

This hierarchical organization provides **logarithmic-time operations** (O(log n)) for all CRUD operations while keeping data physically sorted for efficient range scans. The fixed page size aligns perfectly with disk block sizes, minimizing I/O operations — the primary performance bottleneck in database systems.

### Interface and Public API

The B-tree Storage Engine exposes a minimal, page-oriented API that abstracts away the complexity of tree navigation and page management. Higher-level components (like the query engine) interact with B-trees through cursors for iteration and direct methods for structural operations.

**Core B-tree Interface Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `btree_create_table` | `Database* db`, `const char* table_name`, `Schema* schema` | `bool` (success) | Creates a new B-tree for a table, allocating root page and storing schema in system catalog |
| `btree_open_cursor` | `BTree* btree`, `int64_t key` | `Cursor*` | Opens a cursor positioned at the specified key (or the first key ≥ it). Creates iterator for traversal |
| `cursor_next` | `Cursor* cursor` | `bool` (has next) | Advances cursor to next key-value pair in sorted order. Returns false at end of tree |
| `cursor_get_row` | `Cursor* cursor`, `Row* row` | `bool` (success) | Retrieves current row at cursor position into provided Row structure |
| `btree_insert` | `BTree* btree`, `int64_t key`, `Row* row` | `bool` (success) | Inserts key-value pair into B-tree, performing node splits if necessary |
| `btree_delete` | `BTree* btree`, `int64_t key` | `bool` (success) | Removes key-value pair from B-tree, performing rebalancing if necessary |
| `btree_update` | `BTree* btree`, `int64_t key`, `Row* row` | `bool` (success) | Updates value for existing key (implemented as delete+insert) |
| `btree_find` | `BTree* btree`, `int64_t key`, `Row* row` | `bool` (found) | Finds key and retrieves associated value without opening cursor |

**Cursor Structure (Iterator Pattern):**

The `Cursor` serves as an iterator over B-tree key-value pairs, abstracting tree traversal and page navigation:

| Field | Type | Description |
|-------|------|-------------|
| `btree` | `BTree*` | Reference to the B-tree being traversed |
| `current_page` | `uint32_t` | Page number of the page currently being examined |
| `cell_index` | `uint32_t` | Index within the current page's cell array (0-based) |
| `end_of_table` | `bool` | Flag indicating cursor has moved past last entry |
| `stack_depth` | `uint8_t` | Current depth in tree traversal stack |
| `stack` | `uint32_t[10]` | Stack of parent page numbers for navigation back up |

**B-tree Page Types and Constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `PAGE_SIZE` | 4096 | Fixed size of all pages in bytes (aligns with disk sectors) |
| `PAGE_TYPE_LEAF` | 0x0D | Page type identifier for leaf pages (stores actual data rows) |
| `PAGE_TYPE_INTERNAL` | 0x05 | Page type identifier for internal pages (stores navigation keys) |
| `PAGE_TYPE_FREELIST` | 0x0A | Page type identifier for free list pages (track unused pages) |
| `TABLE_MAX_PAGES` | 100 | Maximum number of pages cached in memory (limits memory usage) |

### Internal Behavior: Page Operations

The B-tree's internal behavior revolves around three core operations: **page layout and serialization**, **tree navigation and search**, and **node maintenance (splits/merges)**. Each operation must handle the delicate balance between in-memory efficiency and on-disk persistence.

#### Page Layout and Organization

Every `PAGE_SIZE` (4096-byte) page follows a precise binary layout that enables efficient random access while accommodating variable-sized records. The page is divided into three regions:

1. **Page Header** (Fixed-size metadata at page start)
2. **Cell Pointer Array** (Growing from header toward center)
3. **Cell Content Area** (Growing from page end toward center)

This **bidirectional allocation** strategy prevents fragmentation when cells of different sizes are inserted and deleted. The header tracks the boundaries between these regions.

**Page Header Structure:**

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `page_type` | `uint8_t` | 1 byte | `PAGE_TYPE_LEAF` or `PAGE_TYPE_INTERNAL` |
| `freeblock_start` | `uint16_t` | 2 bytes | Offset to first free block in cell area (0 if none) |
| `cell_count` | `uint16_t` | 2 bytes | Number of cells (key-value pairs) in this page |
| `cell_content_start` | `uint16_t` | 2 bytes | Offset where cell content area begins (grows upward) |
| `fragmented_free_bytes` | `uint8_t` | 1 byte | Count of fragmented free bytes in cell area |
| `rightmost_child` | `uint32_t` | 4 bytes | For internal pages: page number of rightmost child |
| *(reserved)* | `uint8_t[5]` | 5 bytes | Padding to align header to 16 bytes total |

![B-tree Page Layout Diagram](./diagrams/btree-page-layout.svg)

**Cell Formats:**

Leaf and internal pages store different cell formats, but both maintain key ordering within the page:

*Leaf Cell Format (stores actual row data):*
```
[varint key][varint row_size][serialized row data...]
```
- **varint key**: Variable-length encoded rowid (primary key)
- **varint row_size**: Variable-length encoded total size of serialized row
- **serialized row data**: Binary representation of all column values

*Internal Cell Format (stores navigation structure):*
```
[varint key][uint32_t child_page]
```
- **varint key**: Separator key (smallest key in the right child subtree)
- **child_page**: 4-byte big-endian page number of left child

#### Tree Navigation Algorithm

Searching for a key in the B-tree follows a deterministic top-down path:

1. **Start at root**: Begin with page number 0 (the root page)
2. **Binary search within page**: 
   - For internal pages: Find the largest key ≤ target key, follow corresponding child pointer
   - For leaf pages: Find the key (exact match) or position where it would be inserted
3. **Recurse downward**: Repeat step 2 until reaching a leaf page
4. **Return result**: If key exists, return its value; otherwise, return "not found"

The algorithm's efficiency comes from two properties: 1) each page contains ~100-200 keys (minimizing height), and 2) binary search within a page examines only O(log n) keys in memory.

**Concrete Example: Finding rowid 42 in a 3-level B-tree**
```
Level 1 (Root, page 0): Keys: [100, 500] → 42 < 100, follow left child (page 1)
Level 2 (Internal, page 1): Keys: [30, 70] → 42 between 30-70, follow middle child (page 4)
Level 3 (Leaf, page 4): Binary search finds key 42 at position 3 → Return row data
```
Total pages read: 3 (root + internal + leaf), far fewer than scanning all leaf pages.

#### Node Maintenance: Splits and Merges

As pages fill with data, the B-tree maintains its balance through **node splitting**. When an insert would overflow a page (exceed `PAGE_SIZE`), the page splits into two, and a separator key propagates upward:

**Leaf Page Split Procedure:**
1. **Detect overflow**: Attempt to insert new cell causes total content > `PAGE_SIZE`
2. **Create new leaf**: Allocate a new page, mark as `PAGE_TYPE_LEAF`
3. **Redistribute cells**: Move approximately half the cells (including new cell) to new page
4. **Update sibling pointers**: Set new page's right sibling pointer to old page's right sibling
5. **Propagate separator**: Insert smallest key from new page into parent as separator
6. **Update parent**: If parent overflows, recursively split upward

**Internal Page Split Procedure:** (similar but handles child pointers)
1. **Create new internal page**: Allocate new page, mark as `PAGE_TYPE_INTERNAL`
2. **Redistribute keys**: Move approximately half the separator keys to new page
3. **Update child pointers**: Adjust `rightmost_child` pointers for both pages
4. **Propagate middle key**: The middle key moves up to parent (not copied to new page)

> **Key Insight**: The tree grows upward, not downward. A split at the root creates a new root with two children, increasing tree height by one. This ensures all leaf pages remain at the same depth, guaranteeing equal search time for all keys.

#### Page Serialization and Deserialization

Converting between in-memory `Page` structures and on-disk byte buffers requires careful attention to **endianness** and **alignment**:

**Serialization Steps (page to disk):**
1. **Write header**: Store all header fields in big-endian format using `store_big_endian_u32`
2. **Write cell pointers**: For each cell, store its offset within the page (2-byte big-endian)
3. **Write cell content**: Copy each cell's binary data to calculated offset
4. **Zero unused space**: Fill remaining bytes between cell pointers and cell content with zeros
5. **Compute checksum** (optional): Store in final 4 bytes for corruption detection

**Deserialization Steps (disk to page):**
1. **Read and validate header**: Check `page_type` is valid, `cell_count` is reasonable
2. **Parse cell pointers**: Read array of offsets, validate they point within page bounds
3. **Map cell content**: Set up pointers to cell data (no copying until accessed)
4. **Verify integrity**: Optional checksum validation

> **Design Principle**: Serialization must be **idempotent** — serializing a page then deserializing it should produce an identical in-memory representation. This is crucial for crash recovery.

### ADR: Fixed vs. Variable-Length Page Sizes

> **Decision: Fixed 4096-byte Page Size**
> - **Context**: The database must store variable-sized rows (from a few bytes to several KB) while maintaining efficient disk I/O. Disk hardware reads/writes in sector-sized blocks (typically 512B-4KB), and filesystems use 4KB pages. Aligning database pages with these boundaries maximizes I/O efficiency.
> - **Options Considered**:
>   1. **Fixed 4096-byte pages**: All pages are exactly 4KB, matching typical disk sector and OS page size
>   2. **Variable-length pages**: Pages grow/shrink to fit their content, potentially saving space
>   3. **Multiple fixed sizes**: Small (1KB), medium (4KB), large (16KB) pages chosen per-table
> - **Decision**: Fixed 4096-byte pages for all B-tree nodes
> - **Rationale**: 
>   - **Predictable I/O**: Each page read/write translates to exactly one disk operation
>   - **Memory alignment**: 4KB boundaries work efficiently with virtual memory and cache lines
>   - **Simpler memory management**: Fixed-size pages enable simple array-based caching
>   - **Industry standard**: SQLite, PostgreSQL, and most databases use fixed-size pages
> - **Consequences**:
>   - **Internal fragmentation**: Pages with small rows waste space (mitigated by storing multiple rows per page)
>   - **Large row handling**: Rows > ~3KB require overflow pages (additional complexity)
>   - **Wasted space for small tables**: A 10-byte row still consumes 4KB on disk initially

**Comparison Table:**

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Fixed 4096-byte** | Predictable I/O, memory alignment, simple implementation, matches hardware | Internal fragmentation for small rows, overflow pages for large rows | **CHOSEN** — best balance of performance and simplicity |
| **Variable-length** | No wasted space, handles any row size naturally | Complex memory management, unpredictable I/O patterns, fragmentation over time | Requires sophisticated allocator, poor cache performance |
| **Multiple fixed sizes** | Can optimize for different table characteristics | Complex sizing decisions, increased metadata overhead, still has fragmentation | Adds complexity without clear benefit for educational implementation |

### Common Pitfalls

⚠️ **Pitfall 1: Ignoring Endianness in Page Serialization**
- **Description**: Writing multi-byte integers (like page numbers or offsets) directly to disk without considering byte order. On little-endian systems (x86), the bytes "0x00 0x00 0x01 0x00" represent 256, but on big-endian systems they represent 65,536.
- **Why it's wrong**: Database files become architecture-dependent — a database created on an x86 machine won't be readable on a PowerPC machine, breaking portability.
- **Fix**: Always use `store_big_endian_u32()` and `load_big_endian_u32()` helper functions for all multi-byte values in page headers and cell pointers. Choose a consistent endianness (big-endian is SQLite's convention) and stick to it.

⚠️ **Pitfall 2: Forgetting to Handle Overflow Cells**
- **Description**: Assuming all rows fit within a single page's free space. A row with a large TEXT or BLOB column (e.g., 10KB document) exceeds the `PAGE_SIZE`.
- **Why it's wrong**: Insertion fails for legitimate large rows, or worse, corrupts adjacent pages by writing beyond page boundaries.
- **Fix**: Implement overflow page chains. When a row exceeds a threshold (e.g., `PAGE_SIZE - 100`), store only the first portion in the leaf cell, with a pointer to an overflow page containing the remainder. Chain multiple overflow pages if needed.

⚠️ **Pitfall 3: Incorrect Key Comparison During Binary Search**
- **Description**: Using simple integer comparison for varint-encoded keys without proper decoding, or comparing serialized row data directly as bytes for compound keys.
- **Why it's wrong**: Varints have variable byte lengths — "128" (0x80 0x01) appears as a two-byte sequence where byte-by-byte comparison gives wrong results. Similarly, string comparisons require collation rules.
- **Fix**: Always decode keys to their native representation (int64_t) before comparison. For compound keys, deserialize each key component individually and compare according to its data type.

⚠️ **Pitfall 4: Not Handling Page Fragmentation After Deletions**
- **Description**: Simply marking deleted cells as "free" without consolidating free space. Over time, the page develops "Swiss cheese" fragmentation with many small gaps unusable for new rows.
- **Why it's wrong**: Page appears to have free space (sum of all gaps) but cannot accommodate new rows because no single gap is large enough. Database performance degrades as inserts fail or cause unnecessary splits.
- **Fix**: Maintain a free block list in page header (`freeblock_start`). When deleting a cell, add its space to the free block list, merging with adjacent free blocks. Defragment periodically by compacting all cells toward the end of page.

⚠️ **Pitfall 5: Missing Parent Pointer Updates During Splits**
- **Description**: After splitting a leaf page, updating the parent's separator key but forgetting to update the child pointer array to include the new page.
- **Why it's wrong**: The new page becomes orphaned — reachable from no parent. Subsequent searches for keys in that page fail, effectively losing data.
- **Fix**: Treat split as atomic operation: 1) allocate new page, 2) redistribute cells, 3) update parent (insert new separator AND child pointer), 4) if parent splits, recurse upward. Use a temporary "workspace" to ensure all-or-nothing completion.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Page Cache** | In-memory array of `Page*` with LRU eviction | Two-level cache: hot pages in RAM, cold pages mmap'd |
| **Serialization** | Manual byte-by-byte packing/unpacking | Protocol Buffers/FlatBuffers schema (overkill for fixed format) |
| **Varint Encoding** | Custom implementation following SQLite varint spec | Google's Protocol Buffers varint implementation |
| **Integrity Checking** | Basic header validation | CRC32 checksums in page footer |

**B. Recommended File/Module Structure:**

```
build-your-own-sqlite/
├── src/
│   ├── main.c                          # REPL entry point
│   ├── database.h/c                    # Database and Pager interfaces
│   ├── btree/                          # B-tree Storage Engine
│   │   ├── btree.h                     # Public B-tree API
│   │   ├── btree.c                     # Tree operations (insert, delete, search)
│   │   ├── page.h                      # Page structure definitions
│   │   ├── page.c                      # Page serialization/deserialization
│   │   ├── cursor.h                    # Cursor interface
│   │   ├── cursor.c                    # Cursor implementation
│   │   └── varint.h/c                  # Variable-length integer encoding
│   ├── storage/                        # Lower-level storage (if separating concerns)
│   │   ├── pager.h/c                   # Page cache and file I/O
│   │   └── freelist.h/c                # Free page management
│   └── util/                           # Shared utilities
│       ├── endian.h/c                  # Endian conversion helpers
│       └── checksum.h/c                # CRC32/checksum utilities
└── test/
    └── test_btree.c                    # B-tree unit tests
```

**C. Infrastructure Starter Code (Page Header and Basic Helpers):**

```c
/* src/util/endian.c - Complete endianness helpers */
#include <stdint.h>

/* Store 32-bit value in big-endian order */
void store_big_endian_u32(uint8_t* buf, uint32_t value) {
    buf[0] = (value >> 24) & 0xFF;
    buf[1] = (value >> 16) & 0xFF;
    buf[2] = (value >> 8) & 0xFF;
    buf[3] = value & 0xFF;
}

/* Load 32-bit value from big-endian order */
uint32_t load_big_endian_u32(const uint8_t* buf) {
    return ((uint32_t)buf[0] << 24) |
           ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8) |
           (uint32_t)buf[3];
}

/* Similar functions for uint16_t */
void store_big_endian_u16(uint8_t* buf, uint16_t value) {
    buf[0] = (value >> 8) & 0xFF;
    buf[1] = value & 0xFF;
}

uint16_t load_big_endian_u16(const uint8_t* buf) {
    return ((uint16_t)buf[0] << 8) | (uint16_t)buf[1];
}
```

```c
/* src/btree/varint.c - Complete varint encoding/decoding */
#include <stdint.h>
#include <stddef.h>

/* Encode 64-bit unsigned integer as SQLite-style varint */
int varint_encode(uint64_t value, uint8_t* out) {
    if (value <= 240) {
        out[0] = (uint8_t)value;
        return 1;
    } else if (value <= 2287) {
        out[0] = (uint8_t)((value - 240) / 256 + 241);
        out[1] = (uint8_t)((value - 240) % 256);
        return 2;
    } else if (value <= 67823) {
        out[0] = 249;
        out[1] = (uint8_t)((value - 2288) / 256);
        out[2] = (uint8_t)((value - 2288) % 256);
        return 3;
    } else if (value <= 16777215) {
        out[0] = 250;
        out[1] = (uint8_t)(value >> 16);
        out[2] = (uint8_t)(value >> 8);
        out[3] = (uint8_t)value;
        return 4;
    } else if (value <= 4294967295ULL) {
        out[0] = 251;
        store_big_endian_u32(&out[1], (uint32_t)value);
        return 5;
    } else if (value <= 1099511627775ULL) {
        out[0] = 252;
        store_big_endian_u32(&out[1], (uint32_t)(value >> 8));
        out[5] = (uint8_t)value;
        return 6;
    } else if (value <= 281474976710655ULL) {
        out[0] = 253;
        store_big_endian_u32(&out[1], (uint32_t)(value >> 16));
        store_big_endian_u16(&out[5], (uint16_t)value);
        return 7;
    } else if (value <= 72057594037927935ULL) {
        out[0] = 254;
        store_big_endian_u32(&out[1], (uint32_t)(value >> 24));
        store_big_endian_u32(&out[5], (uint32_t)value);
        return 8;
    } else {
        out[0] = 255;
        store_big_endian_u32(&out[1], (uint32_t)(value >> 32));
        store_big_endian_u32(&out[5], (uint32_t)value);
        return 9;
    }
}

/* Decode varint bytes to 64-bit unsigned integer */
int varint_decode(const uint8_t* in, uint64_t* out) {
    uint8_t first = in[0];
    if (first <= 240) {
        *out = first;
        return 1;
    } else if (first <= 248) {
        *out = 240 + 256 * (first - 241) + in[1];
        return 2;
    } else if (first == 249) {
        *out = 2288 + 256 * in[1] + in[2];
        return 3;
    } else if (first == 250) {
        *out = (in[1] << 16) | (in[2] << 8) | in[3];
        return 4;
    } else if (first == 251) {
        *out = load_big_endian_u32(&in[1]);
        return 5;
    } else if (first == 252) {
        *out = ((uint64_t)load_big_endian_u32(&in[1]) << 8) | in[5];
        return 6;
    } else if (first == 253) {
        *out = ((uint64_t)load_big_endian_u32(&in[1]) << 16) | load_big_endian_u16(&in[5]);
        return 7;
    } else if (first == 254) {
        *out = ((uint64_t)load_big_endian_u32(&in[1]) << 24) | load_big_endian_u32(&in[5]);
        return 8;
    } else { /* first == 255 */
        *out = ((uint64_t)load_big_endian_u32(&in[1]) << 32) | load_big_endian_u32(&in[5]);
        return 9;
    }
}
```

**D. Core Logic Skeleton Code:**

```c
/* src/btree/page.c - Page serialization skeleton */
#include "page.h"
#include "util/endian.h"

/* Serialize Page structure to 4096-byte buffer */
void page_serialize(const Page* page, uint8_t* buffer) {
    // TODO 1: Write page header fields to buffer[0..15]
    //   - Store page_type as single byte
    //   - Store freeblock_start, cell_count, cell_content_start as big-endian uint16_t
    //   - Store fragmented_free_bytes as single byte
    //   - For internal pages: store rightmost_child as big-endian uint32_t at offset 8
    
    // TODO 2: Write cell pointer array starting at offset 16
    //   - For each cell i (0..cell_count-1), write 2-byte big-endian offset
    //   - Cell pointers must be in sorted order by key
    
    // TODO 3: Write cell content area from the end backward
    //   - Each cell's content starts at its offset in the page
    //   - Ensure cells don't overlap and don't cross cell_content_start boundary
    
    // TODO 4: Fill unused space between cell pointers and cell content with zeros
    
    // TODO 5: Optional: Compute and store CRC32 checksum in last 4 bytes
}

/* Deserialize 4096-byte buffer to Page structure */
bool page_deserialize(Page* page, const uint8_t* buffer) {
    // TODO 1: Read and validate page header
    //   - Check page_type is valid (LEAF, INTERNAL, or FREELIST)
    //   - Load cell_count, ensure it's ≤ maximum possible for page size
    //   - Verify cell_content_start > (16 + cell_count*2) and ≤ PAGE_SIZE
    
    // TODO 2: Read cell pointer array
    //   - Allocate memory for cell_offsets array of size cell_count
    //   - Load each 2-byte offset, verify it points within cell content area
    
    // TODO 3: Set up cell data pointers (don't copy data yet)
    //   - For each cell, set cell_data[i] = &buffer[cell_offsets[i]]
    
    // TODO 4: Optional: Verify CRC32 checksum if present
    
    // TODO 5: Initialize free block list from freeblock_start
    
    return true;
}
```

```c
/* src/btree/btree.c - B-tree insertion with splitting skeleton */
#include "btree.h"
#include "page.h"
#include "cursor.h"

bool btree_insert(BTree* btree, int64_t key, Row* row) {
    // TODO 1: Serialize row to binary format
    //   - Use row_serialize() function (from storage layer)
    //   - Calculate total cell size = varint(key) + varint(row_size) + row_data
    
    // TODO 2: Find appropriate leaf page for insertion
    //   - Use btree_find_leaf_page(btree, key) to get target page number
    //   - Load page via pager_get_page()
    
    // TODO 3: Check if page has enough free space
    //   - Calculate free_space = page->header.cell_content_start - 
    //                           (16 + page->header.cell_count * 2)
    //   - If free_space >= cell_size, proceed to step 7
    
    // TODO 4: Handle page overflow (page needs to split)
    //   - Allocate new page via pager_allocate_page()
    //   - Distribute cells between old and new page (including new cell)
    //   - Determine separator key (smallest key in new page)
    
    // TODO 5: Update parent page
    //   - If current page is root (page 0), create new root
    //   - Else insert separator key into parent page
    //   - If parent overflows, recursively split upward
    
    // TODO 6: Clean up: if split occurred, determine which page (old or new) 
    //         should receive the new cell based on key comparison
    
    // TODO 7: Insert cell into leaf page
    //   - Find correct insertion position (maintain sorted order)
    //   - Make space: shift cell pointers and cell content as needed
    //   - Write cell data at calculated offset
    //   - Update cell_count and cell_content_start in header
    
    // TODO 8: Mark page as dirty via pager_mark_dirty()
    
    return true;
}
```

```c
/* src/btree/cursor.c - B-tree traversal skeleton */
#include "cursor.h"
#include "btree.h"

Cursor* btree_open_cursor(BTree* btree, int64_t key) {
    Cursor* cursor = malloc(sizeof(Cursor));
    // TODO 1: Initialize cursor fields
    //   - Set btree pointer, end_of_table = false, stack_depth = 0
    
    // TODO 2: Start at root page (page 0)
    //   - Set current_page = 0
    //   - Push page 0 onto stack (for parent navigation)
    
    // TODO 3: Traverse down to appropriate leaf
    //   - While current page is INTERNAL:
    //     * Binary search for largest key ≤ target key
    //     * Follow corresponding child pointer
    //     * Push current page onto stack before descending
    //     * Update current_page to child page number
    
    // TODO 4: Position at key in leaf page
    //   - Binary search within leaf page cells
    //   - If exact match found: set cell_index to position
    //   - If not found: set cell_index to insertion position (first key > target)
    
    // TODO 5: Handle end-of-table case
    //   - If key > all keys in leaf and leaf has no right sibling:
    //     * Set end_of_table = true
    
    return cursor;
}

bool cursor_next(Cursor* cursor) {
    // TODO 1: Check end_of_table flag - return false if true
    
    // TODO 2: Advance cell_index within current page
    //   - cell_index++
    //   - If cell_index < page->header.cell_count, return true
    
    // TODO 3: Move to next leaf page (right sibling)
    //   - Get right sibling pointer from page header
    //   - If sibling == 0 (no more leaves): set end_of_table = true, return false
    //   - Update current_page to sibling, reset cell_index = 0
    
    // TODO 4: Update traversal stack
    //   - When moving to sibling, may need to pop stack entries
    //   - Keep stack consistent for potential upward traversal
    
    return true;
}
```

**E. Language-Specific Hints (C):**
- **Memory Management**: Use `malloc()`/`free()` for dynamic structures like `Cursor` and `Row`. For page data, the `Pager` manages a fixed array of `PAGE_SIZE` buffers.
- **File I/O**: Use `open()`, `read()`, `write()`, `fsync()` with `O_RDWR | O_CREAT` flags. Position file pointer using `lseek()` or `pread()`/`pwrite()` for atomic operations.
- **Error Handling**: All B-tree functions should return `bool` (success/fail) with an error message accessible via `btree_get_error()`. Use `errno` for system call errors.
- **Portability**: Always include `<stdint.h>` for fixed-width types. Use `#ifdef __BIG_ENDIAN__` for endianness detection if not using the provided helpers.
- **Debugging**: Compile with `-g -fsanitize=address` for memory error detection. Use `hexdump -C database_file` to inspect raw page contents.

**F. Milestone Checkpoint (Milestone 3 & 4):**
- **Command to Test**: `make test-btree && ./test_btree`
- **Expected Output**: 
  ```
  Test 1: Page serialization/deserialization... PASS
  Test 2: Leaf cell insertion in sorted order... PASS  
  Test 3: Internal page navigation... PASS
  Test 4: B-tree insertion with 1000 rows... PASS
  Test 5: Leaf page split on overflow... PASS
  Test 6: Range scan via cursor... PASS
  All 6 tests passed.
  ```
- **Manual Verification**: 
  1. Create a test database: `./sqlite test.db "CREATE TABLE users (id INT, name TEXT)"`
  2. Insert 500 rows: Use a script to insert rows with sequential ids
  3. Verify file size is approximately `500 * (avg_row_size) / 4096 * 4096` bytes
  4. Run `SELECT * FROM users ORDER BY id` and verify rows return in exact id order
- **Signs of Problems**:
  - **File size not increasing**: Pages not being allocated for new data
  - **Rows returned out of order**: Cell sorting within pages is incorrect
  - **Crash after many inserts**: Page split logic has infinite recursion
  - **Data corruption after restart**: Serialization/deserialization mismatch

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Insert fails with "no space" but file has free pages** | Free page list corruption | Dump freelist page contents with hex editor | Rebuild freelist by scanning all pages |
| **Binary search finds wrong key** | Varint encoding/decoding mismatch | Print raw bytes of keys being compared | Ensure `varint_encode`/`varint_decode` are inverse operations |
| **Page split creates orphaned page** | Parent pointer not updated | After split, traverse from root to verify all leaves reachable | Ensure split updates parent's cell pointer array, not just separator |
| **Cursor stops early during range scan** | Right sibling pointer incorrect | Check leaf page header's right sibling field | When splitting leaf, update both old and new page's sibling pointers |
| **Database grows 4MB after first insert** | Root page splitting too early | Check split threshold logic | Only split when page truly full, not when free space < cell size |


## 5.3 Component: Query Execution Engine (Milestone 5, 6, 7, 8)

> **Milestone(s):** 5 (SELECT Execution), 6 (INSERT/UPDATE/DELETE), 7 (WHERE Clause and Indexes), 8 (Query Planner)

The Query Execution Engine is the **processing brain** of the database — the component responsible for transforming parsed SQL statements (ASTs) into actual data operations, reading and modifying table rows, applying filters, selecting optimal access paths, and assembling final result sets.

### Mental Model: The Assembly Line

Imagine a **car manufacturing assembly line**. Raw materials (table data) enter at the beginning, then pass through a series of specialized workstations (operators) that each perform a specific transformation:

1. **Feeder Station (Cursor/Scan)**: Retrieves raw rows from storage, feeding them onto the conveyor belt
2. **Filter Station (WHERE)**: Removes rows that don't meet quality criteria (predicate conditions)
3. **Transformation Station (Projection)**: Modifies each row by removing unnecessary parts (columns) or adding computed features
4. **Sorting Station (ORDER BY)**: Reorders items on the conveyor belt into a specific sequence
5. **Quality Control (Constraint Checking)**: For INSERT/UPDATE operations, verifies each modified row meets schema requirements before allowing it into the warehouse (storage)

Just as an assembly line can be reconfigured for different car models, the query engine builds **execution plans** — customized pipelines of operators optimized for each specific query. The **query planner** acts as the factory foreman, deciding which workstations to use and in what order based on the query requirements and available resources (indexes, table sizes).

This assembly line model emphasizes several key principles:
- **Streaming processing**: Rows flow through operators one at a time (or in small batches), minimizing memory usage
- **Operator composition**: Complex queries chain simple, single-purpose operators together
- **Push vs. pull semantics**: Operators can either *pull* rows from upstream (demand-driven) or have rows *pushed* to them (data-driven)
- **Materialization points**: Some operators may need to collect all rows before proceeding (like sorting), creating temporary "buffers" in the line

### Interface and Public API

The Query Execution Engine exposes a clean interface to the database frontend, accepting parsed ASTs and returning result sets or modification counts. All execution happens through the central `database_execute()` function, which coordinates parsing, planning, and execution.

#### Core Execution Functions

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `database_execute(db, sql)` | `db` (`Database*`), `sql` (`const char*`) | `int` (error code) | Primary public API: parses SQL, creates execution plan, executes it, and returns success/error |
| `execute_select(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `ResultSet*` | Executes a SELECT AST, returning a collection of rows (caller must free) |
| `execute_insert(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `int64_t` (rows affected) | Executes INSERT, returns number of rows inserted (or -1 on error) |
| `execute_update(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `int64_t` (rows affected) | Executes UPDATE, returns number of rows modified |
| `execute_delete(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `int64_t` (rows affected) | Executes DELETE, returns number of rows removed |
| `execute_create_table(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `bool` (success) | Executes CREATE TABLE, initializes B-tree and schema catalog |
| `execute_create_index(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `bool` (success) | Executes CREATE INDEX, builds secondary index B-tree |
| `explain_query(db, ast)` | `db` (`Database*`), `ast` (`ASTNode*`) | `char*` (explanation text) | Returns textual description of execution plan for a query |

#### Execution State and Result Structures

| Structure Name | Fields | Description |
|----------------|--------|-------------|
| `ResultSet` | `rows` (`Row**`), `row_count` (`uint32_t`), `column_names` (`char**`), `column_count` (`uint32_t`), `current_pos` (`uint32_t`) | In-memory collection of result rows with metadata for client consumption |
| `ExecutionPlan` | `root_operator` (`Operator*`), `estimated_cost` (`double`), `estimated_rows` (`uint32_t`) | Complete query execution strategy with cost estimates |
| `Operator` (base type) | `type` (`OperatorType`), `children` (`Operator**`), `next()` function pointer, `reset()` function pointer, `context` (`void*`) | Abstract base for all query operators (table scan, filter, projection, etc.) |
| `TableScanOperator` | `base` (`Operator`), `cursor` (`Cursor*`), `table_name` (`char*`), `schema` (`Schema*`) | Operator that iterates through all rows in a table |
| `IndexScanOperator` | `base` (`Operator`), `cursor` (`Cursor*`), `index_name` (`char*`), `table_name` (`char*`), `key_range` (`KeyRange*`) | Operator that uses an index to find rows efficiently |
| `FilterOperator` | `base` (`Operator`), `predicate` (`Expression*`), `child` (`Operator*`) | Operator that filters rows based on WHERE condition |
| `ProjectionOperator` | `base` (`Operator`), `column_indices` (`int*`), `column_count` (`uint32_t`), `child` (`Operator*`) | Operator that selects specific columns from rows |
| `InsertOperator` | `base` (`Operator`), `table_name` (`char*`), `columns` (`char**`), `values` (`Expression**`), `value_count` (`uint32_t`) | Operator that inserts new rows into a table |
| `UpdateOperator` | `base` (`Operator`), `table_name` (`char*`), `assignments` (`Assignment*`), `assignment_count` (`uint32_t`), `where_clause` (`Expression*`) | Operator that modifies existing rows |
| `DeleteOperator` | `base` (`Operator`), `table_name` (`char*`), `where_clause` (`Expression*`) | Operator that removes rows from a table |

### Internal Behavior: Operators and Planning

The execution engine follows a classic **volcano model** (iterator model), where each operator implements a standard `next()` interface that returns one row at a time or signals end-of-data. This pull-based model enables **pipelined execution** — rows flow through operator chains without materializing intermediate results (except where necessary, like sorting).

#### Operator Pipeline Construction

When the engine receives an AST, it goes through three phases:

1. **Logical planning**: Convert AST into a logical operator tree
2. **Physical planning**: Choose physical implementations for each logical operator
3. **Optimization**: Apply transformations to improve efficiency

For a simple `SELECT name, age FROM users WHERE age > 21`, the pipeline construction works as follows:

```
Parse Tree → Logical Plan → Physical Plan → Optimized Plan
     ↓             ↓             ↓              ↓
   AST      Project(name,age) TableScan   IndexScan(age)
             ↓               ↓             ↓
          Filter(age>21)   users table   age index
             ↓
         TableScan
           users
```

The physical plan might start as a table scan, but the optimizer could transform it to use an index if one exists on the `age` column.

#### Execution Algorithm Walkthrough

Let's trace the execution of a SELECT query with a WHERE clause:

1. **Query Reception**: `database_execute()` receives SQL text and calls `parse_statement()` to produce an AST
2. **Plan Generation**: The planner examines the AST and available indexes:
   - For `SELECT * FROM employees WHERE department = 'Engineering'`:
   - If no index exists: creates a `TableScanOperator` with a `FilterOperator` on top
   - If a `department` index exists: creates an `IndexScanOperator` with key range `['Engineering', 'Engineering']`
3. **Operator Initialization**: The chosen plan's operators are initialized:
   - Open B-tree cursors at appropriate starting positions
   - Allocate expression evaluation contexts
   - Set up constraint checking state
4. **Result Production**: The engine repeatedly calls `next()` on the root operator:
   ```
   // Simplified execution loop
   ResultSet* results = create_result_set();
   Operator* plan = build_execution_plan(ast);
   
   while (plan->next() == ROW_AVAILABLE) {
       Row* row = plan->get_current_row();
       add_row_to_result_set(results, row);
   }
   ```
5. **Cleanup**: All operators are reset, cursors closed, and temporary resources freed

#### Constraint Enforcement During DML Operations

For INSERT, UPDATE, and DELETE operations, the engine must enforce schema constraints:

1. **NOT NULL**: During INSERT and UPDATE, verify that columns marked NOT NULL receive non-NULL values
2. **UNIQUE**: Check that no duplicate values exist for uniquely constrained columns (requires index lookups)
3. **Primary Key**: Ensure PRIMARY KEY values are unique and not NULL (special case of UNIQUE + NOT NULL)
4. **Foreign Key**: (If implemented) Verify referential integrity between tables

The constraint checking algorithm for INSERT:

> 1. For each column with NOT NULL constraint:
>    - If the provided value is NULL → raise constraint violation error
> 2. For each UNIQUE constraint (including primary key):
>    - Use index lookup (or table scan if no index) to check if value already exists
>    - If duplicate found → raise constraint violation error
> 3. If all constraints pass:
>    - Proceed with B-tree insertion via `btree_insert()`

#### Index Maintenance Operations

Secondary indexes require maintenance during data modifications:

- **INSERT**: For each indexed column, insert (column_value, rowid) pair into index B-tree
- **UPDATE**: If indexed column changes, delete old (old_value, rowid) from index, insert new (new_value, rowid)
- **DELETE**: Remove all (column_value, rowid) entries for the deleted row from all indexes

> **Critical Insight**: Index maintenance must be atomic with the base table modification — either both succeed or both fail. This is typically achieved by performing index operations within the same transaction boundary and rolling back all changes if any constraint fails.

#### Query Planning and Optimization

The planner's role is to select the most efficient execution strategy. For our educational implementation, we focus on three key decisions:

1. **Access Path Selection**: Full table scan vs. index scan
2. **Predicate Pushdown**: Applying WHERE filters as early as possible in the pipeline
3. **Projection Pushdown**: Discarding unnecessary columns early to reduce data movement

The planner uses a simple **cost model** based on estimated I/O operations:

```
Cost(TableScan) = Number of pages in table
Cost(IndexScan) = Height of index B-tree + Number of matching rows / rows per leaf page
```

For a WHERE clause like `column = value`, the planner:
1. Checks if an index exists on `column`
2. Estimates selectivity (fraction of rows matching):
   - Equality on unique column: selectivity = 1/total_rows
   - Equality on non-unique column: use index statistics if available, else heuristic (e.g., 0.1)
3. Compares costs and chooses the cheaper plan

### ADR: Cost-Based vs. Heuristic Index Selection

> **Decision: Hybrid Cost-Based with Heuristic Fallbacks**
> 
> - **Context**: Our embedded database needs to choose between table scans and index scans efficiently. Full cost-based optimization requires detailed statistics (histograms, cardinality estimates) that are expensive to maintain in a simple implementation. Pure heuristics might choose poor plans for unusual data distributions.
> - **Options Considered**:
>   1. **Pure cost-based optimization**: Collect detailed statistics (row counts, value distributions) for each table and index, compute precise costs for each plan
>   2. **Pure heuristic rules**: Use fixed rules (e.g., "always use index for equality on primary key", "never use index for non-selective predicates")
>   3. **Hybrid approach**: Use lightweight statistics (table size, index height) for cost estimation, with heuristic fallbacks when statistics are unavailable
> - **Decision**: Implement hybrid cost-based optimization with heuristic fallbacks. Maintain minimal statistics (table page count, index height, unique flag) and use simple cost formulas. When statistics are missing, use conservative heuristics.
> - **Rationale**:
>   - Pure cost-based is overkill for our educational scope — maintaining histograms doubles the complexity
>   - Pure heuristics fail on edge cases (e.g., index on boolean column with 50/50 distribution should not be used)
>   - Hybrid gives reasonable decisions with minimal overhead: we track only what's already available (B-tree heights, row counts from leaf pages)
>   - Aligns with SQLite's approach: basic ANALYZE statistics, but sensible defaults when statistics absent
> - **Consequences**:
>   - Need to store/update simple statistics (table row counts, index uniqueness flags)
>   - `CREATE INDEX` must gather initial statistics
>   - `INSERT/UPDATE/DELETE` must update row counts
>   - Plan choices are good but not optimal for skewed distributions without ANALYZE
>   - EXPLAIN output shows both estimated costs and actual chosen plan

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| Pure Cost-Based | Optimal plans for all data distributions | Requires histogram maintenance, ANALYZE command, significant complexity | Over-engineering for educational scope |
| Pure Heuristic | Simple to implement, no statistics needed | Can choose terrible plans for unusual data (e.g., index on gender column) | Unacceptable for learning proper DB principles |
| Hybrid | Reasonable plans with minimal overhead, teaches real-world tradeoffs | Still requires some statistics tracking | **CHOSEN**: Best balance of educational value and implementation feasibility |

### Common Pitfalls

⚠️ **Pitfall: Forgetting to Reset Operators Between Executions**
- **Description**: After executing a query, operators retain their state (cursor positions, buffer contents). If reused for another query without reset, they may return wrong results or no results.
- **Why It's Wrong**: Operator reuse is common in prepared statements or repeated queries. Stale state causes silent data corruption.
- **Fix**: Implement a `reset()` method for each operator that reinitializes internal state to starting conditions. Always call `reset()` before executing a plan.

⚠️ **Pitfall: Not Handling NULL in Expression Evaluation**
- **Description**: Expressions like `WHERE age > 21` produce NULL when `age` is NULL. Comparison operators must follow **three-valued logic** (TRUE, FALSE, UNKNOWN).
- **Why It's Wrong**: SQL semantics require `NULL > 21` to evaluate to UNKNOWN (treated as FALSE in WHERE). Getting this wrong returns incorrect rows.
- **Fix**: Implement proper three-valued logic in expression evaluator. All comparison operators should return a special `BOOL_NULL` value when any operand is NULL.

⚠️ **Pitfall: Index-Only Scan Optimization Missed**
- **Description**: When a query only needs columns that are all in an index (e.g., `SELECT indexed_col FROM table`), the engine can read just the index without touching the table.
- **Why It's Wrong**: Reading both index and table pages wastes I/O. This optimization can double performance for covered queries.
- **Fix**: During planning, check if all requested columns are in an index. If so, use index-only scan: modify `IndexScanOperator` to return rows directly from index leaf cells without table lookup.

⚠️ **Pitfall: Expression Evaluation Context Not Reset Between Rows**
- **Description**: Expression evaluators that cache intermediate results or reuse allocated memory between rows can leak data from previous rows.
- **Why It's Wrong**: A computed column like `SELECT salary * 1.1` might incorrectly multiply previous row's result if context isn't cleared.
- **Fix**: Create a fresh evaluation context for each row, or explicitly clear all temporary values in the context before evaluating the next row.

⚠️ **Pitfall: B-tree Page Pinning in Nested Loops**
- **Description**: When using nested index lookups (e.g., index scan feeding table lookups), the engine might hold multiple B-tree pages pinned simultaneously, exceeding the page cache limit.
- **Why It's Wrong**: Page cache thrashing occurs, causing excessive disk I/O as pages are constantly loaded and evicted.
- **Fix**: Implement careful cursor management: unpin previous page when moving to next, use LRU page replacement policy in pager, or materialize rowids first then fetch rows in batches.

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Expression Evaluation | Recursive tree walking | Bytecode interpreter (like SQLite's VDBE) |
| Operator Execution | Volcano (iterator) model | Vectorized/batch processing |
| Cost Estimation | Heuristic rules only | Lightweight statistics + cost formulas |
| Plan Caching | No caching (replan each time) | Simple LRU cache for prepared statements |

#### B. Recommended File/Module Structure

```
build-your-own-sqlite/
├── src/
│   ├── main.c                          # REPL entry point
│   ├── database.c/.h                   # Database layer, includes database_execute()
│   ├── parser/                         # From Milestone 1-2
│   │   ├── lexer.c/.h
│   │   ├── parser.c/.h
│   │   └── ast.c/.h
│   ├── execution/                      # Query Execution Engine (this component)
│   │   ├── executor.c/.h               # Top-level execute_*() functions
│   │   ├── operators.c/.h              # All operator implementations
│   │   ├── planner.c/.h                # Query planning and optimization
│   │   ├── expression.c/.h             # Expression evaluation
│   │   └── result_set.c/.h             # ResultSet management
│   ├── storage/                        # From Milestone 3-4
│   │   ├── btree.c/.h
│   │   ├── pager.c/.h
│   │   └── cursor.c/.h
│   ├── transaction/                    # From Milestone 9-10
│   │   ├── transaction.c/.h
│   │   └── wal.c/.h
│   └── schema/
│       ├── catalog.c/.h                # System tables (sqlite_master)
│       └── constraint.c/.h             # Constraint checking
└── test/
    ├── test_executor.c                 # Unit tests for execution engine
    └── integration.sql                 # End-to-end SQL test scripts
```

#### C. Infrastructure Starter Code

**Complete expression evaluation context:**

```c
// expression_context.h
#ifndef EXPRESSION_CONTEXT_H
#define EXPRESSION_CONTEXT_H

#include <stdbool.h>
#include "ast.h"
#include "row.h"

typedef struct {
    Row* current_row;           // Row being evaluated
    Schema* table_schema;       // Schema for column references
    void* user_context;         // For custom functions (optional)
} ExpressionContext;

// Expression evaluation result with three-valued logic support
typedef enum {
    BOOL_FALSE = 0,
    BOOL_TRUE = 1,
    BOOL_NULL = 2
} TriStateBool;

// Public API
ExpressionContext* create_expression_context(Schema* schema);
void destroy_expression_context(ExpressionContext* context);
TriStateBool evaluate_expression(ExpressionContext* context, ASTNode* expr, ColumnValue* result);
bool evaluate_predicate(ExpressionContext* context, ASTNode* where_clause);

#endif
```

```c
// expression_context.c
#include "expression_context.h"
#include <stdlib.h>
#include <string.h>

ExpressionContext* create_expression_context(Schema* schema) {
    ExpressionContext* ctx = malloc(sizeof(ExpressionContext));
    if (!ctx) return NULL;
    
    ctx->current_row = NULL;
    ctx->table_schema = schema;
    ctx->user_context = NULL;
    return ctx;
}

void destroy_expression_context(ExpressionContext* context) {
    free(context);
}

TriStateBool evaluate_predicate(ExpressionContext* context, ASTNode* where_clause) {
    if (!where_clause) {
        // No WHERE clause means all rows match
        return BOOL_TRUE;
    }
    
    ColumnValue result;
    TriStateBool eval_result = evaluate_expression(context, where_clause, &result);
    
    // Convert ColumnValue to TriStateBool
    if (result.type == DATA_TYPE_NULL) {
        return BOOL_NULL;
    }
    if (result.type == DATA_TYPE_BOOL) {
        return result.value.bool_val ? BOOL_TRUE : BOOL_FALSE;
    }
    // For non-boolean results, SQL treats any non-zero/non-NULL as TRUE
    return BOOL_TRUE;
}

// Helper function to get column value by name
static bool get_column_value(ExpressionContext* ctx, const char* col_name, ColumnValue* out) {
    if (!ctx->current_row || !ctx->table_schema) return false;
    
    // Find column index in schema
    int col_index = -1;
    for (uint32_t i = 0; i < ctx->table_schema->column_count; i++) {
        if (strcmp(ctx->table_schema->columns[i].name, col_name) == 0) {
            col_index = i;
            break;
        }
    }
    
    if (col_index < 0 || col_index >= ctx->current_row->column_count) {
        return false;
    }
    
    *out = ctx->current_row->columns[col_index];
    return true;
}
```

#### D. Core Logic Skeleton Code

**Table scan operator implementation:**

```c
// operators.c - TableScanOperator implementation

typedef struct {
    Operator base;           // Base operator fields
    Cursor* cursor;          // B-tree cursor for iteration
    char* table_name;        // Table being scanned
    Schema* schema;          // Table schema for deserialization
    Row current_row;         // Buffer for current row
    bool row_valid;          // Whether current_row contains valid data
} TableScanOperator;

static OperatorResult table_scan_next(Operator* op) {
    TableScanOperator* ts = (TableScanOperator*)op;
    
    // TODO 1: Check if this is the first call (cursor may not be positioned)
    //         If cursor is NULL, open cursor at start of table using btree_open_cursor()
    //         with key = 0 (or first key)
    
    // TODO 2: Advance cursor using cursor_next()
    //         If cursor->end_of_table is true, return OPERATOR_EOF
    
    // TODO 3: Use cursor_get_row() to retrieve current row into ts->current_row
    //         If deserialization fails, return OPERATOR_ERROR
    
    // TODO 4: Set ts->row_valid = true and return OPERATOR_ROW_READY
}

static void table_scan_reset(Operator* op) {
    TableScanOperator* ts = (TableScanOperator*)op;
    
    // TODO 1: If cursor exists, close it (need cursor_close() function)
    // TODO 2: Set cursor = NULL to force re-open on next next() call
    // TODO 3: Set row_valid = false
    // TODO 4: Free any allocated memory in current_row
}

static Row* table_scan_get_current_row(Operator* op) {
    TableScanOperator* ts = (TableScanOperator*)op;
    
    // TODO 1: Check if row_valid is true
    // TODO 2: Return pointer to current_row if valid, NULL otherwise
}

Operator* create_table_scan_operator(Database* db, const char* table_name) {
    // TODO 1: Allocate TableScanOperator memory
    // TODO 2: Initialize base operator fields (type = OPERATOR_TABLE_SCAN)
    // TODO 3: Set table_name (strdup)
    // TODO 4: Lookup table schema from database catalog
    // TODO 5: Set function pointers: next, reset, get_current_row
    // TODO 6: Initialize other fields: cursor = NULL, row_valid = false
    // TODO 7: Return cast to Operator*
}
```

**Filter operator implementation:**

```c
// operators.c - FilterOperator implementation

typedef struct {
    Operator base;               // Base operator fields
    ASTNode* predicate;          // WHERE clause expression tree
    Operator* child;             // Operator providing input rows
    ExpressionContext* eval_ctx; // Context for expression evaluation
    Row* current_row;            // Current row from child (not owned)
} FilterOperator;

static OperatorResult filter_next(Operator* op) {
    FilterOperator* filter = (FilterOperator*)op;
    
    // TODO 1: Loop until we find a row matching predicate or child is exhausted
    while (true) {
        // TODO 2: Call child->next() to get next input row
        //         If child returns OPERATOR_EOF, return OPERATOR_EOF
        //         If child returns OPERATOR_ERROR, propagate error
        
        // TODO 3: Get current row from child using child->get_current_row()
        //         Store pointer in filter->current_row
        
        // TODO 4: Set current row in evaluation context: eval_ctx->current_row = filter->current_row
        
        // TODO 5: Evaluate predicate using evaluate_predicate(eval_ctx, filter->predicate)
        //         Handle three-valued logic: BOOL_NULL treated as FALSE in WHERE
        
        // TODO 6: If predicate evaluates to BOOL_TRUE, return OPERATOR_ROW_READY
        //         If BOOL_FALSE or BOOL_NULL, continue loop to check next row
    }
}

Operator* create_filter_operator(Operator* child, ASTNode* predicate, Schema* schema) {
    // TODO 1: Allocate FilterOperator memory
    // TODO 2: Initialize base operator fields (type = OPERATOR_FILTER)
    // TODO 3: Set child operator (take ownership)
    // TODO 4: Set predicate (AST tree)
    // TODO 5: Create expression context using create_expression_context(schema)
    // TODO 6: Set function pointers
    // TODO 7: Return cast to Operator*
}
```

**Query planner for SELECT with WHERE:**

```c
// planner.c - Basic plan generation

ExecutionPlan* create_select_plan(Database* db, ASTNode* select_ast) {
    // Extract SELECT statement components from AST
    // TODO 1: Get table name from FROM clause
    // TODO 2: Get column list for projection
    // TODO 3: Get WHERE clause predicate
    
    // TODO 4: Check system catalog for indexes on this table
    //         For each index, check if it can be used for WHERE clause predicates
    
    // TODO 5: Choose between table scan and index scan:
    //         a. If no WHERE clause or no usable index → table scan
    //         b. If equality predicate on indexed column → consider index scan
    //         c. Estimate costs for both options
    
    // TODO 6: Build operator tree bottom-up:
    //         Root = create_projection_operator(column_indices)
    //         If WHERE clause: add create_filter_operator(predicate)
    //         Add scan operator (table or index) at bottom
    
    // TODO 7: Create ExecutionPlan, set root_operator and cost estimates
    // TODO 8: Return plan
}
```

#### E. Language-Specific Hints (C)

- **Memory Management**: Use `malloc()`/`free()` consistently. For operator trees, implement a recursive `destroy_operator()` that frees child operators.
- **Function Pointers**: The operator interface uses function pointers for virtual dispatch. Define a struct with function pointers and a void pointer for context.
- **Error Propagation**: Return error codes up the call stack, but also set a thread-local/global error message for detailed diagnostics.
- **String Handling**: Use `strdup()` for copying strings, but remember to free them. Consider a simple string pool allocator for repeated strings like column names.
- **Integer Types**: Use `int64_t` for rowids and counts to ensure 64-bit capacity even on 32-bit systems.
- **const Correctness**: Mark input parameters as `const` where possible (e.g., `const char* sql`).

#### F. Milestone Checkpoint

**After implementing Milestone 5 (SELECT Execution):**
- Run test: `SELECT * FROM users;` should return all rows in rowid order
- Verify: Each row shows all columns with correct values
- Test: `SELECT name, email FROM users;` should return only those two columns
- Expected output format: column headers followed by rows
- Debug hint: If no rows returned, check that `cursor_next()` advances properly and `deserialize_leaf_cell()` succeeds

**After implementing Milestone 6 (INSERT/UPDATE/DELETE):**
- Test sequence:
  ```sql
  INSERT INTO users (name, age) VALUES ('Alice', 30);
  SELECT * FROM users WHERE name = 'Alice';  -- Should return 1 row
  UPDATE users SET age = 31 WHERE name = 'Alice';
  SELECT age FROM users WHERE name = 'Alice';  -- Should return 31
  DELETE FROM users WHERE name = 'Alice';
  SELECT COUNT(*) FROM users WHERE name = 'Alice';  -- Should return 0
  ```
- Verify: Constraints are enforced (try `INSERT INTO users (age) VALUES (NULL)` on NOT NULL name column)
- Debug hint: If UPDATE doesn't work, check that `btree_update()` modifies the correct leaf cell

**After implementing Milestone 7 (WHERE and Indexes):**
- Test sequence:
  ```sql
  CREATE INDEX idx_age ON users(age);
  EXPLAIN SELECT * FROM users WHERE age > 25;  -- Should show index scan
  SELECT * FROM users WHERE age > 25 AND age < 35;  -- Should use index
  ```
- Verify: Index scan is faster than table scan (noticeable with 1000+ rows)
- Debug hint: If index not used, check that WHERE clause analyzer correctly identifies indexable predicates

**After implementing Milestone 8 (Query Planner):**
- Test: `EXPLAIN QUERY PLAN SELECT * FROM users WHERE id = 5 AND age > 20;`
- Expected output shows chosen plan, estimated rows, and cost
- Verify: With both id (primary) and age indexes, planner chooses id index (more selective)
- Debug hint: If cost estimates seem wrong, check statistics gathering in `CREATE INDEX`

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Query returns no rows but data exists | Cursor not advancing | Add debug prints in `cursor_next()` and `table_scan_next()` | Ensure `cursor_next()` updates `cell_index` and checks `end_of_table` |
| WHERE clause ignores NULL values | Two-valued instead of three-valued logic | Test `SELECT * FROM t WHERE col > 10` when col is NULL | Implement `BOOL_NULL` result and treat as FALSE in WHERE |
| Index not used for obvious query | Statistics missing or cost model broken | Run `EXPLAIN` to see chosen plan, check index metadata | Ensure `CREATE INDEX` gathers basic stats (row count, unique flag) |
| Memory leak with repeated queries | Operators not properly freed | Use valgrind or address sanitizer | Implement `destroy_operator()` that recursively frees children |
| Constraint violation not detected | UNIQUE check missing index lookup | Try inserting duplicate values in indexed column | In INSERT, for each UNIQUE constraint, search index for existing value |
| Query slow with large table | No pipelining (materializing all rows) | Monitor memory usage during `SELECT * FROM big_table` | Ensure operators use iterator pattern, not collecting all rows first |

---


## 5.4 Component: Transactions & WAL (Milestone 9, 10)

> **Milestone(s):** 9 (Transactions: BEGIN/COMMIT/ROLLBACK), 10 (WAL Mode)

This component ensures **data integrity** and **crash recovery** by implementing ACID (Atomicity, Consistency, Isolation, Durability) transactions. Think of it as the database's safety net—it guarantees that your data remains correct and persistent even in the face of system crashes, power failures, or unexpected errors. This is achieved through two primary mechanisms: a **rollback journal** (the classic approach) and **Write-Ahead Logging (WAL)** (a more advanced, concurrent-friendly technique).

### Mental Model: The Bank Ledger

Imagine you're a bank teller handling a customer's complex transaction—transferring money between three accounts, updating balances, and recording the transfer in a ledger. You cannot risk a power outage in the middle leaving some accounts updated and others unchanged. How do you guarantee safety?

1.  **The Classic Ledger (Rollback Journal):** Before making any permanent changes to the main account book, you write down the *original values* of the accounts on a separate "scratch pad" (the journal). You then make the changes in the main book. If the process completes, you ceremoniously tear up the scratch pad. If the lights go out mid-update, when power returns you look at the scratch pad and use it to *undo* any partial changes, restoring the main book to its original, consistent state. This ensures **atomicity**—the transaction is all-or-nothing.

2.  **The Modern Flight Recorder (Write-Ahead Log):** Instead of recording the *old* state, you record the *intended changes* in a continuously appended log file (the flight recorder). You then apply these changes to the main book. The critical rule: you must **write to the log before you write to the main book**. If a crash occurs, you "replay" the log from the last known good point to rebuild the intended state. This log also allows multiple tellers (readers) to continue reading from the main book while one is writing new entries to the log, because readers can check the log for the latest updates. This improves **concurrency**.

Both models guarantee that the "main book" (the database file) never enters an inconsistent, half-updated state visible after a crash. The **Transaction Manager** is the head teller coordinating this process, tracking which transactions are active, what they've changed, and ensuring the journal/WAL rules are followed.

### Interface and Public API

The transaction subsystem exposes a minimal API to the query execution engine, primarily through the `Database` and `Pager` components. Transaction control commands (`BEGIN`, `COMMIT`, `ROLLBACK`) are parsed as SQL statements and routed to this subsystem.

#### Data Structures

| Name | Fields (Type) | Description |
| :--- | :--- | :--- |
| `TransactionState` | `state` (`TransactionStatus`), `journal` (`Journal*`), `dirty_pages` (`uint32_t*`), `dirty_count` (`uint32_t`) | Represents the state of a single database connection's transaction. The `journal` pointer is `NULL` when not in a transaction or when using WAL mode (which has its own separate structure). The `dirty_pages` array tracks which pages have been modified in the current transaction for efficient journaling. |
| `Journal` | `file_descriptor` (`int`), `filename` (`char*`), `in_transaction` (`bool`), `page_size` (`uint32_t`) | Represents a rollback journal file. When active, it holds a file descriptor to the journal and knows the page size to write full page images. |
| `WAL` | `file_descriptor` (`int`), `filename` (`char*`), `header` (`WALHeader*`), `frame_index` (`uint32_t`), `read_lock` (`uint32_t`), `write_lock` (`bool`) | Represents the Write-Ahead Log file. The header contains metadata like the WAL format version, page size, and checkpoint information. The `frame_index` tracks the next frame to write. Locks manage concurrent access. |
| `WALHeader` | `magic` (`uint32_t`), `version` (`uint32_t`), `page_size` (`uint32_t`), `checkpoint_seq` (`uint32_t`), `salt1` (`uint32_t`), `salt2` (`uint32_t`), `checksum` (`uint32_t`) | On-disk header of a WAL file. The `magic` number identifies a valid WAL file. `salts` are random numbers changed on each checkpoint to detect obsolete WAL files. |
| `WALFrame` | `page_number` (`uint32_t`), `db_size` (`uint32_t`), `data` (`uint8_t[PAGE_SIZE]`), `checksum` (`uint32_t`) | A single frame in the WAL, containing a full page image (`data`) for the given `page_number`. `db_size` is the database size in pages at the moment the frame was written. |

**TransactionStatus Enum:**
- `TRANSACTION_IDLE`: No active transaction (default state).
- `TRANSACTION_ACTIVE`: A `BEGIN` has been issued; writes are accumulating.
- `TRANSACTION_COMMITTING`: A `COMMIT` is in progress; journal/WAL is being finalized.
- `TRANSACTION_ROLLING_BACK`: A `ROLLBACK` is in progress; changes are being undone.

#### Public API Functions

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `database_execute(db, sql)` | `db` (`Database*`), `sql` (`const char*`) | `int` (error code) | The primary public API. It parses the SQL, and if the statement is `BEGIN`, `COMMIT`, or `ROLLBACK`, it calls the corresponding transaction functions below. For other statements, it ensures they are executed within a transaction context (implicit or explicit). |
| `transaction_begin(db)` | `db` (`Database*`) | `bool` | Starts a new transaction. If not already in a transaction, sets `TransactionState.state` to `TRANSACTION_ACTIVE` and initializes the journal or WAL. For a rollback journal, this creates the journal file and writes a journal header. |
| `transaction_commit(db)` | `db` (`Database*`) | `bool` | Commits the current transaction. For a rollback journal: 1) Write a commit record to the journal, 2) `fsync()` the journal, 3) Write all dirty pages to the main database file, 4) `fsync()` the database, 5) Delete the journal file. For WAL: 1) Write a commit mark (a special WAL frame), 2) `fsync()` the WAL, 3) Release the write lock. The changes become durable and visible to new readers. |
| `transaction_rollback(db)` | `db` (`Database*`) | `bool` | Rolls back the current transaction. For a rollback journal: read the journal file backwards, copying saved original pages back to the main database, then delete the journal. For WAL: simply discard the uncommitted frames (by not writing a commit mark) and release the write lock. |
| `pager_get_page(pager, page_num)` | `pager` (`Pager*`), `page_num` (`uint32_t`) | `void*` (page buffer) | **Modified for transactions:** If a transaction is active and this is the first modification to the page, it calls `journal_page_before_write()` to save the original page to the rollback journal or marks the page as "dirty" for WAL. |
| `pager_flush_page(pager, page_num)` | `pager` (`Pager*`), `page_num` (`uint32_t`) | `void` | **Modified for transactions:** For rollback journal, this writes the dirty page to the main database file (but the journal ensures atomicity). For WAL, this writes the page to the WAL file as a new frame instead of the main database. |
| `wal_read_page(pager, page_num)` | `pager` (`Pager*`), `page_num` (`uint32_t`) | `bool` | (WAL mode only) Checks the WAL file for the most recent version of a page. If found, loads it into the cache. If not, falls back to reading from the main database file. This is called by `pager_get_page` when in WAL mode. |
| `wal_checkpoint(db, force)` | `db` (`Database*`), `force` (`bool`) | `bool` | (WAL mode only) Moves committed pages from the WAL back into the main database file and truncates the WAL. Called automatically when the WAL grows too large or can be invoked manually. |

### Internal Behavior: Journaling and Recovery

The core responsibility is to ensure **atomicity** (all changes in a transaction happen, or none do) and **durability** (committed changes survive a crash). The behavior differs significantly between the two journaling modes.

#### Rollback Journal Operation

The rollback journal is the simpler model. It follows a **shadow paging** technique: before modifying a database page in memory, you write the *entire original page* to a separate journal file. This creates a backup copy.

![Transaction State Machine](./diagrams/transaction-state.svg)

**1. Transaction Begin (`transaction_begin`):**
    1.  Check the current `TransactionState.state`. If not `TRANSACTION_IDLE`, return an error (nested transactions not supported in this simple design).
    2.  Create a journal file: typically `{database_filename}-journal`.
    3.  Write a journal header to the file (containing a magic number and page size).
    4.  `fsync()` the journal file to ensure the header is on disk.
    5.  Set `TransactionState.state = TRANSACTION_ACTIVE`.

**2. Page Modification (within `pager_get_page`):**
    1.  When the B-tree or execution engine requests a page to modify, `pager_get_page` is called.
    2.  If a transaction is active (`TRANSACTION_ACTIVE`) and this page is not already marked as dirty, call `journal_page_before_write(journal, page_num, original_page_data)`.
    3.  This function appends a **journal record** to the journal file: a header with the page number followed by the full `PAGE_SIZE` bytes of the original page image.
    4.  The page is added to the `TransactionState.dirty_pages` list.

**3. Transaction Commit (`transaction_commit`):**
    1.  Write a special **commit record** (e.g., a journal header with a committed flag) to the journal file.
    2.  `fsync()` the journal file. This is the **critical durability point**. Once this sync completes, we have a complete log of how to undo the transaction on disk.
    3.  **Write Phase:** For each page in `dirty_pages`, call `pager_flush_page` to write the modified page to the main database file.
    4.  `fsync()` the main database file to ensure all dirty pages are physically written.
    5.  Delete the journal file. This is the **atomicity point**. If the delete succeeds, the transaction is permanently committed. If the system crashes after the sync but before the delete, the recovery process will see the journal on next startup and roll forward (re-apply the changes).
    6.  Clear the dirty pages list and set `TransactionState.state = TRANSACTION_IDLE`.

**4. Transaction Rollback (`transaction_rollback`):**
    1.  Read the journal file *backwards*. For each journal record found (original page data):
    2.  Write that original data back to its corresponding page in the main database file.
    3.  After all pages are restored, delete the journal file.
    4.  Clear the dirty pages list and set `state = TRANSACTION_IDLE`.

**5. Crash Recovery (on `database_open`):**
    1.  When opening a database, check for the existence of a journal file.
    2.  If a journal file exists:
        - **If it contains a commit record:** The database crashed after commit but before deleting the journal. The changes are durable in the main file. We can safely delete the journal (this is a **hot journal**).
        - **If it does NOT contain a commit record:** The database crashed mid-transaction. Perform a rollback (as in step 4 above) to restore the database to its pre-transaction state.
    3.  This process guarantees the database file is always left in a consistent state.

#### Write-Ahead Log (WAL) Operation

WAL inverts the write order: changes are written to the log *before* they are applied to the main database. This allows readers to operate on the main database while a writer appends to the log.

![WAL Checkpoint Flowchart](./diagrams/wal-checkpoint-flow.svg)

**1. WAL Mode Initialization:**
    1.  When the database is opened in WAL mode (e.g., via `PRAGMA journal_mode=WAL`), a `WAL` object is initialized and a `-wal` file is created or opened.
    2.  The `WALHeader` is read/initialized. The `salt` values are critical for identifying valid WAL files tied to the current database.

**2. Transaction Begin in WAL Mode:**
    1.  Similar to journal mode, set `state = TRANSACTION_ACTIVE`.
    2.  Acquire the **WAL write lock** (a single writer is allowed). This is typically a file lock on the WAL file or a designated byte range within it.

**3. Page Modification & Writing in WAL Mode:**
    1.  When a page is first modified, it is marked dirty in the page cache.
    2.  On `pager_flush_page` (or at transaction commit), the dirty page is not written to the main database. Instead, it's appended to the WAL file as a `WALFrame`.
    3.  Each frame includes the page number, the database size at the time of writing, the page data, and a checksum for integrity.

**4. Transaction Commit in WAL Mode:**
    1.  All dirty pages of the transaction are written as frames to the WAL.
    2.  A special **commit frame** (or a frame with a commit flag in its header) is written to mark the end of the transaction.
    3.  `fsync()` the WAL file. This is the durability point.
    4.  Release the WAL write lock. The transaction is now committed and visible.

**5. Reading in WAL Mode (`wal_read_page`):**
    1.  When a reader (or the same connection reading its own writes) needs a page, `pager_get_page` calls `wal_read_page`.
    2.  `wal_read_page` first checks the WAL index (often a hash map of page number → most recent frame offset) to see if a newer version exists in the WAL.
    3.  If found, it reads the page data from the WAL frame. If not, it reads from the main database file.
    4.  Readers maintain a **read lock** by recording the current end of the WAL (the maximum frame index they might read). This prevents the checkpoint process from overwriting WAL frames that are still needed by active readers.

**6. Checkpointing (`wal_checkpoint`):**
    1.  A checkpoint is the process of transferring committed page changes from the WAL back to the main database file.
    2.  It acquires the write lock, ensuring no active writers.
    3.  It reads WAL frames up to the oldest active reader's read lock.
    4.  For each frame, it writes the page data to the correct location in the main database file.
    5.  It updates the WAL header to indicate the new checkpoint position and changes the `salt` values.
    6.  It can optionally truncate the WAL file to free space.
    7.  Checkpoints can be automatic (based on WAL size) or manual.

**7. Crash Recovery in WAL Mode:**
    1.  On startup, if a WAL file exists, the database is in WAL mode.
    2.  The WAL header is validated (correct magic, compatible salts).
    3.  The database replays all frames from the last checkpoint position to the end of the WAL, applying them to the main database file. This **rolls forward** all committed transactions that weren't yet checkpointed.
    4.  The WAL is then reset, and a new checkpoint is performed.

> **Key Insight:** WAL's performance advantage comes from **sequential writes** (appending to a log file is faster than random writes to a database file) and **reader-writer concurrency** (readers don't block on writers and vice versa, as they access different files).

### ADR: Rollback Journal vs. Write-Ahead Log (WAL)

> **Decision: Implement Rollback Journal First, Then WAL as an Advanced Milestone**
>
> - **Context**: We need to provide ACID transaction guarantees (Milestone 9) and later improve concurrency with multiple readers/writers (Milestone 10). The educational goal is to understand both classical and modern journaling techniques. The project must remain approachable for learners while allowing for incremental complexity.
> - **Options Considered**:
>     1.  **Rollback Journal Only**: Implement the classic SQLite rollback journal. Simpler to understand and implement, but offers poor write concurrency (writers lock the entire database).
>     2.  **WAL Only**: Implement only Write-Ahead Logging. More complex initially, but provides better concurrency and performance from the start. Might obscure the foundational journaling concept.
>     3.  **Both, with Rollback as Default**: Implement both mechanisms, using rollback journal as the default mode (for simplicity and compatibility), with an option to switch to WAL mode via a `PRAGMA`. This matches SQLite's own design.
> - **Decision**: Implement both, with the rollback journal as the initial focus for Milestone 9 and WAL as an extension for Milestone 10. The `Pager` component will abstract the journaling interface, allowing both backends to be plugged in.
> - **Rationale**:
>     - **Pedagogical Progression**: The rollback journal directly implements the "copy-on-write" shadow paging concept, which is intuitive for understanding atomicity. WAL builds on this by inverting the write order, which is a more advanced concept.
>     - **Incremental Complexity**: Learners can first master the fundamentals of journaling, recovery, and `fsync()` semantics with the simpler model before tackling the concurrent reader/writer challenges of WAL.
>     - **Alignment with Real SQLite**: This two-mode approach mirrors SQLite's actual architecture, giving learners direct insight into a widely deployed system.
> - **Consequences**:
>     - **Positive**: Clear learning path, covers two important industry patterns, code can demonstrate the trade-offs directly.
>     - **Negative**: Increased code complexity, need to maintain two journaling paths and ensure the abstraction doesn't leak.

| Option | Pros | Cons | Chosen? |
| :--- | :--- | :--- | :--- |
| **Rollback Journal Only** | Simpler implementation, easier to debug, less code. | Poor concurrency (write locks block all readers), slower for many small writes due to random I/O. | No |
| **WAL Only** | Better read/write concurrency, sequential writes are faster, modern default for many uses. | More complex initial implementation (frame indexing, checkpointing, reader locking). Obscures fundamental journaling concept. | No |
| **Both, Rollback First** | Teaches both fundamental and advanced concepts. Matches real-world SQLite. Allows users to choose based on workload. | Most implementation work. Requires a clean abstraction layer between pager and journal. | **Yes** |

### Common Pitfalls

⚠️ **Pitfall: Forgetting `fsync()` Before Deleting the Journal**
- **Description**: Writing journal records or a commit marker to the journal file but not calling `fsync()` before deleting it or writing to the main database. The OS may keep writes in its buffer cache, so a crash can lose the journal data.
- **Why it's Wrong**: This breaks **durability**. If the power fails after the write() call returns but before data reaches the disk platter, the journal is incomplete or missing. Recovery either can't happen or will corrupt the database.
- **Fix**: Always `fsync()` the journal file after writing the commit record (in rollback journal) or after writing the transaction's frames (in WAL). This ensures the log is durable before proceeding.

⚠️ **Pitfall: Not Handling Partial Writes (Torn Pages)**
- **Description**: Assuming that a `write()` system call for an entire `PAGE_SIZE` block is atomic. On some systems, especially with power loss, a sector write (typically 512 bytes) may be only partially completed.
- **Why it's Wrong**: A partially written page in the main database file is corrupted and unrecoverable if the journal only saved the original page *before* this partial write.
- **Fix**: Use a **checksum** in both journal records and WAL frames. On recovery, verify checksums before applying pages. Additionally, ensure the disk sector size is a divisor of your `PAGE_SIZE` (4096 is usually safe as it's a multiple of 512).

⚠️ **Pitfall: Letting WAL Grow Unbounded**
- **Description**: In WAL mode, committing transactions append frames but never running a checkpoint. The WAL file grows indefinitely, consuming disk space and slowing down reads (which must search through a huge log).
- **Why it's Wrong**: Defeats the purpose of WAL; eventually, performance degrades and disk fills up.
- **Fix**: Implement an automatic checkpoint trigger. Common heuristics: checkpoint when the WAL file exceeds a certain size (e.g., 1000 pages) or when a certain number of frames have been written since the last checkpoint. Also provide a manual `PRAGMA wal_checkpoint` command.

⚠️ **Pitfall: Incorrect Lock Order Leading to Deadlock**
- **Description**: In WAL mode, multiple processes/threads need to coordinate via file locks. If the order of acquiring the "read lock" and "write lock" is not consistent, deadlock can occur (e.g., Process A holds read lock and wants write lock, Process B holds write lock and wants a read lock).
- **Why it's Wrong**: The database hangs indefinitely.
- **Fix**: Implement a strict locking protocol. SQLite's approach: Use a single write lock (exclusive). Readers don't take a lock on the database file; instead, they read a "read mark" from the WAL header atomically. The checkpoint process waits until all readers have released (by checking that their read mark is beyond the frames to be checkpointed).

⚠️ **Pitfall: Not Rolling Back Implicit Transactions**
- **Description**: In SQLite, every DML statement (INSERT, UPDATE, DELETE) runs within a transaction. If the user doesn't issue an explicit `BEGIN`, an **implicit transaction** is automatically started and committed. Failing to journal this implicit transaction means a crash mid-statement could leave partial changes.
- **Why it's Wrong**: Breaks atomicity at the statement level, a basic SQL expectation.
- **Fix**: In `database_execute`, if no explicit transaction is active, automatically call `transaction_begin` before executing a DML statement and `transaction_commit` after successful execution (or `transaction_rollback` on error).

### Implementation Guidance

**A. Technology Recommendations Table**

| Component | Simple Option | Advanced Option |
| :--- | :--- | :--- |
| **Journaling Backend** | Rollback Journal (single file append, `fsync`, random restore). | Write-Ahead Log with frame indexing, reader marks, and checkpointing. |
| **Concurrency Control** | Single writer lock via `flock()` or `fcntl()` on the database file. | WAL mode with shared-memory for reader marks (or a separate lock file). |
| **Crash Detection** | Check for leftover `-journal` or `-wal` files on startup. | Validate WAL header magic and salt values to detect obsolete logs. |

**B. Recommended File/Module Structure**

```
build-your-own-sqlite/
  src/
    database.c           # Contains database_open/close, transaction_begin/commit/rollback
    database.h
    pager.c              # pager_get_page, pager_flush_page (modified for journaling)
    pager.h
    journal.c            # Rollback journal implementation
    journal.h
    wal.c                # Write-Ahead Log implementation (Milestone 10)
    wal.h
    btree.c              # Calls into pager, unaware of journaling mode
    btree.h
  include/
    constants.h          # PAGE_SIZE, journal/wal magic numbers
  test/
    test_transactions.c  # Unit tests for rollback and recovery
    test_wal.c
```

**C. Infrastructure Starter Code**

Here is a complete, usable implementation for a simple rollback journal. This handles the file operations and page recording, allowing the learner to focus on integrating it with the transaction state machine.

```c
/* journal.h */
#ifndef JOURNAL_H
#define JOURNAL_H

#include <stdbool.h>
#include <stdint.h>

#define JOURNAL_MAGIC 0x4A524E4C  // "JRNL" in ASCII

typedef struct {
    int file_descriptor;
    char* filename;
    bool in_transaction;
    uint32_t page_size;
} Journal;

typedef struct {
    uint32_t magic;
    uint32_t page_size;
    uint32_t reserved[62];  // Pad to 256 bytes total
} JournalHeader;

Journal* journal_open(const char* database_filename);
bool journal_begin_transaction(Journal* journal);
bool journal_page_before_write(Journal* journal, uint32_t page_num, const void* page_data);
bool journal_commit(Journal* journal);
bool journal_rollback(Journal* journal, int db_fd);
void journal_close(Journal* journal);

#endif
```

```c
/* journal.c */
#include "journal.h"
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

Journal* journal_open(const char* database_filename) {
    Journal* journal = malloc(sizeof(Journal));
    if (!journal) return NULL;

    size_t len = strlen(database_filename) + 8; // for "-journal"
    journal->filename = malloc(len);
    snprintf(journal->filename, len, "%s-journal", database_filename);

    journal->file_descriptor = -1;
    journal->in_transaction = false;
    journal->page_size = 0; // Set later
    return journal;
}

bool journal_begin_transaction(Journal* journal) {
    if (journal->in_transaction) return false;

    // Open journal file, create if doesn't exist, truncate if does.
    journal->file_descriptor = open(journal->filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (journal->file_descriptor < 0) return false;

    // Write journal header
    JournalHeader header = {0};
    header.magic = JOURNAL_MAGIC;
    header.page_size = journal->page_size;
    ssize_t written = write(journal->file_descriptor, &header, sizeof(header));
    if (written != sizeof(header)) {
        close(journal->file_descriptor);
        journal->file_descriptor = -1;
        return false;
    }

    // Flush header to disk
    if (fsync(journal->file_descriptor) != 0) {
        close(journal->file_descriptor);
        journal->file_descriptor = -1;
        return false;
    }

    journal->in_transaction = true;
    return true;
}

bool journal_page_before_write(Journal* journal, uint32_t page_num, const void* page_data) {
    if (!journal->in_transaction) return false;

    // Write page number header
    ssize_t written = write(journal->file_descriptor, &page_num, sizeof(page_num));
    if (written != sizeof(page_num)) return false;

    // Write the entire original page
    written = write(journal->file_descriptor, page_data, journal->page_size);
    if (written != journal->page_size) return false;

    // Note: We don't fsync after every page for performance.
    // The final commit fsync ensures all writes are durable.
    return true;
}

bool journal_commit(Journal* journal) {
    if (!journal->in_transaction) return false;

    // Write a zero page number as a commit marker (simplification)
    uint32_t commit_marker = 0;
    ssize_t written = write(journal->file_descriptor, &commit_marker, sizeof(commit_marker));
    if (written != sizeof(commit_marker)) return false;

    // CRITICAL: Ensure all journal data is on disk before proceeding
    if (fsync(journal->file_descriptor) != 0) return false;

    journal->in_transaction = false;
    close(journal->file_descriptor);
    journal->file_descriptor = -1;

    // Delete the journal file (atomic commit point)
    if (unlink(journal->filename) != 0) {
        // If delete fails, the journal remains. Recovery will replay it (hot journal).
        // This is acceptable; the transaction is still considered committed.
    }
    return true;
}

bool journal_rollback(Journal* journal, int db_fd) {
    if (!journal->in_transaction) return false;

    // Seek to just after the header
    if (lseek(journal->file_descriptor, sizeof(JournalHeader), SEEK_SET) < 0) {
        return false;
    }

    uint32_t page_num;
    void* page_buffer = malloc(journal->page_size);
    if (!page_buffer) return false;

    // Read journal records backwards is complex. For simplicity, we read sequentially.
    // Real SQLite reads backwards using the file size.
    while (read(journal->file_descriptor, &page_num, sizeof(page_num)) == sizeof(page_num)) {
        if (page_num == 0) break; // Commit marker reached (should not happen in rollback)
        if (read(journal->file_descriptor, page_buffer, journal->page_size) != journal->page_size) {
            free(page_buffer);
            return false;
        }
        // Write original page back to database file at correct offset
        off_t offset = page_num * journal->page_size;
        if (lseek(db_fd, offset, SEEK_SET) < 0 ||
            write(db_fd, page_buffer, journal->page_size) != journal->page_size) {
            free(page_buffer);
            return false;
        }
    }

    free(page_buffer);
    journal->in_transaction = false;
    close(journal->file_descriptor);
    journal->file_descriptor = -1;
    unlink(journal->filename); // Delete the journal
    return true;
}

void journal_close(Journal* journal) {
    if (journal->file_descriptor >= 0) {
        close(journal->file_descriptor);
    }
    free(journal->filename);
    free(journal);
}
```

**D. Core Logic Skeleton Code**

Now, the learner must integrate this journal with the `Pager` and `TransactionState`. Here are the key integration points with TODOs.

```c
/* In pager.c */
#include "journal.h"
#include "database.h"

void* pager_get_page(Pager* pager, uint32_t page_num) {
    // ... existing code to load page into cache ...

    // TODO 1: Check if a transaction is active (pager->db->transaction_state.state == TRANSACTION_ACTIVE)
    // TODO 2: If active and this page is not already in the dirty_pages list:
    //   a. Call journal_page_before_write(pager->journal, page_num, original_page_data)
    //   b. Add page_num to dirty_pages list
    // TODO 3: Return the page pointer
}

void pager_flush_page(Pager* pager, uint32_t page_num) {
    // TODO 1: If using rollback journal: write the cached page to the main database file (as usual)
    // TODO 2: If using WAL: write the page as a new frame to the WAL file instead
    // TODO 3: Clear the dirty flag for this page in cache
}
```

```c
/* In database.c */
bool transaction_begin(Database* db) {
    // TODO 1: If db->transaction_state.state != TRANSACTION_IDLE, return false
    // TODO 2: Initialize dirty_pages list (dynamic array or static max)
    // TODO 3: If journal mode is rollback:
    //   a. Ensure db->pager->journal is open (journal_open)
    //   b. Set journal->page_size = PAGE_SIZE
    //   c. Call journal_begin_transaction(db->pager->journal)
    // TODO 4: If WAL mode: acquire WAL write lock
    // TODO 5: Set db->transaction_state.state = TRANSACTION_ACTIVE
    // TODO 6: Return true
}

bool transaction_commit(Database* db) {
    // TODO 1: If state != TRANSACTION_ACTIVE, return false
    // TODO 2: Set state = TRANSACTION_COMMITTING
    // TODO 3: If rollback journal:
    //   a. Call journal_commit(db->pager->journal)
    //   b. If successful, flush all dirty pages to main database file via pager_flush_page
    //   c. fsync() the database file
    // TODO 4: If WAL mode:
    //   a. Write all dirty pages to WAL as frames
    //   b. Write a commit frame
    //   c. fsync() the WAL file
    //   d. Release WAL write lock
    // TODO 5: Clear dirty_pages list
    // TODO 6: Set state = TRANSACTION_IDLE
    // TODO 7: Return true
}

bool transaction_rollback(Database* db) {
    // TODO 1: If state != TRANSACTION_ACTIVE, return false
    // TODO 2: Set state = TRANSACTION_ROLLING_BACK
    // TODO 3: If rollback journal:
    //   a. Call journal_rollback(db->pager->journal, db->pager->file_descriptor)
    //   b. Invalidate cached pages that were dirty (they contain uncommitted changes)
    // TODO 4: If WAL mode:
    //   a. Simply release the WAL write lock (uncommitted frames are ignored)
    //   b. Invalidate cached dirty pages
    // TODO 5: Clear dirty_pages list
    // TODO 6: Set state = TRANSACTION_IDLE
    // TODO 7: Return true
}
```

**E. Language-Specific Hints (C)**
- Use `fcntl(fd, F_FULLFSYNC)` on macOS for true durability (if available), or `fsync()` on Linux.
- For file locking in WAL mode, use `fcntl()` with `F_SETLK` / `F_GETLK` for portable advisory locks.
- When reading/writing multi-byte integers in journal/WAL headers, use the provided `store_big_endian_u32` and `load_big_endian_u32` functions to ensure consistent byte order across architectures.
- Manage dynamic arrays for `dirty_pages` using `realloc`. Start with a small initial capacity (e.g., 10).

**F. Milestone Checkpoint (Milestone 9 - Transactions)**
1.  **Test Basic Transaction Cycle**: Run a test program that:
    ```c
    database_execute(db, "BEGIN");
    database_execute(db, "INSERT INTO users VALUES (1, 'Alice')");
    database_execute(db, "COMMIT");
    ```
    Verify that the row is persisted and a subsequent `SELECT` returns it.
2.  **Test Rollback**: Run:
    ```c
    database_execute(db, "BEGIN");
    database_execute(db, "INSERT INTO users VALUES (2, 'Bob')");
    database_execute(db, "ROLLBACK");
    ```
    Verify that Bob is *not* in the table.
3.  **Crash Recovery Test**:
    - Insert a row within a transaction but `kill -9` the process before it commits.
    - Restart your program and open the same database file. The journal file should be detected and the partial insert rolled back automatically (no row added).
    - Verify this by counting rows.

**G. Debugging Tips**
| Symptom | Likely Cause | How to Diagnose | Fix |
| :--- | :--- | :--- | :--- |
| After a crash, the database is empty or corrupted. | Journal file not being `fsync()`'ed before delete, or recovery logic not triggered. | Check if a `-journal` file exists after the crash. Add debug prints to `database_open` to see if it's detected. | Ensure `fsync()` is called after writing the commit record. Ensure recovery routine is called on open. |
| WAL mode: Readers see stale data, not recent commits. | `wal_read_page` not checking WAL, or read lock mechanism broken. | Add logging to `wal_read_page` to see if it's checking the WAL index. Check if reader's read mark is being set correctly. | Ensure the WAL index is updated on write. Ensure readers read the latest WAL header to get the current end. |
| "Database is locked" errors in WAL mode with single writer. | Deadlock in locking protocol, or a lock not released after crash. | Use `lsof` or `fuser` to see which process holds the lock file. Check for missing `close()` or `unlock` in error paths. | Implement a robust lock file with timeouts. Ensure all code paths release locks (use `goto cleanup` pattern). |


## 6. Interactions and Data Flow

> **Milestone(s):** 5 (SELECT Execution), 6 (INSERT/UPDATE/DELETE), 7 (WHERE Clause and Indexes), 8 (Query Planner), 9 (Transactions), 10 (WAL Mode)

This section traces the complete journey of SQL operations through our database engine, showing how the components we've designed work together to transform human-readable SQL into persistent data changes. Think of this as following a package through a logistics system — from order placement (SQL input) to delivery confirmation (result set or commit confirmation), with each component acting as a specialized processing station along the route.

Understanding these data flows is critical for debugging, performance optimization, and appreciating the system's overall architecture. We'll examine two representative scenarios: a read-only `SELECT` query that demonstrates the **volcano model** of pipelined execution, and an `INSERT` within a transaction that showcases **ACID guarantees** via journaling.

### Data Flow: SELECT Query

**Mental Model: The Assembly Line for Information Retrieval**

Imagine a factory assembly line where raw materials (disk pages) enter at one end and finished products (formatted result rows) emerge at the other. Each workstation (operator) performs a specific transformation: the **Table Scan** station fetches raw page data, the **Filter** station discards non-conforming items, and the **Projection** station packages only the requested columns. The conveyor belt moves items one at a time (row-by-row processing), minimizing work-in-progress inventory (memory usage). This is the essence of the **volcano execution model** — a pipelined, iterator-based approach where each operator implements a `next()` method to pull rows from its children.

The SELECT query flow exemplifies the **read path** through our database, involving all major components except transaction journaling. The sequence diagram below illustrates the component interactions:

![SELECT Query Execution Sequence](./diagrams/select-sequence.svg)

Let's trace through a concrete example: `SELECT name, age FROM users WHERE age > 25 ORDER BY name`. We'll walk through each step in detail.

#### Step-by-Step SELECT Execution

1. **SQL Input and Initialization**
   - The user or application calls `database_execute(db, "SELECT name, age FROM users WHERE age > 25")`.
   - The `Database` struct receives the SQL string and passes it to the parser component.

2. **Lexical Analysis and Parsing**
   - `lexer_create()` initializes a `Lexer` with the SQL source string.
   - `lexer_next_token()` is called repeatedly, producing `Token` objects:
     - `TOKEN_SELECT` → `TOKEN_IDENTIFIER("name")` → `TOKEN_IDENTIFIER("age")` → `TOKEN_FROM` → `TOKEN_IDENTIFIER("users")` → `TOKEN_WHERE` → `TOKEN_IDENTIFIER("age")` → `TOKEN_GREATER` → `TOKEN_NUMBER(25)`
   - The `Parser` consumes these tokens via recursive descent, building an `AST`:
     - Root node: `AST_SELECT`
     - Children: Column list (`AST_COLUMN_REF` for "name", "age"), table reference ("users"), WHERE clause (`AST_BINARY_OP` with operator `>` comparing column "age" to literal 25)

3. **Query Planning and Optimization**
   - `create_select_plan()` analyzes the `AST` and database catalog:
     - Checks if table "users" exists in schema catalog (returns error if not)
     - Examines available indexes: if an index exists on `age`, considers index scan
     - Estimates costs: full table scan reads all pages; index scan reads index pages plus table pages for matching rows
     - Based on our simple cost model, chooses the cheaper plan (often table scan for low selectivity)
   - Builds an `ExecutionPlan` tree of `Operator` nodes:
     ```
     ProjectionOperator (columns: name, age index positions)
         ↓
     FilterOperator (predicate: age > 25)
         ↓
     TableScanOperator (table: users)
     ```
   - Each operator is initialized with necessary context: `TableScanOperator` gets a `Cursor` for the users table B-tree; `FilterOperator` gets the parsed WHERE clause `ASTNode`; `ProjectionOperator` gets column indices [0, 2] assuming `name` is column 0, `age` is column 2.

4. **Plan Execution via Volcano Iterator Model**
   - The execution engine calls `next()` on the root operator (`ProjectionOperator`), initiating the pipeline:
   
   1. **Table Scan Layer (`TableScanOperator.next()`)**:
      - Calls `btree_open_cursor()` on the "users" table's B-tree with no specific key (full scan)
      - The `Cursor` is positioned at the first leaf page (minimum rowid)
      - On each `next()` call:
        - `cursor_next()` advances to next cell in current leaf page
        - If end of page, follows right sibling pointer to next leaf page
        - `cursor_get_row()` deserializes current cell into a `Row` structure with `ColumnValue` array
        - Returns `OPERATOR_ROW_READY` with the raw row

   2. **Filter Layer (`FilterOperator.next()`)**:
      - Calls child (`TableScanOperator`) `next()` in a loop
      - For each row received:
        - Creates `ExpressionContext` with current row and table schema
        - Calls `evaluate_predicate(context, where_clause)` which recursively evaluates the `AST_BINARY_OP`
        - If result is `BOOL_TRUE`, passes row up; if `BOOL_FALSE`, continues loop; if `BOOL_NULL`, treats as false
        - Implements **short-circuit evaluation**: for `AND`/`OR` expressions, stops evaluating once outcome determined

   3. **Projection Layer (`ProjectionOperator.next()`)**:
      - Receives filtered row from child operator
      - Extracts only specified columns using precomputed indices
      - Creates new `Row` with subset of `ColumnValue` array
      - Returns `OPERATOR_ROW_READY` with projected row

5. **Result Set Construction and Return**
   - The execution engine collects rows from the pipeline into a `ResultSet`
   - `ResultSet` stores:
     - `rows`: array of projected `Row` pointers
     - `column_names`: ["name", "age"] extracted from original query
     - `row_count`: number of rows matching WHERE clause
   - When pipeline signals `OPERATOR_EOF` (cursor reaches end of table), execution completes
   - `database_execute()` returns the `ResultSet*` to caller

6. **Cleanup**
   - After caller processes results, `ResultSet` is freed, releasing all `Row` memory
   - Each operator's `reset()` function is called if plan might be reused
   - `Cursor` is closed, releasing any page pins in `Pager` cache

#### Key Data Transformations Through the Pipeline

The following table shows how data morphs at each stage for a sample users table with columns [id, name, email, age]:

| Stage | Data Format | Example Transformation | Memory Management |
|-------|-------------|------------------------|-------------------|
| Disk Page | `Page` buffer (4096 bytes) | Raw bytes with `PageHeader`, cell pointers, serialized cells | `Pager` cache manages page buffers |
| Table Scan | `Row` structure | `deserialize_leaf_cell()` converts bytes to `ColumnValue` array: `{1, "Alice", "alice@email.com", 30}` | `Row` allocated per row, freed after projection |
| Filter | `Row` (unchanged format) | `evaluate_predicate()` tests `age > 25` → `BOOL_TRUE` | Row passes through if predicate matches |
| Projection | `Row` (subset columns) | Extracts columns 1 & 3: `{"Alice", 30}` | New `Row` allocated with only needed columns |
| Result Set | `ResultSet` + `Row` array | Array of projected rows plus metadata | Caller responsible for freeing via `resultset_free()` |

#### Common SELECT-Specific Optimizations

Two optimizations occur transparently during execution:

1. **Predicate Pushdown**: The WHERE clause is evaluated as early as possible (immediately after table scan), avoiding unnecessary projection work for filtered-out rows.
2. **Index-Only Scan**: If query uses only indexed columns (e.g., `SELECT age FROM users WHERE age > 25` with index on `age`), the `IndexScanOperator` can satisfy the query directly from index leaf cells without accessing table data.

> **Design Insight:** The volcano model's elegance lies in its modularity. Each operator worries only about its specific transformation and knows nothing about upstream or downstream operators beyond the `next()` interface. This allows arbitrary plan recombination — the same `FilterOperator` works whether its child is a `TableScanOperator`, `IndexScanOperator`, or even another `FilterOperator`.

### Data Flow: INSERT with Transaction

**Mental Model: The Bank Teller Protocol**

Imagine depositing money at a bank. The teller doesn't immediately vault your cash; instead, they:
1. **Begin transaction**: Record your intent in a temporary ledger
2. **Verify constraints**: Check your account exists, validate deposit amount
3. **Prepare changes**: Update account balance in their working copy
4. **Commit**: Only after verifying everything, they make changes permanent by updating the main ledger and discarding temporary records
5. **Rollback (if error)**: If something fails (invalid account, system crash), they tear up the temporary record as if nothing happened.

This protocol ensures **atomicity** (all-or-nothing) and **durability** (survives crashes). Our database's transaction system follows the same pattern using a **rollback journal** (or WAL in advanced mode).

The INSERT-with-transaction flow demonstrates the **write path** with ACID guarantees. Unlike SELECT's read-only flow, this path modifies persistent state and must handle failures gracefully.

#### Step-by-Step INSERT Execution Within Transaction

Let's trace `BEGIN; INSERT INTO users (name, age) VALUES ('Bob', 42); COMMIT;`:

1. **Transaction Initiation (`BEGIN`)**:
   - `transaction_begin(db)` is called
   - `TransactionState` transitions from `TRANSACTION_IDLE` to `TRANSACTION_ACTIVE`
   - `journal_begin_transaction()` creates/opens journal file, writes `JournalHeader` with magic number `JOURNAL_MAGIC`
   - A list of dirty pages (`dirty_pages` array) is initialized to track modified pages

2. **INSERT Statement Processing**:
   - Parser produces `AST_INSERT` node with:
     - Table name: "users"
     - Columns: ["name", "age"]
     - Values: `AST_LITERAL_STRING("Bob")`, `AST_LITERAL_NUMBER(42)`
   - `create_select_plan()` builds `InsertOperator` (not a pipeline operator; executes immediately)
   - Before any modification, constraint checking occurs:
     - NOT NULL constraints: If "name" or "age" were declared NOT NULL, values are present ✓
     - UNIQUE constraints: Check if any unique index would be violated (requires B-tree lookup)
     - Foreign key constraints: If implemented, verify referenced row exists

3. **B-tree Insertion with Journaling**:
   - `InsertOperator.execute()` calls `btree_insert()` with:
     - Key: Next available rowid (or specified PRIMARY KEY)
     - Value: Serialized row data: `{'Bob', 42}` + default values for other columns
   - `btree_insert()` performs B-tree traversal to find correct leaf page
   - **CRITICAL: Before modifying any page**:
     - `journal_page_before_write(journal, page_num, original_data)` saves original page content (4096 bytes) to journal file
     - Page is added to `dirty_pages` list in `TransactionState`
   - B-tree insertion proceeds:
     - If leaf page has space: insert cell, update cell pointers, adjust `PageHeader.cell_count`
     - If leaf page full: split page, create new leaf, redistribute cells, update parent internal page(s) (with journaling for each modified page)
   - New row is now in memory page cache (`Pager.pages[]`) marked dirty

4. **Transaction Commit (`COMMIT`)**:
   - `transaction_commit(db)` is called
   - `TransactionState` transitions to `TRANSACTION_COMMITTING`
   - **Atomicity Protocol (Two-Phase)**:
     1. **Prepare phase**: Journal file is flushed to disk (`fsync()`) ensuring all original page images are durable
     2. **Commit point**: A special commit record is appended to journal and flushed
     3. **Apply phase**: All dirty pages in `dirty_pages` are written to main database file via `pager_flush_page()`
     4. **Cleanup**: Journal file is deleted (or header zeroed), transaction transitions to `TRANSACTION_IDLE`
   - If crash occurs:
     - Before commit point: Journal contains original pages → recovery will restore them
     - After commit point: Journal contains commit record → recovery will reapply changes
     - After cleanup: No journal → transaction was fully committed

5. **Post-Commit Cleanup**:
   - `dirty_pages` list is cleared
   - Page cache remains with modified pages (now identical to disk)
   - Locks (if any) are released
   - Returns success to caller

#### Journaling vs. WAL Mode Comparison

The flow differs significantly when using **Write-Ahead Logging (WAL)** mode (Milestone 10):

| Step | Rollback Journal Mode | WAL Mode |
|------|-----------------------|----------|
| Transaction Start | Create `*-journal` file | Open/continue `*-wal` file |
| Before Page Modify | Copy original page to journal | Nothing (no copy needed) |
| Page Modification | Modify page in cache | Modify page in cache |
| Commit Preparation | Flush journal to disk | Append all modified pages as `WALFrame`s to WAL file |
| Commit Point | Write commit record to journal | Write commit marker to WAL |
| Apply Changes | Write dirty pages to main DB | Nothing (pages stay in WAL) |
| Cleanup | Delete journal file | Update WAL header checkpoint pointer |
| Read Path | Read directly from main DB | Check WAL first for latest page versions |

**WAL Advantage**: Multiple readers can continue reading from main DB while writer appends to WAL. Checkpoint process (`wal_checkpoint()`) periodically moves WAL pages back to main DB.

#### Error Scenarios and Recovery

The transaction system handles various failure modes:

| Failure Point | Detection | Automatic Recovery |
|---------------|-----------|-------------------|
| Crash before commit | Journal exists without commit record | `journal_rollback()` on next open restores original pages from journal |
| Crash during commit | Journal has partial commit record | Treat as uncommitted → rollback |
| Crash after commit | Journal has commit record | `journal_rollback()` reapplies changes (forward recovery) |
| Disk full during write | `write()` returns error | Transaction rolls back, journal cleaned up |
| Constraint violation | Check before journaling | Immediate error, transaction remains active (can retry with new values) |

> **Critical Design Principle:** The **write-ahead rule** — never write changes to the main database file before logging enough information to recover. This rule ensures atomicity even with crashes at arbitrary points.

#### Performance Implications and Write Amplification

Each INSERT involves multiple disk operations:

| Operation | Typical I/O Cost | Notes |
|-----------|-----------------|-------|
| Journal write (each page) | 4KB write | Sequential append (fast on SSDs) |
| Database page write | 4KB write | Random write (slower on HDDs) |
| fsync (commit) | ~10ms | Forces write to persistent storage |
| WAL append (WAL mode) | 4KB write | Sequential, shared for multiple pages |
| Checkpoint (WAL mode) | Random writes | Background process, doesn't block commits |

This **write amplification** (writing data multiple times) is the price of durability. Techniques like **group commit** (batching multiple transactions) and **non-synchronous modes** (`PRAGMA synchronous=OFF`) can optimize at the cost of safety.

#### Concurrency Considerations

Our simple implementation uses **single-writer** semantics (one transaction at a time). However, the data flow supports extension to:

1. **Read-Uncommitted**: Readers see dirty pages in cache
2. **Read-Committed**: Readers see only committed pages (check journal/WAL status)
3. **Snapshot Isolation**: Readers see database state at transaction start (copy of page cache)

The WAL mode naturally supports **concurrent readers** because:
- Writer appends new pages to WAL (sequential writes)
- Readers check WAL index for latest page versions
- Old readers continue using pre-commit WAL frames until they finish

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option (Rollback Journal) | Advanced Option (WAL Mode) |
|-----------|----------------------------------|----------------------------|
| Transaction Journaling | Single journal file per transaction with page snapshots | Circular WAL file with frame headers and checksums |
| Crash Recovery | Rollback journal with forward/backward recovery | WAL replay with checkpoint coordination |
| Concurrency | Exclusive writer lock, readers wait | Writer lock on WAL append, readers check WAL index |
| Durability Guarantee | `fsync()` on journal then database | `fsync()` on WAL then optional database sync |

#### B. Recommended File/Module Structure

```
build-your-own-sqlite/
├── src/
│   ├── main.c                          # REPL entry point
│   ├── database.c/.h                   # Database layer, orchestrates components
│   ├── parser/                         # Milestone 1-2
│   │   ├── lexer.c/.h                  # Tokenization
│   │   └── parser.c/.h                 # AST construction
│   ├── storage/                        # Milestone 3-4
│   │   ├── btree.c/.h                  # B-tree operations
│   │   ├── pager.c/.h                  # Page cache and file I/O
│   │   └── serialization.c/.h          # Row serialization/deserialization
│   ├── execution/                      # Milestone 5-8
│   │   ├── operators.c/.h              # Volcano model operators
│   │   ├── planner.c/.h                # Query planning and cost estimation
│   │   └── expression.c/.h             # Expression evaluation
│   └── transaction/                    # Milestone 9-10
│       ├── journal.c/.h                # Rollback journal implementation
│       ├── wal.c/.h                    # WAL mode implementation
│       └── lock.c/.h                   # Concurrency control (optional)
└── test/
    ├── test_parser.c                   # Milestone 1-2 tests
    ├── test_storage.c                  # Milestone 3-4 tests
    ├── test_execution.c                # Milestone 5-8 tests
    └── test_transaction.c              # Milestone 9-10 tests
```

#### C. Infrastructure Starter Code: Simple Journal Implementation

```c
/* transaction/journal.c - Complete rollback journal implementation */
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include "journal.h"

Journal* journal_open(const char* database_filename) {
    Journal* journal = malloc(sizeof(Journal));
    if (!journal) return NULL;
    
    // Create journal filename: database filename + "-journal"
    size_t len = strlen(database_filename) + 8; // "-journal" + null
    journal->filename = malloc(len);
    snprintf(journal->filename, len, "%s-journal", database_filename);
    
    journal->file_descriptor = -1;
    journal->in_transaction = false;
    journal->page_size = PAGE_SIZE;
    
    return journal;
}

bool journal_begin_transaction(Journal* journal) {
    if (journal->in_transaction) {
        fprintf(stderr, "Transaction already active\n");
        return false;
    }
    
    // Open journal file for writing
    journal->file_descriptor = open(journal->filename, 
                                    O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (journal->file_descriptor < 0) {
        perror("Failed to open journal file");
        return false;
    }
    
    // Write journal header
    JournalHeader header = {
        .magic = JOURNAL_MAGIC,
        .page_size = journal->page_size,
        .reserved = {0}
    };
    
    ssize_t written = write(journal->file_descriptor, &header, sizeof(header));
    if (written != sizeof(header)) {
        perror("Failed to write journal header");
        close(journal->file_descriptor);
        journal->file_descriptor = -1;
        return false;
    }
    
    journal->in_transaction = true;
    return true;
}

bool journal_page_before_write(Journal* journal, uint32_t page_num, 
                              const void* page_data) {
    if (!journal->in_transaction) {
        fprintf(stderr, "No active transaction\n");
        return false;
    }
    
    // Write page number followed by page data
    uint32_t page_num_be = htonl(page_num);
    ssize_t written = write(journal->file_descriptor, &page_num_be, 
                           sizeof(page_num_be));
    if (written != sizeof(page_num_be)) {
        perror("Failed to write page number to journal");
        return false;
    }
    
    written = write(journal->file_descriptor, page_data, journal->page_size);
    if (written != journal->page_size) {
        perror("Failed to write page data to journal");
        return false;
    }
    
    return true;
}

bool journal_commit(Journal* journal) {
    if (!journal->in_transaction) {
        fprintf(stderr, "No active transaction\n");
        return false;
    }
    
    // Write commit record (page number 0 indicates commit)
    uint32_t commit_marker = 0;
    uint32_t commit_marker_be = htonl(commit_marker);
    if (write(journal->file_descriptor, &commit_marker_be, 
              sizeof(commit_marker_be)) != sizeof(commit_marker_be)) {
        perror("Failed to write commit marker");
        return false;
    }
    
    // Ensure journal is durable before applying changes
    if (fsync(journal->file_descriptor) < 0) {
        perror("Failed to sync journal");
        return false;
    }
    
    journal->in_transaction = false;
    
    // Journal will be deleted after database pages are written
    return true;
}

bool journal_rollback(Journal* journal, int db_fd) {
    if (!journal->in_transaction) {
        // Check if journal exists from crashed transaction
        if (access(journal->filename, F_OK) != 0) {
            return true; // No journal, nothing to rollback
        }
        
        journal->file_descriptor = open(journal->filename, O_RDONLY);
        if (journal->file_descriptor < 0) {
            perror("Failed to open journal for recovery");
            return false;
        }
    }
    
    // Read and restore pages from journal
    lseek(journal->file_descriptor, sizeof(JournalHeader), SEEK_SET);
    
    uint32_t page_num_be;
    uint8_t page_buffer[PAGE_SIZE];
    
    while (read(journal->file_descriptor, &page_num_be, sizeof(page_num_be)) 
           == sizeof(page_num_be)) {
        uint32_t page_num = ntohl(page_num_be);
        
        if (page_num == 0) {
            // Commit marker reached
            break;
        }
        
        if (read(journal->file_descriptor, page_buffer, PAGE_SIZE) != PAGE_SIZE) {
            fprintf(stderr, "Truncated journal file\n");
            break;
        }
        
        // Restore page to database file
        lseek(db_fd, page_num * PAGE_SIZE, SEEK_SET);
        write(db_fd, page_buffer, PAGE_SIZE);
    }
    
    // Clean up journal file
    close(journal->file_descriptor);
    journal->file_descriptor = -1;
    unlink(journal->filename);
    
    return true;
}

void journal_close(Journal* journal) {
    if (journal->file_descriptor >= 0) {
        close(journal->file_descriptor);
    }
    free(journal->filename);
    free(journal);
}
```

#### D. Core Logic Skeleton Code: Volcano Iterator Implementation

```c
/* execution/operators.c - Skeleton for volcano model operators */
#include "operators.h"

// Table Scan Operator
Operator* create_table_scan_operator(Database* db, const char* table_name) {
    TableScanOperator* tscan = malloc(sizeof(TableScanOperator));
    if (!tscan) return NULL;
    
    // Initialize base operator
    tscan->base.type = OPERATOR_TABLE_SCAN;
    tscan->base.next = table_scan_next;
    tscan->base.reset = table_scan_reset;
    tscan->base.context = tscan;
    
    // Initialize table-specific fields
    tscan->table_name = strdup(table_name);
    tscan->cursor = NULL;
    tscan->schema = database_get_schema(db, table_name);
    tscan->row_valid = false;
    
    return (Operator*)tscan;
}

// TODO: Implement the next() method for table scan
OperatorResult table_scan_next(Operator* op) {
    TableScanOperator* tscan = (TableScanOperator*)op->context;
    
    // TODO 1: If cursor is NULL, open cursor at start of table
    //   Call btree_open_cursor() with NULL key for full table scan
    
    // TODO 2: If row_valid is true from previous call, free previous row
    //   This happens when parent operator didn't consume the row
    
    // TODO 3: Advance cursor using cursor_next()
    //   If cursor_next() returns false (end of table), return OPERATOR_EOF
    
    // TODO 4: Get current row using cursor_get_row()
    //   This deserializes B-tree cell into Row structure
    
    // TODO 5: Set row_valid to true and return OPERATOR_ROW_READY
    
    return OPERATOR_ERROR;
}

// Filter Operator
Operator* create_filter_operator(Operator* child, ASTNode* predicate, 
                                Schema* schema) {
    FilterOperator* filter = malloc(sizeof(FilterOperator));
    if (!filter) return NULL;
    
    filter->base.type = OPERATOR_FILTER;
    filter->base.next = filter_next;
    filter->base.reset = filter_reset;
    filter->base.context = filter;
    
    filter->child = child;
    filter->predicate = predicate;
    filter->eval_ctx = create_expression_context(schema);
    filter->current_row = NULL;
    
    return (Operator*)filter;
}

// TODO: Implement filter's next() method
OperatorResult filter_next(Operator* op) {
    FilterOperator* filter = (FilterOperator*)op->context;
    
    // TODO 1: Loop until we find a row matching predicate or reach end
    while (true) {
        // TODO 2: Get next row from child operator using child->next()
        //   If child returns OPERATOR_EOF, return OPERATOR_EOF
        
        // TODO 3: Set current_row in expression context for evaluation
        //   filter->eval_ctx->current_row = child's row
        
        // TODO 4: Evaluate predicate using evaluate_predicate()
        //   This recursively evaluates the WHERE clause AST
        
        // TODO 5: Check result:
        //   - If BOOL_TRUE: keep row, return OPERATOR_ROW_READY
        //   - If BOOL_FALSE: free row, continue loop
        //   - If BOOL_NULL: treat as false (SQL semantics), free row, continue
    }
}

// Projection Operator
Operator* create_projection_operator(Operator* child, 
                                    const char** column_names, 
                                    uint32_t column_count,
                                    Schema* schema) {
    ProjectionOperator* proj = malloc(sizeof(ProjectionOperator));
    if (!proj) return NULL;
    
    proj->base.type = OPERATOR_PROJECTION;
    proj->base.next = projection_next;
    proj->base.reset = projection_reset;
    proj->base.context = proj;
    
    proj->child = child;
    
    // TODO 1: Map column names to indices in schema
    //   proj->column_indices = malloc(column_count * sizeof(int))
    //   For each column name, find its position in schema->columns
    
    proj->column_count = column_count;
    
    return (Operator*)proj;
}

// TODO: Implement projection's next() method
OperatorResult projection_next(Operator* op) {
    ProjectionOperator* proj = (ProjectionOperator*)op->context;
    
    // TODO 1: Get next row from child operator
    //   If child returns OPERATOR_EOF, return OPERATOR_EOF
    
    // TODO 2: Create new Row structure for projected columns
    //   Row* projected = malloc(sizeof(Row))
    //   projected->column_count = proj->column_count
    
    // TODO 3: For each column index in proj->column_indices
    //   Copy ColumnValue from child's row to projected row
    //   Note: For TEXT/BLOB types, need to copy data, not just pointer
    
    // TODO 4: Free original row from child (projection consumes it)
    
    // TODO 5: Return OPERATOR_ROW_READY with projected row
    return OPERATOR_ERROR;
}

// Execute SELECT plan
ResultSet* execute_select_plan(ExecutionPlan* plan) {
    ResultSet* result = malloc(sizeof(ResultSet));
    if (!result) return NULL;
    
    result->rows = NULL;
    result->row_count = 0;
    result->current_pos = 0;
    
    // TODO 1: Allocate initial row array (e.g., 16 rows)
    //   result->rows = malloc(16 * sizeof(Row*))
    
    // TODO 2: Execute pipeline using volcano model
    //   while (plan->root_operator->next() == OPERATOR_ROW_READY) {
    //     Get row from operator
    //     Add to result->rows array (realloc if needed)
    //     result->row_count++
    //   }
    
    // TODO 3: Handle OPERATOR_EOF (normal termination)
    
    // TODO 4: Handle OPERATOR_ERROR (clean up and return NULL)
    
    return result;
}
```

#### E. Language-Specific Hints (C)

1. **Memory Management**: 
   - Use `malloc()`/`free()` consistently. Every allocation needs a corresponding free.
   - For complex structures, implement `type_free()` functions that recursively free nested members.

2. **Error Handling**:
   - Return `bool` for success/failure, with error messages in global/context variables.
   - Use `errno` for system call errors, `perror()` for logging.

3. **File I/O**:
   - Always check return values of `read()`, `write()`, `fsync()`.
   - Use `O_DIRECT` flag for aligned I/O if implementing advanced pager.

4. **Serialization**:
   - Use `htonl()`, `ntohl()` for portable integer serialization.
   - For variable-length data, store length prefix before value.

5. **Debugging**:
   - Compile with `-g -fsanitize=address` for memory error detection.
   - Use `gdb` with breakpoints at operator `next()` functions to trace execution.

#### F. Milestone Checkpoint for SELECT Flow

**After implementing Milestone 5 (SELECT Execution):**

1. **Test Command**: 
   ```bash
   ./sqlite test.db "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);"
   ./sqlite test.db "INSERT INTO users VALUES (1, 'Alice', 30);"
   ./sqlite test.db "INSERT INTO users VALUES (2, 'Bob', 25);"
   ./sqlite test.db "SELECT name, age FROM users WHERE age > 20;"
   ```

2. **Expected Output**:
   ```
   name|age
   Alice|30
   Bob|25
   ```

3. **Verification Steps**:
   - Check that rows are returned in rowid order (1 then 2).
   - Verify column projection works: only `name` and `age` shown, not `id`.
   - Test edge cases: `SELECT * FROM users` returns all columns.
   - Test empty result: `SELECT * FROM users WHERE age > 100` returns header only.

4. **Debug Signs**:
   - **No rows returned**: Check `cursor_next()` logic and page parsing.
   - **Wrong columns shown**: Verify column indices in projection.
   - **Memory leaks**: Use `valgrind` or AddressSanitizer.
   - **Infinite loop**: Ensure `OPERATOR_EOF` is properly returned.

#### G. Debugging Tips for Data Flow Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| SELECT returns no rows but INSERT succeeded | Cursor not advancing past first page | Add debug prints in `cursor_next()` showing page transitions | Ensure right sibling pointer followed when cell_index >= cell_count |
| WHERE clause filtering incorrect rows | Expression evaluation error | Test `evaluate_predicate()` independently with test rows | Check operator precedence and NULL handling in expression evaluator |
| Memory usage grows unbounded during large SELECT | Rows not freed in pipeline | Trace `Row` allocations with `malloc` hooks or ASan | Ensure each operator frees rows it consumes before fetching next |
| INSERT works but SELECT doesn't see new data | Dirty pages not flushed | Check if `Pager` marks pages dirty on modification | Call `pager_flush_page()` on commit or when cache full |
| Transaction commit leaves database corrupted | Journal not synced before applying changes | Check journal file contents after crash simulation | Ensure `fsync()` on journal before writing to main database |
| WAL mode: Readers see stale data | WAL index not checked | Verify `wal_read_page()` is called in `pager_get_page()` | Update reader's WAL frame pointer on transaction boundaries |

---


## 7. Error Handling and Edge Cases

> **Milestone(s):** All (Error handling is cross-cutting and relevant to every milestone)

Robust error handling is what separates a toy implementation from a production-ready database. Unlike applications that can simply crash, a database must handle failures gracefully while preserving data integrity and providing clear diagnostics. This section defines the systematic approach to managing errors and edge cases across all components, ensuring the database remains reliable even when faced with malformed inputs, hardware failures, or unexpected conditions.

### Error Categories and Recovery

Think of error handling in a database as a **multi-layer defense system**. Each layer has specific responsibilities for detecting, containing, and recovering from different types of failures. The outermost layer handles user input errors, middle layers manage runtime conditions, and the innermost layer deals with catastrophic failures requiring recovery procedures.

> **Key Design Principle:** **Fail fast, recover gracefully.** Detect errors as early as possible to minimize damage, but provide recovery paths for runtime failures to avoid data loss.

#### 1. User Input Errors (SQL Level)

These errors originate from malformed or invalid SQL statements and should be caught during parsing or early execution phases.

| Failure Mode | Detection Method | Recovery Strategy | Error Message Format |
|--------------|------------------|-------------------|---------------------|
| **Syntax Error** | `Parser` fails to match grammar rules during `parse_statement()` | Return error to caller, no state changes | `"Syntax error at position X: unexpected token 'Y'"` |
| **Semantic Error** | `Query Execution Engine` validates AST against schema during `create_select_plan()` | Return error before any execution | `"Table 'users' does not exist"` or `"Column 'email' not found in table 'users'"` |
| **Type Mismatch** | Expression evaluator during `evaluate_expression()` detects incompatible types | Return error during predicate evaluation | `"Type mismatch: cannot compare INTEGER with TEXT"` |
| **Constraint Violation** | `btree_insert()` or `btree_update()` checks constraints before modification | Roll back current operation, return error | `"NOT NULL constraint failed: users.email"` or `"UNIQUE constraint failed: users.username"` |

**Recovery Protocol:** These errors are **non-destructive** and require no recovery beyond cleaning up temporary memory. The transaction (if active) remains in `TRANSACTION_ACTIVE` state, allowing the user to fix and retry the statement.

#### 2. Runtime Resource Errors

These errors occur when the system encounters resource limitations during normal operation.

| Failure Mode | Detection Method | Recovery Strategy | Cleanup Required |
|--------------|------------------|-------------------|------------------|
| **Out of Memory** | `malloc()` or `calloc()` returns `NULL` during allocation | Immediately roll back current operation, free allocated memory, return error | Free any partially allocated structures in reverse order |
| **Disk Full** | `write()` system call returns `ENOSPC` during `pager_flush_page()` or journal operations | If in transaction: initiate rollback via `journal_rollback()`. If not: return error and mark database as read-only | Close file handles, flush any pending writes that succeeded |
| **File Not Found** | `open()` returns `ENOENT` during `database_open()` | Return error, cannot proceed | None (database not opened) |
| **Permission Denied** | `open()` or `write()` returns `EACCES` | Return error, abort operation | Close any opened file descriptors |

**Recovery Protocol:** For disk-full conditions during a transaction, the system must use the **rollback journal** to restore the database to its pre-transaction state. The journal provides the original pages needed for backward recovery. After successful rollback, the database returns to `TRANSACTION_IDLE` state with an error message indicating the disk is full.

#### 3. Data Corruption and Integrity Errors

These are the most serious errors, indicating that on-disk data structures have become inconsistent.

| Failure Mode | Detection Method | Recovery Strategy | Severity Level |
|--------------|------------------|-------------------|----------------|
| **Page Checksum Mismatch** | `page_deserialize()` verifies stored checksum against computed checksum | Mark page as corrupted, attempt to read from `WAL` if in WAL mode, otherwise return error | High (requires manual repair) |
| **B-tree Invariant Violation** | `btree_insert()` detects improper key ordering or invalid child pointers | Abort current operation, return error, suggest `VACUUM` operation | Medium (tree may need rebalancing) |
| **Journal/WAL Corruption** | `journal_open()` or `wal_read_page()` detects invalid magic numbers or checksums | If in recovery: ignore corrupted journal, database remains as-is. If active: abort transaction | Critical (may require backup restoration) |
| **Schema Catalog Inconsistency** | `database_open()` detects mismatched system table structures | Attempt to rebuild schema from table roots, log warning | Medium (automatic recovery attempted) |

**Recovery Protocol:** Corruption detection follows a **defensive hierarchy**:
1. First, check if the corrupted page exists in the `WAL` (if WAL mode is active)
2. If not, check if a valid backup exists in the rollback journal
3. If neither, mark the page as corrupted in memory to prevent further reads/writes
4. Return a detailed error to the user with the page number and corruption type

> **Critical Insight:** **Never silently ignore corruption.** Always propagate corruption errors to the user while preventing further damage through read-only mode or controlled shutdown.

#### 4. Transaction and Concurrency Errors

These errors arise from concurrent access attempts or transaction state violations.

| Failure Mode | Detection Method | Recovery Strategy | User Impact |
|--------------|------------------|-------------------|-------------|
| **Deadlock** | Lock manager detects circular wait (if implemented) | Choose victim transaction, roll it back via `transaction_rollback()`, allow others to proceed | Victim receives error, must retry |
| **Transaction State Error** | `transaction_commit()` called without active transaction | Return error, no state change | User must begin new transaction |
| **Write Conflict** | WAL mode: concurrent write attempted while another holds write lock | Block until lock available or timeout | Timeout returns busy error |
| **Rollback Journal Already Exists** | `journal_begin_transaction()` finds existing journal from crashed transaction | Perform **hot journal recovery** before starting new transaction | Automatic recovery, user may see delay |

**Recovery Protocol:** The **hot journal recovery** process is critical for handling crashes during transactions:
1. On `database_open()`, check for existence of `-journal` file
2. If journal exists and has valid header (`JOURNAL_MAGIC`), database was not cleanly closed
3. Apply `journal_rollback()` to restore all pages modified in that transaction
4. Delete the journal file after successful recovery
5. Open database normally

#### 5. Internal Consistency Errors (Assertions)

These are programming errors or impossible conditions that indicate bugs in the database implementation itself.

| Failure Mode | Detection Method | Response Strategy | Debugging Aid |
|--------------|------------------|-------------------|---------------|
| **Invalid Page Type** | `btree_open_cursor()` receives page with type not `PAGE_TYPE_LEAF` or `PAGE_TYPE_INTERNAL` | Log error with page number, abort operation | Include stack trace in debug builds |
| **Negative Free Space** | `page_deserialize()` calculates free space as negative value | Mark page as corrupted, prevent further use | Dump page hex contents to log |
| **Cursor Stack Overflow** | `Cursor.stack_depth` exceeds maximum (10) during tree traversal | Return error, close cursor | Log traversal path (page numbers) |

**Recovery Protocol:** Internal errors trigger a **fail-safe shutdown**:
1. Log detailed diagnostic information (page contents, stack trace if available)
2. Roll back any active transaction using emergency rollback path
3. Close all file descriptors
4. Return error code indicating internal error

### Key Edge Cases

Edge cases are unusual but valid scenarios that must be handled correctly to ensure robustness. These often reveal subtle bugs in naive implementations.

#### 1. SQL Parsing and Lexing Edge Cases

| Edge Case | Description | Correct Handling | Common Mistake |
|-----------|-------------|------------------|----------------|
| **Escaped Quotes in Strings** | `'It''s raining'` should parse as single string with embedded single quote | Lexer replaces doubled quotes with single quote in token value | Treating `''` as empty string followed by new string |
| **Unicode Identifiers** | Table/column names can contain non-ASCII characters: `SELECT café FROM places` | Tokenizer accepts Unicode letters in identifiers (beyond ASCII) | Restricting to `[A-Za-z0-9_]` only |
| **Mixed Case Keywords** | `SeLeCt * FrOm tAbLe` is valid SQL | Case-insensitive keyword matching during tokenization | Case-sensitive matching requiring exact uppercase |
| **Exponential Notation** | `1.23e-4` should parse as floating point literal | Lexer state machine handles `[eE][+-]?[0-9]+` exponent part | Treating `e` as identifier character |
| **Trailing Commas** | `SELECT a, b, FROM table` has syntax error but `SELECT a, b,` (no FROM) is invalid | Parser detects missing required clause after comma | Incorrectly allowing or rejecting based on position |

#### 2. B-tree and Storage Edge Cases

| Edge Case | Description | Correct Handling | Implementation Challenge |
|-----------|-------------|------------------|-------------------------|
| **Empty Tree** | Table with zero rows still has root page | Root page exists as leaf with `cell_count = 0` | Forgetting to allocate initial root page |
| **Single Key Overflow** | Value exceeds `PAGE_SIZE - header_size`, requires overflow pages | Store first page of value in cell, overflow page numbers in continuation | Treating as error instead of implementing overflow chain |
| **Maximum Depth** | B-tree grows beyond cursor stack capacity (10 levels) | Reject insertion with "tree too deep" error or implement rebalancing | Stack overflow during traversal |
| **Concurrent Modification During Scan** | Row deleted by another transaction while cursor iterating | Cursor maintains snapshot via copy-on-write or MVCC | Returning deleted row or skipping next row |
| **Page Fragmentation** | Repeated inserts/deletes create small free gaps unusable for new cells | Implement freeblock coalescing during insertion | Page appears to have free space but insertion fails |

#### 3. Type System and Expression Edge Cases

| Edge Case | Description | Correct Handling | Three-Valued Logic Result |
|-----------|-------------|------------------|---------------------------|
| **NULL Comparisons** | `NULL = NULL`, `NULL > 5`, `NULL AND TRUE` | All comparisons with `NULL` return `BOOL_NULL` | `NULL = NULL` → `NULL` (not `TRUE`) |
| **Implicit Type Conversion** | `'123' > 100` compares string to integer | Convert string to integer if possible, else return `BOOL_NULL` | `'123' > 100` → `TRUE`, `'abc' > 100` → `NULL` |
| **Division by Zero** | `salary / 0` in SELECT or WHERE clause | Return `NULL` for arithmetic error | `10 / 0` → `NULL` (not crash) |
| **Integer Overflow** | `9223372036854775807 + 1` exceeds 64-bit signed | Cap at maximum or return `NULL` | Implement saturation arithmetic or error |
| **Floating Point Equality** | `0.1 + 0.2 = 0.3` yields false due to precision | Use epsilon comparison for equality (`abs(a-b) < 1e-12`) | Exact comparison gives incorrect results |

#### 4. Transaction and Crash Recovery Edge Cases

| Edge Case | Description | Correct Handling | Atomicity Guarantee |
|-----------|-------------|------------------|---------------------|
| **Torn Page** | Power failure during page write leaves half-old, half-new content | Journal stores complete original page; recovery restores full page | Page is atomically restored to pre-transaction state |
| **Journal Written but Not Committed** | Journal pages saved, commit record not written before crash | Recovery sees incomplete transaction, performs rollback | Transaction fully rolled back (atomicity preserved) |
| **WAL Wrap-Around** | WAL file grows very large, checkpoint cannot keep up | Force checkpoint when WAL exceeds threshold, block new writes | Maintain read consistency during checkpoint |
| **Nested Transactions** | User issues `BEGIN` when already in transaction | Reject with "cannot start transaction within transaction" error | Simple implementation avoids complexity |
| **Crash During Checkpoint** | System fails while copying pages from WAL to main database | Next open sees partial checkpoint, continues from last complete checkpoint | WAL maintains consistency through checksums |

#### 5. File System and Operating System Edge Cases

| Edge Case | Description | Correct Handling | Impact on Durability |
|-----------|-------------|------------------|----------------------|
| **Filesystem Full Mid-Transaction** | Journal pages written, but database page write fails with ENOSPC | Roll back using journal (which is on same filesystem!) | May fail if journal also cannot write; critical design flaw |
| **Database on Network Drive** | `fsync()` may not guarantee durability over network | Use `fdatasync()` and accept weaker guarantees or warn user | Durability not guaranteed without proper network filesystem support |
| **File Permission Change** | Database file becomes read-only after opening | Next write attempt fails; roll back transaction if active | Graceful error vs. crash |
| **Symbolic Link Attack** | Database file replaced by symlink to sensitive file | Check file inode after opening, refuse to follow symlinks | Security vulnerability if not handled |

#### 6. Query Execution Edge Cases

| Edge Case | Description | Correct Handling | Performance Consideration |
|-----------|-------------|------------------|---------------------------|
| **Large Result Set** | `SELECT * FROM big_table` returns millions of rows | Stream results using cursor, avoid materializing all in memory | Memory exhaustion if materialized |
| **Self-Join** | `SELECT * FROM t1, t1` (cartesian product with itself) | Assign table aliases automatically or require explicit aliases | Infinite loop without alias distinction |
| **Correlated Subquery** | `SELECT * FROM t1 WHERE id IN (SELECT t1_id FROM t2 WHERE t2.val > t1.val)` | Naive execution re-evaluates subquery for each outer row; optimize with joins | Extremely poor performance if not optimized |
| **Index on Expression** | `CREATE INDEX idx ON users(UPPER(name))` | Not supported in basic implementation; reject with clear error | Would require storing function results in index |
| **Vacuum After Massive Delete** | 90% of rows deleted, leaving sparse pages | Implement `VACUUM` command to rebuild compact storage | B-tree remains sparse but correct |

### Implementation Guidance

Error handling in C requires careful attention to resource cleanup and state consistency. Unlike garbage-collected languages, C programs must explicitly free resources on all code paths, including error paths.

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Reporting | Return error codes, store message in context object | Structured error objects with codes, messages, and metadata |
| Memory Management | Manual `malloc/free` with cleanup goto labels | Arena allocators for transaction-scoped memory |
| File I/O Error Detection | Check return values of `read/write/fsync` | Use `ferror()` and `errno` for detailed diagnostics |
| Corruption Detection | Basic checksums (CRC32) on pages | Cryptographic hashes (SHA-256) for critical structures |

#### B. Error Handling Infrastructure

Create a consistent error handling pattern across all components:

**Error Code Enumeration:**
```c
typedef enum {
    ERR_OK = 0,
    ERR_SYNTAX,
    ERR_SEMANTIC,
    ERR_IO,
    ERR_NOMEM,
    ERR_CORRUPT,
    ERR_CONSTRAINT,
    ERR_INTERNAL,
    ERR_BUSY,
    ERR_LOCKED
} ErrorCode;
```

**Error Context Structure:**
```c
typedef struct {
    ErrorCode code;
    char message[256];
    const char* component;
    int line;
    const char* file;
} ErrorContext;

// Global thread-local error context
__thread ErrorContext g_last_error = {0};
```

#### C. Resource Cleanup Pattern

The classic C pattern for cleanup on error uses `goto` labels:

```c
int process_statement(Database* db, const char* sql) {
    AST* ast = NULL;
    ExecutionPlan* plan = NULL;
    ResultSet* rs = NULL;
    int ret = ERR_OK;
    
    ast = parse_statement(sql);
    if (ast == NULL) {
        ret = ERR_SYNTAX;
        goto cleanup;
    }
    
    plan = create_select_plan(db, ast);
    if (plan == NULL) {
        ret = ERR_SEMANTIC;
        goto cleanup;
    }
    
    rs = execute_plan(db, plan);
    if (rs == NULL) {
        ret = ERR_INTERNAL;
        goto cleanup;
    }
    
    // Process result set...
    
cleanup:
    if (rs) destroy_result_set(rs);
    if (plan) destroy_execution_plan(plan);
    if (ast) destroy_ast(ast);
    
    if (ret != ERR_OK) {
        // Log error if needed
        fprintf(stderr, "Error: %s\n", get_last_error_message());
    }
    
    return ret;
}
```

#### D. Core Error Handling Skeleton

**Error Reporting Helper:**
```c
// TODO 1: Set error code in thread-local context
// TODO 2: Format error message with vsnprintf for safety
// TODO 3: Record component name and location (__FILE__, __LINE__)
// TODO 4: Return error code for convenience in return statements
ErrorCode set_error(ErrorCode code, const char* component, 
                    const char* file, int line, const char* fmt, ...) {
    // Implementation
}

// Macro for convenience
#define SET_ERROR(code, ...) \
    set_error(code, __func__, __FILE__, __LINE__, __VA_ARGS__)
```

**Memory Allocation with Error Handling:**
```c
// TODO 1: Attempt allocation with malloc/calloc
// TODO 2: If allocation fails, set ERR_NOMEM error
// TODO 3: Return NULL on failure (caller checks)
void* db_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL && size > 0) {
        SET_ERROR(ERR_NOMEM, "Failed to allocate %zu bytes", size);
    }
    return ptr;
}

// TODO 1: Check for NULL pointer before freeing
// TODO 2: Set pointer to NULL after freeing to prevent double-free
// TODO 3: No error on free (nothing to do if memory already gone)
void db_free(void** ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
```

**File I/O with Comprehensive Error Checking:**
```c
// TODO 1: Attempt file open with specified flags
// TODO 2: Check for -1 return value indicating error
// TODO 3: Use errno to determine specific error type
// TODO 4: Map POSIX errors to database error codes
// TODO 5: Set descriptive error message including filename
int db_open_file(const char* filename, int flags) {
    int fd = open(filename, flags, 0644);
    if (fd < 0) {
        ErrorCode code = ERR_IO;
        switch (errno) {
            case ENOENT: code = ERR_IO; break;
            case EACCES: code = ERR_IO; break;
            case ENOSPC: code = ERR_NOMEM; break; // Treat as memory error
        }
        SET_ERROR(code, "Failed to open '%s': %s", 
                 filename, strerror(errno));
    }
    return fd;
}

// TODO 1: Attempt write operation
// TODO 2: Check for partial writes (return value < requested size)
// TODO 3: Retry on EINTR (interrupted system call)
// TODO 4: Handle ENOSPC (disk full) specially for recovery
ssize_t db_write_full(int fd, const void* buf, size_t count) {
    ssize_t total_written = 0;
    while (total_written < count) {
        ssize_t written = write(fd, (char*)buf + total_written, 
                               count - total_written);
        if (written < 0) {
            if (errno == EINTR) continue; // Interrupted, retry
            if (errno == ENOSPC) {
                SET_ERROR(ERR_NOMEM, "Disk full while writing");
            } else {
                SET_ERROR(ERR_IO, "Write failed: %s", strerror(errno));
            }
            return -1;
        }
        if (written == 0) {
            SET_ERROR(ERR_IO, "Write returned 0 bytes unexpectedly");
            return -1;
        }
        total_written += written;
    }
    return total_written;
}
```

**Transaction Error Recovery Skeleton:**
```c
// TODO 1: Check if journal file exists on startup
// TODO 2: Validate journal header magic and checksum
// TODO 3: If valid, perform rollback of all pages in journal
// TODO 4: Delete journal file after successful recovery
// TODO 5: If recovery fails, mark database as corrupted
bool recover_from_crash(const char* db_filename) {
    char journal_filename[1024];
    snprintf(journal_filename, sizeof(journal_filename), 
             "%s-journal", db_filename);
    
    // TODO: Check if journal exists
    // TODO: Open journal and read header
    // TODO: Verify JOURNAL_MAGIC
    // TODO: For each page in journal, read original content
    // TODO: Write original content back to database file at correct offset
    // TODO: Sync database file
    // TODO: Delete journal file
    // TODO: Return success/failure
    
    return false; // Placeholder
}
```

**Corruption Detection Helper:**
```c
// TODO 1: Compute simple checksum (CRC32 or Adler-32) of page data
// TODO 2: Store checksum in page header (reserved field)
// TODO 3: On read, verify checksum matches
// TODO 4: On mismatch, attempt recovery from WAL/journal
// TODO 5: If no recovery possible, mark page as bad and error
uint32_t compute_page_checksum(const void* page_data, size_t page_size) {
    // Simple XOR checksum for illustration
    const uint32_t* words = (const uint32_t*)page_data;
    uint32_t checksum = 0;
    for (size_t i = 0; i < page_size / sizeof(uint32_t); i++) {
        checksum ^= words[i];
    }
    return checksum;
}
```

#### E. Language-Specific Hints for C

1. **Always check return values** of system calls and library functions. C doesn't throw exceptions.
2. **Use `errno` immediately** after a failed system call, as it may be overwritten by subsequent calls.
3. **Initialize all structs** to zero using `memset` or `calloc` to avoid undefined behavior.
4. **Prevent integer overflow** in calculations, especially when computing buffer sizes.
5. **Use `size_t` for sizes** and array indices to avoid signed/unsigned mismatch.
6. **Validate input parameters** at function entry, especially pointer parameters.
7. **Document preconditions and postconditions** for each function in comments.
8. **Test error paths** as rigorously as normal paths; many bugs hide in error handling code.

#### F. Debugging Error Handling

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Database file grows uncontrollably | Journal not deleted after crash recovery | Check for leftover `-journal` files on startup | Implement proper hot journal recovery |
| "Disk full" error but disk has space | Filesystem metadata exhaustion (inodes) | Check `df -i` for inode usage | Implement cleanup of temporary files |
| Random corruption after many transactions | Memory leak overwriting valid memory | Use Valgrind to detect out-of-bounds writes | Add bounds checking to array accesses |
| Error message shows wrong line number | `__LINE__` macro in wrong place in macro | Check macro expansion with `gcc -E` | Use `__LINE__` directly, not through nested macros |
| Database works in debug but crashes in release | Uninitialized variables optimized away | Compile with `-O0 -g` and use AddressSanitizer | Initialize all variables explicitly |


## 8. Testing Strategy
> **Milestone(s):** All (Testing is cross‑cutting and must be applied at every development stage)

This section defines a **systematic testing approach** for building a reliable SQLite implementation. Unlike applications where you can test in‑memory behavior alone, a database engine must validate both **correct logical behavior** and **correct persistent state** across crashes. The strategy adopts a **fail‑fast philosophy** — catching bugs immediately at the component level through unit tests, then validating integration through end‑to‑end scenarios, and finally stress‑testing with real‑world workloads. Each milestone includes concrete checkpoints to validate functionality before proceeding, preventing cascading failures that would be difficult to debug later.

### Testing Approach and Tools

> **Mental Model: The Construction Inspector**  
> Think of testing as a construction inspector visiting a building site at each phase of construction. The inspector doesn't wait until the entire skyscraper is built to check the foundation—they visit after each major component (foundation, framing, plumbing, electrical) and verify it meets specifications *before* the next layer depends on it. Similarly, our testing strategy validates each architectural layer (parser, storage, execution, transactions) in isolation *before* testing their integration, ensuring any defect is caught at its source rather than manifesting as mysterious behavior layers above.

Our testing approach follows a **layered pyramid** with increasing scope and decreasing speed:

![System Component Diagram](./diagrams/sys-component.svg)

**Layer 1: Unit Tests (Fastest, Most Isolated)**  
Each component is tested in isolation with mocked dependencies. These tests run in milliseconds and validate the internal logic of individual functions and data structures.

| Test Type | Target | Tools/Techniques | Frequency |
|-----------|--------|------------------|-----------|
| **Parser Unit Tests** | `Lexer`, `Parser`, `AST` functions | Direct function calls with string inputs, validate token/AST output | After every change |
| **B‑tree Unit Tests** | `Page` serialization, `BTree` operations | In‑memory page buffers, verify binary layout | After every change |
| **Execution Unit Tests** | Individual `Operator` implementations | Mock `Cursor` and `Row` inputs, verify output rows | After every change |
| **Transaction Unit Tests** | `Journal`, `WAL` format handling | Temporary files, verify journal/WAL file contents | Before integration |

**Key testing techniques for unit tests:**
1. **Property‑based testing**: For components like `varint_encode`/`varint_decode`, test that encode‑decode round‑trips preserve values for all integers 0‑2⁶³‑1.
2. **Fuzz testing**: For the parser, feed random byte sequences to `lexer_next_token` to ensure no buffer overflows or crashes.
3. **Boundary testing**: For B‑tree operations, test minimum/maximum key sizes, empty pages, and completely full pages.
4. **Error injection**: Simulate `ENOSPC` (disk full) errors in `pager_flush_page` to verify graceful error recovery.

**Layer 2: Integration Tests (Medium Scope)**  
These tests validate interactions between components, such as the full pipeline from SQL string to B‑tree modification.

| Integration Point | What's Tested | Setup Required |
|-------------------|---------------|----------------|
| **Parser → Execution** | `parse_statement` → `execute_select` with simple queries | In‑memory database with test schema |
| **Storage → Execution** | `btree_insert` followed by `execute_select` returns inserted data | Temporary file database |
| **Transaction → Storage** | `transaction_begin` → `btree_insert` → `transaction_rollback` restores original state | Temporary file with journal |
| **WAL → Concurrent Access** | Writer in WAL mode with active transaction, readers see consistent snapshot | Multiple process/thread test harness |

**Layer 3: End‑to‑End Tests (Slowest, Full System)**  
These tests mimic real user scenarios by executing complete SQL scripts and verifying both results and persistent state.

| Scenario Category | Example Test | Validation Method |
|-------------------|--------------|-------------------|
| **SQL Compliance** | All SQL‑92 subset statements our engine supports | Compare results with SQLite3 on same inputs |
| **Crash Recovery** | Kill process mid‑transaction, restart, verify atomicity | Use `kill -9` simulation, check database consistency |
| **Concurrency** | Multiple readers while writer commits in WAL mode | Thread‑based test with synchronization barriers |
| **Performance** | Insert 100,000 rows, measure time and disk space | Benchmark against baseline with performance regression detection |

**Tools and Infrastructure:**

| Tool/Component | Purpose | Rationale |
|----------------|---------|-----------|
| **C Test Framework** | `check` (https://libcheck.github.io/check/) | Lightweight, supports fixture setup/teardown, test suites, XML output |
| **File‑Based Test Harness** | Custom `test_harness.c` | Manages temporary database files, cleans up after tests, isolates test state |
| **SQLite3 Reference** | System SQLite3 binary | Golden‑master comparison for query results |
| **Memory Leak Detection** | Valgrind (memcheck) | Detects `db_malloc`/`db_free` mismatches in test runs |
| **Coverage Analysis** | gcov/lcov | Identifies untested code paths for each milestone |
| **Fuzzing Engine** | AFL (American Fuzzy Lop) | Discovers crashes in parser and page deserialization |

> **Key Insight: The Golden‑Master Approach**  
> For SQL correctness testing, we use SQLite3 itself as a "golden master"—we execute each test query against both our implementation and SQLite3, then compare results. This automatically validates our engine against decades of refinement. Discrepancies are either bugs in our implementation or intentional differences in our supported SQL subset.

**ADR: Unit‑Test‑First vs. Integration‑Test‑First**

**Decision: Unit‑Test‑First with Component Isolation**

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| **Unit‑Test‑First** | Catches bugs early at component level; Fast feedback (seconds); Clearer failure isolation; Enforces modular design | Can miss integration issues; Requires careful mocking; May over‑specify internal behavior | **Yes** |
| **Integration‑Test‑First** | Tests real usage patterns; Less mocking overhead; Validates component interactions | Slow feedback (minutes); Difficult to debug failures; May allow component‑level bugs to persist | No |
| **Hybrid Approach** | Balance of fast unit and comprehensive integration tests | More complex test infrastructure; Risk of duplicated test coverage | Partial (we add integration tests after unit) |

**Rationale**: For a complex system like a database, component‑level correctness is paramount—a bug in `page_deserialize` could corrupt every query. Unit tests provide the rapid feedback loop needed for iterative development. Integration tests are added once components are individually verified, ensuring they work together correctly.

**Consequences**: Developers must write mock implementations for dependencies (e.g., a `MockPager` that simulates disk I/O in memory). This upfront cost pays off through faster debugging and more reliable components.

### Milestone Checkpoints

Each milestone includes specific verification steps. The checkpoints progress from **manual verification** (early milestones) to **automated test suites** (later milestones).

**Milestone 1: SQL Tokenizer**

| Test Category | Specific Tests to Write | Expected Outcome |
|---------------|------------------------|------------------|
| **Keyword Recognition** | `SELECT`, `FROM`, `WHERE` in upper, lower, mixed case | Correct `TOKEN_SELECT`, `TOKEN_FROM`, `TOKEN_WHERE` tokens |
| **String Literals** | `'hello'`, `'it''s escaped'`, `''` (empty string) | `TOKEN_STRING` with correct value (`"it's escaped"` after unescaping) |
| **Numeric Literals** | `123`, `3.14`, `-42`, `1.23e-4` | `TOKEN_NUMBER` with correct numeric representation |
| **Operators** | `=`, `!=`, `<`, `>`, `AND`, `OR` | Corresponding operator tokens |
| **Identifiers** | `table_name`, `column1`, `_private` | `TOKEN_IDENTIFIER` with correct text |
| **Error Handling** | Unterminated string `'hello`, invalid identifier `1table` | `lexer_get_error` returns descriptive error message |

**Checkpoint Procedure:**
1. Compile tokenizer with test harness: `gcc -o test_tokenizer lexer.c test_tokenizer.c`
2. Run test suite: `./test_tokenizer`
3. Manually verify with sample SQL: `echo "SELECT * FROM users WHERE name = 'Alice'" | ./tokenizer_debug`
4. Expected: Each token printed with type and value, no crashes on malformed input.

**Milestone 2: SQL Parser (AST)**

| Test Category | Test Input | AST Validation Points |
|---------------|------------|----------------------|
| **SELECT Statements** | `SELECT id, name FROM users WHERE age > 18` | `AST_SELECT` node with: 2 column refs, table name `"users"`, `WHERE` clause as `AST_BINARY_OP` (`TOKEN_GREATER`) |
| **INSERT Statements** | `INSERT INTO users (id, name) VALUES (1, 'Alice')` | `AST_INSERT` node with table `"users"`, column list, value expressions |
| **CREATE TABLE** | `CREATE TABLE users (id INT, name TEXT NOT NULL)` | `AST_CREATE_TABLE` with column definitions including types and constraints |
| **Expression Parsing** | `age >= 21 AND (status = 'active' OR override = TRUE)` | Correct operator precedence: `AND` above `OR`, parentheses grouping |
| **Error Recovery** | `SELECT FROM WHERE` (missing elements) | `get_parser_error` returns syntax error at correct position |

**Checkpoint Procedure:**
1. Run parser tests: `./test_parser`
2. Use AST visualizer (if built) to inspect tree structure: `./ast_dump "SELECT * FROM t"`
3. Verify round‑trip for complex nested expressions.
4. Ensure no memory leaks in AST creation/destruction (run with Valgrind).

**Milestone 3: B‑tree Page Format**

| Test Category | Test Operation | Validation Method |
|---------------|----------------|-------------------|
| **Page Serialization** | Create `Page` in memory, call `page_serialize` | Byte‑by‑byte comparison with expected layout from spec |
| **Page Deserialization** | Load byte buffer from file, call `page_deserialize` | All `PageHeader` fields match original values |
| **Leaf Cell Operations** | Add `LeafCell` with key=5, value="test", retrieve | Binary search finds correct cell, `deserialize_leaf_cell` returns correct key/value |
| **Internal Cell Operations** | Add separator key=10, child pointer=3, retrieve | Navigation to child page pointer works correctly |
| **Free Space Management** | Insert then delete cells multiple times | `freeblock_start` and `fragmented_free_bytes` updated correctly |
| **Endianness** | Serialize on big‑endian machine, deserialize on little‑endian | Use `store_big_endian_u32`/`load_big_endian_u32` for cross‑platform consistency |

**Checkpoint Procedure:**
1. Run page unit tests: `./test_page`
2. Inspect hex dump of serialized page: `./page_dump test.page 0`
3. Verify page layout matches SQLite file format diagram: ![B-tree Page Layout Diagram](./diagrams/btree-page-layout.svg)
4. Test with maximum‑size cells (overflow pages not yet implemented).

**Milestone 4: Table Storage**

| Test Category | Test Scenario | Verification |
|---------------|---------------|--------------|
| **CREATE TABLE** | `CREATE TABLE users (id INT, name TEXT)` | Root B‑tree page created, schema catalog updated |
| **INSERT Single Row** | `INSERT INTO users VALUES (1, 'Alice')` | Row retrievable via `btree_find` with correct `rowid` |
| **B‑tree Splitting** | Insert 100 rows (force multiple splits) | All rows retrievable in sorted order, tree height increases appropriately |
| **Full Table Scan** | Iterate through all rows with `cursor_next` | Returns all rows in primary key order |
| **Persistence** | Close database, reopen, query inserted rows | Data persists across sessions |

**Checkpoint Procedure:**
1. Run table storage tests: `./test_table_storage`
2. Manual test via REPL: Create table, insert rows, `SELECT *`, verify output.
3. Inspect database file with hex editor: Confirm B‑tree structure looks valid.
4. Test crash recovery: Kill process mid‑insert, restart, verify no partial data.

**Milestone 5: SELECT Execution (Table Scan)**

| Test Category | SQL Query | Expected Result |
|---------------|-----------|-----------------|
| **Project All Columns** | `SELECT * FROM users` | All columns of all rows, in `rowid` order |
| **Project Subset** | `SELECT name, id FROM users` | Only specified columns, correct order |
| **Empty Table** | `SELECT * FROM empty_table` | Zero rows returned |
| **Large Result Set** | 10,000 rows inserted, then `SELECT *` | All rows returned without crash |
| **Error: Missing Table** | `SELECT * FROM nonexistent` | `ERR_SEMANTIC` error returned |

**Checkpoint Procedure:**
1. Run SELECT execution tests: `./test_select_execution`
2. Compare with SQLite3: `./our_db test.db "SELECT * FROM users"` vs `sqlite3 test.db "SELECT * FROM users"`
3. Validate memory usage: No leaks during large result set processing.
4. Test with mixed data types (INT, TEXT, BLOB, NULL).

**Milestone 6: INSERT/UPDATE/DELETE**

| Test Category | Operation Sequence | Validation |
|---------------|-------------------|------------|
| **INSERT Constraints** | `INSERT INTO users (id) VALUES (NULL)` with `NOT NULL` constraint | `ERR_CONSTRAINT` returned |
| **UPDATE All Rows** | `UPDATE users SET active = 1 WHERE 1=1` | All rows modified, correct count returned |
| **UPDATE With WHERE** | `UPDATE users SET score = score + 5 WHERE score < 50` | Only matching rows updated |
| **DELETE Cascade** | `DELETE FROM users WHERE id > 100` | Rows removed, subsequent SELECT excludes them |
| **Rowid Reuse** | Insert, delete, insert again | New row gets new `rowid` (no reuse required) |
| **B‑tree Rebalancing** | Insert 1000 rows, delete every other row | Tree remains balanced, all operations efficient |

**Checkpoint Procedure:**
1. Run DML operation tests: `./test_dml_operations`
2. Test atomicity: Begin transaction, INSERT, ROLLBACK, verify no rows inserted.
3. Verify constraint enforcement: UNIQUE constraints (if implemented).
4. Performance test: Time 10,000 INSERTs versus SQLite3.

**Milestone 7: WHERE Clause and Indexes**

| Test Category | Test Query | Validation Focus |
|---------------|------------|------------------|
| **WHERE Equality** | `SELECT * FROM users WHERE id = 42` | Returns exactly matching row(s) |
| **WHERE Range** | `SELECT * FROM users WHERE age BETWEEN 18 AND 65` | Returns correct range, sorted by indexed column |
| **Boolean Logic** | `SELECT * FROM users WHERE active = 1 AND (age > 21 OR override = 1)` | Correct three‑valued logic (`BOOL_TRUE`/`FALSE`/`NULL`) |
| **Index Creation** | `CREATE INDEX idx_age ON users(age)` | Index B‑tree built, contains all existing rows |
| **Index Scan** | `SELECT name FROM users WHERE age = 30` (with index) | Uses index scan, not full table scan |
| **Index Maintenance** | INSERT/UPDATE/DELETE after index created | Index stays synchronized with table |

**Checkpoint Procedure:**
1. Run WHERE and index tests: `./test_where_index`
2. Verify index usage via `EXPLAIN`: Output shows `OPERATOR_INDEX_SCAN`.
3. Test performance: Compare `SELECT` with index vs without on 10,000 rows (should be dramatically faster).
4. Test NULL handling in indexes: `WHERE column IS NULL`.

**Milestone 8: Query Planner**

| Test Category | Scenario | Planner Behavior |
|---------------|----------|------------------|
| **Index Selection** | Table with indexes on `a`, `b`; query `WHERE a = 5 AND b > 10` | Chooses most selective index based on estimated cost |
| **Full Scan Fallback** | Query with no usable WHERE clause or indexed columns | Chooses `OPERATOR_TABLE_SCAN` |
| **Cost Estimation** | Large table (simulated) vs small index | Estimates lower cost for index scan, chooses it |
| **EXPLAIN Output** | `EXPLAIN SELECT ...` | Human‑readable plan showing operators, estimated rows |
| **Multi‑Index Consideration** | Query with OR conditions that could use multiple indexes | Considers union of indexes (if implemented) or falls back to table scan |

**Checkpoint Procedure:**
1. Run query planner tests: `./test_query_planner`
2. Manually test with various table sizes (simulate with statistics).
3. Verify `EXPLAIN` output matches actual execution path.
4. Test edge cases: Empty table, all rows matching WHERE, no rows matching.

**Milestone 9: Transactions (BEGIN/COMMIT/ROLLBACK)**

| Test Category | Transaction Sequence | ACID Property Tested |
|---------------|----------------------|----------------------|
| **Atomic Commit** | `BEGIN; INSERT; COMMIT;` | Durability: Data persists after crash |
| **Atomic Rollback** | `BEGIN; INSERT; ROLLBACK;` | Atomicity: No trace of INSERT after rollback |
| **Crash Recovery** | Kill process after `BEGIN; INSERT;` but before COMMIT | Atomicity: No partial changes after restart |
| **Isolation** | Two concurrent connections (simulated) | Changes not visible until COMMIT |
| **Constraint Violation Rollback** | `BEGIN; INSERT (violates constraint); COMMIT;` | Rolls back entire transaction on error |

**Checkpoint Procedure:**
1. Run transaction tests: `./test_transactions`
2. Simulate crashes: Use test harness that kills process at specific points.
3. Verify journal file format: Contents match `JournalHeader` specification.
4. Test hot journal recovery: Create journal manually, restart database, verify recovery.

**Milestone 10: WAL Mode**

| Test Category | Concurrent Scenario | Expected Behavior |
|---------------|---------------------|-------------------|
| **Reader During Write** | Connection 1: `BEGIN; INSERT;` (no commit) Connection 2: `SELECT *` | Reader sees snapshot from before writer's BEGIN |
| **Checkpoint** | Write many pages, commit, run checkpoint | WAL frames merged to main DB, WAL truncated |
| **WAL Recovery** | Crash after WAL write but before checkpoint | Recovery replays WAL on next open |
| **Multiple Readers** | 5 concurrent readers while writer commits | All readers get consistent view, no blocking |
| **WAL File Growth** | Long‑running transaction with many writes | WAL grows, checkpoint controls size |

**Checkpoint Procedure:**
1. Run WAL tests: `./test_wal`
2. Verify concurrency: Use thread‑based test to simulate multiple connections.
3. Inspect WAL file header: Magic number, checksums valid.
4. Performance test: Compare WAL mode vs rollback journal for 10,000 INSERTs.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Test Framework** | `check` (libcheck) | Custom test harness with TAP output |
| **Mocking** | Manual stub functions | `cmocka` for mock generation |
| **Memory Checking** | Valgrind (external) | Custom allocator with guard pages |
| **Coverage** | gcov + lcov | llvm‑cov with HTML reports |
| **Fuzzing** | Simple random input generator | AFL‑fuzz with corpus minimization |
| **Golden Master** | Shell script calling sqlite3 binary | In‑process SQLite3 library linkage |

**B. Recommended File/Module Structure:**

```
build-your-own-sqlite/
├── src/
│   ├── parser/           # Milestone 1,2
│   │   ├── lexer.c
│   │   ├── parser.c
│   │   └── ast.c
│   ├── storage/          # Milestone 3,4
│   │   ├── page.c
│   │   ├── btree.c
│   │   └── pager.c
│   ├── execution/        # Milestone 5,6,7,8
│   │   ├── operators.c
│   │   ├── planner.c
│   │   └── eval.c
│   └── transaction/      # Milestone 9,10
│       ├── journal.c
│       └── wal.c
├── include/              # Headers for each module
├── tests/                # Test suites
│   ├── unit/
│   │   ├── test_lexer.c
│   │   ├── test_parser.c
│   │   ├── test_page.c
│   │   └── ...
│   ├── integration/
│   │   ├── test_select.c
│   │   └── test_transaction.c
│   └── e2e/
│       └── test_sql_compliance.c
├── tools/                # Testing utilities
│   ├── hexdump.c        # Page inspector
│   ├── ast_dump.c       # AST visualizer
│   └── compare_sqlite3.sh # Golden master
└── Makefile
```

**C. Infrastructure Starter Code (Test Harness):**

```c
/* tests/test_harness.h */
#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

/* Creates a temporary database file for testing */
char* create_test_db(const char* prefix);

/* Deletes temporary file and cleans up */
void cleanup_test_db(const char* filename);

/* Asserts two ResultSet are equal */
void assert_resultset_equal(ResultSet* actual, ResultSet* expected);

/* Runs a test SQL against both our engine and SQLite3, compares results */
void compare_with_sqlite3(const char* db_path, const char* sql);

#endif
```

```c
/* tests/test_harness.c */
#include "test_harness.h"
#include "../src/database.h"

char* create_test_db(const char* prefix) {
    char template[] = "/tmp/testdb_XXXXXX";
    int fd = mkstemp(template);
    if (fd == -1) {
        perror("mkstemp");
        return NULL;
    }
    close(fd);
    
    /* Initialize empty database file with our header */
    Database* db = database_open(template);
    if (!db) {
        unlink(template);
        return NULL;
    }
    database_close(db);
    
    char* result = strdup(template);
    if (!result) {
        unlink(template);
        return NULL;
    }
    return result;
}

void cleanup_test_db(const char* filename) {
    if (filename) {
        unlink(filename);
        /* Also clean up journal/WAL files */
        char journal_file[256];
        snprintf(journal_file, sizeof(journal_file), "%s-journal", filename);
        unlink(journal_file);
        
        char wal_file[256];
        snprintf(wal_file, sizeof(wal_file), "%s-wal", filename);
        unlink(wal_file);
    }
}
```

**D. Core Logic Skeleton Code (Example Test for Lexer):**

```c
/* tests/unit/test_lexer.c */
#include <check.h>
#include "../src/parser/lexer.h"

START_TEST(test_lexer_keywords_case_insensitive) {
    Lexer* lexer = lexer_create("SeLeCt FrOm WhErE");
    
    Token* token = lexer_next_token(lexer);
    ck_assert_ptr_nonnull(token);
    ck_assert_int_eq(token->type, TOKEN_SELECT);
    
    token = lexer_next_token(lexer);
    ck_assert_ptr_nonnull(token);
    ck_assert_int_eq(token->type, TOKEN_FROM);
    
    token = lexer_next_token(lexer);
    ck_assert_ptr_nonnull(token);
    ck_assert_int_eq(token->type, TOKEN_WHERE);
    
    token = lexer_next_token(lexer);
    ck_assert_ptr_nonnull(token);
    ck_assert_int_eq(token->type, TOKEN_EOF);
    
    lexer_destroy(lexer);
}
END_TEST

START_TEST(test_lexer_string_with_escaped_quotes) {
    Lexer* lexer = lexer_create("'it''s escaped'");
    
    Token* token = lexer_next_token(lexer);
    ck_assert_ptr_nonnull(token);
    ck_assert_int_eq(token->type, TOKEN_STRING);
    
    /* TODO 1: Verify token->value contains the unescaped string "it's escaped" */
    /* TODO 2: Check that the lexer position advanced past the closing quote */
    /* TODO 3: Test empty string '' */
    /* TODO 4: Test unterminated string error case */
    
    lexer_destroy(lexer);
}
END_TEST

Suite* lexer_suite(void) {
    Suite* s = suite_create("Lexer");
    TCase* tc = tcase_create("Core");
    
    tcase_add_test(tc, test_lexer_keywords_case_insensitive);
    tcase_add_test(tc, test_lexer_string_with_escaped_quotes);
    
    suite_add_tcase(s, tc);
    return s;
}

int main(void) {
    int number_failed;
    Suite* s = lexer_suite();
    SRunner* sr = srunner_create(s);
    
    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

**E. Language‑Specific Hints (C):**

1. **Use `check` Framework**: Install via `apt‑get install check` or `brew install check`. Link with `‑lcheck`.
2. **Memory Leak Detection**: Run tests with `valgrind ‑‑leak‑check=full ./test_lexer`.
3. **Isolate Tests**: Each test case should create its own temporary files to avoid interference.
4. **Error Code Testing**: Use `ck_assert_int_eq(database_execute(db, sql), ERR_OK)` to verify success.
5. **Coverage Reports**: Compile with `‑fprofile‑arcs ‑ftest‑coverage`, run tests, then `gcov ‑b lexer.c`.
6. **Debugging Failing Tests**: Use `gdb ./test_lexer` and `run ‑‑suite lexer ‑‑debug`.

**F. Milestone Checkpoint (Example for Milestone 1):**

1. **Build and Run**: `make test_lexer && ./test_lexer`
2. **Expected Output**: 
   ```
   Running suite(s): Lexer
   100%: Checks: 15, Failures: 0, Errors: 0
   ```
3. **Manual Verification**: 
   ```bash
   echo "SELECT 'test''string', 123.45 FROM table" | ./tools/tokenizer_debug
   ```
   Should output tokens with correct types and values.
4. **Signs of Problems**: 
   - Segmentation fault → Likely buffer overflow in token value copying
   - Wrong token type → State machine logic error
   - Memory leak → `lexer_destroy` not freeing all allocated memory

**G. Debugging Tips for Tests:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Test passes alone but fails in suite** | Global/static state not reset between tests | Add `setup()`/`teardown()` functions in test case | Initialize all static variables in `setup()` |
| **Valgrind reports "conditional jump depends on uninitialized value"** | Struct fields used before initialization | Compile with `‑fsanitize=undefined` to pinpoint | Use `memset` or designated initializers |
| **Test works on Linux but fails on macOS** | Endianness or file locking differences | Check `store_big_endian_u32` usage, verify `fcntl` vs `flock` | Use portable byte‑order macros, abstract file locking |
| **Golden‑master comparison fails on floating‑point values** | Floating‑point rounding differences | Compare with tolerance (ε = 1e‑10) | Use `fabs(a‑b) < epsilon` instead of `a == b` |
| **Transaction test occasionally deadlocks** | Race condition in lock acquisition | Add debug logging to lock/unlock operations | Implement lock timeout or detection |
| **WAL test fails after checkpoint** | WAL header not updated atomically | Check `fsync` order and checksum recalculation | Follow WAL write‑ahead rule strictly |


## 9. Debugging Guide

> **Milestone(s):** All (Debugging is cross‑cutting and essential for every development stage)

Building a database engine is complex, with bugs manifesting at various layers—from SQL parsing to B‑tree corruption to transaction isolation violations. This section provides a structured approach to diagnosing and fixing common issues, along with practical techniques for inspecting database state during development. Think of debugging a database as **forensic investigation**: you have symptoms (queries returning wrong data, crashes, corrupted files), evidence (page dumps, AST representations, execution traces), and you need to reconstruct what happened and why.

### Common Bugs: Symptom → Cause → Fix

This table catalogs frequently encountered issues, organized by component. Each entry follows the pattern: **Symptom** (what the user observes), **Root Cause** (the underlying implementation flaw), and **Fix** (specific corrective action).

| Component | Symptom | Root Cause | Fix |
|-----------|---------|------------|-----|
| **SQL Parser (Milestone 1‑2)** | Tokenizer fails on string literals containing escaped quotes: `'O''Brien'` → syntax error. | Lexer doesn't implement SQL‑style escape handling (two single quotes represent one literal quote). | In `lexer_next_token()`, when in string literal mode, check for `''` sequence and emit a single quote character instead of ending the string. |
| **SQL Parser (Milestone 1‑2)** | `SELECT * FROM t WHERE a = 1 AND b = 2 OR c = 3` yields incorrect precedence (OR evaluated before AND). | Expression parser doesn't implement proper operator precedence; processes left‑to‑right without precedence groups. | Implement precedence‑climbing or shunting‑yard algorithm in expression parsing. Group AND above OR: `(a = 1 AND b = 2) OR c = 3`. |
| **SQL Parser (Milestone 1‑2)** | `CREATE TABLE t (id INT, name TEXT)` parses but `CREATE TABLE t(id INT,name TEXT)` (no spaces after comma) fails. | Parser expects whitespace tokens between identifiers and punctuation; doesn't handle optional whitespace robustly. | Ensure parser `peek()`/`consume()` functions skip whitespace tokens automatically before checking for expected tokens. |
| **B‑tree Storage (Milestone 3‑4)** | Inserting a row works, but subsequent SELECT returns corrupted data or crashes. | **Endianness mismatch**: page header fields written in host byte order (little‑endian) but read assuming big‑endian, or vice versa. | Use `store_big_endian_u16()/u32()` and `load_big_endian_u16()/u32()` consistently for all multi‑byte fields in page headers and cell pointers. |
| **B‑tree Storage (Milestone 3‑4)** | After many INSERT/DELETE operations, database file grows excessively but many pages appear empty. | **Page fragmentation**: deleted cells leave gaps; free‑block list management doesn't coalesce adjacent free blocks. | Implement `page_defragment()` that compacts cell content area during page modification when fragmentation exceeds threshold (e.g., >25%). |
| **B‑tree Storage (Milestone 3‑4)** | Node splitting causes infinite recursion or tree corruption (keys out of order). | **Separator key selection incorrect**: when splitting leaf, middle key not promoted correctly; internal page cell pointers misaligned. | Follow B‑tree algorithm: leaf split → copy middle key to parent; internal split → promote middle key to parent, adjust child pointers. Verify key ordering after split with `btree_validate()`. |
| **Table Storage (Milestone 4)** | `INSERT` with explicit rowid works, but auto‑generated rowids collide (duplicate key error). | Rowid generation uses simple increment from last rowid without checking for existing max rowid in tree. | Implement `btree_get_max_rowid()` that traverses to rightmost leaf to find maximum rowid, then increment. Or use SQLite‑style algorithm: max(rowid, max existing)+1. |
| **Table Storage (Milestone 4)** | `SELECT` returns rows in seemingly random order, not rowid order. | Leaf pages not linked via right‑most child pointers, or cursor traversal follows child pointers incorrectly. | Ensure leaf page headers include `rightmost_child` (SQLite's right‑most pointer) forming linked list. `cursor_next()` should follow this chain after exhausting current page cells. |
| **Query Execution (Milestone 5‑7)** | `WHERE a = 5 AND b = 10` returns rows where only one condition matches, not both. | Boolean expression evaluator uses short‑circuit `&&` instead of three‑valued logic with `BOOL_NULL` handling. | Implement three‑valued logic: `BOOL_TRUE & BOOL_FALSE → BOOL_FALSE`, `BOOL_TRUE & BOOL_NULL → BOOL_NULL`, etc. Use truth tables for AND/OR. |
| **Query Execution (Milestone 5‑7)** | `SELECT * FROM t WHERE col > 100` uses full table scan even with index on `col`. | Query planner doesn't recognize index suitability for range queries; only considers equality lookups. | Extend `create_select_plan()` to check index for any compatible predicate: `col > const`, `col < const`, `col BETWEEN const1 AND const2`. |
| **Query Execution (Milestone 6)** | `UPDATE t SET col = col + 1` modifies same row multiple times (infinite loop). | Cursor positioned at first row, updates it, then `cursor_next()` re‑visits updated row because B‑tree reorganization changed cursor position. | For UPDATE/DELETE, collect rowids first (in a temporary list), then perform modifications by rowid after scan completes. Or use cursor that remains valid across modifications. |
| **Transactions (Milestone 9)** | After power loss during transaction, database is corrupted (some changes applied, some not). | **Torn page**: transaction commit didn't use atomic write (journal not properly synced before overwriting database pages). | Implement **write‑ahead rule**: 1) Write original page to journal, 2) `fsync()` journal, 3) Write modified page to database, 4) `fsync()` database, 5) Delete journal. Missing step 2 or 4 causes corruption. |
| **Transactions (Milestone 9)** | `ROLLBACK` doesn't restore original values for updated rows. | `journal_page_before_write()` not called for every page modification, or journal record format doesn't store enough data to restore. | Ensure every `pager_get_page()` that will be modified calls `journal_page_before_write()` if transaction active. Journal must store full page image (4096 bytes), not just row delta. |
| **WAL Mode (Milestone 10)** | WAL file grows indefinitely, consuming disk space. | Checkpoint not triggered automatically; `wal_checkpoint()` only called manually. | Implement automatic checkpoint when WAL exceeds N pages (e.g., 1000) or after M transactions (e.g., 100). Use `wal_checkpoint(db, false)` for incremental checkpointing. |
| **WAL Mode (Milestone 10)** | Concurrent reader sees stale data after writer commits. | Reader doesn't check WAL for newer page versions; reads directly from main database file. | In `pager_get_page()`, if WAL mode active, call `wal_read_page()` first. If page exists in WAL with higher transaction number, return that version. |
| **Memory/Resource** | Database crashes with `ERR_NOMEM` after many queries, even with small datasets. | **Memory leak**: `Row`, `ResultSet`, `AST` structures allocated but never freed after query execution. | Implement comprehensive cleanup: `destroy_resultset()`, `destroy_ast()`, `destroy_row()`. Use `db_malloc()`/`db_free()` wrappers that track allocations in debug builds. |
| **Concurrency** | Deadlock when two transactions run `UPDATE` on different tables in opposite order. | Naive table‑level locking without deadlock detection or timeout. | Implement deadlock detection via wait‑for graph, or simpler: use timeout and retry with exponential backoff. SQLite uses opportunistic locking with retry. |

> **Key Insight:** Database bugs often manifest **far from their cause**—a parsing bug might corrupt B‑tree pages, a transaction bug might cause wrong query results. Always trace symptoms back through the data flow: wrong output → query execution → B‑tree traversal → page layout → serialization format.

### Debugging Techniques and Tools

Effective database debugging requires inspecting internal state at multiple abstraction levels. These techniques help you see what's *really* happening inside your engine.

#### 1. Diagnostic Queries and EXPLAIN

The most direct way to inspect database behavior is through special diagnostic commands:

| Command | Purpose | Implementation |
|---------|---------|----------------|
| `EXPLAIN QUERY PLAN SELECT ...` | Shows which indexes (if any) the query planner selected and the execution order. | Implement `explain_query()` that returns textual description of `ExecutionPlan` tree. |
| `.schema` | Lists all tables and indexes with their definitions. | Query internal `sqlite_master` table (system catalog) you maintain. |
| `.stats` | Shows cache hit rates, page counts, transaction statistics. | Add counters to `Pager` (cache hits/misses) and `BTree` (split counts). |
| `.dump PAGE N` | Hex/ASCII dump of raw page content for forensic analysis. | Implement `pager_dump_page()` that prints page header and cell pointers. |

**Mental Model:** Think of these as **medical imaging**—X‑rays (EXPLAIN shows query structure), MRI (schema shows database anatomy), and ultrasound (page dump shows tissue-level detail).

#### 2. AST Visualization

When SQL parsing behaves unexpectedly, visualize the Abstract Syntax Tree:

```c
// Simple text visualization
void ast_debug_print(ASTNode* node, int depth) {
    for (int i = 0; i < depth; i++) printf("  ");
    printf("%s", ast_type_to_string(node->type));
    if (node->value.str_val) printf(": %s", node->value.str_val);
    printf("\n");
    for (int i = 0; i < node->child_count; i++) {
        ast_debug_print(node->children[i], depth + 1);
    }
}
```

Use this to verify:
- WHERE clause expression tree structure (operator precedence)
- SELECT column list vs. table references
- INSERT value expressions matching column count

#### 3. Page Inspector Tool

A standalone page inspection utility is invaluable for B‑tree debugging:

```
$ ./page_inspector database.db 5
Page 5 (LEAF):
  Header: type=0x0D, cell_count=42, free_start=3200, cell_start=3800
  Cell pointers: [0:100, 1:150, 2:200, ...]
  Cell 0: rowid=100, data=[INT:42, TEXT:'hello', ...]
  Free blocks: 3200-3800 (600 bytes fragmented)
```

Implement this as:
1. `pager_get_page()` to load raw page
2. `page_deserialize()` to parse header
3. Hex dump of cell content area with annotation for recognized row formats

#### 4. Execution Tracing

Add fine‑grained logging to the query execution engine:

```c
#define TRACE(level, fmt, ...) \
    if (debug_level >= level) fprintf(stderr, "[%s] " fmt, __func__, __VA_ARGS__)

// In table_scan_operator_next():
TRACE(2, "Scanning page %u, cell %u\n", cursor->current_page, cursor->cell_index);
// In evaluate_predicate():
TRACE(3, "Evaluating col=%lld against predicate, result=%d\n", 
      row->columns[col_idx].value.int_val, result);
```

Control verbosity via environment variable:
```bash
DEBUG_LEVEL=3 ./database_cli "SELECT * FROM users WHERE age > 25"
```

#### 5. Consistency Checking

Implement a **validation** routine that checks B‑tree invariants after every modification in debug builds:

```c
bool btree_validate(BTree* btree, uint32_t page_num, int64_t min_key, int64_t max_key) {
    Page* page = pager_get_page(btree->pager, page_num);
    // Check: page type valid, cell count reasonable, keys in range [min_key, max_key]
    // For internal pages: recursively validate each child with appropriate key ranges
    // Report violations with detailed error messages
}
```

Run this after INSERT/UPDATE/DELETE operations to catch corruption early.

#### 6. Golden‑Master Testing

Compare your database's output with **SQLite3** for the same operations:

```bash
# Your implementation
./your_db test.db "CREATE TABLE t(id INT, name TEXT)"
./your_db test.db "INSERT INTO t VALUES (1, 'Alice')"
./your_db test.db "SELECT * FROM t" > your_output.txt

# Reference SQLite3
sqlite3 ref.db "CREATE TABLE t(id INT, name TEXT)"
sqlite3 ref.db "INSERT INTO t VALUES (1, 'Alice')"
sqlite3 ref.db "SELECT * FROM t" > ref_output.txt

diff your_output.txt ref_output.txt
```

Automate this with a script that runs hundreds of query patterns.

#### 7. Fuzzing and Crash Simulation

Deliberately inject failures to test robustness:

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Random page corruption** | Flip random bits in page buffer before write | Test error detection and recovery |
| **Sudden power loss** | `kill -9` the process mid‑transaction | Verify crash recovery works |
| **Memory allocation failure** | Hook `db_malloc()` to fail after N allocations | Test error handling paths |
| **Disk full simulation** | Intercept `write()` calls to return ENOSPC | Test I/O error handling |

#### 8. Interactive Debugging with GDB/LLDB

For C implementations, use debugger watchpoints and backtraces:

**Common Debugger Recipes:**
- `watch *(uint32_t*)page->header.cell_count` – break when cell count changes
- `bt full` after crash – get complete stack trace with local variables
- `x/40xb page->data` – examine raw bytes at memory address
- `info registers` – check for alignment issues (esp. on ARM)

**GDB Script for B‑tree traversal:**
```
define btree_walk
  set $page = btree->root_page
  while ($page != 0)
    printf "Page %u\n", $page
    set $header = (PageHeader*)pager_get_page(btree->pager, $page)
    printf "  Type: %u, Cells: %u\n", $header->page_type, $header->cell_count
    if ($header->page_type == PAGE_TYPE_INTERNAL)
      set $page = $header->rightmost_child
    else
      set $page = 0
    end
  end
end
```

> **Principle:** Debugging a database requires **systematic observation** at multiple layers. Start from user‑visible symptoms, then enable progressively deeper instrumentation until you isolate the faulty component and specific code path.

### Implementation Guidance

This section provides concrete debugging utilities and code patterns to implement the techniques described above.

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Debug Tracing | `fprintf(stderr, ...)` with compile‑time `DEBUG` flag | Structured logging with log levels, rotation, and syslog integration |
| Page Inspection | Command‑line tool reading raw database file | Interactive GUI visualizing B‑tree structure and page layout |
| AST Visualization | Text‑based tree printing with indentation | Graphviz DOT generation for graphical AST rendering |
| Consistency Checking | Assertions and validation functions called in debug builds | Continuous background validation thread in production |

#### B. Recommended File/Module Structure

```
project-root/
  src/
    main.c                 # Entry point with CLI
    database.c/.h          # Core Database type and high-level operations
    debug/                 # Debugging utilities
      inspector.c/.h       # Page inspector and B-tree visualizer
      trace.c/.h           # Execution tracing with configurable levels
      validate.c/.h        # B-tree consistency checker
    tools/
      page_inspector.c     # Standalone page dump utility
      sqlite_compare.py    # Golden-master testing script
  tests/
    fuzz/                  # Fuzzing tests
    corruption/            # Crash recovery tests
```

#### C. Infrastructure Starter Code

**Complete debug tracing module (trace.c):**

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "trace.h"

// Global debug level (0=off, 1=errors, 2=info, 3=verbose, 4=trace)
int g_trace_level = 0;
FILE* g_trace_file = NULL;

void trace_init(int level, const char* filename) {
    g_trace_level = level;
    if (filename && strcmp(filename, "stderr") == 0) {
        g_trace_file = stderr;
    } else if (filename && strcmp(filename, "stdout") == 0) {
        g_trace_file = stdout;
    } else if (filename) {
        g_trace_file = fopen(filename, "a");
        if (!g_trace_file) g_trace_file = stderr;
    } else {
        g_trace_file = stderr;
    }
}

void trace_printf(int level, const char* func, const char* file, int line, 
                  const char* fmt, ...) {
    if (level > g_trace_level || !g_trace_file) return;
    
    // Timestamp (simple version)
    fprintf(g_trace_file, "[%s:%d] ", func, line);
    
    va_list args;
    va_start(args, fmt);
    vfprintf(g_trace_file, fmt, args);
    va_end(args);
    
    fprintf(g_trace_file, "\n");
    fflush(g_trace_file);
}

void trace_page_dump(const char* label, const void* page_data, size_t size) {
    if (g_trace_level < 4) return;
    
    const unsigned char* bytes = (const unsigned char*)page_data;
    fprintf(g_trace_file, "=== PAGE DUMP: %s ===\n", label);
    
    for (size_t i = 0; i < size; i += 16) {
        fprintf(g_trace_file, "%04zx: ", i);
        for (size_t j = 0; j < 16; j++) {
            if (i + j < size) {
                fprintf(g_trace_file, "%02x ", bytes[i + j]);
            } else {
                fprintf(g_trace_file, "   ");
            }
            if (j == 7) fprintf(g_trace_file, " ");
        }
        fprintf(g_trace_file, " |");
        for (size_t j = 0; j < 16 && i + j < size; j++) {
            unsigned char c = bytes[i + j];
            fprintf(g_trace_file, "%c", (c >= 32 && c < 127) ? c : '.');
        }
        fprintf(g_trace_file, "|\n");
    }
    fflush(g_trace_file);
}

void trace_close(void) {
    if (g_trace_file && g_trace_file != stderr && g_trace_file != stdout) {
        fclose(g_trace_file);
    }
    g_trace_file = NULL;
}
```

**Corresponding header (trace.h):**

```c
#ifndef TRACE_H
#define TRACE_H

#include <stddef.h>

void trace_init(int level, const char* filename);
void trace_close(void);
void trace_page_dump(const char* label, const void* page_data, size_t size);

#define TRACE(level, ...) \
    trace_printf(level, __func__, __FILE__, __LINE__, __VA_ARGS__)

#define TRACE_ERROR(...)   TRACE(1, __VA_ARGS__)
#define TRACE_INFO(...)    TRACE(2, __VA_ARGS__)
#define TRACE_VERBOSE(...) TRACE(3, __VA_ARGS__)
#define TRACE_TRACE(...)   TRACE(4, __VA_ARGS__)

#endif
```

#### D. Core Logic Skeleton Code

**Page inspector tool skeleton (tools/page_inspector.c):**

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include "debug/inspector.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <database.db> <page_number>\n", argv[0]);
        fprintf(stderr, "       %s <database.db> --all\n", argv[0]);
        return 1;
    }
    
    const char* filename = argv[1];
    const char* page_arg = argv[2];
    
    // TODO 1: Open database file with db_open_file()
    // TODO 2: If page_arg is "--all", iterate through all pages
    // TODO 3: Otherwise, parse page number and inspect single page
    // TODO 4: For each page, read PAGE_SIZE bytes into buffer
    // TODO 5: Parse page header using load_big_endian_u16/u32
    // TODO 6: Print page type, cell count, free space information
    // TODO 7: Dump cell pointer array
    // TODO 8: For leaf pages, attempt to deserialize and display row data
    // TODO 9: For internal pages, display child page pointers and separator keys
    // TODO 10: Highlight any inconsistencies (pointers out of range, etc.)
    
    return 0;
}
```

**B-tree validation function skeleton (debug/validate.c):**

```c
#include "debug/validate.h"
#include "storage/page.h"
#include "storage/pager.h"
#include <stdio.h>

bool btree_validate(BTree* btree, uint32_t page_num, 
                    int64_t min_key, int64_t max_key, 
                    ValidationContext* ctx) {
    if (page_num == 0) return true;
    
    // TODO 1: Get page using pager_get_page()
    // TODO 2: Verify page header sanity: page_type is LEAF or INTERNAL
    // TODO 3: Check cell_count doesn't exceed maximum possible for page size
    // TODO 4: Verify free_start and cell_content_start are within page bounds
    // TODO 5: For each cell pointer, verify it points within cell content area
    // TODO 6: If leaf page:
    //   - Verify each key is within [min_key, max_key]
    //   - Verify keys are in ascending order
    //   - Attempt to deserialize each row to ensure no corruption
    // TODO 7: If internal page:
    //   - Verify child page numbers are valid (non-zero, < max pages)
    //   - Recursively validate each child with updated key ranges
    //   - Validate rightmost_child pointer
    // TODO 8: Track validation errors in ctx->errors array
    // TODO 9: Return false if any invariant violated
    
    return true;
}

void print_validation_report(ValidationContext* ctx) {
    printf("=== B-tree Validation Report ===\n");
    printf("Pages checked: %u\n", ctx->pages_checked);
    printf("Errors found: %u\n", ctx->error_count);
    for (uint32_t i = 0; i < ctx->error_count; i++) {
        printf("  [%u] Page %u: %s\n", 
               i + 1, 
               ctx->errors[i].page_num,
               ctx->errors[i].message);
    }
    if (ctx->error_count == 0) {
        printf("✓ B-tree structure is valid\n");
    }
}
```

#### E. Language-Specific Hints (C)

- **Memory Debugging:** Use `valgrind --leak-check=full ./your_db test.db "SELECT 1"` to detect memory leaks. For more detailed analysis, `valgrind --tool=memcheck --track-origins=yes`.
- **Address Sanitizer:** Compile with `-fsanitize=address,undefined` to catch buffer overflows and undefined behavior.
- **GDB Pretty‑Printers:** Create GDB Python scripts to pretty‑print your data structures:
  ```python
  class PagePrinter:
      def __init__(self, val):
          self.val = val
      def to_string(self):
          return f"Page(type={self.val['header']['page_type']}, cells={self.val['header']['cell_count']})"
  ```
- **Core Dumps:** Enable core dumps with `ulimit -c unlimited`. After crash, analyze with `gdb ./your_db core`.

#### F. Milestone Checkpoint

After implementing debugging utilities, verify they work:

```bash
# 1. Build with debug symbols
cc -g -O0 -DDEBUG=1 -o your_db *.c

# 2. Create test database
./your_db test.db "CREATE TABLE users(id INT, name TEXT)"
./your_db test.db "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')"

# 3. Run page inspector on root page (usually page 1)
./tools/page_inspector test.db 1

# Expected output:
# Page 1 (LEAF):
#   Header: type=0x0D, cell_count=2, free_start=4050, cell_start=4000
#   Cell 0: rowid=1, columns=[INT:1, TEXT:'Alice']
#   Cell 1: rowid=2, columns=[INT:2, TEXT:'Bob']
#   Free space: 50 bytes

# 4. Run validation
./your_db test.db ".validate"

# Expected output:
# === B-tree Validation Report ===
# Pages checked: 1
# Errors found: 0
# ✓ B-tree structure is valid

# 5. Enable trace logging
DEBUG_LEVEL=3 ./your_db test.db "SELECT * FROM users" 2>&1 | head -20

# Should show trace messages:
# [table_scan_operator_next] Scanning page 1, cell 0
# [evaluate_predicate] No WHERE clause, returning TRUE
# [cursor_get_row] Retrieved rowid=1
```

**Signs of problems:**
- Page inspector shows garbled text or impossible values (negative cell counts) → serialization bug
- Validation reports "key out of order" → B‑tree insertion/split bug
- Trace shows operator calling sequence wrong (filter before scan) → query planner bug

#### G. Debugging Tips Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Query returns wrong number of rows** | Cursor `end_of_table` flag set prematurely, or leaf page chain broken. | Use page inspector to verify leaf page `rightmost_child` pointers form chain. Add trace to `cursor_next()`. | Ensure `cursor_next()` follows rightmost pointer when current page exhausted. |
| **Database file size grows but data doesn't** | Freelist pages not reused; deleted pages remain in file. | Check page type bytes: `PAGE_TYPE_FREELIST` (0x0A). Count freelist pages vs. used pages. | Implement freelist tracking and reuse when allocating new pages. |
| **Transaction commit extremely slow** | `fsync()` called too frequently (e.g., after every page write). | Trace I/O operations; count `fsync()` calls per transaction. | Batch page writes and call `fsync()` once at commit (journal) or checkpoint (WAL). |
| **Memory usage grows unbounded** | `ResultSet` accumulates all rows instead of streaming. | Monitor memory with `valgrind --tool=massif`. Check if `ResultSet` freed after query. | Implement streaming result sets: operator pipeline yields one row at a time. |
| **Crash in `deserialize_leaf_cell()`** | Cell content corrupted or schema mismatch (columns vs. data). | Use `trace_page_dump()` before deserialization. Check varint encoding/decoding. | Add bounds checking: verify cell data length matches expected based on schema types. |
| **Index not used for obvious query** | Query planner cost estimation wrong (overestimates index cost). | Use `EXPLAIN QUERY PLAN` to see chosen plan. Check statistics in index metadata. | Improve cost model: index cost = logN(page accesses) vs. table scan = all pages. |


## 10. Future Extensions

> **Milestone(s):** All (Extensions build upon the foundational components from all milestones)

This section explores **potential enhancements** that could transform our minimal SQLite implementation into a more feature-rich database engine. While the core project focuses on essential database concepts, these extensions represent natural progression points for learners who want to dive deeper. Each extension builds upon the existing architectural components, demonstrating how a well-designed foundation enables incremental feature development.

The architecture we've established—with clean separation between parser, storage engine, query execution engine, and transaction manager—provides **extension points** for new functionality. By understanding how to extend each layer, you'll gain insight into how real database systems evolve from simple beginnings to sophisticated engines.

### Possible Extension Ideas

#### **1. JOIN Operations (Multi-Table SELECT)**

**Mental Model:** The Dating Service Matchmaker
> Think of a JOIN as a matchmaking service that connects people from two different guest lists based on shared interests. The database must examine every possible pair of rows from two tables, check if they satisfy the matching condition (like having the same "interest code"), and introduce them by combining their information into a single result row.

**Current Design Limitations:**
- The `SELECT` parser currently only supports single-table queries
- The query execution engine has no operators for combining rows from multiple tables
- The query planner cannot optimize across table boundaries

**Extension Approach:**

1. **Parser Extension:** Enhance the SQL grammar to support `JOIN` clauses:
   ```sql
   SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE users.country = 'US'
   ```
   - Add `TOKEN_JOIN`, `TOKEN_ON`, `TOKEN_INNER`, `TOKEN_OUTER` tokens to the lexer
   - Extend the `AST_SELECT` node to include a linked list of join specifications
   - Each join specification contains: left table, right table, join type (INNER, LEFT, etc.), and ON condition expression

2. **Execution Engine Extension:** Implement JOIN operators:
   - **Nested Loop Join:** Simplest algorithm—for each row in left table, scan entire right table
   - **Hash Join:** Build hash table from smaller table, probe with larger table (more efficient for equality joins)
   - **Sort-Merge Join:** Sort both tables by join key, then merge (efficient for large datasets)

3. **Join Operator Data Structures:**

| Operator Type | Field Name | Type | Description |
|---------------|------------|------|-------------|
| `OPERATOR_NESTED_LOOP_JOIN` | `left_child` | `Operator*` | Operator producing left table rows |
| | `right_child` | `Operator*` | Operator producing right table rows |
| | `join_condition` | `ASTNode*` | ON clause expression (e.g., `users.id = orders.user_id`) |
| | `join_type` | `JoinType` | `INNER_JOIN`, `LEFT_JOIN`, etc. |
| | `right_buffer` | `Row*` | Buffer for right table rows (materialized for repeated scans) |
| `OPERATOR_HASH_JOIN` | `build_child` | `Operator*` | Operator for build side (typically smaller table) |
| | `probe_child` | `Operator*` | Operator for probe side |
| | `join_key_expr` | `ASTNode*` | Expression to extract join key from row |
| | `hash_table` | `HashTable*` | In-memory hash table mapping join keys to row lists |
| `JoinType` enum | `INNER_JOIN` | `0` | Include only matching rows from both tables |
| | `LEFT_OUTER_JOIN` | `1` | Include all rows from left table, matched rows from right |
| | `RIGHT_OUTER_JOIN` | `2` | Include all rows from right table, matched rows from left |
| | `FULL_OUTER_JOIN` | `3` | Include all rows from both tables |

**ADR: Nested Loop vs. Hash Join Implementation Order**

> **Decision: Implement Nested Loop Join First, Then Hash Join**
> - **Context:** JOINs are complex operations with multiple implementation algorithms of varying complexity. Learners need to understand the fundamental concept before optimizing.
> - **Options Considered:**
>   1. **Nested Loop Join Only:** Simplest to implement but inefficient for large tables
>   2. **Hash Join Only:** More efficient but requires hash table implementation and memory management
>   3. **Both with Planner Choice:** Most complete but doubles implementation complexity
> - **Decision:** Start with nested loop join, then add hash join as an optimization
> - **Rationale:** Nested loop join has minimal prerequisites (just two iterators and a condition evaluator), making it the best pedagogical starting point. Once learners understand the JOIN concept, they can appreciate why hash joins are more efficient and implement them.
> - **Consequences:** Initial JOIN performance will be poor on large tables, but the architecture will support adding optimized join algorithms later.

**Implementation Steps:**
1. Extend the SQL grammar in the parser to recognize `JOIN ... ON ...` syntax
2. Add a `NestedLoopJoinOperator` that implements the double-nested iteration algorithm
3. Enhance the query planner to recognize JOIN patterns and insert appropriate operators
4. Test with small tables to verify correctness before optimizing

---

#### **2. Prepared Statements and Query Caching**

**Mental Model:** The Recipe Card File
> Think of a prepared statement as a recipe card you write once and reuse many times. The parsing and planning work happens during "preparation" (writing the card), leaving only the ingredient substitution (parameter binding) and cooking (execution) for each use. This saves time when making the same dish repeatedly.

**Current Design Limitations:**
- Every query is parsed, planned, and optimized from scratch each time
- No parameterized query support (`SELECT * FROM users WHERE id = ?`)
- No caching of execution plans for repeated queries

**Extension Approach:**

1. **API Extension:** Add prepared statement interface:
   ```c
   typedef struct PreparedStatement {
       const char* sql;
       AST* ast;
       ExecutionPlan* plan;
       ParameterBinding* bindings;
       int param_count;
   } PreparedStatement;
   
   PreparedStatement* db_prepare(Database* db, const char* sql);
   ResultSet* db_execute_prepared(PreparedStatement* stmt, ... /* parameters */);
   void db_finalize(PreparedStatement* stmt);
   ```

2. **Plan Cache:** Simple LRU cache mapping SQL strings to `ExecutionPlan*`:
   ```c
   typedef struct PlanCache {
       PlanCacheEntry** entries;
       size_t capacity;
       size_t size;
       uint64_t clock_hand;  // For clock replacement algorithm
   } PlanCache;
   ```

3. **Parameter Binding:** Extend expression evaluation to handle placeholders:
   - Replace `AST_PARAMETER` nodes with bound values during execution
   - Type checking between parameters and expected types

| Component | Modification Required | Benefit |
|-----------|----------------------|---------|
| **Parser** | Recognize `?` and `:name` parameter tokens | Enable parameterized queries |
| **AST** | Add `AST_PARAMETER` node type | Represent placeholder positions |
| **Query Planner** | Cache plans with parameterized WHERE clauses | Avoid re-planning for same query pattern |
| **Execution Engine** | Bind parameter values before expression evaluation | Execute with different values efficiently |

**Common Pitfalls:**
⚠️ **Pitfall: Plan Cache Invalidation on Schema Changes**
- **Description:** Cached execution plans become invalid when tables or indexes are modified
- **Why Wrong:** Using a stale plan might use dropped indexes or incorrect statistics
- **Fix:** Add schema version number to database; include it in cache key

⚠️ **Pitfall: Parameter Type Mismatch**
- **Description:** Binding a string to an integer column without conversion
- **Why Wrong:** Type mismatches cause execution errors or incorrect results
- **Fix:** Implement type coercion rules or strict type checking at bind time

---

#### **3. Virtual File System (VFS) Layer**

**Mental Model:** The Universal Power Adapter
> Think of the VFS layer as a universal power adapter that lets your database "plug into" different electrical systems (file systems) without rewiring. Whether you're storing data on Windows NTFS, Linux ext4, or in memory, the VFS adapter provides a consistent interface so the storage engine works unchanged.

**Current Design Limitations:**
- Direct file I/O calls (`open`, `read`, `write`, `fsync`) are scattered throughout the code
- Porting to different platforms requires modifying I/O calls everywhere
- Testing I/O error handling is difficult without mocking file operations

**Extension Approach:**

1. **VFS Interface Definition:**
   ```c
   typedef struct VFS {
       // File operations
       int (*xOpen)(const char* filename, int flags, int* fd);
       int (*xRead)(int fd, void* buf, size_t count, off_t offset);
       int (*xWrite)(int fd, const void* buf, size_t count, off_t offset);
       int (*xSync)(int fd);
       int (*xClose)(int fd);
       off_t (*xFileSize)(int fd);
       
       // Locking operations (for concurrent access)
       int (*xLock)(int fd, int lock_type);
       int (*xUnlock)(int fd, int lock_type);
       
       // Platform-specific
       const char* name;
       int (*xRandomness)(void* buf, size_t count);  // For encryption/randomness
   } VFS;
   ```

2. **Default Implementations:**
   - **UnixVFS:** Uses POSIX file operations
   - **WindowsVFS:** Uses Windows API file operations  
   - **InMemoryVFS:** Stores "files" in RAM (useful for testing)
   - **EncryptedVFS:** Wraps another VFS with transparent encryption

3. **Pager Integration:** Modify `Pager` to use VFS interface instead of direct system calls:
   ```c
   typedef struct Pager {
       VFS* vfs;
       int file_descriptor;
       // ... existing fields
   } Pager;
   
   // Instead of open(), use:
   pager->vfs->xOpen(filename, O_RDWR|O_CREAT, &pager->file_descriptor);
   ```

**ADR: Interface-Based vs. Conditional Compilation for Platform Support**

> **Decision: Use VFS Interface Pattern**
> - **Context:** The database needs to run on multiple operating systems with different file APIs
> - **Options Considered:**
>   1. **Conditional Compilation:** `#ifdef _WIN32` throughout the codebase
>   2. **Interface Abstraction:** VFS interface with platform-specific implementations
>   3. **Build-Time Selection:** Separate source files for each platform
> - **Decision:** Implement VFS interface abstraction
> - **Rationale:** Interface pattern provides cleaner separation, enables mock VFS for testing, and allows runtime selection of storage backends (e.g., choosing encrypted VFS at runtime)
> - **Consequences:** Slight overhead from indirect function calls, but enables powerful testing and flexibility

**Testing Benefits:**
- **Mock VFS:** Simulate disk full, permission errors, and corrupted writes
- **In-Memory VFS:** Run tests without touching actual disk (faster, more reliable)
- **Fault Injection:** Test crash recovery by simulating power loss mid-write

---

#### **4. Aggregation Functions and GROUP BY**

**Mental Model:** The Classroom Grade Calculator
> Think of GROUP BY as organizing students into groups by their class (Math, Science, Art), then aggregation functions as calculating statistics for each group (COUNT of students, AVG of grades, MAX score). The database must partition rows, compute aggregates per partition, and output summary rows instead of individual records.

**Current Design Limitations:**
- Only scalar expressions supported in SELECT clause
- No mechanism to group rows or compute aggregates across multiple rows
- Result sets always contain one row per source row

**Extension Approach:**

1. **SQL Parser Enhancement:**
   - Recognize aggregation functions: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`
   - Parse `GROUP BY` clause with column list
   - Distinguish between aggregation and scalar expressions

2. **Aggregation Operator Design:**
   ```c
   typedef struct AggregationOperator {
       Operator base;
       Operator* child;
       AggregationGroup* groups;  // Hash table of groups
       AggregationSpec* specs;    // What to compute per group
       int num_specs;
       bool group_by_present;
       int* group_by_columns;     // Column indices for GROUP BY
       int num_group_columns;
   } AggregationOperator;
   
   typedef struct AggregationSpec {
       AggregationFunc func;      // COUNT, SUM, etc.
       Expression* argument;      // Expression to aggregate (NULL for COUNT(*))
       DataType output_type;
       void* accumulator;         // Per-group accumulator state
   } AggregationSpec;
   ```

3. **Execution Algorithm:**
   ```
   1. Read next row from child operator
   2. Extract GROUP BY column values → compute hash key
   3. Look up or create group in hash table
   4. Update all aggregation accumulators for that group:
      - COUNT(*): increment counter
      - SUM(col): add value to running sum
      - MIN(col): track minimum seen
   5. Repeat until child operator exhausted
   6. Iterate through hash table, finalize aggregates (e.g., compute AVG from sum/count)
   7. Emit one row per group
   ```

**Aggregation State Management:**

| Function | Accumulator Type | Update Logic | Finalize Logic |
|----------|-----------------|--------------|----------------|
| `COUNT(*)` | `int64_t` | `acc++` | Return `acc` |
| `SUM(col)` | Depends on type | `acc += value` | Return `acc` |
| `AVG(col)` | `struct { sum; count; }` | `sum += value; count++` | Return `sum / count` |
| `MIN(col)` | Same as column type | `if (value < acc) acc = value` | Return `acc` |
| `MAX(col)` | Same as column type | `if (value > acc) acc = value` | Return `acc` |

**Common Pitfalls:**
⚠️ **Pitfall: Mixing Aggregated and Non-Aggregated Columns**
- **Description:** `SELECT name, COUNT(*) FROM users` without GROUP BY is invalid
- **Why Wrong:** `name` is per-row, `COUNT(*)` is per-group—incompatible scopes
- **Fix:** Parser/planner must detect this semantic error early

⚠️ **Pitfall: NULL Handling in Aggregates**
- **Description:** `COUNT(col)` vs `COUNT(*)` treat NULLs differently
- **Why Wrong:** `COUNT(col)` should skip NULL values; `COUNT(*)` counts all rows
- **Fix:** Implement SQL-standard NULL semantics for each function

---

#### **5. Subqueries (Scalar, EXISTS, IN)**

**Mental Model:** The Russian Nesting Dolls
> Think of subqueries as Russian nesting dolls—queries inside queries. A scalar subquery is like a tiny doll that produces a single value. An EXISTS subquery is a detective checking if any matching row exists. An IN subquery is a bouncer checking credentials against a guest list. Each requires different execution strategies.

**Current Design Limitations:**
- No support for queries within queries
- WHERE clause only supports simple expressions, not subquery results
- No mechanism to correlate inner query with outer query rows

**Extension Approach:**

1. **Subquery Types and Their Execution:**

| Subquery Type | Returns | Example | Execution Strategy |
|---------------|---------|---------|-------------------|
| **Scalar** | Single value | `SELECT name FROM users WHERE id = (SELECT manager_id FROM dept WHERE ...)` | Execute inner query first, cache single result |
| **EXISTS** | Boolean | `SELECT * FROM orders WHERE EXISTS (SELECT 1 FROM customers WHERE ...)` | Stop inner query at first matching row |
| **IN** | Boolean (membership) | `SELECT * FROM products WHERE category_id IN (SELECT id FROM categories WHERE active=1)` | Materialize inner query results, build hash set |
| **Correlated** | Depends on outer row | `SELECT * FROM employees e WHERE salary > (SELECT AVG(salary) FROM employees WHERE dept = e.dept)` | Execute inner query repeatedly for each outer row |

2. **AST Extension:** Add `AST_SUBQUERY` node type that contains a full `SELECT` statement AST

3. **Query Planner Extension:** 
   - **Uncorrelated Subqueries:** Evaluate once, materialize result
   - **Correlated Subqueries:** Convert to JOINs where possible, or use nested iteration
   - **EXISTS Optimization:** Use semi-join (stop after first match)

4. **Execution Operator:** 
   ```c
   typedef struct SubqueryOperator {
       Operator base;
       Operator* outer_child;     // For correlated subqueries
       SubqueryType type;
       AST* subquery_ast;
       ExecutionPlan* subquery_plan;
       MaterializedResult* cached_result;  // For uncorrelated subqueries
       int correlation_column;    // Which outer column correlates to inner
   } SubqueryOperator;
   ```

**Performance Considerations:**
- **Correlated Subquery Problem:** Executing inner query for every outer row can be O(n²)
- **Optimization:** The planner should attempt to "decorrelate" by rewriting as JOIN
- **Materialization:** For `IN` subqueries, building a hash set once is more efficient than repeated linear scans

**Example Decorrelation Transformation:**
```sql
-- Correlated subquery (slow)
SELECT * FROM employees e 
WHERE salary > (SELECT AVG(salary) FROM employees WHERE dept = e.dept)

-- Decorrelated equivalent (faster)
SELECT e.* FROM employees e 
JOIN (SELECT dept, AVG(salary) as avg_salary FROM employees GROUP BY dept) d
ON e.dept = d.dept 
WHERE e.salary > d.avg_salary
```

---

#### **6. Full-Text Search Index**

**Mental Model:** The Book Index at the Back of a Textbook
> While a regular index helps find specific terms (like finding the page for "B-tree"), a full-text search index is like the comprehensive index at the back of a textbook that helps you find all occurrences of concepts, synonyms, and related terms, even when they appear in different forms ("query", "queries", "querying").

**Current Design Limitations:**
- B-tree indexes only support exact matches or prefix matches
- No support for searching text with stemming, ranking, or phrase matching
- Cannot handle queries like "database NEAR/3 implementation"

**Extension Approach:**

1. **Inverted Index Data Structure:**
   ```
   Document 1: "database system implementation"
   Document 2: "distributed database query"
   
   Inverted Index:
   "database" → [Doc1(pos1), Doc2(pos2)]
   "system" → [Doc1(pos2)]
   "implementation" → [Doc1(pos3)]
   "distributed" → [Doc2(pos1)]
   "query" → [Doc2(pos3)]
   ```

2. **Storage Format Extension:** Create new page type `PAGE_TYPE_FULLTEXT`:
   - **Dictionary B-tree:** Maps terms to posting list locations
   - **Posting Lists:** Compressed lists of (document_id, position) pairs
   - **Document Table:** Maps document_id to primary key

3. **Full-Text Search Operator:**
   ```c
   typedef struct FullTextSearchOperator {
       Operator base;
       const char* search_query;    // User's query string
       FullTextIndex* index;
       PostingListIterator* current_iterator;
       ResultRanking* rankings;     // For relevance scoring
       double score_threshold;      // Minimum relevance score
   } FullTextSearchOperator;
   ```

4. **Query Syntax Extension:**
   ```sql
   -- Basic full-text search
   SELECT * FROM docs WHERE fts_match(content, 'database implementation')
   
   -- With ranking
   SELECT *, fts_rank(content, 'query optimization') as relevance 
   FROM docs 
   WHERE fts_match(content, 'query optimization') 
   ORDER BY relevance DESC
   
   -- Phrase search
   SELECT * FROM docs WHERE fts_match(content, '"B-tree node"')
   ```

**Tokenizer Integration:**
- Reuse the SQL lexer's tokenization logic for word breaking
- Add stemming (running → run), stop-word removal (the, a, an), and case folding
- Support for multiple languages with different tokenization rules

**Common Pitfalls:**
⚠️ **Pitfall: Index Size Explosion**
- **Description:** Full-text indexes can be larger than original text (2-3x)
- **Why Wrong:** Storing every word position for large documents consumes excessive space
- **Fix:** Use compression (delta encoding for positions), exclude stop words, limit indexing to important fields

⚠️ **Pitfall: Transactional Consistency**
- **Description:** Maintaining full-text index during INSERT/UPDATE/DELETE
- **Why Wrong:** Index can become inconsistent with table data
- **Fix:** Treat index updates as part of the same transaction; use same WAL/journal

---

#### **7. User-Defined Functions (UDFs)**

**Mental Model:** The App Store for Your Database
> Think of UDFs as downloadable apps that add new capabilities to your database. Just as a smartphone becomes more useful with apps for specific tasks (calculator, weather, maps), your database becomes more powerful with functions for domain-specific calculations (geodistance, text processing, statistical analysis).

**Current Design Limitations:**
- Only built-in functions available (`COUNT`, `SUM`, etc.)
- Cannot extend SQL with domain-specific logic
- Business logic must be implemented in application code

**Extension Approach:**

1. **UDF Registration API:**
   ```c
   typedef struct UDF {
       const char* name;
       DataType return_type;
       DataType* arg_types;
       int num_args;
       bool deterministic;  // Same inputs → same output (enables optimization)
       void* user_data;
       UDFFunction func;    // Implementation function pointer
   } UDF;
   
   // Registration function
   bool db_register_udf(Database* db, UDF* udf);
   
   // Example UDF implementation
   ColumnValue geodistance_udf(ExpressionContext* ctx, ColumnValue* args, int num_args, void* user_data) {
       double lat1 = args[0].float_val;
       double lon1 = args[1].float_val;
       double lat2 = args[2].float_val;
       double lon2 = args[3].float_val;
       // Calculate distance...
       ColumnValue result;
       result.type = DATA_TYPE_FLOAT;
       result.float_val = distance;
       return result;
   }
   ```

2. **SQL Integration:**
   ```sql
   -- Register from SQL
   CREATE FUNCTION geodistance(lat1 FLOAT, lon1 FLOAT, lat2 FLOAT, lon2 FLOAT) 
   RETURNS FLOAT
   AS EXTERNAL 'libgeo.so!geodistance_impl';
   
   -- Use in queries
   SELECT name, geodistance(lat, lon, 40.7, -74.0) as distance 
   FROM businesses 
   WHERE geodistance(lat, lon, 40.7, -74.0) < 5.0
   ORDER BY distance;
   ```

3. **Security Considerations:**
   - **Sandboxing:** UDFs in C can crash the database or access arbitrary memory
   - **Solution:** Consider interpreted language (Lua, JavaScript) for safer UDFs
   - **Permissions:** Differentiate between trusted (C) and untrusted (scripted) UDFs

**Performance Optimizations:**
- **Deterministic UDFs:** Can be cached/memoized (same inputs → reuse previous result)
- **Stateless UDFs:** Can be parallelized across threads
- **Vectorized UDFs:** Process batches of rows for better cache locality

**Extension Points in Current Architecture:**
1. **Parser:** Recognize `CREATE FUNCTION` syntax
2. **Catalog:** Store UDF metadata alongside table schemas
3. **Expression Evaluator:** Call UDF implementation during expression evaluation
4. **Query Planner:** Apply optimizations based on UDF properties (deterministic, stateless)

---

#### **8. Foreign Key Constraints**

**Mental Model:** The Family Relationship Registry
> Think of foreign key constraints as official family registries that track relationships. A "child" row (order) must reference an existing "parent" row (customer). The registry ensures no orphan records exist and cascade rules define what happens when parents are deleted (reject, cascade, set null, set default).

**Current Design Limitations:**
- Tables exist in isolation with no relationship enforcement
- Application code must maintain referential integrity
- No cascade behavior for updates/deletes

**Extension Approach:**

1. **Constraint Storage in Schema Catalog:**
   ```c
   typedef struct ForeignKeyConstraint {
       const char* name;
       const char* table_name;        // Child table
       const char* column_names[16];  // Child columns
       const char* ref_table_name;    // Parent table  
       const char* ref_column_names[16]; // Parent columns
       ForeignKeyAction on_delete;    // CASCADE, SET NULL, etc.
       ForeignKeyAction on_update;    // CASCADE, RESTRICT, etc.
       int num_columns;
   } ForeignKeyConstraint;
   ```

2. **Constraint Enforcement Points:**
   - **INSERT into child table:** Check parent exists
   - **UPDATE child table:** Check new parent exists
   - **DELETE from parent table:** Apply cascade rule
   - **UPDATE parent primary key:** Apply cascade rule

3. **Referential Integrity Validation:** Utility to check all foreign key relationships:
   ```c
   bool validate_foreign_keys(Database* db, ValidationContext* ctx);
   ```

4. **SQL Syntax Support:**
   ```sql
   CREATE TABLE orders (
       id INTEGER PRIMARY KEY,
       customer_id INTEGER,
       amount DECIMAL(10,2),
       FOREIGN KEY (customer_id) REFERENCES customers(id)
           ON DELETE CASCADE
           ON UPDATE RESTRICT
   );
   ```

**Cascade Action Implementation:**

| Action | Description | Implementation |
|--------|-------------|----------------|
| `RESTRICT` | Prevent parent modification if children exist | Check before modification, fail if any children |
| `CASCADE` | Modify children to match parent changes | Recursively apply same operation to children |
| `SET NULL` | Set child foreign key to NULL | Update child rows to NULL |
| `SET DEFAULT` | Set child foreign key to column default | Update child rows to default value |
| `NO ACTION` | Defer constraint check until transaction commit | Check at commit time, rollback if violation |

**Performance Implications:**
- **Index Requirement:** Foreign key columns should be indexed for efficient constraint checking
- **Cascade Chains:** Deep cascade chains can cause many recursive operations
- **Deferred Constraints:** Allow batching constraint checks for bulk operations

**Common Pitfalls:**
⚠️ **Pitfall: Circular Dependencies**
- **Description:** Table A references B, B references A (directly or indirectly)
- **Why Wrong:** Impossible to insert first row without violating constraint
- **Fix:** Support deferred constraint checking, allow temporarily invalid state within transaction

⚠️ **Pitfall: Orphan Rows After Failed Transaction**
- **Description:** Child rows inserted, parent insertion fails, child becomes orphan
- **Why Wrong:** Violates referential integrity
- **Fix:** All modifications are atomic within transaction; rollback removes orphan

---

### Extension Prioritization Guide

For learners considering which extensions to implement first, here's a suggested priority based on educational value and complexity:

| Extension | Educational Value | Implementation Complexity | Prerequisites |
|-----------|------------------|--------------------------|---------------|
| **JOIN Operations** | High (fundamental to relational model) | Medium | Working SELECT, B-tree cursors |
| **GROUP BY & Aggregates** | High (shows set operations) | Medium | Working SELECT, expression evaluation |
| **Subqueries** | Medium (query composition) | High | JOINs, query planning |
| **Prepared Statements** | Medium (performance optimization) | Low | Query planning, expression evaluation |
| **VFS Layer** | Medium (abstraction design) | Low-Medium | Pager component |
| **Foreign Keys** | Medium (constraint systems) | Medium | Schema catalog, transaction rollback |
| **Full-Text Search** | High (specialized indexing) | High | B-tree indexes, tokenizer |
| **User-Defined Functions** | Medium (extensibility) | Medium | Expression evaluation, catalog |

**Extension Dependency Graph:**
```
Prepared Statements
    ↑
    └─────┬─────┐
          ↓     ↓
     JOINs    Aggregates
          ↓     ↓
       Subqueries ←─┐
          ↓         │
       Query        │
       Planner      │
    Extensions      │
          ↓         ↓
       Foreign    Full-Text
        Keys       Search
          └─────┬─────┘
                ↓
            UDFs & VFS
```

> **Key Insight:** Extensions should be implemented **incrementally**, testing each thoroughly before moving to the next. The beauty of the modular architecture is that each component can be enhanced independently, allowing you to choose extensions based on your interests and learning goals.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Extension | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **JOINs** | Nested loop join with brute-force scanning | Hash join with in-memory hash tables |
| **Prepared Statements** | Simple plan cache (fixed-size array) | LRU cache with cost-based eviction |
| **VFS Layer** | Interface with Unix/Windows implementations | Pluggable VFS with encryption wrapper |
| **Aggregates** | In-memory hash table for grouping | Spill-to-disk for large groups |
| **Subqueries** | Materialize all results, then process | Decorrelation to JOINs when possible |
| **Full-Text Search** | Simple inverted index (word → doc IDs) | Positional index with ranking (BM25) |

**B. Recommended File/Module Structure:**

```
project-root/
├── src/
│   ├── parser/              # Existing parser
│   │   ├── parser.c         # Extend for new SQL syntax
│   │   └── ast.c            # Add new AST node types
│   ├── execution/           # Execution engine
│   │   ├── operators.c      # Add new operator implementations
│   │   ├── join.c           # JOIN operators
│   │   ├── aggregate.c      # GROUP BY and aggregation
│   │   └── subquery.c       # Subquery execution
│   ├── storage/            # Storage engine
│   │   ├── vfs.c           # Virtual File System interface
│   │   │   ├── unix_vfs.c  # Unix implementation
│   │   │   ├── win_vfs.c   # Windows implementation
│   │   │   └── mem_vfs.c   # In-memory for testing
│   │   ├── fulltext.c      # Full-text index implementation
│   │   └── constraint.c    # Foreign key constraint checking
│   └── catalog/            # Schema catalog
│       └── udf.c           # UDF registration and management
└── include/
    ├── vfs.h               # VFS interface definitions
    ├── join.h              # JOIN operator structures
    └── udf.h               # UDF API
```

**C. Infrastructure Starter Code (VFS Interface):**

```c
/* vfs.h - Virtual File System Interface */
#ifndef VFS_H
#define VFS_H

#include <stddef.h>
#include <stdint.h>

typedef struct VFS VFS;

/* File open flags */
#define VFS_OPEN_READONLY      0x0001
#define VFS_OPEN_READWRITE     0x0002
#define VFS_OPEN_CREATE        0x0004
#define VFS_OPEN_EXCLUSIVE     0x0008
#define VFS_OPEN_DELETEONCLOSE 0x0010
#define VFS_OPEN_TEMPORARY     0x0020

/* Lock types */
#define VFS_LOCK_SHARED        1
#define VFS_LOCK_RESERVED      2
#define VFS_LOCK_PENDING       3
#define VFS_LOCK_EXCLUSIVE     4

/* VFS method table */
struct VFS {
    /* File operations */
    int (*xOpen)(VFS* vfs, const char* filename, int flags, int* fd);
    int (*xDelete)(VFS* vfs, const char* filename, int sync_dir);
    int (*xAccess)(VFS* vfs, const char* filename, int flags, int* result);
    int (*xFullPathname)(VFS* vfs, const char* relative, char* absolute, size_t size);
    
    /* File descriptor operations */
    int (*xClose)(VFS* vfs, int fd);
    int (*xRead)(VFS* vfs, int fd, void* buf, size_t count, off_t offset);
    int (*xWrite)(VFS* vfs, int fd, const void* buf, size_t count, off_t offset);
    int (*xTruncate)(VFS* vfs, int fd, off_t size);
    int (*xSync)(VFS* vfs, int fd, int full_sync);
    int (*xFileSize)(VFS* vfs, int fd, off_t* size);
    
    /* Locking operations */
    int (*xLock)(VFS* vfs, int fd, int lock_type);
    int (*xUnlock)(VFS* vfs, int fd, int lock_type);
    int (*xCheckReservedLock)(VFS* vfs, int fd, int* result);
    
    /* Platform-specific */
    const char* name;                     /* Name of this VFS */
    int (*xRandomness)(VFS* vfs, void* buf, size_t count);  /* Random bytes */
    int (*xSleep)(VFS* vfs, int microseconds);              /* Sleep */
    int (*xCurrentTime)(VFS* vfs, double* time);            /* Current time */
    
    /* Application data */
    void* app_data;
};

/* Default VFS implementations */
VFS* vfs_get_default(void);               /* Platform-specific default */
VFS* vfs_create_memory(void);             /* In-memory VFS for testing */
VFS* vfs_create_wrapper(VFS* base_vfs);   /* For encryption/compression */

#endif /* VFS_H */
```

**D. Core Logic Skeleton Code (Nested Loop Join Operator):**

```c
/* join.c - JOIN operator implementations */
#include "join.h"
#include "execution.h"
#include "expression.h"

typedef struct NestedLoopJoinState {
    Row left_row;
    Row right_row;
    bool left_row_valid;
    bool right_row_valid;
    bool left_exhausted;
    bool need_new_left;
} NestedLoopJoinState;

static OperatorResult nested_loop_join_next(Operator* op) {
    NestedLoopJoinOperator* join = (NestedLoopJoinOperator*)op;
    NestedLoopJoinState* state = (NestedLoopJoinState*)op->context;
    ExpressionContext* expr_ctx = join->expression_context;
    
    while (1) {
        /* TODO 1: If we need a new left row, fetch one from left child */
        /*   - Call left_child->next() to get next left row */
        /*   - If left child returns OPERATOR_EOF, set left_exhausted = true */
        /*   - If we get a row, store it in state->left_row and set left_row_valid = true */
        /*   - Reset right child to start (call right_child->reset() if available) */
        /*   - Set need_new_left = false */
        
        /* TODO 2: If left side exhausted, return OPERATOR_EOF */
        
        /* TODO 3: Fetch next right row from right child */
        /*   - Call right_child->next() */
        /*   - If OPERATOR_EOF: need_new_left = true; continue outer loop */
        /*   - If we get a row, store in state->right_row, right_row_valid = true */
        
        /* TODO 4: Set up expression context with both rows */
        /*   - expr_ctx->current_row should point to combined row structure */
        /*   - expr_ctx->user_context might store left/right row pointers */
        
        /* TODO 5: Evaluate join condition */
        /*   - Call evaluate_predicate(expr_ctx, join->join_condition) */
        /*   - If result is BOOL_TRUE: join match found! */
        /*   - If BOOL_FALSE: continue loop (try next right row) */
        /*   - If BOOL_NULL: treat as no match (SQL semantics) */
        
        /* TODO 6: If match found, construct output row */
        /*   - For INNER JOIN: combine columns from left and right */
        /*   - For LEFT JOIN: combine all left columns, right columns or NULLs */
        /*   - Set op->current_row to combined row */
        /*   - Return OPERATOR_ROW_READY */
    }
    
    return OPERATOR_ERROR;
}

Operator* create_nested_loop_join_operator(
    Operator* left_child,
    Operator* right_child,
    ASTNode* join_condition,
    JoinType join_type,
    Schema* left_schema,
    Schema* right_schema
) {
    /* TODO: Allocate and initialize NestedLoopJoinOperator */
    /* TODO: Set up expression context with both schemas */
    /* TODO: Initialize state structure */
    /* TODO: Set operator's next function pointer to nested_loop_join_next */
    /* TODO: Set operator's reset function to reset both children */
    return NULL;
}
```

**E. Language-Specific Hints (C):**

1. **Memory Management for Extensions:**
   ```c
   /* For hash joins and aggregates, consider using memory pools */
   typedef struct MemoryPool {
       void** blocks;
       size_t block_size;
       size_t used;
       size_t block_count;
   } MemoryPool;
   
   void* pool_alloc(MemoryPool* pool, size_t size);
   void pool_reset(MemoryPool* pool);  /* Reuse memory without freeing */
   ```

2. **Thread Safety for Prepared Statement Cache:**
   ```c
   #include <pthread.h>
   
   pthread_rwlock_t plan_cache_lock = PTHREAD_RWLOCK_INITIALIZER;
   
   /* Reader */
   pthread_rwlock_rdlock(&plan_cache_lock);
   plan = find_in_cache(sql);
   pthread_rwlock_unlock(&plan_cache_lock);
   
   /* Writer */
   pthread_rwlock_wrlock(&plan_cache_lock);
   insert_into_cache(sql, plan);
   pthread_rwlock_unlock(&plan_cache_lock);
   ```

3. **Testing Extensions:**
   ```c
   /* Use the in-memory VFS for fast, isolated tests */
   void test_join() {
       VFS* mem_vfs = vfs_create_memory();
       Database* db = database_open_vfs(":memory:", mem_vfs);
       
       /* Create test tables, insert data */
       database_execute(db, "CREATE TABLE left (id INT, val TEXT)");
       database_execute(db, "CREATE TABLE right (id INT, val TEXT)");
       
       /* Test JOIN */
       ResultSet* rs = database_execute(db, 
           "SELECT * FROM left JOIN right ON left.id = right.id");
       
       /* Verify results */
       assert(rs->row_count == expected_count);
   }
   ```

**F. Milestone Checkpoint for JOIN Implementation:**

**What to Test:**
```bash
# Create test database
./sqlite test.db

# Create two related tables
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);

# Insert test data
INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');
INSERT INTO orders VALUES (101, 1, 99.99), (102, 1, 49.99), (103, 2, 29.99);

# Test INNER JOIN
SELECT users.name, orders.amount 
FROM users 
INNER JOIN orders ON users.id = orders.user_id 
WHERE amount > 50.0;

# Expected output:
# name   | amount
# -------+-------
# Alice  | 99.99

# Test LEFT JOIN (users without orders)
SELECT users.name, orders.amount 
FROM users 
LEFT JOIN orders ON users.id = orders.user_id;

# Should show all 3 rows (Alice's 2 orders, Bob's 1 order)
```

**Signs of Correct Implementation:**
1. JOIN returns correct number of rows (Cartesian product filtered by condition)
2. Column references from both tables resolve correctly
3. NULL handling works for OUTER JOINs (missing matches show NULL)
4. Performance is reasonable for small tables (but will be slow for large ones)

**Common Implementation Bugs:**
- **Infinite loop:** Forgetting to advance right child before trying same left row again
- **Memory leak:** Not freeing temporary rows when moving to next left row
- **Wrong column mapping:** Mixing up left/right schema positions in combined row

---

> **Final Insight:** The true test of a well-designed system is not just what it does today, but how easily it can be extended tomorrow. Each extension described here builds upon the solid foundation established in the core milestones, demonstrating the power of modular, layered architecture. By implementing even one of these extensions, you'll deepen your understanding of database internals and gain valuable experience in evolving complex software systems.


## 11. Glossary

> **Milestone(s):** All (reference for terminology used throughout all milestones)

This glossary defines key technical terms, acronyms, and domain-specific vocabulary used throughout this design document. It serves as a quick reference for concepts that may be unfamiliar, ensuring consistent understanding across the entire project. Terms are organized alphabetically for easy lookup.

### Terms and Definitions

| Term | Definition |
|------|------------|
| **Abstract Syntax Tree (AST)** | A hierarchical tree structure that represents the syntactic structure of an SQL statement after parsing. Each node in the tree corresponds to a grammatical construct (e.g., a `SELECT` statement, an expression, a column reference), with child nodes representing subcomponents. The `AST` struct and `ASTNode` are the primary in-memory representations. |
| **ACID** | An acronym for Atomicity, Consistency, Isolation, and Durability—the four key properties that guarantee reliable processing of database transactions. Our implementation ensures Atomicity (transactions are all-or-nothing), Consistency (transactions preserve database invariants), Isolation (concurrent transactions don't interfere), and Durability (committed transactions survive crashes). |
| **Atomicity** | The property that ensures a transaction is treated as a single, indivisible unit of work—either all of its operations are completed and persisted, or none of them are. In our system, atomicity is achieved via the rollback journal or Write-Ahead Log (WAL). |
| **Atomicity Protocol** | The two-phase process for ensuring atomic commits: (1) prepare/write phase where all changes are durably logged, and (2) commit/apply phase where changes are made permanent. The rollback journal and WAL implement variations of this protocol. |
| **Backward Recovery** | The process of restoring the database to a previous consistent state by undoing uncommitted changes, typically using a rollback journal that contains original page images. Contrast with **forward recovery**. |
| **Binary Search** | An O(log n) search algorithm that finds a target value within a sorted array by repeatedly dividing the search interval in half. Used within B-tree pages to locate cells quickly, since keys within a page are maintained in sorted order. |
| **B-tree** | A self-balancing tree data structure that maintains sorted data and allows efficient insertion, deletion, and search operations. Our implementation uses B-trees for both table storage (with `rowid` as key) and secondary indexes (with indexed column values as keys). Nodes correspond to fixed-size `Page` structures on disk. |
| **Boundary Testing** | A testing technique that focuses on values at the edges of input domains (e.g., minimum/maximum values, empty strings, zero rows). Important for database components handling variable-length data, page boundaries, and overflow conditions. |
| **Cell** | A key-value pair stored within a B-tree page. In leaf pages, a **LeafCell** stores a row (with `rowid` as key and serialized row data as value). In internal pages, an **InternalCell** stores a separator key and a child page pointer for tree navigation. |
| **Checkpointing** | The process of moving committed changes from the Write-Ahead Log (WAL) back into the main database file. This prevents the WAL from growing indefinitely and reduces recovery time after a crash. The `wal_checkpoint()` function implements this process. |
| **Concurrent Readers** | Multiple database connections or threads that can query the database simultaneously while a writer holds the write lock. In WAL mode, readers can access the database while a transaction is active because they read from the main database file or older WAL frames, not the active transaction's uncommitted changes. |
| **Consistency Checking** | The process of verifying data structure invariants (e.g., B-tree ordering, page linkage, cell counts) to detect corruption early. Implemented via `btree_validate()` and other validation functions. |
| **Coverage Analysis** | Measuring which code paths, branches, or statements are exercised by tests, typically using code coverage tools. Helps ensure comprehensive testing of edge cases and error handling code. |
| **Crash Simulation** | Artificially terminating a process (e.g., via `kill -9` or simulated power failure) to test recovery mechanisms. Essential for validating that the rollback journal or WAL correctly restores database consistency. |
| **Cursor** | An iterator-like object (`Cursor` struct) that maintains position within a B-tree while traversing key-value pairs. It tracks the current page, cell index, and navigation stack for efficient forward/backward movement through sorted data. Used by scan operators to retrieve rows. |
| **Data Definition Language (DDL)** | SQL statements that define or modify database schema structures, such as `CREATE TABLE`, `CREATE INDEX`, and `ALTER TABLE`. Contrast with **Data Manipulation Language (DML)**. |
| **Data Manipulation Language (DML)** | SQL statements that manipulate data within existing tables, such as `SELECT`, `INSERT`, `UPDATE`, and `DELETE`. Contrast with **Data Definition Language (DDL)**. |
| **Database** | The primary database handle (`Database` struct) that aggregates the pager, schema catalog, and transaction state. The `database_open()` function creates this object, and `database_execute()` is the main public API for executing SQL statements. |
| **Defensive Hierarchy** | A layered approach to corruption handling where each level attempts to detect and handle errors before they propagate upward. For example, page-level checksums detect corruption, B-tree validation detects structural issues, and transaction rollback recovers from incomplete operations. |
| **Deserialization** | The process of converting bytes from disk (e.g., from a `Page`'s data area) into in-memory structures (e.g., `Row`, `ColumnValue`). Functions like `page_deserialize()` and `deserialize_leaf_cell()` perform this conversion. Contrast with **serialization**. |
| **Dirty Pages** | Pages in the page cache that have been modified in memory but not yet written to disk. The transaction manager tracks these pages in `TransactionState.dirty_pages` to ensure they're properly flushed during commit or restored during rollback. |
| **Durability** | The property that guarantees once a transaction is committed, its changes will persist even in the event of a system crash. Achieved by forcing modified pages to stable storage (via `fsync`) before reporting transaction completion to the user. |
| **Endianness** | The byte order used to store multi-byte integers in memory or on disk. Our implementation uses big-endian (network byte order) for all multi-byte integers in page headers and cell pointers, ensuring consistent interpretation across different hardware architectures. Functions `store_big_endian_u16()`, `load_big_endian_u32()`, etc., handle conversions. |
| **End-to-End Test** | A test that exercises the entire system from user input (SQL text) to output (result set or side effect), verifying that all components work together correctly. Often uses golden-master testing against SQLite3. |
| **Error Injection** | A testing technique that deliberately introduces failures (e.g., simulated I/O errors, memory allocation failures) to verify error handling and recovery code paths work correctly. |
| **Escaped Quotes** | The SQL convention of representing a single quote character within a string literal by doubling it (e.g., `'It''s a test'`). The lexer must recognize this pattern during string tokenization to avoid prematurely terminating the string. |
| **Execution Plan** | A tree of operators (`ExecutionPlan` struct) that represents the chosen strategy for executing a query. The query planner generates this structure, and the execution engine evaluates it to produce results. |
| **Fail-Fast** | A design principle where errors are detected and reported as early as possible, minimizing the damage caused by continuing with invalid data or inconsistent state. Our parser, deserialization functions, and validation routines follow this principle. |
| **Forensic Investigation** | The methodical process of examining symptoms (e.g., crashes, corrupted data) to determine the root cause, often using logging, debugging tools, and systematic hypothesis testing. The debugging guide outlines this process. |
| **Forward Recovery** | The process of reapplying changes from a log to bring the database forward to a consistent state after a crash. Write-Ahead Logging uses forward recovery by replaying WAL frames. Contrast with **backward recovery**. |
| **Free Block List** | A linked list within a B-tree page that tracks regions of free space created by cell deletions. The `PageHeader.freeblock_start` field points to the first free block, allowing efficient reuse of space without immediate defragmentation. |
| **fsync** | A system call that forces the operating system to write all buffered data for a file descriptor to stable storage. Critical for ensuring durability in database systems; used by `journal_commit()`, `wal_checkpoint()`, and other functions that write persistent state. |
| **Golden-Master Approach** | A testing methodology where output from our implementation is compared against a reference implementation (SQLite3) to verify correctness. Used extensively in integration and end-to-end testing. |
| **Golden-Master Testing** | See **Golden-Master Approach**. |
| **Hash Join** | A join algorithm that builds a hash table from one relation (typically the smaller) and probes it with rows from the other relation. More efficient than nested loop joins for equality joins on large datasets. Represented by `OPERATOR_HASH_JOIN`. |
| **Hot Journal** | A rollback journal file that contains a commit record but hasn't been deleted, indicating a crash occurred during transaction commit. The recovery mechanism (`recover_from_crash()`) automatically detects and processes hot journals on database open. |
| **Hot Journal Recovery** | The automatic recovery process that occurs when opening a database file with a hot journal present. The system reads the journal and either completes the commit (if the commit record exists) or rolls back (if not) to ensure atomicity. |
| **Index-Only Scan** | A query execution strategy where all required columns are available in a secondary index, eliminating the need to access the main table. Improves performance by reducing I/O. |
| **Integration Point** | An interface between two components (e.g., parser to execution engine, B-tree to pager) that requires careful testing to ensure data flows correctly and errors propagate appropriately. |
| **Isolation** | The property that ensures concurrent execution of transactions leaves the database in the same state as if transactions were executed sequentially. Our implementation provides snapshot isolation in WAL mode (readers see a consistent snapshot). |
| **Journal** | A separate file (rollback journal) that stores original page images before they are modified, enabling transaction rollback and crash recovery. The `Journal` struct manages this file, with `journal_page_before_write()` saving pages and `journal_rollback()` restoring them. |
| **JOIN** | An SQL operation that combines rows from two or more tables based on a related column between them. Types include `INNER_JOIN`, `LEFT_OUTER_JOIN`, etc. Our implementation can be extended with operators like `NestedLoopJoinOperator`. |
| **LeafCell** | A cell in a B-tree leaf page that stores a key (typically `rowid`) and a value (serialized row data). The format includes a variable-length header with column types and values. |
| **Lexer** | The component (`Lexer` struct) that performs lexical analysis, breaking SQL text into a sequence of tokens (keywords, identifiers, literals, operators). `lexer_next_token()` is the core function that scans input and returns `Token` objects. |
| **Materialization** | Storing intermediate query results (e.g., subquery results, join outputs) fully in memory or on disk rather than processing them pipelined. Can improve performance for certain operations but increases memory usage. |
| **Memory Leak Detection** | Finding memory that is allocated but never freed, using tools like Valgrind or address sanitizers. Critical for long-running database processes that must manage memory carefully. |
| **Memory Pool** | A custom allocator (`MemoryPool` struct) that manages fixed-size or variable-size memory blocks efficiently, reducing fragmentation and allocation overhead for database operations. |
| **Mock** | A simulated component used during testing to isolate the component under test from its dependencies (e.g., a mock pager that simulates disk I/O failures). |
| **Multi-Version Concurrency Control (MVCC)** | A concurrency control method that allows multiple readers and writers to access the database simultaneously by maintaining multiple versions of data items. Our WAL implementation provides a form of MVCC where readers see a consistent snapshot. |
| **Nested Loop Join** | A join algorithm that iterates over each row in the outer relation and, for each, scans the inner relation for matching rows. Simple but can be inefficient for large tables. The `NestedLoopJoinState` tracks iteration state. |
| **Operator** | A processing unit in the query execution engine's volcano model. Each operator (`Operator` struct) implements a `next()` method that returns rows. Operators are chained into a pipeline (e.g., `TableScanOperator` → `FilterOperator` → `ProjectionOperator`). |
| **Operator Precedence** | Rules defining the order in which operators (e.g., `AND` vs `OR`, `*` vs `+`) are evaluated in expressions. The parser must respect these rules when building expression ASTs, typically by implementing a precedence-climbing or similar algorithm. |
| **Overflow Page** | An additional page used to store large rows or values that don't fit within a single leaf cell. The cell contains a pointer to the overflow page(s), with data spread across multiple pages. |
| **Page** | A fixed-size unit (typically 4096 bytes, `PAGE_SIZE`) of disk I/O and memory allocation. The `Page` struct represents this unit, with a `PageHeader` followed by a data area containing cell pointers and cell content. |
| **Page Cache** | An in-memory buffer of recently accessed disk pages managed by the `Pager`. `pager_get_page()` returns a page from cache if present; otherwise, it loads from disk. Dirty pages are written back via `pager_flush_page()`. |
| **PageHeader** | The metadata section at the start of each `Page` that describes the page's type (`PAGE_TYPE_LEAF` or `PAGE_TYPE_INTERNAL`), cell count, free space offset, and other layout information. |
| **Pager** | The component (`Pager` struct) that manages the page cache and file I/O. It handles reading/writing pages to disk, caching, and maintaining file length. `pager_open()` initializes it, `pager_get_page()` retrieves pages, and `pager_flush_page()` writes dirty pages. |
| **Performance Regression** | When new code changes make operations slower than before. Detected through benchmarking and monitored during development, especially for critical paths like B-tree operations and query execution. |
| **Pipelined Execution** | A query execution strategy where rows flow through operators one at a time without materializing intermediate results, reducing memory overhead. The volcano model enables pipelined execution. |
| **Predicate Pushdown** | A query optimization technique that applies WHERE clause filters as early as possible in the execution pipeline (e.g., in a `FilterOperator` immediately after a scan), reducing the number of rows processed by downstream operators. |
| **Prepared Statement** | A precompiled SQL statement (`PreparedStatement` struct) that can be executed multiple times with different parameter values. Improves performance by avoiding repeated parsing and planning. Created by `db_prepare()` and executed by `db_execute_prepared()`. |
| **Property-Based Testing** | A testing methodology that verifies certain properties hold for a wide range of randomly generated inputs (e.g., "serialization followed by deserialization yields the original value"). Useful for testing parsers, serialization, and B-tree invariants. |
| **Query Execution Engine** | The component responsible for executing SQL operations by evaluating an execution plan. It implements the volcano model with operators for scanning, filtering, projecting, and modifying data. |
| **Read-Eval-Print Loop (REPL)** | An interactive shell that reads SQL commands, executes them, and prints results. Used for manual testing and exploration of the database. |
| **Recursive Descent Parsing** | A top-down parsing technique where each grammar rule is implemented as a mutually recursive function. Our SQL parser uses this approach: `parse_select_statement()`, `parse_expression()`, etc., call each other based on the grammar. |
| **Regression Detection** | Catching when new code changes break existing functionality, typically through a comprehensive test suite that runs automatically (e.g., via continuous integration). |
| **ResultSet** | An in-memory collection (`ResultSet` struct) that holds the output of a `SELECT` query, including rows, column names, and a cursor for iterating through results. Returned by `execute_select()`. |
| **Rollback Journal** | The classic journaling method where original page contents are saved to a separate journal file before modification. If the transaction rolls back or crashes, the original pages are restored. Implemented by the `Journal` component. |
| **Row** | The in-memory representation (`Row` struct) of a table record, containing a `rowid` and an array of `ColumnValue` structures for each column. Rows are serialized into leaf cells for storage and deserialized during query execution. |
| **Secondary Index** | An additional B-tree that provides an alternative access path to table rows, keyed by column values other than the primary key. `CREATE INDEX` builds a secondary index, and the query planner may choose an index scan over a table scan. |
| **Separator Key** | A key stored in an internal B-tree page that divides the key ranges of its child subtrees. All keys in the left child are ≤ the separator, and all keys in the right child are > the separator. |
| **Serialization** | The process of converting in-memory structures (e.g., `Row`, `Page`) into bytes for storage on disk. Functions like `page_serialize()` and cell serialization routines perform this conversion. Contrast with **deserialization**. |
| **Short-Circuit Evaluation** | An optimization where boolean expression evaluation stops as soon as the outcome is determined (e.g., `false AND anything` is `false`). Used in `evaluate_predicate()` to improve performance. |
| **Subquery** | A query nested inside another query, represented by `AST_SUBQUERY`. Can be correlated (references outer query columns) or uncorrelated. The `SubqueryOperator` executes subqueries. |
| **Temporary Database** | A database file created solely for a single test and deleted afterwards, ensuring test isolation and avoiding side effects. Used extensively in our testing strategy. |
| **Test Harness** | A framework or set of utilities for running tests, reporting results, and managing test fixtures. Our testing approach includes a custom test harness that runs golden-master comparisons and crash recovery tests. |
| **Three-Valued Logic** | Boolean logic with three possible values: `BOOL_TRUE`, `BOOL_FALSE`, and `BOOL_NULL` (unknown). Used in SQL for expression evaluation where `NULL` values propagate through operations. The `evaluate_predicate()` function returns a `TriStateBool`. |
| **Token** | The smallest meaningful unit of SQL text, produced by the lexer. The `Token` struct represents a token with type (e.g., `TOKEN_SELECT`, `TOKEN_IDENTIFIER`), value string, and position in the input. |
| **Tokenization** | The process of converting SQL text into a sequence of tokens, performed by the lexer. Also called lexical analysis. |
| **Torn Page** | A partially written page that results from a power failure or crash during a write operation. Journaling mechanisms prevent torn pages by ensuring either the original or the new page is intact, never a mixture. |
| **TransactionState** | A structure (`TransactionState` struct) that tracks the current transaction's status (`TRANSACTION_ACTIVE`, `TRANSACTION_COMMITTING`, etc.), dirty pages, and journal reference. |
| **User-Defined Function (UDF)** | A function defined by the user (not built into SQL) that can be called from SQL statements. Registered via `db_register_udf()` and invoked during expression evaluation. |
| **Variable-Length Integer (varint)** | A compact encoding for integers that uses fewer bytes for small values. Our implementation uses SQLite's varint encoding for keys and row IDs; `varint_encode()` and `varint_decode()` handle conversion. |
| **Virtual File System (VFS)** | An abstraction layer (`VFS` struct) that encapsulates file operations (open, read, write, sync, lock), enabling portability across platforms and facilitating testing with in-memory or simulated file systems. `vfs_get_default()` returns the platform-specific implementation. |
| **Volcano Model** | An iterator-based query execution model where each operator implements a `next()` method that returns the next row or `OPERATOR_EOF`. Operators are arranged in a tree, and rows flow from leaf operators upward. Our query execution engine follows this model. |
| **Write Amplification** | The phenomenon where writing data causes more physical writes than logically required (e.g., writing a journal entry and then the main database page). WAL reduces write amplification compared to rollback journal in some scenarios. |
| **Write-Ahead Logging (WAL)** | A journaling method where changes are appended to a separate log file (the WAL) before being written to the main database. Provides better concurrency and often better performance than rollback journal. The `WAL` struct manages the WAL file, and `wal_read_page()` serves pages from it. |
| **Write-Ahead Rule** | The fundamental principle that recovery information must be written to stable storage before the corresponding changes are made to the database. Both rollback journal and WAL adhere to this rule to ensure atomicity. |

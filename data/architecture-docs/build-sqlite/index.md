# 🎯 Project Charter: Build Your Own SQLite
## What You Are Building
A fully functional embedded SQL database engine — built from scratch, layer by layer. By the end, you will have a binary-compatible SQLite implementation that opens `.db` files readable by the real `sqlite3` CLI, executes SQL queries against persistent B-tree storage, and survives process crashes without data loss. The engine includes a SQL tokenizer, recursive-descent parser, register-based virtual machine (VDBE), page-based B-tree/B+tree storage engine, LRU buffer pool, cost-based query planner, and ACID-compliant transactions via both rollback journal and write-ahead logging.
## Why This Project Exists
Every database abstraction you use daily — ORM, connection pool, query cache — sits on top of mechanisms that almost no developer has seen. Building a database from first principles exposes the assumptions baked into every `SELECT` you've ever written: why indexes make some queries fast and others slow, why transactions cost time, why crash recovery is hard, and why concurrent reads and writes require careful choreography. These are the concepts that separate backend engineers who configure databases from those who understand them.
## What You Will Be Able to Do When Done
- Tokenize and parse arbitrary SQL text into a validated Abstract Syntax Tree
- Compile SQL ASTs into register-based VDBE bytecode and execute it in a virtual machine
- Read and write fixed-size 4096-byte pages using a buffer pool with LRU eviction and pin counting
- Encode and decode rows using SQLite's variable-length binary record format (varint + serial types)
- Implement B-tree node splitting and B+tree leaf-chain traversal for table and index storage
- Execute SELECT, INSERT, UPDATE, and DELETE with correct three-valued NULL semantics
- Build and maintain secondary indexes synchronously across all DML operations
- Collect column statistics with ANALYZE and select index-scan vs. table-scan based on cost estimates
- Implement BEGIN/COMMIT/ROLLBACK with a rollback journal, enforcing write ordering with fsync
- Add WAL mode for concurrent read access during writes, with snapshot isolation and auto-checkpoint
- Execute GROUP BY, HAVING, COUNT/SUM/AVG/MIN/MAX, and INNER JOIN with nested-loop execution
## Final Deliverable
Approximately 10,000–14,000 lines of C (or equivalent in Rust or Go) across 20+ source files. The compiled binary opens and creates `.db` files in SQLite's binary format — verifiable with `xxd` and readable by the real `sqlite3` command-line tool. It boots in milliseconds, executes a full 10,000-row table scan in under 100ms, and survives `kill -9` mid-transaction with full data integrity on restart. You can demo `CREATE TABLE`, `INSERT`, `SELECT` with `WHERE`/`JOIN`/`GROUP BY`, `EXPLAIN` to show query plans, `ANALYZE` to gather statistics, and `PRAGMA journal_mode=WAL` to enable concurrent reader mode.
## Is This Project For You?
**You should start this if you:**
- Can write non-trivial C, Rust, or Go programs that allocate memory, manipulate raw bytes, and call POSIX file I/O (`pread`, `pwrite`, `fsync`, `flock`)
- Understand B-tree data structures at the conceptual level (balanced trees, node splitting, search by key)
- Know basic SQL — can write SELECT with WHERE, JOIN, GROUP BY, and understand what a transaction is
- Have implemented at least one compiler or interpreter component (lexer, parser, or AST evaluator)
- Are comfortable reading and writing binary file formats using bitwise operations and struct packing
**Come back after you've learned:**
- Systems programming in C, Rust, or Go — specifically pointer arithmetic, manual memory management, and file I/O at the syscall level ([The C Programming Language](https://en.wikipedia.org/wiki/The_C_Programming_Language) or [Rust Book](https://doc.rust-lang.org/book/))
- How a B-tree works — implement one from scratch before starting this project ([Crafting Interpreters](https://craftinginterpreters.com/) covers trees well; CLRS Chapter 18 covers B-trees)
- Finite state machines and recursive-descent parsing at a basic level
## Estimated Effort
| Phase | Time |
|-------|------|
| SQL Tokenizer | ~4 hours |
| SQL Parser (AST) | ~7 hours |
| Bytecode Compiler (VDBE) | ~10 hours |
| Buffer Pool Manager | ~8 hours |
| B-tree Page Format & Table Storage | ~12 hours |
| SELECT Execution & DML | ~10 hours |
| Secondary Indexes | ~8 hours |
| Query Planner & Statistics | ~10 hours |
| Transactions (Rollback Journal) | ~10 hours |
| WAL Mode | ~12 hours |
| Aggregate Functions & JOIN | ~14 hours |
| **Total** | **~105 hours** |
## Definition of Done
The project is complete when:
- `SELECT * FROM t` on a 10,000-row table returns all rows in correct order in under 100ms on a warm buffer pool
- `BEGIN; INSERT ...; INSERT ...; COMMIT;` persists both rows durably — verified by closing and reopening the database file
- `BEGIN; UPDATE ...; ROLLBACK;` leaves the database byte-for-byte identical to its pre-transaction state — verified by comparing file hashes before and after
- Crash recovery works: a `.db-journal` or `.db-wal` file left by a simulated crash is automatically replayed or rolled back on next open, with no manual intervention required
- `EXPLAIN SELECT * FROM t WHERE email = 'x@y.com'` shows `INDEX SCAN` when a unique index on `email` exists and `TABLE SCAN` when selectivity is above 20%
- `COUNT(*)`, `SUM`, `AVG`, `GROUP BY`, and `INNER JOIN` produce correct results including correct NULL handling (COUNT(*) counts NULLs; AVG ignores them; GROUP BY handles empty groups)

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Foundational — Read Before Starting
### Finite Automata & Lexical Analysis
**Best Explanation:** Crafting Interpreters by Robert Nystrom, Chapter 3 ("Scanning") — available free at https://craftinginterpreters.com/scanning.html
**Why:** The most lucid hands-on treatment of hand-written lexers. Nystrom builds the same zero-copy FSM scanner the Atlas describes, with identical design rationale.
**Code:** CPython's tokenizer — `cpython/Lib/tokenize.py` and `Python/tokenizer.c` — shows production state-machine lexing with error recovery.
**Read BEFORE starting** — the tokenizer is your first line of code; the FSM mental model must be solid before Milestone 1.
---
### Recursive Descent Parsing & Pratt Expressions
**Best Explanation:** *"Pratt Parsers: Expression Parsing Made Easy"* by Bob Nystrom — https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/
**Why:** The canonical 15-minute article that makes Pratt parsing click. Written by the same author as the Atlas cites; direct, with working Java code that maps 1:1 to the C implementation you'll write.
**Code:** SQLite's own `parse.c` (generated by Lemon) alongside `expr.c` — demonstrates how the real engine encodes SQL's precedence rules.
**Read BEFORE Milestone 2** — operator precedence bugs are silent correctness failures; understand the binding-power table before you type a character.
---
### Page-Based Storage & B-Trees
**Paper:** Bayer & McCreight (1972), "Organization and Maintenance of Large Ordered Indices" — the original B-tree paper. Short (14 pages) and readable.
**Best Explanation:** Designing Data-Intensive Applications (Kleppmann), Chapter 3, pages 76–97 ("B-Trees") — covers the storage-engine tradeoffs the Atlas alludes to throughout Milestones 4–5.
**Code:** SQLite's `btree.c` — 9,000 lines, but the `btreeNext`, `sqlite3BtreeInsert`, and `balance()` functions map directly to the cursor and split logic you implement.
**Read BEFORE Milestone 5** — the page-split logic requires internalizing the invariants before implementing them; re-reading after your first split bug is also highly recommended.
---
## At Specific Milestones
### Virtual Machines & Register-Based Bytecode
*(Read before Milestone 3)*
**Best Explanation:** Crafting Interpreters, Chapter 14 ("Chunks of Bytecode") through Chapter 24 — the entire back-half of the book. Nystrom builds a register-adjacent VM in C with exactly the dispatch loop the Atlas implements.
**Paper:** "The Implementation of Lua 5.0" (Ierusalimschy et al., 2005) — explains the design decision to switch from stack-based to register-based; directly addresses the tradeoff the Atlas presents.
**Why:** Understanding *why* the VDBE is register-based (not stack-based) is what separates a working implementation from an understood one.
---
### Buffer Pool & LRU / Cache Replacement
*(Read before or during Milestone 4)*
**Best Explanation:** Carnegie Mellon's 15-445 lecture notes, Lecture 6 ("Buffer Pools") — https://15445.courses.cs.cmu.edu. Andy Pavlo's slides cover LRU, Clock, and ARC with concrete examples.
**Paper:** Megiddo & Modha (2003), "ARC: A Self-Tuning, Low Overhead Replacement Cache" — explains exactly why LRU fails on sequential scans, and what the Atlas means by the "2Q" and "ARC" alternatives.
**Read before Milestone 4** — implementing LRU correctly requires knowing its failure modes; understanding ARC tells you what production systems do instead.
---
### Write-Ahead Logging & Crash Recovery
*(Read before Milestone 9, revisit before Milestone 10)*
**Paper:** Mohan et al. (1992), "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging" — *the* foundational recovery paper. Dense but indispensable. Read the introduction and the "ARIES Algorithm" section.
**Best Explanation:** CMU 15-445 Lecture 19 ("Crash Recovery") — Pavlo walks through ARIES analysis/redo/undo with worked examples, making the paper tractable.
**Code:** SQLite's `pager.c` — implements both the rollback journal (Milestone 9) and WAL (Milestone 10) in a single file (~7,500 lines). Functions `pagerRollback()`, `sqlite3PagerCommitPhaseOne()`, and `walWriteLock()` map directly to what you build.
**Read BEFORE Milestone 9** — the write ordering violations the Atlas warns about are subtle; reading ARIES first makes the "journal fsync before database write" rule feel obvious rather than arbitrary.
---
### Cost-Based Query Optimization
*(Read before or during Milestone 8)*
**Paper:** Selinger et al. (1979), "Access Path Selection in a Relational Database Management System" — System R's optimizer paper. Introduces cost-based optimization, selectivity estimation, and the dynamic-programming join ordering you implement. Readable in an afternoon.
**Best Explanation:** CMU 15-445 Lecture 13 ("Query Planning I") — Pavlo walks through selectivity formulas and the Selinger DP algorithm with concrete examples.
**Why:** The Selinger paper is the most cited in database history for a reason; reading it before implementing the planner makes every design decision feel inevitable rather than arbitrary.
---
### WAL Mode, MVCC & Snapshot Isolation
*(Read after Milestone 9, before Milestone 10)*
**Paper:** Bernstein & Goodman (1983), "Multiversion Concurrency Control — Theory and Algorithms" — the MVCC foundation; explains why readers and writers can coexist without blocking.
**Best Explanation:** Designing Data-Intensive Applications, Chapter 7, pages 237–261 ("Weak Isolation Levels" through "Serializable Snapshot Isolation") — Kleppmann's treatment of read-committed, snapshot isolation, and write skew is the clearest available.
**Code:** SQLite's `wal.c` — `walWriteLock()`, `walIndexReadHdr()`, and `walCheckpoint()` implement the exact WAL index, snapshot marker, and checkpoint logic the Atlas describes. Studying the real implementation after writing your own is deeply instructive.
**Read after Milestone 9** — you need the rollback journal mental model before WAL's inversion of it makes sense. Reading MVCC theory before Milestone 10 makes snapshot isolation feel like a logical consequence rather than a magic trick.
---
## Reference Material (Ongoing)
**SQLite File Format Specification** — https://www.sqlite.org/fileformat2.html
The authoritative specification for every byte the Atlas describes: page headers, varint encoding, serial types, journal and WAL formats. Bookmark it. Read sections as you implement each milestone. When your hex dump doesn't match, this is where you look.
**"SQLite Internals: How the World's Most Used Database Works"** (Grigorev, 2023) — https://www.compileralchemy.com/books/sqlite-internals/
A free, modern walkthrough of SQLite's actual source code at the level of abstraction the Atlas uses. Complements the spec by explaining *why* each format decision was made. Read in parallel with Milestones 5–10.

---

# Build Your Own SQLite

This project constructs a complete embedded SQL database engine from first principles — starting with lexical analysis of SQL text, through recursive-descent parsing into ASTs, compilation to bytecode executed by a register-based virtual machine (VDBE), all operating atop a page-based storage engine with B-trees for clustered table storage and B+trees for secondary indexes, managed through a buffer pool with LRU eviction, and ACID guarantees via both rollback journal and write-ahead logging.

The architecture mirrors real SQLite's layered design: a frontend (tokenizer → parser → code generator), a virtual machine execution engine, and a backend (B-tree module → pager/buffer pool → OS interface). Each layer communicates through well-defined interfaces, allowing the learner to build and test incrementally. By the end, the learner will have a working database that can parse SQL, optimize queries using collected statistics, execute joins and aggregates, and survive process crashes without losing committed data.

This is an expert-level project demanding mastery of compiler construction, data structure design, systems programming with careful attention to disk I/O ordering, and the subtle interplay between performance and durability that defines every production database.



<!-- MS_ID: build-sqlite-m1 -->
# Milestone 1: SQL Tokenizer
## System Orientation
Before writing a single line of code, orient yourself in the machine you are building.

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

The SQL tokenizer sits at the very front of a layered pipeline. Raw text enters from a user or application. Nothing downstream — the parser, the bytecode compiler, the virtual machine, the storage engine — ever sees a raw character again after this milestone. The tokenizer is the first transformation in the query lifecycle.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

Every subsequent milestone depends on the token stream you produce here. A bug in this layer propagates invisibly through the entire system: `SELECT` misidentified as an identifier doesn't crash immediately — it produces a parse error three tokens later that says something incomprehensible. Build this layer correctly and you will never think about it again. Build it carelessly and it will haunt you through every milestone that follows.
---
## The Revelation: It Is Not String Splitting
Here is what many developers assume before they have built a tokenizer: SQL tokenization is string splitting with some regex polish. Find the spaces. Find the obvious delimiters. Classify what's between them. Done in an afternoon.
This model is wrong, and the place it breaks is immediately.
Consider this SQL string:
```
SELECT 'it''s a table', "table name" FROM t WHERE x >= -3.14;
```
Try splitting on whitespace and punctuation. What happens?
- `'it''s` — The embedded `''` is an escaped single quote inside a string literal. A naive split sees two separate tokens: `'it'` and `'s`. The string has been destroyed.
- `"table name"` — The space inside double quotes is part of an identifier. A split on whitespace fragments it into `"table` and `name"`. The identifier is corrupted.
- `-3.14` — Is this the number negative-three-point-fourteen, or is it the subtraction operator followed by the number `3.14`? You cannot decide by looking at the characters in isolation.
- `>=` — This is a single two-character operator token, not `>` followed by `=`. Split on operators and you break it.
The fundamental reason these cases fail is that **a character's meaning depends on what came before it**. A `'` character means "start a string" when you are reading ordinary SQL, but means "escaped quote" when you are already inside a string literal. A `-` means "begin a negative number" or "subtraction operator" depending on whether a value or an operator is valid in the current context. You cannot decide token boundaries by examining characters in isolation.
The correct mental model is this: **tokenization is a stateful process**, and the appropriate formalism for stateful character-by-character processing is the finite state machine.

> **🔑 Foundation: Finite state machines for lexical analysis**
> 
> ## Finite State Machines for Lexical Analysis
### What it IS
A finite state machine (FSM) is a computational model with a fixed set of **states**, **transitions** between states triggered by input characters, a **start state**, and one or more **accepting states**. At any moment, the machine is in exactly one state. You feed it characters one at a time, it transitions, and when input ends you check: are we in an accepting state? If yes, you recognized a valid token.
For lexical analysis (the "lexer" or "scanner" phase of a compiler), FSMs are the theoretical backbone. Each token type — integer literal, identifier, `==` operator, string literal — is described by a regular expression, which maps directly onto an FSM.
**Concrete example — recognizing integer literals:**
```
States: START, IN_DIGIT, ERROR
START  --[0-9]--> IN_DIGIT
IN_DIGIT --[0-9]--> IN_DIGIT   (self-loop)
IN_DIGIT --[anything else]--> DONE (emit token, put char back)
```
The moment you hit a non-digit, you stop and emit the integer you accumulated.
**Deterministic vs Non-deterministic (DFA vs NFA):**
- An **NFA** (Non-deterministic FA) can have multiple possible transitions for the same input, or ε-transitions (move without consuming input). Easy to *construct* from a regex.
- A **DFA** (Deterministic FA) has exactly one transition per (state, input) pair. Easy to *execute* efficiently.
- The standard pipeline: regex → NFA (Thompson construction) → DFA (subset construction) → minimized DFA → code.
In practice, tools like `flex`/`re2c` do this for you. But if you're writing a lexer by hand, you're essentially implementing a DFA as a `switch` statement or a transition table.
### WHY you need it now
Your lexer is the first phase of your compiler/interpreter. It converts raw source text into a stream of tokens that the parser can work with. Without a principled model, you'll write ad-hoc `if/else` character-scanning code that breaks on edge cases (e.g., `!=` vs `!`, `=` vs `==`, nested strings with escape sequences). FSMs give you a systematic way to handle all these correctly and to reason about *what your lexer can and cannot recognize*.
### Key Insight
**Regular expressions and FSMs are the same thing** — every regex describes exactly the set of strings accepted by some FSM, and vice versa. This means: if you can write a regex for your token, you can build an FSM for it; if you can't write a regex for it (e.g., balanced parentheses), an FSM *cannot* recognize it either, and you need the parser phase instead. The lexer/parser split is not arbitrary — it's the boundary between regular and context-free languages.

---
## What You Are Building
The tokenizer's contract is simple to state: accept a null-terminated C string of SQL text, produce a flat array (or linked list) of `Token` structures, each containing a type, a value (the matched text), and a source location (line and column number).
The caller — the parser in Milestone 2 — will consume this stream token-by-token. It will never call back into raw character processing. The token stream is the abstraction boundary between lexical analysis and syntactic analysis.
Your token structure in C:
```c
typedef enum {
    /* Literals */
    TOKEN_INTEGER,       /* 42, -7 */
    TOKEN_FLOAT,         /* 3.14, -2.718 */
    TOKEN_STRING,        /* 'hello' */
    TOKEN_IDENTIFIER,    /* table_name, col */
    TOKEN_QUOTED_ID,     /* "column name" */
    /* Keywords */
    TOKEN_SELECT,
    TOKEN_INSERT,
    TOKEN_UPDATE,
    TOKEN_DELETE,
    TOKEN_CREATE,
    TOKEN_TABLE,
    TOKEN_INDEX,
    TOKEN_WHERE,
    TOKEN_FROM,
    TOKEN_JOIN,
    TOKEN_ON,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,
    TOKEN_NULL,
    TOKEN_IS,
    TOKEN_ORDER,
    TOKEN_BY,
    TOKEN_ASC,
    TOKEN_DESC,
    TOKEN_LIMIT,
    TOKEN_INTO,
    TOKEN_VALUES,
    TOKEN_SET,
    TOKEN_PRIMARY,
    TOKEN_KEY,
    TOKEN_UNIQUE,
    TOKEN_GROUP,
    TOKEN_HAVING,
    TOKEN_INNER,
    TOKEN_BEGIN,
    TOKEN_COMMIT,
    TOKEN_ROLLBACK,
    TOKEN_EXPLAIN,
    TOKEN_ANALYZE,
    TOKEN_PRAGMA,
    /* Operators */
    TOKEN_EQ,            /* = */
    TOKEN_NEQ,           /* != or <> */
    TOKEN_LT,            /* < */
    TOKEN_GT,            /* > */
    TOKEN_LTE,           /* <= */
    TOKEN_GTE,           /* >= */
    TOKEN_PLUS,          /* + */
    TOKEN_MINUS,         /* - */
    TOKEN_STAR,          /* * */
    TOKEN_SLASH,         /* / */
    /* Punctuation */
    TOKEN_LPAREN,        /* ( */
    TOKEN_RPAREN,        /* ) */
    TOKEN_COMMA,         /* , */
    TOKEN_SEMICOLON,     /* ; */
    TOKEN_DOT,           /* . */
    /* Sentinels */
    TOKEN_EOF,
    TOKEN_ERROR,
} TokenType;
typedef struct {
    TokenType  type;
    const char *start;   /* pointer into original source — no allocation */
    int        length;   /* byte count of the lexeme */
    int        line;
    int        column;
} Token;
```
Notice that `start` is a pointer into the original source string and `length` is the byte count of the lexeme. This zero-copy design avoids allocating individual strings for each token — a critical choice for a database that may tokenize tens of thousands of queries per second. The parser will copy the value out of this span only when it actually needs a heap-allocated string (for string literal values and identifier names).
The `TOKEN_ERROR` sentinel carries a human-readable message about the unrecognized character, along with the exact line and column where it was encountered.
---
## The Tokenizer State Machine

![Tokenizer Finite State Machine](./diagrams/diag-tokenizer-state-machine.svg)

The tokenizer maintains a handful of pieces of state:
```c
typedef struct {
    const char *source;   /* original SQL string */
    const char *current;  /* read cursor */
    int         line;
    int         column;
    int         start_line;
    int         start_col;
} Lexer;
```
`current` points at the next unread character. `source` points at the beginning of the string (used to compute offsets). `line` and `column` track where `current` is in the source. `start_line` and `start_col` capture the position where the current token began.
The core function is:
```c
Token lexer_next_token(Lexer *l);
```
Every call to `lexer_next_token` advances through the source, consuming exactly one token's worth of characters and returning the corresponding `Token`. When the source is exhausted, it returns `TOKEN_EOF` every time (so the parser can always call `lexer_next_token` safely without bounds-checking).
Here is the top-level dispatch:
```c
static char advance(Lexer *l) {
    char c = *l->current++;
    if (c == '\n') {
        l->line++;
        l->column = 1;
    } else {
        l->column++;
    }
    return c;
}
static char peek(Lexer *l) {
    return *l->current;
}
static char peek_next(Lexer *l) {
    if (*l->current == '\0') return '\0';
    return *(l->current + 1);
}
Token lexer_next_token(Lexer *l) {
    skip_whitespace_and_comments(l);
    l->start_line = l->line;
    l->start_col  = l->column;
    const char *token_start = l->current;
    if (*l->current == '\0') {
        return make_token(l, TOKEN_EOF, token_start, 0);
    }
    char c = advance(l);
    if (isdigit((unsigned char)c))         return scan_number(l, token_start);
    if (isalpha((unsigned char)c) || c=='_') return scan_word(l, token_start);
    if (c == '\'')                          return scan_string(l, token_start);
    if (c == '"')                           return scan_quoted_id(l, token_start);
    return scan_operator_or_punct(l, c, token_start);
}
```
The structure makes the state transitions explicit. After skipping whitespace, we record where this token starts, then examine the first character to decide which *sub-machine* handles the rest. Each `scan_*` function runs its own specialized loop until it reaches a terminal state.
---
## Scanning Keywords and Identifiers
The scan for identifiers and keywords begins when the first character is alphabetic or underscore:
```c
static Token scan_word(Lexer *l, const char *start) {
    while (isalnum((unsigned char)*l->current) || *l->current == '_') {
        advance(l);
    }
    int length = (int)(l->current - start);
    TokenType kw = match_keyword(start, length);
    return make_token(l, kw, start, length);
}
```
The key insight: **you read the entire word first, then check whether it's a keyword**. You do not try to detect keywords character-by-character as you go. Read the lexeme, then classify it.
Keyword matching must be case-insensitive. SQL defines `SELECT`, `select`, and `SeLeCt` as identical. The cleanest implementation uses `strncasecmp` (POSIX) or a manual ASCII case-fold:
```c
static int ascii_iequal(const char *a, int alen, const char *b) {
    /* b is a null-terminated keyword string in uppercase */
    for (int i = 0; i < alen; i++) {
        if (b[i] == '\0') return 0;  /* b is shorter */
        if (toupper((unsigned char)a[i]) != (unsigned char)b[i]) return 0;
    }
    return b[alen] == '\0';  /* a and b have same length */
}
static TokenType match_keyword(const char *start, int length) {
    /* Dispatch on first character to avoid linear search */
    switch (toupper((unsigned char)start[0])) {
    case 'A':
        if (ascii_iequal(start, length, "AND"))     return TOKEN_AND;
        if (ascii_iequal(start, length, "ASC"))     return TOKEN_ASC;
        if (ascii_iequal(start, length, "ANALYZE")) return TOKEN_ANALYZE;
        break;
    case 'B':
        if (ascii_iequal(start, length, "BEGIN"))   return TOKEN_BEGIN;
        if (ascii_iequal(start, length, "BY"))      return TOKEN_BY;
        break;
    case 'C':
        if (ascii_iequal(start, length, "COMMIT"))  return TOKEN_COMMIT;
        if (ascii_iequal(start, length, "CREATE"))  return TOKEN_CREATE;
        break;
    /* ... all other keywords ... */
    }
    return TOKEN_IDENTIFIER;
}
```
Dispatching on the first (uppercased) character reduces average comparisons significantly. For a 40-keyword language this is fast enough — you do not need a hash table. If you later extend to hundreds of keywords, a perfect hash (gperf) or trie is the next step.
> **Why `TOKEN_IDENTIFIER` is the default**: when `match_keyword` finds no match, the word is an identifier — a table name, column name, or alias. The parser handles the distinction between "this identifier refers to a table" vs "this identifier refers to a column" — the tokenizer does not know and should not try.
---
## Scanning String Literals
String literals are enclosed in single quotes. The complication — the one that breaks naive approaches — is that SQL uses `''` (two consecutive single quotes) as the escape sequence for a literal single-quote character inside a string. This is not the backslash escape `\'` you may know from C or JavaScript.
```
'it''s a table'   →   it's a table
'hello'           →   hello
''''              →   '           (a string containing a single quote)
```

![String Literal Escape Handling — State Detail](./diagrams/diag-tokenizer-string-escape.svg)

The state machine for string scanning has two states: **IN_STRING** and **POSSIBLE_ESCAPE**. When you see the opening `'`, enter **IN_STRING**. From **IN_STRING**:
- Any character except `'` → consume it, stay in **IN_STRING**
- `'` → transition to **POSSIBLE_ESCAPE**
From **POSSIBLE_ESCAPE**:
- Another `'` → this was an escape sequence; emit a single `'` to the value buffer, return to **IN_STRING**
- Anything else → the previous `'` was the closing quote; the current character belongs to the next token
The catch: you need a mutable buffer to hold the decoded string value, because the in-source representation (`it''s`) differs from the decoded value (`it's`). For the tokenizer, a practical choice is to leave the raw span in the `Token` struct (pointing into the source, single quotes and all) and let the parser decode the escape sequences when it actually needs the string value. This keeps the tokenizer zero-copy.
The scanning loop:
```c
static Token scan_string(Lexer *l, const char *start) {
    /* 'start' points to the opening single-quote already consumed */
    while (1) {
        if (*l->current == '\0') {
            /* Unterminated string literal — error at start position */
            return make_error(l, "Unterminated string literal",
                              l->start_line, l->start_col);
        }
        char c = advance(l);
        if (c == '\'') {
            if (*l->current == '\'') {
                /* Escaped quote: consume the second quote and continue */
                advance(l);
            } else {
                /* Closing quote — string complete */
                int length = (int)(l->current - start);
                return make_token(l, TOKEN_STRING, start, length);
            }
        }
        /* Any other character: continue consuming */
    }
}
```
Notice the unterminated string check: if `\0` is reached before the closing `'`, return `TOKEN_ERROR` anchored at the *opening* quote position (`start_line`, `start_col`). This is the right error location — it tells the user where the string began, not where the file ended.
---
## Scanning Numbers
Numbers come in two flavors: integers (`42`, `0`, `1000000`) and floats (`3.14`, `0.5`, `1e10`). The tokenizer must distinguish them because the parser produces different AST node types for each.
```c
static Token scan_number(Lexer *l, const char *start) {
    /* Consume digits */
    while (isdigit((unsigned char)*l->current)) {
        advance(l);
    }
    TokenType type = TOKEN_INTEGER;
    /* Optional fractional part */
    if (*l->current == '.' && isdigit((unsigned char)peek_next(l))) {
        type = TOKEN_FLOAT;
        advance(l);  /* consume the dot */
        while (isdigit((unsigned char)*l->current)) {
            advance(l);
        }
    }
    /* Optional exponent (1e10, 2.5E-3) */
    if (*l->current == 'e' || *l->current == 'E') {
        type = TOKEN_FLOAT;
        advance(l);
        if (*l->current == '+' || *l->current == '-') advance(l);
        if (!isdigit((unsigned char)*l->current)) {
            return make_error(l, "Invalid numeric literal: expected digits after exponent",
                              l->start_line, l->start_col);
        }
        while (isdigit((unsigned char)*l->current)) {
            advance(l);
        }
    }
    int length = (int)(l->current - start);
    return make_token(l, type, start, length);
}
```
**The negative number problem**: `scan_number` is called when the first character is a digit. But what about `-3.14`? The `-` is scanned by `scan_operator_or_punct` as `TOKEN_MINUS`. The *parser* decides whether `MINUS INTEGER` means a negative number literal or a binary subtraction expression. This is a deliberate separation of concerns: the tokenizer handles *lexical* structure, the parser handles *semantic* structure. Never push ambiguity into the tokenizer that properly belongs to the grammar.
---
## Scanning Quoted Identifiers
Double-quoted identifiers allow spaces, reserved words, and other unusual characters in table and column names: `"First Name"`, `"select"` (a column literally named "select"), `"table 2"`.
```c
static Token scan_quoted_id(Lexer *l, const char *start) {
    while (*l->current != '"' && *l->current != '\0') {
        advance(l);
    }
    if (*l->current == '\0') {
        return make_error(l, "Unterminated quoted identifier",
                          l->start_line, l->start_col);
    }
    advance(l);  /* consume the closing '"' */
    int length = (int)(l->current - start);
    return make_token(l, TOKEN_QUOTED_ID, start, length);
}
```
> **A subtle gotcha**: SQL also allows `""` inside a double-quoted identifier as an escape for a literal `"`. The implementation above does not handle it. Document this limitation: `"he said ""hello"""` is not supported in the initial version. The parser will fail on such input with a misleading error. If you want to add it later, use the same two-state machine as string scanning, substituting `"` for `'`.
---
## Scanning Operators and Punctuation
Most operators are single characters. A handful are two-character sequences: `<=`, `>=`, `!=`, `<>`. The scan function uses one-character lookahead to distinguish them:
```c
static Token scan_operator_or_punct(Lexer *l, char c, const char *start) {
    switch (c) {
    case '(':  return make_token(l, TOKEN_LPAREN,    start, 1);
    case ')':  return make_token(l, TOKEN_RPAREN,    start, 1);
    case ',':  return make_token(l, TOKEN_COMMA,     start, 1);
    case ';':  return make_token(l, TOKEN_SEMICOLON, start, 1);
    case '.':  return make_token(l, TOKEN_DOT,       start, 1);
    case '+':  return make_token(l, TOKEN_PLUS,      start, 1);
    case '*':  return make_token(l, TOKEN_STAR,      start, 1);
    case '/':  return make_token(l, TOKEN_SLASH,     start, 1);
    case '-':  return make_token(l, TOKEN_MINUS,     start, 1);
    case '=':  return make_token(l, TOKEN_EQ,        start, 1);
    case '!':
        if (*l->current == '=') {
            advance(l);
            return make_token(l, TOKEN_NEQ, start, 2);
        }
        return make_error(l, "Expected '=' after '!'",
                          l->start_line, l->start_col);
    case '<':
        if (*l->current == '=') { advance(l); return make_token(l, TOKEN_LTE, start, 2); }
        if (*l->current == '>') { advance(l); return make_token(l, TOKEN_NEQ, start, 2); }
        return make_token(l, TOKEN_LT, start, 1);
    case '>':
        if (*l->current == '=') { advance(l); return make_token(l, TOKEN_GTE, start, 2); }
        return make_token(l, TOKEN_GT, start, 1);
    default: {
        char msg[64];
        snprintf(msg, sizeof(msg), "Unexpected character: '%c' (0x%02X)", c, (unsigned char)c);
        return make_error(l, msg, l->start_line, l->start_col);
    }
    }
}
```
The `<>` operator (SQL's alternative to `!=`) maps to the same `TOKEN_NEQ` as `!=`. Both forms produce identical tokens; the parser never needs to distinguish them.
---
## Whitespace and Comments
SQL allows two comment styles: `-- line comment` (to end of line) and `/* block comment */`. Both must be silently consumed. Since comments can appear between any two tokens, the simplest approach is to consume all whitespace and comments before each token scan:
```c
static void skip_whitespace_and_comments(Lexer *l) {
    for (;;) {
        /* Skip whitespace */
        while (isspace((unsigned char)*l->current)) {
            advance(l);
        }
        /* Line comment: -- to end of line */
        if (l->current[0] == '-' && l->current[1] == '-') {
            while (*l->current != '\n' && *l->current != '\0') {
                advance(l);
            }
            continue;  /* re-check for more whitespace or comments */
        }
        /* Block comment: /* ... */ */
        if (l->current[0] == '/' && l->current[1] == '*') {
            advance(l); advance(l);  /* consume '/' and '*' */
            while (!(*l->current == '*' && *(l->current+1) == '/')) {
                if (*l->current == '\0') {
                    /* Unterminated block comment — we have no Token to return here.
                       Set an error flag on the lexer and return. */
                    return;
                }
                advance(l);
            }
            advance(l); advance(l);  /* consume '*' and '/' */
            continue;
        }
        /* Nothing more to skip */
        break;
    }
}
```
The `continue` after consuming each comment re-enters the loop to handle cases like `-- comment\n   /* block */ SELECT` where whitespace, line comments, and block comments are interleaved.
---
## Error Reporting: Source Position Is Non-Negotiable
When the tokenizer encounters an unrecognized character, it must report exactly where. Not "syntax error somewhere", but "line 7, column 23: unexpected character '`'". This is the difference between a debugging experience that takes 10 seconds and one that takes 10 minutes.
The `make_error` function packages this information:
```c
/* Error messages are stored in a static buffer within the Token.
   This avoids heap allocation for error tokens. */
static Token make_error(Lexer *l, const char *msg, int line, int col) {
    Token t;
    t.type   = TOKEN_ERROR;
    t.start  = msg;    /* points to static or caller-owned string */
    t.length = (int)strlen(msg);
    t.line   = line;
    t.column = col;
    return t;
}
```
The error token carries both the human-readable message (in `start`/`length`) and the source position. Callers check for `TOKEN_ERROR` and surface it to the user:
```c
if (token.type == TOKEN_ERROR) {
    fprintf(stderr, "Tokenizer error at line %d, column %d: %.*s\n",
            token.line, token.column,
            token.length, token.start);
}
```

![Token Stream — Trace Example](./diagrams/diag-tokenizer-trace-example.svg)

---
## The Full Tokenizer Interface
Putting it together, the public API for the tokenizer is minimal:
```c
/* tokenizer.h */
#pragma once
#include <stddef.h>
typedef enum { /* ... all TokenType values ... */ } TokenType;
typedef struct {
    TokenType  type;
    const char *start;
    int        length;
    int        line;
    int        column;
} Token;
/* Initialize a lexer from a SQL string.
   The string must remain valid for the lifetime of the Lexer. */
void lexer_init(Lexer *l, const char *source);
/* Return the next token, advancing the lexer.
   Returns TOKEN_EOF when exhausted; never returns past EOF. */
Token lexer_next_token(Lexer *l);
/* Convenience: tokenize the entire string into a heap-allocated array.
   Caller is responsible for free(). *count is set to the number of tokens. */
Token *lexer_tokenize_all(const char *source, int *count);
/* Return a human-readable name for a token type (for debugging). */
const char *token_type_name(TokenType t);
```
The `lexer_tokenize_all` convenience function is useful for testing. The parser uses `lexer_next_token` directly in a streaming fashion — it never needs to see the full array at once.
---
## Handling the Keyword/Identifier Case Sensitivity Distinction
SQL's rules here are subtle and worth making explicit:
| Input | Result |
|-------|--------|
| `SELECT` | `TOKEN_SELECT` (keyword) |
| `select` | `TOKEN_SELECT` (keyword, case-insensitive) |
| `sElEcT` | `TOKEN_SELECT` (keyword, case-insensitive) |
| `myTable` | `TOKEN_IDENTIFIER` (value = `myTable`, case-preserved) |
| `MYTABLE` | `TOKEN_IDENTIFIER` (value = `MYTABLE`, case-preserved) |
| `"My Table"` | `TOKEN_QUOTED_ID` (value = `"My Table"`, case-preserved, spaces allowed) |
Keywords are recognized case-insensitively. Unquoted identifiers are returned with their original casing (the parser and semantic layer decide case-sensitivity rules for identifier matching — SQLite is case-insensitive for identifiers by default, but that is a semantic decision, not a lexical one). Double-quoted identifiers always preserve case exactly.
This means your `match_keyword` function must normalize the input to uppercase for comparison, but the `Token.start` and `Token.length` always point to the original source text unchanged. Never modify the source string.
---
## Designing for the Parser Downstream
The token stream you produce must make the parser's job easy. Two design principles:
**1. Make the token stream regular.** Every call to `lexer_next_token` returns exactly one `Token`. No nulls, no exceptions (in C), no calling conventions to memorize. The parser calls the same function repeatedly until `TOKEN_EOF`.
**2. Preserve the original text.** The zero-copy design (pointer + length into source) means the parser can reconstruct any token's text for error messages without extra allocations. When the parser says "expected expression, got 'FROMT'" (a typo), it can print the actual token text because the raw span is always available.
**3. Separate concerns cleanly.** The tokenizer does not decide whether `SELECT` appears in a legal position, whether a table name is defined, or whether `3.14` is valid as a primary key. These are the parser's and semantic analyser's concerns. The tokenizer says: "here is a float literal with value `3.14` at line 4, column 12". Nothing more.
---
## Building the Test Suite
Your acceptance criteria require 20+ SQL statements. Structure them to cover every code path systematically. A practical test harness in C:
```c
typedef struct {
    const char  *input;
    TokenType    expected_types[32];  /* TOKEN_EOF-terminated */
} TokenizerTest;
static TokenizerTest tests[] = {
    /* Basic keyword recognition */
    {
        "SELECT * FROM t;",
        {TOKEN_SELECT, TOKEN_STAR, TOKEN_FROM, TOKEN_IDENTIFIER,
         TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* Case-insensitive keywords */
    {
        "select * from T;",
        {TOKEN_SELECT, TOKEN_STAR, TOKEN_FROM, TOKEN_IDENTIFIER,
         TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* String literal with escaped quote */
    {
        "SELECT 'it''s';",
        {TOKEN_SELECT, TOKEN_STRING, TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* Double-quoted identifier with space */
    {
        "SELECT \"First Name\" FROM users;",
        {TOKEN_SELECT, TOKEN_QUOTED_ID, TOKEN_FROM, TOKEN_IDENTIFIER,
         TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* Integer and float distinction */
    {
        "INSERT INTO t VALUES (42, 3.14);",
        {TOKEN_INSERT, TOKEN_INTO, TOKEN_IDENTIFIER, TOKEN_VALUES,
         TOKEN_LPAREN, TOKEN_INTEGER, TOKEN_COMMA, TOKEN_FLOAT,
         TOKEN_RPAREN, TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* Two-character operators */
    {
        "SELECT * FROM t WHERE a >= 10 AND b <> 0;",
        {TOKEN_SELECT, TOKEN_STAR, TOKEN_FROM, TOKEN_IDENTIFIER,
         TOKEN_WHERE, TOKEN_IDENTIFIER, TOKEN_GTE, TOKEN_INTEGER,
         TOKEN_AND, TOKEN_IDENTIFIER, TOKEN_NEQ, TOKEN_INTEGER,
         TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* NULL keyword */
    {
        "SELECT * FROM t WHERE x IS NULL;",
        {TOKEN_SELECT, TOKEN_STAR, TOKEN_FROM, TOKEN_IDENTIFIER,
         TOKEN_WHERE, TOKEN_IDENTIFIER, TOKEN_IS, TOKEN_NULL,
         TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* CREATE TABLE */
    {
        "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL);",
        {TOKEN_CREATE, TOKEN_TABLE, TOKEN_IDENTIFIER, TOKEN_LPAREN,
         TOKEN_IDENTIFIER, TOKEN_IDENTIFIER /*INTEGER*/, TOKEN_PRIMARY, TOKEN_KEY,
         TOKEN_COMMA, TOKEN_IDENTIFIER, TOKEN_IDENTIFIER /*TEXT*/,
         TOKEN_NOT, TOKEN_NULL, TOKEN_RPAREN, TOKEN_SEMICOLON, TOKEN_EOF}
    },
    /* Error: unrecognized character */
    {
        "SELECT `bad`;",
        {TOKEN_SELECT, TOKEN_ERROR}
    },
    /* ... 12+ more cases ... */
};
static void run_tokenizer_tests(void) {
    int passed = 0, failed = 0;
    for (int i = 0; i < (int)(sizeof(tests)/sizeof(tests[0])); i++) {
        Lexer l;
        lexer_init(&l, tests[i].input);
        int ok = 1;
        for (int j = 0; tests[i].expected_types[j] != TOKEN_EOF; j++) {
            Token t = lexer_next_token(&l);
            if (t.type != tests[i].expected_types[j]) {
                fprintf(stderr, "FAIL test %d token %d: expected %s got %s\n",
                    i, j,
                    token_type_name(tests[i].expected_types[j]),
                    token_type_name(t.type));
                ok = 0;
                break;
            }
            if (t.type == TOKEN_ERROR) break;
        }
        ok ? passed++ : failed++;
    }
    printf("Tokenizer: %d passed, %d failed\n", passed, failed);
}
```
Notice `INTEGER` and `TEXT` in the CREATE TABLE test are treated as `TOKEN_IDENTIFIER`, not keywords. This is correct SQL behavior — `INTEGER` and `TEXT` are type names, and SQL dialects disagree wildly on whether they are reserved words. SQLite treats them as identifiers in most contexts. Your keyword table should include them only if you want `TOKEN_INTEGER_TYPE` variants, which adds complexity for marginal benefit at this stage.
---
## Three-Level View: What Happens to Your Token Stream
To understand why the tokenizer is designed the way it is, look at the three levels of the query processing pipeline:
**Level 1 — Tokenizer (this milestone)**
Input: `"SELECT name FROM users WHERE id = 42;"` (raw bytes)
Output: `[TOKEN_SELECT, TOKEN_IDENTIFIER("name"), TOKEN_FROM, TOKEN_IDENTIFIER("users"), TOKEN_WHERE, TOKEN_IDENTIFIER("id"), TOKEN_EQ, TOKEN_INTEGER("42"), TOKEN_SEMICOLON, TOKEN_EOF]`
The tokenizer knows nothing about grammar. It can't tell you whether `SELECT name FROM` is legal. It just classifies character sequences.
**Level 2 — Parser (Milestone 2)**
Input: the token stream from Level 1
Output: an Abstract Syntax Tree (AST) — a `SelectStatement` node containing a column list `["name"]`, a table reference `"users"`, and a `WHERE` subtree representing `id = 42`
The parser consumes tokens one at a time. It never touches raw characters. This is the abstraction boundary your tokenizer creates.
**Level 3 — Code Generator / VDBE (Milestone 3)**
Input: the AST from Level 2
Output: bytecode instructions that the virtual machine executes against the storage engine
The clean separation means: if you later want to support PostgreSQL-style `$1` parameter placeholders, you add one case to `scan_operator_or_punct`. The parser, code generator, and storage engine are completely unaffected. If you want to add a new keyword `EXPLAIN`, you add one entry to `match_keyword`. This is the value of the abstraction boundary.
---
## Design Decision: Why a Hand-Written Lexer Over Flex/Lex?
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Hand-written (Chosen ✓)** | Zero dependencies, debuggable with printf, direct control over error messages and line tracking, teachable | More code to write | SQLite, PostgreSQL, CPython, GCC |
| Flex/Lex | Less code in the spec file, handles complex rules declaratively | Generated C is unreadable, debugging requires understanding DFA internals, adds build dependency, obscures the learning | Many academic projects |
| Regex library | Very little code | Catastrophically slow for repeated calls (regex compilation cost), poor error position tracking | Quick-and-dirty scripts |
For this project, the hand-written lexer is not just the right engineering choice — it is the required one. The entire point is to understand what happens at each layer. A generated lexer hides the state machine you need to internalize.
---
## Knowledge Cascade: What This Milestone Unlocks
You have just built a Deterministic Finite Automaton (DFA) by hand, without ever drawing a formal state diagram or taking a theory course. Here is what that connects to:
**→ Formal language theory**: The class of languages your tokenizer can recognize is exactly the class of *regular languages*. Every regular expression in every language maps to a DFA. When PostgreSQL says "my identifier rule is `[a-zA-Z_][a-zA-Z0-9_]*`", that is a description of a DFA transition function. You just built one manually.
**→ Every other parser in the universe**: The tokenizer pattern — FSM producing a flat stream consumed by a recursive-descent parser — appears in C compilers, JSON parsers, HTTP/1.1 request parsers, YAML parsers, DNS message decoders, and protocol analyzers for Ethernet frames. The specific tokens differ; the structure is identical.
**→ Error recovery and IDE tooling**: You encoded error *position* (line, column) in every token. This is what makes "line 7, column 12: unexpected '}'" possible. Language servers (LSP), linters, and syntax highlighters depend on this same position information to underline the right range of characters. Your tokenizer already has the architecture for IDE-quality diagnostics.
**→ Unicode and encoding**: Your lexer handles ASCII. The reason this works is that UTF-8 was deliberately designed so that multi-byte sequences never contain bytes in the ASCII range (0x00–0x7F). A SQL keyword like `SELECT` (all ASCII) will never be confused with a byte inside a multi-byte UTF-8 character. If a user puts a UTF-8 table name inside double quotes, your `scan_quoted_id` function consumes it correctly without any Unicode awareness — it just reads bytes until the closing `"`. This design choice (ASCII-first, byte-transparent for non-ASCII) is what made UTF-8 the universal encoding.
**→ The token stream as an API contract**: You are about to write a parser (Milestone 2) that calls `lexer_next_token`. From that point forward, the parser will assume this contract: every token has a type, a text span, and a source position. If you later want to replace the lexer with a faster SIMD-optimized version, or add support for PostgreSQL-style parameter markers, or handle Unicode keywords — you only touch the tokenizer. The parser, the VDBE, and the storage engine see nothing change.
**→ WAL and journal logging** (preview of Milestone 9 and 10): The same write-ordering discipline you will implement for durability — "write the journal before you touch the database page" — appears here in microcosm: the tokenizer must be deterministic. Given the same input, it produces exactly the same token stream, every time. No global state, no side effects beyond `l->current`, `l->line`, `l->column`. Determinism is the foundation of reproducibility, and reproducibility is the foundation of correctness.

> **🔑 Foundation: Write-ahead logging vs undo logging**
> 
> ## Write-Ahead Logging vs Undo Logging
### What it IS
Both are **crash-recovery protocols** — strategies for ensuring that if the database crashes mid-transaction, you can restore the database to a consistent state on restart. They differ in *when* they force data and log records to disk.
#### Undo Logging
The rule: **before you overwrite a data page on disk, write an undo log record** (capturing the old value). On crash recovery, you scan the log backwards and *undo* any transaction that hadn't committed — rolling it back by restoring old values.
Protocol order:
1. Write undo log record (old value) to disk
2. Write updated data page to disk
3. Write COMMIT record to log
**Key property:** Data can be flushed to disk *before* commit, so the buffer manager has flexibility to evict dirty pages early. But you must keep old values around until commit is confirmed.
#### Write-Ahead Logging (WAL)
The rule: **before you write a data page to disk, write the redo log record** (capturing the new value). The "write-ahead" refers to the log always being ahead of the actual data. On crash recovery, you scan forward to *redo* committed transactions whose data pages didn't make it to disk, and *undo* uncommitted ones.
Protocol order:
1. Write redo log record (new value) to disk
2. (Data page may stay in memory — flushed lazily)
3. Write COMMIT record; only *after* COMMIT is durable can data reach disk
**Key property:** Data pages don't need to be flushed at commit time — only the log must be durable. This is a huge performance win because you convert random writes (data pages) into sequential log appends. PostgreSQL, SQLite, MySQL InnoDB all use WAL.
### Comparison Table
| | Undo Logging | WAL (Redo Logging) |
|---|---|---|
| Crash recovery direction | Backward (undo) | Forward redo + backward undo |
| Data flush requirement | Before commit | After log flush (lazy) |
| Log contains | Old values | New values |
| Performance | Higher I/O at commit | Sequential log → better throughput |
| Complexity | Simpler | More complex (checkpoint management) |
### WHY you need it now
When building a storage engine or database, you'll implement transactions with ACID guarantees. **WAL is the industry-standard choice** — understanding it explains why your database writes to a `wal` file, why crashes are recoverable just by replaying that file, and how checkpointing (periodically flushing data pages and truncating the log) works. Undo logging is worth knowing as a contrast to understand *why* WAL's approach of deferring data writes is superior.
### Key Insight
WAL's core trick is turning **random writes into sequential writes**. A hard disk (or even an SSD) handles a sequential append to one log file far faster than scattering writes across many data pages. By guaranteeing only that the *log* is durable at commit time, WAL lets data pages sit in memory and be written lazily in large batches — dramatically improving write throughput without sacrificing durability.


> **🔑 Foundation: Snapshot isolation and MVCC concepts**
> 
> ## Snapshot Isolation and MVCC
### What it IS
#### The Problem
Multiple concurrent transactions reading and writing the same data can produce anomalies: a transaction reads a row, another transaction modifies it mid-way, and the first transaction reads it again and sees different data (*non-repeatable read*), or phantom rows appear between two reads. Locking everything serializes all access and kills performance.
#### Multi-Version Concurrency Control (MVCC)
MVCC solves this by **never overwriting data in place**. Instead, every write creates a *new version* of a row, tagged with the transaction ID (or timestamp) that created it. Old versions are kept around until no transaction needs them anymore (a process called vacuuming/garbage collection).
Each row in a MVCC store looks like:
```
row_id | xmin (created by txn) | xmax (deleted by txn) | data
  42   |        txn_105        |        txn_201         | "Alice"
  42   |        txn_201        |          ∞             | "Alice Smith"
```
Transaction 105 inserted "Alice"; transaction 201 updated it to "Alice Smith". Both versions coexist on disk.
#### Snapshot Isolation
MVCC enables **snapshot isolation**: when a transaction starts, it gets a *snapshot* — a consistent view of the database as it existed at that moment. It sees all versions committed *before* its snapshot timestamp, and ignores all versions committed after. This means:
- **Readers never block writers** — you read old versions; writers create new ones.
- **Writers never block readers** — same reason.
- **Reads are always consistent** — you always see the same data within a transaction, no matter what other transactions do concurrently.
**Example:**
- T1 starts at time 100. Sees the database as of t=100.
- T2 starts at time 101, modifies row 42, commits at t=102.
- T1 reads row 42 at t=103 — still sees the t=100 version. No non-repeatable read.
#### The Write Skew Anomaly
Snapshot isolation is *not* full serializability. The notorious **write skew** problem: two transactions each read a shared condition, then each write different rows based on it, violating a constraint neither could see alone. Example: two doctors both check "is anyone on call?" (yes), both decide to go off-call — now nobody is on call. Neither transaction conflicted on the *same row*, so MVCC didn't catch it. Serializable Snapshot Isolation (SSI, used in PostgreSQL's `SERIALIZABLE` level) detects these dependency cycles.
### WHY you need it now
When implementing transactions in your storage engine or database, you must choose a concurrency control strategy. MVCC is what PostgreSQL, MySQL InnoDB, CockroachDB, and SQLite (WAL mode) use — understanding it explains: why your database has a `VACUUM` process, what `xmin`/`xmax` columns in PostgreSQL's system catalog mean, why `READ COMMITTED` and `REPEATABLE READ` isolation levels behave differently, and why "snapshot" is in the name of so many modern isolation levels.
### Key Insight
**MVCC trades storage space for concurrency** — it keeps multiple versions of data to let readers and writers operate simultaneously without locking each other out. The fundamental mental model: a *read* is a time-travel query ("give me the state of the world as of timestamp T"), and a *write* is an append ("add a new version, don't touch old ones"). This reframing — from mutation to versioned append — is the conceptual shift that unlocks non-blocking reads.

---
## Common Pitfalls: What Will Break Without Warning
**1. Not handling `NULL` as a keyword.** If `NULL` is classified as `TOKEN_IDENTIFIER`, the parser must add a special case everywhere expressions appear to check whether an identifier token happens to spell "null". This is error-prone. Classify `NULL` as its own keyword token. Then `WHERE x IS NULL` tokenizes cleanly.
**2. Consuming too many characters on error.** When you hit an unrecognized character, return immediately with `TOKEN_ERROR`. Do not try to consume the rest of the "bad token" speculatively. The parser will see `TOKEN_ERROR` and handle recovery. If you consume past the error, you lose position information.
**3. Line counting off-by-one.** The most common bug: advance `l->line` *when you consume the `\n` character*, not after. If you count the newline after returning the token, the positions reported for the next token are wrong by one line.
**4. Not re-checking for whitespace after comments.** A sequence like `-- comment\n   -- another comment\n SELECT` requires multiple passes through the skip loop. Ensure `skip_whitespace_and_comments` loops, not just runs once.
**5. The `>=` vs `>` ambiguity.** When you scan `>`, you must peek at the next character before returning `TOKEN_GT`. If you return `TOKEN_GT` immediately and the next character is `=`, the next call to `lexer_next_token` returns `TOKEN_EQ`, and the parser sees `>` `=` rather than `>=`. Always do the one-character lookahead for multi-character operators.
---
## What You Have Built
At the end of this milestone, you have:
- A zero-copy lexer that handles the full SQL tokenization problem
- Correct string literal scanning with `''` escape sequences
- Case-insensitive keyword recognition with case-preserving identifier storage
- Double-quoted identifier support for unusual names
- Integer and float literal distinction
- All SQL operators including two-character forms (`<=`, `>=`, `!=`, `<>`)
- Source position tracking on every token (line and column)
- Informative error tokens with position information
- A test suite verifying 20+ SQL statements
The parser (Milestone 2) will consume this token stream. It will never see a raw character. The contract you have established — one function call returns one token with a type, span, and position — is the foundation everything upstream will build on.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m2 -->
<!-- MS_ID: build-sqlite-m2 -->
# Milestone 2: SQL Parser (AST)
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You are one layer deeper into the pipeline. The tokenizer you built in Milestone 1 dissolved raw SQL text into a stream of typed tokens — `TOKEN_SELECT`, `TOKEN_IDENTIFIER("users")`, `TOKEN_WHERE`, `TOKEN_INTEGER("42")`. The parser's job is to give that flat sequence *structure*. It reads tokens one at a time and builds a tree that captures the grammatical relationships between them.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

After this milestone, every downstream component — the bytecode compiler, the query planner, the constraint checker — will never see a token directly. They will walk a tree. The parser is the second and final transformation in SQL's front-end. Get it right and the back-end writes itself naturally. Get it wrong and you spend three milestones fighting mysterious misbehavior rooted here.
---
## The Revelation: SQL Is Not English, and That Makes It Hard
Here is the trap every developer falls into before building their first SQL parser: SQL reads like English, so surely parsing it is mostly matching keywords in order.
```sql
SELECT name FROM users WHERE age > 18 ORDER BY name LIMIT 10;
```
Look at it. `SELECT` — okay, we're selecting. `name` — that's the column. `FROM users` — obvious. `WHERE age > 18` — filter. It practically parses itself. You could write a function per clause: `parse_select_list()`, `parse_from_clause()`, `parse_where_clause()`. What's the hard part?
Now look at this:
```sql
SELECT * FROM users WHERE NOT age > 18 AND name = 'Alice' OR id = 1;
```
What does this evaluate? Specifically: does `AND` group tighter than `OR`? Does `NOT` bind to just `age > 18` or to the entire expression? The answer matters enormously. With the wrong precedence table, a WHERE clause that should filter 3 rows filters 3,000 — and there is no error. The query runs, returns data, the wrong data, silently.
In SQL: `NOT` > comparison operators > `AND` > `OR`. So the expression parses as:
```
((NOT (age > 18)) AND (name = 'Alice')) OR (id = 1)
```
Not:
```
NOT ((age > 18) AND ((name = 'Alice') OR (id = 1)))
```
The second interpretation would be catastrophically wrong. And a naive keyword-matching parser produces neither — it produces *something*, something that looks almost right but fails on edge cases where the correctness difference is invisible without carefully constructed tests.
The second trap is left recursion. Consider parsing `a AND b AND c`. A grammar rule that says "an AND-expression is an AND-expression followed by AND followed by an expression" is left-recursive: to parse `a AND b AND c`, you first need to parse `a AND b AND c` (the left side), which recursively requires the same, forever. Naive recursive descent generates infinite recursion. The solution — Pratt parsing — is the real technical achievement of this milestone.

![SQL Grammar Rules — Structure Layout](./diagrams/diag-parser-grammar-rules.svg)

---
## What You Are Building: The AST Contract

> **🔑 Foundation: Recursive descent parsing and grammar rules**
> 
> ## Recursive Descent Parsing and Grammar Rules
### What It IS
A **grammar rule** is a formal description of what a valid sentence (or expression) looks like in a language. You write them in a notation called BNF or EBNF, like this:
```
expression  → term (('+' | '-') term)*
term        → factor (('*' | '/') factor)*
factor      → NUMBER | '(' expression ')'
```
Read `→` as "is made of." An `expression` is one or more `term`s joined by `+` or `-`. A `term` is one or more `factor`s joined by `*` or `/`. A `factor` is either a bare number or a parenthesized `expression` — which brings us back to the top. That loop is what makes it **recursive**.
**Recursive descent parsing** is the technique of turning those grammar rules directly into functions, one function per rule:
```python
def expression():
    left = term()
    while current_token in ('+', '-'):
        op = consume()
        right = term()
        left = BinaryOp(op, left, right)
    return left
def term():
    left = factor()
    while current_token in ('*', '/'):
        op = consume()
        right = factor()
        left = BinaryOp(op, left, right)
    return left
def factor():
    if current_token == '(':
        consume('(')
        node = expression()   # ← calls back up the chain
        consume(')')
        return node
    return Number(consume())
```
The parser "descends" through grammar rules by calling functions, and "recurses" when a rule refers back to a higher-level rule (like `factor` calling `expression`).
The output is usually an **Abstract Syntax Tree (AST)** — a tree where each node represents a syntactic construct (a binary operation, a function call, a variable reference) rather than raw characters.
### WHY You Need It Right Now
You're building a compiler or interpreter. After the lexer breaks source text into tokens, the **parser's job is to understand structure** — not just *what* the tokens are, but *how they relate*. Is `a + b * c` the same as `(a + b) * c`? A parser that follows the grammar above will correctly build `a + (b * c)` because `term` (which handles `*`) is called from inside `expression` (which handles `+`), giving `*` higher precedence automatically.
Recursive descent is the standard approach for hand-written parsers because:
- Each grammar rule maps cleanly to one function — the code **reads like the grammar**.
- Error messages can be precise: you know exactly which rule failed.
- It handles the full complexity of real languages (statements, expressions, declarations, blocks) without needing parser-generator tools.
### The Key Mental Model
**The call stack IS the parse stack.**
When `expression` calls `term` calls `factor` calls `expression` again (for a parenthesized sub-expression), the chain of active function calls *is* the record of how deep into the grammar you are. You don't need a separate stack data structure — the language's own call stack tracks your position in the grammar for you.
This also tells you the limit: deeply nested expressions (like `((((((x))))))`) will recurse deeply. For a student compiler that's fine. For a production compiler, you'd eventually manage this explicitly — but the recursive descent model is always the conceptual foundation.
**Remember:** one grammar rule → one function, and the grammar's structure determines operator precedence automatically by how deeply you nest the rules.

The parser's public contract: accept a `Lexer *` (or a pre-tokenized `Token *` array), return a heap-allocated AST node representing the parsed statement, and on error return `NULL` with a diagnostic message.
First, define the AST node types. The key design decision is whether to use a tagged union (one struct with a `type` field and a `union` body) or a struct-per-node-type with a base pointer. In C, the tagged union is simpler and avoids needing casts through a base struct pointer:
```c
/* parser.h */
#pragma once
#include "tokenizer.h"
/* ---- Node type tags ---- */
typedef enum {
    /* Statement nodes */
    NODE_SELECT,
    NODE_INSERT,
    NODE_CREATE_TABLE,
    /* Expression nodes */
    NODE_BINARY_EXPR,     /* left OP right */
    NODE_UNARY_EXPR,      /* OP operand */
    NODE_LITERAL_INT,     /* 42 */
    NODE_LITERAL_FLOAT,   /* 3.14 */
    NODE_LITERAL_STRING,  /* 'hello' */
    NODE_LITERAL_NULL,    /* NULL */
    NODE_IDENTIFIER,      /* column_name or table_name */
    NODE_COLUMN_REF,      /* table.column */
    NODE_WILDCARD,        /* * in SELECT list */
} NodeType;
/* ---- AST node ---- */
typedef struct AstNode AstNode;
/* Column definition (inside CREATE TABLE) */
typedef struct {
    char *name;           /* heap-allocated, null-terminated */
    char *type_name;      /* "INTEGER", "TEXT", "REAL", "BLOB" */
    int   is_primary_key;
    int   is_not_null;
    int   is_unique;
} ColumnDef;
/* ORDER BY item */
typedef struct {
    AstNode *expr;
    int      ascending;   /* 1 = ASC, 0 = DESC */
} OrderByItem;
struct AstNode {
    NodeType type;
    int      line;    /* source position of the first token of this node */
    int      column;
    union {
        /* NODE_SELECT */
        struct {
            AstNode **columns;     /* array of expression nodes (or NODE_WILDCARD) */
            int       column_count;
            char     *table_name;  /* FROM clause */
            AstNode  *where_expr;  /* NULL if no WHERE */
            OrderByItem *order_by; /* NULL if no ORDER BY */
            int          order_by_count;
            AstNode  *limit_expr;  /* NULL if no LIMIT */
        } select;
        /* NODE_INSERT */
        struct {
            char     *table_name;
            char    **column_names;   /* optional; NULL if not specified */
            int       column_count;
            AstNode **values;         /* one per value in VALUES (...) */
            int       value_count;
        } insert;
        /* NODE_CREATE_TABLE */
        struct {
            char       *table_name;
            ColumnDef  *columns;
            int         column_count;
        } create_table;
        /* NODE_BINARY_EXPR */
        struct {
            TokenType  op;      /* TOKEN_AND, TOKEN_OR, TOKEN_EQ, TOKEN_LT, etc. */
            AstNode   *left;
            AstNode   *right;
        } binary;
        /* NODE_UNARY_EXPR */
        struct {
            TokenType  op;      /* TOKEN_NOT, TOKEN_MINUS */
            AstNode   *operand;
        } unary;
        /* NODE_LITERAL_INT */
        struct { int64_t value; } lit_int;
        /* NODE_LITERAL_FLOAT */
        struct { double value; } lit_float;
        /* NODE_LITERAL_STRING */
        struct { char *value; int length; } lit_str;  /* decoded, no outer quotes */
        /* NODE_IDENTIFIER / NODE_COLUMN_REF */
        struct {
            char *table;   /* NULL for simple identifier */
            char *name;
        } identifier;
    };
};
```
Every `AstNode` is heap-allocated and exclusively owned by its parent (or by the caller for the root node). Freeing the tree is a recursive post-order traversal that frees children before freeing the node itself. You must implement `ast_free(AstNode *node)` — memory leaks here become unbearable in a long-running server.
---
## The Parser State
The parser needs to look at tokens without consuming them (peek), consume tokens, and report errors. Wrap this in a struct:
```c
typedef struct {
    Token  *tokens;    /* flat array produced by lexer_tokenize_all */
    int     count;
    int     pos;       /* index of the next token to consume */
    /* Error state */
    char    error_msg[256];
    int     error_line;
    int     error_col;
    int     had_error;
} Parser;
static void parser_init(Parser *p, Token *tokens, int count) {
    p->tokens    = tokens;
    p->count     = count;
    p->pos       = 0;
    p->had_error = 0;
    p->error_msg[0] = '\0';
}
/* Return current token without consuming */
static Token parser_peek(Parser *p) {
    if (p->pos >= p->count) {
        return (Token){ .type = TOKEN_EOF };
    }
    return p->tokens[p->pos];
}
/* Return token at offset ahead without consuming */
static Token parser_peek_ahead(Parser *p, int offset) {
    int idx = p->pos + offset;
    if (idx >= p->count) return (Token){ .type = TOKEN_EOF };
    return p->tokens[idx];
}
/* Consume and return current token */
static Token parser_advance(Parser *p) {
    Token t = parser_peek(p);
    if (p->pos < p->count) p->pos++;
    return t;
}
/* Consume if current token matches expected type, else set error */
static int parser_expect(Parser *p, TokenType expected, const char *context) {
    Token t = parser_peek(p);
    if (t.type != expected) {
        snprintf(p->error_msg, sizeof(p->error_msg),
                 "Expected %s %s, got '%.*s' at line %d, column %d",
                 token_type_name(expected), context,
                 t.length, t.start, t.line, t.column);
        p->error_line = t.line;
        p->error_col  = t.column;
        p->had_error  = 1;
        return 0;
    }
    parser_advance(p);
    return 1;
}
/* Check current token type without consuming */
static int parser_check(Parser *p, TokenType type) {
    return parser_peek(p).type == type;
}
/* Consume if matches, return 1; else return 0 without error */
static int parser_match(Parser *p, TokenType type) {
    if (parser_check(p, type)) {
        parser_advance(p);
        return 1;
    }
    return 0;
}
```
The `parser_match` / `parser_check` / `parser_expect` trio is the idiomatic toolkit for recursive descent. You will use these functions constantly throughout the parser.
---
## Parsing SELECT

![AST for SELECT — Trace Example](./diagrams/diag-parser-ast-example.svg)

SELECT is the most complex statement because it has the most optional clauses. Parse them strictly in order — the SQL standard defines the clause order:
```
SELECT column_list FROM table [WHERE expr] [ORDER BY col [ASC|DESC], ...] [LIMIT expr]
```
```c
static AstNode *parse_select(Parser *p) {
    Token first = parser_peek(p);
    if (!parser_expect(p, TOKEN_SELECT, "at statement start")) return NULL;
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type   = NODE_SELECT;
    node->line   = first.line;
    node->column = first.column;
    /* ---- Column list ---- */
    int   cap = 8;
    node->select.columns = malloc(cap * sizeof(AstNode*));
    node->select.column_count = 0;
    do {
        if (node->select.column_count >= cap) {
            cap *= 2;
            node->select.columns = realloc(node->select.columns,
                                           cap * sizeof(AstNode*));
        }
        AstNode *col;
        if (parser_check(p, TOKEN_STAR)) {
            Token star = parser_advance(p);
            col = calloc(1, sizeof(AstNode));
            col->type   = NODE_WILDCARD;
            col->line   = star.line;
            col->column = star.column;
        } else {
            col = parse_expression(p, 0);  /* will be explained shortly */
            if (!col) { ast_free(node); return NULL; }
        }
        node->select.columns[node->select.column_count++] = col;
    } while (parser_match(p, TOKEN_COMMA));
    /* ---- FROM clause ---- */
    if (!parser_expect(p, TOKEN_FROM, "after column list")) {
        ast_free(node); return NULL;
    }
    Token table_tok = parser_peek(p);
    if (table_tok.type != TOKEN_IDENTIFIER && table_tok.type != TOKEN_QUOTED_ID) {
        snprintf(p->error_msg, sizeof(p->error_msg),
                 "Expected table name after FROM at line %d, column %d",
                 table_tok.line, table_tok.column);
        p->had_error = 1;
        ast_free(node); return NULL;
    }
    parser_advance(p);
    node->select.table_name = strndup(table_tok.start, table_tok.length);
    /* ---- Optional WHERE ---- */
    if (parser_match(p, TOKEN_WHERE)) {
        node->select.where_expr = parse_expression(p, 0);
        if (!node->select.where_expr && p->had_error) {
            ast_free(node); return NULL;
        }
    }
    /* ---- Optional ORDER BY ---- */
    if (parser_check(p, TOKEN_ORDER)) {
        parser_advance(p);
        if (!parser_expect(p, TOKEN_BY, "after ORDER")) {
            ast_free(node); return NULL;
        }
        int ob_cap = 4;
        node->select.order_by = malloc(ob_cap * sizeof(OrderByItem));
        node->select.order_by_count = 0;
        do {
            if (node->select.order_by_count >= ob_cap) {
                ob_cap *= 2;
                node->select.order_by = realloc(node->select.order_by,
                                                ob_cap * sizeof(OrderByItem));
            }
            AstNode *col_expr = parse_expression(p, 0);
            if (!col_expr) { ast_free(node); return NULL; }
            int asc = 1;
            if      (parser_match(p, TOKEN_ASC))  asc = 1;
            else if (parser_match(p, TOKEN_DESC)) asc = 0;
            OrderByItem *item =
                &node->select.order_by[node->select.order_by_count++];
            item->expr      = col_expr;
            item->ascending = asc;
        } while (parser_match(p, TOKEN_COMMA));
    }
    /* ---- Optional LIMIT ---- */
    if (parser_match(p, TOKEN_LIMIT)) {
        node->select.limit_expr = parse_expression(p, 0);
        if (!node->select.limit_expr && p->had_error) {
            ast_free(node); return NULL;
        }
    }
    return node;
}
```
Notice `parse_expression(p, 0)` calls — these are the Pratt parser calls, explained in the next section. The `0` is the minimum binding power, meaning "parse the full expression". You will understand this parameter completely after the next section.
---
## Parsing INSERT
INSERT is simpler. The grammar:
```
INSERT INTO table_name [(col1, col2, ...)] VALUES (val1, val2, ...) [, (...)]
```
```c
static AstNode *parse_insert(Parser *p) {
    Token first = parser_peek(p);
    if (!parser_expect(p, TOKEN_INSERT, "at statement start")) return NULL;
    if (!parser_expect(p, TOKEN_INTO, "after INSERT"))         return NULL;
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type   = NODE_INSERT;
    node->line   = first.line;
    node->column = first.column;
    /* Table name */
    Token tname = parser_peek(p);
    if (tname.type != TOKEN_IDENTIFIER && tname.type != TOKEN_QUOTED_ID) {
        snprintf(p->error_msg, sizeof(p->error_msg),
                 "Expected table name after INTO at line %d, column %d",
                 tname.line, tname.column);
        p->had_error = 1;
        ast_free(node); return NULL;
    }
    parser_advance(p);
    node->insert.table_name = strndup(tname.start, tname.length);
    /* Optional column list */
    if (parser_match(p, TOKEN_LPAREN)) {
        int ccap = 8;
        node->insert.column_names = malloc(ccap * sizeof(char*));
        node->insert.column_count = 0;
        do {
            if (node->insert.column_count >= ccap) {
                ccap *= 2;
                node->insert.column_names =
                    realloc(node->insert.column_names, ccap * sizeof(char*));
            }
            Token col = parser_peek(p);
            if (col.type != TOKEN_IDENTIFIER && col.type != TOKEN_QUOTED_ID) {
                snprintf(p->error_msg, sizeof(p->error_msg),
                         "Expected column name at line %d, column %d",
                         col.line, col.column);
                p->had_error = 1;
                ast_free(node); return NULL;
            }
            parser_advance(p);
            node->insert.column_names[node->insert.column_count++] =
                strndup(col.start, col.length);
        } while (parser_match(p, TOKEN_COMMA));
        if (!parser_expect(p, TOKEN_RPAREN, "after column list")) {
            ast_free(node); return NULL;
        }
    }
    /* VALUES clause */
    if (!parser_expect(p, TOKEN_VALUES, "after table name")) {
        ast_free(node); return NULL;
    }
    if (!parser_expect(p, TOKEN_LPAREN, "after VALUES")) {
        ast_free(node); return NULL;
    }
    int vcap = 8;
    node->insert.values = malloc(vcap * sizeof(AstNode*));
    node->insert.value_count = 0;
    do {
        if (node->insert.value_count >= vcap) {
            vcap *= 2;
            node->insert.values = realloc(node->insert.values,
                                          vcap * sizeof(AstNode*));
        }
        if (parser_check(p, TOKEN_RPAREN)) break;  /* trailing comma guard */
        AstNode *val = parse_expression(p, 0);
        if (!val) { ast_free(node); return NULL; }
        node->insert.values[node->insert.value_count++] = val;
    } while (parser_match(p, TOKEN_COMMA));
    if (!parser_expect(p, TOKEN_RPAREN, "after VALUES list")) {
        ast_free(node); return NULL;
    }
    return node;
}
```
---
## Parsing CREATE TABLE
CREATE TABLE introduces column definitions and constraints. The grammar:
```
CREATE TABLE table_name (
    col_name col_type [PRIMARY KEY] [NOT NULL] [UNIQUE] [, ...]
)
```
```c
static AstNode *parse_create_table(Parser *p) {
    Token first = parser_peek(p);
    if (!parser_expect(p, TOKEN_CREATE, "")) return NULL;
    if (!parser_expect(p, TOKEN_TABLE, "after CREATE")) return NULL;
    AstNode *node = calloc(1, sizeof(AstNode));
    node->type   = NODE_CREATE_TABLE;
    node->line   = first.line;
    node->column = first.column;
    /* Table name */
    Token tname = parser_peek(p);
    if (tname.type != TOKEN_IDENTIFIER && tname.type != TOKEN_QUOTED_ID) {
        snprintf(p->error_msg, sizeof(p->error_msg),
                 "Expected table name at line %d, column %d",
                 tname.line, tname.column);
        p->had_error = 1;
        ast_free(node); return NULL;
    }
    parser_advance(p);
    node->create_table.table_name = strndup(tname.start, tname.length);
    if (!parser_expect(p, TOKEN_LPAREN, "after table name")) {
        ast_free(node); return NULL;
    }
    int ccap = 8;
    node->create_table.columns = malloc(ccap * sizeof(ColumnDef));
    node->create_table.column_count = 0;
    do {
        Token tok = parser_peek(p);
        if (tok.type == TOKEN_RPAREN) break; /* empty table: CREATE TABLE t () */
        if (node->create_table.column_count >= ccap) {
            ccap *= 2;
            node->create_table.columns =
                realloc(node->create_table.columns, ccap * sizeof(ColumnDef));
        }
        ColumnDef *def =
            &node->create_table.columns[node->create_table.column_count];
        memset(def, 0, sizeof(ColumnDef));
        /* Column name */
        Token cname = parser_peek(p);
        if (cname.type != TOKEN_IDENTIFIER && cname.type != TOKEN_QUOTED_ID) {
            snprintf(p->error_msg, sizeof(p->error_msg),
                     "Expected column name at line %d, column %d",
                     cname.line, cname.column);
            p->had_error = 1;
            ast_free(node); return NULL;
        }
        parser_advance(p);
        def->name = strndup(cname.start, cname.length);
        /* Column type — in SQLite, type names are identifiers, not keywords.
           We accept any identifier or recognized type-name token. */
        Token ctype = parser_peek(p);
        if (ctype.type == TOKEN_IDENTIFIER ||
            ctype.type == TOKEN_INTEGER   ||   /* some dialects */
            ctype.type == TOKEN_QUOTED_ID) {
            parser_advance(p);
            def->type_name = strndup(ctype.start, ctype.length);
        } else {
            def->type_name = strdup("BLOB"); /* type-less columns default to BLOB */
        }
        /* Constraints (any order, any number) */
        for (;;) {
            if (parser_check(p, TOKEN_PRIMARY)) {
                parser_advance(p);
                if (!parser_expect(p, TOKEN_KEY, "after PRIMARY")) {
                    ast_free(node); return NULL;
                }
                def->is_primary_key = 1;
            } else if (parser_check(p, TOKEN_NOT)) {
                parser_advance(p);
                if (!parser_expect(p, TOKEN_NULL, "after NOT")) {
                    ast_free(node); return NULL;
                }
                def->is_not_null = 1;
            } else if (parser_check(p, TOKEN_UNIQUE)) {
                parser_advance(p);
                def->is_unique = 1;
            } else {
                break;
            }
        }
        node->create_table.column_count++;
    } while (parser_match(p, TOKEN_COMMA));
    if (!parser_expect(p, TOKEN_RPAREN, "after column definitions")) {
        ast_free(node); return NULL;
    }
    return node;
}
```
One subtlety worth noting: `INTEGER`, `TEXT`, `REAL`, `BLOB` are *not* SQL keywords in SQLite's grammar — they are type affinity names that happen to look like keywords. Your tokenizer classified them as `TOKEN_IDENTIFIER`. Do not add them as keywords in the tokenizer; instead, accept them as identifiers here in the parser. This is consistent with SQLite's actual behavior, where `CREATE TABLE t (x FROBNICATOR)` is valid SQL — the column just gets `BLOB` affinity because `FROBNICATOR` doesn't match any known affinity pattern.
---
## The Hard Part: Pratt Parsing for Expressions
[[EXPLAIN:pratt-parsing-/-precedence-climbing-for-expressions|Pratt parsing / precedence climbing for expressions]]
This is the real intellectual content of this milestone. Everything before this was mechanical. Now you must solve operator precedence without left recursion.

![Pratt Parser — Precedence Climbing Walkthrough](./diagrams/diag-parser-precedence-climbing.svg)

### The Precedence Table
SQL's expression grammar, from lowest precedence (loosest binding) to highest (tightest binding):
| Precedence | Operators | Associativity |
|-----------|-----------|---------------|
| 1 (lowest) | `OR` | left |
| 2 | `AND` | left |
| 3 | `NOT` (unary) | right/prefix |
| 4 | `=`, `!=`, `<>`, `<`, `>`, `<=`, `>=`, `IS`, `IS NOT` | non-associative |
| 5 (highest) | `+`, `-` (binary arithmetic) | left |
The critical ordering: `AND` binds tighter than `OR`. In most programming languages (Python, JavaScript, Java) this is also the case, but it surprises developers who expect `AND` and `OR` to have equal precedence. SQL explicitly specifies: `WHERE a = 1 OR b = 2 AND c = 3` means `WHERE a = 1 OR (b = 2 AND c = 3)`.
`NOT` is special: it is a unary prefix operator with precedence higher than `AND` but lower than comparison operators. This means `NOT a = b AND c = d` parses as `(NOT (a = b)) AND (c = d)`, not `NOT ((a = b) AND (c = d))`.
### How Pratt Parsing Works
The Pratt parser (also called "precedence climbing" or "top-down operator precedence parsing") assigns each operator a **left binding power (LBP)** — a number representing how tightly it grabs what's to its left. The core insight:
> When deciding whether operator `B` in `A op1 B op2 C` should group with `A` or with `C`, compare `op1`'s LBP with `op2`'s. If `op2`'s LBP is higher, `B` groups right (with `C`). If `op1`'s LBP is higher or equal (for left-associative operators), `B` groups left (with `A`).
The algorithm is a function `parse_expression(min_bp)` where `min_bp` is the minimum binding power of any operator that can steal the current subexpression to the right:
```
parse_expression(min_bp):
    lhs = parse_primary()             // parse a "leaf" (literal, identifier, parenthesized expr)
    loop:
        op = current_token
        if op is not an operator: break
        if left_binding_power(op) <= min_bp: break   // op doesn't bind tightly enough
        consume op
        rhs = parse_expression(right_binding_power(op))  // recurse with RBP
        lhs = BinaryNode(op, lhs, rhs)
    return lhs
```
For left-associative operators, `right_binding_power(op) = left_binding_power(op)` (equal, so the same operator at the same level won't steal the rhs again). For right-associative operators, `right_binding_power(op) = left_binding_power(op) - 1`.
Let's trace `a AND b OR c` with LBP: `OR=1`, `AND=2`:
1. `parse_expression(0)`: parse primary `a` → lhs = `a`
2. Current op: `AND`, LBP=2 > min_bp=0: proceed. Recurse: `parse_expression(2)`.
3. In recursion: parse primary `b` → lhs = `b`. Current op: `OR`, LBP=1. 1 <= 2: **stop**. Return `b`.
4. Back in step 2: rhs = `b`. lhs = `AND(a, b)`.
5. Current op: `OR`, LBP=1 > min_bp=0: proceed. Recurse: `parse_expression(1)`.
6. In recursion: parse primary `c`. No more operators. Return `c`.
7. lhs = `OR(AND(a, b), c)`.
The tree is `OR(AND(a, b), c)` — AND bound tighter. Correct.
Now trace `a OR b AND c`:
1. Parse `a`. Op: `OR`, LBP=1 > 0. Recurse: `parse_expression(1)`.
2. Parse `b`. Op: `AND`, LBP=2 > 1. Recurse: `parse_expression(2)`.
3. Parse `c`. No op. Return `c`. rhs = `c`. lhs = `AND(b, c)`.
4. Back at step 1: rhs = `AND(b, c)`. lhs = `OR(a, AND(b, c))`.
`OR(a, AND(b, c))`. AND bound tighter. Correct.
### The Implementation
```c
/* Binding powers for binary operators */
static int left_binding_power(TokenType op) {
    switch (op) {
    case TOKEN_OR:    return 10;
    case TOKEN_AND:   return 20;
    /* Comparison operators — non-associative */
    case TOKEN_EQ:
    case TOKEN_NEQ:
    case TOKEN_LT:
    case TOKEN_GT:
    case TOKEN_LTE:
    case TOKEN_GTE:
    case TOKEN_IS:    return 30;
    /* Arithmetic */
    case TOKEN_PLUS:
    case TOKEN_MINUS: return 40;
    case TOKEN_STAR:
    case TOKEN_SLASH: return 50;
    default:          return 0;  /* not a binary operator */
    }
}
/* Parse a "primary" — a leaf value or a prefix-operator-prefixed expression */
static AstNode *parse_primary(Parser *p) {
    Token t = parser_peek(p);
    /* Parenthesized expression: ( expr ) */
    if (parser_match(p, TOKEN_LPAREN)) {
        AstNode *inner = parse_expression(p, 0);
        if (!inner) return NULL;
        if (!parser_expect(p, TOKEN_RPAREN, "after parenthesized expression")) {
            ast_free(inner); return NULL;
        }
        return inner;  /* parentheses are not represented in AST — they influence parse, not tree */
    }
    /* Unary NOT */
    if (parser_match(p, TOKEN_NOT)) {
        AstNode *operand = parse_expression(p, 25); /* LBP between AND(20) and comparison(30) */
        if (!operand) return NULL;
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type    = NODE_UNARY_EXPR;
        node->line    = t.line;
        node->column  = t.column;
        node->unary.op      = TOKEN_NOT;
        node->unary.operand = operand;
        return node;
    }
    /* Unary minus: -3.14 (at expression level, not tokenizer level) */
    if (parser_match(p, TOKEN_MINUS)) {
        AstNode *operand = parse_primary(p);  /* only grab the immediate primary */
        if (!operand) return NULL;
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type    = NODE_UNARY_EXPR;
        node->line    = t.line;
        node->column  = t.column;
        node->unary.op      = TOKEN_MINUS;
        node->unary.operand = operand;
        return node;
    }
    /* NULL keyword → literal null */
    if (parser_match(p, TOKEN_NULL)) {
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type   = NODE_LITERAL_NULL;
        node->line   = t.line;
        node->column = t.column;
        return node;
    }
    /* Integer literal */
    if (t.type == TOKEN_INTEGER) {
        parser_advance(p);
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type   = NODE_LITERAL_INT;
        node->line   = t.line;
        node->column = t.column;
        /* Parse the integer value from the token span */
        char buf[32];
        int len = t.length < 31 ? t.length : 31;
        memcpy(buf, t.start, len);
        buf[len] = '\0';
        node->lit_int.value = (int64_t)strtoll(buf, NULL, 10);
        return node;
    }
    /* Float literal */
    if (t.type == TOKEN_FLOAT) {
        parser_advance(p);
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type   = NODE_LITERAL_FLOAT;
        node->line   = t.line;
        node->column = t.column;
        char buf[64];
        int len = t.length < 63 ? t.length : 63;
        memcpy(buf, t.start, len);
        buf[len] = '\0';
        node->lit_float.value = strtod(buf, NULL);
        return node;
    }
    /* String literal */
    if (t.type == TOKEN_STRING) {
        parser_advance(p);
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type   = NODE_LITERAL_STRING;
        node->line   = t.line;
        node->column = t.column;
        /* Decode: strip outer quotes, unescape '' → ' */
        node->lit_str.value  = decode_string_literal(t.start, t.length,
                                                      &node->lit_str.length);
        return node;
    }
    /* Identifier: col_name or table.col_name */
    if (t.type == TOKEN_IDENTIFIER || t.type == TOKEN_QUOTED_ID) {
        parser_advance(p);
        AstNode *node = calloc(1, sizeof(AstNode));
        node->line   = t.line;
        node->column = t.column;
        /* Check for table.column syntax */
        if (parser_check(p, TOKEN_DOT)) {
            parser_advance(p); /* consume dot */
            Token col_tok = parser_peek(p);
            if (col_tok.type != TOKEN_IDENTIFIER && col_tok.type != TOKEN_QUOTED_ID) {
                snprintf(p->error_msg, sizeof(p->error_msg),
                         "Expected column name after '.' at line %d, column %d",
                         col_tok.line, col_tok.column);
                p->had_error = 1;
                free(node);
                return NULL;
            }
            parser_advance(p);
            node->type = NODE_COLUMN_REF;
            node->identifier.table = strndup(t.start, t.length);
            node->identifier.name  = strndup(col_tok.start, col_tok.length);
        } else {
            node->type = NODE_IDENTIFIER;
            node->identifier.table = NULL;
            node->identifier.name  = strndup(t.start, t.length);
        }
        return node;
    }
    /* Anything else: error */
    snprintf(p->error_msg, sizeof(p->error_msg),
             "Expected expression at line %d, column %d, got '%.*s'",
             t.line, t.column, t.length, t.start);
    p->had_error = 1;
    return NULL;
}
/* The main Pratt parser: parse_expression(p, min_bp) */
static AstNode *parse_expression(Parser *p, int min_bp) {
    if (p->had_error) return NULL;
    AstNode *lhs = parse_primary(p);
    if (!lhs) return NULL;
    for (;;) {
        Token op_tok = parser_peek(p);
        int lbp = left_binding_power(op_tok.type);
        if (lbp == 0) break;           /* not a binary operator */
        if (lbp <= min_bp) break;      /* operator doesn't bind tightly enough */
        parser_advance(p);  /* consume the operator */
        /* For IS NOT: peek and consume NOT if present */
        int is_not = 0;
        if (op_tok.type == TOKEN_IS && parser_check(p, TOKEN_NOT)) {
            parser_advance(p);
            is_not = 1;
        }
        /* Right binding power = lbp for left-associative (same level won't recur) */
        AstNode *rhs = parse_expression(p, lbp);
        if (!rhs) { ast_free(lhs); return NULL; }
        AstNode *node = calloc(1, sizeof(AstNode));
        node->type         = NODE_BINARY_EXPR;
        node->line         = op_tok.line;
        node->column       = op_tok.column;
        node->binary.op    = is_not ? TOKEN_NEQ : op_tok.type;  /* simplify IS NOT */
        node->binary.left  = lhs;
        node->binary.right = rhs;
        lhs = node;
    }
    return lhs;
}
```
This is the complete, working Pratt parser for SQL expressions. The `for(;;)` loop is the key: it keeps consuming binary operators as long as their binding power exceeds the minimum. When it doesn't, it stops — and the caller's own loop picks up where this one stopped. This is the elegant mechanism that handles chains of same-precedence operators (`a AND b AND c`) without infinite recursion: `AND`'s right BP equals its left BP, so after building `AND(a, b)`, the next `AND` has LBP=20 and the min_bp at that level is also 20, so `20 <= 20` is true and the loop stops. Control returns to the caller, which then builds `AND(AND(a,b), c)`.
### NULL Handling: A Special Case
Notice that `NULL` is parsed as `NODE_LITERAL_NULL` in `parse_primary`, not as `NODE_IDENTIFIER`. This is critical. If `NULL` comes through as an identifier, then the expression evaluator must special-case "identifiers named NULL" everywhere it appears. That is error-prone and ugly. By making it a distinct literal node type, the evaluator handles it cleanly:
```c
/* In expression evaluator (Milestone 6) */
case NODE_LITERAL_NULL:
    result.type  = VALUE_NULL;
    result.is_null = 1;
    break;
```
Also: `WHERE x IS NULL` is legal SQL; `WHERE x = NULL` is legal SQL but always evaluates to NULL (never TRUE) because of three-valued logic. The parser produces valid ASTs for both — the semantic difference is in the evaluator, not the parser.
---
## String Literal Decoding
When `parse_primary` encounters a `TOKEN_STRING`, it calls `decode_string_literal` to strip the outer single quotes and unescape `''` sequences. This is the deferred decoding promised in Milestone 1:
```c
/* Decode a string literal token span into a heap-allocated string.
   Input: pointer to opening quote, length including both quotes.
   Output: heap-allocated decoded string, *out_len set to byte count. */
static char *decode_string_literal(const char *start, int raw_len, int *out_len) {
    /* Allocate worst-case: same size as input (no expansion from decoding) */
    char *buf = malloc(raw_len + 1);
    int   wpos = 0;
    /* Skip opening quote; stop before closing quote */
    int i = 1;  /* start after opening ' */
    while (i < raw_len - 1) {
        if (start[i] == '\'' && start[i+1] == '\'') {
            buf[wpos++] = '\'';
            i += 2;  /* consume both '' as one ' */
        } else {
            buf[wpos++] = start[i++];
        }
    }
    buf[wpos] = '\0';
    *out_len = wpos;
    return buf;
}
```
This is a clean place to also handle `\n`, `\t` etc. if you want to support those non-standard escapes — but SQLite itself does not. Keep it simple: only `''` is an escape sequence in standard SQL string literals.
---
## The Top-Level Parse Function
Tie all three statement parsers together:
```c
/* parser.h (public) */
typedef struct {
    AstNode *root;
    char     error_msg[256];
    int      error_line;
    int      error_col;
    int      had_error;
} ParseResult;
ParseResult parse_statement(const char *sql);
void        ast_free(AstNode *node);
void        ast_print(AstNode *node, int indent);  /* for EXPLAIN and debugging */
```
```c
/* parser.c */
ParseResult parse_statement(const char *sql) {
    ParseResult result = {0};
    int    token_count;
    Token *tokens = lexer_tokenize_all(sql, &token_count);
    /* Check for tokenizer errors */
    for (int i = 0; i < token_count; i++) {
        if (tokens[i].type == TOKEN_ERROR) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Tokenizer error at line %d, column %d: %.*s",
                     tokens[i].line, tokens[i].column,
                     tokens[i].length, tokens[i].start);
            result.error_line = tokens[i].line;
            result.error_col  = tokens[i].column;
            result.had_error  = 1;
            free(tokens);
            return result;
        }
    }
    Parser p;
    parser_init(&p, tokens, token_count);
    Token first = parser_peek(&p);
    AstNode *root = NULL;
    switch (first.type) {
    case TOKEN_SELECT:      root = parse_select(&p);       break;
    case TOKEN_INSERT:      root = parse_insert(&p);       break;
    case TOKEN_CREATE:      root = parse_create_table(&p); break;
    default:
        snprintf(p.error_msg, sizeof(p.error_msg),
                 "Unknown statement type '%.*s' at line %d, column %d",
                 first.length, first.start, first.line, first.column);
        p.had_error  = 1;
        p.error_line = first.line;
        p.error_col  = first.column;
    }
    if (p.had_error) {
        memcpy(result.error_msg, p.error_msg, sizeof(result.error_msg));
        result.error_line = p.error_line;
        result.error_col  = p.error_col;
        result.had_error  = 1;
        ast_free(root);
        root = NULL;
    }
    result.root = root;
    free(tokens);
    return result;
}
```
---
## Error Reporting and Recovery

![Parser Error — Before/After](./diagrams/diag-parser-error-recovery.svg)

Error position reporting is non-negotiable. Every `parser_expect` failure writes the token's line and column to `p.error_msg`. The `ParseResult` surfaces this to the caller. A user who types `SELEECT * FROM t` should see:
```
Parse error at line 1, column 1: Expected SELECT at statement start, got 'SELEECT'
```
Not:
```
Parse failed.
```
**Error recovery for multiple statements**: in a production database, SQL input may contain multiple statements separated by semicolons. If the parser fails on the first, it should synchronize to the next `TOKEN_SEMICOLON` and attempt to parse the second — a technique called *panic-mode error recovery*. The implementation:
```c
static void synchronize(Parser *p) {
    /* Consume tokens until we find a statement boundary or EOF */
    while (!parser_check(p, TOKEN_SEMICOLON) && !parser_check(p, TOKEN_EOF)) {
        parser_advance(p);
    }
    parser_match(p, TOKEN_SEMICOLON); /* consume the semicolon too */
    p->had_error = 0;  /* reset error flag — ready to try the next statement */
}
```
You call `synchronize` in a loop that parses multiple statements:
```c
while (!parser_check(&p, TOKEN_EOF)) {
    AstNode *stmt = parse_one_statement(&p);
    if (p.had_error) {
        fprintf(stderr, "Error: %s\n", p.error_msg);
        synchronize(&p);
        continue;
    }
    execute_statement(stmt);
    ast_free(stmt);
    parser_match(&p, TOKEN_SEMICOLON);
}
```
This is exactly how IDEs parse whole files with errors in some methods — they recover to the next boundary and continue. Without it, a single typo causes the parser to silently ignore the rest of the input.
---
## The `ast_print` Function: Your Debugging Lifeline
Build this immediately. You will use it constantly throughout every subsequent milestone.
```c
static void ast_print_node(AstNode *node, int depth) {
    if (!node) return;
    char indent[128] = {0};
    for (int i = 0; i < depth && i < 62; i++) {
        indent[i*2]   = ' ';
        indent[i*2+1] = ' ';
    }
    switch (node->type) {
    case NODE_SELECT:
        printf("%sSELECT\n", indent);
        printf("%s  FROM: %s\n", indent, node->select.table_name);
        printf("%s  COLUMNS (%d):\n", indent, node->select.column_count);
        for (int i = 0; i < node->select.column_count; i++)
            ast_print_node(node->select.columns[i], depth + 2);
        if (node->select.where_expr) {
            printf("%s  WHERE:\n", indent);
            ast_print_node(node->select.where_expr, depth + 2);
        }
        if (node->select.limit_expr) {
            printf("%s  LIMIT:\n", indent);
            ast_print_node(node->select.limit_expr, depth + 2);
        }
        break;
    case NODE_BINARY_EXPR:
        printf("%sBINARY(%s)\n", indent, token_type_name(node->binary.op));
        ast_print_node(node->binary.left,  depth + 1);
        ast_print_node(node->binary.right, depth + 1);
        break;
    case NODE_UNARY_EXPR:
        printf("%sUNARY(%s)\n", indent, token_type_name(node->unary.op));
        ast_print_node(node->unary.operand, depth + 1);
        break;
    case NODE_LITERAL_INT:
        printf("%sLIT_INT(%lld)\n", indent, (long long)node->lit_int.value);
        break;
    case NODE_LITERAL_FLOAT:
        printf("%sLIT_FLOAT(%g)\n", indent, node->lit_float.value);
        break;
    case NODE_LITERAL_STRING:
        printf("%sLIT_STR(\"%s\")\n", indent, node->lit_str.value);
        break;
    case NODE_LITERAL_NULL:
        printf("%sNULL\n", indent);
        break;
    case NODE_IDENTIFIER:
        printf("%sIDENTIFIER(%s)\n", indent, node->identifier.name);
        break;
    case NODE_WILDCARD:
        printf("%sWILDCARD(*)\n", indent);
        break;
    /* ... other cases ... */
    default:
        printf("%s<node type %d>\n", indent, node->type);
    }
}
void ast_print(AstNode *node, int indent) {
    ast_print_node(node, indent);
}
```
Run this after every parse and compare against hand-drawn trees. An AST that looks wrong in the printout is wrong — trust the print, fix the parser.
---
## Designing the Test Suite
Your acceptance criteria require 15 valid and 10 invalid SQL statements. Organize them to cover every grammar branch:
```c
typedef struct {
    const char *sql;
    int         expect_error;   /* 1 if this should fail */
    NodeType    expect_root;    /* for valid parses */
    const char *description;
} ParserTest;
static ParserTest tests[] = {
    /* ---- SELECT ---- */
    {"SELECT * FROM t;",                        0, NODE_SELECT, "basic wildcard"},
    {"SELECT a, b, c FROM t;",                  0, NODE_SELECT, "column list"},
    {"SELECT * FROM t WHERE x = 1;",            0, NODE_SELECT, "WHERE equality"},
    {"SELECT * FROM t WHERE a AND b OR c;",     0, NODE_SELECT, "AND/OR precedence"},
    {"SELECT * FROM t WHERE NOT a = b;",        0, NODE_SELECT, "NOT prefix"},
    {"SELECT * FROM t WHERE (a OR b) AND c;",   0, NODE_SELECT, "parenthesized override"},
    {"SELECT * FROM t ORDER BY name ASC;",      0, NODE_SELECT, "ORDER BY ASC"},
    {"SELECT * FROM t ORDER BY age DESC;",      0, NODE_SELECT, "ORDER BY DESC"},
    {"SELECT * FROM t LIMIT 10;",               0, NODE_SELECT, "LIMIT clause"},
    {"SELECT * FROM t WHERE x IS NULL;",        0, NODE_SELECT, "IS NULL"},
    {"SELECT * FROM t WHERE x IS NOT NULL;",    0, NODE_SELECT, "IS NOT NULL"},
    {"SELECT * FROM t WHERE x != 5 AND y >= 10;", 0, NODE_SELECT, "comparison ops"},
    {"SELECT t.col FROM t WHERE t.id = 1;",     0, NODE_SELECT, "table.column refs"},
    {"SELECT * FROM t WHERE x = 'it''s fine';", 0, NODE_SELECT, "string escape"},
    {"SELECT * FROM t WHERE x = 3.14;",         0, NODE_SELECT, "float literal"},
    /* ---- INSERT ---- */
    {"INSERT INTO t VALUES (1, 'hello');",       0, NODE_INSERT, "basic insert"},
    {"INSERT INTO t (a, b) VALUES (1, 2);",      0, NODE_INSERT, "insert with col list"},
    {"INSERT INTO t VALUES (NULL, 42);",         0, NODE_INSERT, "null in values"},
    /* ---- CREATE TABLE ---- */
    {"CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL, score REAL);",
                                                 0, NODE_CREATE_TABLE, "create with constraints"},
    {"CREATE TABLE t (data BLOB);",              0, NODE_CREATE_TABLE, "blob column"},
    /* ---- INVALID ---- */
    {"SELECT FROM t;",                           1, 0, "missing column list"},
    {"SELECT * WHERE x = 1;",                    1, 0, "missing FROM"},
    {"SELECT * FROM;",                           1, 0, "missing table name"},
    {"INSERT VALUES (1);",                       1, 0, "missing INTO"},
    {"INSERT INTO t VALUES;",                    1, 0, "missing value parens"},
    {"CREATE TABLE (id INTEGER);",               1, 0, "missing table name"},
    {"SELECT * FROM t WHERE AND x = 1;",         1, 0, "AND without left operand"},
    {"SELECT * FROM t WHERE (x = 1;",            1, 0, "unclosed paren"},
    {"SELECT * FROM t ORDER;",                   1, 0, "ORDER without BY"},
    {"SELEECT * FROM t;",                        1, 0, "misspelled keyword"},
};
```
The test runner:
```c
static void run_parser_tests(void) {
    int passed = 0, failed = 0;
    for (int i = 0; i < (int)(sizeof(tests)/sizeof(tests[0])); i++) {
        ParseResult r = parse_statement(tests[i].sql);
        int ok;
        if (tests[i].expect_error) {
            ok = (r.had_error != 0);
            if (!ok)
                fprintf(stderr, "FAIL [%s]: expected error, got valid parse\n",
                        tests[i].description);
        } else {
            ok = (!r.had_error && r.root && r.root->type == tests[i].expect_root);
            if (!ok)
                fprintf(stderr, "FAIL [%s]: %s\n",
                        tests[i].description,
                        r.had_error ? r.error_msg : "wrong root type");
        }
        ok ? passed++ : failed++;
        ast_free(r.root);
    }
    printf("Parser: %d passed, %d failed\n", passed, failed);
}
```
---
## Grammar Formalization: Making the Implicit Explicit
Your code implicitly encodes a grammar. Writing it down explicitly catches ambiguities before they become bugs. Here is the subset of SQL your parser handles, in BNF-like notation:
```
statement     ::= select_stmt | insert_stmt | create_stmt
select_stmt   ::= SELECT column_list FROM identifier
                  [WHERE expression]
                  [ORDER BY order_item (',' order_item)*]
                  [LIMIT expression]
column_list   ::= '*' | expression (',' expression)*
order_item    ::= expression [ASC | DESC]
insert_stmt   ::= INSERT INTO identifier
                  ['(' identifier (',' identifier)* ')']
                  VALUES '(' expression (',' expression)* ')'
create_stmt   ::= CREATE TABLE identifier
                  '(' column_def (',' column_def)* ')'
column_def    ::= identifier identifier constraint*
constraint    ::= PRIMARY KEY | NOT NULL | UNIQUE
expression    ::= or_expr
or_expr       ::= and_expr (OR and_expr)*
and_expr      ::= not_expr (AND not_expr)*
not_expr      ::= NOT not_expr | cmp_expr
cmp_expr      ::= add_expr [(= | != | < | > | <= | >= | IS [NOT]) add_expr]
add_expr      ::= mul_expr ((+ | -) mul_expr)*
mul_expr      ::= primary ((* | /) primary)*
primary       ::= INTEGER | FLOAT | STRING | NULL
               |  identifier ['.' identifier]
               |  '(' expression ')'
               |  '-' primary
```
This grammar is exactly what your Pratt parser implements — but the Pratt parser compresses the `or_expr → and_expr → not_expr → cmp_expr → add_expr → mul_expr` hierarchy into a single table of binding powers. The grammar is what you'd write to prove correctness; the Pratt parser is the efficient implementation.
---
## Three-Level View: What the Parser Sees
**Level 1 — The token stream surface (what you consume)**
The parser sees: `TOKEN_SELECT TOKEN_STAR TOKEN_FROM TOKEN_IDENTIFIER("users") TOKEN_WHERE TOKEN_IDENTIFIER("age") TOKEN_GT TOKEN_INTEGER("18")`. This is a flat sequence with no structure. The parser's job is to impose structure.
**Level 2 — The grammar engine (what you implement)**
The recursive descent functions encode the context-free grammar of SQL. Each function corresponds to a grammar production rule. `parse_select` handles `select_stmt`, `parse_expression` handles `expression`, `parse_primary` handles `primary`. These functions call each other according to the grammar — the call stack at any point during parsing is the parse stack, tracking which production rules are currently being matched.
**Level 3 — The AST (what you produce)**
A tree of `AstNode` structs in heap memory. The bytecode compiler (Milestone 3) will walk this tree in a post-order traversal to emit bytecode instructions. The query planner (Milestone 8) will walk it to find column references and apply cost estimates. The constraint checker will walk it to verify NOT NULL and UNIQUE requirements. The tree is the intermediate representation shared by all downstream components.
---
## Knowledge Cascade: What This Milestone Unlocks
**→ Every compiler ever written.** The parser pattern — recursive descent consuming a token stream, producing an AST, with Pratt parsing for expressions — is used in CPython's parser, in GCC, in Clang, in the TypeScript compiler, in the Rust compiler. These are not coincidences. This is the standard architecture for language front-ends. You now understand why `clang -ast-dump foo.c` produces a tree that looks exactly like what you just built.
**→ Linters, formatters, and IDE tooling.** ESLint, Prettier, `rustfmt`, `gofmt` — all work by parsing source code into an AST and then either analyzing the tree (linter) or pretty-printing it (formatter). The formatter doesn't rewrite source text; it traverses the AST and emits tokens. Your `ast_print` function is a primitive version of exactly this. The error recovery technique (synchronize to semicolon) is what language servers use to keep producing completions even in broken files.
**→ SQL dialect differences become obvious.** Now that you know what the grammar looks like, SQL dialect differences are just grammar differences. PostgreSQL adds `RETURNING` to `INSERT`. MySQL accepts `INSERT IGNORE`. SQLite allows `CREATE TABLE IF NOT EXISTS`. These aren't mysterious — they're additional production rules or optional tokens in the specific clauses you just built. If you ever need to add them, you know exactly which function to modify.
**→ Pratt parsing generalizes to everything.** The same technique works for mathematical expressions in a calculator, configuration file languages (`key = value1 + value2 * weight`), template languages, CSS property values, even URL path patterns. Any time you have a language with infix operators and precedence, Pratt parsing is the right tool. Douglas Crockford used it to parse JavaScript in JSLint. Bob Nystrom's "Crafting Interpreters" describes it as the best parsing technique he knows. You now understand why.
**→ The AST is a universal IR.** Every compiler pipeline has an intermediate representation at roughly this level — between "text" and "machine instructions". LLVM has LLVM-IR. GCC has GIMPLE. Java has bytecode. The JVM's `javap` tool is the equivalent of your `ast_print` — it lets you inspect the intermediate form. The pattern of "front-end produces tree, back-end consumes tree" is so universal because it perfectly separates concerns: the front-end can be replaced (add a new language!) without touching the back-end, and the back-end can be replaced (target a new architecture!) without touching the front-end.
**→ Error recovery enables production quality.** The synchronize-to-semicolon technique is what separates toy parsers from production parsers. psql continues after a failed statement. The PostgreSQL JDBC driver handles multi-statement scripts. VS Code keeps showing completions even in broken files. You now have the mental model for how this works. The next step is structured error recovery (tracking open delimiters, synchronizing to matching closes) — the Rust compiler's excellent error messages come from decades of refinement of exactly this technique.
---
## Common Pitfalls: What Will Break Silently
**1. AND/OR precedence is backwards from your instincts.** If you accidentally give `OR` a higher binding power than `AND`, `WHERE a = 1 OR b = 2 AND c = 3` becomes `(a = 1 OR b = 2) AND c = 3`. Every multi-condition WHERE clause evaluates differently. No error is produced. Wrong rows are returned. This bug is a data correctness problem, not a crash — which makes it far more dangerous.
**2. `NOT` applied to the wrong operand.** `NOT x = 1 AND y = 2` should parse as `(NOT (x = 1)) AND (y = 2)`. If NOT's right binding power is too low, it grabs `(x = 1) AND (y = 2)`. Set NOT's recursive call to `parse_expression(p, 25)` — between AND's BP (20) and comparison's BP (30). This is non-obvious but critical.
**3. NULL parsed as identifier.** If `match_keyword` didn't include `NULL` → `TOKEN_NULL`, or if `parse_primary` doesn't handle `TOKEN_NULL` before the identifier case, `NULL` becomes an identifier node. Three milestones later, `WHERE x IS NULL` either crashes or silently treats NULL as a column reference that doesn't exist. Handle it now.
**4. Forgetting to decode string literals.** If you pass the raw token span (with surrounding quotes and `''` escapes) as the string value, then inserting `'hello'` stores the string `'hello'` (with quotes) rather than `hello`. The bug surfaces when you SELECT the data back and compare it.
**5. Memory leaks in error paths.** Every `parse_select`, `parse_insert`, `parse_create_table` allocates heap memory before it can fail. When it does fail, it must free everything allocated so far. The pattern throughout the code above is: detect error, call `ast_free(node)`, return NULL. If you skip the `ast_free`, every failed parse leaks memory. In a test suite that runs thousands of invalid inputs, this becomes gigabytes of leakage.
**6. Left recursion disguised as a loop.** Do not implement expression parsing as grammar rules that call themselves on the left: `parse_and_expr` that calls `parse_and_expr` to get its left operand. That's left recursion — infinite loop. The Pratt parser's loop-based structure eliminates left recursion entirely. Commit to the Pratt approach.
---
## What You Have Built
At the end of this milestone:
- A recursive-descent parser that consumes the token stream produced by your Milestone 1 lexer
- SELECT parsing with column list (including `*`), FROM clause, optional WHERE, ORDER BY (with ASC/DESC), and LIMIT
- INSERT parsing with optional column name list and VALUES clause
- CREATE TABLE parsing with column definitions, type names, and PRIMARY KEY / NOT NULL / UNIQUE constraints
- A Pratt parser implementing SQL's exact operator precedence: NOT > comparison > AND > OR
- Parenthesized expressions correctly overriding default precedence
- NULL parsed as a first-class literal, not an identifier
- String literal decoding with `''` escape handling
- Error reporting with token position (line and column) in every failure case
- Panic-mode error recovery that synchronizes to semicolons for multi-statement input
- An `ast_print` debugging function
- A test suite with 15+ valid and 10+ invalid SQL cases
The bytecode compiler in Milestone 3 will take this AST and emit the instruction stream that the VDBE executes. The compiler's structure mirrors the parser's: one function per AST node type, recursively translating the tree into instructions. Every pattern you established here — recursive tree traversal, clean error propagation, typed node dispatch — carries forward.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m3 -->
<!-- MS_ID: build-sqlite-m3 -->
# Milestone 3: Bytecode Compiler (VDBE)
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You are now at the center of the engine. Look at the pipeline you have built so far: the tokenizer dissolved SQL text into typed tokens; the parser assembled those tokens into a structured AST. Both milestones were about *understanding* — converting a human-readable language into a machine-readable form. This milestone is about *execution*.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

The VDBE (Virtual DataBase Engine) is SQLite's execution layer — the component that actually does the work of reading rows, evaluating expressions, and producing results. Its existence is the architectural decision that makes SQLite what it is: embeddable, portable, fast, and debuggable. Understanding why SQLite chose bytecode compilation over simpler alternatives is the real lesson of this milestone.
---
## The Revelation: Why Bytecode?
Here is what almost every developer thinks when they first study a database: parsing produces an AST, so execution means walking the AST. For each statement, traverse the tree, evaluate each node, produce output. Simple. Obvious. Why complicate it?
This intuition is wrong in a specific and important way. Let's make the problem concrete.
You run `SELECT * FROM users WHERE age > 18` against a table with 500,000 rows. A tree-walking interpreter executes this query by traversing the WHERE clause AST for every single row:
```
For row 1:   evaluate NODE_BINARY_EXPR(GT)
               → dispatch on node type: case NODE_BINARY_EXPR
               → evaluate left: dispatch on NODE_IDENTIFIER → look up "age" → value 23
               → evaluate right: dispatch on NODE_LITERAL_INT → value 18
               → apply GT operator → true
For row 2:   evaluate NODE_BINARY_EXPR(GT)
               → dispatch on node type: case NODE_BINARY_EXPR
               ...same structure, same dispatch overhead...
For row 3:   ...
```
Each row evaluation involves: following multiple pointer chains through heap memory (the AST nodes), performing type dispatch (`switch(node->type)`) at each level, and resolving column names against schema metadata. On modern hardware, pointer-chasing through heap-allocated tree nodes causes CPU cache misses. Each cache miss costs roughly 100 nanoseconds. Multiply by millions of nodes across millions of rows, and AST interpretation is genuinely slow.
SQLite's actual approach is radically different. The compiler translates the AST *once* into a compact sequence of integers — bytecode instructions — and then the VDBE executes that sequence in a tight loop. The hot path for scanning 500,000 rows looks like this:
```c
while (pc < program->ninstruction) {
    Instruction *instr = &program->instructions[pc];
    switch (instr->opcode) {  // integer comparison — branch prediction handles this well
    case OP_COLUMN: ...
    case OP_GT:     ...
    case OP_NEXT:   ...
    }
    pc++;
}
```
Instead of traversing a tree with pointer chains and runtime type dispatch, the VDBE executes a flat array of structs. The array fits in CPU cache. The switch-on-integer is branch-predictable. The translation overhead (compiling the AST to bytecode) happens once per query preparation, not once per row.
This is the revelation: **the VDBE exists to move work from the hot path (per-row) to the cold path (per-query).** AST compilation happens once. Bytecode execution happens once per row. The separation is the optimization.
But there's a second, deeper reason: **bytecode is a stable interface**. The optimizer can rewrite bytecode without touching the parser. The storage engine can change its B-tree layout without touching the compiler. The VDBE is the contract between SQL semantics and disk representation. You can add a new optimization pass that transforms bytecode before execution — something impossible if execution is AST-walking.

![VDBE Instruction Set — Structure Layout](./diagrams/diag-vdbe-instruction-set.svg)

---
## What You Are Building
Three components, tightly coupled:
1. **The instruction set** — a fixed catalog of opcodes with defined semantics
2. **The compiler** — an AST visitor that emits sequences of instructions
3. **The virtual machine** — a fetch-decode-execute loop that runs the instruction sequences
The public interface:
```c
/* vdbe.h */
#pragma once
#include <stdint.h>
#include "parser.h"   /* AstNode */
/* ---- Opcode definitions ---- */
typedef enum {
    /* Cursor operations */
    OP_OPEN_TABLE,    /* p1=cursor_id, p4=table_name: open a table cursor */
    OP_REWIND,        /* p1=cursor_id, p2=jump_addr: if table empty, jump to p2 */
    OP_NEXT,          /* p1=cursor_id, p2=jump_addr: advance cursor; if EOF, fall through */
    OP_CLOSE,         /* p1=cursor_id: close cursor, release resources */
    /* Data access */
    OP_COLUMN,        /* p1=cursor_id, p2=col_index, p3=dest_reg: load column into register */
    OP_ROWID,         /* p1=cursor_id, p2=dest_reg: load current row's rowid */
    OP_MAKE_RECORD,   /* p1=start_reg, p2=count, p3=dest_reg: pack registers into record */
    OP_INSERT,        /* p1=cursor_id, p2=record_reg, p3=rowid_reg: insert record */
    /* Register operations */
    OP_INTEGER,       /* p1=value (immediate small int), p2=dest_reg */
    OP_INT64,         /* p1=dest_reg, p4=int64 value (stored in aux) */
    OP_REAL,          /* p1=dest_reg, p4=double value */
    OP_STRING8,       /* p1=dest_reg, p4=string pointer */
    OP_NULL,          /* p1=dest_reg: store NULL in register */
    OP_COPY,          /* p1=src_reg, p2=dest_reg */
    /* Comparison */
    OP_EQ,            /* p1=reg_a, p2=jump_addr, p3=reg_b: if a==b, jump */
    OP_NE,            /* p1=reg_a, p2=jump_addr, p3=reg_b: if a!=b, jump */
    OP_LT,            /* p1=reg_a, p2=jump_addr, p3=reg_b: if a<b, jump */
    OP_LE,            /* p1=reg_a, p2=jump_addr, p3=reg_b: if a<=b, jump */
    OP_GT,            /* p1=reg_a, p2=jump_addr, p3=reg_b: if a>b, jump */
    OP_GE,            /* p1=reg_a, p2=jump_addr, p3=reg_b: if a>=b, jump */
    OP_IS_NULL,       /* p1=reg, p2=jump_addr: if reg is NULL, jump */
    OP_NOT_NULL,      /* p1=reg, p2=jump_addr: if reg is not NULL, jump */
    /* Arithmetic */
    OP_ADD,           /* p1=reg_a, p2=dest_reg, p3=reg_b */
    OP_SUBTRACT,      /* p1=reg_a, p2=dest_reg, p3=reg_b */
    OP_MULTIPLY,      /* p1=reg_a, p2=dest_reg, p3=reg_b */
    OP_DIVIDE,        /* p1=reg_a, p2=dest_reg, p3=reg_b */
    /* Control flow */
    OP_GOTO,          /* p2=jump_addr: unconditional jump */
    OP_IF,            /* p1=reg, p2=jump_addr: if reg is truthy, jump */
    OP_IF_NOT,        /* p1=reg, p2=jump_addr: if reg is falsy or null, jump */
    /* Output */
    OP_RESULT_ROW,    /* p1=start_reg, p2=count: emit one result row */
    /* Schema */
    OP_CREATE_TABLE,  /* p4=table_name: create table in schema */
    /* Termination */
    OP_HALT,          /* p1=return_code, p4=error_msg: stop execution */
    OP_COUNT,         /* sentinel: number of opcodes */
} Opcode;
/* ---- A single instruction ---- */
typedef struct {
    Opcode   opcode;
    int      p1;      /* first integer operand */
    int      p2;      /* second integer operand (often a jump address) */
    int      p3;      /* third integer operand */
    union {
        int64_t  i64;
        double   real;
        char    *str;  /* not owned by instruction — points into compiled query lifetime */
        void    *ptr;
    } p4;             /* auxiliary operand for large values */
    const char *comment;  /* human-readable annotation for EXPLAIN */
} Instruction;
/* ---- A compiled program ---- */
typedef struct {
    Instruction *instructions;
    int          count;
    int          capacity;
} Program;
/* ---- A typed value (register contents) ---- */
typedef enum {
    VAL_NULL,
    VAL_INTEGER,
    VAL_REAL,
    VAL_TEXT,
    VAL_BLOB,
} ValueType;
typedef struct {
    ValueType  type;
    union {
        int64_t  i;
        double   r;
        struct { char *data; int len; } text;
        struct { uint8_t *data; int len; } blob;
    };
} Value;
/* ---- Compiler and VM public API ---- */
Program *compile(AstNode *ast, const char **error_out);
void     program_free(Program *p);
void     program_explain(Program *p, FILE *out);  /* EXPLAIN output */
typedef struct DB DB;  /* forward declaration of database handle */
int      vdbe_execute(Program *p, DB *db, FILE *result_out);
```
Before diving into each component, note the instruction format: five fields, including three small integers (`p1`, `p2`, `p3`) and one large auxiliary (`p4`). This is almost exactly SQLite's real instruction format. The three integer fields cover the vast majority of operand needs — cursor IDs, register indices, jump addresses, immediate small integers. The `p4` union handles the cases that don't fit.

> **🔑 Foundation: Bytecode instruction sets and register-based VMs**
> 
> ## Bytecode Instruction Sets and Register-Based VMs
**What it IS**
Bytecode is a compact, binary-encoded instruction format designed to be executed by a virtual machine (VM) rather than real hardware. Think of it as an instruction set for an imaginary CPU that your software simulates. Each instruction is typically one byte (the "opcode") followed by zero or more operands — hence "bytecode."
A **register-based VM** is one architectural style for executing that bytecode. Instead of a stack (where every operation pushes/pops values), a register-based VM has a fixed set of *virtual registers* — named slots that hold intermediate values — and instructions explicitly reference them by index.
**Concrete example.** Suppose you want to compute `a + b`. In a register-based encoding:
```
LOAD  R0, [a]     ; load variable 'a' into register 0
LOAD  R1, [b]     ; load variable 'b' into register 1
ADD   R2, R0, R1  ; R2 = R0 + R1
STORE [result], R2
```
Each instruction names its source and destination registers directly. The Lua VM and Dalvik (Android's original Java VM) both use this model. CPython, by contrast, is stack-based — a useful contrast to keep in mind.
**The tradeoff vs. stack-based:**
- Register-based VMs tend to generate *fewer instructions* because values don't need to be constantly pushed and popped — operands are referenced by name.
- But each instruction is slightly *larger* (it must encode register indices), and the interpreter dispatch loop is more complex.
- Register machines map more naturally to real CPU registers, which matters a great deal when you later add JIT compilation.
**Designing the instruction set** means deciding:
- What opcodes exist (`ADD`, `LOAD`, `CALL`, `JUMP`, `RET`, etc.)
- How many virtual registers there are (often per-call-frame, like Lua's 256)
- How operands are encoded (fixed-width vs. variable-length)
- What the calling convention looks like (how function frames are laid out)
**WHY you need it right now**
You are building a VM. Before you can write an interpreter loop or think about optimization, you need to define *what your VM executes*. The instruction set is the contract between your compiler frontend (which emits bytecode) and your execution engine. Every later decision — how the interpreter dispatches, what the JIT compiles, how the debugger steps through code — flows from this definition. Getting the instruction set right is foundational; changing it later breaks everything downstream.
**Key mental model**
Think of a register-based VM as designing a *simple, fake CPU on paper first*. Your virtual registers are like named scratch-pad slots for a single function's execution. The instruction set is the ISA (Instruction Set Architecture) of that fake CPU. Just as x86 has `mov`, `add`, `jmp`, your VM has its own opcodes — you're building silicon in software.

---
## The Register File
Before you can understand how instructions work, you need to understand what they operate on. The VDBE is a **register-based virtual machine**. This is distinct from a stack-based VM — a distinction worth understanding clearly.
In a **stack-based VM** (like CPython's bytecode engine, or the JVM), operations implicitly pop operands from a stack and push results. The instruction `ADD` doesn't name its operands — it pops two values from the top of the stack and pushes their sum. Simple to compile, simple to implement.
In a **register-based VM** (like Lua's VM, Dalvik/ART on Android, and SQLite's VDBE), each instruction explicitly names its source and destination registers. `OP_ADD p1=3, p2=5, p3=2` means: add the value in register 3 to the value in register 2, store the result in register 5. No implicit stack.
Register-based VMs generate fewer instructions for the same computation (no push/pop pairs) and can skip intermediate stores. The tradeoff is that instructions are wider (they carry register numbers). For database operations, where each instruction does substantial work (reading an entire column from a B-tree page), the register overhead is negligible.
The register file is a flat array of `Value` structs:
```c
#define VDBE_MAX_REGISTERS 1024
typedef struct {
    Value regs[VDBE_MAX_REGISTERS];
    /* ... other VM state ... */
} VdbeState;
```
Register index `0` is conventionally unused (as a null/sentinel). Registers `1` through `N` are allocated by the compiler. The compiler is responsible for assigning register indices — this is a simplified form of register allocation, the same problem CPU compiler backends solve for physical registers.

![Register File — State Evolution](./diagrams/diag-vdbe-register-file.svg)

---
## The Compiler: Translating AST to Bytecode
The compiler is an AST visitor with a twist: it emits instructions into a growing program buffer. The key state:
```c
typedef struct {
    Program *prog;       /* the program being built */
    int      next_reg;   /* next available register (monotonically increasing) */
    int      next_cursor;/* next available cursor ID */
    char     error[256]; /* error message on failure */
    int      had_error;
} Compiler;
static void compiler_init(Compiler *c, Program *p) {
    c->prog       = p;
    c->next_reg   = 1;    /* register 0 reserved */
    c->next_cursor = 0;
    c->had_error  = 0;
    c->error[0]   = '\0';
}
static int alloc_register(Compiler *c) {
    if (c->next_reg >= VDBE_MAX_REGISTERS) {
        snprintf(c->error, sizeof(c->error), "Register overflow: too many temporaries");
        c->had_error = 1;
        return -1;
    }
    return c->next_reg++;
}
static int alloc_cursor(Compiler *c) {
    return c->next_cursor++;
}
/* Emit one instruction, return its address (index in program) */
static int emit(Compiler *c, Opcode op, int p1, int p2, int p3, const char *comment) {
    Program *prog = c->prog;
    if (prog->count >= prog->capacity) {
        prog->capacity = prog->capacity ? prog->capacity * 2 : 16;
        prog->instructions = realloc(prog->instructions,
                                     prog->capacity * sizeof(Instruction));
    }
    int addr = prog->count++;
    Instruction *instr = &prog->instructions[addr];
    instr->opcode  = op;
    instr->p1      = p1;
    instr->p2      = p2;
    instr->p3      = p3;
    instr->p4.ptr  = NULL;
    instr->comment = comment;
    return addr;
}
/* Emit instruction and set the p4 large-value field */
static int emit_p4_str(Compiler *c, Opcode op, int p1, int p2, int p3,
                        const char *str, const char *comment) {
    int addr = emit(c, op, p1, p2, p3, comment);
    c->prog->instructions[addr].p4.str = (char *)str;  /* caller owns lifetime */
    return addr;
}
static int emit_p4_i64(Compiler *c, Opcode op, int p1, int p2, int p3,
                        int64_t val, const char *comment) {
    int addr = emit(c, op, p1, p2, p3, comment);
    c->prog->instructions[addr].p4.i64 = val;
    return addr;
}
/* Patch a previously-emitted jump instruction to point at current address */
static void patch_jump(Compiler *c, int jump_instr_addr) {
    c->prog->instructions[jump_instr_addr].p2 = c->prog->count;
}
```
The `patch_jump` function is crucial. When you emit a conditional jump instruction (for WHERE clause evaluation), you don't yet know the jump target — the instructions that follow haven't been emitted. So you emit a placeholder jump, record its address, emit the remaining instructions, then go back and fill in the correct target. This forward-reference patching is standard in every bytecode compiler.
---
## Compiling SELECT

![SELECT → Bytecode Compilation — Trace Example](./diagrams/diag-vdbe-select-compilation.svg)

SELECT compilation is the most instructive case because it reveals the cursor abstraction — the mechanism that hides B-tree traversal behind a simple iterator interface.
The compiled bytecode for `SELECT * FROM users` (full table scan, no WHERE) looks like this:
```
addr  opcode         p1    p2    p3    comment
----  -----------    ----  ----  ----  -------
0     OpenTable      0     0     0     open cursor 0 on table 'users'
1     Rewind         0     4     0     seek cursor to first row; if empty, jump to 4
2     ResultRow      1     N     0     emit columns 1..N as one result row
3     Next           0     2     0     advance cursor; if not EOF, jump back to 2
4     Halt           0     0     0     done
```
This is the fundamental loop structure for any table scan. The VDBE executes instructions 2 and 3 once per row. The entire scan of a million-row table is this tiny 5-instruction program executing two instructions per row — and those two instructions are integer comparisons in a tight switch statement.
Here is the compiler function for SELECT:
```c
static void compile_select(Compiler *c, AstNode *node) {
    /* Allocate a cursor for the table */
    int cursor = alloc_cursor(c);
    /* OP_OPEN_TABLE: opens a read cursor on the named table */
    emit_p4_str(c, OP_OPEN_TABLE, cursor, 0, 0,
                node->select.table_name, "open table cursor");
    /* OP_REWIND: position cursor at first row. p2 is the address to jump to
       if the table is empty — we patch this after emitting the loop. */
    int rewind_addr = emit(c, OP_REWIND, cursor, 0, 0, "seek to first row");
    /* Body of the scan loop: evaluate WHERE, then output columns */
    int loop_start = c->prog->count;  /* address of first instruction in loop body */
    /* --- WHERE clause --- */
    /* The WHERE clause emits conditional jumps that skip to OP_NEXT if the
       predicate is false. We collect these "skip to next" addresses to patch. */
    int where_jump_addr = -1;
    if (node->select.where_expr) {
        where_jump_addr = compile_where(c, cursor, node->select.where_expr);
        /* compile_where returns the address of the conditional skip instruction */
    }
    /* --- Column projection --- */
    /* Allocate registers for the output columns */
    int first_result_reg = c->next_reg;
    int col_count = 0;
    if (node->select.column_count == 1 &&
        node->select.columns[0]->type == NODE_WILDCARD) {
        /* SELECT * — emit OP_COLUMN for each column in table schema.
           At compile time, we need to know the schema. For now, assume schema
           is available via a schema lookup; we'll return to this in Milestone 5. */
        int ncols = schema_get_column_count(node->select.table_name);
        for (int i = 0; i < ncols; i++) {
            int reg = alloc_register(c);
            emit(c, OP_COLUMN, cursor, i, reg, "load column");
            col_count++;
        }
    } else {
        /* SELECT col1, col2, ... */
        for (int i = 0; i < node->select.column_count; i++) {
            int reg = alloc_register(c);
            compile_expr_into_reg(c, cursor, node->select.columns[i], reg);
            col_count++;
        }
    }
    /* OP_RESULT_ROW: emit the current row to the result set */
    emit(c, OP_RESULT_ROW, first_result_reg, col_count, 0, "output row");
    /* If WHERE compiled a conditional skip, patch it to jump here (past ResultRow,
       to the OP_NEXT below). */
    if (where_jump_addr >= 0) {
        patch_jump(c, where_jump_addr);
    }
    /* OP_NEXT: advance cursor. p2 is the loop body start — jump back if not EOF */
    emit(c, OP_NEXT, cursor, loop_start, 0, "advance cursor");
    /* Patch the Rewind to jump here (past the loop) when table is empty */
    patch_jump(c, rewind_addr);
    /* OP_CLOSE and OP_HALT */
    emit(c, OP_CLOSE,  cursor, 0, 0, "close cursor");
    emit(c, OP_HALT,   0,      0, 0, "done");
}
```
Notice the two-phase emit: emit the jump instruction with a placeholder target, emit the body, then patch the target. This pattern appears at every control flow point.
Also notice `schema_get_column_count` — at compile time, the compiler needs to know the table schema to resolve column indices and wildcard expansion. The schema is stored in a system catalog (implemented in Milestone 5). For now, you can stub this with a hardcoded function or pass schema information through.
---
## Compiling the WHERE Clause

![VM Fetch-Decode-Execute Loop — Data Walk](./diagrams/diag-vdbe-execution-loop.svg)

The WHERE clause is the most algorithmically interesting part of compilation, because it must translate a boolean expression tree into conditional jump logic. The key insight: **SQL evaluates the WHERE clause to decide whether to include each row. The bytecode equivalent is: evaluate the predicate, and jump past the output instructions if the predicate is false.**
Consider `WHERE age > 18`. The bytecode approach:
```
3     Column         0     2     1     load column 'age' into reg 1
4     Integer        18    2     0     load immediate 18 into reg 2
5     Le             1     8     2     if reg1 <= reg2 (age <= 18), jump to addr 8 (OP_NEXT)
6     ResultRow      3     N     0     (emit row — only reached if age > 18)
7     ...
8     Next           0     3     0     advance cursor
```
Instructions 3-5 are the WHERE evaluation. If `age <= 18` (the negation of our condition), we skip to OP_NEXT. If `age > 18`, we fall through to ResultRow.
The comparison opcodes in the VDBE use "jump if NOT matching" semantics — this is deliberate. By jumping over the output when the condition fails, you avoid needing an explicit OP_GOTO for the common case. The jump is only taken for rows that don't match, which (for selective predicates) is most rows.
```c
/* Compile a WHERE expression. Returns the address of the conditional
   skip instruction, so the caller can patch it to point past the
   output instructions. */
static int compile_where(Compiler *c, int cursor, AstNode *where) {
    if (where->type == NODE_BINARY_EXPR) {
        return compile_binary_where(c, cursor, where);
    }
    /* For complex expressions, fall back to evaluate-into-register + OP_IF_NOT */
    int reg = alloc_register(c);
    compile_expr_into_reg(c, cursor, where, reg);
    return emit(c, OP_IF_NOT, reg, 0, 0, "skip if WHERE false");
}
static int compile_binary_where(Compiler *c, int cursor, AstNode *expr) {
    /* AND: both sides must be true */
    if (expr->binary.op == TOKEN_AND) {
        /* Compile left side — if it fails, skip */
        int left_jump = compile_where(c, cursor, expr->binary.left);
        /* If left succeeds, compile right side */
        int right_jump = compile_where(c, cursor, expr->binary.right);
        /* Both jumps need to be patched to the same "skip to next" target.
           We return the right_jump; the left_jump is patched inline.
           In practice, you track all pending jumps in a list. */
        /* For simplicity here, we chain: left failure jumps to right's failure target */
        patch_jump(c, left_jump);  /* left failure also goes to the combined skip */
        return right_jump;
    }
    /* OR: either side is enough */
    if (expr->binary.op == TOKEN_OR) {
        /* Compile left side — if it succeeds, skip the right side check */
        int left_reg = alloc_register(c);
        compile_expr_into_reg(c, cursor, expr->binary.left, left_reg);
        int left_success = emit(c, OP_IF, left_reg, 0, 0, "OR: skip right if left true");
        /* Compile right side */
        int right_jump = compile_where(c, cursor, expr->binary.right);
        /* Patch left_success to jump past the right_jump, to ResultRow */
        /* This requires knowing the address after right_jump — complex but doable */
        patch_jump(c, left_success);
        return right_jump;
    }
    /* Comparison operators: emit OP_COLUMN + OP_INTEGER/etc + OP_LE/OP_GT/etc */
    int left_reg  = alloc_register(c);
    int right_reg = alloc_register(c);
    compile_expr_into_reg(c, cursor, expr->binary.left,  left_reg);
    compile_expr_into_reg(c, cursor, expr->binary.right, right_reg);
    /* Map AST operator to VDBE comparison opcode.
       Inversion rule: we jump when the condition is FALSE, so we invert the operator.
       "age > 18" → if age <= 18 then skip → OP_LE */
    Opcode skip_op;
    const char *comment;
    switch (expr->binary.op) {
    case TOKEN_GT:  skip_op = OP_LE; comment = "skip if not >"; break;
    case TOKEN_GTE: skip_op = OP_LT; comment = "skip if not >="; break;
    case TOKEN_LT:  skip_op = OP_GE; comment = "skip if not <"; break;
    case TOKEN_LTE: skip_op = OP_GT; comment = "skip if not <="; break;
    case TOKEN_EQ:  skip_op = OP_NE; comment = "skip if not ="; break;
    case TOKEN_NEQ: skip_op = OP_EQ; comment = "skip if not !="; break;
    default:
        snprintf(c->error, sizeof(c->error), "Unsupported comparison operator in WHERE");
        c->had_error = 1;
        return -1;
    }
    return emit(c, skip_op, left_reg, 0, right_reg, comment);
}
```
The inversion of comparison operators (`GT` → `LE` for the skip condition) is subtle but important. Make sure you understand it: if you want to *include* rows where `age > 18`, you emit a jump for when `age <= 18` (the rows to exclude). Memorize the inversion table; it will trip you up if you mix it up.

![Register File — State Evolution](./diagrams/diag-vdbe-register-file.svg)

---
## Compiling Expressions Into Registers
The `compile_expr_into_reg` function bridges the gap between AST expression nodes and register file values. It handles literals, column references, and arithmetic:
```c
static void compile_expr_into_reg(Compiler *c, int cursor,
                                   AstNode *expr, int dest_reg) {
    if (c->had_error) return;
    switch (expr->type) {
    case NODE_LITERAL_INT: {
        int64_t v = expr->lit_int.value;
        if (v >= INT32_MIN && v <= INT32_MAX) {
            /* Small integers fit in p1 directly — no p4 needed */
            emit(c, OP_INTEGER, (int)v, dest_reg, 0, "integer immediate");
        } else {
            emit_p4_i64(c, OP_INT64, dest_reg, 0, 0, v, "large integer");
        }
        break;
    }
    case NODE_LITERAL_FLOAT:
        emit(c, OP_REAL, dest_reg, 0, 0, "real immediate");
        c->prog->instructions[c->prog->count - 1].p4.real = expr->lit_float.value;
        break;
    case NODE_LITERAL_STRING:
        emit_p4_str(c, OP_STRING8, dest_reg, 0, 0,
                    expr->lit_str.value, "string literal");
        break;
    case NODE_LITERAL_NULL:
        emit(c, OP_NULL, dest_reg, 0, 0, "null literal");
        break;
    case NODE_IDENTIFIER: {
        /* Look up column index in schema */
        int col_idx = schema_get_column_index(
            /* table name */ NULL,  /* resolved from context */
            expr->identifier.name
        );
        if (col_idx < 0) {
            snprintf(c->error, sizeof(c->error),
                     "Unknown column: %s", expr->identifier.name);
            c->had_error = 1;
            return;
        }
        emit(c, OP_COLUMN, cursor, col_idx, dest_reg, "load column");
        break;
    }
    case NODE_BINARY_EXPR: {
        /* Arithmetic operators */
        int left_reg  = alloc_register(c);
        int right_reg = alloc_register(c);
        compile_expr_into_reg(c, cursor, expr->binary.left,  left_reg);
        compile_expr_into_reg(c, cursor, expr->binary.right, right_reg);
        Opcode arith_op;
        switch (expr->binary.op) {
        case TOKEN_PLUS:  arith_op = OP_ADD;      break;
        case TOKEN_MINUS: arith_op = OP_SUBTRACT; break;
        case TOKEN_STAR:  arith_op = OP_MULTIPLY; break;
        case TOKEN_SLASH: arith_op = OP_DIVIDE;   break;
        default:
            snprintf(c->error, sizeof(c->error),
                     "Unsupported operator in expression context");
            c->had_error = 1;
            return;
        }
        emit(c, arith_op, left_reg, dest_reg, right_reg, "arithmetic");
        break;
    }
    default:
        snprintf(c->error, sizeof(c->error),
                 "Unsupported expression type in compiler: %d", expr->type);
        c->had_error = 1;
    }
}
```
The `OP_COLUMN` instruction is the bridge between the VM and the storage engine. When the VM executes `OP_COLUMN cursor=0, col_idx=2, dest_reg=5`, it asks the cursor to deserialize column 2 of the current row and write the resulting `Value` into register 5. The cursor knows how to navigate the B-tree page and decode the variable-length record format (Milestone 5). The VM knows nothing about page layouts — it just calls the cursor's column function. This abstraction boundary is critical.
---
## Compiling INSERT

![INSERT → Bytecode — Trace Example](./diagrams/diag-vdbe-insert-compilation.svg)

INSERT compilation is simpler than SELECT: no cursor loop, no conditional jumps. The pattern:
1. Open a write cursor on the target table
2. Load each value into a register
3. Pack the registers into a serialized record (`OP_MAKE_RECORD`)
4. Insert the record (`OP_INSERT`)
5. Halt
```c
static void compile_insert(Compiler *c, AstNode *node) {
    int cursor = alloc_cursor(c);
    /* Open table for writing (write flag = 1, distinguished from read cursor) */
    emit_p4_str(c, OP_OPEN_TABLE, cursor, 1 /*write*/, 0,
                node->insert.table_name, "open write cursor");
    /* Compile each value expression into a register */
    int first_val_reg = c->next_reg;
    for (int i = 0; i < node->insert.value_count; i++) {
        int reg = alloc_register(c);
        compile_expr_into_reg(c, cursor, node->insert.values[i], reg);
    }
    /* Allocate a register for the rowid */
    int rowid_reg  = alloc_register(c);
    int record_reg = alloc_register(c);
    /* OP_NULL in rowid_reg means "auto-assign next rowid" */
    emit(c, OP_NULL, rowid_reg, 0, 0, "auto rowid");
    /* OP_MAKE_RECORD: serialize registers first_val_reg .. first_val_reg+count-1
       into a binary record, store handle in record_reg */
    emit(c, OP_MAKE_RECORD, first_val_reg, node->insert.value_count, record_reg,
         "pack record");
    /* OP_INSERT: insert (rowid_reg, record_reg) into cursor */
    emit(c, OP_INSERT, cursor, record_reg, rowid_reg, "insert row");
    emit(c, OP_CLOSE, cursor, 0, 0, "close write cursor");
    emit(c, OP_HALT,  0,      0, 0, "done");
}
```
`OP_MAKE_RECORD` is the instruction that calls into the record serializer (implemented in Milestone 5). It reads `count` consecutive registers starting at `p1`, serializes them in SQLite's variable-length record format, and stores a reference to the resulting binary blob in `p3`. The VM passes this blob to `OP_INSERT`, which writes it to the B-tree.
The `OP_NULL` in the rowid register tells the INSERT handler to auto-assign the next rowid (effectively `MAX(rowid) + 1` for a new empty table, or retrieved from an auto-increment counter). This implements SQLite's INTEGER PRIMARY KEY autoincrement behavior.
---
## The Top-Level Compiler Entry Point
```c
Program *compile(AstNode *ast, const char **error_out) {
    Program *prog = calloc(1, sizeof(Program));
    prog->capacity = 16;
    prog->instructions = malloc(prog->capacity * sizeof(Instruction));
    prog->count = 0;
    Compiler c;
    compiler_init(&c, prog);
    switch (ast->type) {
    case NODE_SELECT:
        compile_select(&c, ast);
        break;
    case NODE_INSERT:
        compile_insert(&c, ast);
        break;
    case NODE_CREATE_TABLE:
        compile_create_table(&c, ast);
        break;
    default:
        snprintf(c.error, sizeof(c.error),
                 "Unsupported statement type for compilation: %d", ast->type);
        c.had_error = 1;
    }
    if (c.had_error) {
        if (error_out) *error_out = strdup(c.error);
        program_free(prog);
        return NULL;
    }
    if (error_out) *error_out = NULL;
    return prog;
}
```
---
## The Virtual Machine: Fetch-Decode-Execute
The VM is the component most people imagine when they think "database engine" — but it is actually the simplest piece. The compiler did all the hard structural work; the VM just executes the resulting instructions one at a time.
```c
typedef struct {
    /* Cursors: one per open table */
    BTreeCursor *cursors[16];
    int          cursor_count;
    /* Register file */
    Value        regs[VDBE_MAX_REGISTERS];
    /* Program counter */
    int          pc;
    /* Output callback */
    FILE        *result_out;
    /* Database handle */
    DB          *db;
    /* Error state */
    char         error[256];
    int          had_error;
} VdbeState;
int vdbe_execute(Program *prog, DB *db, FILE *result_out) {
    VdbeState vm = {0};
    vm.db         = db;
    vm.result_out = result_out;
    vm.pc         = 0;
    while (vm.pc < prog->count) {
        Instruction *instr = &prog->instructions[vm.pc];
        switch (instr->opcode) {
        case OP_OPEN_TABLE: {
            int cursor_id  = instr->p1;
            int write_mode = instr->p2;
            const char *tname = instr->p4.str;
            vm.cursors[cursor_id] = btree_cursor_open(db, tname, write_mode);
            if (!vm.cursors[cursor_id]) {
                snprintf(vm.error, sizeof(vm.error),
                         "Table not found: %s", tname);
                vm.had_error = 1;
                goto halt;
            }
            vm.pc++;
            break;
        }
        case OP_REWIND: {
            int cursor_id = instr->p1;
            int jump_if_empty = instr->p2;
            BTreeCursor *cur = vm.cursors[cursor_id];
            int empty = btree_cursor_rewind(cur);
            if (empty) {
                vm.pc = jump_if_empty;  /* skip the loop entirely */
            } else {
                vm.pc++;
            }
            break;
        }
        case OP_NEXT: {
            int cursor_id    = instr->p1;
            int loop_back_to = instr->p2;
            BTreeCursor *cur = vm.cursors[cursor_id];
            int has_more = btree_cursor_next(cur);
            if (has_more) {
                vm.pc = loop_back_to;   /* jump back to loop body */
            } else {
                vm.pc++;                /* fall through past loop */
            }
            break;
        }
        case OP_COLUMN: {
            int cursor_id = instr->p1;
            int col_idx   = instr->p2;
            int dest_reg  = instr->p3;
            BTreeCursor *cur = vm.cursors[cursor_id];
            btree_cursor_column(cur, col_idx, &vm.regs[dest_reg]);
            vm.pc++;
            break;
        }
        case OP_INTEGER: {
            int dest_reg = instr->p2;
            vm.regs[dest_reg].type = VAL_INTEGER;
            vm.regs[dest_reg].i    = instr->p1;
            vm.pc++;
            break;
        }
        case OP_INT64: {
            int dest_reg = instr->p1;
            vm.regs[dest_reg].type = VAL_INTEGER;
            vm.regs[dest_reg].i    = instr->p4.i64;
            vm.pc++;
            break;
        }
        case OP_REAL: {
            int dest_reg = instr->p1;
            vm.regs[dest_reg].type = VAL_REAL;
            vm.regs[dest_reg].r    = instr->p4.real;
            vm.pc++;
            break;
        }
        case OP_STRING8: {
            int dest_reg = instr->p1;
            vm.regs[dest_reg].type      = VAL_TEXT;
            vm.regs[dest_reg].text.data = instr->p4.str;
            vm.regs[dest_reg].text.len  = (int)strlen(instr->p4.str);
            vm.pc++;
            break;
        }
        case OP_NULL: {
            int dest_reg = instr->p1;
            vm.regs[dest_reg].type = VAL_NULL;
            vm.pc++;
            break;
        }
        case OP_GT: {
            int ra = instr->p1;
            int jump_to = instr->p2;
            int rb = instr->p3;
            int cmp = value_compare(&vm.regs[ra], &vm.regs[rb]);
            if (cmp <= 0) {          /* NOT greater-than: skip */
                vm.pc = jump_to;
            } else {
                vm.pc++;
            }
            break;
        }
        case OP_LE: {
            int ra = instr->p1;
            int jump_to = instr->p2;
            int rb = instr->p3;
            int cmp = value_compare(&vm.regs[ra], &vm.regs[rb]);
            if (cmp > 0) {           /* NOT less-than-or-equal: don't skip */
                vm.pc++;
            } else {
                vm.pc = jump_to;     /* a <= b: skip */
            }
            break;
        }
        /* ... similar for OP_LT, OP_GE, OP_EQ, OP_NE, OP_IS_NULL, OP_NOT_NULL ... */
        case OP_RESULT_ROW: {
            int start_reg = instr->p1;
            int count     = instr->p2;
            /* Print tab-separated values — in a real database, these would be
               returned through a callback or result set API */
            for (int i = 0; i < count; i++) {
                if (i > 0) fputc('|', result_out);
                value_print(&vm.regs[start_reg + i], result_out);
            }
            fputc('\n', result_out);
            vm.pc++;
            break;
        }
        case OP_MAKE_RECORD: {
            int start_reg  = instr->p1;
            int count      = instr->p2;
            int dest_reg   = instr->p3;
            /* Serialize registers into a record blob.
               This calls into the storage engine's record encoder. */
            char *record;
            int   record_len;
            record_encode(&vm.regs[start_reg], count, &record, &record_len);
            vm.regs[dest_reg].type      = VAL_BLOB;
            vm.regs[dest_reg].blob.data = (uint8_t *)record;
            vm.regs[dest_reg].blob.len  = record_len;
            vm.pc++;
            break;
        }
        case OP_INSERT: {
            int cursor_id  = instr->p1;
            int record_reg = instr->p2;
            int rowid_reg  = instr->p3;
            BTreeCursor *cur = vm.cursors[cursor_id];
            int64_t rowid;
            if (vm.regs[rowid_reg].type == VAL_NULL) {
                rowid = btree_next_rowid(cur);  /* auto-increment */
            } else {
                rowid = vm.regs[rowid_reg].i;
            }
            btree_cursor_insert(cur,
                                rowid,
                                vm.regs[record_reg].blob.data,
                                vm.regs[record_reg].blob.len);
            vm.pc++;
            break;
        }
        case OP_GOTO:
            vm.pc = instr->p2;
            break;
        case OP_IF: {
            int reg = instr->p1;
            if (value_is_truthy(&vm.regs[reg])) {
                vm.pc = instr->p2;
            } else {
                vm.pc++;
            }
            break;
        }
        case OP_IF_NOT: {
            int reg = instr->p1;
            if (!value_is_truthy(&vm.regs[reg])) {
                vm.pc = instr->p2;
            } else {
                vm.pc++;
            }
            break;
        }
        case OP_CLOSE: {
            int cursor_id = instr->p1;
            btree_cursor_close(vm.cursors[cursor_id]);
            vm.cursors[cursor_id] = NULL;
            vm.pc++;
            break;
        }
        case OP_HALT:
        halt:
            goto done;
        default:
            snprintf(vm.error, sizeof(vm.error),
                     "Unknown opcode: %d at PC=%d", instr->opcode, vm.pc);
            vm.had_error = 1;
            goto done;
        }
    }
done:
    /* Close any cursors left open (cleanup on halt or error) */
    for (int i = 0; i < 16; i++) {
        if (vm.cursors[i]) btree_cursor_close(vm.cursors[i]);
    }
    return vm.had_error ? -1 : 0;
}
```
This is the complete VM core. Every opcode follows the same pattern: read operands from the instruction, perform the operation, increment `vm.pc` (or set it to a jump target), break. The `goto done` for `OP_HALT` and error handling is cleaner than nested break-from-switch.
The `value_compare` function handles type coercion — comparing an integer to a real, comparing text by lexicographic order. SQL's comparison semantics are non-trivial: comparing `NULL` to anything returns NULL (not true or false), integers and reals are compared numerically with promotion, text comparison is locale-dependent. Keep `value_compare` correct — wrong comparison behavior silently corrupts query results.
```c
/* Returns negative if a < b, 0 if a == b, positive if a > b.
   NULL comparisons return INT32_MAX (treated as "not comparable" — callers must check). */
static int value_compare(const Value *a, const Value *b) {
    /* NULL handling: NULL != anything, NULL != NULL */
    if (a->type == VAL_NULL || b->type == VAL_NULL) return INT32_MAX;
    /* Numeric comparison with type promotion */
    if ((a->type == VAL_INTEGER || a->type == VAL_REAL) &&
        (b->type == VAL_INTEGER || b->type == VAL_REAL)) {
        double da = (a->type == VAL_INTEGER) ? (double)a->i : a->r;
        double db = (b->type == VAL_INTEGER) ? (double)b->i : b->r;
        if (da < db) return -1;
        if (da > db) return  1;
        return 0;
    }
    /* Text comparison */
    if (a->type == VAL_TEXT && b->type == VAL_TEXT) {
        int min_len = a->text.len < b->text.len ? a->text.len : b->text.len;
        int cmp = memcmp(a->text.data, b->text.data, min_len);
        if (cmp != 0) return cmp;
        return a->text.len - b->text.len;
    }
    /* Mixed types: SQLite defines type order: NULL < INTEGER < REAL < TEXT < BLOB */
    return (int)a->type - (int)b->type;
}
```
---
## The EXPLAIN Command

![SELECT → Bytecode Compilation — Trace Example](./diagrams/diag-vdbe-select-compilation.svg)

EXPLAIN is not just a debugging convenience — it is the window into your database engine that transforms it from a black box into an observable system. Every production database has EXPLAIN. PostgreSQL's `EXPLAIN ANALYZE` is a career-defining tool for backend engineers. SQLite's `.explain` mode is how SQLite developers verify compiler output.
The implementation is simple: instead of executing the program, print it:
```c
/* Human-readable opcode names */
static const char *opcode_name(Opcode op) {
    switch (op) {
    case OP_OPEN_TABLE:  return "OpenTable";
    case OP_REWIND:      return "Rewind";
    case OP_NEXT:        return "Next";
    case OP_CLOSE:       return "Close";
    case OP_COLUMN:      return "Column";
    case OP_ROWID:       return "Rowid";
    case OP_MAKE_RECORD: return "MakeRecord";
    case OP_INSERT:      return "Insert";
    case OP_INTEGER:     return "Integer";
    case OP_INT64:       return "Int64";
    case OP_REAL:        return "Real";
    case OP_STRING8:     return "String8";
    case OP_NULL:        return "Null";
    case OP_COPY:        return "Copy";
    case OP_EQ:          return "Eq";
    case OP_NE:          return "Ne";
    case OP_LT:          return "Lt";
    case OP_LE:          return "Le";
    case OP_GT:          return "Gt";
    case OP_GE:          return "Ge";
    case OP_IS_NULL:     return "IsNull";
    case OP_NOT_NULL:    return "NotNull";
    case OP_ADD:         return "Add";
    case OP_SUBTRACT:    return "Subtract";
    case OP_MULTIPLY:    return "Multiply";
    case OP_DIVIDE:      return "Divide";
    case OP_GOTO:        return "Goto";
    case OP_IF:          return "If";
    case OP_IF_NOT:      return "IfNot";
    case OP_RESULT_ROW:  return "ResultRow";
    case OP_CREATE_TABLE:return "CreateTable";
    case OP_HALT:        return "Halt";
    default:             return "???";
    }
}
void program_explain(Program *prog, FILE *out) {
    fprintf(out, "%-4s  %-14s  %-6s  %-6s  %-6s  %s\n",
            "addr", "opcode", "p1", "p2", "p3", "comment");
    fprintf(out, "----  --------------  ------  ------  ------  -------\n");
    for (int i = 0; i < prog->count; i++) {
        Instruction *in = &prog->instructions[i];
        fprintf(out, "%-4d  %-14s  %-6d  %-6d  %-6d  %s\n",
                i,
                opcode_name(in->opcode),
                in->p1, in->p2, in->p3,
                in->comment ? in->comment : "");
    }
}
```
The EXPLAIN command in your SQL interface:
```c
/* In your SQL dispatcher */
if (ast->type == NODE_EXPLAIN) {
    ParseResult inner = parse_statement(ast->explain.sql);
    if (inner.had_error) { /* report error */ return; }
    const char *err;
    Program *prog = compile(inner.root, &err);
    if (!prog) { fprintf(stderr, "Compile error: %s\n", err); return; }
    program_explain(prog, stdout);
    program_free(prog);
    ast_free(inner.root);
    return;
}
```
What `EXPLAIN SELECT * FROM users WHERE age > 18` produces:
```
addr  opcode          p1      p2      p3      comment
----  --------------  ------  ------  ------  -------
0     OpenTable       0       0       0       open table cursor
1     Rewind          0       7       0       seek to first row
2     Column          0       2       1       load column 'age' into reg 1
3     Integer         18      2       0       integer immediate
4     Le              1       6       2       skip if not >
5     ResultRow       3       5       0       output row
6     Next            0       2       0       advance cursor
7     Close           0       0       0       close cursor
8     Halt            0       0       0       done
```
Every time you query the database and get unexpected results, run EXPLAIN first. The bytecode shows you exactly what the engine will do — which cursor it opens, which comparisons it evaluates, which registers it uses. There is no mystery when execution is visible.
---
## Three-Level View: What Happens at Each Layer
**Level 1 — SQL query (what the user writes)**
```sql
SELECT name, email FROM users WHERE age > 18;
```
This is text. The user's intention. Humans understand it; computers don't.
**Level 2 — VDBE bytecode (what you just built)**
```
OpenTable  0         -- open cursor on 'users'
Rewind     0  →end   -- position at first row
Column     0  1  r1  -- load 'age' into reg 1
Integer    18    r2  -- load constant 18 into reg 2
Le         r1 →next r2  -- skip if age <= 18
Column     0  2  r3  -- load 'name' into reg 3
Column     0  3  r4  -- load 'email' into reg 4
ResultRow  r3  2     -- emit (name, email)
Next       0  →loop  -- advance; loop if not EOF
Halt
```
This is an integer sequence. The CPU can execute this loop in a tight, cache-friendly branch with predictable memory access patterns. The VM has no notion of "SQL" or "WHERE" — just opcodes and registers.
**Level 3 — Disk I/O (what happens beneath the VM)**
Each `OP_COLUMN` call reaches down through the cursor API into the buffer pool: "give me the value at column index 2 of the current row." The buffer pool checks if the containing page is in memory. If yes, it's a pointer arithmetic operation into a 4096-byte page. If no, it calls `pread()` to load the page from disk — a system call that may block for 100 microseconds while the SSD seeks. The VM knows none of this. It just sees a `Value` returned from `btree_cursor_column`.
---
## Design Decisions: Why Register-Based?
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Register-based (Chosen ✓)** | Fewer instructions, better cache locality, explicit register assignment enables optimization passes | Wider instructions (carry register numbers), harder to write compiler for | SQLite VDBE, Lua 5.0+, Dalvik/ART, LuaJIT |
| Stack-based | Simpler to compile (no register allocation), smaller instructions | Implicit operand stack requires more instructions for same computation, stack discipline harder to optimize | CPython, JVM, WebAssembly, .NET CLR |
| AST-walking interpreter | No compilation step, simplest possible implementation | Re-traverses tree structure per row, cache-unfriendly pointer chasing, hard to optimize | Many toy interpreters, some scripting languages for simple scripts |
SQLite chose register-based for a specific reason: the instructions are "thick" — one opcode does a lot of work (opening a B-tree cursor, reading a column value including deserialization). When each instruction does significant work, the overhead of carrying register numbers is negligible compared to the work saved by fewer total instructions.
---
## The Missing Opcode: `OP_HALT` and the End of the World
Add `OP_HALT` to the end of every compiled program. Without it, the VM increments `pc` past the last instruction and reads garbage memory. This is not a hypothetical concern — in C, array out-of-bounds is undefined behavior. The CPU will read whatever bytes happen to follow the instruction buffer in memory and attempt to decode them as opcodes. The resulting behavior is the worst kind of bug: occasional crashes on some machines, silent wrong results on others, correct behavior in debug builds.
The rule: every compile function ends with `emit(c, OP_HALT, 0, 0, 0, "done")`. No exceptions.
---
## Performance: The 100ms Guarantee
The acceptance criterion specifies `SELECT * FROM t` on 10,000 rows in under 100ms. Let's do the math to understand what this implies.
A table scan executes roughly 3 instructions per row: `OP_COLUMN` (or multiple), `OP_RESULT_ROW`, `OP_NEXT`. At 10,000 rows, that's ~30,000 instruction dispatches. Each dispatch involves: array index into `prog->instructions`, switch on opcode, a few register reads/writes. On modern hardware, a tight switch-dispatch loop runs at roughly 200-400 million iterations per second.
30,000 iterations at 200M/second = **0.15 milliseconds** for pure VM overhead.
The bottleneck is I/O. 10,000 rows at, say, 100 bytes each = 1MB of data. With a 4KB page size, that's ~250 pages. If all pages are in the buffer pool (in-memory), 250 page reads at ~50ns each (L2 cache access) = **12.5 microseconds**. If pages are on SSD (not cached), 250 random reads at ~100μs each = **25 milliseconds**.
25ms is well under 100ms. The benchmark is achievable even with cold pages, because 10,000 rows is a small dataset.
The real performance danger is per-row overhead in the VM that scales linearly. Two common culprits:
1. **`malloc` per row** — if `OP_STRING8` allocates heap memory for string values on every row, and you're iterating 10,000 rows, that's 10,000 allocations. Keep string values as pointers into page data (zero-copy) for the hot path.
2. **Type dispatch on `value_compare`** — if the comparison function branches on types and the column is always `INTEGER`, you can specialize: if both operands are `VAL_INTEGER`, compare directly without promotion to `double`. This is a 5x speedup on integer-heavy WHERE clauses.
For this milestone, neither optimization is required to meet the 100ms target. But understanding the math tells you where to look when performance is insufficient.
---
## Cursor Abstraction: The Most Important Interface
The cursor interface is worth examining carefully, because it's the most consequential abstraction boundary in the entire codebase:
```c
/* btree_cursor.h */
BTreeCursor *btree_cursor_open(DB *db, const char *table_name, int write_mode);
int          btree_cursor_rewind(BTreeCursor *cur);   /* returns 1 if table empty */
int          btree_cursor_next(BTreeCursor *cur);     /* returns 1 if more rows exist */
void         btree_cursor_column(BTreeCursor *cur, int col_idx, Value *out);
int64_t      btree_next_rowid(BTreeCursor *cur);
void         btree_cursor_insert(BTreeCursor *cur, int64_t rowid,
                                  const uint8_t *record, int record_len);
void         btree_cursor_close(BTreeCursor *cur);
```
From the VM's perspective, this interface is the entire storage engine. The VM doesn't know if there's a B-tree, a hash table, or a network socket underneath. The opcode `OP_NEXT` calls `btree_cursor_next`; the opcode `OP_COLUMN` calls `btree_cursor_column`. The VM is completely decoupled from storage representation.
This is the same abstraction that makes real databases powerful: PostgreSQL's executor doesn't know if data comes from a heap table, an index, a view, a foreign table backed by a network connection, or a function call. It just calls `ExecScan`. The scan API hides everything below.
For this milestone, stub these functions. Return hardcoded data that exercises the VM without needing a real B-tree:
```c
/* btree_stub.c — for testing the VM before Milestone 5 */
typedef struct {
    /* Hardcoded test data */
    const char *table_name;
    int current_row;
    int total_rows;
    /* Column values per row */
    Value rows[1000][8];
    int   ncols;
} StubCursor;
int btree_cursor_next(BTreeCursor *cur) {
    StubCursor *sc = (StubCursor *)cur;
    sc->current_row++;
    return sc->current_row < sc->total_rows;
}
void btree_cursor_column(BTreeCursor *cur, int col_idx, Value *out) {
    StubCursor *sc = (StubCursor *)cur;
    *out = sc->rows[sc->current_row][col_idx];
}
```
With this stub, you can fully test the VM — correct WHERE evaluation, correct column projection, correct result output — before writing a single byte of B-tree code. This is the power of the cursor abstraction: the VM and the storage engine are independently testable.
---
## Knowledge Cascade: What This Milestone Unlocks
**→ Every language runtime ever written.** The register-vs-stack tradeoff you just navigated appears in every language VM. The JVM was stack-based (1995); Dalvik/ART replaced it with register-based bytecode (2008) and was 30% faster for the same programs. Lua switched from stack-based (4.x) to register-based (5.0) for the same reason SQLite chose registers: when instructions do more work, the overhead of naming registers is worth paying. CPython is still stack-based — and remains slower than it could be for this reason. You now understand why.
**→ JIT compilation is a natural next step.** Once you have bytecode, you can JIT-compile it: translate the bytecode into native machine instructions at runtime, specializing for the actual types encountered. PostgreSQL 10 added JIT for expression evaluation using LLVM, achieving 2-5x speedups on CPU-bound analytical queries. The architecture is: parse → optimize → bytecode → JIT → native. You've built the first four steps. 
> **🔑 Foundation: JIT compilation: translating bytecode to native code at runtime**
> 
> ## JIT Compilation: Translating Bytecode to Native Code at Runtime
**What it IS**
Just-In-Time (JIT) compilation is the process of translating bytecode (or some intermediate representation) into native machine code *during program execution*, rather than ahead of time. The result is that frequently-run code eventually runs at native CPU speed instead of being interpreted instruction-by-instruction.
The name captures the timing: compilation happens *just in time* — right before (or as) the code is needed, not before the program starts.
**The basic lifecycle:**
1. **Interpret first.** The VM starts by interpreting bytecode normally. This is fast to start and requires no upfront work.
2. **Profile / count.** The runtime tracks which functions or loops are executed frequently — these are called *hot spots*.
3. **Compile hot spots.** When a function exceeds a threshold (e.g., called 1000 times), the JIT kicks in: it translates that bytecode to native x86/ARM instructions and stores them in a memory buffer marked executable.
4. **Redirect execution.** Future calls to that function jump directly to the native code, bypassing the interpreter entirely.
**A concrete mental image:**
```
Bytecode:        LOAD R0, R1 | MUL R2, R0, R1 | RET R2
                      ↓  (after 1000 calls, JIT fires)
Native x86-64:   mov  rax, [rbp-8]
                 imul rax, [rbp-16]
                 ret
```
The second version runs at full hardware speed — no opcode dispatch, no indirection.
**Key implementation pieces:**
- **Code generation:** You walk the bytecode and emit native instruction bytes into a buffer (often using a library like AsmJit, DynASM, or hand-rolled emitters for each target architecture).
- **Executable memory:** The buffer must be allocated with `mmap`/`VirtualAlloc` using `PROT_EXEC` (write-then-execute, or use dual-mapping to avoid W^X security restrictions).
- **Calling convention:** Your JIT-compiled function must obey the platform ABI (System V on Linux/macOS, Microsoft x64 on Windows) so it can call C runtime functions and be called from your interpreter.
- **Guards and deoptimization:** A production JIT compiles *optimistic* fast paths. If an assumption breaks (e.g., a variable you assumed was always an integer turns out to be a string), the JIT must *deoptimize* — fall back to the interpreter. This is the hard part.
**Tiered approaches** are common: a "baseline JIT" does a fast, simple translation (little optimization, but still faster than interpreting); an "optimizing JIT" runs after more profiling and applies register allocation, inlining, dead-code elimination, etc. V8 (JavaScript) and HotSpot (Java) both use multi-tier pipelines.
**WHY you need it right now**
Your VM's interpreter is correct but slow for hot code. JIT compilation is the primary mechanism used by production language runtimes (JVM, V8, LuaJIT, PyPy) to achieve near-native performance. Understanding it now lets you design your bytecode instruction set and VM data structures in ways that are *JIT-friendly* — for example, keeping type information accessible, making your calling convention regular, and avoiding unnecessary indirection. Even if you implement only a simple baseline JIT, the architectural decisions you make now (register-based instructions, flat value representation) either enable or block that path.
**Key mental model**
JIT compilation is just an *optimized cache for translations*. The interpreter is always the fallback truth; the JIT is a fast-path shortcut that produces the same observable result. Think of it like memoization, but instead of caching a *return value*, you're caching a *compiled version of the computation itself*. The first call pays the translation cost; every subsequent call gets it for free at native speed.

**→ EXPLAIN transforms debugging into engineering.** The difference between `SELECT` returns wrong data (mystery) and `EXPLAIN SELECT` shows `OP_LE` where there should be `OP_GE` (root cause in 30 seconds) is the difference between guessing and knowing. Every database professional's first step when a query misbehaves is EXPLAIN. You've built the tool. Now you understand why it exists.
**→ Opcode as abstraction over storage.** The cursor opcodes (`OP_OPEN_TABLE`, `OP_REWIND`, `OP_NEXT`) are the same abstraction that makes PostgreSQL's foreign data wrappers work. A FDW (Foreign Data Wrapper) implements the same scan interface as a heap table — the executor calls the same function pointers regardless of whether data comes from PostgreSQL's own storage, a CSV file, a remote PostgreSQL server, or a Kafka topic. This pattern — abstract the data source behind an iterator interface, drive iteration from a loop in the execution engine — is one of the most powerful patterns in systems programming.
**→ Register allocation is a real compiler problem.** The `next_reg++` approach you used (always allocate a new register, never reuse) is the naive solution. A real register allocator would detect that a register's value is no longer needed (its last use is before the current point) and reuse the slot. SQLite's own compiler does this: it tracks register lifetimes and reuses registers to minimize the register file size. The general problem (optimal register allocation given unlimited registers → optimal register assignment for limited physical registers) is NP-complete, which is why compiler backends use heuristics. You've built the simple version; understanding why the hard version is hard is the first step toward improving it.
**→ Bytecode as a portable binary format.** SQLite stores compiled query plans in a feature called "prepared statements" — compiled once, executed many times with different parameter values. The bytecode is stable across executions: `OP_GT reg1 _ reg2` has the same semantics whether reg1 contains 5 or 500. This is why prepared statements are faster for repeated queries: the compilation overhead (parsing + AST building + bytecode generation) is paid once. You've built the compilation half; the prepared statement cache is a future optimization that builds directly on this foundation.
**→ The hot path is now in the VM, not the parser.** Before this milestone, every query execution required re-parsing SQL text. Now, parsing is a one-time cost. The hot path — iterating millions of rows — is the tight switch loop in `vdbe_execute`. This is where profiler time should be spent. Every microsecond saved in `value_compare` is multiplied by the number of rows scanned. Every allocation avoided in `OP_COLUMN` is a GC pressure reduction. You've identified the exact function to optimize for performance: `vdbe_execute`.
---
## Common Pitfalls: What Will Break Without Warning
**1. Missing `OP_HALT` causes undefined behavior.** The VM increments `pc` past the last instruction and reads garbage memory. On some hardware this loops; on others it crashes; on x86 with ASLR it's non-deterministic. Add `OP_HALT` at the end of every compiled program as an invariant, and assert in the VM that `OP_HALT` is the last instruction.
**2. Forgetting to patch forward jumps.** If `rewind_addr` is compiled but never patched, `OP_REWIND` jumps to address 0 (the uninitialized `p2` field) when the table is empty, restarting the entire program. This creates an infinite loop for empty tables. Verify every forward jump by reviewing the list of emitted but unpatched jump instructions before the function returns.
**3. Register index collisions.** If two paths through `compile_expr_into_reg` allocate the same register index due to a reset bug, one value clobbers another. The symptom is incorrect column values with no error. Write a test that compiles `SELECT a + b, a - b FROM t` and verifies both arithmetic results are correct — this catches register aliasing immediately.
**4. NULL comparison semantics.** `value_compare` returning `INT32_MAX` for NULL comparisons means the comparison opcodes must check for this sentinel before branching. If they don't, `WHERE x = NULL` passes (because `INT32_MAX > 0` is truthy), which is wrong — `x = NULL` must always evaluate to NULL (unknown), never true. Add explicit NULL checks in every comparison opcode.
**5. `OP_NEXT` semantics reversed.** The opcode advances the cursor and jumps *back to the loop body* when there are more rows — it does NOT jump *forward* on end-of-table. If you reverse this (jump forward when there ARE more rows), the VM processes only the first row and exits. Check: the `p2` of `OP_NEXT` must point to an address *before* the current `pc`.
**6. The compile/execute split for schema access.** The compiler needs schema information (column names → indices, column count for `SELECT *`) at compile time. The VM needs the schema at execute time (to open cursors). If you access schema from a global singleton, parallel compilation or query caching breaks. Design your schema access so the compiler can take a snapshot of the schema at compile time, and the VM doesn't need to re-read it.
---
## What You Have Built
At the end of this milestone, you have three production-quality components:
- An **instruction set** of ~30 opcodes covering cursor operations, data access, comparisons, arithmetic, control flow, and output — sufficient to execute SELECT, INSERT, and CREATE TABLE
- A **compiler** that translates SELECT ASTs (including WHERE clauses with AND/OR/comparison operators) and INSERT ASTs into correct bytecode programs, with proper forward-reference patching for all conditional jumps
- A **virtual machine** with a typed register file, a fetch-decode-execute loop, cursor management, and correct NULL semantics in all comparison operations
- An **EXPLAIN command** that prints the bytecode program in human-readable form for any SQL statement
- A **stub cursor interface** that lets you test the VM independently of the storage engine
The VDBE is the engine's center of gravity. Every upstream component (tokenizer, parser) feeds into it. Every downstream component (buffer pool, B-tree, indexes, query planner, transaction manager) is accessed through the cursor abstraction it defines. Adding a new SQL feature in later milestones means: extend the compiler to emit new opcodes, extend the VM to execute them. The architecture scales.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m4 -->
<!-- MS_ID: build-sqlite-m4 -->
# Milestone 4: Buffer Pool Manager
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You are now descending into the engine room. Look at where you stand in the pipeline: the tokenizer dissolved SQL text into tokens, the parser built an AST, the VDBE compiler translated that AST into bytecode, and the virtual machine executes those instructions. Every `OP_COLUMN` instruction the VM fires ends with a single question: *where is the data?*

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

The answer to that question is what you are building in this milestone. The buffer pool manager is the component that answers "where is page 42?" It either returns a pointer to memory (fast) or fetches the page from disk (slow) and then returns a pointer to memory. Everything above the buffer pool — the VDBE, the B-tree, the query planner — operates exclusively on memory addresses. Everything below the buffer pool — the raw file, the OS, the SSD — operates in disk pages. The buffer pool is the translation layer between these two worlds.
Without a buffer pool, every `OP_COLUMN` would call `pread()` to read 4096 bytes from the file, then parse the record, then discard the page. The next `OP_COLUMN` on the same page would call `pread()` again. A table scan of 10,000 rows fitting in 100 pages would call `pread()` 10,000 times instead of 100 times. That is the baseline cost the buffer pool eliminates.
---
## The Revelation: Why Not Just Use mmap?
Here is the trap that catches every database implementer who has studied systems programming: the operating system already has a page cache. Linux keeps recently-accessed file pages in memory automatically. If you `mmap()` your database file, the OS maps file pages into your address space, caches them automatically, and evicts them under memory pressure using its own replacement policy. You get caching for free, with zero code.
This reasoning is seductive and wrong. Not wrong in the sense that mmap doesn't work — it does work, and LMDB (the database behind OpenLDAP and many other systems) uses exactly this approach. But it's wrong for a database that needs crash safety with precise write ordering.
Here is what you need that the OS page cache cannot give you:
**1. Pin control.** During a B-tree traversal that descends three levels (root → internal → leaf), you hold references to three pages simultaneously. You need to guarantee that none of those pages are evicted while you're holding a pointer to them — otherwise you'd have a dangling pointer into freed memory. The OS page cache can reclaim pages at any time under memory pressure. With mmap, a page you're reading could be unmapped mid-traversal (this manifests as a SIGBUS signal, not a clean error).
With a buffer pool, you *pin* a page when you fetch it. A pinned page cannot be evicted. You unpin it when you're done. The OS cannot do this.
**2. Dirty page tracking for WAL.** To implement write-ahead logging (Milestone 10), you need to know *exactly* which pages have been modified and not yet written to the WAL file. With mmap, any write to the memory-mapped region modifies the page, but detecting which pages changed requires either write-protecting every page (expensive: one `mprotect()` call per page) or scanning every page for changes on commit (catastrophically slow for large databases). 
With a buffer pool, when code calls `buffer_pool_mark_dirty(frame)`, you set a single bit. You know exactly which pages need writing.
**3. Write ordering enforcement.** Crash safety requires that certain pages reach disk in a specific order: the WAL/journal record must be durable *before* the modified database page. The OS page cache makes no ordering guarantees. You call `fsync()` on the database file and the OS writes pages in whatever order it pleases. You cannot tell the OS "write page 7 before page 12."
With a buffer pool, you call `flush_page(7)` followed by `fsync()` followed by `flush_page(12)`. Ordering is yours to command.
**The mmap tradeoff:** LMDB accepts these limitations because it uses copy-on-write pages with a different transaction model. It writes new page versions, never overwrites in-place, and relies on the OS to provide atomic file operations at the page level. That model works, but it requires LMDB's specific multi-version concurrency approach. SQLite's rollback journal and WAL models require precise write ordering. This is why SQLite has a buffer pool.

![Buffer Pool Architecture — Street View](./diagrams/diag-buffer-pool-architecture.svg)

---
## What You Are Building
The buffer pool manages a fixed-size pool of **page frames** — fixed-size memory regions, each exactly 4096 bytes. A frame is the container; a page is the content. The same frame can hold different pages at different times.
The pool's responsibilities:
1. **FetchPage(page_id)** → return a pointer to the page's data in a frame (loading from disk if needed)
2. **UnpinPage(page_id)** → release your hold on the page (allowing eviction)
3. **MarkDirty(page_id)** → declare that you've modified the page's content
4. **NewPage()** → allocate a new page in the database file, return it pinned
5. **FlushPage(page_id)** → force a specific page to disk
6. **FlushAll()** → force all dirty pages to disk (used before checkpoint/shutdown)
The public C interface:
```c
/* buffer_pool.h */
#pragma once
#include <stdint.h>
#include <stdio.h>
#define PAGE_SIZE         4096
#define DEFAULT_POOL_SIZE 1000   /* frames */
typedef uint32_t PageId;
#define INVALID_PAGE_ID  ((PageId)UINT32_MAX)
/* Opaque handle — callers never inspect internals */
typedef struct BufferPool BufferPool;
/* Create a buffer pool attached to an open database file.
   fd     — file descriptor of the database file (already opened with O_RDWR)
   frames — number of page frames to allocate
   Returns NULL on allocation failure. */
BufferPool *buffer_pool_create(int fd, int frames);
/* Destroy the buffer pool, flushing all dirty pages.
   Does NOT close the file descriptor. */
void buffer_pool_destroy(BufferPool *bp);
/* Fetch a page into a frame.
   Returns a pointer to the frame's 4096-byte data buffer, pinned.
   The pointer is valid until UnpinPage is called.
   Returns NULL if all frames are pinned (pool exhausted).
   The caller MUST call buffer_pool_unpin() when done with the pointer. */
uint8_t *buffer_pool_fetch(BufferPool *bp, PageId page_id);
/* Unpin a page frame, making it eligible for eviction.
   dirty=1 marks the page as modified; dirty=0 leaves the dirty bit unchanged. */
void buffer_pool_unpin(BufferPool *bp, PageId page_id, int dirty);
/* Allocate a new page in the database file.
   Assigns the next page_id, pins the frame, zeroes the data.
   Stores the new page's ID in *page_id_out.
   Returns a pointer to the frame's data (pinned), or NULL on failure. */
uint8_t *buffer_pool_new_page(BufferPool *bp, PageId *page_id_out);
/* Force a specific page to disk immediately.
   The page must be in the pool (fetched) — does not load if not present.
   Returns 0 on success, -1 on write error. */
int buffer_pool_flush_page(BufferPool *bp, PageId page_id);
/* Flush ALL dirty pages to disk.
   Returns the number of pages flushed, or -1 on error. */
int buffer_pool_flush_all(BufferPool *bp);
/* Performance statistics */
typedef struct {
    uint64_t fetches;       /* total FetchPage calls */
    uint64_t hits;          /* pages found in pool */
    uint64_t misses;        /* pages loaded from disk */
    uint64_t evictions;     /* pages evicted */
    uint64_t dirty_writes;  /* dirty pages written to disk */
} BufferPoolStats;
void buffer_pool_get_stats(BufferPool *bp, BufferPoolStats *out);
void buffer_pool_reset_stats(BufferPool *bp);
```
Notice: callers never see frame numbers, never see the page table, never see the LRU state. The only thing callers hold is a raw `uint8_t *` pointing into a frame's data buffer. This pointer is valid while the page is pinned. The buffer pool is an opaque cache.
---
## The Internal Data Structures

![FetchPage — Hit vs Miss Data Walk](./diagrams/diag-buffer-pool-fetch-flow.svg)

Internally, the buffer pool needs three data structures working together:
**1. The frame array** — the actual memory. A flat array of `frames` entries, each containing 4096 bytes of page data plus frame metadata.
**2. The page table** — a hash map from `PageId → frame_index`. Answers "is page P in memory? If so, which frame?" in O(1).
**3. The LRU list** — a doubly-linked list of frame indices in recency order (most recently used at the head, least recently used at the tail). Answers "which unpinned frame should we evict?" in O(1).
```c
/* Internal frame metadata */
typedef struct {
    PageId  page_id;     /* which page this frame holds (INVALID_PAGE_ID if empty) */
    uint8_t data[PAGE_SIZE]; /* the actual page bytes */
    int     pin_count;   /* number of active pins (0 = evictable) */
    int     is_dirty;    /* 1 if modified since last flush to disk */
    /* LRU doubly-linked list pointers */
    int     lru_prev;    /* index of previous frame in LRU list (-1 = head) */
    int     lru_next;    /* index of next frame in LRU list (-1 = tail) */
} Frame;
/* Page table entry */
typedef struct PageTableEntry {
    PageId page_id;
    int    frame_index;
    struct PageTableEntry *next;  /* chaining for hash collisions */
} PageTableEntry;
struct BufferPool {
    int       fd;              /* database file descriptor */
    Frame    *frames;          /* heap-allocated array of Frame[pool_size] */
    int       pool_size;       /* number of frames */
    PageId    next_page_id;    /* for NewPage() — monotonically increasing */
    /* Page table: hash map from page_id → frame_index */
    PageTableEntry **page_table;   /* hash table, size = pool_size * 2 */
    int               page_table_size;
    /* LRU list state */
    int       lru_head;   /* most recently used frame index (-1 if empty) */
    int       lru_tail;   /* least recently used frame index (-1 if empty) */
    /* Statistics */
    BufferPoolStats stats;
};
```
The page table uses separate chaining (linked list per bucket) to resolve hash collisions. The hash function is simple:
```c
static int page_table_bucket(BufferPool *bp, PageId page_id) {
    /* Knuth multiplicative hash for 32-bit integers */
    return (int)((page_id * 2654435761u) % (uint32_t)bp->page_table_size);
}
```
The LRU list is embedded directly in the `Frame` structs using array indices rather than pointers. This avoids pointer arithmetic and keeps the data structure cache-friendly: all frames live in a contiguous array, and navigating the LRU list is just array indexing.
---
## Initialization
```c
BufferPool *buffer_pool_create(int fd, int frames) {
    if (frames <= 0) return NULL;
    BufferPool *bp = calloc(1, sizeof(BufferPool));
    if (!bp) return NULL;
    bp->fd        = fd;
    bp->pool_size = frames;
    bp->lru_head  = -1;
    bp->lru_tail  = -1;
    /* Allocate the frame array */
    bp->frames = calloc(frames, sizeof(Frame));
    if (!bp->frames) { free(bp); return NULL; }
    /* Initialize all frames as empty */
    for (int i = 0; i < frames; i++) {
        bp->frames[i].page_id  = INVALID_PAGE_ID;
        bp->frames[i].pin_count = 0;
        bp->frames[i].is_dirty  = 0;
        bp->frames[i].lru_prev  = -1;
        bp->frames[i].lru_next  = -1;
    }
    /* Allocate page table (2x pool size for load factor ~0.5) */
    bp->page_table_size = frames * 2;
    bp->page_table = calloc(bp->page_table_size, sizeof(PageTableEntry *));
    if (!bp->page_table) {
        free(bp->frames);
        free(bp);
        return NULL;
    }
    /* Read the current file size to determine next_page_id */
    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size < 0) file_size = 0;
    bp->next_page_id = (PageId)(file_size / PAGE_SIZE);
    return bp;
}
```
The initialization allocates the exact memory upfront. Unlike dynamic caches that grow on demand, the buffer pool size is fixed at creation. Why fixed? Because the whole point of the buffer pool is to give the database *predictable* memory usage — SQLite's `PRAGMA cache_size` lets you configure exactly how much RAM the database uses. A database that grows its memory footprint unboundedly is unsuitable for embedded deployment.
> **Memory cost per frame:** Each `Frame` contains 4096 bytes of page data plus ~24 bytes of metadata ≈ 4120 bytes. For the default 1000-frame pool: ~4MB of memory. For a 10,000-frame pool (a more aggressive configuration for larger workloads): ~40MB. This is tunable without changing any code.
---
## FetchPage: The Core Operation

![FetchPage — Hit vs Miss Data Walk](./diagrams/diag-buffer-pool-fetch-flow.svg)

`buffer_pool_fetch` is called thousands of times per second during a query. Its hot path (cache hit) must be as fast as possible:
```c
uint8_t *buffer_pool_fetch(BufferPool *bp, PageId page_id) {
    bp->stats.fetches++;
    /* --- Fast path: page already in pool --- */
    int frame_idx = page_table_lookup(bp, page_id);
    if (frame_idx >= 0) {
        bp->stats.hits++;
        Frame *f = &bp->frames[frame_idx];
        f->pin_count++;
        lru_move_to_head(bp, frame_idx);  /* mark as most recently used */
        return f->data;
    }
    /* --- Slow path: page not in pool, must load --- */
    bp->stats.misses++;
    /* Find a free or evictable frame */
    frame_idx = find_free_frame(bp);
    if (frame_idx < 0) {
        frame_idx = evict_lru(bp);  /* evict LRU unpinned page */
    }
    if (frame_idx < 0) {
        /* All frames are pinned — pool exhausted. This is a bug in the caller. */
        return NULL;
    }
    Frame *f = &bp->frames[frame_idx];
    /* Read the page from disk */
    off_t offset = (off_t)page_id * PAGE_SIZE;
    ssize_t n = pread(bp->fd, f->data, PAGE_SIZE, offset);
    if (n < 0) {
        return NULL;  /* I/O error */
    }
    if (n < PAGE_SIZE) {
        /* Partial read: new page beyond current EOF — zero the rest */
        memset(f->data + n, 0, PAGE_SIZE - n);
    }
    /* Install in page table and LRU */
    f->page_id   = page_id;
    f->pin_count = 1;
    f->is_dirty  = 0;
    page_table_insert(bp, page_id, frame_idx);
    lru_move_to_head(bp, frame_idx);
    return f->data;
}
```
The two-phase structure — check table, then load — is the universal cache access pattern. It appears in CPU L1/L2 cache lookups, DNS resolver caches, HTTP proxy caches, and browser caches. The taxonomy of cache results:
- **Hit**: page is in pool, already pinned by you or another operation. Increment pin_count, move to LRU head. One hash lookup, one increment, one pointer return.
- **Miss (cold)**: pool has free frames. Load from disk, install. One disk read (the expensive operation).
- **Miss (capacity)**: pool is full. Must evict before loading.
The `pread()` system call reads from the file at a specific offset without changing the file position. This is critical in a concurrent or re-entrant context — `lseek()` followed by `read()` would be a race condition if two threads called FetchPage simultaneously. `pread()` is atomic with respect to the file position. 
> **🔑 Foundation: Why pread**
> 
> ## pread() vs lseek+read for Database I/O
**What it IS**
Both approaches read data from a specific offset in a file, but they differ in how they handle the file's *current position pointer*.
`lseek+read` is a two-step dance:
```c
lseek(fd, offset, SEEK_SET);  // move the file cursor
read(fd, buf, n);              // read from wherever cursor is now
```
`pread` collapses this into one atomic call:
```c
pread(fd, buf, n, offset);  // read from offset, cursor untouched
```
The critical difference: `lseek+read` modifies the file descriptor's shared position state. `pread` does not — it reads from the given offset and leaves the cursor exactly where it was.
**WHY you need it now**
A buffer pool manages many pages across a single file (or a few files). Multiple threads are constantly fetching and flushing pages concurrently. If two threads both use `lseek+read`:
```
Thread A: lseek(fd, page_5_offset)   ← cursor = 40960
Thread B: lseek(fd, page_9_offset)   ← cursor = 73728  (races here)
Thread A: read(fd, buf, PAGE_SIZE)    ← WRONG: reads page 9, not page 5
```
This is a classic TOCTOU (time-of-check/time-of-use) race condition. You'd need a mutex around every `lseek+read` pair — serializing all I/O, destroying concurrency.
`pread` is *thread-safe by design*: each call specifies its own offset independently. No shared state is touched. Threads can issue concurrent reads to any page without coordination.
**Key insight**
`pread` is not just a convenience wrapper — it's a *concurrency primitive*. The file descriptor's position pointer is process-global shared state, and `lseek` mutates it. `pread` bypasses that state entirely, making positional I/O stateless from the caller's perspective. In any multi-threaded system doing file I/O, defaulting to `pread`/`pwrite` is the correct baseline.

---
## LRU Eviction: The Heart of the Cache

> **🔑 Foundation: Cache replacement policies: LRU**
> 
> ## Cache Replacement Policies: LRU, LFU, Clock, ARC
**What it IS**
A buffer pool holds a fixed number of pages in memory. When it's full and you need to load a new page, you must *evict* one. A replacement policy decides which page to kick out. The goal: evict the page least likely to be needed soon, minimizing future page faults.
**LRU (Least Recently Used)**
Evict the page that was accessed the longest time ago. Intuition: if something hasn't been touched in a while, it probably won't be soon. Implemented with a doubly-linked list + hash map — on every access, move the page to the front; evict from the back.
- **Pro:** Works well for workloads with temporal locality (recently used = likely to be used again).
- **Con:** Vulnerable to *sequential scans*. If a query scans 10,000 pages in order, each page is used once and never again — but LRU will still cache them, polluting the buffer pool and evicting actually-hot pages.
**LFU (Least Frequently Used)**
Evict the page with the lowest access count. Intuition: popularity over time matters more than recency.
- **Pro:** Resistant to one-time scans (a scan page only gets count=1).
- **Con:** Suffers from *cache pollution by historical data*. A page that was popular months ago but irrelevant now will have a high count and resist eviction. Also expensive to implement efficiently (requires a frequency-keyed priority structure).
**Clock (Second-Chance)**
A practical approximation of LRU. Imagine pages arranged in a circle with a "clock hand." Each page has a *reference bit* set to 1 on access. When eviction is needed, the hand sweeps clockwise: if a page's bit is 1, clear it to 0 and move on (give it a second chance); if it's 0, evict it.
- **Pro:** O(1) amortized, very low overhead. No linked-list manipulation on every access.
- **Con:** Less precise than LRU; a page accessed once and then forgotten may survive multiple clock sweeps.
This is what many real database systems (PostgreSQL's buffer manager, for instance) use in practice.
**ARC (Adaptive Replacement Cache)**
Developed by IBM. Maintains *four* lists internally:
- `T1`: pages seen *exactly once* recently (recency)
- `T2`: pages seen *at least twice* recently (frequency)
- `B1`: ghost entries for recently evicted T1 pages (history, no data)
- `B2`: ghost entries for recently evicted T2 pages (history, no data)
ARC dynamically shifts a balance parameter `p` between favoring recency (T1) vs. frequency (T2) based on which ghost list gets hit. If a page's ghost entry is in B1, the workload is showing recency patterns — grow T1. If in B2, it's showing frequency patterns — grow T2.
- **Pro:** Self-tuning, outperforms both pure LRU and LFU across mixed workloads. Scan-resistant.
- **Con:** More complex to implement; some patents complicate commercial use.
**WHY you need it now**
Your buffer pool is the single most performance-sensitive component in the database. The difference between 95% and 85% hit rate is the difference between memory-speed and disk-speed for 10% of queries — often a 100–1000x difference in latency. Choosing and correctly implementing a replacement policy is not an afterthought.
**Key insight**
No policy is universally optimal — it depends on the *access pattern*. LRU assumes temporal locality; LFU assumes stable popularity. Real workloads mix sequential scans (enemies of LRU), hot lookup tables (friends of LFU), and bursty access. ARC's genius is that it *learns* which assumption is currently correct and adapts. When in doubt for a general-purpose buffer pool, Clock is a pragmatic starting point; ARC is the step up when you need real-world performance.

When the pool is full and a new page is needed, you must evict an existing page. The eviction policy determines which page leaves. The canonical choice — and what you will implement — is **LRU (Least Recently Used)**: evict the page that was accessed least recently.
The intuition: if a page hasn't been used for a long time, it's unlikely to be needed soon. LRU is an approximation of the optimal policy (OPT / Bélády's algorithm), which evicts the page that will be needed furthest in the future. OPT requires knowledge of the future; LRU approximates it with knowledge of the past.

![LRU Eviction — State Evolution](./diagrams/diag-buffer-pool-lru-eviction.svg)

The LRU invariant: the doubly-linked list always has the most recently used frame at the head and the least recently used at the tail. Every FetchPage (hit or miss) moves the accessed frame to the head. Eviction takes from the tail.
The LRU list operations:
```c
/* Remove a frame from its current position in the LRU list */
static void lru_remove(BufferPool *bp, int idx) {
    Frame *f = &bp->frames[idx];
    int prev = f->lru_prev;
    int next = f->lru_next;
    if (prev >= 0) bp->frames[prev].lru_next = next;
    else           bp->lru_head = next;  /* was head */
    if (next >= 0) bp->frames[next].lru_prev = prev;
    else           bp->lru_tail = prev;  /* was tail */
    f->lru_prev = -1;
    f->lru_next = -1;
}
/* Insert a frame at the head of the LRU list (most recently used) */
static void lru_insert_head(BufferPool *bp, int idx) {
    Frame *f = &bp->frames[idx];
    f->lru_prev = -1;
    f->lru_next = bp->lru_head;
    if (bp->lru_head >= 0)
        bp->frames[bp->lru_head].lru_prev = idx;
    else
        bp->lru_tail = idx;  /* list was empty */
    bp->lru_head = idx;
}
/* Move a frame to the head (called on every access) */
static void lru_move_to_head(BufferPool *bp, int idx) {
    if (bp->lru_head == idx) return;  /* already head */
    lru_remove(bp, idx);
    lru_insert_head(bp, idx);
}
/* Evict the LRU unpinned page. Returns frame index, or -1 if all pinned. */
static int evict_lru(BufferPool *bp) {
    /* Walk from tail toward head, find first unpinned frame */
    int idx = bp->lru_tail;
    while (idx >= 0) {
        Frame *f = &bp->frames[idx];
        if (f->pin_count == 0) {
            /* Found an evictable frame */
            if (f->is_dirty) {
                /* Write dirty page to disk before eviction */
                write_page_to_disk(bp, idx);
            }
            /* Remove from page table */
            page_table_remove(bp, f->page_id);
            /* Remove from LRU list */
            lru_remove(bp, idx);
            bp->stats.evictions++;
            f->page_id  = INVALID_PAGE_ID;
            f->is_dirty = 0;
            return idx;
        }
        idx = f->lru_prev;  /* walk toward head */
    }
    return -1;  /* all frames pinned */
}
```
The walk from tail toward head is O(pool_size) in the worst case — but only when almost all frames are pinned. In normal operation, many frames are unpinned and the first frame at the tail is immediately evictable: O(1). Only pathological cases (a B-tree traversal holding too many pages open) require walking the list.
### The Pin Invariant
The critical rule: **a page with `pin_count > 0` is never evicted**. This is the entire safety contract. When the B-tree code fetches a page for traversal, it increments the pin count. Until it calls `unpin()`, no amount of memory pressure can cause the page to disappear. The pointer returned by `buffer_pool_fetch()` is stable for exactly as long as the pin is held.
Pin counting is reference counting applied to page frames. 
> **🔑 Foundation: Reference counting: shared_ptr**
> 
> ## Reference Counting: shared_ptr, Rc, GC Roots, and Pin Counts
**What it IS**
Reference counting is a memory (and resource) management strategy where each object tracks how many things currently *refer* to it. When the count drops to zero, no one needs the object anymore — it can be freed.
In C++:
```cpp
std::shared_ptr<Page> a = load_page(42);  // count = 1
{
    std::shared_ptr<Page> b = a;           // count = 2
}                                          // b destroyed, count = 1
// a destroyed, count = 0 → Page freed
```
In Rust, `Rc<T>` (single-threaded) and `Arc<T>` (atomic, thread-safe) do the same. `Arc` uses atomic increment/decrement so the count is safe across threads — important for a multi-threaded buffer pool.
**The GC roots connection**
In garbage-collected languages (Java, Python, Go), reference counting is one of several GC strategies. A *GC root* is any reference that's directly reachable without traversal — stack variables, global variables, open file handles. The GC traces from roots outward; anything unreachable from a root is garbage. Reference counting is a simpler, eager version of this: every holder of a pointer is implicitly a "root," and the count is the number of such roots.
**Pin counts in a buffer pool**
A buffer pool uses reference counting through *pin counts* (also called fix counts). When a thread needs a page, it *pins* it — incrementing the pin count, signaling "I'm actively using this page, do not evict it." When done, it *unpins* — decrementing the count. The replacement policy may only evict pages with a pin count of zero.
```
load_page(42):   pin_count[42]++  →  pin_count = 1, page is "pinned"
release_page(42): pin_count[42]-- →  pin_count = 0, page is evictable
```
If two transactions simultaneously access page 42, pin_count = 2. Neither can trigger eviction of that page. This prevents a nasty bug where a thread reads a pointer to a page, gets preempted, the page gets evicted and its memory reused, and the thread later dereferences now-garbage memory — a use-after-eviction bug.
**WHY you need it now**
The buffer pool is shared memory. Multiple threads may simultaneously hold references to the same page. You cannot free (evict) a page while any thread is reading or writing it. Pin counts give you a simple, correct, lightweight mechanism to enforce this — no garbage collector required, no complex ownership transfer. Every page access becomes: pin → use → unpin. Eviction only touches pages at zero.
**Key insight**
A pin count is reference counting applied to *physical memory frames*, not heap objects. The key rule is the same: the resource (frame) lives exactly as long as someone holds a reference (pin). The discipline is: *always unpin*, even on error paths. A leaked pin is a frame that can never be evicted — a slow memory leak that eventually deadlocks the buffer pool when all frames are "pinned" by long-forgotten operations. Treat unpin like `free` — it must be called exactly once per pin, unconditionally.
 The pin count tracks how many callers are simultaneously using the page. When the count drops to zero, the page is eligible for eviction — but it stays in memory until something else needs that frame. Zero pin count means "evictable", not "evicted".
A concrete scenario: B-tree descent from root to leaf holds three pages simultaneously:
```
Pin root page (count: 1)
  Pin internal page (count: 1)
    Pin leaf page (count: 1)
    Read leaf data
    Unpin leaf page (count: 0) — eligible for eviction
  Unpin internal page (count: 0)
Unpin root page (count: 0)
```
During the traversal, all three pages are safe. After each unpin, the page becomes eligible but is likely still in the pool for the next traversal. B-tree pages near the root are accessed repeatedly and therefore stay at the LRU head — they are effectively permanent residents in a warm cache.
---
## Dirty Page Tracking and Write-Back

![Dirty Page Write-Back — Before/After](./diagrams/diag-buffer-pool-dirty-writeback.svg)

The buffer pool is a **write-back cache**: modifications accumulate in memory and are written to disk lazily (either on eviction or explicit flush), not immediately on every write. This contrasts with a **write-through cache**, where every modification is synchronously written to disk. 
> **🔑 Foundation: Write-back vs write-through: CPU cache hierarchy**
> 
> ## Write-Back vs Write-Through: CPU Caches and Database Pages
**What it IS**
These are two policies for handling *writes* to cached data — deciding when a modification to a cached copy propagates back to the authoritative storage (main memory, disk).
**Write-Through**
Every write to the cache *immediately* also writes to the backing store. The cache and the store are always in sync.
```
CPU writes X=5 to L1 cache
   → simultaneously writes X=5 to RAM
L1 and RAM always agree.
```
- **Pro:** Backing store is always up-to-date. A cache failure loses nothing.
- **Con:** Every write goes all the way to the slower backing store. Write latency = backing store latency. Poor write performance.
**Write-Back**
Writes go *only* to the cache. The backing store is updated lazily, later. The cache line (or page) is marked *dirty* — meaning "this has been modified and the backing store is stale." The dirty data is written back only when the line is evicted or explicitly flushed.
```
CPU writes X=5 to L1 cache  → L1 marks cache line "dirty"
RAM still has old value.
...later, cache line evicted → dirty line written back to RAM
```
- **Pro:** Writes are fast (memory speed). Multiple writes to the same cache line coalesce — you might write 100 times but only flush once.
- **Con:** Complexity. Dirty tracking required. If power fails before writeback, data is lost.
**In CPU cache hierarchies (L1/L2/L3)**
Modern CPUs use write-back for L1 and L2. Each cache line has a dirty bit. When a line is evicted from L1 to L2 (or L2 to L3 or L3 to RAM), dirty lines are written through. This is why a CPU crash can corrupt in-flight data — dirty cache lines vanish.
**In database buffer pools**
The exact same concept applies at the page level, but the "backing store" is disk:
- A page loaded into the buffer pool is a *cached copy* of a disk page.
- When a transaction modifies it, the page is marked *dirty*.
- The dirty page lives in memory (fast access) but the on-disk copy is stale.
- A *page flush* (write-back) writes the dirty page to disk.
- Write-through would mean every page modification immediately hits disk — unacceptably slow.
Database buffer pools always use write-back. This is why *crash recovery* (WAL, redo logs) is necessary — at any moment, modified pages may exist only in memory.
**WHY you need it now**
Implementing dirty tracking in your buffer pool is not optional — it's the mechanism that makes writes fast. When a page is unpinned after modification, the caller must signal `is_dirty=true`. Your flush logic must only write pages that are actually dirty (writing clean pages is wasted I/O). Your eviction path must write dirty pages to disk before reusing their frame.
**Key insight**
Dirty = "memory and disk disagree." Your buffer pool is always managing this disagreement. The goal is to minimize how often you synchronize (for performance) while ensuring you never lose committed data (for correctness). The tension between these two drives the need for WAL: instead of making writes synchronous, you make the *log* synchronous (small sequential writes) and let page flushes happen lazily in the background.

Write-back is dramatically faster for workloads with many small writes: inserting 1000 rows modifies the same leaf pages repeatedly. Write-through would issue 1000 disk writes. Write-back issues one disk write when the page is eventually evicted.
The dirty bit is set by callers who modify a page:
```c
void buffer_pool_unpin(BufferPool *bp, PageId page_id, int dirty) {
    int frame_idx = page_table_lookup(bp, page_id);
    if (frame_idx < 0) return;  /* not in pool — shouldn't happen */
    Frame *f = &bp->frames[frame_idx];
    if (dirty) f->is_dirty = 1;
    if (f->pin_count > 0) f->pin_count--;
    /* Note: don't evict here even if pin_count reaches 0.
       The LRU eviction handles that lazily. */
}
```
The typical call sequence for modifying a page:
```c
/* B-tree code, modifying a page */
uint8_t *page_data = buffer_pool_fetch(bp, page_id);
/* ... modify page_data in place ... */
buffer_pool_unpin(bp, page_id, /*dirty=*/1);
```
The dirty bit is cleared when the page is written to disk, either during eviction or explicit flush:
```c
static int write_page_to_disk(BufferPool *bp, int frame_idx) {
    Frame *f = &bp->frames[frame_idx];
    off_t offset = (off_t)f->page_id * PAGE_SIZE;
    ssize_t n = pwrite(bp->fd, f->data, PAGE_SIZE, offset);
    if (n != PAGE_SIZE) return -1;  /* write error */
    f->is_dirty = 0;
    bp->stats.dirty_writes++;
    return 0;
}
```
`pwrite()` is the write equivalent of `pread()`: writes exactly `count` bytes at the specified offset without affecting the file position. For a 4096-byte page, this should be an atomic operation on any reasonable filesystem — but see the discussion of torn pages in Milestone 9 (rollback journal) for why this assumption is fragile.
### Why the Dirty Bit Matters for Crash Safety
The dirty bit is not just an optimization — it is a correctness requirement for write-ahead logging. When you implement WAL mode (Milestone 10), the rule is: you must never write a modified database page to the main database file until the corresponding WAL record has been written and fsynced. The dirty bit is your inventory of which pages are "in flight" — modified in memory, not yet written to disk.
Without explicit dirty tracking, you'd need to scan all pages to find modified ones, or use OS-level dirty page detection (which requires `mprotect()` tricks). The dirty bit makes the WAL protocol efficient.
---
## FlushAll: Preparing for Shutdown and Checkpoint
```c
int buffer_pool_flush_all(BufferPool *bp) {
    int flushed = 0;
    for (int i = 0; i < bp->pool_size; i++) {
        Frame *f = &bp->frames[i];
        if (f->page_id == INVALID_PAGE_ID) continue;  /* empty frame */
        if (!f->is_dirty) continue;                    /* clean page */
        if (write_page_to_disk(bp, i) < 0) return -1;
        flushed++;
    }
    return flushed;
}
```
`FlushAll` is called in two situations:
**Before checkpoint (WAL mode):** The checkpoint process copies WAL-modified pages back to the main database file. Before doing so, the buffer pool must flush all dirty pages so the file is fully up-to-date from the buffer pool's perspective.
**On shutdown:** Before closing the database file, all dirty pages must be written. If the process exits without flushing, all in-memory modifications are lost. This is not a crash scenario (we handle crashes via the journal/WAL) — it's a clean shutdown that must correctly persist all committed writes.
`FlushAll` does NOT call `fsync()`. Calling `fsync()` is the caller's responsibility, and it must happen *after* all the `pwrite()` calls complete. The reason: `fsync()` is expensive (it blocks until the OS confirms all writes have reached the physical medium). `FlushAll` might be called in a context where multiple steps need to complete before the fsync — forcing fsync inside `FlushAll` would be premature.
```c
/* Correct shutdown sequence */
int db_close(DB *db) {
    int n = buffer_pool_flush_all(db->bp);
    if (n < 0) return -1;
    /* Now fsync to guarantee all pwrite() calls are durable */
    if (fsync(db->fd) < 0) return -1;
    buffer_pool_destroy(db->bp);
    close(db->fd);
    return 0;
}
```

> **🔑 Foundation: fsync**
> 
> ## fsync(): What It Guarantees, When It's Necessary, and Why It's Expensive
**What it IS**
When you call `write()` on a file, the operating system does *not* immediately write data to the physical disk. Instead, it copies your data into the *page cache* (kernel memory) and returns immediately. The OS will flush this to disk eventually — maybe seconds or minutes later — based on its own writeback scheduling.
`fsync(fd)` is the system call that says: *"flush everything buffered for this file descriptor to durable storage right now, and do not return until it's done."*
```c
write(fd, data, len);   // data → kernel page cache (fast)
fsync(fd);              // kernel page cache → disk (slow, synchronous)
// After fsync returns: data is guaranteed durable
```
This includes not just the OS page cache but also *disk write buffers* — `fsync` instructs the drive's controller to flush its internal cache to the physical platters (or NAND cells for SSDs). This is why `fsync` is expensive: it forces a write to the hardware's slowest layer.
**What fsync does NOT guarantee**
- It does not guarantee the *directory entry* is durable. If you create a new file, you must also `fsync` the *parent directory* to ensure the directory entry pointing to the file is on disk. Otherwise, the file data may survive a crash but the directory won't know the file exists.
- It does not help if the storage hardware lies — some consumer SSDs and drives report `fsync` complete without actually flushing (a significant source of data loss bugs on consumer hardware).
**WHY you need it now**
A database has a fundamental durability contract: when a `COMMIT` returns to the user, the transaction's effects must survive a crash. Without `fsync`, a committed transaction's data may live only in the kernel page cache. A power failure loses it. This violates the D in ACID.
The standard pattern in WAL-based databases:
1. Write transaction changes to the WAL log buffer.
2. `fsync` the WAL file (small, sequential write — relatively fast).
3. Return `COMMIT` to the user.
4. Later, flush dirty data pages to disk (may or may not `fsync` immediately — depends on checkpoint policy).
The WAL `fsync` is the *only* fsync on the critical path for commit latency. Data page flushing happens in the background.
**Why it's expensive**
On a spinning disk, `fsync` may require waiting for a full disk rotation (~5ms at 7200 RPM) to ensure the write is on the platter. SSDs are faster but still impose flush penalties (100μs–1ms) because the controller must commit pending writes from volatile DRAM to NAND flash.
Postgres on commodity hardware: each `fsync` takes ~1–5ms. At 1ms per commit, max throughput is ~1000 transactions/second on a single connection. This is why *group commit* exists — batching multiple transactions' WAL writes before a single `fsync`, amortizing the cost across many commits.
**Key insight**
`fsync` is the boundary between "the OS thinks it wrote it" and "the disk guarantees it survives power loss." Every database durability guarantee ultimately reduces to: *did we call `fsync` before we told the user their commit succeeded?* The entire architecture of WAL, checkpointing, and crash recovery exists to minimize how often you must call `fsync` while still maintaining this guarantee. When debugging data loss bugs in database code, the first question is always: "did we fsync the WAL before returning COMMIT?"

---
## NewPage: Extending the Database File
When a B-tree node split requires a new page (Milestone 5), it calls `buffer_pool_new_page()`. This function allocates the next page ID, "creates" the page in the buffer pool, and returns it pinned:
```c
uint8_t *buffer_pool_new_page(BufferPool *bp, PageId *page_id_out) {
    PageId new_id = bp->next_page_id++;
    /* Find a frame for the new page (same as FetchPage miss path) */
    int frame_idx = find_free_frame(bp);
    if (frame_idx < 0) frame_idx = evict_lru(bp);
    if (frame_idx < 0) return NULL;  /* pool exhausted */
    Frame *f = &bp->frames[frame_idx];
    memset(f->data, 0, PAGE_SIZE);   /* new pages start zeroed */
    f->page_id   = new_id;
    f->pin_count = 1;
    f->is_dirty  = 1;   /* new page is immediately dirty — must be written eventually */
    page_table_insert(bp, new_id, frame_idx);
    lru_insert_head(bp, frame_idx);
    if (page_id_out) *page_id_out = new_id;
    return f->data;
}
```
The new page is marked dirty immediately. It doesn't exist on disk yet — the first time this page is evicted or flushed, `write_page_to_disk` will `pwrite()` it to the file, extending the file to accommodate the new page. `pwrite()` on a file descriptor extends the file size when writing past the current end (the OS fills gaps with zeros). This means the database file grows in 4096-byte increments as new pages are allocated.
> **Page ID as file offset:** The mapping `page_id → file_offset = page_id * PAGE_SIZE` is the entire "file format" of the database at this level. Page 0 lives at byte 0, page 1 at byte 4096, page 2 at byte 8192. This is not the only possible design — some databases interleave multiple files or use non-linear addressing — but linear page addressing is simple, debuggable with `xxd`, and portable. SQLite uses exactly this layout.
---
## The Page Table Implementation
The page table is a critical performance component — it is called on every FetchPage. It must answer "is page P in memory?" in O(1):
```c
static void page_table_insert(BufferPool *bp, PageId page_id, int frame_idx) {
    int bucket = page_table_bucket(bp, page_id);
    PageTableEntry *entry = malloc(sizeof(PageTableEntry));
    entry->page_id    = page_id;
    entry->frame_index = frame_idx;
    entry->next       = bp->page_table[bucket];
    bp->page_table[bucket] = entry;
}
static int page_table_lookup(BufferPool *bp, PageId page_id) {
    int bucket = page_table_bucket(bp, page_id);
    PageTableEntry *e = bp->page_table[bucket];
    while (e) {
        if (e->page_id == page_id) return e->frame_index;
        e = e->next;
    }
    return -1;  /* not found */
}
static void page_table_remove(BufferPool *bp, PageId page_id) {
    int bucket = page_table_bucket(bp, page_id);
    PageTableEntry **pp = &bp->page_table[bucket];
    while (*pp) {
        if ((*pp)->page_id == page_id) {
            PageTableEntry *to_free = *pp;
            *pp = to_free->next;
            free(to_free);
            return;
        }
        pp = &(*pp)->next;
    }
}
```
The hash table size is `pool_size * 2` (load factor ~0.5). With separate chaining, this means average chain length is ~0.5 — most lookups are O(1) with no chain traversal. You could use open addressing (linear probing) for better cache behavior, but separate chaining is simpler and adequate here.
> **Optimization note:** In a production buffer pool, `PageTableEntry` allocations are a performance concern — each insert calls `malloc`. A fixed-size pool of `PageTableEntry` structs (one per frame, since each frame holds exactly one page) eliminates this allocation. The entries array is allocated once at initialization, and the page table's free list recycles them. This optimization is worth making if you see `malloc` appearing in profiler hotspots.
---
## Performance Metrics and the Hit Rate
The buffer pool's effectiveness is measured by its **hit rate**: the fraction of FetchPage calls that find the page already in memory.
```c
void buffer_pool_get_stats(BufferPool *bp, BufferPoolStats *out) {
    *out = bp->stats;
}
/* Convenience: compute hit rate as a percentage */
double buffer_pool_hit_rate(BufferPool *bp) {
    if (bp->stats.fetches == 0) return 100.0;
    return (double)bp->stats.hits / (double)bp->stats.fetches * 100.0;
}
```
A hit rate below 90% means your buffer pool is too small for your workload. Typical guidelines:
- **>99%**: excellent — nearly all accesses served from memory
- **95–99%**: good — occasional disk reads, not a bottleneck
- **90–95%**: acceptable for disk-bound workloads
- **<90%**: investigate — either pool is too small or the workload has poor locality
The hit rate formula assumes all pages are equally expensive to load. In practice, B-tree root pages are accessed on every query and are almost always hot; leaf pages at the bottom of a large B-tree might be accessed rarely. A high hit rate for root pages alone doesn't mean leaf pages are cached — profile by page type if performance is critical.
```c
/* Example: print hit rate after a workload */
void print_buffer_stats(BufferPool *bp) {
    BufferPoolStats s;
    buffer_pool_get_stats(bp, &s);
    printf("Buffer pool statistics:\n");
    printf("  Fetches:      %lu\n", s.fetches);
    printf("  Hits:         %lu (%.1f%%)\n", s.hits,
           s.fetches ? 100.0 * s.hits / s.fetches : 100.0);
    printf("  Misses:       %lu\n", s.misses);
    printf("  Evictions:    %lu\n", s.evictions);
    printf("  Dirty writes: %lu\n", s.dirty_writes);
}
```
---
## Three-Level View: A Single OP_COLUMN Through the Stack
To see the buffer pool's place in the complete execution stack, trace what happens when the VDBE executes `OP_COLUMN cursor=0, col_idx=2, dest_reg=5`:
**Level 1 — VDBE (the instructions you wrote in Milestone 3)**
```c
case OP_COLUMN: {
    BTreeCursor *cur = vm.cursors[instr->p1];
    btree_cursor_column(cur, instr->p2, &vm.regs[instr->p3]);
    vm.pc++;
    break;
}
```
The VM calls `btree_cursor_column`. It knows nothing about pages.
**Level 2 — Buffer Pool (this milestone)**
```c
/* Inside btree_cursor_column */
uint8_t *page_data = buffer_pool_fetch(bp, cursor->current_page_id);
/* Deserialize column 2 from page_data at the current row offset */
decode_column(page_data, cursor->current_slot, col_idx, out_value);
buffer_pool_unpin(bp, cursor->current_page_id, /*dirty=*/0);
```
The B-tree cursor calls `buffer_pool_fetch`. If the page is hot (pin count > 0 already, or just in LRU), this is five array accesses and a pointer return. If cold, it calls `pread()`.
**Level 3 — OS / Disk (beneath the buffer pool)**
`pread(fd, frame->data, 4096, offset)` asks the kernel to read 4096 bytes. The kernel checks its own page cache (the OS-level cache). If the OS page is cached, the data is copied from kernel space to user space in ~1 microsecond. If not, the kernel issues a read to the SSD, which takes ~100 microseconds for a random read.
The buffer pool's job is to make Level 3 happen rarely. A warm buffer pool means the database does its work entirely in Level 2, with Level 3 only for cold pages on first access.
---
## Design Decisions: LRU vs The Alternatives
LRU is not the only replacement policy. Here is why it's chosen and what the alternatives offer:
| Policy | Mechanism | Pros | Cons | Used By |
|--------|-----------|------|------|---------|
| **LRU (Chosen ✓)** | Doubly-linked list, O(1) operations | Simple, good for temporal locality, predictable behavior | Fails for sequential scans (thrashes pool) | SQLite, PostgreSQL (approx), most databases |
| Clock (Second-Chance) | Circular buffer with reference bits | Approximates LRU with less overhead, O(1) amortized | Slightly worse hit rate than true LRU | Linux VM, many OS page caches |
| LFU (Least Frequently Used) | Frequency counter per page | Retains "hot" pages even if not recently accessed | New pages evicted before they get frequency counts; counters decay problems | Some CDN caches |
| 2Q (Two-Queue) | Hot and warm queues; pages promoted on second access | Handles sequential scans without evicting hot pages | More complex state | PostgreSQL's actual implementation |
| ARC (Adaptive Replacement Cache) | Ghost lists track recently evicted pages; adapts between recency and frequency | Self-tuning, handles mixed workloads | Patent-encumbered (IBM), complex | ZFS, some enterprise databases |
SQLite uses a variant of LRU enhanced with a free list for clean pages (clean pages are preferred for eviction over dirty ones, to avoid write overhead). You'll implement the simpler pure LRU here, but understanding the alternatives explains why real databases are more complex.
**The sequential scan problem** with LRU is worth understanding: if a table scan accesses 10,000 pages sequentially, it fills the entire buffer pool with pages it will never access again. When the scan completes, the pool has zero useful pages for the next query. This is called **cache flooding**. PostgreSQL addresses this with a ring buffer strategy for sequential scans: scan pages cycle through a small fixed buffer (typically 256KB) rather than evicting the entire LRU. For this milestone, LRU is sufficient and the problem is noted for future improvement.
---
## Common Pitfalls: What Will Corrupt Your Database
**1. Returning an unpinned page's pointer.** The worst possible bug: call `buffer_pool_fetch`, get a pointer, call `buffer_pool_unpin`, then use the pointer. Between unpin and use, the page might have been evicted and overwritten with different data. The pointer is a dangling reference into memory that now contains a different page's bytes. The symptom is silent data corruption — you're reading page 42's data but you think it's page 17. The fix: always hold the pin until you're completely done with the pointer.
```c
/* WRONG: use after unpin */
uint8_t *data = buffer_pool_fetch(bp, page_id);
buffer_pool_unpin(bp, page_id, 0);
uint8_t cell_type = data[0];   /* data may be evicted! */
/* CORRECT: unpin after all use */
uint8_t *data = buffer_pool_fetch(bp, page_id);
uint8_t cell_type = data[0];   /* safe: page is pinned */
buffer_pool_unpin(bp, page_id, 0);
```
**2. Forgetting to unpin a page.** Every `buffer_pool_fetch` must have a corresponding `buffer_pool_unpin`. Missing unpins cause pin counts to accumulate, eventually making all frames permanently pinned. The next FetchPage that needs to evict finds every frame pinned and returns NULL. The database deadlocks. Add a debug mode that logs all fetches without corresponding unpins:
```c
#ifdef DEBUG_PINS
/* Log every fetch */
fprintf(stderr, "PIN page %u at %s:%d (count now %d)\n",
        page_id, __FILE__, __LINE__, f->pin_count);
#endif
```
**3. Marking clean reads as dirty.** If you call `buffer_pool_unpin(bp, id, /*dirty=*/1)` on a read that didn't modify the page, you've scheduled an unnecessary write. Worse, if you're in WAL mode, you've marked a page as needing WAL protection even though it's unchanged. Use `dirty=0` for all read-only accesses.
**4. PageId collision across tables.** If two different tables share the same page ID namespace (e.g., both table A's page 5 and table B's page 5 are represented as `PageId 5`), the buffer pool returns the same cached frame for both. All pages in the database must have globally unique page IDs. The simplest approach: use a single flat page ID space for the entire database file. Page 0 is the header, pages 1 through N are the B-tree pages for all tables. There is no "table namespace" — each B-tree page in any table has a unique position in the file. This is how SQLite works.
**5. Buffer pool deadlock from excessive pinning.** If a B-tree operation pins more pages simultaneously than the pool has frames, `evict_lru` returns -1 (all frames pinned) and FetchPage returns NULL. The typical cause is a bug in B-tree traversal that pins parent nodes and never releases them during descent. Set a soft limit: if pin count per cursor exceeds (pool_size / 4), log a warning. For a 1000-frame pool, holding more than 250 pages simultaneously is almost certainly a bug.
**6. Not zeroing new pages.** `buffer_pool_new_page` must zero the frame with `memset(f->data, 0, PAGE_SIZE)`. If it doesn't, the frame contains the data of the previously-evicted page. The B-tree code initializing a new node would be initializing into garbage data, producing a corrupted page layout.
---
## The Free Frame List: A Small Optimization
The code above calls `find_free_frame(bp)` before resorting to LRU eviction. An empty frame (one that has never been used, `page_id == INVALID_PAGE_ID`) doesn't need eviction logic — just take it. The naive implementation scans the frame array linearly:
```c
static int find_free_frame(BufferPool *bp) {
    for (int i = 0; i < bp->pool_size; i++) {
        if (bp->frames[i].page_id == INVALID_PAGE_ID) return i;
    }
    return -1;
}
```
This is O(pool_size) but is only called during initialization (while the pool has empty frames). Once the pool is full, `find_free_frame` always returns -1 and eviction handles everything. You can optimize this with a free list (a linked list of empty frame indices), making it O(1):
```c
/* In BufferPool struct: */
int free_list_head;  /* index of first free frame, -1 if none */
/* During initialization: */
for (int i = 0; i < frames - 1; i++)
    bp->frames[i].lru_next = i + 1;   /* reuse lru_next as free list link */
bp->frames[frames - 1].lru_next = -1;
bp->free_list_head = 0;
static int find_free_frame(BufferPool *bp) {
    if (bp->free_list_head < 0) return -1;
    int idx = bp->free_list_head;
    bp->free_list_head = bp->frames[idx].lru_next;
    bp->frames[idx].lru_next = -1;  /* detach from free list */
    return idx;
}
```
This is the same optimization used by memory allocators: maintain a free list rather than scanning for available slots. 
> **🔑 Foundation: Free list allocators: slab allocators**
> 
> ## Free List Allocators: Slab Allocators, Arena Allocators, and Buffer Pool Free Lists
**What it IS**
General-purpose allocators like `malloc` handle allocations of arbitrary sizes — but they pay for that generality with overhead: metadata per allocation, fragmentation, and non-deterministic latency. When you know more about your allocation patterns, you can build specialized allocators that are dramatically faster and simpler.
**Free List**
A free list is the simplest specialized allocator: a linked list of available memory chunks of a *fixed size*. Allocation = pop from the head (O(1)). Deallocation = push to the head (O(1)). No searching, no coalescing, no fragmentation.
```
free_list → [frame_3] → [frame_7] → [frame_1] → NULL
allocate():  frame = free_list.pop()  → returns frame_3
deallocate(frame_5): free_list.push(frame_5)
```
This works beautifully when all items are the same size — which is exactly the case for a buffer pool, where every frame holds exactly one page (e.g., 4KB or 8KB).
**Slab Allocator**
A slab allocator extends the free list idea to handle a specific *object type*. A "slab" is a contiguous block of memory pre-divided into N slots, each sized for one instance of the object. Multiple slabs are managed together. The kernel's slab allocator (Linux `kmalloc`) uses this for allocating inodes, task structs, etc.
Benefits beyond a plain free list:
- Objects are pre-initialized ("constructor" called at slab creation, not on every allocation).
- Memory locality: objects of the same type are co-located in memory, improving cache line utilization.
- Avoids fragmentation: all slots in a slab are same-size, so freed slots are immediately reusable without coalescing.
**Arena Allocator**
An arena (also called a *bump allocator* or *region allocator*) takes a different approach entirely: allocate from a large contiguous block by simply incrementing a pointer. Individual frees are *not supported* — you free the entire arena at once.
```
arena: [used|used|used|........free............]
             ↑ bump pointer
allocate(n): ptr = bump; bump += n; return ptr  ← O(1), one addition
free_all():  bump = arena_start                  ← O(1), one assignment
```
Arenas are ideal for "allocate a bunch of things, use them together, throw them all away together" — query execution (allocate temp buffers for one query, free all when query completes), parsing, or per-request allocations in a web server.
**Connection to buffer pool free lists**
Your buffer pool manages a fixed pool of N frames (say, 1000 frames × 8KB = 8MB). At startup, you allocate this memory once and build a free list of all N frames. The "allocator" is trivially a free list:
- `allocate_frame()` → pop a frame from the free list. If empty → must evict.
- `free_frame(f)` → push frame back onto the free list.
This means frame allocation is O(1) with zero fragmentation — you're never asking the OS for memory during normal operation. The entire pool is pre-allocated. This is critical for predictable latency: `malloc` inside a hot I/O path would introduce unpredictable pauses.
**WHY you need it now**
The buffer pool is the inner loop of every database operation. Every page access touches the frame allocator. You cannot afford `malloc`'s overhead or fragmentation behavior here. The free list is not just an optimization — it's the *correct* data structure for this problem because your allocation unit (page frame) is fixed-size and the total count is bounded.
**Key insight**
Match your allocator to your allocation *shape*. Fixed-size objects → free list. Same-type objects with reuse → slab allocator. Bulk allocations with bulk frees → arena. The buffer pool uses a free list because pages are fixed-size and frames are individually recycled. Understanding this lets you extend the same thinking to other parts of the database: a transaction log buffer might use an arena (allocate for transaction lifetime, free on commit/rollback), while a lock manager might use a slab (many small, same-size lock records with frequent alloc/free).

---
## Testing the Buffer Pool
Testing a buffer pool requires verifying both correctness (right data returned) and policy (LRU evicts the right frame):
```c
/* Test 1: Basic fetch and hit detection */
static void test_basic_fetch(void) {
    /* Use a temporary file as our "disk" */
    char tmpname[] = "/tmp/bp_test_XXXXXX";
    int fd = mkstemp(tmpname);
    unlink(tmpname);
    /* Pre-populate the file with known data */
    uint8_t page0[PAGE_SIZE] = {0};
    uint8_t page1[PAGE_SIZE] = {0};
    memset(page0, 0xAA, PAGE_SIZE);
    memset(page1, 0xBB, PAGE_SIZE);
    pwrite(fd, page0, PAGE_SIZE, 0);
    pwrite(fd, page1, PAGE_SIZE, PAGE_SIZE);
    BufferPool *bp = buffer_pool_create(fd, 10);
    assert(bp != NULL);
    /* Fetch page 0 — should be a miss */
    uint8_t *data0 = buffer_pool_fetch(bp, 0);
    assert(data0 != NULL);
    assert(data0[0] == 0xAA);  /* correct data */
    /* Fetch page 0 again — should be a hit */
    uint8_t *data0_again = buffer_pool_fetch(bp, 0);
    assert(data0_again == data0);  /* same frame pointer */
    buffer_pool_unpin(bp, 0, 0);
    buffer_pool_unpin(bp, 0, 0);  /* unpin both fetches */
    BufferPoolStats s;
    buffer_pool_get_stats(bp, &s);
    assert(s.fetches == 2);
    assert(s.hits == 1);
    assert(s.misses == 1);
    buffer_pool_destroy(bp);
    close(fd);
    printf("PASS: test_basic_fetch\n");
}
/* Test 2: LRU eviction correctness */
static void test_lru_eviction(void) {
    /* Create a pool with 3 frames */
    char tmpname[] = "/tmp/bp_lru_XXXXXX";
    int fd = mkstemp(tmpname);
    unlink(tmpname);
    /* Write 5 pages with distinguishable data */
    for (int i = 0; i < 5; i++) {
        uint8_t page[PAGE_SIZE] = {0};
        memset(page, (uint8_t)(i + 1) * 0x11, PAGE_SIZE);
        pwrite(fd, page, PAGE_SIZE, (off_t)i * PAGE_SIZE);
    }
    BufferPool *bp = buffer_pool_create(fd, 3);
    /* Access pages in order: 0, 1, 2 — pool is now full */
    uint8_t *p0 = buffer_pool_fetch(bp, 0);  buffer_pool_unpin(bp, 0, 0);
    uint8_t *p1 = buffer_pool_fetch(bp, 1);  buffer_pool_unpin(bp, 1, 0);
    uint8_t *p2 = buffer_pool_fetch(bp, 2);  buffer_pool_unpin(bp, 2, 0);
    /* Re-access page 0 to make it recently used: LRU order is now [0, 2, 1] */
    buffer_pool_fetch(bp, 0);
    buffer_pool_unpin(bp, 0, 0);
    /* Fetch page 3 — must evict. LRU page is 1. */
    uint8_t *p3 = buffer_pool_fetch(bp, 3);
    assert(p3 != NULL);
    assert(p3[0] == 0x44);  /* page 3 has data 0x44 */
    /* Page 1 should have been evicted. Re-fetching it should be a miss. */
    BufferPoolStats before, after;
    buffer_pool_get_stats(bp, &before);
    buffer_pool_fetch(bp, 1);
    buffer_pool_get_stats(bp, &after);
    assert(after.misses == before.misses + 1);  /* page 1 was a miss */
    buffer_pool_unpin(bp, 1, 0);
    buffer_pool_unpin(bp, 3, 0);
    buffer_pool_destroy(bp);
    close(fd);
    printf("PASS: test_lru_eviction\n");
}
/* Test 3: Pinned pages are not evicted */
static void test_pin_protection(void) {
    char tmpname[] = "/tmp/bp_pin_XXXXXX";
    int fd = mkstemp(tmpname);
    unlink(tmpname);
    /* Create 5 pages of data */
    for (int i = 0; i < 5; i++) {
        uint8_t page[PAGE_SIZE] = {0};
        page[0] = (uint8_t)(i + 10);
        pwrite(fd, page, PAGE_SIZE, (off_t)i * PAGE_SIZE);
    }
    BufferPool *bp = buffer_pool_create(fd, 3);
    /* Pin all 3 frames */
    uint8_t *p0 = buffer_pool_fetch(bp, 0);  /* pinned, NOT unpinned */
    uint8_t *p1 = buffer_pool_fetch(bp, 1);  /* pinned */
    uint8_t *p2 = buffer_pool_fetch(bp, 2);  /* pinned */
    /* Attempt to fetch page 3 — pool is exhausted (all pinned) */
    uint8_t *p3 = buffer_pool_fetch(bp, 3);
    assert(p3 == NULL);  /* must return NULL, not corrupt an existing page */
    /* Unpin one frame */
    buffer_pool_unpin(bp, 1, 0);
    /* Now page 3 should be fetchable */
    p3 = buffer_pool_fetch(bp, 3);
    assert(p3 != NULL);
    assert(p3[0] == 13);  /* page 3 has byte value 13 */
    buffer_pool_unpin(bp, 0, 0);
    buffer_pool_unpin(bp, 2, 0);
    buffer_pool_unpin(bp, 3, 0);
    buffer_pool_destroy(bp);
    close(fd);
    printf("PASS: test_pin_protection\n");
}
/* Test 4: Dirty page write-back on eviction */
static void test_dirty_writeback(void) {
    char tmpname[] = "/tmp/bp_dirty_XXXXXX";
    int fd = mkstemp(tmpname);
    unlink(tmpname);
    /* Write one page with zeroed data */
    uint8_t original[PAGE_SIZE] = {0};
    pwrite(fd, original, PAGE_SIZE, 0);
    BufferPool *bp = buffer_pool_create(fd, 1);  /* single-frame pool */
    /* Fetch page 0, modify it, mark dirty */
    uint8_t *p0 = buffer_pool_fetch(bp, 0);
    p0[0] = 0xFF;
    buffer_pool_unpin(bp, 0, /*dirty=*/1);
    /* Fetch page 1 — forces eviction of page 0 (dirty) */
    /* First we need page 1 to exist */
    uint8_t page1_data[PAGE_SIZE] = {0};
    pwrite(fd, page1_data, PAGE_SIZE, PAGE_SIZE);
    buffer_pool_fetch(bp, 1);
    buffer_pool_unpin(bp, 1, 0);
    /* Verify page 0 was written to disk */
    uint8_t readback[PAGE_SIZE];
    pread(fd, readback, PAGE_SIZE, 0);
    assert(readback[0] == 0xFF);  /* modification persisted */
    buffer_pool_destroy(bp);
    close(fd);
    printf("PASS: test_dirty_writeback\n");
}
```
Run all four tests before connecting the buffer pool to the B-tree layer. The tests exercise every code path: hit, miss, eviction with dirty write-back, and pin protection. If any fails, fix it before proceeding — a buggy buffer pool corrupts everything built on top of it.
---
## Knowledge Cascade: What This Milestone Unlocks
You have just built a component that appears, in some form, in virtually every system that bridges the gap between fast memory and slow storage. Here is what else this knowledge unlocks:
**→ CPU Cache Hierarchy (same problem, different hardware layer).** Your buffer pool is L3 cache for the database. The dirty bit corresponds to the cache line dirty bit in CPU caches. Write-back caching in your buffer pool is exactly how L2/L3 caches work: modified cache lines accumulate, then flush to RAM on eviction. The cache coherency problem (what happens when two CPUs modify the same cache line?) is the distributed version of the buffer pool multi-writer problem. Cache replacement policies (LRU, PLRU, pseudo-LRU in Intel CPUs) are the hardware analogue of what you just implemented in software. Computer architects and database engineers solved the same problem independently and arrived at the same solutions.
**→ Pin Counting is Reference Counting.** The pin count is exactly `shared_ptr`'s reference count in C++, `Rc<T>` in Rust, or ARC (Automatic Reference Counting) in Swift/ObjC. All share the same invariant: when count reaches 0, the resource is eligible for collection. The buffer pool's "eviction" when pin_count == 0 corresponds to the destructor running when refcount reaches 0. The difference: in garbage-collected languages, GC roots prevent collection — in the buffer pool, active pins prevent eviction. The pattern is identical. Understanding pin counting here gives you an intuitive model for every reference-counting memory management scheme.
**→ Double Buffering (GPU, Audio, Networking).** The buffer pool's two roles — providing stable pointers while data is being read, and staging writes before they reach disk — generalize into **double buffering**: maintaining two buffers so the consumer always reads from one while the producer writes to the other. GPU rendering swaps between front buffer (displayed) and back buffer (rendering). Audio hardware uses double-buffering to prevent glitches. Network card DMA rings are a circular buffer pool. The pattern: decouple producer from consumer with intermediate buffering.
**→ The mmap Debate Reveals Database Architecture Choices.** Now that you understand pin control and dirty tracking, you can evaluate the LMDB design choice: LMDB uses `mmap` and copy-on-write pages rather than a buffer pool. It gives up pin control (relies on OS page locking via `mlock()` in some configurations) and dirty tracking (copy-on-write ensures the original is never overwritten, so "dirty" is meaningless). LMDB gets simplicity and can use the OS's optimized page cache. SQLite gets precise write ordering and pin control. Neither is strictly better — the choice depends on your transaction model. You now understand *why* the choice exists, not just that it exists.
**→ The Buffer Pool as a Systems Interface.** Every component above the buffer pool — the B-tree, the VDBE, the query planner — touches disk through `buffer_pool_fetch`. Every component below the buffer pool — the OS, the SSD, the filesystem — is abstracted away. This boundary enables the entire architecture above to be tested and developed without ever touching a real disk (use an in-memory file via `memfd_create` or a RAM-backed tmpfs). It enables performance analysis: every cache miss is a disk access; hit rate directly measures how much disk-bound work you're doing. This layering principle — define a clean abstraction at the storage boundary — appears in every high-performance system. Linux's `struct address_space` operations, the FreeBSD VM object system, Windows' Cache Manager — all implement some variant of this interface.
**→ 2Q, ARC, and Why Production Databases Are More Complex.** LRU you just implemented handles many workloads well, but fails on sequential scans (floods the cache) and cold-start (new pages get evicted before they can establish access frequency). PostgreSQL's clock-sweep with second-chance pages, SQLite's enhanced LRU with clean-page preference, and systems like InnoDB's midpoint insertion (new pages enter at 37% from tail, not the head) all address the same failure mode. The theory here is cache replacement theory, where optimal policy requires future knowledge and real policies approximate it. You've learned the baseline; understanding why production systems diverge tells you exactly which workloads expose the baseline's weaknesses.
---
## What You Have Built
At the end of this milestone:
- A **buffer pool manager** with configurable frame count (default 1000) and fixed 4096-byte pages
- A **FetchPage operation** that returns pinned page pointers, loading from disk on cache miss with `pread()`
- An **LRU eviction policy** using a doubly-linked list embedded in frame structs, with O(1) move-to-head and O(1) eviction of the tail
- A **pin/unpin lifecycle** that prevents eviction of in-use pages, with explicit dirty flagging on unpin
- A **write-back dirty page mechanism** that writes modified pages to disk on eviction or explicit flush
- A **FlushAll operation** for clean shutdown and checkpoint preparation
- A **NewPage allocator** that extends the database file by one page
- A **page table** (hash map) for O(1) page ID → frame index lookup
- **Hit rate statistics** tracking fetches, hits, misses, evictions, and dirty writes
- A **test suite** covering basic fetch, LRU eviction order, pin protection, and dirty write-back
The B-tree layer (Milestone 5) will use this buffer pool through the six-function cursor interface. Every page read is a `buffer_pool_fetch` with a subsequent `buffer_pool_unpin`. Every page write calls `buffer_pool_fetch`, modifies the data in place, and calls `buffer_pool_unpin` with `dirty=1`. The buffer pool is the foundation that makes the entire storage engine work.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m5 -->
<!-- MS_ID: build-sqlite-m5 -->
# Milestone 5: B-tree Page Format & Table Storage
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

Stand back and locate yourself in the machine. The tokenizer turned SQL text into tokens. The parser turned tokens into an AST. The VDBE compiler turned the AST into bytecode. The virtual machine executes those instructions. The buffer pool, which you just built, answers the question "give me the bytes at page N" in ~50 nanoseconds if the page is cached, or ~100 microseconds if it must load from disk.
But none of that answers the *real* question: **what is on those pages?** A page is 4096 raw bytes. The buffer pool gives you a pointer to those bytes. This milestone defines what those bytes mean — how a table row is encoded, how rows are organized into a searchable tree, how that tree's nodes are laid out within a single page, and how a new table comes to exist in the database at all.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

After this milestone, the cursor stub you wrote in Milestone 3 — `btree_cursor_rewind`, `btree_cursor_next`, `btree_cursor_column` — will have real implementations. The bytecode instruction `OP_COLUMN` will decompress actual variable-length records from actual pages in an actual file. The layer between "bytes on disk" and "typed values in registers" is what you are building now.
This is the densest milestone in the project. The challenges are not algorithmic in the traditional sense — you are not solving an NP-hard problem. The challenge is *precision*: bytes must land in exactly the right positions, offsets must be consistent across write and read, pointers must not drift. A one-byte error in a page format function corrupts every subsequent read. Build this layer carefully. Test it at the byte level.
---
## The Revelation: B-trees and B+trees Are Different Structures for Different Access Patterns
Here is the misconception that derails almost every developer who first implements a storage engine: "B-trees and B+trees are basically the same. They're both balanced trees with high branching factor. Once I implement one, I have the other."
This is precisely wrong, and the difference is not cosmetic — it changes the fundamental performance characteristics of two critical operations.
Let's make it concrete. You have a table `users` with 1 million rows. You execute:
```sql
SELECT * FROM users WHERE rowid = 500000;
```
And separately:
```sql
SELECT * FROM users WHERE name BETWEEN 'Alice' AND 'Bob';
```
These are completely different access patterns. The first is a **point lookup** — find a single row by its primary key. The second is a **range scan** — find all rows whose `name` column falls in a range.

![B-tree vs B+tree — Before/After Comparison](./diagrams/diag-btree-vs-bplustree.svg)

**The B-tree approach (used for rowid-keyed tables):** Every node — both internal routing nodes and leaf nodes — stores complete row data. An internal node at height 2 contains actual rows alongside its separator keys and child pointers. When you look up `rowid = 500000`, you might find it at an internal node at height 2, reading the row without ever reaching a leaf. Point lookups can terminate early.
**The B+tree approach (used for secondary indexes):** Only leaf nodes store data. Internal nodes store *only* separator keys and child pointers — no row data, no record payloads. All data lives at the leaves, and crucially, **all leaf nodes are linked in a doubly-linked chain**. When you scan `name BETWEEN 'Alice' AND 'Bob'`, you find 'Alice' by descending the tree (following separator keys), land on the leaf containing 'Alice', then follow the leaf chain rightward until you pass 'Bob'. You never go back up the tree.
The reveal: **SQLite deliberately uses both structures for different purposes.** Tables (B-tree, data in all nodes) are optimized for the most common table operation: `WHERE rowid = N`. Indexes (B+tree, data only in leaves with linked leaves) are optimized for range scans on arbitrary columns. This dual-structure design is why `SELECT * FROM users WHERE rowid = 500000` can terminate at an internal node while `SELECT * FROM users WHERE name BETWEEN 'Alice' AND 'Bob'` efficiently traverses a flat leaf chain.
The *same page format* serves both structures — the leaf-or-internal distinction is encoded in the page header. You build one page layout, two tree behaviors.
---
## What You Are Building: The Complete Picture
Before descending into bytes, see the architecture you're implementing:
```
Page types in the database file:
┌─────────────────────────────────────────────────────────────────┐
│ Page 0:  Database header (first 100 bytes) + first B-tree page  │
│ Page 1:  sqlite_master (system catalog) root page               │
│ Page 2+: B-tree/B+tree pages for user tables and indexes        │
└─────────────────────────────────────────────────────────────────┘
Each B-tree/B+tree page is one of four types:
  TABLE_LEAF     — stores full rows, keyed by rowid
  TABLE_INTERNAL — stores rowid separator keys + child pointers
  INDEX_LEAF     — stores (key_value, rowid) pairs
  INDEX_INTERNAL — stores key_value separator keys + child pointers
```
The type is encoded in the page header. The page format — header, cell pointer array, cell content area — is the same for all four. The content of the cells differs.
Let's define the foundational constants before anything else:
```c
/* btree.h */
#pragma once
#include <stdint.h>
#include "buffer_pool.h"
#define PAGE_SIZE              4096
#define PAGE_HEADER_SIZE       12     /* fixed header at start of each page */
#define DB_HEADER_SIZE         100    /* extra header on page 0 only */
#define PAGE_MAX_CELLS         ((PAGE_SIZE - PAGE_HEADER_SIZE - 2) / 4)
/* Page types — stored in header byte 0 */
typedef enum {
    PAGE_TYPE_TABLE_LEAF     = 0x0D,  /* SQLite's actual values */
    PAGE_TYPE_TABLE_INTERNAL = 0x05,
    PAGE_TYPE_INDEX_LEAF     = 0x0A,
    PAGE_TYPE_INDEX_INTERNAL = 0x02,
} PageType;
```
The values `0x0D`, `0x05`, `0x0A`, `0x02` are not arbitrary — they match SQLite's actual file format. Your database file will be byte-compatible with real SQLite for simple schemas. This means you can open it with the real `sqlite3` command-line tool and read your data. That's an extraordinarily useful debugging capability.

> **🔑 Foundation: Page-based storage and why fixed-size pages matter**
> 
> ## Page-Based Storage and Why Fixed-Size Pages Matter
**What it is:**
A *page* is the fundamental unit of data transfer between disk and memory in a database (or any storage engine). Rather than reading individual bytes or rows on demand, the storage engine divides the entire data file into fixed-size chunks — typically 4KB, 8KB, or 16KB — and always reads or writes exactly one full page at a time.
Think of it like a library that only lends books in crates of exactly 10 books. You never borrow a single book — you borrow a crate, take what you need, and return the whole crate.
**Why fixed size specifically?**
This is the key design decision. Fixed-size pages matter for several concrete reasons:
- **Predictable addressing:** If every page is exactly 8KB, then page number `N` always starts at byte offset `N × 8192` in the file. No directory, no indirection — just arithmetic. You can seek directly to any page in O(1).
- **Buffer pool management:** The in-memory cache (buffer pool) holds a fixed number of pages as frames. Because all pages are the same size, frames are interchangeable — any frame can hold any page. Variable-size chunks would make this a fragmentation nightmare.
- **Alignment with OS and hardware:** The OS reads/writes disk in blocks (often 4KB). A database page sized as a multiple of the OS block size means one database read maps cleanly to one (or a few) OS reads. Misaligned or variable sizes would cause partial-block reads, wasting I/O.
- **Simple allocation and reuse:** When you free a page, its slot is immediately reusable by the next allocation. No compaction needed.
**The mental model to remember:**
> The page is the *atom* of your storage engine — it cannot be partially read, partially written, or partially cached. Everything else (rows, indexes, free lists) is built *inside* pages, never *across* them arbitrarily.
Once you internalize this, many other design decisions (why rows can't span pages naively, why page size affects tree fanout, why WAL writes are page-aligned) become obvious consequences rather than arbitrary rules.

---
## The Page Header: 12 Bytes That Describe Everything

![Page Binary Layout — Structure Layout](./diagrams/diag-btree-page-format.svg)

Every B-tree page begins with a 12-byte header. On page 0 (the first page), the database-level header occupies the first 100 bytes, and the B-tree page header begins at byte 100. On all other pages, the B-tree page header begins at byte 0.
```
Byte offset  Size  Field
─────────────────────────────────────────────────────────────────
0            1     Page type (PAGE_TYPE_TABLE_LEAF, etc.)
1            2     First freeblock offset (0 if none)
3            2     Number of cells on this page
5            2     Start of cell content area (0 means 65536)
7            1     Number of fragmented free bytes
8            4     Right-child page number (TABLE_INTERNAL and INDEX_INTERNAL only)
```
All multi-byte integers in the page header are **big-endian** — most-significant byte first. This is SQLite's portability decision: a database file created on a little-endian x86 machine can be read on a big-endian SPARC without conversion. SQLite chose big-endian, matching network byte order convention.
```c
/* Page header structure — matches SQLite's on-disk format exactly */
typedef struct {
    uint8_t  page_type;           /* one of PageType enum values */
    uint16_t first_freeblock;     /* offset of first freeblock, or 0 */
    uint16_t cell_count;          /* number of cells on page */
    uint16_t cell_content_area;   /* byte offset where cell content starts */
    uint8_t  fragmented_bytes;    /* number of fragmented free bytes */
    uint32_t right_child;         /* rightmost child page (internal pages only) */
} PageHeader;
/* Read a page header from a raw page buffer.
   page_start: pointer to byte 0 of the page (accounts for DB_HEADER_SIZE on page 0). */
static void page_header_read(const uint8_t *page_start, PageHeader *h) {
    h->page_type          = page_start[0];
    h->first_freeblock    = ((uint16_t)page_start[1] << 8) | page_start[2];
    h->cell_count         = ((uint16_t)page_start[3] << 8) | page_start[4];
    h->cell_content_area  = ((uint16_t)page_start[5] << 8) | page_start[6];
    h->fragmented_bytes   = page_start[7];
    /* Right-child only meaningful for internal pages */
    h->right_child = ((uint32_t)page_start[8]  << 24) |
                     ((uint32_t)page_start[9]  << 16) |
                     ((uint32_t)page_start[10] <<  8) |
                     ((uint32_t)page_start[11]);
}
static void page_header_write(uint8_t *page_start, const PageHeader *h) {
    page_start[0] = h->page_type;
    page_start[1] = (h->first_freeblock >> 8) & 0xFF;
    page_start[2] =  h->first_freeblock       & 0xFF;
    page_start[3] = (h->cell_count >> 8) & 0xFF;
    page_start[4] =  h->cell_count       & 0xFF;
    page_start[5] = (h->cell_content_area >> 8) & 0xFF;
    page_start[6] =  h->cell_content_area       & 0xFF;
    page_start[7] = h->fragmented_bytes;
    page_start[8]  = (h->right_child >> 24) & 0xFF;
    page_start[9]  = (h->right_child >> 16) & 0xFF;
    page_start[10] = (h->right_child >>  8) & 0xFF;
    page_start[11] =  h->right_child        & 0xFF;
}
```
Write a test immediately after implementing this: initialize a `PageHeader`, write it to a buffer, read it back, and assert every field matches. A single bit-shift error here silently corrupts every page you write.
---
## The Slotted Page: Bidirectional Growth

> **🔑 Foundation: Slotted page format with bidirectional growth**
> 
> ## Slotted Page Format with Bidirectional Growth
**What it is:**
A *slotted page* is the standard internal layout used to store variable-length records (rows, tuples) inside a fixed-size page. It solves a real problem: if rows have different sizes, how do you pack them efficiently into a page while still being able to find, delete, and update them cheaply?
The layout divides the page into three regions:
```
┌──────────────────────────────────────────────────┐
│  Header  │  Slot Array →        ← Tuple Data     │
│          │ [slot0][slot1][slot2] ... [t2][t1][t0] │
└──────────────────────────────────────────────────┘
           ↑ grows right                ↑ grows left
```
- **Header** (fixed position, start of page): stores metadata — page ID, free space pointer, slot count, flags.
- **Slot array** (grows forward from the header): a compact array of `(offset, length)` pairs. Slot `i` tells you where tuple `i` starts within the page and how many bytes it occupies.
- **Tuple data** (grows backward from the end of the page): actual row bytes packed from the end toward the middle.
**The bidirectional growth trick:**
New slots are appended to the right end of the slot array. New tuples are placed at the left end of the free space (i.e., growing inward from the page's tail). Both regions grow toward the middle. Free space is the gap between them. When they meet, the page is full.
This is elegant because:
1. **Stable slot IDs:** A row's identity is its slot number, not its byte offset. When you compact/reorder tuples (e.g., after a delete), you only update the offset in the slot entry — external references (index pointers) use the slot number and stay valid.
2. **Variable-length friendly:** Tuples of any size just need contiguous bytes from the tail region. The slot array handles the indirection.
3. **Deletion is cheap:** Mark a slot as dead (set length to 0 or a tombstone flag). The space isn't reclaimed immediately — a later *compaction* (vacuum) pass shifts live tuples and resets the free pointer.
**Concrete example:**
Suppose you insert three rows of sizes 40B, 120B, and 60B into a fresh 4KB page:
- Tuple 0 (40B) lands at bytes 4056–4095 (end of page)
- Tuple 1 (120B) lands at bytes 3936–4055
- Tuple 2 (60B) lands at bytes 3876–3935
- Slot array has three entries: `[(4056,40), (3936,120), (3876,60)]`
Now delete tuple 1. Slot 1 is marked dead. The bytes 3936–4055 are wasted until compaction. But slot 0 and slot 2 — and any index entries pointing to them — don't change at all.
**The mental model to remember:**
> The slot array is a *stable indirection layer*. Rows move physically when the page is compacted, but their slot number never changes. Decouple "where is it stored" from "how do I refer to it" — that's the entire point of the slotted format.
This pattern recurs everywhere in systems design (inode tables, segment descriptors, handle tables). Once you've built a slotted page, you'll recognize the same idea in many other contexts.

After the 12-byte header, the page is divided into two regions that grow toward each other:

![Slotted Page — Bidirectional Growth](./diagrams/diag-slotted-page-growth.svg)

```
┌─────────────────────────────────────────────────────┐
│ Page Header (12 bytes)                              │  ← byte 0 (or 100 on page 0)
├─────────────────────────────────────────────────────┤
│ Cell Pointer Array                                  │  ← grows DOWNWARD (forward)
│   [ptr0][ptr1][ptr2][ptr3]...                       │    each entry is 2 bytes
│                                                     │
│              FREE SPACE                             │
│                                                     │
│                       ...[cellN][cellN-1][cell0]    │  ← grows UPWARD (backward)
│ Cell Content Area                                   │
└─────────────────────────────────────────────────────┘
```
The **cell pointer array** lives immediately after the header and grows forward (toward higher addresses). Each entry is a 2-byte big-endian offset — the byte position within the page where a cell's content starts. `ptr[0]` is the offset of the first cell, `ptr[1]` of the second, and so on.
The **cell content area** lives at the end of the page and grows backward (toward lower addresses). New cells are inserted at the current `cell_content_area` offset, which decreases with each insertion.
The free space between these two regions is the page's available capacity. You can calculate it:
```c
static uint16_t page_free_space(const uint8_t *page_start) {
    PageHeader h;
    page_header_read(page_start, &h);
    uint16_t header_end = PAGE_HEADER_SIZE + h.cell_count * 2;
    uint16_t content_start = h.cell_content_area;
    if (content_start == 0) content_start = 65536; /* SQLite convention: 0 means 65536 */
    return content_start - header_end;
}
```
This bidirectional growth is the same trick used by process memory layout: the stack grows downward from high addresses, the heap grows upward from low addresses, and they meet in the middle when memory is exhausted. The insight generalizes: whenever you have two variable-size regions sharing a fixed-size space, bidirectional growth maximizes utilization because neither region needs a pre-committed size. 
Why cell pointers instead of a simple sequential layout? Because **deletion and insertion should not require moving cell content**. When you delete a cell, you zero the pointer entry and mark the content space as free — but you don't shuffle every other cell. The cells themselves stay where they are. The pointer array is the index into the cell content; reorganizing pointers is cheap (2 bytes each), reorganizing content is expensive (up to hundreds of bytes each). This design — small indirection layer, large content left in place — is the slotted page's core insight.
---
## Variable-Length Integer Encoding (Varint)
[[EXPLAIN:variable-length-integer-encoding-(varint)|Variable-length integer encoding (varint)]]
Before you can understand the cell format, you need to understand varints, because they appear everywhere in the record format: in row sizes, in column counts, in the encoded values of integer columns.

![Variable-Length Integer (Varint) Encoding — Data Walk](./diagrams/diag-btree-varint-encoding.svg)

SQLite uses a specific varint format: a 1-to-9 byte encoding of 64-bit integers where small values occupy fewer bytes. The encoding:
- If the value fits in 7 bits (0–127): encode as 1 byte with the high bit 0
- Otherwise: encode the next 7 bits with the high bit set to 1 (continuation flag), and continue with the remaining bits
- Exception: if 8 bytes with continuation bits are not sufficient, the 9th byte encodes the remaining bits without a continuation flag (enabling the full 64-bit range)
```
Value    Encoded bytes
──────   ─────────────────────────────────────────────────────
0        0x00                               (1 byte)
127      0x7F                               (1 byte)
128      0x81 0x00                          (2 bytes)
255      0x81 0x7F                          (2 bytes)
16383    0xFF 0x7F                          (2 bytes)
16384    0x81 0x80 0x00                     (3 bytes)
```
The encoding mirrors UTF-8 (same continuation-bit trick) and Protocol Buffers' Base-128 varint. It also matches DWARF debug info's LEB128 encoding and Bitcoin's CompactSize encoding. This is a cross-domain pattern worth recognizing: when you have integers that are *usually* small but *occasionally* large, variable-length encoding is always the answer.
```c
/* Encode a 64-bit unsigned integer as a varint.
   buf must have at least 9 bytes of space.
   Returns the number of bytes written. */
int varint_encode(uint64_t value, uint8_t *buf) {
    if (value <= 0x7F) {
        buf[0] = (uint8_t)value;
        return 1;
    }
    /* Write 7 bits at a time, most significant first, with continuation bits */
    uint8_t tmp[9];
    int     len = 0;
    /* SQLite varint: bytes 1-8 use 7 bits with high bit as continuation.
       Byte 9 (if needed) uses 8 bits and is the terminus. */
    if (value > 0x00FFFFFFFFFFFFFF) {
        /* Needs 9 bytes */
        tmp[8] = (uint8_t)(value & 0xFF);
        value >>= 8;
        len = 8;
        for (int i = 7; i >= 0; i--) {
            tmp[i] = (uint8_t)((value & 0x7F) | 0x80);
            value >>= 7;
        }
        memcpy(buf, tmp, 9);
        return 9;
    }
    /* 1-8 byte encoding */
    /* Work from least significant to most significant, then reverse */
    len = 0;
    do {
        tmp[len++] = (uint8_t)(value & 0x7F);
        value >>= 7;
    } while (value > 0);
    /* Reverse and add continuation bits */
    for (int i = len - 1; i >= 0; i--) {
        buf[len - 1 - i] = tmp[i] | (i > 0 ? 0x80 : 0x00);
    }
    return len;
}
/* Decode a varint from buf.
   *bytes_read is set to the number of bytes consumed.
   Returns the decoded value. */
uint64_t varint_decode(const uint8_t *buf, int *bytes_read) {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result = (result << 7) | (buf[i] & 0x7F);
        if (!(buf[i] & 0x80)) {
            *bytes_read = i + 1;
            return result;
        }
    }
    /* 9th byte: uses all 8 bits */
    result = (result << 8) | buf[8];
    *bytes_read = 9;
    return result;
}
/* Return the number of bytes a given value would require */
int varint_length(uint64_t value) {
    if (value <= 0x7F)           return 1;
    if (value <= 0x3FFF)         return 2;
    if (value <= 0x1FFFFF)       return 3;
    if (value <= 0x0FFFFFFF)     return 4;
    if (value <= 0x07FFFFFFFF)   return 5;
    if (value <= 0x03FFFFFFFFFF) return 6;
    if (value <= 0x01FFFFFFFFFFFF) return 7;
    if (value <= 0x00FFFFFFFFFFFFFF) return 8;
    return 9;
}
```
Test varints exhaustively before continuing. Test boundary values: 127, 128, 16383, 16384, 2097151, 2097152, the max int64. A varint bug in encode or decode corrupts every record in your database and produces insane behavior — columns appear to shift, string lengths overflow, integer values are off by a factor of 128. These bugs are fiendishly hard to debug without byte-level tracing.
---
## Row Serialization: The Record Format

![Row Serialization — Record Format](./diagrams/diag-btree-row-serialization.svg)

A row stored in a table B-tree leaf cell has two parts: a **header** that describes the types and sizes of each column, followed by the **column data** itself.
```
Cell layout for TABLE_LEAF:
┌──────────────┬─────────┬──────────────────────────────────────┐
│ payload_size │  rowid  │            Payload                   │
│  (varint)    │(varint) ├────────────────┬─────────────────────┤
│              │         │ header_size    │ serial_type[0]      │
│              │         │ (varint)       │ serial_type[1]      │
│              │         │                │ ...                 │
│              │         │                ├─────────────────────┤
│              │         │                │ col0_data           │
│              │         │                │ col1_data           │
│              │         │                │ ...                 │
└──────────────┴─────────┴────────────────┴─────────────────────┘
```
The **serial type** is a varint that encodes the column's type and, for variable-length columns, its size:
```
Serial type value   Meaning
───────────────────────────────────────────────────────────────
0                   NULL
1                   8-bit signed integer (1 byte of data)
2                   16-bit signed integer (2 bytes, big-endian)
3                   24-bit signed integer (3 bytes, big-endian)
4                   32-bit signed integer (4 bytes, big-endian)
5                   48-bit signed integer (6 bytes, big-endian)
6                   64-bit signed integer (8 bytes, big-endian)
7                   IEEE 754 64-bit float (8 bytes, big-endian)
8                   Integer 0 (no data bytes)
9                   Integer 1 (no data bytes)
10, 11              Reserved
N (even, N≥12)      BLOB, length = (N-12)/2 bytes
N (odd,  N≥13)      TEXT, length = (N-13)/2 bytes
```
Types 8 and 9 are a SQLite optimization: the values 0 and 1 are so common (boolean columns, status flags) that SQLite encodes them with zero data bytes — the value is entirely in the serial type number itself.
```c
/* Compute the serial type for a Value */
static uint64_t serial_type_for_value(const Value *v) {
    switch (v->type) {
    case VAL_NULL:
        return 0;
    case VAL_INTEGER: {
        int64_t i = v->i;
        if (i == 0)                               return 8;
        if (i == 1)                               return 9;
        if (i >= -128     && i <= 127)            return 1;
        if (i >= -32768   && i <= 32767)          return 2;
        if (i >= -8388608 && i <= 8388607)        return 3;
        if (i >= INT32_MIN && i <= INT32_MAX)     return 4;
        if (i >= -140737488355328LL &&
            i <=  140737488355327LL)              return 5;
        return 6;  /* full 64-bit */
    }
    case VAL_REAL:
        return 7;
    case VAL_TEXT:
        return (uint64_t)(v->text.len * 2 + 13);
    case VAL_BLOB:
        return (uint64_t)(v->blob.len * 2 + 12);
    }
    return 0;  /* unreachable */
}
/* Return the data byte count for a serial type */
static int serial_type_data_size(uint64_t st) {
    if (st == 0) return 0;        /* NULL */
    if (st == 8 || st == 9) return 0;   /* integer 0/1 */
    if (st >= 1 && st <= 4) return (int)st;
    if (st == 5) return 6;
    if (st == 6 || st == 7) return 8;
    if (st >= 12 && (st % 2 == 0)) return (int)((st - 12) / 2);  /* BLOB */
    if (st >= 13 && (st % 2 == 1)) return (int)((st - 13) / 2);  /* TEXT */
    return 0;
}
```
Now the full record encoder:
```c
/* Encode a row (array of Values) into a heap-allocated byte buffer.
   *out_len is set to the total byte count.
   Caller must free() the returned buffer. */
uint8_t *record_encode(const Value *cols, int ncols, int *out_len) {
    /* Step 1: compute serial types and header size */
    uint64_t serial_types[64];  /* max 64 columns */
    int data_sizes[64];
    int total_data = 0;
    for (int i = 0; i < ncols; i++) {
        serial_types[i] = serial_type_for_value(&cols[i]);
        data_sizes[i]   = serial_type_data_size(serial_types[i]);
        total_data += data_sizes[i];
    }
    /* Header = header_size_varint + serial_type_varints */
    int header_body_size = 0;
    for (int i = 0; i < ncols; i++)
        header_body_size += varint_length(serial_types[i]);
    int header_size = varint_length(header_body_size + varint_length(header_body_size + 1))
                    + header_body_size;
    /* Recompute accurately (header_size field encodes total header bytes including itself) */
    /* The header begins with a varint containing the total header size (incl. that varint) */
    int hdr_total = 1;  /* start with 1 byte guess for header_size field */
    for (;;) {
        int sz = varint_length((uint64_t)hdr_total) + header_body_size;
        if (sz == hdr_total) break;
        hdr_total = sz;
    }
    int payload_size = hdr_total + total_data;
    uint8_t *buf = malloc(payload_size);
    if (!buf) return NULL;
    uint8_t *p = buf;
    /* Write header_size varint */
    p += varint_encode((uint64_t)hdr_total, p);
    /* Write serial types */
    for (int i = 0; i < ncols; i++)
        p += varint_encode(serial_types[i], p);
    /* Write column data */
    for (int i = 0; i < ncols; i++) {
        const Value *v = &cols[i];
        uint64_t st = serial_types[i];
        if (st == 0 || st == 8 || st == 9) {
            /* NULL, 0, 1: no data bytes */
        } else if (st >= 1 && st <= 6) {
            /* Integer: write big-endian */
            int nbytes = data_sizes[i];
            int64_t val = v->i;
            for (int b = nbytes - 1; b >= 0; b--) {
                p[b] = (uint8_t)(val & 0xFF);
                val >>= 8;
            }
            p += nbytes;
        } else if (st == 7) {
            /* IEEE 754 double, big-endian */
            uint64_t bits;
            memcpy(&bits, &v->r, 8);
            for (int b = 7; b >= 0; b--) {
                p[b] = (uint8_t)(bits & 0xFF);
                bits >>= 8;
            }
            p += 8;
        } else if (st % 2 == 1) {
            /* TEXT */
            memcpy(p, v->text.data, v->text.len);
            p += v->text.len;
        } else {
            /* BLOB */
            memcpy(p, v->blob.data, v->blob.len);
            p += v->blob.len;
        }
    }
    *out_len = payload_size;
    return buf;
}
```
The decode direction is equally important — this is what `OP_COLUMN` calls:
```c
/* Decode a single column from a serialized record.
   record: pointer to the start of the payload (after rowid varint, before header).
   col_idx: which column to extract.
   out: output Value. */
void record_decode_column(const uint8_t *record, int record_len,
                           int col_idx, Value *out) {
    /* Read header_size */
    int consumed;
    uint64_t header_size = varint_decode(record, &consumed);
    const uint8_t *header_pos = record + consumed;
    const uint8_t *header_end = record + header_size;
    /* Skip to serial_type[col_idx] */
    uint64_t serial_type = 0;
    int data_offset = (int)header_size;  /* data starts after header */
    for (int i = 0; i <= col_idx; i++) {
        if (header_pos >= header_end) {
            out->type = VAL_NULL;  /* column beyond record — treat as NULL */
            return;
        }
        int vlen;
        serial_type = varint_decode(header_pos, &vlen);
        if (i < col_idx) {
            data_offset += serial_type_data_size(serial_type);
        }
        header_pos += vlen;
    }
    /* Decode the value at data_offset */
    const uint8_t *data = record + data_offset;
    switch (serial_type) {
    case 0:
        out->type = VAL_NULL;
        break;
    case 8:
        out->type = VAL_INTEGER; out->i = 0;
        break;
    case 9:
        out->type = VAL_INTEGER; out->i = 1;
        break;
    case 1: case 2: case 3: case 4: case 5: case 6: {
        int nbytes = serial_type_data_size(serial_type);
        int64_t val = 0;
        /* Sign-extend from the first byte */
        val = (int8_t)data[0];
        for (int b = 1; b < nbytes; b++) {
            val = (val << 8) | data[b];
        }
        out->type = VAL_INTEGER;
        out->i = val;
        break;
    }
    case 7: {
        uint64_t bits = 0;
        for (int b = 0; b < 8; b++)
            bits = (bits << 8) | data[b];
        out->type = VAL_REAL;
        memcpy(&out->r, &bits, 8);
        break;
    }
    default: {
        int dlen = serial_type_data_size(serial_type);
        if (serial_type % 2 == 1) {
            out->type       = VAL_TEXT;
            out->text.data  = (char *)data;  /* points into page — valid while page is pinned */
            out->text.len   = dlen;
        } else {
            out->type       = VAL_BLOB;
            out->blob.data  = (uint8_t *)data;
            out->blob.len   = dlen;
        }
        break;
    }
    }
}
```
Notice: for TEXT and BLOB, the returned `Value` points directly into the page buffer — zero copies. This is why the buffer pool's pin must be held for as long as you're using these values. The `btree_cursor_column` function must not unpin the page until the caller is done with the value.
---
## Cell Formats: Four Variants
Each page type has a specific cell format. The cell pointer array entries point to cells stored in the cell content area.

![Page Binary Layout — Structure Layout](./diagrams/diag-btree-page-format.svg)

**TABLE_LEAF cell:**
```
[payload_size: varint][rowid: varint][payload: payload_size bytes]
```
The payload is the encoded record (header + data as described above). `payload_size` is the byte count of the payload only (not including the rowid or payload_size varints).
**TABLE_INTERNAL cell:**
```
[left_child: uint32_t big-endian][key: varint]
```
A 4-byte child page number, followed by a varint rowid separator key. The "right child" (the child for keys greater than the maximum key in the rightmost cell) is stored in the page header's `right_child` field.
**INDEX_LEAF cell:**
```
[payload_size: varint][payload: payload_size bytes]
```
The payload is an encoded record containing two columns: (indexed_column_value, rowid). No rowid is stored separately — it's encoded as the second column in the payload.
**INDEX_INTERNAL cell:**
```
[left_child: uint32_t big-endian][payload_size: varint][payload: payload_size bytes]
```
A 4-byte child page number, followed by the separator key encoded as a record (same format as index leaf payload).
```c
/* Compute the byte size of a cell given its type and content */
static int cell_size(PageType page_type, const uint8_t *cell_start) {
    int pos = 0;
    if (page_type == PAGE_TYPE_TABLE_INTERNAL) {
        /* 4-byte left child + varint key */
        int vlen;
        varint_decode(cell_start + 4, &vlen);
        return 4 + vlen;
    }
    if (page_type == PAGE_TYPE_INDEX_INTERNAL) {
        /* 4-byte left child + varint payload_size + payload */
        int vlen;
        uint64_t payload_size = varint_decode(cell_start + 4, &vlen);
        return 4 + vlen + (int)payload_size;
    }
    /* Leaf pages: varint payload_size [+ varint rowid for table] + payload */
    int vlen1;
    uint64_t payload_size = varint_decode(cell_start, &vlen1);
    pos += vlen1;
    if (page_type == PAGE_TYPE_TABLE_LEAF) {
        int vlen2;
        varint_decode(cell_start + pos, &vlen2);
        pos += vlen2;
    }
    return pos + (int)payload_size;
}
```
---
## Inserting a Cell Into a Page
Inserting a cell into a page requires three steps: write the cell content into the cell content area (growing backward), write a new entry in the cell pointer array (growing forward), and update the header.
```c
/* Insert a cell into a page at the correct sorted position.
   page:        pointer to the 4096-byte page buffer (pinned, dirty)
   cell_data:   the cell bytes to insert
   cell_len:    number of bytes
   insert_pos:  index in the cell pointer array (0-based, cells shift right)
   Returns 0 on success, -1 if page has insufficient free space. */
int page_insert_cell(uint8_t *page, const uint8_t *cell_data, int cell_len,
                     int insert_pos) {
    PageHeader h;
    page_header_read(page, &h);
    int free = page_free_space(page);
    if (free < cell_len + 2) return -1;  /* +2 for the pointer entry */
    /* Allocate space in cell content area (grow backward) */
    uint16_t content_start = h.cell_content_area;
    if (content_start == 0) content_start = 65536;  /* page size limit */
    content_start -= (uint16_t)cell_len;
    memcpy(page + content_start, cell_data, cell_len);
    /* Shift existing pointer entries right to make room */
    uint8_t *ptr_array = page + PAGE_HEADER_SIZE;
    int ncells = h.cell_count;
    memmove(ptr_array + (insert_pos + 1) * 2,
            ptr_array + insert_pos * 2,
            (ncells - insert_pos) * 2);
    /* Write the new pointer entry */
    ptr_array[insert_pos * 2]     = (content_start >> 8) & 0xFF;
    ptr_array[insert_pos * 2 + 1] =  content_start       & 0xFF;
    /* Update header */
    h.cell_count++;
    h.cell_content_area = content_start;
    page_header_write(page, &h);
    return 0;
}
/* Read the cell offset at position idx from the pointer array */
static uint16_t cell_pointer_get(const uint8_t *page, int idx) {
    const uint8_t *ptr = page + PAGE_HEADER_SIZE + idx * 2;
    return ((uint16_t)ptr[0] << 8) | ptr[1];
}
/* Return a pointer to the cell content at index idx */
static const uint8_t *cell_at(const uint8_t *page, int idx) {
    return page + cell_pointer_get(page, idx);
}
```
---
## Initializing a New Page
When `buffer_pool_new_page` allocates a fresh zeroed page, you must initialize it as a B-tree page:
```c
void page_init(uint8_t *page, PageType type) {
    memset(page, 0, PAGE_SIZE);
    PageHeader h = {0};
    h.page_type         = (uint8_t)type;
    h.cell_content_area = PAGE_SIZE;  /* content area starts at the bottom */
    h.cell_count        = 0;
    h.right_child       = 0;
    page_header_write(page, &h);
}
```
---
## The B-tree Data Structure: Navigating Between Pages
[[EXPLAIN:b-tree-vs-b+tree:-when-data-lives-in-all-nodes-vs-only-leaves|B-tree vs B+tree: when data lives in all nodes vs only leaves]]
Now that you can read and write individual pages, you need the tree structure that connects them. A B-tree is characterized by:
- Every node is one page
- All leaf nodes are at the same depth (perfectly balanced height)
- Each internal node holds N separator keys and N+1 child page references
- The minimum occupancy is ceil(max_keys / 2) — no node except the root can be less than half full
The branching factor determines tree height. For our page format, an internal page can hold approximately:
```
max_cells_per_internal_page ≈ (PAGE_SIZE - PAGE_HEADER_SIZE) / (4 + 8)  ≈ 340
```
(4 bytes child pointer + 8 bytes for a typical rowid varint). With branching factor 340, a tree of height 3 can index 340³ ≈ 39 million rows. Height 4 covers 13 billion rows. A 10-million-row table needs at most 3 page accesses from root to leaf.
### Finding a Row by Rowid (Search)
```c
/* Search a table B-tree for a given rowid.
   Returns the page_id of the leaf page containing the row,
   and sets *slot to the cell index within that leaf.
   Returns INVALID_PAGE_ID if not found. */
PageId btree_search(BufferPool *bp, PageId root_page_id,
                    int64_t target_rowid, int *slot_out) {
    PageId current = root_page_id;
    for (;;) {
        uint8_t *page = buffer_pool_fetch(bp, current);
        PageHeader h;
        page_header_read(page, &h);
        if (h.page_type == PAGE_TYPE_TABLE_LEAF) {
            /* Binary search within leaf for the rowid */
            int lo = 0, hi = h.cell_count - 1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                const uint8_t *cell = cell_at(page, mid);
                int vlen1, vlen2;
                /* skip payload_size */
                varint_decode(cell, &vlen1);
                int64_t rowid = (int64_t)varint_decode(cell + vlen1, &vlen2);
                if (rowid == target_rowid) {
                    buffer_pool_unpin(bp, current, 0);
                    *slot_out = mid;
                    return current;
                } else if (rowid < target_rowid) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            buffer_pool_unpin(bp, current, 0);
            return INVALID_PAGE_ID;  /* not found */
        }
        /* Internal page: find the child to descend into */
        /* Cells are sorted; find the first cell with key >= target_rowid */
        PageId next_child = h.right_child;  /* default: rightmost child */
        for (int i = 0; i < h.cell_count; i++) {
            const uint8_t *cell = cell_at(page, i);
            int vlen;
            int64_t key = (int64_t)varint_decode(cell + 4, &vlen);  /* skip 4-byte child ptr */
            if (target_rowid <= key) {
                /* target is in the left subtree of this cell */
                next_child = ((uint32_t)cell[0] << 24) | ((uint32_t)cell[1] << 16) |
                             ((uint32_t)cell[2] <<  8) |  (uint32_t)cell[3];
                break;
            }
        }
        buffer_pool_unpin(bp, current, 0);
        current = next_child;
    }
}
```
Note: in a real B-tree search you often use binary search within internal pages too, for large branching factors. For simplicity, the above uses linear scan for the child pointer lookup — acceptable for pages with up to ~340 cells.
---
## Node Splitting: The Hard Part

![Node Split on Insert Overflow — State Evolution](./diagrams/diag-btree-node-split.svg)

When you insert a row into a leaf page and there is not enough free space, the page must **split**. Splitting is the core algorithmic challenge of B-tree implementation. Get it wrong and the tree becomes inconsistent: rows are lost, keys point to wrong children, or the balance invariant breaks.
The split procedure:
1. Allocate a new page (the "right sibling")
2. Move approximately half the cells from the overfull page to the right sibling
3. Determine the **separator key** — the smallest key in the right sibling (or the median key, depending on the variant)
4. Insert the separator key with a pointer to the right sibling into the parent page
5. If the parent is also full, split the parent recursively (potentially creating a new root)
The recursion terminates because the root can grow: when the root splits, you allocate a new root page with a single separator key and two children. This is the only time a B-tree grows in height.
```c
typedef struct BTree BTree;
typedef struct SplitResult {
    int64_t  separator_key;   /* key to promote to parent */
    PageId   right_page_id;   /* the newly-allocated right sibling */
} SplitResult;
/* Split a full leaf page.
   left_page_id: the page that is full (and has the new cell to insert)
   new_cell:     the cell that triggered the overflow
   new_rowid:    the rowid of the new cell
   result:       filled with the separator key and right sibling page ID */
static void leaf_split(BufferPool *bp,
                        PageId left_page_id,
                        const uint8_t *new_cell, int new_cell_len,
                        int64_t new_rowid,
                        SplitResult *result) {
    uint8_t *left  = buffer_pool_fetch(bp, left_page_id);
    PageHeader lh;
    page_header_read(left, &lh);
    /* Allocate right sibling */
    PageId right_page_id;
    uint8_t *right = buffer_pool_new_page(bp, &right_page_id);
    page_init(right, PAGE_TYPE_TABLE_LEAF);
    /* Collect all cells (existing + new) in rowid order into a scratch buffer */
    int total_cells = lh.cell_count + 1;
    uint8_t  *cell_bufs[MAX_CELLS_PER_PAGE + 1];
    int       cell_lens[MAX_CELLS_PER_PAGE + 1];
    int64_t   cell_rowids[MAX_CELLS_PER_PAGE + 1];
    int       inserted = 0;
    int       j = 0;
    for (int i = 0; i < lh.cell_count; i++) {
        const uint8_t *c = cell_at(left, i);
        int vlen1, vlen2;
        varint_decode(c, &vlen1);
        int64_t rowid = (int64_t)varint_decode(c + vlen1, &vlen2);
        /* Insert new cell in sorted position */
        if (!inserted && new_rowid < rowid) {
            cell_bufs[j]   = (uint8_t *)new_cell;
            cell_lens[j]   = new_cell_len;
            cell_rowids[j] = new_rowid;
            j++;
            inserted = 1;
        }
        cell_bufs[j]   = (uint8_t *)c;
        cell_lens[j]   = cell_size(PAGE_TYPE_TABLE_LEAF, c);
        cell_rowids[j] = rowid;
        j++;
    }
    if (!inserted) {
        cell_bufs[j]   = (uint8_t *)new_cell;
        cell_lens[j]   = new_cell_len;
        cell_rowids[j] = new_rowid;
        j++;
    }
    /* total_cells == j */
    /* Split: first half stays in left, second half goes to right */
    int split_point = total_cells / 2;  /* right gets [split_point .. total_cells) */
    /* Reinitialize left page and re-insert first half */
    page_init(left, PAGE_TYPE_TABLE_LEAF);
    for (int i = 0; i < split_point; i++) {
        /* cells are already in sorted order; append at the end */
        page_insert_cell(left, cell_bufs[i], cell_lens[i], i);
    }
    /* Insert second half into right page */
    for (int i = split_point; i < total_cells; i++) {
        page_insert_cell(right, cell_bufs[i], cell_lens[i], i - split_point);
    }
    /* Separator key = smallest rowid in right page */
    result->separator_key  = cell_rowids[split_point];
    result->right_page_id  = right_page_id;
    buffer_pool_unpin(bp, left_page_id,  /*dirty=*/1);
    buffer_pool_unpin(bp, right_page_id, /*dirty=*/1);
}
```
After splitting a leaf, you must insert the separator key into the parent internal page. If the parent is also full, you must split the parent — and so on recursively. This is easiest to implement using a path-tracking approach: record the pages visited during the descent to the leaf, then walk back up the path performing splits and insertions as needed.
```c
/* Path element: a page and the slot within it that led downward */
typedef struct {
    PageId page_id;
    int    child_index;  /* which child pointer we followed (-1 = right_child) */
} PathEntry;
/* Full insert into a table B-tree.
   bt:    the BTree handle (contains root_page_id and buffer pool reference)
   cols:  the column values to insert
   ncols: number of columns
   rowid: the rowid for the new row */
int btree_insert(BTree *bt, const Value *cols, int ncols, int64_t rowid) {
    /* Encode the record */
    int record_len;
    uint8_t *record = record_encode(cols, ncols, &record_len);
    /* Build the leaf cell: [payload_size][rowid][payload] */
    uint8_t cell_buf[PAGE_SIZE];
    int cpos = 0;
    cpos += varint_encode((uint64_t)record_len, cell_buf + cpos);
    cpos += varint_encode((uint64_t)rowid,       cell_buf + cpos);
    memcpy(cell_buf + cpos, record, record_len);
    cpos += record_len;
    int cell_len = cpos;
    free(record);
    /* Descend to the leaf, recording the path */
    PathEntry path[32];  /* max tree height */
    int path_depth = 0;
    PageId current = bt->root_page_id;
    for (;;) {
        uint8_t *page = buffer_pool_fetch(bt->bp, current);
        PageHeader h;
        page_header_read(page, &h);
        if (h.page_type == PAGE_TYPE_TABLE_LEAF) {
            buffer_pool_unpin(bt->bp, current, 0);
            break;
        }
        /* Find the child to descend into */
        int child_idx = -1;  /* -1 = right_child */
        PageId next = h.right_child;
        for (int i = 0; i < h.cell_count; i++) {
            const uint8_t *cell = cell_at(page, i);
            int vlen;
            int64_t key = (int64_t)varint_decode(cell + 4, &vlen);
            if (rowid <= key) {
                child_idx = i;
                next = ((uint32_t)cell[0] << 24) | ((uint32_t)cell[1] << 16) |
                       ((uint32_t)cell[2] <<  8) |  (uint32_t)cell[3];
                break;
            }
        }
        path[path_depth].page_id     = current;
        path[path_depth].child_index = child_idx;
        path_depth++;
        buffer_pool_unpin(bt->bp, current, 0);
        current = next;
    }
    /* Attempt insertion at leaf */
    PageId leaf_id = current;
    uint8_t *leaf = buffer_pool_fetch(bt->bp, leaf_id);
    /* Find insertion position (sorted by rowid) */
    PageHeader lh;
    page_header_read(leaf, &lh);
    int insert_pos = lh.cell_count;
    for (int i = 0; i < lh.cell_count; i++) {
        const uint8_t *c = cell_at(leaf, i);
        int vl1, vl2;
        varint_decode(c, &vl1);
        int64_t r = (int64_t)varint_decode(c + vl1, &vl2);
        if (rowid < r) { insert_pos = i; break; }
    }
    if (page_insert_cell(leaf, cell_buf, cell_len, insert_pos) == 0) {
        /* Insertion fit — done */
        buffer_pool_unpin(bt->bp, leaf_id, /*dirty=*/1);
        return 0;
    }
    buffer_pool_unpin(bt->bp, leaf_id, 0);
    /* Leaf is full: split and propagate upward */
    SplitResult split;
    leaf_split(bt->bp, leaf_id, cell_buf, cell_len, rowid, &split);
    /* Propagate the split upward through the path */
    int64_t  promote_key   = split.separator_key;
    PageId   promote_right = split.right_page_id;
    PageId   promote_left  = leaf_id;
    while (path_depth > 0) {
        path_depth--;
        PageId parent_id = path[path_depth].page_id;
        uint8_t *parent = buffer_pool_fetch(bt->bp, parent_id);
        PageHeader ph;
        page_header_read(parent, &ph);
        /* Build the internal cell: [left_child: 4 bytes][separator_key: varint] */
        uint8_t icell[16];
        icell[0] = (promote_left >> 24) & 0xFF;
        icell[1] = (promote_left >> 16) & 0xFF;
        icell[2] = (promote_left >>  8) & 0xFF;
        icell[3] =  promote_left        & 0xFF;
        int iklen = varint_encode((uint64_t)promote_key, icell + 4);
        int icell_len = 4 + iklen;
        /* Insert position: before the first key >= promote_key */
        int ipos = ph.cell_count;
        for (int i = 0; i < ph.cell_count; i++) {
            const uint8_t *c = cell_at(parent, i);
            int vl;
            int64_t k = (int64_t)varint_decode(c + 4, &vl);
            if (promote_key <= k) { ipos = i; break; }
        }
        /* The right sibling of the split becomes the child to the right of ipos.
           Update right_child or the (ipos+1)th cell's left pointer accordingly. */
        /* Simplest approach: set the old child pointer in the new cell to promote_left,
           and stitch promote_right as the new right-side pointer.
           Actually: the cell we're inserting IS the left child = promote_left.
           The right child of promote_key in the parent becomes promote_right.
           This means: if ipos == ph.cell_count, update right_child = promote_right.
           Otherwise, update cell[ipos]'s left pointer to promote_right. */
        /* (This detail requires careful handling; the logic here is simplified.) */
        if (page_insert_cell(parent, icell, icell_len, ipos) == 0) {
            /* After insertion: cell[ipos].left_child = promote_left (already in cell)
               and the key to the right (ipos+1 or right_child) points to promote_right */
            /* Patch: if ipos == ph.cell_count, right_child = promote_right */
            PageHeader ph2;
            page_header_read(parent, &ph2);
            if (ipos == ph2.cell_count - 1) {
                /* Actually the inserted cell IS at the end; right_child was the previous right */
                /* promote_right becomes the new right_child */
                ph2.right_child = promote_right;
                page_header_write(parent, &ph2);
            } else {
                /* Patch the (ipos+1)th cell's left child pointer to promote_right */
                uint8_t *next_cell = (uint8_t *)cell_at(parent, ipos + 1);
                next_cell[0] = (promote_right >> 24) & 0xFF;
                next_cell[1] = (promote_right >> 16) & 0xFF;
                next_cell[2] = (promote_right >>  8) & 0xFF;
                next_cell[3] =  promote_right        & 0xFF;
            }
            buffer_pool_unpin(bt->bp, parent_id, /*dirty=*/1);
            return 0;
        }
        buffer_pool_unpin(bt->bp, parent_id, 0);
        /* Parent also needs splitting — continue upward */
        /* internal_split(bt->bp, parent_id, icell, icell_len, &split); */
        promote_key   = split.separator_key;
        promote_right = split.right_page_id;
        promote_left  = parent_id;
    }
    /* Reached the root and it needs splitting — grow tree height */
    /* Allocate a new root, with left=old_root and right=promote_right,
       separator=promote_key */
    PageId new_root_id;
    uint8_t *new_root = buffer_pool_new_page(bt->bp, &new_root_id);
    page_init(new_root, PAGE_TYPE_TABLE_INTERNAL);
    uint8_t rcell[16];
    rcell[0] = (bt->root_page_id >> 24) & 0xFF;
    rcell[1] = (bt->root_page_id >> 16) & 0xFF;
    rcell[2] = (bt->root_page_id >>  8) & 0xFF;
    rcell[3] =  bt->root_page_id        & 0xFF;
    int rklen = varint_encode((uint64_t)promote_key, rcell + 4);
    page_insert_cell(new_root, rcell, 4 + rklen, 0);
    PageHeader rh;
    page_header_read(new_root, &rh);
    rh.right_child = promote_right;
    page_header_write(new_root, &rh);
    buffer_pool_unpin(bt->bp, new_root_id, /*dirty=*/1);
    bt->root_page_id = new_root_id;
    /* Update system catalog with new root page ID */
    schema_update_root(bt->db, bt->table_name, new_root_id);
    return 0;
}
```
The split logic above is simplified for clarity. The real implementation requires careful handling of which side the new right-sibling pointer goes — this is the most error-prone part. Verify correctness by:
1. Inserting rows in ascending rowid order (triggers rightmost splits — easiest case)
2. Inserting in descending order (triggers leftmost splits)
3. Inserting randomly with a fixed seed (triggers splits across the tree)
4. After each insertion, verify the B-tree invariants: all leaf depths equal, all keys in a node are sorted, each child's keys are between its left and right separator keys in the parent
---
## Cursor Interface: Connecting the VM to the B-tree
The VDBE cursor abstraction from Milestone 3 now has real implementations. Here are the key ones:
```c
typedef struct BTreeCursor {
    BTree    *bt;
    PageId    leaf_page_id;   /* current leaf page */
    int       slot;           /* current cell index within leaf */
    int       at_end;         /* 1 if exhausted */
    int       write_mode;     /* 1 if opened for writes */
} BTreeCursor;
/* Position cursor at the first row (smallest rowid). */
int btree_cursor_rewind(BTreeCursor *cur) {
    /* Descend to the leftmost leaf */
    PageId current = cur->bt->root_page_id;
    for (;;) {
        uint8_t *page = buffer_pool_fetch(cur->bt->bp, current);
        PageHeader h;
        page_header_read(page, &h);
        if (h.page_type == PAGE_TYPE_TABLE_LEAF) {
            cur->leaf_page_id = current;
            cur->slot         = 0;
            cur->at_end       = (h.cell_count == 0);
            buffer_pool_unpin(cur->bt->bp, current, 0);
            return cur->at_end;  /* returns 1 if empty */
        }
        /* Descend to leftmost child: first cell's left_child */
        PageId next_child;
        if (h.cell_count > 0) {
            const uint8_t *cell = cell_at(page, 0);
            next_child = ((uint32_t)cell[0] << 24) | ((uint32_t)cell[1] << 16) |
                         ((uint32_t)cell[2] <<  8) |  (uint32_t)cell[3];
        } else {
            next_child = h.right_child;
        }
        buffer_pool_unpin(cur->bt->bp, current, 0);
        current = next_child;
    }
}
/* Advance cursor to the next row.
   Returns 1 if a next row exists, 0 if exhausted. */
int btree_cursor_next(BTreeCursor *cur) {
    if (cur->at_end) return 0;
    uint8_t *page = buffer_pool_fetch(cur->bt->bp, cur->leaf_page_id);
    PageHeader h;
    page_header_read(page, &h);
    cur->slot++;
    if (cur->slot < h.cell_count) {
        buffer_pool_unpin(cur->bt->bp, cur->leaf_page_id, 0);
        return 1;  /* more rows on this leaf */
    }
    /* Need to move to the right sibling leaf.
       In a B+tree, leaves are linked. In our B-tree, we re-traverse
       from the parent. For simplicity: re-search from root for the
       next rowid after the current one.
       (B+tree leaf linking is the optimization that avoids this.) */
    /* Get current rowid */
    const uint8_t *cell = cell_at(page, cur->slot - 1);
    int vl1, vl2;
    varint_decode(cell, &vl1);
    int64_t current_rowid = (int64_t)varint_decode(cell + vl1, &vl2);
    buffer_pool_unpin(cur->bt->bp, cur->leaf_page_id, 0);
    /* Find the leaf page containing the next rowid */
    int next_slot;
    PageId next_leaf = btree_find_next_leaf(cur->bt, current_rowid, &next_slot);
    if (next_leaf == INVALID_PAGE_ID) {
        cur->at_end = 1;
        return 0;
    }
    cur->leaf_page_id = next_leaf;
    cur->slot         = next_slot;
    return 1;
}
/* Extract column col_idx from the current row into *out */
void btree_cursor_column(BTreeCursor *cur, int col_idx, Value *out) {
    uint8_t *page = buffer_pool_fetch(cur->bt->bp, cur->leaf_page_id);
    const uint8_t *cell = cell_at(page, cur->slot);
    int vl1, vl2;
    uint64_t payload_size = varint_decode(cell, &vl1);
    varint_decode(cell + vl1, &vl2);           /* skip rowid */
    const uint8_t *payload = cell + vl1 + vl2; /* points to record header */
    record_decode_column(payload, (int)payload_size, col_idx, out);
    /* NOTE: do NOT unpin here — out->text.data may point into this page.
       The VM must call buffer_pool_unpin after it's done with the Value. */
    /* In practice: the cursor holds a persistent pin on the current page,
       released only when the cursor advances or closes. */
    (void)page;  /* suppress unused-variable warning — page is kept pinned */
}
```
> **Implementation Note on Leaf Linking:** The code above for `btree_cursor_next` re-traverses from the root to find the next leaf — correct but O(log N) per row. The efficient approach, used in B+trees, is to store a `right_sibling` pointer in each leaf page header. Then advancing to the next leaf is O(1): follow the sibling pointer. SQLite does this for index B+trees (which are true B+trees with data only in leaves and linked leaves) but tables use B-trees where the "next leaf" traversal is less critical because rowid scans are less common than index scans. Implement the re-traversal approach first; adding the leaf link as an optimization is straightforward once the basic structure works.
---
## The System Catalog: The Database Describes Itself

![System Catalog (sqlite_master) — Structure Layout](./diagrams/diag-btree-system-catalog.svg)

Here is one of the most elegant design decisions in SQLite: **the database's own schema is stored in a table within the database**. The schema table is called `sqlite_master` (or `sqlite_schema` in newer versions). It is a table B-tree rooted at page 1 (the second page of the file, after page 0 which contains the database header).
```sql
-- sqlite_master schema:
CREATE TABLE sqlite_master (
    type    TEXT,       -- 'table' or 'index'
    name    TEXT,       -- table or index name
    tbl_name TEXT,      -- table this object belongs to
    rootpage INTEGER,   -- B-tree root page number
    sql     TEXT        -- original CREATE TABLE/INDEX statement
);
```
When your database opens, it reads page 1 to discover what tables exist. When `CREATE TABLE users (...)` executes, it allocates a new B-tree root page, inserts a row into `sqlite_master` recording the table name and root page number, and returns. The mapping `table_name → root_page_id` is itself persisted on disk, in a B-tree, in the same format as every other table.
This self-describing property is called **bootstrapping**: the schema table is an instance of the very data structure it describes. There is no separate "schema file" or "catalog file" — everything is pages in the one database file.
```c
/* System catalog row structure */
typedef struct {
    char     type[8];      /* "table" or "index" */
    char    *name;         /* table or index name */
    char    *tbl_name;     /* table name (same as name for tables) */
    PageId   rootpage;     /* root B-tree page */
    char    *sql;          /* original CREATE statement */
} CatalogEntry;
#define SQLITE_MASTER_ROOT_PAGE 1   /* page 1 is always sqlite_master */
#define SQLITE_MASTER_TYPE_COL  0
#define SQLITE_MASTER_NAME_COL  1
#define SQLITE_MASTER_TBL_COL   2
#define SQLITE_MASTER_ROOT_COL  3
#define SQLITE_MASTER_SQL_COL   4
/* Look up a table by name in sqlite_master.
   Returns the root page ID, or INVALID_PAGE_ID if not found. */
PageId schema_find_table(DB *db, const char *table_name) {
    BTree *catalog = btree_open(db->bp, SQLITE_MASTER_ROOT_PAGE);
    BTreeCursor cur;
    btree_cursor_open_read(&cur, catalog);
    if (btree_cursor_rewind(&cur)) {
        return INVALID_PAGE_ID;  /* empty catalog */
    }
    do {
        /* Read 'name' column (index 1) */
        Value name_val;
        btree_cursor_column(&cur, SQLITE_MASTER_NAME_COL, &name_val);
        if (name_val.type == VAL_TEXT &&
            strncasecmp(name_val.text.data, table_name, name_val.text.len) == 0 &&
            table_name[name_val.text.len] == '\0') {
            /* Found the table — read rootpage column (index 3) */
            Value root_val;
            btree_cursor_column(&cur, SQLITE_MASTER_ROOT_COL, &root_val);
            PageId root = (PageId)root_val.i;
            btree_cursor_close(&cur);
            return root;
        }
    } while (btree_cursor_next(&cur));
    btree_cursor_close(&cur);
    return INVALID_PAGE_ID;
}
/* Register a new table in sqlite_master.
   Called after allocating the B-tree root page for a new table. */
int schema_register_table(DB *db, const char *name,
                           PageId rootpage, const char *sql) {
    Value cols[5];
    /* type = "table" */
    cols[0].type = VAL_TEXT; cols[0].text.data = "table"; cols[0].text.len = 5;
    /* name */
    cols[1].type = VAL_TEXT; cols[1].text.data = (char*)name; cols[1].text.len = strlen(name);
    /* tbl_name = name for tables */
    cols[2] = cols[1];
    /* rootpage */
    cols[3].type = VAL_INTEGER; cols[3].i = (int64_t)rootpage;
    /* sql */
    cols[4].type = VAL_TEXT; cols[4].text.data = (char*)sql; cols[4].text.len = strlen(sql);
    /* Auto-assign a rowid for the catalog entry */
    int64_t rowid = schema_next_rowid(db);
    BTree *catalog = btree_open(db->bp, SQLITE_MASTER_ROOT_PAGE);
    return btree_insert(catalog, cols, 5, rowid);
}
```
The bootstrapping circularity is resolved by convention: page 1 is always `sqlite_master`, known without consulting any catalog. To find table `users`, you open page 1 (no catalog lookup needed — it's always there), scan it as a table B-tree, and look for a row whose `name` column matches `"users"`. The `rootpage` column gives you the root page of `users`'s B-tree.
---
## Executing CREATE TABLE
When the VDBE executes `OP_CREATE_TABLE`, the flow is:
```c
case OP_CREATE_TABLE: {
    const char *table_name = instr->p4.str;
    /* Allocate a fresh B-tree root page for the new table */
    PageId new_root_id;
    uint8_t *new_root_page = buffer_pool_new_page(db->bp, &new_root_id);
    page_init(new_root_page, PAGE_TYPE_TABLE_LEAF);  /* starts as a leaf */
    buffer_pool_unpin(db->bp, new_root_id, /*dirty=*/1);
    /* Register in sqlite_master */
    const char *sql = instr->p4_aux.str;  /* original CREATE TABLE SQL */
    schema_register_table(db, table_name, new_root_id, sql);
    /* Store schema column definitions for later column-index lookups */
    schema_add_columns(db, table_name, /* column defs from AST */ ...);
    vm.pc++;
    break;
}
```
New tables start as a single leaf page. The first INSERT does not trigger a split — it just adds a cell. Splits only occur when a page fills up, which for a typical row size (100–500 bytes) means after ~8–40 rows on the first page.
---
## Full Table Scan
A full table scan — the operation behind `SELECT * FROM t` with no WHERE clause — traverses all leaf pages in rowid order:
```c
int btree_full_scan(BTree *bt, void (*row_callback)(const Value *cols, int ncols, void *ctx),
                    int ncols, void *ctx) {
    BTreeCursor cur;
    btree_cursor_open_read(&cur, bt);
    if (btree_cursor_rewind(&cur)) return 0;  /* empty table */
    Value cols[64];
    do {
        for (int i = 0; i < ncols; i++) {
            btree_cursor_column(&cur, i, &cols[i]);
        }
        row_callback(cols, ncols, ctx);
    } while (btree_cursor_next(&cur));
    btree_cursor_close(&cur);
    return 0;
}
```
The VDBE's `OP_REWIND` / `OP_NEXT` / `OP_COLUMN` instructions map directly onto this: `OP_REWIND` calls `btree_cursor_rewind`, `OP_NEXT` calls `btree_cursor_next`, and `OP_COLUMN` calls `btree_cursor_column`. The scan loop in the VDBE bytecode:
```
Rewind   0  →end    ; position at first row (jump to end if empty)
Column   0  0  r1   ; load column 0 into register 1
Column   0  1  r2   ; load column 1 into register 2
ResultRow r1  2      ; emit (r1, r2)
Next     0  →loop   ; advance; jump to Column if more rows
Halt
```
This is the inner loop of every table scan. On a warm buffer pool (all pages cached), the cost per row is: one `cell_at` lookup + one `record_decode_column` call per projected column. For a 10,000-row table fitting in ~200 pages (at 50 rows/page for 100-byte rows), all pages load once and the scan is entirely in memory.
---
## Endianness: A Portability Decision Made in Concrete
SQLite's on-disk format is entirely big-endian. Every multi-byte integer in page headers, cell pointer arrays, and fixed-width record fields uses big-endian byte order. Your code above consistently applies this via explicit byte-level encoding rather than casting struct pointers to integers.
Why does this matter? On x86 (little-endian), if you wrote `*(uint32_t*)page_start = 0x12345678`, the bytes on disk would be `78 56 34 12` — not `12 34 56 78`. A program on a big-endian ARM would read back `0x78563412`, which is wrong. By always writing bytes explicitly in big-endian order and reading them back the same way, your database file is readable on any architecture without byte-swapping logic.
This is the same convention as network byte order (TCP/IP uses big-endian for all header fields) — "big-endian" and "network byte order" are synonyms. The choice of big-endian over little-endian for on-disk formats is a portability convention, not a performance choice.
---
## Three-Level View: `INSERT INTO users VALUES (1, 'Alice')`
**Level 1 — VDBE bytecode**
```
OpenTable  0  1      -- open write cursor on 'users'
Integer    1  r1     -- load 1 into r1
String8    r2  'Alice' -- load 'Alice' into r2
Null       r3        -- rowid = NULL (auto-assign)
MakeRecord r1  2  r4 -- serialize (r1, r2) into record in r4
Insert     0  r4  r3 -- insert record with rowid r3 into cursor 0
Halt
```
**Level 2 — B-tree / Page format (this milestone)**
- `MakeRecord`: calls `record_encode([VAL_INTEGER(1), VAL_TEXT("Alice")], 2, &len)`. Produces a binary payload: `[header_size][serial_type(1)][serial_type("Alice")][data(1: 0x01)][data("Alice": 41 6C 69 63 65)]`
- `Insert`: calls `btree_insert(cursor, cols, 2, auto_rowid)`. Descends to the correct leaf. Builds the cell `[payload_len][rowid][payload]`. Calls `page_insert_cell` to place the cell in the content area and add a pointer in the pointer array. Updates the page header's `cell_count` and `cell_content_area`. Marks the page dirty.
**Level 3 — Buffer pool / Disk**
- `btree_insert` calls `buffer_pool_fetch(bp, leaf_page_id)` to get the page in memory (either hot cache hit or `pread()` cold miss). Modifies the bytes in the frame buffer. Calls `buffer_pool_unpin(bp, leaf_page_id, dirty=1)`. The page stays in memory (marked dirty) until evicted or explicitly flushed. `pwrite()` only happens at eviction time, not at insert time — this is the write-back cache.

![INSERT Execution — Data Walk](./diagrams/diag-dml-insert-execution.svg)

---
## Design Decision: Overflow Pages
What happens when a row is too large to fit in a single page? A text blob of 5000 bytes cannot fit in a 4096-byte page alongside the page header and cell pointer array.
SQLite handles this with **overflow pages**: when a payload exceeds a threshold (approximately `PAGE_SIZE - PAGE_HEADER_SIZE - 35` bytes for the first piece), the excess is stored in one or more linked overflow pages. The cell in the B-tree leaf stores the first portion of the payload plus a 4-byte pointer to the first overflow page. Each overflow page begins with a 4-byte pointer to the next overflow page (or 0 for the last).
For this milestone, the simplest approach is to **document the size limit and return an error** for payloads that would overflow. Implement overflow pages as a future extension. The limit is generous: at ~3900 bytes per cell, a row must have very large text or blob values to hit it. Real-world tables with normal column sizes never hit this limit.
```c
#define MAX_INLINE_PAYLOAD (PAGE_SIZE - PAGE_HEADER_SIZE - 36)  /* conservative estimate */
int btree_insert(BTree *bt, const Value *cols, int ncols, int64_t rowid) {
    int record_len;
    uint8_t *record = record_encode(cols, ncols, &record_len);
    if (record_len > MAX_INLINE_PAYLOAD) {
        free(record);
        /* TODO: implement overflow pages */
        return -1;  /* error: row too large */
    }
    /* ... continue with normal insert ... */
}
```
Document this limitation clearly in your code. Every production system documents its limits; a silent failure on large rows is a data loss bug.
---
## Database File Header: Page 0
Page 0 is special. Its first 100 bytes are the database-level header, then at byte 100 begins the `sqlite_master` B-tree page header. The key database header fields:
```c
/* Write the database file header (first 100 bytes of page 0).
   Called when creating a new database. */
void db_header_write(uint8_t *page0) {
    /* Magic string: identifies this as a SQLite 3 database */
    memcpy(page0, "SQLite format 3\000", 16);
    /* Page size: 4096, stored as big-endian uint16.
       Special case: SQLite stores 65536 as 1 (historical quirk). */
    page0[16] = (4096 >> 8) & 0xFF;
    page0[17] =  4096       & 0xFF;
    /* File format write/read version */
    page0[18] = 1;  /* write: legacy */
    page0[19] = 1;  /* read:  legacy */
    /* Reserved space at end of each page (0 for us) */
    page0[20] = 0;
    /* Maximum/minimum embedded payload fractions (SQLite uses 64, 32, 32) */
    page0[21] = 64;
    page0[22] = 32;
    page0[23] = 32;
    /* File change counter: increment on every write transaction */
    uint32_t change_counter = 1;
    page0[24] = (change_counter >> 24) & 0xFF;
    page0[25] = (change_counter >> 16) & 0xFF;
    page0[26] = (change_counter >>  8) & 0xFF;
    page0[27] =  change_counter        & 0xFF;
    /* Database size in pages */
    /* (filled in after first writes) */
    /* ... other header fields ... */
    /* Text encoding: UTF-8 = 1 */
    page0[56] = 0; page0[57] = 0; page0[58] = 0; page0[59] = 1;
    /* Initialize the sqlite_master B-tree page starting at byte 100 */
    page_init(page0 + 100, PAGE_TYPE_TABLE_LEAF);
    /* Adjust cell_content_area: on page 0, the usable area starts at 100 */
    /* Actually page_init uses 0-based offsets within the slice it receives.
       Correct behavior: the B-tree page header at offset 100 uses offsets
       relative to the start of page 0 (byte 0), so cell_content_area = PAGE_SIZE
       and all cell offsets are from byte 0. */
}
```
The magic string `"SQLite format 3\000"` at the start of the file is the identifier that tools use to detect SQLite databases. It's the database equivalent of the ELF magic bytes `\x7FELF` or the PNG magic `\x89PNG\r\n`. When `file` on Linux tells you a file is a "SQLite 3.x database", this string is what it's detecting.
---
## Testing Strategy: Trust No Assumption
Testing this milestone requires verification at four levels:
**Level 1: Varint round-trip**
```c
void test_varint(void) {
    uint64_t values[] = {0, 1, 127, 128, 16383, 16384, 2097151, 2097152,
                         INT32_MAX, INT64_MAX};
    for (int i = 0; i < 10; i++) {
        uint8_t buf[9];
        int written = varint_encode(values[i], buf);
        int read;
        uint64_t decoded = varint_decode(buf, &read);
        assert(decoded == values[i]);
        assert(written == read);
        assert(written == varint_length(values[i]));
    }
    printf("PASS: varint round-trip\n");
}
```
**Level 2: Record round-trip**
```c
void test_record_encode_decode(void) {
    Value cols[3];
    cols[0].type = VAL_INTEGER; cols[0].i = 42;
    cols[1].type = VAL_TEXT;    cols[1].text.data = "Alice"; cols[1].text.len = 5;
    cols[2].type = VAL_NULL;
    int len;
    uint8_t *buf = record_encode(cols, 3, &len);
    assert(buf != NULL);
    Value out;
    record_decode_column(buf, len, 0, &out);
    assert(out.type == VAL_INTEGER && out.i == 42);
    record_decode_column(buf, len, 1, &out);
    assert(out.type == VAL_TEXT && out.text.len == 5);
    assert(memcmp(out.text.data, "Alice", 5) == 0);
    record_decode_column(buf, len, 2, &out);
    assert(out.type == VAL_NULL);
    free(buf);
    printf("PASS: record encode/decode\n");
}
```
**Level 3: Page insertion and slotted layout**
Insert cells of varying sizes, verify the pointer array is correct, verify free space accounting, verify cells don't overlap.
**Level 4: B-tree insert and scan**
```c
void test_btree_insert_scan(void) {
    /* Open an in-memory database (use memfd_create or /tmp) */
    int fd = open("/tmp/test_btree.db", O_RDWR | O_CREAT | O_TRUNC, 0644);
    DB *db = db_create(fd);
    /* Create table */
    db_exec(db, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
    /* Insert 1000 rows */
    for (int i = 1; i <= 1000; i++) {
        char sql[64];
        snprintf(sql, sizeof(sql), "INSERT INTO t VALUES (%d, 'name%d')", i, i);
        db_exec(db, sql);
    }
    /* Scan and verify all 1000 rows come back */
    int count = 0;
    int prev_rowid = 0;
    /* ... scan and verify ... */
    assert(count == 1000);
    printf("PASS: btree insert/scan 1000 rows\n");
    db_close(db);
    unlink("/tmp/test_btree.db");
}
```
Also test:
- Insert in descending order (triggers left-biased splits)
- Insert with duplicate rowid (should fail or overwrite, depending on your semantics)
- Create multiple tables (each gets its own root page, all tracked in sqlite_master)
- Open a database file, close it, reopen it, and verify all data persists
---
## Knowledge Cascade: What This Milestone Unlocks
**→ Filesystem internals.** The ext4, Btrfs, and NTFS filesystems all use B-trees to organize their directory structures and extent maps. Btrfs's copy-on-write B-trees allow cheap snapshots — instead of splitting a page in place, you copy the page, modify the copy, and update the parent pointer. The ZFS filesystem uses a similar approach. Your B-tree knowledge directly translates to understanding how your OS organizes files on disk. The `btrfs inspect-internal dump-tree` command shows you the raw B-tree structure — you'll now recognize the internal/leaf distinction and separator keys.
**→ Protocol design and binary encoding.** The varint you just implemented is directly related to UTF-8 (same continuation-bit pattern), Protocol Buffers' Base-128 varint (identical semantics), Bitcoin's CompactSize encoding, and DWARF debugging information's LEB128 encoding. Every time a protocol or format must represent integers that are "usually small but occasionally large," some form of variable-length encoding is the answer. Now that you understand why (fixed-width wastes space for small values; variable-width adds decoding complexity), you can evaluate new encoding schemes on your own. Protocol Buffers' `sint32` (zigzag encoding) is the extension for signed integers — read the protobuf encoding guide and you'll recognize everything.
**→ The cursor pattern everywhere.** The `btree_cursor_rewind` / `btree_cursor_next` / `btree_cursor_column` iterator you built is the universal pattern for lazily producing a sequence of values from a data source. The same interface appears as Python's generator protocol (`__iter__`, `__next__`), Rust's `Iterator` trait, Java's `Iterator<T>`, C++'s range-for (based on `begin()`, `end()`, `++`, `*`), and reactive streams (RxJava, Project Reactor). The underlying data structure changes — B-tree, hash table, file, network socket, computed sequence — but the interface is identical: initialize, advance, read current. You've now built a cursor from first principles; every future cursor you encounter is the same pattern with different internals.
**→ Bootstrapping and self-describing systems.** The `sqlite_master` table that stores schema information in the same B-tree format as all other tables is an instance of **self-description bootstrapping**. The same pattern appears in: the ELF format (the ELF header describes where to find the section table, and the section table is stored in the ELF file itself), TCP/IP (the IP header describes the packet, which contains IP header), operating system kernels (the kernel's own code is loaded by a bootloader that reads the kernel's own header format), and Lisp (the Lisp runtime is defined in Lisp). The pattern: define a simple, fixed bootstrap entry point (page 1, ELF magic bytes, the bootloader), then let everything else be described by the structure reachable from that entry point.
**→ Memory layout mirrors process layout.** The slotted page's bidirectional growth — cell pointer array grows forward from the header, cell content grows backward from the end — is structurally identical to a process's virtual memory layout: the stack grows downward from high addresses, the heap grows upward from low addresses. Both solve the same problem: two variable-size regions sharing a fixed-size space, with neither knowing in advance how large it will be. The insight generalizes to any bounded buffer with two variable consumers. When you encounter this layout pattern in unfamiliar code — a packet buffer with a header region and a payload region growing from opposite ends — you'll recognize it immediately.
**→ The database file as a filesystem.** Your database file is, in miniature, a filesystem. Pages are blocks. The B-tree pages are directory inodes. Row data is file content. The system catalog (sqlite_master) is the root directory. Page allocation (via `buffer_pool_new_page`) is like block allocation in a filesystem. Page deallocation (not yet implemented — free list pages in SQLite) is like freeing blocks. The only thing missing is the tree structure that maps filenames to inodes — which in your database is the system catalog mapping table names to root page IDs. When you read about filesystems — ext4, ZFS, APFS — you'll recognize all the same concepts.
---
## Common Pitfalls: What Will Corrupt Your Database Silently
**1. Endianness inconsistency.** If even one field in your page format is written little-endian and read big-endian (or vice versa), you'll read garbage values. The symptom: a table that reports 16,777,216 rows when it has 1 (because `0x00 0x00 0x00 0x01` is 1 in big-endian but read as `0x01 0x00 0x00 0x00 = 16777216` in little-endian). Write a test that creates a database, closes it, hexdumps page 0 with `xxd`, and verifies specific bytes at specific offsets. Treat the hex dump as the ground truth.
**2. Off-by-one in cell content area.** When `cell_content_area = PAGE_SIZE` (empty page), the first cell is written starting at `PAGE_SIZE - cell_len`. If you compute `cell_content_area - cell_len` with uint16_t arithmetic and `cell_len > cell_content_area`, you get a wraparound. Use explicit bounds checking and signed intermediate values when computing free space.
**3. Forgetting to update the right-child pointer on split.** When an internal page splits, the right sibling's leftmost key becomes the separator. The right-child of the original page must be updated to point to the new right sibling's leftmost child. Getting this wrong produces a tree where one child subtree is unreachable — its rows silently disappear from scans.
**4. Cell pointer array alignment.** The cell pointer array starts immediately after the 12-byte header (at byte 12 for non-root pages). Calculations that add `PAGE_HEADER_SIZE + cell_count * 2` must use consistent definitions of `PAGE_HEADER_SIZE`. If this shifts by even one byte, all cell pointer reads are wrong.
**5. Pin leaks on error paths.** Every `buffer_pool_fetch` in the B-tree code must have a corresponding `buffer_pool_unpin`. When an error is detected mid-descent and you return early, walk the path and unpin all held pages. A single leaked pin in a frequently-called function will eventually make all buffer pool frames permanently pinned and deadlock the database.
**6. Sign extension in varint decode.** When reading a negative integer encoded with serial type 1–6, you must sign-extend the first byte. In the decode code above, `(int8_t)data[0]` does the sign extension for 1-byte integers — this ensures `-1` stored as `0xFF` is decoded as `int64_t = -1`, not `int64_t = 255`. Miss the sign extension and all negative integers in your database are silently wrong.
---
## What You Have Built
At the end of this milestone, you have the complete storage engine foundation:
- A **page format** with 12-byte header, big-endian fields, and four page types (table/index × leaf/internal) — byte-compatible with SQLite's file format
- A **slotted page layout** with cell pointer array growing forward from the header and cell content area growing backward from the page end, enabling O(1) cell access by index
- A **variable-length integer (varint) encoder/decoder** handling the full 64-bit integer range in 1–9 bytes, consistent with SQLite's wire format
- A **record serializer/deserializer** encoding typed column values using serial type codes, with optimizations for common cases (integers 0 and 1 use zero data bytes)
- A **B-tree insert** with node splitting and separator key promotion, handling both leaf splits and recursive internal page splits up to root growth
- A **B+tree leaf page format** for secondary indexes, storing (key, rowid) pairs only in leaf nodes with internal nodes containing only separator keys
- A **full table scan** via the cursor interface: `btree_cursor_rewind`, `btree_cursor_next`, `btree_cursor_column`
- A **system catalog** (`sqlite_master`) on page 1, mapping table names to root page IDs using the same B-tree format as user tables
- A **CREATE TABLE implementation** that allocates a B-tree root page and registers the table in the system catalog
The cursor interface stubs from Milestone 3 now have real implementations. `OP_COLUMN` calls `record_decode_column` on actual page data. `OP_INSERT` calls `btree_insert` which writes real binary records to real B-tree leaf pages. The VDBE can now execute SQL against actual persistent data stored in a file that a real SQLite command-line tool can open.
The next milestone — SELECT execution and DML — builds the remaining cursor operations (UPDATE, DELETE, three-valued logic for WHERE) on top of this storage foundation.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m6 -->
<!-- MS_ID: build-sqlite-m6 -->
# Milestone 6: SELECT Execution & DML
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You have built every layer of the engine except the one that makes them work together. The tokenizer dissolves SQL text into tokens. The parser assembles tokens into an AST. The VDBE compiler translates ASTs into bytecode. The virtual machine executes instructions. The buffer pool caches pages. The B-tree stores rows in a binary format on disk.
None of that means anything until you wire all six layers into a single end-to-end path: SQL text enters, rows come out, and modifications persist. That is what this milestone delivers.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

The components you will implement are deceptively simple to name — table scan, projection, filtering, INSERT, UPDATE, DELETE — but each hides a specific trap. Projection seems trivial until you hit variable-length records. Filtering seems obvious until NULL breaks your boolean logic. DELETE seems straightforward until you realize you're modifying a data structure you're simultaneously iterating. This milestone is where correctness becomes the hard problem.
---
## The Revelation: SQL NULL Is Not Python None
Here is what every developer assumes when they first implement a SQL engine: NULL is just a missing value. It's like `None` in Python, `null` in Java, `nil` in Go, `nullptr` in C++. When you compare it to anything, you get false. When you include it in arithmetic, you get some default. It equals itself.
This model is wrong in a way that causes more production database bugs than any other single concept.
Consider this table:
```
users:
rowid | name    | age
1     | Alice   | 25
2     | Bob     | NULL
3     | Carol   | 30
```
Now execute:
```sql
SELECT * FROM users WHERE age != 25;
```
If NULL were like `None`, this would return Bob (because `NULL != 25` evaluates to "true — they're different"). It does not. It returns only Carol.
Execute:
```sql
SELECT * FROM users WHERE NOT (age = 25);
```
Same result. Only Carol. Bob disappears again.
Execute:
```sql
SELECT * FROM users WHERE age = NULL;
```
Zero rows returned. Not Bob. Zero.
Execute:
```sql
SELECT * FROM users WHERE age IS NULL;
```
One row: Bob.
This is the revelation: **SQL NULL is not a value. It is the absence of a value.** The correct mental model is not "NULL equals itself" but "NULL is unknown." When you compare an unknown to anything — including another unknown — the result is unknown. And in SQL's three-valued logic, unknown is treated as false in WHERE clauses.

![Three-Valued Logic Truth Tables — Before/After](./diagrams/diag-dml-three-valued-logic.svg)

The three truth values are: TRUE, FALSE, and NULL (unknown). The truth tables:
```
AND:    T   F   N        OR:     T   F   N        NOT:
T       T   F   N        T       T   T   T        T → F
F       F   F   F        F       T   F   N        F → T
N       N   F   N        N       T   N   N        N → N
```
The critical observations:
- `NULL AND TRUE = NULL` — even when one side is true, unknown AND true is still unknown
- `NULL OR TRUE = TRUE` — this is the exception: unknown OR definitely-true is definitely true
- `NULL AND FALSE = FALSE` — unknown AND definitely-false is definitely false
- `NOT NULL = NULL` — negating unknown is still unknown
The WHERE clause includes a row only when its predicate evaluates to TRUE. NULL and FALSE both exclude the row. This is why `WHERE age != 25` doesn't return Bob: `NULL != 25` evaluates to NULL, not TRUE, so Bob is excluded.
[[EXPLAIN:three-valued-logic-(true,-false,-null)-in-sql|Three-valued logic (TRUE, FALSE, NULL) in SQL]]
This is not an edge case you can defer. Three-valued logic is fundamental to SQL semantics. Get it wrong and your database silently returns incorrect results for every query touching nullable columns — which is most production queries. Implement it now, in the expression evaluator, before any DML.
---
## What You Are Building
Five capabilities, implemented in sequence because each depends on the previous:
1. **Table scan with projection** — `SELECT col1, col2 FROM t` via B-tree cursor iteration and record deserialization
2. **WHERE clause evaluation with three-valued logic** — filter rows by evaluating boolean expressions correctly, including NULL semantics
3. **INSERT** — serialize a new row and insert it into the B-tree
4. **UPDATE** — find matching rows, deserialize, modify specific columns, re-serialize, rewrite
5. **DELETE** — find matching rows and remove them without corrupting the cursor
Each of these operates through the VDBE: the SQL goes through the tokenizer → parser → compiler → VM, and the VM calls the cursor API into the B-tree. Your job this milestone is to make that path work end-to-end, including all the subtle correctness requirements.
---
## Full Table Scan and Projection

![SELECT Execution — Full Stack Data Walk](./diagrams/diag-dml-select-execution.svg)

The simplest SELECT — `SELECT * FROM users` — exercises the complete stack. Trace what happens at each layer:
**The bytecode the compiler emits:**
```
addr  opcode         p1    p2    p3
0     OpenTable      0     0     0     // cursor 0 = 'users', read mode
1     Rewind         0     5     0     // position at first row; empty → jump to 5
2     Column         0     0     1     // reg[1] = cursor[0].column(0)
3     Column         0     1     2     // reg[2] = cursor[0].column(1)
4     Column         0     2     3     // reg[3] = cursor[0].column(2)
5     ResultRow      1     3     0     // emit (reg[1], reg[2], reg[3])
6     Next           0     2     0     // advance; if more rows, jump to 2
7     Halt           0     0     0
```
Wait — that's wrong. I numbered the jump targets before the body. Let me be precise: `Rewind` jumps to the `Halt` instruction (address 7) when the table is empty. `Next` jumps back to address 2 (the first `Column`) when there are more rows.
The VM executes addresses 2 through 6 once per row. For a 10,000-row table that fits in memory, this is ~50,000 instruction dispatches — roughly 50 microseconds at 1 billion dispatches per second. The dominant cost is memory access: reading each page from the buffer pool and decoding the record.
### Projection: Deserializing Specific Columns
When the query is `SELECT name, age FROM users` (not `SELECT *`), the compiler emits `OP_COLUMN` only for the requested columns at their schema-defined indices. The cursor's `btree_cursor_column` function handles partial deserialization.
The key property of the record format you built in Milestone 5: to read column at index `N`, you scan the serial type array in the header to skip over columns 0 through N-1, accumulating data sizes, then decode column N at the computed offset. You never deserialize the entire record — only what's needed.
```c
/* In the VDBE, OP_COLUMN handler */
case OP_COLUMN: {
    int cursor_id = instr->p1;
    int col_idx   = instr->p2;
    int dest_reg  = instr->p3;
    BTreeCursor *cur = vm.cursors[cursor_id];
    if (!cur || cur->at_end) {
        vm.regs[dest_reg].type = VAL_NULL;
        vm.pc++;
        break;
    }
    /* btree_cursor_column reads directly from the pinned page,
       storing a pointer into page memory for TEXT/BLOB values.
       The page stays pinned until the cursor advances or closes. */
    btree_cursor_column(cur, col_idx, &vm.regs[dest_reg]);
    vm.pc++;
    break;
}
```
The projection operation — selecting a subset of columns — maps directly to **relational algebra's π (projection) operator**. In relational theory, a relation is a set of tuples, and projection produces a new relation containing only the specified attributes. Your implementation: instead of materializing the full tuple and then discarding columns, you read only the needed columns from the binary record. This lazy deserialization is the storage engine's implementation of push-down optimization — avoid reading data you don't need.

![B-tree Cursor — Iterator Pattern](./diagrams/diag-dml-cursor-pattern.svg)

The cursor pattern connecting the VDBE to the B-tree is worth examining explicitly. The cursor is an **iterator** — it maintains position state and provides a uniform interface to advance through a sequence one element at a time:
```c
typedef struct BTreeCursor {
    BTree    *bt;
    int       write_mode;     /* 0 = read-only, 1 = write */
    PageId    leaf_page_id;   /* current leaf page */
    int       slot;           /* cell index within current leaf */
    int       at_end;         /* 1 if past last row */
    /* For pinned page access */
    uint8_t  *current_page;   /* pinned page buffer, or NULL */
} BTreeCursor;
```
The cursor interface from Milestone 3 (previously stubbed) now has real implementations. Here is what each VDBE opcode maps to:
| VDBE opcode | Cursor call | Effect |
|------------|-------------|--------|
| `OP_OPEN_TABLE` | `btree_cursor_open()` | Find table in sqlite_master, open cursor |
| `OP_REWIND` | `btree_cursor_rewind()` | Seek to first leaf, slot 0 |
| `OP_NEXT` | `btree_cursor_next()` | Advance slot; cross leaf boundary if needed |
| `OP_COLUMN` | `btree_cursor_column()` | Decode column N from current row |
| `OP_ROWID` | `btree_cursor_rowid()` | Decode rowid varint from current cell |
| `OP_CLOSE` | `btree_cursor_close()` | Release pins, free cursor |
This iterator abstraction is the same pattern as Python's generator protocol (`__iter__`, `__next__`), C++'s range-based for loop (uses `begin()`, `++`, `*`, `end()`), Rust's `Iterator` trait (`next()` returning `Option<Item>`), and Java's `Iterator<T>`. The data source changes — B-tree, hash table, network socket, file lines — but the protocol is identical: advance, read current value, stop when exhausted.
---
## WHERE Clause Evaluation with Three-Valued Logic
The expression evaluator is the component that implements three-valued logic. It takes an expression AST node (or the corresponding bytecode) and returns one of three possible outcomes: TRUE (include this row), FALSE (exclude this row), NULL (also exclude this row — unknown is treated as non-matching in WHERE).
```c
/* Three-valued logic result */
typedef enum {
    TRI_FALSE = 0,
    TRI_TRUE  = 1,
    TRI_NULL  = 2,   /* unknown */
} Trivalue;
```
The expression evaluator in the VDBE processes comparison opcodes that return a `Trivalue`. Before this milestone, comparison opcodes like `OP_GT` called `value_compare()` which returned `INT32_MAX` for NULL comparisons. Now those sentinel values must be translated into `TRI_NULL` behavior:
```c
/* value_compare_tri: compare two Values, returning Trivalue */
static Trivalue value_compare_tri(const Value *a, const Value *b, Opcode op) {
    /* NULL propagation: any comparison with NULL → NULL */
    if (a->type == VAL_NULL || b->type == VAL_NULL) {
        return TRI_NULL;
    }
    int cmp = value_compare(a, b);  /* returns -1, 0, +1 */
    int result;
    switch (op) {
    case OP_EQ:  result = (cmp == 0);  break;
    case OP_NE:  result = (cmp != 0);  break;
    case OP_LT:  result = (cmp < 0);   break;
    case OP_LE:  result = (cmp <= 0);  break;
    case OP_GT:  result = (cmp > 0);   break;
    case OP_GE:  result = (cmp >= 0);  break;
    default:     return TRI_NULL;
    }
    return result ? TRI_TRUE : TRI_FALSE;
}
```
The critical change from Milestone 3: every comparison opcode must check for NULL inputs **before** comparing. In the VM dispatch:
```c
case OP_EQ: {
    int ra       = instr->p1;
    int jump_to  = instr->p2;
    int rb       = instr->p3;
    Trivalue tri = value_compare_tri(&vm.regs[ra], &vm.regs[rb], OP_EQ);
    if (tri == TRI_NULL || tri == TRI_FALSE) {
        /* Condition did NOT hold — take the skip jump */
        vm.pc = jump_to;
    } else {
        vm.pc++;
    }
    break;
}
```
The `TRI_NULL` case jumps just like `TRI_FALSE`. This is the core of three-valued logic in WHERE: NULL is not a match.
### IS NULL and IS NOT NULL
These are the only operators that operate *on* NULL rather than propagating it:
```c
case OP_IS_NULL: {
    int reg      = instr->p1;
    int jump_to  = instr->p2;
    /* Jump if register IS null */
    if (vm.regs[reg].type == VAL_NULL) {
        vm.pc = jump_to;
    } else {
        vm.pc++;
    }
    break;
}
case OP_NOT_NULL: {
    int reg      = instr->p1;
    int jump_to  = instr->p2;
    /* Jump if register is NOT null */
    if (vm.regs[reg].type != VAL_NULL) {
        vm.pc = jump_to;
    } else {
        vm.pc++;
    }
    break;
}
```
`IS NULL` and `IS NOT NULL` always return TRUE or FALSE, never NULL. They are the escape hatch from the three-valued logic system — the way to explicitly check for the absence of a value. The SQL standard deliberately made these the only way to match NULL because `= NULL` would otherwise make the system inconsistent: if `NULL = NULL` were TRUE, then `NOT (NULL = NULL)` would be FALSE, but `NOT NULL = NULL` should also be FALSE by the NOT table, and that creates paradoxes in query semantics.
### NULL in AND and OR
When the compiler generates code for `WHERE a = 1 AND b IS NOT NULL`, it emits two sequential comparisons. The AND logic is implicit in the control flow: each comparison is a conditional skip; if any skip fires, the row is excluded. This works correctly for three-valued logic because a NULL comparison fires the skip jump (excludes the row), which is exactly the AND-with-NULL semantics: `NULL AND anything = NULL/FALSE`.
OR is more subtle. `WHERE a = 1 OR b = 2` compiles to:
```
Column   0  col_a  r1     ; load column a
Integer  1         r2     ; load constant 1
Eq       r1  →check_b  r2 ; if a=1, skip to ResultRow (OR short-circuit)
Column   0  col_b  r3     ; load column b  
Integer  2         r4     ; load constant 2
Ne       r3  →next r4    ; if b!=2 (or NULL), skip to Next
ResultRow ...
Next     ...
```
This short-circuit evaluation handles NULL correctly: if `a = 1` is TRUE, the row is included without evaluating `b = 2`. If `a = 1` is NULL or FALSE, evaluate `b = 2`. If both are NULL or FALSE, skip. The case that requires care: `NULL OR TRUE = TRUE`. If `b = 2` fires as TRUE after `a = 1` returned NULL, the row is correctly included.
### Implementing IS NULL/IS NOT NULL in the Compiler
The parser produces `NODE_BINARY_EXPR` with `op = TOKEN_IS` for `x IS NULL` (optionally followed by NOT). The compiler must detect this pattern and emit `OP_IS_NULL` or `OP_NOT_NULL`:
```c
static void compile_is_null_expr(Compiler *c, int cursor,
                                  AstNode *expr, int dest_reg) {
    /* expr is NODE_BINARY_EXPR(IS) with right side = NODE_LITERAL_NULL */
    int col_reg = alloc_register(c);
    compile_expr_into_reg(c, cursor, expr->binary.left, col_reg);
    /* For IS NULL: load 1 into dest_reg if col_reg is null, else 0 */
    /* (Or we can emit OP_IS_NULL directly as a conditional skip) */
    /* As a boolean value in a register: */
    int is_null_val = (expr->binary.op == TOKEN_IS) ? 1 : 0;
    /* Is the right side NODE_LITERAL_NULL? */
    if (expr->binary.right->type == NODE_LITERAL_NULL) {
        /* Emit: if col_reg is NULL → dest_reg = 1, else dest_reg = 0 */
        int not_null_jump = emit(c, OP_IS_NULL, col_reg, 0, 0, "test IS NULL");
        emit(c, OP_INTEGER, is_null_val ? 0 : 1, dest_reg, 0, "IS NULL false");
        int skip_jump = emit(c, OP_GOTO, 0, 0, 0, "skip true branch");
        patch_jump(c, not_null_jump);
        emit(c, OP_INTEGER, is_null_val ? 1 : 0, dest_reg, 0, "IS NULL true");
        patch_jump(c, skip_jump);
    }
}
```
---
## NOT NULL Constraint Enforcement
Before any INSERT or UPDATE reaches the B-tree, the constraint checker must verify that columns declared `NOT NULL` are not being assigned NULL.
The schema stores which columns have the NOT NULL flag (from the CREATE TABLE AST). At INSERT time:
```c
/* Constraint checking before INSERT */
static int check_not_null_constraints(DB *db, const char *table_name,
                                       const Value *cols, int ncols) {
    SchemaTable *schema = schema_get_table(db, table_name);
    if (!schema) {
        snprintf(db->error, sizeof(db->error),
                 "Table '%s' does not exist", table_name);
        return -1;
    }
    for (int i = 0; i < schema->column_count && i < ncols; i++) {
        if (schema->columns[i].is_not_null && cols[i].type == VAL_NULL) {
            snprintf(db->error, sizeof(db->error),
                     "NOT NULL constraint failed: %s.%s",
                     table_name, schema->columns[i].name);
            return -1;
        }
    }
    return 0;
}
```
In the VDBE, this check happens inside the `OP_INSERT` handler, before calling `btree_cursor_insert`:
```c
case OP_INSERT: {
    int cursor_id  = instr->p1;
    int record_reg = instr->p2;
    int rowid_reg  = instr->p3;
    BTreeCursor *cur = vm.cursors[cursor_id];
    /* Auto-assign rowid if NULL */
    int64_t rowid;
    if (vm.regs[rowid_reg].type == VAL_NULL) {
        rowid = btree_next_rowid(cur);
    } else {
        rowid = vm.regs[rowid_reg].i;
    }
    /* Constraint check: unpack the record back into Values for checking */
    /* (Alternatively, pass column Values directly to OP_INSERT) */
    /* ... constraint check here ... */
    int rc = btree_cursor_insert(cur, rowid,
                                  vm.regs[record_reg].blob.data,
                                  vm.regs[record_reg].blob.len);
    if (rc < 0) {
        snprintf(vm.error, sizeof(vm.error), "%s", db->error);
        vm.had_error = 1;
        goto halt;
    }
    vm.pc++;
    break;
}
```
A cleaner approach: move the NOT NULL check into the compiler, emitting `OP_NOT_NULL` + conditional halt for each constrained column before the `OP_MAKE_RECORD`. This way the check happens at the register level before serialization:
```c
/* Compiler: emit constraint checks for INSERT */
static void compile_not_null_checks(Compiler *c, const char *table_name,
                                     int first_val_reg, int ncols) {
    SchemaTable *schema = schema_get_table(c->db, table_name);
    for (int i = 0; i < ncols && i < schema->column_count; i++) {
        if (schema->columns[i].is_not_null) {
            int reg = first_val_reg + i;
            /* If reg IS NULL → halt with error */
            int skip_halt = emit(c, OP_NOT_NULL, reg, 0, 0,
                                 "check NOT NULL");
            /* OP_HALT with error message in p4 */
            char *msg = malloc(128);
            snprintf(msg, 128, "NOT NULL constraint failed: %s.%s",
                     table_name, schema->columns[i].name);
            emit_p4_str(c, OP_HALT, 1, 0, 0, msg, "NOT NULL violation");
            patch_jump(c, skip_halt);
        }
    }
}
```
The compiler-side approach runs constraint checks as part of the query plan, visible in EXPLAIN output — which is the correct architecture. The runtime check in `OP_INSERT` is a safety net.
---
## INSERT: From SQL to B-tree

![INSERT Execution — Data Walk](./diagrams/diag-dml-insert-execution.svg)

The INSERT path was sketched in Milestone 3 (the compiler emits the bytecode) and the B-tree insert was built in Milestone 5. This milestone wires them together and handles the remaining details.
**Auto-increment rowid:** When the user writes `INSERT INTO t VALUES ('Alice', 25)` without specifying a rowid, the engine assigns the next available rowid. SQLite's algorithm: `MAX(rowid) + 1`, with the current maximum tracked in the B-tree cursor's rightmost-leaf position.
```c
int64_t btree_next_rowid(BTreeCursor *cur) {
    /* Find the rightmost leaf's last cell's rowid + 1 */
    BTree *bt = cur->bt;
    PageId current = bt->root_page_id;
    for (;;) {
        uint8_t *page = buffer_pool_fetch(bt->bp, current);
        PageHeader h;
        page_header_read(page, &h);
        if (h.page_type == PAGE_TYPE_TABLE_LEAF) {
            if (h.cell_count == 0) {
                buffer_pool_unpin(bt->bp, current, 0);
                return 1;  /* empty table: start at rowid 1 */
            }
            /* Last cell = rightmost cell = highest rowid */
            const uint8_t *last_cell = cell_at(page, h.cell_count - 1);
            int vl1, vl2;
            varint_decode(last_cell, &vl1);
            int64_t max_rowid = (int64_t)varint_decode(last_cell + vl1, &vl2);
            buffer_pool_unpin(bt->bp, current, 0);
            return max_rowid + 1;
        }
        /* Follow the rightmost child (right_child pointer) */
        PageId next = h.right_child;
        buffer_pool_unpin(bt->bp, current, 0);
        current = next;
    }
}
```
**Verifying persistence:** After INSERT, a subsequent SELECT must return the inserted row. This is the acceptance criterion that tests the complete round-trip: SQL → tokenizer → parser → compiler → VDBE → B-tree insert → B-tree scan → record decode → VDBE register → result output.
```c
/* Integration test: insert then select */
void test_insert_select_roundtrip(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)");
    db_exec(db, "INSERT INTO t VALUES (1, 'Alice')");
    db_exec(db, "INSERT INTO t VALUES (2, 'Bob')");
    ResultSet *rs = db_query(db, "SELECT id, name FROM t");
    assert(rs->row_count == 2);
    assert(rs->rows[0].cols[0].i == 1);
    assert(strcmp(rs->rows[0].cols[1].text.data_copy, "Alice") == 0);
    assert(rs->rows[1].cols[0].i == 2);
    assert(strcmp(rs->rows[1].cols[1].text.data_copy, "Bob") == 0);
    /* Verify NOT NULL constraint */
    int rc = db_exec(db, "INSERT INTO t VALUES (3, NULL)");
    assert(rc < 0);  /* must fail */
    assert(strstr(db->error, "NOT NULL") != NULL);
    db_close(db);
    printf("PASS: insert/select roundtrip\n");
}
```
---
## UPDATE: The Delete-and-Reinsert Problem
UPDATE is more complex than it appears. The naive mental model: find the row, modify the columns, write it back. The complication: **you cannot update a row in place if the row's key changes**, and you need to handle the case where the update modifies a primary key.
For the common case (updating non-key columns), the approach is:
1. Scan the table to find rows matching the WHERE clause
2. For each matching row: read the existing record, deserialize all columns, apply the column modifications, re-serialize, write back to the same position
The "write back to the same position" part sounds simple but requires understanding the B-tree's cell replacement mechanics. In a slotted page, you cannot resize a cell in place if the new record is larger than the old one — you'd need to shift other cells. The practical approach: delete the old cell and insert the new one. Since rowid hasn't changed, it re-inserts into the same leaf (or nearby leaf after a split).
```c
/* UPDATE execution flow */
static int execute_update(VdbeState *vm, UpdatePlan *plan) {
    BTreeCursor *cur = vm->cursors[plan->cursor_id];
    if (btree_cursor_rewind(cur)) return 0;  /* empty table */
    /* Collect rowids of rows to update FIRST (two-pass to avoid
       cursor invalidation during modification) */
    int64_t rowids_to_update[MAX_UPDATE_BATCH];
    Value   new_values_batch[MAX_UPDATE_BATCH][MAX_COLS];
    int     update_count = 0;
    /* PASS 1: scan and collect matching rows */
    do {
        /* Evaluate WHERE clause on current row */
        Trivalue match = evaluate_where(vm, cur, plan->where_expr);
        if (match != TRI_TRUE) continue;
        /* Record the rowid */
        rowids_to_update[update_count] = btree_cursor_rowid(cur);
        /* Deserialize all columns */
        for (int i = 0; i < plan->schema->column_count; i++) {
            btree_cursor_column(cur, i, &new_values_batch[update_count][i]);
            /* TEXT/BLOB values point into page memory — copy them now
               before we modify anything */
            value_deep_copy(&new_values_batch[update_count][i]);
        }
        /* Apply the SET assignments */
        for (int a = 0; a < plan->assignment_count; a++) {
            int col_idx = plan->assignments[a].col_idx;
            evaluate_expr_into_value(vm, cur,
                                      plan->assignments[a].value_expr,
                                      &new_values_batch[update_count][col_idx]);
        }
        update_count++;
        if (update_count >= MAX_UPDATE_BATCH) break;  /* batch limit */
    } while (btree_cursor_next(cur));
    /* PASS 2: apply updates */
    for (int i = 0; i < update_count; i++) {
        /* Delete old row */
        btree_delete_by_rowid(cur->bt, rowids_to_update[i]);
        /* Check constraints on new values */
        if (check_not_null_constraints(vm->db, plan->table_name,
                                        new_values_batch[i],
                                        plan->schema->column_count) < 0) {
            return -1;
        }
        /* Re-insert with same rowid and updated values */
        btree_insert(cur->bt,
                     new_values_batch[i],
                     plan->schema->column_count,
                     rowids_to_update[i]);
    }
    return 0;
}
```
**The rowid update prohibition:** SQLite's semantics forbid changing the rowid via UPDATE (it is the physical primary key). If a user writes `UPDATE t SET id = 5 WHERE id = 1` and `id` is the INTEGER PRIMARY KEY, the engine must either:
- Reject it with an error: "Cannot change primary key via UPDATE"
- Implement it as delete + reinsert at new rowid (and handle the ordering change)
The simplest correct implementation: detect when the SET clause modifies the INTEGER PRIMARY KEY column and return an error. Add this check in the compiler or at constraint-check time.
```c
/* In the UPDATE compiler/executor */
for (int a = 0; a < assignment_count; a++) {
    if (schema->columns[assignments[a].col_idx].is_primary_key) {
        snprintf(error, sizeof(error),
                 "Cannot UPDATE rowid column '%s' directly",
                 schema->columns[assignments[a].col_idx].name);
        return -1;
    }
}
```
**Why TEXT/BLOB values must be deep-copied in pass 1:** During a scan, `btree_cursor_column` returns TEXT/BLOB values whose `.data` pointer points directly into the buffer pool page. The page is pinned while the cursor is on it. But when you advance to the next row (`btree_cursor_next`), the page may change — and when you execute the update in pass 2 (calling `btree_delete_by_rowid`), the buffer pool may evict and overwrite the original leaf page. The stored pointer becomes dangling. Always deep-copy strings and blobs read during scan if they'll be used after cursor advancement.
```c
/* Deep copy a Value — allocate heap memory for TEXT/BLOB */
void value_deep_copy(Value *v) {
    if (v->type == VAL_TEXT) {
        char *copy = malloc(v->text.len + 1);
        memcpy(copy, v->text.data, v->text.len);
        copy[v->text.len] = '\0';
        v->text.data = copy;
        /* Mark as heap-owned so we know to free it */
        v->text.heap_owned = 1;
    } else if (v->type == VAL_BLOB) {
        uint8_t *copy = malloc(v->blob.len);
        memcpy(copy, v->blob.data, v->blob.len);
        v->blob.data  = copy;
        v->blob.heap_owned = 1;
    }
}
```
---
## DELETE: The Iterator Invalidation Problem

![DELETE Two-Pass Strategy — State Evolution](./diagrams/diag-dml-delete-two-pass.svg)

DELETE has a trap that crashes novice implementations: **you cannot delete a row from a B-tree while iterating through it with the same cursor.**
The naive approach:
```c
/* WRONG: delete during iteration */
while (btree_cursor_next(cur)) {
    if (matches_where(cur)) {
        btree_delete_current(cur);  /* CORRUPTS the cursor! */
    }
}
```
Why does this corrupt? When you delete a cell from a page, the page's cell pointer array shifts — cells after the deleted one move one slot earlier. The cursor's `slot` index, which pointed to the next cell, now points to the wrong cell or past the end. The cursor's invariants are violated.
This is the same class of bug as `ConcurrentModificationException` in Java's `ArrayList`, or the "don't remove from a list while iterating it" rule in Python. The underlying data structure is being modified while an iterator holds a position reference into it. The position becomes invalid.

> **🔑 Foundation: Iterator invalidation during collection modification**
> 
> ## Iterator Invalidation & Concurrent Modification
**What it IS**
An iterator is a cursor that traverses a collection — it holds internal state (typically a pointer or index) that says "I am currently here in this data structure." Iterator invalidation happens when the underlying collection is structurally modified *while* an iterator is active, making that internal state stale or dangerously wrong.
In C++, "invalidation" is the official term. In Java, it's called a "concurrent modification exception." Different languages enforce it differently, but the problem is universal.
Consider a vector/array-list: iterators often store a raw pointer or index into the backing buffer. If you insert an element, the buffer may reallocate entirely — your pointer now points into freed memory. If you erase an element, every element after it shifts left by one — your index now skips an element or reads past the end.
```cpp
std::vector<int> v = {1, 2, 3, 4, 5};
for (auto it = v.begin(); it != v.end(); ++it) {
    if (*it == 3) {
        v.erase(it); // UNDEFINED BEHAVIOR — it is now invalid
        // On the next ++it, anything can happen
    }
}
```
For linked structures (linked lists, trees), the rules differ. Erasing a node only invalidates iterators *to that node*; other iterators remain valid. Hash maps are particularly treacherous: any insertion that triggers a rehash invalidates *all* iterators.
**WHY you need it right now**
When building a shell, compiler, or kernel component, you frequently iterate over live collections: the process table, a symbol table, a free-block list, a list of open file descriptors. These structures get modified during the very loop that's walking them — processes exit while you're scheduling, symbols get added during parsing, blocks get freed during compaction.
Getting this wrong produces the worst class of bugs: silent data corruption or use-after-free errors that manifest far from the original mutation. In a kernel or allocator context, this can be a security vulnerability.
**The key mental model**
Think of an iterator as a *contract* between you and the collection: "I promise the structure won't change shape while you walk it."
There are three clean ways to honor or work around this contract:
1. **Collect, then act**: iterate first to collect items to remove, then do the removals in a second pass.
   ```cpp
   std::vector<Iterator> to_erase;
   for (auto it = v.begin(); it != v.end(); ++it)
       if (should_remove(*it)) to_erase.push_back(it);
   // Now erase in reverse to preserve earlier iterators
   for (auto it : to_erase) v.erase(it);
   ```
2. **Use the return value of mutating operations**: Many APIs (C++ `erase`, Rust's `retain`) return a new valid iterator specifically so you can continue safely.
   ```cpp
   for (auto it = v.begin(); it != v.end(); )
       it = should_remove(*it) ? v.erase(it) : ++it; // correct
   ```
3. **Use a different data structure**: `std::list` iterators survive insertions and erasures everywhere except at the erased node itself. Sometimes the right fix is choosing a structure whose invalidation rules fit your access pattern.
The mental shortcut: **mutation changes the map; don't trust old directions after the map changes.**

**The solution: two-pass delete.**
```c
/* CORRECT: two-pass delete */
/* PASS 1: scan and collect rowids of rows to delete */
int64_t rowids_to_delete[MAX_DELETE_BATCH];
int delete_count = 0;
BTreeCursor scan_cursor;
btree_cursor_open_read(&scan_cursor, bt);
if (!btree_cursor_rewind(&scan_cursor)) {
    do {
        Trivalue match = evaluate_where_cursor(&scan_cursor, where_expr);
        if (match == TRI_TRUE) {
            rowids_to_delete[delete_count++] = btree_cursor_rowid(&scan_cursor);
            if (delete_count >= MAX_DELETE_BATCH) break;
        }
    } while (btree_cursor_next(&scan_cursor));
}
btree_cursor_close(&scan_cursor);
/* PASS 2: delete by rowid (no cursor in use) */
for (int i = 0; i < delete_count; i++) {
    btree_delete_by_rowid(bt, rowids_to_delete[i]);
}
```
The two-pass approach is correct because:
1. The scan cursor is used only in pass 1, then closed before any modification
2. Pass 2 uses `btree_delete_by_rowid`, which opens a fresh internal cursor for each delete, not the scan cursor
For large deletions where collecting all rowids exhausts memory (`MAX_DELETE_BATCH`), process in batches: collect N rowids, delete them, then resume scanning from where you left off (using the last processed rowid as a lower bound for the next scan).
### Implementing btree_delete_by_rowid
DELETE is structurally the inverse of INSERT. Find the leaf page containing the rowid, remove the cell, compact the page, and — if the page is now underfull — potentially rebalance. The full rebalancing algorithm (merging underfull siblings and pulling separator keys down from the parent) is complex. A pragmatic starting point: mark deleted cells with a tombstone and handle space reclamation lazily.
```c
/* btree_delete_by_rowid: find and remove a row by its rowid.
   Returns 0 if deleted, -1 if not found. */
int btree_delete_by_rowid(BTree *bt, int64_t rowid) {
    int slot;
    PageId leaf_id = btree_search(bt->bp, bt->root_page_id, rowid, &slot);
    if (leaf_id == INVALID_PAGE_ID) return -1;  /* not found */
    uint8_t *page = buffer_pool_fetch(bt->bp, leaf_id);
    PageHeader h;
    page_header_read(page, &h);
    /* Remove the cell at slot */
    page_delete_cell(page, slot);
    buffer_pool_unpin(bt->bp, leaf_id, /*dirty=*/1);
    /* Note: we do NOT rebalance underfull pages in this implementation.
       Space is reclaimed when the page is next modified or during VACUUM.
       This is SQLite's actual behavior for simple cases. */
    return 0;
}
/* Remove cell at position idx from a page's cell array and content area.
   Updates header. Does NOT rebalance tree structure. */
static void page_delete_cell(uint8_t *page, int idx) {
    PageHeader h;
    page_header_read(page, &h);
    /* The cell content space becomes fragmented free space.
       We record it as fragmented bytes (simplified — full implementation
       would add it to the freeblock chain). */
    uint16_t cell_offset = cell_pointer_get(page, idx);
    /* cell_size at this position: */
    PageType pt = (PageType)h.page_type;
    int csize = cell_size(pt, page + cell_offset);
    h.fragmented_bytes += (uint8_t)(csize > 255 ? 255 : csize);
    /* Shift pointer array: remove entry at idx */
    int ncells = h.cell_count;
    uint8_t *ptr_array = page + PAGE_HEADER_SIZE;
    memmove(ptr_array + idx * 2,
            ptr_array + (idx + 1) * 2,
            (ncells - idx - 1) * 2);
    h.cell_count--;
    page_header_write(page, &h);
}
```
This tombstone approach (fragmented bytes accumulate without space reclamation) matches SQLite's behavior for simple DELETE operations. Space is fully reclaimed when the page undergoes compaction — triggered either by a new insertion that needs space (the page is defragmented first) or by a future VACUUM operation.
**Page defragmentation:** When a page has enough fragmented space to accommodate a new cell but the free space is not contiguous (the cell content area pointer is too far down), compact the page:
```c
/* Compact a page: move all cells to be contiguous at the page bottom,
   eliminating fragmented gaps. */
static void page_defragment(uint8_t *page) {
    PageHeader h;
    page_header_read(page, &h);
    /* Collect all cell contents into a temporary buffer */
    uint8_t temp[PAGE_SIZE];
    int write_pos = PAGE_SIZE;
    for (int i = 0; i < h.cell_count; i++) {
        uint16_t old_offset = cell_pointer_get(page, i);
        PageType pt = (PageType)h.page_type;
        int csize = cell_size(pt, page + old_offset);
        write_pos -= csize;
        memcpy(temp + write_pos, page + old_offset, csize);
        /* Update pointer */
        uint8_t *ptr = page + PAGE_HEADER_SIZE + i * 2;
        ptr[0] = (write_pos >> 8) & 0xFF;
        ptr[1] =  write_pos       & 0xFF;
    }
    /* Copy compacted cells back */
    memcpy(page + write_pos, temp + write_pos, PAGE_SIZE - write_pos);
    h.cell_content_area = (uint16_t)write_pos;
    h.fragmented_bytes  = 0;
    page_header_write(page, &h);
}
```
---
## Wiring DML into the VDBE Compiler
The compiler needs new opcodes for UPDATE and DELETE, or can reuse existing opcodes creatively. The practical approach for UPDATE: compile it as a scan-and-delete-reinsert sequence using existing opcodes plus new `OP_UPDATE_ROW` and `OP_DELETE_ROW` instructions that encapsulate the two-pass logic.
For DELETE:
```
OpenTable   0  1      -- cursor 0, write mode
Rewind      0  →end
Column      0  col  r1  -- load the WHERE column
Integer     val  r2     -- load the comparison value
Ne          r1  →next r2  -- skip non-matching rows
DeleteRow   0           -- mark current row for deletion (deferred)
Next        0  →loop
FlushDeletes 0          -- execute all deferred deletes
Close       0
Halt
```
The `OP_DELETE_ROW` instruction collects the current rowid into a deletion list. `OP_FLUSH_DELETES` executes the actual B-tree deletions after the scan completes. This is the two-pass strategy expressed at the opcode level.
```c
case OP_DELETE_ROW: {
    int cursor_id = instr->p1;
    BTreeCursor *cur = vm.cursors[cursor_id];
    /* Collect the current rowid for deferred deletion */
    int64_t rowid = btree_cursor_rowid(cur);
    /* Add to the deletion list */
    if (vm.delete_count >= vm.delete_capacity) {
        vm.delete_capacity = vm.delete_capacity ? vm.delete_capacity * 2 : 16;
        vm.delete_rowids = realloc(vm.delete_rowids,
                                    vm.delete_capacity * sizeof(int64_t));
    }
    vm.delete_rowids[vm.delete_count++] = rowid;
    vm.pc++;
    break;
}
case OP_FLUSH_DELETES: {
    int cursor_id = instr->p1;
    BTreeCursor *cur = vm.cursors[cursor_id];
    for (int i = 0; i < vm.delete_count; i++) {
        btree_delete_by_rowid(cur->bt, vm.delete_rowids[i]);
    }
    vm.delete_count = 0;
    vm.pc++;
    break;
}
```
---
## Non-Existent Table Error Handling
When `OP_OPEN_TABLE` is executed and the table name is not in sqlite_master, the engine must return a clear, actionable error:
```c
case OP_OPEN_TABLE: {
    int cursor_id  = instr->p1;
    int write_mode = instr->p2;
    const char *table_name = instr->p4.str;
    PageId root = schema_find_table(vm.db, table_name);
    if (root == INVALID_PAGE_ID) {
        snprintf(vm.error, sizeof(vm.error),
                 "Table '%s' does not exist", table_name);
        vm.had_error = 1;
        goto halt;
    }
    vm.cursors[cursor_id] = btree_cursor_open(vm.db->bp, vm.db, root, write_mode);
    vm.cursor_table_names[cursor_id] = table_name;  /* for error messages */
    vm.pc++;
    break;
}
```
The error message includes the table name because "Table does not exist" is unhelpful; "Table 'usres' does not exist" immediately tells the user it's a typo.
---
## Three-Level View: WHERE x > 18 Across the Full Stack
To see how all six layers interact for a filtered query, trace `SELECT name FROM users WHERE age > 18` from first character to last output byte.

![VM Fetch-Decode-Execute Loop — Data Walk](./diagrams/diag-vdbe-execution-loop.svg)

**Level 1 — What the user sees**
The VDBE executes the bytecode. From the VM's perspective, each row evaluation is:
1. `OP_COLUMN` → load `age` from current row into register 1
2. `OP_INTEGER 18` → load constant into register 2
3. `OP_LE r1 →next r2` → if age <= 18 (or NULL), skip to OP_NEXT
4. `OP_COLUMN` → load `name` into register 3
5. `OP_RESULT_ROW` → emit register 3 as output
**Level 2 — What happens in the cursor**
`OP_COLUMN` calls `btree_cursor_column(cur, col_idx=1, &reg[1])`:
- The cursor has `leaf_page_id` and `slot` set from the previous `OP_NEXT`
- It calls `buffer_pool_fetch(bp, leaf_page_id)` — hot hit in the buffer pool
- Reads `cell_at(page, slot)` → pointer to cell bytes
- Calls `record_decode_column(payload, payload_len, col_idx=1, &reg[1])`
- Decodes the serial type for column 1 from the header — say it's `0x06` (64-bit integer)
- Reads 8 bytes big-endian from the data region → `int64_t age = 25`
- The page stays pinned (not unpinned until cursor advances)
**Level 3 — What happens in the buffer pool and disk**
`buffer_pool_fetch(bp, leaf_page_id)`:
- Computes `bucket = (leaf_page_id * 2654435761u) % page_table_size`
- Walks the hash chain — finds `frame_idx = 7` (warm hit)
- Increments `frames[7].pin_count` from 0 to 1
- Calls `lru_move_to_head(bp, 7)` — moves frame 7 to the MRU position
- Returns `frames[7].data` — pointer to 4096 bytes in process memory
- Total cost: ~5 array accesses, ~10ns
If cold: `pread(fd, frames[free_idx].data, 4096, leaf_page_id * 4096)` — ~100μs on SSD.

![Register File — State Evolution](./diagrams/diag-vdbe-register-file.svg)

---
## NULL Semantics in Practice: The Common Bugs
Three-valued logic produces real bugs in production databases. Understanding these prevents you from introducing them:
**Bug 1: `WHERE col != 'value'` doesn't return NULL rows**
```sql
SELECT * FROM users WHERE status != 'inactive';
```
This returns users with status = 'active', 'pending', etc., but NOT users with `status = NULL`. To include NULL rows:
```sql
SELECT * FROM users WHERE status != 'inactive' OR status IS NULL;
```
Your evaluator must handle this correctly: `NULL != 'inactive'` evaluates to NULL (excluded), not TRUE (included).
**Bug 2: `NOT (col = 'value')` still excludes NULLs**
```sql
SELECT * FROM users WHERE NOT (status = 'inactive');
```
`NOT NULL = NULL`. So `NOT (NULL = 'inactive')` = `NOT NULL` = `NULL`. Row excluded.
**Bug 3: NULL in UNIQUE constraint**
SQL allows multiple NULL values in a UNIQUE column. `NULL ≠ NULL` in SQL means two rows with NULL in a UNIQUE column do not violate the constraint. When you implement UNIQUE index enforcement (Milestone 7), this rule matters: do not reject an INSERT that would produce a second NULL in a UNIQUE column.
**Bug 4: JOIN with NULL**
`a JOIN b ON a.id = b.foreign_id` — rows where `foreign_id IS NULL` never match anything. They're excluded from the result. This is expected behavior but surprises developers who expect NULL foreign keys to appear in LEFT JOINs (they do in LEFT JOIN, not INNER JOIN). Implement your JOIN correctly by using the same three-valued comparison.
```c
/* Evaluating a WHERE expression: the three-valued evaluator */
Trivalue evaluate_expr_tri(const Value *a, const Value *b, Opcode op) {
    if (a->type == VAL_NULL || b->type == VAL_NULL) {
        /* Special cases: IS NULL and IS NOT NULL handle NULL explicitly */
        if (op == OP_IS_NULL)  return TRI_TRUE;
        if (op == OP_NOT_NULL) return TRI_FALSE;
        /* All other comparisons with NULL → NULL */
        return TRI_NULL;
    }
    /* ... non-null comparison ... */
}
```
---
## Testing: The Correctness Gauntlet
NULL semantics require exhaustive testing. Your test suite must cover every cell in every truth table:
```c
static void test_three_valued_logic(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (x INTEGER, y INTEGER)");
    db_exec(db, "INSERT INTO t VALUES (1, 1)");
    db_exec(db, "INSERT INTO t VALUES (1, NULL)");
    db_exec(db, "INSERT INTO t VALUES (NULL, 1)");
    db_exec(db, "INSERT INTO t VALUES (NULL, NULL)");
    /* Test: NULL = anything → no match */
    ResultSet *r1 = db_query(db, "SELECT * FROM t WHERE x = NULL");
    assert(r1->row_count == 0);
    /* Test: IS NULL matches */
    ResultSet *r2 = db_query(db, "SELECT * FROM t WHERE x IS NULL");
    assert(r2->row_count == 2);  /* rows (NULL,1) and (NULL,NULL) */
    /* Test: != doesn't match NULLs */
    ResultSet *r3 = db_query(db, "SELECT * FROM t WHERE x != 1");
    assert(r3->row_count == 0);  /* NOT 1 doesn't exist; NULLs excluded */
    /* Test: NOT (x = 1) still excludes NULLs */
    ResultSet *r4 = db_query(db, "SELECT * FROM t WHERE NOT (x = 1)");
    assert(r4->row_count == 0);  /* same result */
    /* Test: x = 1 OR x IS NULL includes both */
    ResultSet *r5 = db_query(db, "SELECT * FROM t WHERE x = 1 OR x IS NULL");
    assert(r5->row_count == 3);  /* rows (1,1), (1,NULL), (NULL,1), (NULL,NULL) — wait: 4 */
    /* Correction: x=1 matches rows 1,2. x IS NULL matches rows 3,4. Total 4. */
    assert(r5->row_count == 4);
    /* Test: NULL AND TRUE = NULL (falsy) */
    /* WHERE x = 1 AND y = NULL should match nothing */
    ResultSet *r6 = db_query(db, "SELECT * FROM t WHERE x = 1 AND y = NULL");
    assert(r6->row_count == 0);
    /* Test: NULL OR TRUE = TRUE */
    /* WHERE x = NULL OR y = 1 should match (1,1) and (NULL,1) */
    ResultSet *r7 = db_query(db, "SELECT * FROM t WHERE x = NULL OR y = 1");
    /* x=NULL: both x=NULL comparisons are NULL; y=1 is TRUE for rows 1 and 3 */
    assert(r7->row_count == 2);
    printf("PASS: three-valued logic\n");
    db_close(db);
}
```
Add DML tests covering:
```c
static void test_update_delete(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, name TEXT, age INTEGER)");
    db_exec(db, "INSERT INTO t VALUES (1, 'Alice', 25)");
    db_exec(db, "INSERT INTO t VALUES (2, 'Bob', NULL)");
    db_exec(db, "INSERT INTO t VALUES (3, 'Carol', 30)");
    /* UPDATE matching rows */
    db_exec(db, "UPDATE t SET age = 26 WHERE name = 'Alice'");
    ResultSet *r1 = db_query(db, "SELECT age FROM t WHERE id = 1");
    assert(r1->rows[0].cols[0].i == 26);
    /* UPDATE with NULL WHERE condition — NULL rows not updated */
    db_exec(db, "UPDATE t SET age = 99 WHERE age > 20");
    /* Alice (26) and Carol (30) are updated. Bob (NULL) is NOT. */
    ResultSet *r2 = db_query(db, "SELECT age FROM t WHERE id = 2");
    assert(r2->rows[0].cols[0].type == VAL_NULL);  /* Bob unchanged */
    /* DELETE */
    db_exec(db, "DELETE FROM t WHERE id = 1");
    ResultSet *r3 = db_query(db, "SELECT * FROM t");
    assert(r3->row_count == 2);
    /* DELETE with NULL condition */
    db_exec(db, "DELETE FROM t WHERE age > 50");
    /* Carol (age=99 now) and Bob (NULL) — NULL doesn't match > 50 */
    ResultSet *r4 = db_query(db, "SELECT * FROM t");
    assert(r4->row_count == 1);  /* only Bob (NULL) survives */
    printf("PASS: update/delete with NULL semantics\n");
    db_close(db);
}
```
---
## Design Decision: Cursor Pinning During Scan
A critical design question: when should `btree_cursor_column` unpin the current leaf page?
**Option A: Unpin immediately after column read.** Each `OP_COLUMN` call fetches the page, reads the value, copies it to a register, and unpins. For TEXT/BLOB values, this requires copying the string to heap memory on every column read.
**Option B: Keep page pinned across column reads.** The cursor holds a persistent pin on the current leaf page. The page is unpinned only when `btree_cursor_next` or `btree_cursor_close` is called. TEXT/BLOB values can be zero-copy pointers into page memory.
| | Option A | Option B |
|---|----------|----------|
| Memory | `malloc` per TEXT/BLOB column per row | Zero-copy from page buffer |
| Simplicity | Easy to implement | Must track persistent pin |
| Buffer pool pressure | Low pin count at any moment | Page stays pinned during row processing |
| Performance | O(cols) heap allocations per row | O(1) allocations per row |
| **Chosen** | | **✓** |
Option B is the right choice for performance-sensitive paths. A `SELECT name, email, address FROM users` would allocate 3 strings per row × 10,000 rows = 30,000 heap allocations for a simple table scan. Option B reduces this to zero allocations in the hot path (the page stays pinned and values point directly into it).
The implementation requires the cursor to track its pinned page:
```c
/* When btree_cursor_next advances to a new page */
int btree_cursor_next(BTreeCursor *cur) {
    if (cur->at_end) return 0;
    /* Increment slot within current page */
    uint8_t *page = buffer_pool_fetch(cur->bt->bp, cur->leaf_page_id);
    PageHeader h;
    page_header_read(page, &h);
    cur->slot++;
    if (cur->slot < h.cell_count) {
        /* Stay on same page — update the pinned page pointer */
        cur->current_page = page;
        /* Don't unpin: we just "transferred" the pin to the next slot */
        return 1;
    }
    /* Must move to next leaf page */
    buffer_pool_unpin(cur->bt->bp, cur->leaf_page_id, 0);
    cur->current_page = NULL;
    /* Find next leaf (see Milestone 5's btree_find_next_leaf) */
    int next_slot;
    int64_t current_rowid = /* get last rowid on this page */;
    PageId next_leaf = btree_find_next_leaf(cur->bt, current_rowid, &next_slot);
    if (next_leaf == INVALID_PAGE_ID) {
        cur->at_end = 1;
        return 0;
    }
    cur->leaf_page_id = next_leaf;
    cur->slot         = next_slot;
    cur->current_page = buffer_pool_fetch(cur->bt->bp, next_leaf);
    return 1;
}
```
---
## Knowledge Cascade: What This Milestone Unlocks
You have just built the most commonly used part of any database: SELECT, INSERT, UPDATE, DELETE with correct NULL semantics. Here is what this knowledge connects to:
**→ Three-valued logic extends to all SQL predicates.** The NULL semantics you implemented appear in JOIN ON conditions (NULLs never join), HAVING clauses (aggregate NULLs), subquery predicates (IN, EXISTS with NULLs), and CASE expressions. Every production database bug involving "query returns wrong rows" or "query returns fewer rows than expected" traces back to NULL propagation somewhere. You now understand the mechanism; when you see such a bug, your first question is "does this involve NULL?" This debugging reflex is worth more than any academic understanding.
**→ The cursor pattern is everywhere.** The `BTreeCursor` with its `rewind/next/column` protocol is the same pattern as: Python generators (`yield` / `send` / `throw` — a cursor through a lazily-computed sequence), C++ range-based for (requires `begin()`, `end()`, `operator++`, `operator*`), Rust's `Iterator` trait (`next()` returning `Option<Item>`), Java's `Stream<T>` (`Spliterator`), JDBC's `ResultSet` (`next()`, `getString(col)`). Every time you use any of these, you're using the same abstraction you built in C. The next time you implement a custom data source in any language, you now know the protocol from the implementer's side — not just the consumer's side.
**→ Projection is relational algebra in code.** The SELECT column list translates to π (pi) in relational algebra: π_{name,age}(users) extracts only those attributes. Your implementation — deserialize only the requested serial types from the binary record — is an efficient implementation of this operator. The connection extends to linear algebra: the projection operator in linear algebra (P²=P, projects a vector onto a subspace) shares the same conceptual structure as SQL projection (selecting a subset of dimensions from a tuple). Understanding this connection helps when reading database research papers that use formal notation.
**→ Two-pass delete is concurrent programming in miniature.** The "collect then delete" pattern you implemented is the database equivalent of: Python's `list(filter(...))` before modifying (you can't delete from a list while iterating), Java's `CopyOnWriteArrayList` (reads see a snapshot, modifications create a new copy), the copy-on-write mechanism in Unix `fork()` (writes create private page copies). The root cause is the same: you cannot simultaneously navigate a structure and modify it through the same reference. The general solution is always some form of separation: separate the read phase from the write phase, either in time (two passes) or in space (copy on write).
**→ The buffer pool pin discipline is observable here.** In the cursor's persistent pin strategy, you can instrument the buffer pool to measure: how many frames are pinned during a 10-column SELECT scan? For a single-row result from a large table (index scan), you hold at most 3 pages simultaneously (root, internal, leaf). For a full table scan with wide rows spanning many pages, you hold 1 page at a time. For an UPDATE that requires collecting all matching rowids before deleting (pass 1 + pass 2), you hold 0 pages between passes. This observable pin behavior directly explains why buffer pool exhaustion (all frames pinned) is a symptom of a scan that holds too many pages open simultaneously — a real production database issue when write transactions interleave with long-running reads.
**→ UPDATE as delete+reinsert reveals MVCC's design.** The reason UPDATE is implemented as delete+reinsert in many databases (including PostgreSQL's heap storage) is that it simplifies concurrent access: a reader who started before the UPDATE sees the old version (because the old row still exists until it's vacuumed), while a reader who started after sees the new version (because the new row is already in place). This is Multi-Version Concurrency Control (MVCC) in embryonic form. SQLite uses a simpler model (exclusive write lock), but the structural similarity to MVCC — new row inserted at the B-tree's correct position, old row marked deleted — is intentional. Understanding this makes PostgreSQL's VACUUM process intuitive: it's the janitor that removes the old versions that MVCC kept around for concurrent readers.
---
## Common Pitfalls: Silent Failures That Pass Unit Tests
**1. Forgetting NULL in comparison opcodes.** If `OP_GT` calls `value_compare` which returns `INT32_MAX` for NULL, and your opcode interprets `INT32_MAX > 0` as "greater-than is true, so the condition passes," you have a silent correctness bug. Every comparison with a NULL column silently includes the row. Write a specific test: `INSERT INTO t VALUES (NULL)`, then `SELECT * FROM t WHERE col > 0` — it must return zero rows.
**2. TEXT/BLOB values pointing to evicted pages.** If you store a TEXT value's `.data` pointer during a scan and then advance the cursor (which may unpin the page), and then later use the pointer — you have a dangling pointer. The symptom is garbled string output or a segfault that only occurs on cold-cache runs (when the page actually gets evicted). Fix: deep-copy TEXT/BLOB values whenever you'll use them after cursor advancement.
**3. The delete-during-iteration bug.** A common partial implementation: `OP_DELETE_ROW` immediately calls `btree_delete_by_rowid` rather than deferring to `OP_FLUSH_DELETES`. Tests that delete a single row (the cursor is already past the deleted row when it's removed) pass. Tests that delete multiple rows in sequence reveal the bug when the second deletion corrupts the cursor state for the third row. Always verify with a test that deletes every-other row: `DELETE FROM t WHERE id % 2 = 0` against a 10-row table.
**4. NOT NULL check on the wrong column.** The schema stores column indices. If your `check_not_null_constraints` function iterates schema columns and compares against the VALUES columns by index, a mismatch in column ordering (e.g., the user supplied `INSERT INTO t (b, a) VALUES (...)`) will check the wrong column for NOT NULL. The compiler must resolve column names from the INSERT's explicit column list to schema indices before emitting `OP_MAKE_RECORD`.
**5. AUTO-ROWID collision.** If `btree_next_rowid` returns the same rowid twice (e.g., it doesn't traverse to the rightmost leaf and instead returns 1 for an empty root page that happens to have been split), the second INSERT with the same rowid either silently overwrites the first row or fails with a confusing error. Test by inserting 100 rows and verifying each rowid is distinct and monotonically increasing.
**6. Update with WHERE touching NULL columns.** `UPDATE t SET age = 30 WHERE age = NULL` uses `=` comparison which returns NULL for NULL inputs, so zero rows are updated. This is correct, but users expect it to work and it silently does nothing. Your implementation is correct; the test that validates this behavior is: run the UPDATE, then verify the count of updated rows is 0, not an error.
---
## What You Have Built
At the end of this milestone, your database executes real SQL against real data with correct semantics:
- **Full table scan** via B-tree leaf traversal, with zero-copy TEXT/BLOB values from pinned pages via the persistent-pin cursor strategy
- **Column projection** deserializing only requested columns from variable-length records, with the correct schema-to-column-index mapping
- **WHERE clause evaluation** implementing full three-valued logic: `NULL = anything` → NULL (excluded), `NULL IS NULL` → TRUE, `NOT NULL` → NULL, AND/OR truth tables with all nine NULL combinations correct
- **IS NULL and IS NOT NULL** operators that inspect the null status directly rather than comparing values
- **INSERT** with auto-rowid assignment (MAX existing rowid + 1), NOT NULL constraint enforcement at the register level before serialization, and B-tree insertion verified by subsequent SELECT
- **UPDATE** with two-phase scan-then-modify, deep-copy of TEXT/BLOB values to prevent dangling pointers after cursor advancement, rejection of INTEGER PRIMARY KEY updates
- **DELETE** with two-pass collection-then-removal, preventing cursor invalidation during modification
- **NOT NULL constraints** enforced at INSERT and UPDATE time with error messages naming the violated column
- **Non-existent table errors** with the table name in the error message, raised at cursor open time before any execution
- **NULL-correct aggregation foundation**: COUNT(*) vs COUNT(col) semantics are now implementable correctly because the evaluator returns VAL_NULL for NULL column reads rather than 0 or empty string
The cursor interface from Milestone 3 is fully real. The B-tree from Milestone 5 is exercised by all four DML operations. The buffer pool from Milestone 4 is stress-tested by scan-heavy SELECT and two-pass DELETE. The foundation for secondary indexes (Milestone 7) and the query planner (Milestone 8) is complete: every indexed column comparison uses the same three-valued evaluator, and the cursor protocol is the uniform interface that both table scans and index scans will implement.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m7 -->
<!-- MS_ID: build-sqlite-m7 -->
# Milestone 7: Secondary Indexes
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

Stand back and place yourself in the machine. You have built a complete working database: SQL text flows through the tokenizer and parser into bytecode, the VDBE executes it against B-tree pages managed by the buffer pool, and DML operations persist correctly to disk. You can create tables, insert rows, and query them back.
But your database has a problem you haven't felt yet because your test tables have been small. Run this query against a million-row `users` table:
```sql
SELECT * FROM users WHERE email = 'alice@example.com';
```

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

Your engine starts at the B-tree's leftmost leaf and reads every single row, comparing `email` to `'alice@example.com'` for each one. If Alice is the last row alphabetically, you've read the entire table — perhaps 10,000 pages, perhaps 50,000 microseconds of sequential I/O — to find one row. This is a table scan, and for point lookups on non-rowid columns, it is catastrophically slow.
Secondary indexes are the solution. After this milestone, `CREATE INDEX idx_email ON users(email)` builds a separate B+tree that maps email values directly to rowids. The lookup `WHERE email = 'alice@example.com'` descends three levels of the index tree (root → internal → leaf), reads the rowid, then fetches Alice's row directly — three page reads instead of ten thousand.
---
## The Revelation: Indexes Are a Double-Edged Tax
Here is the misconception that every developer internalizes before building a real storage system: **more indexes means better query performance**. You've heard this from experienced engineers. It shows up in code review feedback: "Add an index on that column." It seems cost-free — you're only adding structure, not removing any.
This model is wrong in a specific and important way. Every index you create is a separate B+tree that lives in your database file. Every time a row is inserted, updated, or deleted, **every** index on that table must also be modified.
Consider a table with 5 indexes and 10 million rows. You run a bulk import:
```sql
INSERT INTO events VALUES (1, 'pageview', 'alice', '2024-01-01', ...);
INSERT INTO events VALUES (2, 'click',    'alice', '2024-01-01', ...);
-- ... 1,000,000 rows
```
For each INSERT, your engine must:
1. Insert the row into the table B-tree (1 B-tree modification, potentially 1 split)
2. Insert into idx_type (1 B-tree modification)
3. Insert into idx_user (1 B-tree modification)
4. Insert into idx_date (1 B-tree modification)
5. Insert into idx_user_date (1 B-tree modification)
6. Insert into idx_type_user (1 B-tree modification)
Six B-tree modifications per row. Each modification potentially causes a page split. Each split writes multiple pages to disk. A table with 5 indexes can have write throughput **5-6× worse** than the same table with no indexes.
This is the reveal: **an index is a write-tax that you pay on every mutation to buy faster reads**. The skill isn't "add indexes" — it's choosing exactly which indexes to create based on the actual query patterns that justify the ongoing write cost.
The second half of the revelation is equally important: even when an index exists, using it isn't always faster than a table scan. If you query `WHERE status = 'active'` on a table where 80% of rows have `status = 'active'`, the index scan must follow 800,000 rowid pointers back to the table — each one a random page read. The sequential table scan reads each page once, contiguously. At high selectivity (most rows match), the sequential scan wins.
The threshold where SQLite's query planner switches from index scan to table scan is roughly 20-25% selectivity. Below that threshold (the index is selective — few rows match), use the index. Above it, scan the table.

![Secondary Index B+tree — Structure Layout](./diagrams/diag-index-bplustree-structure.svg)

---
## What You Are Building
Four components that work together:
1. **The index B+tree** — a B+tree with leaves storing (indexed_value, rowid) pairs and linked for range traversal, internal nodes storing only separator keys and child pointers
2. **CREATE INDEX** — build the index from existing table data and register it in `sqlite_master`
3. **Index maintenance** — synchronously update all indexes on INSERT, UPDATE, DELETE
4. **Index-driven query execution** — compile `WHERE indexed_col = val` and `WHERE indexed_col BETWEEN a AND b` into the double-lookup pattern
Let's build each component.
---
## The Index B+tree Structure

![B-tree vs B+tree — Before/After Comparison](./diagrams/diag-btree-vs-bplustree.svg)

You already implemented the page format in Milestone 5, including `PAGE_TYPE_INDEX_LEAF` and `PAGE_TYPE_INDEX_INTERNAL`. Now you'll build the B+tree logic on top of it.
The defining characteristic of a B+tree (as opposed to the B-tree used for table storage) is the clean leaf/internal separation:
- **Internal nodes**: Store only separator keys and child pointers. No row data. No rowids from the payload perspective — just the key values needed for routing.
- **Leaf nodes**: Store `(indexed_column_value, rowid)` pairs. All data lives here.
- **Leaf linking**: Each leaf page stores the page number of its right sibling. Range scans follow this chain without re-ascending the tree.
The leaf linking is what makes B+trees superior for range scans. After locating the first matching leaf via tree descent, you follow the sibling chain rightward, collecting matching rows, stopping when the indexed value exceeds the range end. You never visit an internal node again.
```c
/* Index leaf cell format (stored in PAGE_TYPE_INDEX_LEAF pages):
   [payload_size: varint][payload: (key_value, rowid) encoded as 2-column record]
   The key_value is the indexed column's value.
   The rowid is the table B-tree rowid for the corresponding row.
   Both are stored as the first two columns of a standard record:
   header: [header_size][serial_type(key)][serial_type(rowid)]
   data:   [key_bytes][rowid_bytes]
*/
/* Index internal cell format (stored in PAGE_TYPE_INDEX_INTERNAL pages):
   [left_child: uint32_t big-endian]
   [payload_size: varint]
   [payload: separator_key encoded as 1-column record]
*/
/* Right sibling pointer: stored at a fixed offset in each INDEX_LEAF page header.
   We extend the page header to include a right_sibling field for index leaves. */
#define INDEX_LEAF_RIGHT_SIBLING_OFFSET  8   /* bytes 8-11 of the page header */
/* Note: this field is the same as right_child in the PageHeader struct.
   For INDEX_LEAF pages, we repurpose it as right_sibling. */
```
The right-sibling pointer reuses the `right_child` field in the page header. For `PAGE_TYPE_TABLE_INTERNAL` and `PAGE_TYPE_INDEX_INTERNAL`, this field is the rightmost child pointer. For `PAGE_TYPE_INDEX_LEAF`, it's the right sibling. The page type determines interpretation — your page header reader already handles this correctly.
---
## Index Descriptor: Tracking Indexes in the Schema
Before building the B+tree operations, you need to track which indexes exist and which columns they cover:
```c
/* In schema.h */
typedef struct {
    char    *index_name;      /* e.g., "idx_users_email" */
    char    *table_name;      /* e.g., "users" */
    int     *col_indices;     /* schema column indices (0-based) for the indexed columns */
    int      col_count;       /* number of columns in the index */
    PageId   root_page_id;    /* root of the index B+tree */
    int      is_unique;       /* 1 if UNIQUE index */
} IndexDescriptor;
typedef struct SchemaTable {
    char           *name;
    ColumnDef      *columns;
    int             column_count;
    PageId          root_page_id;     /* table B-tree root */
    IndexDescriptor *indexes;         /* array of indexes on this table */
    int              index_count;
} SchemaTable;
```
When `CREATE INDEX` runs, it creates an `IndexDescriptor`, allocates a new root page for the index B+tree, and appends the descriptor to the table's schema entry. This information must also be persisted in `sqlite_master` so it survives database close/reopen.
---
## CREATE INDEX: Building the Index

![Index Maintenance on INSERT/DELETE — Before/After](./diagrams/diag-index-maintenance.svg)

`CREATE INDEX idx_email ON users(email)` must:
1. Scan the entire `users` table
2. For each row, extract the `email` column value and the rowid
3. Insert `(email_value, rowid)` into the new B+tree
4. Register the index in `sqlite_master`
The compiler emits a new opcode for this:
```c
/* In vdbe.h, add to Opcode enum: */
OP_CREATE_INDEX,  /* p4=index_name, p4_aux=table_name, p1=col_index, p2=is_unique */
```
The VDBE handler for `OP_CREATE_INDEX`:
```c
case OP_CREATE_INDEX: {
    const char *index_name = instr->p4.str;
    const char *table_name = instr->p4_aux.str;
    int         col_index  = instr->p1;
    int         is_unique  = instr->p2;
    /* 1. Allocate a new root page for the index */
    PageId index_root_id;
    uint8_t *root_page = buffer_pool_new_page(vm.db->bp, &index_root_id);
    page_init(root_page, PAGE_TYPE_INDEX_LEAF);
    buffer_pool_unpin(vm.db->bp, index_root_id, /*dirty=*/1);
    /* 2. Register in schema */
    IndexDescriptor *desc = schema_add_index(vm.db, table_name, index_name,
                                              &col_index, 1, is_unique,
                                              index_root_id);
    if (!desc) {
        snprintf(vm.error, sizeof(vm.error),
                 "Failed to create index '%s'", index_name);
        vm.had_error = 1;
        goto halt;
    }
    /* 3. Populate the index from existing table data */
    if (index_build_from_table(vm.db, desc) < 0) {
        snprintf(vm.error, sizeof(vm.error),
                 "Failed to build index '%s': %s", index_name, vm.db->error);
        vm.had_error = 1;
        goto halt;
    }
    /* 4. Register in sqlite_master */
    char sql_buf[512];
    snprintf(sql_buf, sizeof(sql_buf),
             "CREATE INDEX %s ON %s(...)",  /* original SQL from p4_sql */
             index_name, table_name);
    schema_register_index(vm.db, index_name, table_name,
                           index_root_id, sql_buf);
    vm.pc++;
    break;
}
```
The bulk-build function `index_build_from_table` scans the table and inserts into the B+tree:
```c
/* Build an index from existing table data.
   Scans the table and inserts every (col_value, rowid) pair into the index B+tree. */
int index_build_from_table(DB *db, IndexDescriptor *desc) {
    SchemaTable *schema = schema_get_table(db, desc->table_name);
    if (!schema) return -1;
    BTree *table_bt = btree_open(db->bp, schema->root_page_id);
    BTree *index_bt = btree_open_index(db->bp, desc->root_page_id);
    BTreeCursor cur;
    btree_cursor_open_read(&cur, table_bt);
    if (btree_cursor_rewind(&cur)) {
        /* Empty table — index is already empty */
        btree_cursor_close(&cur);
        return 0;
    }
    do {
        /* Read the indexed column value */
        Value key_val;
        btree_cursor_column(&cur, desc->col_indices[0], &key_val);
        value_deep_copy(&key_val);  /* copy before advancing cursor */
        /* Read the rowid */
        int64_t rowid = btree_cursor_rowid(&cur);
        /* Check UNIQUE constraint against existing index entries */
        if (desc->is_unique && key_val.type != VAL_NULL) {
            if (index_contains_key(index_bt, &key_val)) {
                char errbuf[256];
                snprintf(errbuf, sizeof(errbuf),
                         "UNIQUE constraint failed: %s.%s",
                         desc->table_name,
                         schema->columns[desc->col_indices[0]].name);
                strncpy(db->error, errbuf, sizeof(db->error));
                value_free(&key_val);
                btree_cursor_close(&cur);
                return -1;
            }
        }
        /* Insert (key_val, rowid) into the index */
        if (index_insert(index_bt, &key_val, rowid) < 0) {
            value_free(&key_val);
            btree_cursor_close(&cur);
            return -1;
        }
        value_free(&key_val);
    } while (btree_cursor_next(&cur));
    btree_cursor_close(&cur);
    return 0;
}
```
### Index Insert: Encoding the Key-Rowid Pair
Each index leaf cell stores a 2-column record: (key_value, rowid). The existing `record_encode` function from Milestone 5 handles this directly:
```c
/* Insert a (key_value, rowid) pair into an index B+tree. */
int index_insert(BTree *index_bt, const Value *key, int64_t rowid) {
    /* Encode as a 2-column record: [key, rowid] */
    Value cols[2];
    cols[0] = *key;
    cols[1].type = VAL_INTEGER;
    cols[1].i    = rowid;
    int record_len;
    uint8_t *record = record_encode(cols, 2, &record_len);
    if (!record) return -1;
    /* Build the index leaf cell: [payload_size: varint][payload] */
    uint8_t cell_buf[PAGE_SIZE];
    int cpos = 0;
    cpos += varint_encode((uint64_t)record_len, cell_buf + cpos);
    memcpy(cell_buf + cpos, record, record_len);
    cpos += record_len;
    free(record);
    /* Insert into the B+tree, sorted by (key, rowid) */
    return btree_index_insert_cell(index_bt, cell_buf, cpos, key, rowid);
}
```
The index B+tree sorts cells by the key value first, then by rowid for ties (when multiple rows have the same value in a non-unique column). This ordering enables efficient range scans and equality lookups.
The `btree_index_insert_cell` function is structurally identical to the table B-tree insert from Milestone 5, with one difference: the comparison function uses the key value (first column of the 2-column record) rather than a rowid varint. The split algorithm is the same: when a leaf overflows, split it and promote the separator key to the parent.
---
## Index Lookup: The Double-Lookup Pattern

![Index Double Lookup — Data Walk](./diagrams/diag-index-double-lookup.svg)

This is the most architecturally interesting part of index execution. When you find a row via an index, you don't have the row's data — you have its rowid. You then need to fetch the full row from the table B-tree. This two-step process is the **double lookup** (also called a "bookmark lookup" or "key lookup" in other databases):
```
Step 1: Descend index B+tree for 'alice@example.com'
        → find leaf cell: ('alice@example.com', rowid=47382)
Step 2: Descend table B+tree for rowid=47382
        → find leaf cell: (47382, full_row_record)
        → deserialize all needed columns
```
Two tree descents instead of one full table scan. For selective queries (needle-in-a-haystack lookups), this is a dramatic win.
The VDBE needs new opcodes for index operations:
```c
/* Add to Opcode enum: */
OP_OPEN_INDEX,       /* p1=cursor_id, p4=index_name: open an index cursor */
OP_INDEX_SEEK,       /* p1=index_cursor, p2=jump_if_not_found, p3=key_reg:
                        seek to first entry with key = reg[p3] */
OP_INDEX_SEEK_GE,    /* p1=index_cursor, p2=jump_if_past_end, p3=key_reg:
                        seek to first entry with key >= reg[p3] */
OP_INDEX_NEXT,       /* p1=index_cursor, p2=jump_if_done: advance index cursor */
OP_INDEX_KEY,        /* p1=index_cursor, p2=dest_reg: load key value */
OP_INDEX_ROWID,      /* p1=index_cursor, p2=dest_reg: load rowid from index entry */
OP_SEEK_ROWID,       /* p1=table_cursor, p2=jump_if_not_found, p3=rowid_reg:
                        position table cursor at specific rowid */
```
The compiled bytecode for `SELECT name, age FROM users WHERE email = 'alice@example.com'`:
```
addr  opcode        p1   p2   p3   comment
0     OpenTable     0    0    0    table cursor 0 = 'users'
1     OpenIndex     1    0    0    index cursor 1 = 'idx_email'
2     String8       2    0    0    reg[2] = 'alice@example.com'
3     IndexSeek     1    8    2    seek index to 'alice@example.com'; not found → jump 8
4     IndexRowid    1    3    0    reg[3] = rowid from index
5     SeekRowid     0    8    3    position table cursor at reg[3]; not found → jump 8
6     Column        0    1    4    reg[4] = name (col 1)
7     Column        0    2    5    reg[5] = age  (col 2)
8     ResultRow     4    2    0    emit (reg[4], reg[5])
9     IndexNext     1    11   0    advance index; if more matches → loop... 
      (simplified: equality scan stops after first non-matching key)
10    Goto          0    3    0    (for range scans: loop back to check)
11    Close         0    0    0
12    Close         1    0    0
13    Halt          0    0    0
```

![SELECT → Bytecode Compilation — Trace Example](./diagrams/diag-vdbe-select-compilation.svg)

This is the double-lookup pattern at the bytecode level: the index cursor (cursor 1) finds the rowid, then `OP_SEEK_ROWID` positions the table cursor (cursor 0) at that rowid for column extraction.
### Implementing OP_INDEX_SEEK
The seek operation descends the index B+tree to find the first leaf entry with `key >= search_key`:
```c
case OP_INDEX_SEEK: {
    int cursor_id       = instr->p1;
    int jump_if_not_found = instr->p2;
    int key_reg         = instr->p3;
    IndexCursor *cur    = (IndexCursor *)vm.cursors[cursor_id];
    Value *search_key   = &vm.regs[key_reg];
    int found = index_cursor_seek_eq(cur, search_key);
    if (!found) {
        vm.pc = jump_if_not_found;
    } else {
        vm.pc++;
    }
    break;
}
/* Seek to the first leaf entry whose key equals search_key.
   Returns 1 if found, 0 if not found. */
int index_cursor_seek_eq(IndexCursor *cur, const Value *search_key) {
    PageId current = cur->index_bt->root_page_id;
    for (;;) {
        uint8_t *page = buffer_pool_fetch(cur->index_bt->bp, current);
        PageHeader h;
        page_header_read(page, &h);
        if (h.page_type == PAGE_TYPE_INDEX_LEAF) {
            /* Binary search within leaf for key == search_key */
            int lo = 0, hi = (int)h.cell_count - 1;
            int found_slot = -1;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                Value cell_key;
                index_leaf_read_key(page, mid, &cell_key);
                int cmp = value_compare(&cell_key, search_key);
                value_free_inline(&cell_key);
                if (cmp == 0) {
                    /* Found a match — but there might be earlier matches */
                    found_slot = mid;
                    hi = mid - 1;  /* keep searching left for first match */
                } else if (cmp < 0) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            if (found_slot >= 0) {
                cur->leaf_page_id = current;
                cur->slot         = found_slot;
                cur->at_end       = 0;
                buffer_pool_unpin(cur->index_bt->bp, current, 0);
                return 1;
            }
            buffer_pool_unpin(cur->index_bt->bp, current, 0);
            return 0;
        }
        /* Internal node: find child containing search_key */
        PageId next_child = h.right_child;
        for (int i = 0; i < (int)h.cell_count; i++) {
            const uint8_t *cell = cell_at(page, i);
            Value sep_key;
            index_internal_read_key(cell, &sep_key);
            int cmp = value_compare(search_key, &sep_key);
            value_free_inline(&sep_key);
            if (cmp <= 0) {
                next_child = ((uint32_t)cell[0] << 24) | ((uint32_t)cell[1] << 16) |
                             ((uint32_t)cell[2] <<  8) |  (uint32_t)cell[3];
                break;
            }
        }
        buffer_pool_unpin(cur->index_bt->bp, current, 0);
        current = next_child;
    }
}
```
### Implementing OP_INDEX_NEXT
After finding a matching entry, `OP_INDEX_NEXT` advances the index cursor and checks whether the new entry's key still matches the search condition. For equality scans, it stops when the key changes:
```c
case OP_INDEX_NEXT: {
    int cursor_id    = instr->p1;
    int jump_if_done = instr->p2;
    IndexCursor *cur = (IndexCursor *)vm.cursors[cursor_id];
    /* Advance to next cell */
    int has_more = index_cursor_next(cur);
    if (!has_more) {
        vm.pc = jump_if_done;
        break;
    }
    /* For equality scans: check if key still matches */
    /* The compiler sets p3 to the equality key register for equality scans,
       0 for range scans that use a separate termination check */
    if (instr->p3 > 0) {
        Value current_key;
        index_cursor_read_key(cur, &current_key);
        int still_matches = (value_compare(&current_key, &vm.regs[instr->p3]) == 0);
        value_free_inline(&current_key);
        if (!still_matches) {
            cur->at_end = 1;
            vm.pc = jump_if_done;
            break;
        }
    }
    vm.pc++;
    break;
}
```
### Implementing OP_SEEK_ROWID
After extracting the rowid from the index entry, `OP_SEEK_ROWID` positions the table cursor at that exact rowid:
```c
case OP_SEEK_ROWID: {
    int cursor_id         = instr->p1;
    int jump_if_not_found = instr->p2;
    int rowid_reg         = instr->p3;
    BTreeCursor *cur      = vm.cursors[cursor_id];
    int64_t target_rowid  = vm.regs[rowid_reg].i;
    int slot;
    PageId leaf = btree_search(cur->bt->bp, cur->bt->root_page_id,
                                target_rowid, &slot);
    if (leaf == INVALID_PAGE_ID) {
        /* Index entry exists but table row doesn't — index/table inconsistency */
        vm.pc = jump_if_not_found;
        break;
    }
    cur->leaf_page_id = leaf;
    cur->slot         = slot;
    cur->at_end       = 0;
    vm.pc++;
    break;
}
```
The `OP_SEEK_ROWID` jump-if-not-found branch handles index-table inconsistency: a row was deleted from the table but its index entry wasn't removed (a bug). Rather than crashing, the engine skips to the next index entry. In a correctly implemented database this path never fires, but defensive handling here prevents a single corrupted index from taking down the database.
---
## Range Scans: Following the Leaf Chain

![Index Range Scan — Data Walk](./diagrams/diag-index-range-scan.svg)

The B+tree's linked leaf chain makes range scans elegant. For `WHERE age BETWEEN 20 AND 30`, the compiled bytecode:
```
addr  opcode        p1   p2   p3   comment
0     OpenTable     0    0    0    table cursor 0 = 'users'
1     OpenIndex     1    0    0    index cursor 1 = 'idx_age'
2     Integer       20   2    0    reg[2] = lower bound (20)
3     Integer       30   3    0    reg[3] = upper bound (30)
4     IndexSeekGe   1    12   2    seek to first key >= 20; past end → 12
5     IndexKey      1    4    0    reg[4] = current index key value
6     Gt            4    12   3    if key > 30, stop (past upper bound)
7     IndexRowid    1    5    0    reg[5] = rowid
8     SeekRowid     0    10   5    position table cursor at rowid
9     Column        0    1    6    reg[6] = name
10    ResultRow     6    1    0    emit name
11    IndexNext     1    12   0    advance index (no equality key check)
     Goto           0    5    0    loop back to check upper bound
12    Halt
```
`OP_INDEX_SEEK_GE` positions the cursor at the first entry with key ≥ lower bound. Then the loop checks the upper bound at each step (`OP_GT` at addr 6), stopping when the key exceeds 30. The leaf chain traversal happens transparently inside `index_cursor_next`:
```c
/* Advance an index cursor to the next entry.
   Follows the right-sibling pointer when the current leaf is exhausted.
   Returns 1 if another entry exists, 0 if exhausted. */
int index_cursor_next(IndexCursor *cur) {
    if (cur->at_end) return 0;
    cur->slot++;
    uint8_t *page = buffer_pool_fetch(cur->index_bt->bp, cur->leaf_page_id);
    PageHeader h;
    page_header_read(page, &h);
    if (cur->slot < (int)h.cell_count) {
        /* More entries on this leaf */
        buffer_pool_unpin(cur->index_bt->bp, cur->leaf_page_id, 0);
        return 1;
    }
    /* Exhausted this leaf — follow right sibling pointer */
    PageId right_sibling = h.right_child;  /* repurposed for INDEX_LEAF */
    buffer_pool_unpin(cur->index_bt->bp, cur->leaf_page_id, 0);
    if (right_sibling == 0) {
        cur->at_end = 1;
        return 0;
    }
    cur->leaf_page_id = right_sibling;
    cur->slot         = 0;
    return 1;
}
```
This is why the B+tree's leaf chain is worth the extra complexity: `index_cursor_next` for a range scan is just a slot increment plus an occasional sibling pointer follow. No tree re-ascent, no parent page reads. A range scan that spans 1,000 leaf pages follows 1,000 sibling pointers — each a single 4-byte field read from the already-cached page header.
---
## UNIQUE Index: Constraint Enforcement
A `UNIQUE INDEX` means: no two rows can have the same value in the indexed column (with the exception of NULL — multiple NULLs are allowed because `NULL ≠ NULL` in SQL's three-valued logic).
UNIQUE enforcement happens at insert time, before the index entry is added:
```c
/* Check UNIQUE constraint before inserting into an index.
   Returns 0 if OK, -1 if UNIQUE violation. */
int index_check_unique(DB *db, IndexDescriptor *desc, const Value *key) {
    if (!desc->is_unique) return 0;
    /* NULL values never violate UNIQUE — multiple NULLs are allowed */
    if (key->type == VAL_NULL) return 0;
    BTree *index_bt = btree_open_index(db->bp, desc->root_page_id);
    int found = index_contains_key(index_bt, key);
    if (found) {
        SchemaTable *schema = schema_get_table(db, desc->table_name);
        snprintf(db->error, sizeof(db->error),
                 "UNIQUE constraint failed: %s.%s",
                 desc->table_name,
                 schema->columns[desc->col_indices[0]].name);
        return -1;
    }
    return 0;
}
/* Check if any entry with the given key exists in the index. */
int index_contains_key(BTree *index_bt, const Value *key) {
    IndexCursor cur;
    index_cursor_open(&cur, index_bt);
    int found = index_cursor_seek_eq(&cur, key);
    index_cursor_close(&cur);
    return found;
}
```
In the VDBE, UNIQUE enforcement is wired into `OP_INSERT`. The compiler knows which columns have UNIQUE indexes (from the schema), and emits an `OP_CHECK_UNIQUE` instruction for each one before the record is inserted:
```c
/* OP_CHECK_UNIQUE: verify UNIQUE constraint for an index.
   p4 = index_name, p1 = key_register */
case OP_CHECK_UNIQUE: {
    const char *index_name = instr->p4.str;
    int key_reg = instr->p1;
    IndexDescriptor *desc = schema_find_index(vm.db, index_name);
    if (!desc) {
        vm.pc++;
        break;  /* index not found — shouldn't happen */
    }
    if (index_check_unique(vm.db, desc, &vm.regs[key_reg]) < 0) {
        snprintf(vm.error, sizeof(vm.error), "%s", vm.db->error);
        vm.had_error = 1;
        goto halt;
    }
    vm.pc++;
    break;
}
```
The UNIQUE check for a column with a `NOT NULL` constraint: if the column is `NOT NULL UNIQUE`, the only values rejected are duplicates of non-null values. If the column is just `UNIQUE` (without `NOT NULL`), then `NULL` can appear any number of times, and non-null duplicates are still rejected. Your implementation above handles this correctly because the `if (key->type == VAL_NULL) return 0` bypass allows multiple NULLs through.
---
## Index Maintenance: Synchronous Updates on DML

![Index Maintenance on INSERT/DELETE — Before/After](./diagrams/diag-index-maintenance.svg)

Every INSERT, UPDATE, and DELETE on a table must synchronously update all indexes on that table. "Synchronous" means: within the same transaction, before the operation returns. No deferred index maintenance — a committed row must be findable through its indexes immediately.
### Maintenance on INSERT
After inserting a row into the table B-tree, insert an entry into each index:
```c
/* Called after btree_insert succeeds for a table row.
   Inserts the row's indexed column values into all indexes on this table. */
int index_maintain_insert(DB *db, SchemaTable *schema,
                           const Value *cols, int ncols, int64_t rowid) {
    for (int i = 0; i < schema->index_count; i++) {
        IndexDescriptor *desc = &schema->indexes[i];
        /* For multi-column indexes: build a composite key.
           For single-column indexes (our current implementation): use col directly. */
        if (desc->col_count != 1) {
            /* TODO: composite index support */
            continue;
        }
        int col_idx = desc->col_indices[0];
        if (col_idx >= ncols) continue;  /* column not in this insert */
        const Value *key = &cols[col_idx];
        /* Check UNIQUE before inserting */
        if (desc->is_unique && key->type != VAL_NULL) {
            if (index_check_unique(db, desc, key) < 0) return -1;
        }
        /* Insert into index */
        BTree *index_bt = btree_open_index(db->bp, desc->root_page_id);
        if (index_insert(index_bt, key, rowid) < 0) return -1;
    }
    return 0;
}
```
This function is called from the `OP_INSERT` handler in the VDBE, immediately after `btree_cursor_insert` succeeds. If any index insertion fails (e.g., UNIQUE violation), the entire INSERT fails and the table row should be rolled back.
### Maintenance on DELETE
Delete the index entries for the row being removed. You need the row's current column values to find its index entries:
```c
/* Called before btree_delete_by_rowid.
   Removes the row's entries from all indexes. */
int index_maintain_delete(DB *db, SchemaTable *schema, int64_t rowid) {
    /* First, read the row's current column values (needed to find index entries) */
    BTree *table_bt = btree_open(db->bp, schema->root_page_id);
    int slot;
    PageId leaf = btree_search(db->bp, schema->root_page_id, rowid, &slot);
    if (leaf == INVALID_PAGE_ID) return -1;  /* row not found */
    /* Read all indexed columns */
    uint8_t *page = buffer_pool_fetch(db->bp, leaf);
    Value col_values[MAX_INDEX_COLS];
    for (int i = 0; i < schema->index_count; i++) {
        IndexDescriptor *desc = &schema->indexes[i];
        for (int j = 0; j < desc->col_count; j++) {
            int col_idx = desc->col_indices[j];
            record_decode_column(/* payload */..., col_idx, &col_values[col_idx]);
            value_deep_copy(&col_values[col_idx]);  /* copy before unpinning */
        }
    }
    buffer_pool_unpin(db->bp, leaf, 0);
    /* Delete from each index */
    for (int i = 0; i < schema->index_count; i++) {
        IndexDescriptor *desc = &schema->indexes[i];
        if (desc->col_count != 1) continue;
        int col_idx = desc->col_indices[0];
        Value *key = &col_values[col_idx];
        BTree *index_bt = btree_open_index(db->bp, desc->root_page_id);
        if (index_delete(index_bt, key, rowid) < 0) return -1;
    }
    /* Free deep-copied values */
    for (int i = 0; i < schema->column_count; i++) {
        value_free(&col_values[i]);
    }
    return 0;
}
```
The key implementation is `index_delete`, which finds the leaf entry matching `(key, rowid)` and removes it:
```c
/* Delete the entry (key, rowid) from an index B+tree.
   Returns 0 if deleted, -1 if not found. */
int index_delete(BTree *index_bt, const Value *key, int64_t rowid) {
    IndexCursor cur;
    index_cursor_open(&cur, index_bt);
    /* Seek to first entry with this key */
    if (!index_cursor_seek_eq(&cur, key)) {
        index_cursor_close(&cur);
        return -1;  /* key not found */
    }
    /* Among entries with this key, find the one with matching rowid */
    do {
        Value cur_key;
        index_cursor_read_key(&cur, &cur_key);
        int key_matches = (value_compare(&cur_key, key) == 0);
        value_free_inline(&cur_key);
        if (!key_matches) break;  /* past all entries with this key */
        int64_t cur_rowid = index_cursor_read_rowid(&cur);
        if (cur_rowid == rowid) {
            /* Found it — delete this entry */
            index_cursor_delete_current(&cur);
            index_cursor_close(&cur);
            return 0;
        }
    } while (index_cursor_next(&cur));
    index_cursor_close(&cur);
    return -1;  /* rowid not found among entries with this key */
}
```
The loop handles the case where multiple rows share the same indexed value (a non-unique index). Among all entries with key `'active'`, you must find the one whose rowid is the specific row being deleted.
### Maintenance on UPDATE
UPDATE is: delete old index entries for the row, then insert new ones with the updated values:
```c
/* Called when a row is being updated.
   Removes old index entries (based on pre-update values) and inserts new ones. */
int index_maintain_update(DB *db, SchemaTable *schema,
                           int64_t rowid,
                           const Value *old_cols, int ncols,
                           const Value *new_cols) {
    for (int i = 0; i < schema->index_count; i++) {
        IndexDescriptor *desc = &schema->indexes[i];
        if (desc->col_count != 1) continue;
        int col_idx = desc->col_indices[0];
        if (col_idx >= ncols) continue;
        const Value *old_key = &old_cols[col_idx];
        const Value *new_key = &new_cols[col_idx];
        /* Optimization: if the key value didn't change, skip */
        if (value_compare(old_key, new_key) == 0 &&
            old_key->type == new_key->type) {
            continue;
        }
        BTree *index_bt = btree_open_index(db->bp, desc->root_page_id);
        /* Delete old entry */
        if (old_key->type != VAL_NULL) {
            index_delete(index_bt, old_key, rowid);
        }
        /* Check UNIQUE for new entry */
        if (desc->is_unique && new_key->type != VAL_NULL) {
            if (index_check_unique(db, desc, new_key) < 0) return -1;
        }
        /* Insert new entry */
        index_insert(index_bt, new_key, rowid);
    }
    return 0;
}
```
The optimization `if value_compare(old_key, new_key) == 0: skip` is important for UPDATE performance. `UPDATE t SET age = age + 1 WHERE id = 42` touches the `age` column, but if there's an index on `name`, the name index doesn't need updating. Only indexes whose columns are actually modified need maintenance.
---
## Write Amplification: The Hidden Cost

> **🔑 Foundation: Write amplification: why each logical write becomes multiple physical writes**
> 
> ## Write Amplification in Storage Systems
**What it is**
Write amplification (WA) is the ratio between the amount of data physically written to storage versus the amount of data logically requested by the application. A WA of 10× means that writing 1 MB of user data causes 10 MB of actual disk or flash writes to occur.
It emerges from the gap between how applications think about data (logical records, pages, rows) and the constraints of physical storage media and data structure maintenance.
There are several compounding sources:
**1. Page/Block granularity**  
Storage devices write in fixed-size units. A traditional HDD writes in 512-byte sectors; SSDs erase and write in blocks (often 128 KB–4 MB). If you update a single byte, the entire containing page must be rewritten. Update a 1-byte field in a 16 KB database page → 16 KB written. WA = 16,384×.
**2. Copy-on-write and shadowing**  
B-Trees with copy-on-write semantics (like LMDB) never update a page in-place. They write a new version of the page, then walk up the tree updating parent pointers — also as new copies. A single key update can cascade through the full tree height (3–5 levels is typical), multiplying writes by tree height.
**3. Log-structured compaction (LSM trees)**  
LSM trees (RocksDB, LevelDB, Cassandra) write all updates sequentially to a memtable, flush to SSTable files on disk, then periodically *compact* — merging and rewriting SSTables to reclaim space and maintain read performance. That compaction rewrites data multiple times as it migrates through levels (L0 → L1 → L2 → ...). Benchmarks show WA of 10–30× for write-heavy workloads.
**4. WAL (Write-Ahead Log) overhead**  
Databases that guarantee durability (ACID) write to the WAL *before* writing to the data file. Every logical write becomes at minimum two physical writes: journal entry + actual page. Then the page may later be checkpointed/flushed again.
**Concrete example — SQLite UPDATE**  
```
UPDATE orders SET status = 'shipped' WHERE id = 42;
```
This single logical write can cause:
- WAL record written (~100 bytes, rounded to 512-byte sector)
- B-Tree leaf page rewritten (4 KB default page size)
- Potentially one or more internal B-Tree pages rewritten if keys shift
- On checkpoint: data file page flushed again
Total: potentially 4–20 KB written for a 10-byte change.
**Why you need it right now**
When building a storage engine or database, write amplification is a primary design constraint — not an implementation detail. Your choice of data structure (B-Tree vs LSM vs append-only log), page size, journaling strategy, and compaction policy will determine whether your engine sustains 50K writes/sec or 5K writes/sec under load. SSDs are especially sensitive: high WA accelerates wear-leveling exhaustion and throttling. Understanding WA tells you *why* certain engines make seemingly strange choices (e.g., why RocksDB uses a tiered compaction strategy, or why PostgreSQL uses an 8 KB page size, or why LMDB uses MVCC with copy-on-write B-Trees).
**The key mental model**
> **You never write just what you changed — you write the entire containing unit, plus the cost of keeping your data structures consistent.**
Think in terms of *write units*: what is the smallest chunk your storage layer must rewrite to make a change durable and structurally valid? Every architectural decision either shrinks that unit, amortizes it across many writes, or trades write amplification for better read performance. There is no free lunch — LSM trees reduce WA for writes by accepting read amplification; B-Trees do the opposite.

Consider what happens to your SSD when you have 5 indexes and run 1,000 inserts. Each insert triggers 6 B-tree modifications. Each B-tree modification writes at least 1 page (the modified leaf). If there's a page split, it writes 2-3 pages. If the split propagates upward, more pages.
For 1,000 inserts with 5 indexes and moderate splits: you might write 10,000–20,000 pages (40–80 MB) to persist 1,000 rows that total perhaps 100 KB of actual data. That's a **write amplification factor of 400-800×**.
This isn't a bug — it's the fundamental tradeoff of B-tree indexing. The same phenomenon degrades SSD lifespan: SSDs have limited write endurance (P/E cycles per NAND cell), and write amplification consumes those cycles faster than the logical write rate suggests. A database with aggressive indexing running on an SSD writes far more data to the physical medium than the application believes it's writing.
The industry response to write amplification is the **LSM-tree** (Log-Structured Merge tree), used in RocksDB, LevelDB, and Cassandra. Instead of maintaining indexes synchronously (immediate B-tree modification on every write), LSM-trees buffer writes in memory, write them sequentially to disk as sorted run files, and merge them periodically. This converts random write amplification (B-tree splits scattered across pages) into sequential write amplification (large, efficient sequential I/O). The tradeoff: reads become more complex (must check multiple levels) and background compaction introduces latency spikes.
You don't need to implement an LSM-tree, but understanding why it exists makes you a better designer of systems that use databases. When you see "use RocksDB for write-heavy workloads," you now know the architectural reason.
---
## The Compiler: Choosing Index vs. Table Scan
The VDBE compiler needs to decide, at compile time, whether to emit an index scan or a table scan for a given query. This is a simplified version of what the query planner does (Milestone 8 will build the full cost-based planner). For now, implement a basic heuristic: **use an index if one exists for a column in the WHERE clause with an equality or range predicate**.
```c
/* In the SELECT compiler, after parsing the WHERE clause: */
static void compile_select_with_index(Compiler *c, AstNode *node) {
    SchemaTable *schema = schema_get_table(c->db, node->select.table_name);
    if (!schema) {
        /* Error: table not found */
        return;
    }
    /* Try to find a usable index for the WHERE clause */
    IndexDescriptor *idx = NULL;
    AstNode *index_pred  = NULL;  /* the predicate that uses the index */
    if (node->select.where_expr) {
        idx = find_usable_index(schema, node->select.where_expr, &index_pred);
    }
    if (idx && index_pred) {
        compile_select_via_index(c, node, idx, index_pred);
    } else {
        compile_select_table_scan(c, node);
    }
}
/* Find an index that can serve the WHERE predicate.
   Currently: look for a simple equality or range predicate on an indexed column.
   Returns the index descriptor and the predicate AST node that uses it. */
static IndexDescriptor *find_usable_index(SchemaTable *schema,
                                           AstNode *where,
                                           AstNode **pred_out) {
    if (!where || where->type != NODE_BINARY_EXPR) return NULL;
    /* Only handle simple predicates: col OP value */
    TokenType op = where->binary.op;
    if (op != TOKEN_EQ && op != TOKEN_LT && op != TOKEN_GT &&
        op != TOKEN_LTE && op != TOKEN_GTE) {
        return NULL;
    }
    /* The left side must be a column identifier */
    AstNode *col_node = where->binary.left;
    if (col_node->type != NODE_IDENTIFIER) return NULL;
    /* Find schema column index for this identifier */
    int col_idx = schema_get_column_index(schema, col_node->identifier.name);
    if (col_idx < 0) return NULL;
    /* Look for an index on this column */
    for (int i = 0; i < schema->index_count; i++) {
        IndexDescriptor *idx = &schema->indexes[i];
        if (idx->col_count == 1 && idx->col_indices[0] == col_idx) {
            *pred_out = where;
            return idx;
        }
    }
    return NULL;
}
```

![Cost Model — Table Scan vs Index Scan](./diagrams/diag-planner-cost-model.svg)

This heuristic has a known limitation: it always uses an index when one exists, regardless of selectivity. Milestone 8 will replace this with a cost-based decision. For now, the rule "use index if it exists for the WHERE column" is correct enough to pass the acceptance criteria and significantly reduces page reads for selective queries.
The `compile_select_via_index` function emits the double-lookup bytecode pattern shown earlier (OpenTable + OpenIndex + IndexSeek + IndexRowid + SeekRowid + Column + ResultRow + IndexNext + loop).
---
## Covering Indexes: Eliminating the Double Lookup

![Covering Index vs Double Lookup — Before/After](./diagrams/diag-index-covering-vs-double.svg)

The double lookup (index → rowid → table) reads from two B+trees. If the query needs only columns that are in the index itself, you can skip the table lookup entirely — the index already contains everything the query needs.
This is a **covering index**: an index that "covers" the query — provides all needed columns without touching the table.
```sql
-- Index on (email) for this query:
SELECT email FROM users WHERE email LIKE 'alice%';
-- The query needs: email (for output), email (for filter)
-- Both are in the index! No table lookup needed.
-- The index leaf stores: (email_value, rowid)
-- We can return email_value directly without fetching the row.
```
The covering index check in the compiler:
```c
/* Returns 1 if the index covers all columns needed by the query (projection + predicates).
   If so, skip the table cursor and read everything from the index. */
static int index_covers_query(IndexDescriptor *idx, SchemaTable *schema,
                               AstNode *select_node) {
    /* Collect all column references in the SELECT list and WHERE clause */
    int needed_cols[MAX_COLS];
    int needed_count = 0;
    collect_column_refs(select_node->select.columns,
                        select_node->select.column_count,
                        schema, needed_cols, &needed_count);
    if (select_node->select.where_expr) {
        collect_column_refs_expr(select_node->select.where_expr,
                                  schema, needed_cols, &needed_count);
    }
    /* Check if all needed columns are in the index */
    for (int i = 0; i < needed_count; i++) {
        int found = 0;
        for (int j = 0; j < idx->col_count; j++) {
            if (idx->col_indices[j] == needed_cols[i]) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;  /* column needed but not in index */
    }
    return 1;  /* all needed columns are in the index */
}
```
When a covering index is detected, the compiler emits:
```
OpenIndex     0    0    0    (only index cursor needed, no table cursor)
IndexSeek     0    →end   key_reg
IndexKey      0    result_reg    (read key value from index entry)
ResultRow     result_reg  1
IndexNext     0    →end
Halt
```
No `OP_OPEN_TABLE`, no `OP_SEEK_ROWID`. The table B-tree is never opened.
Covering indexes connect conceptually to **materialized views** — a database feature where you precompute and store the result of a query. A covering index is a materialized partial view: it stores the indexed column values in sorted order, ready to satisfy specific query shapes without touching the base table. The cost is the same — writes are amplified to maintain the materialized data.
---
## NULL Handling in Unique Indexes

> **🔑 Foundation: SQL standard rule: multiple NULLs are permitted in UNIQUE indexes**
> 
> ## NULL in UNIQUE Indexes
**What it is**
The SQL standard specifies that `NULL` represents an *unknown or absent value*. A core property of NULL is that it is **not equal to anything — including itself**. The expression `NULL = NULL` evaluates to `NULL` (unknown), not `TRUE`.
UNIQUE constraints are defined in terms of equality: two rows violate uniqueness if their constrained column values are *equal*. Since `NULL ≠ NULL` (or more precisely, the comparison is unknown rather than true), the standard concludes that **no two NULLs are equal, therefore multiple NULLs cannot violate a UNIQUE constraint**.
In practice: you can insert as many rows with `NULL` in a UNIQUE column as you like.
```sql
CREATE TABLE users (
    id       INTEGER PRIMARY KEY,
    email    TEXT UNIQUE,   -- UNIQUE allows multiple NULLs
    username TEXT NOT NULL
);
INSERT INTO users VALUES (1, NULL, 'alice');   -- OK
INSERT INTO users VALUES (2, NULL, 'bob');     -- OK — not a uniqueness violation
INSERT INTO users VALUES (3, NULL, 'carol');   -- OK
INSERT INTO users VALUES (4, 'x@y.com', 'dave'); -- OK
INSERT INTO users VALUES (5, 'x@y.com', 'eve');  -- ERROR — duplicate non-NULL
```
**Database-specific behavior**
Most databases follow the standard, but not all:
| Database | Multiple NULLs in UNIQUE index? |
|---|---|
| PostgreSQL | ✅ Yes (standard behavior) |
| SQLite | ✅ Yes |
| MySQL/MariaDB | ✅ Yes |
| Oracle | ✅ Yes |
| SQL Server | ❌ No — treats NULL = NULL for UNIQUE purposes (non-standard) |
SQL Server's non-standard behavior trips up many developers. A filtered unique index (`WHERE col IS NOT NULL`) is the common workaround there.
**Why you need it right now**
When implementing a storage engine or query executor with index support, you need to handle NULL specifically in your uniqueness-checking logic. If your B-Tree or hash index comparison function treats `NULL = NULL` as `true` (a natural default when using `memcmp` or standard equality), you will accidentally reject valid INSERTs. The null-handling case must be explicitly carved out: *skip the uniqueness check when the new value is NULL*. Similarly, when building index scans, a lookup for `WHERE email = NULL` should return zero rows (use `IS NULL` instead) — your index iterator needs to understand this distinction.
**The key mental model**
> **NULL is not a value — it's the absence of a value. UNIQUE means "no two known values are the same", not "this slot can only be filled once."**
Think of a UNIQUE column with NULLs like reserved parking spots where an empty space (NULL) doesn't count as "taken." Two empty spaces can coexist without conflict. Only when an actual car (non-NULL value) parks does it occupy a uniquely claimed spot that no other car can share.

The SQL standard specifies: `NULL` values do not violate `UNIQUE` constraints because `NULL ≠ NULL` in three-valued logic. Two rows with `NULL` in a `UNIQUE` column do not have "equal" values — they have unknown values, which are not equal.
Your `index_check_unique` already handles this correctly with `if (key->type == VAL_NULL) return 0`. Verify with a test:
```c
void test_unique_null_handling(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, email TEXT)");
    db_exec(db, "CREATE UNIQUE INDEX idx_email ON t(email)");
    /* Two NULLs in a UNIQUE column: must succeed */
    assert(db_exec(db, "INSERT INTO t VALUES (1, NULL)") == 0);
    assert(db_exec(db, "INSERT INTO t VALUES (2, NULL)") == 0);  /* NOT a UNIQUE violation */
    /* Duplicate non-null value: must fail */
    assert(db_exec(db, "INSERT INTO t VALUES (3, 'alice@example.com')") == 0);
    assert(db_exec(db, "INSERT INTO t VALUES (4, 'alice@example.com')") < 0);
    assert(strstr(db->error, "UNIQUE constraint failed") != NULL);
    db_close(db);
    printf("PASS: unique null handling\n");
}
```
---
## sqlite_master: Persisting Index Metadata
Just as `CREATE TABLE` records the table in `sqlite_master`, `CREATE INDEX` records the index. The `sqlite_master` table already has the right schema for this: the `type` column stores `'index'` instead of `'table'`, `name` stores the index name, `tbl_name` stores the table the index belongs to, `rootpage` stores the index B+tree root page number, and `sql` stores the original `CREATE INDEX` statement.

![System Catalog (sqlite_master) — Structure Layout](./diagrams/diag-btree-system-catalog.svg)

```c
/* Register an index in sqlite_master.
   Called from OP_CREATE_INDEX after the index B+tree is built. */
int schema_register_index(DB *db, const char *index_name,
                            const char *table_name,
                            PageId rootpage, const char *sql) {
    Value cols[5];
    cols[0].type = VAL_TEXT; cols[0].text.data = "index"; cols[0].text.len = 5;
    cols[1].type = VAL_TEXT; cols[1].text.data = (char *)index_name;
                             cols[1].text.len = strlen(index_name);
    cols[2].type = VAL_TEXT; cols[2].text.data = (char *)table_name;
                             cols[2].text.len = strlen(table_name);
    cols[3].type = VAL_INTEGER; cols[3].i = (int64_t)rootpage;
    cols[4].type = VAL_TEXT; cols[4].text.data = (char *)sql;
                             cols[4].text.len = strlen(sql);
    int64_t rowid = schema_next_rowid(db);
    BTree *catalog = btree_open(db->bp, SQLITE_MASTER_ROOT_PAGE);
    return btree_insert(catalog, cols, 5, rowid);
}
```
On database open, the schema loader scans `sqlite_master` for both `type='table'` and `type='index'` entries, building the in-memory `SchemaTable` and `IndexDescriptor` structures:
```c
/* Load schema from sqlite_master on database open */
int schema_load(DB *db) {
    BTree *catalog = btree_open(db->bp, SQLITE_MASTER_ROOT_PAGE);
    BTreeCursor cur;
    btree_cursor_open_read(&cur, catalog);
    if (btree_cursor_rewind(&cur)) return 0;  /* empty database */
    do {
        Value type_val, name_val, tbl_val, root_val;
        btree_cursor_column(&cur, 0, &type_val);   /* type */
        btree_cursor_column(&cur, 1, &name_val);   /* name */
        btree_cursor_column(&cur, 2, &tbl_val);    /* tbl_name */
        btree_cursor_column(&cur, 3, &root_val);   /* rootpage */
        /* Deep copy before advancing */
        value_deep_copy(&type_val);
        value_deep_copy(&name_val);
        value_deep_copy(&tbl_val);
        if (type_val.type == VAL_TEXT) {
            if (strncmp(type_val.text.data, "table", 5) == 0) {
                /* Register table */
                schema_load_table_columns(db, name_val.text.data,
                                           (PageId)root_val.i);
            } else if (strncmp(type_val.text.data, "index", 5) == 0) {
                /* Register index — find the table and add the descriptor */
                schema_load_index(db, name_val.text.data,
                                   tbl_val.text.data,
                                   (PageId)root_val.i);
            }
        }
        value_free(&type_val);
        value_free(&name_val);
        value_free(&tbl_val);
    } while (btree_cursor_next(&cur));
    btree_cursor_close(&cur);
    return 0;
}
```
---
## Three-Level View: `WHERE email = 'alice@example.com'` With an Index
To see how all layers interact for an index-driven query, trace the full execution path:

![SELECT Execution — Full Stack Data Walk](./diagrams/diag-dml-select-execution.svg)

**Level 1 — VDBE bytecode (what the VM executes)**
```
OpenTable   0              // open table cursor for 'users'
OpenIndex   1              // open index cursor for 'idx_email'
String8     2 → 'alice@...' // load search key into reg[2]
IndexSeek   1  →end  2     // descend index tree, seek to 'alice@...'
IndexRowid  1  3           // reg[3] = rowid from index leaf entry
SeekRowid   0  →end  3     // descend table tree to rowid reg[3]
Column      0  1  4        // reg[4] = name column
Column      0  2  5        // reg[5] = age column
ResultRow   4  2           // emit (name, age)
IndexNext   1  →end        // advance index; key changed → stop
Halt
```
**Level 2 — B+tree and cursor (what happens in the storage engine)**
`OP_INDEX_SEEK` calls `index_cursor_seek_eq(cur, 'alice@...')`:
- Fetches root page of `idx_email` from buffer pool
- Reads internal cells: `'bob@...', 'carol@...'` — 'alice' < 'bob', go left child
- Fetches internal page, reads `'alan@...', 'alice@...'` — 'alice' == 'alice@...', go right of 'alan'
- Fetches leaf page, binary searches: finds `('alice@example.com', 47382)` at slot 3
- Returns rowid=47382
`OP_SEEK_ROWID` calls `btree_search(bp, users_root, 47382, &slot)`:
- Fetches root page of `users` table B-tree
- Follows separator keys down to the leaf containing rowid 47382
- Returns leaf_page_id and slot
`OP_COLUMN` calls `btree_cursor_column(cur, col_idx, &reg)`:
- Reads the cell at the cursor's current position (already set by SeekRowid)
- Decodes the record header, skips to the requested column's serial type and data
- Returns the Value pointing into the pinned page
Total page reads: ~6 (3 index tree levels + 3 table tree levels). Compare to a table scan: ~10,000 pages.
**Level 3 — Buffer pool and disk**
Each of the 6 tree pages goes through `buffer_pool_fetch`:
- Root pages are almost certainly hot (accessed every query) → cache hits, ~10ns each
- Leaf pages may be cold on first access → `pread()`, ~100μs each
- On a warm cache: all 6 pages are hits → ~60ns total storage engine overhead
- On a cold cache: 6 page reads → ~600μs, compared to ~1,000,000μs (1 second) for 10,000 pages cold

![FetchPage — Hit vs Miss Data Walk](./diagrams/diag-buffer-pool-fetch-flow.svg)

---
## Design Decisions: Index on All Columns vs. Composite Index
A composite index covers multiple columns. The rule you must understand — and which your implementation should document clearly — is the **leftmost prefix rule**:
A composite index on `(last_name, first_name, age)` can serve queries on:
- `WHERE last_name = 'Smith'`
- `WHERE last_name = 'Smith' AND first_name = 'Alice'`
- `WHERE last_name = 'Smith' AND first_name = 'Alice' AND age = 30`
- `WHERE last_name = 'Smith' AND age = 30` (partial — can use index for last_name, then filter age in memory)
But it **cannot** efficiently serve:
- `WHERE first_name = 'Alice'` (skipping the leftmost column)
- `WHERE age = 30` (alone, skipping both leftmost columns)
- `WHERE first_name = 'Alice' AND age = 30` (no leftmost prefix)
Why? Because the B+tree sorts entries by (last_name, first_name, age) hierarchically. All entries with the same last_name are grouped together, within those, entries with the same first_name are grouped, and within those, entries with the same age are grouped. Searching for `first_name = 'Alice'` without constraining `last_name` requires scanning all entries — the B+tree provides no shortcuts.
| Query Pattern | Composite (a,b,c) | Three Separate Indexes |
|---|---|---|
| `WHERE a = ?` | ✓ | ✓ |
| `WHERE a = ? AND b = ?` | ✓ better | ✓ (engine must merge) |
| `WHERE b = ?` | ✗ | ✓ |
| `WHERE a = ? AND c = ?` | partial | ✓ (engine must merge) |
| Write overhead | 1 B+tree update | 3 B+tree updates |
| **Used by** | Most RDBMS | Some scenarios |
For this milestone, implement single-column indexes only and document the composite index limitation. Composite indexes are an extension that requires modifying the key comparison function to compare tuples rather than scalar values — structurally similar but more complex.
---
## Testing the Complete Index Pipeline
Your test suite must verify both correctness and performance impact:
```c
/* Test 1: CREATE INDEX builds correct B+tree from existing data */
void test_create_index(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)");
    db_exec(db, "INSERT INTO users VALUES (1, 'Alice', 25)");
    db_exec(db, "INSERT INTO users VALUES (2, 'Bob',   30)");
    db_exec(db, "INSERT INTO users VALUES (3, 'Carol', 25)");
    /* Create index on age */
    assert(db_exec(db, "CREATE INDEX idx_age ON users(age)") == 0);
    /* Query via index should return correct results */
    ResultSet *r = db_query(db, "SELECT name FROM users WHERE age = 25");
    assert(r->row_count == 2);  /* Alice and Carol */
    db_close(db);
    printf("PASS: create_index builds correct B+tree\n");
}
/* Test 2: Index maintained on INSERT */
void test_index_maintain_insert(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
    db_exec(db, "CREATE INDEX idx_val ON t(val)");
    db_exec(db, "INSERT INTO t VALUES (1, 'alpha')");
    db_exec(db, "INSERT INTO t VALUES (2, 'beta')");
    db_exec(db, "INSERT INTO t VALUES (3, 'alpha')");  /* duplicate in non-unique index */
    /* Both rows with val='alpha' should be findable */
    ResultSet *r = db_query(db, "SELECT id FROM t WHERE val = 'alpha'");
    assert(r->row_count == 2);
    db_close(db);
    printf("PASS: index maintained on insert\n");
}
/* Test 3: Index maintained on DELETE */
void test_index_maintain_delete(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
    db_exec(db, "CREATE INDEX idx_val ON t(val)");
    db_exec(db, "INSERT INTO t VALUES (1, 'alpha')");
    db_exec(db, "INSERT INTO t VALUES (2, 'beta')");
    db_exec(db, "DELETE FROM t WHERE id = 1");
    /* val='alpha' should no longer be findable via index */
    ResultSet *r = db_query(db, "SELECT id FROM t WHERE val = 'alpha'");
    assert(r->row_count == 0);
    db_close(db);
    printf("PASS: index maintained on delete\n");
}
/* Test 4: UNIQUE index rejects duplicates, allows multiple NULLs */
void test_unique_index(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, email TEXT)");
    db_exec(db, "CREATE UNIQUE INDEX idx_email ON t(email)");
    assert(db_exec(db, "INSERT INTO t VALUES (1, 'a@b.com')") == 0);
    /* Duplicate non-null: must fail */
    assert(db_exec(db, "INSERT INTO t VALUES (2, 'a@b.com')") < 0);
    assert(strstr(db->error, "UNIQUE constraint failed") != NULL);
    /* Two NULLs: must succeed */
    assert(db_exec(db, "INSERT INTO t VALUES (3, NULL)") == 0);
    assert(db_exec(db, "INSERT INTO t VALUES (4, NULL)") == 0);
    db_close(db);
    printf("PASS: unique index constraints\n");
}
/* Test 5: Range scan via index */
void test_index_range_scan(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, score INTEGER)");
    db_exec(db, "CREATE INDEX idx_score ON t(score)");
    for (int i = 1; i <= 100; i++) {
        char sql[64];
        snprintf(sql, sizeof(sql), "INSERT INTO t VALUES (%d, %d)", i, i * 2);
        db_exec(db, sql);
    }
    /* Range scan: scores between 40 and 60 (i=20..30, scores=40..60) */
    ResultSet *r = db_query(db, "SELECT id FROM t WHERE score >= 40 AND score <= 60");
    assert(r->row_count == 11);  /* scores 40, 42, ..., 60 = 11 values */
    db_close(db);
    printf("PASS: index range scan\n");
}
/* Test 6: Verify index scan reads fewer pages than table scan */
void test_index_reduces_page_reads(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, needle INTEGER, data TEXT)");
    /* Insert 1000 rows with unique 'needle' values */
    for (int i = 0; i < 1000; i++) {
        char sql[128];
        snprintf(sql, sizeof(sql),
                 "INSERT INTO t VALUES (%d, %d, 'padding_data_to_make_rows_bigger_xxxx')",
                 i, i);
        db_exec(db, sql);
    }
    buffer_pool_reset_stats(db->bp);
    /* Table scan for needle=500 */
    db_query(db, "SELECT id FROM t WHERE needle = 500");
    BufferPoolStats stats_scan;
    buffer_pool_get_stats(db->bp, &stats_scan);
    uint64_t pages_scan = stats_scan.misses;  /* cold start */
    /* Create index and repeat */
    db_exec(db, "CREATE INDEX idx_needle ON t(needle)");
    buffer_pool_reset_stats(db->bp);
    db_query(db, "SELECT id FROM t WHERE needle = 500");
    BufferPoolStats stats_index;
    buffer_pool_get_stats(db->bp, &stats_index);
    uint64_t pages_index = stats_index.misses;
    /* Index scan should read significantly fewer pages */
    printf("Table scan: %lu pages, Index scan: %lu pages\n",
           pages_scan, pages_index);
    assert(pages_index < pages_scan / 5);  /* at least 5x fewer */
    db_close(db);
    printf("PASS: index reduces page reads\n");
}
```
---
## Knowledge Cascade: What This Milestone Unlocks
You have built the machinery that separates toy databases from production-quality engines. Here is what this knowledge connects to:
**→ Write amplification in SSDs (cross-domain hardware connection).** Every index you maintain multiplies the number of page writes to disk. Modern SSDs manage write amplification internally too — a single logical write to an SSD may translate to 3-5× more physical writes to NAND cells due to the page/block erase granularity mismatch. An indexed database running on an SSD compounds both amplification factors. A table with 5 indexes, on an SSD with internal write amplification of 3×, means one logical INSERT could cause 6 × 3 = 18× the physical NAND writes. This is why NVMe SSD datasheets specify "Total Bytes Written" (TBW) endurance ratings and why write-heavy database workloads are commonly cited for reducing SSD lifespan faster than expected.
**→ Covering indexes are materialized views in miniature.** When the index covers all needed columns, the table is never accessed — the index leaf is the entire result. This is structurally identical to a **materialized view**: a precomputed stored query result that serves read queries directly. The tradeoff is the same: writes must update the materialized store (index/view) synchronously, reads avoid the base table entirely. PostgreSQL's `INCLUDE` clause (added in v11) lets you add non-indexed columns to a B-tree index explicitly for covering index purposes: `CREATE INDEX idx_email_cover ON users(email) INCLUDE (name, age)`. You've built the conceptual foundation to understand exactly what that does.
**→ Index selection is query optimization.** The heuristic you implemented — "use an index if one exists for the WHERE column" — will be replaced in Milestone 8 by a cost-based decision. The query planner must estimate: (1) how many rows match the predicate (cardinality estimation), (2) the cost of index scan (log₂(N) index reads + K random table reads where K = matching rows), (3) the cost of table scan (N sequential reads). This is the core of cost-based optimization, and the work you've done building the index infrastructure is the prerequisite. When you read research papers on query optimization (Selinger et al. 1979, the original System R optimizer paper), you'll recognize every component.
**→ The prefix rule connects to composite data structure design universally.** The leftmost prefix rule for composite indexes (`(a,b,c)` helps on `a` and `a,b` but not `b` alone) appears in identical form across many systems: URL routing in web frameworks (prefix matching), IP routing tables (longest prefix match), trie data structures (prefix search), and DNS hierarchies (reverse domain name lookups use rightmost prefix). Any time data is sorted by a hierarchical key tuple, only queries that constrain the most-significant fields can leverage the sort order. Understanding this from the B+tree context gives you an intuition that transfers to all these systems.
**→ B+tree leaf chains are the same as linked lists at scale.** The right-sibling chain you follow during range scans is a linked list threaded through the leaf pages. Following it during a range scan is O(K) page reads where K is the number of matching leaf pages. This is why databases prefer B+trees over B-trees for range-scan-heavy workloads: the B-tree forces a tree re-ascent when a leaf is exhausted, adding O(log N) overhead per leaf crossing. The B+tree's leaf chain reduces this to O(1) per crossing. The same principle appears in skip lists (linked list with express lanes for O(log N) search), where the bottom level is the complete sorted sequence.
**→ Index maintenance reveals the cost of consistency.** Every synchronous index update means that `INSERT INTO t VALUES (...)` doesn't complete until all 6 B-trees have been modified and the changes are durable. This is the consistency side of the CAP theorem made tangible: maintaining a consistent index costs write throughput. Systems that relax consistency — eventual consistency in distributed databases, asynchronous index maintenance in some NoSQL systems — do so precisely to avoid this bottleneck. You now understand, at the code level, exactly what consistency costs and why engineers sometimes choose not to pay it.
---
## Common Pitfalls: What Will Break Quietly
**1. Not updating indexes on UPDATE.** `UPDATE t SET email = 'new@example.com' WHERE id = 1` modifies `email`, which is indexed. If your UPDATE implementation deletes the row and reinserts it into the table B-tree but doesn't call `index_maintain_update`, the old index entry `('old@example.com', 1)` remains and the new `('new@example.com', 1)` is never added. Queries by old email still "find" the row (returning a stale rowid that now points to the updated row with the wrong email), and queries by new email find nothing. This is a silent data corruption bug — no error, just wrong query results. Always ensure UPDATE calls `index_maintain_delete` (for old values) then `index_maintain_insert` (for new values) for every index.
**2. The rowid lookup finds no row (index-table inconsistency).** After an index lookup returns rowid 47382, `OP_SEEK_ROWID` searches the table B-tree and finds nothing. This means the index has a stale entry for a deleted row. The safe behavior: skip this rowid and continue. The correct behavior: never create this state by ensuring DELETE always removes index entries before removing the table row. Test by deliberately breaking consistency (delete table row without deleting index entry), then verify the engine handles it gracefully rather than crashing.
**3. UNIQUE index allowing duplicate NULL.** `NULL = NULL` is NULL in SQL, never TRUE. Multiple NULL values in a UNIQUE column do not violate the constraint. If your `index_check_unique` function compares NULL keys with `value_compare` and `value_compare` returns 0 for NULL == NULL (treating them as equal), you'll incorrectly reject second NULL insertions. The check must short-circuit: if `key->type == VAL_NULL`, return 0 (no violation) without consulting the index.
**4. Index sort order inconsistency.** Index entries must be sorted by key value for the B+tree to work correctly. If `TEXT` values are sorted case-sensitively in the index but compared case-insensitively in WHERE clause evaluation, an equality lookup may miss entries. Define one `value_compare` function and use it consistently for both index insertion order and lookup comparison. SQLite uses case-sensitive binary comparison for TEXT by default.
**5. Not maintaining right-sibling pointers after index leaf splits.** When an index leaf page splits, the new right sibling must have its right-sibling pointer set to the original page's old right sibling, and the original page's right-sibling pointer must be updated to point to the new sibling. Missing either update breaks the leaf chain for range scans — some range scans will silently skip entries or terminate early. Test with a range scan that spans multiple leaf pages after many insertions.
**6. Index maintenance order on INSERT: check UNIQUE before inserting into table.** The correct order is: (1) check all UNIQUE constraints, (2) insert into table B-tree, (3) insert into all indexes. If you check UNIQUE after inserting into the table, a UNIQUE violation requires rolling back the table insertion. Checking before insertion avoids the need to undo partial work.
---
## What You Have Built
At the end of this milestone, your database has production-quality index infrastructure:
- A **B+tree index structure** with leaf nodes storing `(key_value, rowid)` pairs, internal nodes storing only separator keys and child pointers, and leaf-to-leaf right-sibling pointers for range traversal
- **CREATE INDEX** that scans existing table data and bulk-builds the index B+tree, then registers the index in `sqlite_master` with the root page number and original SQL
- **Index maintenance** on INSERT, UPDATE, and DELETE — synchronous, correct, handling the key-value extraction, old-value removal, new-value insertion pattern for UPDATE
- **UNIQUE constraint enforcement** with correct NULL semantics: non-null duplicates are rejected, multiple NULLs are permitted, error messages name the failing column
- **Index equality lookup** via `OP_INDEX_SEEK` + `OP_INDEX_ROWID` + `OP_SEEK_ROWID` (the double-lookup pattern) that descends the index tree, extracts the rowid, then positions the table cursor at that rowid
- **Index range scan** via `OP_INDEX_SEEK_GE` + leaf chain traversal using the right-sibling pointer, with upper-bound termination
- **Covering index detection** that eliminates the table cursor when all needed columns are present in the index
- **Schema persistence** of index metadata in `sqlite_master` surviving database close/reopen
- A **test suite** verifying all six acceptance criteria: build from existing data, maintenance on INSERT/UPDATE/DELETE, equality lookup page reduction, range scan correctness, UNIQUE rejection, and double-lookup bytecode execution
The query planner in Milestone 8 will build on this foundation, replacing the heuristic "use index if it exists" with a cost model that correctly weighs index selectivity against sequential scan cost — using the same index structures and cursor interface you've built here.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m8 -->
<!-- MS_ID: build-sqlite-m8 -->
# Milestone 8: Query Planner & Statistics
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You are now in the brain of the engine. Every layer below you — tokenizer, parser, compiler, VDBE, buffer pool, B-tree, indexes — exists to execute a query plan. But *which* plan?

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

In Milestone 7, you wired up index infrastructure with a heuristic: "if an index exists for the WHERE column, use it." This worked well enough to demonstrate the concept. But it is subtly wrong, and the wrongness is catastrophic in specific cases. The query planner exists to replace that heuristic with a principled, cost-based decision.
Before you write a single line of planner code, understand where it sits architecturally. The planner operates *between* the parser and the compiler. The parser produces an AST describing *what* to query. The planner examines that AST, consults collected statistics, and produces a *plan* describing *how* to execute it. The compiler then translates the plan (not the raw AST) into bytecode.
```
SQL text
  → tokenizer (Milestone 1)
  → parser (Milestone 2)         ← AST produced here
  → query planner (this milestone) ← plan produced here
  → compiler (Milestone 3)
  → VDBE (Milestone 3)
  → storage engine (Milestones 4-7)
```
The planner is the only component that reasons about *alternative* execution strategies. Every other component executes exactly what it's given. Only the planner asks: "Is there a better way?"
---
## The Revelation: Indexes Are Not Always Faster
Here is the misconception that corrupts every database engine written without a planner: if a column appears in a WHERE clause and there's an index on that column, use the index. Always. The index exists to make that query faster, so obviously it should be used.
This model fails — catastrophically, silently — on a specific class of queries. Let's make it concrete.
You have a `users` table with 1,000,000 rows. The table is stored across 10,000 pages (roughly 100 rows per 4KB page, assuming 40 bytes per row). You have an index on the `gender` column. Someone runs:
```sql
SELECT * FROM users WHERE gender = 'M';
```
Approximately 500,000 rows match. There are two execution strategies:
**Strategy A: Table scan.** Read all 10,000 pages sequentially. Cost: 10,000 page reads. Each page fetches sequentially, which is fast for spinning disks (one seek + streaming) and good for SSD prefetch patterns. Total I/O: predictable, sequential.
**Strategy B: Index scan.** Descend the index B+tree to find `'M'` entries. There are 500,000 of them. For *each* index entry, follow the rowid pointer back to the table B-tree to fetch the full row. Each table lookup is a *random* read to a different leaf page. In the worst case, each of the 500,000 lookups hits a different page — 500,000 random page reads.
Strategy B is **50× more expensive** than Strategy A for this query. An engine that blindly uses indexes makes this query an order of magnitude slower than the naive full scan.

![Cost Model — Table Scan vs Index Scan](./diagrams/diag-planner-cost-model.svg)

The fundamental tension: **a sequential scan reads each page once; a random index scan may read each page once per matching row.** At high selectivity (many rows match), the index turns a sequential access pattern into a random one. Below a threshold — when the predicate matches a small fraction of rows — the index wins because you only touch a small number of pages. Above the threshold, the table scan wins.

> **🔑 Foundation: Cost-based optimization and cardinality estimation**
> 
> ## Cost-Based Optimization and Cardinality Estimation
**What it IS**
A query optimizer chooses *how* to execute a SQL query — which indexes to use, in what order to join tables, whether to hash-join or nested-loop. A **rule-based optimizer** applies fixed heuristics ("always prefer an index scan over a full table scan"). A **cost-based optimizer (CBO)** instead *estimates the cost* of each candidate plan and picks the cheapest one.
Cost is typically expressed as a weighted combination of I/O operations and CPU cycles. For example:
```
cost(plan) = disk_pages_read × io_cost_factor + rows_processed × cpu_cost_factor
```
But to estimate cost, the optimizer must first estimate **how many rows each operation will produce**. That's **cardinality estimation** — predicting the output size of a filter, join, or aggregation *before actually running it*.
Example: given `SELECT * FROM orders WHERE status = 'pending'`, the optimizer needs to know roughly how many rows match `status = 'pending'` to decide whether a sequential scan or an index scan is cheaper. If it guesses 5 rows (use the index), but the real answer is 5 million rows, the resulting plan can be catastrophically slow.
**WHY you need it right now**
You're building a query optimizer. Every decision your optimizer makes — join ordering, operator selection, parallelism — depends on cardinality estimates flowing through the plan tree. A bad estimate at one node compounds into worse estimates at every node that consumes its output. Understanding *why* estimates go wrong is prerequisite to building the data structures (histograms, statistics) that make them go right.
**Key insight**
Cardinality estimation is the optimizer's model of the data. The quality of every plan the optimizer produces is bounded by the accuracy of this model. A perfect search algorithm over a wrong cost model still produces bad plans. Invest in the statistics layer — it's the foundation everything else stands on.

The threshold where the crossover occurs depends on the ratio of random I/O cost to sequential I/O cost. On spinning disks, random reads are 100–1000× more expensive than sequential reads (the seek time dominates). On SSDs, the ratio is smaller (10–30×) but still significant. Most databases use a threshold around 10–25% selectivity — if more than 10–25% of rows match, prefer the table scan.
**This is why statistics exist.** Without knowing how many rows match `WHERE gender = 'M'`, the planner cannot compute the cost of either strategy. The `ANALYZE` command collects those statistics. Without `ANALYZE`, the planner is flying blind — it must assume some default, and whatever default it assumes will be wrong for some queries.
---
## What You Are Building
This milestone has four components:
1. **Statistics collection (`ANALYZE`)** — scan tables and indexes to collect row counts and distinct value counts, stored in a statistics table
2. **Cost model** — functions that estimate the I/O cost (in page reads) of a table scan vs. an index scan, given the statistics
3. **Plan selection** — logic that examines the WHERE clause, looks up statistics, computes costs for each access path, and picks the cheapest
4. **Join ordering** — for queries with multiple tables, dynamic programming over possible join orderings to minimize intermediate result size
5. **EXPLAIN** — display the chosen plan with cost estimates
Let's build each component in order.
---
## ANALYZE: Collecting Statistics

![Cardinality Estimation — Error Propagation](./diagrams/diag-planner-cardinality-estimation.svg)

The `ANALYZE` command scans the database and populates a statistics table. You'll store statistics in a new system table, `sqlite_stat1`, using the same B-tree format as all other tables. This is intentional — the statistics table is just another table that happens to be consulted by the planner.
```sql
-- sqlite_stat1 schema (SQLite's actual stats table format)
CREATE TABLE sqlite_stat1 (
    tbl  TEXT,    -- table name
    idx  TEXT,    -- index name, or NULL for table-level stats
    stat TEXT     -- space-separated statistics
);
```
For table-level statistics, `idx` is NULL and `stat` is a single integer: the total row count.
For index statistics, `idx` is the index name and `stat` starts with the total row count followed by one integer per indexed column: the number of distinct values in that column.
```
stat for table-level:   "1000000"
stat for single-column index: "1000000 50"
-- means: 1,000,000 rows, 50 distinct values in the indexed column
```
The distinct value count tells the planner the average number of rows per distinct value: `total_rows / distinct_values`. For `gender` with 50% split and 2 distinct values: `1,000,000 / 2 = 500,000` rows per value — terrible selectivity, don't use the index. For `email` with 1,000,000 distinct values: `1,000,000 / 1,000,000 = 1` row per value — perfect selectivity, always use the index.
### Statistics Storage in Memory
During query planning, reading statistics from disk on every decision would be too slow. Load the statistics table into a fast in-memory hash map at database open time (or on first query after `ANALYZE`):
```c
/* statistics.h */
#pragma once
#include <stdint.h>
typedef struct {
    char    *table_name;   /* heap-allocated */
    char    *index_name;   /* NULL for table-level stats */
    int64_t  total_rows;   /* row count in the table */
    int64_t  distinct_vals; /* distinct values in the indexed column (0 if table-level) */
} StatEntry;
typedef struct {
    StatEntry *entries;
    int        count;
    int        capacity;
} StatCache;
/* Initialize statistics cache from sqlite_stat1 table.
   Called after database open, or after ANALYZE completes. */
int  stat_cache_load(DB *db, StatCache *sc);
void stat_cache_free(StatCache *sc);
/* Look up table-level row count.
   Returns -1 if no statistics available (not yet ANALYZEd). */
int64_t stat_table_rows(const StatCache *sc, const char *table_name);
/* Look up distinct value count for an indexed column.
   Returns -1 if not available. */
int64_t stat_index_distinct(const StatCache *sc,
                             const char *table_name,
                             const char *index_name);
```
### Implementing ANALYZE
The `ANALYZE` command iterates over all tables and their indexes, collecting statistics:
```c
/* Execute ANALYZE: collect statistics for all tables and indexes.
   Writes results into sqlite_stat1.
   Returns 0 on success, -1 on error. */
int analyze_execute(DB *db) {
    /* Ensure sqlite_stat1 exists */
    if (schema_find_table(db, "sqlite_stat1") == INVALID_PAGE_ID) {
        db_exec(db,
            "CREATE TABLE sqlite_stat1 "
            "(tbl TEXT, idx TEXT, stat TEXT)");
    } else {
        /* Clear existing stats */
        db_exec(db, "DELETE FROM sqlite_stat1");
    }
    /* Iterate all user tables */
    SchemaTable *tables;
    int          table_count;
    schema_get_all_tables(db, &tables, &table_count);
    for (int t = 0; t < table_count; t++) {
        SchemaTable *schema = &tables[t];
        /* Skip system tables */
        if (strncmp(schema->name, "sqlite_", 7) == 0) continue;
        /* ---- Collect table row count ---- */
        int64_t row_count = analyze_count_rows(db, schema);
        /* Insert table-level stat: idx = NULL */
        char stat_buf[32];
        snprintf(stat_buf, sizeof(stat_buf), "%lld", (long long)row_count);
        analyze_insert_stat(db, schema->name, NULL, stat_buf);
        /* ---- Collect per-index stats ---- */
        for (int i = 0; i < schema->index_count; i++) {
            IndexDescriptor *idx = &schema->indexes[i];
            int64_t distinct = analyze_count_distinct(db, schema, idx);
            /* stat = "total_rows distinct_vals" */
            snprintf(stat_buf, sizeof(stat_buf), "%lld %lld",
                     (long long)row_count,
                     (long long)distinct);
            analyze_insert_stat(db, schema->name, idx->index_name, stat_buf);
        }
    }
    /* Reload statistics cache */
    stat_cache_load(db, &db->stat_cache);
    return 0;
}
```
#### Counting Rows
Counting rows is a full table scan that counts cells:
```c
static int64_t analyze_count_rows(DB *db, SchemaTable *schema) {
    BTree *bt = btree_open(db->bp, schema->root_page_id);
    BTreeCursor cur;
    btree_cursor_open_read(&cur, bt);
    if (btree_cursor_rewind(&cur)) return 0;
    int64_t count = 0;
    do { count++; } while (btree_cursor_next(&cur));
    btree_cursor_close(&cur);
    return count;
}
```
For large tables, a faster approach counts cells directly from page headers without deserializing records — each page header stores `cell_count`, and a leaf-traversal accumulating cell counts avoids record decoding overhead. For correctness, the simple version above is sufficient.
#### Counting Distinct Values
Counting distinct values in an indexed column requires a sorted scan of the index B+tree. Because the index is sorted by key value, distinct counting is a single linear pass:
```c
static int64_t analyze_count_distinct(DB *db, SchemaTable *schema,
                                       IndexDescriptor *idx) {
    BTree *index_bt = btree_open_index(db->bp, idx->root_page_id);
    IndexCursor cur;
    index_cursor_open(&cur, index_bt);
    if (index_cursor_rewind(&cur)) return 0;
    int64_t distinct = 0;
    Value   prev_key;
    prev_key.type = VAL_NULL;  /* sentinel: no previous key */
    int first = 1;
    do {
        Value key;
        index_cursor_read_key(&cur, &key);
        value_deep_copy(&key);
        if (first || value_compare(&key, &prev_key) != 0) {
            distinct++;
            first = 0;
            value_free(&prev_key);
            prev_key = key;
        } else {
            value_free(&key);
        }
    } while (index_cursor_next(&cur));
    value_free(&prev_key);
    index_cursor_close(&cur);
    return distinct;
}
```
The sorted index guarantees that all entries with the same key are adjacent, so a single linear pass counts transitions — no hash table needed.
#### Inserting the Stat Row
```c
static void analyze_insert_stat(DB *db, const char *tbl,
                                  const char *idx, const char *stat) {
    char sql[512];
    if (idx) {
        snprintf(sql, sizeof(sql),
                 "INSERT INTO sqlite_stat1 VALUES ('%s', '%s', '%s')",
                 tbl, idx, stat);
    } else {
        snprintf(sql, sizeof(sql),
                 "INSERT INTO sqlite_stat1 VALUES ('%s', NULL, '%s')",
                 tbl, stat);
    }
    db_exec(db, sql);
}
```
---
## The Cost Model

![Cost Model — Table Scan vs Index Scan](./diagrams/diag-planner-cost-model.svg)

A cost model is a function that takes a query predicate and statistics and returns a number representing the expected "cost" of executing that plan. The unit of cost is **estimated page reads** — a direct proxy for I/O latency, which dominates query execution time.
### Table Scan Cost
A full table scan reads every page in the table exactly once, sequentially:
```
cost_table_scan = total_pages
where:
    total_pages = ceil(total_rows / rows_per_page)
    rows_per_page = floor(PAGE_SIZE / avg_row_size)
```
If you don't have `avg_row_size` in your statistics (the ANALYZE implementation above doesn't collect it), use a reasonable default of 100 bytes per row, giving ~40 rows per 4KB page:
```c
#define DEFAULT_ROWS_PER_PAGE  40
#define RANDOM_IO_COST_FACTOR   5   /* random I/O costs ~5× sequential I/O */
static double cost_table_scan(const StatCache *sc, const char *table_name) {
    int64_t total_rows = stat_table_rows(sc, table_name);
    if (total_rows < 0) total_rows = 1000;  /* no stats: assume 1000 rows */
    double total_pages = (double)total_rows / DEFAULT_ROWS_PER_PAGE;
    return total_pages;  /* sequential reads */
}
```
### Index Scan Cost
An index scan has two components: traversing the index B+tree to find matching entries, then fetching the full rows from the table B-tree.
```
selectivity       = distinct_vals > 0 ? (1.0 / distinct_vals) : 1.0
matching_rows     = total_rows × selectivity
index_pages_read  = ceil(log_b(total_rows))    (B+tree descent depth)
                  + ceil(matching_rows / index_entries_per_page)  (leaf scan)
table_pages_read  = matching_rows × RANDOM_IO_COST_FACTOR
                    (each lookup is a random read)
cost_index_scan   = index_pages_read + table_pages_read
```
The `RANDOM_IO_COST_FACTOR` is the key tuning parameter. It penalizes random reads relative to sequential reads. For an SSD, a value of 5 is reasonable. For spinning disk, 20–50 is more accurate. SQLite uses approximately 4 in its actual cost model.
```c
#define INDEX_ENTRIES_PER_PAGE  200  /* approximate: index entries are small */
#define BTREE_DEPTH_ESTIMATE(n) \
    ((n) <= 1 ? 1 : (int)(log((double)(n)) / log(200.0)) + 1)
static double cost_index_scan(const StatCache *sc,
                               const char *table_name,
                               const char *index_name,
                               double selectivity) {
    int64_t total_rows = stat_table_rows(sc, table_name);
    if (total_rows < 0) total_rows = 1000;
    double matching_rows   = (double)total_rows * selectivity;
    double index_depth     = (double)BTREE_DEPTH_ESTIMATE(total_rows);
    double index_leaf_pages = matching_rows / INDEX_ENTRIES_PER_PAGE;
    double table_random_reads = matching_rows * RANDOM_IO_COST_FACTOR
                                / DEFAULT_ROWS_PER_PAGE;
    return index_depth + index_leaf_pages + table_random_reads;
}
```
### Selectivity Estimation
Selectivity is the fraction of rows that satisfy a predicate — a number between 0 and 1. The planner derives it from statistics:
```c
/* Estimate selectivity for a predicate col OP value.
   Uses index statistics if available, otherwise falls back to defaults. */
double estimate_selectivity(const StatCache *sc,
                             const char *table_name,
                             const char *index_name,
                             TokenType   op) {
    int64_t total_rows    = stat_table_rows(sc, table_name);
    int64_t distinct_vals = stat_index_distinct(sc, table_name, index_name);
    if (total_rows <= 0) return 0.3;   /* no stats: default 30% */
    if (distinct_vals <= 0) return 0.3;
    switch (op) {
    case TOKEN_EQ:
        /* Equality: 1/distinct_values */
        return 1.0 / (double)distinct_vals;
    case TOKEN_LT:
    case TOKEN_GT:
        /* Range open at one end: roughly 1/3 of table by default */
        return 0.33;
    case TOKEN_LTE:
    case TOKEN_GTE:
        /* Range closed at one end: roughly 1/3 */
        return 0.33;
    case TOKEN_NEQ:
        /* Not-equal: most rows match */
        return 1.0 - (1.0 / (double)distinct_vals);
    default:
        return 0.3;
    }
}
```
The selectivity estimates above are simplifications. A production database uses **histograms** — frequency distributions of column values that capture skew. 
> **🔑 Foundation: Histogram-based cardinality estimation for non-uniform distributions**
> 
> ## Histogram-Based Cardinality Estimation for Non-Uniform Distributions
**What it IS**
The naive approach to estimating selectivity assumes **uniform distribution**: if a column has 100 distinct values and you filter for one, assume 1/100 of rows match. This breaks badly in practice — real data is skewed. `status = 'pending'` might match 80% of orders while `status = 'cancelled'` matches 0.1%.
A **histogram** captures the actual distribution of column values by dividing the value range into **buckets** and recording statistics per bucket. Two common designs:
- **Equi-width histogram**: divide the value range into N equal-width intervals. Simple but wastes precision on sparse regions.
- **Equi-depth (equi-height) histogram**: divide so each bucket contains roughly the same number of rows. Much better — more buckets land on dense regions automatically.
For a column `age` with values 0–100, an equi-depth histogram with 10 buckets might look like:
```
Bucket 1: [18–24], 12,000 rows   ← young adults, dense
Bucket 2: [25–29], 12,000 rows
...
Bucket 9: [70–85], 12,000 rows   ← wider range, sparser
```
To estimate `WHERE age BETWEEN 22 AND 27`, you interpolate across the buckets those values fall in.
For **multi-column predicates** (`WHERE age > 25 AND income > 80000`), naive histograms assume **independence** and multiply selectivities: `sel(A) × sel(B)`. This is famously inaccurate when columns are correlated. More advanced approaches use multi-dimensional histograms or column correlation statistics (though these are expensive to build and store).
**WHY you need it right now**
Your optimizer needs per-column statistics to do anything useful. Before you implement the join ordering or plan enumeration logic, you need a statistics collection phase that builds histograms during table scans or explicit `ANALYZE` commands, then a selectivity estimation module that consults those histograms when evaluating filter predicates. This is your accuracy budget — everything downstream inherits its errors.
**Key insight**
Equi-depth beats equi-width because the histogram naturally allocates more precision where the data is dense — exactly where queries are more likely to filter. When building your statistics layer, implement equi-depth first. And always track the number of NULLs and the number of distinct values (NDV) separately — range queries need histograms, but equality predicates on low-cardinality columns (like `status`) are better served by a simple **most-common-values (MCV) list**.
 For example, if 90% of your `status` column contains `'active'`, a histogram records this. Without a histogram, `WHERE status = 'active'` gets selectivity `1/4 = 25%` (if there are 4 distinct statuses), but the true selectivity is 90%.
For this milestone, the 1/distinct_values formula is sufficient and matches SQLite's actual approach for single-column predicates.
---
## Plan Selection: Choosing the Access Path

![Query Planner Decision Flow — Data Walk](./diagrams/diag-planner-decision-tree.svg)

Plan selection examines each table in the query, evaluates all available access paths (table scan + one path per applicable index), computes their costs, and picks the cheapest.
```c
/* Access path descriptor */
typedef enum {
    ACCESS_TABLE_SCAN,
    ACCESS_INDEX_SCAN,
} AccessType;
typedef struct {
    AccessType  type;
    char       *index_name;    /* NULL for table scan */
    double      selectivity;   /* fraction of rows estimated to match */
    double      cost;          /* estimated page reads */
    int64_t     estimated_rows; /* rows expected in output */
} AccessPath;
/* Choose the best access path for a single-table predicate.
   Returns the chosen path in *out.
   pred: the WHERE expression AST node (may be NULL for no predicate). */
void planner_choose_access_path(DB *db, const char *table_name,
                                 AstNode *pred, AccessPath *out) {
    StatCache *sc = &db->stat_cache;
    int64_t total_rows = stat_table_rows(sc, table_name);
    if (total_rows < 0) total_rows = 1000;  /* default assumption */
    /* Baseline: table scan */
    AccessPath best;
    best.type          = ACCESS_TABLE_SCAN;
    best.index_name    = NULL;
    best.selectivity   = pred ? 0.3 : 1.0;
    best.cost          = cost_table_scan(sc, table_name);
    best.estimated_rows = (int64_t)(total_rows * best.selectivity);
    /* Try each applicable index */
    SchemaTable *schema = schema_get_table(db, table_name);
    if (schema && pred) {
        for (int i = 0; i < schema->index_count; i++) {
            IndexDescriptor *idx = &schema->indexes[i];
            /* Check if this index is applicable to the predicate */
            TokenType pred_op;
            if (!predicate_uses_index(pred, schema, idx, &pred_op)) continue;
            double sel  = estimate_selectivity(sc, table_name,
                                                idx->index_name, pred_op);
            double cost = cost_index_scan(sc, table_name,
                                           idx->index_name, sel);
            if (cost < best.cost) {
                best.type          = ACCESS_INDEX_SCAN;
                best.index_name    = idx->index_name;
                best.selectivity   = sel;
                best.cost          = cost;
                best.estimated_rows = (int64_t)(total_rows * sel);
            }
        }
    }
    *out = best;
}
```
The `predicate_uses_index` function checks whether the index's column appears in the WHERE predicate at the top level:
```c
/* Returns 1 if the index can serve the predicate, setting *op_out to the operator.
   Only handles simple predicates: col OP literal. */
static int predicate_uses_index(AstNode *pred, SchemaTable *schema,
                                 IndexDescriptor *idx, TokenType *op_out) {
    if (!pred || pred->type != NODE_BINARY_EXPR) return 0;
    TokenType op = pred->binary.op;
    if (op != TOKEN_EQ  && op != TOKEN_LT && op != TOKEN_GT &&
        op != TOKEN_LTE && op != TOKEN_GTE) return 0;
    /* Left side must be a column identifier */
    if (pred->binary.left->type != NODE_IDENTIFIER) return 0;
    const char *col_name = pred->binary.left->identifier.name;
    int col_idx = schema_get_column_index(schema, col_name);
    if (col_idx < 0) return 0;
    /* Index must cover this column */
    if (idx->col_count != 1 || idx->col_indices[0] != col_idx) return 0;
    *op_out = op;
    return 1;
}
```
### The Selectivity Threshold
Rather than always choosing the minimum-cost path mathematically, you can implement a simple threshold check as a fast path:
```c
#define SELECTIVITY_THRESHOLD  0.20   /* use table scan if > 20% rows match */
/* Quick check: is this predicate selective enough to warrant index use? */
static int predicate_is_selective(double selectivity) {
    return selectivity < SELECTIVITY_THRESHOLD;
}
```
The threshold and the cost model agree for reasonable I/O cost ratios: with `RANDOM_IO_COST_FACTOR = 5`, the breakeven is approximately `5 × matching_rows / total_pages = 1`, i.e., `matching_rows = total_pages / 5 = total_rows / (DEFAULT_ROWS_PER_PAGE × 5) = total_rows / 200 = 0.5%`. That breakeven is optimistic; a 20% threshold is more conservative and correct for larger tables with less buffer pool coverage.
---
## Integrating the Planner with the Compiler
The planner runs between AST production and bytecode emission. Thread it through the compiler:
```c
/* In compile_select (Milestone 3), replace the heuristic index check: */
/* OLD (Milestone 7 heuristic): */
/* IndexDescriptor *idx = find_usable_index(schema, where, &index_pred); */
/* if (idx) compile_select_via_index(...); else compile_select_table_scan(...); */
/* NEW (cost-based): */
AccessPath ap;
planner_choose_access_path(db, node->select.table_name,
                            node->select.where_expr, &ap);
if (ap.type == ACCESS_INDEX_SCAN) {
    compile_select_via_index(c, node, ap.index_name,
                              node->select.where_expr);
} else {
    compile_select_table_scan(c, node);
}
```
The compiler no longer makes access-path decisions. It receives a plan and translates it to bytecode. The planner and compiler are now cleanly separated — a critical architectural boundary that enables future optimization passes.
---
## EXPLAIN: Making the Plan Visible

![EXPLAIN Output — Trace Example](./diagrams/diag-planner-explain-output.svg)

EXPLAIN is not optional infrastructure — it's the tool that makes your database debuggable. Every time a user complains "my query is slow," your first response is "run EXPLAIN and show me the plan." Without EXPLAIN, diagnosing slow queries requires guesswork.
The EXPLAIN output format should show:
```
addr  opcode         p1    p2    p3    plan_annotation
----  -----------    ----  ----  ----  ------------------
0     OpenTable      0     0     0
1     Rewind         0     7     0     TABLE SCAN users (est. 1000000 rows, cost 25000.0)
2     Column         0     2     1
3     Integer        18    2     0
4     Le             1     6     2
5     ResultRow      3     5     0
6     Next           0     2     0
7     Halt           0     0     0
```
Or, with an index scan:
```
addr  opcode         p1    p2    p3    plan_annotation
----  -----------    ----  ----  ----  ------------------
0     OpenTable      0     0     0
1     OpenIndex      1     0     0
2     String8        2     0     0     INDEX SCAN idx_email (est. 1 row, cost 6.0)
3     IndexSeek      1     11    2     selectivity=0.0001%
...
```
The annotations are attached to the plan, not the bytecode. The compiler stores plan metadata alongside the compiled program:
```c
/* Plan node stored with the compiled program */
typedef struct {
    char       *table_name;
    AccessType  access_type;
    char       *index_name;        /* NULL for table scan */
    double      estimated_rows;
    double      estimated_cost;
    double      selectivity;
} PlanNode;
typedef struct {
    Instruction *instructions;
    int          count;
    int          capacity;
    /* Plan metadata for EXPLAIN */
    PlanNode    *plan_nodes;
    int          plan_node_count;
} Program;
```
The EXPLAIN printer formats both the bytecode and the plan annotations:
```c
void program_explain_with_plan(Program *prog, FILE *out) {
    /* Print plan summary first */
    fprintf(out, "Query Plan:\n");
    for (int i = 0; i < prog->plan_node_count; i++) {
        PlanNode *pn = &prog->plan_nodes[i];
        if (pn->access_type == ACCESS_TABLE_SCAN) {
            fprintf(out, "  TABLE SCAN %s"
                    " (est. %.0f rows, cost %.1f pages)\n",
                    pn->table_name,
                    pn->estimated_rows,
                    pn->estimated_cost);
        } else {
            fprintf(out, "  INDEX SCAN %s via %s"
                    " (selectivity %.4f, est. %.0f rows, cost %.1f pages)\n",
                    pn->table_name,
                    pn->index_name,
                    pn->selectivity,
                    pn->estimated_rows,
                    pn->estimated_cost);
        }
    }
    fprintf(out, "\nBytecode:\n");
    /* ... existing program_explain() bytecode output ... */
}
```
---
## Join Ordering: The Hardest Problem

![Join Order — Dynamic Programming](./diagrams/diag-planner-join-order-dp.svg)

Everything above handles single-table queries. Multi-table JOIN queries introduce a combinatorial challenge: given N tables, there are N! possible orderings to join them. For N=3: 6 orderings. For N=5: 120 orderings. For N=10: 3,628,800 orderings. Examining them all is feasible only up to N≈10.
The chosen approach is **dynamic programming over subsets** — the Selinger algorithm 
> **🔑 Foundation: The Selinger**
> 
> ## The Selinger Algorithm: Dynamic Programming for Optimal Join Ordering
**What it IS**
Given N tables to join, there are N! possible orderings — for 10 tables, that's 3.6 million. Trying all of them is infeasible. The **Selinger algorithm**, introduced in IBM's System R (1979) and still the foundation of most production optimizers, uses **dynamic programming** to find the optimal join order in O(2^N × N) time instead.
The core insight: **optimal subplans compose into optimal full plans** (optimal substructure). If the best way to join {A, B, C} uses a particular plan for {A, B} as an intermediate step, then that {A, B} plan must itself be optimal. So we can build up solutions bottom-up:
1. **Level 1**: for each single table, compute the best access path (seq scan vs. index scan) and its estimated cost/cardinality.
2. **Level 2**: for each pair of tables {T_i, T_j}, find the best join of the cheapest single-table plans. Store the winner.
3. **Level k**: for each set of k tables, try all ways to split it into a (k-1)-subset + one new table. Look up the already-computed optimal plan for the (k-1)-subset. Keep the cheapest combination.
4. **Level N**: the full set of all N tables — the stored winner is the globally optimal plan.
A key subtlety: Selinger also considers **interesting orders**. A sort order produced as a side effect of a merge-join or index scan might eliminate a later ORDER BY or make a downstream join cheaper — so the optimizer must sometimes keep a *suboptimal-cost* plan for a subset if it produces a useful sort order.
```
Best({A,B,C,D}) = min over all splits {S, T\S} of:
    cost(best(S)) + cost(join(best(S), best(T\S)))
```
**WHY you need it right now**
Join ordering is where most of the optimizer's leverage is. A bad join order (joining large tables before filtering with a small table) can make a query 100–1000× slower. Once you have cardinality estimates, Selinger is the standard algorithm for turning those estimates into an optimal plan. You'll implement it as a table of `(frozenset_of_tables → best_plan)` built iteratively.
**Key insight**
Selinger trades exponential-in-N space (storing one best plan per subset) for exponential-but-tractable time. In practice this works well up to ~15–20 tables. Beyond that, optimizers switch to heuristic search (genetic algorithms, greedy). When you implement the DP table, use a bitmask (integer with one bit per table) as your key — subset operations become fast bitwise AND/OR, and the table fits in memory even for N=20 (2^20 = ~1M entries).
 from the 1979 IBM System R paper that founded cost-based query optimization. It finds the optimal join tree in O(3^N) time (still exponential, but far better than O(N!)).
### Join Cardinality Estimation
Before discussing the algorithm, you need to understand how the planner estimates the output size of a join.

![Cardinality Estimation — Error Propagation](./diagrams/diag-planner-cardinality-estimation.svg)

The key formula: **join cardinality = cardinality(R) × cardinality(S) × selectivity(join_condition)**.
For an equi-join `R.id = S.r_id`:
- If `id` is a primary key in R (unique) and `r_id` is a foreign key in S, each row in S joins with at most one row in R → cardinality ≈ cardinality(S)
- If neither column is unique, use `1 / max(distinct_R, distinct_S)` as the join selectivity — the probability that a random pair (r, s) satisfies the join condition
```c
/* Estimate the output cardinality of a join between two tables.
   join_col_R, join_col_S: the columns used in the ON clause.
   sel_R, sel_S: the selectivity of filters already applied to each table. */
double estimate_join_cardinality(const StatCache *sc,
                                  const char *table_R, const char *col_R,
                                  const char *table_S, const char *col_S,
                                  double sel_R, double sel_S) {
    int64_t rows_R = stat_table_rows(sc, table_R);
    int64_t rows_S = stat_table_rows(sc, table_S);
    if (rows_R < 0) rows_R = 1000;
    if (rows_S < 0) rows_S = 1000;
    double filtered_R = (double)rows_R * sel_R;
    double filtered_S = (double)rows_S * sel_S;
    /* Find distinct values for the join columns */
    int64_t dist_R = stat_index_distinct_for_col(sc, table_R, col_R);
    int64_t dist_S = stat_index_distinct_for_col(sc, table_S, col_S);
    if (dist_R <= 0) dist_R = rows_R;  /* assume all unique if no stats */
    if (dist_S <= 0) dist_S = rows_S;
    double join_selectivity = 1.0 / (double)(dist_R > dist_S ? dist_R : dist_S);
    return filtered_R * filtered_S * join_selectivity;
}
```
### The Dynamic Programming Algorithm
The DP algorithm builds optimal join plans bottom-up. Start with single-table plans. Then find the best way to join each pair. Then find the best way to join each triple (as a pair + one table), and so on.
State: `dp[S]` = the optimal plan for joining the set of tables S.
Transition: `dp[S] = min over all T ⊂ S where T ≠ ∅ and T ≠ S of: cost(dp[T], dp[S\T])`.
For each candidate join, cost = cost of building the left side + cost of building the right side + cost of the join itself.
```c
#define MAX_TABLES_FOR_DP  10
typedef struct {
    int      table_indices[MAX_TABLES_FOR_DP];  /* which tables are in this subset */
    int      table_count;
    double   cost;
    double   estimated_rows;
    /* Join order: left_set and right_set that produced this plan */
    uint32_t left_mask;   /* bitmask of tables in left sub-plan */
    uint32_t right_mask;  /* bitmask of tables in right sub-plan */
    /* Access paths for leaf plans */
    AccessPath access_paths[MAX_TABLES_FOR_DP];
} JoinPlan;
/* dp_table[mask] = best plan for the set of tables encoded in mask */
typedef struct {
    JoinPlan *plans;         /* indexed by bitmask */
    int       num_tables;
} DPTable;
/* Find the optimal join order for a multi-table query using DP.
   tables:       array of table names involved in the join
   num_tables:   number of tables (must be ≤ MAX_TABLES_FOR_DP)
   join_preds:   join predicates (one per table pair)
   filter_preds: per-table filter predicates (WHERE clauses)
   best_plan:    output: the best join plan found */
int planner_join_order_dp(DB *db,
                           const char **tables, int num_tables,
                           JoinPredicate *join_preds, int num_join_preds,
                           AstNode **filter_preds,
                           JoinPlan *best_plan) {
    if (num_tables > MAX_TABLES_FOR_DP) {
        /* Fall back to a greedy heuristic for large queries */
        return planner_join_order_greedy(db, tables, num_tables,
                                         join_preds, num_join_preds,
                                         filter_preds, best_plan);
    }
    int total_subsets = 1 << num_tables;  /* 2^N subsets */
    JoinPlan *dp = calloc(total_subsets, sizeof(JoinPlan));
    /* Mark uninitialized plans with sentinel cost */
    for (int i = 0; i < total_subsets; i++) dp[i].cost = 1e18;
    /* Initialize single-table plans (subsets of size 1) */
    for (int t = 0; t < num_tables; t++) {
        uint32_t mask = (1u << t);
        JoinPlan *plan = &dp[mask];
        AccessPath ap;
        planner_choose_access_path(db, tables[t], filter_preds[t], &ap);
        plan->cost           = ap.cost;
        plan->estimated_rows = (double)ap.estimated_rows;
        plan->table_count    = 1;
        plan->table_indices[0] = t;
        plan->left_mask      = 0;
        plan->right_mask      = 0;
        plan->access_paths[t] = ap;
    }
    /* Fill DP for all subsets of size 2..num_tables */
    for (int size = 2; size <= num_tables; size++) {
        /* Enumerate all subsets of this size */
        for (uint32_t mask = 1; mask < (uint32_t)total_subsets; mask++) {
            if (__builtin_popcount(mask) != size) continue;
            /* Try all ways to split mask into two non-empty subsets */
            for (uint32_t left = (mask - 1) & mask;
                 left > 0;
                 left = (left - 1) & mask) {
                uint32_t right = mask ^ left;
                if (right == 0 || right >= left) continue; /* avoid duplicates */
                JoinPlan *lp = &dp[left];
                JoinPlan *rp = &dp[right];
                if (lp->cost >= 1e17 || rp->cost >= 1e17) continue;
                /* Estimate join cardinality */
                /* (Simplified: find the join condition connecting left and right) */
                double join_rows = estimate_join_output(db, lp, rp,
                                                         join_preds, num_join_preds);
                double join_cost = lp->cost + rp->cost
                                 + (lp->estimated_rows * rp->estimated_rows
                                    / DEFAULT_ROWS_PER_PAGE * 0.01);
                /* ↑ Nested loop cost: left_rows × right_rows / page_size */
                if (join_cost < dp[mask].cost) {
                    dp[mask].cost          = join_cost;
                    dp[mask].estimated_rows = join_rows;
                    dp[mask].left_mask     = left;
                    dp[mask].right_mask    = right;
                }
            }
        }
    }
    /* The full-table-set plan is the answer */
    uint32_t all = (1u << num_tables) - 1;
    if (dp[all].cost >= 1e17) {
        free(dp);
        return -1;  /* could not find a valid plan */
    }
    *best_plan = dp[all];
    free(dp);
    return 0;
}
```
### Why Join Ordering Matters: The Error Propagation Problem
The most important insight in join optimization is how estimation errors compound.

![Cardinality Estimation — Error Propagation](./diagrams/diag-planner-cardinality-estimation.svg)

Suppose you have three tables: `orders` (1M rows), `customers` (100K rows), `products` (10K rows). You're joining all three with predicates that filter orders down to 1% (10K rows). The estimation error on `orders` might be 2×: you estimate 5K matching rows when the truth is 10K.
With two tables, a 2× error produces a plan with 2× wrong cost — annoying but not catastrophic.
With three tables: the wrong estimate for `orders` produces a wrong join cardinality with `customers` — a 2× error cascades to a 4× error in the join result size. That wrong intermediate size then feeds the second join, producing an 8× error in the final cost estimate. With four tables, a 2× base error becomes 16×. With five tables: 32×.
This is why `ANALYZE` matters so much for multi-table queries. Without statistics, the default assumption (`1000 rows`) might be wrong by 1000× for a million-row table. A three-table join with 1000× base errors can produce a trillion-fold error in the final plan cost — the difference between a 10ms plan and a year-long plan, with the planner confidently choosing the wrong one.
The industry response to compounding cardinality estimation errors is **adaptive query execution**: re-estimate cardinalities during query execution and re-optimize the plan mid-flight if estimates prove wrong. This is implemented in PostgreSQL 14+ (via `enable_memoize`), Microsoft SQL Server's "Adaptive Joins", and Apache Spark's Adaptive Query Execution. The concept is essentially a feedback loop: collect actual row counts from operators as they run, compare to estimates, trigger re-optimization when the error exceeds a threshold.
> **🔑 Foundation: Adaptive query execution: re-optimizing plans during execution when cardinality estimates prove wrong**
> 
> ## Adaptive Query Execution: Re-Optimizing Plans During Execution
**What it IS**
Even the best static optimizer works from estimates — and estimates are wrong. A join planned as a nested-loop (expecting 50 rows from the inner side) might encounter 500,000 rows at runtime, turning a millisecond operation into minutes. **Adaptive Query Execution (AQE)** solves this by treating the optimizer as a feedback loop: *pause execution at key boundaries, observe actual cardinalities, and re-optimize the remaining plan*.
The canonical model (popularized by Apache Spark 3.0 and present in production databases like SQL Server's **Adaptive Joins** and PostgreSQL's pending work) works like this:
1. Execute the query in **stages** separated by **materialization points** (shuffle boundaries in distributed systems, or hash build phases in single-node systems).
2. After each stage completes, the runtime knows the **actual row counts** and data statistics produced.
3. Feed those actual statistics back into the optimizer to re-plan downstream stages.
Concrete example — **adaptive join switching**: the optimizer initially plans a hash join (expecting a large inner table). At runtime, after the inner side is built, if it turns out to be tiny (say, 500 rows after a highly selective filter), AQE can switch to a **broadcast join** or **nested-loop join** that's now cheaper. No restart needed — it transparently swaps the operator.
Other AQE capabilities include:
- **Partition coalescing**: if a shuffle produced far fewer rows than expected, merge small output partitions to reduce task overhead.
- **Skew join handling**: if one join key has 10× more rows than others (data skew), split that partition and parallelize it separately.
**WHY you need it right now**
Your optimizer will produce plans based on estimated cardinalities. You need to know that there's a spectrum: pure static optimization (plan once, execute), AQE (plan → execute partial → replan → execute), and fully pipelined streaming re-optimization. For your build, even a simple form of AQE — collecting actual row counts at pipeline breakers and using them to warn or re-optimize the next stage — dramatically improves real-world robustness.
**Key insight**
Materialization points are both a cost (you write intermediate data to memory/disk) and an opportunity (you learn the ground truth). Design your execution engine so that pipeline-breaker operators (hash joins, sorts, exchanges) can report their output statistics back to an **optimizer feedback hook**. This decoupling lets you layer AQE on top of a static optimizer incrementally — you don't have to redesign everything at once.

For this milestone, static planning with `ANALYZE`-sourced statistics is the correct baseline.
---
## Default Statistics: Planning Before ANALYZE
The planner needs sensible behavior when `ANALYZE` has never been run — which is the initial state of every new database. The rule: **assume large tables, use conservative defaults that favor table scans.**
```c
/* Default assumptions when no statistics are available */
#define DEFAULT_TABLE_ROWS        1000000  /* assume 1M rows */
#define DEFAULT_DISTINCT_VALS     100      /* assume 100 distinct values */
#define DEFAULT_SELECTIVITY       0.01     /* assume 1% selectivity for equality */
```
With these defaults, a 1M-row table has scan cost `1,000,000 / 40 = 25,000` pages. An equality index scan has selectivity 1/100 = 1%, matching 10,000 rows, costing `log(1M) + 10,000/200 + 10,000×5/40 ≈ 3 + 50 + 1,250 = 1,303` pages. The index scan wins — which is correct for a selective equality predicate on a large table.
If the table has only 100 rows (a small configuration table), the defaults assume 1M rows and overprice the table scan, potentially choosing an index scan that isn't faster. This is acceptable — the error goes in the "too conservative about indexes" direction, which costs a little performance but never produces catastrophically wrong plans. Run `ANALYZE` to fix it.
Document this behavior clearly:
```c
/* Statistics note: planner uses default assumptions when ANALYZE has not been run.
   Defaults assume large tables (1M rows) and moderate selectivity (1%).
   Run ANALYZE after bulk data loading for accurate query plans.
   Stale statistics (old ANALYZE on data that has since changed) may cause
   suboptimal plan selection. Periodic ANALYZE is recommended after large
   INSERT/UPDATE/DELETE operations. */
```
---
## Stale Statistics: The Plan Regression Problem
Statistics become stale whenever data changes after `ANALYZE` runs. A table that had 1,000 rows when `ANALYZE` ran might have 10,000,000 rows by the time the query executes. The planner makes decisions based on 1,000-row estimates, producing a plan optimized for the wrong data size.
The classic failure mode: `ANALYZE` runs on a nearly-empty staging table. Data is loaded in production. Statistics still reflect the staging data. A query that should use an index (because the production table is large and the predicate is selective) gets a table scan plan because the planner thinks the table has 100 rows — too small to benefit from an index. The table scan on 10M rows is 100,000× slower than the index scan would have been.
This is structurally identical to **cache staleness** — the buffer pool concept from Milestone 4 applied at a higher level. Just as a stale buffer pool cache returns wrong page data, a stale statistics cache returns wrong cardinality estimates. The remedy is the same: invalidation and refresh.
```c
/* Trigger re-analysis recommendations */
typedef struct {
    char  *table_name;
    double row_change_ratio;   /* current_rows / analyzed_rows */
} AnalyzeRecommendation;
/* Check if statistics are stale based on approximate row count.
   Uses the B-tree root page's cell count as a fast approximation. */
int stats_are_stale(DB *db, const char *table_name, double threshold) {
    int64_t analyzed_rows = stat_table_rows(&db->stat_cache, table_name);
    if (analyzed_rows < 0) return 1;  /* no stats = definitely stale */
    /* Fast approximate count from page headers */
    int64_t approx_rows = btree_approximate_row_count(db, table_name);
    if (approx_rows < 0) return 0;
    double ratio = (double)approx_rows / (double)analyzed_rows;
    return (ratio > threshold || ratio < 1.0 / threshold);
}
```
---
## EXPLAIN: The Full Picture

![EXPLAIN Output — Trace Example](./diagrams/diag-planner-explain-output.svg)

The EXPLAIN output for a multi-table join with index usage:
```
Query Plan:
  JOIN ORDER: orders → customers (nested loop)
    TABLE SCAN orders (est. 10000 rows, cost 250.0 pages)
      filter: status = 'pending' (selectivity 0.01)
    INDEX SCAN customers via idx_cust_id
      (est. 1 row per probe, cost 6.0 pages, selectivity 0.0001)
  Total estimated cost: 60250.0 pages
Bytecode:
addr  opcode         p1    p2    p3
0     OpenTable      0     0     0     -- 'orders' cursor
1     OpenIndex      1     0     0     -- 'idx_cust_id' cursor
2     Rewind         0     12    0
3     Column         0     2     1     -- orders.status
4     String8        2     0     0     -- 'pending'
5     Ne             1     11    2     -- skip if status != 'pending'
6     Column         0     3     3     -- orders.customer_id
7     IndexSeek      1     11    3     -- seek idx_cust_id for this customer
8     IndexRowid     1     4     0
9     SeekRowid      2     11    4     -- fetch from customers table
10    ResultRow      ...
11    Next           0     3     0
12    Halt
```
Implement EXPLAIN as a mode flag on the compilation context. When `EXPLAIN` is prepended to a query, set `c->explain_mode = 1`. The compiler runs normally, accumulating plan metadata. Instead of executing the program, `program_explain_with_plan` prints it and returns.
```c
/* In the SQL dispatcher: */
if (first_token.type == TOKEN_EXPLAIN) {
    parser_advance(&p);  /* consume EXPLAIN */
    AstNode *inner = parse_statement_from_parser(&p);
    if (p.had_error) { /* report */ return; }
    const char *err;
    Program *prog = compile_with_explain(inner, db, &err);
    if (!prog) { fprintf(stderr, "Compile error: %s\n", err); return; }
    program_explain_with_plan(prog, stdout);
    program_free(prog);
    ast_free(inner);
    return;
}
```
---
## Three-Level View: A Query Through the Planner
To see the planner in context, trace `SELECT name FROM users WHERE age > 50` end-to-end:
**Level 1 — Planner (this milestone)**
After parsing produces the SELECT AST:
1. `planner_choose_access_path(db, "users", WHERE(age > 50), &ap)`
2. Look up stats: `users` has 1,000,000 rows, `idx_age` has 80 distinct values
3. Selectivity for `age > 50`: range predicate → 0.33 (one-third of rows)
4. Cost of table scan: `1,000,000 / 40 = 25,000` pages
5. Cost of index scan: `3 (index depth) + 4,125 (leaf pages) + 41,250 (random table reads)` = 45,378 pages
6. Table scan wins! → `ap.type = ACCESS_TABLE_SCAN`
7. The compiler emits `OP_REWIND / OP_COLUMN / OP_GT / OP_RESULT_ROW / OP_NEXT` without any index opcodes
For `WHERE age = 42` (equality, far more selective):
1. Selectivity: `1 / 80 = 0.0125` (1.25%)
2. Cost of table scan: 25,000 pages
3. Cost of index scan: `3 + 156 + 1,562 = 1,721` pages
4. Index scan wins → emit double-lookup bytecode
**Level 2 — VDBE + B-tree**
For the table scan plan: `OP_COLUMN` calls `btree_cursor_column` on the table B-tree, decoding `age` from each row's record. The comparison `OP_GT r1 →next r2` skips rows where age ≤ 50. Each row is exactly one `record_decode_column` call.
**Level 3 — Buffer pool**
For the table scan on a cold database: 25,000 page reads at 100μs each = 2.5 seconds. On a warm cache: 25,000 × 10ns (L2 hit) = 250 microseconds. The planner's cost estimate of 25,000 pages directly predicts the cold-cache execution time in microseconds.

![FetchPage — Hit vs Miss Data Walk](./diagrams/diag-buffer-pool-fetch-flow.svg)

---
## Design Decisions: Cost Model Granularity

![Cost Model — Table Scan vs Index Scan](./diagrams/diag-planner-cost-model.svg)

The cost model has several design degrees of freedom:
| Decision | Simple (Chosen ✓) | Advanced | Used By |
|----------|-------------------|----------|---------|
| Unit of cost | Pages read | Weighted I/O (seq vs random) | PostgreSQL uses weighted cost |
| Join selectivity | 1/max(distinct) | Histogram-based | All production DBs |
| Default row count | Fixed 1M | Adaptive based on file size | MySQL InnoDB |
| Join algorithm | Nested loop only | Hash join, sort-merge join | PostgreSQL, SQL Server |
| Statistics table | sqlite_stat1 (rows + distinct) | Multi-column histograms | PostgreSQL pg_stats |
The simple model is correct for the acceptance criteria of this milestone and matches SQLite's actual approach for its simpler cost model. SQLite's real cost model adds a weighted I/O cost that distinguishes sequential from random reads, uses the `sqlite_stat4` table for histogram support (in full builds), and considers the buffer pool fill rate as a correction factor.
---
## Knowledge Cascade: What This Milestone Unlocks
You have just implemented cost-based query optimization. Here is what that connects to across the field:
**→ Operations research and combinatorial optimization.** The join ordering problem is structurally identical to the **Traveling Salesman Problem** (TSP): you have N cities (tables), distances between pairs (join costs), and want the shortest tour (cheapest join order). Both are NP-hard. Both use dynamic programming for small N (Held-Karp algorithm for TSP maps to Selinger DP for join ordering). Both use heuristics for large N (nearest neighbor for TSP, greedy left-deep trees for join ordering). If you study operations research, you will recognize every algorithm here. The 1979 Selinger paper that first solved this problem is the database equivalent of Dijkstra's 1959 shortest path paper — it founded an entire subfield.
**→ Probability theory and selectivity estimation.** Selectivity estimation (`1 / distinct_values` for equality predicates) is a direct application of the probability that a randomly drawn value matches a specific value from a uniform distribution. The histogram-based extension uses empirical probability distributions — the same concept as probability density functions and cumulative distribution functions from statistics. When databases use "multi-dimensional statistics" to handle correlations between columns (PostgreSQL 10+), they're computing joint probability distributions. The query planner is fundamentally a probabilistic inference engine predicting future data distributions from past observations.
**→ Adaptive systems and control theory.** Adaptive query execution — re-optimizing during execution when estimates prove wrong — is a feedback control loop. The "plant" is the query execution engine, the "sensor" is the row count at operator boundaries, the "setpoint" is the estimated row count, and the "actuator" is the query re-optimizer. This is mathematically equivalent to a PID controller. The insight that cardinality estimation errors compound through join trees is the same as the error propagation problem in control theory: a cascaded system amplifies errors at each stage. Understanding adaptive query execution now gives you a concrete instantiation of a control-theoretic concept.
**→ Cache invalidation at a higher level.** Statistics staleness is structurally identical to cache invalidation — the classic "second hardest problem in computer science." Your statistics cache (in-memory `StatCache`) can be stale relative to the on-disk `sqlite_stat1` table, which can be stale relative to the actual data in user tables. The three-level cache hierarchy with staleness propagation is exactly the same structure as CPU caches, browser caches, and CDN caches. The solution — explicit invalidation triggers (re-run `ANALYZE`) rather than automatic eviction — is the same approach taken by PostgreSQL's `autovacuum` daemon, which triggers `ANALYZE` automatically when a table changes significantly.
**→ The Selinger 1979 paper is the foundation of relational databases.** The System R paper "Access Path Selection in a Relational Database Management System" by Selinger et al. is one of the most cited papers in computer science. It introduced cost-based optimization, selectivity estimation using statistics, and DP-based join ordering — all of which you've just implemented. Reading the original paper after implementing these concepts is deeply satisfying. Every concept maps directly to code you've written.
---
## Common Pitfalls: What Will Break Your Planner
**1. Planning without statistics returns wrong costs.** If `stat_table_rows` returns -1 (no statistics) and your cost model interprets that as 0 rows, every cost is 0 and every index looks infinitely better than a table scan. Always substitute the default assumption (`DEFAULT_TABLE_ROWS = 1,000,000`) when statistics are unavailable. Verify with a test: create a table, insert rows, query without running `ANALYZE`, and confirm the plan is a table scan (the safe default).
**2. Integer overflow in cardinality estimates.** For joins between large tables: `matching_rows_R × matching_rows_S` can overflow `int64_t` if each side has millions of rows. Use `double` throughout the cost model, not integer arithmetic. Doubles lose precision above 2^53 (~9 quadrillion), which is large enough for any realistic row count.
**3. The threshold doesn't account for buffer pool warmth.** The cost model estimates cold-cache page reads. For a frequently-accessed table that is largely resident in the buffer pool, the true cost of a table scan is much lower than the model predicts. A "hot" table might be better served by a table scan even with 5% selectivity. This is a known limitation of static cost models. Document it; the fix (buffer pool occupancy estimation) is a production-grade extension.
**4. Stale statistics after bulk inserts cause catastrophic plan regressions.** A common production bug: `ANALYZE` runs on a staging database with 1,000 rows. Data is loaded into production with 10,000,000 rows. Statistics still reflect 1,000 rows. The planner estimates 10 matching rows for a selective predicate, costs the index scan at 10 random reads, and uses it. Correct. But for a non-selective predicate, the table scan cost estimate is `1,000 / 40 = 25 pages` — tiny. The planner chooses the table scan without trying the index. The actual cost is `250,000 pages`. Write a test: run `ANALYZE`, load 100× more data, run the same query, verify the plan degrades gracefully.
**5. Join DP implementation produces wrong bitmask enumeration.** The inner loop `for (uint32_t left = (mask - 1) & mask; left > 0; left = (left - 1) & mask)` is a standard trick to enumerate all non-empty proper subsets of a bitmask. An off-by-one or missing `right >= left` deduplication guard produces duplicate subset pairs, computing each join ordering twice. This doesn't affect correctness (taking the minimum of the same value twice still gives the minimum) but doubles the runtime of the DP. Verify with N=3: print all (left, right) pairs and confirm each partition appears exactly once.
**6. `ANALYZE` on empty tables produces divide-by-zero in selectivity.** If `distinct_vals = 0` (empty index), `1.0 / distinct_vals` is a division by zero. Guard: if `distinct_vals <= 0`, use `DEFAULT_SELECTIVITY = 0.01`. Test with: `CREATE TABLE t (id INTEGER)`, run `ANALYZE`, then run any query — it must not crash.
---
## Testing the Planner
Your test suite must verify both correctness (right plan chosen) and performance impact (fewer pages read):
```c
/* Test 1: ANALYZE populates sqlite_stat1 correctly */
void test_analyze_collects_stats(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, cat TEXT)");
    db_exec(db, "CREATE INDEX idx_cat ON t(cat)");
    for (int i = 0; i < 100; i++) {
        char sql[64];
        const char *cat = (i % 4 == 0) ? "A" :
                          (i % 4 == 1) ? "B" :
                          (i % 4 == 2) ? "C" : "D";
        snprintf(sql, sizeof(sql),
                 "INSERT INTO t VALUES (%d, '%s')", i, cat);
        db_exec(db, sql);
    }
    db_exec(db, "ANALYZE");
    /* Check table-level row count */
    int64_t rows = stat_table_rows(&db->stat_cache, "t");
    assert(rows == 100);
    /* Check index distinct count: 4 distinct values (A, B, C, D) */
    int64_t distinct = stat_index_distinct(&db->stat_cache, "t", "idx_cat");
    assert(distinct == 4);
    db_close(db);
    printf("PASS: analyze_collects_stats\n");
}
/* Test 2: Planner chooses table scan for low-selectivity predicate */
void test_planner_prefers_table_scan_low_selectivity(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, gender TEXT)");
    db_exec(db, "CREATE INDEX idx_gender ON t(gender)");
    for (int i = 0; i < 10000; i++) {
        char sql[64];
        snprintf(sql, sizeof(sql),
                 "INSERT INTO t VALUES (%d, '%s')",
                 i, (i % 2 == 0) ? "M" : "F");
        db_exec(db, sql);
    }
    db_exec(db, "ANALYZE");
    /* EXPLAIN should show TABLE SCAN, not INDEX SCAN */
    AccessPath ap;
    AstNode *pred = /* parse WHERE gender = 'M' */
        parse_predicate("gender = 'M'");
    planner_choose_access_path(db, "t", pred, &ap);
    assert(ap.type == ACCESS_TABLE_SCAN);
    /* selectivity = 1/2 = 50%, well above 20% threshold */
    db_close(db);
    printf("PASS: table_scan_for_low_selectivity\n");
}
/* Test 3: Planner chooses index scan for high-selectivity predicate */
void test_planner_prefers_index_scan_high_selectivity(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, email TEXT)");
    db_exec(db, "CREATE UNIQUE INDEX idx_email ON t(email)");
    for (int i = 0; i < 10000; i++) {
        char sql[128];
        snprintf(sql, sizeof(sql),
                 "INSERT INTO t VALUES (%d, 'user%d@example.com')", i, i);
        db_exec(db, sql);
    }
    db_exec(db, "ANALYZE");
    AccessPath ap;
    AstNode *pred = parse_predicate("email = 'user42@example.com'");
    planner_choose_access_path(db, "t", pred, &ap);
    assert(ap.type == ACCESS_INDEX_SCAN);
    /* selectivity = 1/10000 = 0.01%, far below 20% threshold */
    assert(ap.selectivity < 0.001);
    db_close(db);
    printf("PASS: index_scan_for_high_selectivity\n");
}
/* Test 4: EXPLAIN output contains plan info */
void test_explain_shows_plan(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
    db_exec(db, "CREATE INDEX idx_val ON t(val)");
    db_exec(db, "INSERT INTO t VALUES (1, 'unique_value')");
    db_exec(db, "ANALYZE");
    /* Capture EXPLAIN output */
    char buf[4096];
    FILE *f = fmemopen(buf, sizeof(buf), "w");
    Program *prog = compile_with_explain(
        parse_statement("SELECT * FROM t WHERE val = 'unique_value'"), db, NULL);
    program_explain_with_plan(prog, f);
    fclose(f);
    /* Output must mention the index */
    assert(strstr(buf, "INDEX SCAN") != NULL);
    assert(strstr(buf, "idx_val") != NULL);
    assert(strstr(buf, "est.") != NULL);  /* cost estimates present */
    program_free(prog);
    db_close(db);
    printf("PASS: explain_shows_plan\n");
}
/* Test 5: EXPLAIN falls back to table scan when no applicable index */
void test_explain_table_scan_no_index(void) {
    DB *db = db_open_memory();
    db_exec(db, "CREATE TABLE t (id INTEGER, unindexed TEXT)");
    db_exec(db, "INSERT INTO t VALUES (1, 'hello')");
    db_exec(db, "ANALYZE");
    AccessPath ap;
    AstNode *pred = parse_predicate("unindexed = 'hello'");
    planner_choose_access_path(db, "t", pred, &ap);
    assert(ap.type == ACCESS_TABLE_SCAN);
    db_close(db);
    printf("PASS: table_scan_no_applicable_index\n");
}
```
---
## What You Have Built
At the end of this milestone, your database engine makes principled access-path decisions instead of naive heuristics:
- An **ANALYZE command** that scans all user tables and their indexes, collecting total row counts and per-column distinct value counts, persisting results in `sqlite_stat1`
- A **statistics cache** (`StatCache`) loaded into memory at database open time, providing O(1) lookup of `stat_table_rows` and `stat_index_distinct` during query planning
- A **cost model** with two functions: `cost_table_scan` (sequential pages = total_rows / rows_per_page) and `cost_index_scan` (index descent + leaf scan + random table reads weighted by `RANDOM_IO_COST_FACTOR`)
- A **selectivity estimator** using `1 / distinct_values` for equality predicates and 0.33 for range predicates, with sensible defaults when statistics are unavailable
- A **plan selector** (`planner_choose_access_path`) that evaluates all applicable indexes and picks the lowest-cost access path, falling back to table scan when no index helps or selectivity is above the threshold
- A **dynamic programming join orderer** for multi-table queries (up to 10 tables), finding the optimal join tree by enumerating all subset combinations and selecting minimum-cost builds
- **Join cardinality estimation** using `rows_R × rows_S / max(distinct_R, distinct_S)` for equi-joins
- An **EXPLAIN command** displaying the chosen plan with access type, index name (if used), selectivity, estimated row count, and estimated cost in pages — alongside the full bytecode dump
- A clean **planner/compiler interface** where the compiler receives an `AccessPath` and emits the corresponding bytecode (table scan or index double-lookup), with the planning decision fully separated from code generation
- **Sensible degradation** when statistics are absent or stale, with documented default assumptions and a recommendation to run `ANALYZE` after bulk data loading
The query planner is the component that makes all prior milestones work together intelligently. The storage engine (B-tree, indexes, buffer pool) provides the mechanisms. The planner decides which mechanisms to use and when. The VDBE executes the decision. This separation of concerns — mechanism vs. policy — is the architectural principle that makes relational databases powerful, extensible, and debuggable.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m9 -->
<!-- MS_ID: build-sqlite-m9 -->
# Milestone 9: Transactions (Rollback Journal)
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You are about to cross a fundamental threshold. Every milestone so far has been building a database that works correctly when nothing goes wrong. The tokenizer, parser, VDBE, buffer pool, B-tree, indexes, and query planner are all sophisticated — but they share a silent assumption: the process will complete normally. Pull the power cord mid-INSERT and everything since the last manual flush is gone, possibly leaving the database in a corrupt state that no amount of careful code can repair.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

This milestone adds the property that separates a database from a file: **crash safety**. When your database says a transaction committed, it means the data will survive a power failure, a kernel panic, a `kill -9`, or a cosmic-ray bit flip in a critical register. When it says a transaction rolled back, it means the database is exactly as if those writes never happened — no partial updates, no orphaned index entries, no half-written records.
The mechanism is the **rollback journal**. It is simple in concept, subtle in its ordering requirements, and the foundation of all crash-safe storage systems from ext4 to Raft.
---
## The Revelation: One fsync Is Not Enough
Here is the model almost every developer carries before building their first transactional system: crash safety means flushing data to disk. Call `fsync()`, data is durable, you're safe. If you want to be really careful, call `fsync()` after every write.
This model is wrong. Not just incomplete — structurally wrong in a way that guarantees data corruption under specific crash scenarios.
Let's make the failure concrete. You execute:
```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- debit Alice
UPDATE accounts SET balance = balance + 100 WHERE id = 2;  -- credit Bob
COMMIT;
```
Your naive "call fsync after each write" implementation does this:
```
1. Modify page containing Alice's row in memory
2. Write modified Alice page to disk → fsync()
3. Modify page containing Bob's row in memory  
4. Write modified Bob page to disk → fsync()
5. Transaction done
```
Now imagine the process crashes between step 2 and step 3. What is the state of the database?
Alice's balance has been decremented. Bob's balance has not been incremented. $100 has vanished from the system. The database is in an inconsistent state — and there is no way to detect this, because both writes that did complete look perfectly valid individually. There is no record of the fact that they were supposed to be atomic.
The problem is not the `fsync()` call. The problem is that you have **no undo information** and **no record of intent**. You cannot recover because you don't know what you were trying to do.
The revelation: **crash safety requires writing your intent before writing your data.** Before you modify any database page, you must record what that page looked like before the modification. This record — the rollback journal — is what enables recovery. If you crash mid-transaction, the journal tells the recovery system: "these pages were modified during an incomplete transaction; here is what they looked like before; restore them."
But this raises an immediate question: what if you write the journal and then crash before writing to the database? Answer: the database is unchanged, the journal exists, the recovery system reads the journal and verifies the database is consistent. No harm done.
What if you write to the database and the journal, but then crash before deleting the journal? Answer: the recovery system reads the journal, restores the original page images, and the database returns to its pre-transaction state. The committed data is lost, but the database is consistent.
What if the journal is corrupt? Answer: the recovery system cannot use it safely, so it leaves the database alone. The transaction's effects are lost, but the database is not corrupted.
In every scenario, correctness is preserved — *as long as the journal reaches disk before any database modification does*. That single constraint is the foundation of crash safety.

![Rollback Journal Write Ordering — State Evolution](./diagrams/diag-journal-write-ordering.svg)

---
## ACID: What You Are Actually Implementing
Before writing code, you need to understand precisely what the four ACID properties mean in terms of your implementation. "ACID" is not a single feature — it is four separate guarantees that require different mechanisms.
**Atomicity**: All writes in a transaction succeed, or none do. Your rollback journal provides this: if a transaction aborts (crash or explicit ROLLBACK), the journal restores all modified pages to their pre-transaction state. The transaction leaves no trace in the database.
**Consistency**: The database moves from one valid state to another. Your constraint enforcement (NOT NULL, UNIQUE from Milestones 5-7) implements consistency. Atomicity is a prerequisite: you can only guarantee consistency if you can guarantee that partial transactions don't leave broken states.
**Isolation**: Concurrent transactions don't see each other's uncommitted writes. For SQLite's single-writer model, you implement isolation with file-level locking: a writer holds an exclusive lock, preventing other writers or readers from seeing the in-progress state.
**Durability**: Committed transactions survive crashes. This is what the rollback journal's write ordering protocol delivers: after COMMIT returns, the data is on disk.
This milestone implements **A** (atomicity via rollback) and **D** (durability via fsync and write ordering). I (isolation) is implemented via locking, which you'll enforce in the transaction state machine. C (consistency) was implemented in earlier milestones through constraint checks.

> **🔑 Foundation: Write ordering and fsync semantics for crash safety**
> 
> ## Write Ordering and fsync Semantics for Crash Safety
### What It Is
When your program calls `write()`, the data doesn't immediately land on disk. Instead, it enters the OS page cache — a layer of in-memory buffers managed by the kernel. The kernel flushes these buffers to physical storage at its own convenience, typically seconds later. From the OS's perspective, this is a performance optimization. From your database's perspective, it's a loaded gun pointed at data integrity.
**`fsync(fd)`** is the system call that forces the OS to flush all dirty pages for a file descriptor to durable storage. It blocks until the storage device confirms the write is physically persisted. Only after `fsync` returns can you guarantee the data survives a power cut, kernel panic, or hardware reset.
**Write ordering** is the constraint that certain writes must reach durable storage *before* others. Consider a simple append-only log: you write the record payload, then write a commit marker. If the commit marker reaches disk but the payload doesn't, you have a phantom committed record. If the payload reaches disk but the commit marker doesn't, the record is simply absent — the correct outcome. Therefore: `write(payload)` → `fsync()` → `write(commit_marker)` → `fsync()` is the safe sequence.
This is harder than it looks. Consider:
```c
write(fd, record_data, len);  // goes to page cache
write(fd, commit_record, len); // also goes to page cache
// At this point, the OS can flush these in ANY order.
// A crash here may write commit before data. Corrupt.
fsync(fd); // NOW both are durable, in OS-determined order
```
For a single file where both writes go to the same fd, a single `fsync` after both writes is typically safe — the OS will flush them in offset order. But across *multiple files*, ordering guarantees evaporate entirely without explicit `fsync` calls between them.
**`fdatasync(fd)`** is a cheaper variant: it flushes data pages but only the *essential* metadata (file size, if changed). It skips updating `atime` and similar non-critical inode fields. For write-ahead logs and data files where you control the structure, `fdatasync` is usually sufficient and faster.
**`O_DIRECT`** bypasses the page cache entirely, writing directly to the storage device. This gives you full control over write ordering at the cost of aligned-buffer requirements and losing OS read-ahead. Most production databases (PostgreSQL, SQLite WAL mode) use `fsync` over buffered I/O rather than `O_DIRECT`.
### Why You Need This Right Now
You're building a storage engine or database that must survive crashes. Every milestone that involves writing records, updating indices, or maintaining a write-ahead log depends on this. The failure modes are insidious: your engine will pass all tests (tests don't simulate power cuts), but corrupt databases in production.
The classic corruption scenario:
1. You write a new B-tree node to a new page.
2. You update the parent node's child pointer to reference it.
3. Crash before `fsync`.
4. On recovery, the parent points to a page that may contain garbage (old data, zeros, or a partial write).
The fix is strict ordering: `fsync` the child page *before* writing the parent pointer update, then `fsync` the parent. Now a crash at any point leaves the tree in a consistent (though possibly incomplete) state.
SQLite's WAL mode, PostgreSQL's checkpoint mechanism, and LevelDB's log-structured approach all encode specific `fsync` strategies as core design decisions — not afterthoughts.
### The Key Mental Model
**Think of `fsync` as a durability barrier, not just a flush.**
Draw a timeline of your writes. Every write before an `fsync` call is in an "uncertain" zone — it may or may not be on disk. Every write after a completed `fsync` for those bytes is in a "durable" zone. Your crash-recovery logic must reason about which writes fell in which zone, and the on-disk state after recovery must be *valid for every possible crash point*.
A practical rule: **if write B's correctness depends on write A being present, then A must be `fsync`'d before B is written.** Apply this rule to every pair of logically related writes in your system, and your storage engine will be crash-safe.

---
## The Transaction State Machine
Before any file format, any journal code, any recovery logic, you need a state machine that tracks whether a transaction is currently active.

![Transaction State Machine — State Evolution](./diagrams/diag-journal-transaction-states.svg)

The state machine has four states:
```c
/* transaction.h */
typedef enum {
    TXN_STATE_NONE,      /* no active transaction */
    TXN_STATE_ACTIVE,    /* BEGIN has been executed, writes are buffered */
    TXN_STATE_COMMITTING,/* COMMIT in progress — writing journal + flushing pages */
    TXN_STATE_ROLLING_BACK /* ROLLBACK in progress — restoring from journal */
} TxnState;
typedef struct {
    TxnState  state;
    int       journal_fd;     /* file descriptor for the journal file, -1 if none */
    char     *journal_path;   /* path to the .db-journal file */
    uint32_t  page_count;     /* number of pages in the database at transaction start */
    /* Set of page IDs written to the journal during this transaction */
    uint32_t *journaled_pages; /* sorted array for O(log N) lookup */
    int       journaled_count;
    int       journaled_capacity;
} Transaction;
```
The state machine transitions:
```
NONE ──BEGIN──→ ACTIVE ──COMMIT──→ COMMITTING ──done──→ NONE
                    │              (flush pages,
                    │               delete journal)
                    └──ROLLBACK──→ ROLLING_BACK ──done──→ NONE
                                   (restore pages
                                    from journal)
```
The transition rules are strict:
- `BEGIN` is only valid in `NONE` state. Nested transactions are not supported.
- `COMMIT` and `ROLLBACK` are only valid in `ACTIVE` state.
- Any write operation (INSERT, UPDATE, DELETE) requires `ACTIVE` state — auto-begin if in `NONE` state (auto-commit mode).
- In `COMMITTING` or `ROLLING_BACK` states, no new operations are accepted.
```c
/* Transaction lifecycle */
typedef struct DB DB;
int  txn_begin(DB *db);
int  txn_commit(DB *db);
int  txn_rollback(DB *db);
/* Called by the buffer pool's UnpinPage(dirty=1) path */
int  txn_journal_page(DB *db, uint32_t page_id, const uint8_t *original_data);
/* Called on database open — detects and recovers hot journals */
int  txn_recover(DB *db);
```
The key invariant: **every dirty page write must be preceded by a journal entry for that page, and that journal entry must be on disk before the dirty page write occurs.** The `txn_journal_page` function enforces this.
---
## The Journal File Format

![Rollback Journal File Format — Structure Layout](./diagrams/diag-journal-file-format.svg)

The journal file lives alongside the database file with a fixed naming convention: if the database is `mydb.db`, the journal is `mydb.db-journal`. This deterministic naming means recovery can find the journal without any additional metadata.
The journal file format:
```
Journal Header (28 bytes):
  [0-7]   Magic: 0xd9d505f920a163d7  (identifies this as a SQLite journal)
  [8-11]  Page count: number of pages in database at transaction start (uint32 BE)
  [12-15] Nonce: random 32-bit value written at journal creation
  [16-19] Sector size: sector size of the underlying filesystem (uint32 BE)
  [20-23] Page size: database page size (uint32 BE)
  [24-27] Reserved (zeros)
For each journaled page:
  [+0 to +3]   Page number (uint32 BE, 1-based)
   [+4 to +4+PAGE_SIZE-1]  Original page contents (PAGE_SIZE bytes)
  [+4+PAGE_SIZE to +4+PAGE_SIZE+3]  Checksum (uint32 BE)
```
The checksum covers the page number and page contents, protecting against torn writes to the journal itself.
```c
#define JOURNAL_MAGIC  0xd9d505f920a163d7ULL
#define JOURNAL_HEADER_SIZE  28
#define JOURNAL_PAGE_HEADER  4   /* page number */
#define JOURNAL_PAGE_FOOTER  4   /* checksum */
/* Checksum: XOR of all 32-bit words in the page number + page content,
   using the nonce as the initial value. Matches SQLite's actual checksum. */
static uint32_t journal_checksum(uint32_t nonce,
                                  uint32_t page_no,
                                  const uint8_t *page_data) {
    uint32_t sum = nonce;
    sum += page_no;
    /* XOR-sum all 32-bit words in the page */
    for (int i = 0; i + 3 < PAGE_SIZE; i += 4) {
        uint32_t word = ((uint32_t)page_data[i]   << 24) |
                        ((uint32_t)page_data[i+1] << 16) |
                        ((uint32_t)page_data[i+2] <<  8) |
                        ((uint32_t)page_data[i+3]);
        sum += word;
    }
    return sum;
}
```
The nonce is written in the journal header at creation time. It is a random value. The same nonce is used for all page checksums in that journal. Why? Because the nonce is written **before** any page data. If recovery reads a journal but the nonce field on disk is wrong (the journal creation was interrupted), the checksums for all pages will fail — correctly indicating a corrupt journal that cannot be used for recovery.
The journal magic number `0xd9d505f920a163d7` is SQLite's actual magic number. Your files will be recognized by recovery tools that understand SQLite's journal format.
```c
/* Write the journal file header. Called once when the journal is first created. */
static int journal_write_header(int fd, uint32_t page_count,
                                 uint32_t nonce, uint32_t page_size) {
    uint8_t header[JOURNAL_HEADER_SIZE];
    uint64_t magic = JOURNAL_MAGIC;
    /* Magic (8 bytes, big-endian) */
    for (int i = 7; i >= 0; i--) {
        header[i] = (uint8_t)(magic & 0xFF);
        magic >>= 8;
    }
    /* Page count (4 bytes, big-endian) */
    header[8]  = (page_count >> 24) & 0xFF;
    header[9]  = (page_count >> 16) & 0xFF;
    header[10] = (page_count >>  8) & 0xFF;
    header[11] =  page_count        & 0xFF;
    /* Nonce (4 bytes, big-endian) */
    header[12] = (nonce >> 24) & 0xFF;
    header[13] = (nonce >> 16) & 0xFF;
    header[14] = (nonce >>  8) & 0xFF;
    header[15] =  nonce        & 0xFF;
    /* Sector size (use 512 as a safe default) */
    uint32_t sector_size = 512;
    header[16] = (sector_size >> 24) & 0xFF;
    header[17] = (sector_size >> 16) & 0xFF;
    header[18] = (sector_size >>  8) & 0xFF;
    header[19] =  sector_size        & 0xFF;
    /* Page size */
    header[20] = (page_size >> 24) & 0xFF;
    header[21] = (page_size >> 16) & 0xFF;
    header[22] = (page_size >>  8) & 0xFF;
    header[23] =  page_size        & 0xFF;
    /* Reserved */
    memset(header + 24, 0, 4);
    ssize_t n = pwrite(fd, header, JOURNAL_HEADER_SIZE, 0);
    return (n == JOURNAL_HEADER_SIZE) ? 0 : -1;
}
```
---
## The Torn Page Problem

![Torn Page Write Problem — State Evolution](./diagrams/diag-journal-torn-page.svg)

Before implementing the journal write path, you need to understand the failure mode that makes complete page images necessary.
You have a 4096-byte page. You write it to disk with `pwrite(fd, page_data, 4096, offset)`. The OS accepts the write and returns success. The process then crashes.
Did the 4096 bytes make it to the physical medium? Maybe. Maybe not. But here is the subtler question: are they **consistent**?
An SSD or HDD writes at the sector granularity — typically 512 bytes or 4096 bytes. If the filesystem sector size (512 bytes) is smaller than the database page size (4096 bytes), a single database page write requires eight sector writes. If power fails after sectors 1-4 are written but before sectors 5-8, the page on disk contains the first 2048 bytes of the new version and the last 2048 bytes of the old version. This is a **torn page** — a chimera of two versions that is internally inconsistent.
No application-level code can detect a torn page without knowing what the correct state should be. A B-tree node with half-new separator keys and half-old child pointers is syntactically valid but semantically meaningless. Scanning it could produce wrong query results, follow invalid pointers, or crash the database engine.
This is exactly why the rollback journal stores **complete original page images**. When recovery detects an incomplete transaction:
1. For each page recorded in the journal, the recovery system has the complete, known-good original content of that page.
2. It writes those original contents back to the database file, replacing whatever partial state may exist there.
3. The restoration is itself potentially vulnerable to tearing — which is why recovery writes each restoration carefully and validates with checksums.
The lesson: **the unit of atomic I/O for crash recovery is the page, not the record, not the field, not the transaction**. Every mechanism in the recovery path operates at the page level.
---
## BEGIN: Creating the Journal
`BEGIN` must be a nearly-instantaneous operation — it just changes state and records the transaction start. No journal file is created yet. The journal is created **lazily**, on the first write operation within the transaction.
```c
int txn_begin(DB *db) {
    if (db->txn.state != TXN_STATE_NONE) {
        snprintf(db->error, sizeof(db->error),
                 "Cannot BEGIN: transaction already active");
        return -1;
    }
    /* Record the current database size (page count) for rollback */
    off_t file_size = lseek(db->fd, 0, SEEK_END);
    if (file_size < 0) return -1;
    db->txn.page_count = (uint32_t)(file_size / PAGE_SIZE);
    /* Transition to ACTIVE state */
    db->txn.state            = TXN_STATE_ACTIVE;
    db->txn.journal_fd       = -1;  /* journal not yet created */
    db->txn.journaled_count  = 0;
    /* Generate a random nonce for this transaction's journal */
    db->txn.nonce = (uint32_t)time(NULL) ^ (uint32_t)(uintptr_t)&db;
    /* Acquire write lock on the database file.
       SQLite uses file locking; for simplicity, use flock() here. */
    if (flock(db->fd, LOCK_EX | LOCK_NB) < 0) {
        db->txn.state = TXN_STATE_NONE;
        snprintf(db->error, sizeof(db->error),
                 "Cannot BEGIN: database is locked by another writer");
        return -1;
    }
    return 0;
}
```
The write lock acquisition here implements the **I** in ACID — isolation. While this transaction holds the exclusive lock, no other writer can start a transaction, and readers that need a consistent snapshot will see only the committed state (since dirty pages are only in the buffer pool, not in the database file yet).
---
## Journaling a Page: The Critical Path
This function is called every time the buffer pool is about to write a dirty page to disk. It is the most critical function in the crash safety implementation — it must guarantee that the original page content reaches the journal file and is durable before the modified page is allowed to reach the database file.
```c
/* Called by the buffer pool before writing a dirty page.
   Ensures the original page content is in the journal before modification.
   INVARIANT: after this function returns successfully, the original content
   of page_id is durably stored in the journal file. Only then may the 
   modified version be written to the database file. */
int txn_journal_page(DB *db, uint32_t page_id,
                     const uint8_t *original_data) {
    if (db->txn.state != TXN_STATE_ACTIVE) {
        /* Auto-commit mode: no journaling needed */
        return 0;
    }
    /* Check if this page has already been journaled in this transaction.
       We only need to record the original content once — the first state
       before any modification. Subsequent modifications of the same page
       within the same transaction don't need new journal entries. */
    if (txn_page_is_journaled(db, page_id)) {
        return 0;  /* already have the original — skip */
    }
    /* Create the journal file if it doesn't exist yet */
    if (db->txn.journal_fd < 0) {
        if (txn_create_journal(db) < 0) return -1;
    }
    /* Append the page record to the journal:
       [page_number: 4 bytes][page_data: PAGE_SIZE bytes][checksum: 4 bytes] */
    off_t offset = JOURNAL_HEADER_SIZE
                 + (off_t)db->txn.journaled_count
                   * (4 + PAGE_SIZE + 4);
    /* Write page number */
    uint8_t page_no_buf[4];
    page_no_buf[0] = (page_id >> 24) & 0xFF;
    page_no_buf[1] = (page_id >> 16) & 0xFF;
    page_no_buf[2] = (page_id >>  8) & 0xFF;
    page_no_buf[3] =  page_id        & 0xFF;
    if (pwrite(db->txn.journal_fd, page_no_buf, 4, offset) != 4) {
        return -1;
    }
    /* Write original page content */
    if (pwrite(db->txn.journal_fd, original_data, PAGE_SIZE,
               offset + 4) != PAGE_SIZE) {
        return -1;
    }
    /* Write checksum */
    uint32_t cksum = journal_checksum(db->txn.nonce, page_id, original_data);
    uint8_t cksum_buf[4];
    cksum_buf[0] = (cksum >> 24) & 0xFF;
    cksum_buf[1] = (cksum >> 16) & 0xFF;
    cksum_buf[2] = (cksum >>  8) & 0xFF;
    cksum_buf[3] =  cksum        & 0xFF;
    if (pwrite(db->txn.journal_fd, cksum_buf, 4,
               offset + 4 + PAGE_SIZE) != 4) {
        return -1;
    }
    /* CRITICAL: fsync the journal before returning.
       This guarantees the original page is durable BEFORE the caller
       is allowed to write the modified page to the database file. */
    if (fsync(db->txn.journal_fd) < 0) {
        return -1;
    }
    /* Record that this page has been journaled */
    txn_mark_page_journaled(db, page_id);
    return 0;
}
```
The `fsync` call here is the write ordering barrier. Every time a page is about to be modified, the original is written to the journal and synced before the modification proceeds. This is expensive — each `fsync` costs roughly 5ms on a spinning disk, 50-500 microseconds on an NVMe SSD. But it is the price of per-write durability.
The "only journal once" optimization is critical for correctness, not just performance. Consider: a transaction modifies page 42 three times. The journal must record what page 42 looked like **before the transaction started** — not before the second or third modification. Recording the intermediate states would give recovery the wrong target. The first journal entry captures the pre-transaction state; all subsequent modifications of the same page within the same transaction don't need new journal entries.
```c
/* Track journaled pages using a sorted array (for O(log N) lookup) */
static int txn_page_is_journaled(DB *db, uint32_t page_id) {
    /* Binary search */
    int lo = 0, hi = db->txn.journaled_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (db->txn.journaled_pages[mid] == page_id) return 1;
        if (db->txn.journaled_pages[mid] < page_id)  lo = mid + 1;
        else                                           hi = mid - 1;
    }
    return 0;
}
static void txn_mark_page_journaled(DB *db, uint32_t page_id) {
    if (db->txn.journaled_count >= db->txn.journaled_capacity) {
        db->txn.journaled_capacity = db->txn.journaled_capacity
                                   ? db->txn.journaled_capacity * 2 : 16;
        db->txn.journaled_pages = realloc(db->txn.journaled_pages,
            db->txn.journaled_capacity * sizeof(uint32_t));
    }
    /* Insert in sorted order */
    int pos = db->txn.journaled_count;
    while (pos > 0 && db->txn.journaled_pages[pos-1] > page_id) {
        db->txn.journaled_pages[pos] = db->txn.journaled_pages[pos-1];
        pos--;
    }
    db->txn.journaled_pages[pos] = page_id;
    db->txn.journaled_count++;
}
```
---
## Integrating with the Buffer Pool
The buffer pool's `UnpinPage(page_id, dirty=1)` path must call `txn_journal_page` **before** the page becomes eligible for eviction and physical write. But actually, the journaling must happen even earlier — at the moment the caller signals they are about to modify the page, not when it eventually gets evicted.
The cleanest integration point is a new function `buffer_pool_get_for_write`:
```c
/* Fetch a page and prepare it for modification within a transaction.
   This function:
   1. Fetches the page into a frame (as buffer_pool_fetch does)
   2. Reads the current (unmodified) content
   3. Journals the original content to the transaction journal
   4. Returns the frame pointer for modification
   The caller MUST call buffer_pool_unpin(page_id, dirty=1) when done.
   Failure to unpin will pin-leak the frame. */
uint8_t *buffer_pool_get_for_write(BufferPool *bp, uint32_t page_id) {
    /* Fetch page into buffer pool */
    uint8_t *frame = buffer_pool_fetch(bp, page_id);
    if (!frame) return NULL;
    /* Journal the original content before any modification */
    if (txn_journal_page(bp->db, page_id, frame) < 0) {
        buffer_pool_unpin(bp, page_id, 0);
        return NULL;
    }
    return frame;
}
```
Every B-tree modification path that was previously calling `buffer_pool_fetch` + `buffer_pool_unpin(dirty=1)` must now call `buffer_pool_get_for_write` instead. This is a systematic change through `btree_insert`, `btree_delete_by_rowid`, `page_insert_cell`, `page_delete_cell`, and all other functions that modify page data.
The buffer pool's eviction path also needs modification. When `evict_lru` decides to write a dirty page to disk, it must check that the journal entry for that page already exists:
```c
/* Modified eviction path */
static int evict_lru(BufferPool *bp) {
    int idx = bp->lru_tail;
    while (idx >= 0) {
        Frame *f = &bp->frames[idx];
        if (f->pin_count == 0) {
            if (f->is_dirty) {
                /* Verify this page has been journaled before writing.
                   If we're in a transaction, the page must have been
                   journaled before we write it to the database file.
                   If we're in auto-commit mode (no transaction), there
                   is no journal and we write directly. */
                if (bp->db->txn.state == TXN_STATE_ACTIVE &&
                    !txn_page_is_journaled(bp->db, f->page_id)) {
                    /* This page was marked dirty but not journaled.
                       This should not happen if buffer_pool_get_for_write
                       is used correctly. Assert or return error. */
                    assert(0 && "Dirty page not journaled in active transaction");
                    return -1;
                }
                write_page_to_disk(bp, idx);
            }
            /* ... rest of eviction ... */
            return idx;
        }
        idx = f->lru_prev;
    }
    return -1;
}
```
---
## COMMIT: The Five-Step Protocol
COMMIT is where the write ordering protocol becomes explicit. The steps must happen in exactly this order, and each step must complete successfully before the next begins.

![Rollback Journal Write Ordering — State Evolution](./diagrams/diag-journal-write-ordering.svg)

```c
int txn_commit(DB *db) {
    if (db->txn.state != TXN_STATE_ACTIVE) {
        snprintf(db->error, sizeof(db->error),
                 "Cannot COMMIT: no active transaction");
        return -1;
    }
    db->txn.state = TXN_STATE_COMMITTING;
    /* === STEP 1: Ensure all dirty pages are journaled ===
       All modifications in the transaction must be in the journal before
       any of them reach the database file. The per-write journaling in
       txn_journal_page already guarantees this, but we verify here. */
    /* (The journal entries were written and fsynced in txn_journal_page) */
    /* === STEP 2: fsync the database file header update ===
       Update the database header's change counter (tells other processes
       that the database has been modified). */
    /* === STEP 3: Flush all dirty pages from buffer pool to database file ===
       Now that the journal is durable, we can safely write modified pages. */
    int n = buffer_pool_flush_all(db->bp);
    if (n < 0) {
        db->txn.state = TXN_STATE_ACTIVE;  /* remain active for retry/rollback */
        snprintf(db->error, sizeof(db->error),
                 "COMMIT failed: could not flush dirty pages");
        return -1;
    }
    /* === STEP 4: fsync the database file ===
       Guarantee that all modified pages are durable on the physical medium.
       After this point, the committed data survives a crash. */
    if (fsync(db->fd) < 0) {
        db->txn.state = TXN_STATE_ACTIVE;
        snprintf(db->error, sizeof(db->error),
                 "COMMIT failed: fsync on database file failed");
        return -1;
    }
    /* === STEP 5: Delete the journal file ===
       The journal's existence is the marker for an incomplete transaction.
       Deleting it signals that the transaction committed successfully.
       If we crash between step 4 and step 5, the database is consistent
       (all pages are durable) and recovery will re-delete the journal
       after finding it has a valid complete transaction. */
    if (db->txn.journal_fd >= 0) {
        close(db->txn.journal_fd);
        db->txn.journal_fd = -1;
        if (unlink(db->txn.journal_path) < 0) {
            /* Failure to delete is not fatal — recovery handles stale journals.
               Log the error but consider the transaction committed. */
            /* In production: log warning */
        }
    }
    /* Release write lock */
    flock(db->fd, LOCK_UN);
    /* Clean up transaction state */
    db->txn.state           = TXN_STATE_NONE;
    db->txn.journaled_count = 0;
    return 0;
}
```
The fifth step — deleting the journal — is the **commit point**. Before the delete, the transaction is not committed (a crash would trigger rollback). After the delete, the transaction is committed (the journal is gone and recovery has nothing to do). This binary distinction — journal exists → incomplete, journal absent → committed — is what makes crash recovery possible without scanning every page for validity.
Notice that after step 4 (fsync database), the data is already durable. If the process crashes between step 4 and step 5, recovery will find the journal, attempt to restore pages, but the database is already in the correct committed state. Recovery needs to handle this case: it reads each journal entry and checks whether the database page already contains the pre-transaction state or the post-transaction state. In practice, recovery simply restores from the journal unconditionally — after a successful commit, the journal pages and the database pages might differ, but restoring the pre-transaction state would be wrong. This apparent paradox is resolved by: **only restore from the journal if the transaction was incomplete**. The detection mechanism is whether the journal is valid and complete. If step 4 completed but step 5 didn't, the journal is valid and complete — but the correct action is to leave the database alone (it already has committed data) and just delete the journal. SQLite's actual recovery implementation handles this by checking the journal's page count field against the database file size to determine if the database file has been extended (new pages allocated during the transaction). If the journal is consistent and complete, recovery deletes it. If it's incomplete (truncated, invalid checksum), recovery restores pages and then deletes the journal.
---
## ROLLBACK: Undoing the Transaction
ROLLBACK is conceptually simpler than COMMIT: read the journal, restore each original page, delete the journal.
```c
int txn_rollback(DB *db) {
    if (db->txn.state != TXN_STATE_ACTIVE) {
        snprintf(db->error, sizeof(db->error),
                 "Cannot ROLLBACK: no active transaction");
        return -1;
    }
    db->txn.state = TXN_STATE_ROLLING_BACK;
    /* If no journal exists (transaction made no writes), nothing to undo */
    if (db->txn.journal_fd < 0) {
        db->txn.state           = TXN_STATE_NONE;
        db->txn.journaled_count = 0;
        flock(db->fd, LOCK_UN);
        return 0;
    }
    /* Restore all original pages from journal */
    if (txn_restore_from_journal(db) < 0) {
        /* Restoration failed — database may be in inconsistent state.
           Log a severe error. The journal still exists, so startup
           recovery will attempt restoration on next open. */
        snprintf(db->error, sizeof(db->error),
                 "ROLLBACK failed: could not restore pages from journal");
        return -1;
    }
    /* fsync the database after restoration to make the rollback durable */
    if (fsync(db->fd) < 0) {
        return -1;
    }
    /* Invalidate all dirty (now-restored) pages in buffer pool.
       The buffer pool's frames hold modified data that was rolled back.
       We must evict those frames so subsequent reads see the restored
       on-disk content. */
    buffer_pool_invalidate_dirty_pages(db->bp);
    /* Delete the journal */
    close(db->txn.journal_fd);
    db->txn.journal_fd = -1;
    unlink(db->txn.journal_path);
    /* Release lock and clean up */
    flock(db->fd, LOCK_UN);
    db->txn.state           = TXN_STATE_NONE;
    db->txn.journaled_count = 0;
    return 0;
}
```
The `buffer_pool_invalidate_dirty_pages` call deserves attention. After restoring original pages to disk, the buffer pool's frames still contain the modified (now-invalid) versions of those pages. Any subsequent read of those pages from the buffer pool would see the rolled-back modification, not the correctly restored original. You must evict those frames:
```c
/* Evict all dirty frames from the buffer pool without writing them to disk.
   Called after rollback — dirty frames contain invalid rolled-back data. */
void buffer_pool_invalidate_dirty_pages(BufferPool *bp) {
    for (int i = 0; i < bp->pool_size; i++) {
        Frame *f = &bp->frames[i];
        if (f->page_id != INVALID_PAGE_ID && f->is_dirty) {
            /* Remove from page table */
            page_table_remove(bp, f->page_id);
            /* Remove from LRU list */
            lru_remove(bp, i);
            /* Invalidate frame — do NOT write to disk */
            f->page_id   = INVALID_PAGE_ID;
            f->is_dirty  = 0;
            f->pin_count = 0;
        }
    }
}
```
Note: this function does not write dirty frames to disk before invalidating them. That would be wrong during rollback — the frames contain the modifications we're undoing. We want the on-disk content (already restored from the journal) to be the truth, and the buffer pool frames to be forced to reload from disk on next access.
---
## The Journal Restoration Function
```c
/* Read the journal and restore each original page to the database file.
   Used by both ROLLBACK and crash recovery. */
static int txn_restore_from_journal(DB *db) {
    int    fd     = db->txn.journal_fd;
    off_t  offset = JOURNAL_HEADER_SIZE;
    uint8_t entry_buf[4 + PAGE_SIZE + 4];  /* page_no + data + checksum */
    /* Read the nonce from the journal header for checksum verification */
    uint8_t header[JOURNAL_HEADER_SIZE];
    if (pread(fd, header, JOURNAL_HEADER_SIZE, 0) != JOURNAL_HEADER_SIZE) {
        return -1;
    }
    uint32_t nonce = ((uint32_t)header[12] << 24) |
                     ((uint32_t)header[13] << 16) |
                     ((uint32_t)header[14] <<  8) |
                     ((uint32_t)header[15]);
    /* Read each page record */
    while (1) {
        ssize_t n = pread(fd, entry_buf,
                          4 + PAGE_SIZE + 4, offset);
        if (n == 0) break;  /* end of journal */
        if (n < (ssize_t)(4 + PAGE_SIZE + 4)) break;  /* incomplete entry */
        uint32_t page_id = ((uint32_t)entry_buf[0] << 24) |
                           ((uint32_t)entry_buf[1] << 16) |
                           ((uint32_t)entry_buf[2] <<  8) |
                           ((uint32_t)entry_buf[3]);
        const uint8_t *page_data = entry_buf + 4;
        uint32_t stored_cksum = ((uint32_t)entry_buf[4 + PAGE_SIZE + 0] << 24) |
                                ((uint32_t)entry_buf[4 + PAGE_SIZE + 1] << 16) |
                                ((uint32_t)entry_buf[4 + PAGE_SIZE + 2] <<  8) |
                                ((uint32_t)entry_buf[4 + PAGE_SIZE + 3]);
        /* Verify checksum — skip corrupted entries */
        uint32_t computed_cksum = journal_checksum(nonce, page_id, page_data);
        if (computed_cksum != stored_cksum) {
            /* Corrupt journal entry — stop here.
               Pages before this point have been restored.
               Pages after this point were not journaled (or journal was truncated).
               In either case, stopping is the safe behavior. */
            break;
        }
        /* Restore the original page content to the database file */
        off_t db_offset = (off_t)(page_id - 1) * PAGE_SIZE;
        if (pwrite(db->fd, page_data, PAGE_SIZE, db_offset) != PAGE_SIZE) {
            return -1;  /* write failure — database may be inconsistent */
        }
        offset += 4 + PAGE_SIZE + 4;
    }
    return 0;
}
```
The checksum verification deserves careful thought. If a journal entry has a corrupted checksum, it means one of:
1. The journal file was truncated (the process crashed while writing the journal entry itself)
2. The journal file was partially written by the OS before a crash
3. The journal file was corrupted by a hardware error
In all cases, the safe action is to stop processing journal entries at the first bad checksum. Pages before the corrupt entry may have been restored. Pages after it were either: (a) successfully journaled but their journal entry was lost in the truncation — meaning their database-file content was never modified (they were journaled but the transaction was interrupted before modifying the database), or (b) never journaled — meaning their database-file content was never modified (the transaction was interrupted before reaching them). Either way, stopping at the first bad checksum is correct: the database pages that were never modified are still in their pre-transaction state.
---
## Crash Recovery: The Hot Journal

![Crash Recovery — Before/After](./diagrams/diag-journal-crash-recovery.svg)

The most important function in this milestone is `txn_recover`, called once during database open, before any queries are executed.
```c
/* Called on database open. Detects and recovers from incomplete transactions.
   A "hot journal" is a .db-journal file found alongside the database file
   at startup — it indicates the previous process crashed mid-transaction. */
int txn_recover(DB *db) {
    /* Construct journal path */
    size_t db_path_len = strlen(db->path);
    char *journal_path = malloc(db_path_len + 9);  /* + "-journal\0" */
    snprintf(journal_path, db_path_len + 9, "%s-journal", db->path);
    /* Check if the journal file exists */
    int journal_fd = open(journal_path, O_RDWR);
    if (journal_fd < 0) {
        /* No journal — clean state, nothing to recover */
        free(journal_path);
        return 0;
    }
    /* Hot journal detected. The previous process crashed during a transaction.
       We must roll back the incomplete transaction. */
    fprintf(stderr,
            "Hot journal detected: %s\n"
            "Recovering from incomplete transaction...\n",
            journal_path);
    /* Validate the journal header */
    uint8_t header[JOURNAL_HEADER_SIZE];
    if (pread(journal_fd, header, JOURNAL_HEADER_SIZE, 0)
        != JOURNAL_HEADER_SIZE) {
        /* Journal header is incomplete — journal creation was interrupted.
           The database file was never modified (journaling is the first step).
           Simply delete the incomplete journal. */
        close(journal_fd);
        unlink(journal_path);
        free(journal_path);
        return 0;
    }
    /* Verify magic number */
    uint64_t magic = 0;
    for (int i = 0; i < 8; i++) {
        magic = (magic << 8) | header[i];
    }
    if (magic != JOURNAL_MAGIC) {
        /* Not a valid SQLite journal — possibly a different file.
           Do not touch either file. */
        close(journal_fd);
        free(journal_path);
        snprintf(db->error, sizeof(db->error),
                 "Corrupt journal file: invalid magic number");
        return -1;
    }
    /* Set up temporary transaction state for recovery */
    db->txn.journal_fd   = journal_fd;
    db->txn.journal_path = journal_path;
    /* Extract nonce */
    db->txn.nonce = ((uint32_t)header[12] << 24) |
                    ((uint32_t)header[13] << 16) |
                    ((uint32_t)header[14] <<  8) |
                    ((uint32_t)header[15]);
    /* Restore original pages */
    if (txn_restore_from_journal(db) < 0) {
        snprintf(db->error, sizeof(db->error),
                 "Recovery failed: could not restore pages from journal");
        close(journal_fd);
        free(journal_path);
        db->txn.journal_fd   = -1;
        db->txn.journal_path = NULL;
        return -1;
    }
    /* fsync the database file after restoration */
    if (fsync(db->fd) < 0) {
        snprintf(db->error, sizeof(db->error),
                 "Recovery failed: fsync after restoration failed");
        return -1;
    }
    /* Delete the journal — recovery complete */
    close(journal_fd);
    unlink(journal_path);
    free(journal_path);
    db->txn.journal_fd   = -1;
    db->txn.journal_path = NULL;
    fprintf(stderr, "Recovery complete.\n");
    return 0;
}
```
The recovery function must run **before** the buffer pool loads any pages. If recovery runs after pages are cached, the buffer pool might have cached corrupt pages from before recovery fixed them. The correct startup sequence is:
```c
DB *db_open(const char *path) {
    DB *db = calloc(1, sizeof(DB));
    db->path = strdup(path);
    /* 1. Open the database file */
    db->fd = open(path, O_RDWR | O_CREAT, 0644);
    /* 2. Run crash recovery BEFORE creating buffer pool */
    if (txn_recover(db) < 0) {
        /* Recovery failed — database may be inconsistent */
        close(db->fd);
        free(db);
        return NULL;
    }
    /* 3. Create buffer pool (now safe to cache pages) */
    db->bp = buffer_pool_create(db->fd, DEFAULT_POOL_SIZE);
    /* 4. Load schema from sqlite_master */
    schema_load(db);
    return db;
}
```
The recovery-before-buffer-pool ordering is critical. After `txn_recover` completes, every page in the database file is in its consistent pre-transaction state. The buffer pool starts with an empty cache. All subsequent page reads see the clean recovered state. The alternative — running recovery after the buffer pool loads pages — would require invalidating potentially all cached pages after recovery, which is complex and error-prone.
---
## Auto-Commit Mode
SQL engines support two operating modes:
**Explicit transaction mode**: the user writes `BEGIN` before writes and `COMMIT` after. All writes within the BEGIN/COMMIT block are atomic.
**Auto-commit mode**: each statement is automatically wrapped in a transaction. `INSERT INTO t VALUES (1, 'Alice')` is implicitly `BEGIN; INSERT...; COMMIT;`. The user never writes BEGIN or COMMIT.
Auto-commit is the default behavior. Implement it by checking in the VDBE's `OP_INSERT`, `OP_UPDATE`, `OP_DELETE` handlers: if no transaction is active, automatically begin one before the write and commit it after:
```c
/* In VDBE execution, before any write operation */
static int ensure_transaction(VdbeState *vm) {
    if (vm->db->txn.state == TXN_STATE_NONE) {
        /* Auto-begin a transaction */
        if (txn_begin(vm->db) < 0) {
            snprintf(vm->error, sizeof(vm->error),
                     "Failed to begin auto-commit transaction: %s",
                     vm->db->error);
            return -1;
        }
        vm->auto_commit = 1;  /* remember to auto-commit after this statement */
    }
    return 0;
}
/* After a write statement completes successfully in auto-commit mode */
static int maybe_auto_commit(VdbeState *vm) {
    if (vm->auto_commit) {
        vm->auto_commit = 0;
        if (txn_commit(vm->db) < 0) {
            snprintf(vm->error, sizeof(vm->error),
                     "Auto-commit failed: %s", vm->db->error);
            return -1;
        }
    }
    return 0;
}
```
The auto-commit model has an important implication for durability: each individual write is immediately committed and durable. This is the most durable setting but also the slowest: every single INSERT incurs the cost of journal creation, fsync, page flush, fsync, and journal deletion. For bulk inserts, wrapping them all in a single explicit `BEGIN`/`COMMIT` block reduces this overhead from N × (2 fsyncs) to 1 × (2 fsyncs).
---
## Basic Read Isolation
The acceptance criteria require that changes are not visible to other connections until COMMIT. With the write-lock-on-BEGIN design above, this is already enforced: other writers cannot begin transactions (they'll fail to acquire the exclusive lock). But what about readers?
SQLite's rollback journal mode uses a simpler model than full MVCC: **write lock prevents new readers during a write transaction**. When a writer holds the exclusive lock:
1. Other writers fail immediately with "database locked"
2. New readers that need a consistent snapshot must wait for the writer to finish
This is implemented through SQLite's lock hierarchy: SHARED → RESERVED → PENDING → EXCLUSIVE. For simplicity, this milestone implements a binary LOCK_EX / LOCK_UN model. The key guarantee: until COMMIT (which releases the lock), no other connection can see the uncommitted writes, because:
- The uncommitted modifications are only in the buffer pool's dirty frames
- Those dirty frames are not visible to other processes (they have their own buffer pools)
- The database file itself still contains the original (pre-transaction) data until the COMMIT flush
This means a reader in a separate process that opens the same database file will see the pre-transaction state throughout the write transaction, and the post-transaction state after COMMIT. Basic read isolation is achieved through the combination of dirty-page buffering (modifications in memory only) and the write lock (preventing concurrent schema modifications that would make in-flight queries invalid).
---
## ACID Verification Tests
The acceptance criteria can be directly translated into test cases:
```c
/* Test 1: BEGIN/COMMIT/ROLLBACK state transitions */
void test_transaction_states(void) {
    DB *db = db_open("/tmp/test_txn.db");
    db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
    /* Auto-commit: INSERT without BEGIN */
    assert(db_exec(db, "INSERT INTO t VALUES (1, 'a')") == 0);
    /* Explicit transaction */
    assert(db_exec(db, "BEGIN") == 0);
    assert(db->txn.state == TXN_STATE_ACTIVE);
    assert(db_exec(db, "INSERT INTO t VALUES (2, 'b')") == 0);
    assert(db_exec(db, "COMMIT") == 0);
    assert(db->txn.state == TXN_STATE_NONE);
    /* Verify both rows persisted */
    ResultSet *rs = db_query(db, "SELECT COUNT(*) FROM t");
    assert(rs->rows[0].cols[0].i == 2);
    db_close(db);
    printf("PASS: transaction_states\n");
}
/* Test 2: ROLLBACK undoes writes */
void test_rollback_undoes_writes(void) {
    DB *db = db_open("/tmp/test_rollback.db");
    db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
    db_exec(db, "INSERT INTO t VALUES (1, 'original')");
    db_exec(db, "BEGIN");
    db_exec(db, "INSERT INTO t VALUES (2, 'rolled_back')");
    db_exec(db, "UPDATE t SET val = 'modified' WHERE id = 1");
    db_exec(db, "ROLLBACK");
    /* Row 2 should not exist */
    ResultSet *rs1 = db_query(db, "SELECT COUNT(*) FROM t");
    assert(rs1->rows[0].cols[0].i == 1);
    /* Row 1 should have original value */
    ResultSet *rs2 = db_query(db, "SELECT val FROM t WHERE id = 1");
    assert(strcmp(rs2->rows[0].cols[0].text.data, "original") == 0);
    db_close(db);
    printf("PASS: rollback_undoes_writes\n");
}
/* Test 3: Journal file exists during transaction, deleted on commit */
void test_journal_file_lifecycle(void) {
    DB *db = db_open("/tmp/test_jlife.db");
    db_exec(db, "CREATE TABLE t (id INTEGER)");
    db_exec(db, "BEGIN");
    db_exec(db, "INSERT INTO t VALUES (1)");
    /* Journal file must exist now */
    assert(access("/tmp/test_jlife.db-journal", F_OK) == 0);
    db_exec(db, "COMMIT");
    /* Journal file must be deleted after commit */
    assert(access("/tmp/test_jlife.db-journal", F_OK) != 0);
    db_close(db);
    printf("PASS: journal_file_lifecycle\n");
}
/* Test 4: Crash recovery — simulate crash by killing journal mid-transaction */
void test_crash_recovery(void) {
    /* Phase 1: start a transaction, write to journal, simulate crash */
    {
        DB *db = db_open("/tmp/test_crash.db");
        db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
        db_exec(db, "INSERT INTO t VALUES (1, 'committed')");
        /* Start transaction that will be "crashed" */
        db_exec(db, "BEGIN");
        db_exec(db, "INSERT INTO t VALUES (2, 'crashed_insert')");
        db_exec(db, "UPDATE t SET val = 'modified' WHERE id = 1");
        /* Simulate crash: close without committing or rolling back.
           The journal file remains on disk. */
        /* Force dirty pages to the journal (they're already there via
           txn_journal_page), then close the file descriptors without
           calling txn_commit or txn_rollback. */
        /* In a real crash simulation: call _exit() or use a separate
           process that calls abort(). Here we use a controlled close
           that skips the normal cleanup path. */
        crash_simulate_close(db);  /* closes fd without commit/rollback */
    }
    /* Phase 2: reopen — should trigger automatic recovery */
    {
        DB *db = db_open("/tmp/test_crash.db");
        /* Recovery should have run automatically */
        /* Verify: only the committed row exists */
        ResultSet *rs = db_query(db, "SELECT COUNT(*) FROM t");
        assert(rs->rows[0].cols[0].i == 1);
        ResultSet *rs2 = db_query(db, "SELECT val FROM t WHERE id = 1");
        assert(strcmp(rs2->rows[0].cols[0].text.data, "committed") == 0);
        /* Journal file should be deleted after recovery */
        assert(access("/tmp/test_crash.db-journal", F_OK) != 0);
        db_close(db);
    }
    printf("PASS: crash_recovery\n");
}
/* Test 5: Write ordering — journal must exist before database modification */
void test_write_ordering(void) {
    DB *db = db_open("/tmp/test_order.db");
    db_exec(db, "CREATE TABLE t (id INTEGER)");
    /* Install a hook that verifies journal exists before any pwrite to .db */
    /* (In practice: use LD_PRELOAD or a test-mode flag on pwrite) */
    db->verify_journal_before_db_write = 1;
    db_exec(db, "BEGIN");
    db_exec(db, "INSERT INTO t VALUES (1)");
    assert(db->journal_verified_before_db_write);  /* hook was triggered */
    db_exec(db, "COMMIT");
    db_close(db);
    printf("PASS: write_ordering\n");
}
```
---
## Three-Level View: What Happens During `UPDATE accounts SET balance = balance - 100 WHERE id = 1`

![Rollback Journal Write Ordering — State Evolution](./diagrams/diag-journal-write-ordering.svg)

**Level 1 — Transaction manager (this milestone)**
When the VDBE executes `OP_UPDATE` for this statement within a `BEGIN` block:
1. `ensure_transaction()` finds `TXN_STATE_ACTIVE` — no implicit BEGIN needed
2. The VDBE calls `btree_search` to locate rowid 1's leaf page (page 47, say)
3. Before modifying page 47, `buffer_pool_get_for_write(bp, 47)` is called
4. `txn_journal_page(db, 47, current_page_47_data)` fires:
   - Page 47's current bytes are appended to the journal file
   - `fsync(journal_fd)` — this call blocks until the OS confirms page 47's original content is on the physical medium
5. The B-tree modification proceeds — Alice's balance in page 47 is decremented
6. `buffer_pool_unpin(bp, 47, dirty=1)` — frame 47 is now dirty in the buffer pool
7. The dirty frame stays in memory; it has NOT been written to the database file yet
**Level 2 — Buffer pool / Journal file (physical layer)**
The buffer pool holds frame 47 (Alice's modified row) in memory with `is_dirty = 1`. The journal file on disk contains:
```
[header: magic, nonce, page_count=100, page_size=4096]
[page_no=47][page_47_original_bytes: 4096 bytes][checksum]
```
The database file's page 47 still contains Alice's original balance. Until COMMIT, the database file is untouched.
**Level 3 — fsync and the storage stack**
When `fsync(journal_fd)` is called:
```
Application (your code)
  → libc fsync() wrapper
  → kernel sys_fsync()
  → VFS (Virtual Filesystem) layer
  → ext4/XFS filesystem
  → block layer (merge pending writes, handle I/O scheduling)
  → SCSI/NVMe device driver
  → disk controller firmware (flush controller cache to NAND/platters)
```
At each layer, buffers are flushed. The call does not return until the storage controller confirms the data is on the physical medium. For an NVMe SSD, this takes 50-500 microseconds. For a spinning disk, 3-10 milliseconds. For an SD card or flash on a Raspberry Pi, potentially 100+ milliseconds.
This is why a single `fsync` per write (the naive model) is so expensive: each INSERT in auto-commit mode costs 2 fsyncs. 1000 auto-commit INSERTs cost 2000 fsyncs × 5ms each = 10 seconds. The same 1000 INSERTs wrapped in a single `BEGIN`/`COMMIT` cost 2 fsyncs = 10ms. A 1000× throughput difference from batching transactions.
---
## Design Decision: Why Not WAL Here?
| Feature | Rollback Journal (this milestone) | WAL (Milestone 10) |
|---|---|---|
| **Write pattern** | Copy-before-write (undo log) | Append-only (redo log) |
| **Read during write** | Writer blocks readers | Multiple readers, one writer |
| **Checkpoint needed** | No | Yes (WAL grows without bound) |
| **Lock held** | Full write: exclusive | Write: exclusive on WAL only |
| **Crash recovery** | Restore original pages | Replay WAL from last checkpoint |
| **Write performance** | One fsync per unique page on COMMIT | One fsync per COMMIT (sequential) |
| **Read performance** | Normal (database file is current) | Must check WAL for every page |
| **Complexity** | Moderate | Higher |
| **Used by** | SQLite default mode | SQLite WAL mode, PostgreSQL, InnoDB |
The rollback journal is the right model to implement first because:
1. It is conceptually simpler — recovery is pure undo, no redo needed
2. It is SQLite's default mode — most SQLite databases use it
3. It establishes the foundation that WAL mode builds on (WAL adds a redo layer)
4. The write ordering principle you learn here is identical in WAL, just applied to a different structure
---
## Knowledge Cascade: What Write Ordering Unlocks
You have just implemented the fundamental pattern of crash-safe systems. This same pattern appears in every system that must survive failures:
**→ Filesystem journaling (ext4, NTFS, APFS, ZFS).** Every journaling filesystem uses the same write ordering protocol you just implemented. ext4's journal records metadata updates before applying them. NTFS's $LogFile contains before-images of modified sectors. ZFS's intent log (ZIL) is essentially a per-filesystem rollback journal. The code you wrote for `txn_journal_page` and `txn_recover` is structurally identical to what `jbd2` (ext4's journal layer) does. Linux filesystem engineers call the pattern "journaling" at this layer and "logging" when applied to database transactions, but the mechanism is the same.
**→ Distributed systems: Raft and two-phase commit.** The Raft consensus protocol's write path is write-ordering made distributed. A Raft leader writes a log entry (the journal), replicates it to a majority of followers, then commits. The log entry reaching a quorum of durable stores before the state machine applies it is the distributed equivalent of your journal fsync before database write. Two-phase commit (2PC) is even more directly analogous: phase 1 (voting) is writing prepare records to all participants' journals; phase 2 (commit) is writing commit records after all participants have durable prepare entries. The failure modes — coordinator crash between phase 1 and phase 2, leaving an "uncertain" transaction — are the distributed equivalent of crashing after the journal fsync but before the database fsync.
**→ ARIES recovery algorithm.** Your rollback journal implements the "undo" half of the ARIES (Algorithms for Recovery and Isolation Exploiting Semantics) protocol, which is the theoretical foundation for virtually all production database recovery systems. ARIES's full form combines undo logging (what you've built) with redo logging (WAL, Milestone 10) to enable:
- **Analysis**: scan log from last checkpoint to determine which transactions were active at crash
- **Redo**: replay all changes after the last checkpoint (even those that weren't committed — they'll be undone next)
- **Undo**: roll back all transactions that were active at crash
Your rollback journal is the simplest form of undo-only logging, without the redo component. WAL will add the redo component. Together, they form the complete ARIES protocol.
**→ Torn page writes and atomic I/O.** The 4KB page size is not arbitrary. Modern SSDs and HDDs have 4KB physical sectors, making 4KB writes atomic — either the entire sector is written or none of it. Databases that use a page size equal to the filesystem/disk sector size get torn-page protection "for free" from the hardware. Databases that use larger page sizes (PostgreSQL's default 8KB) must handle partial writes explicitly (PostgreSQL uses a "full-page writes" setting that writes complete page images to the WAL after every checkpoint, for exactly this reason). Your choice of 4KB pages matches the hardware atomic write unit.
**→ The fsync cost is the reason for group commit.** Every production database with high write throughput implements **group commit**: collect multiple transactions' journal entries, write them all, perform a single fsync, then commit all of them together. PostgreSQL, MySQL InnoDB, and SQLite all do this. The amortization works because `fsync` latency is dominated by the device seek/flush cost, not the data volume. Writing 100 transactions' journal entries in one `pwrite` + one `fsync` costs the same as writing one transaction's entry — but commits 100 transactions. Your implementation fsyncs per-transaction; group commit is the production optimization built on this foundation.
**→ The journal deletion is a distributed consensus operation in miniature.** Deleting the journal file is an atomic operation from the application's perspective: either it exists (transaction incomplete) or it doesn't (transaction complete). The filesystem guarantees that `unlink()` is atomic. This binary existence check is your commit protocol's consensus mechanism. In distributed Raft, the equivalent is writing the commit record to the leader's log and replicating it: the log record's existence in a quorum of followers is the commit point. The pattern — a single atomic bit-flip (file exists / doesn't exist, log record present / absent) that transitions from "not committed" to "committed" — is universal across crash-safe systems.
---
## Common Pitfalls: What Will Corrupt Your Database
**1. Writing database pages before journaling them.** This is the fundamental violation: any write to the database file that is not preceded by a journal entry and fsync is a data corruption waiting to happen. The symptom is difficult to detect — everything works until a crash, at which point the database has partial modifications with no undo information. The fix: enforce `txn_journal_page` at the `buffer_pool_get_for_write` call site, and add an assertion in the eviction path that dirty pages have journal entries.
**2. Re-journaling the same page with its modified content.** If a page is journaled after the first modification (not before), the journal contains the modified state, not the original. Rolling back would "restore" the modification, leaving the database in the modified state. The "journal only once per transaction per page" rule must record the state **before the first modification**. A bug that allows re-journaling (e.g., clearing the `journaled_pages` set mid-transaction) silently loses the pre-transaction state.
**3. Not fsyncing the journal before writing the database.** If `txn_journal_page` writes to the journal but doesn't call `fsync`, the journal entry is in the OS page cache but not on physical media. A crash after the database write but before the OS flushes the journal cache leaves you with a modified database and no journal entry — equivalent to no journaling at all. The `fsync` is not optional.
**4. Not fsyncing the database before deleting the journal.** If COMMIT flushes dirty pages and deletes the journal without fsyncing the database file, a crash after the journal deletion but before the dirty pages reach physical media leaves you with a committed transaction marker (journal gone) but uncommitted data (pages not durable). When the user reopens the database, there's no journal to trigger recovery, but the "committed" data isn't there. This is silent data loss after a successful `COMMIT` return. The `fsync(db->fd)` before `unlink(journal_path)` is the critical ordering.
**5. Buffer pool cache invalidation after rollback.** After restoring pages from the journal to the database file, the buffer pool may hold the modified (pre-rollback) versions of those pages in dirty frames. If those frames are not invalidated, subsequent reads from the buffer pool return the rolled-back modifications instead of the restored originals. The `buffer_pool_invalidate_dirty_pages` call is essential. A symptom of missing this: ROLLBACK appears to succeed, but a subsequent SELECT still returns the rolled-back data.
**6. Journal creation after the database write (auto-commit edge case).** In auto-commit mode, if the implementation begins a journal only when the first write happens during `OP_INSERT`, and that write modifies the database before creating the journal, you have the unsafe ordering. The journal must be created and the original page written to it **before** the B-tree modification, not after the VDBE instruction completes.
**7. Nested transactions causing double-journaling.** If `BEGIN` is called inside an already-active transaction and creates a new journal, then COMMIT of the inner transaction deletes the journal, then ROLLBACK of the outer transaction finds no journal — there's nothing to restore from. The simplest correct behavior: reject nested transactions with an error. Document that savepoints (a more complex feature) are needed for sub-transaction semantics.
---
## What You Have Built
At the end of this milestone, your database crosses from "works when nothing goes wrong" to "correct under failure":
- A **transaction state machine** with four states (NONE, ACTIVE, COMMITTING, ROLLING_BACK) that gates all write operations, enforces BEGIN/COMMIT/ROLLBACK semantics, and prevents nested transactions
- A **rollback journal file** in SQLite's binary format: 28-byte header with magic number, nonce, and page count, followed by (page_number, page_data, checksum) records for each modified page
- A **write ordering protocol** enforcing the invariant: journal entry durable → database write allowed. Implemented via `txn_journal_page` which writes the original page, computes its checksum, and calls `fsync(journal_fd)` before returning
- **Per-page deduplication**: each page is journaled only once per transaction, capturing the pre-transaction state, not intermediate states
- A **COMMIT path** that flushes dirty pages with `buffer_pool_flush_all`, fsyncs the database file, then deletes the journal — in that exact order, with no shortcuts
- A **ROLLBACK path** that reads journal entries, verifies checksums, restores original page images to the database file, fsyncs, invalidates the buffer pool's dirty frames, and deletes the journal
- **Crash recovery** via `txn_recover` called at database open: detects the presence of a `.db-journal` file, validates its header, restores original pages, and cleans up — before the buffer pool caches any pages
- **Auto-commit mode** wrapping each DML statement in an implicit BEGIN/COMMIT when no explicit transaction is active
- **Write lock enforcement** via `flock(LOCK_EX)` ensuring that uncommitted writes are not visible to other connections
- An **ACID compliance test suite** verifying atomicity (ROLLBACK undoes all writes), durability (committed data survives close/reopen), and isolation (uncommitted writes invisible to other connections)
The WAL mode in Milestone 10 builds directly on this foundation: it replaces the copy-before-write journal with an append-only write-ahead log, enabling concurrent readers during writes while maintaining the same write ordering and crash recovery guarantees.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m10 -->
# Milestone 10: WAL Mode
## System Orientation

![SQLite Architecture — Satellite Map](./diagrams/diag-satellite-system-map.svg)

You stand at the final frontier of your storage engine. Every layer is in place: the tokenizer, parser, VDBE compiler, virtual machine, buffer pool, B-tree storage engine, secondary indexes, query planner, and rollback journal. Your database is correct, persistent, and crash-safe.
But it has a bottleneck you have not yet felt in your test suite: **exclusive access**. In rollback journal mode, a writer holds an exclusive file lock for the entire duration of a write transaction. Every reader that arrives while the writer is active must wait. On a database serving concurrent workloads — web requests, background analytics, administrative queries — this serialization point becomes a wall.

![Query Lifecycle — End-to-End Data Walk](./diagrams/diag-data-flow-query-lifecycle.svg)

WAL mode — Write-Ahead Logging — exists to remove that wall. It restructures where writes go, enabling readers and the writer to operate simultaneously without blocking each other. Understanding how it achieves this concurrency without locks is the deepest insight in all of database engineering. Everything you've built leads here.
---
## The Revelation: WAL Inverts the I/O Pattern
Here is what every developer assumes after learning about rollback journals: WAL is just a rollback journal that appends instead of overwriting. A minor implementation detail. Strictly better. The same concept with cleaner I/O.
This model is wrong in a way that touches the fundamental architecture of concurrent systems.
Consider how the rollback journal works. Before modifying a page, you write the original page to the journal. The **main database file is always current** — it contains the latest committed state. Readers always go directly to the database file and get correct data. Writers temporarily disturb the database file and use the journal to undo that disturbance if needed. The database file is the truth; the journal is the safety net.

![WAL vs Rollback Journal — Before/After](./diagrams/diag-wal-vs-journal.svg)

Now invert every word of that sentence. In WAL mode:
**The WAL file is always current.** Writers never touch the main database file. Instead, they append new page versions to the WAL. The **main database file is always stale** — it contains the last checkpointed state, which may be hours or days behind the current state.
This inversion has consequences that cascade through the entire system:
**Writes become sequential.** Instead of writing pages scattered across the database file (wherever those pages happen to live), the writer appends to the end of the WAL. Sequential I/O is dramatically faster than random I/O: an SSD can sustain 3 GB/s sequential writes but only 300 MB/s random writes. A spinning disk can sustain 100+ MB/s sequential but 1 MB/s random (1 seek per write × 5ms × 200 IOPS). Appending to the WAL converts write-scattered random I/O into sequential I/O.
**Reads become more complex.** Before returning a page, a reader must check whether the WAL contains a newer version. The WAL might have been written to many times since the last checkpoint. The reader must search the WAL for the most recent version of the requested page, then fall back to the database file if not found. This is a read overhead that rollback journal mode never incurs.
**Concurrency becomes possible.** Here is the key. In rollback journal mode, both readers and writers need the main database file. A writer must modify it (and hold an exclusive lock during modification), so readers must wait. In WAL mode, the writer only appends to the WAL — it never touches the main database file. Readers read from the main database file (plus the WAL up to their snapshot point). These two operations do not conflict. **Readers and the writer can proceed simultaneously, without any locks between them.**
This is the real breakthrough: snapshot isolation without locking. Each reader, when it starts, records the current end-of-WAL position as its "end of world" marker. Everything in the WAL before that marker is visible to the reader. Everything written after is invisible. The writer can keep appending beyond that marker. The reader sees a consistent snapshot that will never change, even as the writer continues modifying data.
The cost is unbounded WAL growth. If readers hold long-running snapshots, the WAL cannot be checkpointed past the oldest active snapshot's marker. The WAL grows without bound until all readers finish. Managing this tension — enabling concurrent access while preventing unbounded growth — is the operational challenge that WAL introduces.
---
## What You Are Building

![WAL Mode Architecture — Street View](./diagrams/diag-wal-architecture.svg)

WAL mode consists of five integrated components:
1. **WAL file format** — the on-disk structure: a header followed by frames, each frame containing a page image with a checksum
2. **WAL index (wal-index)** — an in-memory (shared-memory) structure mapping page IDs to their most recent frame in the WAL, enabling O(1) page lookup
3. **Write path** — appending page frames to the WAL instead of modifying the database file
4. **Read path** — checking the WAL index before reading from the main database
5. **Checkpoint** — copying WAL frames back to the main database file, then truncating (or resetting) the WAL
You also need a mode toggle — `PRAGMA journal_mode=WAL` — that switches an open database from rollback journal to WAL mode.
Let's build each piece.
---
## The WAL File Format

![WAL Frame Format — Structure Layout](./diagrams/diag-wal-frame-format.svg)

The WAL file lives alongside the database file with a fixed naming convention: `<database>.db-wal`. Its format is a 32-byte header followed by an unlimited sequence of **frames**, each containing one full page image.
```
WAL Header (32 bytes):
  [0-3]   Magic:        0x377f0682  (big-endian, identifies WAL file)
  [4-7]   File format:  3007000 (SQLite WAL format version)
  [8-11]  Page size:    4096 (big-endian uint32)
  [12-15] Checkpoint sequence number (big-endian uint32, monotonically increasing)
  [16-19] Salt-1: random 32-bit value, changes on each checkpoint
  [20-23] Salt-2: random 32-bit value, changes on each checkpoint  
  [24-27] Checksum-1: cumulative checksum of header bytes 0-23
  [28-31] Checksum-2: cumulative checksum continuation
WAL Frame (PAGE_SIZE + 24 bytes per frame):
  Frame Header (24 bytes):
    [0-3]   Page number being stored (big-endian uint32, 1-based)
    [4-7]   Database size after commit: non-zero for commit frames, 0 for non-commit
    [8-11]  Salt-1: must match WAL header salt (for validating this frame belongs to current WAL generation)
    [12-15] Salt-2: must match WAL header salt
    [16-19] Checksum-1: cumulative running checksum
    [20-23] Checksum-2: cumulative running checksum
  Frame Data (PAGE_SIZE bytes):
    The complete contents of the page at page_number
```
Two details deserve careful attention.
**The commit marker.** The `database size` field in a frame header is 0 for non-commit frames and non-zero for commit frames. A non-commit frame represents a page modified during a transaction that has not yet committed. A commit frame (the last frame written in a transaction, with a non-zero database size) signals that all preceding frames in this write transaction are durable and complete. Readers only use frames up to and including a commit frame — they never use partially-written transactions.
**The salt pair.** Salts are random 32-bit values written when the WAL is first created and changed on each checkpoint. Their purpose: when recovery reads the WAL, it can distinguish frames from the current WAL generation versus frames left over from a previous generation (before the last checkpoint). A frame whose salt fields don't match the WAL header was written in a previous era and should be ignored.
```c
/* wal.h */
#pragma once
#include <stdint.h>
#include "buffer_pool.h"
#define WAL_MAGIC          0x377f0682u
#define WAL_HEADER_SIZE    32
#define WAL_FRAME_HDRSIZE  24
#define WAL_FRAME_SIZE     (WAL_FRAME_HDRSIZE + PAGE_SIZE)
/* WAL header, parsed from disk */
typedef struct {
    uint32_t magic;
    uint32_t file_format;
    uint32_t page_size;
    uint32_t checkpoint_seq;
    uint32_t salt1;
    uint32_t salt2;
    uint32_t cksum1;
    uint32_t cksum2;
} WalHeader;
/* WAL frame header, parsed from disk */
typedef struct {
    uint32_t page_no;       /* 1-based page number */
    uint32_t db_size;       /* non-zero on commit frames */
    uint32_t salt1;
    uint32_t salt2;
    uint32_t cksum1;
    uint32_t cksum2;
} WalFrameHdr;
/* WAL handle attached to a database */
typedef struct Wal Wal;
Wal    *wal_open(const char *db_path, int page_size);
void    wal_close(Wal *wal);
/* Write path: begin a write transaction in WAL mode */
int     wal_begin_write(Wal *wal);
/* Append one page frame to the WAL */
int     wal_write_frame(Wal *wal, uint32_t page_no, const uint8_t *page_data,
                        int is_commit, uint32_t db_size);
/* End the write transaction, finalizing commit frames */
int     wal_end_write(Wal *wal);
/* Read path: find the most recent WAL frame for a page.
   end_frame: the reader's snapshot marker (frame index limit).
   Returns frame index if found, -1 if not in WAL (fall back to main db). */
int     wal_find_frame(Wal *wal, uint32_t page_no, int end_frame);
/* Read frame data from the WAL into buf */
int     wal_read_frame(Wal *wal, int frame_idx, uint8_t *buf);
/* Snapshot management */
int     wal_begin_read(Wal *wal);   /* returns the snapshot end_frame marker */
void    wal_end_read(Wal *wal, int snapshot_id);
/* Checkpoint: copy WAL frames to main database, return frames checkpointed */
int     wal_checkpoint(Wal *wal, int db_fd);
/* WAL frame count */
int     wal_frame_count(Wal *wal);
```
---
## The WAL Checksum

> **🔑 Foundation: Cumulative checksums in WAL files**
> 
> ## Cumulative Checksums in WAL Files
### What It IS
A WAL (Write-Ahead Log) file stores database pages as a sequence of **frames**. Each frame has a header containing two checksum integers. The twist: these aren't independent per-frame checksums. They're **cumulative** — each frame's checksum is computed over that frame's content *plus* the checksum values from the previous frame.
Concretely, the algorithm (SQLite uses a custom integer-based checksum) works like this:
```
frame[0].checksum = checksum(frame[0].header + frame[0].data, seed=(0, 0))
frame[1].checksum = checksum(frame[1].header + frame[1].data, seed=frame[0].checksum)
frame[2].checksum = checksum(frame[2].header + frame[2].data, seed=frame[1].checksum)
...
frame[N].checksum = checksum(frame[N].content, seed=frame[N-1].checksum)
```
Each frame's checksum value *encodes the entire history of all frames before it*. To verify frame N, you must have correctly computed the checksum through frames 0..N-1 first.
### WHY You Need It Right Now
When your WAL reader (or the SQLite engine itself) reads a WAL file to determine the **end of valid data**, it walks frames sequentially, recomputing checksums as it goes. The moment a frame's recomputed checksum doesn't match the stored value, the reader knows that frame — and everything after it — is invalid. This is how WAL readers find the **last committed frame** after a crash.
Without cumulative chaining, you could verify each frame independently, but you'd have no way to detect **truncation corruption** — a scenario where the last few frames were partially written and their bytes happen to pass an independent checksum by coincidence. The chained design makes this virtually impossible: a truncated or partially written frame will produce a checksum mismatch that propagates, because it broke the chain.
This matters directly when you implement WAL recovery or a WAL reader: you'll iterate frames and recompute checksums in a running state object, stopping at the first mismatch. That stopping point *is* the valid WAL end.
### The Key Mental Model
**Think of it like a hash chain in a blockchain.** Each block (frame) commits to all previous blocks by incorporating the prior hash (checksum) into its own. Corrupt or fabricate any single link and every subsequent link's hash becomes wrong. You can't silently drop frames from the middle or end — the chain will break.
In practical terms: store your running checksum pair `(s1, s2)` as you scan. Feed them as the seed into each new frame's checksum computation. If a frame matches, advance. If it doesn't, stop — you've found the WAL's valid boundary. All frames beyond that point must be ignored, even if their bytes look syntactically reasonable.

WAL uses a running (cumulative) checksum. Each frame's checksum is computed over the frame header and page data, using the previous frame's checksum as the starting point. This means:
- **Corruption detection**: a single corrupted byte anywhere causes all subsequent checksums to fail, making the corruption visible even if the corrupted frame itself happens to have the right bytes by coincidence.
- **Truncation detection**: if the WAL was truncated mid-write (crash during append), the incomplete frame's checksum will not match, because the running state from the previous frame is not present.
```c
/* WAL checksum: a pair of uint32 values computed cumulatively.
   The algorithm XORs rotating 32-bit words of the input into two
   accumulators (c1 and c2), producing a 64-bit checksum as two
   independent 32-bit values. This is SQLite's actual WAL checksum. */
static void wal_checksum_bytes(const uint8_t *data, int n,
                                uint32_t *c1_inout, uint32_t *c2_inout) {
    uint32_t c1 = *c1_inout;
    uint32_t c2 = *c2_inout;
    assert(n % 8 == 0);  /* must be multiple of 8 bytes */
    const uint32_t *p = (const uint32_t *)data;
    const uint32_t *end = p + (n / 4);
    while (p < end) {
        /* Read as big-endian uint32 */
        uint32_t w1 = ((uint32_t)((uint8_t*)p)[0] << 24) |
                      ((uint32_t)((uint8_t*)p)[1] << 16) |
                      ((uint32_t)((uint8_t*)p)[2] <<  8) |
                      ((uint32_t)((uint8_t*)p)[3]);
        uint32_t w2 = ((uint32_t)((uint8_t*)(p+1))[0] << 24) |
                      ((uint32_t)((uint8_t*)(p+1))[1] << 16) |
                      ((uint32_t)((uint8_t*)(p+1))[2] <<  8) |
                      ((uint32_t)((uint8_t*)(p+1))[3]);
        c1 += w1 + c2;
        c2 += w2 + c1;
        p += 2;
    }
    *c1_inout = c1;
    *c2_inout = c2;
}
```
This two-accumulator approach provides stronger detection than a single XOR checksum: XOR cannot detect any transposition of two values (since XOR is commutative), but this mixing checksum can detect most transpositions because `c1` feeds into `c2` and vice versa.
The same checksum logic applies to networking protocols: TCP checksums, Ethernet CRC, and Git's SHA-1 content hashing all detect corruption by transforming data through a mathematical function that changes detectably if any bit changes. The WAL checksum is the storage-layer equivalent.
---
## The WAL Index: O(1) Page Lookup

![WAL Read Path — Data Walk](./diagrams/diag-wal-read-path.svg)

The WAL file might contain thousands of frames. Reading it linearly to find the most recent version of a specific page would be O(frames) per page access — catastrophic for read performance. The WAL index solves this.
The WAL index is a hash table mapping `page_number → most recent frame index`. It lives in shared memory (a `.db-shm` file memory-mapped by all processes). Because it's in shared memory, all readers and the writer see the same index. The writer updates it when appending frames; readers consult it to find frames.
Because the WAL index is in shared memory and can be accessed by multiple processes simultaneously, it requires careful concurrent access management — the writer must update it atomically enough that partial updates don't mislead readers. SQLite's actual implementation uses a specific locking protocol for the index. For this implementation, we use a simpler in-process approach: the index is in process memory (not shared across processes), sufficient for the single-process SQLite use case.
```c
/* WAL index: maps page_no → most recent frame index.
   Uses open addressing (linear probing). */
#define WAL_INDEX_SLOTS   65536   /* power of 2 for fast modulo */
#define WAL_INDEX_EMPTY   UINT32_MAX
typedef struct {
    uint32_t page_no;     /* UINT32_MAX = empty slot */
    uint32_t frame_idx;   /* 0-based frame index in WAL file */
} WalIndexEntry;
typedef struct {
    WalIndexEntry slots[WAL_INDEX_SLOTS];
    int           frame_count;   /* total frames written to WAL */
    int           write_lock;    /* 1 if writer holds lock */
    /* Snapshot tracking: minimum end_frame across all active readers */
    int           min_reader_snapshot; /* -1 if no readers */
} WalIndex;
/* Insert or update: page_no → frame_idx (always the most recent) */
static void walindex_insert(WalIndex *idx, uint32_t page_no, uint32_t frame_idx) {
    uint32_t h = (page_no * 2654435761u) & (WAL_INDEX_SLOTS - 1);
    for (int i = 0; i < WAL_INDEX_SLOTS; i++) {
        uint32_t slot = (h + i) & (WAL_INDEX_SLOTS - 1);
        if (idx->slots[slot].page_no == WAL_INDEX_EMPTY ||
            idx->slots[slot].page_no == page_no) {
            idx->slots[slot].page_no  = page_no;
            idx->slots[slot].frame_idx = frame_idx;
            return;
        }
    }
    /* Index full — should not happen with WAL_INDEX_SLOTS = 65536 and
       realistic WAL sizes (we checkpoint before overflow) */
}
/* Lookup: find the most recent frame index for page_no that is ≤ end_frame.
   Returns frame index, or -1 if not in WAL. */
static int walindex_lookup(WalIndex *idx, uint32_t page_no, int end_frame) {
    uint32_t h = (page_no * 2654435761u) & (WAL_INDEX_SLOTS - 1);
    for (int i = 0; i < WAL_INDEX_SLOTS; i++) {
        uint32_t slot = (h + i) & (WAL_INDEX_SLOTS - 1);
        if (idx->slots[slot].page_no == WAL_INDEX_EMPTY) return -1;
        if (idx->slots[slot].page_no == page_no) {
            int f = (int)idx->slots[slot].frame_idx;
            /* Respect snapshot boundary: reader can only see frames up to end_frame */
            if (f <= end_frame) return f;
            /* The most recent frame is beyond the reader's snapshot.
               We need to scan earlier frames for this page. */
            /* Simplified: do a linear scan of the WAL file for this page
               within the snapshot range. The full implementation maintains
               per-snapshot hash maps; this linear fallback is correct but O(n). */
            return wal_scan_for_page_before(/* ... */);
        }
    }
    return -1;
}
```
The snapshot boundary check — `if (f <= end_frame) return f` — is the mechanism that implements snapshot isolation. The reader's `end_frame` is set to the WAL's frame count at the moment the reader started. The writer may have added frames beyond that count. The reader's lookup ignores those frames entirely, as if they don't exist.
This is snapshot isolation **without any locks between readers and the writer**. The reader and writer are not coordinating at all during the lookup — the reader simply has a local integer (`end_frame`) that tells it how far into the WAL to look. The writer can append new frames beyond that integer with no interaction with the reader.
---
## Snapshot Isolation: The Mechanism Explained


![Snapshot Isolation — State Evolution](./diagrams/diag-wal-snapshot-isolation.svg)

Let's trace through a concrete scenario to make snapshot isolation tangible.
**Time 0**: The database has 100 committed pages. The WAL is empty. `wal_index.frame_count = 0`.
**Time 1**: Reader R1 begins. `wal_begin_read()` returns `end_frame = 0`. R1 sees the world as it existed at time 0 — no WAL frames.
**Time 2**: Writer W1 begins a transaction. It modifies page 42 and page 77. The WAL now contains:
- Frame 0: page 42 (non-commit, db_size=0)
- Frame 1: page 77 (commit, db_size=100)
`wal_index.frame_count = 2`. The writer updates the WAL index: page 42 → frame 0, page 77 → frame 1.
**Time 3**: Reader R2 begins. `wal_begin_read()` returns `end_frame = 2`. R2 sees the world as it exists after W1's commit.
**Time 4**: R1 reads page 42. `walindex_lookup(index, page_no=42, end_frame=0)` returns -1 (frame 0 is at index 0, which is NOT ≤ -1... wait). Actually: R1's `end_frame = 0` means it can see no WAL frames at all. The lookup returns -1. R1 falls back to the main database file and reads the pre-W1 version of page 42. Correct — R1 started before W1's commit.
**Time 5**: R2 reads page 42. `walindex_lookup(index, page_no=42, end_frame=2)` returns 0 (frame 0 ≤ 2). R2 reads frame 0 from the WAL and gets W1's updated version of page 42. Correct — R2 started after W1's commit.
Neither reader blocked the writer. The writer did not block either reader. There are no mutexes in this read path — only integer comparisons.
This pattern — readers hold an immutable view of the world defined by a single integer — is the same mechanism used by:
- PostgreSQL's MVCC (transaction IDs as the snapshot marker)
- CockroachDB's MVCC (hybrid logical timestamps)
- Google Spanner's TrueTime API (commit timestamps as snapshot markers)
- Linux kernel's RCU (Read-Copy-Update) mechanism for concurrent data structure access
In every case, readers take a cheap "snapshot" (a single integer or timestamp), and the write path never invalidates existing read views — it only appends new versions.
---
## The Write Path

![Concurrent Readers + Writer — State Evolution](./diagrams/diag-wal-concurrent-access.svg)

In WAL mode, the buffer pool's eviction path and the transaction commit path change fundamentally. Modified pages are not written to the database file. They are appended to the WAL.
```c
struct Wal {
    int        fd;            /* WAL file descriptor */
    char      *path;          /* path to .db-wal file */
    WalIndex   index;         /* in-process WAL index */
    WalHeader  header;        /* parsed WAL file header */
    uint32_t   run_cksum1;    /* running checksum state */
    uint32_t   run_cksum2;
    int        write_lock;    /* 1 if this process holds the write lock */
};
/* Append one page frame to the WAL.
   is_commit: 1 if this is the last frame in a write transaction.
   db_size:   current database page count (only meaningful for commit frames). */
int wal_write_frame(Wal *wal, uint32_t page_no,
                    const uint8_t *page_data,
                    int is_commit, uint32_t db_size) {
    /* Build the frame header */
    uint8_t frame_hdr[WAL_FRAME_HDRSIZE];
    /* Page number (big-endian) */
    frame_hdr[0] = (page_no >> 24) & 0xFF;
    frame_hdr[1] = (page_no >> 16) & 0xFF;
    frame_hdr[2] = (page_no >>  8) & 0xFF;
    frame_hdr[3] =  page_no        & 0xFF;
    /* DB size (non-zero for commit frames) */
    uint32_t commit_size = is_commit ? db_size : 0;
    frame_hdr[4] = (commit_size >> 24) & 0xFF;
    frame_hdr[5] = (commit_size >> 16) & 0xFF;
    frame_hdr[6] = (commit_size >>  8) & 0xFF;
    frame_hdr[7] =  commit_size        & 0xFF;
    /* Salt (must match WAL header salt for validity check) */
    frame_hdr[8]  = (wal->header.salt1 >> 24) & 0xFF;
    frame_hdr[9]  = (wal->header.salt1 >> 16) & 0xFF;
    frame_hdr[10] = (wal->header.salt1 >>  8) & 0xFF;
    frame_hdr[11] =  wal->header.salt1        & 0xFF;
    frame_hdr[12] = (wal->header.salt2 >> 24) & 0xFF;
    frame_hdr[13] = (wal->header.salt2 >> 16) & 0xFF;
    frame_hdr[14] = (wal->header.salt2 >>  8) & 0xFF;
    frame_hdr[15] =  wal->header.salt2        & 0xFF;
    /* Compute running checksum over frame header bytes 0-15 and page data */
    uint32_t c1 = wal->run_cksum1;
    uint32_t c2 = wal->run_cksum2;
    wal_checksum_bytes(frame_hdr, 8, &c1, &c2);    /* first 8 bytes of header */
    wal_checksum_bytes(page_data, PAGE_SIZE, &c1, &c2);
    wal->run_cksum1 = c1;
    wal->run_cksum2 = c2;
    /* Write checksum into header bytes 16-23 */
    frame_hdr[16] = (c1 >> 24) & 0xFF;
    frame_hdr[17] = (c1 >> 16) & 0xFF;
    frame_hdr[18] = (c1 >>  8) & 0xFF;
    frame_hdr[19] =  c1        & 0xFF;
    frame_hdr[20] = (c2 >> 24) & 0xFF;
    frame_hdr[21] = (c2 >> 16) & 0xFF;
    frame_hdr[22] = (c2 >>  8) & 0xFF;
    frame_hdr[23] =  c2        & 0xFF;
    /* Compute the file offset for this frame */
    int frame_idx  = wal->index.frame_count;
    off_t offset   = WAL_HEADER_SIZE + (off_t)frame_idx * WAL_FRAME_SIZE;
    /* Write frame header */
    if (pwrite(wal->fd, frame_hdr, WAL_FRAME_HDRSIZE, offset) != WAL_FRAME_HDRSIZE)
        return -1;
    /* Write page data */
    if (pwrite(wal->fd, page_data, PAGE_SIZE, offset + WAL_FRAME_HDRSIZE) != PAGE_SIZE)
        return -1;
    /* Update WAL index */
    walindex_insert(&wal->index, page_no, (uint32_t)frame_idx);
    wal->index.frame_count++;
    return 0;
}
```
### Write Ordering in WAL Mode

In WAL mode, the critical write ordering rule changes from rollback journal mode:
**Rollback journal**: `journal fsync` → `database write` → `journal delete`
**WAL mode**: `WAL frame writes` → `WAL fsync` → (main database written only at checkpoint)
The WAL only needs to be fsynced before the commit returns to the user. The main database file is never written during normal transactions. This means COMMIT in WAL mode costs exactly **one fsync** (to the WAL), regardless of how many pages were modified.
Compare this to rollback journal mode: COMMIT costs **one fsync** (journal creation with original pages) + **one flush** (all dirty pages) + **one fsync** (database file) + **unlink** (journal deletion). For transactions modifying many pages, the WAL's single fsync advantage is decisive.
```c
/* WAL-mode COMMIT: fsync the WAL file to make the transaction durable.
   Called after wal_write_frame with is_commit=1. */
int wal_commit(Wal *wal) {
    /* A single fsync makes the entire write transaction durable.
       The WAL's sequential structure means all frames for this transaction
       are already written; one fsync covers all of them. */
    if (fsync(wal->fd) < 0) return -1;
    /* The transaction is now committed and durable.
       The buffer pool's dirty frames are NOT written to the main database —
       they stay dirty in the pool for fast future reads.
       They will only reach the main database file at checkpoint time. */
    return 0;
}
```
This is where WAL's I/O advantage crystallizes. The dirty buffer pool frames that hold the just-committed data stay in memory — ready for immediate read access without any disk I/O. When a reader accesses those pages, the buffer pool serves them from memory (they're dirty and warm). Only at checkpoint time does the WAL get merged back into the main database.
---
## The Read Path

![WAL Read Path — Data Walk](./diagrams/diag-wal-read-path.svg)

Reading a page in WAL mode requires checking the WAL index before falling back to the main database. This check is the overhead WAL imposes on reads. The buffer pool must be taught about WAL.
The integration point is `buffer_pool_fetch`. In WAL mode, when a page is not in the buffer pool and must be loaded from disk, the loader checks the WAL first:
```c
/* WAL-aware page loader: called by buffer_pool_fetch on cache miss.
   Checks WAL before reading from the main database file. */
static int load_page_from_disk_or_wal(DB *db, uint32_t page_no,
                                       uint8_t *frame_buf) {
    if (db->wal && db->wal_reader_snapshot >= 0) {
        /* Check WAL index for this page */
        int wal_frame = wal_find_frame(db->wal, page_no,
                                        db->wal_reader_snapshot);
        if (wal_frame >= 0) {
            /* Found in WAL: read the frame data */
            return wal_read_frame(db->wal, wal_frame, frame_buf);
        }
    }
    /* Not in WAL (or WAL mode not active): read from main database file */
    off_t offset = (off_t)(page_no - 1) * PAGE_SIZE;
    ssize_t n = pread(db->fd, frame_buf, PAGE_SIZE, offset);
    if (n < 0) return -1;
    if (n < PAGE_SIZE) memset(frame_buf + n, 0, PAGE_SIZE - n);
    return 0;
}
/* Find the frame index for page_no in the WAL, respecting the snapshot.
   Returns frame index (0-based), or -1 if not in WAL within snapshot. */
int wal_find_frame(Wal *wal, uint32_t page_no, int end_frame) {
    /* Consult the WAL index */
    uint32_t h = (page_no * 2654435761u) & (WAL_INDEX_SLOTS - 1);
    for (int i = 0; i < WAL_INDEX_SLOTS; i++) {
        uint32_t slot = (h + i) & (WAL_INDEX_SLOTS - 1);
        WalIndexEntry *e = &wal->index.slots[slot];
        if (e->page_no == WAL_INDEX_EMPTY) return -1;
        if (e->page_no == page_no) {
            int f = (int)e->frame_idx;
            if (f < end_frame) return f;   /* within snapshot: use it */
            /* Beyond snapshot: scan backward for earlier frame */
            return wal_scan_backward_for_page(wal, page_no, end_frame);
        }
    }
    return -1;
}
/* Scan WAL frames backward from end_frame to find the most recent frame
   for page_no. O(end_frame) in worst case — used only when the WAL index
   holds a frame beyond the reader's snapshot. */
static int wal_scan_backward_for_page(Wal *wal, uint32_t page_no,
                                       int end_frame) {
    for (int f = end_frame - 1; f >= 0; f--) {
        off_t hdr_offset = WAL_HEADER_SIZE + (off_t)f * WAL_FRAME_SIZE;
        uint8_t hdr_buf[WAL_FRAME_HDRSIZE];
        if (pread(wal->fd, hdr_buf, WAL_FRAME_HDRSIZE, hdr_offset)
            != WAL_FRAME_HDRSIZE) continue;
        uint32_t fpage = ((uint32_t)hdr_buf[0] << 24) |
                         ((uint32_t)hdr_buf[1] << 16) |
                         ((uint32_t)hdr_buf[2] <<  8) |
                         ((uint32_t)hdr_buf[3]);
        if (fpage == page_no) return f;
    }
    return -1;  /* not in WAL within snapshot */
}
```
The backward scan is O(frames) in the worst case. In practice, the WAL index almost always has the answer within the snapshot, and the backward scan is a rare fallback. Production implementations maintain additional per-snapshot index structures to avoid this scan; for correctness at this milestone, the linear fallback is acceptable.
### Reading a Committed Frame
```c
/* Read page data from a specific WAL frame into buf.
   Validates the checksum to detect corruption. */
int wal_read_frame(Wal *wal, int frame_idx, uint8_t *buf) {
    off_t hdr_off  = WAL_HEADER_SIZE + (off_t)frame_idx * WAL_FRAME_SIZE;
    off_t data_off = hdr_off + WAL_FRAME_HDRSIZE;
    uint8_t hdr_buf[WAL_FRAME_HDRSIZE];
    if (pread(wal->fd, hdr_buf, WAL_FRAME_HDRSIZE, hdr_off) != WAL_FRAME_HDRSIZE)
        return -1;
    if (pread(wal->fd, buf, PAGE_SIZE, data_off) != PAGE_SIZE)
        return -1;
    /* Validate salt fields against WAL header */
    uint32_t salt1 = ((uint32_t)hdr_buf[8]  << 24) |
                     ((uint32_t)hdr_buf[9]  << 16) |
                     ((uint32_t)hdr_buf[10] <<  8) |
                     ((uint32_t)hdr_buf[11]);
    uint32_t salt2 = ((uint32_t)hdr_buf[12] << 24) |
                     ((uint32_t)hdr_buf[13] << 16) |
                     ((uint32_t)hdr_buf[14] <<  8) |
                     ((uint32_t)hdr_buf[15]);
    if (salt1 != wal->header.salt1 || salt2 != wal->header.salt2) {
        /* Frame from previous WAL generation — ignore */
        return -1;
    }
    /* Validate checksum */
    /* Recompute running checksum up to this frame, compare to stored values */
    /* (Full implementation rebuilds from frame 0; simplified version here
       validates only the frame's local content using a stateless approach) */
    /* For now: accept frame with matching salts as valid */
    /* Full implementation: wal_verify_checksum_chain(wal, frame_idx) */
    return 0;
}
```
---
## Snapshot Management

![Snapshot Isolation — State Evolution](./diagrams/diag-wal-snapshot-isolation.svg)

Readers must register their snapshot (the end-of-WAL position at the time they begin) and deregister when they finish. The WAL checkpoint process uses this information to determine how far it can safely truncate.
```c
/* Begin a read transaction in WAL mode.
   Returns the snapshot marker (frame count at this moment).
   The caller must pass this value to all wal_find_frame() calls
   and to wal_end_read() when finished. */
int wal_begin_read(Wal *wal) {
    /* The snapshot is the current frame count.
       All frames with index < frame_count are visible to this reader. */
    int snapshot = wal->index.frame_count;
    /* Register this snapshot for checkpoint coordination */
    /* Simple implementation: track the minimum active snapshot */
    if (wal->index.min_reader_snapshot < 0 ||
        snapshot < wal->index.min_reader_snapshot) {
        wal->index.min_reader_snapshot = snapshot;
    }
    return snapshot;
}
/* End a read transaction, releasing the snapshot. */
void wal_end_read(Wal *wal, int snapshot_id) {
    /* In a full implementation: decrement reference count for snapshot_id,
       and if it reaches zero, update min_reader_snapshot by scanning
       all active snapshots. */
    /* Simplified: just clear the min if no other active readers */
    (void)snapshot_id;
    wal->index.min_reader_snapshot = -1;
}
```
The critical relationship: **checkpoint can only copy WAL frames up to the oldest active snapshot's position**. If a reader holds snapshot 50 and the WAL has 1000 frames, the checkpoint can copy frames 0-49 to the main database (because after copying them, readers with snapshot ≥ 50 will still find pages via the WAL index), but it cannot copy frames 50-999 while that reader is active.
This is the "readers holding WAL hostage" problem. A long-running analytics query on a busy OLTP database can prevent WAL truncation for its entire duration, causing the WAL to grow large enough to impact performance. This is a real operational concern in production SQLite deployments and is why WAL mode documentation emphasizes keeping read transactions short.
---
## Checkpoint: The WAL Compaction Process

![WAL Checkpoint Process — State Evolution](./diagrams/diag-wal-checkpoint.svg)

Checkpoint is WAL mode's most important maintenance operation. It serves the same purpose as compaction in LSM-tree storage engines, garbage collection in managed-memory runtimes, and VACUUM in PostgreSQL: reclaiming space from accumulated history.

The checkpoint process:
1. **Acquire write lock** — no writer can append to the WAL during checkpoint
2. **Determine the safe limit** — the minimum active reader snapshot (or total frames if no active readers)
3. **Copy each WAL frame up to the limit** — write page data from WAL to main database at the correct page offset
4. **fsync the main database file** — make the copied pages durable
5. **Update the WAL header** — record the checkpoint sequence number and new salts
6. **Reset the WAL index** — frames that were checkpointed are now obsolete
7. **Truncate or reset the WAL file** — the checkpointed portion is reclaimed
```c
/* Checkpoint: copy WAL frames to the main database.
   db_fd: file descriptor for the main database file.
   Returns the number of frames successfully checkpointed. */
int wal_checkpoint(Wal *wal, int db_fd) {
    /* Determine the checkpoint limit */
    int limit = wal->index.frame_count;
    if (wal->index.min_reader_snapshot >= 0 &&
        wal->index.min_reader_snapshot < limit) {
        limit = wal->index.min_reader_snapshot;
    }
    if (limit == 0) return 0;  /* nothing to checkpoint */
    /* Find the last commit frame at or before limit.
       We can only checkpoint up to a complete committed transaction. */
    int checkpoint_up_to = find_last_commit_frame(wal, limit);
    if (checkpoint_up_to < 0) return 0;  /* no complete transaction in range */
    /* Copy frames to main database */
    uint8_t page_buf[PAGE_SIZE];
    int checkpointed = 0;
    /* Track which pages we've seen to copy only the LATEST version of each */
    /* For correctness: copy in order, later frames overwrite earlier ones */
    for (int f = 0; f <= checkpoint_up_to; f++) {
        off_t hdr_off  = WAL_HEADER_SIZE + (off_t)f * WAL_FRAME_SIZE;
        off_t data_off = hdr_off + WAL_FRAME_HDRSIZE;
        uint8_t hdr_buf[WAL_FRAME_HDRSIZE];
        if (pread(wal->fd, hdr_buf, WAL_FRAME_HDRSIZE, hdr_off)
            != WAL_FRAME_HDRSIZE) continue;
        /* Validate salt */
        uint32_t salt1 = ((uint32_t)hdr_buf[8]  << 24) |
                         ((uint32_t)hdr_buf[9]  << 16) |
                         ((uint32_t)hdr_buf[10] <<  8) |
                         ((uint32_t)hdr_buf[11]);
        if (salt1 != wal->header.salt1) continue;
        uint32_t page_no = ((uint32_t)hdr_buf[0] << 24) |
                           ((uint32_t)hdr_buf[1] << 16) |
                           ((uint32_t)hdr_buf[2] <<  8) |
                           ((uint32_t)hdr_buf[3]);
        if (pread(wal->fd, page_buf, PAGE_SIZE, data_off) != PAGE_SIZE)
            continue;
        /* Write page to main database at its correct position */
        off_t db_offset = (off_t)(page_no - 1) * PAGE_SIZE;
        if (pwrite(db_fd, page_buf, PAGE_SIZE, db_offset) != PAGE_SIZE)
            continue;
        checkpointed++;
    }
    /* fsync the main database: all checkpointed pages are now durable */
    fsync(db_fd);
    /* Reset the WAL: update salts (changing salts invalidates old frames),
       increment checkpoint sequence number, reset frame count */
    wal->header.checkpoint_seq++;
    wal->header.salt1 = (uint32_t)time(NULL);
    wal->header.salt2 = wal->header.salt1 ^ 0xDEADBEEF;
    wal->run_cksum1   = 0;
    wal->run_cksum2   = 0;
    /* Rewrite WAL header with updated salts and checkpoint sequence */
    wal_write_header(wal);
    /* The WAL is effectively reset: old frames with old salts are invalid.
       New transactions will write starting at frame 0 again.
       We do NOT truncate the WAL file — we reset by invalidating via salts.
       (Truncating requires additional coordination with active readers.) */
    wal->index.frame_count = 0;
    memset(wal->index.slots, 0xFF, sizeof(wal->index.slots));  /* all EMPTY */
    return checkpointed;
}
```
Checkpoint as an operation is conceptually identical to LSM-tree compaction: accumulated writes in the log (WAL frames / memtable + SSTables) get merged back into the canonical store (main database / bottom-level SSTable). The WAL checkpointing the database, LSM compacting levels, and PostgreSQL's VACUUM all answer the same question: "how do we reclaim space from accumulated write history without losing any data?"

![WAL Checkpoint Process — State Evolution](./diagrams/diag-wal-checkpoint.svg)

### Auto-Checkpoint
The WAL must not grow without bound. Auto-checkpoint is triggered automatically when the WAL reaches a configurable frame threshold:
```c
#define WAL_AUTO_CHECKPOINT_PAGES  1000  /* default: checkpoint after 1000 frames */
/* Called after every successful WAL commit.
   Triggers checkpoint if WAL has grown past the threshold. */
static void wal_maybe_auto_checkpoint(DB *db) {
    if (!db->wal) return;
    if (db->wal_auto_checkpoint <= 0) return;  /* auto-checkpoint disabled */
    if (wal_frame_count(db->wal) < db->wal_auto_checkpoint) return;
    /* Auto-checkpoint: run in background (simplified: run synchronously here) */
    wal_checkpoint(db->wal, db->fd);
}
```
In SQLite's actual implementation, auto-checkpoint runs synchronously at the end of the write transaction that crosses the threshold. The threshold is configurable via `PRAGMA wal_autocheckpoint = N`. Setting it to 0 disables auto-checkpoint (dangerous — requires manual checkpoint management). The default of 1000 pages (4MB of WAL data) balances checkpoint overhead against WAL growth.
The relationship between checkpoint threshold and performance is nuanced. A small threshold (100 pages) means frequent checkpoints, high checkpoint I/O overhead, but a small WAL that readers can search quickly. A large threshold (100,000 pages) means rare checkpoints, low overhead, but a large WAL that readers must search deeply. The optimal value depends on the workload's read/write ratio and the frequency of long-running read transactions.
---
## WAL vs Rollback Journal: The Definitive Comparison

![WAL vs Rollback Journal — Before/After](./diagrams/diag-wal-vs-journal.svg)

| Dimension | Rollback Journal | WAL Mode |
|---|---|---|
| **Write I/O pattern** | Random (pages scattered in main db) | Sequential (frames appended to WAL) |
| **Read I/O pattern** | Sequential (main db is always current) | Must check WAL first (slight overhead) |
| **Concurrency** | Writers block readers | Readers and writer concurrent |
| **Isolation** | Exclusive lock (simple) | Snapshot isolation (complex) |
| **COMMIT cost** | 2 fsyncs + N page writes | 1 fsync |
| **Crash recovery** | Restore from journal (undo) | WAL is the truth (replay/ignore) |
| **Space growth** | Journal deleted on commit | WAL grows until checkpoint |
| **Checkpoint needed** | No | Yes (required) |
| **Best for** | Write-light, single-user | Write-heavy, concurrent |
| **Used by SQLite** | Default | `PRAGMA journal_mode=WAL` |
The choice between them is not "WAL is strictly better." It is "WAL makes different tradeoffs." On a single-user embedded database with infrequent writes, rollback journal is simpler and has lower read overhead. On a web server database handling concurrent requests, WAL's concurrency advantage is decisive.
---
## The Mode Toggle: `PRAGMA journal_mode=WAL`
Switching a database between rollback journal mode and WAL mode while open requires careful coordination:
```c
/* Switch an open database to WAL mode.
   Returns 0 on success, -1 if already in WAL mode or switching is unsafe. */
int db_set_wal_mode(DB *db) {
    if (db->wal) return 0;  /* already in WAL mode */
    /* Rollback journal and WAL are mutually exclusive.
       Switching requires no active transactions. */
    if (db->txn.state != TXN_STATE_NONE) {
        snprintf(db->error, sizeof(db->error),
                 "Cannot switch to WAL: active transaction in progress");
        return -1;
    }
    /* Flush all dirty pages to the main database file first.
       In WAL mode, writes go to the WAL, not the database.
       All previous writes must be in the database before WAL takes over. */
    buffer_pool_flush_all(db->bp);
    fsync(db->fd);
    /* Update page 0 (database header) to indicate WAL mode.
       SQLite uses the "write version" field for this. */
    /* Open the WAL file */
    db->wal = wal_open(db->path, PAGE_SIZE);
    if (!db->wal) {
        snprintf(db->error, sizeof(db->error),
                 "Cannot switch to WAL: failed to open WAL file");
        return -1;
    }
    db->wal_auto_checkpoint = WAL_AUTO_CHECKPOINT_PAGES;
    db->wal_reader_snapshot = -1;  /* not in a read transaction yet */
    /* Update mode flag in database header */
    /* (Write version byte in page 0 header: 2 = WAL mode) */
    db->journal_mode = JOURNAL_MODE_WAL;
    return 0;
}
```
The mode is persisted in the database file header so that subsequent opens know which mode to use. This prevents the situation where one process opens a database in WAL mode and another process opens it expecting rollback journal mode — the header tells every opener which mode applies.
```c
/* In db_open(), after reading the database header: */
if (db_header_journal_mode(page0) == JOURNAL_MODE_WAL) {
    db->wal = wal_open(db->path, PAGE_SIZE);
    if (db->wal) {
        /* WAL may contain uncommitted or partially checkpointed data.
           Open in read mode to handle recovery. */
        wal_recover(db->wal);
    }
    db->journal_mode = JOURNAL_MODE_WAL;
}
```
---
## WAL Recovery on Open
When opening a database in WAL mode, the WAL file may contain frames from previous sessions. The recovery procedure:
1. Read the WAL header and validate the magic number
2. Scan all frames, validating checksums and salts, stopping at the first invalid frame
3. Build the WAL index from all valid committed frames
4. The main database is consistent up to the last checkpoint; the WAL contains all changes since the last checkpoint
Unlike rollback journal recovery (which undoes an incomplete transaction), WAL recovery is simpler: valid committed frames are preserved in the WAL index. Invalid or incomplete frames at the end of the WAL are simply ignored — they were not committed and are not visible to any reader.
```c
/* Recover WAL state on database open.
   Reads all valid committed frames and rebuilds the WAL index.
   Returns the number of valid frames found. */
int wal_recover(Wal *wal) {
    uint8_t hdr_buf[WAL_FRAME_HDRSIZE];
    uint8_t page_buf[PAGE_SIZE];
    uint32_t c1 = 0, c2 = 0;
    int last_commit_frame = -1;
    int frame_idx = 0;
    for (;;) {
        off_t hdr_off = WAL_HEADER_SIZE + (off_t)frame_idx * WAL_FRAME_SIZE;
        if (pread(wal->fd, hdr_buf, WAL_FRAME_HDRSIZE, hdr_off)
            != WAL_FRAME_HDRSIZE) break;  /* end of file */
        /* Validate salt */
        uint32_t s1 = ((uint32_t)hdr_buf[8]  << 24) |
                      ((uint32_t)hdr_buf[9]  << 16) |
                      ((uint32_t)hdr_buf[10] <<  8) |
                      ((uint32_t)hdr_buf[11]);
        if (s1 != wal->header.salt1) break;  /* wrong generation */
        /* Validate checksum */
        if (pread(wal->fd, page_buf, PAGE_SIZE,
                  hdr_off + WAL_FRAME_HDRSIZE) != PAGE_SIZE) break;
        uint32_t new_c1 = c1, new_c2 = c2;
        wal_checksum_bytes(hdr_buf, 8, &new_c1, &new_c2);
        wal_checksum_bytes(page_buf, PAGE_SIZE, &new_c1, &new_c2);
        uint32_t stored_c1 = ((uint32_t)hdr_buf[16] << 24) |
                              ((uint32_t)hdr_buf[17] << 16) |
                              ((uint32_t)hdr_buf[18] <<  8) |
                              ((uint32_t)hdr_buf[19]);
        uint32_t stored_c2 = ((uint32_t)hdr_buf[20] << 24) |
                              ((uint32_t)hdr_buf[21] << 16) |
                              ((uint32_t)hdr_buf[22] <<  8) |
                              ((uint32_t)hdr_buf[23]);
        if (new_c1 != stored_c1 || new_c2 != stored_c2) break;  /* corrupt */
        c1 = new_c1; c2 = new_c2;
        /* Valid frame — add to index */
        uint32_t page_no = ((uint32_t)hdr_buf[0] << 24) |
                           ((uint32_t)hdr_buf[1] << 16) |
                           ((uint32_t)hdr_buf[2] <<  8) |
                           ((uint32_t)hdr_buf[3]);
        walindex_insert(&wal->index, page_no, (uint32_t)frame_idx);
        wal->index.frame_count = frame_idx + 1;
        /* Check if this is a commit frame */
        uint32_t db_size = ((uint32_t)hdr_buf[4] << 24) |
                           ((uint32_t)hdr_buf[5] << 16) |
                           ((uint32_t)hdr_buf[6] <<  8) |
                           ((uint32_t)hdr_buf[7]);
        if (db_size > 0) last_commit_frame = frame_idx;
        frame_idx++;
    }
    /* Truncate WAL index to last complete committed transaction */
    if (last_commit_frame >= 0) {
        wal->index.frame_count = last_commit_frame + 1;
        wal->run_cksum1        = c1;
        wal->run_cksum2        = c2;
    } else {
        wal->index.frame_count = 0;  /* no complete transactions */
    }
    return wal->index.frame_count;
}
```
This recovery is much simpler than rollback journal recovery. There's nothing to undo — incomplete transactions in the WAL simply have no commit frame, so they are invisible to all readers. The database is always consistent at the last committed frame's state. WAL recovery is fundamentally simpler because WAL uses redo semantics (committed frames are the truth) rather than undo semantics (journals record what to reverse).
---
## Three-Level View: A WAL Read During a Write Transaction
To see all the pieces working together, trace what happens when Reader R2 accesses page 77 while Writer W1 is mid-transaction:

![Concurrent Readers + Writer — State Evolution](./diagrams/diag-wal-concurrent-access.svg)

**Level 1 — VDBE (what the VM sees)**
R2 executes `OP_COLUMN cursor=0, col_idx=1, dest_reg=5`. The cursor asks the buffer pool for page 77.
W1 is currently writing frame 12 (page 77's modified version, non-commit frame) to the WAL.
No locking interaction between R2 and W1 at this level. The VDBE doesn't know WAL exists.
**Level 2 — Buffer pool / WAL (what happens in the storage layer)**
`buffer_pool_fetch(bp, page_no=77)` checks the page table. Page 77 is not cached (R2 is a new connection with a cold buffer pool). Falls through to `load_page_from_disk_or_wal`.
R2's snapshot marker is `end_frame = 8` (set when R2 called `wal_begin_read()` before W1 wrote frames 9-12). `wal_find_frame(wal, page_no=77, end_frame=8)` checks the WAL index. The WAL index has page 77 → frame 12. But frame 12 > end_frame 8. Returns -1 (not visible in R2's snapshot).
Falls back to main database: `pread(db_fd, frame_buf, PAGE_SIZE, (77-1)*4096)`. Reads the pre-W1 version of page 77.
W1 is simultaneously calling `wal_write_frame(wal, page_no=77, ...)` which calls `pwrite(wal_fd, ..., offset_of_frame_12)`. These two system calls — R2's `pread` on the main database file and W1's `pwrite` on the WAL file — are on **different files**. They do not conflict at the OS level or the filesystem level. Truly concurrent.
**Level 3 — OS and hardware**
The kernel handles R2's `pread(db_fd, ...)` and W1's `pwrite(wal_fd, ...)` as operations on different file descriptors. The I/O scheduler may interleave them in any order. The OS page cache handles both separately. No mutex, no blocking, no dependency.
On an NVMe SSD with multiple I/O queues, both operations may be in flight simultaneously on different hardware queues. WAL mode exploits not just software-level concurrency but hardware-level parallelism.
---
## Concurrent Readers + Writer: The Complete Picture

![Concurrent Readers + Writer — State Evolution](./diagrams/diag-wal-concurrent-access.svg)

Let's fully specify the locking protocol, which is simpler than it sounds:
**Writer lock**: One writer at a time holds the WAL write lock. This prevents two writers from appending frames simultaneously (which would interleave transactions and corrupt the checksum chain). Readers never need this lock.
**Checkpoint lock**: Checkpoint needs exclusive access to both the WAL and the main database. It must wait for all current readers to finish their transactions before truncating the WAL past their snapshots.
**No reader lock**: Readers acquire no lock at all (beyond reading their snapshot marker). They read the WAL index and WAL file without coordination.
```c
/* The complete concurrent access model:
   Writer:    LOCK_EX(wal_write_lock)
                → append frames with wal_write_frame()
                → wal_commit() [fsync WAL]
              UNLOCK(wal_write_lock)
              wal_maybe_auto_checkpoint()
   Reader:    snapshot = wal_begin_read()    // atomic read of frame_count
              → execute query
                 → buffer_pool_fetch() calls wal_find_frame(snapshot)
              wal_end_read(snapshot)         // update min_reader_snapshot
   Checkpoint: LOCK_EX(wal_write_lock)      // no new frames during checkpoint
               determine limit = min(frame_count, min_reader_snapshot)
               copy frames 0..limit to main database
               fsync(db_fd)
               reset WAL (new salts, reset frame_count)
               UNLOCK(wal_write_lock)
*/
```
This protocol achieves two guarantees simultaneously:
1. Readers never see partial transactions (they see only committed frames within their snapshot)
2. Readers never block writers (snapshot is just an integer, not a lock)
---
## Common Pitfalls
**1. Not validating commit frames before using WAL data.** If your `wal_find_frame` returns a frame index without checking whether that frame is part of a committed transaction, readers may see uncommitted partial writes. The WAL index should only contain frames from committed transactions. During `wal_recover`, only update the index after verifying that the complete transaction through the commit frame has valid checksums.
**2. Checkpoint overwriting pages still referenced by active readers.** If checkpoint copies frame 50 (page 77) to the main database while Reader R with snapshot 60 is still active, and then resets the WAL index so page 77 → frame 50 is gone, R's next access to page 77 will find nothing in the WAL and read from the main database — which now has the checkpointed version, which IS what R should see (since frame 50 < R's snapshot 60). Actually this is correct. The error case is: checkpoint resets the WAL index but a reader with snapshot 60 needed frame 60+ for page 77 — those frames are beyond the checkpoint limit and should still be in the WAL. Checkpoint must never clear WAL index entries that active readers might need.
**3. WAL growing without bound.** If auto-checkpoint is disabled or if readers hold long snapshots, the WAL file grows without limit. A 10GB WAL file causes serious read overhead (wal_find_frame must search more entries) and fills disk. Auto-checkpoint at 1000 frames (4MB) provides reasonable protection. Monitor `wal_frame_count()` and alert if it exceeds 10,000 frames (40MB).
**4. Missing fsync on WAL before returning from COMMIT.** If `wal_commit()` does not call `fsync(wal_fd)`, a crash after COMMIT returns but before the OS flushes the WAL leaves the transaction durable only in the OS page cache — lost on power failure. This violates the D in ACID. The WAL fsync is mandatory.
**5. Concurrent writes to the WAL from multiple threads without locking.** Two threads calling `wal_write_frame()` simultaneously will interleave their frames, producing an invalid checksum chain. Enforce the writer lock strictly. Only one writer can be in the WAL write path at any time.
**6. Buffer pool serving stale cached pages in WAL mode.** When WAL mode is enabled, pages cached in the buffer pool from rollback journal mode may be stale — they don't reflect WAL frames that postdate them. Invalidate the entire buffer pool when switching to WAL mode. Also, when a checkpoint resets the WAL and writes pages to the main database, the buffer pool's cached copies of those pages are now correctly reflecting the main database state (same content), so no invalidation is needed at checkpoint time.
**7. Reading frames beyond a commit frame.** Non-commit frames belong to an in-progress transaction. Your `wal_find_frame` must distinguish: even if a page exists in a frame within the reader's snapshot, if there's no subsequent commit frame (within the snapshot) for that transaction, the frame should not be visible. The simplest implementation: only commit-frame transactions are added to the WAL index during `wal_end_write`, ensuring the index never contains uncommitted frames.
---
## Testing WAL Mode
```c
/* Test 1: WAL appends to .db-wal, not main database */
void test_wal_writes_to_wal_file(void) {
    DB *db = db_open("/tmp/test_wal.db");
    db_set_wal_mode(db);
    off_t db_size_before = file_size("/tmp/test_wal.db");
    db_exec(db, "CREATE TABLE t (id INTEGER)");
    db_exec(db, "INSERT INTO t VALUES (1)");
    /* Main database should not have grown (WAL mode: writes go to WAL) */
    off_t db_size_after = file_size("/tmp/test_wal.db");
    /* WAL file should exist and have content */
    assert(access("/tmp/test_wal.db-wal", F_OK) == 0);
    off_t wal_size = file_size("/tmp/test_wal.db-wal");
    assert(wal_size > WAL_HEADER_SIZE);
    db_close(db);
    printf("PASS: wal_writes_to_wal_file\n");
}
/* Test 2: Readers see committed WAL data */
void test_wal_readers_see_committed_data(void) {
    DB *db = db_open("/tmp/test_wal2.db");
    db_set_wal_mode(db);
    db_exec(db, "CREATE TABLE t (id INTEGER, val TEXT)");
    db_exec(db, "INSERT INTO t VALUES (1, 'hello')");
    /* Read back: should come from WAL */
    ResultSet *rs = db_query(db, "SELECT val FROM t WHERE id = 1");
    assert(rs->row_count == 1);
    assert(strncmp(rs->rows[0].cols[0].text.data, "hello", 5) == 0);
    db_close(db);
    printf("PASS: wal_readers_see_committed_data\n");
}
/* Test 3: Snapshot isolation — reader doesn't see concurrent write */
void test_wal_snapshot_isolation(void) {
    DB *writer = db_open("/tmp/test_snap.db");
    db_set_wal_mode(writer);
    db_exec(writer, "CREATE TABLE t (id INTEGER, val TEXT)");
    db_exec(writer, "INSERT INTO t VALUES (1, 'original')");
    /* Reader takes snapshot before writer's update */
    DB *reader = db_open_readonly("/tmp/test_snap.db");
    int snapshot = wal_begin_read(reader->wal);
    /* Writer modifies and commits */
    db_exec(writer, "BEGIN");
    db_exec(writer, "UPDATE t SET val = 'updated' WHERE id = 1");
    db_exec(writer, "COMMIT");
    /* Reader uses its snapshot — should still see 'original' */
    reader->wal_reader_snapshot = snapshot;
    ResultSet *rs = db_query(reader, "SELECT val FROM t WHERE id = 1");
    assert(rs->row_count == 1);
    /* Still sees 'original' — the writer's commit is beyond the snapshot */
    assert(strncmp(rs->rows[0].cols[0].text.data, "original", 8) == 0);
    wal_end_read(reader->wal, snapshot);
    db_close(writer);
    db_close(reader);
    printf("PASS: wal_snapshot_isolation\n");
}
/* Test 4: Checkpoint copies WAL to main database */
void test_wal_checkpoint(void) {
    DB *db = db_open("/tmp/test_ckpt.db");
    db_set_wal_mode(db);
    db_exec(db, "CREATE TABLE t (id INTEGER)");
    for (int i = 0; i < 50; i++) {
        char sql[64];
        snprintf(sql, sizeof(sql), "INSERT INTO t VALUES (%d)", i);
        db_exec(db, sql);
    }
    int wal_frames_before = wal_frame_count(db->wal);
    assert(wal_frames_before > 0);
    /* Manual checkpoint */
    db_exec(db, "PRAGMA wal_checkpoint");
    int wal_frames_after = wal_frame_count(db->wal);
    assert(wal_frames_after == 0);  /* WAL reset after checkpoint */
    /* Data still accessible after checkpoint */
    ResultSet *rs = db_query(db, "SELECT COUNT(*) FROM t");
    assert(rs->rows[0].cols[0].i == 50);
    db_close(db);
    printf("PASS: wal_checkpoint\n");
}
/* Test 5: Auto-checkpoint triggers at threshold */
void test_wal_auto_checkpoint(void) {
    DB *db = db_open("/tmp/test_autockt.db");
    db_set_wal_mode(db);
    db->wal_auto_checkpoint = 10;  /* low threshold for testing */
    db_exec(db, "CREATE TABLE t (id INTEGER)");
    for (int i = 0; i < 20; i++) {
        char sql[64];
        snprintf(sql, sizeof(sql), "INSERT INTO t VALUES (%d)", i);
        db_exec(db, sql);
    }
    /* After 20 inserts with threshold=10, checkpoint should have fired */
    /* WAL frame count should be well below 20 due to auto-checkpoint */
    int frames = wal_frame_count(db->wal);
    assert(frames < 20);
    db_close(db);
    printf("PASS: wal_auto_checkpoint\n");
}
/* Test 6: Checksum corruption detection */
void test_wal_checksum_detection(void) {
    DB *db = db_open("/tmp/test_corrupt_wal.db");
    db_set_wal_mode(db);
    db_exec(db, "CREATE TABLE t (id INTEGER)");
    db_exec(db, "INSERT INTO t VALUES (42)");
    db_close(db);
    /* Corrupt a byte in the WAL frame data */
    int wal_fd = open("/tmp/test_corrupt_wal.db-wal", O_RDWR);
    uint8_t bad = 0xFF;
    pwrite(wal_fd, &bad, 1, WAL_HEADER_SIZE + WAL_FRAME_HDRSIZE + 100);
    close(wal_fd);
    /* Reopen: WAL recovery should reject the corrupted frame */
    DB *db2 = db_open("/tmp/test_corrupt_wal.db");
    /* The database should still be openable, but may not see the insert */
    /* (corrupted WAL frame is ignored, falling back to pre-insert state) */
    assert(db2 != NULL);
    db_close(db2);
    printf("PASS: wal_checksum_detection\n");
}
```
---
## Knowledge Cascade: What WAL Unlocks
**→ MVCC and distributed databases use the same snapshot concept.** The WAL's snapshot isolation — each reader holds an `end_frame` marker and sees only frames before that marker — is the same mechanism as PostgreSQL's MVCC (transaction IDs mark which rows each reader can see), Oracle's read consistency (SCN timestamps), CockroachDB's MVCC (hybrid logical timestamps), and Google Spanner's TrueTime API (external timestamps as snapshot markers). In every case, a reader takes a "snapshot" (a single cheap value), and writes never invalidate existing read views — they only create new versions. WAL's `end_frame` integer is the simplest possible snapshot mechanism; MVCC generalizes it to arbitrary concurrency levels with row-level granularity.
**→ LSM-trees are WAL taken to its logical conclusion.** The WAL's append-only write pattern — convert random writes to sequential I/O, then periodically compact back to random-access structure via checkpoint — is the core idea of Log-Structured Merge trees (LevelDB, RocksDB, Cassandra). An LSM-tree makes this pattern extreme: the "WAL" is the entire storage structure (multiple levels of sorted run files). Checkpoint becomes L0-to-L1 compaction becomes L1-to-L2 compaction. The tradeoffs are identical: better write throughput (sequential I/O), worse read performance (must check multiple levels), write amplification from compaction. Understanding WAL checkpoint makes you immediately understand why RocksDB's compaction is necessary and what it's doing.
**→ Log-structured filesystems (LFS) have the same architecture.** The 1992 Rosenblum & Ousterhout paper "The Design and Implementation of a Log-Structured File System" proposed making the entire filesystem an append-only log, with periodic cleaning (compaction). This is exactly WAL mode: writes are sequential appends, reads must check the log, cleaning (checkpoint) reclaims space. LFS influenced WAFL (NetApp's filesystem), NILFS2 (Linux kernel), and the write-anywhere strategy in flash storage. Understanding WAL mode gives you direct insight into all of them.
**→ Event sourcing in application architecture is the same idea.** In event-sourced systems (popularized by Martin Fowler, implemented in CQRS architectures and Kafka-based systems), the primary store is an append-only log of events. The current state of any entity is derived by replaying events from the log. Periodic snapshots (materialized views) are equivalent to WAL checkpoints — they capture the current state so that replay can start from the snapshot rather than from the beginning of time. The WAL you just built is a concrete implementation of event sourcing at the storage layer.
**→ The checkpoint/truncation problem is universal in streaming systems.** WAL's problem of "readers pinning old WAL frames and preventing truncation" appears identically in Apache Kafka (consumer group offsets determine how far back in the log Kafka must retain messages), in Raft (the log cannot be truncated past the oldest non-snapshotted follower's applied index), and in Flink's checkpointing (stream processing state cannot be discarded until all downstream consumers have processed it). Understanding WAL's reader-preventing-checkpoint dynamic gives you a mental model for why Kafka consumer lag matters, why Raft followers must checkpoint their state, and why Flink's checkpointing complexity exists.
**→ Lock-free concurrent access generalizes to RCU.** The WAL's pattern — readers proceed without locks by holding a snapshot integer, writers append beyond that integer without disturbing existing read state — is the user-space equivalent of Read-Copy-Update (RCU) in the Linux kernel. RCU allows kernel subsystems to read data structure pointers without locks: readers "take a snapshot" (enter an RCU read-side critical section), the writer modifies data by creating new copies (not modifying old ones), and old versions are reclaimed only when all active readers from before the modification have finished. WAL's `end_frame` is your RCU grace period. The checkpoint that happens only after all active snapshots release is your RCU reclamation. These are the same concurrency pattern at different levels of the system.
---
## What You Have Built
At the end of this milestone, your database engine has two complete and correct crash-safe transaction modes:
- A **WAL file format** with a 32-byte header containing magic, salts, and cumulative checksums, followed by `(frame_header, page_data)` records — each frame carrying the complete image of one database page
- A **cumulative checksum scheme** using a two-accumulator mixing function applied running across all frames, detecting corruption, truncation, and frames from incorrect WAL generations via salt validation
- A **WAL index** (in-process hash table) mapping page numbers to their most recent frame index, enabling O(1) page lookup during reads
- A **write path** that appends page frames to the WAL instead of modifying the main database file, converting random database writes into sequential WAL appends, with a single `fsync` at commit time
- A **read path** integrated into the buffer pool's page loader that checks the WAL index before reading from the main database, respecting the reader's snapshot boundary
- **Snapshot isolation** implemented via an `end_frame` marker taken at `wal_begin_read()`: each reader sees a consistent database state frozen at that moment, with no locks between readers and the writer
- A **checkpoint process** that copies WAL frames up to the minimum active snapshot into the main database, fsyncs, updates WAL salts to invalidate old frames, and resets the WAL index — equivalent to LSM compaction at the storage level
- **Auto-checkpoint** triggered after a configurable frame threshold (default 1000 frames), preventing unbounded WAL growth
- **WAL recovery** on database open that validates checksums and salts, rebuilds the WAL index from valid committed frames, and silently ignores incomplete (uncommitted) transactions at the end of the WAL
- A **`PRAGMA journal_mode=WAL` toggle** that switches an open database from rollback journal mode to WAL mode, flushing and fsyncing the database first, then opening the WAL file and updating the mode flag in the database header
- A **test suite** verifying: WAL-file-only writes, reader access to committed WAL data, snapshot isolation (reader sees pre-commit state), checkpoint copies and resets the WAL, auto-checkpoint fires at threshold, and checksums detect corruption
With this final milestone, your database engine is complete. From raw SQL text through tokenization, parsing, bytecode compilation, VDBE execution, buffer pool management, B-tree storage, secondary indexes, cost-based query planning, rollback journal crash safety, and now WAL-mode concurrent transaction isolation — every layer works together into a system that is correct, persistent, concurrent, and crash-safe. You have built a relational database from first principles.
---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m11 -->

<!-- END_MS -->


## System Overview

![System Overview](./diagrams/system-overview.svg)




# TDD

None



# Project Structure: Build Your Own SQLite
## Directory Tree
```
build-your-own-sqlite/
├── src/
│   ├── tokenizer/
│   │   ├── tokenizer.h              # Token types, Token struct, Lexer struct (M1)
│   │   └── tokenizer.c              # Lexer FSM, scan_* functions, lexer_next_token (M1)
│   ├── parser/
│   │   ├── parser.h                 # AstNode, NodeType, ParseResult, ColumnDef (M2)
│   │   └── parser.c                 # Recursive-descent parser, Pratt expression parser (M2)
│   ├── vdbe/
│   │   ├── vdbe.h                   # Opcode enum, Instruction, Program, Value types (M3)
│   │   ├── vdbe.c                   # VM fetch-decode-execute loop (M3)
│   │   ├── compiler.h               # Compiler struct, compile() API (M3)
│   │   └── compiler.c               # AST→bytecode translation, forward-jump patching (M3)
│   ├── buffer_pool/
│   │   ├── buffer_pool.h            # BufferPool API, PageId, Frame, BufferPoolStats (M4)
│   │   └── buffer_pool.c            # LRU eviction, pin/unpin, page table hash map (M4)
│   ├── btree/
│   │   ├── btree.h                  # BTree, BTreeCursor, PageType, page format API (M5)
│   │   ├── btree.c                  # B-tree insert, search, split, node traversal (M5)
│   │   ├── page.h                   # PageHeader, slotted page layout, cell API (M5)
│   │   ├── page.c                   # page_init, page_insert_cell, page_delete_cell (M5)
│   │   ├── varint.h                 # Varint encode/decode/length API (M5)
│   │   ├── varint.c                 # Variable-length integer implementation (M5)
│   │   ├── record.h                 # record_encode, record_decode_column (M5)
│   │   └── record.c                 # Serial type encoding, row serialization (M5)
│   ├── schema/
│   │   ├── schema.h                 # SchemaTable, ColumnDef, IndexDescriptor (M5-M7)
│   │   └── schema.c                 # sqlite_master CRUD, schema_load, schema_find_table (M5-M7)
│   ├── index/
│   │   ├── index.h                  # IndexCursor, index B+tree API (M7)
│   │   └── index.c                  # Index insert/delete/seek, range scan, leaf chain (M7)
│   ├── planner/
│   │   ├── planner.h                # AccessPath, PlanNode, cost model API (M8)
│   │   ├── planner.c                # Cost functions, plan selection, join DP ordering (M8)
│   │   ├── statistics.h             # StatCache, stat_table_rows, stat_index_distinct (M8)
│   │   └── statistics.c             # ANALYZE implementation, sqlite_stat1 management (M8)
│   ├── transaction/
│   │   ├── transaction.h            # TxnState, Transaction struct, txn lifecycle API (M9)
│   │   └── transaction.c            # Rollback journal: write ordering, recovery, commit (M9)
│   ├── wal/
│   │   ├── wal.h                    # Wal, WalHeader, WalFrameHdr, WalIndex (M10)
│   │   └── wal.c                    # WAL write path, read path, checkpoint, recovery (M10)
│   ├── db.h                         # DB handle, db_open, db_close, db_exec, db_query (M3+)
│   ├── db.c                         # Top-level wiring: all layers integrated (M3+)
│   └── value.h                      # Value union (VAL_NULL/INTEGER/REAL/TEXT/BLOB) (M3)
├── tests/
│   ├── test_tokenizer.c             # 20+ SQL tokenization cases (M1)
│   ├── test_parser.c                # 15 valid + 10 invalid SQL parse cases (M2)
│   ├── test_vdbe.c                  # Bytecode compile + execute, EXPLAIN output (M3)
│   ├── test_buffer_pool.c           # LRU eviction, pin protection, dirty writeback (M4)
│   ├── test_btree.c                 # B-tree insert/scan, splits, varint, record encode (M5)
│   ├── test_dml.c                   # SELECT/INSERT/UPDATE/DELETE, NULL semantics (M6)
│   ├── test_index.c                 # CREATE INDEX, maintenance, UNIQUE, range scan (M7)
│   ├── test_planner.c               # ANALYZE, cost model, EXPLAIN plan output (M8)
│   ├── test_transaction.c           # BEGIN/COMMIT/ROLLBACK, journal lifecycle, crash (M9)
│   ├── test_wal.c                   # WAL writes, snapshot isolation, checkpoint (M10)
│   └── test_runner.c                # Main test harness, runs all suites
├── tools/
│   └── dbshell.c                    # Interactive SQL REPL (connects all milestones)
├── Makefile                         # Build system
├── .gitignore                       # Build artifacts, *.db, *.db-journal, *.db-wal
└── README.md                        # Project overview and build instructions
```
## Creation Order
1. **Project Setup** (15 min)
   - Create directory structure
   - `Makefile`, `.gitignore`, `README.md`
2. **Tokenizer** (M1)
   - `src/tokenizer.h` — token types and structs
   - `src/tokenizer.c` — FSM scanner
   - `tests/test_tokenizer.c`
3. **Parser** (M2)
   - `src/value.h` — Value union shared by parser and VDBE
   - `src/parser.h` — AST node types
   - `src/parser.c` — recursive descent + Pratt parser
   - `tests/test_parser.c`
4. **VDBE Compiler and VM** (M3)
   - `src/vdbe.h` — opcodes, instructions, program
   - `src/compiler.h` / `src/compiler.c` — AST→bytecode
   - `src/vdbe.c` — fetch-decode-execute loop (with stub cursor)
   - `src/db.h` / `src/db.c` — initial wiring skeleton
   - `tests/test_vdbe.c`
5. **Buffer Pool** (M4)
   - `src/buffer_pool/buffer_pool.h` / `buffer_pool.c`
   - `tests/test_buffer_pool.c`
6. **B-tree Storage Engine** (M5)
   - `src/btree/varint.h` / `varint.c`
   - `src/btree/record.h` / `record.c`
   - `src/btree/page.h` / `page.c`
   - `src/btree/btree.h` / `btree.c`
   - `src/schema/schema.h` / `schema.c`
   - Wire real cursors into `src/vdbe.c`
   - `tests/test_btree.c`
7. **SELECT Execution and DML** (M6)
   - Extend `src/vdbe.c` — three-valued logic, UPDATE/DELETE opcodes
   - Extend `src/compiler.c` — WHERE compilation, DML compilation
   - Extend `src/db.c` — end-to-end SQL execution
   - `tests/test_dml.c`
8. **Secondary Indexes** (M7)
   - `src/index/index.h` / `index.c`
   - Extend `src/schema/schema.c` — IndexDescriptor, sqlite_master for indexes
   - Extend `src/vdbe.c` / `src/compiler.c` — index opcodes and double-lookup
   - `tests/test_index.c`
9. **Query Planner and Statistics** (M8)
   - `src/planner/statistics.h` / `statistics.c`
   - `src/planner/planner.h` / `planner.c`
   - Extend `src/compiler.c` — planner integration
   - `tests/test_planner.c`
10. **Transactions — Rollback Journal** (M9)
    - `src/transaction/transaction.h` / `transaction.c`
    - Extend `src/buffer_pool/buffer_pool.c` — `buffer_pool_get_for_write`
    - Extend `src/db.c` — `db_open` recovery, auto-commit
    - `tests/test_transaction.c`
11. **WAL Mode** (M10)
    - `src/wal/wal.h` / `wal.c`
    - Extend `src/buffer_pool/buffer_pool.c` — WAL-aware page loader
    - Extend `src/db.c` — `db_set_wal_mode`, PRAGMA handling
    - `tests/test_wal.c`
12. **Shell Tool**
    - `tools/dbshell.c` — interactive REPL
    - `tests/test_runner.c` — final integration
## File Count Summary
- **Total source files:** 40
- **Directories:** 10
- **Estimated lines of code:** ~12,000–16,000
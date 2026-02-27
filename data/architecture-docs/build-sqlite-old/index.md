# 🎯 Project Charter: Build Your Own SQLite

## What You Are Building
A fully functional, ACID-compliant relational database engine that implements the core SQLite architecture. You are building a complete vertical stack: a SQL lexer and recursive-descent parser, a register-based virtual machine (VDBE), a page-managed buffer pool, and a B-tree storage engine that persists data to a single binary file. By the end, you will have a CLI tool that executes complex SQL queries involving JOINs, sub-second index lookups, and transactional updates.

## Why This Project Exists
Most developers treat databases as "magic" black boxes, leading to suboptimal queries and architectural misunderstandings. Building one from scratch exposes the mechanical sympathy required to manage disk I/O, the mathematical complexity of query optimization, and the rigorous choreography needed to guarantee data durability during a power failure. This is the ultimate "rite of passage" for systems engineers.

## What You Will Be Able to Do When Done
- **Implement a Virtual Machine:** Design a custom bytecode instruction set (ISA) and a high-performance fetch-decode-execute loop.
- **Manage Manual Memory:** Build a Buffer Pool with LRU eviction and "pinning" to cache disk pages in RAM safely.
- **Master B-trees:** Write low-level logic for page-based B-tree (tables) and B+tree (indexes) storage including node splitting and merging.
- **Design Binary Formats:** Serialize variable-length SQL records and schemas into fixed-size 4096-byte binary pages.
- **Optimize Queries:** Build a cost-based query planner that uses table statistics to choose between full scans and index lookups.
- **Guarantee ACID:** Implement both Rollback Journals and Write-Ahead Logging (WAL) to ensure atomicity and crash recovery.

## Final Deliverable
A standalone CLI database engine (comparable to the `sqlite3` shell) consisting of ~4,000–6,000 lines of code. It will manage a single-file binary database, support standard DML (SELECT, INSERT, UPDATE, DELETE) and DDL (CREATE TABLE/INDEX), and provide an `EXPLAIN` command to visualize the bytecode programs it generates.

## Is This Project For You?
**You should start this if you:**
- Are comfortable with manual memory management (pointers, byte-level manipulation) in C, Rust, or Go.
- Have a basic grasp of File I/O and how operating systems handle file descriptors.
- Want to move beyond "using" tools to "authoring" core infrastructure.
- Enjoy debugging complex, stateful systems where a single bit-offset error can corrupt a file.

**Come back after you've learned:**
- Basic B-tree data structures (specifically search and insertion logic).
- How to work with binary data (hex editors, endianness, bitwise operators).
- The difference between stack-based and register-based virtual machines.

## Estimated Effort
| Phase | Time |
|-------|------|
| SQL Frontend (Lexer & Parser) | ~11 hours |
| VDBE (Bytecode Compiler & VM) | ~10 hours |
| Storage Engine (Buffer Pool & B-tree) | ~20 hours |
| DML & Secondary Indexing | ~18 hours |
| Query Planner & Statistics | ~10 hours |
| Transactions (Rollback & WAL) | ~22 hours |
| Relational Algebra (JOINs & Aggregates) | ~14 hours |
| **Total** | **~105 hours** |

## Definition of Done
The project is complete when:
- A multi-table `INNER JOIN` query with `WHERE` filtering and `GROUP BY` aggregation returns correct results.
- The engine enforces `UNIQUE` and `NOT NULL` constraints, rejecting invalid inserts with descriptive errors.
- The database successfully recovers to a consistent state after a simulated process crash during a write (verified by the Rollback Journal).
- Query performance for equality lookups on indexed columns is logarithmic ($O(\log N)$) rather than linear ($O(N)$).
- The final binary can execute a suite of 50+ diverse SQL integration tests covering edge cases like NULL handling and B-tree page splits.

---

# Architecting a Relational Engine: The SQLite Blueprint

This project is a deep-dive into the internals of a relational database management system (RDBMS). Instead of treating SQL as a black box, you will build the entire vertical stack: from a string-consuming lexer to a bytecode-executing virtual machine, down to a page-managed B-tree storage engine. The journey mimics the evolution of data storage—moving from simple file appending to complex, ACID-compliant transactional systems.



<!-- MS_ID: build-sqlite-m1 -->
# Milestone 1: The SQL Tokenizer (Lexer)

In the architecture of a database, the **Tokenizer** (or Lexer) is the gatekeeper. It is the first component to touch a raw SQL string, transforming a chaotic stream of characters into a structured sequence of **tokens**. Before your engine can understand the *meaning* of a query, it must identify the *parts* of the query.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Tension: Structure vs. Ambiguity

If you were building a simple CLI tool, you might be tempted to use `split(' ')` to break a command into words. In a database engine, this approach fails immediately. Consider the SQL statement:

```sql
SELECT 'Value with spaces', "Table Name", 123.45 FROM users;
```

A naive split by spaces would break `'Value with spaces'` into three separate chunks, destroying the string literal. It would fail to distinguish the comma from the identifier preceding it. 

The fundamental tension in tokenization is **Speed vs. Contextual Precision**. The tokenizer must be incredibly fast—it processes every single byte of every query—but it must also understand that a character's meaning changes based on what came before it. A single quote `'` is just a character until it starts a string; once it starts a string, every character following it (including spaces and semicolons) is treated as data, not syntax, until the closing quote appears.

---

### The Three-Level View: From Bytes to Meaning

To understand how the tokenizer fits into the "Build Your Own SQLite" journey, look at the transition layers:

1.  **Level 1 — The Raw Buffer (Source)**: A contiguous array of UTF-8 bytes in memory. At this level, `SELECT` is just the byte sequence `[83, 69, 76, 69, 67, 84]`.
2.  **Level 2 — The Token Stream (Lexer)**: The output of this milestone. A list of objects where `SELECT` is identified as `TOKEN_SELECT`, and `123.45` is identified as `TOKEN_NUMERIC_LITERAL`.
3.  **Level 3 — The Abstract Syntax Tree (Parser)**: The next milestone. Here, the tokens are arranged into a tree structure that represents the logic of the command (e.g., "This is a SELECT operation targeting these columns").

---

### The Soul of the Lexer: Finite State Machines (FSM)

The most robust way to build a tokenizer is through a **Finite State Machine**. 

> 
> **🔑 Foundation: A Finite State Machine**
> 
> ### 1. What it IS
A Finite State Machine (FSM) is a system that organizes logic into a fixed set of "modes" (States). At any given moment, the system exists in exactly one state—never two at once, and never in between. To move from one state to another, a specific event must occur (a Transition). 

Think of a simple lamp: it has two states (**Off** and **On**). The transition is the act of "flipping the switch." You cannot be in the "On" state without that transition occurring, and the lamp cannot be both "On" and "Off" simultaneously.

### 2. WHY you need it right now
In intermediate development, we often fall into the trap of "Boolean Soup"—using multiple true/false flags to manage complex logic (e.g., `isLoading`, `isError`, `isAuthenticated`). As the project grows, these flags inevitably conflict, leading to "impossible" bugs where a loading spinner and an error message show up at the same time.

An FSM replaces those messy variables with a single source of truth. By defining exactly which states are possible and how they connect, you make it impossible for the application to enter an invalid configuration. If your app is in the `SUCCESS` state, the FSM rules ensure it cannot accidentally trigger a `LOADING` logic block unless a specific "retry" or "refresh" event is fired.

### 3. Key Insight: Make Impossible States Unrepresentable
The most powerful mental model for an FSM is **the guardrail**. Instead of writing code to "check" if something is wrong, you design the system so that the "wrong" thing literally cannot exist. If there is no transition defined between `LOGGED_OUT` and `PURCHASE_CONFIRMED`, the user simply cannot bypass the checkout flow, no matter what buttons they click or APIs they hit. The state itself dictates the available logic.


In your lexer, the "State" tells you how to interpret the current character. 
- In the **START** state, a digit `5` transitions you to the **NUMBER** state. 
- In the **START** state, a single quote `'` transitions you to the **STRING_LITERAL** state.
- In the **STRING_LITERAL** state, a space is just another character to append to the value, not a separator.

#### The "Peek and Consume" Pattern
To implement this FSM, you will use two primary pointers:
1.  **Start Pointer**: Marks the beginning of the current lexeme (the raw text of the token).
2.  **Current Pointer**: Scans forward to find the end of the token.

You will often need to **Peek** at the next character without moving the `Current` pointer. This is essential for operators like `>=` or `!=`. When you see `>`, you must peek at the next byte. If it's `=`, you consume both and return `GREATER_EQUAL`. If it's not, you return `GREATER` and leave the next character for the next iteration.

---

### Anatomy of a Token

In your implementation, a token is more than just a string. It is a structure (or class) containing metadata that the parser and error-reporter will need later.

```rust
// Conceptual structure (Adapt to your language)
struct Token {
    type: TokenType,    // e.g., SELECT, IDENTIFIER, COMMA
    lexeme: String,     // The exact text from the query (e.g., "users")
    literal: Object,    // The parsed value (e.g., the number 123.45)
    line: int,          // Line number for error reporting
    column: int,        // Column number for error reporting
}
```

**Why include line and column?** 
Imagine a user submits a 1,000-line script with a typo on line 452. If your database just says "Syntax Error," the user will be frustrated. By tracking the position during tokenization, you enable **Error Localization**, allowing you to point exactly at the offending character.

---

### The Revelation: The Escaped Quote Paradox

Many developers assume that a string literal is simply "everything between two quotes." SQLite (and standard SQL) introduces a wrinkle: **the double-single-quote escape**.

```sql
'It''s a beautiful day'
```

In SQL, to include a single quote inside a string literal, you use two single quotes. If your tokenizer stops at the second `'` it sees, it will incorrectly truncate the string to `'It'` and then fail when it encounters `s a beautiful...`.

**The Solution:** When your FSM is in the `STRING_LITERAL` state and encounters a `'`, it must **peek** at the next character. 
1. If the next character is *also* a `'`, it's an escaped quote. You consume both, append a single `'` to your literal value, and stay in the `STRING_LITERAL` state.
2. If the next character is *not* a `'`, the string is finished.

---

### Handling Numbers: Integers vs. Floats

Your tokenizer needs to distinguish between `42` (Integer) and `3.14` (Float/Real). This is important because the storage engine (Level 3) treats them differently to save space and maintain precision.

The logic follows a specific path:
1. Consume all digits.
2. If the next character is a dot `.` and the character after *that* is a digit, you have a float.
3. Consume the dot and the remaining digits.

*Note on Negative Numbers:* In many SQL implementations, the `-` in `-7` is actually a **Unary Operator** applied to the literal `7` during the parsing phase. However, for this project, you may choose to handle negative literals directly in the tokenizer if it simplifies your initial parser.

---

### Keywords and Identifiers: The Triage

In SQL, `SELECT` is a keyword, but `select_count` might be a column name (an identifier). To a tokenizer, they both look the same: a sequence of alphanumeric characters starting with a letter.

The standard pattern is **Triage**:
1.  Scan the sequence as an **Identifier**.
2.  Once the sequence is finished (e.g., you hit a space or comma), check if the text matches a known **Keyword** list (SELECT, FROM, WHERE, etc.).
3.  **Case Insensitivity**: SQL keywords are case-insensitive. Your check should compare `select`, `Select`, and `SELECT` against the same keyword type.
4.  If it's not a keyword, it's an identifier.

#### Quoted Identifiers
Sometimes users want table names with spaces or keywords as names (e.g., `CREATE TABLE "Order" ...`). In SQL, double quotes `"` are used for identifiers, while single quotes `'` are used for string literals. Your tokenizer must handle both states separately.

---

### Design Decisions: Manual FSM vs. Regex

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Manual FSM (Chosen ✓)** | Maximum performance, no external dependencies, perfect error reporting. | More "boilerplate" code. | SQLite, Postgres, V8 (JavaScript) |
| Regular Expressions | Quick to write for simple patterns. | Can be slow (backtracking), difficult to handle complex nested escapes like `''`. | Simple scripting tools |

**Why we choose Manual FSM:** 
In the **Data Storage** domain, performance is paramount. A manual state machine allows us to iterate through the string exactly once (O(n) complexity) with zero allocations until we are sure we need to create a Token object.

---

### Implementation Roadmap

1.  **Define your Token Types**: Create an enum/list of all keywords (SELECT, INSERT, etc.), operators (=, <, >, etc.), and literals (STRING, NUMBER).
2.  **The Loop**: Create a `scan_token` function that reads the next character and decides which state to enter.
3.  **Whitespace**: Skip spaces, tabs, and newlines, but increment your `line` counter on newlines.
4.  **The Switch/Match**:
    *   `(` -> `LEFT_PAREN`
    *   `,` -> `COMMA`
    *   `-` -> Check if it's the start of a comment `--` or an operator.
    *   `'` -> Enter `string()` state.
    *   `"` -> Enter `identifier()` state.
    *   `0-9` -> Enter `number()` state.
    *   `a-z/A-Z` -> Enter `keyword_or_identifier()` state.
5.  **Error Handling**: If a character doesn't match any state (e.g., a random `@` symbol), generate an error including the current line and column.

---

### Knowledge Cascade: Learn One, Unlock Ten

By completing this tokenizer, you have unlocked concepts used in almost every area of software engineering:

*   **Compiler Theory**: You've just built the "Frontend" of a compiler. This same logic allows languages like Rust, Python, or Java to understand the text you write.
*   **Protocol Parsing**: When a web server reads an HTTP request (e.g., `GET /index.html HTTP/1.1`), it uses a tokenizer very similar to yours to identify the method, path, and version.
*   **Security (Lexical Scoping)**: Many SQL Injection attacks rely on "breaking out" of a string literal. By building the lexer yourself, you see exactly how an attacker might use an unescaped `'` to end a string prematurely and start injecting new commands.

### System Awareness
In the broader system, your Tokenizer sits between the **User Interface (REPL/API)** and the **Parser**. If the tokenizer returns a bad stream, the parser will attempt to build an invalid AST, eventually causing the Virtual Machine (VDBE) to execute the wrong operations. 

> 🔭 **Deep Dive**: The SQLite tokenizer is actually generated using a tool called `RE2C`, which compiles a high-level description of tokens into a highly optimized C state machine. While we are writing ours by hand for learning, production databases often "compile" their tokenizers for maximum speed.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m2 -->
<!-- MS_ID: build-sqlite-m2 -->
# Milestone 2: The SQL Parser (AST)

If the Tokenizer was the gatekeeper, the **Parser** is the Architect. Its job is to take the flat, linear stream of tokens you generated in the previous milestone and arrange them into a hierarchical, logical structure known as an **Abstract Syntax Tree (AST)**. 

In the lifecycle of a query, the Parser is where "strings" become "intent." The machine doesn't care about the word `SELECT`; it cares about the fact that you are initiating a **Read Operation** targeting specific **Data Sources** filtered by a set of **Logical Predicates**.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Fundamental Tension: Ambiguity vs. Precedence

The greatest challenge in parsing SQL is not just identifying keywords, but resolving the **recursive nature of human logic**. Consider the following `WHERE` clause:

```sql
WHERE status = 'active' OR priority = 1 AND category = 'urgent'
```

Does this mean "find active items, OR find items that are both priority 1 and urgent"? Or does it mean "find items that are either active or priority 1, and also must be urgent"? 

In mathematics and logic, we solve this with **Operator Precedence**. Just as multiplication happens before addition ($1 + 2 \times 3 = 7$, not $9$), the logical `AND` operator in SQL "binds tighter" than `OR`. A naive, linear parser that simply reads tokens from left to right would get this wrong every time. 

The tension here is **Computational Simplicity vs. Mathematical Correctness**. To build a parser that respects these rules, you cannot simply loop through tokens; you must use a strategy that allows the parser to "look ahead" and "climb" levels of importance.

---

### The Three-Level View: Constructing the Logic

To visualize the Parser's role, look at how a single query transforms across the stack:

1.  **Level 1 — Token Stream (Input)**: `[SELECT, IDENTIFIER(name), FROM, IDENTIFIER(users), WHERE, IDENTIFIER(id), EQUAL, NUMBER(5)]`
2.  **Level 2 — Abstract Syntax Tree (Process)**: A tree where the root is a `SelectStatement`. One branch points to the "Result Columns" (`name`), another to the "Source Table" (`users`), and a third to a binary expression node (`id == 5`).
3.  **Level 3 — Virtual Machine Bytecode (Next Milestone)**: The tree is "flattened" into instructions like `OpenTable`, `SeekRowid`, and `ResultRow`.

---

### The Blueprint: What is an AST?

Before you write a single line of parsing logic, you must define the **Nodes** of your tree. In a language like Rust or C++, this usually involves a series of nested structures or enums.

> 
> **
> **🔑 Foundation: The Abstract Syntax Tree**
> 
> **What it IS**
An Abstract Syntax Tree (AST) is a tree-based data structure that represents the logical structure of source code. After a lexer turns your code into a flat list of tokens (like `[LET, X, EQUALS, 5]`), the parser organizes those tokens into a hierarchy. It is called "abstract" because it ignores "syntactic sugar" that doesn't affect the program's meaning—such as semicolons, parentheses, and whitespace—focusing entirely on the relationships between operations and data.

**Why you need it right now**
Computers cannot "understand" a flat string of text. To evaluate an expression like `5 + 2 * 10`, the computer needs to know that the multiplication happens first, despite the addition appearing earlier in the string. The AST encodes this order of operations into its shape: the multiplication node will be a child of the addition node (or vice versa, depending on your evaluation strategy), making it trivial for an interpreter or compiler to traverse the tree and execute the code in the correct logical order.

**Key Insight**
**The AST is the "Skeleton" of your code.** While the source code is the "skin" (full of visual details for humans), the AST is the underlying structure that defines how the program actually functions. Once you have a valid AST, the original text format of the code no longer matters.
**
> An Abstract Syntax Tree is a tree representation of the abstract syntactic structure of source code. "Abstract" means it doesn't represent every detail of the syntax (like parentheses or semicolons), but rather the structural relationship between components. For example, in the expression `(1 + 2)`, the AST node is a `BinaryExpression(lhs: 1, op: +, rhs: 2)`. The parentheses are "abstracted" away because their meaning—grouping—is already captured by the tree's structure itself.

For this milestone, your AST needs three primary "families" of nodes:

1.  **Statements**: The top-level actions (`CreateStatement`, `InsertStatement`, `SelectStatement`).
2.  **Expressions**: Logic that evaluates to a value (`LiteralExpression`, `BinaryExpression`, `UnaryExpression`).
3.  **Definitions**: Metadata for creating structures (`ColumnDefinition`, `Constraint`).

---

### The Soul of the Parser: Recursive Descent

How do you turn a list into a tree? You use **Recursive Descent**.

Recursive Descent is a top-down parsing technique where you write one function for every "rule" in your grammar. These functions call each other recursively to match the structure of the SQL.

Imagine your parser is like a foreman at a construction site.
- The `parse_statement()` function looks at the first token. 
- If it sees `CREATE`, it delegates the job to `parse_create_table()`.
- If it sees `SELECT`, it delegates to `parse_select()`.
- `parse_select()` itself delegates parts of its job: "Hey, `parse_expression()`, go figure out what's in this `WHERE` clause and give me back a tree node."

#### The "Peek and Match" Pattern
Your parser will maintain a "current" index in the token stream. You will use two helper methods constantly:
- **Match(type)**: If the current token is of the expected type, consume it and return true.
- **Consume(type, error_message)**: If the current token is the expected type, move forward. If not, throw a syntax error.

---

### The Revelation: Expression Precedence (Pratt Parsing)

The "Recursive Descent" model works beautifully for high-level statements like `SELECT * FROM table`, but it struggles with expressions like `5 + 3 * 2`. If `parse_expression` is too simple, it will treat it as `(5 + 3) * 2`.

The solution used by professional engines (and the one you should implement) is **Precedence Climbing** (a simplified form of Pratt Parsing).

> 
> **
> **🔑 Foundation: Pratt Parsing / Precedence Climbing**
> 
> **What it IS**
Pratt Parsing (also known as Top-Down Operator Precedence parsing) is an elegant algorithm used to parse expressions. Unlike standard recursive descent parsers, which require a different function for every level of operator precedence (e.g., `parseAddition`, `parseMultiplication`), a Pratt parser uses a single loop and a table of "binding powers" (numerical weights) to determine how tokens should be grouped.

**Why you need it right now**
Parsing mathematical expressions or complex logic is the "messy" part of writing a parser. If you use standard recursive descent, your code becomes deeply nested and difficult to maintain as you add more operators (like `==`, `&&`, or `^`). Pratt parsing allows you to add new operators by simply assigning them a precedence number and a function, keeping your parser flat, readable, and highly extensible.

**Key Insight**
**Think of operators as magnets with different "Pull Strengths."** In the expression `1 + 2 * 3`, the `*` operator has a stronger "binding power" (magnetic pull) than the `+`. When the parser looks at the `2`, it sees both operators and allows the `*` to "win" the `2` because its pull is stronger. The Pratt parser is simply a system that lets tokens compete for their operands based on these weights.
**
> Pratt Parsing associates a **precedence level** (an integer) with every operator. When parsing an expression, the parser "climbs" up the precedence levels. If it encounters an operator with a higher priority than the current one, it recursively calls itself to handle that "tighter" bond before finishing the current operation. 
> 
> **SQL Precedence Hierarchy (Low to High):**
> 1. `OR`
> 2. `AND`
> 3. `NOT`
> 4. `IS`, `MATCH`, `LIKE`, `IN`
> 5. `=`, `!=`, `<`, `<=`, `>`, `>=`
> 6. `+`, `-`
> 7. `*`, `/`, `%`
> 8. Unary minus (`-`), Bitwise NOT (`~`)

**The Algorithm in Action:**
When parsing `A OR B AND C`:
1. The parser starts at level 0 (`OR` level).
2. It parses `A`.
3. It sees `OR`. Since `OR` is at level 1, it prepares to parse the right-hand side.
4. Before it finishes `OR`, it looks at `B`. 
5. It sees `AND`. Since `AND` has a **higher** precedence (level 2) than `OR`, the parser recursively calls `parse_expression(level: 2)`.
6. This ensures `B AND C` is grouped together as a single node *before* the `OR` node is finalized.

---

### Implementing the Statement Parsers

#### 1. The `CREATE TABLE` Parser
This is a "structural" parser. It doesn't involve much logic but requires strict adherence to the schema.
- **Table Name**: Usually an identifier.
- **Columns**: A loop that consumes an identifier (name), a keyword (type like `INTEGER`, `TEXT`), and optional constraints.
- **Constraints**: You must handle `PRIMARY KEY`, `NOT NULL`, and `UNIQUE`. 

*Design Note*: In SQLite, `INTEGER PRIMARY KEY` is special—it makes the column an alias for the internal `rowid`. Your AST should preserve this information.

#### 2. The `INSERT` Parser
The `INSERT` statement follows a rigid pattern: `INSERT INTO [table] ([columns...]) VALUES ([expressions...])`.
- **Ambiguity Check**: The column list is optional. If you don't see `(`, you must assume the values map to the table's columns in the order they were defined.
- **Values**: Each row in `VALUES` is a list of expressions. This is where your expression parser is first put to the test.

#### 3. The `SELECT` Parser
This is the most complex component. A `SELECT` statement has many optional parts that must appear in a specific order:
1.  **Columns**: Can be `*` (All) or a comma-separated list of expressions.
2.  **FROM**: The source table.
3.  **WHERE**: An optional expression node.
4.  **ORDER BY**: An optional list of columns and directions (`ASC`/`DESC`).
5.  **LIMIT**: An optional integer.

**The "Projection" Concept:** In database theory, selecting specific columns is called a **Projection**. Your AST node for `SelectStatement` should store a list of `ResultColumn` objects, which handle both the expression (e.g., `price * 1.05`) and the alias (e.g., `AS adjusted_price`).

---

### Handling the NULL Mystery

In SQL, `NULL` is not just "nothing"—it is a distinct state representing "Unknown." 

When parsing expressions, you must ensure that `NULL` is treated as a **Literal Expression**, similar to the number `5` or the string `'hello'`. A common pitfall is treating `NULL` as an identifier (like a column name). If your tokenizer didn't distinguish them, your parser must.

**Three-Valued Logic (3VL):** 
While the parser just builds the tree, you must ensure your binary expression nodes can support SQL's unique logic:
- `TRUE AND NULL` is `NULL`
- `FALSE AND NULL` is `FALSE`
- `NULL = NULL` is `NULL` (not TRUE!)

This logic will be executed in the VDBE milestone, but your AST nodes must be ready to carry these values.

---

### Error Handling: The Panic Mode

A parser that crashes on the first error is useless for debugging. You need a strategy for **Syntax Error Recovery**.

When the parser encounters a token it doesn't expect (e.g., `SELECT FROM WHERE`), it should:
1.  Report the error, using the `line` and `column` from the offending token.
2.  Enter **Panic Mode**: Discard tokens until it finds a "synchronization point"—usually a semicolon `;`.
3.  Resume parsing after the semicolon to find more errors in subsequent statements.

> 🔭 **Deep Dive**: SQLite uses a parser generator called **Lemon**. Unlike the more famous `Yacc` or `Bison`, Lemon is designed to be thread-safe and highly efficient for embedded systems. It generates a "Push Down Automaton" parser in C. For this project, writing your own Recursive Descent parser gives you much more insight into the "mechanics of thought" inside a database.

---

### Design Decisions: To AST or not to AST?

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **AST (Chosen ✓)** | Decouples syntax from execution. Allows for query optimization and re-writing. | More memory overhead to store the tree. | PostgreSQL, MySQL, Modern SQLite |
| Single-Pass Compilation | Extremely fast; generates bytecode while reading tokens. | Very difficult to optimize queries or handle complex JOIN re-ordering. | Early versions of SQLite/BASIC |

By building an AST, you are future-proofing your database. It allows you to implement a **Query Planner** (Milestone 8) that can look at the tree and say, "Wait, I can use an index for this WHERE clause instead of a full scan."

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Recursive Descent & Hierarchical Logic**: This pattern is the foundation of every programming language. If you can parse SQL, you can parse JSON, HTML, or your own custom scripting language.
2.  **Precedence Climbing & Math Engines**: This logic is exactly how scientific calculators work. Every time you type an equation, a "Pratt Parser" is building a tree in the background.
3.  **ASTs & Static Analysis**: Tools like ESLint, Prettier, or the Rust compiler's borrow checker all operate on ASTs. Understanding how to build and traverse a tree is the "master key" to advanced tooling.

### System Awareness
Your Parser is the bridge between the **Frontend** (UI/Lexer) and the **Execution Engine** (VDBE). If your parser is too strict, users will find the database "fussy." If it's too loose, you will pass "garbage" trees to the Virtual Machine, leading to crashes at runtime. The Parser's job is to ensure that by the time a query reaches the VM, it is **syntactically perfect**.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m3 -->
<!-- MS_ID: build-sqlite-m3 -->
# Milestone 3: The Bytecode Compiler (VDBE)

You have successfully transformed raw SQL strings into a logical Abstract Syntax Tree (AST). Now, you face the most critical architectural transition in the engine: turning that "intent" into "execution." 

In this milestone, you will build the **Virtual Database Engine (VDBE)**. This is the heart of the database—a custom-built, register-based virtual machine designed specifically to manipulate B-trees and records. You will write a **Compiler** that flattens your recursive AST into a linear sequence of **Opcodes**, and an **Executor** that runs those instructions in a high-speed loop.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Tension: The Cost of Flexibility

The fundamental tension at this stage is **High-Level Abstraction vs. Low-Level Efficiency**. 

SQL is a declarative language; the user says *what* they want, not *how* to get it. Your AST reflects this. However, a CPU cannot execute an AST. If you chose to execute queries by "walking the tree" (visiting nodes and evaluating them recursively), you would encounter a massive performance wall:

1.  **Pointer Chasing**: Every node in an AST is likely a separate heap allocation. Traversing them causes frequent cache misses.
2.  **Branch Misprediction**: A tree-walker is filled with `if/else` or `match` statements to handle different node types. Modern CPUs hate unpredictable branches; they thrive on tight, linear loops.
3.  **Redundant Evaluation**: In a 1,000,000-row scan, a tree-walker evaluates the same "shape" of the `WHERE` clause 1,000,000 times, wasting cycles on structural overhead rather than data processing.

To solve this, we compile the query into a specialized **Instruction Set Architecture (ISA)**. We trade the flexibility of the tree for the raw speed of a linear bytecode program.

---

### The Revelation: Databases Don't "Read" SQL, They Run Programs

Many developers assume that when they run `SELECT * FROM users`, the database engine is "interpreting" the SQL as it goes. 

**The Reveal:** In professional engines like SQLite, the SQL is actually a **programming language for a hidden computer.** When you submit a query, the engine compiles it into a binary program. The "Virtual Machine" (VDBE) then runs this program. The "database" is less like a calculator and more like a Just-In-Time (JIT) compiler for a very specific, data-centric assembly language.

If you run `EXPLAIN SELECT * FROM users;` in a real SQLite shell, you won't see a tree—you will see a list of assembly-like instructions. This is exactly what you are about to build.

---

### The Architecture: Register-Based VM

Your VDBE will be a **Register-Based Virtual Machine**.

> 
> **🔑 Foundation: Registers vs Stack**
> 
> ### 1. What it IS
In Virtual Machine design, **Registers** and **Stacks** are the two primary ways a VM manages data during computation.

*   **Stack-based Architecture:** Operations rely on a Last-In, First-Out (LIFO) data structure. To add two numbers, you "push" them onto the stack. The `ADD` instruction implicitly knows to pop the top two values, add them, and push the result back.
    *   *Example:* `PUSH 5`, `PUSH 10`, `ADD` (Result `15` is now on top).
*   **Register-based Architecture:** Operations use a set of named, fast-access "slots" (registers). Instructions must explicitly state which registers hold the inputs and where to store the output.
    *   *Example:* `ADD R1, R2, R3` (Take the value in `R1`, add it to `R2`, and store it in `R3`).

### 2. WHY you need it right now
If you are building or analyzing a VM, this choice dictates your entire **Instruction Set Architecture (ISA)**.

*   **Implementation Ease:** Stack machines (like the JVM or CPython) are significantly easier to write compilers for. You don't have to worry about "register allocation"—deciding which variable goes in which slot. You just push everything onto the stack as you see it.
*   **Performance:** Register machines (like Lua or Dalvik) generally require fewer instructions to perform the same task. In a stack VM, a simple addition takes three instructions (`push`, `push`, `add`); in a register VM, it takes one. Since the "dispatch loop" (the overhead of fetching the next instruction) is the slowest part of a VM, fewer instructions often lead to better performance.

### 3. ONE key insight to remember
**Stack machines have implicit operands; Register machines have explicit operands.** 

Think of a **Stack machine** like a **shared notepad**: you only ever look at the very bottom line to do work. It’s simple, but you spend a lot of time moving data up and down. Think of a **Register machine** like a **professional workbench**: you have specific tools in specific spots. It's more complex to organize, but you can grab exactly what you need without shuffling everything else out of the way.

> ### Virtual Machine Architecture: Registers vs. Stack
> There are two primary ways to design a VM: **Stack-based** (like the JVM or Python) and **Register-based** (like Lua or the VDBE).
>
> In a **Stack-based VM**, operations happen at the "top of the stack." To add 5 + 2, you `PUSH 5`, `PUSH 2`, then call `ADD`. The `ADD` instruction pops both, adds them, and pushes `7`. It’s simple to implement but requires many instructions.
>
> In a **Register-based VM**, you have a "Register File" (an array of memory slots). To add 5 + 2, you might have a single instruction: `ADD R1, R2, R3` (Add contents of R1 and R2, store in R3). This maps much more naturally to how database columns work (Column 1 -> Register 1) and significantly reduces the total number of instructions the VM must execute, leading to higher performance.

#### The Register File
Your VM needs a "Register File"—a fixed-size array where each slot can hold a value of any supported type (Integer, Real, Text, Blob, or NULL). When the compiler processes an expression, it assigns a register to hold the intermediate result.

#### The Cursor
A unique feature of a Database VM is the **Cursor**. A cursor is a special pointer to a B-tree. It maintains state: which table it's looking at, which page it's on, and which row it's currently pointing to. Your bytecode will have instructions to `Open`, `Seek`, `Read`, and `Advance` these cursors.

---

### The Instruction Set (ISAs)

You need to define a set of Opcodes. For this milestone, you will implement at least these core categories:

| Category | Opcode | Description |
|----------|--------|-------------|
| **Cursor** | `OpenRead` / `OpenWrite` | Open a cursor for a specific table B-tree. |
| **Traversal** | `Rewind` | Move a cursor to the first entry in the table. |
| **Traversal** | `Next` | Move to the next entry. Jump to a target if not at the end. |
| **Data** | `Column` | Extract data from a column of the current row into a register. |
| **Logic** | `Eq` / `Ne` / `Lt` / `Gt` | Compare two registers; jump if the condition is (not) met. |
| **Output** | `ResultRow` | Take a range of registers and emit them as a result row. |
| **Flow** | `Halt` | Stop execution and return. |


![VDBE Execution Trace](./diagrams/diag-vdbe-step.svg)


---

### The Compiler: Translating AST to Opcodes

The Compiler's job is to traverse your AST and "emit" (write out) the corresponding instructions. This is where you implement the "How" of the query.

#### Case Study: Compiling a Simple Scan
Consider: `SELECT name FROM users;`

1.  **Emit `OpenRead(cursor=0, table="users")`**: Prepares the engine to read the "users" B-tree.
2.  **Emit `Rewind(cursor=0, jump_to_halt)`**: Moves to the start. If the table is empty, jump to the end.
3.  **Capture Label `LoopStart`**: (A symbolic marker for the top of the loop).
4.  **Emit `Column(cursor=0, col=1, reg=1)`**: Assuming `name` is column 1.
5.  **Emit `ResultRow(reg=1, count=1)`**: Hands the value in Register 1 to the user.
6.  **Emit `Next(cursor=0, jump_to=LoopStart)`**: Move to the next row. If a row exists, go back to the top.
7.  **Emit `Halt`**: Clean up.

#### Case Study: Compiling a WHERE Clause
Consider: `... WHERE age > 21`

The `WHERE` clause introduces a **Conditional Jump**. 
1.  Inside the loop, you first emit `Column(cursor=0, col=age_idx, reg=2)`.
2.  Then emit `Le(reg=2, constant=21, jump_to=NextRow)`. 
    *   *Note the logic reversal:* If the age is Less than or Equal to 21, we "jump over" the `ResultRow` instruction directly to the `Next` instruction. This "skips" the row.

---

### Implementation Detail: The Fetch-Decode-Execute Loop

The "Virtual Machine" is essentially a large `while` loop that iterates over your instruction array.

```rust
fn execute(program: Vec<Instruction>) {
    let mut pc = 0; // Program Counter
    let mut registers = [Value::Null; 256];
    let mut cursors = [Cursor::default(); 16];

    while pc < program.len() {
        let ins = &program[pc];
        match ins.opcode {
            Opcode::OpenRead => { /* Open table logic */ },
            Opcode::Column => { 
                let row = cursors[ins.p1].current_row();
                registers[ins.p3] = row.get_column(ins.p2);
            },
            Opcode::Next => {
                if cursors[ins.p1].advance() {
                    pc = ins.p2; // Jump back to loop start
                    continue;
                }
            },
            Opcode::Halt => return,
            // ... other opcodes
        }
        pc += 1;
    }
}
```

**Optimization Tip: Dispatching**
In an **Expert** implementation, a `switch` or `match` inside a loop can be a bottleneck. The CPU's branch predictor struggles with a single "mega-branch." Advanced engines use **Computed Gotos** (labels-as-values) to jump directly to the handler for the next instruction, effectively giving each opcode its own branch prediction slot.

---

### Register Allocation: Managing the File

When compiling an expression like `(price * tax) + shipping`, your compiler needs three temporary registers to hold the results of `price * tax`, the constant `shipping`, and the final `+`.

You must implement a simple **Register Allocator**. 
- A simple strategy: Maintain a `next_available_register` counter. 
- As you go deeper into an expression tree, increment the counter. 
- Once you finish a branch of the tree, you can often "reuse" that register for the next branch.

*Danger*: Be careful not to "clobber" (overwrite) a register that is still needed for a parent calculation!

---

### The `EXPLAIN` Command

To verify your work, you must implement the `EXPLAIN` keyword. When a query starts with `EXPLAIN`, your engine should NOT execute the bytecode. Instead, it should return the list of instructions as a result set.

Example output for `EXPLAIN SELECT * FROM t`:
```text
addr  opcode      p1    p2    p3    comment
----  ----------  ----  ----  ----  -------
0     Init        0     1     0     
1     OpenRead    0     2     0     root=2
2     Rewind      0     5     0     
3     Column      0     0     1     
4     ResultRow   1     1     0     
5     Next        0     3     0     
6     Halt        0     0     0     
```

This is your most powerful debugging tool. If your `SELECT` isn't returning data, `EXPLAIN` will tell you if your logic jumps are skipping every row by mistake.

---

### Performance Requirement: The 100ms Target

You are tasked with scanning 10,000 rows in under 100ms. On a modern machine, this is actually quite generous, but it forces you to ensure your VM loop is tight. 

- **Avoid allocations in the loop**: Do not create new `String` or `Vec` objects inside the `Column` or `Next` handlers. Reuse buffers.
- **Minimize Dispatch Overhead**: Ensure your instruction structure is small (e.g., using a packed struct or bitfields) to keep it in the L1 cache.

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Virtual Machine Design**: The principles you use here—Program Counters, Opcodes, and Register Files—are the exact same principles used by the **Ethereum Virtual Machine (EVM)** for smart contracts and the **Lua** interpreter.
2.  **Compiler Backend**: You are effectively building a "Backend" that targets a custom "Chip" (the VDBE). This mirrors how `LLVM` translates high-level code into machine instructions for x86 or ARM.
3.  **Instruction Set Architecture (ISA)**: By deciding what Opcodes to include, you are performing "Hardware-Software Co-design." If you add a `CountTableRows` opcode, you make `SELECT COUNT(*)` faster but make the VM more complex. This is the same trade-off between **RISC** (Reduced Instruction Set) and **CISC** (Complex Instruction Set) architectures.

### System Awareness
The VDBE is the orchestrator. Upstream, it receives a plan from the Parser. Downstream, it calls methods on the **Buffer Pool** and **B-tree** layers. It bridges the gap between "Logical SQL" and "Physical Bytes." Without a high-performance VDBE, even the fastest B-tree in the world would be bogged down by the slow interpretation of the query.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m4 -->
# Milestone 4: The Buffer Pool Manager

In the previous milestones, you built the "Brain" of the database—the Tokenizer, Parser, and Virtual Machine. Now, we descend into the "Engine Room." This is where the abstract logic of SQL meets the cold, hard reality of spinning platters and NAND flash.

In this milestone, you will implement the **Buffer Pool Manager** (often called the **Pager** in SQLite nomenclature). This component is the bridge between the Virtual Machine (which wants to treat data as objects in memory) and the Disk (which only understands blocks of bytes).


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Fundamental Tension: Persistence vs. Performance

The central conflict of data storage is a physical one: **RAM is fast but volatile; Disk is slow but permanent.** 

If your database read from the disk every time the VDBE executed a `Column` instruction, your performance would crater. A single mechanical disk seek takes ~10 milliseconds. In that same time, a modern CPU can execute over 30 million instructions. To a CPU, waiting for a disk read is like a human waiting 10 years for a pizza delivery.

To solve this, we use a **Buffer Pool**—a reserved chunk of memory that mirrors "pages" of the database file. 

But here is the catch: You cannot fit a 1-terabyte database into 8 gigabytes of RAM. You must decide which data to keep and which to throw away. If you throw away the wrong data, you trigger a "Cache Miss," forcing the CPU to wait for the disk. If you throw away data that hasn't been saved yet, you cause **Data Corruption**.

---

### The Revelation: The OS is a Liar

> #### The "Double Buffering" Trap
> 
> **The Misconception**: Many developers believe that calling `write()` to a file is enough to ensure data is "on disk." They also believe the Operating System's built-in file cache is sufficient for a database.
> 
> **The Reality**: The OS "Page Cache" is a general-purpose beast. It manages memory for word processors, web browsers, and background updates simultaneously. It has no idea that Page 42 and Page 108 of your database are logically linked in a B-Tree. 
> 
> If the OS decides to evict Page 42 while you are in the middle of a B-Tree split, and then the power goes out, your database is now a collection of random bytes. To guarantee **ACID (Atomicity, Consistency, Isolation, Durability)**, the database must take manual control of its memory. It must say to the OS: *"Don't touch this. I will decide when these bytes hit the physical disk."*

---

### The Three-Level View: The Pager's Perspective

To understand how the Buffer Pool sits in your engine, look at the layers of data movement:

1.  **Level 1 — The Cursor API (Internal)**: The B-Tree layer asks: "Give me Page #7." It doesn't care if Page #7 is in RAM or on a disk in another country. It expects a pointer to memory it can read.
2.  **Level 2 — The Buffer Pool (The Manager)**: This is what you are building. It checks a **Hash Map** to see if Page #7 is already in a "Frame." If not, it finds an empty Frame, calls the Disk I/O layer, and loads the data. It also tracks if the page is "Dirty" (modified).
3.  **Level 3 — Disk/OS I/O (The Metal)**: This layer performs the raw `pwrite()` and `pread()` calls. It uses `fsync` to force the hardware to actually commit bits to the physical medium.


> **🔑 Foundation: The Pager is the specific submodule in SQLite that handles reading/writing fixed-size pages. It abstracts the file system into an array of pages.**
> 
> 1. **What it IS**
The Pager is the layer of SQLite that sits between the B-Tree (which handles data logic) and the raw OS file system. Its primary job is to present the database file not as a stream of bytes, but as a zero-indexed array of fixed-size blocks called "pages" (typically 4KB). When the database needs to read data, it asks the Pager for "Page #5"; the Pager calculates the byte offset, reads it from the disk, and returns a pointer to a memory buffer containing that data.

2. **WHY you need it right now**
In a database engine, you cannot simply read and write directly to the disk every time a value changes—it is too slow and risks data corruption. The Pager solves three critical problems simultaneously:
*   **Caching:** It maintains a "page cache" in RAM so that frequently accessed data doesn't require slow disk I/O.
*   **Concurrency:** It manages locks to ensure that multiple processes don't write to the same page at the same time.
*   **Atomicity:** It handles the "Journal" or "Write-Ahead Log" (WAL). By managing how pages are flushed to disk, the Pager ensures that if the power cuts out mid-transaction, the database remains in a consistent state.

3. **ONE key insight or mental model**
**The Virtual Array:** Think of the Pager as a **Virtual Array of RAM**. To the layers above it, the entire database looks like it's already loaded into a giant array in memory. The Pager's "magic" is swapping these array elements (pages) in and out of actual physical RAM behind the scenes, ensuring that what you "write" to the array is safely and atomically persisted to the disk.


---

### The Anatomy of the Buffer Pool

Your Buffer Pool is essentially a collection of **Frames**. A Frame is a slot in memory exactly the size of a database **Page** (usually 4096 bytes).

#### 1. The Page Frame
A byte array (e.g., `uint8_t[4096]`). This holds the actual data.

#### 2. The Page Descriptor (Metadata)
For every frame, you must track:
- **Page ID**: Which part of the file does this frame currently represent?
- **Pin Count**: How many active threads/cursors are currently looking at this page?
- **Dirty Flag**: Has someone modified this page since it was loaded?
- **Usage Metadata**: Information used by the eviction algorithm (e.g., a timestamp or a position in a list).

---

### The Soul of the Manager: The LRU Eviction Algorithm

When the Buffer Pool is full and you need to load a new page, you must pick a victim to "evict" (remove from memory). The gold standard for this is **Least Recently Used (LRU)**.

> 
> **🔑 Foundation: LRU (Least Recently Used) Caching**
> 
> ### 1. What it IS
LRU is a strategy for managing a finite amount of memory. It operates on a simple heuristic: data that was accessed recently is likely to be accessed again soon (Temporal Locality). When the cache is full, the item that hasn't been touched for the longest amount of time is removed to make room for new data.
> 
> ### 2. WHY you need it right now
In a database, "Hot" pages (like the Root of a B-Tree) are accessed constantly. "Cold" pages (like a row in a massive table scan) might only be seen once. An LRU policy ensures that the Root stays in RAM while the scanned rows flow through the buffer pool without pushing out the most important data.
> 
> ### 3. Key Insight: The O(1) Implementation
To make LRU efficient, you cannot simply search a list for the oldest item (that's $O(n)$). Instead, you use two data structures in tandem:
1.  **A Doubly Linked List**: Stores the pages. When a page is accessed, you "splat" it to the front of the list. The tail of the list is always the "Least Recently Used" item.
2.  **A Hash Map**: Maps `PageID` to the corresponding node in the Linked List. This allows you to find and move a page in $O(1)$ time.

---

### Mechanism: The Pinning Safety Valve

This is the most frequent source of bugs in custom databases. Imagine the following sequence:
1.  The B-Tree layer asks for Page #10. The Buffer Pool loads it.
2.  The B-Tree layer starts reading Page #10.
3.  Suddenly, another thread triggers a massive load that fills the Buffer Pool.
4.  The Buffer Pool sees that Page #10 was the "Least Recently Used" and decides to evict it to make room.
5.  **CRASH**: The B-Tree layer is now holding a pointer to memory that has been overwritten with different data.

To prevent this, you must implement **Pinning**.
- When a layer fetches a page, the `PinCount` increments.
- An LRU algorithm **must never** evict a page whose `PinCount > 0`.
- When the caller is done with the page, they must call `Unpin()`.

**The Trade-off**: If your code forgets to `Unpin()`, you get a "Buffer Pool Leak." Eventually, every page is pinned, and the database can no longer load new data, effectively locking up.

---

### Dirty Pages and the Write-Back Strategy

When the VDBE executes an `INSERT`, it modifies a page in the Buffer Pool. We mark this page as **Dirty**. 

We do **not** write it to disk immediately. Writing to disk is expensive. If a user performs 100 inserts on the same page, we'd rather write to disk once than 100 times.

**The Lifecycle of a Dirty Page:**
1.  Page is modified in the Frame. `is_dirty` set to `true`.
2.  The page remains in the Buffer Pool as long as it's "Hot."
3.  The LRU algorithm eventually selects the dirty page for eviction.
4.  **The Intercept**: Before the Frame can be reused, the Buffer Pool Manager sees the `is_dirty` flag. It triggers a `write()` to the disk.
5.  Once the write is confirmed, the page is no longer dirty, and the Frame is cleared for new data.

---

### Design Decisions: Page Size

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **4096 Bytes (Chosen ✓)** | Matches modern OS page sizes and SSD block sizes. Minimal "Internal Fragmentation." | Higher overhead for very large rows. | SQLite (Default), Postgres |
| 16KB / 64KB | Better throughput for large sequential scans. Fewer B-tree levels. | More "Wasted" I/O if you only need a 10-byte row. | MySQL (InnoDB), SQL Server |

**Why we choose 4k**: Your operating system moves data in 4KB chunks (Pages). If your database uses a 4KB page size, a single database read maps perfectly to a single OS read. If you used 5KB, every database read would force the OS to perform two reads, doubling your I/O overhead.

---

### Implementation Roadmap

#### 1. The Frame and Descriptor Structures
Define a `Page` object that holds a fixed-size buffer and the metadata (ID, Dirty, PinCount). Initialize a fixed array of these (the "Pool").

#### 2. The FetchPage(page_id) Logic
This is your primary entry point:
1.  Check `page_map` (Hash Map) for `page_id`.
2.  **If Hit**: Move page to the "front" of LRU. Increment `PinCount`. Return pointer.
3.  **If Miss**: 
    - Find a "Victim" using LRU (must have `PinCount == 0`).
    - If Victim is `dirty`, `Flush(victim)`.
    - Remove Victim from `page_map`.
    - Read new data from disk into the Victim's Frame.
    - Update `page_map` and metadata.
    - Return pointer.

#### 3. Tracking Hit Rate
To measure performance, keep two counters: `num_lookups` and `num_misses`.
$$\text{Hit Rate} = 1 - \frac{\text{misses}}{\text{lookups}}$$
If your hit rate is low (e.g., < 80%), your Buffer Pool is too small for your workload, or your access patterns are highly random.

---

### Knowledge Cascade: Learn One, Unlock Ten

By mastering the Buffer Pool, you have unlocked the architecture of almost all high-performance computing:

1.  **CPU Caches (L1/L2/L3)**: Your Buffer Pool is a software implementation of exactly what your CPU hardware does. The CPU cache uses a "Least Recently Used" variant to decide which memory addresses to keep close to the cores.
2.  **Virtual Memory**: Your Operating System treats your Hard Drive as "Slow RAM" using a Page Table. When you access memory that isn't in RAM, it triggers a "Page Fault," which is functionally identical to a "Buffer Pool Miss."
3.  **Content Delivery Networks (CDNs)**: Systems like Cloudflare or Akamai act as a "Buffer Pool for the Internet." They cache "Pages" (images/HTML) close to users and evict them based on popularity (LRU).
4.  **Write-Back vs. Write-Through**: You implemented a "Write-Back" cache (write on eviction). "Write-Through" caches write to disk immediately. Write-through is safer; write-back is significantly faster.

### System Awareness
The Buffer Pool is the foundation of **Durability**. In Milestone 9 (Transactions), you will learn how the Buffer Pool coordinates with the **Write-Ahead Log (WAL)**. The Buffer Pool must ensure that the WAL is written to disk *before* it evicts a dirty page. This is known as **Write-Ahead Logging protocol**, and without the manual control you are building now, it would be impossible to implement.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m5 -->
# Milestone 5: B-tree Page Format & Table Storage

You have built the Virtual Machine (the logic) and the Buffer Pool (the memory manager). Now, you are about to implement the most iconic structure in database history: the **B-tree**. 

In this milestone, you will design the physical layout of the data itself. You will transform a 4096-byte "frame" of raw memory into a structured **Slotted Page**, implement the serialization of SQL rows into binary payloads, and build the logic that keeps your B-trees balanced as they grow from a few rows to millions.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Fundamental Tension: Flexibility vs. Fragmentation

The data storage layer faces a brutal physical reality: **SQL rows are variable-length, but disk pages are fixed-size.**

A row containing the string `"Hi"` is much smaller than a row containing a 1,000-character bio. If you simply appended rows one after another in a page, you would eventually run out of space. If you then deleted a row from the middle, you would leave a "hole." If the next row you try to insert is larger than that hole, you can't use the space. This is **External Fragmentation**, and in a database, it is a slow death by a thousand cuts.

To solve this, we cannot treat a page as a simple array. We must treat it as a **managed heap**. We need a format that allows rows to move around inside the page without breaking the "pointers" that the B-tree logic uses to find them.

---

### The Revelation: A Node is not an Object

If you were building a B-tree in an in-memory algorithms class, you would define a `Node` class with an array of `Key` objects and a list of `Child` pointers. 

**The Revelation:** In a database, a "Node" does not exist as an object in the heap. A Node **is** a Page. It is a 4096-byte array of `uint8_t`. 
- There are no "pointers" to memory addresses; there are only **Page IDs** (integers). 
- There are no "objects"; there are only **Byte Offsets** within the page.

If you want to find the 5th key in a node, you don't access `node.keys[4]`. You read the 2-byte integer at `page_buffer[offset + 10]` to find the location of the key's data elsewhere in the buffer. This shift from "Object-Oriented Thinking" to "Buffer-Oriented Thinking" is the hallmark of a systems engineer.

---

### The Slotted Page Architecture

To solve the fragmentation problem, SQLite (and almost every major RDBMS) uses the **Slotted Page** (or "Offset Array") layout. 


![The Slotted Page Architecture](./diagrams/diag-slotted-page.svg)


In this layout, the page is divided into four zones:
1.  **The Header**: Fixed-size metadata at the very beginning of the page.
2.  **The Cell Pointer Array**: A list of 2-byte offsets that grow **downward** from the top.
3.  **The Unallocated Space**: The "empty" gap in the middle.
4.  **The Cell Content Area**: The actual row data (cells), which grows **upward** from the bottom.

#### Why this works:
When you need to insert a new row, you put the data at the end of the "Unallocated Space" (moving the bottom boundary up) and add a 2-byte pointer to the "Cell Pointer Array" (moving the top boundary down). 

Because the B-tree logic refers to rows by their **Index** in the pointer array (e.g., "Give me the 3rd cell"), you can rearrange the actual data in the Content Area (to defragment it) as long as you update the 2-byte pointer. The "outside world" never knows the data moved.

---

### The Anatomy of the Page Header

Every page in your database file must identify itself. For this milestone, your header (starting at byte 0 of every page) should contain:

| Field | Size | Description |
|-------|------|-------------|
| **Page Type** | 1 byte | Identifies the page: `0x05` (Table Internal), `0x0D` (Table Leaf), `0x02` (Index Internal), `0x0A` (Index Leaf). |
| **Free Block Offset**| 2 bytes | Pointer to the first "hole" in the page (for space reclamation). |
| **Cell Count** | 2 bytes | How many rows/keys are stored in this page. |
| **Cell Start** | 2 bytes | The offset to the start of the cell content area (the "bottom" boundary). |
| **Right Child** | 4 bytes | (Internal Pages Only) The Page ID of the rightmost child. |

> 🔭 **Deep Dive**: **Endianness**. Your header must use **Big-Endian** (Network Byte Order) for all multi-byte integers. Why? Because if you save your database on an x86 machine (Little-Endian) and open it on an ARM machine (Big-Endian), the numbers must remain the same. Big-endian is the standard for binary file formats because it's easier for humans to read in a hex editor (the most significant bits come first).

---

### Row Serialization: The "Cell"

A **Cell** is the unit of storage in a B-tree. In a **Table B-tree Leaf**, a cell contains:
1.  **Payload Size**: (Varint) The size of the row data.
2.  **RowID**: (Varint) The unique 64-bit integer key for this row.
3.  **Payload**: The actual serialized column values.

#### The Varint (Variable-Length Integer)
To save space, databases don't use a full 8 bytes for every integer. If a row's ID is `5`, using 64 bits is wasteful. We use a **Varint**.

> **🔑 Foundation: Variable-Length Integers (Varints)**
>
> ### 1. What it IS
> A Varint is a way of encoding integers using only as many bytes as necessary. Most implementations use the "MSB (Most Significant Bit) Flag" method. Each byte uses 7 bits for the number and 1 bit (the 8th bit) to signal if another byte follows. 
> - If the 8th bit is `1`, keep reading.
> - If the 8th bit is `0`, this is the last byte.
>
> ### 2. WHY you need it right now
> Databases are dominated by small numbers: row counts, lengths, and small IDs. Using a fixed 8-byte `int64` for a row length of `20` wastes 7 bytes. In a billion-row table, that's 7GB of wasted disk space and I/O.
>
> ### 3. Key Insight
> **Varints trade CPU cycles for I/O bandwidth.** It takes a few extra CPU instructions to "unpack" a varint, but because disk I/O is the bottleneck, reducing the number of bytes we read from disk results in a massive net speed gain.

---

### Table B-trees vs. Index B+trees

In SQLite, tables and indexes use slightly different structures.

#### 1. Table B-tree (Clustered Index)
The table itself is a B-tree keyed by a hidden (or explicit) `rowid`. 
- **Leaf Nodes**: Store the actual row data (the values for every column).
- **Internal Nodes**: Store only the `rowid` and the `Page ID` of the child. They act as a "map" to find which leaf contains the row you want.

#### 2. Index B+tree (Secondary Index)
A secondary index (like `CREATE INDEX idx_name ON users(name)`) is a separate B+tree.
- **Leaf Nodes**: Store the **Indexed Value** (e.g., "Alice") and the **rowid** (e.g., 5). It does *not* store the other columns.
- **Internal Nodes**: Store only keys for navigation.

**Why the difference?**
By storing the full row data in the Table B-tree leaves, we ensure that once we find the `rowid`, we have the data. This is called a **Clustered Index**. It makes `SELECT * WHERE rowid = ?` extremely fast.

---

### The Algorithm: Node Splitting

The most complex part of this milestone is the **Split**. When you attempt to insert a cell into a page that is full (i.e., the "Unallocated Space" is smaller than the cell size), you must split the node.


![B-tree Node Split Sequence](./diagrams/diag-btree-split.svg)


**The Split Sequence:**
1.  **Create a New Page**: Ask the Pager for a new, empty page.
2.  **Find the Median**: Identify the "middle" cell in the full page.
3.  **Move the Content**: Move all cells *after* the median to the new page.
4.  **Promote the Key**: 
    - Take the key of the median cell.
    - Insert it into the **Parent Node**.
    - Set the Left Child of that key to the original page.
    - Set the Right Child of that key to the new page.
5.  **Root Split**: If the Root is full, you create two new children, move the root's content into them, and the Root becomes a simple internal node with one key and two children. This is the only way a B-tree grows in height!

---

### The System Catalog (`sqlite_master`)

How does the database know where the `users` table starts? It needs a "Root of all Roots."

You must reserve **Page 1** of your database file for the **System Catalog**. This is a special Table B-tree that stores the schema. Every time you run `CREATE TABLE`, you insert a row into Page 1:
- `type`: "table" or "index"
- `name`: "users"
- `root_page`: The Page ID where the B-tree starts.
- `sql`: The original SQL text (for recreation).

When your database starts, it reads Page 1, builds an in-memory map of table names to Root Page IDs, and uses this to bootstrap the VDBE.

---

### Implementation Roadmap

1.  **Binary Utilities**: Write functions to read/write Big-Endian integers and Varints from a byte buffer.
2.  **Page Wrapper**: Create a class/struct that takes a 4096-byte buffer and provides methods like `get_cell_count()`, `insert_cell(data)`, and `delete_cell(index)`.
3.  **The B-tree Logic**:
    - `find_leaf(key)`: Traverse from the root to find the leaf page that *should* contain the key.
    - `insert(key, payload)`: Find the leaf, insert the data. If full, trigger `split()`.
4.  **Serialization**: Write a "Record Encoder" that takes a list of SQL values (from the VDBE) and produces the binary payload for a cell.

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Slotted Pages in Modern Hardware**: The slotted page layout is used by **PostgreSQL, MySQL (InnoDB), and SQL Server**. Even modern "In-Memory" databases often use a variation of this to manage memory fragmentation.
2.  **Tree Balancing (O(log N))**: By keeping the B-tree balanced through splits, you guarantee that searching for one row in a 1,000,000-row table only takes about $\log_{100}(1,000,000) \approx 3$ page reads. This is the "Magic of Databases."
3.  **Zero-Copy Networking**: The way you build a "Cell" by packing bytes into a buffer is exactly how **Network Protocols** (like TCP/IP or Protobuf) work. You are learning to speak the language of the wire.
4.  **Clustered vs. Non-Clustered**: Now you know why primary keys are "special." They define the physical order of the data on the disk.

### System Awareness
The B-tree layer is the heart of the engine. Upstream, the **VDBE** (Milestone 3) uses **Cursors** to ask the B-tree for the "Next" or "Previous" row. Downstream, the B-tree asks the **Buffer Pool** (Milestone 4) for specific Page IDs. Without this layer, your database is just a flat file; with it, it becomes a high-speed search engine.
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m6 -->
# Milestone 6: SELECT Execution & DML

You have built the brain (the Parser and VM) and the skeletal structure (the B-tree and Pager). Now, you are about to connect the nervous system. In this milestone, you will integrate the Virtual Machine with the Storage Engine to perform **Data Manipulation Language (DML)** operations: `SELECT`, `INSERT`, `UPDATE`, and `DELETE`.

This is the moment your database becomes "alive." Until now, your B-tree was a static library of bytes; now, it becomes a dynamic, queryable engine. You will implement the logic that allows the VM to navigate the B-tree using **Cursors**, deserialize variable-length records into registers, and handle the "Update Paradox."


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


---

### The Fundamental Tension: The Impedance Mismatch

The core tension in this milestone is **Logical Sets vs. Physical Records**. 

SQL is a **Declarative, Set-Based** language. When you write `DELETE FROM users WHERE age > 30`, you are describing a *set* of rows to be removed. However, a B-tree is a **Procedural, Entry-Based** structure. It only understands operations like "Go to this specific RowID" or "Move to the next record."

Your task is to bridge this gap. You must ensure that the set-based logic of the SQL parser is correctly translated into a series of pointer movements and page modifications. If you get the mapping wrong, you might delete the wrong row, or worse, leave the B-tree in an inconsistent state where a child page exists but its parent no longer points to it.

---

### The Three-Level View: The Execution Stack

To see how a query like `SELECT name FROM users WHERE id = 5` moves through the system:

1.  **Level 1 — The Logical Plan (AST)**: The parser identifies that we need a specific row from the `users` table where the primary key is `5`.
2.  **Level 2 — The Bytecode Execution (VM)**: The VM opens a **Cursor** on the `users` B-tree, executes a `Seek` to find RowID `5`, and extracts the `name` column into a register.
3.  **Level 3 — The Physical Storage (B-tree/Pager)**: The B-tree layer traverses internal pages, uses the Buffer Pool to load Page #12, finds the record in the slotted page's cell content area, and returns the raw bytes.

---

### The Soul of Iteration: The Cursor Pattern

In database internals, a **Cursor** is the primary abstraction for record navigation. 

> **🔑 Foundation: The Cursor Pattern**
>
> ### 1. What it IS
> A Cursor is a stateful object that represents a "position" within a database structure (like a Table B-tree or an Index). It keeps track of which page it is currently on and which "slot" (index in the cell pointer array) it is pointing to. 
>
> ### 2. WHY you need it right now
> The Virtual Machine (VM) should not need to know the complexities of B-tree page splitting or depth. By using a Cursor, the VM can simply say `Next()` or `Prev()`. The Cursor handles the heavy lifting: if it reaches the end of Page #5, it knows to look at the "Right Child" or "Parent" to find Page #6.
>
> ### 3. Key Insight: Decoupling Logical and Physical
> **The Cursor is a "Bookmark" for the VM.** Without it, every opcode would have to re-calculate the path from the root of the B-tree to the data. With a Cursor, the database maintains a "hot" path to the data, significantly reducing the CPU cost of sequential scans.

When you implement the `OpenRead` or `OpenWrite` opcodes, you are instantiating a Cursor. This cursor must be stored in the VM's `cursors[]` array, allowing subsequent opcodes (`Column`, `Next`, `Insert`) to reference it by index.

---

### Mechanism 1: SELECT and Projection

A `SELECT` statement involves two main phases: **Scanning** and **Projection**.

#### 1. The Scan (`Rewind` & `Next`)
To execute `SELECT * FROM users`, the compiler generates a `Rewind` opcode followed by a loop containing `Next`.
- `Rewind`: Tells the cursor to find the "leftmost" leaf in the B-tree. If the table is empty, it jumps to the end of the program.
- `Next`: Tells the cursor to move to the next slot in the current page. If it's at the last slot, it uses the B-tree's sibling pointers (or parent traversal) to move to the next leaf page.

#### 2. The Projection (`Column` & Record Deserialization)
The `Column` opcode is responsible for **Projection**—extracting specific data from a raw B-tree cell. 

Recall that a B-tree cell contains a **Record**. This record is a serialized sequence of values. To extract "Column 2", you cannot just jump to a fixed offset because columns like `TEXT` or `BLOB` are variable-length. 

**The Record Format Header:**
SQLite-style records start with a **Header Size** (varint) followed by a series of **Serial Types** (one varint per column).
- If Serial Type is `1`, the column is an 8-bit integer.
- If Serial Type is `13`, the column is a string of length `(13-13)/2 = 0`? No, the formula is usually `(N-12)/2` for strings.
- You must parse this header to find the byte offset of the specific column the VM is asking for.

---

### The Revelation Arc: The Update Paradox

Most developers approaching database design have a simple mental model for `UPDATE`: "Just find the bytes on the disk and overwrite them with the new values."

**The Scenario:**
You have a row: `(ID: 1, Name: "Bob", Bio: "Short bio")`.
The user executes: `UPDATE users SET Bio = "[A 500-page novel...]" WHERE ID = 1;`

**The Problem:**
The original row occupied 50 bytes. The new row occupies 50,000 bytes. It **physically cannot fit** in the same slot in the slotted page. It might not even fit on the same **page**. 

**The Reveal:**
In a B-tree storage engine, an `UPDATE` is almost never a simple overwrite. It is actually a **DELETE** followed by an **INSERT**. 
1. The engine deletes the old, small record.
2. The engine attempts to insert the new, large record.
3. This insertion triggers a **Node Split** (Milestone 5) because the page is now over-capacity.
4. The B-tree's structure changes entirely.

**Key Insight:** Because updates change row sizes, they can cause "Write Amplification." A single-row update might force the database to re-balance three levels of the B-tree and write 5+ new pages to disk. This is why "Primary Keys" (RowIDs) are usually immutable—changing the key would require moving the row to a completely different part of the tree.

---

### Mechanism 2: WHERE and Three-Valued Logic (3VL)

When the VM evaluates a `WHERE` clause, it uses comparison opcodes (`Eq`, `Gt`, `Ne`). These opcodes must respect SQL's **Three-Valued Logic**.

In most programming languages, a boolean is `TRUE` or `FALSE`. In SQL, it is `TRUE`, `FALSE`, or `NULL` (Unknown). 

> 🔭 **Deep Dive**: **3VL Comparison Rules**. 
> - `5 = 5` is `TRUE`.
> - `5 = 6` is `FALSE`.
> - `5 = NULL` is `NULL`.
> - `NULL = NULL` is `NULL`.
> 
> For a full matrix of logical operations (AND/OR/NOT) in 3VL, refer to **ISO/IEC 9075 (SQL Standard)**. The critical takeaway for your VM: a `WHERE` clause only includes a row if the expression evaluates to **exactly TRUE**. If it evaluates to `FALSE` or `NULL`, the row is skipped.

Your comparison opcodes must check the "Serial Type" of the values in the registers. Comparing an `INTEGER` to a `TEXT` string requires **Type Affinity** rules (e.g., converting the string "123" to the number 123 before comparing).

---

### Mechanism 3: INSERT and Auto-increment

When executing an `INSERT`, your VM must handle the `rowid`. 
- If the user provides an ID (`INSERT INTO t(id, val) VALUES(10, 'hi')`), the B-tree attempts to insert that specific key. If it exists, it must return a `UNIQUE CONSTRAINT` error.
- If the user provides `NULL` for an `INTEGER PRIMARY KEY`, or omits it, you must implement **Auto-increment**.

**The Max-RowID Optimization:**
To find the next ID, don't scan the whole table. Since the Table B-tree is keyed by `rowid`, the largest ID is always in the **rightmost** leaf. Your engine should maintain a `last_inserted_rowid` in the table's metadata or perform a quick "Seek to End" to find the current maximum.

---

### The "Sawing Off the Branch" Problem (DELETE)

There is a subtle but deadly bug waiting in your `DELETE` implementation. Consider:
`DELETE FROM users WHERE age > 20;`

The VM loop looks like this:
1. `Rewind` cursor.
2. Check `age`. If > 20, call `Delete`.
3. Call `Next` on the cursor.

**The Trap:** When you call `Delete` on a B-tree, the page might be re-balanced or merged with a neighbor. The cursor, which was pointing to "Slot 4 of Page #10," is now pointing at garbage or the wrong row. Calling `Next` after a `Delete` often leads to skipped rows or crashes.

**The Solution: Two-Pass Delete.**
1. **Pass 1 (Collect)**: Scan the table and collect the `rowid` of every matching row into a temporary list (or a VM register array).
2. **Pass 2 (Execute)**: Iterate through the collected IDs and perform a point-delete for each one. This ensures that B-tree mutations don't corrupt the "Scanning Cursor."

---

### Constraint Enforcement: NOT NULL

Before the `Insert` or `Update` opcodes commit data to the B-tree, they must validate **Constraints**. 

The `NOT NULL` constraint is checked during the **MakeRecord** phase. As the VM assembles the binary payload from registers, it checks the table's schema (stored in your System Catalog). If a column is marked `NOT NULL` but the register contains a `NULL` value, the VM must abort the transaction and return a descriptive error: `Error: NOT NULL constraint failed: users.name`.

---

### Design Decisions: Write Amplification

| Feature | Impact | Trade-off |
|--------|------|---------|
| **In-Place Update** | High Performance | Only works if the new data is exactly the same size as the old data. |
| **Delete + Insert (Chosen ✓)** | High Reliability | Causes **Write Amplification**. Updating 1 byte might cause a 4KB page write. |

**Why we choose Delete + Insert**: It is the only way to support variable-length columns (`VARCHAR`, `TEXT`) and maintain B-tree balance. While it increases disk I/O, it simplifies the storage engine logic immensely and prevents "shattered" records that span across non-contiguous fragments.

---

### Implementation Roadmap

1.  **Table Validation**: In the `OpenRead/Write` opcodes, check the System Catalog. If the table name doesn't exist, throw an error.
2.  **The Record Decoder**: Implement a utility that takes a raw byte buffer and a column index, and returns a `Value` object (Integer, String, etc.) by parsing the Record Header.
3.  **The Cursor Wrapper**: Enhance your B-tree `find()` and `insert()` methods to work with a `Cursor` object that tracks `currentPage` and `currentSlot`.
4.  **The Delete Opcode**: Implement the two-pass logic or a "Cursor Revalidation" step to prevent corruption during scans.
5.  **3VL Logic**: Update your comparison opcodes (`Eq`, `Lt`, etc.) to handle `NULL` correctly according to SQL standards.

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **The Cursor Pattern**: This is identical to **Iterators** in C++, **Generators** in Python, or **Streams** in Java. You are learning how to process data that is too large to fit in memory by using a "window" (the cursor).
2.  **Write Amplification**: This concept is vital for **SSD Engineering**. SSDs cannot overwrite individual bytes; they must erase entire blocks (e.g., 256KB) to change one bit. Your B-tree's page-level updates mirror the physical reality of modern flash memory.
3.  **Three-Valued Logic (3VL)**: This is a form of **Kleene Logic**. Understanding how "Unknown" propagates through expressions is the foundation of modern data science and missing-data handling.
4.  **The Record Format**: The way you pack metadata (headers) before data is the basis of **Protobuf**, **Avro**, and even **TCP Packet Headers**. You are learning to design efficient binary protocols.

### System Awareness
In the context of the whole engine, Milestone 6 is the **Integration Point**. You are finally using the `Pager` (M4), the `B-tree` (M5), and the `VM` (M3) together. If the Pager doesn't flush dirty pages, your `INSERT`s will vanish. If your `B-tree` split logic is buggy, your `SELECT` will find only half the rows. This is where you prove that your architectural layers are truly solid.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m7 -->
# Milestone 7: Secondary Indexes

You have a functioning database. You can create tables, insert rows, and retrieve them. But if you have a million rows and you want to find the user with the email `dev@null.com`, your engine currently has to perform a **Full Table Scan**. It must load every single page from disk, deserialize every row, and check the email column. Even with an efficient Buffer Pool, this is $O(N)$—a linear crawl that kills performance as your data grows.

In this milestone, you will implement **Secondary Indexes** using **B+trees**. You will build the "Search Engine" of your database, allowing the Virtual Machine to jump directly to the data it needs in $O(\log N)$ time. You will implement index maintenance (the "write tax"), range scans, and the critical "Double Lookup" pattern.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


---

### The Fundamental Tension: The "Index Tax"

The tension in indexing is **Read Speed vs. Write Latency & Storage Space.**

Every index you add to a table is a secondary, physical structure that must be stored on disk and kept in perfect synchronization with the main table. There is no such thing as a "free" index:
1.  **Storage Cost**: An index on a `name` column requires storing every name a second time, plus the overhead of B-tree internal pages.
2.  **Write Penalty**: When you `INSERT` a row, you no longer just write to the Table B-tree. You must also navigate and insert into *every* associated Index B+tree. This turns a single write into multiple writes, increasing the risk of fragmentation and I/O wait.
3.  **The "Stale Data" Risk**: If a bug in your engine updates the Table but fails to update the Index, your queries will return "ghost" rows or miss existing ones.

The engineering challenge is to make the retrieval so much faster ($O(\log N)$ vs $O(N)$) that the user is willing to pay the "tax" on every write.

---

### The Revelation Arc: The "Thin" Tree vs. The "Thick" Tree

#### The Misconception
Many developers beginning their journey into database internals assume that an index is just a "fast copy" of the table. They imagine that if they create an index on `email`, the index contains the full row data sorted by email so the engine can just read it and be done.

#### The Scenario
Imagine a table `Users` with columns: `ID, Email, Bio, ProfilePicture (BLOB)`. 
The `ProfilePicture` is 100KB. If you have 10,000 users, and you create an index on `Email`, should the index also store the 100KB `ProfilePicture`? 

#### The Reveal
If the index stored the full row, it would be as large as the table itself. Your Buffer Pool would be exhausted by duplicate data. Instead, indexes are **"Thin."** 

An index entry (a "Cell" in the Index B+tree) contains only two things:
1.  **The Key**: The value being indexed (e.g., `dev@null.com`).
2.  **The Pointer**: The `RowID` of the corresponding row in the Table B-tree.

**Key Insight**: An index does not usually give you the answer; it gives you the **Address** of the answer. Finding a row via an index is a two-stage journey called the **Double Lookup**. 

---

### Mechanism: The Double-Lookup Walk

When the VDBE executes a query like `SELECT * FROM users WHERE email = 'dev@null.com'`, and an index exists on `email`, it follows this path:

1.  **Search the Index B+tree**: Use the B+tree search algorithm to find the leaf page containing `'dev@null.com'`.
2.  **Extract the RowID**: The leaf cell for `'dev@null.com'` contains the value `42`.
3.  **Search the Table B-tree**: The VM then takes that `RowID: 42` and performs a second, high-speed lookup in the main Table B-tree. 
4.  **Retrieve the Row**: The Table B-tree provides the actual row containing the `Bio` and `ProfilePicture`.


![The Double-Lookup Walk](./diagrams/diag-index-lookup.svg)


---

### B-tree vs. B+tree: Why the "+" Matters

In Milestone 5, you built a B-tree for table storage. For indexes, we specifically use the **B+tree** variant. 

> **🔑 Foundation: The B+tree Variant**
>
> ### 1. What it IS
> A B+tree is a variation of the B-tree where:
> 1.  **Data only lives in Leaves**: Internal nodes *only* store keys for navigation.
> 2.  **Linked Leaves**: Every leaf page contains a pointer to the "Next" leaf page in sorted order.
>
> ### 2. WHY you need it right now
> Databases love **Range Scans** (e.g., `WHERE price BETWEEN 10 AND 50`). In a standard B-tree, performing a range scan requires "backtracking" up to the parent and down to the next child repeatedly. In a B+tree, you find the start of the range (`10`) and then simply follow the "Next Page" pointers horizontally across the leaves until you hit `50`.
>
> ### 3. Key Insight
> **B+trees turn Random I/O into Sequential I/O.** By linking the leaves, the B+tree allows the engine to treat the bottom layer of the tree as a sorted, linked list, which is significantly faster for modern disk controllers to read.

---

### Implementation: The Index Cell Format

An Index B+tree page uses the same **Slotted Page Architecture** as your tables, but the **Cells** (the data units) are formatted differently. 

An Index Cell must store a composite value. Even if you are only indexing one column, the cell actually stores a tuple: `(IndexValue, RowID)`.

**Why include the RowID in the key?**
In SQL, you can have multiple rows with the same value (e.g., ten users named "Smith"). If the index only stored the value "Smith," it wouldn't know which `RowID` to return. By storing `(Smith, 10)` and `(Smith, 12)`, every entry in the index becomes unique and sorted.

**Handling NULLs**: 
In SQL, `NULL` is usually treated as the "smallest" possible value for sorting purposes. Your index comparison logic must handle `NULL` values consistently so that `WHERE col IS NULL` can also use the index.

---

### Index Maintenance: The Synchronous Tax

You must modify your `INSERT`, `UPDATE`, and `DELETE` execution logic to maintain all associated indexes. This logic must be **Synchronous**—the user must not see the `INSERT` as "complete" until the indexes are updated.

#### 1. INSERT Hook
1.  Insert the row into the Table B-tree to get the `RowID`.
2.  For every index on that table:
    *   Extract the value of the indexed column(s) from the new row.
    *   Construct an Index Cell: `(Value, RowID)`.
    *   Insert into the Index B+tree.

#### 2. DELETE Hook
1.  Read the row from the Table B-tree *before* deleting it to get the current column values.
2.  For every index:
    *   Construct the Index Cell `(OldValue, RowID)`.
    *   Perform a point-delete in the Index B+tree.
3.  Delete the row from the Table B-tree.

#### 3. UPDATE Hook
An update is the most expensive operation. If the indexed column changes, you must perform an **Index Delete** of the old value and an **Index Insert** of the new value. 

---

### The Unique Constraint: Free Enforcement

A `UNIQUE` index is a special type of index that performs a "Double Duty." In addition to speeding up reads, it enforces business logic.

When you attempt to insert `(Value: 'Smith', RowID: 15)` into a `UNIQUE` index:
1.  The B+tree performs a search for `Value: 'Smith'`.
2.  If it finds *any* entry with the value `'Smith'`, the insertion fails with a `UNIQUE CONSTRAINT VIOLATION`.
3.  Because the search happens in $O(\log N)$, enforcing uniqueness is extremely cheap. Without an index, checking for uniqueness would require a full table scan on every single insert!

---

### VDBE Integration: New Opcodes

To use the index, your Bytecode Compiler needs new instructions that understand how to navigate "Thin" trees.

| Opcode | Parameters | Logic |
|--------|------------|-------|
| **`IdxGE`** | `Cursor, Label, Reg` | **Index Greater-Equal**: Seek the index cursor to the first entry $\ge$ value in `Reg`. If not found, jump to `Label`. |
| **`IdxRowid`** | `IdxCursor, OutReg` | Extract the `RowID` from the current index cell and put it in `OutReg`. |
| **`SeekRowid`** | `TableCursor, RowidReg`| Move the table cursor to the row identified by the ID in `RowidReg`. |

**The VDBE Loop for an Indexed Query:**
1.  `IdxGE(idx_cursor, end_label, target_value)`
2.  `LoopStart:`
3.  `IdxRowid(idx_cursor, r1)`
4.  `SeekRowid(table_cursor, r1)`
5.  `Column(table_cursor, ...)`
6.  `ResultRow`
7.  `Next(idx_cursor, LoopStart)`
8.  `Halt`

---

### Optimization: The "Covering Index"

There is one exception to the "Double Lookup" rule. If your query only asks for columns that are already present inside the index, the VDBE can skip the second lookup entirely.

**The Scenario**:
`CREATE INDEX idx_name ON users(name);`
`SELECT name FROM users WHERE name = 'Alice';`

Since the index already contains the `name` "Alice", the engine has all the information it needs. It can emit the `ResultRow` using the data inside the index cell. This is a **Covering Index**, and it is the fastest possible way to retrieve data in a relational engine.

---

### Design Decisions: B+Tree vs. Hash Index

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **B+Tree (Chosen ✓)** | Supports range scans (`<`, `>`, `BETWEEN`). Stays balanced naturally. | Slightly slower equality lookups ($O(\log N)$). | SQLite, Postgres, MySQL |
| Hash Index | Perfect $O(1)$ equality lookups. | No range scans. Hash collisions can degrade to $O(N)$. | Memory engines, Redis |

**Why we choose B+Tree**: In SQL, users constantly perform range queries and `ORDER BY` operations. A B+tree provides sorted results "for free" because of its internal structure. A Hash Index would require a full sort of the results every time.

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Index Maintenance and LSM-Trees**: You've seen that updating B-tree indexes is expensive because of random writes. **Log-Structured Merge-Trees (LSM)** (used in RocksDB or Cassandra) solve this by buffering index updates in memory and writing them sequentially.
2.  **Search Engines**: The "Inverted Index" used by systems like **Elasticsearch** is essentially a "Thin" index where the Key is a word and the "RowID" is a list of Document IDs. The fundamental concept is the same: mapping a value to its location.
3.  **Composite Indexing**: Now that you understand the tuple `(Value, RowID)`, you can see how multi-column indexes work: `(Col1, Col2, RowID)`. This explains the "Leftmost Prefix" rule: an index on `(FirstName, LastName)` can help you find "John", but it cannot help you find everyone with the last name "Smith" without a full scan.
4.  **Covering Indexes**: This is a core trick in **Database Tuning**. By adding an extra column to an index (even if you don't search by it), you can "cover" the query and eliminate the expensive disk seek of the Double Lookup.

### System Awareness
In the System Map, the Secondary Index is a "Satellite" structure. Upstream, the **Query Planner** (Milestone 8) must now decide: "Is it cheaper to do a Full Table Scan or an Index Scan?" Downstream, the Index uses the same **Buffer Pool** and **Pager** as the table. If your Buffer Pool is too small to hold both the Index and the Table's "hot" pages, you will experience **Cache Thrashing**, where the system spends all its time swapping pages in and out.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m8 -->
# Milestone 8: The Query Planner & Statistics

In the previous milestones, you built a powerful engine capable of navigating B-trees and executing complex bytecode. But your engine is currently "blind." When it receives a query like `SELECT * FROM users WHERE age > 25 AND status = 'active'`, it has no idea whether it should scan the whole table or use an index on `age`. If you have an index on `status` as well, it doesn't know which one will filter out more rows.

In this milestone, you will build the **Query Planner**. This is the "brain" of the database that performs **Cost-Based Optimization (CBO)**. You will implement the `ANALYZE` command to gather metadata about your data, build a mathematical model to predict I/O costs, and write the logic that chooses the most efficient execution path.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


---

### The Fundamental Tension: The Random I/O Penalty

The tension at the heart of query planning is **Sequential I/O vs. Random I/O.**

In Milestone 7, we celebrated the B+tree index for allowing us to jump directly to a row. However, "jumping" has a hidden cost. When you perform a **Full Table Scan**, the Pager reads pages sequentially. Modern OS kernels and disk controllers are optimized for this; they "read ahead," pulling the next 10 pages into memory before you even ask for them. Sequential reading is like a high-speed train on a straight track.

When you use an **Index Scan** (the "Double Lookup"), you are performing Random I/O.
1. Read an Index Page.
2. Jump to a Table Page (possibly far away on disk).
3. Jump back to the next Index Page.
4. Jump to a different Table Page.

Every "jump" risks a **Buffer Pool Miss**. If the table is large, these jumps force the disk head (in HDDs) to move or the flash controller (in SSDs) to open new blocks. On mechanical disks, a random seek is **100x to 1000x slower** than a sequential read. Even on NVMe SSDs, the overhead of random access is significant due to the loss of read-ahead optimizations.

The Planner’s job is to answer one question: **"Is the number of rows I'm skipping worth the cost of the random jumps I'm adding?"**

---

### The Revelation Arc: The Index Trap

#### The Misconception
As a developer, your instinct is likely: *"If an index exists on a column in the WHERE clause, I should always use it. Why would I ever scan the whole table if I have a shortcut?"*

#### The Scenario
Imagine a table `Votes` with 1,000,000 rows. You have an index on the column `voted_at`. 
You run the query: `SELECT * FROM Votes WHERE voted_at > '2000-01-01'`.
It turns out that 950,000 people have voted since the year 2000. 

#### The Reveal
If you use the index, you will perform **950,000 random lookups** into the main table. You will visit almost every page in the database, but you will do it in a fragmented, chaotic order, jumping back and forth. 

If you just perform a **Full Table Scan**, you read every page exactly once, in order, using high-speed sequential I/O. 

**The Reveal:** For queries with **low selectivity** (queries that return a large percentage of the table), a Full Table Scan is significantly faster than an Index Scan. The Query Planner's primary duty is to detect these "low selectivity" traps and force the engine to stay on the "straight track" of a sequential scan.

---

### The Three-Level View: The Planner's Eye

The Query Planner operates as a translation layer between the user's intent and the VM's execution.

1.  **Level 1 — The Logical Request (AST)**: "I want all users where age is 30."
2.  **Level 2 — The Candidate Plans (Optimization)**: The Planner looks at the schema. 
    *   Plan A: Full Table Scan + Filter by age. 
    *   Plan B: Index Scan on `idx_age` + Double Lookup.
3.  **Level 3 — The Physical Execution (Bytecode)**: The Planner calculates that Plan B is 50x cheaper. It tells the Compiler to emit `IdxGE` and `SeekRowid` opcodes instead of a `Rewind` loop.

---

### Phase 1: The `ANALYZE` Command (Statistics Collection)

To make a "Cost-Based" decision, the Planner needs data about the data. It needs **Statistics**. 

You must implement the `ANALYZE` command. When run, it performs a full scan of the database and populates a system table (e.g., `sqlite_stat1`). 

> **🔑 Foundation: Selectivity and Cardinality**
>
> ### 1. What it IS
> **Cardinality** is the number of *distinct* values in a column. If a `gender` column has values "Male", "Female", and "Other", its cardinality is 3, regardless of whether the table has 10 rows or 10 million.
> **Selectivity** is the "filtering power" of a value. It is calculated as $\frac{1}{\text{Cardinality}}$. A high-cardinality column (like `email`) has high selectivity (it filters out almost everything). A low-cardinality column (like `is_active`) has low selectivity.
>
> ### 2. WHY you need it right now
> If you query `WHERE email = 'a@b.com'`, the Planner knows (via cardinality stats) that this will likely return 1 row. Using an index is a no-brainer. If you query `WHERE is_active = 1`, the Planner knows this might return 50% of the table. It might choose to ignore the index entirely.
>
> ### 3. Key Insight
> **Statistics are a "Snapshot in Time."** The database does not update these stats on every `INSERT` because it would be too slow. This is why databases sometimes "get slow" over time until you manually run `ANALYZE` to refresh the Planner's view of the world.

**What `ANALYZE` should collect:**
- **Table Level**: Total number of rows (`nRow`).
- **Index Level**: For each column in the index, the average number of rows that share the same value. (e.g., "In the `last_name` index, there are an average of 15 rows per name").

---

### Phase 2: The Cost Model

Now you must define the "Math of Performance." A cost model assigns a numerical value (usually representing I/O operations) to a plan.

$$ \text{Total Cost} = (\text{CPU Cost}) + (\text{I/O Cost}) $$

In our engine, I/O is the dominant factor. Let's build a simplified model:

#### 1. Full Table Scan Cost
$$ \text{Cost}_{scan} = \text{TotalPagesInTable} $$
*Intuition: We have to touch every page once.*

#### 2. Index Scan Cost
$$ \text{Cost}_{index} = \text{PagesInIndex} + (\text{EstimatedMatchingRows} \times \text{RandomIOFactor}) $$

- **EstimatedMatchingRows**: Calculated using your stats. If you have 1,000 rows and the column has a cardinality of 10, you estimate 100 matching rows for an equality predicate (`col = ?`).
- **RandomIOFactor**: This is a "weight" you assign to the penalty of a random seek. In your implementation, start with a value of **4.0**. This means one random jump is considered 4x more expensive than one sequential page read.

**The Threshold**:
Your Planner should compare the costs. If $\text{Cost}_{index} < \text{Cost}_{scan}$, use the index. 
*Note: Most engines find that the "tipping point" is around 20-30%. If a query matches more than 20% of the table, the index is usually slower than a scan.*

---

### Phase 3: Choosing the "Access Path"

When the Compiler receives an AST for a `SELECT` statement, it now calls the Planner before emitting opcodes.

**The Planning Logic:**
1.  Identify all columns in the `WHERE` clause.
2.  Find all indexes that involve those columns.
3.  For each candidate index:
    *   Estimate how many rows will be returned based on the operator.
    *   `=` (Equality): $\text{TotalRows} / \text{Cardinality}$.
    *   `<` or `>` (Range): Usually estimated at **33%** of the table if no better stats exist (this is a **Heuristic**).
    *   `BETWEEN`: Usually estimated at **10%**.
4.  Calculate the cost of using that index vs. the cost of a full scan.
5.  Select the plan with the lowest cost.

---

### Phase 4: Join Optimization (The N-Body Problem)

This is the most complex part of query planning. When a user joins three tables (`A JOIN B JOIN C`), there are $3! = 6$ possible orders to join them.
1. $(A \bowtie B) \bowtie C$
2. $(B \bowtie A) \bowtie C$
3. $(C \bowtie B) \bowtie A$
...and so on.


> **🔑 Foundation: Why Join Order Matters: Joining a 10-row table to a 1-million-row table is much faster if you start with the 10-row table and use an index to find the matching millionth-row entries**
> 
> **1. What it IS**
Join order complexity refers to the mathematical explosion of possible sequences in which a database can combine multiple tables to satisfy a query. While the final result of a join is the same regardless of the order (e.g., Joining Table A to B is logically the same as B to A), the physical path the database takes to get there—the "execution plan"—determines the performance. In a query with $N$ tables, the number of possible join orders grows factorially ($N!$), meaning for a 10-table join, there are over 3.6 million possible ways to execute the query.

**2. WHY you need it right now**
In this project, you are dealing with asymmetrical data sizes (e.g., a small `Users` table and a massive `Logs` table). If the database engine chooses an inefficient join order, it might attempt to load the massive table into memory or perform a full table scan before applying filters. Understanding join order complexity allows you to diagnose "slow queries" that look fine on paper but are failing because the Query Optimizer is overwhelmed by the search space or lacks the statistics to pick the cheapest path.

**3. Key Insight: The "Funnel" Principle**
Think of a join sequence as a funnel. Your goal is to **discard data as early as possible.** If you join two small tables first to create a tiny intermediate result, the subsequent join against a million-row table becomes a series of quick, indexed lookups. If you join the large tables first, you are forcing the database to carry a massive "heavy" dataset through every subsequent step of the process.


#### The Nested Loop Join Strategy
For this milestone, we use the **Nested Loop Join** (the only join algorithm you've implemented so far). 


![Nested Loop Join Trace](./diagrams/diag-nested-loop-join.svg)


**The Planner's Goal for Joins:**
Find the "Outer Loop" that is the smallest, and an "Inner Loop" that has an index on the join column.

**Estimation (Cardinality of a Join):**
If you join `Users` and `Orders` on `Users.id = Orders.user_id`:
$$ \text{EstimatedRows} = \frac{\text{Rows(Users)} \times \text{Rows(Orders)}}{\max(\text{Cardinality(Users.id)}, \text{Cardinality(Orders.user_id)})} $$

Your Planner should iterate through possible join orders (for small numbers of tables, you can just try them all) and pick the one with the lowest total estimated cost.

---

### Phase 5: Integrating with `EXPLAIN`

The `EXPLAIN` command you built in Milestone 3 must now be upgraded. Instead of just showing opcodes, it should first show the **Plan Description**. 

When a user runs `EXPLAIN QUERY PLAN SELECT...`, your engine should output a human-readable summary:
- `SCAN TABLE users` (If it chose a full scan)
- `SEARCH TABLE users USING INDEX idx_age (age>?)` (If it chose an index)
- `NESTED LOOP JOIN` (For joins)

This allows you to verify that your Planner is actually working. If you add 1,000,000 rows and the `EXPLAIN` still says `SCAN TABLE`, you know your cost model or your statistics are broken.

---

### Design Decisions: Heuristics vs. Statistics

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Heuristics (Rules of Thumb)** | Instant. No need for `ANALYZE`. Works even on empty tables. | Often wrong. Doesn't know if a column is "skewed" (e.g., 99% of users are from one country). | MySQL (Early versions), SQLite (Simple queries) |
| **Statistics (Chosen ✓)** | High precision. Adapts to the actual data distribution. | Requires the `ANALYZE` tax. Stats can become stale. | SQLite, Postgres, SQL Server |

**Why we choose Statistics**: An Expert-level database must handle data "skew." If your `status` column has two values, "active" (99%) and "banned" (1%), a heuristic would say selectivity is 50% for both. Statistics tell the Planner: "Use the index for 'banned', but scan the table for 'active'."

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Cost Modeling in Distributed Systems**: The same logic is used in **Apache Spark** or **Presto**. The "Cost" there isn't just disk I/O; it's **Network Shuffles**. The Planner tries to minimize moving data between servers.
2.  **Machine Learning for Query Planning**: Modern research (like **PostgreSQL's Learned Optimizer**) uses Neural Networks to predict query costs instead of manual formulas. They "learn" the Random I/O penalty of the specific hardware they are running on.
3.  **The N-Body Problem**: Join ordering is a **Combinatorial Optimization** problem. As the number of tables grows, databases switch from "Try every order" to **Genetic Algorithms** or **Dynamic Programming** to find a "good enough" plan without taking forever to plan it.
4.  **Static Analysis in Compilers**: Your Query Planner is a form of **Profile-Guided Optimization (PGO)**. Just as C++ compilers can use execution traces to optimize code, your DB uses `ANALYZE` to optimize the data path.

### System Awareness
The Query Planner is the **Navigator** of your engine. It sits between the **Parser** (M2) and the **Bytecode Compiler** (M3). It is the only part of the system that "understands" the data distribution. Without it, your B-trees (M5/M7) are like libraries with no index cards—you know the information is there, but you have no efficient way to find it.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m9 -->
<!-- MS_ID: build-sqlite-m9 -->
# Milestone 9: Transactions (Rollback Journal)

Up to this point, your database has been a "fair-weather" friend. If the power stays on and your code never crashes, it works perfectly. But in the world of data storage, we must build for the "storm." A database that cannot survive a sudden power loss or a kernel panic is not a database—it is a temporary buffer with a file extension.

In this milestone, you will implement **ACID Transactions** using a **Rollback Journal**. This is the classic SQLite mechanism for ensuring that either a query happens completely, or it doesn't happen at all. You will move from simple file writes to a choreographed sequence of operations that use the disk's physical properties to guarantee safety.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Fundamental Tension: The "Torn Page" and the Lie of the Cache

The core tension in this milestone is **Speed vs. Absolute Durability.**

When you call `write()` in your code, the Operating System (OS) doesn't actually write to the disk. It copies your data into a kernel buffer (the Page Cache) and tells your program, "Done!" This makes your app feel fast. However, if the power fails five seconds later, that data—which the OS promised was "written"—is gone forever.

Even worse is the **Torn Page** problem. A database page is typically 4096 bytes. Most hardware (especially older HDDs or cheap SSDs) can only guarantee that a 512-byte "sector" is written atomically. If a crash occurs while the drive is halfway through writing your 4KB page, the file will contain 2KB of new data and 2KB of old data. This "Frankenstein Page" is physically corrupt; your B-tree pointers will point to garbage, and your database is effectively destroyed.

The only way to solve this is to accept a performance penalty: we must force the hardware to wait until the data is physically on the platter (or NAND flash) before we proceed. This is the **fsync()** tax.

---

### The Revelation Arc: The Crash Paradox

#### The Misconception
Most developers think about crashes as something that happens *between* operations. You think: "I'll write the data, and if I crash *after* that, it’s fine." 

#### The Scenario
You are executing an `UPDATE` that modifies three different pages in your B-tree: a Leaf page, an Internal node, and the Root. You call `write()` for Page A, then Page B, then Page C. 

#### The Reveal
**Computers don't just crash between operations; they crash *during* them.** 

If the system fails while writing Page B, you now have:
- Page A: New version.
- Page B: Half-new, half-old (Torn).
- Page C: Old version.

Your B-tree is now logically inconsistent. Page A might point to a record that Page B doesn't know about. There is no "Undo" button on a raw file. 

**The Reveal:** Durability is not about writing data; it's about **Write Ordering**. To survive a crash, you must follow a protocol where you save the "Original State" to a safe place **before** you touch the main file. If you crash during the write, you can use that "Original State" to repair the damage.

---

### The Three-Level View: The Transaction Stack

To understand how a transaction works, look at how the `COMMIT` command traverses the stack:

1.  **Level 1 — The API (SQL)**: The user executes `BEGIN TRANSACTION` and then `COMMIT`. This sets the logical boundaries of the work.
2.  **Level 2 — The Transaction Manager (VDBE/Pager)**: This layer creates the `.db-journal` file. It intercepts all "Dirty Page" requests from the Buffer Pool and copies the *original* bytes into the journal before allowing the Buffer Pool to modify them in RAM.
3.  **Level 3 — The OS/Disk (I/O)**: This layer uses `fsync()` to create **I/O Barriers**. It ensures the journal is "Hardened" to the disk before the main database file is touched.

---

### Mechanism: The Atomic Commit Choreography

SQLite’s Rollback Journal uses a specific "dance" to ensure atomicity. You will implement this sequence in your Pager.


![Atomic Write Choreography](./diagrams/diag-journal-flow.svg)


#### Step 1: Preparation (The Journal)
When the first write occurs after a `BEGIN`:
1.  Open a new file named `[your-db-name]-journal`.
2.  **The Header**: Write a header to the journal containing the original size of the database file.
3.  **The Undo Log**: For every page the VDBE wants to change, read the *original* page from the `.db` file and write it to the journal.

#### Step 2: The First Barrier (fsync)
Before you write a single modified byte to the main `.db` file, you must call `fsync()` on the journal file. 
- **Why?** If you crash during Step 3, the journal *must* be complete and valid on the disk so the recovery logic can find it. If the journal is only in the OS cache and not on disk, it's useless.

#### Step 3: The Update
Now, and only now, write the "Dirty" (modified) pages from your Buffer Pool into the main `.db` file. 

#### Step 4: The Second Barrier (fsync)
Call `fsync()` on the main `.db` file. This ensures all the new data is physically safe.

#### Step 5: The Commit
Delete the journal file (or truncate it to zero bytes). 
- **The Magic Moment**: The exact microsecond the journal is deleted is the "Commit Point." If you crash a microsecond before this, the database will roll back. If you crash a microsecond after, the changes are permanent.

---

### The Concept: ACID Properties

By implementing this choreography, you are fulfilling the **ACID** contract.

> **🔑 Foundation: ACID**
> 
> ### 1. What it IS
ACID is an acronym representing the four primary requirements of a reliable database transaction:
- **Atomicity**: The "All or Nothing" rule. If a transaction has 10 steps and fails on step 9, the first 8 steps are undone.
- **Consistency**: The database moves from one valid state to another, never violating constraints (like `NOT NULL`).
- **Isolation**: Concurrent transactions don't interfere with each other. One person's "half-finished" work isn't visible to another.
- **Durability**: Once a transaction is committed, it remains committed even if the power fails or the OS crashes.

### 2. WHY you need it right now
Without ACID, you cannot build financial systems, inventory trackers, or even a simple blog. If two users buy the last item in a store at the same time, or if the server reboots during a checkout, ACID is what prevents the store from losing money or corrupting its inventory.

### 3. Key Insight: Durability is the most expensive
Of the four, **Durability** is the one that slows your database down the most. It requires `fsync()`, which forces the CPU to stop and wait for the physical disk (thousands of times slower than RAM). Most "High-Performance" databases that claim to be 100x faster than SQLite are simply lying about Durability—they skip the `fsync()` and hope the power doesn't go out.


---

### Mechanism: The "Hot Journal" and Crash Recovery

How does the database "heal" itself? You must implement **Crash Recovery logic** that runs every time your database opens.

**The Recovery Logic:**
1.  Look for a file named `[db-name]-journal`.
2.  If it doesn't exist, the previous session closed cleanly. Continue as normal.
3.  If it *does* exist, it might be a **Hot Journal**. A journal is "Hot" if:
    - It exists on disk.
    - It is not empty.
    - The process that created it is no longer running (or doesn't hold an exclusive lock).
4.  **The Rollback**:
    - Read the original page images from the journal.
    - Write them back into the main `.db` file at their original offsets.
    - Truncate the `.db` file to the original size stored in the journal header (this removes any pages that were appended during the failed transaction).
    - `fsync()` the `.db` file.
    - Delete the journal.

**Result**: The database is now exactly as it was before the failed transaction started. No "Torn Pages," no partial data.

---

### The Locking Protocol: Managing Concurrency

To prevent one user from reading Page A while another user is halfway through writing the journal, you need a **Locking State Machine**.

SQLite uses five locking states for the database file:

1.  **UNLOCKED**: No one is touching the file.
2.  **SHARED**: Multiple users can read, but no one can write.
3.  **RESERVED**: One user *intends* to write soon. Others can still read, but no other user can start a write.
4.  **PENDING**: A writer is waiting for all current readers to finish so it can take an Exclusive lock. No new readers are allowed.
5.  **EXCLUSIVE**: One writer is modifying the file. No one else can read or write.

**The "Read Isolation" Rule**: In Rollback Journal mode, readers see the "Old" data on the disk because the writer hasn't overwritten it yet. Once the writer starts writing to the `.db` file (Step 3 of the choreography), it must hold an **EXCLUSIVE** lock so that no reader sees a "partial" or "torn" page.

---

### Implementation Detail: The Journal File Format

Don't just dump bytes into the journal. It needs structure so the recovery logic can validate it.

| Field | Size | Description |
|-------|------|-------------|
| **Magic Number** | 8 bytes | A constant (e.g., `0xd9d505f9`) to identify this as a valid journal. |
| **Page Count** | 4 bytes | Number of pages stored in this journal. |
| **Random Nonce** | 4 bytes | Used for calculating checksums. |
| **Original DB Size**| 4 bytes | Size of the DB in pages before the transaction. |
| **Page Record** | Variable | `[Page Number (4b)] + [Original Page Data (4KB)] + [Checksum (4b)]` |

**The Checksum**: For every page in the journal, calculate a simple checksum. During recovery, if a page's checksum doesn't match, it means the crash happened *while* the journal was being written (Step 1). In this case, the journal is not "Hot" and should be ignored, because the main `.db` file hasn't been touched yet!

---

### Design Decisions: Rollback vs. WAL

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Rollback Journal (Chosen ✓)** | Simpler to implement. Single-file database when idle. Better for "Read-Heavy" workloads with rare writes. | Writers block all readers. `fsync()` happens twice per transaction. | SQLite (Legacy default) |
| Write-Ahead Log (WAL) | Readers don't block writers. Faster writes (only one `fsync`). | More complex. Requires a shared-memory file (`-shm`). Database is spread across multiple files. | SQLite (Modern default), Postgres |

**Why we choose Rollback Journal**: It is the best way to learn the fundamentals of **Atomic Commit**. WAL (which you will build in the next milestone) is an optimization, but the Rollback Journal is the "Soul" of durability.

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **I/O Barriers and Fences**: The `fsync()` call is a software-level **I/O Barrier**. In multi-core programming, you use **Memory Barriers** (like `std::atomic_thread_fence` in C++ or `atomic::fence` in Rust) to ensure that one CPU core sees writes from another core in the correct order.
2.  **Self-Healing Systems**: The "Hot Journal" recovery pattern is the basis for **Journaling File Systems** like **ext4** or **NTFS**. When your computer reboots after a crash and says "Fixing C: drive," it is performing a rollback exactly like the one you are building.
3.  **Distributed Consensus**: In distributed systems (like **Raft** or **Paxos**), nodes use a "Log" to agree on state. If a leader crashes, the followers use the log to determine which "Transaction" was actually committed to the majority of the cluster.
4.  **Version Control (Git)**: Git's internal object store is effectively a giant, immutable append-only log. It avoids "Torn Pages" by writing a file to a temporary name and then using an **Atomic Rename** (an OS-level atomic operation) to commit it.

---

### System Awareness

In the architecture of your engine, the Transaction Manager sits inside the **Pager**. It coordinates between the **Buffer Pool** (which holds the "Dirty" pages) and the **VFS (Virtual File System)** (which talks to the disk). 

Without this milestone, your database is a toy. With it, you have built a system that can be trusted with real-world data. You have conquered the "Crash Paradox" and tamed the entropy of the physical disk.
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m10 -->
In Milestone 9, you conquered the "Crash Paradox" using the Rollback Journal. It gave your database a "Soul of Durability." But that durability came with a heavy cost: **Strict Seriality.** In Rollback mode, if a writer is updating the database, no one else can read. If a reader is scanning a table, the writer must wait. This "Stop the World" approach is the death of performance in high-concurrency environments like web servers.

In this milestone, you will implement **Write-Ahead Logging (WAL)**. This is the modern engine of SQLite (and PostgreSQL, and MySQL). You will move from an "Undo" architecture to a "Redo" architecture, enabling **Snapshot Isolation** where readers and writers can operate simultaneously without ever stepping on each other's toes.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


### The Fundamental Tension: The Concurrency Bottlecap

The tension here is **Data Consistency vs. Concurrent Access.**

In a Rollback Journal system, the main database file always represents the "Current Committed Truth." To change it, a writer must overwrite pages in place. To prevent a reader from seeing a "half-written" (torn) page, the writer must kick all readers out of the building (Exclusive Lock). This creates a **Concurrency Bottlecap**: your database's throughput is limited by the fact that only one "type" of operation can happen at a time.

If your database is powering a website with 1,000 users reading posts and 1 user writing a new post, that 1 writer will periodically "freeze" the experience for all 1,000 readers. In modern systems, we demand **Non-blocking Reads**. We want readers to see a consistent version of the world *as it existed when they started*, even if a writer is currently busy carving out a new reality in the background.

---

### The Revelation Arc: The Ghost in the Machine

#### The Misconception
You likely believe that for a reader to get a consistent view of the data, they must look at the actual database file, and therefore, that file cannot be changing while they look at it.

#### The Scenario
Imagine a reader starts a long-running query: `SELECT SUM(balance) FROM accounts`. 
Halfway through the scan, a writer performs a transfer: 
`UPDATE accounts SET balance = balance - 100 WHERE id = 1;` 
`UPDATE accounts SET balance = balance + 100 WHERE id = 999;`

#### The Reveal
In a Rollback system, if the reader processed account #1 *before* the update and account #999 *after* the update, the total sum would be off by $100. The reader saw a "ghost" state that never actually existed. To prevent this, Rollback mode blocks the writer entirely.

**The Reveal:** We don't need to block the writer. Instead of the writer modifying the main file, the writer can just "shout" their changes into a separate log file (the WAL). 
- The **Writer** appends new versions of pages to the end of the WAL file. 
- The **Reader** ignores those new versions for now. They continue reading the "Old Truth" from the main database file. 

Because the main file never changes during the transaction, the reader's view remains perfectly consistent. The writer and reader literally operate on different files, moving the bottleneck from "Locking" to "I/O Management."

---

### The Three-Level View: The WAL Perspective

To understand how the WAL shifts the database's behavior, look at the layers of execution:

1.  **Level 1 — The Logical View (SQL)**: The user sets `PRAGMA journal_mode=WAL;`. Transactions look identical to the user, but `COMMIT` suddenly becomes significantly faster because it no longer requires multiple `fsync()` calls to the main database file.
2.  **Level 2 — The Log Manager (Pager/WAL)**: This layer redirects all writes to a `.db-wal` file. It maintains a **WAL Index** (usually a shared-memory file) to track which page versions are in the log versus the main file. It acts as a router for the Buffer Pool.
3.  **Level 3 — Disk/OS I/O**: Instead of random-writing pages across the database file (which is slow), the system performs a single, high-speed **Sequential Append** to the WAL file.

---

### Mechanism 1: The Redo Log (The WAL File)

In Rollback mode, we recorded "Undo" data (how to go back). In WAL mode, we record **"Redo" data** (how to go forward).

The WAL file is an append-only sequence of **WAL Frames**. Each frame contains:
- **Page Number**: Which page is this?
- **Page Data**: The full 4096 bytes of the new version.
- **Checksums**: Two 32-bit integers to detect corruption.

**The Append Advantage**: Appending to a file is the fastest operation a disk can perform. There is no "seeking" involved. The disk head stays in one place, and the bits flow. This is why WAL mode often results in a 10x-100x improvement in write throughput compared to the Rollback Journal.

---

### Mechanism 2: The WAL Index (`-shm`)

If the writer is putting all the new data into a separate file, how does a reader find it? 

If a reader needs Page #5, they first have to check: "Is there a newer version of Page #5 in the WAL file?" Scanning a 100MB WAL file for every page read would be $O(N)$ and would destroy your performance. 

To solve this, you must implement the **WAL Index** (the `wal-index`). 
- This is a hash table or a compact array stored in a separate file (ending in `-shm`). 
- It maps `Page Number` $\rightarrow$ `Last Frame Index in WAL`.

When the Pager asks for a page:
1.  Check the WAL Index. 
2.  **If Hit**: Page #5 is found at Frame #102 in the WAL. Read it from the WAL file.
3.  **If Miss**: Page #5 is NOT in the WAL Index. Read it from the main `.db` file.

> 🔭 **Deep Dive**: **Shared Memory (`mmap`)**. In production SQLite, the `-shm` file is mapped into the process's memory using `mmap()`. This allows multiple processes to see the same index instantly without performing disk I/O. For your intermediate implementation, you can simulate this with a global data structure or a simple file-backed cache if you are not working in a multi-process environment.

---

### Mechanism 3: Snapshot Isolation

This is the "Brain" of the WAL. How do we ensure a reader doesn't see a write that finished *after* the reader started?

Every time a writer finishes a transaction, it records a **Commit Mark** (a special bit or flag in the WAL frame). The WAL Index tracks these commit marks as "End-of-Transaction" points.

When a **Reader** starts:
1.  It notes the current **WAL Read Mark** (the index of the last committed frame in the WAL at that exact moment).
2.  During its entire lifetime, the reader **only** looks at frames in the WAL that are $\le$ its Read Mark.
3.  Even if a writer appends 1,000 new frames while the reader is working, the reader ignores them. To that reader, the database is "frozen" in time at the moment they started.


![WAL Concurrency: Reader vs Writer](./diagrams/diag-wal-snapshots.svg)


---

### Mechanism 4: The Checkpoint (Merging Reality)

If we keep appending to the WAL, it will eventually fill up the entire disk. We need to move those changes back into the main database file so we can truncate the WAL. This process is called **Checkpointing**.

> 
> **🔑 Foundation: Checkpointing Depth**
> 
> ### What it IS
**Checkpointing depth** is a configuration parameter that determines the granularity at which a system saves its intermediate state during a complex computation. In the context of deep learning, it specifically refers to "gradient checkpointing"—a technique where you decide how many layers of a neural network to skip before saving the mathematical outputs (activations) to memory. 

Instead of storing the activations for every single layer during the forward pass (which is the default), a specific "depth" tells the system to only store a few "anchor" points. If the system needs a missing piece of data later, it simply re-calculates it on the fly starting from the nearest saved anchor.

### WHY you need it right now
You need to manage checkpointing depth when you hit the **Memory Wall**. As models grow larger (e.g., LLMs or high-resolution vision transformers), the activations alone can easily exceed the available VRAM on a GPU, leading to "Out of Memory" (OOM) errors.

By increasing the checkpointing depth (saving less frequently), you drastically reduce the memory footprint of your model. This allows you to:
1. Train larger models on consumer-grade hardware.
2. Use larger batch sizes, which can lead to more stable training.
3. Fit longer sequences (context lengths) into memory.

The trade-off is a roughly 20–30% increase in computation time, as you are trading "compute" to save "space."

### Key Insight: The "Save Game" Analogy
Think of checkpointing depth like **save points in a difficult video game**. 
* **Low Depth (Save often):** You save after every room. If you die, you restart exactly where you were, but your memory card fills up instantly with save files.
* **High Depth (Save rarely):** You only save at the end of each level. You save massive amounts of space on your memory card, but if you fail, you have to replay the entire level to get back to where you were. 

**Checkpointing depth is your strategy for balancing "how much I have to replay" vs. "how much space I have on the card."**

> **Checkpointing** is the process of synchronizing the "Log" (the temporary list of changes) with the "Main Store" (the actual database file). It involves reading the latest version of every page from the WAL and writing it to its proper location in the `.db` file. Without a checkpoint, the WAL grows infinitely. Furthermore, reading from a massive WAL index becomes slower and slower. Checkpointing "cleans the slate," allowing the database to stay compact and fast.

**The Checkpoint Algorithm:**
1.  **Identify the Barrier**: Find the oldest "Read Mark" among all active readers. You can only checkpoint up to this mark.
2.  **The Transfer**: For every page in the WAL (up to the barrier):
    *   Read the latest version from the WAL.
    *   Write it to the corresponding offset in the `.db` file.
3.  **The Flush**: Call `fsync()` on the `.db` file.
4.  **The Reset**: Once all readers have finished using the old WAL frames, you can "Restart" the WAL from byte 0.

*Key Insight*: A checkpoint can only safely move pages that are no longer being looked at by *any* active reader. If a "Zombie Reader" holds onto a snapshot from 10 minutes ago, the WAL cannot be fully checkpointed. This is the primary cause of WAL file bloat.

---

### The Revelation: Redo Logs are the Heart of High Availability

Now that you understand WAL, you understand how modern cloud databases like **Amazon Aurora** or **Google Spanner** work. 

In a traditional database, you replicate the *entire database file* to a backup server. In a modern WAL-based system, you don't send the files; you only stream the **WAL Frames** over the network. The backup server (the "Replica") just keeps "Redoing" the frames it receives from the primary. This is called **Log Shipping**, and it's how we achieve sub-second failover in the cloud. You are building the "Seed" of a distributed system.

---

### Design Decisions: The WAL Frame Checksum

In Milestone 9, we worried about **Torn Pages**. In WAL mode, we handle this using **Cumulative Checksums**.

Every WAL frame contains a checksum that is a function of:
1.  The Page Data.
2.  The Checksum of the **Previous** frame.

This creates a "Chain of Integrity." If a crash occurs and the last frame in the WAL is only partially written (torn), the checksum will fail. But more importantly, because the checksum is cumulative, we can detect if a frame in the middle of the log was corrupted. If a checksum fails, the WAL reader stops immediately and treats the rest of the file as "junk," ensuring that partial or corrupted transactions are never "Redone" into the main database.

---

### Implementation Roadmap

#### 1. The WAL File Structure
Implement a `WalManager` that opens `[db-name]-wal`. Define a `write_frame(page_no, data)` method that appends the header, data, and calculates the cumulative checksum using a robust algorithm like CRC-32 or Fletcher's checksum.

#### 2. The WAL-Index (Search Logic)
Create a lookup table (PageID $\rightarrow$ FrameIndex). 
- *Intermediate Hack*: Rebuild this index from the WAL file every time the database starts by scanning the frame headers. In-memory, store it as a `HashMap<PageID, u32>`.

#### 3. Integrated Pager (The Search)
Update your Pager's `fetch_page(id)` logic to check the WAL before the main file:
```rust
// Conceptual Logic
fn fetch_page(id: PageID) -> Page {
    if let Some(frame_idx) = wal_index.lookup(id, current_read_mark) {
        return read_from_wal(frame_idx); // Redo source
    }
    return read_from_main_db(id); // Main source
}
```

#### 4. The Auto-Checkpoint
In your `COMMIT` logic, check the size of the WAL file. If it exceeds 1,000 pages (approx 4MB), trigger a `wal_checkpoint(PASSIVE)`.
- **PASSIVE**: Checkpoint as many frames as possible without blocking any readers.
- **RESTART**: Block until all readers are gone, then wipe the WAL.

---

### Design Decisions — WAL vs. Rollback Journal

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Rollback Journal** | Simpler. Single-file database when idle. Better for purely read-heavy workloads with very rare writes. | Writers block all readers. `fsync()` happens twice per transaction. | SQLite (Legacy), Simple Embedded DBs |
| **WAL Mode ✓** | **Concurrent Readers & Writers**. High write throughput (Sequential I/O). | Higher complexity. Requires managing the WAL Index. DB is spread across multiple files. | SQLite (Modern), PostgreSQL, MySQL |

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Redo Logs and Replay**: This "Redo" logic is the foundation of **Event Sourcing** in distributed systems. Instead of storing the "Current State" of an object, you store a log of every change and "Replay" it.
2.  **Snapshot Isolation**: You've now implemented a form of **MVCC (Multi-Version Concurrency Control)**. This is exactly how **Git** handles branches—multiple versions of a file coexist, and you "check out" a specific snapshot.
3.  **LSM-Trees (Log-Structured Merge-Trees)**: WAL mode is essentially a tiny, one-level LSM-tree. Systems like **BigTable** and **Cassandra** take this to the extreme, where the *entire database* is just a series of sorted logs.
4.  **CPU Write Buffering**: Your CPU has a "Store Buffer." When you write to memory, the CPU appends the write to a buffer (the WAL) and continues executing. It only "Checkpoints" that buffer into the cache when the bus is free.

### System Awareness
The WAL is the "Concurrency Booster" of your engine. It sits inside the **Pager** layer. When WAL is active, the Pager's role changes from "Disk Coordinator" to "Version Switcher." Without the WAL, your database is robust but serial. With it, your database becomes a concurrent powerhouse, ready to handle the high-traffic demands of a real-world application.

---
<!-- END_MS -->


<!-- MS_ID: build-sqlite-m11 -->
<!-- MS_ID: build-sqlite-m11 -->
# Milestone 11: Aggregate Functions & JOIN

You have built a database that can store, index, and retrieve individual rows with surgical precision. But data is rarely useful in isolation. Real-world questions—"What is the total revenue per region?" or "Show me the names of users who bought this specific item"—require the engine to perform **Relational Algebra**.

In this milestone, you will implement the two most powerful tools in the SQL arsenal: **Aggregate Functions** (COUNT, SUM, AVG, etc.) and **INNER JOINs**. You will move from a single-table execution model to a multi-cursor orchestration model. You will learn how to compute summaries over millions of rows without exhausting your RAM and how to connect disparate tables using the foundational **Nested Loop Join** algorithm.


![The SQLite Satellite Map](./diagrams/diag-l0-map.svg)


---

### The Fundamental Tension: The Multi-Cursor Explosion

The tension in this milestone is **Declarative Simplicity vs. Combinatorial Complexity.**

In a single-table query, the cost of the query is linear: $O(N)$ for a scan or $O(\log N)$ for an index lookup. When you introduce a **JOIN**, you are no longer looking at a list; you are looking at a **Cartesian Product**. If Table A has 1,000 rows and Table B has 1,000 rows, there are 1,000,000 possible combinations of those rows.

As the developer, you must ensure that your Virtual Machine (VDBE) can manage multiple cursors simultaneously without losing its place. If your join logic is inefficient, a simple three-table join can turn a sub-second query into a multi-minute "hang" that saturates the CPU. You are shifting from being a "Librarian" who finds one book to a "Matchmaker" who must find every valid pair in a crowded room.

---

### The Revelation Arc: The Join Misconception

#### The Misconception
If you look at most SQL tutorials online, they explain `JOIN` using **Venn Diagrams**. They show two overlapping circles and say, "The INNER JOIN is the intersection where the circles meet."

#### The Scenario
You are building the execution logic for your database. You have an AST node representing `users JOIN orders ON users.id = orders.user_id`. You look at the Venn diagram. How do you turn that "overlapping circle" into a series of byte-level operations on a B-tree? 

#### The Reveal
**A JOIN is not a mathematical intersection; it is a Nested Loop.**

Computers do not "intersect" tables. They iterate. To a database engine, an `INNER JOIN` is physically a `for` loop inside another `for` loop.
1.  **Outer Loop**: Open a cursor on Table A and move to the first row.
2.  **Inner Loop**: Open a cursor on Table B. For the current row of Table A, scan Table B to find rows where the condition (the "Predicate") is true.
3.  **The Match**: When a match is found, "stitch" the columns of both rows together and emit them as a single `ResultRow`.
4.  **Repeat**: Move Table A to the next row and reset the Table B cursor to the beginning.

**The Reveal**: The Venn diagram is a logical abstraction for the user. The **Nested Loop** is the physical reality of the machine. Once you realize this, you understand why Join Order (Milestone 8) and Secondary Indexes (Milestone 7) are so critical: if the Inner Loop has to do a full scan for every row of the Outer Loop, your database will crawl. If the Inner Loop can use an index to "jump" to the matching row, the $O(N \times M)$ cost collapses back toward $O(N \times \log M)$.

---

### The Three-Level View: Joining the Dots

To understand how a Join executes, look at the layers of the engine:

1.  **Level 1 — The API (SQL)**: `SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id`. The user expresses a relationship between two entities.
2.  **Level 2 — The Execution Engine (VDBE)**: The VM manages two independent B-tree cursors. It uses a "Rewind" on the outer cursor and a "Seek" or "Rewind" on the inner cursor. It tracks the "Nested Loop" state using program counter jumps.
3.  **Level 3 — The Storage Engine (B-tree)**: The cursors perform independent page-loads. One cursor might be at Page 50 of `users.db`, while the other is bouncing between Page 2 and Page 10 of `orders.db`.

---

### Mechanism 1: The Nested Loop Join (NLJ)

You will implement the **Nested Loop Join** as your baseline join algorithm. 


![Nested Loop Join Trace](./diagrams/diag-nested-loop-join.svg)


#### The VDBE Opcodes for Join
To support joins, your compiler must emit a structure like this:

| Addr | Opcode | Parameters | Logic |
|------|--------|------------|-------|
| 0 | `OpenRead` | `0 (users)` | Open outer cursor. |
| 1 | `OpenRead` | `1 (orders)` | Open inner cursor. |
| 2 | `Rewind` | `0, 10` | Start outer loop. Jump to 10 if empty. |
| 3 | `Column` | `0, 0, r1` | Get `users.id` into Register 1. |
| 4 | `SeekGE` | `1, 9, r1` | **Inner Loop Optimization**: Jump to `orders` row matching `r1`. |
| 5 | `Column` | `0, 1, r2` | Get `users.name`. |
| 6 | `Column` | `1, 1, r3` | Get `orders.amount`. |
| 7 | `ResultRow` | `r2, 2` | Output the joined row. |
| 8 | `Next` | `1, 5` | Next row in inner loop (if multiple orders per user). |
| 9 | `Next` | `0, 3` | Next row in outer loop. |
| 10| `Halt` | | Done. |

**Key Insight**: Notice how the `Next` instruction at Address 8 loops back to the *inner* logic, while the `Next` at Address 9 loops back to the *outer* logic. This is the exact structure of a nested `for` loop in bytecode.

---

### Mechanism 2: Aggregate State Management

Aggregate functions like `SUM`, `COUNT`, and `AVG` pose a different challenge: they are **Stateful**. 

Unlike a simple `SELECT name`, which looks at one row at a time, `SUM(price)` needs to remember the total it has seen so far. You must implement the concept of an **Accumulator Register**.

> 
> **🔑 Foundation: Aggregation State**
> 
> ### 1. What it IS
Aggregation state is the temporary memory held by a database to track progress during a "Folding" operation (where many rows are reduced to one result). 
- For `COUNT`, the state is an integer (starts at 0).
- For `SUM`, the state is a number (starts at 0 or NULL).
- For `AVG`, the state is a **tuple**: `(running_sum, running_count)`.
- For `MIN`/`MAX`, the state is the "best value seen so far."

### 2. WHY you need it right now
If you try to compute an average by loading all 1,000,000 rows into an array and then dividing, you will crash your engine with an `OutOfMemory` error. Streaming aggregation allows you to process 1,000,000 rows while only ever holding **two numbers** in memory: the sum and the count.

### 3. Key Insight: The Initialize-Step-Finalize Lifecycle
Aggregates in SQL follow a three-step lifecycle:
1.  **Init**: Reset the accumulator register to a default (0 or NULL).
2.  **Step**: For every row found by the scan, update the accumulator (e.g., `acc = acc + new_value`).
3.  **Finalize**: Perform the final math (e.g., for `AVG`, divide `sum` by `count`).


#### Handling NULLs in Aggregates
The SQL standard defines specific (and sometimes counter-intuitive) rules for NULLs in aggregates:
- `COUNT(*)`: Counts every row, even if all columns are NULL.
- `COUNT(col)`: Only counts rows where `col` is **NOT NULL**.
- `SUM(col)` / `AVG(col)`: Ignores NULL values completely. If all values in a group are NULL, the result of `SUM` is **NULL**, not 0.
- `AVG` must return a **REAL** (floating point), even if the input column is an integer. 

---

### Mechanism 3: GROUP BY (The Bucket Strategy)

`GROUP BY` takes aggregation to the next level. Instead of one global accumulator, you need **one accumulator per group**.

If you run `SELECT category, SUM(price) FROM products GROUP BY category`, and you have 50 categories, you need 50 separate `SUM` counters. 

**There are two primary ways to implement this:**

#### 1. Sort-Based Grouping (The Streaming Way)
If the input data is **already sorted** by the grouping column (e.g., via an index), you only need one accumulator.
- As you scan, if the `category` is the same as the last row, update the current accumulator.
- If the `category` changes, "Finalize" the current accumulator, emit the result row, and reset the accumulator for the new category.

#### 2. Hash-Based Grouping (The Memory Way)
If the data is unsorted, you must use a **Hash Map** in memory.
- The Key of the map is the grouping value (`category`).
- The Value of the map is the **Aggregation State** object.
- For every row, look up the category in the map and update its state.
- Once the scan is finished, iterate through the Hash Map and emit one row per entry.

**Implementation Tip**: For this milestone, if you haven't built a robust internal Hash Map yet, you can implement the **Sort-Based** approach by first performing a "Sort" operation on the input data (or requiring an index).

---

### Mechanism 4: The HAVING Clause

The `HAVING` clause is a filter that applies **after** aggregation. This is the "Aha!" moment of SQL query order. 

Consider: `SELECT category FROM products GROUP BY category HAVING SUM(price) > 1000`.
- The `WHERE` clause filters rows **before** they are grouped (Physical rows).
- The `HAVING` clause filters rows **after** they are grouped (Logical groups).

In your VDBE, the `HAVING` logic must be placed *after* the `Finalize` step of the aggregate but *before* the `ResultRow` opcode. It acts exactly like a `WHERE` clause, but its inputs are the values in the accumulator registers rather than the column values from the B-tree cursors.

---

### The Revelation Arc: Joins and Three-Valued Logic

When joining tables, you will encounter the "Missing Data" problem. 
`SELECT * FROM users JOIN orders ON users.id = orders.user_id`

If a user exists but has **zero orders**, the `INNER JOIN` logic will result in the inner loop (orders) finding 0 matches. The `ResultRow` is never reached. Consequently, the user "disappears" from the results. 

**The Reveal**: This is why `NULL` handling is so critical in joins. If you were implementing an `OUTER JOIN`, you would need logic to say: "If the inner loop finishes without finding a single match, emit a row anyway, but fill the 'orders' columns with `NULL`." While this milestone focuses on `INNER JOIN`, ensuring your `Column` opcode can return a `Value::Null` for non-existent matches is the first step toward supporting all join types.

---

### Design Decisions: Join Algorithms

| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Nested Loop Join (Chosen ✓)** | Simple to implement. Works with any join condition (`>`, `<`, `!=`). Uses very little memory. | $O(N \times M)$ performance. Deadly without indexes. | SQLite, All RDBMS (as fallback) |
| **Hash Join** | Extremely fast ($O(N+M)$) for equality joins. | Only works for `=` joins. Requires building a large hash table in memory. | PostgreSQL, MySQL 8.0+, SQL Server |
| **Merge Join** | Very fast if both sides are already sorted. | Requires sorting both tables first, which can be $O(N \log N)$. | Oracle, PostgreSQL |

**Why we choose NLJ**: In the context of an embedded database like SQLite, memory is often scarce. A Nested Loop Join (especially an **Indexed Nested Loop Join**) provides a great balance of code simplicity and performance. It allows us to leverage the B-trees we've already spent so much time perfecting.

---

### Knowledge Cascade: Learn One, Unlock Ten

1.  **Nested Loop Joins & Relational Algebra**: This is the "Multiplication" of the database world. By mastering the nested loop, you've understood how databases implement the **Cross Product** ($\times$) and **Join** ($\bowtie$) operators that form the basis of relational theory.
2.  **Aggregation State & Streaming Analytics**: The "Initialize-Step-Finalize" pattern is the exact same logic used in **MapReduce** (Google) and **Apache Flink**. "Map" is the scan, "Shuffle" is the grouping, and "Reduce" is the aggregation step.
3.  **Venn Diagrams vs. Loops**: This insight—that mathematical sets are physically loops—applies to **Computer Graphics** (clipping pixels), **Computational Geometry** (intersecting shapes), and **Search Engines** (intersecting keyword lists).
4.  **NULL Propagation**: Your work with aggregates and NULLs is a primer for **Functional Programming**. The way NULL "poisons" an `AVG` or `SUM` is identical to how the `Option` or `Maybe` monad works in languages like Rust or Haskell.

---

### System Awareness

In the System Map, we are now at the **highest level of the Execution Engine**. 
- Upstream: The **Query Planner** (Milestone 8) is deciding which table should be the "Outer" one.
- Downstream: The **VDBE** is managing multiple **Cursors** (Milestone 6) and pulling pages through the **Buffer Pool** (Milestone 4).

If your Buffer Pool is too small to hold the "hot" pages of both tables being joined, the system will start **Thrashing**. Each iteration of the outer loop will push the inner table's pages out of the cache, and each iteration of the inner loop will push the outer table's pages out. This is why tuning your Buffer Pool size is often the first step in fixing a "slow join."

---
<!-- END_MS -->




# TDD

A high-performance, embedded relational engine prioritizing ACID durability via Write-Ahead Logging and efficient data retrieval through a page-based B-tree storage engine and a register-based virtual machine.



<!-- TDD_MOD_ID: build-sqlite-m1 -->
# Module Design: SQL Tokenizer (build-sqlite-m1)

## 1. Module Charter

The **SQL Tokenizer** (Lexer) is the entry point of the SQL execution pipeline. Its primary responsibility is to transform a raw UTF-8 string buffer into a linear stream of discrete, typed **Tokens**. This module performs lexical analysis only; it does not validate SQL grammar or schema existence. 

**Core Responsibilities:**
- Scan the input buffer in a single O(N) pass.
- Categorize character sequences into keywords, identifiers, literals (string/numeric), and operators.
- Track source locations (line and column) for precise error reporting in downstream modules.
- Normalize case for keywords while preserving case for quoted identifiers and string literals.

**Non-Goals:**
- Validating if a table exists.
- Checking if a `SELECT` statement has a `FROM` clause.
- Handling multi-statement scripts (this module processes one "Statement Block" at a time).

**Invariants:**
- The output token stream must preserve the logical order of the input.
- White space and comments are discarded after serving as token delimiters.
- Every byte of input must either contribute to a token or trigger a lexical error.

---

## 2. File Structure

Implementation should follow this numbered sequence to establish dependencies:

1. `src/include/tokenizer/token_types.h`: Definition of the `TokenType` enumeration.
2. `src/include/tokenizer/token.h`: The `Token` structure and literal value unions.
3. `src/include/tokenizer/scanner.h`: The state machine container definition.
4. `src/tokenizer/scanner.c`: Core logic for character classification and state transitions.
5. `src/tokenizer/keywords.c`: Lookup table for SQL reserved words.

---

## 3. Complete Data Model

### 3.1 TokenType Enum
The `TokenType` distinguishes between syntax, data, and metadata.

| Category | Type | Example |
| :--- | :--- | :--- |
| **Keywords** | `TK_SELECT`, `TK_INSERT`, `TK_CREATE`, `TK_TABLE`, `TK_FROM`, `TK_WHERE`, `TK_JOIN`, `TK_ORDER`, `TK_BY`, `TK_LIMIT`, `TK_AND`, `TK_OR`, `TK_NOT`, `TK_NULL`, `TK_VALUES`, `TK_PRIMARY`, `TK_KEY` | `SELECT`, `FROM` |
| **Literals** | `TK_STRING`, `TK_INTEGER`, `TK_FLOAT`, `TK_IDENTIFIER` | `'it''s'`, `42`, `3.14`, `users` |
| **Operators** | `TK_EQUAL`, `TK_BANG_EQUAL`, `TK_LESS`, `TK_LESS_EQUAL`, `TK_GREATER`, `TK_GREATER_EQUAL` | `=`, `!=`, `<=`, `>=` |
| **Symbols** | `TK_LEFT_PAREN`, `TK_RIGHT_PAREN`, `TK_COMMA`, `TK_DOT`, `TK_SEMICOLON`, `TK_STAR` | `(`, `)`, `,`, `.`, `;`, `*` |
| **Special** | `TK_EOF`, `TK_ERROR` | (End of input / Lexical error) |

### 3.2 Token Structure
For expert-level performance, use a "String View" approach to avoid excessive allocations.

```c
typedef enum {
    VAL_INT,
    VAL_FLOAT,
    VAL_STR,
    VAL_NONE
} LiteralType;

typedef struct {
    TokenType type;
    const char* start;   // Pointer to start of lexeme in source buffer
    size_t length;       // Length of lexeme
    int line;
    int col;
    
    // Literal interpretation (optional at this stage, but helpful)
    struct {
        LiteralType type;
        union {
            int64_t i_val;
            double f_val;
        } as;
    } literal;
} Token;
```

### 3.3 Scanner State
```c
typedef struct {
    const char* source;  // The full SQL string
    const char* start;   // Start of current token being scanned
    const char* current; // Current character pointer
    int line;
    int col;
} Scanner;
```


![Lexer Architecture](./diagrams/tdd-diag-m1-arch.svg)

*(Visual: Pipeline showing `Buffer -> Scanner -> Token Stream -> ResultSet`)*

---

## 4. Interface Contracts

### 4.1 `void scanner_init(Scanner* scanner, const char* source)`
- **Purpose**: Initialize the FSM state.
- **Pre-condition**: `source` must be a null-terminated UTF-8 string.
- **Post-condition**: Scanner pointers are set to the first byte; line and col set to 1.

### 4.2 `Token scanner_next_token(Scanner* scanner)`
- **Purpose**: The core FSM execution. Scans until the next valid token is found.
- **Returns**: A `Token` struct. If an error is found, `type` is `TK_ERROR` and `lexeme` contains the error message.
- **Constraints**: 
    - Single pass (O(1) amortized).
    - Must skip all whitespace (space, tab, newline, carriage return).

### 4.3 `bool scanner_is_at_end(Scanner* scanner)`
- **Returns**: True if `*current == '\0'`.

---

## 5. Algorithm Specification: The FSM

The tokenizer is a **Deterministic Finite Automaton (DFA)**. The state is implicitly determined by the current character under the `current` pointer.

### 5.1 Basic Character Classification
Use a lookup table or macros for fast classification:
- `IS_DIGIT(c)`: `(c >= '0' && c <= '9')`
- `IS_ALPHA(c)`: `(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_')`
- `IS_WHITESPACE(c)`: `(c == ' ' || c == '\r' || c == '\t' || c == '\n')`

### 5.2 The Main Loop (Algorithm: `scan_token`)
1. **Skip Whitespace**: 
    - While `IS_WHITESPACE(*current)`:
        - If `\n`, increment `line`, reset `col`.
        - Else, increment `col`.
        - Advance `current`.
2. **Mark Start**: Set `scanner->start = scanner->current`.
3. **Switch on `*current`**:
    - **Single-byte symbols**: `(`, `)`, `,`, `.`, `;`, `*`, `+`, `-`. Return corresponding type.
    - **Operators with lookahead**: 
        - If `!`, peek next. If `=`, return `TK_BANG_EQUAL`. Else return `TK_ERROR`.
        - If `=`, return `TK_EQUAL`.
        - If `<`, peek next. If `=`, return `TK_LESS_EQUAL`. If `>`, return `TK_BANG_EQUAL` (SQL style). Else return `TK_LESS`.
    - **String Literals**: If `'`, call `scan_string()`.
    - **Identifiers/Keywords**: If `IS_ALPHA`, call `scan_identifier()`.
    - **Numbers**: If `IS_DIGIT`, call `scan_number()`.


![Tokenizer State Machine](./diagrams/tdd-diag-m1-fsm.svg)

*(Visual: State machine diagram with edges for String, Number, and Identifier branching)*

### 5.3 `scan_string()` (The Escaped Quote Logic)
1. Advance past initial `'`.
2. Loop:
    - If at end: return `TK_ERROR` ("Unterminated string").
    - If `\n`: increment `line`.
    - If `'`:
        - **Peek** next. If it is also `'`:
            - This is an escaped quote. Consume both. Stay in loop.
        - Else:
            - The string is closed. Advance past `'` and break.
    - Advance `current`.
3. Return `TK_STRING`.


![SQL String Escape Logic](./diagrams/tdd-diag-m1-esc.svg)

*(Visual: Detail of the string escape state: ' -> [' -> ' -> stay] vs [non-' -> stay] vs [end -> exit])*

### 5.4 `scan_number()` (Int vs Float)
1. Consume all digits.
2. If `.` is found AND next is a digit:
    - Consume `.`.
    - Consume all subsequent digits.
    - Return `TK_FLOAT`.
3. Else: return `TK_INTEGER`.

### 5.5 `scan_identifier()` (Keyword Triage)
1. Consume all `IS_ALPHA` or `IS_DIGIT` characters.
2. Extract the lexeme.
3. Perform a case-insensitive comparison against the keyword table.
4. If match: return `TK_[KEYWORD]`.
5. Else: return `TK_IDENTIFIER`.

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| Unterminated String | `scan_string` | Return `TK_ERROR`, advance to EOF | Yes: "Unterminated string literal" |
| Invalid Character | `scan_token` default case | Return `TK_ERROR`, advance one byte | Yes: "Unexpected character '@'" |
| Malformed Number | `scan_number` | Return `TK_ERROR`, consume until whitespace | Yes: "Invalid numeric format" |
| Unterminated Quoted Ident | `scan_quoted_ident` | Return `TK_ERROR`, advance to EOF | Yes: "Unterminated identifier" |

---

## 7. Implementation Sequence

### Phase 1: Skeleton & Symbols (2 Hours)
- Define `TokenType` and `Token` structs.
- Implement `scanner_init` and `scanner_next_token` for single-byte symbols only.
- **Checkpoint**: Scanning `( , ; )` should return 4 tokens followed by `TK_EOF`.

### Phase 2: Literals & Lookahead (3 Hours)
- Implement `scan_string` with escape logic.
- Implement `scan_number`.
- Implement multi-byte operators (`>=`, `!=`).
- **Checkpoint**: `SELECT 'it''s' FROM t WHERE x >= 10.5;` should correctly identify the string `'it''s'` as one token and `10.5` as a float.

### Phase 3: Keywords & Identifiers (2 Hours)
- Implement `scan_identifier` and the keyword lookup table.
- Ensure case-insensitivity (e.g., `sElEcT` matches `TK_SELECT`).
- **Checkpoint**: Full query `SELECT * FROM users` returns `[TK_SELECT, TK_STAR, TK_FROM, TK_IDENTIFIER]`.

---

## 8. Test Specification

### 8.1 Happy Path: Complex SELECT
- **Input**: `SELECT id, name FROM "Users" WHERE salary > 50000.00;`
- **Expected Output**:
    1. `TK_SELECT`, "SELECT"
    2. `TK_IDENTIFIER`, "id"
    3. `TK_COMMA`, ","
    4. `TK_IDENTIFIER`, "name"
    5. `TK_FROM`, "FROM"
    6. `TK_IDENTIFIER`, "Users" (Length 5, skip quotes)
    7. `TK_WHERE`, "WHERE"
    8. `TK_IDENTIFIER`, "salary"
    9. `TK_GREATER`, ">"
    10. `TK_FLOAT`, "50000.00"
    11. `TK_SEMICOLON`, ";"
    12. `TK_EOF`

### 8.2 Edge Case: Escaped Quotes
- **Input**: `'O''Reilly'`
- **Expected**: `TK_STRING`, lexeme `'O''Reilly'`. (Note: The literal value interpretation would be `O'Reilly`, but the lexeme in the token should match the source).

### 8.3 Edge Case: Whitespace and Lines
- **Input**: 
  ```sql
  SELECT 
  * 
  FROM 
  t;
  ```
- **Expected**: `TK_STAR` should report `line: 2`, `TK_FROM` should report `line: 3`.

### 8.4 Failure Case: Unterminated String
- **Input**: `SELECT 'unclosed string`
- **Expected**: `TK_SELECT`, followed by `TK_ERROR` with message containing "Unterminated".

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Throughput** | > 200 MB/s | Use `clock_gettime` to measure time to tokenize a repeating 1MB SQL string 100 times. |
| **Pass Count** | Exactly 1 | Verify that the `current` pointer never decrements (no backtracking). |
| **Allocation** | 0 per Token | Ensure `Token.lexeme` is a pointer/length into the existing buffer, not a `strdup`. |

---

## 10. Memory Layout of Hot Structures

Since the Tokenizer is often the bottleneck in high-volume query ingestion (e.g., bulk inserts), the `Scanner` state should fit within a single CPU cache line (64 bytes).

**Scanner Layout (Offsets):**
- `0x00`: `source` pointer (8 bytes)
- `0x08`: `start` pointer (8 bytes)
- `0x10`: `current` pointer (8 bytes)
- `0x18`: `line` (4 bytes)
- `0x1C`: `col` (4 bytes)
- `0x20`: Padding/Reserved (32 bytes)
- **Total**: 64 bytes.

The `TokenType` enum should be 1 byte if possible (packed) to keep `Token` structs small during large stream operations.
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m2 -->
# Module Design: SQL Parser (AST) (build-sqlite-m2)

## 1. Module Charter

The **SQL Parser** is the structural architect of the database engine. It consumes the linear stream of tokens produced by the Tokenizer and constructs a hierarchical **Abstract Syntax Tree (AST)** that represents the logical intent of the query. Its primary goal is to enforce the SQL grammar and resolve operator precedence, ensuring that the subsequent Bytecode Compiler (VDBE) receives a syntactically perfect and unambiguous tree.

**Core Responsibilities:**
- Implement a **Recursive Descent** parser for high-level SQL statements (`SELECT`, `INSERT`, `CREATE TABLE`).
- Implement a **Pratt Parser** (Precedence Climbing) for complex expressions to correctly handle mathematical and logical operator precedence (e.g., `AND` vs `OR`).
- Transform raw token strings (like escaped quotes `'it''s'`) into normalized literal values within AST nodes.
- Provide "Panic Mode" error recovery to detect multiple syntax errors in a single pass.
- Utilize an **Arena Allocator** for AST nodes to guarantee $O(1)$ allocation and sub-millisecond cleanup.

**Non-Goals:**
- **Semantic Validation**: This module does not check if a table or column exists in the database schema.
- **Type Checking**: It does not validate if you are adding a string to an integer (this happens in the VDBE/Binder).
- **Optimization**: No query rewriting occurs here; the AST is a "raw" syntactic representation.

**Invariants:**
- Every node in the AST must maintain a reference to its source token for error localization.
- The parser must be stateless regarding the database file; it only operates on the token stream.
- An empty token stream must result in a `NULL` or `EMPTY_STMT` node, never a crash.

---

## 2. File Structure

Implementation follows this numbered sequence:

1. `src/include/parser/ast_nodes.h`: Definition of AST node types and the Tagged Union structure.
2. `src/include/parser/arena.h`: Interface for the Arena Allocator.
3. `src/parser/arena.c`: Implementation of the contiguous memory block allocator.
4. `src/parser/expr_parser.c`: The Pratt Parsing logic for expressions.
5. `src/parser/stmt_parser.c`: Recursive descent logic for `SELECT`, `INSERT`, and `CREATE`.
6. `src/parser/parser.c`: Top-level entry point and error recovery logic.

---

## 3. Complete Data Model

### 3.1 AST Node Memory Layout (Expert Level)
To achieve the performance target of 1000 nodes in < 1ms, we avoid individual `malloc` calls. We use a **Contiguous Tagged Union** pattern. Each node is exactly 64 bytes to align with cache lines.

| Offset | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | `uint32_t` | `type` | `AST_SELECT`, `AST_EXPR_BINARY`, etc. |
| 0x04 | `uint32_t` | `line` | Source line for error reporting. |
| 0x08 | `uint32_t` | `col` | Source column for error reporting. |
| 0x0C | `uint32_t` | `_pad` | Alignment padding. |
| 0x10 | `union` | `data` | Variant-specific data (48 bytes). |

### 3.2 AST Node Variants

```c
typedef enum {
    NODE_SELECT, NODE_INSERT, NODE_CREATE_TABLE,
    NODE_EXPR_BINARY, NODE_EXPR_UNARY, NODE_EXPR_LITERAL, NODE_EXPR_COLUMN,
    NODE_COLUMN_DEF, NODE_LIMIT_OFFSET
} ASTNodeType;

// Expression Data
typedef struct {
    TokenType op;
    struct ASTNode* left;
    struct ASTNode* right;
} BinaryExpr;

// Select Statement Data
typedef struct {
    struct ASTNode* result_columns; // Linked list or Array of expressions
    const char* table_name;
    struct ASTNode* where_clause;
    struct ASTNode* limit_clause;
    bool is_distinct;
} SelectStmt;

// Literal Data
typedef struct {
    LiteralType type;
    union {
        int64_t i_val;
        double f_val;
        char* s_val; // Unescaped string
    } value;
} LiteralExpr;
```


![AST Node Memory Layout](./diagrams/tdd-diag-m2-ast-node.svg)

*(Visual: A memory block diagram showing the 64-byte node with a pointer pointing to children nodes in the same Arena block)*

---

## 4. Interface Contracts

### 4.1 `ASTNode* parser_parse(TokenStream* stream)`
- **Purpose**: Main entry point for the parser.
- **Input**: A stream/iterator of tokens from Milestone 1.
- **Output**: Root of the AST.
- **Error**: Returns `NULL` and populates the `ParserError` context if syntax is invalid.

### 4.2 `ASTNode* parser_parse_expression(ParserContext* ctx, int min_precedence)`
- **Purpose**: The Pratt implementation.
- **Precedence Logic**: Only consumes tokens with binding power greater than `min_precedence`.
- **Edge Case**: Handles parenthesized expressions by resetting `min_precedence` to 0.

### 4.3 `void parser_arena_free(Arena* arena)`
- **Purpose**: Wipes all AST nodes in a single $O(1)$ memory release.
- **Constraint**: Must be called after the Bytecode Compiler has finished processing the tree.

---

## 5. Algorithm Specification: Pratt Expression Parsing

Traditional recursive descent for expressions results in deep, inefficient stacks (e.g., `parse_or` -> `parse_and` -> `parse_equality`...). Pratt parsing replaces this with a simple loop and a precedence table.

### 5.1 Precedence Table (Binding Power)

| Token Type | Precedence | Associativity |
| :--- | :--- | :--- |
| `TK_OR` | 10 | Left |
| `TK_AND` | 20 | Left |
| `TK_NOT` | 30 | Right (Unary) |
| `TK_EQUAL`, `TK_BANG_EQUAL` | 40 | Left |
| `TK_LESS`, `TK_GREATER` | 50 | Left |
| `TK_PLUS`, `TK_MINUS` | 60 | Left |
| `TK_STAR`, `TK_SLASH` | 70 | Left |

### 5.2 Procedure: `parse_expression(precedence)`
1. **Prefix Step**: 
    - Peek at the current token.
    - If `TK_INTEGER`, `TK_STRING`, `TK_NULL`: Return a `NODE_EXPR_LITERAL`.
    - If `TK_IDENTIFIER`: Return a `NODE_EXPR_COLUMN`.
    - If `TK_MINUS` or `TK_NOT`: Call `parse_expression(UNARY_PRECEDENCE)` and return `NODE_EXPR_UNARY`.
    - If `TK_LEFT_PAREN`: Call `parse_expression(0)`, then consume `TK_RIGHT_PAREN`.
2. **Infix Loop**:
    - While `get_precedence(peek_token()) > precedence`:
        - `token = consume()`
        - `left_node = create_binary_node(op: token, left: left_node, right: parse_expression(token.precedence))`
3. **Return `left_node`**.


![Pratt Parsing Precedence Climbing](./diagrams/tdd-diag-m2-pratt-flow.svg)

*(Visual: Flowchart showing the "Climbing" loop: Parse Prefix -> Peek Next Op -> If Precedence > Current -> Recurse)*

---

## 6. Algorithm Specification: SELECT Statement

The `SELECT` statement is parsed using traditional Recursive Descent rules.

### 6.1 Procedure: `parse_select()`
1. Consume `TK_SELECT`.
2. **Result Columns**: 
    - Loop: 
        - If `TK_STAR`, create a special "All" node.
        - Else, call `parse_expression(0)`.
        - Check for optional `TK_AS` + `TK_IDENTIFIER` (Alias).
        - If `TK_COMMA`, continue loop; else break.
3. **FROM Clause**:
    - Consume `TK_FROM`.
    - Consume `TK_IDENTIFIER` (Table Name).
4. **WHERE Clause (Optional)**:
    - If `TK_WHERE`:
        - `where_node = parse_expression(0)`.
5. **LIMIT Clause (Optional)**:
    - If `TK_LIMIT`:
        - `limit_node = parse_expression(0)`.
6. Consume `TK_SEMICOLON` or `TK_EOF`.


![SELECT Statement Data Flow](./diagrams/tdd-diag-m2-select-flow.svg)

*(Visual: Sequential flowchart showing the checkpoints: [SELECT] -> [COLS] -> [FROM] -> [WHERE?] -> [LIMIT?])*

---

## 7. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `UnexpectedToken` | `consume()` | Skip to next `TK_SEMICOLON` | Yes: "Expected 'FROM', found 'WHERE' at line 1:12" |
| `MismatchedParen` | `parse_expression` | Panic to next statement | Yes: "Unclosed parenthesis" |
| `MissingColumnDef`| `parse_create` | Abort `CREATE TABLE` node | Yes: "Table must have at least one column" |
| `IncompleteExpr`  | `parse_expression` | Return error node | Yes: "Trailing operator in expression" |

---

## 8. Implementation Sequence

### Phase 1: Expression Pratt Parsing (Estimated: 4 Hours)
- Implement `arena_alloc`.
- Implement `parse_expression` for literals, identifiers, and binary operators.
- **Checkpoint**: Running `parser_parse_expression` on `1 + 2 * 3` should produce a tree where `+` is the root and `*` is the right child.

### Phase 2: Statement Recursive Descent (Estimated: 4 Hours)
- Implement `parse_create_table` with column constraints.
- Implement `parse_insert`.
- Implement `parse_select` with `WHERE` and `LIMIT`.
- **Checkpoint**: `SELECT * FROM users WHERE id = 5;` should produce a `SelectStmt` node with a `BinaryExpr` node in the `where_clause` field.

---

## 9. Test Specification

### 9.1 Happy Path: Operator Precedence
- **Input**: `SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3`
- **Logic**: `AND` (20) > `OR` (10).
- **Expected AST Structure**:
    - Root: `SelectStmt`
    - Where: `BinaryExpr(OR)`
        - Left: `BinaryExpr(=, a, 1)`
        - Right: `BinaryExpr(AND)`
            - Left: `BinaryExpr(=, b, 2)`
            - Right: `BinaryExpr(=, c, 3)`

### 9.2 Edge Case: String Unescaping
- **Input**: `INSERT INTO t VALUES ('It''s logic');`
- **Expected**: The `LiteralExpr` node for the string should contain the value `It's logic` (one single quote), not `It''s logic`.

### 9.3 Failure Case: Invalid Syntax
- **Input**: `SELECT FROM table;` (Missing column list)
- **Expected**: `parser_parse` returns `NULL`. `ParserError` contains `line: 1, col: 8, message: "Expected expression or '*' before 'FROM'"`.

---

## 10. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Parsing Latency** | < 1ms for 1k nodes | Time `parser_parse` on a query with a 500-clause `WHERE` (e.g., `x=1 AND x=2 AND ...`). |
| **Memory Efficiency** | 0 Small Mallocs | Use a memory profiler (Valgrind/Instruments) to ensure only large Arena blocks are allocated. |
| **Recursion Depth** | > 500 Levels | Ensure stack safety by testing deeply nested parentheses `((((...1...))))`. |

---

## 11. Column Constraints Representation

When parsing `CREATE TABLE`, the parser must capture column attributes. These are stored in a `ColumnDef` node:

```c
typedef struct {
    const char* name;
    DataType type; // INT, TEXT, etc.
    bool is_primary_key;
    bool is_not_null;
    bool is_unique;
} ColumnDef;
```
If multiple constraints appear (e.g., `PRIMARY KEY NOT NULL`), the parser simply flips multiple booleans on the same `ColumnDef` node.


![Grammar Ambiguity](./diagrams/m2-alt-reality.svg)

<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m3 -->
# Module Design: Bytecode Compiler & VDBE (build-sqlite-m3)

## 1. Module Charter

The **Virtual Database Engine (VDBE)** is the execution core of the database. It acts as a specialized, register-based virtual machine designed to perform high-speed data manipulation by abstracting B-tree operations into a linear instruction set. The **Bytecode Compiler** acts as the "backend" of the SQL engine, translating the high-level, recursive Abstract Syntax Tree (AST) into a flat array of VDBE instructions (Opcodes).

**Core Responsibilities:**
- **Code Generation**: Traverse the AST and emit a sequence of opcodes that implement the query logic.
- **Register Management**: Allocate and manage a virtual register file for intermediate expression results and column data.
- **Control Flow**: Implement jumps and loops required for table scans, `WHERE` clause filtering, and `JOIN` logic.
- **Abstraction**: Provide a "Cursor" interface that hides the complexity of B-tree traversal from the high-level execution logic.
- **Execution**: Run the bytecode in a highly optimized fetch-decode-execute loop.

**Non-Goals:**
- **Parsing**: This module assumes a valid AST is provided by Milestone 2.
- **Physical I/O**: The VDBE calls B-tree/Pager methods; it does not handle file descriptors or buffer eviction directly.
- **Optimization**: This milestone focuses on a "straight-line" compiler; cost-based optimization is deferred to Milestone 8.

**Invariants:**
- Every bytecode program must end with a `Halt` instruction to prevent program counter overflow.
- Registers are "typed" at runtime; an operation on incompatible types (e.g., `Add` on a String and Integer) must trigger a VM-level error.
- Cursors must be explicitly opened before access and are automatically closed when the VM halts.

---

## 2. File Structure

The implementation follows this sequence:

1. `src/include/vdbe/opcodes.h`: Definition of the Instruction Set Architecture (ISA).
2. `src/include/vdbe/value.h`: The `Value` tagged union for register storage.
3. `src/include/vdbe/vdbe.h`: The `Vdbe` machine state and `Instruction` struct.
4. `src/vdbe/compiler.c`: Recursive visitor that emits bytecode from AST nodes.
5. `src/vdbe/vm.c`: The core execution loop (Fetch-Decode-Execute).
6. `src/vdbe/explain.c`: Human-readable formatting for bytecode programs.

---

## 3. Complete Data Model

### 3.1 The Instruction Word (Byte-Level Layout)
To ensure high performance and cache alignment, each instruction is exactly 16 bytes.

| Offset | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | `uint8_t` | `opcode` | The operation to perform (e.g., `OP_Column`). |
| 0x01 | `int8_t` | `p1` | Usually a cursor index or register index. |
| 0x02 | `int16_t` | `p2` | Usually a jump target (PC) or column index. |
| 0x04 | `int32_t` | `p3` | Usually a destination register or constant value. |
| 0x08 | `void*` | `p4` | Pointer to complex data (strings, blobs) or metadata. |


![VDBE Runtime Architecture](./diagrams/tdd-diag-m3-vm-arch.svg)

*(Visual: Pipeline showing AST -> Compiler -> Opcode Array -> VM Loop -> Register/Cursor State)*

### 3.2 Register Value (`Value` Struct)
Registers are dynamic. They store the "Live" data during a query.

```c
typedef enum {
    VAL_NULL,
    VAL_INT,
    VAL_FLOAT,
    VAL_TEXT,
    VAL_BLOB
} ValueType;

typedef struct {
    ValueType type;
    union {
        int64_t i;
        double r;
        struct {
            char* z;
            int n;
        } s; // String/Blob
    } u;
} Mem; // In SQLite, registers are called "Mem" units
```

### 3.3 The VDBE State
```c
typedef struct {
    Instruction* aOp;    // Array of instructions
    int nOp;            // Number of instructions
    int pc;             // Program Counter
    Mem* aReg;          // Register File (e.g., 256 registers)
    VdbeCursor** apCsr; // Array of Cursors (e.g., 16 cursors)
    int nReg;
    int nCsr;
    char* zErrMsg;      // Runtime error message
} Vdbe;
```

---

## 4. Interface Contracts

### 4.1 `Vdbe* vdbe_compile(ASTNode* root)`
- **Purpose**: Translates AST to Bytecode.
- **Input**: AST root.
- **Output**: Allocated `Vdbe` object ready for execution.
- **Error**: Returns `NULL` on `TableNotFound` or `RegisterOverflow` (if an expression is too deep).

### 4.2 `VdbeResult vdbe_step(Vdbe* p)`
- **Purpose**: Executes the program until a row is produced or the program ends.
- **Return Values**:
    - `VDBE_ROW`: A result row is available in the designated registers.
    - `VDBE_DONE`: Execution finished successfully.
    - `VDBE_ERROR`: A runtime error occurred (e.g., Type Mismatch).

### 4.3 `void vdbe_explain(Vdbe* p)`
- **Purpose**: Prints the instruction array to `stdout` in a tabular format for debugging.

---

## 5. Algorithm Specification: Instruction Set (ISA)

The VDBE implements a "Dataflow Assembly." Below are the critical opcodes required for Milestone 3.

| Opcode | P1 | P2 | P3 | Action |
| :--- | :--- | :--- | :--- | :--- |
| `Integer` | Value | Reg | - | `aReg[P2] = (int64_t)P1` |
| `String` | - | Reg | P4 | `aReg[P2] = (string)P4` |
| `OpenRead` | Csr | Root | - | Open cursor P1 on B-tree starting at P2. |
| `Rewind` | Csr | Jump | - | Move Csr P1 to first row. If empty, `pc = P2`. |
| `Next` | Csr | Jump | - | Move Csr P1 to next row. If successful, `pc = P2`. |
| `Column` | Csr | Col | Reg | `aReg[P3] = aCsr[P1].data[P2]` |
| `ResultRow` | Reg | Count | - | Output `Count` registers starting at `Reg`. |
| `Halt` | - | - | - | Terminate execution. |
| `Add` | RegL | RegR | RegOut| `aReg[P3] = aReg[P1] + aReg[P2]` |
| `Lt` | RegL | Jump | RegR | If `aReg[P1] < aReg[P3]`, `pc = P2`. |

---

## 6. Algorithm Specification: Code Generation

The compiler uses a recursive `code_gen(node)` function. It maintains a `next_reg` counter that resets for each statement.

### 6.1 Compiling `SELECT * FROM table`
1. **Emit `OpenRead(0, root_page)`**: P1=0 (Cursor 0), P2=Root of table.
2. **Emit `Rewind(0, halt_label)`**: P1=0. P2 will be patched later.
3. **Loop Start Label**: `L1`
4. **For each column `i` in table**:
   - **Emit `Column(0, i, i+1)`**: Extract column `i` into Register `i+1`.
5. **Emit `ResultRow(1, num_cols)`**: Output registers.
6. **Emit `Next(0, L1)`**: Jump back to `L1` if more rows exist.
7. **Label `halt_label`**: Emit `Halt`.

### 6.2 Backpatching Jumps
Since the compiler emits code linearly, the jump target for `Rewind` (the end of the loop) isn't known until the loop is finished.
1. Store the index of the `Rewind` instruction.
2. Compile the loop body.
3. After `Next` is emitted, set `aOp[rewind_idx].p2 = current_pc`.


![AST to Bytecode Mapping](./diagrams/tdd-diag-m3-compile-flow.svg)

*(Visual: Sequential steps of compiling a SELECT: [Emit Open] -> [Mark L1] -> [Emit Column] -> [Emit Next L1] -> [Backpatch Rewind])*

---

## 7. Algorithm Specification: Register Allocation

To prevent register collisions in complex expressions like `(a + b) * (c + d)`, the compiler uses a **Stack-Based Allocation** approach during expression traversal.

**Procedure: `allocate_expr(node)`**
1. If `node` is a Literal:
   - `r = next_reg++`
   - Emit `Integer` or `String` into `r`.
   - Return `r`.
2. If `node` is Binary (`+`):
   - `reg_left = allocate_expr(node->left)`
   - `reg_right = allocate_expr(node->right)`
   - `reg_out = next_reg++`
   - Emit `Add(reg_left, reg_right, reg_out)`
   - **Important**: We do *not* decrement `next_reg` yet to ensure parent nodes can safely use `reg_out` without child nodes overwriting it. Registers are recycled at the end of the statement.

---

## 8. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `RegisterOverflow` | Compiler | Abort compilation, free `Vdbe` | Yes: "Expression too complex" |
| `TypeMismatch` | VM (`Add` op) | `Halt` with error code | Yes: "Cannot add TEXT and INT" |
| `InvalidCursor` | VM (`Column` op) | `Halt` with error code | No (Internal bug) |
| `TableNotFound` | Compiler | Abort compilation | Yes: "Table 'x' does not exist" |

---

## 9. Implementation Sequence

### Phase 1: ISA & VM Skeleton (4 Hours)
- Define `Opcode` enum and `Instruction` struct.
- Implement the `vdbe_step` while loop with a `switch(opcode)` block.
- Implement `Halt`, `Integer`, and `String` opcodes.
- **Checkpoint**: Manually construct an instruction array `[Integer 42 R1, Halt]` and verify `aReg[1].i == 42` after execution.

### Phase 2: Simple Scan Compiler (4 Hours)
- Implement `vdbe_compile` for `SELECT * FROM table`.
- Implement `OpenRead`, `Rewind`, `Column`, `Next`, and `ResultRow` in the VM.
- Integrate with the (stubbed) B-tree layer to "scan" a hardcoded array of records.
- **Checkpoint**: Compiling `SELECT * FROM users` produces the correct opcode sequence (verified via `vdbe_explain`).

### Phase 3: Expression & Logic (3 Hours)
- Implement `code_gen` for Binary Expressions (`+`, `-`, `*`).
- Implement comparison opcodes (`Eq`, `Lt`, `Gt`) and conditional jumps.
- Implement `WHERE` clause compilation (conditional jump to `Next`).
- **Checkpoint**: `SELECT name FROM t WHERE age > 21` correctly emits a `Lt` or `Le` instruction that jumps over the `ResultRow`.

---

## 10. Test Specification

### 10.1 Happy Path: Constant Arithmetic
- **Input AST**: `SELECT 1 + 2;`
- **Expected Bytecode**:
    1. `Integer 1, R1`
    2. `Integer 2, R2`
    3. `Add R1, R2, R3`
    4. `ResultRow R3, 1`
    5. `Halt`
- **Verification**: `vdbe_step` returns `VDBE_ROW`, `aReg[R3]` contains `VAL_INT(3)`.

### 10.2 Edge Case: Empty Table
- **Scenario**: `SELECT * FROM empty_table;`
- **Logic**: `Rewind` should immediately jump to the instruction *after* the `Next` opcode.
- **Verification**: `vdbe_step` returns `VDBE_DONE` immediately without ever hitting `ResultRow`.

### 10.3 Failure Case: Register Exhaustion
- **Input**: A query with 300 nested additions.
- **Expected**: Compiler returns `NULL` and sets error to "Register limit exceeded."

---

## 11. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Instruction Dispatch**| < 10ns / op | Run a loop of 10M `Add` instructions; measure total time / 10M. |
| **Full Table Scan** | 1M rows/sec | Scan a table with 1M rows (cached in RAM) and count `VDBE_ROW` returns. |
| **Memory Overhead** | < 64KB / Vdbe | Measure `sizeof(Vdbe) + (nOp * 16) + (nReg * sizeof(Mem))`. |

---

## 12. State Machine: VM Execution

The VM follows a strict fetch-decode-execute cycle.

1. **FETCH**: `instr = aOp[pc]`
2. **DECODE**: Extract P1, P2, P3, P4.
3. **EXECUTE**: `switch(instr.opcode)`
4. **INCREMENT**: `pc++` (unless a jump occurred).
5. **HALT**: If `opcode == OP_Halt` or `pc >= nOp`.

**Illegal Transitions**:
- Jumping to `pc < 0` or `pc >= nOp`.
- Accessing `apCsr[i]` where `i >= nCsr`.
- Accessing `aReg[i]` where `i >= nReg`.


![Register Usage Map](./diagrams/m3-reg-use.svg)

<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m4 -->
# Module Design: Buffer Pool Manager (build-sqlite-m4)

## 1. Module Charter

The **Buffer Pool Manager** (Pager) is the central memory management authority of the storage engine. It abstracts the physical database file into a logical array of fixed-size **Pages** (4096 bytes), providing a "virtual memory" layer for the B-tree and Virtual Machine. Its primary duty is to cache frequently accessed disk blocks in a finite set of memory **Frames**, utilizing an **LRU (Least Recently Used)** eviction policy to maximize cache hit rates. This module is responsible for the "Durability Soul" of the database, ensuring that modified (dirty) pages are safely flushed to disk and that active pages are "pinned" to prevent use-after-free corruption. It does not understand the content of the pages (B-trees, records); it treats them as opaque byte buffers.

**Invariants:**
- A page with a `pin_count > 0` must **never** be selected for eviction.
- A "dirty" page must be written to disk (and `fsync`'d if in synchronous mode) before its frame can be repurposed for a different page ID.
- Every `fetch_page` call must eventually be balanced by an `unpin_page` call to prevent pool exhaustion.

---

## 2. File Structure

Implementation should follow this sequence to ensure structural integrity:

1. `src/include/storage/pager_types.h`: Definition of constants (PAGE_SIZE), Frame descriptors, and Error codes.
2. `src/include/storage/lru_cache.h`: Interface for the O(1) LRU eviction policy.
3. `src/storage/lru_cache.c`: Implementation of the Doubly Linked List + Hash Map for LRU.
4. `src/include/storage/pager.h`: Main Pager interface for fetching, pinning, and flushing pages.
5. `src/storage/pager.c`: Core logic for disk I/O coordination, frame management, and dirty tracking.

---

## 3. Complete Data Model

### 3.1 Frame Descriptor (Memory Layout)
Each frame in the buffer pool is associated with a metadata descriptor. These descriptors are stored in a contiguous array to minimize cache misses during lookup.

| Offset | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | `uint32_t` | `page_id` | Physical ID of the page currently in this frame. |
| 0x04 | `uint32_t` | `pin_count` | Number of active references (prevents eviction). |
| 0x08 | `bool` | `is_dirty` | True if the frame has been modified and requires flush. |
| 0x09 | `bool` | `is_valid` | True if the frame contains actual disk data. |
| 0x10 | `void*` | `data` | Pointer to the 4096-byte memory block. |
| 0x18 | `LRUNode*` | `lru_node` | Pointer to the node in the eviction list. |

### 3.2 Buffer Pool Structure
The `Pager` object maintains the state of the entire caching system.

```c
#define PAGE_SIZE 4096

typedef struct {
    int fd;                 // File descriptor for the .db file
    uint32_t num_frames;    // Size of the buffer pool (e.g., 1000)
    PageDescriptor* frames; // Array of metadata descriptors
    void* buffer_pool_mem;  // Contiguous block of (num_frames * 4096) bytes
    LRUCache* lru;          // Eviction policy controller
    HashTable* page_table;  // Maps page_id -> frame_index
    pthread_mutex_t latch;  // Global lock for metadata synchronization
} Pager;
```


![Pager Architecture](./diagrams/tdd-diag-m4-pager-arch.svg)

*(Visual: Architecture showing B-tree calling Pager -> Pager checking PageTable -> If Miss, LRU selects victim -> Pager calls OS `pread`/`pwrite`)*

---

## 4. Interface Contracts

### 4.1 `void* pager_fetch_page(Pager* p, uint32_t page_id)`
- **Purpose**: Retrieves a pointer to a page's data.
- **Constraints**: 
    - If `page_id` is in cache: Update LRU status, increment `pin_count`, return pointer.
    - If `page_id` is NOT in cache: Execute Eviction/Load algorithm (Section 5.2).
- **Errors**: 
    - `POOL_FULL`: If all frames are pinned (`pin_count > 0`), no eviction is possible.
    - `IO_ERROR`: If `pread` fails.

### 4.2 `void pager_unpin_page(Pager* p, uint32_t page_id, bool is_dirty)`
- **Purpose**: Releases a reference to a page.
- **Logic**: Decrement `pin_count`. If `is_dirty` is true, set the frame's `is_dirty` flag. 
- **Invariant**: Must be called after every `fetch_page`.

### 4.3 `void pager_flush_all(Pager* p)`
- **Purpose**: Forces all dirty pages to disk.
- **Use Case**: Shutdown or Transaction Commit.

---

## 5. Algorithm Specification

### 5.1 O(1) LRU Strategy
To achieve O(1) eviction, the `LRUCache` uses a **Doubly Linked List** (for order) and a **Hash Map** (for location).

1. **Accessing a Page**:
   - Find node in Hash Map.
   - Remove node from its current position in the Linked List.
   - Move node to the **Head** of the Linked List (Most Recently Used).
2. **Evicting a Page**:
   - Start at the **Tail** of the Linked List (Least Recently Used).
   - Check `pin_count` of the frame associated with that node.
   - If `pin_count > 0`, move to the previous node (searching for an unpinned victim).
   - Once an unpinned victim is found: Remove from Hash Map, remove from List, return `frame_index`.


![LRU Eviction Algorithm](./diagrams/tdd-diag-m4-lru-algo.svg)

*(Visual: Linked List showing [Head/MRU] <-> [Node] <-> [Tail/LRU] and a HashMap pointing into the nodes)*

### 5.2 The Pager Fetch/Evict Logic (The Implementation Path)
When `pager_fetch_page(page_id)` is called:

1. **Lock** the Pager latch.
2. **Check Page Table**:
   - If `page_id` exists in `page_table`:
     - `frame_idx = page_table[page_id]`
     - Increment `frames[frame_idx].pin_count`.
     - `lru_touch(p->lru, frame_idx)`.
     - **Unlock** and return `frames[frame_idx].data`.
3. **If Miss (Not in Cache)**:
   - **Identify Victim**: `victim_idx = lru_get_victim(p->lru)`.
   - If no victim found (all pinned): **Unlock**, return `ERR_POOL_FULL`.
4. **Evict Victim**:
   - `PageDescriptor* victim = &p->frames[victim_idx]`.
   - If `victim->is_dirty`:
     - `lseek(p->fd, victim->page_id * PAGE_SIZE, SEEK_SET)`.
     - `write(p->fd, victim->data, PAGE_SIZE)`.
     - `fsync(p->fd)` (depending on durability mode).
   - Remove `victim->page_id` from `page_table`.
5. **Load New Page**:
   - `lseek(p->fd, page_id * PAGE_SIZE, SEEK_SET)`.
   - `read(p->fd, victim->data, PAGE_SIZE)`.
   - Update `victim` metadata: `page_id = page_id`, `is_dirty = false`, `pin_count = 1`, `is_valid = true`.
   - Add `page_id -> victim_idx` to `page_table`.
   - `lru_add(p->lru, victim_idx)`.
6. **Unlock** and return `victim->data`.

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `POOL_FULL` | `lru_get_victim` | B-tree must wait or abort transaction. | Yes: "Out of memory frames" |
| `DISK_FULL` | `write` during flush | Attempt to rollback; enter Read-Only mode. | Yes: "Disk full, write failed" |
| `IO_ERROR` | `read`/`write` | Retry once, then panic (Database is unstable). | Yes: "Physical I/O error" |
| `PAGE_CORRUPTION`| Checksum check | Reject page, return `ERR_CORRUPT`. | Yes: "Database file is corrupt" |

---

## 7. Implementation Sequence

### Phase 1: LRU Infrastructure (Estimated: 3-4 Hours)
- Implement Doubly Linked List with `prev/next` pointers.
- Implement Hash Map (or simple array if `num_frames` is small) to store `frame_index -> ListNode*`.
- **Checkpoint**: Test `lru_touch` and `lru_get_victim` with dummy indices. Verify that the oldest touched index is returned as the victim.

### Phase 2: Pager Machinery (Estimated: 4-5 Hours)
- Implement `pager_init` to `mmap` or `malloc` the buffer pool memory.
- Implement `pager_fetch_page` with the Disk Read logic.
- Implement `pager_unpin_page` with the Dirty Flag logic.
- **Checkpoint**: Fetch Page 0, write "Hello" into the pointer, unpin with `is_dirty=true`. Close pager. Reopen pager and fetch Page 0—verify "Hello" persists on disk.

### Phase 3: Concurrency & Stress (Estimated: 2 Hours)
- Add `pthread_mutex` around the descriptor updates.
- Stress test: Multiple threads fetching and unpinning random page IDs.
- **Checkpoint**: Ensure no deadlocks and that the `pin_count` logic successfully prevents the pool from being "over-evicted."

---

## 8. Test Specification

### 8.1 Happy Path: Cache Hit
- **Setup**: Fetch Page 1, Unpin Page 1.
- **Action**: Fetch Page 1 again.
- **Expected**: `page_table` lookup succeeds; no `read()` syscall is issued (verify via mock I/O).

### 8.2 Edge Case: Victim Pinning
- **Setup**: Pool size = 2. Fetch Page 1 (pin=1), Fetch Page 2 (pin=1).
- **Action**: Attempt to Fetch Page 3.
- **Expected**: Return `ERR_POOL_FULL` because no unpinned victims exist.

### 8.3 Failure Case: Write Failure on Eviction
- **Setup**: Mark Page 1 as dirty.
- **Action**: Fetch Page 100 (triggering Page 1 eviction). Simulate `write()` failure.
- **Expected**: Frame 1 remains dirty and pinned; `fetch_page` returns `IO_ERROR`.

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **LRU Lookup** | O(1) | Profile with 10k frames; lookup time must remain constant. |
| **Cache Hit Latency** | < 500ns | Average time of `pager_fetch_page` when page is resident. |
| **I/O Alignment** | 4096-byte | Use `posix_memalign` and verify addresses are 4k aligned for Direct I/O. |

---

## 10. Concurrency Specification

The Pager uses a **Single Global Latch (Mutex)** strategy for the metadata. While this limits parallelism, it is the standard "Intermediate" approach to avoid complex fine-grained locking.

**Lock Ordering:**
1. Acquire `Pager.latch`.
2. Perform Hash Table/LRU operations.
3. If I/O is needed: **Release** `Pager.latch` during the actual `read()`/`write()` to allow other threads to access *resident* pages, then **Re-acquire** to update metadata.
4. Release `Pager.latch`.

**Thread Safety Invariant**: A pointer returned by `fetch_page` is safe to use as long as the caller holds the "Pin." No other thread can evict that frame while the `pin_count > 0`.
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m5 -->
# Module Design: B-tree Page Format & Table Storage (build-sqlite-m5)

## 1. Module Charter

The **B-tree Page Format & Table Storage** module is the physical heart of the database. It defines how logical SQL rows are mapped into fixed-size 4096-byte blocks on disk. This module implements **Clustered Storage**, where the primary table data is stored directly within the leaves of a B-tree, keyed by a 64-bit `rowid`. It manages the **Slotted Page** architecture to handle variable-length records efficiently while minimizing fragmentation. 

**Core Responsibilities:**
- Implement binary serialization/deserialization for page headers and cell pointer arrays.
- Provide variable-length integer (Varint) encoding to minimize metadata overhead.
- Manage the "dual-growth" slotted page layout (pointers grow down, data grows up).
- Execute B-tree balancing operations, specifically **Node Splitting**, when a page exceeds its 4KB capacity.
- Abstract the physical layout into a "Cell" interface for the Virtual Machine.

**Non-Goals:**
- This module does not handle the Buffer Pool (it requests pages from the Pager).
- It does not handle WAL or Rollback Journals (it assumes the Pager provides a consistent view).
- It does not implement SQL parsing or execution logic.

**Invariants:**
- Every page must be exactly 4096 bytes when flushed to the Pager.
- In a Table B-tree, internal nodes store only `rowid` separators and child pointers; leaf nodes store actual row payloads.
- All multi-byte integers on disk must be stored in **Big-Endian** format.

---

## 2. File Structure

Implementation follows this sequence to build from primitive encodings to complex tree structures:

1. `src/include/storage/varint.h`: Varint encoding/decoding macros and prototypes.
2. `src/storage/varint.c`: Implementation of the 1-9 byte variable integer logic.
3. `src/include/storage/page_format.h`: Struct definitions for Page Headers and Cell Pointers.
4. `src/storage/page_format.c`: Logic for initializing pages and managing the slotted heap.
5. `src/include/storage/btree.h`: The B-tree manager and Cursor definitions.
6. `src/storage/btree.c`: Implementation of Search, Insert, and the Node Split algorithm.

---

## 3. Complete Data Model

### 3.1 On-Disk Page Layout (Byte-Level)

The page is a 4096-byte buffer. The header occupies the first few bytes, followed by the cell pointer array.

| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | 1 byte | `page_type` | `0x05`: Table Internal, `0x0D`: Table Leaf. |
| 0x01 | 2 bytes | `freeblock_ptr` | Offset to the first free block in the page heap (Big-Endian). |
| 0x03 | 2 bytes | `cell_count` | Number of cells stored in this page (Big-Endian). |
| 0x05 | 2 bytes | `content_start` | Offset to the start of the cell content area (Big-Endian). |
| 0x07 | 1 byte | `fragmented_free` | Number of fragmented free bytes. |
| 0x08 | 4 bytes | `right_child` | (Internal Only) Page ID of the right-most child. |

**Cell Pointer Array**: Starts at offset `0x08` (for leaves) or `0x0C` (for internal nodes). It is an array of `uint16_t` offsets pointing to the start of cells in the content area.


![Slotted Page Binary Format](./diagrams/tdd-diag-m5-slotted-layout.svg)


![Slotted Page Memory Map](./diagrams/m5-layout.svg)


### 3.2 Cell Formats (Binary Payload)

#### Table B-tree Leaf Cell (Data Node)
| Field | Type | Description |
| :--- | :--- | :--- |
| `payload_size` | Varint | Total size of the record data. |
| `rowid` | Varint | The 64-bit primary key. |
| `payload` | Bytes | Serialized SQL row (column values). |

#### Table B-tree Internal Cell (Navigation Node)
| Field | Type | Description |
| :--- | :--- | :--- |
| `left_child` | uint32_t | Page ID of the child containing keys ≤ `rowid`. |
| `rowid` | Varint | The separator key. |

### 3.3 Memory Structures
```c
typedef struct {
    uint8_t type;
    uint16_t cell_count;
    uint16_t content_start; // Marks the "top" of the bottom heap
    uint32_t right_child;   // Only for Internal
    uint8_t* data;          // Pointer to the 4096-byte buffer from Pager
} BtreePage;

typedef struct {
    uint32_t payload_size;
    int64_t rowid;
    uint8_t* pPayload;      // Pointer into the page buffer
} BtreeCell;
```

---

## 4. Interface Contracts

### 4.1 `int varint_encode(uint8_t* out, uint64_t value)`
- **Purpose**: Encodes a 64-bit integer into 1-9 bytes.
- **Constraints**: `out` must have at least 9 bytes of space.
- **Returns**: Number of bytes written.

### 4.2 `uint32_t page_free_space(BtreePage* p)`
- **Purpose**: Calculate remaining bytes in the "gap" between pointers and content.
- **Formula**: `content_start - (header_size + cell_count * 2)`.

### 4.3 `int btree_insert(Pager* p, uint32_t root_id, int64_t rowid, const uint8_t* data, uint32_t len)`
- **Purpose**: Insert a new row into the B-tree.
- **Logic**:
    - Traverse to leaf.
    - If space exists: Insert cell, update pointers.
    - If no space: Execute `btree_split()`.
- **Errors**: `ERR_PAGE_OVERFLOW`, `ERR_DUPLICATE_KEY`.

---

## 5. Algorithm Specification

### 5.1 Varint Encoding (MSB Flag)
1. If `value <= 0x7F`: Write 1 byte.
2. For larger values: 
   - Use the 8th bit of each byte as a "More" flag (1 = continue, 0 = last byte).
   - The 9th byte is a special case in SQLite: it uses all 8 bits for data.
3. **Logic**:
   ```c
   while (value > 0x7F && count < 8) {
       out[count++] = (value & 0x7F) | 0x80;
       value >>= 7;
   }
   out[count++] = value & 0xFF;
   ```
   *Note: Bytes are written MSB-first.*

### 5.2 B-tree Leaf Search (`find_leaf`)
1. Start at `root_page_id`.
2. Fetch page from Pager.
3. If `page_type == 0x0D` (Leaf): Return this page.
4. If `page_type == 0x05` (Internal):
   - Perform binary search on the `cell_pointer_array`.
   - Find the first cell where `cell.rowid >= search_rowid`.
   - If found: `next_page = cell.left_child`.
   - If not found: `next_page = page.right_child`.
   - Repeat from step 2 with `next_page`.

### 5.3 Node Split Algorithm (`btree_split`)
This is triggered when an `insert` exceeds the `page_free_space`.

1. **Identify the Median**: Find the middle cell in the overflowing page.
2. **Create Sibling**: Request a new page `P_new` from the Pager. Initialize it with the same `page_type`.
3. **Redistribute**:
   - Move all cells after the median from the old page `P_old` to `P_new`.
   - Update `P_old->cell_count` and `P_new->cell_count`.
   - Adjust `content_start` for both.
4. **Promote**:
   - If `P_old` was the Root:
     - Create a new Root page `P_root_new`. Set type to Internal.
     - Add a cell to `P_root_new` with `rowid = median.rowid` and `left_child = P_old`.
     - Set `P_root_new->right_child = P_new`.
   - If `P_old` was not Root:
     - Insert `(median.rowid, P_old)` into the parent of `P_old`.
     - Update parent's `right_child` or neighboring pointer to `P_new`.


![B-tree Node Split Sequence](./diagrams/tdd-diag-m5-split-steps.svg)


![B-tree Split Sequence](./diagrams/m5-split-visual.svg)


---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `PageOverflow` | `btree_insert` | Trigger `btree_split()`. If split fails (no disk), abort. | No (Handled internally) |
| `DuplicateRowID` | `btree_insert` | Return error to VM; roll back transaction. | Yes: "UNIQUE constraint failed" |
| `InvalidPageType` | `fetch_page` | Pager/Btree verify header byte. Panic if unknown. | Yes: "Database disk image is malformed" |
| `CellTooLarge` | `btree_insert` | If cell > 4KB, trigger Overflow Page logic (or reject in simple build). | Yes: "Row size too large" |

---

## 7. Implementation Sequence

### Phase 1: Binary Primitives (Estimated: 2-3 Hours)
- Implement `varint_encode` and `varint_decode`.
- Implement Big-Endian read/write utilities (`get_u16`, `put_u16`, `get_u32`).
- **Checkpoint**: Encode the number `150`. Verify it occupies 2 bytes: `0x81 0x16`. Decode it back.

### Phase 2: Slotted Page Management (Estimated: 4-5 Hours)
- Implement `page_init`, `page_add_cell`, and `page_get_cell`.
- Logic for shifting `content_start` and updating the `cell_pointer_array`.
- **Checkpoint**: Initialize a 4KB buffer. Add 5 cells of varying sizes. Verify `cell_count` is 5 and the pointers correctly resolve to the payloads.

### Phase 3: B-tree Logic (Estimated: 6-8 Hours)
- Implement `btree_find_leaf` (Recursive or Iterative).
- Implement `btree_insert` with the split logic.
- **Checkpoint**: Insert 100 rows with sequential IDs into a tree. Verify that the tree height grows from 1 to 2 once the first page fills.

---

## 8. Test Specification

### 8.1 Happy Path: Sequential Insert
- **Input**: Insert 10 rows with IDs 1-10.
- **Expected**: `page_get_cell(0)` returns ID 1, `page_get_cell(9)` returns ID 10. `cell_count` is 10.

### 8.2 Edge Case: Random Insertion
- **Input**: Insert IDs 10, 5, 20, 1.
- **Expected**: The `cell_pointer_array` must be kept sorted by `rowid`. `page_get_cell(0)` must point to ID 1 even though it was inserted last.

### 8.3 Failure Case: Page Full
- **Input**: Insert 500 rows of 100 bytes each.
- **Expected**: `btree_insert` should successfully split pages. Total pages in the file should increase. A `find(250)` must successfully navigate from root to the correct leaf.

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Search Depth** | $O(\log_{100} N)$ | For 1M rows, verify `btree_find_leaf` fetches $\le$ 4 pages. |
| **Serialization** | < 10μs / row | Measure time for `page_add_cell` for 1000 iterations. |
| **Space Efficiency** | > 85% Fill | Measure total data size / (total pages * 4096) after many random inserts. |

---

## 10. Concurrency Specification

**Model: Single-Writer, Multi-Reader (SWMR)**

1. **Reader Path**:
   - Acquires `SHARED` lock on the B-tree Root.
   - As it descends, it unpins parent pages once the child is pinned (Crabbing).
2. **Writer Path**:
   - Acquires `EXCLUSIVE` lock on the B-tree.
   - During a split, it must hold pins on the Parent, the Full Page, and the New Sibling.
   - **Lock Ordering**: Always lock Page IDs in ascending order or Parent-then-Child to avoid deadlocks.

**Torn Page Protection**: 
The B-tree layer relies on the **Pager** to handle atomicity. However, if a split is interrupted, the "Right Child" pointer in the parent must be updated **last** to ensure that an old version of the tree remains reachable until the new pages are fully flushed.

---

## 11. Wire Format: Varint Detail

The varint layout is a sequence of 1-9 bytes.
- Bytes 1-8: `1` in MSB means "another byte follows". `0` in MSB means "this is the last byte". Use the remaining 7 bits for data.
- Byte 9: All 8 bits are data.

**Example: The number 1000**
- Binary: `11 11101000` (10 bits)
- Grouped in 7s: `0000111` and `1101000`
- Add MSB flags: `1 0000111` (0x87) and `0 1101000` (0x68)
- Hex Output: `0x87 0x68`


![SQLite Varint Bit Layout](./diagrams/tdd-diag-m5-varint-layout.svg)

<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m6 -->
# Module Design: SELECT Execution & DML (build-sqlite-m6)

## 1. Module Charter

The **SELECT Execution & DML** module is the functional bridge between the high-level Virtual Machine (VDBE) and the low-level B-tree storage engine. It provides the mechanisms required to navigate B-tree structures, deserialize binary records into typed registers, and modify data while maintaining structural integrity and constraints. 

**Core Responsibilities:**
- Implement the **Cursor** abstraction for stateful B-tree navigation (Forward/Backward scans).
- Provide a **Record Deserializer** capable of parsing the SQLite-style variable-length binary format into VDBE `Value` objects.
- Execute **Data Manipulation Language (DML)** operations: `INSERT` (row creation), `UPDATE` (modification via delete-and-insert), and `DELETE` (removal with space reclamation).
- Enforce **NOT NULL** and basic schema constraints during the write path.
- Coordinate with the Pager to manage **Lock Upgrades** (moving from Shared to Reserved/Exclusive) during write transactions.

**Non-Goals:**
- **Secondary Index Maintenance**: Handled in Milestone 7.
- **Complex Query Planning**: Handled in Milestone 8.
- **Transaction Recovery**: Handled by the Pager/Journal in Milestone 9/10.

**Invariants:**
- A cursor must always point to a valid cell, the EOF marker, or be explicitly marked as "Invalid" after a B-tree mutation.
- Column extraction must be zero-copy where possible (referencing the Buffer Pool frames directly for strings/blobs).
- No write operation can proceed without first upgrading the database lock state to `RESERVED`.

---

## 2. File Structure

Implementation follows this sequence to ensure the navigation API is solid before building complex DML logic:

1. `src/include/vdbe/cursor.h`: Definition of the `VdbeCursor` state and navigation constants.
2. `src/vdbe/cursor.c`: Implementation of `CursorNext`, `CursorRewind`, and `CursorSeek`.
3. `src/include/vdbe/record.h`: Serialization/Deserialization logic for the SQLite Record Format.
4. `src/vdbe/record.c`: Logic for parsing Serial Types and calculating byte offsets within cells.
5. `src/vdbe/dml_ops.c`: Integration of VDBE opcodes (`OP_Column`, `OP_Insert`, `OP_Delete`) with the B-tree layer.

---

## 3. Complete Data Model

### 3.1 VdbeCursor Structure
The cursor tracks the "Current Position" of a scan within a Table B-tree.

| Field | Type | Description |
| :--- | :--- | :--- |
| `btree_root` | `uint32_t` | The Page ID of the B-tree root this cursor is exploring. |
| `curr_page` | `uint32_t` | The Page ID of the leaf node currently being read. |
| `cell_idx` | `uint16_t` | The index in the slotted page's cell pointer array. |
| `is_eof` | `bool` | Set to true when `Next()` moves past the last cell in the last leaf. |
| `is_writable` | `bool` | Flag indicating if the cursor was opened for DML operations. |
| `cache_status` | `uint8_t` | Metadata for optimizing repeated column access on the same row. |

### 3.2 The Record Format (Wire Format)
Data is stored as a "Record" within the B-tree cell payload. This format uses a header to define the types of all columns before the data begins.

**Record Layout:**
- **Header Size**: Varint (Total bytes in the header, including this varint).
- **Serial Types**: A sequence of Varints (One per column).
- **Data**: The actual column values, concatenated.

**Serial Type Mapping (The SQLite Protocol):**
| Serial Type | Content Size | Meaning |
| :--- | :--- | :--- |
| `0` | 0 bytes | NULL |
| `1` | 1 byte | 8-bit signed integer |
| `2` | 2 bytes | 16-bit signed integer |
| `3` | 3 bytes | 24-bit signed integer |
| `4` | 4 bytes | 32-bit signed integer |
| `6` | 8 bytes | 64-bit signed integer |
| `7` | 8 bytes | IEEE 754 64-bit float |
| `8` | 0 bytes | The integer 0 (constant) |
| `9` | 0 bytes | The integer 1 (constant) |
| `N >= 12, even` | `(N-12)/2` | BLOB |
| `N >= 13, odd` | `(N-13)/2` | UTF-8 String |

{{DIAGRAM:tdd-diag-m6-record-layout}}
*(Visual: [HeaderSize][ST1][ST2][ST3][Data1][Data2][Data3]. Show offsets: Data2 starts at HeaderSize + Size(Data1))*

---

## 4. Interface Contracts

### 4.1 `int cursor_move_to(VdbeCursor* pCur, int64_t rowid)`
- **Purpose**: Perform a point-lookup in the Table B-tree.
- **Logic**: Uses B-tree traversal to find the leaf and slot for the target `rowid`.
- **Return**: `SQLITE_OK` if found, `SQLITE_NOTFOUND` if missing.

### 4.2 `VdbeValue record_get_column(const uint8_t* pPayload, int col_idx)`
- **Purpose**: Extract a specific column from a raw payload.
- **Algorithm**: See Section 5.1.
- **Constraints**: Must handle cases where the record has fewer columns than the schema (returns NULL).

### 4.3 `int op_insert(Vdbe* p, int cursor_idx, int reg_payload, int reg_rowid)`
- **Purpose**: VDBE opcode handler for inserting a row.
- **Logic**:
    - Retrieve binary payload from `reg_payload`.
    - Retrieve `rowid` from `reg_rowid`.
    - Call `btree_insert`.
- **Errors**: `ConstraintViolation` (Duplicate rowid).

---

## 5. Algorithm Specification

### 5.1 Record Deserialization (The Column Skip)
Since columns are variable-length, finding Column $N$ requires scanning the header.

1. **Parse Header Size**: Read first varint to find where data begins (`data_offset`).
2. **Scan Serial Types**:
   - Initialize `current_data_ptr = data_offset`.
   - For `i` from 0 to `col_idx - 1`:
     - Determine `size` of column `i` based on its Serial Type.
     - `current_data_ptr += size`.
3. **Extract Value**:
   - Read Serial Type for `col_idx`.
   - If `type == 0`: return `Value(NULL)`.
   - If `type == 1`: return `Value((int8_t)pPayload[current_data_ptr])`.
   - ... (Repeat for all types).
   - If `type >= 13`: return `Value(string_view(pPayload + current_data_ptr, (type-13)/2))`.

### 5.2 The Two-Pass Delete Algorithm
To prevent cursor corruption during a `DELETE FROM ... WHERE ...` scan:

1. **Pass 1 (Identify)**:
   - Open Scan Cursor.
   - For each row:
     - Evaluate `WHERE` clause logic in VM.
     - If true: Store `rowid` in a temporary `vector<int64_t>`.
2. **Pass 2 (Mutate)**:
   - For each `rowid` in the vector:
     - Perform a point-lookup (`SeekRowid`).
     - Call `btree_delete` on that specific row.
3. **Invalidation**: Mark all active cursors on this table as "Needs Reset" if they were pointing to the deleted rows.


![DML Write Pipeline](./diagrams/tdd-diag-m6-dml-flow.svg)

*(Visual: Flowchart showing Evaluation Loop -> ID Collection -> Batch Deletion)*

---

## 6. State Machine: VdbeCursor Lifecycle

A cursor's state determines which opcodes are valid.

| State | Transition Event | Next State | Action |
| :--- | :--- | :--- | :--- |
| **UNINITIALIZED**| `OpenRead` / `OpenWrite`| **BEFORE_FIRST** | Allocate cursor, pin root page. |
| **BEFORE_FIRST** | `Rewind` / `Seek` | **VALID_ROW** | Navigate to first/specific cell. |
| **VALID_ROW** | `Next` (Success) | **VALID_ROW** | Advance `cell_idx`, handle page boundaries. |
| **VALID_ROW** | `Next` (End of Table) | **EOF** | Mark `is_eof = true`. |
| **EOF** | `Rewind` | **VALID_ROW** | Reset to beginning. |
| **ANY** | `BtreeMutation` | **INVALID** | Clear cached page pointers. |

**Illegal Transitions**:
- Calling `Column` on a cursor in `EOF` state.
- Calling `Insert` on a cursor opened with `OpenRead`.
- Calling `Next` on an `UNINITIALIZED` cursor.


![Cursor State Machine](./diagrams/tdd-diag-m6-cursor-sm.svg)


---

## 7. Locking & Concurrency: The Upgrade

This module introduces the **Lock Upgrade** logic required for DML.

1. **Shared Lock**: Acquired by `OpenRead`. Multiple readers permitted.
2. **Reserved Lock**: Acquired when the first `Insert/Update/Delete` opcode is encountered. 
   - Only one process can hold `RESERVED`. 
   - New readers can still enter `SHARED`.
3. **Exclusive Lock**: Acquired during the final `PagerFlush` (Commit). No readers allowed.

**Implementation Path**:
- When `OP_Insert` is called:
  - If `pager->lock_state < RESERVED`:
    - Attempt `pager_upgrade_lock(RESERVED)`.
    - If failed (another writer exists): Return `SQLITE_BUSY`.

---

## 8. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `ConstraintViolation` | `OP_Insert` / `MakeRecord` | Abort VM loop, trigger Rollback. | Yes: "NOT NULL constraint failed" |
| `TypeMismatch` | `PredicateEvaluator` | Return `NULL` or Error based on strictness. | Yes: "Cannot compare INT and BLOB" |
| `NoSuchTable` | `OP_OpenRead` | Abort compilation/execution. | Yes: "Table 'x' not found" |
| `CorruptRecord` | `RecordDeserializer` | Return `INTERNAL_ERROR`, stop VM. | Yes: "Database file is malformed" |

---

## 9. Implementation Sequence

### Phase 1: Cursor Navigation (4-5 Hours)
- Implement `VdbeCursor` struct and `cursor_rewind`.
- Implement B-tree leaf-to-leaf logic in `cursor_next` (handling the "Right Child" transition).
- **Checkpoint**: Open a cursor on a 3-page B-tree. Manually call `Next()` and verify it visits every row in order.

### Phase 2: Record Deserialization (3-4 Hours)
- Implement `RecordHeader` parsing.
- Implement `record_get_column` with the "skip" algorithm.
- **Checkpoint**: Provide a raw byte buffer containing `[Header: 3, Types: 1, 1, Data: 0x05, 0x0A]`. Extract Col 1, verify it returns integer `10`.

### Phase 3: DML Integration (3 Hours)
- Implement `OP_Column`, `OP_Insert`, and `OP_Delete` logic.
- Integrate Lock Upgrade logic in the Pager/VDBE boundary.
- **Checkpoint**: Run `INSERT INTO t VALUES (1, 'test')` and then `SELECT * FROM t`. Verify the row is persistent.

---

## 10. Test Specification

### 10.1 Happy Path: Full Table Scan
- **Input**: Table with 10,000 rows. `SELECT * FROM table;`
- **Expected**: `vdbe_step` returns `VDBE_ROW` exactly 10,000 times. Time taken < 100ms.

### 10.2 Edge Case: Column Overflow
- **Input**: Table has 3 columns. Query asks for `Column(idx=5)`.
- **Expected**: `record_get_column` returns `VAL_NULL` safely without reading out of bounds.

### 10.3 Failure Case: Lock Conflict
- **Setup**: Process A holds a `RESERVED` lock.
- **Action**: Process B attempts an `INSERT`.
- **Expected**: `op_insert` returns `SQLITE_BUSY`.

---

## 11. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Row Scan** | 10k rows < 100ms | Measure `SELECT COUNT(*)` on 10k cached rows. |
| **Column Extract** | < 200ns | Measure time to extract the 10th column of a wide row. |
| **Record Encoding**| < 1μs | Measure time to convert 5 registers into one binary B-tree cell. |

---

## 12. Record Encoding Byte-Level Detail

When the VM prepares an `INSERT`, it uses `OP_MakeRecord`.

1. **Calculate Header Size**: Sum of varint sizes for each column's type.
2. **Buffer Allocation**: `Total = HeaderSize + Sum(DataSizes)`.
3. **Write Header**:
   - Write `HeaderSize` as varint.
   - Write `SerialType` for each register.
4. **Write Data**:
   - Append raw bytes for each register in order.
   - *Note*: Integers must be Big-Endian.


![Example Record Encoding](./diagrams/m6-record-example.svg)

<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m7 -->
# Module Design: Secondary Indexes (build-sqlite-m7)

## 1. Module Charter

The **Secondary Index** module manages non-clustered B+tree structures that enable high-speed data retrieval based on column values other than the primary `rowid`. It functions as a mapping layer that translates an indexed value (or a composite tuple of values) into the corresponding `rowid` for a "Double Lookup" into the main Table B-tree. This module is responsible for the synchronous maintenance of index integrity during all DML operations (`INSERT`, `UPDATE`, `DELETE`), ensuring that indexes never become stale. It implements the **B+tree** variant where data resides exclusively in leaf nodes and leaves are horizontally linked to optimize range scans. Finally, it provides the primary mechanism for enforcing **UNIQUE** constraints at the storage level.

**Core Responsibilities:**
- Implement the **Index B+tree** page format, distinguishing between internal navigation nodes and leaf data nodes.
- Manage **Index Cell** serialization, which combines indexed column values with the `rowid` to ensure unique, sortable keys.
- Implement **Synchronous Maintenance** hooks that trigger on every table modification.
- Provide B+tree traversal algorithms for **Equality Lookups** ($O(\log N)$) and **Range Scans** ($O(N)$ following leaf pointers).
- Enforce **UNIQUE** constraints by performing a "pre-check" search before insertion.

**Non-Goals:**
- This module does not decide *when* to use an index; that is the responsibility of the Query Planner (Milestone 8).
- It does not handle the primary table data (clustered storage).
- It does not manage its own memory (uses the Buffer Pool from Milestone 4).

**Invariants:**
- Every entry in a secondary index B+tree must contain a `rowid` pointing to a valid row in the associated Table B-tree.
- Leaf nodes in the Index B+tree must be linked in ascending key order via the `next_leaf` pointer.
- For `UNIQUE` indexes, no two cells may have the same indexed value (NULLs excepted per SQL standard).

---

## 2. File Structure

Implementation follows this sequence to establish the B+tree variant before integrating with the VM:

1. `src/include/storage/index_format.h`: Byte-level definitions for Index Page headers and Cell layouts.
2. `src/storage/index_page.c`: Logic for managing Index B+tree pages (splitting, cell insertion).
3. `src/include/storage/index_manager.h`: High-level API for index creation and maintenance.
4. `src/storage/index_manager.c`: The "Maintenance Hook" logic and Unique Constraint checks.
5. `src/vdbe/index_ops.c`: Implementation of index-specific VDBE opcodes (`IdxGE`, `IdxRowid`, `IdxDelete`).

---

## 3. Complete Data Model

### 3.1 Index B+tree Page Layout (Binary Format)

Index pages use the same 4096-byte frame as tables but utilize different `page_type` identifiers and header fields.

| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | 1 byte | `page_type` | `0x02`: Index Internal, `0x0A`: Index Leaf. |
| 0x01 | 2 bytes | `freeblock_ptr` | Offset to the first free block (Big-Endian). |
| 0x03 | 2 bytes | `cell_count` | Number of index entries in this page. |
| 0x05 | 2 bytes | `content_start` | Offset to the start of the cell heap. |
| 0x07 | 1 byte | `frag_free` | Fragmented free bytes count. |
| 0x08 | 4 bytes | `next_leaf` | **(Leaf Only)** Page ID of the next leaf in sorted order. |
| 0x08 | 4 bytes | `right_child` | **(Internal Only)** Page ID of the right-most child. |

{{DIAGRAM:tdd-diag-m7-index-layout}}

![Index Page Architecture](./diagrams/m7_page_layout.svg)


### 3.2 Index Cell Layout (The "Search Key")

In a secondary index, the "Key" we search for must be unique to maintain B+tree stability. SQLite achieves this by appending the `rowid` to the user's indexed columns.

#### Index Leaf Cell
| Field | Type | Description |
| :--- | :--- | :--- |
| `payload_size` | Varint | Total size of the key + rowid record. |
| `record_data` | Bytes | **The Record**: Contains the indexed columns followed by the `rowid`. |

*Example*: An index on `Name` for a row with `Name='Alice', RowID=5` becomes a record: `[Header: 0x03, Type1: String(5), Type2: Int(5)][Data: 'Alice', 5]`.

#### Index Internal Cell
| Field | Type | Description |
| :--- | :--- | :--- |
| `left_child` | uint32_t | Page ID of the child tree. |
| `payload_size` | Varint | Size of the separator key. |
| `record_data` | Bytes | The separator record (IndexValue + RowID). |

### 3.3 Key Comparison logic
Indices require a strict comparison function.
- **NULLs**: In SQL, NULL is considered "smaller" than all other values.
- **Types**: Comparison must follow Type Affinity (Integer < Float < String < Blob).
- **Tie-Breaking**: If the indexed values are identical (e.g., two people named "Smith"), the `rowid` (the last element in the record) acts as the tie-breaker to ensure a deterministic sort order.

---

## 4. Interface Contracts

### 4.1 `int index_insert(IndexManager* mgr, uint32_t root_id, Value* cols, int64_t rowid, bool is_unique)`
- **Purpose**: Inserts a new entry into the B+tree.
- **Logic**:
    - Serialize `cols + rowid` into a Record.
    - If `is_unique`, perform `index_find(root_id, cols)` first.
    - Traverse to the correct leaf and insert.
- **Errors**: `ERR_UNIQUE_VIOLATION` if `is_unique` and key exists.

### 4.2 `int index_cursor_seek(IndexCursor* pCur, Value* search_key, int op)`
- **Purpose**: Position an index cursor for range or equality scans.
- **Logic**:
    - Use B+tree traversal to find the first cell $\ge$ `search_key`.
    - Set cursor to `is_eof` if no such key exists.
- **Operations**: `SEEK_EQ` (Equality), `SEEK_GE` (Greater-Equal).

### 4.3 `int64_t index_cursor_rowid(IndexCursor* pCur)`
- **Purpose**: Extract the `rowid` from the current index entry.
- **Return**: The 64-bit RowID used for the double-lookup.

---

## 5. Algorithm Specification

### 5.1 The "Double Lookup" Query Flow
Executing `SELECT * FROM table WHERE col = 'val'` using an index:

1. **VDBE Instruction**: `IdxGE(cursor=index_csr, label=end, reg=target_val)`.
2. **Implementation**:
   - `index_cursor_seek` finds the first entry in the B+tree where `IndexValue >= 'val'`.
3. **Loop Body**:
   - `IdxRowid(cursor=index_csr, reg=r1)`: Extracts the `rowid` from the index record.
   - `SeekRowid(cursor=table_csr, reg=r1)`: Uses the B-tree implementation from M5 to jump to the actual row in the main table.
   - `Column(cursor=table_csr, ...)`: Extracts the full data.
   - `Next(cursor=index_csr, label=loop_start)`: Advances to the next cell in the index leaf.
4. **Range Check**: If the next index cell has a value `> 'val'`, the loop terminates (for equality).


![Double-Lookup Sequence](./diagrams/tdd-diag-m7-index-lookup.svg)


![Double Lookup Sequence](./diagrams/m7_double_lookup.svg)


### 5.2 B+Tree Leaf Linking & Range Scan
Unlike the Table B-tree (M5), the Index B+tree must support fast horizontal traversal.

1. **During `index_split`**:
   - When Leaf `L1` splits into `L1` and `L2`:
   - `L2->next_leaf = L1->next_leaf`.
   - `L1->next_leaf = L2_PageID`.
2. **During `index_cursor_next`**:
   - `cursor->cell_idx++`.
   - If `cursor->cell_idx >= page->cell_count`:
     - `next_id = page->header.next_leaf`.
     - If `next_id == 0`: `cursor->is_eof = true`.
     - Else: Fetch `next_id`, set `cursor->cell_idx = 0`.

### 5.3 Maintenance: The Write Tax
Every Table DML operation must trigger index updates.

**Algorithm: `table_insert_with_indexes(Table* t, Row* r, int64_t rowid)`**
1. Insert `r` into `t->btree` at `rowid`.
2. For each `Index* idx` in `t->indexes`:
   - `Value* vals = extract_columns(r, idx->col_indices)`.
   - `index_insert(idx, vals, rowid)`.
   - If `index_insert` returns `ERR_UNIQUE_VIOLATION`:
     - **Rollback**: Delete `r` from `t->btree` (and any previous indexes).
     - Return error to VM.

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `UniqueConstraintFailed` | `index_insert` | Abort DML, trigger transaction rollback. | Yes: "UNIQUE constraint failed: table.col" |
| `StaleIndexEntry` | `SeekRowid` | If RowID found in Index but not in Table, panic. | Yes: "Database malformed: orphaned index" |
| `IndexPageFull` | `index_page` | Trigger `index_split`. | No |
| `TypeMismatch` | `index_compare` | Apply SQL conversion rules or return error. | Yes: "Comparison error" |

---

## 7. Implementation Sequence

### Phase 1: B+Tree Leaf Linking (Estimated: 4-5 Hours)
- Modify the `page_split` logic from Milestone 5 to support the `next_leaf` pointer at offset 0x08.
- Implement the horizontal `cursor_next` logic that follows the `next_leaf` pointer.
- **Checkpoint**: Insert enough index entries to cause 3 splits. Manually iterate from the first leaf using only `next_leaf` pointers and verify all entries are present in sorted order.

### Phase 2: Unique Constraint Enforcement (Estimated: 2-3 Hours)
- Implement `index_find(root, values)` which searches the B+tree.
- Add a check in `index_insert` that fails if a match is found and `is_unique` is true.
- **Checkpoint**: Attempt to insert the same value twice into a UNIQUE index. Verify the second call returns `ERR_UNIQUE_VIOLATION`.

### Phase 3: VDBE Integration (Estimated: 4 Hours)
- Implement `OP_IdxGE`, `OP_IdxRowid`, and `OP_IdxNext`.
- Update the Bytecode Compiler to detect when an index can be used (Simple equality check for now).
- **Checkpoint**: Run `EXPLAIN SELECT * FROM t WHERE indexed_col = 5`. Verify it uses `IdxGE` instead of `Rewind`.

---

## 8. Test Specification

### 8.1 Happy Path: Equality Lookup
- **Input**: Table with index on `age`. Rows with ages `[20, 21, 22, 21]`.
- **Query**: `SELECT rowid FROM t WHERE age = 21`
- **Expected**: Index search finds RowID for first 21, `IdxNext` finds RowID for second 21, then terminates. Total 2 RowIDs returned.

### 8.2 Edge Case: Indexing NULLs
- **Input**: Insert `NULL` into an indexed column.
- **Expected**: Index B+tree should store the NULL at the very beginning of the first leaf. `WHERE col IS NULL` should successfully use the index.

### 8.3 Failure Case: Maintenance Failure
- **Action**: Manually delete a row from the Table B-tree but *not* the Index.
- **Query**: Use the index to find that row.
- **Expected**: `SeekRowid` fails to find the row in the table. The system must report a corruption error.

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Point Lookup** | $O(\log N)$ | Measure time for 1M row table; search must take < 5 page fetches. |
| **Range Scan** | > 500k entries/sec | Measure time to scan 10% of a 1M entry index. |
| **Maintenance Overhd**| < 2x write time | Compare `INSERT` time with 0 indexes vs 1 index. |

---

## 10. Concurrency: Synchronous Updates

In this intermediate build, we enforce **Atomic Multi-Tree Updates**.

1. When a transaction modifies a table, it must hold an `EXCLUSIVE` lock on the entire database file.
2. The index update happens within the same atomic block as the table update.
3. If the index update fails (e.g., Unique constraint), the Pager's **Rollback Journal** (M9) will be used to undo the change to the main table.

**Invariant**: No reader will ever see a state where the Table B-tree is updated but the Index B+tree is not.

---

## 11. Wire Format: Index Cell Detail

The `record_data` in an index cell is a standard SQL Record (from M6).

**Example for Index on (City, Zip):**
Row: `City='London', Zip=10001, RowID=500`
1. **Values**: `['London', 10001, 500]`
2. **Serial Types**: `[13+(6*2)=25, 4 (32-bit int), 4 (32-bit int)]`
3. **Record Header**: `[Size: 4, 25, 4, 4]`
4. **Record Data**: `['L','o','n','d','o','n', 0x00, 0x00, 0x27, 0x11, 0x00, 0x00, 0x01, 0xF4]`


![Index Cell Layout](./diagrams/tdd-diag-m7-index-cell.svg)

<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m8 -->
# Module Design: Query Planner & Statistics (build-sqlite-m8)

## 1. Module Charter

The **Query Planner & Statistics** module is the intelligence layer of the RDBMS. It acts as a strategic advisor to the Bytecode Compiler, transforming a declarative SQL AST into the most efficient physical execution plan. By utilizing a **Cost-Based Optimizer (CBO)**, it evaluates multiple "Access Paths" (e.g., Table Scan vs. Index Scan) and selects the one that minimizes predicted Disk I/O. 

**Core Responsibilities:**
- **Statistics Collection (`ANALYZE`)**: Persistently store metadata about table volume and column cardinality in a system catalog.
- **Selectivity Estimation**: Calculate the "filtering power" of `WHERE` clause predicates based on available statistics or heuristics.
- **Cost Modeling**: Quantify the predicted performance of a plan candidate using a weighted formula that penalizes random I/O.
- **Join Optimization**: Determine the optimal nesting order for multi-table joins to minimize intermediate result sets (the "Funnel Principle").
- **Plan Generation**: Emit a refined set of high-level execution directives that the VDBE Compiler uses to generate final bytecode.

**Non-Goals:**
- This module does not perform physical data access; it queries the metadata catalog.
- It does not handle syntax validation (assumed valid from Milestone 2).
- It does not implement low-level B-tree splits (handled in Milestone 5/7).

**Invariants:**
- The Planner must always fall back to a Full Table Scan if no suitable indexes exist or if statistics are missing.
- Cost estimates must be reproducible and deterministic for a given set of statistics.
- System statistics tables must be accessible via standard internal B-tree cursors.

---

## 2. File Structure

Implementation follows this sequence to ensure the data source (Stats) exists before the logic (Planner) is built:

1. `src/include/optimizer/stats.h`: Definition of the `sqlite_stat1` equivalent internal structures.
2. `src/optimizer/analyzer.c`: Implementation of the `ANALYZE` command (B-tree sampling logic).
3. `src/include/optimizer/cost_model.h`: Cost constants and the `PlanCandidate` structure.
4. `src/optimizer/selectivity.c`: Logic for estimating row counts from predicates (`=`, `<`, `>`, `BETWEEN`).
5. `src/optimizer/planner.c`: The core search algorithm for selecting access paths and join orders.

---

## 3. Complete Data Model

### 3.1 Statistics Storage (System Catalog)
Statistics are stored in an internal table named `sys_stats`. This table is a standard Table B-tree.

| Column | Type | Description |
| :--- | :--- | :--- |
| `tbl` | TEXT | Name of the table. |
| `idx` | TEXT | Name of the index (NULL if table-level stats). |
| `stat` | TEXT | A space-separated string of integers: `nRow nDist1 nDist2 ...` |

**Example**: `users | idx_age | 1000000 500`
- `1000000`: Total rows in the table.
- `500`: Average number of rows per distinct value in the index. (Selectivity = 500 / 1,000,000).

### 3.2 In-Memory Optimizer Structures

```c
typedef struct {
    uint64_t row_count;      // Total rows (nRow)
    uint32_t page_count;     // Total pages in B-tree (from Pager)
    double avg_row_size;     // bytes
} TableStats;

typedef struct {
    char* index_name;
    uint64_t distinct_values; // Cardinality
    double selectivity;       // range [0.0 - 1.0]
    uint32_t root_page;
} IndexStats;

typedef enum {
    PATH_SCAN,
    PATH_INDEX
} AccessPathType;

typedef struct {
    AccessPathType type;
    IndexStats* index;       // NULL if Table Scan
    double est_cost;         // The calculated "Cost"
    uint64_t est_rows;       // Predicted output rows
} PlanCandidate;
```


![Cost Calculation Data Flow](./diagrams/tdd-diag-m8-stats-model.svg)

*(Visual: Relationship diagram: Table -> has multiple IndexStats -> used by PlanCandidate to calculate Cost)*

---

## 4. Interface Contracts

### 4.1 `void optimizer_analyze_table(const char* table_name)`
- **Purpose**: Scans the table B-tree and all associated index B+trees to update `sys_stats`.
- **Constraint**: Must hold a `SHARED` lock on the table. For massive tables, it may sample every Nth page.
- **Output**: Updates/Inserts rows in the `sys_stats` B-tree.

### 4.2 `double estimate_selectivity(Predicate* pred, IndexStats* stats)`
- **Purpose**: Predicts what fraction of the table matches a WHERE clause.
- **Rules**:
    - Equality (`=`): If index exists, `1.0 / stats->distinct_values`. Else, default `0.1`.
    - Range (`<`, `>`): Default `0.33` (Rule of thumb).
    - In-list (`IN (...)`): `num_elements / stats->distinct_values`.
- **Return**: A double between `0.0` and `1.0`.

### 4.3 `PlanCandidate optimizer_find_best_path(TableContext* tbl, WhereClause* where)`
- **Purpose**: Iterates through all available indexes and compares costs.
- **Algorithm**: See Section 5.2.
- **Error**: Returns `PATH_SCAN` as default if errors occur.

---

## 5. Algorithm Specification

### 5.1 The `ANALYZE` Procedure (Phase 1)
To populate statistics without blocking the DB for hours:

1. **Table Scan**:
   - Initialize `row_counter = 0`.
   - Iterate through the Table B-tree leaf sequence.
   - For every row, increment `row_counter`.
2. **Index Scan**:
   - For each index:
     - Initialize `distinct_counter = 1`.
     - Iterate through Index B+tree. 
     - Compare current key to `prev_key`. If different, `distinct_counter++`.
3. **Persist**:
   - Calculate `avg_rows_per_value = row_counter / distinct_counter`.
   - Write `row_counter` and `avg_rows_per_value` to `sys_stats` using an internal `INSERT` (VDBE).

### 5.2 The Cost Model (Phase 2)
The planner uses a weighted formula where Random I/O is the primary "currency".

**Constants:**
- `SEQ_PAGE_COST = 1.0` (Cost to read one page sequentially)
- `RAND_PAGE_COST = 4.0` (Cost to seek to a random page - SSD/HDD penalty)
- `CPU_ROW_COST = 0.01` (Cost to process one row in the VM)

**Formula 1: Table Scan Cost**
$$Cost_{scan} = (TablePages \times SEQ\_PAGE\_COST) + (TotalRows \times CPU\_ROW\_COST)$$

**Formula 2: Index Scan Cost (Double Lookup)**
1. `MatchRows = TotalRows * Selectivity(Predicate)`
2. `IndexPagesRead = MatchRows / RowsPerPageIndex`
3. `TablePagesRead = MatchRows` (Assuming every match is a random seek)
4. $$Cost_{index} = (IndexPagesRead \times SEQ\_PAGE\_COST) + (TablePagesRead \times RAND\_PAGE\_COST) + (MatchRows \times CPU\_ROW\_COST)$$


![Planner Decision Tree](./diagrams/tdd-diag-m8-planner-flow.svg)

*(Visual: Decision Tree: Start -> For each Index -> Calc Selectivity -> Calc Cost -> Compare with Scan -> Choose Min)*

### 5.3 Join Order Optimization (Greedy Approach)
For a query `A JOIN B JOIN C`:

1. Start with the smallest table (lowest `nRow`) as the **Outer Loop**.
2. For the next table, evaluate which one has an index on the join predicate (e.g., `B.id = A.b_id`).
3. If multiple tables have indexes, pick the one where the index scan cost is lowest.
4. If no indexes exist, pick the smallest remaining table to minimize the product of the nested loops.

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `StatisticsStale` | `optimizer_find_best_path` | If `sys_stats` is older than a threshold or missing, use **Heuristics** (default selectivity). | No (Silent fallback) |
| `PlannerTimeout` | `join_optimizer` | If join permutations > 1000, abort exhaustive search and use **Greedy** search. | No (Silent fallback) |
| `InternalStatsCorrupt`| `analyzer` | If `sys_stats` format is unreadable, delete the stats row and use heuristics. | Yes (Warning) |
| `DivisionByZero` | `selectivity` | If `distinct_values == 0`, assume cardinality of 1. | No |

---

## 7. Implementation Sequence

### Phase 1: `ANALYZE` Infrastructure (Estimated: 5-6 Hours)
- Implement the `sys_stats` table creation in the system catalog.
- Write the `analyzer_scan` logic that traverses B-trees and counts keys.
- **Checkpoint**: Run `ANALYZE users;`. Verify that the `sys_stats` table contains a row for `users` with the correct row count.

### Phase 2: Cost Model & Selectivity (Estimated: 4-5 Hours)
- Implement the `CostModel` functions for Scan and Index paths.
- Implement the `SelectivityEstimator` with default heuristics for equality and ranges.
- **Checkpoint**: Create a table with 10,000 rows and an index. Manually calculate the cost. Verify the code's output matches your calculation.

### Phase 3: Access Path Selection (Estimated: 4 Hours)
- Integrate the planner into the Bytecode Compiler.
- Modify `vdbe_compile_select` to call `find_best_path`.
- **Checkpoint**: Run `EXPLAIN SELECT * FROM t WHERE age = 20`. 
    - Case A (No index): `EXPLAIN` shows `Rewind`.
    - Case B (With index): `EXPLAIN` shows `IdxGE`.

---

## 8. Test Specification

### 8.1 Happy Path: Index Choice
- **Setup**: Table `T` with 1M rows. Index `I` on `col`.
- **Action**: Query `SELECT * FROM T WHERE col = 5`.
- **Expected**: Planner chooses `PATH_INDEX` because $Cost_{index} \approx 4.0 \ll Cost_{scan} \approx 10,000.0$.

### 8.2 Edge Case: High Selectivity (The Scan Choice)
- **Setup**: Table `T` with 1M rows. Index `I` on `status`. Only 2 statuses exist (Active/Inactive).
- **Action**: Query `WHERE status = 'Active'`.
- **Expected**: Selectivity is 0.5. $Cost_{index} = 500,000 \times 4.0 = 2,000,000$. $Cost_{scan} \approx 10,000$. Planner chooses `PATH_SCAN`.

### 8.3 Failure Case: Missing Stats
- **Setup**: New table `T2` created, but `ANALYZE` not yet run.
- **Action**: Query `WHERE col = 5`.
- **Expected**: Planner falls back to **Heuristics**. It assumes `col` is unique enough to justify an index if one exists.

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Planning Latency** | < 1ms | Measure `vdbe_compile` time for a 3-table join. |
| **ANALYZE Speed** | 100k rows/sec | Time the `ANALYZE` command on a medium table. |
| **Join Permutations**| Cap at 1000 | Ensure complex joins (10+ tables) don't hang the compiler. |

---

## 10. Concurrency Specification

**Model: Read-Only Metadata Access**

1. The Planner acquires a `SHARED` lock on the `sys_stats` table at the start of compilation.
2. It reads the necessary rows into a local cache (stack-allocated) to avoid repeated B-tree lookups during plan enumeration.
3. The lock is released as soon as the bytecode is emitted.
4. **Race Condition**: If `ANALYZE` is running while a query is being planned, the Planner sees either the old stats or the new stats (Atomicity is guaranteed by the Pager). Stale stats are acceptable; incorrect stats are not.

---

## 11. Heuristics Table (When Stats are Missing)

| Condition | Default Selectivity |
| :--- | :--- |
| `column = constant` | 0.05 (Assume 20 distinct values) |
| `column > constant` | 0.33 |
| `column BETWEEN c1 AND c2` | 0.10 |
| `column LIKE 'prefix%'` | 0.10 |
| `column IS NULL` | 0.01 |
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m9 -->
# Module Design: Transactions (Rollback Journal) (build-sqlite-m9)

## 1. Module Charter

The **Transactions (Rollback Journal)** module is the guardian of the database's "Durability Soul." Its primary mission is to implement **ACID Atomicity** and **Crash Recovery** using the classic Rollback Journal mechanism. This module ensures that even in the event of a power failure, OS crash, or application panic, the database remains in a consistent state—either fully reflecting a completed transaction or appearing as if the transaction never began. It manages the physical choreography of writing "Undo" data to a separate journal file before any modifications touch the main database.

**Core Responsibilities:**
- Implement a **Multi-state Locking Machine** (Shared, Reserved, Pending, Exclusive) to coordinate concurrent access.
- Manage the creation, population, and synchronization (`fsync`) of the **Rollback Journal** file.
- Intercept the Pager's write requests to ensure original page data is archived before being overwritten (the **Undo Log**).
- Execute the **Atomic Commit Choreography**, strictly ordering I/O operations and hardware flushes.
- Perform **Hot Journal Recovery** on startup, detecting interrupted transactions and restoring the database to its last consistent state.

**Non-Goals:**
- This module does not handle Write-Ahead Logging (WAL); that is Milestone 10.
- it does not implement row-level locking; locking is performed at the database file level.
- It does not handle logical "Undo" for individual SQL statements within a transaction (savepoints), only the full transaction rollback.

**Invariants:**
- The Rollback Journal must be physically flushed to disk (`fsync`) before the first modified page is written to the database file.
- A "Hot Journal" (a journal existing without an accompanying exclusive lock) must always trigger a recovery before any other operation.
- The database file must be `fsync`'d before the journal file is deleted or truncated during a commit.

---

## 2. File Structure

Implementation must proceed in this specific order to establish the locking safety net before implementing file mutations:

1. `src/include/storage/lock_manager.h`: Definition of the 5-state locking machine and OS-specific lock types.
2. `src/storage/lock_manager.c`: Implementation of advisory file locking (using `fcntl` on Unix or `LockFileEx` on Windows).
3. `src/include/storage/journal_format.h`: Byte-level layout for the journal header and page records.
4. `src/storage/journal.c`: Logic for creating the journal and appending original page images.
5. `src/storage/recovery.c`: The startup "Healer" logic that detects and processes hot journals.
6. `src/storage/pager_transaction.c`: Integration with the Pager to trigger journaling on `pager_write()`.

---

## 3. Complete Data Model

### 3.1 The Rollback Journal Wire Format (Byte-Level)

The journal file (`.db-journal`) is a sequence of a header followed by multiple page records.

**Journal Header (Offset 0x00)**
| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | 8 bytes | `magic` | Constant: `0xd9d505f920a1c3b7`. |
| 0x08 | 4 bytes | `page_count` | Number of page records in this journal (Big-Endian). |
| 0x0C | 4 bytes | `nonce` | Random integer used to salt the checksums. |
| 0x10 | 4 bytes | `db_size_pages` | Total size of the DB file in pages before transaction (Big-Endian). |
| 0x14 | 4 bytes | `sector_size` | The sector size of the underlying disk (usually 512). |
| 0x18 | 4 bytes | `page_size` | The page size of the database (usually 4096). |

**Journal Page Record (Appended)**
| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | 4 bytes | `page_no` | The physical page number in the `.db` file. |
| 0x04 | 4096 bytes| `data` | The **original** 4KB content of the page. |
| 0x1004 | 4 bytes | `checksum` | Cumulative checksum of `nonce + page_no + data`. |

### 3.2 Locking State Machine

| State | Readers? | Writers? | Description |
| :--- | :--- | :--- | :--- |
| **UNLOCKED** | Yes | Yes | Initial state. No file locks held. |
| **SHARED** | Many | No | Multiple processes can read. No one can write. |
| **RESERVED** | Many | 1 (Intended)| One process intends to write. New readers can still enter. |
| **PENDING** | Current | 1 (Waiting) | Writer waiting for current readers to exit. No new readers allowed. |
| **EXCLUSIVE** | 0 | 1 (Active) | Writer is modifying the file. Absolute isolation. |


![Database Locking State Machine](./diagrams/tdd-diag-m9-locks.svg)

*(Visual: State diagram showing transitions: Unlocked -> Shared -> Reserved -> Pending -> Exclusive)*

---

## 4. Interface Contracts

### 4.1 `int pager_begin_transaction(Pager* p)`
- **Purpose**: Transitions from `SHARED` to `RESERVED`.
- **Pre-condition**: Pager must already hold a `SHARED` lock.
- **Errors**: `SQLITE_BUSY` if another process holds a `RESERVED` or `EXCLUSIVE` lock.

### 4.2 `int pager_write(Pager* p, uint32_t page_no)`
- **Purpose**: Intercepts a write attempt.
- **Logic**: 
    1. If `page_no` is not already in the journal:
        - Read original page from disk.
        - Append to journal file.
        - Increment `page_count` in journal header (in-memory).
    2. Mark the Buffer Pool frame as `dirty`.
- **Invariants**: Must be called *before* any byte in the memory frame is changed.

### 4.3 `int pager_commit(Pager* p)`
- **Purpose**: Executes the Atomic Commit Dance.
- **Logic**: See Algorithm 5.1.
- **Recovery**: If any `fsync` fails, the transaction is considered failed and must be rolled back.

### 4.4 `int pager_rollback(Pager* p)`
- **Purpose**: Manually undoes changes.
- **Logic**: Reads journal, writes original pages back to DB, deletes journal, drops locks.

---

## 5. Algorithm Specification

### 5.1 The Atomic Commit Dance (The "Dance of Durability")

This procedure is the core of Milestone 9. It ensures that the database is never in a state where a crash causes permanent corruption.

1. **Journal Flush**: Call `fsync()` on the journal file. This ensures the "Undo Log" is safe on the physical platter.
2. **Lock Upgrade (Exclusive)**: Transition from `RESERVED` to `EXCLUSIVE`. This requires waiting for all `SHARED` readers to close their handles. (State moves to `PENDING` then `EXCLUSIVE`).
3. **Database Write**: Write all `dirty` pages from the Buffer Pool into the main `.db` file at their original offsets.
4. **Database Flush**: Call `fsync()` on the `.db` file. The new data is now hardened.
5. **Cleanup (The Commit Point)**:
    - Delete the journal file.
    - **Note**: On some file systems, you must also `fsync()` the parent directory to ensure the deletion is persistent.
6. **Lock Downgrade**: Move back to `SHARED` or `UNLOCKED`.


![Atomic Commit Sequence (Rollback)](./diagrams/tdd-diag-m9-write-seq.svg)

*(Visual: Sequence diagram: Pager -> [fsync Journal] -> [Lock Exclusive] -> [Write DB] -> [fsync DB] -> [Delete Journal])*

### 5.2 Hot Journal Recovery (Startup Procedure)

This algorithm must run whenever a database connection is opened.

1. **Check for Journal**: Does `[db-name]-journal` exist?
2. **Determine "Hotness"**: Attempt to acquire a `SHARED` lock on the database file.
    - If successful, and the journal exists, the journal is **HOT** (it means the previous process died while holding an exclusive lock and left the journal behind).
3. **The Rollback**:
    - Acquire an `EXCLUSIVE` lock on the database (to prevent others from reading during recovery).
    - Read the `page_size` and `db_size_pages` from the journal header.
    - For each page record in the journal:
        - Read `page_no`, `data`, and `checksum`.
        - Verify `checksum(nonce, page_no, data)`. 
        - If checksum fails, the crash happened during journal writing; stop and ignore the rest of the journal.
        - Write `data` into the `.db` file at `page_no * page_size`.
    - Truncate the `.db` file to `db_size_pages * page_size` (removes any appended pages).
    - `fsync()` the `.db` file.
4. **Cleanup**: Delete the journal and release the `EXCLUSIVE` lock.


![Crash Recovery Algorithm](./diagrams/tdd-diag-m9-recovery-flow.svg)

*(Visual: Flowchart: Start -> Journal Exists? -> Can lock Shared? -> If Yes, Rollback -> Delete Journal -> Finish)*

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `JournalCorrupt` | `recovery` | If magic number or checksums fail, the transaction was never started/half-journaled; delete journal safely. | Yes (Warning) |
| `LockDeadlock` | `lock_manager` | If `PENDING` state times out, release all locks and return `BUSY`. | Yes: "Database is locked" |
| `PartialWrite` | `write()` | If disk fills during journal write, delete journal and abort transaction before touching DB. | Yes: "Disk full" |
| `FsyncFailed` | `fsync()` | Hardware error. Panic and close connection. Database remains in "Recovery Needed" state. | Yes: "I/O Error" |

---

## 7. Implementation Sequence

### Phase 1: Locking State Machine (Estimated: 3-4 Hours)
- Implement `src/storage/lock_manager.c`.
- Use `fcntl` (Unix) with `F_SETLK`. Map states:
    - `SHARED`: Read lock on bytes 100-101.
    - `RESERVED`: Write lock on byte 102.
    - `PENDING`: Write lock on byte 103.
    - `EXCLUSIVE`: Write lock on bytes 100-101.
- **Checkpoint**: Run two processes. Verify Process B cannot acquire `RESERVED` if Process A holds it. Verify Process A cannot acquire `EXCLUSIVE` if Process B holds `SHARED`.

### Phase 2: Journaling & fsync (Estimated: 4-5 Hours)
- Implement `pager_write` to copy pages to the journal.
- Implement the `fsync` ordering logic in `pager_commit`.
- **Checkpoint**: Perform an `UPDATE`. Kill the process (using `SIGKILL`) *after* the `fsync` of the journal but *before* the deletion of the journal. Verify the `.db-journal` file exists and contains the correct "original" page data.

### Phase 3: Crash Recovery (Estimated: 3-4 Hours)
- Implement `pager_recover_if_needed` in the database open path.
- Implement the checksum and truncation logic.
- **Checkpoint**: Run the "killed" database from Phase 2. Verify that the `SELECT` query returns the **old** data (rollback successful) and the journal file is automatically deleted.

---

## 8. Test Specification

### 8.1 Happy Path: Successful Commit
- **Setup**: `UPDATE users SET name = 'New' WHERE id = 1`.
- **Action**: Call `COMMIT`.
- **Expected**: 
    - Journal is created then deleted.
    - Database file contains 'New'.
    - No journal file exists on disk.

### 8.2 Failure Case: Crash Mid-Transaction
- **Setup**: Start `UPDATE`.
- **Action**: Manually write garbage to a database page in a hex editor while a journal exists. Run `recovery`.
- **Expected**: The "garbage" page is overwritten by the original page from the journal.

### 8.3 Failure Case: Lock Contention
- **Setup**: Thread A has an open `SELECT` cursor (holding `SHARED`).
- **Action**: Thread B calls `BEGIN TRANSACTION` followed by an `INSERT`.
- **Expected**: Thread B succeeds in `RESERVED`, but `pager_commit` returns `SQLITE_BUSY` (or waits) at the `PENDING` stage because Thread A is still reading.

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Commit Latency** | < 50ms | Time from `COMMIT` command to return (includes 2-3 `fsync` calls). |
| **Fsync Count** | Exactly 2-3 | Use `strace -e fsync` to count calls per transaction. |
| **Recovery Time** | < 100ms | Measure time to recover from a 100-page "Hot Journal". |

---

## 10. Concurrency Specification: The Lock Protocol

To ensure no deadlocks and absolute consistency:

1. **Reader Path**:
    - `UNLOCKED` -> `SHARED`.
    - Cannot move to `SHARED` if a `PENDING` or `EXCLUSIVE` lock is on the file.
2. **Writer Path**:
    - Must be in `SHARED` first.
    - `SHARED` -> `RESERVED` (Only one writer allowed).
    - `RESERVED` -> `PENDING` (Prevents new readers from entering, but lets current readers finish).
    - `PENDING` -> `EXCLUSIVE` (Waits for `SHARED` count to hit 0).
3. **Wait Policy**: If a lock cannot be acquired, the manager should sleep for 10ms and retry up to 100 times before returning `SQLITE_BUSY`.

---

## 11. Recovery Checksum Algorithm

Use the **Fletcher-32** checksum for speed and reliability.

```c
uint32_t calculate_checksum(uint32_t nonce, uint32_t page_no, uint8_t* data) {
    uint32_t s1 = nonce + page_no;
    uint32_t s2 = s1;
    for(int i=0; i < 4096; i++) {
        s1 += data[i];
        s2 += s1;
    }
    return (s1 & 0xffff) | (s2 << 16);
}
```
*Note: In production, checksums are verified to ensure that a "Torn Write" to the journal doesn't result in restoring garbage to the database.*
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m10 -->
# Module Design: Write-Ahead Logging (WAL) Mode (build-sqlite-m10)

## 1. Module Charter

The **Write-Ahead Logging (WAL) Mode** module provides a high-concurrency durability alternative to the Rollback Journal. Its primary mission is to decouple readers from writers, allowing multiple simultaneous readers to query a consistent snapshot of the database while a single writer appends changes to a separate log. This module implements **Snapshot Isolation** by ensuring that readers only see transactions committed before their start time. It replaces the "Undo" logic of the Rollback Journal with a "Redo" logic, where the main database file remains untouched during transactions, and updates are periodically merged back via a **Checkpointer**.

**Core Responsibilities:**
- Manage the **WAL File** (`.db-wal`) as a sequential append-only redo log.
- Implement the **WAL Index** (`.db-shm`) as a memory-mapped hash table for O(1) page lookups.
- Coordinate **Snapshot Isolation** using "Read Marks" to track the visibility of WAL frames for active readers.
- Execute the **Checkpointing** process to migrate pages from the WAL back to the main database file.
- Provide high-performance **Cumulative Checksumming** to detect torn writes or log corruption.

**Non-Goals:**
- This module is not a replacement for the B-tree layer; it is an I/O redirection layer within the Pager.
- It does not support multiple simultaneous writers (SQLite remains 1-writer-N-readers).
- It does not handle logical SQL undo; it only manages physical page versions.

**Invariants:**
- A reader must never see a WAL frame with an index greater than the "Commit Mark" present at the reader's start time.
- The main database file must never be modified while a reader is accessing a version of a page that has not yet been checkpointed.
- WAL frames must be appended in strictly increasing transaction order.

---

## 2. File Structure

Implementation must follow this sequence to build the log structure before the indexing/concurrency logic:

1. `src/include/storage/wal_format.h`: Byte-level definitions for WAL headers and frame structures.
2. `src/storage/wal_checksum.c`: Implementation of the 64-bit cumulative checksum algorithm.
3. `src/include/storage/wal_index.h`: Memory-mapped WAL Index structure (shared-memory).
4. `src/storage/wal_index.c`: Logic for updating and searching the WAL Index.
5. `src/storage/wal_manager.c`: Core logic for appending frames and coordinating snapshots.
6. `src/storage/checkpoint.c`: The checkpointer logic for merging log data into the main DB.

---

## 3. Complete Data Model

### 3.1 The WAL File Format (Byte-Level)

The WAL file consists of a fixed 32-byte header followed by zero or more frames.

**WAL Header (Offset 0x00)**
| Offset | Size | Field | Description | Endian |
| :--- | :--- | :--- | :--- | :--- |
| 0x00 | 4 bytes | `magic` | `0x377f0682` (normal) or `0x377f0683` (big-endian). | Native |
| 0x04 | 4 bytes | `version` | WAL format version (currently 3007000). | Big |
| 0x08 | 4 bytes | `page_size` | Database page size (e.g., 4096). | Big |
| 0x0C | 4 bytes | `checkpoint_seq`| Incremental counter for checkpoint operations. | Big |
| 0x10 | 4 bytes | `salt_1` | Random value for checksum initialization. | Big |
| 0x14 | 4 bytes | `salt_2` | Second random value for checksum initialization. | Big |
| 0x18 | 4 bytes | `checksum_1` | First 32 bits of header checksum. | Big |
| 0x1C | 4 bytes | `checksum_2` | Second 32 bits of header checksum. | Big |

**WAL Frame (Repeated)**
Each frame encapsulates one database page.
| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | 4 bytes | `page_no` | The original Page ID in the `.db` file. |
| 0x04 | 4 bytes | `db_size` | Size of DB file in pages after this commit (0 if not a commit). |
| 0x08 | 4 bytes | `salt_1` | Copy of `salt_1` from header. |
| 0x0C | 4 bytes | `salt_2` | Copy of `salt_2` from header. |
| 0x10 | 4 bytes | `checksum_1` | Cumulative checksum up to this point (Part 1). |
| 0x14 | 4 bytes | `checksum_2` | Cumulative checksum up to this point (Part 2). |
| 0x18 | `PAGE_SIZE` | `page_data` | The raw 4KB page content. |


![WAL File Binary Layout](./diagrams/tdd-diag-m10-wal-layout.svg)


### 3.2 The WAL Index (`-shm` file)

The WAL Index is a memory-mapped file that enables fast lookups. For this implementation, we use a simplified version of the SQLite WAL-Index.

**WAL Index Header**
| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | 4 bytes | `version` | Index version. |
| 0x04 | 4 bytes | `is_init` | Initialization flag. |
| 0x08 | 4 bytes | `mx_frame` | Index of the last valid frame in the WAL. |
| 0x0C | 4 bytes | `n_page` | Number of pages in the DB according to the WAL. |
| 0x10 | 32 bytes | `read_marks` | Array of 8 `uint32_t` marks for concurrent readers. |

**Hash Table Blocks**
Following the header, the index contains blocks of 16-bit integers mapping `PageID % HashSize -> FrameIndex`. This allows the engine to skip scanning the WAL file.

---

## 4. Interface Contracts

### 4.1 `int wal_append_frames(Wal* pWal, Page** apPages, int nPages, uint32_t dbSize)`
- **Purpose**: Appends a set of modified pages to the log as a single atomic transaction.
- **Constraints**: 
    - `nPages` must be $\ge 1$.
    - `dbSize` is the post-transaction file size (only set on the last frame).
- **Errors**: `DISK_FULL`, `IO_ERROR`, `WAL_TOO_LARGE`.

### 4.2 `int wal_find_page(Wal* pWal, uint32_t page_no, uint32_t read_mark, void** ppOut)`
- **Purpose**: Search for a page version visible to a specific reader.
- **Logic**: 
    1. Search the WAL Index for the highest `frame_idx` such that `WAL_INDEX[frame_idx].page_no == page_no` AND `frame_idx <= read_mark`.
    2. If found, read from `.db-wal` at `32 + frame_idx * (24 + PAGE_SIZE)`.
- **Return**: `SQLITE_OK` if found in WAL, `SQLITE_NOTFOUND` if the reader must fallback to the `.db` file.

### 4.3 `int wal_checkpoint(Wal* pWal, int mode)`
- **Purpose**: Merge WAL frames into the main database.
- **Modes**:
    - `PASSIVE`: Checkpoint as much as possible without blocking readers.
    - `FULL`: Wait for all readers to finish, then checkpoint everything.
- **Errors**: `CHECKPOINT_BLOCKED`.

---

## 5. Algorithm Specification

### 5.1 Cumulative Checksumming
The WAL uses a running checksum to detect torn writes. The checksum of Frame $N$ depends on the checksum of Frame $N-1$.

1. **Initialize**: `s1 = salt_1`, `s2 = salt_2`.
2. **Step 1 (Header)**: Iterate through the first 24 bytes of the header. Update `s1, s2` using the Fletcher-64 variant.
3. **Step 2 (Frames)**: For each frame:
    - Update `s1, s2` using the 8-byte Frame Header (excluding the checksum fields themselves).
    - Update `s1, s2` using the 4096-byte Page Data.
    - Store current `s1, s2` in the Frame Header's checksum fields.
4. **Validation**: During recovery, if a frame's calculated checksum doesn't match its stored checksum, the log is truncated at the *previous* frame.

### 5.2 The Snapshot Read (Read Mark Logic)
To support concurrent readers, the system uses a set of "Read Marks" in the shared memory.

1. **Reader Starts**:
    - Acquire a `SHARED` lock on the database.
    - Read `mx_frame` (the last committed frame index) from the WAL Index.
    - Find an available `read_mark` slot in the WAL Index and set it to `mx_frame`.
    - This `read_mark` is the reader's "Horizon."
2. **Reading Page #X**:
    - Call `wal_find_page(X, horizon)`.
    - If found, use the WAL version.
    - If not found, use the main `.db` file version.
3. **Reader Finishes**:
    - Set its `read_mark` slot back to `0xFFFFFFFF` (Available).


![Snapshot Isolation (Read Marks)](./diagrams/tdd-diag-m10-isolation.svg)


### 5.3 Passive Checkpointing
1. **Determine the Limit**:
    - Scan all active `read_marks` in the WAL Index.
    - `min_mark = min(all_active_read_marks)`. 
    - This is the highest frame that *all* current readers have already moved past (or haven't reached yet).
2. **The Redo Loop**:
    - For `i` from `last_checkpointed_frame + 1` to `min_mark`:
        - Read `page_no` and `data` from WAL Frame `i`.
        - Write `data` to `.db` file at `page_no * page_size`.
3. **Flush**: `fsync()` the `.db` file.
4. **Update Metadata**: Update `last_checkpointed_frame` in the WAL Index.

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `WALChecksumMismatch` | `wal_open` | Truncate WAL at last valid frame. This represents a crash mid-write. | No (Automatic) |
| `CheckpointBlocked` | `wal_checkpoint`| Return `SQLITE_BUSY`. Trigger auto-checkpoint later when readers exit. | Yes (If manual PRAGMA) |
| `IndexCorrupt` | `wal_find_page` | Rebuild WAL Index from raw `.db-wal` file (Single scan). | No |
| `SHM_MappingFailed` | `wal_open` | Fallback to Rollback Journal mode or return error. | Yes |

---

## 7. Implementation Sequence

### Phase 1: WAL File Format (Estimated: 4-5 Hours)
- Implement `wal_checksum.c` with the cumulative logic.
- Implement `wal_append_frames` to write the header and serialized frames.
- **Checkpoint**: Run a loop that appends 100 random pages. Verify the `.db-wal` file size is exactly `32 + 100 * (24 + 4096)` bytes and that checksums are mathematically valid.

### Phase 2: WAL Index & Search (Estimated: 4-5 Hours)
- Implement `mmap` logic for the `-shm` file.
- Implement `wal_index_append` to update the hash table whenever a frame is written.
- Implement `wal_find_page`.
- **Checkpoint**: Manually append a new version of Page #1 to the WAL. Call `wal_find_page(1)`. Verify it returns the new version from the WAL, not the old one from the DB.

### Phase 3: Checkpointing & Recovery (Estimated: 4-5 Hours)
- Implement `wal_checkpoint(PASSIVE)`.
- Implement `wal_open` recovery logic (detecting and truncating invalid frames).
- **Checkpoint**: Fill WAL with 2000 pages. Run `wal_checkpoint`. Verify the `.db` file is updated and the WAL file is truncated/reset.

---

## 8. Test Specification

### 8.1 Happy Path: Simultaneous Read/Write
- **Setup**: One thread starts a long `SELECT` (holds read mark at frame 10).
- **Action**: Another thread performs 5 `INSERT`s (appends 50 frames to WAL).
- **Expected**: The reader thread continues to see the data as it was at frame 10, even though the WAL now contains 60 frames.

### 8.2 Edge Case: Log Wrapping
- **Setup**: WAL reaches `auto_checkpoint` limit.
- **Action**: Checkpoint runs successfully.
- **Expected**: The next `INSERT` should overwrite the WAL from the beginning (offset 32), reset `mx_frame` to 1.

### 8.3 Failure Case: Torn Write
- **Setup**: Append a frame but intentionally corrupt its checksum footer.
- **Action**: Re-open the database.
- **Expected**: `wal_open` detects the mismatch, truncates the WAL at the previous valid frame, and allows the DB to function normally.

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Write Throughput** | > 5,000 tx/sec | Run 10k single-row INSERTS in WAL mode. Compare to Rollback Mode. |
| **Lookup Latency** | < 200ns | Measure `wal_find_page` time when the page is in a 1000-frame WAL. |
| **Checkpt Overhead** | < 10% CPU | Monitor CPU usage during a background passive checkpoint. |

---

## 10. Concurrency Specification

**Concurrency Logic: Snapshot Isolation (SI)**

1. **The Writer**:
    - Acquires `RESERVED` lock on the `.db` file (Only one writer).
    - Appends to WAL.
    - Updates `mx_frame` in SHM *after* `fsync` of WAL.
2. **The Readers**:
    - No locks on `.db` file required.
    - Acquire `SHARED` lock on the SHM `read_mark` array to prevent a checkpointer from overwriting frames they are using.
3. **The Checkpointer**:
    - Acquires `EXCLUSIVE` lock on the SHM `read_mark` array to find the "Safe Point."
    - Acquires `CHECKPOINTER` lock to ensure only one checkpoint happens at a time.

**Deadlock Avoidance**: Locks must always be acquired in the order: `SHM` -> `WAL` -> `DB`.

---

## 11. Memory Layout: WAL Index Hash Table

To achieve O(1) lookups, the `-shm` file uses a simple hash table.
- **Bucket Count**: 1024 per block.
- **Entry**: `uint16_t` (The index of the frame).
- **Collision Handling**: Linear probing within the bucket block.

```c
uint32_t hash_page(uint32_t page_no) {
    return (page_no * 383) % 1024;
}
```
*Note: We multiply by a prime (383) to distribute Page IDs across the 1024 buckets.*
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-sqlite-m11 -->
# Module Design: Aggregate Functions & JOIN (build-sqlite-m11)

## 1. Module Charter

The **Aggregate Functions & JOIN** module is the high-level orchestration layer of the SQL execution engine. It extends the Virtual Machine (VDBE) from simple single-table scans to complex relational operations involving multiple data sources and data reduction logic. This module is responsible for managing the lifecycle of **Nested Loop Joins (NLJ)**, where multiple cursors are synchronized to find matching rows across tables, and **Streaming Aggregation**, where results are accumulated into stateful registers. 

**Core Responsibilities:**
- Implement the **Inner/Outer Cursor synchronization** logic within the VDBE loop.
- Provide the **Accumulator** abstraction for aggregate functions (`SUM`, `COUNT`, `AVG`, `MIN`, `MAX`).
- Manage **Group-By Bucket Management** using an in-memory hash table for unsorted grouping.
- Enforce **Three-Valued Logic (3VL)** during join predicate evaluation.
- Implement the **Finalization** phase of aggregation to compute derived values like averages or counts of non-null fields.

**Non-Goals:**
- This module does not perform query optimization (handled in Milestone 8).
- It does not implement Hash Joins or Merge Joins (only Nested Loop Joins are within scope).
- it does not handle subqueries or Common Table Expressions (CTEs).

**Invariants:**
- Every `INNER JOIN` result row must satisfy the join predicate; non-matching rows are strictly excluded.
- Aggregate functions must ignore `NULL` values except for `COUNT(*)`.
- The `HAVING` clause must only be evaluated after the global or grouped aggregation is finalized.

---

## 2. File Structure

Implementation proceeds in this order to establish the aggregation state before the multi-cursor join logic:

1. `src/include/vdbe/aggregate.h`: Definitions for `AggContext` and function pointers for `step`/`finalize`.
2. `src/vdbe/aggregate.c`: Implementation of standard SQL aggregate logic (SUM, COUNT, etc.).
3. `src/include/vdbe/hash_group.h`: Memory-mapped hash table for `GROUP BY` operations.
4. `src/vdbe/hash_group.c`: Logic for bucket allocation and collision resolution during grouping.
5. `src/vdbe/join_engine.c`: Integration of multi-cursor logic into the `vdbe_step` loop.
6. `src/vdbe/opcodes_m11.c`: Implementation of `AggStep`, `AggFinalize`, and `Next` (Multi-cursor variant).

---

## 3. Complete Data Model

### 3.1 Aggregation Context (`AggContext`)
Unlike standard registers that store static values, an aggregation register stores a pointer to an `AggContext`.

| Offset | Type | Field | Description |
| :--- | :--- | :--- | :--- |
| 0x00 | `int64_t` | `count` | Number of non-null values encountered. |
| 0x08 | `double` | `sum` | Running total for numeric columns. |
| 0x10 | `Value` | `min_max` | Current extreme value for MIN/MAX operations. |
| 0x20 | `bool` | `has_data` | Flag to distinguish between `SUM=0` and `SUM=NULL` (empty set). |
| 0x21 | `uint8_t` | `func_type` | Enum: `AGG_SUM`, `AGG_COUNT`, `AGG_AVG`, etc. |

### 3.2 Join Execution State (`JoinContext`)
The VDBE tracks the hierarchy of cursors during a join.

| Field | Type | Description |
| :--- | :--- | :--- |
| `cursor_id` | `uint32_t` | The B-tree cursor index. |
| `parent_id` | `int32_t` | The index of the outer cursor (-1 if outermost). |
| `is_matched` | `bool` | Used for OUTER joins (set to true if inner match found). |
| `reg_base` | `uint32_t` | The starting register where this table's columns are projected. |

### 3.3 Hash Grouping Entry
For `GROUP BY`, we use an in-memory hash table where the key is the grouping value and the value is the `AggContext`.

```c
typedef struct HashEntry {
    Value key;               // The grouping key (e.g., 'CategoryA')
    AggContext* accumulators; // Array of contexts for each aggregate in SELECT
    struct HashEntry* next;  // Chaining for collisions
} HashEntry;
```


![Aggregate State Machine](./diagrams/tdd-diag-m11-agg-sm.svg)

*(Visual: State machine for a register: Empty -> [AggStep] -> Accumulating -> [AggFinalize] -> ResultValue)*

---

## 4. Interface Contracts

### 4.1 `void agg_step(AggContext* ctx, Value* input)`
- **Purpose**: Updates the accumulator with a new row's value.
- **Constraints**: 
    - If `input` is `NULL`, most functions (SUM, AVG) increment nothing.
    - `COUNT(*)` ignores the `input` value and increments `ctx->count`.
- **Edge Case**: If types are incompatible (e.g., SUM on a BLOB), set `VDBE_ERROR`.

### 4.2 `Value agg_finalize(AggContext* ctx)`
- **Purpose**: Computes the final result (e.g., `sum / count` for AVG).
- **Return**: 
    - `SUM` returns `NULL` if `has_data` is false.
    - `COUNT` returns `0` if `has_data` is false.

### 4.3 `int vdbe_exec_join_step(Vdbe* p, JoinContext* jc)`
- **Purpose**: Moves the cursors for a Nested Loop Join.
- **Logic**: See Algorithm 5.1.
- **Errors**: `SQLITE_CORRUPT` if cursors become desynchronized.

---

## 5. Algorithm Specification

### 5.1 Nested Loop Join (NLJ) Execution
The VDBE implements joins as nested loops using the `Rewind` and `Next` opcodes across multiple cursors.

**Procedure: `exec_join_loop`**
1. **Outer Cursor**: Open `Csr[0]`. `Rewind(0)`.
2. **Inner Cursor Initialization**:
   - For each row in `Csr[0]`:
     - Open `Csr[1]`.
     - **Optimization Check**: If an index exists on the join predicate (e.g., `Csr[1].fk = Csr[0].id`):
       - Perform `IdxSeek(Csr[1], Csr[0].id)`.
     - Else:
       - `Rewind(1)`.
3. **Evaluation**:
   - For each row in `Csr[1]`:
     - Evaluate the `JoinPredicate`.
     - If `TRUE`:
       - Emit `ResultRow` (Columns from `Csr[0]` + `Csr[1]`).
     - Else:
       - Continue to `Next(1)`.
4. **Advance**:
   - When `Csr[1]` hits EOF, `Next(0)`.
   - Reset `Csr[1]` to top.


![Nested Loop Join Flow](./diagrams/tdd-diag-m11-nlj-flow.svg)

*(Visual: Flowchart showing nested loops: Outer Loop -> Reset Inner -> Inner Loop -> Predicate -> Match? -> Output)*

### 5.2 Streaming Aggregation (No Grouping)
When the query is `SELECT SUM(x) FROM t`, the compiler emits:

1. `Null R1`: Initialize accumulator register.
2. `Rewind Csr[0]`: Start scan.
3. `Column Csr[0], col_x, R2`: Load value.
4. `AggStep R1, R2`: Update sum in R1.
5. `Next Csr[0], label_3`: Loop.
6. `AggFinalize R1`: Compute final sum.
7. `ResultRow R1`: Output result.

### 5.3 Hash-Based GROUP BY
For queries with `GROUP BY`, we use a Hash Map to manage multiple aggregation states.

1. **Step Phase**:
   - For each row:
     - `key = Column(group_by_col)`.
     - `hash_idx = hash(key) % table_size`.
     - `entry = find_or_create_entry(hash_idx, key)`.
     - For each aggregate in query:
       - `agg_step(entry->accumulators[i], Column(agg_col))`.
2. **Finalize Phase**:
   - Iterate through every bucket in the Hash Map.
   - For each `entry`:
     - `val = agg_finalize(entry->accumulators)`.
     - If `HAVING` clause exists:
       - Evaluate `HAVING` on `val`.
       - If `FALSE`, skip entry.
     - `ResultRow(key, val)`.

---

## 6. Error Handling Matrix

| Error | Detected By | Recovery | User-Visible? |
| :--- | :--- | :--- | :--- |
| `AmbiguousColumn` | Compiler | Abort compilation; requires table prefix (e.g., `t1.id`). | Yes |
| `AggOnNonNumeric` | `agg_step` | Return `NULL` for the result or `0` for count. | Yes (Warning) |
| `HashMemoryExhausted`| `hash_group` | Fallback to Sort-Based grouping (requires temporary disk file). | Yes |
| `DivisionByZero` | `agg_finalize`| In AVG, if count is 0, return `NULL`. | No |

---

## 7. Implementation Sequence

### Phase 1: Aggregate Accumulator Logic (Estimated: 4-5 Hours)
- Implement `AggContext` and the `agg_step`/`agg_finalize` functions for `SUM`, `COUNT`, and `MIN`.
- Extend the `Value` struct to support the `AGG_CONTEXT` type.
- **Checkpoint**: Manually feed 5 `Value` objects (10, 20, NULL, 30, 40) into `agg_step` for a `SUM` context. Verify `agg_finalize` returns `100`.

### Phase 2: Nested Loop Join (Estimated: 6-7 Hours)
- Implement multi-cursor support in the VDBE `pc` (Program Counter) logic.
- Implement the `SeekGE` optimization for inner-loop lookups.
- **Checkpoint**: Perform a join between a 10-row table and a 5-row table. Verify `ResultRow` is called 50 times (Cartesian product) if no predicate is provided.

### Phase 3: GROUP BY Bucket Management (Estimated: 4-5 Hours)
- Implement the in-memory `HashEntry` table.
- Implement the `AggStep` opcode to look up the correct context based on the grouping key.
- **Checkpoint**: Run `SELECT cat, COUNT(*) FROM t GROUP BY cat`. Verify the result set has one row per unique `cat` value.

---

## 8. Test Specification

### 8.1 Happy Path: Indexed Inner Join
- **Setup**: `Table A {id: 1, 2}`, `Table B {uid: 1, 1, 2}`. Index on `B.uid`.
- **Query**: `SELECT * FROM A JOIN B ON A.id = B.uid`
- **Expected**: 3 rows: `(1, 1), (1, 1), (2, 2)`.

### 8.2 Edge Case: Aggregates with NULLs
- **Input**: `[10, NULL, 20]`
- **Action**: `SELECT SUM(val), COUNT(val), COUNT(*)`
- **Expected**: `SUM=30`, `COUNT(val)=2`, `COUNT(*)=3`.

### 8.3 Failure Case: Ambiguous Join
- **Input**: `SELECT id FROM A JOIN B ON A.id = B.id` (Both tables have `id`).
- **Expected**: Compiler error: "Ambiguous column name: id".

---

## 9. Performance Targets

| Operation | Target | How to Measure |
| :--- | :--- | :--- |
| **Indexed Join** | $O(N \log M)$ | Join 1k rows to 1M rows; total page fetches should be $\approx 1000 \times 4$. |
| **Hash Grouping** | < 100ns / row | Measure `AggStep` overhead in a 100k row scan. |
| **Agg Memory** | 64 bytes / group | `sizeof(AggContext)` + hash overhead. |

---

## 10. Concurrency Specification

**Multi-Cursor Isolation**:
1. All cursors within a single VDBE program share a single **Read Snapshot** (M10).
2. Even if an `INSERT` happens in a separate connection during the join, the cursors will only see rows committed at the start of the `SELECT`.
3. **Locking**: The engine acquires a `SHARED` lock for every table involved in the join to prevent a concurrent `EXCLUSIVE` lock (Rollback Journal mode).

---

## 11. Wire Format: Aggregation Register Layout

When an aggregate function is active, the VDBE Register (a `Mem` struct) points to a heap-allocated buffer.

**Binary Layout of Aggregation Register:**
- `0x00`: Type Tag (`0xFF` for Internal Context)
- `0x01`: Function ID (`0x01`=SUM, `0x02`=COUNT, etc.)
- `0x08`: Pointer to `AggContext` structure.

**Accumulator for AVG (Specific Example):**
- `SumPart`: `double` (8 bytes) at `AggContext + 0x08`.
- `CountPart`: `uint64_t` (8 bytes) at `AggContext + 0x00`.
- **Finalize Formula**: `return (count == 0) ? NULL : (sum / count)`.
<!-- END_TDD_MOD -->


# Project Structure: Build Your Own SQLite

## Directory Tree

```text
sqlite-clone/
├── src/
│   ├── include/                # Header Files (Public & Internal)
│   │   ├── tokenizer/          # M1: Lexical analysis definitions
│   │   │   ├── token_types.h   # TokenType enumeration
│   │   │   ├── token.h         # Token struct & literal unions
│   │   │   └── scanner.h       # Scanner/FSM state definition
│   │   ├── parser/             # M2: Syntactic analysis definitions
│   │   │   ├── ast_nodes.h     # AST node types (tagged unions)
│   │   │   └── arena.h         # Memory arena allocator interface
│   │   ├── vdbe/               # M3, M6, M11: Virtual Machine definitions
│   │   │   ├── opcodes.h       # VDBE ISA (Instruction Set)
│   │   │   ├── value.h         # Value/Mem tagged union for registers
│   │   │   ├── vdbe.h          # VM state & Instruction layout
│   │   │   ├── cursor.h        # M6: B-tree cursor abstraction
│   │   │   ├── record.h        # M6: SQLite binary record format
│   │   │   └── aggregate.h     # M11: Aggregation context & functions
│   │   ├── storage/            # M4, M5, M7, M9, M10: Disk & B-tree
│   │   │   ├── pager_types.h   # M4: Frame/Page size constants
│   │   │   ├── lru_cache.h     # M4: O(1) eviction interface
│   │   │   ├── pager.h         # M4: Buffer pool manager interface
│   │   │   ├── varint.h        # M5: Varint encoding/decoding
│   │   │   ├── page_format.h   # M5: Slotted page binary layout
│   │   │   ├── btree.h         # M5: B-tree search/insert/split
│   │   │   ├── index_format.h  # M7: Index B+tree layout
│   │   │   ├── index_manager.h # M7: Index maintenance interface
│   │   │   ├── lock_manager.h  # M9: 5-state file locking
│   │   │   ├── journal_format.h# M9: Rollback journal binary layout
│   │   │   ├── wal_format.h    # M10: WAL redo log headers/frames
│   │   │   └── wal_index.h     # M10: Shared memory WAL index
│   │   └── optimizer/          # M8: Query planning definitions
│   │       ├── stats.h         # sys_stats table structures
│   │       └── cost_model.h    # Cost constants & plan structures
│   ├── tokenizer/              # M1: Lexer Implementation
│   │   ├── scanner.c           # FSM logic & char classification
│   │   └── keywords.c          # SQL keyword lookup table
│   ├── parser/                 # M2: Parser Implementation
│   │   ├── arena.c             # O(1) AST node allocator
│   │   ├── expr_parser.c       # Pratt Parser (Precedence Climbing)
│   │   ├── stmt_parser.c       # Recursive Descent for SELECT/INSERT
│   │   └── parser.c            # Top-level entry & error recovery
│   ├── vdbe/                   # VM & Execution Implementation
│   │   ├── compiler.c          # M3: AST to Bytecode translator
│   │   ├── vm.c                # M3: Fetch-Decode-Execute loop
│   │   ├── explain.c           # M3: Bytecode disassembler
│   │   ├── cursor.c            # M6: B-tree navigation logic
│   │   ├── record.c            # M6: Record serialization logic
│   │   ├── dml_ops.c           # M6: OP_Insert/OP_Delete handlers
│   │   ├── index_ops.c         # M7: OP_IdxGE/OP_IdxRowid handlers
│   │   ├── aggregate.c         # M11: SUM/COUNT step & finalize
│   │   ├── hash_group.c        # M11: Group-By hash table management
│   │   ├── join_engine.c       # M11: Nested Loop Join logic
│   │   └── opcodes_m11.c       # M11: New aggregation opcodes
│   ├── storage/                # Storage & Durability Implementation
│   │   ├── lru_cache.c         # M4: Doubly Linked List + HashMap
│   │   ├── pager.c             # M4: Page fetch/evict/flush
│   │   ├── varint.c            # M5: 1-9 byte integer encoding
│   │   ├── page_format.c       # M5: Header & Cell management
│   │   ├── btree.c             # M5: Page splitting & traversal
│   │   ├── index_page.c        # M7: B+tree leaf linking logic
│   │   ├── index_manager.c     # M7: Synchronous write-tax hooks
│   │   ├── lock_manager.c      # M9: fcntl/advisory locking
│   │   ├── journal.c           # M9: Undo-log population
│   │   ├── recovery.c          # M9: Hot journal rollback logic
│   │   ├── pager_transaction.c # M9: Pager-level ACID coordination
│   │   ├── wal_checksum.c      # M10: Fletcher-64 cumulative sums
│   │   ├── wal_index.c         # M10: SHM hash table lookups
│   │   ├── wal_manager.c       # M10: Log appending & snapshots
│   │   └── checkpoint.c        # M10: Redo-log to DB merging
│   └── optimizer/              # Optimizer Implementation
│       ├── analyzer.c          # M8: B-tree sampling (ANALYZE)
│       ├── selectivity.c       # M8: Row count estimation
│       └── planner.c           # M8: Access path & join selection
├── build/                      # Build Artifacts (Binary outputs)
├── tests/                      # Test Suite (Module unit tests)
├── Makefile                    # Build System
├── .gitignore                  # Ignore build/ & .db files
└── README.md                   # Setup & Usage guide
```

## Creation Order

1.  **Project Scaffolding**
    *   Setup `Makefile`, `src/include/`, and basic project directories.
2.  **SQL Frontend (M1 - M2)**
    *   Implement `tokenizer/` and `parser/`. Ensure AST can represent `SELECT 1+2`.
3.  **The Virtual Machine (M3)**
    *   Implement `vdbe/vm.c` and `vdbe/compiler.c` for constant arithmetic.
4.  **Buffer Pool & Pager (M4)**
    *   Implement `storage/pager.c` and `lru_cache.c`. This is the foundation for all disk I/O.
5.  **B-tree Storage (M5)**
    *   Implement `varint.c` and `btree.c`. Achieve "Root-only" table storage.
6.  **Data Navigation & DML (M6)**
    *   Connect VM to B-tree via `cursor.c` and `record.c`. Enable `SELECT *` scans.
7.  **Secondary Indexes (M7)**
    *   Implement `index_manager.c` and `index_page.c`. Enable $O(\log N)$ lookups.
8.  **Query Optimization (M8)**
    *   Implement `planner.c` and `analyzer.c`. Teach the VM when to use indexes.
9.  **Atomicity & Rollback (M9)**
    *   Implement `lock_manager.c` and `journal.c`. Enable `BEGIN/COMMIT`.
10. **Concurrency Optimization (M10)**
    *   Implement `wal_manager.c` and `checkpoint.c`. Enable simultaneous read/write.
11. **Relational Power (M11)**
    *   Implement `join_engine.c` and `aggregate.c` to complete the SQL engine.

## File Count Summary
- **Total files**: 58
- **Directories**: 14
- **Estimated lines of code**: ~8,000 - 12,000 LOC

# 📚 Beyond the Atlas: Further Reading

## I. Language Frontend (Lexing & Parsing)

### 1. Finite State Machines for Lexing
*   **Best Explanation**: *Crafting Interpreters* by Robert Nystrom, **Chapter 4: Scanning**.
*   **Code**: SQLite Source — [`src/tokenize.c`](https://github.com/sqlite/sqlite/blob/master/src/tokenize.c).
*   **Why**: Nystrom provides the cleanest visual transition from raw characters to a state machine, while the SQLite source shows how this is implemented for a real SQL dialect.
*   **Timing**: **Read BEFORE Milestone 1**. It establishes the "Peek and Consume" mental model required for the very first line of code.

### 2. Pratt Parsing (Precedence Climbing)
*   **Paper**: Vaughan Pratt (1973), *Top Down Operator Precedence*.
*   **Best Explanation**: ["Pratt Parsers: Expression Parsing Made Easy"](https://journal.stuffwithstuff.com/2011/03/19/pratt-parsers-expression-parsing-made-easy/) by Robert Nystrom.
*   **Code**: Rust-Analyzer — [`parser/src/grammar/expressions.rs`](https://github.com/rust-lang/rust-analyzer/blob/master/crates/parser/src/grammar/expressions.rs).
*   **Why**: This is the most elegant algorithm for handling operator precedence. Nystrom's blog post is widely considered the definitive modern explanation.
*   **Timing**: **Read BEFORE Milestone 2**. Without this, your expression parser will struggle with mathematical order of operations.

## II. The Virtual Machine (VDBE)

### 3. Register-Based Virtual Machines
*   **Paper**: Ierusalimschy et al. (2005), *The Implementation of Lua 5.0* (Section 4: The Virtual Machine).
*   **Best Explanation**: *Architecture of SQLite*, **"The Virtual Database Engine"** section on sqlite.org.
*   **Code**: SQLite Source — [`src/vdbe.c`](https://github.com/sqlite/sqlite/blob/master/src/vdbe.c) (The `sqlite3VdbeExec` function).
*   **Why**: SQLite is the most famous example of a register-based VM in the world. The Lua paper explains why registers outperform stack machines.
*   **Timing**: **Read during Milestone 3**. Understanding the fetch-decode-execute loop is essential while you are building your opcode dispatcher.

## III. Storage Engine & B-Trees

### 4. The B-Tree Data Structure
*   **Paper**: Douglas Comer (1979), *The Ubiquitous B-Tree*.
*   **Best Explanation**: *Database Internals* by Alex Petrov, **Chapter 2: B-Tree Basics**.
*   **Code**: SQLite Source — [`src/btree.c`](https://github.com/sqlite/sqlite/blob/master/src/btree.c).
*   **Why**: Comer’s paper is the seminal text. Petrov’s book provides the best modern context for how B-Trees function specifically on disk (vs. in memory).
*   **Timing**: **Read BEFORE Milestone 5**. You cannot design the physical page layout without understanding the tree-balancing math.

### 5. Slotted Pages & Record Format
*   **Spec**: [SQLite Database File Format](https://www.sqlite.org/fileformat.html), **Section 2.0: B-Tree Pages**.
*   **Best Explanation**: "Database Internals" by Alex Petrov, **Chapter 1: Storage Engine** (Section on Slotted Pages).
*   **Why**: This official spec is the "Holy Grail" for this project. It defines the exact byte-offsets for headers and cells.
*   **Timing**: **Read BEFORE Milestone 5 & 6**. This is required foundational knowledge for serializing SQL rows into bytes.

## IV. Buffer Management

### 6. Buffer Pool Replacement Policies (LRU)
*   **Paper**: Effelsberg & Haerder (1984), *Principles of Database Buffer Management*.
*   **Best Explanation**: CMU Database Systems (15-445) **Lecture 06: Memory Management**.
*   **Code**: Postgres Source — [`src/backend/storage/buffer/freelist.c`](https://github.com/postgres/postgres/blob/master/src/backend/storage/buffer/freelist.c).
*   **Why**: This lecture and paper explain the "Double Buffering" problem and why LRU is the industry standard for database pagers.
*   **Timing**: **Read AFTER Milestone 4**. You will have built a basic LRU, and this will help you appreciate why modern DBs use more complex variants like LRU-K or Clock.

## V. Query Optimization

### 7. Cost-Based Optimization (CBO)
*   **Paper**: Selinger et al. (1979), *Access Path Selection in a Relational Database Management System*.
*   **Best Explanation**: [SQLite Query Planner Architecture](https://www.sqlite.org/queryplanner.html).
*   **Why**: The Selinger paper is the foundation of every SQL optimizer in existence. The SQLite doc explains how it applies specifically to nested-loop joins.
*   **Timing**: **Read BEFORE Milestone 8**. It explains the math behind "Selectivity" and "Random I/O penalty" that you will code.

## VI. Durability & Transactions (ACID)

### 8. Write-Ahead Logging (WAL)
*   **Paper**: C. Mohan et al. (1992), *ARIES: A Transaction Recovery Method*.
*   **Best Explanation**: [SQLite WAL Mode](https://www.sqlite.org/wal.html) documentation.
*   **Code**: SQLite Source — [`src/wal.c`](https://github.com/sqlite/sqlite/blob/master/src/wal.c).
*   **Why**: ARIES is the research foundation for WAL. The SQLite documentation provides the most accessible explanation of the `-shm` (shared memory) index.
*   **Timing**: **Read BEFORE Milestone 10**. WAL is highly complex; you need the mental model of "Redo Logs" before touching the code.

### 9. Atomic Commit & fsync
*   **Best Explanation**: [Atomic Commit In SQLite](https://www.sqlite.org/atomiccommit.html) official documentation.
*   **Why**: This document details the "Dance of fsyncs" required to ensure data isn't lost during a power failure.
*   **Timing**: **Read during Milestone 9**. It explains exactly when to create, flush, and delete the rollback journal.

## VII. Advanced Execution

### 10. Join Algorithms & Nested Loops
*   **Best Explanation**: *Database System Concepts* (Silberschatz et al.), **Chapter 15.5: Join Operations**.
*   **Code**: SQLite Source — [`src/where.c`](https://github.com/sqlite/sqlite/blob/master/src/where.c) (Look for `whereLoopAdd`).
*   **Why**: SQLite almost exclusively uses Nested Loop Joins. This resource explains why that is efficient for embedded use cases compared to Hash Joins.
*   **Timing**: **Read BEFORE Milestone 11**. This will clarify how the Virtual Machine's Program Counter (PC) moves between multiple cursors.
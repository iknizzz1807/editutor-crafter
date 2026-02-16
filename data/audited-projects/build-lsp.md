# AUDIT & FIX: build-lsp

## CRITIQUE
- **Audit Finding #1 is VALID and CRITICAL**: LSP specifies positions as (line, character) where 'character' is a UTF-16 code unit offset. This does NOT correspond to byte offsets, Unicode codepoints, or grapheme clusters. A server using byte offsets or UTF-8 offsets will produce off-by-one errors for any non-ASCII text (emoji, CJK, etc.). This is one of the most common LSP implementation bugs and must be addressed explicitly.
- **Audit Finding #2 is VALID**: The LSP spec says document version numbers are monotonically increasing. Out-of-order notifications can occur in practice (race conditions). The server should validate version numbers and reject or reorder stale updates.
- **M1 Content-Length pitfall is under-specified**: Content-Length is in BYTES, but the JSON body may contain multi-byte UTF-8 characters. Calculating Content-Length from string length (character count) instead of byte length is a common bug.
- **M2 is missing the critical position encoding issue**: The entire milestone talks about 'range-based text changes' but never mentions that positions use UTF-16 code units. This causes every subsequent milestone's position calculations to be wrong for non-ASCII content.
- **M3 is too ambitious for a single milestone**: Completion, hover, go-to-definition, AND find-all-references is typically 4 separate features. Find-all-references is significantly more complex than go-to-definition (requires indexing all references, not just declarations).
- **M3 missing parser/AST implementation**: The milestone assumes an AST and symbol table exist but no milestone builds them. Either M2 should include basic parsing, or a dedicated milestone is needed.
- **M4 'Diagnostic spam' pitfall is real but no debouncing mechanism is described**: Sending diagnostics on every keystroke overwhelms the editor. A debounce/throttle mechanism should be explicitly required.
- **Missing milestone for workspace features**: textDocument/references, workspace/symbol, and rename require cross-file analysis. The current milestones only cover single-file features.
- **Missing 'progress' and 'cancellation' support**: Long-running operations (like indexing a workspace) should support $/progress and $/cancelRequest. These are important for UX.
- **The project doesn't specify what language the LSP server is FOR**: An LSP server must target a specific language to analyze. This should be explicitly stated (e.g., a simple language like Markdown, JSON, or a toy language).

## FIXED YAML
```yaml
id: build-lsp
name: Build Your Own LSP Server
description: Language Server Protocol implementation for a simple target language
difficulty: expert
estimated_hours: "55-90"
essence: >
  JSON-RPC message transport implementing bidirectional editor-server communication
  protocol, incremental document synchronization with version tracking and UTF-16
  position mapping, and AST-based semantic analysis for real-time code intelligence
  features like completion, hover, and navigation.
why_important: >
  Building an LSP server teaches fundamental compiler frontend techniques (lexing,
  parsing, semantic analysis) while learning how modern IDEs achieve language-agnostic
  tooling through standardized protocols, skills directly applicable to developer
  tools, IDE extensions, and language toolchain development.
learning_outcomes:
  - Implement JSON-RPC 2.0 message framing and request-response handling over stdio
  - Handle UTF-16 code unit position encoding required by the LSP specification
  - Design incremental document synchronization with version tracking
  - Build a parser and AST for a target language to support semantic analysis
  - Build symbol table and scope resolution for semantic token analysis
  - Implement completion engine with context-aware suggestion ranking
  - Create goto-definition using AST traversal and symbol reference tracking
  - Design diagnostic system with error recovery and debounced publishing
  - Handle concurrent document state updates with version validation
skills:
  - JSON-RPC Protocol
  - Abstract Syntax Trees
  - Incremental Parsing
  - Semantic Analysis
  - Symbol Resolution
  - Editor Integration
  - Concurrent State Management
  - Protocol Design
  - UTF-16 Position Encoding
tags:
  - build-from-scratch
  - compilers
  - completion
  - diagnostics
  - expert
  - go
  - language-server
  - protocols
  - refactoring
  - rust
  - typescript
architecture_doc: architecture-docs/build-lsp/index.md
languages:
  recommended:
    - TypeScript
    - Rust
    - Go
  also_possible:
    - Python
    - C#
resources:
  - type: specification
    name: "LSP Specification 3.17"
    url: "https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/"
  - type: article
    name: "LSP Tutorial (VS Code)"
    url: "https://code.visualstudio.com/api/language-extensions/language-server-extension-guide"
  - type: specification
    name: "JSON-RPC 2.0 Specification"
    url: "https://www.jsonrpc.org/specification"
prerequisites:
  - type: skill
    name: JSON-RPC protocol basics
  - type: skill
    name: Parsing and AST construction
  - type: skill
    name: IDE extension concepts
  - type: skill
    name: Concurrency
notes: >
  This project requires choosing a target language to analyze. Recommended: a simple
  language like a subset of Markdown, JSON, TOML, or a toy programming language.
  The LSP server provides intelligence features for this target language.
milestones:
  - id: build-lsp-m1
    name: JSON-RPC Transport and LSP Initialization
    description: >
      Implement the JSON-RPC 2.0 transport layer over stdin/stdout with
      Content-Length framing, and handle the LSP initialization handshake.
    acceptance_criteria:
      - JSON-RPC messages are framed with 'Content-Length: <byte_count>\r\n\r\n' headers where the length is the byte count (not character count) of the JSON body
      - Partial message reads are handled correctly; the reader buffers until Content-Length bytes are received before parsing
      - Initialize request is handled: client capabilities are received, server returns a ServerCapabilities object listing supported features
      - initialized notification is received and the server transitions from 'initializing' to 'ready' state; requests before initialization return error code -32002 (ServerNotInitialized)
      - Shutdown request returns success (null result); the subsequent exit notification causes the server process to terminate with exit code 0
      - Unknown methods return JSON-RPC error with code -32601 (MethodNotFound)
      - Notifications (no 'id' field) do not receive responses; requests (with 'id') always receive a response or error
    pitfalls:
      - Content-Length is in BYTES not characters; a JSON body with multi-byte UTF-8 characters has a different byte length than character length
      - The header section ends with \r\n\r\n (double CRLF); parsing only \n\n will fail with some editors
      - Sending a response to a notification is a protocol violation; check for presence of 'id' field
      - The LSP spec requires the server to send an error for requests received before initialization completes
      - stdout must be used exclusively for JSON-RPC messages; any debug logging to stdout corrupts the protocol stream; use stderr for logging
    concepts:
      - JSON-RPC 2.0 message format
      - Content-Length framing protocol
      - LSP lifecycle (initialize → initialized → shutdown → exit)
    skills:
      - JSON-RPC protocol implementation
      - Byte-level stream reading and buffering
      - LSP capability negotiation
      - Protocol state machine implementation
    deliverables:
      - JSON-RPC message parser and serializer with Content-Length byte-counting framing
      - Initialize/initialized handshake handler returning server capabilities
      - Shutdown and exit handler for clean server termination
      - Error handling for unknown methods and pre-initialization requests
      - Stderr-based logging that doesn't interfere with the protocol stream
    estimated_hours: "8-12"

  - id: build-lsp-m2
    name: Document Synchronization and Position Mapping
    description: >
      Track document open/change/close lifecycle, apply incremental edits,
      and implement UTF-16 to internal offset conversion.
    acceptance_criteria:
      - textDocument/didOpen stores the full document content keyed by URI and triggers initial parsing/analysis
      - textDocument/didChange applies edits (full or incremental sync) and updates the stored document to the new version
      - textDocument/didClose removes the document from the server's in-memory store
      - Incremental sync correctly applies range-based text changes using (line, character) positions where 'character' is a UTF-16 code unit offset per the LSP specification
      - Position mapping utility converts LSP UTF-16 (line, character) positions to internal byte offsets (or codepoint offsets) and vice versa, correctly handling multi-byte characters (emoji, CJK, surrogate pairs)
      - Version numbers are tracked per document; stale notifications with lower version numbers than the current document version are logged and discarded
      - After each document change, the document is re-parsed to update the AST and symbol information
    pitfalls:
      - LSP positions use UTF-16 code unit offsets; characters outside the Basic Multilingual Plane (e.g., emoji 😀 = U+1F600) occupy 2 UTF-16 code units but 4 UTF-8 bytes; using byte offsets or UTF-32 codepoints directly produces wrong positions
      - Incremental edits specify a range to replace; an empty range is an insertion, and empty newText is a deletion
      - Version numbers should be validated; out-of-order notifications can occur if the editor sends changes faster than the server processes them
      - Off-by-one errors in line/character conversion are extremely common; line is 0-indexed, character is 0-indexed UTF-16 offset within the line
      - Full document sync (sending the entire text on every change) is simpler but uses more bandwidth; incremental sync is more complex but required for large files
    concepts:
      - Document lifecycle in LSP
      - UTF-16 position encoding
      - Incremental text synchronization
      - Version-based consistency
    skills:
      - Text document state management
      - UTF-16 ↔ UTF-8/byte offset conversion
      - Incremental content synchronization
      - Position and range calculations
    deliverables:
      - didOpen/didClose/didChange handlers managing document lifecycle
      - Incremental edit application using LSP range-based edits
      - UTF-16 position mapper converting between LSP positions and internal offsets
      - Document version tracking with stale update detection
      - Basic parser for target language producing an AST on each document change
    estimated_hours: "10-15"

  - id: build-lsp-m3
    name: Core Language Features
    description: >
      Implement completion, hover, and go-to-definition using the AST and
      symbol table built from the parsed target language.
    acceptance_criteria:
      - textDocument/completion returns a CompletionList with relevant items based on cursor position context (e.g., keywords, variable names, function names in scope)
      - Completion items include at minimum: label, kind (variable, function, keyword, etc.), and insertText
      - textDocument/hover returns a Hover response with MarkupContent containing type signature or documentation for the symbol under the cursor, or null if no symbol
      - textDocument/definition returns a Location (file URI, range) pointing to the symbol's declaration site, or null if the symbol cannot be resolved
      - Symbol table is constructed from the AST with scope tracking; nested scopes correctly shadow outer scope declarations
      - All position inputs are converted from UTF-16 to internal offsets; all position outputs are converted back to UTF-16
    pitfalls:
      - Stale AST/symbol table after document change causes incorrect results; ensure re-parsing happens before responding to feature requests
      - Cursor position for completion is between characters; the trigger context (prefix text) must be extracted correctly
      - Scope visibility: a variable declared on line 10 should not appear in completions for a position on line 5
      - Performance on large files: full re-parse on every keystroke adds latency; consider lazy/incremental strategies for large documents
      - Completion triggered mid-identifier should filter results by the partial identifier prefix
    concepts:
      - Symbol table construction and lookup
      - AST traversal for semantic information
      - Scope-based name resolution
      - Completion context analysis
    skills:
      - Abstract syntax tree traversal
      - Symbol table construction and queries
      - Context-aware code completion
      - Cross-reference resolution
    deliverables:
      - Completion provider returning context-filtered completion items with labels and kinds
      - Hover provider returning type/documentation information for symbols
      - Go-to-definition resolving symbol references to declaration locations
      - Symbol table with scope tracking for the target language
    estimated_hours: "15-22"

  - id: build-lsp-m4
    name: Diagnostics and Code Actions
    description: >
      Publish diagnostics (errors, warnings) after document changes with
      debouncing, and provide code actions (quick fixes) for common issues.
    acceptance_criteria:
      - textDocument/publishDiagnostics pushes diagnostics to the client after document changes, including parse errors, undefined symbols, and type mismatches (as applicable to target language)
      - Diagnostic publishing is debounced: rapid successive edits (keystrokes) delay diagnostic computation until a quiet period (e.g., 200-500ms after last edit)
      - Each diagnostic includes: range (start/end position), severity (Error, Warning, Information, Hint), message, and source (server name)
      - When a document is closed or has no errors, an empty diagnostics array is published to clear stale diagnostics in the editor
      - textDocument/codeAction returns applicable code actions for the given range or diagnostic context
      - At least one quick-fix code action is implemented that includes WorkspaceEdit text edits to resolve a specific diagnostic (e.g., adding a missing import, fixing a typo)
      - Code actions specify the kind (quickfix, refactor, etc.) for editor UI categorization
    pitfalls:
      - Sending diagnostics on every keystroke overwhelms the editor and the server; debouncing is essential
      - Stale diagnostics from a previous version of the document must be cleared; always publish for the current document version
      - Diagnostic ranges must use UTF-16 positions; incorrect positions cause highlighting of wrong text in the editor
      - Publishing diagnostics for a closed document causes ghost diagnostics in the editor; always clear on didClose
      - Code actions must be fast since they're requested on every cursor movement in some editors; avoid expensive computation
    concepts:
      - Static analysis and error detection
      - Debounce/throttle patterns
      - Quick fix code actions with text edits
      - WorkspaceEdit format
    skills:
      - Static code analysis implementation
      - Diagnostic message formatting with positions
      - Code action provider design
      - Debounce timer implementation
    deliverables:
      - Diagnostic publisher with debounced analysis after document changes
      - Parse error and semantic error diagnostics with severity and range
      - Diagnostic clearing on document close and error-free documents
      - Code action provider with at least one quick-fix generating WorkspaceEdit
      - Debounce timer delaying analysis until edits settle
    estimated_hours: "12-18"
```
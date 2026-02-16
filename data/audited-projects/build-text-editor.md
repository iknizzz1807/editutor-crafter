# AUDIT & FIX: build-text-editor

## CRITIQUE
- **Audit Finding #2 is WRONG**: The audit suggests Milestone 4 (Text Editing) should precede Milestone 2 (Screen Refresh). This is backwards. You need screen refresh BEFORE text editing so you can visually verify cursor movement and rendering. The original ordering (Raw Mode → Screen Refresh → File Viewing → Text Editing) is correct. Editing without visual feedback is pedagogically absurd.
- **Audit Finding #1 is VALID**: Terminal window size detection (ioctl TIOCGWINSZ or fallback via cursor-probing escape sequences) is missing. Without it, screen refresh and scrolling calculations use hardcoded dimensions, which will break on any non-standard terminal size.
- **Missing pitfall in M1**: No mention of handling SIGWINCH for terminal resize events after startup.
- **M2 pitfall is weak**: 'Flickering' is mentioned but the root cause (multiple small write() calls vs. a single buffered write) isn't explicit in the AC.
- **M3 AC is over-scoped**: Line numbers in the gutter are a nice-to-have, not a core requirement for file viewing. Tab rendering (mentioned as pitfall) should be in AC since tabs vs. spaces is a guaranteed correctness issue.
- **M5 Undo/Redo is ambitious for 3-4 hours**: Implementing a proper command pattern with undo/redo stacks for multi-operation undo is typically more than 3-4 hours, especially alongside file save. The time estimate is optimistic.
- **M7 Syntax Highlighting**: No mention of performance concerns for re-highlighting on every edit. The AC says 'multi-line strings' but doesn't address the state propagation problem (a change on line 5 can affect highlighting on line 500).
- **Gap buffer vs. array**: The project mentions gap buffer in tags and learning outcomes but never requires it in any milestone AC. Milestone 4 should explicitly address the data structure choice and its performance implications.
- **No mention of UTF-8**: The entire project assumes ASCII. Any real text editor must handle multi-byte characters for cursor positioning and rendering. This is a significant omission.

## FIXED YAML
```yaml
id: build-text-editor
name: Build Your Own Text Editor
description: Vim-like terminal text editor built from scratch using raw terminal I/O
difficulty: advanced
estimated_hours: "25-40"
essence: >
  Direct terminal control through ANSI escape sequences and raw mode configuration,
  implementing efficient text buffer manipulation with gap buffer or array-based data
  structures for insert/delete operations at cursor position.
why_important: >
  Building a text editor demystifies how tools like vim and emacs work at the system
  level, teaching low-level I/O, terminal control, and the data structure trade-offs
  that impact performance in interactive applications.
learning_outcomes:
  - Configure terminal raw mode by manipulating termios flags to disable canonical input processing
  - Detect terminal dimensions via ioctl TIOCGWINSZ and handle SIGWINCH resize events
  - Implement cursor positioning and screen rendering using ANSI escape sequences
  - Design gap buffer or array-based data structure for efficient text insertion and deletion at cursor
  - Handle edge cases in text manipulation including tab rendering, line wrapping, scrolling, and viewport management
  - Implement file I/O operations with proper error handling for loading and saving text
  - Build incremental search with real-time highlighting and cursor navigation
  - Parse and tokenize source code to implement syntax highlighting with state machines
  - Debug low-level terminal interactions and handle platform-specific termios differences
skills:
  - Terminal I/O Programming
  - ANSI Escape Sequences
  - Raw Mode Configuration
  - Gap Buffer Implementation
  - File I/O Operations
  - State Machine Design
  - Incremental Search
  - Syntax Tokenization
tags:
  - advanced
  - build-from-scratch
  - c
  - cursor
  - gap-buffer
  - go
  - rope
  - rust
  - syntax-highlighting
architecture_doc: architecture-docs/build-text-editor/index.md
languages:
  recommended:
    - C
    - Rust
    - Go
  also_possible: []
resources:
  - name: "Build Your Own Text Editor"
    url: "https://viewsourcecode.org/snaptoken/kilo/"
    type: tutorial
  - name: "antirez/kilo source"
    url: "https://github.com/antirez/kilo"
    type: reference
  - name: "Hecto (Rust version)"
    url: "https://philippflenker.com/hecto/"
    type: tutorial
prerequisites:
  - type: skill
    name: Terminal I/O
  - type: skill
    name: C or systems language
  - type: skill
    name: Basic data structures
milestones:
  - id: build-text-editor-m1
    name: Raw Mode and Input
    description: >
      Put terminal in raw mode, detect terminal dimensions, and read keypresses
      one at a time including multi-byte escape sequences.
    acceptance_criteria:
      - Raw mode disables echo, canonical buffering, and signal processing (ICANON, ECHO, ISIG, IXON, IEXTEN, ICRNL, OPOST, BRKINT, INPCK, ISTRIP flags correctly toggled)
      - Individual keypresses are read without waiting for Enter key confirmation
      - Arrow keys, Home, End, Page Up, Page Down, and Delete are recognized from multi-byte escape sequences and mapped to distinct internal key constants
      - Terminal window dimensions (rows and columns) are detected via ioctl TIOCGWINSZ with fallback to cursor-probing escape sequence method
      - Terminal is restored to original cooked mode on normal exit, on error, and on signal (SIGINT, SIGTERM) via atexit handler or equivalent
      - SIGWINCH signal handler detects terminal resize and updates stored dimensions
    pitfalls:
      - Not restoring terminal on crash or signal leaves terminal in broken state requiring 'reset' command
      - Ctrl+C sends SIGINT which kills the process before cleanup unless signal is caught or disabled
      - Different terminal emulators send different escape sequences for the same keys (xterm vs screen vs tmux)
      - Forgetting to disable OPOST causes \n to render as \r\n breaking output alignment
      - ioctl TIOCGWINSZ can fail in some environments (piped stdin); fallback cursor-probe technique must be implemented
    concepts:
      - Terminal modes (canonical vs raw)
      - termios structure and flag manipulation
      - Signal handling for cleanup (SIGINT, SIGTERM, SIGWINCH)
      - Terminal size detection methods
    skills:
      - System-level programming
      - Terminal I/O control
      - Signal handling and cleanup
      - POSIX termios API
    deliverables:
      - Terminal raw mode setup disabling echo and canonical line buffering with all required flag changes
      - Keypress reader processing individual characters and multi-byte escape sequences
      - Special key handling mapping escape sequences to internal key constants for arrow keys, home, end, page up/down, and delete
      - Terminal size detection using ioctl with cursor-probe fallback
      - Graceful cleanup restoring terminal settings on editor exit via atexit and signal handlers
    estimated_hours: "3-4"

  - id: build-text-editor-m2
    name: Screen Refresh and Cursor
    description: >
      Clear screen, render rows of tildes (empty buffer), position cursor using
      ANSI escape sequences, and display a status bar. All output must be
      batched into a single write to avoid flicker.
    acceptance_criteria:
      - Screen refresh redraws all visible rows using ANSI cursor positioning sequences (ESC[H, ESC[K, ESC[?25l/h)
      - All escape sequences and row content are appended to an in-memory write buffer and flushed in a single write() call to avoid flicker
      - Cursor movement commands (arrow keys, home, end, page up/down) update both logical position and visible cursor location
      - Status bar at the bottom shows current filename (or [No Name]), total lines, cursor row/column, and dirty indicator
      - Welcome message is displayed centered when the buffer is empty
      - Cursor position is 0-indexed internally but displayed as 1-indexed in the status bar and converted to 1-indexed for ANSI sequences
    pitfalls:
      - Writing escape sequences in multiple small write() calls causes visible flicker; must batch into single write
      - ANSI cursor positioning is 1-indexed; off-by-one errors are extremely common
      - Screen size detection from M1 must be used; hardcoded dimensions will break on non-standard terminals
      - Forgetting to hide/show cursor during redraw causes cursor flicker artifacts
      - Not clearing each line with ESC[K causes stale characters to persist when line content shrinks
    concepts:
      - VT100/ANSI escape sequences
      - Screen write buffering
      - Terminal graphics
      - Cursor coordinate systems
    skills:
      - ANSI escape sequence manipulation
      - Screen buffer management
      - Terminal graphics programming
      - Cursor positioning control
    deliverables:
      - ANSI escape sequence writer controlling cursor position, screen clearing, and cursor visibility
      - Write buffer that accumulates all output for a frame and flushes in a single write() syscall
      - Status bar displaying filename, line count, cursor position, and modification status
      - Cursor movement handling for all navigation keys keeping cursor within valid bounds
    estimated_hours: "3-4"

  - id: build-text-editor-m3
    name: File Viewing and Scrolling
    description: >
      Load a file from disk into a line-based buffer, render its contents,
      and implement vertical and horizontal scrolling with proper tab rendering.
    acceptance_criteria:
      - File loader reads specified file path and stores each line as an editable row in a dynamic array structure
      - Vertical scrolling adjusts viewport offset so cursor line is always visible on screen (scroll up when cursor above viewport, scroll down when below)
      - Horizontal scrolling adjusts column offset so cursor column is always visible for lines longer than terminal width
      - Tab characters are rendered as spaces (configurable tab stop width, default 8) with correct column accounting for cursor positioning
      - Files with no trailing newline are handled correctly without adding or losing content
      - Empty files open correctly showing the welcome screen or empty buffer
    pitfalls:
      - Storing lines with their trailing newline characters causes rendering bugs; strip newlines on load
      - Tab characters occupy variable visual width; cursor column calculation must use rendered width not byte offset
      - Memory allocation for lines must handle arbitrarily long lines without truncation
      - Horizontal scroll offset must be accounted for in both rendering and cursor positioning
      - Large files may cause noticeable load time; line storage should use efficient dynamic array growth
    concepts:
      - File I/O and line parsing
      - Dynamic arrays with amortized growth
      - Viewport scrolling and offset management
      - Tab stop rendering
    skills:
      - File I/O operations
      - Dynamic memory management
      - Viewport scrolling implementation
      - Line-based text buffer manipulation
    deliverables:
      - File loading reading text file contents into line-based dynamic array buffer structure
      - Vertical scrolling adjusting viewport row offset when cursor moves beyond visible area
      - Horizontal scrolling adjusting viewport column offset for lines wider than terminal width
      - Tab rendering converting tab characters to spaces at correct tab stop positions
    estimated_hours: "3-5"

  - id: build-text-editor-m4
    name: Text Editing
    description: >
      Implement character insertion, deletion, line splitting, and line joining.
      Choose and implement either a gap buffer or simple dynamic array for
      each row's character storage.
    acceptance_criteria:
      - Inserting a character at cursor position shifts remaining text right and advances cursor by one column
      - Backspace removes the character before cursor and moves cursor left; Delete removes the character at cursor without moving
      - Enter key splits current line into two lines at cursor position, creating a new line below with the text after the cursor
      - Backspace at beginning of a line (column 0) appends current line's content to the previous line and removes the current line
      - Each row stores characters in a gap buffer or dynamic array that provides O(1) amortized insert/delete at cursor position
      - A dirty flag is set on any edit operation and displayed in the status bar to indicate unsaved changes
      - Tab rendering from M3 remains correct after insertions and deletions within lines containing tabs
    pitfalls:
      - Cursor at end of line must allow appending characters but not advancing past the last character in navigation mode
      - Deleting at the beginning of the first line or backspacing at position (0,0) must be a no-op
      - Memory reallocation on every character insert is O(n); use amortized doubling strategy or gap buffer
      - Forgetting to update the rendered (tab-expanded) version of a row after edits causes display corruption
      - Line split/join must correctly update total line count and adjust cursor position
    concepts:
      - Text buffer data structures (gap buffer, dynamic array, piece table)
      - Amortized complexity for insert/delete operations
      - Dirty tracking for unsaved changes
    skills:
      - In-place text buffer editing
      - String manipulation and insertion
      - Memory reallocation strategies
      - Line joining and splitting operations
    deliverables:
      - Character insertion adding typed characters at current cursor position in the row buffer
      - Character deletion removing characters with backspace and delete keys
      - Line splitting creating a new line below when Enter is pressed at cursor position
      - Line joining merging current line with previous when backspacing at column 0
      - Dirty flag tracking modification state displayed in status bar
    estimated_hours: "5-7"

  - id: build-text-editor-m5
    name: Save, Quit Confirmation, and Undo
    description: >
      Save buffer to disk with error handling, prompt for confirmation on
      quit with unsaved changes, and implement undo/redo.
    acceptance_criteria:
      - Save writes all buffer lines joined by newlines to the file path and clears the dirty flag, displaying byte count and line count in status message
      - Save-as prompts for a filename when no filename is set (new file) and stores the provided name for future saves
      - Quit with unsaved changes requires the user to press Ctrl-Q multiple times (e.g., 3 times) or confirm, preventing accidental data loss
      - Write errors (permission denied, disk full) are caught and displayed as status bar messages without crashing
      - Undo reverses the most recent edit operation (insert, delete, split, join) restoring the previous buffer and cursor state
      - Redo re-applies the last undone operation; performing a new edit after undo clears the redo stack
      - Undo/redo history supports at least 100 operations without excessive memory usage
    pitfalls:
      - Writing to the original file directly risks data loss on error; consider writing to a temp file and renaming (atomic save)
      - Undo across line splits/joins must restore both line content and cursor position
      - Grouping rapid sequential character inserts into a single undo operation improves UX but adds complexity
      - Memory for undo history can grow unboundedly; cap history length or total memory usage
      - Forgetting to clear the redo stack on new edits causes confusing redo behavior
    concepts:
      - File writing with error handling
      - Atomic file save pattern
      - Undo architectures (command pattern, memento)
      - Redo stack invalidation
    skills:
      - File system operations and error handling
      - Command pattern implementation
      - Undo/redo architecture design
      - User interaction and confirmation prompts
    deliverables:
      - File saving writing buffer contents to disk with status confirmation message
      - Save-as prompt for unnamed files requesting filename input
      - Quit confirmation preventing accidental exit with unsaved changes
      - Undo stack recording edit operations for reversal
      - Redo stack allowing re-application of previously undone operations
    estimated_hours: "4-6"

  - id: build-text-editor-m6
    name: Incremental Search
    description: >
      Implement incremental search that highlights matches in real-time
      as the user types the query, with forward/backward navigation.
    acceptance_criteria:
      - Incremental search updates the highlighted match and moves the cursor to the first match after each character typed in the query
      - Forward search (e.g., arrow down or Ctrl-N) finds the next occurrence after current cursor position, wrapping to the start of the document
      - Backward search (e.g., arrow up or Ctrl-P) finds the previous occurrence before cursor, wrapping to the end of the document
      - Escape key cancels search and restores cursor to the original position before search began
      - Enter key confirms search and leaves cursor at the current match position
      - Search query is displayed in the status/message bar with visual feedback
    pitfalls:
      - Search wrapping at end/beginning of file must handle correctly without infinite loops when no matches exist
      - Case sensitivity should be consistent (default case-insensitive or configurable)
      - Restoring scroll position and cursor on cancel requires saving state before search begins
      - Incremental re-search on every keypress can be slow on very large files; consider limiting search scope to visible region for highlight
    concepts:
      - Text searching algorithms
      - Incremental search UX patterns
      - State save and restore
    skills:
      - String searching algorithms
      - Incremental UI state management
      - Pattern matching implementation
      - Search result navigation
    deliverables:
      - Incremental search highlighting and navigating to matches as user types search query
      - Forward and backward search navigation jumping between match occurrences with wrapping
      - Search prompt in status bar area accepting query input with special key handling
      - State save/restore preserving cursor and scroll position on search cancel
    estimated_hours: "2-4"

  - id: build-text-editor-m7
    name: Syntax Highlighting
    description: >
      Add syntax highlighting using a state machine tokenizer that colors
      keywords, strings, comments, and numbers for at least one language.
    acceptance_criteria:
      - File extension maps to the correct syntax highlighting ruleset for that language (e.g., .c → C, .py → Python)
      - Keywords (if, else, for, while, return, etc.) and type names (int, char, etc.) are displayed in distinct highlight colors using separate color groups
      - String literals enclosed in single or double quotes are highlighted, including handling of escape sequences within strings (e.g., \" does not end the string)
      - Single-line comments (// or #) and multi-line block comments (/* ... */) are highlighted differently from code
      - Number literals (integers, floats, hex) are highlighted in a distinct color
      - Multi-line comment and string state is propagated across lines: a change on one line that opens or closes a multi-line construct triggers re-highlighting of subsequent lines
      - Highlighting is updated after each edit without noticeable delay for files under 10,000 lines
    pitfalls:
      - Multi-line strings and comments require tracking highlight state at the start of each line; a naive per-line tokenizer breaks on these constructs
      - Escape sequences in strings (e.g., \", \\) must be handled to avoid prematurely ending the string highlight
      - Performance degrades on large files if every edit triggers full-file re-tokenization; track per-line highlight state and only re-highlight from the changed line forward
      - Relying solely on regex for tokenization leads to incorrect results for nested or context-dependent constructs
      - ANSI color codes add bytes to the output buffer; cursor positioning must not count these as visible columns
    concepts:
      - Lexical analysis and tokenization
      - Finite state machines for syntax parsing
      - ANSI SGR color codes
      - Incremental re-highlighting strategies
    skills:
      - Lexical analysis and tokenization
      - State machine design
      - ANSI color code application
      - Performance optimization for interactive editing
    deliverables:
      - Language detection selecting syntax rules based on file extension
      - Keyword and type highlighting coloring reserved words and type names in distinct colors
      - String and comment highlighting with multi-line span support and escape sequence handling
      - Number literal highlighting distinguishing numeric values from identifiers
      - Per-line highlight state tracking enabling incremental re-highlighting after edits
    estimated_hours: "5-8"
```
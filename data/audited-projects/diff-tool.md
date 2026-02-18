# AUDIT & FIX: diff-tool

## CRITIQUE
- **Encoding detection (M1 AC 1)**: Claiming to 'handle different encodings such as UTF-8 and Latin-1' with automatic detection is an unreasonably hard problem for a beginner project. Charset detection (chardet, etc.) is heuristic and unreliable. A proper approach is to assume UTF-8 (the de facto standard) and fail gracefully on decoding errors, or accept an encoding flag.
- **LCS space complexity (M2 AC 4)**: Stating 'Achieve O(mn) time and space complexity' as an acceptance criterion is technically accurate for the naive LCS but is a performance bottleneck for any non-trivial file. The pitfall mentions Hirschberg but the AC actually requires O(mn) space as a *goal*, which is contradictory. The AC should accept O(mn) time but explicitly require at minimum a rolling-array O(min(m,n)) space optimization.
- **M2 deliverable contradiction**: The deliverables list 'Memory optimization to reduce space usage for large files' but the AC requires O(mn) space. These are contradictory.
- **M3 off-by-one**: The pitfall mentions 1-indexed line numbers but no AC explicitly requires verifying against `diff -u` output for correctness.
- **M4 side-by-side**: The deliverables mention 'Side-by-side display option' but no AC requires it. Deliverables must match ACs.
- **Missing**: No AC for handling binary files gracefully (detecting and refusing or warning).
- **Missing**: No AC for the unified diff header lines (`--- a/file` and `+++ b/file`).
- **Exit code semantics**: M4 AC 5 correctly mentions exit codes (0 same, 1 different) which matches POSIX diff behavior—good.
- **Myers' algorithm**: The essence mentions Myers' O(ND) but no milestone actually implements it. Either remove it from the essence or add an optional milestone.

## FIXED YAML
```yaml
id: diff-tool
name: Diff Tool
description: Text diff using LCS algorithm with unified output format
difficulty: beginner
estimated_hours: "12-18"
essence: >
  Sequence comparison through dynamic programming to compute the Longest
  Common Subsequence between two text files, then transforming the LCS result
  into a minimal edit script rendered in unified diff format with context
  lines and hunk headers.
why_important: >
  Building this teaches core algorithmic thinking through dynamic programming
  tables, backtracking, and space-time tradeoffs while creating a practical
  tool that powers version control systems like Git.
learning_outcomes:
  - Implement Longest Common Subsequence with dynamic programming tabulation and backtracking
  - Optimize LCS space complexity using rolling arrays to O(min(m,n))
  - Generate unified diff format with context lines, hunk headers, and file headers
  - Parse and tokenize text files into comparable line-based sequences
  - Build edit scripts that classify lines as added, deleted, or unchanged
  - Implement command-line argument parsing with multiple output formats
  - Debug algorithm correctness using small test cases and edge conditions
skills:
  - Dynamic Programming
  - Algorithm Analysis
  - String Processing
  - Edit Distance Concepts
  - CLI Development
  - File I/O Operations
  - Output Formatting
  - Space Optimization
tags:
  - algorithms
  - beginner-friendly
  - go
  - hunks
  - javascript
  - lcs
  - patch
  - python
  - tool
architecture_doc: architecture-docs/diff-tool/index.md
languages:
  recommended:
    - Python
    - JavaScript
    - Go
  also_possible:
    - C
    - Rust
    - Java
resources:
  - name: "Myers' Diff Algorithm Tutorial"
    url: "http://simplygenius.net/Article/DiffTutorial1"
    type: tutorial
  - name: "The Myers Difference Algorithm"
    url: "https://nathaniel.ai/myers-diff/"
    type: article
  - name: Wikipedia - LCS
    url: "https://en.wikipedia.org/wiki/Longest_common_subsequence"
    type: reference
prerequisites:
  - type: skill
    name: Dynamic programming basics
  - type: skill
    name: File I/O
  - type: skill
    name: String manipulation
milestones:
  - id: diff-tool-m1
    name: Line Tokenization
    description: Read two text files and split into line arrays for comparison.
    estimated_hours: "2-3"
    concepts:
      - File encoding
      - Line endings
      - Text normalization
    skills:
      - File I/O operations
      - String manipulation and parsing
      - Line ending normalization
      - Error handling for encoding issues
    acceptance_criteria:
      - Read files assuming UTF-8 encoding; accept an optional --encoding flag to override
      - On decoding errors, print a warning to stderr and either abort or replace invalid bytes (configurable)
      - Detect and warn (but do not crash) if input appears to be a binary file (contains null bytes in first 8KB)
      - Split content by newlines preserving empty lines in the sequence; normalize CRLF and CR to LF
      - Track whether each file ends with a trailing newline (required for correct unified diff output)
      - Report line counts for each input file to stderr in verbose mode
    pitfalls:
      - Automatic encoding detection (chardet) is unreliable; default to UTF-8 and let user override
      - Binary files will cause encoding errors or produce meaningless diffs; detect and warn early
      - Trailing newline presence affects diff output ('No newline at end of file' marker)
      - Large files can exhaust memory if loaded entirely; for this beginner project, full load is acceptable but note the limitation
    deliverables:
      - File reader with configurable encoding and error handling
      - Line splitter normalizing line endings to LF
      - Binary file detection heuristic (null byte check)
      - Trailing newline tracking for each input file

  - id: diff-tool-m2
    name: LCS Algorithm
    description: Implement Longest Common Subsequence using dynamic programming.
    estimated_hours: "4-5"
    concepts:
      - Dynamic programming
      - 2D matrices
      - Backtracking
      - Space optimization
    skills:
      - Dynamic programming implementation
      - 2D array manipulation
      - Algorithm optimization
      - Space-time complexity analysis
    acceptance_criteria:
      - Build LCS length table from both input line sequences with O(mn) time complexity
      - Backtrack through the table to recover the actual longest common subsequence
      - Handle edge cases: empty files, identical files, completely different files, single-line files
      - Optimize space to O(min(m,n)) using a rolling two-row array (do NOT allocate full m×n matrix for files > 10K lines)
      - Verify LCS correctness: the recovered subsequence length must equal the value in the table cell [m][n]
      - For files up to 10,000 lines each, complete in under 5 seconds on commodity hardware
    pitfalls:
      - Off-by-one errors in matrix indexing (table is (m+1) × (n+1), indices are 1-based)
      - Not handling empty sequences (0-length file should produce all-additions or all-deletions diff)
      - Full O(mn) space matrix causes OOM for large files; rolling array optimization is essential
      - Backtracking requires the full matrix OR Hirschberg's divide-and-conquer; rolling array alone loses backtrack path—use two-pass Hirschberg or accept full matrix for backtracking with size limits
    deliverables:
      - LCS length computation with rolling-array space optimization
      - Backtracking implementation to recover actual LCS (may require full matrix for small inputs or Hirschberg for large)
      - Edge case handling for empty, identical, and fully different inputs
      - Performance test demonstrating acceptable runtime for 10K-line files

  - id: diff-tool-m3
    name: Diff Generation
    description: Convert LCS result into unified diff format with hunks and context.
    estimated_hours: "4-5"
    concepts:
      - Edit scripts
      - Unified diff format
      - Hunk generation
    skills:
      - Data structure transformation
      - Output formatting
      - Line number tracking (1-indexed)
      - Context window management
    acceptance_criteria:
      - Classify each line as unchanged (context), added (+), or deleted (-) based on LCS result
      - Generate unified diff file headers ('--- a/file1' and '+++ b/file2') with timestamps or labels
      - Generate @@ hunk headers with correct 1-indexed line ranges for both files (e.g., @@ -1,3 +1,4 @@)
      - Group changes into hunks with 3 lines of context by default; merge hunks whose context lines overlap
      - Append 'No newline at end of file' marker when applicable (per POSIX diff spec)
      - Verify output by piping to 'patch' and confirming it applies cleanly to produce the target file
    pitfalls:
      - Unified diff format is 1-indexed; off-by-one errors are the most common bug
      - Hunk line counts must accurately reflect the number of lines in each side (including context)
      - Context overlap between adjacent hunks requires merging into a single hunk
      - Files with no common lines produce a single hunk covering everything
      - The 'No newline at end of file' marker must appear on its own line starting with backslash
    deliverables:
      - Edit script generator classifying lines from LCS comparison
      - Hunk builder grouping nearby changes with configurable context
      - Hunk merger for overlapping context regions
      - Unified diff formatter producing patch-compatible output

  - id: diff-tool-m4
    name: CLI and Color Output
    description: Build command-line interface with colored output and standard options.
    estimated_hours: "3-4"
    concepts:
      - CLI argument parsing
      - ANSI colors
      - Exit codes
      - TTY detection
    skills:
      - Command-line argument parsing
      - Terminal output formatting
      - TTY detection for smart defaults
      - Process exit code management
    acceptance_criteria:
      - Accept two file paths as positional command-line arguments
      - Color output: red for deletions (-), green for additions (+), cyan for hunk headers (@@); enabled by default when stdout is a TTY
      - "--no-color" flag disables ANSI codes; auto-disable when output is piped (not a TTY)
      - "--context N" flag sets the number of context lines (default 3); validate N >= 0
      - "--unified" flag (default) selects unified format; "--side-by-side" flag selects side-by-side format
      - Exit code 0 if files are identical, 1 if files differ, 2 on error (matching POSIX diff convention)
      - Print usage/help on --help or invalid arguments
    pitfalls:
      - ANSI codes corrupt output when piped to a file; must detect TTY (isatty)
      - Windows terminals need special handling for ANSI (enable virtual terminal processing or use a library)
      - Exit code 2 for errors is the POSIX convention; many forget to distinguish error from 'files differ'
      - Side-by-side format requires knowing terminal width (ioctl TIOCGWINSZ or environment variable COLUMNS)
    deliverables:
      - Argument parser for file paths, flags, and options with validation
      - ANSI color output with automatic TTY detection
      - Context line count configuration via --context flag
      - Side-by-side display option computing column widths from terminal size
      - Correct exit codes following POSIX diff conventions
```
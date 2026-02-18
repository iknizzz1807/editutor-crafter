# AUDIT & FIX: json-parser

## CRITIQUE
- **Number Validation Placement (Confirmed):** M3 includes 'Number format validation rejects leading zeros, lone minus signs, and trailing decimal points.' This is logically a lexer responsibility (M1), not a post-parsing error handling step. The tokenizer should reject malformed number literals at scan time, not after the parse tree is built.
- **Stack Overflow Claim (Confirmed):** M2 says 'without stack overflow up to configurable depth' but then uses recursive descent, which IS bounded by the system stack. For truly deep nesting (10,000+ levels), you need an explicit stack on the heap. The AC should either limit depth and be honest about it, or require an explicit stack implementation.
- **M1 Missing: Number Scanning Rigor:** The JSON spec has very specific number rules (no leading zeros except 0 itself, no trailing decimal, must have digit after decimal, etc.). M1 lists these as pitfalls but doesn't include them as AC. If the tokenizer accepts '007' as valid, the parser can't reject it later without re-examining the lexeme.
- **Missing: Trailing Comma Policy:** JSON spec forbids trailing commas, but many real-world JSON generators produce them. The AC should explicitly state that trailing commas are a syntax error per RFC 8259.
- **Missing: Top-Level Value Constraint:** RFC 8259 allows any JSON value at the top level (not just object/array). Earlier RFC 4627 required object/array. The AC should specify which RFC is being implemented.
- **M2 Missing: Duplicate Key Handling:** M3 lists duplicate keys as a pitfall but no AC addresses them. RFC 8259 says keys SHOULD be unique but doesn't make duplicates a syntax error. The AC should document the chosen behavior.
- **Unicode Surrogate Pairs (M3):** Listed as an AC but implementing UTF-16 surrogate pair decoding (\uD800\uDC00) is significantly complex and arguably beyond beginner scope. Should be clearly marked as advanced or optional with basic \uXXXX being the baseline.
- **Missing: Conformance Test Suite:** The project should reference the JSONTestSuite (github.com/nst/JSONTestSuite) as a validation tool.

## FIXED YAML
```yaml
id: json-parser
name: JSON Parser
description: "RFC 8259-compliant JSON parser with lexer, recursive descent parser, and comprehensive error handling."
difficulty: beginner
estimated_hours: "10-15"
essence: >
  Lexical analysis via character scanning to tokenize JSON character
  streams into typed tokens, followed by recursive descent parsing to
  validate context-free grammar rules and construct native hierarchical
  data structures, with strict RFC 8259 compliance for number formats,
  string escapes, and structural validation.
why_important: >
  Building a JSON parser teaches foundational compiler construction
  skills (lexing, recursive descent parsing, error reporting) that
  apply directly to building interpreters, DSLs, configuration parsers,
  API processors, and understanding how programming languages work
  internally.
learning_outcomes:
  - Implement a lexer that tokenizes JSON using character-by-character scanning
  - Validate JSON number formats per RFC 8259 at the lexer level
  - Build a recursive descent parser that produces native data structures
  - Handle string escape sequences including basic Unicode escapes
  - Implement nesting depth limits to prevent stack overflow
  - Generate error messages with line number and column position
  - Validate against the JSON specification (RFC 8259)
  - Test against the JSONTestSuite conformance corpus
skills:
  - Lexical Analysis / Tokenization
  - Recursive Descent Parsing
  - Error Handling with Position Tracking
  - String Escape Processing
  - RFC Specification Compliance
  - Context-Free Grammar Implementation
tags:
  - beginner-friendly
  - c
  - compilers
  - configuration
  - javascript
  - parsing
  - python
  - recursive-descent
  - tokenization
  - validation
architecture_doc: architecture-docs/json-parser/index.md
languages:
  recommended:
    - Python
    - JavaScript
    - C
  also_possible:
    - Rust
    - Go
resources:
  - name: "JSON Specification (RFC 8259)"
    url: https://www.rfc-editor.org/rfc/rfc8259
    type: specification
  - name: "JSON.org Visual Grammar"
    url: https://www.json.org/json-en.html
    type: specification
  - name: Crafting Interpreters
    url: https://craftinginterpreters.com/
    type: book
  - name: JSONTestSuite
    url: https://github.com/nst/JSONTestSuite
    type: test-suite
prerequisites:
  - type: skill
    name: Basic programming (strings, arrays, dictionaries)
  - type: skill
    name: Understanding of JSON format (objects, arrays, strings, numbers)
milestones:
  - id: json-parser-m1
    name: "Tokenizer with Strict Number Validation"
    description: >
      Build a lexer that converts a JSON input string into a stream of
      typed tokens, with RFC 8259-compliant number validation at scan time.
    acceptance_criteria:
      - Tokenizer emits tokens for: "string, number, true, false, null, '{', '}', '[', ']', ':', ','"
      - "Whitespace (space, tab, newline, carriage return) is consumed between tokens without emitting"
      - String tokens handle escape sequences: \\\", \\\\, \\/, \\b, \\f, \\n, \\r, \\t
      - String tokens handle basic Unicode escapes: \\uXXXX where X is a hex digit
      - Number validation per RFC 8259: "no leading zeros (except '0' itself), no trailing decimal point, no lone minus"
      - "Numbers support optional minus, integer part, optional decimal fraction, optional exponent (e/E with optional +/-)"
      - "Invalid number formats (e.g., '007', '1.', '.5', '+1') produce error tokens with position"
      - "Unterminated strings produce error with line and column of the opening quote"
      - "Each token records line number and column number"
      - "EOF token emitted at end of input"
    pitfalls:
      - "Accepting '007' as valid number—JSON forbids leading zeros"
      - "Accepting '.5' or '1.'—JSON requires digit before and after decimal point"
      - "Not handling escaped forward slash (\\/) which is valid JSON"
      - Scientific notation: "'1e10', '1E-3', '1.5e+2' are all valid; '1e' is not"
      - "Unterminated string spanning multiple lines—JSON strings cannot contain literal newlines"
    concepts:
      - Lexical analysis
      - Token types
      - RFC 8259 number grammar
      - Escape sequence handling
    skills:
      - Character-by-character scanning
      - Number format validation from specification
      - Escape sequence processing
      - Position tracking
    deliverables:
      - "Token type enumeration for all JSON token categories"
      - "Token struct with type, value, line, and column"
      - "String tokenizer with escape sequence handling"
      - "Number tokenizer with RFC 8259 format validation"
      - "Keyword scanner for true, false, null"
      - "Whitespace consumer"
      - "Error token emission with position"
    estimated_hours: "3-4"

  - id: json-parser-m2
    name: "Recursive Descent Parser"
    description: >
      Parse the token stream into native language data structures (dict,
      list, string, number, bool, None/null) using recursive descent.
    acceptance_criteria:
      - "Parser consumes token stream and returns native data structure (dict/list/string/number/bool/null)"
      - Objects parse as key-value pairs: keys must be strings, values can be any JSON type
      - "Arrays parse as ordered lists of any JSON type"
      - Nested structures parse correctly: objects containing arrays containing objects, etc.
      - "Empty objects {} and empty arrays [] parse correctly"
      - "Trailing commas are rejected as syntax errors per RFC 8259 (e.g., [1,2,] is invalid)"
      - Any JSON value type is accepted at the top level (per RFC 8259: not just objects/arrays)
      - "Nesting depth is limited to a configurable maximum (e.g., 512); exceeding produces error"
      - "Extra tokens after the root value produce an error (e.g., '{}[]' is invalid)"
      - Syntax errors produce descriptive messages: "'Expected : but found , at line 3, column 12'"
    pitfalls:
      - "Accepting trailing commas silently—common in JavaScript but invalid JSON"
      - "Stack overflow on deeply nested input—must enforce depth limit"
      - Empty object/array: forgetting to check for closing brace/bracket immediately after opening
      - Duplicate keys: RFC 8259 says SHOULD be unique; decide behavior (last-wins or error) and document
      - "Not consuming the full input—silently ignoring garbage after the root value"
    concepts:
      - Recursive descent parsing
      - JSON grammar rules
      - Depth limiting
      - Error reporting
    skills:
      - Recursive function design
      - Token consumption and expectation
      - Native data structure construction
      - Error message formatting
    deliverables:
      - "parse_value() dispatching to parse_object, parse_array, parse_string, parse_number, parse_bool, parse_null"
      - parse_object() reading key: value pairs into dictionary
      - "parse_array() reading values into list"
      - "Nesting depth counter with configurable limit"
      - "Trailing token check after root value"
      - "Error messages with line, column, and expected vs found token"
    estimated_hours: "3-4"

  - id: json-parser-m3
    name: "Edge Cases, Unicode & Conformance Testing"
    description: >
      Handle advanced edge cases, Unicode surrogate pairs (optional),
      and validate against the JSONTestSuite conformance corpus.
    acceptance_criteria:
      - "All 'y_' (must accept) tests from JSONTestSuite pass"
      - "All 'n_' (must reject) tests from JSONTestSuite are correctly rejected with errors"
      - "Basic Unicode escapes (\\u0041 = 'A') are decoded to proper characters in output strings"
      - "UTF-16 surrogate pairs (\\uD83D\\uDE00) are decoded to correct UTF-8/native character (advanced, may be optional)"
      - "Lone surrogate (\\uD800 without low surrogate) is rejected as an error"
      - "Numbers at extremes of float precision parse without crash (e.g., 1e308, 1e-308)"
      - "Deeply nested input at depth limit is accepted; depth limit + 1 is rejected"
      - "Empty string input produces an error (not a crash or empty result)"
      - "Parser handles all 'i_' (implementation-defined) tests with documented behavior"
    pitfalls:
      - Surrogate pair decoding is complex: high surrogate (D800-DBFF) must be followed by low surrogate (DC00-DFFF)
      - Float precision: very large or very small exponents may produce infinity or zero—decide behavior
      - "JSONTestSuite includes tricky edge cases like '0e1' (valid), '0e' (invalid), '\\u0000' (null character in string)"
      - Null bytes inside strings: some implementations crash; must handle gracefully
    concepts:
      - Conformance testing
      - Unicode processing
      - Edge case handling
      - Specification compliance
    skills:
      - Test suite integration
      - Unicode encoding/decoding
      - Specification reading and implementation
      - Boundary condition testing
    deliverables:
      - JSONTestSuite integration: automated run of y_, n_, and i_ test cases
      - "Unicode \\uXXXX basic escape decoding"
      - "Surrogate pair decoding (optional/advanced)"
      - "Documented behavior for implementation-defined cases"
      - Edge case handling: empty input, extreme numbers, null bytes
    estimated_hours: "3-5"
```
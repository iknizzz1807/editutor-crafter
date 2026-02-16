# AUDIT & FIX: markdown-renderer

## CRITIQUE
- **Missing Link Reference Definitions**: `[text][id]` and `[id]: url` style link references are a core CommonMark feature and a major source of parser complexity (they require two-pass parsing: first pass collects definitions, second pass resolves references). Completely absent.
- **Escaping Phase Incorrect**: The audit correctly notes that escaping should happen during HTML generation, but the milestones don't clearly separate parsing from rendering. Escaping raw HTML entities in the source (`&amp;` etc.) needs to happen at parse time, while escaping *generated* content happens at render time. These are different operations.
- **Raw HTML Pass-Through Missing**: CommonMark allows inline HTML (e.g., `<div>content</div>`) to pass through to the output. The milestones don't address this, but it's a spec requirement AND a security concern (XSS).
- **XSS Sanitization Critical**: M4 mentions 'Input Sanitization' as a skill but there's no acceptance criterion for preventing XSS attacks in generated HTML. If raw HTML pass-through is supported, `<script>` tags MUST be handled.
- **List Parsing Underspecified**: The list milestone is oversimplified. CommonMark list parsing is one of the HARDEST parts of the spec—it involves 'lazy continuation lines', complex indentation rules (continuation content must be indented to the column after the list marker), and interaction between lists and other block elements.
- **Missing Blockquote in M1 Deliverables but Not AC**: Blockquote parser is in M1 deliverables but blockquote handling has no acceptance criteria.
- **AST Not Required in Any AC**: The project describes building an AST in the learning outcomes but no milestone requires one. A student could use regex-replace and 'pass'. The AST should be an explicit deliverable.
- **Tables Extension Missing**: While not in core CommonMark, GFM-style tables are ubiquitous and would make the project more complete. Could be an optional extension.
- **Difficulty Rating Questionable**: 'Beginner' is appropriate for basic Markdown, but handling the FULL CommonMark spec (link references, list edge cases, raw HTML) is intermediate at minimum.

## FIXED YAML
```yaml
id: markdown-renderer
name: Markdown Renderer
description: >-
  Markdown to HTML converter implementing CommonMark specification with
  block parsing, inline formatting, nested lists, link references, and
  HTML generation with XSS prevention.
difficulty: beginner-to-intermediate
estimated_hours: "15-22"
essence: >-
  Two-pass text parsing with block-level state machine and inline-level
  delimiter matching to tokenize Markdown syntax, construct an abstract
  syntax tree representing document structure, resolve link references,
  and generate valid sanitized HTML through recursive tree traversal.
why_important: >-
  Building this project develops fundamental parser and compiler skills
  applicable to template engines, code formatters, and DSL interpreters,
  while teaching how text processing tools used by millions of developers
  daily actually work under the hood.
learning_outcomes:
  - Implement two-pass block parsing (first pass identifies block structure, second pass processes inline content)
  - Build a recursive descent or state-machine parser for nested block and inline elements
  - Design an abstract syntax tree (AST) representing document hierarchy
  - Handle CommonMark list parsing with indentation tracking and continuation lines
  - Implement link reference definitions with two-pass resolution
  - Generate valid HTML with proper escaping and optional XSS sanitization
  - Write test suites using CommonMark spec examples to verify compliance
skills:
  - Block-Level Parsing
  - Inline Parsing with Delimiter Matching
  - AST Construction
  - State Machines
  - Tree Traversal
  - Link Reference Resolution
  - HTML Generation
  - XSS Prevention
tags:
  - beginner-friendly
  - go
  - html-generation
  - javascript
  - parser
  - python
  - syntax
architecture_doc: architecture-docs/markdown-renderer/index.md
languages:
  recommended:
    - Python
    - JavaScript
    - Go
  also_possible:
    - Rust
    - Ruby
resources:
  - name: CommonMark Spec
    url: https://spec.commonmark.org/
    type: specification
  - name: CommonMark Spec Examples (Test Suite)
    url: https://spec.commonmark.org/0.31.2/
    type: specification
  - name: "CommonMark Parsing Strategy"
    url: https://spec.commonmark.org/0.31.2/#appendix-a-parsing-strategy
    type: specification
prerequisites:
  - type: skill
    name: Regular expressions
  - type: skill
    name: String manipulation
  - type: skill
    name: HTML basics
  - type: skill
    name: Tree data structures
milestones:
  - id: markdown-renderer-m1
    name: Block Elements & AST Construction
    description: >-
      Parse block-level elements (headings, paragraphs, code blocks,
      blockquotes, horizontal rules) and construct an abstract syntax
      tree representing document structure.
    acceptance_criteria:
      - Parse input line-by-line, classifying each line as belonging to a block type
      - ATX headings (# through ######) parsed into heading AST nodes with level 1-6
      - Consecutive non-blank lines grouped into paragraph AST nodes
      - Blank lines separate paragraphs and close open blocks
      - Fenced code blocks (triple backticks or triple tildes) parsed into code block AST nodes with optional language identifier
      - Indented code blocks (4+ spaces or 1 tab indentation) parsed into code block AST nodes
      - Blockquotes (lines starting with >) parsed into blockquote AST nodes supporting nesting (>> for nested)
      - Horizontal rules (3+ dashes, asterisks, or underscores with optional spaces) parsed into thematic break AST nodes
      - Build AST with document root containing ordered list of block-level child nodes
      - Collect link reference definitions ([label]: url "title") during first pass and store in reference map
      - Link reference definitions are NOT rendered as output; they only provide URL targets for reference links
      - Verify AST structure by implementing a debug print showing tree hierarchy
    pitfalls:
      - Setext headings (underlined with === or ---) require lookahead and conflict with horizontal rules and list items
      - Indented code blocks inside list items require tracking list indentation context
      - Fenced code block content is NOT parsed for Markdown syntax; everything inside is literal text
      - Blockquote continuation lines (without >) are allowed for lazy continuation per CommonMark
      - Link reference definitions can appear ANYWHERE in the document but must be collected before inline parsing
      - Reference labels are case-insensitive and whitespace-normalized
    concepts:
      - Two-pass parsing strategy (blocks first, then inlines)
      - Abstract syntax tree construction
      - Line-by-line block classification
      - Link reference definition collection
    skills:
      - Text parsing and line classification
      - AST design and construction
      - State machine for block context tracking
      - Map/dictionary for reference storage
    deliverables:
      - Line-by-line block parser classifying lines into block types
      - AST with node types: document, heading, paragraph, code_block, blockquote, thematic_break
      - ATX heading parser (# through ######)
      - Fenced code block parser with language identifier extraction
      - Indented code block parser
      - Blockquote parser with nesting support
      - Link reference definition collector
      - AST debug printer for verification
    estimated_hours: "4-6"

  - id: markdown-renderer-m2
    name: Inline Elements & Delimiter Matching
    description: >-
      Parse inline formatting within block-level content: emphasis,
      code spans, links, images, and autolinks with proper delimiter
      matching and reference resolution.
    acceptance_criteria:
      - Process inline content within each block node's text content (paragraphs, headings, blockquote content)
      - Bold (**text** or __text__) produces strong AST nodes
      - Italic (*text* or _text_) produces emphasis AST nodes
      - Nested emphasis (**bold *and italic* text**) produces correctly nested strong and emphasis nodes
      - Underscore emphasis does NOT trigger in the middle_of_words (intraword emphasis rules per CommonMark)
      - Inline code (`code`) produces code_span AST nodes preserving inner whitespace; backtick strings can use multiple backticks (`` ` ``)
      - Links [text](url "title") produce link AST nodes with text, url, and optional title
      - Reference links [text][ref] and [text] (shortcut) resolve against collected reference definitions
      - Images ![alt](url "title") produce image AST nodes with alt text, url, and optional title
      - Backslash escaping (\*) produces literal characters instead of triggering Markdown syntax
      - Hard line breaks: two trailing spaces or backslash before newline produce <br> elements
      - Mismatched delimiters (**bold* is handled gracefully (literal asterisks, not malformed HTML)
    pitfalls:
      - CommonMark emphasis parsing uses a delimiter stack algorithm, NOT simple regex; regex cannot handle nesting correctly
      - Underscore rules are complex: _foo_bar is not emphasis but *foo*bar IS emphasis
      - Inline code backtick matching: the opening and closing backtick strings must have the same number of backticks
      - Link text can contain other inline formatting (bold, italic, code) but NOT nested links
      - Reference link labels are case-insensitive; [FOO] and [foo] refer to the same reference
      - Backslash escapes only work for ASCII punctuation characters
    concepts:
      - Delimiter stack algorithm for emphasis parsing
      - Inline element nesting rules
      - Reference link resolution
      - Backslash escape handling
    skills:
      - Delimiter matching algorithms
      - Recursive inline parsing
      - Reference resolution from collected definitions
      - Escape sequence handling
    deliverables:
      - Emphasis parser using delimiter stack for bold and italic with nesting
      - Inline code span parser handling single and multiple backtick delimiters
      - Link parser for inline links [text](url) and reference links [text][ref]
      - Image parser for ![alt](url)
      - Backslash escape handler for literal punctuation
      - Hard line break detection (trailing spaces or backslash)
      - Inline AST nodes: strong, emphasis, code_span, link, image, text, hard_break
    estimated_hours: "4-6"

  - id: markdown-renderer-m3
    name: Lists with Nesting
    description: >-
      Parse ordered and unordered lists with proper nesting based on
      indentation, tight/loose distinction, and list continuation rules.
    acceptance_criteria:
      - Unordered list markers (-, *, +) followed by space start list items
      - Ordered list markers (1., 2., etc.) followed by space start ordered list items
      - Ordered lists start with their first item's number (1. starts at 1, 3. starts at 3)
      - Nested lists detected by indentation: content indented to the column after the list marker belongs to the parent item
      - Support at least 3 levels of list nesting with correct HTML structure
      - Tight lists (no blank lines between items) render items WITHOUT <p> tags inside <li>
      - Loose lists (blank lines between any items) render ALL items WITH <p> tags inside <li>
      - List items can contain multiple paragraphs if continuation lines are properly indented
      - List items can contain other block elements: code blocks, blockquotes, nested lists
      - Switching list markers (- to * or - to 1.) starts a new list
    pitfalls:
      - List continuation indentation is measured from the column AFTER the list marker + space, NOT a fixed number of spaces
      - A blank line between items makes the ENTIRE list loose, not just the affected items
      - '- foo' starts a list but '-foo' does not (space after marker is required)
      - Indented code blocks inside list items need 4 spaces PLUS the list indentation
      - Lazy continuation lines (not indented) are allowed for paragraph content in list items, making parsing ambiguous
      - Mixed list types at the same level should start separate lists
    concepts:
      - Indentation-based nesting with column tracking
      - Tight vs loose list distinction
      - List continuation and lazy continuation
      - Block elements inside list items
    skills:
      - Indentation tracking with column-based parsing
      - Recursive structure building for nested lists
      - Tight/loose list classification
      - Multi-line list item content handling
    deliverables:
      - Unordered list parser recognizing -, *, + markers
      - Ordered list parser recognizing numeric-dot markers with start number
      - Nesting detector tracking indentation columns for hierarchical lists
      - Tight/loose list classifier based on blank line presence
      - List item content parser handling multi-paragraph and nested block elements
      - AST nodes: ordered_list (with start number), unordered_list, list_item
    estimated_hours: "4-6"

  - id: markdown-renderer-m4
    name: HTML Generation & Safety
    description: >-
      Convert AST to valid HTML output with proper escaping, optional
      raw HTML pass-through, and XSS prevention.
    acceptance_criteria:
      - Recursive AST traversal generates HTML by visiting each node and producing appropriate tags
      - Special characters in TEXT content are escaped: & → &amp;, < → &lt;, > → &gt;, " → &quot;
      - Escaping occurs during generation, NOT during parsing (AST stores raw text)
      - HTML elements are properly nested with correct opening and closing tags
      - Void elements use correct syntax: <hr>, <br>, <img> (no closing tag needed in HTML5)
      - Code block content is escaped (no Markdown or HTML processing inside code)
      - Generated HTML passes basic structural validation (matching open/close tags)
      - Handle raw HTML in Markdown source: by default, pass through (CommonMark spec) OR sanitize (configurable)
      - XSS sanitization mode: strip <script>, <iframe>, on* event attributes, and javascript: URLs from passed-through HTML
      - Link URLs are sanitized: reject javascript:, data:, and vbscript: URL schemes
      - Optional: wrap output in HTML5 document template (<!DOCTYPE html>, <head>, <body>)
      - Run generated HTML against at least 50 CommonMark spec examples and verify output matches expected
    pitfalls:
      - Double-escaping: if you escape during parsing AND generation, &amp; becomes &amp;amp;
      - Raw HTML pass-through is required by CommonMark spec but is a major XSS vector; always offer sanitization mode
      - Self-closing tags: <hr/> and <hr> are both valid in HTML5; be consistent
      - Link URLs can contain special characters that need different escaping rules than content text
      - CommonMark spec has hundreds of edge cases; use the official test suite for verification
      - Pretty-printing (indentation) can add whitespace that changes rendering in some contexts (e.g., inside <pre>)
    concepts:
      - AST-to-HTML recursive traversal
      - HTML entity escaping
      - Raw HTML handling and XSS prevention
      - CommonMark spec compliance testing
    skills:
      - Recursive tree-to-string serialization
      - HTML entity escaping
      - URL sanitization
      - Test suite verification against specification
    deliverables:
      - Recursive AST visitor generating HTML for each node type
      - HTML entity escaper for content text
      - URL sanitizer rejecting dangerous schemes
      - Raw HTML handler with configurable pass-through or sanitization
      - XSS sanitization stripping dangerous elements and attributes
      - CommonMark spec test runner verifying output against expected HTML
      - Optional HTML5 document wrapper template
    estimated_hours: "4-6"
```